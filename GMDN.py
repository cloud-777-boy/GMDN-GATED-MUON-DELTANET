#GATED MUON DELTANET
#LAFAYETTE J COMPTON 2026
#DELTA RULE IS IMPLICT SGD. SO WHY NOT USE MUON?

#!/usr/bin/env python3
"""
6-way ablation: {Muon, AdamW} outer optimizer  x  {muon, adam, delta} inner rule
=================================================================================

Grid:
    outer in {muon, adam}     -- optimizer on the SLOW weights (training time)
    inner in {muon, adam, delta} -- optimizer on the FAST weights (test time,
                                    inside the recurrence)

Scientific controls baked in:
  * All three inner rules share the IDENTICAL chunkwise parallelization:
    parallel intra-chunk read + errors vs. the frozen chunk-start state.
    Cells differ ONLY in the chunk-boundary preconditioner:
        delta:  S <- a*S + sum_s beta_s e_s k_s^T          (raw gradient)
        muon:   M <- mu*M + G;  S <- a*S + b * msign(M)     (spectral precond.)
        adam:   M,V EMAs;       S <- a*S + b * M/(sqrt(V)+eps) (elementwise)
    => "delta" here is a parallel-delta variant, not exact WY-form GatedDeltaNet.
       This is deliberate: differences are attributable to the preconditioner.
  * Muon-outer uses the conventional hybrid: Muon on 2D hidden matrices,
    AdamW on embeddings / lm_head / norms / biases / convs. The Adam-outer row
    uses AdamW everywhere with the same AdamW hyperparameters for that group.
  * Paired data order: every condition consumes the same seeded stream.
  * Val set = first --val-batches of the stream, identical across conditions.

Usage:
    python train_ablation.py --smoke                 # tiny CPU/GPU pipe check, ~1 min
    python train_ablation.py --sanity                # all 6 cells, 125M, 1000 steps each
    python train_ablation.py                         # all 6 cells, full budget
    python train_ablation.py --conditions adam:delta # one cell
    python train_ablation.py --data wikitext         # alt corpus

Data: streams HuggingFaceFW/fineweb-edu (sample-10BT) or wikitext-103 via
`datasets` + `tiktoken` (pip install datasets tiktoken). Falls back to a
synthetic Markov corpus (learnable, vocab 512) if those aren't installed.

Logs every --log-interval steps: train loss, val loss/ppl, lr, grad norm,
tok/s, mean gates. CSV per condition in results/<tag>/, summary at the end.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Newton-Schulz msign (shared by inner-muon rule and outer Muon optimizer)
# ---------------------------------------------------------------------------

def msign(M: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    dt = M.dtype
    X = M.float()
    transposed = X.shape[-2] > X.shape[-1]
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * (A @ A)) @ X
    if transposed:
        X = X.mT
    return X.to(dt)


# ---------------------------------------------------------------------------
# Fast-weight mixer: unified chunkwise engine, three inner rules
# ---------------------------------------------------------------------------

class FastWeightMixer(nn.Module):
    """Token mixing via fast weights S (per head), inner rule in
    {'delta', 'muon', 'adam'}. State math runs in fp32 for all cells."""

    def __init__(self, d_model: int, n_heads: int, inner: str,
                 chunk: int = 64, ns_steps: int = 5, conv_kernel: int = 4):
        super().__init__()
        assert inner in ("delta", "muon", "adam")
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.inner, self.chunk, self.ns_steps = inner, chunk, ns_steps

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.conv = nn.Conv1d(3 * d_model, 3 * d_model, conv_kernel,
                              groups=3 * d_model, padding=conv_kernel - 1)
        self.n_gates = 4 if inner == "adam" else 3   # alpha, beta, mu (, nu)
        self.gates = nn.Linear(d_model, self.n_gates * n_heads, bias=True)
        self.norm = nn.RMSNorm(self.dh)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.last_gate_means: dict[str, float] = {}
        self.reset_gate_bias()

    def reset_gate_bias(self):
        with torch.no_grad():
            bias = self.gates.bias.view(self.n_gates, self.h)
            bias[0].fill_(3.0)    # alpha ~ 0.95 (remember-mostly)
            bias[1].fill_(-1.0)   # beta  ~ 0.27
            bias[2].fill_(1.0)    # mu    ~ 0.73
            if self.n_gates == 4:
                bias[3].fill_(2.0)  # nu ~ 0.88

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Three-phase chunkwise forward (FLA-style schedule):
          phase 1: all O(T*C*dh) matmuls batched across every chunk (amp dtype)
          phase 2: sequential scan over N=T/C chunks, only dh x dh state math (fp32)
          phase 3: one batched output matmul across all chunks
        Identical math to the naive per-chunk loop; only the schedule changes.
        Uses G_i = Uv_i - S_i @ Kw_i and o_i = (Q_i - A_i K_i) S_i^T + A_i V_i,
        which hold because intra-chunk errors are linear in the frozen S_i."""
        B, T, D = x.shape
        C, H, dh = self.chunk, self.h, self.dh
        assert T % C == 0, "seq_len must be a multiple of chunk size"
        N = T // C

        qkv = self.conv(self.qkv(x).transpose(1, 2))[..., :T].transpose(1, 2)
        qkv = F.silu(qkv)
        q, k, v = qkv.chunk(3, dim=-1)
        q = F.normalize(q.view(B, T, H, dh).transpose(1, 2), dim=-1)
        k = F.normalize(k.view(B, T, H, dh).transpose(1, 2), dim=-1)
        v = v.view(B, T, H, dh).transpose(1, 2)

        g = torch.sigmoid(self.gates(x)).view(B, T, self.n_gates, H)
        g = g.permute(2, 0, 3, 1)                             # (nG, B, H, T)
        alpha, beta, mu = g[0], g[1], g[2]
        nu = g[3] if self.n_gates == 4 else None
        if not torch.compiler.is_compiling():                 # skip side effect in
            self.last_gate_means = {                          # compiled graphs
                "alpha": alpha.detach().mean(), "beta": beta.detach().mean(),
                "mu": mu.detach().mean(),
                "nu": nu.detach().mean() if nu is not None else float("nan"),
            }

        # chunked views: (B, H, N, C, dh) / (B, H, N, C)
        qn = q.view(B, H, N, C, dh)
        kn = k.view(B, H, N, C, dh)
        vn = v.view(B, H, N, C, dh)
        an = alpha.view(B, H, N, C)
        bn = beta.view(B, H, N, C)
        mn = mu.view(B, H, N, C)

        # per-rule token weights w inside the chunk gradient
        if self.inner == "delta":
            w = bn
        elif self.inner == "muon":                # residual mu decay to chunk end
            logmu = torch.log(mn.clamp_min(1e-6))
            w = torch.exp(torch.flip(torch.cumsum(torch.flip(logmu, [-1]), -1),
                                     [-1]) - logmu)
        else:
            w = torch.ones_like(bn)

        # ---- phase 1: batched across all chunks ----
        mask = torch.tril(torch.ones(C, C, device=x.device, dtype=torch.bool))
        A = (qn @ kn.mT) * bn.unsqueeze(-2)                   # (B,H,N,C,C)
        A = A.masked_fill(~mask, 0.0)
        Kw = (kn * w.unsqueeze(-1)).mT @ kn                   # k^T diag(w) k
        Uv = (vn * w.unsqueeze(-1)).mT @ kn                   # v^T diag(w) k
        Qt = qn - A @ kn                                      # (B,H,N,C,dh)
        Ov = A @ vn                                           # (B,H,N,C,dh)

        # per-chunk gate scalars (fp32)
        a_c = torch.exp(torch.log(an.clamp_min(1e-6)).float().sum(-1))
        b_c = bn.float().mean(-1)
        mu_c = torch.exp(torch.log(mn.clamp_min(1e-6)).float().sum(-1))
        if nu is not None:
            nun = nu.view(B, H, N, C)
            nu_c = torch.exp(torch.log(nun.clamp_min(1e-6)).float().sum(-1))

        # ---- phase 2: sequential scan, dh x dh state math in fp32 ----
        Uv32, Kw32 = Uv.float(), Kw.float()
        S = x.new_zeros(B, H, dh, dh, dtype=torch.float32)
        M = torch.zeros_like(S)
        V2 = torch.zeros_like(S) if self.inner == "adam" else None
        states = []
        for i in range(N):
            states.append(S)
            G = Uv32[:, :, i] - S @ Kw32[:, :, i]
            ai = a_c[:, :, i, None, None]
            if self.inner == "delta":
                S = ai * S + G
            elif self.inner == "muon":
                M = mu_c[:, :, i, None, None] * M + G
                S = ai * S + b_c[:, :, i, None, None] * msign(M, self.ns_steps)
            else:  # adam
                mi = mu_c[:, :, i, None, None]
                ni = nu_c[:, :, i, None, None]
                M = mi * M + (1 - mi) * G
                V2 = ni * V2 + (1 - ni) * G.square()
                S = ai * S + b_c[:, :, i, None, None] * M / (V2.sqrt() + 1e-6)

        # ---- phase 3: batched output across all chunks ----
        Sst = torch.stack(states, dim=2).to(qn.dtype)         # (B,H,N,dh,dh)
        o = Qt @ Sst.mT + Ov                                  # (B,H,N,C,dh)
        o = o.reshape(B, H, T, dh)
        o = self.norm(o).to(x.dtype)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.out(o)


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Block(nn.Module):
    def __init__(self, d_model, n_heads, inner, chunk, ns_steps=5):
        super().__init__()
        self.n1 = nn.RMSNorm(d_model)
        self.mixer = FastWeightMixer(d_model, n_heads, inner, chunk, ns_steps)
        self.n2 = nn.RMSNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False), nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False))

    def forward(self, x):
        x = x + self.mixer(self.n1(x))
        return x + self.mlp(self.n2(x))


class LM(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, inner, chunk=64,
                 ns_steps=5):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, inner, chunk, ns_steps) for _ in range(n_layers))
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        self.lm_head.weight = self.emb.weight                 # tied
        self.apply(self._init)
        for blk in self.blocks:                               # scaled residual init
            for w in (blk.mixer.out.weight, blk.mlp[2].weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * n_layers))
            blk.mixer.reset_gate_bias()   # _init zeroed it; restore gate priors

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        x = self.emb(idx)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.norm_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(),
                               targets.reshape(-1))
        return logits, loss

    def gate_means(self):
        keys = ("alpha", "beta", "mu", "nu")
        vals = {k: [] for k in keys}
        for blk in self.blocks:
            for k in keys:
                v = blk.mixer.last_gate_means.get(k, float("nan"))
                v = v.item() if torch.is_tensor(v) else v
                if v == v:
                    vals[k].append(v)
        return {k: (sum(v) / len(v) if v else float("nan")) for k, v in vals.items()}


# ---------------------------------------------------------------------------
# Outer Muon optimizer (2D hidden matrices only; AdamW handles the rest)
# ---------------------------------------------------------------------------

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom) if group["nesterov"] else buf
                u = msign(u, group["ns_steps"])
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                if group["weight_decay"]:
                    p.mul_(1 - lr * group["weight_decay"])
                p.add_(u, alpha=-lr * scale)


def build_optimizers(model: LM, outer: str, args):
    hidden2d, other = [], []
    emb_ids = {id(model.emb.weight)}
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and id(p) not in emb_ids:
            hidden2d.append(p)
        else:
            other.append(p)
    adamw_kw = dict(lr=args.adam_lr, betas=(0.9, 0.95),
                    weight_decay=args.weight_decay)
    if outer == "muon":
        return [Muon(hidden2d, lr=args.muon_lr, weight_decay=args.weight_decay),
                torch.optim.AdamW(other, **adamw_kw)]
    return [torch.optim.AdamW(hidden2d, **adamw_kw),
            torch.optim.AdamW(other, **adamw_kw)]


def lr_factor(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))


# ---------------------------------------------------------------------------
# Data: fineweb-edu / wikitext streaming, or synthetic Markov fallback
# ---------------------------------------------------------------------------

def synthetic_stream(seed: int, vocab: int = 512):
    """Learnable Markov chain: next = (5*cur + 17) % V with 10% uniform noise."""
    gen = torch.Generator().manual_seed(seed)
    cur = torch.randint(0, vocab, (1,), generator=gen).item()
    while True:
        block = torch.empty(8192, dtype=torch.long)
        for i in range(8192):
            if torch.rand((), generator=gen).item() < 0.10:
                cur = torch.randint(0, vocab, (1,), generator=gen).item()
            else:
                cur = (5 * cur + 17) % vocab
            block[i] = cur
        yield block


def hf_stream(source: str, seed: int):
    from datasets import load_dataset            # noqa: deferred import
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    if source == "fineweb":
        ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
    else:
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    for row in ds:
        text = row["text"]
        if text.strip():
            yield torch.tensor(enc.encode_ordinary(text) + [enc.eot_token],
                               dtype=torch.long)


def batch_iter(stream, batch_size: int, seq_len: int, device):
    buf = torch.empty(0, dtype=torch.long)
    need = batch_size * seq_len + 1
    while True:
        while buf.numel() < need:
            buf = torch.cat([buf, next(stream)])
        chunk_, buf = buf[:need], buf[need - 1:]
        x = chunk_[:-1].view(batch_size, seq_len).to(device)
        y = chunk_[1:].view(batch_size, seq_len).to(device)
        yield x, y


def make_data(args, device):
    if args.data == "synthetic":
        return batch_iter(synthetic_stream(args.data_seed), args.micro_bs,
                          args.seq_len, device), 512
    try:
        return batch_iter(hf_stream(args.data, args.data_seed), args.micro_bs,
                          args.seq_len, device), 50257
    except Exception as e:                        # noqa: broad on purpose
        print(f"[warn] '{args.data}' unavailable ({type(e).__name__}: {e}); "
              f"FALLING BACK TO SYNTHETIC DATA. pip install datasets tiktoken "
              f"for real runs.")
        return batch_iter(synthetic_stream(args.data_seed), args.micro_bs,
                          args.seq_len, device), 512


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_batches, use_amp, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in val_batches:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=use_amp):
            _, loss = model(x, y)
        total += loss.item()
        n += 1
    model.train()
    loss = total / max(n, 1)
    return loss, math.exp(min(loss, 20))


def run_condition(outer: str, inner: str, args, device, results_dir: str):
    cond = f"outer-{outer}_inner-{inner}"
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    data, vocab = make_data(args, device)
    val_batches = [next(data) for _ in range(args.val_batches)]  # paired val set

    model = LM(vocab, args.d_model, args.n_layers, args.n_heads, inner,
               chunk=args.chunk, ns_steps=args.ns_steps).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {cond} | params {n_params/1e6:.1f}M | vocab {vocab} | "
          f"{args.steps} steps x {args.micro_bs}x{args.seq_len}x{args.accum} "
          f"tok/step ===")

    opts = build_optimizers(model, outer, args)
    eager_model = model                       # for gate stats / checkpointing
    if args.compile:
        try:                                  # CUDA graphs: kills launch overhead
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except Exception as e:
            print(f"[warn] reduce-overhead compile failed ({e}); default mode")
            model = torch.compile(model)
    base_lrs = [[g["lr"] for g in opt.param_groups] for opt in opts]
    use_amp = device.type == "cuda"

    csv_path = os.path.join(results_dir, f"{cond}.csv")
    json_path = os.path.join(results_dir, f"{cond}_metrics.json")
    with open(csv_path, "w") as f:
        f.write("step,train_loss,val_loss,val_ppl,lr_hidden,grad_norm,tok_s,"
                "alpha,beta,mu,nu,elapsed_s\n")
    run_meta = dict(condition=cond, outer=outer, inner=inner,
                    params_M=n_params / 1e6, vocab=vocab,
                    tokens_per_step=args.micro_bs * args.seq_len * args.accum,
                    torch_version=torch.__version__,
                    device=(torch.cuda.get_device_name(0)
                            if device.type == "cuda" else "cpu"),
                    args=vars(args))
    log_rows: list[dict] = []

    def flush_json():                       # incremental & crash-safe
        tmp = json_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(meta=run_meta, log=log_rows), f, indent=1)
        os.replace(tmp, json_path)

    model.train()
    t0 = t_last = time.time()
    loss_acc, loss_n, tok_since = 0.0, 0, 0
    grad_norm = float("nan")

    for step in range(1, args.steps + 1):
        fac = lr_factor(step - 1, args.warmup, args.steps)
        for opt, bases in zip(opts, base_lrs):
            for g, b in zip(opt.param_groups, bases):
                g["lr"] = b * fac
        for opt in opts:
            opt.zero_grad(set_to_none=True)

        for _ in range(args.accum):
            x, y = next(data)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=use_amp):
                _, loss = model(x, y)
            (loss / args.accum).backward()
            loss_acc += loss.item()
            loss_n += 1
            tok_since += x.numel()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
        for opt in opts:
            opt.step()

        if step % args.log_interval == 0 or step == args.steps:
            val_loss, val_ppl = evaluate(model, val_batches, use_amp, device)
            if args.compile:                  # stats skipped in compiled graphs;
                with torch.no_grad():         # one eager fwd repopulates them
                    eager_model(val_batches[0][0])
            now = time.time()
            tok_s = tok_since / max(now - t_last, 1e-9)
            gm = eager_model.gate_means()
            train_loss = loss_acc / max(loss_n, 1)
            lr_h = opts[0].param_groups[0]["lr"]
            print(f"[{cond}] step {step:>6}/{args.steps} | "
                  f"loss {train_loss:.4f} | val {val_loss:.4f} "
                  f"(ppl {val_ppl:.2f}) | lr {lr_h:.2e} | gnorm {grad_norm:.2f} | "
                  f"{tok_s/1e3:.1f}k tok/s | "
                  f"a {gm['alpha']:.2f} b {gm['beta']:.2f} m {gm['mu']:.2f} | "
                  f"{(now - t0)/60:.1f} min")
            with open(csv_path, "a") as f:
                f.write(f"{step},{train_loss:.6f},{val_loss:.6f},{val_ppl:.4f},"
                        f"{lr_h:.6e},{grad_norm:.4f},{tok_s:.1f},"
                        f"{gm['alpha']:.4f},{gm['beta']:.4f},{gm['mu']:.4f},"
                        f"{gm['nu']:.4f},{now - t0:.1f}\n")
            log_rows.append(dict(
                step=step, train_loss=round(train_loss, 6),
                val_loss=round(val_loss, 6), val_ppl=round(val_ppl, 4),
                lr_hidden=lr_h, grad_norm=round(grad_norm, 4),
                tok_s=round(tok_s, 1),
                tokens_seen=step * args.micro_bs * args.seq_len * args.accum,
                alpha=round(gm["alpha"], 4), beta=round(gm["beta"], 4),
                mu=round(gm["mu"], 4), nu=round(gm["nu"], 4),
                elapsed_s=round(now - t0, 1)))
            flush_json()
            loss_acc, loss_n, tok_since, t_last = 0.0, 0, 0, now

    val_loss, val_ppl = evaluate(model, val_batches, use_amp, device)
    result = dict(condition=cond, params_M=n_params / 1e6, final_val_loss=val_loss,
                  final_val_ppl=val_ppl, minutes=(time.time() - t0) / 60)
    run_meta["final"] = result
    flush_json()
    if not args.no_save:
        ckpt_path = os.path.join(results_dir, f"{cond}.pt")
        torch.save(dict(
            model_state=eager_model.state_dict(),
            model_config=dict(vocab=vocab, d_model=args.d_model,
                              n_layers=args.n_layers, n_heads=args.n_heads,
                              inner=inner, chunk=args.chunk),
            condition=cond, outer=outer, inner=inner,
            step=args.steps, seed=args.seed, data_seed=args.data_seed,
            args=vars(args), torch_version=torch.__version__,
            final_val_loss=val_loss, final_val_ppl=val_ppl,
        ), ckpt_path)
        print(f"[{cond}] checkpoint -> {ckpt_path}")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [(o, i) for o in ("muon", "adam") for i in ("muon", "adam", "delta")]


def main():
    p = argparse.ArgumentParser(description="6-way outer x inner optimizer ablation")
    p.add_argument("--conditions", default="all",
                   help="'all' or comma-separated outer:inner pairs, "
                        "e.g. 'muon:muon,adam:delta'")
    p.add_argument("--data", default="fineweb",
                   choices=["fineweb", "wikitext", "synthetic"])
    p.add_argument("--steps", type=int, default=9600)
    p.add_argument("--log-interval", type=int, default=1000)
    p.add_argument("--val-batches", type=int, default=16)
    p.add_argument("--warmup", type=int, default=500)
    # 125M config
    p.add_argument("--d-model", type=int, default=768)
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--ns-steps", type=int, default=5,
                   help="Newton-Schulz iterations in the inner-muon rule")
    p.add_argument("--micro-bs", type=int, default=16)
    p.add_argument("--accum", type=int, default=8)
    # optimizers
    p.add_argument("--adam-lr", type=float, default=4e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.1)
    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-seed", type=int, default=1234,
                   help="kept identical across conditions => paired data order")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-save", action="store_true",
                   help="skip saving model.pt checkpoints (e.g. for LR sweeps)")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (extra warmup, faster steady-state)")
    p.add_argument("--sanity", action="store_true",
                   help="all 6 conditions at full 125M scale, 1000 steps each")
    p.add_argument("--smoke", action="store_true",
                   help="tiny model, synthetic data, ~30 steps: pipeline check")
    args = p.parse_args()

    if args.sanity:
        args.steps, args.log_interval, args.warmup = 1000, 250, 100
        args.val_batches = 8
        args.conditions = "all"
    if args.smoke:
        args.d_model, args.n_layers, args.n_heads = 128, 2, 4
        args.seq_len, args.chunk, args.micro_bs, args.accum = 256, 64, 4, 1
        args.steps, args.log_interval, args.warmup = 30, 10, 5
        args.val_batches, args.data, args.conditions = 4, "synthetic", "all"

    if args.conditions == "all":
        conditions = ALL_CONDITIONS
    else:
        conditions = [tuple(c.split(":")) for c in args.conditions.split(",")]
        for o, i in conditions:
            assert o in ("muon", "adam") and i in ("muon", "adam", "delta"), (o, i)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # tensor cores for fp32 state math
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    tag = args.tag or time.strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join("results", tag)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"device={device} | conditions={conditions} | results -> {results_dir}")

    summary = []
    for outer, inner in conditions:
        summary.append(run_condition(outer, inner, args, device, results_dir))

    print("\n" + "=" * 78)
    print(f"{'condition':<28} {'params':>8} {'val loss':>10} {'val ppl':>10} "
          f"{'minutes':>8}")
    for r in summary:
        print(f"{r['condition']:<28} {r['params_M']:>7.1f}M "
              f"{r['final_val_loss']:>10.4f} {r['final_val_ppl']:>10.2f} "
              f"{r['minutes']:>8.1f}")
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
