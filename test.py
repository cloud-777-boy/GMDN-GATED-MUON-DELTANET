"""GPU verification + benchmark for the fused Triton state scan.

    python3 test_triton.py

Part 1 is the CORRECTNESS PROOF: kernel in IEEE-fp32 mode vs eager with TF32
disabled — same precision on both sides, tight tolerance, realistic PSD Kw
inputs (spectral norm <~ 1, like real training; adversarially-scaled random
K makes the scan exponentially unstable and amplifies rounding into false
failures). Part 2 measures the TF32-vs-fp32 rounding gap (informational).
Part 3 benchmarks.
"""
import time
import torch

import fast_scan
from fast_scan import scan_eager, _TritonScan, HAS_TRITON
from train_ablation import FastWeightMixer

assert torch.cuda.is_available() and HAS_TRITON, "needs CUDA + triton"
dev = "cuda"
torch.manual_seed(0)


def rel(x, y):
    return (x - y).abs().max().item() / (y.abs().max().item() + 1e-30)


def make_inputs(B=4, H=12, N=16, C=64, dh=64):
    """Realistic scan inputs: Kw = k^T diag(w) k with normalized k, w in (0,0.3)
    (PSD, spectral <~ 1); Uv at matching scale."""
    k = torch.nn.functional.normalize(torch.randn(B, H, N, C, dh, device=dev), dim=-1)
    v = torch.randn(B, H, N, C, dh, device=dev) / dh ** 0.5
    w = torch.rand(B, H, N, C, device=dev) * 0.3
    K = (k * w.unsqueeze(-1)).mT @ k
    U = (v * w.unsqueeze(-1)).mT @ k
    gates = [(torch.rand(B, H, N, device=dev) * r + o)
             for r, o in ((0.2, 0.75), (0.5, 0.2), (0.5, 0.3), (0.2, 0.7))]
    return [U, K] + gates


def compare(precision, tol):
    fast_scan.set_precision(precision)
    torch.backends.cuda.matmul.allow_tf32 = (precision == "tf32")
    ok_all = True
    for rule, name in ((0, "delta"), (1, "muon"), (2, "adam")):
        for ns in (3, 5):
            base = make_inputs()
            ins_e = [t.clone().requires_grad_(True) for t in base]
            ins_t = [t.clone().requires_grad_(True) for t in base]
            S_e = scan_eager(*ins_e, rule, ns)
            S_t = _TritonScan.apply(*ins_t, rule, ns)
            fdiff = rel(S_t, S_e)
            gS = torch.randn_like(S_e)
            torch.autograd.backward(S_e, gS)
            torch.autograd.backward(S_t, gS)
            bdiff = max(rel(a.grad, b.grad) for a, b in zip(ins_t, ins_e)
                        if b.grad is not None and b.grad.abs().max() > 0)
            ok = fdiff < tol and bdiff < tol
            ok_all &= ok
            print(f"  rule={name:<6} ns={ns}  fwd rel {fdiff:.2e}  "
                  f"bwd worst rel {bdiff:.2e}  {'OK' if ok else 'FAIL'}")
    return ok_all


print("=" * 70)
print("1) CORRECTNESS: kernel(ieee fp32) vs eager(strict fp32) — must be tight")
print("=" * 70)
assert compare("ieee", 1e-3), "IEEE-mode mismatch => real kernel bug, report it"

print()
print("=" * 70)
print("2) TF32 rounding gap: kernel(tf32) vs eager(tf32-enabled) — informational")
print("=" * 70)
compare("tf32", 5e-2)

print()
print("=" * 70)
print("3) mixer fwd+bwd benchmark @ 125M-layer scale (B=16, T=1024, d=768)")
print("=" * 70)
torch.backends.cuda.matmul.allow_tf32 = True
x = torch.randn(16, 1024, 768, device=dev)


def bench(mixer, iters=10):
    def step():
        with torch.autocast("cuda", torch.bfloat16):
            y = mixer(x)
        y.square().mean().backward()
        mixer.zero_grad(set_to_none=True)
    step(); step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


for inner in ("delta", "muon", "adam"):
    m = FastWeightMixer(768, 12, inner, chunk=64, ns_steps=3).to(dev)
    fast_scan.set_backend("eager")
    t_e = bench(m)
    fast_scan.set_backend("triton")
    fast_scan.set_precision("tf32")
    t_t = bench(m)
    fast_scan.set_precision("ieee")
    t_i = bench(m)
    print(f"  inner={inner:<6} eager {t_e:7.1f} ms | triton-tf32 {t_t:6.1f} ms "
          f"({t_e/t_t:4.1f}x) | triton-ieee {t_i:6.1f} ms ({t_e/t_i:4.1f}x)")

fast_scan.set_backend("auto")
fast_scan.set_precision("tf32")
print("\nPart 1 OK => kernel math verified on-device. Train with --scan auto; "
      "pick --scan-precision by part 3 (tf32 matches eager's precision).")