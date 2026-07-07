"""
Fused fast-weight state scan (phase 2 of the chunkwise mixer).
================================================================

Computes, per (batch, head), the sequential chunk recurrence over N chunks:

    inputs per chunk i:  Uv_i, Kw_i  (dh x dh),  scalars a_i, b_i, mu_i, nu_i
    G_i = Uv_i - S_i @ Kw_i
    rule 0 (delta): S_{i+1} = a_i S_i + G_i
    rule 1 (muon):  M_{i+1} = mu_i M_i + G_i
                    S_{i+1} = a_i S_i + b_i * msign(M_{i+1})     [NS iterations]
    rule 2 (adam):  M_{i+1} = mu_i M_i + (1-mu_i) G_i
                    V_{i+1} = nu_i V_i + (1-nu_i) G_i^2
                    S_{i+1} = a_i S_i + b_i * M_{i+1} / (sqrt(V_{i+1}) + 1e-6)
    output: the PRE-update states S_0..S_{N-1}  (read by phase 3)

Three implementations, identical math:
  * scan_eager        -- PyTorch loop, autograd backward (reference / CPU / fallback)
  * manual_backward   -- hand-derived VJP in PyTorch (validates the derivation;
                         the Triton backward is a transcription of this function)
  * Triton kernels    -- one fused kernel launch fwd, one bwd; state pinned in
                         registers/SMEM across the whole scan

The Newton-Schulz iteration count NS is a unified knob: the same value flows
through the eager path, the manual backward, and the Triton kernels
(--ns-steps in the trainer).
"""

from __future__ import annotations
import torch

NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
_EPS_NORM = 1e-7
_EPS_ADAM = 1e-6

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

_BACKEND = "auto"          # 'auto' | 'triton' | 'eager'
_PRECISION = "tf32"        # 'tf32' | 'ieee'  (tl.dot input precision)


def set_precision(name: str):
    global _PRECISION
    assert name in ("tf32", "ieee")
    _PRECISION = name


def set_backend(name: str):
    global _BACKEND
    assert name in ("auto", "triton", "eager")
    _BACKEND = name


# ---------------------------------------------------------------------------
# Eager reference (autograd-able)
# ---------------------------------------------------------------------------

def _msign_sq(M: torch.Tensor, ns: int) -> torch.Tensor:
    """Newton-Schulz msign for square matrices (dv == dk)."""
    X = M / (M.norm(dim=(-2, -1), keepdim=True) + _EPS_NORM)
    for _ in range(ns):
        A = X @ X.mT
        X = NS_A * X + (NS_B * A + NS_C * (A @ A)) @ X
    return X


def scan_eager(U, K, a, b, m, n, rule: int, ns: int) -> torch.Tensor:
    """U, K: (B, H, N, dh, dh); a, b, m, n: (B, H, N). Returns pre-update
    states (B, H, N, dh, dh). Autograd provides the backward."""
    B, H, N, dh, _ = U.shape
    S = U.new_zeros(B, H, dh, dh)
    M = torch.zeros_like(S)
    V = torch.zeros_like(S)
    states = []
    for i in range(N):
        states.append(S)
        G = U[:, :, i] - S @ K[:, :, i]
        ai = a[:, :, i, None, None]
        if rule == 0:
            S = ai * S + G
        elif rule == 1:
            M = m[:, :, i, None, None] * M + G
            S = ai * S + b[:, :, i, None, None] * _msign_sq(M, ns)
        else:
            mi = m[:, :, i, None, None]
            ni = n[:, :, i, None, None]
            M = mi * M + (1 - mi) * G
            V = ni * V + (1 - ni) * G.square()
            S = ai * S + b[:, :, i, None, None] * M / (V.sqrt() + _EPS_ADAM)
    return torch.stack(states, dim=2)


# ---------------------------------------------------------------------------
# Hand-derived backward (PyTorch mirror of the Triton bwd kernel)
# ---------------------------------------------------------------------------

def _ns_step(X):
    A = X @ X.mT
    return NS_A * X + (NS_B * A + NS_C * (A @ A)) @ X


def _msign_vjp(Mi, Ybar, ns: int):
    """VJP of Y = msign(Mi) (square). Recomputes the NS chain.
    Per-iteration VJP for X' = NS_A X + (NS_B A + NS_C A^2) X, A = X X^T:
        Bbar = Xbar' X^T
        Abar = NS_B Bbar + NS_C (Bbar A + A Bbar)          [A symmetric]
        Xbar = NS_A Xbar' + (NS_B A + NS_C A^2) Xbar' + (Abar + Abar^T) X
    Normalization X0 = Mi / (||Mi||_F + eps):
        Mbar = Xbar0/nrm - (<Xbar0, Mi>/nrm^2) * Mi/||Mi||_F
    """
    nf = Mi.norm(dim=(-2, -1), keepdim=True)
    nrm = nf + _EPS_NORM
    X0 = Mi / nrm
    Xbar = Ybar
    for kk in range(ns):
        k = ns - 1 - kk
        Xk = X0
        for _ in range(k):
            Xk = _ns_step(Xk)
        A = Xk @ Xk.mT
        Bm = NS_B * A + NS_C * (A @ A)
        Bbar = Xbar @ Xk.mT
        Abar = NS_B * Bbar + NS_C * (Bbar @ A + A @ Bbar)
        Xbar = NS_A * Xbar + Bm @ Xbar + (Abar + Abar.mT) @ Xk
    dot = (Xbar * Mi).sum(dim=(-2, -1), keepdim=True)
    return Xbar / nrm - dot / nrm.square() * (Mi / nf.clamp_min(1e-12))


def manual_backward(U, K, a, b, m, n, S_all, M_all, V_all, gS,
                    rule: int, ns: int):
    """Reverse scan. S_all/M_all/V_all hold PRE-update values per chunk.
    Returns dU, dK, da, db, dm, dn."""
    B, H, N, dh, _ = U.shape
    dU = torch.zeros_like(U)
    dK = torch.zeros_like(K)
    da = torch.zeros_like(a)
    db = torch.zeros_like(b)
    dm = torch.zeros_like(m)
    dn = torch.zeros_like(n)
    Sb = U.new_zeros(B, H, dh, dh)     # dL/dS_{i+1}
    Mb = torch.zeros_like(Sb)          # dL/dM_{i+1} (as input to chunk i+1)
    Vb = torch.zeros_like(Sb)
    for i in reversed(range(N)):
        S = S_all[:, :, i]
        Ui, Ki = U[:, :, i], K[:, :, i]
        ai = a[:, :, i, None, None]
        bi = b[:, :, i, None, None]
        mi = m[:, :, i, None, None]
        G = Ui - S @ Ki
        da[:, :, i] = (Sb * S).sum(dim=(-2, -1))
        if rule == 0:
            Gb = Sb
        elif rule == 1:
            Mp = M_all[:, :, i]
            Mi = mi * Mp + G
            Y = _msign_sq(Mi, ns)
            db[:, :, i] = (Sb * Y).sum(dim=(-2, -1))
            Mtot = Mb + _msign_vjp(Mi, bi * Sb, ns)
            dm[:, :, i] = (Mtot * Mp).sum(dim=(-2, -1))
            Mb = mi * Mtot
            Gb = Mtot
        else:
            Mp, Vp = M_all[:, :, i], V_all[:, :, i]
            ni = n[:, :, i, None, None]
            Mi = mi * Mp + (1 - mi) * G
            Vi = ni * Vp + (1 - ni) * G.square()
            den = Vi.sqrt() + _EPS_ADAM
            upd = Mi / den
            db[:, :, i] = (Sb * upd).sum(dim=(-2, -1))
            ub = bi * Sb
            Mtot = Mb + ub / den
            Vtot = Vb - ub * Mi / den.square() / (2 * Vi.sqrt().clamp_min(1e-12))
            dm[:, :, i] = (Mtot * (Mp - G)).sum(dim=(-2, -1))
            dn[:, :, i] = (Vtot * (Vp - G.square())).sum(dim=(-2, -1))
            Mb = mi * Mtot
            Vb = ni * Vtot
            Gb = (1 - mi) * Mtot + (1 - ni) * 2 * G * Vtot
        dU[:, :, i] = Gb
        dK[:, :, i] = -S.mT @ Gb
        Sb = ai * Sb - Gb @ Ki.mT + gS[:, :, i]
    return dU, dK, da, db, dm, dn


# ---------------------------------------------------------------------------
# Triton kernels (transcription of the validated math above)
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _dot(a, b, IEEE: tl.constexpr):
        if IEEE:
            return tl.dot(a, b, input_precision="ieee")
        return tl.dot(a, b)

    @triton.jit
    def _ns_step_t(X, IEEE: tl.constexpr):
        A = _dot(X, tl.trans(X), IEEE)
        return 3.4445 * X + _dot(-4.7750 * A + 2.0315 * _dot(A, A, IEEE), X, IEEE)

    @triton.jit
    def _scan_fwd_kernel(U_ptr, K_ptr, A_ptr, B_ptr, MU_ptr, NU_ptr,
                         S_ptr, M_ptr, V_ptr,
                         N, s_bh, s_n, s_bhv,
                         DH: tl.constexpr, RULE: tl.constexpr,
                         NS: tl.constexpr, IEEE: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, DH)
        idx = r[:, None] * DH + r[None, :]
        mb = pid * s_bh
        vb = pid * s_bhv
        S = tl.zeros((DH, DH), dtype=tl.float32)
        M = tl.zeros((DH, DH), dtype=tl.float32)
        V = tl.zeros((DH, DH), dtype=tl.float32)
        for i in range(N):
            off = mb + i * s_n + idx
            tl.store(S_ptr + off, S)
            if RULE != 0:
                tl.store(M_ptr + off, M)
            if RULE == 2:
                tl.store(V_ptr + off, V)
            Ui = tl.load(U_ptr + off)
            Ki = tl.load(K_ptr + off)
            a = tl.load(A_ptr + vb + i)
            G = Ui - _dot(S, Ki, IEEE)
            if RULE == 0:
                S = a * S + G
            elif RULE == 1:
                b = tl.load(B_ptr + vb + i)
                mu = tl.load(MU_ptr + vb + i)
                M = mu * M + G
                nrm = tl.sqrt(tl.sum(M * M)) + 1e-7
                X = M / nrm
                for _ in tl.static_range(NS):
                    X = _ns_step_t(X, IEEE)
                S = a * S + b * X
            else:
                b = tl.load(B_ptr + vb + i)
                mu = tl.load(MU_ptr + vb + i)
                nu = tl.load(NU_ptr + vb + i)
                M = mu * M + (1 - mu) * G
                V = nu * V + (1 - nu) * G * G
                S = a * S + b * M / (tl.sqrt(V) + 1e-6)

    @triton.jit
    def _msign_vjp_t(Mi, Ybar, NS: tl.constexpr, IEEE: tl.constexpr):
        nf = tl.sqrt(tl.sum(Mi * Mi))
        nrm = nf + 1e-7
        X0 = Mi / nrm
        Xbar = Ybar
        for kk in tl.static_range(NS):
            Xk = X0
            for _ in tl.static_range(NS - 1 - kk):
                Xk = _ns_step_t(Xk, IEEE)
            A = _dot(Xk, tl.trans(Xk), IEEE)
            Bm = -4.7750 * A + 2.0315 * _dot(A, A, IEEE)
            Bbar = _dot(Xbar, tl.trans(Xk), IEEE)
            Abar = -4.7750 * Bbar + 2.0315 * (_dot(Bbar, A, IEEE) + _dot(A, Bbar, IEEE))
            Xbar = 3.4445 * Xbar + _dot(Bm, Xbar, IEEE) \
                + _dot(Abar + tl.trans(Abar), Xk, IEEE)
        dot = tl.sum(Xbar * Mi)
        return Xbar / nrm - dot / (nrm * nrm) * (Mi / tl.maximum(nf, 1e-12))

    @triton.jit
    def _scan_bwd_kernel(U_ptr, K_ptr, A_ptr, B_ptr, MU_ptr, NU_ptr,
                         S_ptr, M_ptr, V_ptr, GS_ptr,
                         DU_ptr, DK_ptr, DA_ptr, DB_ptr, DMU_ptr, DNU_ptr,
                         N, s_bh, s_n, s_bhv,
                         DH: tl.constexpr, RULE: tl.constexpr,
                         NS: tl.constexpr, IEEE: tl.constexpr):
        pid = tl.program_id(0)
        r = tl.arange(0, DH)
        idx = r[:, None] * DH + r[None, :]
        mb = pid * s_bh
        vb = pid * s_bhv
        Sb = tl.zeros((DH, DH), dtype=tl.float32)
        Mb = tl.zeros((DH, DH), dtype=tl.float32)
        Vb = tl.zeros((DH, DH), dtype=tl.float32)
        for j in range(N):
            i = N - 1 - j
            off = mb + i * s_n + idx
            S = tl.load(S_ptr + off)
            Ui = tl.load(U_ptr + off)
            Ki = tl.load(K_ptr + off)
            a = tl.load(A_ptr + vb + i)
            G = Ui - _dot(S, Ki, IEEE)
            tl.store(DA_ptr + vb + i, tl.sum(Sb * S))
            if RULE == 0:
                Gb = Sb
            elif RULE == 1:
                b = tl.load(B_ptr + vb + i)
                mu = tl.load(MU_ptr + vb + i)
                Mp = tl.load(M_ptr + off)
                Mi = mu * Mp + G
                nrm = tl.sqrt(tl.sum(Mi * Mi)) + 1e-7
                Y = Mi / nrm
                for _ in tl.static_range(NS):
                    Y = _ns_step_t(Y, IEEE)
                tl.store(DB_ptr + vb + i, tl.sum(Sb * Y))
                Mtot = Mb + _msign_vjp_t(Mi, b * Sb, NS, IEEE)
                tl.store(DMU_ptr + vb + i, tl.sum(Mtot * Mp))
                Mb = mu * Mtot
                Gb = Mtot
            else:
                b = tl.load(B_ptr + vb + i)
                mu = tl.load(MU_ptr + vb + i)
                nu = tl.load(NU_ptr + vb + i)
                Mp = tl.load(M_ptr + off)
                Vp = tl.load(V_ptr + off)
                Mi = mu * Mp + (1 - mu) * G
                Vi = nu * Vp + (1 - nu) * G * G
                sq = tl.sqrt(Vi)
                den = sq + 1e-6
                upd = Mi / den
                tl.store(DB_ptr + vb + i, tl.sum(Sb * upd))
                ub = b * Sb
                Mtot = Mb + ub / den
                Vtot = Vb - ub * Mi / (den * den) / (2 * tl.maximum(sq, 1e-12))
                tl.store(DMU_ptr + vb + i, tl.sum(Mtot * (Mp - G)))
                tl.store(DNU_ptr + vb + i, tl.sum(Vtot * (Vp - G * G)))
                Mb = mu * Mtot
                Vb = nu * Vtot
                Gb = (1 - mu) * Mtot + (1 - nu) * 2 * G * Vtot
            tl.store(DU_ptr + off, Gb)
            tl.store(DK_ptr + off, -_dot(tl.trans(S), Gb, IEEE))
            Sb = a * Sb - _dot(Gb, tl.trans(Ki), IEEE) + tl.load(GS_ptr + off)


class _TritonScan(torch.autograd.Function):

    # (num_warps, num_stages) ladder: stages=1 kills the pipelining buffers
    # that blow the shared-memory budget for tile-heavy scan iterations.
    _CONFIGS = ((4, 1), (2, 1), (8, 1), (1, 1))
    _cfg_cache: dict = {}

    @staticmethod
    def _launch(kernel, grid, key, args, consts):
        from triton.runtime.errors import OutOfResources
        cached = _TritonScan._cfg_cache.get(key)
        if cached is not None:
            w, s = cached
            kernel[grid](*args, **consts, num_warps=w, num_stages=s)
            return
        last = None
        for w, s in _TritonScan._CONFIGS:
            try:
                kernel[grid](*args, **consts, num_warps=w, num_stages=s)
                _TritonScan._cfg_cache[key] = (w, s)
                return
            except OutOfResources as e:
                last = e
        raise last

    @staticmethod
    def forward(ctx, U, K, a, b, m, n, rule, ns):
        B, H, N, dh, _ = U.shape
        BH = B * H
        Uc = U.contiguous()
        Kc = K.contiguous()
        ac, bc, mc, nc = (t.contiguous() for t in (a, b, m, n))
        S_all = torch.empty_like(Uc)
        M_all = torch.empty_like(Uc) if rule != 0 else Uc.new_empty(1)
        V_all = torch.empty_like(Uc) if rule == 2 else Uc.new_empty(1)
        ieee = _PRECISION == "ieee"
        _TritonScan._launch(
            _scan_fwd_kernel, (BH,), ("fwd", rule, ns, dh, ieee),
            (Uc, Kc, ac, bc, mc, nc, S_all, M_all, V_all,
             N, N * dh * dh, dh * dh, N),
            dict(DH=dh, RULE=rule, NS=ns, IEEE=ieee))
        ctx.save_for_backward(Uc, Kc, ac, bc, mc, nc, S_all, M_all, V_all)
        ctx.rule, ctx.ns, ctx.ieee = rule, ns, ieee
        return S_all

    @staticmethod
    def backward(ctx, gS):
        Uc, Kc, ac, bc, mc, nc, S_all, M_all, V_all = ctx.saved_tensors
        B, H, N, dh, _ = Uc.shape
        BH = B * H
        dU = torch.empty_like(Uc)
        dK = torch.empty_like(Kc)
        da = torch.empty_like(ac)
        db = torch.empty_like(bc)
        dm = torch.empty_like(mc)
        dn = torch.empty_like(nc)
        _TritonScan._launch(
            _scan_bwd_kernel, (BH,), ("bwd", ctx.rule, ctx.ns, dh, ctx.ieee),
            (Uc, Kc, ac, bc, mc, nc, S_all, M_all, V_all, gS.contiguous(),
             dU, dK, da, db, dm, dn,
             N, N * dh * dh, dh * dh, N),
            dict(DH=dh, RULE=ctx.rule, NS=ctx.ns, IEEE=ctx.ieee))
        return dU, dK, da, db, dm, dn, None, None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def fast_state_scan(U, K, a, b, m, n, rule: int, ns: int) -> torch.Tensor:
    """Pre-update state stack (B, H, N, dh, dh). Backend per set_backend():
    'triton' when available on CUDA with power-of-two dh >= 16, else eager."""
    use_triton = (
        _BACKEND != "eager" and HAS_TRITON and U.is_cuda
        and U.shape[-1] >= 16 and (U.shape[-1] & (U.shape[-1] - 1)) == 0)
    if _BACKEND == "triton" and not use_triton:
        raise RuntimeError("triton backend requested but unavailable "
                           "(need CUDA + triton + power-of-two head dim >= 16)")
    if use_triton:
        return _TritonScan.apply(U, K, a, b, m, n, rule, ns)
    return scan_eager(U, K, a, b, m, n, rule, ns)