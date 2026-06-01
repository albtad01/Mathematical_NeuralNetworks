"""
Replication of Figure 4 from E, Ma, Wojtowytsch, Wu (2020).

Heat-map of test errors in (log10(m), log10(n)) space for:
  Left  – conventional scaling:  f_m = sum_j  a_j sigma(b_j^T x)
  Right – mean-field scaling:    f_m = (1/m) sum_j a_j sigma(b_j^T x)

Settings:  d=20,  eta=0.0005 (full) or 1e-3 (small),  stop when training loss < 1e-7
Target:    f*(x) = ReLU(x_1),  x ~ Uniform([0,1]^20)
Optimizer: full-batch GD (matching the paper)

m range: log10(m) = 2.0 … 4.5  (paper range; m up to ~30 000)
         Important: the phase boundary and dark low-error region only appear
         at large m (log10(m) > 3), which is why the range must reach 4.5.

MAX_ITER = 3_000_000 with stagnation detection:
  - For LARGE m (NTK regime): converges in ~10 000 iterations → fast.
  - For SMALL m (few params): needs many iterations; each step is cheap (small
    matrices). Stagnation break exits if loss hasn't improved by 1% over 300K
    consecutive iterations (handles cells where 10⁻⁷ is unreachable).
  Manual gradients (no autograd graph) keep per-step overhead minimal.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch

# MPS → CUDA → CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# ── grid settings ─────────────────────────────────────────────────────────────

D   = 20
TOL = 1e-7

SMALL_GRID = False

# Stagnation: break if loss hasn't dropped by STAG_FRAC over STAG_CHECK iters
STAG_CHECK = 300_000
STAG_FRAC  = 0.01        # 1 % improvement threshold

if SMALL_GRID:
    LR       = 1e-3
    MAX_ITER = 500_000
    log_m_vals = np.linspace(2.0, 4.5, 6)
    log_n_vals = np.linspace(1.8, 2.65, 4)
    grid_label = "SMALL (6×4)"
else:
    LR       = 5e-4
    MAX_ITER = 3_000_000   # paper: run until convergence
    log_m_vals = np.linspace(2.0, 4.5, 15)
    log_n_vals = np.linspace(1.8, 2.65, 12)
    grid_label = "FULL (15×12)"

M_vals = np.unique(np.round(10 ** log_m_vals).astype(int))
N_vals = np.unique(np.round(10 ** log_n_vals).astype(int))
log_M  = np.log10(M_vals)
log_N  = np.log10(N_vals)
total  = len(M_vals) * len(N_vals)

print(f"Grid mode : {grid_label}")
print(f"Grid size : {len(M_vals)} widths × {len(N_vals)} n-values = {total} cells")
print(f"  m range : {M_vals[0]} – {M_vals[-1]}")
print(f"  n range : {N_vals[0]} – {N_vals[-1]}")
print(f"  max_iter: {MAX_ITER:,}   tol={TOL}   lr={LR}")
print()

# ── data & init ───────────────────────────────────────────────────────────────

def make_data(n, d, seed):
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.uniform(0, 1, (n, d)), dtype=torch.float32, device=DEVICE)
    y = torch.relu(X[:, 0])
    return X, y

def init_params(m, d, mf, seed):
    rng = np.random.default_rng(seed)
    # conv: a ~ N(0, 1/√m) so initial f is O(1);  mf: a ~ N(0,1)
    a_np = rng.normal(0, 1.0 if mf else 1.0 / np.sqrt(m), m).astype(np.float32)
    B_np = rng.normal(0, 1.0 / np.sqrt(d), (m, d)).astype(np.float32)
    a = torch.tensor(a_np, device=DEVICE)
    B = torch.tensor(B_np, device=DEVICE)
    return a, B

# ── training (manual gradients, no autograd) ──────────────────────────────────

def train(n, m, d, lr, tol, max_iters, mf, seed, print_every):
    """
    Full-batch GD with manual gradients.
    No autograd: avoids graph-creation overhead over millions of iterations.
    Returns (test_error, iterations_run).
    """
    label = "mf  " if mf else "conv"

    X_tr, y_tr = make_data(n, d, seed)
    X_te, y_te = make_data(5000, d, seed + 99999)

    a, B = init_params(m, d, mf, seed)
    scale = 1.0 / m if mf else 1.0   # output scale factor

    loss_val   = float('inf')
    prev_check = float('inf')

    for it in range(1, max_iters + 1):
        # ── forward ──────────────────────────────────────────────────────────
        Z   = X_tr @ B.T             # (n, m) pre-activations
        H   = Z.clamp(min=0)         # (n, m) ReLU activations
        f   = (H @ a) * scale        # (n,)   predictions
        r   = f - y_tr               # (n,)   residuals
        loss_val = (0.5 * r.pow(2).mean()).item()

        if loss_val < tol:
            break

        # Stagnation: break if stuck at a local/global min (e.g. m too small)
        if it % STAG_CHECK == 0:
            if loss_val > prev_check * (1.0 - STAG_FRAC):
                break
            prev_check = loss_val

        if it % print_every == 0:
            print(f"      [{label}] iter {it:7d}  loss={loss_val:.3e}", flush=True)

        # ── gradients ────────────────────────────────────────────────────────
        # ∂L/∂f_i = r_i / n  →  ∂L/∂a = scale · Hᵀ (r/n)
        rn     = r / n
        grad_a = scale * (H.T @ rn)              # (m,)

        # ∂L/∂B_j = (scale · a_j / n) Σ_i r_i 1[z_ij > 0] x_i
        ind    = (Z > 0).float()                 # (n, m)
        grad_B = (scale * a.unsqueeze(0)) * ind  # (n, m)  broadcast aⱼ
        grad_B = (rn.unsqueeze(1) * grad_B).T @ X_tr  # (m, d)

        # ── GD step ──────────────────────────────────────────────────────────
        a = a - lr * grad_a
        B = B - lr * grad_B

    with torch.no_grad():
        Z_te = X_te @ B.T
        H_te = Z_te.clamp(min=0)
        f_te = (H_te @ a) * scale
        te   = (0.5 * (f_te - y_te).pow(2).mean()).item()

    return te, it

# ── main loop (with cache) ────────────────────────────────────────────────────

# Use v2 cache names — old files had the wrong m range (3.5 instead of 4.5).
_cache_files = ('te_conv_v2.npy', 'te_mf_v2.npy', 'log_M_v2.npy', 'log_N_v2.npy')
_cache_ok    = all(os.path.isfile(f) for f in _cache_files)

if _cache_ok:
    print("Loading cached arrays …")
    te_conv = np.load('te_conv_v2.npy')
    te_mf   = np.load('te_mf_v2.npy')
    log_M   = np.load('log_M_v2.npy')
    log_N   = np.load('log_N_v2.npy')
    print(f"  te_conv shape: {te_conv.shape}  te_mf shape: {te_mf.shape}")
else:
    te_conv = np.full((len(N_vals), len(M_vals)), np.nan)
    te_mf   = np.full((len(N_vals), len(M_vals)), np.nan)

    t0   = time.time()
    done = 0

    for ni, n in enumerate(N_vals):
        for mi, m in enumerate(M_vals):
            print(f"\n  [{done+1}/{total}]  cell  n={n}, m={m}", flush=True)
            pe = max(50_000, MAX_ITER // 20)

            tc, it_c  = train(n, m, D, LR, TOL, MAX_ITER, mf=False,
                              seed=ni*1000+mi, print_every=pe)
            tm, it_mf = train(n, m, D, LR, TOL, MAX_ITER, mf=True,
                              seed=ni*1000+mi, print_every=pe)

            te_conv[ni, mi] = tc
            te_mf[ni, mi]   = tm
            done += 1

            elapsed   = time.time() - t0
            remaining = (elapsed / done) * (total - done)
            print(f"  --> [{done:3d}/{total}]  n={n:4d}  m={m:6d}  "
                  f"conv_te={tc:.2e} ({it_c} iters)  "
                  f"mf_te={tm:.2e} ({it_mf} iters)  "
                  f"elapsed={elapsed:.0f}s  ETA={remaining:.0f}s",
                  flush=True)

    print(f"\nAll {total} cells done in {time.time()-t0:.0f}s.")

    np.save('te_conv_v2.npy', te_conv)
    np.save('te_mf_v2.npy',   te_mf)
    np.save('log_M_v2.npy',   log_M)
    np.save('log_N_v2.npy',   log_N)
    print("Arrays saved.")

# ── plot ──────────────────────────────────────────────────────────────────────

print("Plotting ...")
log_tc = np.log10(np.clip(te_conv, 1e-8, None))
log_tm = np.log10(np.clip(te_mf,   1e-8, None))
vmin, vmax = -6.0, -0.6
levels = np.linspace(vmin, vmax, 28)

# Three-column layout: [heatmap | heatmap | colorbar]
fig, (ax1, ax2, cax) = plt.subplots(
    1, 3,
    figsize=(14, 5),
    gridspec_kw={'width_ratios': [1, 1, 0.05], 'wspace': 0.30}
)

for ax, Z, title, dlines in [
        (ax1, log_tc, 'Conventional scaling', True),
        (ax2, log_tm, 'Mean-field scaling',   False)]:
    cf = ax.contourf(log_M, log_N, Z, levels=levels,
                     cmap='hot_r', vmin=vmin, vmax=vmax, extend='both')
    ax.contour(log_M, log_N, Z, levels=levels[::2],
               colors='white', linewidths=0.4, alpha=0.5)
    if dlines:
        # m = n/(d+1)  and  m = n  phase boundaries
        # Note: m=n/(d+1) line has log10(m) = log10(n) - log10(d+1) ≈ log10(n) - 1.32
        # which falls in range [0.48, 1.33] — just left of the plot boundary at 2.0.
        # m=n line has log10(m) = log10(n), visible from (1.8,1.8) to (2.65,2.65).
        ax.plot(log_N - np.log10(D + 1), log_N, 'k--', linewidth=1.2,
                label=r'$m=n/(d{+}1)$')
        ax.plot(log_N, log_N, 'k--', linewidth=1.2, label=r'$m=n$')
    ax.set_xlim(log_M[0], log_M[-1])
    ax.set_ylim(log_N[0], log_N[-1])
    ax.set_xlabel(r'$\log_{10}(m)$')
    ax.set_ylabel(r'$\log_{10}(n)$')
    ax.set_title(title)

# Colorbar in its own dedicated axis
cbar = fig.colorbar(cf, cax=cax,
                    ticks=np.arange(vmin, vmax + 0.1, 0.6))
cbar.set_label(r'$\log_{10}(\mathrm{test\ error})$', labelpad=10)

fig.suptitle(
    r'Figure 4  ($d=20,\ \eta=5\times10^{-4}$, stop at train loss $<10^{-7}$)',
    y=1.01)
plt.savefig('figure4.pdf', bbox_inches='tight')
plt.savefig('figure4.png', dpi=150, bbox_inches='tight')
print("Saved figure4.pdf / figure4.png")
plt.close()
