"""
Replication of Figure 5 from E, Ma, Wojtowytsch, Wu (2020).

Two-panel plot:
  Left  : log₁₀(test error) vs log₁₀(width m)  — nn, rf (analytical), rf (GD).
  Right : Path norm          vs log₁₀(width m)  — nn model only.

Setup (Figure 5 caption)
-------------------------
  n = 200,  d = 20,  learning rate η = 0.001.
  GD stopped when training error < 10⁻⁸ (10⁻⁷ for float32 on GPU).
  Dashed lines at  m = n/(d+1)  (left)  and  m = n  (right).
  Mean ± std over N_SEEDS independent runs shown.

Models
------
  nn      : two-layer ReLU network  f(x) = Σⱼ aⱼ σ(bⱼᵀ x)
            trained with full-batch GD on (a, B) jointly.
  rf_ana  : random features — bⱼ fixed at init, aⱼ solved analytically
            (minimum-norm least squares via SVD).  ← paper's rf
  rf_gd   : random features — bⱼ fixed at init, aⱼ trained with GD.
            Converges to same solution as rf_ana (convex linear problem).

Dashed lines
------------
  m = n/(d+1) ≈ 9.5  : nn interpolation threshold  (m·(d+1) = n parameters = data)
  m = n       = 200   : rf interpolation threshold  (m free params in a = n data)

Path norm :  Σⱼ |aⱼ| · ‖bⱼ‖

Data
----
  Teacher network (m* = 5 hidden units, fixed random weights).
  x ~ N(0, Iₐ/d)  so  ‖x‖ ≈ 1 almost surely.
  Output normalised to std(y_train) = 1.
  n = 200 train,  N_TE = 2 000 test.

Device priority: MPS (Apple) → CUDA → CPU.
  MPS / CUDA : float32,  EFF_TOL = 1e-7
  CPU        : float64,  EFF_TOL = 1e-8  (paper value)

Why MAX_ITERS = 2_000_000
--------------------------
  The paper runs GD until convergence (no iteration cap).  With lr = 0.001,
  reaching training error 10⁻⁸ from ~0.5 requires exp(-lr·λ_min·t) < 2×10⁻⁸,
  i.e. t > 16/(0.001·λ_min).  For small m (m ≈ 5, poorly conditioned NTK),
  λ_min can be O(0.01), requiring ~1.6M steps.  With n = 200 (tiny), each step
  takes microseconds on GPU, so 2M iterations per run is fast in absolute time.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time
import os

# ── device & dtype ─────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')

if DEVICE.type == 'cpu':
    DTYPE   = torch.float64
    EFF_TOL = 1e-8      # achievable with float64
else:
    DTYPE   = torch.float32
    EFF_TOL = 1e-7      # float32 precision floor
print(f"Device: {DEVICE}  dtype: {DTYPE}  effective_tol: {EFF_TOL:.0e}")

# ── parameters ─────────────────────────────────────────────────────────────────
N         = 200          # training samples
D         = 20           # input dimension
N_TE      = 2000         # test samples
LR        = 0.001        # GD learning rate (paper: 0.001)
TRAIN_TOL = 1e-8         # paper stopping criterion (see EFF_TOL above)
MAX_ITERS = 2_000_000    # large cap — paper runs until convergence
N_SEEDS   = 6            # independent runs for mean / std
DATA_SEED = 0

# Stagnation detection: break if loss hasn't dropped by STAG_FRAC over
# STAG_CHECK consecutive iterations (handles m < m* where 10⁻⁸ is unreachable).
STAG_CHECK = 200_000     # check every N iters
STAG_FRAC  = 0.01        # need at least 1% improvement to continue

# Width sweep: log₁₀(m) from ~0.3 to ~3.6  (m ≈ 2 … 4000)
M_VALS = np.unique(
    np.round(np.logspace(np.log10(2), np.log10(4000), 22)).astype(int)
)
M_VALS = M_VALS[M_VALS >= 1]

# Threshold comments (see module docstring)
M_LEFT  = N / (D + 1)   # ≈  9.5  (nn interpolation threshold: m·(d+1) = n)
M_RIGHT = float(N)       # = 200   (rf interpolation threshold: m = n)

CACHE_FILE = 'figure5_results_v2.npz'

# ── data (numpy → torch) ───────────────────────────────────────────────────────
rng_d  = np.random.default_rng(DATA_SEED)

# x ~ N(0, I/d)  so  ‖x‖ ≈ 1  (each component has std 1/√d)
X_np   = rng_d.normal(0, 1.0 / np.sqrt(D), (N + N_TE, D)).astype(np.float64)

# Teacher: m*=5, bⱼ* ~ N(0, I/d),  aⱼ* ~ N(0,1)
B_t    = rng_d.normal(0, 1.0 / np.sqrt(D), (5, D)).astype(np.float64)
a_t    = rng_d.normal(0, 1.0, 5).astype(np.float64)
y_np   = np.maximum(X_np @ B_t.T, 0.0) @ a_t   # (N+N_TE,)

# Normalise: std(y_train) = 1
y_np  /= y_np[:N].std()

def to_dev(arr):
    return torch.tensor(arr, dtype=DTYPE, device=DEVICE)

X_tr = to_dev(X_np[:N]);   y_tr = to_dev(y_np[:N])
X_te = to_dev(X_np[N:]);   y_te = to_dev(y_np[N:])
print(f"y_tr  mean={y_tr.mean().item():.3f}  std={y_tr.std().item():.3f}")

# ── helper: reproducible random tensors ───────────────────────────────────────
def rand_tensor(shape, std, seed):
    """Generate a CPU random tensor then move to DEVICE (MPS-safe)."""
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(*shape, dtype=DTYPE, generator=g).to(DEVICE) * std

# ── random-features: analytical minimum-norm LS (paper's rf) ─────────────────
def solve_rf(m: int, seed: int):
    """
    Minimum-norm least-squares  a* = H† y  via thin SVD.
    This is the rf model from the paper: B fixed, a solved analytically.
    """
    B  = rand_tensor((m, D), 1.0 / D**0.5, seed)          # (m, D)
    H  = (X_tr @ B.T).clamp(min=0)                        # (N, m)

    # Thin SVD: H = U S Vhᵀ  →  H† = Vh diag(1/S) Uᵀ
    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    thresh    = S[0] * 1e-10 if S.numel() > 0 else torch.tensor(1e-30)
    S_pinv    = torch.where(S > thresh, 1.0 / S, torch.zeros_like(S))
    a         = Vh.T @ (S_pinv * (U.T @ y_tr))            # (m,)  min-norm solution

    H_te     = (X_te @ B.T).clamp(min=0)
    test_err = 0.5 * ((H_te @ a - y_te) ** 2).mean().item()
    pnorm    = (a.abs() * B.norm(dim=1)).sum().item()
    return test_err, pnorm

# ── random-features: GD on a only, B fixed ────────────────────────────────────
def gd_rf(m: int, lr: float, tol: float, max_iters: int, seed: int):
    """
    Train rf with GD: B fixed at init, gradient descent on a only.
    Problem is convex (linear regression), so GD converges to global optimum.
    For m ≤ n (overdetermined): same unique LS solution as rf_analytical.
    For m > n (underdetermined): converges to a solution that depends on init
    (not necessarily minimum-norm, unlike rf_analytical from a=0).
    """
    B    = rand_tensor((m, D), 1.0 / D**0.5, seed)
    H_tr = (X_tr @ B.T).clamp(min=0)                      # (N, m) — fixed
    a    = rand_tensor((m,), 1.0 / m**0.5, seed + 999983) # same init as nn

    train_err  = float('inf')
    prev_check = float('inf')
    for it in range(max_iters):
        r         = H_tr @ a - y_tr
        train_err = 0.5 * (r ** 2).mean().item()
        if train_err < tol:
            break

        # Stagnation: for rf this shouldn't trigger (convex), but just in case
        if it > 0 and it % STAG_CHECK == 0:
            if train_err > prev_check * (1.0 - STAG_FRAC):
                break
            prev_check = train_err

        grad_a = H_tr.T @ r / N
        a      = a - lr * grad_a

    H_te     = (X_te @ B.T).clamp(min=0)
    test_err = 0.5 * ((H_te @ a - y_te) ** 2).mean().item()
    pnorm    = (a.abs() * B.norm(dim=1)).sum().item()
    return test_err, pnorm, train_err

# ── neural-network GD (manual, no autograd) ────────────────────────────────────
def gd_nn(m: int, lr: float, tol: float, max_iters: int, seed: int):
    """
    Full-batch GD on  f(x) = Σⱼ aⱼ σ(bⱼᵀ x).
    Gradients computed manually — avoids autograd overhead, GPU-friendly.
    Returns (test_error, path_norm, final_train_error).

    Stagnation break: for m < m* = 5, the teacher can't be exactly represented
    so training error > 0 at any local/global minimum. We detect this and stop
    early rather than running all MAX_ITERS steps.
    """
    B = rand_tensor((m, D), 1.0 / D**0.5, seed)           # (m, D)
    a = rand_tensor((m,),   1.0 / m**0.5, seed + 999983)  # (m,)

    train_err  = float('inf')
    prev_check = float('inf')

    for it in range(max_iters):
        Z  = X_tr @ B.T              # (N, m)  pre-activations
        H  = Z.clamp(min=0)          # (N, m)  ReLU activations
        r  = H @ a - y_tr            # (N,)    residuals

        train_err = 0.5 * (r ** 2).mean().item()
        if train_err < tol:
            break

        # Stagnation check: break if not making meaningful progress
        if it > 0 and it % STAG_CHECK == 0:
            if train_err > prev_check * (1.0 - STAG_FRAC):
                break   # stuck at local/global minimum, accept current solution
            prev_check = train_err

        # ∂L/∂a  =  Hᵀ r / N
        grad_a = H.T @ r / N                             # (m,)

        # ∂L/∂bⱼ = (aⱼ/N) Σᵢ rᵢ 𝟙[zᵢⱼ>0] xᵢ
        ind    = (Z > 0).to(DTYPE)                       # (N, m)  indicator
        grad_B = (r.unsqueeze(1) * ind).T @ X_tr         # (m, D)
        grad_B.mul_(a.unsqueeze(1) / N)

        a = a - lr * grad_a
        B = B - lr * grad_B

    H_te     = (X_te @ B.T).clamp(min=0)
    test_err = 0.5 * ((H_te @ a - y_te) ** 2).mean().item()
    pnorm    = (a.abs() * B.norm(dim=1)).sum().item()
    return test_err, pnorm, train_err

# ── sweep over widths ──────────────────────────────────────────────────────────
if os.path.isfile(CACHE_FILE):
    print(f"\nLoading cache from {CACHE_FILE} …")
    c = np.load(CACHE_FILE)
    M_VALS        = c['M_VALS']
    nn_te_mean    = c['nn_te_mean'];    nn_te_std    = c['nn_te_std']
    nn_pn_mean    = c['nn_pn_mean'];    nn_pn_std    = c['nn_pn_std']
    rf_te_mean    = c['rf_te_mean'];    rf_te_std    = c['rf_te_std']
    rfgd_te_mean  = c['rfgd_te_mean'];  rfgd_te_std  = c['rfgd_te_std']
    print(f"  Loaded {len(M_VALS)} width values.")
else:
    n_m         = len(M_VALS)
    nn_te_all   = np.full((n_m, N_SEEDS), np.nan)
    nn_pn_all   = np.full((n_m, N_SEEDS), np.nan)
    rf_te_all   = np.full((n_m, N_SEEDS), np.nan)
    rfgd_te_all = np.full((n_m, N_SEEDS), np.nan)

    t0 = time.time()
    for i, m in enumerate(M_VALS):
        for s in range(N_SEEDS):
            seed = s * 10007 + i * 37 + 1

            te_rf, _                  = solve_rf(m, seed)
            rf_te_all[i, s]           = te_rf

            te_rfgd, _, _             = gd_rf(m, LR, EFF_TOL, MAX_ITERS, seed)
            rfgd_te_all[i, s]         = te_rfgd

            te_nn, pn_nn, tr_err      = gd_nn(m, LR, EFF_TOL, MAX_ITERS, seed)
            nn_te_all[i, s]           = te_nn
            nn_pn_all[i, s]           = pn_nn

        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (n_m - i - 1)
        print(f"  m={m:5d} | "
              f"nn_te={nn_te_all[i].mean():.2e}  "
              f"rf_te={rf_te_all[i].mean():.2e}  "
              f"rfgd_te={rfgd_te_all[i].mean():.2e}  "
              f"pn={nn_pn_all[i].mean():.2f} | "
              f"{elapsed:.0f}s  ETA {eta:.0f}s")

    nn_te_mean   = np.nanmean(nn_te_all,   1);  nn_te_std   = np.nanstd(nn_te_all,   1)
    nn_pn_mean   = np.nanmean(nn_pn_all,   1);  nn_pn_std   = np.nanstd(nn_pn_all,   1)
    rf_te_mean   = np.nanmean(rf_te_all,   1);  rf_te_std   = np.nanstd(rf_te_all,   1)
    rfgd_te_mean = np.nanmean(rfgd_te_all, 1);  rfgd_te_std = np.nanstd(rfgd_te_all, 1)

    np.savez(CACHE_FILE,
             M_VALS=M_VALS,
             nn_te_mean=nn_te_mean,   nn_te_std=nn_te_std,
             nn_pn_mean=nn_pn_mean,   nn_pn_std=nn_pn_std,
             rf_te_mean=rf_te_mean,   rf_te_std=rf_te_std,
             rfgd_te_mean=rfgd_te_mean, rfgd_te_std=rfgd_te_std)
    print(f"Saved {CACHE_FILE}  (total {time.time()-t0:.0f}s)")

# ── plot ────────────────────────────────────────────────────────────────────────
log10_m = np.log10(M_VALS.astype(float))
BLUE    = '#4878cf'
ORANGE  = '#d65f5f'
GREEN   = '#3a9a3a'
BG      = '#e8ecf4'

def shaded_log(ax, x, mean, std, color, label):
    """Plot mean line + ±std shading, both in log₁₀ space."""
    eps   = 1e-12
    lm    = np.log10(np.clip(mean, eps, None))
    lup   = np.log10(np.clip(mean + std, eps, None))
    ldown = np.log10(np.clip(np.maximum(mean - std, eps), eps, None))
    ax.plot(x, lm, color=color, linewidth=1.8, label=label)
    ax.fill_between(x, ldown, lup, color=color, alpha=0.20)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor('white')

# ── Left: test error ───────────────────────────────────────────────────────────
shaded_log(ax1, log10_m, nn_te_mean,   nn_te_std,   BLUE,   'nn')
shaded_log(ax1, log10_m, rf_te_mean,   rf_te_std,   ORANGE, 'rf (analytical)')
shaded_log(ax1, log10_m, rfgd_te_mean, rfgd_te_std, GREEN,  'rf (GD)')
ax1.axvline(np.log10(M_LEFT),  color='black', linewidth=1.3, linestyle='--',
            label=r'$m=n/(d{+}1)$')
ax1.axvline(np.log10(M_RIGHT), color='gray',  linewidth=1.3, linestyle='--',
            label=r'$m=n$')
ax1.set_xlabel(r'$\log_{10}(\mathrm{width})$', fontsize=11)
ax1.set_ylabel(r'$\log_{10}(\mathrm{test\ error})$', fontsize=11)
ax1.legend(fontsize=9)
ax1.set_facecolor(BG)
ax1.grid(True, linestyle='-', linewidth=0.5, alpha=0.6, color='white')
for sp in ax1.spines.values():
    sp.set_linewidth(0)

# ── Right: path norm ───────────────────────────────────────────────────────────
ax2.plot(log10_m, nn_pn_mean, color=BLUE, linewidth=1.8)
ax2.fill_between(log10_m,
                 np.maximum(nn_pn_mean - nn_pn_std, 0),
                 nn_pn_mean + nn_pn_std,
                 color=BLUE, alpha=0.20)
ax2.axvline(np.log10(M_LEFT),  color='black', linewidth=1.3, linestyle='--')
ax2.axvline(np.log10(M_RIGHT), color='gray',  linewidth=1.3, linestyle='--')
ax2.set_xlabel(r'$\log_{10}(\mathrm{width})$', fontsize=11)
ax2.set_ylabel('Path norm', fontsize=11)
ax2.set_facecolor(BG)
ax2.grid(True, linestyle='-', linewidth=0.5, alpha=0.6, color='white')
for sp in ax2.spines.values():
    sp.set_linewidth(0)

plt.tight_layout()
plt.savefig('figure5.pdf', bbox_inches='tight', facecolor='white')
plt.savefig('figure5.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved figure5.pdf / figure5.png")
plt.close()
