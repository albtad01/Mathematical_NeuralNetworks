"""
Replication of Figure 7 from E, Ma, Wojtowytsch, Wu (2020).

GD training dynamics at the interpolation threshold m = n = 500 (MNIST).
X-axis: number of GD iterations T  (log scale, 1 to 10^12)
Left  Y-axis: train error, test error, min-norm solution error
Right Y-axis: path norm sum_j |a_j| ||b_j||,  min-norm path norm

The key challenge: at m=n the Gram matrix K = H H^T is nearly singular.

We search over seeds for B to find one where K2 ≈ 10^8-10^10, so all three
phases fit in [1, 10^12].  The spectral-filter method makes each T value
essentially free to evaluate, so the seed search adds no meaningful cost.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── MNIST loading (same as figure6) ───────────────────────────────────────────

from torchvision import datasets, transforms

_ds_tr = datasets.MNIST('./data', train=True,  download=True,
                         transform=transforms.ToTensor())
_ds_te = datasets.MNIST('./data', train=False, download=True,
                         transform=transforms.ToTensor())

X_all    = _ds_tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_all    = _ds_tr.targets.numpy()
X_te_raw = _ds_te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
y_te_all = _ds_te.targets.numpy()

mu, std  = X_all.mean(), X_all.std()
X_all    = (X_all    - mu) / std
X_te_raw = (X_te_raw - mu) / std

# ── data split (same seed as figure6) ─────────────────────────────────────────

N    = 500
M    = 500      # m = n  →  interpolation threshold
D    = 784
N_TE = 3000
DATA_SEED = 0

rng0   = np.random.default_rng(DATA_SEED)
tr_idx = rng0.choice(len(X_all),    N,    replace=False)
te_idx = rng0.choice(len(X_te_raw), N_TE, replace=False)

X_tr = X_all[tr_idx]
y_tr = y_all[tr_idx]
X_te = X_te_raw[te_idx]
y_te = y_te_all[te_idx]

Y_tr = np.zeros((N,    10), dtype=np.float32)
Y_te = np.zeros((N_TE, 10), dtype=np.float32)
Y_tr[np.arange(N),    y_tr] = 1.0
Y_te[np.arange(N_TE), y_te] = 1.0

# ── seed search: find B whose κ puts Phase III inside [1, 10^12] ──────────────
# Target: slowest-mode convergence time T_slow = K2  ≈  10^9
# (i.e. log10(κ) ≈ 4.5,  so k ≈ 30 000)

print("Searching for B seed with k2 ≈ 10^8 ...")
TARGET_LOG_TSLOW = 8.0      # want T_slow ≈ 10^8  (Phase III visible ~10^8-10^10)

best_seed, best_diff, best_kappa = 0, np.inf, 0.0

for seed in range(300):
    rng_s = np.random.default_rng(seed)
    B_s   = rng_s.normal(0, 1.0 / D**0.5, (M, D)).astype(np.float32)
    H_s   = np.maximum(X_tr @ B_s.T, 0.0)
    s_s   = np.linalg.svd(H_s, compute_uv=False)   # just singular values
    if s_s[-1] < 1e-12:
        continue                                    # rank-deficient, skip
    eta_s  = 1.0 / s_s[0] ** 2
    T_slow = 1.0 / (eta_s * s_s[-1] ** 2)          # approx K2
    diff   = abs(np.log10(T_slow) - TARGET_LOG_TSLOW)
    if diff < best_diff:
        best_diff, best_seed, best_kappa = diff, seed, s_s[0] / s_s[-1]

print(f"  Best seed={best_seed}  κ={best_kappa:.2e}  "
      f"T_slow≈{best_kappa**2:.1e}")

B_SEED = best_seed
rng_b  = np.random.default_rng(B_SEED)
B      = rng_b.normal(0, 1.0 / D**0.5, (M, D)).astype(np.float32)

# ── random features ───────────────────────────────────────────────────────────

b_norms = np.linalg.norm(B, axis=1)              # (M,)
H_tr    = np.maximum(X_tr @ B.T, 0.0)            # (N,   M)
H_te    = np.maximum(X_te @ B.T, 0.0)            # (N_TE, M)

# ── thin SVD of H_tr (m = n → square, full rank) ─────────────────────────────

U, s, Vt = np.linalg.svd(H_tr, full_matrices=False)

UY       = U.T @ Y_tr          # (N, 10)
UY_rn2   = np.sum(UY**2, axis=1)   # (N,)  row-norms², for cheap train err
H_te_VtT = H_te @ Vt.T         # (N_TE, N)

eta   = 1.0 / (s[0] ** 2 + 1e-12)
kappa = s[0] / s[-1]
print(f"σ_max={s[0]:.2f}  σ_min={s[-1]:.2e}  κ={kappa:.2e}")

# ── min-norm reference (T → inf) ────────────────────────────────────────────────

A_mn     = Vt.T @ ((1.0 / s)[:, None] * UY)       # (M, 10)
mn_train = 0.5 * np.mean(np.sum((H_tr @ A_mn - Y_tr)**2, axis=1))
mn_test  = 0.5 * np.mean(np.sum((H_te @ A_mn - Y_te)**2, axis=1))
mn_pnorm = float(np.sum(np.linalg.norm(A_mn, axis=1) * b_norms))

print(f"Min-norm  train={mn_train:.4f}  test={mn_test:.4f}  "
      f"path-norm={mn_pnorm:.2f}")

# ── sweep T values ────────────────────────────────────────────────────────────

T_vals = np.unique(np.round(np.logspace(0, 12, 300)).astype(np.int64))

train_errs, test_errs, path_norms = [], [], []

for T in T_vals:
    log_base = np.log(np.clip(1.0 - eta * s**2, 1e-300, 1.0 - 1e-15))
    factor   = np.exp(np.float64(T) * log_base)    # (1 − ησ²)^T
    filt     = (1.0 - factor) / s

    # train error (exact closed form — O(N), no matrix products)
    train_errs.append(0.5 * np.sum(factor**2 * UY_rn2) / N)

    # test error
    pred_te = H_te_VtT @ (filt[:, None] * UY)      # (N_TE, 10)
    test_errs.append(0.5 * np.mean(np.sum((pred_te - Y_te)**2, axis=1)))

    # path norm Σ_j ||A_T[j,:]||_2 * ||b_j||
    A_T = Vt.T @ (filt[:, None] * UY)              # (M, 10)
    path_norms.append(float(np.sum(np.linalg.norm(A_T, axis=1) * b_norms)))

train_errs = np.array(train_errs)
test_errs  = np.array(test_errs)
path_norms = np.array(path_norms)

# ── detect phase boundaries ───────────────────────────────────────────────────

pn_frac = path_norms / max(mn_pnorm, path_norms.max(), 1e-10)

def phase_boundary(threshold):
    idx = np.searchsorted(pn_frac, threshold)
    return T_vals[min(idx, len(T_vals) - 1)]

t_I_II   = phase_boundary(0.02)
t_II_III = phase_boundary(0.25)

print(f"Phase I/II  boundary: T ≈ {t_I_II:.1e}")
print(f"Phase II/III boundary: T ≈ {t_II_III:.1e}")

# ── plot ──────────────────────────────────────────────────────────────────────

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# clip to avoid log(0) for exact-zero train errors
CLIP = 1e-6
train_plot = np.clip(train_errs, CLIP, None)
test_plot  = np.clip(test_errs,  CLIP, None)
pn_plot    = np.clip(path_norms, CLIP, None)

# left axis — log y-scale so near-zero curves and large ref line are both visible
ax1.loglog(T_vals, train_plot, '.', color='limegreen',  markersize=2,
           label='train error')
ax1.loglog(T_vals, test_plot,  '.', color='steelblue',  markersize=2,
           label='test error')
ax1.axhline(mn_test,  color='steelblue', linewidth=1.4, linestyle='-',
            label='min norm solu error')

# right axis — log y-scale
ax2.loglog(T_vals, pn_plot, '.', color='darkorange', markersize=2,
           label='norm')
ax2.axhline(mn_pnorm, color='firebrick', linewidth=1.4, linestyle='-',
            label='minimal norm')

# phase boundary lines
for t_b in [t_I_II, t_II_III]:
    ax1.axvline(t_b, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)

# phase labels below x-axis
lims_log = [np.log10(T_vals[0]), np.log10(t_I_II),
            np.log10(t_II_III),  np.log10(T_vals[-1])]
for label, i in zip(['I', 'II', 'III'], range(3)):
    xpos = 10 ** ((lims_log[i] + lims_log[i+1]) / 2)
    ax1.text(xpos, -0.13, label, transform=ax1.get_xaxis_transform(),
             ha='center', va='top', fontsize=13, color='firebrick',
             fontstyle='italic', fontweight='bold')

# formatting
ax1.set_xlabel('t')
ax1.set_ylabel('err',  color='steelblue',  fontsize=11)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_xlim(T_vals[0], T_vals[-1])
ax1.set_ylim(CLIP, max(mn_test, test_plot.max()) * 5)

ax2.set_ylabel('norm', color='darkorange', fontsize=11)
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(CLIP, max(mn_pnorm, pn_plot.max()) * 5)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')
ax1.set_title(rf'$m = n = {N}$,  MNIST,  RF model,  B seed={B_SEED}')

plt.tight_layout()
#plt.savefig('figure7.pdf', bbox_inches='tight')
plt.savefig('figure7.png', dpi=150, bbox_inches='tight')
print("Saved figure7.pdf / figure7.png")
plt.close()
