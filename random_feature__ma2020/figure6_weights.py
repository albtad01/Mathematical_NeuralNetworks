"""
Neuron-weight profile at the interpolation threshold (m = n = 500) for the
RF model on MNIST.

For each GD iteration count T (and the min-norm limit), we plot
  a_j * ||b_j||   vs   neuron index j
for the digit-0 vs rest binary sub-problem (class 0 column of A).

This shows how early stopping (small T) keeps coefficients small and smooth,
while large T (→ min-norm) produces large, noisy coefficients that overfit the
ill-conditioned directions near the interpolation threshold.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── MNIST loading (same preprocessing as figure6.py) ─────────────────────────

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

# ── settings — identical seed / split as figure6.py ──────────────────────────

N    = 500      # training samples  (= m  →  interpolation threshold)
M    = 500      # neurons
D    = 784
SEED = 0
N_TE = 3000

T_VALUES = [10_000, 100_000, 1_000_000, 100_000_000, 10**12]
CLASS    = 0    # digit whose binary sub-problem we inspect

# ── data ──────────────────────────────────────────────────────────────────────

rng0   = np.random.default_rng(SEED)
tr_idx = rng0.choice(len(X_all),    N,    replace=False)
te_idx = rng0.choice(len(X_te_raw), N_TE, replace=False)

X_tr = X_all[tr_idx]
y_tr = y_all[tr_idx]
X_te = X_te_raw[te_idx]
y_te = y_te_all[te_idx]

# One-hot labels (N × 10); we extract column CLASS below
Y_tr = np.zeros((N, 10), dtype=np.float32)
Y_tr[np.arange(N), y_tr] = 1.0
y_cls = Y_tr[:, CLASS]   # (N,) binary labels for the chosen digit

# ── random features ───────────────────────────────────────────────────────────

rng = np.random.default_rng(SEED + 1)          # same offset as figure6 wi=0
B   = rng.normal(0, 1.0 / D**0.5, (M, D)).astype(np.float32)  # (M, D)

H_tr = np.maximum(X_tr @ B.T, 0.0)   # (N, M)
H_te = np.maximum(X_te @ B.T, 0.0)   # (N_TE, M)

b_norms = np.linalg.norm(B, axis=1)  # ||b_j||  for j=0..M-1

# ── SVD + spectral filter ─────────────────────────────────────────────────────

U, s, Vt = np.linalg.svd(H_tr, full_matrices=False)  # r = min(N,M) = N = M = 500
Uy        = U.T @ y_cls            # (r,)
H_te_VtT  = H_te @ Vt.T           # (N_TE, r)  — reused across all T

eta = 1.0 / (s[0]**2 + 1e-12)

def gd_solution(T):
    """Return A_T (length-M weight vector) for the chosen class via spectral filter."""
    log_base = np.log(np.clip(1.0 - eta * s**2, 1e-300, 1.0 - 1e-15))
    factor   = np.exp(np.float64(T) * log_base)
    filt     = (1.0 - factor) / s          # (r,)
    return Vt.T @ (filt * Uy)             # (M,)  weight vector

def min_norm_solution():
    """Min-norm via lstsq (= limit T → ∞)."""
    a, _, _, _ = np.linalg.lstsq(H_tr, y_cls, rcond=None)
    return a

def test_error(a):
    pred = H_te @ a
    return np.mean((pred > 0.5) != (y_te == CLASS))

def path_norm(a):
    return float(np.sum(np.abs(a) * b_norms))

# ── compute all solutions ─────────────────────────────────────────────────────

solutions = {}
for T in T_VALUES:
    solutions[T] = gd_solution(T)
solutions['min-norm'] = min_norm_solution()

# ── plot ──────────────────────────────────────────────────────────────────────

# Colors: light blue for small T, dark blue for large T, red for min-norm
colors = ['#a8d8f0', '#5baee3', '#2176c7', '#0d4d91', '#062a4e']
labels = T_VALUES + ['min-norm']
idx    = np.arange(M)

fig, ax = plt.subplots(figsize=(12, 5))

# Plot T values from largest to smallest so small-T (light) sits on top
for T, col in zip(reversed(T_VALUES), reversed(colors)):
    a    = solutions[T]
    coef = a * b_norms
    te   = test_error(a)
    pn   = path_norm(a)
    ax.scatter(idx, coef, s=2, color=col, alpha=0.7, zorder=2,
               label=f'T={T:.0e}  (te={te:.3f}, pn={pn:.1f})')

# Min-norm on top in red
a_mn  = solutions['min-norm']
coef_mn = a_mn * b_norms
te_mn   = test_error(a_mn)
pn_mn   = path_norm(a_mn)
ax.scatter(idx, coef_mn, s=2, color='firebrick', alpha=0.8, zorder=3,
           label=f'Min-norm  (te={te_mn:.3f}, pn={pn_mn:.1f})')

ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
ax.set_xlabel('Index of neurons')
ax.set_ylabel(r'$a_j \|\mathbf{b}_j\|$')
ax.set_title(
    rf'Neuron weights at interpolation threshold  ($m = n = {N}$,  MNIST,  digit {CLASS} vs rest)')
ax.legend(fontsize=8, markerscale=4, loc='upper right')

plt.tight_layout()
plt.savefig('figure6_weights.pdf', bbox_inches='tight')
plt.savefig('figure6_weights.png', dpi=150, bbox_inches='tight')
print("Saved figure6_weights.pdf / figure6_weights.png")
plt.close()
