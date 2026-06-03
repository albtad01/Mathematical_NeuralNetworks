"""
Replication of Figure 6 from E, Ma, Wojtowytsch, Wu (2020).

Left:  Test error (blue) and smallest eigenvalue of the Gram matrix (red)
       vs number of random features m, for n=500 MNIST samples.
       Model: min-norm RF solution  f_m(x) = ReLU(x B^T) A,  B fixed random.

Right: Test error vs m for GD solutions stopped at iterations T.
       GD starts from A_0=0; finite T acts as early-stopping regularisation.
       Computed exactly via SVD spectral filter — no loop over T steps needed.

Key equation (GD from 0, full-batch, step η, multiclass 10 outputs):
  A_T  = V diag(filter_T(σ)) U^T Y
  filter_T(σ) = [1 − (1 − η σ²)^T] / σ
  → T → ∞: A_T → min-norm solution (pseudoinverse)
  → small T: suppresses small singular values (early-stop regularisation)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# ── MNIST loading ──────────────────────────────────────────────────────────────

try:
    from torchvision import datasets, transforms
    _ds_tr = datasets.MNIST('./data', train=True,  download=True,
                             transform=transforms.ToTensor())
    _ds_te = datasets.MNIST('./data', train=False, download=True,
                             transform=transforms.ToTensor())
    X_all    = _ds_tr.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_all    = _ds_tr.targets.numpy()
    X_te_raw = _ds_te.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
    y_te_all = _ds_te.targets.numpy()
    print("MNIST loaded via torchvision.")
except Exception as e:
    raise RuntimeError(
        "Could not load MNIST. Install torchvision or place mnist npz in ./data.\n"
        f"Original error: {e}")

# Global normalisation (standard MNIST values)
mu, sigma = X_all.mean(), X_all.std()
X_all    = (X_all    - mu) / sigma
X_te_raw = (X_te_raw - mu) / sigma

# ── experiment parameters ─────────────────────────────────────────────────────

N      = 500        # training samples
N_TE   = 3000       # test samples (subset of 10k for speed)
D      = 784        # input dimension (28×28)
SEED   = 0          # master seed

T_VALUES = [10_000, 100_000, 1_000_000, 100_000_000, 10**12]

# Log-spaced widths from ~5 to ~20 000; denser near m=n=500
_main = np.round(np.logspace(0.7, 4.3, 28)).astype(int)
_fine = np.arange(350, 700, 30)                   # extra points near m=n
WIDTHS = np.unique(np.concatenate([_main, _fine]))

print(f"n={N}, d={D}, n_te={N_TE}")
print(f"T values: {[f'{T:.0e}' for T in T_VALUES]}")
print(f"Widths ({len(WIDTHS)}): {WIDTHS[0]} … {WIDTHS[-1]}")

# ── data selection ────────────────────────────────────────────────────────────

rng0 = np.random.default_rng(SEED)

tr_idx = rng0.choice(len(X_all),    N,    replace=False)
te_idx = rng0.choice(len(X_te_raw), N_TE, replace=False)

X_tr = X_all[tr_idx]          # (N,   D)
y_tr = y_all[tr_idx]
X_te = X_te_raw[te_idx]       # (N_TE, D)
y_te = y_te_all[te_idx]

# One-hot training labels
Y_tr = np.zeros((N, 10), dtype=np.float32)
Y_tr[np.arange(N), y_tr] = 1.0

# ── helpers ───────────────────────────────────────────────────────────────────

def relu_features(X, B):
    """Compute H = ReLU(X B^T).  X: (n,d), B: (m,d) → H: (n,m)."""
    return np.maximum(X @ B.T, 0.0)

def classify(pred):
    return np.argmax(pred, axis=1)

def test_error(pred, labels):
    return np.mean(classify(pred) != labels)

# ── main experiment loop ──────────────────────────────────────────────────────

mn_te   = []                        # min-norm test error
lam_min = []                        # smallest eigenvalue of Gram matrix
gd_te   = {T: [] for T in T_VALUES} # GD test error per T
gd_pn   = {T: [] for T in T_VALUES} # GD path norm per T

t0 = time.time()

for wi, m in enumerate(WIDTHS):
    print(f"\n[{wi+1}/{len(WIDTHS)}]  m={m}", flush=True)

    rng = np.random.default_rng(SEED + 1 + wi)
    B = rng.normal(0, 1.0 / D**0.5, (m, D)).astype(np.float32)
    b_norms = np.linalg.norm(B, axis=1)   # (m,)  ||b_j||

    H_tr = relu_features(X_tr, B)   # (N, m)
    H_te = relu_features(X_te, B)   # (N_TE, m)

    # ── min-norm solution via lstsq (SVD-based, handles both m<n and m>n) ────
    A_mn, _, _, _ = np.linalg.lstsq(H_tr, Y_tr, rcond=None)  # (m, 10)
    te_mn = test_error(H_te @ A_mn, y_te)
    mn_te.append(te_mn)

    # ── smallest eigenvalue of Gram matrix K = H_tr H_tr^T ───────────────────
    K = H_tr @ H_tr.T                       # (N, N)
    eigs = np.linalg.eigvalsh(K)            # sorted ascending
    pos = eigs[eigs > eigs[-1] * 1e-8]     # ignore numerical zeros
    lam_min.append(float(pos[0]) if len(pos) > 0 else np.nan)

    # ── GD spectral filter via thin SVD of H_tr ───────────────────────────────
    # H_tr = U diag(s) Vt  (shapes: U (N,r), s (r,), Vt (r,m), r=min(N,m))
    U, s, Vt = np.linalg.svd(H_tr, full_matrices=False)
    UY = U.T @ Y_tr          # (r, 10)
    H_te_VtT = H_te @ Vt.T  # (N_TE, r)  ← precompute; used for all T

    eta = 1.0 / (s[0]**2 + 1e-12)   # safe step: 1 / σ_max^2

    for T in T_VALUES:
        # filter_T(σ) = [1 − (1 − η σ²)^T] / σ,  computed in log-space
        log_base = np.log(np.clip(1.0 - eta * s**2, 1e-300, 1.0 - 1e-15))
        factor   = np.exp(np.float64(T) * log_base)       # (1-ησ²)^T
        filt     = (1.0 - factor) / s                     # (r,)
        # pred = H_te Vt^T diag(filt) UY = H_te_VtT @ (filt * UY)
        coeff_T = filt[:, None] * UY                      # (r, 10)
        pred_T  = H_te_VtT @ coeff_T                      # (N_TE, 10)
        gd_te[T].append(test_error(pred_T, y_te))

        # path norm: Σ_j ||A_T[j,:]||_2 * ||b_j||_2
        A_T    = Vt.T @ coeff_T                           # (m, 10)
        a_norms = np.linalg.norm(A_T, axis=1)             # (m,)
        gd_pn[T].append(float(np.sum(a_norms * b_norms)))

    elapsed = time.time() - t0
    eta_sec = elapsed / (wi+1) * (len(WIDTHS) - wi - 1)
    print(f"  mn_te={te_mn:.4f}  λ_min={lam_min[-1]:.2e}  "
          f"gd_te(T=1e12)={gd_te[T_VALUES[-1]][-1]:.4f}  "
          f"elapsed={elapsed:.0f}s  ETA={eta_sec:.0f}s", flush=True)

print(f"\nAll done in {time.time()-t0:.0f}s")

# ── plot ───────────────────────────────────────────────────────────────────────

mn_te   = np.array(mn_te)
lam_min = np.array(lam_min, dtype=float)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5))

# ── Left plot: min-norm test error + smallest eigenvalue ──────────────────────
ax1r = ax1.twinx()

ax1.loglog(WIDTHS, mn_te, color='blue', marker='o', markersize=4,
           linewidth=1.6, label='Min-norm Sol.', zorder=4)
ax1.axvline(N, color='k', linewidth=1.2, zorder=5)

# Gram eigenvalue (only plot where positive and finite)
valid = np.isfinite(lam_min) & (lam_min > 0)
ax1r.loglog(WIDTHS[valid], lam_min[valid], color='red', marker='*',
            markersize=5, linewidth=1.4, zorder=3)

# Fix the blue y-axis range to the test-error data (no spike artefact)
te_min = np.nanmin(mn_te)
te_max = np.nanmax(mn_te)
ax1.set_ylim(te_min * 0.7, te_max * 2.0)

# Fix red y-axis range to eigenvalue data
lv = lam_min[valid]
ax1r.set_ylim(lv.min() * 0.3, lv.max() * 3.0)

ax1.set_xlabel('Number of features: m')
ax1.set_ylabel('Test error', color='blue', fontsize=11)
ax1.tick_params(axis='y', labelcolor='blue')
ax1r.set_ylabel('Smallest eigenvalue', color='red', fontsize=11)
ax1r.tick_params(axis='y', labelcolor='red')
ax1.set_title(f'MNIST, n={N}')

# Legend in upper-right; include both series
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color='blue',  marker='o', markersize=4,
           linewidth=1.6, label='Min-norm Sol.'),
    Line2D([0], [0], color='red', marker='*', markersize=5,
           linewidth=1.4, label='Smallest eigenvalue'),
]
ax1.legend(handles=handles, loc='upper right', fontsize=8,
           framealpha=0.9)

# Light gridlines on the blue (left) axis only
ax1.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5,
         color='gray', alpha=0.4)

# "m = n" annotation above the vertical line, using axes-fraction y
ax1.annotate('m = n', xy=(N, 1), xycoords=('data', 'axes fraction'),
             xytext=(N * 1.6, 0.97), textcoords=('data', 'axes fraction'),
             fontsize=8, va='top', ha='left',
             arrowprops=dict(arrowstyle='-', color='black', lw=0.8))

# ── Right plot: GD test error at different T ──────────────────────────────────
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']

for (T, col, mk) in zip(T_VALUES, colors, markers):
    ax2.loglog(WIDTHS, gd_te[T], color=col, marker=mk, markersize=4,
               linewidth=1.5, label=f'T={T:.0e}')

ax2.axvline(N, color='k', linewidth=1.2)
ax2.annotate('m = n', xy=(N, 1), xycoords=('data', 'axes fraction'),
             xytext=(N * 1.6, 0.97), textcoords=('data', 'axes fraction'),
             fontsize=8, va='top', ha='left',
             arrowprops=dict(arrowstyle='-', color='black', lw=0.8))
ax2.set_xlabel('Number of features')
ax2.set_ylabel('Test error')
ax2.legend(fontsize=9)
ax2.set_title(f'MNIST, n={N}')
ax2.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.4)

# ── Third plot: path norm for different T ─────────────────────────────────────
for (T, col, mk) in zip(T_VALUES, colors, markers):
    ax3.semilogx(WIDTHS, gd_pn[T], color=col, marker=mk, markersize=4,
                 linewidth=1.5, label=f'T={T:.0e}')

ax3.axvline(N, color='k', linewidth=1.2)
ax3.annotate('m = n', xy=(N, 1), xycoords=('data', 'axes fraction'),
             xytext=(N * 1.6, 0.97), textcoords=('data', 'axes fraction'),
             fontsize=8, va='top', ha='left',
             arrowprops=dict(arrowstyle='-', color='black', lw=0.8))
ax3.set_xlabel('Number of features')
ax3.set_ylabel(r'Path norm  $\sum_j \|A_j\|\,\|b_j\|$')
ax3.legend(fontsize=9)
ax3.set_title(f'MNIST, n={N}')
ax3.grid(True, which='both', linestyle=':', linewidth=0.5, color='gray', alpha=0.4)


plt.tight_layout()
plt.savefig('figure6.pdf', bbox_inches='tight')
plt.savefig('figure6.png', dpi=150, bbox_inches='tight')
print("Saved figure6.pdf / figure6.png")
plt.close()
