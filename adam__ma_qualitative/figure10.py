"""
Replication of Figure 10 from E, Ma, Wojtowytsch, Wu (2020).

Three typical behavior patterns for Adam trajectories.

Setup (from paper caption)
--------------------------
  Network : fully-connected, hidden layers  256 -> 128 -> 64, ReLU activations
  Data    : 2 classes of CIFAR-10 (airplane=0, automobile=1),
            1000 samples per class  (2000 total)
  LR      : nu = 0.001
  Adam    : parameters reparametrised as  alpha = 1 - alpha·nu,  beta = 1 - b·nu
            where  alpha = PyTorch beta2 (second-moment / v decay)
                   beta = PyTorch beta1 (first-moment  / m decay)

Loss function
-------------
  The paper (Ma & Wu 2021, p.3) explicitly states: "The quadratic loss is used
  for all the experiments, including the classification problems."
  MSE with one-hot targets is used, NOT cross-entropy.
  With cross-entropy the network converges smoothly; with MSE the sharper
  curvature of the loss landscape produces the spike / oscillation / divergence
  instabilities even under full-batch Adam.

Batch size
----------
  Full batch (all 2000 samples per step), as stated in the paper.

Regimes
-------
  Spike       a =   1, b = 100  ->  betas = (beta, alpha) = (0.900, 0.999)
  Oscillation a =  10, b =  10  ->  betas = (0.990, 0.990)
  Divergence  a = 100, b =   1  ->  betas = (0.999, 0.900)

Figure layout: 2 rows x 3 columns  (log y-axis throughout)
  Row 1 : full 0 - 1000-iteration loss curve
  Row 2 : zoomed view
            Spike:       iterations 400 - 800
            Oscillation: iterations 800 - 1000
            Divergence:  iterations 800 - 1000
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time

# ── device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

# ── hyperparameters ────────────────────────────────────────────────────────────
LR          = 0.001
N_ITERS     = 1000        # gradient steps  (paper Fig 10 uses 1000)
N_PER_CLASS = 1000        # CIFAR-10 samples per class  (2 classes -> 2000 total)
CLASSES     = [0, 1]      # airplane, automobile
SEED        = 0

# (a, b, key, display-label)
# PyTorch: beta1 = 1 − b·LR  (first moment / m)
#          beta2 = 1 − a·LR  (second moment / v)
REGIMES = [
    (  1, 100, 'spike',       'a=1, b=100'),
    ( 10,  10, 'oscillation', 'a=10, b=10'),
    (100,   1, 'divergence',  'a=100, b=1'),
]

# x-range for the zoomed (row-2) panels
ZOOM = {
    'spike':       (400,  800),
    'oscillation': (800, 1000),
    'divergence':  (600, 800),
}

# ── CIFAR-10 ───────────────────────────────────────────────────────────────────
print("Loading CIFAR-10 …")
_ds = datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.ToTensor())

X_list, y_list = [], []
counts = {c: 0 for c in CLASSES}
for img, label in _ds:
    if label in CLASSES and counts[label] < N_PER_CLASS:
        X_list.append(img.numpy().reshape(-1).astype(np.float32))
        y_list.append(CLASSES.index(label))   # remap to 0 / 1
        counts[label] += 1
    if all(v >= N_PER_CLASS for v in counts.values()):
        break

X_np = np.array(X_list)           # (2000, 3072)
y_np = np.array(y_list, np.int64)

# z-score normalise
mu, sd = X_np.mean(), X_np.std()
X_np   = (X_np - mu) / sd

# shuffle once
rng  = np.random.default_rng(SEED)
perm = rng.permutation(len(X_np))
X_np, y_np = X_np[perm], y_np[perm]

# Use float64 on CPU for all computations.
# Reason: the divergence regime (a=100, b=1) drives the loss past 10^30.
# float32 overflows to Inf at ~3.4e38 after only ~200-400 steps, cutting the
# curve short. float64 (max ~1.8e308) lets all 1000 iterations complete.
# MPS does not support float64, so we force CPU regardless of hardware.
COMPUTE_DEVICE = torch.device('cpu')
DTYPE          = torch.float64

X_t = torch.tensor(X_np, dtype=DTYPE, device=COMPUTE_DEVICE)
y_t = torch.tensor(y_np, device=COMPUTE_DEVICE)

n_classes = len(CLASSES)
y_onehot  = torch.zeros(len(y_np), n_classes, dtype=DTYPE, device=COMPUTE_DEVICE)
y_onehot.scatter_(1, y_t.unsqueeze(1), 1.0)

n = X_t.shape[0]
print(f"Data: {X_t.shape}  dtype=float64  cpu  |  class counts: {counts}")

# ── network ────────────────────────────────────────────────────────────────────
class FCNet(nn.Module):
    """FC:  3072 → 256 → 128 → 64 → 2,  ReLU activations."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3072, 256), nn.ReLU(),
            nn.Linear( 256, 128), nn.ReLU(),
            nn.Linear( 128,  64), nn.ReLU(),
            nn.Linear(  64,   2),
        )
    def forward(self, x):
        return self.net(x)

# Square (MSE) loss — paper: "quadratic loss is used for all experiments"
criterion = nn.MSELoss()

# Shared initialisation — identical for all three regimes (float64)
torch.manual_seed(SEED)
init_state = FCNet().to(dtype=DTYPE).state_dict()

# ── full-batch Adam ─────────────────────────────────────────────────────────────
def run_adam(init_state, X, Y_onehot, lr, beta1, beta2, n_iters, label=''):
    """
    Full-batch Adam with MSE loss for n_iters steps.
    X        : (N, D) input tensor
    Y_onehot : (N, C) one-hot target tensor  ← required for MSE
    Logs loss at every single step.
    Returns np.ndarray of shape (n_iters,).
    """
    model = FCNet().to(dtype=DTYPE, device=COMPUTE_DEVICE)
    model.load_state_dict({k: v.to(dtype=DTYPE, device=COMPUTE_DEVICE)
                           for k, v in init_state.items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                  betas=(beta1, beta2), eps=1e-8)
    losses   = []
    t0       = time.time()
    diverged = False

    for it in range(n_iters):
        optimizer.zero_grad()
        loss = criterion(model(X), Y_onehot)   # MSE vs one-hot targets
        val  = loss.item()
        losses.append(val)

        if not np.isfinite(val):
            if not diverged:
                print(f"  [{label}] NaN/Inf at iter {it}", flush=True)
            diverged = True

        if diverged:
            continue

        loss.backward()

        nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in model.parameters()
        )
        if nan_grad:
            if not diverged:
                print(f"  [{label}] NaN gradient at iter {it}", flush=True)
            diverged = True
            continue

        optimizer.step()

        if it % 200 == 0:
            print(f"  [{label}]  iter {it:4d}  loss = {val:.3e}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    print(f"  [{label}]  done — final loss = {losses[-1]:.3e}"
          f"  ({time.time()-t0:.0f}s)")
    return np.array(losses, dtype=np.float64)

# ── run all three regimes ──────────────────────────────────────────────────────
all_losses = {}
t_total = time.time()

for a, b, key, disp in REGIMES:
    beta1 = 1.0 - b * LR   # first-moment decay  (PyTorch beta1)
    beta2 = 1.0 - a * LR   # second-moment decay  (PyTorch beta2)
    print(f"\n{'='*60}")
    print(f"  {disp}   β₁={beta1:.3f}  β₂={beta2:.3f}")
    print(f"{'='*60}")
    all_losses[key] = run_adam(
        init_state, X_t, y_onehot, LR, beta1, beta2,
        N_ITERS, label=disp)

print(f"\nAll regimes done in {time.time()-t_total:.0f}s")

# ── colours ────────────────────────────────────────────────────────────────────
# Match the paper: uniform light blue-gray plot backgrounds, single blue line,
# white figure padding between plots.
PLOT_BG    = '#dde4f0'    # light blue-gray — same for all six subplots
LINE_COLOR = '#3a5f8a'    # medium navy-blue — same for all lines
FIG_BG     = 'white'      # white space between / around the subplots

# ── plotting helper ─────────────────────────────────────────────────────────────
def semilogy_safe(ax, x, y, xlim=None, stride=1, **kw):
    """Log-y plot; skips non-positive / non-finite values and optional x crop.
    stride > 1 subsamples the data (used for the full-view row to match the
    paper's smoother overview appearance)."""
    mask = np.isfinite(y) & (y > 0)
    if xlim is not None:
        mask &= (x >= xlim[0]) & (x <= xlim[1])
    idx = np.where(mask)[0][::stride]
    if len(idx):
        ax.semilogy(x[idx], y[idx], **kw)

iters = np.arange(N_ITERS, dtype=np.float64)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor(FIG_BG)

for col, (a, b, key, disp) in enumerate(REGIMES):
    losses = all_losses[key]
    z0, z1 = ZOOM[key]

    for row, (xlim, stride) in enumerate([
            ((0, N_ITERS), 1),   # full view  — every 10th point (smoother overview)
            ((z0, z1),      1),   # zoomed view — every point     (full resolution)
    ]):
        ax = axes[row, col]
        ax.set_facecolor(PLOT_BG)

        semilogy_safe(ax, iters, losses, xlim=xlim, stride=stride,
                      color=LINE_COLOR, linewidth=0.7)

        ax.set_xlim(*xlim)
        ax.set_xlabel('number of iterations', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, which='both', linestyle=':', linewidth=0.4,
                alpha=0.6, color='white')

        for sp in ax.spines.values():
            sp.set_linewidth(0)   # no visible border — matches paper style

        if row == 0:
            ax.set_title(key, fontsize=11)
            ax.text(0.97, 0.97, disp,
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.25', fc=PLOT_BG,
                              alpha=0.0, ec='none'))
        if col == 0:
            ax.set_ylabel('loss', fontsize=8)

#fig.suptitle(
#    'Figure 10: Three typical Adam behavior patterns\n'
#    'FC(256, 128, 64)  CIFAR-10 two-class  $\eta=0.001$  batch=64',
#    fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('figure10.pdf', bbox_inches='tight', facecolor=FIG_BG)
plt.savefig('figure10.png', dpi=150, bbox_inches='tight', facecolor=FIG_BG)
print("Saved figure10.pdf / figure10.png")
plt.close()
