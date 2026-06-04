"""
Replication of Figure 11 from E, Ma, Wojtowytsch, Wu (2020).

Three panels
------------
  Left   : Adam training curve with default parameters (alpha=0.9, beta=0.999),
            i.e. betas=(0.9, 0.999) → spike regime (a=1, b=100).
  Middle : Heat map of the average training loss (log₁₀) over the last 1 000
            of 1 500 Adam iterations, across a grid of (a, b) values.
  Right  : Classification of Adam behaviour:
              spikes / small oscillation / large oscillation.

Setup (Figure 11 caption)
--------------------------
  Network  : FC  3072 -> 256 -> 256 -> 128 -> 2,  ReLU activations
  Data     : 2 classes of CIFAR-10 (airplane=0, automobile=1),
             500 samples per class = 1 000 total
  LR       : η = 0.001
  Loss     : square (MSE) with one-hot targets
  Iters    : 1 500 total;  statistics over last 1 000
  Grid     : a, b ∈ [0.1, 100] log-spaced (N_GRID points each)
             beta2 = 1 - a·nu  (PyTorch betas[1], second-moment decay)
             beta1 = 1 - b·nu  (PyTorch betas[0], first-moment  decay)

Default for left panel: (alpha, beta) = (0.9, 0.999) in standard DL convention:
  alpha = beta1 = 0.9  ->  b = (1 - 0.9 ) / nu = 100
  beta  = beta2 = 0.999 ->  a = (1 - 0.999) / nu =   1
  -> a=1, b=100: the spike regime from Figure 10.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time
import os

# ── device ─────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
# float32 is sufficient: divergence cases overflow to Inf (detected and capped),
# and the spike / small-oscillation regimes stay well within float32 range.
DTYPE = torch.float32
print(f"Device: {DEVICE}  dtype: float32")

# ── hyperparameters ────────────────────────────────────────────────────────────
LR          = 0.001
N_ITERS     = 1500        # Adam iterations per run
STATS_WIN   = 1000        # average loss over the last STATS_WIN iterations
N_PER_CLASS = 500         # CIFAR-10 samples per class (2 classes → 1 000 total)
CLASSES     = [0, 1]      # airplane, automobile
SEED        = 0

# Grid: a and b from 0.1 to 100 log-spaced
N_GRID  = 18
A_VALS  = np.logspace(np.log10(0.1), np.log10(100), N_GRID)
B_VALS  = np.logspace(np.log10(0.1), np.log10(100), N_GRID)

# Default run for the left panel (spike regime)
LEFT_A, LEFT_B = 1, 100    # -> betas=(0.9, 0.999)

# Classification thresholds
LARGE_OSC_MEAN  = 0.12    # mean loss above this → large oscillation
SPIKE_RATIO     = 50.0    # max/min ratio above this → spikes (if mean is low)

# Cache file for the expensive grid sweep
CACHE_FILE = 'figure11_grid.npz'

# ── CIFAR-10 ───────────────────────────────────────────────────────────────────
print("Loading CIFAR-10 …")
_ds = datasets.CIFAR10('./data', train=True, download=True,
                        transform=transforms.ToTensor())

X_list, y_list = [], []
counts = {c: 0 for c in CLASSES}
for img, label in _ds:
    if label in CLASSES and counts[label] < N_PER_CLASS:
        X_list.append(img.numpy().reshape(-1).astype(np.float32))
        y_list.append(CLASSES.index(label))
        counts[label] += 1
    if all(v >= N_PER_CLASS for v in counts.values()):
        break

X_np = np.array(X_list)
y_np = np.array(y_list, np.int64)

mu, sd = X_np.mean(), X_np.std()
X_np   = (X_np - mu) / sd

rng  = np.random.default_rng(SEED)
perm = rng.permutation(len(X_np))
X_np, y_np = X_np[perm], y_np[perm]

X_t      = torch.tensor(X_np, dtype=DTYPE, device=DEVICE)
y_t      = torch.tensor(y_np, device=DEVICE)
n_classes = len(CLASSES)
y_onehot  = torch.zeros(len(y_np), n_classes, dtype=DTYPE, device=DEVICE)
y_onehot.scatter_(1, y_t.unsqueeze(1), 1.0)
print(f"Data: {X_t.shape}  |  class counts: {counts}")

# ── network ────────────────────────────────────────────────────────────────────
class FCNet(nn.Module):
    """FC: 3072 -> 256 -> 256 -> 128 -> 2, ReLU activations."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3072, 256), nn.ReLU(),
            nn.Linear( 256, 256), nn.ReLU(),
            nn.Linear( 256, 128), nn.ReLU(),
            nn.Linear( 128,   2),
        )
    def forward(self, x):
        return self.net(x)

criterion = nn.MSELoss()

torch.manual_seed(SEED)
init_state = FCNet().to(dtype=DTYPE).state_dict()
n_params   = sum(p.numel() for p in FCNet().parameters())
print(f"FCNet parameters: {n_params:,}")

# ── Adam runner ────────────────────────────────────────────────────────────────
def run_adam(a, b):
    """
    Full-batch Adam for N_ITERS steps.
    Returns np.ndarray of shape (N_ITERS,) with losses.
    Inf is stored for NaN/Inf steps; remaining steps are filled with Inf.
    """
    beta1 = float(np.clip(1.0 - b * LR, 1e-6, 1 - 1e-6))
    beta2 = float(np.clip(1.0 - a * LR, 1e-6, 1 - 1e-6))

    model = FCNet().to(dtype=DTYPE, device=DEVICE)
    model.load_state_dict({k: v.to(dtype=DTYPE, device=DEVICE)
                           for k, v in init_state.items()})
    optimizer = torch.optim.Adam(model.parameters(), lr=LR,
                                  betas=(beta1, beta2), eps=1e-8)
    losses   = []
    diverged = False

    for it in range(N_ITERS):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_onehot)
        val  = loss.item()

        if not np.isfinite(val):
            diverged = True

        losses.append(val if np.isfinite(val) else np.inf)

        if diverged:
            losses.extend([np.inf] * (N_ITERS - len(losses)))
            break

        loss.backward()

        nan_grad = any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in model.parameters())
        if nan_grad:
            losses.extend([np.inf] * (N_ITERS - len(losses)))
            break

        optimizer.step()

    return np.array(losses)

# ── behaviour classifier ────────────────────────────────────────────────────────
def classify(losses):
    """
    'large_oscillation' : mean loss over last STATS_WIN steps is large, or Inf.
    'spikes'            : mean is small but max/min ratio is large
                          (periodic large spikes punctuating a low-loss trend).
    'small_oscillation' : bounded oscillation near a moderate loss value.
    """
    last = losses[-STATS_WIN:]

    if not np.all(np.isfinite(last)):
        return 'large_oscillation'

    mean_l = float(np.mean(last))
    if mean_l > LARGE_OSC_MEAN:
        return 'large_oscillation'

    max_l = float(np.max(last))
    min_l = float(np.min(last))
    ratio = max_l / max(min_l, 1e-14)
    if ratio > SPIKE_RATIO:
        return 'spikes'

    return 'small_oscillation'

# ── left panel: default (spike) run ───────────────────────────────────────────
print(f"\n--- Left panel: a={LEFT_A}, b={LEFT_B} "
      f"→ betas=({1-LEFT_B*LR:.3f}, {1-LEFT_A*LR:.3f}) ---")
losses_default = run_adam(LEFT_A, LEFT_B)
print(f"  Behaviour: {classify(losses_default)}")
print(f"  Final loss: {losses_default[-1]:.3e}")

# ── grid sweep (cached) ────────────────────────────────────────────────────────
if os.path.isfile(CACHE_FILE):
    print(f"\nLoading grid cache from {CACHE_FILE} …")
    _c         = np.load(CACHE_FILE, allow_pickle=True)
    mean_grid  = _c['mean_grid']
    class_grid = _c['class_grid']
    print(f"  Loaded {N_GRID}×{N_GRID} grid.")
else:
    print(f"\n--- Grid sweep: {N_GRID}×{N_GRID} = {N_GRID**2} runs ---")
    mean_grid  = np.full((N_GRID, N_GRID), np.nan)
    class_grid = np.empty((N_GRID, N_GRID), dtype=object)
    t0         = time.time()

    for i, b in enumerate(B_VALS):
        for j, a in enumerate(A_VALS):
            L = run_adam(a, b)
            finite = L[-STATS_WIN:][np.isfinite(L[-STATS_WIN:])]
            mean_grid[i, j]  = float(np.mean(finite)) if len(finite) > 0 else np.inf
            class_grid[i, j] = classify(L)

        elapsed = time.time() - t0
        eta     = elapsed / (i + 1) * (N_GRID - i - 1)
        pct     = [class_grid[i, j] for j in range(N_GRID)]
        print(f"  b={b:6.2f}  "
              f"sp={pct.count('spikes')}  "
              f"sm={pct.count('small_oscillation')}  "
              f"lg={pct.count('large_oscillation')}  "
              f"{elapsed:.0f}s / ETA {eta:.0f}s")

    np.savez(CACHE_FILE, mean_grid=mean_grid, class_grid=class_grid,
             A_VALS=A_VALS, B_VALS=B_VALS)
    print(f"Saved {CACHE_FILE}  (total {time.time()-t0:.0f}s)")

# ── plot ────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor('white')

# ── helper: shared axis ticks for the two grid panels ────────────────────────
def grid_ticks(ax, a_vals, b_vals, n=6):
    """Show ~n evenly-spaced log-scale tick labels."""
    step = max(1, N_GRID // n)
    xt   = np.arange(N_GRID)[::step] + 0.5
    yt   = np.arange(N_GRID)[::step] + 0.5
    ax.set_xticks(xt)
    ax.set_xticklabels([f'{a_vals[k]:.1f}' for k in range(N_GRID)[::step]],
                       fontsize=6, rotation=45)
    ax.set_yticks(yt)
    ax.set_yticklabels([f'{b_vals[k]:.1f}' for k in range(N_GRID)[::step]],
                       fontsize=6)
    ax.set_xlabel('a', fontsize=10)
    ax.set_ylabel('b', fontsize=10)

# ── Left: training curve ──────────────────────────────────────────────────────
ax = axes[0]
iters = np.arange(1, N_ITERS + 1)
mask  = np.isfinite(losses_default) & (losses_default > 0)
ax.semilogy(iters[mask], losses_default[mask],
            color='#3a5f8a', linewidth=0.9)
ax.set_xlabel('number of iterations', fontsize=9)
ax.set_ylabel('loss', fontsize=9)
ax.set_xlim(0, N_ITERS)
ax.set_facecolor('#dde4f0')
ax.grid(True, which='both', linestyle=':', linewidth=0.4, alpha=0.5, color='white')
for sp in ax.spines.values():
    sp.set_linewidth(0)

# ── Middle: log10(mean loss) heat map ────────────────────────────────────────
ax    = axes[1]
# Rows indexed by B_VALS (i=0 -> smallest b), imshow row 0 is at top → flipud
log_m = np.log10(np.clip(mean_grid, 1e-8, None))
im    = ax.imshow(np.flipud(log_m), aspect='auto',
                  cmap='inferno_r', vmin=-7, vmax=-1,
                  extent=[0, N_GRID, 0, N_GRID])
grid_ticks(ax, A_VALS, B_VALS[::-1])   # B flipped → largest b at top
ax.set_title('loss', fontsize=11)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# ── Right: behaviour classification ──────────────────────────────────────────
ax = axes[2]
COLOR = {
    'spikes':            '#f5e0cc',   # light peach  (upper-left)
    'small_oscillation': '#8b1a1a',   # dark red      (diagonal band)
    'large_oscillation': '#000000',   # black         (lower-right)
}
rgb_img = np.zeros((N_GRID, N_GRID, 3))
for i in range(N_GRID):
    for j in range(N_GRID):
        c = class_grid[i, j]
        rgb_img[i, j] = mcolors.to_rgb(COLOR.get(c, 'gray'))

ax.imshow(np.flipud(rgb_img), aspect='auto',
          extent=[0, N_GRID, 0, N_GRID])
grid_ticks(ax, A_VALS, B_VALS[::-1])

# Region labels
ax.text(0.58, 0.88, 'spikes',
        transform=ax.transAxes, fontsize=9, color='black')
ax.text(0.20, 0.50, 'small\noscillation',
        transform=ax.transAxes, fontsize=8, color='white')
ax.text(0.55, 0.12, 'large oscillation',
        transform=ax.transAxes, fontsize=9, color='white')

# Legend
legend_elems = [
    Patch(facecolor=COLOR['spikes'],            label='spikes'),
    Patch(facecolor=COLOR['small_oscillation'], label='small oscillation'),
    Patch(facecolor=COLOR['large_oscillation'], label='large oscillation',
          edgecolor='white', linewidth=0.5),
]
ax.legend(handles=legend_elems, loc='upper left', fontsize=7, framealpha=0.7)

plt.tight_layout()
plt.savefig('figure11.pdf', bbox_inches='tight', facecolor='white')
plt.savefig('figure11.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Saved figure11.pdf / figure11.png")
plt.close()
