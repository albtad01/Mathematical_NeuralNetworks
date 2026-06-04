"""
Replication of escape phenomenon (Zhu et al. 2019, Figure 3).

GD converges to a sharp minimum (theta*_GD) with ~100 % training accuracy
(memorising corrupted labels).  SGD, initialised at θ*_GD, escapes that
sharp minimum and settles at a flatter one with better test accuracy.

Setup (Zhu et al. 2019, Section 5.3)
--------------------------------------
  Dataset : FashionMNIST
              1 000 samples with correct labels
              +  200 samples with random labels
              = 1 200 training samples total
  Network : LeNet-like, ~11 244 parameters
              Conv2d(1->4, 5x5, pad=2) -> ReLU -> MaxPool(2)
              -> Linear(784->14) -> ReLU -> Linear(14->10)
  Loss    : cross-entropy
  LR      : nu = 0.07  (same constant for both GD and SGD)
  GD iters: 3 000  (to reach theta*_GD near a global minimum)
  SGD batch: m = 20
  Total iters plotted: 14 000

Left : Training accuracy (%) vs iteration
Right: Test accuracy    (%) vs iteration
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import time
import os

# ── device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Device: {DEVICE}")

# ── parameters ────────────────────────────────────────────────────────────────
N_CORRECT   = 1000    # samples with correct labels
N_RANDOM    = 200     # samples with random (corrupted) labels
N_TR        = N_CORRECT + N_RANDOM   # 1 200 total training samples
N_TE        = 1000    # test samples
N_CLS       = 10      # FashionMNIST classes
SEED        = 0

GD_ITERS    = 3000    # full-batch GD iterations to reach θ*_GD
TOTAL_ITERS = 14000   # total iterations shown on x-axis
LR          = 0.07    # learning rate — same for GD and SGD
SGD_BATCH   = 20      # mini-batch size (Zhu et al.: m = 20)
LOG_EVERY   = 1       # record accuracy at every single iteration

# SGD learning-rate decay: start large (η=LR) to escape the sharp minimum,
# then reduce so SGD can reconverge at the flat minimum it finds.
# Milestones are global iteration numbers.
SGD_LR_MILESTONES = [6000, 10000]   # global iters to halve the LR
SGD_LR_GAMMA      = 0.3             # 0.07 → 0.021 → 0.0063
SMOOTH_WIN        = 200             # rolling-average window for plotting

# ── FashionMNIST ──────────────────────────────────────────────────────────────
print("Loading FashionMNIST ...")
_ds_tr = datasets.FashionMNIST('./data', train=True,  download=True,
                                transform=transforms.ToTensor())
_ds_te = datasets.FashionMNIST('./data', train=False, download=True,
                                transform=transforms.ToTensor())

# Keep spatial dimensions (1×28×28) for the convolutional network.
X_all    = _ds_tr.data.float().unsqueeze(1) / 255.0   # (60000, 1, 28, 28)
y_all    = _ds_tr.targets.numpy()
X_te_raw = _ds_te.data.float().unsqueeze(1) / 255.0   # (10000, 1, 28, 28)
y_te_all = _ds_te.targets.numpy()

mu, std  = X_all.mean(), X_all.std()
X_all    = (X_all    - mu) / std
X_te_raw = (X_te_raw - mu) / std

# ── data split ────────────────────────────────────────────────────────────────
rng    = np.random.default_rng(SEED)
tr_idx = rng.choice(len(X_all),    N_TR, replace=False)
te_idx = rng.choice(len(X_te_raw), N_TE, replace=False)

X_tr_np = X_all[tr_idx].numpy()
y_tr_np = y_all[tr_idx].copy()
X_te_np = X_te_raw[te_idx].numpy()
y_te_np = y_te_all[te_idx]

# ── corrupt last N_RANDOM training labels ─────────────────────────────────────
y_tr_np[N_CORRECT:] = rng.integers(0, N_CLS, size=N_RANDOM)
print(f"Training: {N_CORRECT} correct + {N_RANDOM} random = {N_TR} total")

X_tr = torch.tensor(X_tr_np, device=DEVICE)
y_tr = torch.tensor(y_tr_np, dtype=torch.long, device=DEVICE)
X_te = torch.tensor(X_te_np, device=DEVICE)
y_te = torch.tensor(y_te_np, dtype=torch.long, device=DEVICE)

# ── model ─────────────────────────────────────────────────────────────────────
class LeNetSmall(nn.Module):
    """
    LeNet-like network for FashionMNIST (1x28x28 -> 10 classes).
    Parameter count: 104 + 10 990 + 150 = 11 244 near 11 330 (Zhu et al.).
      Conv2d(1, 4, 5, padding=2) -> ReLU -> MaxPool(2)  # 4x14x14 = 784 features
      Linear(784, 14) -> ReLU -> Linear(14, 10)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(784, 14), nn.ReLU(),
            nn.Linear(14, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


criterion = nn.CrossEntropyLoss()


@torch.no_grad()
def accuracy(model, X, y_int):
    return (model(X).argmax(1) == y_int).float().mean().item() * 100.0


def gd_step(model, X, y_int, lr):
    """Single full-batch GD step with cross-entropy loss."""
    model.zero_grad()
    loss = criterion(model(X), y_int)
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p -= lr * p.grad
    return loss.item()


# ── shared initialisation ─────────────────────────────────────────────────────
torch.manual_seed(SEED)
_tmp = LeNetSmall()
n_params = sum(p.numel() for p in _tmp.parameters())
print(f"LeNetSmall parameters: {n_params:,}")
init_state = _tmp.state_dict()
del _tmp

# ── cache files ───────────────────────────────────────────────────────────────
GD_LOGS_FILE   = 'figure8_gd_logs_lenet.npz'
GD_SWITCH_FILE = 'figure8_gd_switch_lenet.pt'
gd_cache_exists = os.path.isfile(GD_LOGS_FILE) and os.path.isfile(GD_SWITCH_FILE)

if gd_cache_exists:
    print(f"\n=== Loading cached GD results ===")
    _gd = np.load(GD_LOGS_FILE)
    iters_gd     = _gd['iters_gd']
    tr_gd        = _gd['tr_gd']
    te_gd        = _gd['te_gd']
    tr_at_switch = float(_gd['tr_at_switch'])
    te_at_switch = float(_gd['te_at_switch'])
    print(f"  GD switch point: train={tr_at_switch:.1f}%  test={te_at_switch:.1f}%")

else:
    # ── Run 1: pure full-batch GD (blue dashed, full x-axis) ─────────────────
    print(f"\n=== Run 1: pure GD ({TOTAL_ITERS} iters, lr={LR}) ===")
    net_gd = LeNetSmall().to(DEVICE)
    net_gd.load_state_dict({k: v.to(DEVICE) for k, v in init_state.items()})

    iters_gd, tr_gd, te_gd = [], [], []
    t0 = time.time()

    for it in range(TOTAL_ITERS + 1):
        if it % LOG_EVERY == 0:
            iters_gd.append(it)
            tr_gd.append(accuracy(net_gd, X_tr, y_tr))
            te_gd.append(accuracy(net_gd, X_te, y_te))
            if it % 1000 == 0:
                print(f"  GD  iter {it:6d}  train={tr_gd[-1]:.1f}%  "
                      f"test={te_gd[-1]:.1f}%  ({time.time()-t0:.0f}s)")
        if it < TOTAL_ITERS:
            gd_step(net_gd, X_tr, y_tr, LR)

    iters_gd = np.array(iters_gd)
    tr_gd    = np.array(tr_gd)
    te_gd    = np.array(te_gd)

    # ── GD warmup: save weights exactly at iteration GD_ITERS ────────────────
    print(f"\n  GD warmup → switch weights at iter {GD_ITERS} ...")
    net_sw = LeNetSmall().to(DEVICE)
    net_sw.load_state_dict({k: v.to(DEVICE) for k, v in init_state.items()})
    for it in range(GD_ITERS):
        loss = gd_step(net_sw, X_tr, y_tr, LR)
        if (it + 1) % 500 == 0:
            print(f"    iter {it+1:5d}  loss={loss:.4f}")

    tr_at_switch = accuracy(net_sw, X_tr, y_tr)
    te_at_switch = accuracy(net_sw, X_te, y_te)
    print(f"  Switch point: train={tr_at_switch:.1f}%  test={te_at_switch:.1f}%")

    np.savez(GD_LOGS_FILE,
             iters_gd=iters_gd, tr_gd=tr_gd, te_gd=te_gd,
             tr_at_switch=tr_at_switch, te_at_switch=te_at_switch)
    torch.save(net_sw.state_dict(), GD_SWITCH_FILE)
    print(f"  Saved {GD_LOGS_FILE} and {GD_SWITCH_FILE}")

# ── Run 2: SGD phase initialised from theta*_GD ───────────────────────────────────
print(f"\n=== Run 2 (SGD phase): lr={LR}  batch={SGD_BATCH} ===")
net_sgd = LeNetSmall().to(DEVICE)
net_sgd.load_state_dict(
    {k: v.to(DEVICE)
     for k, v in torch.load(GD_SWITCH_FILE, map_location=DEVICE).items()})

dataset     = TensorDataset(X_tr, y_tr)
loader      = DataLoader(dataset, batch_size=SGD_BATCH, shuffle=True,
                         generator=torch.Generator().manual_seed(SEED))
loader_iter = iter(loader)
optimizer   = torch.optim.SGD(net_sgd.parameters(), lr=LR)
# Convert global-iteration milestones to SGD-phase iteration milestones
_sgd_milestones = [m - GD_ITERS for m in SGD_LR_MILESTONES if m > GD_ITERS]
scheduler   = torch.optim.lr_scheduler.MultiStepLR(
                  optimizer, milestones=_sgd_milestones, gamma=SGD_LR_GAMMA)

# First log point is the GD solution at the switch
iters_sgd = [GD_ITERS]
tr_sgd    = [tr_at_switch]
te_sgd    = [te_at_switch]

SGD_PHASE = TOTAL_ITERS - GD_ITERS
t0 = time.time()

for it in range(1, SGD_PHASE + 1):
    try:
        Xb, yb = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        Xb, yb = next(loader_iter)

    optimizer.zero_grad()
    criterion(net_sgd(Xb), yb).backward()
    optimizer.step()
    #scheduler.step()

    global_it = GD_ITERS + it
    if it % LOG_EVERY == 0:
        iters_sgd.append(global_it)
        tr_sgd.append(accuracy(net_sgd, X_tr, y_tr))
        te_sgd.append(accuracy(net_sgd, X_te, y_te))
        if it % 1000 == 0:
            print(f"  SGD iter {global_it:6d}  train={tr_sgd[-1]:.1f}%  "
                  f"test={te_sgd[-1]:.1f}%  ({time.time()-t0:.0f}s)")

# ── plot ──────────────────────────────────────────────────────────────────────
iters_gd  = np.array(iters_gd)
tr_gd     = np.array(tr_gd)
te_gd     = np.array(te_gd)
iters_sgd = np.array(iters_sgd)
tr_sgd    = np.array(tr_sgd)
te_sgd    = np.array(te_sgd)

def smooth(y, w):
    """Centred rolling average with window w; edges use smaller windows."""
    return np.convolve(y, np.ones(w) / w, mode='same')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for ax, gd_y, sgd_y, ylabel in [
        (ax1, tr_gd, tr_sgd, 'Train Accuracy (%)'),
        (ax2, te_gd, te_sgd, 'Test Accuracy (%)')]:

    # raw GD curve (full-batch -> no noise, no smoothing needed)
    ax.plot(iters_gd, gd_y, 'b--', linewidth=1.5, label='GD', alpha=0.9)

    # SGD: faint raw trace + bold smoothed trend
    ax.plot(iters_sgd, sgd_y,
            color='green', linewidth=0.4, alpha=0.25)
    ax.plot(iters_sgd, smooth(sgd_y, SMOOTH_WIN),
            color='green', linewidth=1.8, label='SGD (smoothed)', alpha=0.95)

    # switch point and LR decay markers
    ax.axvline(GD_ITERS, color='gray', linewidth=1.0, linestyle=':', alpha=0.7)
    #for ms in SGD_LR_MILESTONES:
    #    ax.axvline(ms, color='orange', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Iteration')
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=11)
    ax.set_xlim(0, TOTAL_ITERS)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4)

#fig.suptitle(
#    rf'Escape phenomenon — corrupted FashionMNIST '
#    rf'($N_{{tr}}={N_TR},\ {N_RANDOM}$ random labels, '
#    rf'$\eta={LR}$, batch={SGD_BATCH}, '
#    rf'GD$\to$SGD at iter {GD_ITERS})',
#    fontsize=9)

plt.tight_layout()
plt.savefig('figure8.pdf', bbox_inches='tight')
plt.savefig('figure8.png', dpi=150, bbox_inches='tight')
print("Saved figure8.pdf / figure8.png")
plt.close()
