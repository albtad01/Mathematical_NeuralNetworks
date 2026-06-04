"""
Replication of Figure 9 from E, Ma, Wojtowytsch, Wu (2020).

Sharpness-nonuniformity diagram for minima selected by GD and SGD
applied to a VGG-type network for FashionMNIST.

Definitions  (Section 5.5, E et al. 2020 — 1-D toy model, extended to multi-D)
-----------
  Sharpness     a = (1/n) sum_i a_i  =  lambda_max(nabla^2*L)
                  largest Hessian eigenvalue, estimated via power iteration.
                  At stability boundary for GD: a = 2/η.

  Non-uniformity s = sqrt( (1/n) sum_i a_i^2 - a^2 )
                  std of per-sample curvatures a_i = v^T nabla^2 L_i(θ) v,
                  where v is the leading eigenvector of nabla^2 L.

Theory predicts that SGD with batch B selects minima satisfying
  s <= C · sqrt(B)
and GD (full batch) clusters near the stability boundary a = 2/nu.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import time

# ── device ────────────────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
# Hessian-vector products (create_graph=True) can fail on MPS;
# fall back to CPU for those computations.
HVP_DEVICE = torch.device('cpu')
print(f"Training device: {DEVICE}  |  HVP device: {HVP_DEVICE}")

# ── parameters ────────────────────────────────────────────────────────────────
N_TR         = 500     # training samples (small so full-batch GD is affordable)
N_TE         = 1000    # test samples
LR           = 0.5     # learning rate shared by all optimizers
                       # → GD stability threshold 2/LR = 4.0
BATCH_SIZES  = [None, 25, 10, 4]  # None = full-batch GD
N_SEEDS      = 8       # independent runs per configuration
MAX_EPOCHS   = 1500    # max training epochs
TOL_LOSS     = 5e-3    # stop when full-batch loss < TOL_LOSS
N_POWER_ITER = 40      # power iterations for lambda_max
DATA_SEED    = 0

# ── FashionMNIST ──────────────────────────────────────────────────────────────
print("Loading FashionMNIST ...")
_ds_tr = datasets.FashionMNIST('./data', train=True,  download=True,
                                transform=transforms.ToTensor())
_ds_te = datasets.FashionMNIST('./data', train=False, download=True,
                                transform=transforms.ToTensor())

X_all    = _ds_tr.data.float().reshape(-1, 1, 28, 28) / 255.0
y_all    = torch.tensor(_ds_tr.targets.numpy())
X_te_raw = _ds_te.data.float().reshape(-1, 1, 28, 28) / 255.0
y_te_all = torch.tensor(_ds_te.targets.numpy())

mu, std  = X_all.mean(), X_all.std()
X_all    = (X_all    - mu) / std
X_te_raw = (X_te_raw - mu) / std

rng    = np.random.default_rng(DATA_SEED)
tr_idx = rng.choice(len(X_all),    N_TR, replace=False)
te_idx = rng.choice(len(X_te_raw), N_TE, replace=False)

X_tr    = X_all[tr_idx].to(DEVICE)
y_tr    = y_all[tr_idx].to(DEVICE)          # integer labels (for accuracy only)
X_te    = X_te_raw[te_idx].to(DEVICE)
y_te    = y_te_all[te_idx].to(DEVICE)

# One-hot targets for MSE loss (Wu & Ma 2021: "use MSE rather than cross-entropy")
N_CLS      = 10
y_tr_oh    = torch.zeros(N_TR, N_CLS, device=DEVICE)
y_tr_oh.scatter_(1, y_tr.unsqueeze(1), 1.0)

# CPU copies for HVP / per-sample computations
X_tr_cpu    = X_tr.cpu()
y_tr_cpu    = y_tr.cpu()        # integer, for nothing currently — kept for safety
y_tr_oh_cpu = y_tr_oh.cpu()    # one-hot on CPU for Hessian computations

# ── VGG-type network ──────────────────────────────────────────────────────────
class VGGSmall(nn.Module):
    """
    Small VGG-type architecture for FashionMNIST (28x28 grayscale):
      Conv(1→16) -> ReLU -> MaxPool -> Conv(16→32) -> ReLU -> MaxPool
      -> Linear(1568→128) -> ReLU -> Linear(128→10)
    ~207 K parameters — small enough for efficient Hessian computation.
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

# MSE (square) loss — Wu & Ma 2021 use this for all experiments
criterion = nn.MSELoss()

# ── training ──────────────────────────────────────────────────────────────────
def train_model(model, X, y_oh, lr, batch_size, max_epochs, tol):
    """
    Train to convergence using MSE loss with one-hot targets y_oh.
    batch_size=None → full-batch GD.
    batch_size=B    → mini-batch SGD.
    """
    n = X.shape[0]

    if batch_size is None:
        label = "GD "
        for epoch in range(max_epochs):
            model.zero_grad()
            loss = criterion(model(X), y_oh)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= lr * p.grad
            if epoch % 300 == 0:
                print(f"      {label} epoch {epoch:5d}  loss={loss.item():.3e}")
            if loss.item() < tol:
                print(f"      {label} converged  epoch={epoch}  loss={loss.item():.3e}")
                return
        print(f"      {label} max_epochs reached  loss={loss.item():.3e}")

    else:
        label = f"SGD(B={batch_size})"
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        for epoch in range(max_epochs):
            perm = torch.randperm(n, device=X.device)
            for i in range(0, n, batch_size):
                idx = perm[i : i + batch_size]
                optimizer.zero_grad()
                criterion(model(X[idx]), y_oh[idx]).backward()
                optimizer.step()
            with torch.no_grad():
                full_loss = criterion(model(X), y_oh).item()
            if epoch % 300 == 0:
                print(f"      {label} epoch {epoch:5d}  loss={full_loss:.3e}")
            if full_loss < tol:
                print(f"      {label} converged  epoch={epoch}  loss={full_loss:.3e}")
                return
        print(f"      {label} max_epochs reached  loss={full_loss:.3e}")

# ── sharpness + nonuniformity (paper definitions, Section 5.5) ───────────────
def compute_sharpness_and_nonuniformity(model, X, y, n_iter=40):
    """
    Returns (sharpness, nonuniformity) following E, Ma, Wojtowytsch, Wu (2020).

    From the 1-D toy model f(x) = 1/(2n) sum_i a_i x^2, extended to multi-dimension:

      Sharpness      a  = (1/n) sum_i a_i  =  lambda_max(nabla^2 L)
                         estimated via Rayleigh-quotient power iteration.

      Non-uniformity s  = sqrt( (1/n) sum_i a_i^2 - a^2 )
                         std of the per-sample curvatures
                         a_i = v^T nabla^2 L_i(theta) v
                         where v is the leading eigenvector of ∇²L.

    Both computations run on CPU (create_graph=True is unreliable on MPS).
    """
    # ── CPU copy ──────────────────────────────────────────────────────────────
    model_cpu = VGGSmall()
    model_cpu.load_state_dict({k: w.cpu() for k, w in model.state_dict().items()})
    model_cpu.eval()

    params = [p for p in model_cpu.parameters() if p.requires_grad]
    n_p    = sum(p.numel() for p in params)
    n      = X.shape[0]

    # ── Phase 1: power iteration -> v and lambda_max ────────────────────────────────
    vec = torch.randn(n_p)
    vec /= vec.norm()
    lam = 0.0

    for _ in range(n_iter):
        model_cpu.zero_grad()
        loss = criterion(model_cpu(X), y)   # y must be one-hot for MSELoss
        grads = torch.autograd.grad(loss, params, create_graph=True)
        g_flat = torch.cat([g.reshape(-1) for g in grads])

        gv     = (g_flat * vec.detach()).sum()
        Hv_tup = torch.autograd.grad(gv, params, retain_graph=False)
        Hv     = torch.cat([h.reshape(-1) for h in Hv_tup]).detach()

        lam = (vec * Hv).sum().item()
        vec = Hv / (Hv.norm() + 1e-12)

    sharpness = lam
    # vec is now the leading eigenvector (detached)

    # ── Phase 2: per-sample curvatures a_i = v^T nabla^2 L_i v ───────────────────────
    a_samples = []
    for i in range(n):
        model_cpu.zero_grad()
        loss_i  = criterion(model_cpu(X[i:i+1]), y[i:i+1])
        grads_i = torch.autograd.grad(loss_i, params, create_graph=True)
        g_i     = torch.cat([g.reshape(-1) for g in grads_i])

        # v^T g_i  ->  d/dθ  ->  v^T (nabla^2 L_i) v
        gv_i     = (g_i * vec).sum()
        Hv_i_tup = torch.autograd.grad(gv_i, params, retain_graph=False)
        Hv_i     = torch.cat([h.reshape(-1) for h in Hv_i_tup]).detach()
        a_i      = (vec * Hv_i).sum().item()
        a_samples.append(a_i)

        if (i + 1) % 100 == 0:
            print(f"        per-sample HVP: {i+1}/{n} done", flush=True)

    a_arr          = np.array(a_samples)
    mean_a         = np.mean(a_arr)
    nonuniformity  = float(np.sqrt(max(np.mean(a_arr ** 2) - mean_a ** 2, 0.0)))

    return sharpness, nonuniformity

# ── main experiment ───────────────────────────────────────────────────────────
configs = [
    (None, 'GD',       'tab:blue',   'o'),
    (25,   'SGD, B=25','tab:orange', '*'),
    (10,   'SGD, B=10','tab:green',  's'),
    (4,    'SGD, B=4', 'tab:red',    '^'),
]

results = {name: {'S': [], 'U': []} for _, name, _, _ in configs}
t0 = time.time()

for batch_size, name, color, marker in configs:
    print(f"\n{'='*65}")
    print(f"  Config: {name}  |  {N_SEEDS} seeds  |  LR={LR}")
    print(f"{'='*65}")

    for seed in range(N_SEEDS):
        print(f"\n  Seed {seed + 1}/{N_SEEDS}")
        torch.manual_seed(seed)
        model = VGGSmall().to(DEVICE)

        # Train (y_tr_oh = one-hot targets for MSE loss)
        train_model(model, X_tr, y_tr_oh, LR, batch_size, MAX_EPOCHS, TOL_LOSS)

        with torch.no_grad():
            logits  = model(X_tr)
            tr_acc  = (logits.argmax(1) == y_tr).float().mean().item() * 100
            tr_loss = criterion(logits, y_tr_oh).item()   # MSE needs one-hot targets
        print(f"    train_acc={tr_acc:.1f}%  train_loss={tr_loss:.3e}")

        # Sharpness + nonuniformity (paper defs, combined to reuse eigenvector v)
        print(f"    Computing sharpness ({N_POWER_ITER} iters) + "
              f"nonuniformity ({N_TR} HVPs) ...", flush=True)
        S, U = compute_sharpness_and_nonuniformity(
                   model, X_tr_cpu, y_tr_oh_cpu, n_iter=N_POWER_ITER)
        print(f"    S = {S:.4f}   U = {U:.4f}")

        results[name]['S'].append(S)
        results[name]['U'].append(U)
        print(f"    Elapsed so far: {time.time() - t0:.0f}s")

print(f"\nAll configurations done in {time.time() - t0:.0f}s")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

for batch_size, name, color, marker in configs:
    S_list = results[name]['S']
    U_list = results[name]['U']
    ax.scatter(S_list, U_list,
               color=color, marker=marker,
               s=(120 if marker == '*' else 80),
               label=name, zorder=3, alpha=0.85)

# Vertical line: GD stability threshold S = 2/nu
stab = 2.0 / LR
ax.axvline(stab, color='black', linewidth=1.5, zorder=4)
ylim_top = ax.get_ylim()[1]
ax.text(stab * 1.02, ylim_top * 0.97, r'$2/\eta$',
        fontsize=12, ha='left', va='top')

# Dashed predicted bounds: U <= C · sqrt(B)
# Estimate C from the mean U / sqrt(B) of each SGD configuration
C_estimates = []
for batch_size, name, _, _ in configs:
    if batch_size is not None and results[name]['U']:
        C_estimates.append(np.mean(results[name]['U']) / np.sqrt(batch_size))
C = float(np.mean(C_estimates)) if C_estimates else 1.0
print(f"Estimated bound constant C = {C:.3f}  →  s_bound = {C:.2f}·√B")

for batch_size, name, color, marker in configs:
    if batch_size is not None:
        ax.axhline(C * np.sqrt(batch_size), color=color,
                   linewidth=1.5, linestyle='--', alpha=0.7, zorder=2)

ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.set_xlabel('sharpness', fontsize=13, fontweight='bold')
ax.set_ylabel('nonuniformity', fontsize=13, fontweight='bold')
ax.set_title('FashionMNIST', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='upper left')
ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.4)

plt.tight_layout()
plt.savefig('figure9.pdf', bbox_inches='tight')
plt.savefig('figure9.png', dpi=150, bbox_inches='tight')
print("Saved figure9.pdf / figure9.png")
plt.close()
