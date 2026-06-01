"""
Replication of Figure 2 from E, Ma, Wojtowytsch, Wu (2020).

Rate of convergence of mean-field GD for Barron vs non-Barron target functions.

Model (mean-field):   f_m(x) = (1/m) sum_j a_j sigma(w_j^T x)
GD dynamics (eq.35):  du_j/dt = -m * grad_{u_j} I
  → discrete update with step eta:
      a -= (eta/n) * H^T r
      B -= (eta/n) * (r[:,None] * mask * a[None,:]).T @ ... (see mf_gd_step)

Barron target:     f*(x) = ReLU(x_1)                          Barron norm = 1
Non-Barron target: f*(x) = ReLU((sum_k x_k - d/2) / sqrt(d)) Barron norm = sqrt(d)
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# ── device ────────────────────────────────────────────────────────────────────

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
print(f"Using device: {DEVICE}")

# ── experiment parameters ─────────────────────────────────────────────────────

M         = 2000      # neurons — needs to be >> d for dimension-independent Barron convergence
N         = 1000      # training samples
N_TEST    = 10_000    # test samples for population risk
ETA       = 0.02      # smaller step → stable for d=250 (t_max = N_STEPS * ETA = 65)
N_STEPS   = 3250      # 3250 * 0.02 = 65
LOG_EVERY = 10        # record every 10 steps  (Δt = 0.2)
SEED      = 0
DIMS      = [30, 100, 250]

# ── target functions ──────────────────────────────────────────────────────────

def barron_target(X):
    """f*(x) = ReLU(x_1).  Barron norm = ||e_1||_1 = 1 (dimension-independent)."""
    return torch.relu(X[:, 0])

def non_barron_target(X):
    """
    f*(x) = ReLU( (sum_k x_k - d/2) / sqrt(d) ).
    Centered so mean≈0; std≈1/sqrt(12) regardless of d.
    Barron norm = ||1/sqrt(d)||_1 = sqrt(d)  →  grows with dimension.
    """
    d = X.shape[1]
    return torch.relu((X.sum(dim=1) - d / 2.0) / d ** 0.5)

# ── data & initialisation ─────────────────────────────────────────────────────

def make_data(n, d, seed):
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.uniform(0, 1, (n, d)), dtype=torch.float32, device=DEVICE)
    return X

def init_params(m, d, seed):
    """a_j ~ N(0,1),  w_j ~ N(0, I/d)."""
    rng = np.random.default_rng(seed)
    a = torch.tensor(rng.normal(0, 1.0,            m   ).astype(np.float32), device=DEVICE)
    B = torch.tensor(rng.normal(0, 1.0/d**0.5, (m, d)).astype(np.float32), device=DEVICE)
    return a, B

# ── mean-field forward & GD step ─────────────────────────────────────────────

def mf_forward(X, a, B):
    """f_m(x) = (1/m) sum_j a_j ReLU(w_j^T x)."""
    return torch.relu(X @ B.T) @ a / len(a)   # (n,)

def mf_gd_step(X, y, a, B, eta):
    """
    Mean-field GD step (from du_j/dt = -m grad_{u_j} I, m's cancel):
        da = (eta/n) * H^T r
        dB[j] = (eta * a_j / n) * X^T (r * 1[z_j > 0])
    """
    n = X.shape[0]
    Z = X @ B.T                                    # (n, m)
    H = torch.relu(Z)                              # (n, m)
    r = H @ a / len(a) - y                         # (n,)  residual

    da = H.T @ r / n                               # (m,)
    mask = (Z > 0).float()                         # (n, m)
    dB = (X.T @ (r.unsqueeze(1) * mask) * a.unsqueeze(0) / n).T  # (m, d)

    return a - eta * da, B - eta * dB

def mse(f, y):
    return (0.5 * (f - y).pow(2).mean()).item()

# ── training run ──────────────────────────────────────────────────────────────

def run(m, n, d, eta, n_steps, target_fn, log_every, seed):
    X_tr = make_data(n,      d, seed)
    X_te = make_data(N_TEST, d, seed + 1)
    y_tr = target_fn(X_tr)
    y_te = target_fn(X_te)

    a, B = init_params(m, d, seed)

    times, emp_risks, pop_risks = [], [], []

    for step in range(n_steps + 1):
        if step % log_every == 0:
            with torch.no_grad():
                emp_risks.append(mse(mf_forward(X_tr, a, B), y_tr))
                pop_risks.append(mse(mf_forward(X_te, a, B), y_te))
            times.append(step * eta)
            print(f"    t={step*eta:5.1f}  emp={emp_risks[-1]:.3e}  "
                  f"pop={pop_risks[-1]:.3e}", flush=True)

        if step < n_steps:
            with torch.no_grad():
                a, B = mf_gd_step(X_tr, y_tr, a, B, eta)

    return np.array(times), np.array(emp_risks), np.array(pop_risks)

# ── run all experiments ───────────────────────────────────────────────────────

results = {}
t0 = time.time()

for target_name, target_fn in [('barron',     barron_target),
                                ('non_barron', non_barron_target)]:
    for d in DIMS:
        print(f"\n[{target_name}  d={d:3d}]  m={M}, n={N}, eta={ETA}, steps={N_STEPS}")
        times, emp, pop = run(M, N, d, ETA, N_STEPS, target_fn,
                              log_every=LOG_EVERY, seed=SEED)
        results[(target_name, d)] = (times, emp, pop)
        print(f"  done in {time.time()-t0:.0f}s  "
              f"final emp={emp[-1]:.3e}  pop={pop[-1]:.3e}")

print(f"\nTotal time: {time.time()-t0:.0f}s")

# ── plot ──────────────────────────────────────────────────────────────────────

BARRON_COLORS     = ['#1f77b4', '#222222', '#8B4513']   # blue, black, brown
NON_BARRON_COLORS = ['#2ca02c', '#ff7f0e', '#d62728']   # green, orange, red

fig, ax = plt.subplots(figsize=(10, 5))

for di, d in enumerate(DIMS):
    times, emp, pop = results[('barron', d)]
    c = BARRON_COLORS[di]
    ax.semilogy(times, emp, linestyle='--', color=c, linewidth=1.2,
                label=f'empirical risk, Barron, d={d}')
    ax.semilogy(times, pop, linestyle='-',  color=c, linewidth=1.5,
                label=f'population risk, Barron, d={d}')

for di, d in enumerate(DIMS):
    times, emp, pop = results[('non_barron', d)]
    c = NON_BARRON_COLORS[di]
    ax.semilogy(times, emp, linestyle='--', color=c, linewidth=1.2,
                label=f'empirical risk, non-Barron, d={d}')
    ax.semilogy(times, pop, linestyle='-',  color=c, linewidth=1.5,
                label=f'population risk, non-Barron, d={d}')

ax.set_xlabel('Time')
ax.set_ylabel('Risk')
ax.set_title('Risk vs time on a logarithmic scale')
ax.set_xlim(0, N_STEPS * ETA)

# legend outside the plot, to the right
ax.legend(fontsize=7, loc='upper left',
          bbox_to_anchor=(1.02, 1), borderaxespad=0)

plt.tight_layout()
plt.savefig('figure2.pdf', bbox_inches='tight')
plt.savefig('figure2.png', dpi=150, bbox_inches='tight')
print("Saved figure2.pdf / figure2.png")
plt.close()
