"""
Figure 3 replication – saves each panel as its own file:
  figure3a.png  –  loss curves,   m=30   (mildly over-parametrized)
  figure3b.png  –  coefficients,  m=30
  figure3c.png  –  loss curves,   m=2000 (highly over-parametrized)
  figure3d.png  –  coefficients,  m=2000
"""

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Model primitives
# ─────────────────────────────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0.0, z)

def relu_grad(z):
    return (z > 0.0).astype(float)

def make_data(n, d, seed):
    """n samples x ~ Uniform([0,1]^d),  target f*(x) = ReLU(x_1)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n, d))
    return X, relu(X[:, 0])

def forward(X, a, B):
    """
    Two-layer NN with conventional scaling:
        f(x) = sum_j  a_j * ReLU(b_j^T x)
    Returns prediction f (n,), pre-activations Z (n,m), activations H (n,m).
    """
    Z = X @ B.T          # (n, m)  –  pre-activations
    H = relu(Z)          # (n, m)  –  post-activations
    f = H @ a            # (n,)    –  output
    return f, Z, H

def mse(f, y):
    return 0.5 * np.mean((f - y) ** 2)

def init_params(m, d, seed):
    """
    Conventional-scaling initialisation:
        a_j  ~ N(0, 1/m)      outer layer
        b_j  ~ N(0, I/d)      inner layer (each entry ~ N(0,1/d))
    Both NN and RF model start from the same draw (fair comparison).
    """
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1.0 / np.sqrt(m), m)
    B = rng.normal(0, 1.0 / np.sqrt(d), (m, d))
    return a, B

# ─────────────────────────────────────────────────────────────────────────────
# GD update rules
# ─────────────────────────────────────────────────────────────────────────────

def gd_step_nn(X, y, a, B, lr):
    """
    Full-batch GD on both a (outer) and B (inner).

    Loss L = (1/2n) ||f - y||^2

    Gradients derived via chain rule:
        dL/da_j = (1/n) sum_i  (f_i - y_i) * H_{ij}
                = H^T r          where r = (f-y)/n

        dL/db_j = (1/n) sum_i  (f_i - y_i) * a_j * 1[z_{ij}>0] * x_i
                = a_j * X^T (r * 1[Z_j > 0])
    """
    n = len(y)
    f, Z, H = forward(X, a, B)
    r = (f - y) / n                              # (n,)  scaled residual

    da = H.T @ r                                 # (m,)
    mask = relu_grad(Z)                          # (n, m)  indicator 1[z>0]
    # (d, n) @ (n, m)  ->  (d, m), then scale each col j by a_j, transpose
    dB = (X.T @ (r[:, None] * mask) * a[None, :]).T   # (m, d)

    return a - lr * da, B - lr * dB

def gd_step_rf(X, y, a, B, lr):
    """
    Random Feature model: B is frozen at its initial value.
    Only the outer layer a is trained (linear regression on fixed features).

        dL/da_j = H^T r    (same as NN, but B never changes)
    """
    n = len(y)
    f, _, H = forward(X, a, B)
    r = (f - y) / n
    return a - lr * (H.T @ r), B   # B returned unchanged

# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def run(m, n, d, lr, n_iters, log_every, seed=0):
    """
    Train both the NN and the RF model from the same initialisation.
    Returns logged losses and the final neuron coefficients a_j ||b_j||.
    """
    X_tr, y_tr = make_data(n, d, seed)
    X_te, y_te = make_data(5000, d, seed + 1)   # held-out test set

    a_nn, B_nn = init_params(m, d, seed)
    a_rf, B_rf = a_nn.copy(), B_nn.copy()        # identical start

    iters = []
    nn_tr, nn_te = [], []
    rf_tr, rf_te = [], []

    for t in range(n_iters + 1):
        if t % log_every == 0:
            iters.append(t)
            f, _, _ = forward(X_tr, a_nn, B_nn);  nn_tr.append(mse(f, y_tr))
            f, _, _ = forward(X_te, a_nn, B_nn);  nn_te.append(mse(f, y_te))
            f, _, _ = forward(X_tr, a_rf, B_rf);  rf_tr.append(mse(f, y_tr))
            f, _, _ = forward(X_te, a_rf, B_rf);  rf_te.append(mse(f, y_te))
        if t < n_iters:
            a_nn, B_nn = gd_step_nn(X_tr, y_tr, a_nn, B_nn, lr)
            a_rf, B_rf = gd_step_rf(X_tr, y_tr, a_rf, B_rf, lr)

    # path-norm-style coefficient: a_j * ||b_j||
    # captures how much each neuron contributes to the function
    coef_nn = a_nn * np.linalg.norm(B_nn, axis=1)
    coef_rf = a_rf * np.linalg.norm(B_rf, axis=1)

    return iters, nn_tr, nn_te, rf_tr, rf_te, coef_nn, coef_rf


# ─────────────────────────────────────────────────────────────────────────────
# Run experiments
# ─────────────────────────────────────────────────────────────────────────────

print("Training (a): m=30,   n=200, d=19  [mildly over-parametrized] ...")
top = run(m=30,   n=200, d=19, lr=1e-3, n_iters=50000, log_every=500)

print("Training (c): m=2000, n=200, d=20  [highly over-parametrized] ...")
bot = run(m=2000, n=200, d=20, lr=1e-3, n_iters=20000, log_every=200)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss(iters, nn_tr, nn_te, rf_tr, rf_te, m, n, d, ax=None, fname=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        created_fig = True
    ax.semilogy(iters, nn_tr, color='steelblue', linestyle='-',  label='NN, train')
    ax.semilogy(iters, nn_te, color='steelblue', linestyle='--', label='NN, test')
    ax.semilogy(iters, rf_tr, color='firebrick', linestyle='-',  label='RF, train')
    ax.semilogy(iters, rf_te, color='firebrick', linestyle='--', label='RF, test')
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Loss')
    ax.set_title(f'm={m}, n={n}, d={d}')
    ax.legend(fontsize=9)
    if created_fig:
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {fname}")


def plot_coef(coef_nn, coef_rf, m, n, d, ax=None, fname=None):
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        created_fig = True
    idx = np.arange(1, m + 1)

    if m <= 50:
        ax.scatter(idx, coef_rf, s=80, marker='*', color='firebrick',
                   label='rf', alpha=0.85, zorder=4)
        ax.vlines(idx, 0, coef_nn, colors='steelblue', linewidth=1.2, alpha=0.7)
        ax.scatter(idx, coef_nn, s=40, color='steelblue',
                   label='nn', alpha=0.95, zorder=5)
    else:
        ax.vlines(idx, 0, coef_nn, colors='steelblue', linewidth=0.4, alpha=0.5, zorder=2)
        ax.scatter(idx, coef_nn, s=4, color='steelblue',
                   label='nn', alpha=0.7, zorder=3)
        ax.scatter(idx, coef_rf, s=12, marker='*', color='firebrick',
                   label='rf', alpha=0.9, zorder=5)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Index of neurons')
    ax.set_ylabel(r'$a_j \|\mathbf{b}_j\|$')
    ax.set_title(f'm={m}, n={n}, d={d}')
    ax.legend(fontsize=9)
    if created_fig:
        plt.tight_layout()
        if fname is not None:
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {fname}")


iters, nn_tr, nn_te, rf_tr, rf_te, coef_nn, coef_rf = bot
# ─────────────────────────────────────────────────────────────────────────────
# Create one combined figure with 4 panels (2x2)
# ─────────────────────────────────────────────────────────────────────────────

fig, axs = plt.subplots(2, 2, figsize=(11, 8))

# top-left: loss (m=30)
iters, nn_tr, nn_te, rf_tr, rf_te, coef_nn, coef_rf = top
plot_loss(iters, nn_tr, nn_te, rf_tr, rf_te, m=30, n=200, d=19, ax=axs[0, 0])

# top-right: coefficients (m=30)
plot_coef(coef_nn, coef_rf, m=30, n=200, d=19, ax=axs[0, 1])

# bottom-left: loss (m=2000)
iters, nn_tr, nn_te, rf_tr, rf_te, coef_nn, coef_rf = bot
plot_loss(iters, nn_tr, nn_te, rf_tr, rf_te, m=2000, n=200, d=20, ax=axs[1, 0])

# bottom-right: coefficients (m=2000)
plot_coef(coef_nn, coef_rf, m=2000, n=200, d=20, ax=axs[1, 1])

plt.tight_layout()
outname = 'figure3_combined.png'
plt.savefig(outname, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {outname}")

print("Done.")
