# Experiments

Numerical replications accompanying the paper review:

> **E, Ma, Wojtowytsch, Wu (2020).** *Towards a Mathematical Understanding of Neural Network-Based Machine Learning: What We Know and What We Don't.* [arXiv:2009.10713](http://arxiv.org/abs/2009.10713)

Each subfolder corresponds to one **model class** and the **paper that is the primary source for those experiments**.  
The mapping follows Section 5 of E et al. 2020, which studies convergence and generalisation along five axes:

| Model | Convergence | Generalisation | Experiment source | Folder |
|---|:---:|:---:|---|---|
| Random feature model | ✓ | ✓ | Ma et al. 2020 – *Slow Deterioration* | [`random_feature__ma2020/`](#random-feature-model--ma-et-al-2020) |
| Two-layer NN (mean-field) | ✓ | – | E et al. 2020 – *Mathematical Understanding* | [`twolayer_meanfield__e2020/`](#two-layer-nn-mean-field--e-et-al-2020) |
| Two-layer NN (conventional) | ✓ | ✓ | Ma – *Quenching Activation* | [`twolayer_conventional__ma_quenching/`](#two-layer-nn-conventional--ma-quenching) |
| Adam | ✓ | ✓ | Ma – *Qualitative Study of Adam* | [`adam__ma_qualitative/`](#adam--ma-qualitative-study) |
| Global minima: GD vs SGD | – | ✓ | Zhu et al. 2019 – *Anisotropic Noise* | [`gd_sgd__zhu2019/`](#gd-vs-sgd--zhu-et-al-2019) |

---

## Random feature model — Ma et al. 2020

**Folder:** [`random_feature__ma2020/`](random_feature__ma2020/)  
**Paper:** Ma, Wu – *Slow Deterioration of the Generalization Error of the Random Feature Model* (2020) [paper](https://proceedings.mlr.press/v107/ma20a.html) 

### Theory (§ 5.4)
The random feature model fixes the hidden weights **B** at initialisation and only trains the output weights **a** — linear regression on fixed random features.  
Training dynamics are governed by a linear ODE driven by the Gram matrix Φ = relu(XB^T):

$$a(t) = \sum_{i:\,\lambda_i>0} \frac{1-e^{-\lambda_i^2 t/(mn)}}{\lambda_i}\,(u_i^T y)\,v_i$$

Three distinct training regimes appear:
1. Test error decreases (large eigenvalues dominate)
2. Test error stays small (stable plateau)
3. Test error deteriorates (small eigenvalues near interpolation threshold contribute)

The double-descent spike occurs at the **interpolation threshold** m = n, where the smallest eigenvalue of the Gram matrix approaches zero and the minimum-norm solution becomes erratic.

### Files
| File | Content |
|---|---|
| `figure6.py` | Training / test loss curves for rf model vs nn model (two panel: m=30, m=2000) |
| `figure6_weights.py` | Weight magnitude plot aⱼ‖bⱼ‖ per neuron index |
| `figure7.py` | Double-descent heat map of test error over (log m, log n) — rf model |

---

## Two-layer NN, mean-field — E et al. 2020

**Folder:** [`twolayer_meanfield__e2020/`](twolayer_meanfield__e2020/)  
**Paper:** E, Ma, Wojtowytsch, Wu – *Towards a Mathematical Understanding* (2020) [arXiv:2009.10713](http://arxiv.org/abs/2009.10713)

### Theory (§ 5.1)
Mean-field scaling writes the network as

$$f_m(x) = \frac{1}{m}\sum_j a_j\,\sigma(w_j^T x), \qquad u_j = (a_j, w_j)$$

and lifts GD on the finite-particle system to a **mean-field PDE** on the distribution ρ of neurons:

$$\partial_t \rho = \nabla(\rho\,\nabla V), \qquad V = \frac{\delta\hat{\mathcal{R}}_n}{\delta\rho}$$

Under two conditions on ρ₀ — support on the cone {|a|² ≤ |w|²} and omni-directionality — the risk $\mathcal{R}(\rho_t)$ converges to the **global infimum**.  
This guarantee does not hold for conventional scaling.

### Files
| File | Content |
|---|---|
| `figure2.py` | Convergence curves comparing mean-field vs conventional GD |

---

## Two-layer NN, conventional — Ma, Quenching

**Folder:** [`twolayer_conventional__ma_quenching/`](twolayer_conventional__ma_quenching/)  
**Paper:** Ma, Wu – *The Quenching-Activation Behavior of the Gradient Descent Dynamics for Two-layer Neural Network Models* [paper link](https://arxiv.org/pdf/2006.14450)

### Theory (§ 5.2)
Conventional scaling:

$$f_m(x;\,a,B) = \sum_j a_j\,\sigma(b_j^T x), \qquad a_j(0)\sim\mathcal{N}(0,\beta^2),\quad b_j(0)\sim\mathcal{N}(0,I/d)$$

**Good news (highly over-parameterised regime):** exponential convergence to a global minimum of the empirical risk.

**Bad news (generalisation):** when m is large, parameters barely move from initialisation (lazy training / NTK regime) and the GD solution is **no better than the random feature model**. With fewer weights than data (m < n), a *quenching* process occurs: only a few neurons remain active and the rest are quenched to near zero, concentrating the representation in sparse neurons — visible in the weight-magnitude plots of Figure 3.

### Files
| File | Content |
|---|---|
| `figure3.py` | Training curves + weight magnitudes aⱼ‖bⱼ‖ (four-panel: m=30/2000, underfit/overfit) |
| `figure4.py` | Heat map of test error in (log m, log n) space — conventional vs mean-field scaling |
| `figure5.py` | Test error & path norm vs log(width) — nn, rf analytical, rf GD; double-descent curve |

---

## Adam — Ma, Qualitative Study

**Folder:** [`adam__ma_qualitative/`](adam__ma_qualitative/)  
**Paper:** Ma, Ying – *A Qualitative Study of the Dynamic Behavior of Adaptive Gradient Algorithms* [paper link](https://proceedings.mlr.press/v145/ma22a/ma22a.pdf)

### Theory (§ 5.6)
Adam with learning rate η and parameters β₁ = 1 − b·η (first moment), β₂ = 1 − a·η (second moment) exhibits three regimes depending on the ratio a/b:

1. **Spike regime** (b >> a, e.g. default β₁=0.9, β₂=0.999 → a=1, b=100): loss makes periodic large spikes while drifting downward — fast initial progress but unstable.
2. **Small oscillation regime** (a ≈ b): stable convergence with bounded oscillations near a low loss value.
3. **Large oscillation / divergence regime** (a >> b): loss diverges or stays large.

### Files
| File | Content |
|---|---|
| `figure10.py` | Training curve for the default (spike) regime; divergence analysis across a/b ratios |
| `figure11.py` | Three-panel: (left) spike training curve; (middle) loss heat map over (a,b) grid; (right) regime classification map |

---

## GD vs SGD — Zhu et al. 2019

**Folder:** [`gd_sgd__zhu2019/`](gd_sgd__zhu2019/)  
**Paper:** Zhu, Wu, Yu, Wu, Ma – *The Anisotropic Noise in Stochastic Gradient Descent: Its Behavior of Escaping from Sharp Minima and Regularization Effects* (2019) [paper link](https://arxiv.org/abs/1803.00195)

### Theory (§ 5.5)
Different optimisers select qualitatively different global minima.  
Key quantities (defined for leading eigenvector **v** of ∇²L):

- **Sharpness** a = λ_max(∇²L): GD clusters near the stability boundary a = 2/η.
- **Non-uniformity** s = std of per-sample curvatures aᵢ = vᵀ∇²Lᵢv: SGD with batch B is predicted to select minima with s ≤ C·√B.

**Escape phenomenon (Figure 8):** GD (full batch) converges to a *sharp* minimum that memorises corrupted labels. SGD, initialised at that sharp minimum, escapes it and settles at a *flatter* minimum with better test accuracy. The escape works because SGD's noise is anisotropic — it preferentially pushes parameters along directions of high curvature.

Note on loss function: the sharpness/non-uniformity analysis requires the Hessian to be well-defined at the minimum. MSE loss satisfies this; cross-entropy does not (Hessian vanishes), which is why the experiments in Figure 9 use MSE.

### Files
| File | Content |
|---|---|
| `figure8.py` | Escape phenomenon: GD → sharp min, then SGD escapes → flat min (FashionMNIST, LeNet) |
| `figure9.py` | Sharpness–non-uniformity scatter plot for GD and SGD with various batch sizes (FashionMNIST, VGG-small) |

---

## Running the experiments

Each script is self-contained.  Run from inside the subfolder so that cache files (`.npy`, `.npz`, `.pt`) land next to the script:

```bash
cd experiments/twolayer_conventional__ma_quenching
python figure5.py
```

Scripts auto-detect MPS (Apple Silicon) → CUDA → CPU and choose dtype / tolerance accordingly.  
Cached results are loaded automatically on subsequent runs; delete the cache files to force a rerun.
