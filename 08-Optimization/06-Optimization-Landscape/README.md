# Optimization Landscape Analysis

> **Navigation**: [← 05-Stochastic-Optimization](../05-Stochastic-Optimization/) | [Optimization](../) | [07-Adaptive-Learning-Rate →](../07-Adaptive-Learning-Rate/)

**Files in this section:**
- [examples.ipynb](examples.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Understanding the geometry of loss surfaces is crucial for deep learning. The optimization landscape determines which minima we can reach, how fast we converge, and how well our models generalize. This topic explores the mathematical tools for analyzing loss landscapes.

## Prerequisites

- Multivariate calculus (gradients, Hessians)
- Linear algebra (eigenvalues)
- Convex optimization basics

## Learning Objectives

1. Characterize critical points (minima, maxima, saddle points)
2. Analyze loss surface geometry
3. Understand the role of curvature
4. Connect landscape properties to generalization

---

## 1. Critical Points

### 1.1 Definition

A point $\mathbf{w}^*$ is a **critical point** if:

$$\nabla f(\mathbf{w}^*) = \mathbf{0}$$

### 1.2 Classification by Hessian

| Type          | Hessian Eigenvalues  | Description       |
| ------------- | -------------------- | ----------------- |
| Local minimum | All $\lambda_i > 0$  | Positive definite |
| Local maximum | All $\lambda_i < 0$  | Negative definite |
| Saddle point  | Mixed signs          | Indefinite        |
| Degenerate    | Some $\lambda_i = 0$ | Semi-definite     |

```
2D Examples:

Local Minimum       Local Maximum       Saddle Point
    ↘  ↙               ↗  ↖              ↗  ↙
     ●                  ●                 ●
    ↗  ↖               ↘  ↙              ↘  ↗

All directions       All directions     Some up, some down
curve up             curve down
```

### 1.3 Second-Order Conditions

**Necessary:** If $\mathbf{w}^*$ is a local minimum, then $\nabla^2 f(\mathbf{w}^*) \succeq 0$

**Sufficient:** If $\nabla f(\mathbf{w}^*) = 0$ and $\nabla^2 f(\mathbf{w}^*) \succ 0$, then $\mathbf{w}^*$ is a local minimum

---

## 2. Curvature and Condition Number

### 2.1 Condition Number

$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

where $\lambda_{\max}$ and $\lambda_{\min}$ are the largest and smallest eigenvalues of the Hessian.

### 2.2 Effect on Optimization

| $\kappa$           | Geometry  | Optimization      |
| ------------------ | --------- | ----------------- |
| $\kappa \approx 1$ | Spherical | Fast convergence  |
| $\kappa \gg 1$     | Elongated | Slow, zigzag path |

```
Well-conditioned (κ ≈ 1):    Ill-conditioned (κ >> 1):

   ╱───╲                         ╱─────────╲
  │     │                       │           │
  │  ●  │                       │     ●     │
  │     │                       │           │
   ╲───╱                         ╲─────────╱

Gradient points                 Gradient points away
toward minimum                  from minimum
```

### 2.3 Curvature in Different Directions

Local curvature in direction $\mathbf{v}$:

$$\kappa_{\mathbf{v}} = \frac{\mathbf{v}^T \nabla^2 f \mathbf{v}}{\|\mathbf{v}\|^2}$$

---

## 3. Saddle Points

### 3.1 Prevalence in High Dimensions

In high-dimensional spaces, saddle points vastly outnumber local minima.

**Intuition:** At a critical point, each eigenvalue is independently positive or negative. The probability of all $d$ eigenvalues being positive decreases exponentially with $d$.

$$P(\text{local min}) \approx \left(\frac{1}{2}\right)^d$$

### 3.2 Saddle Point Problem

Gradient descent can get stuck near saddle points:

- Gradient is small
- Can take exponentially long to escape

```
Saddle Point Dynamics:

f(x,y) = x² - y²

      ↑ y
      │    ╲     ╱
      │     ╲   ╱
      ├──────●──────→ x
      │     ╱   ╲
      │    ╱     ╲

Attracted along y-axis
Repelled along x-axis
GD moves slowly near saddle
```

### 3.3 Escaping Saddles

Methods to escape saddle points:

- **Negative curvature exploitation:** Follow directions of negative eigenvalues
- **Noise:** SGD noise helps escape
- **Cubic regularization:** Add cubic terms to Newton step

---

## 4. Loss Surface Geometry

### 4.1 Convex vs Non-convex

| Convex Loss                          | Non-convex Loss         |
| ------------------------------------ | ----------------------- |
| Single global minimum                | Multiple local minima   |
| GD converges to global               | GD converges to local   |
| Examples: Linear/logistic regression | Examples: Deep networks |

### 4.2 Flatness and Sharpness

**Sharp minimum:** High curvature, large $\lambda_{\max}$
$$f(\mathbf{w} + \boldsymbol{\epsilon}) - f(\mathbf{w}) \approx \frac{1}{2}\boldsymbol{\epsilon}^T \mathbf{H} \boldsymbol{\epsilon} \text{ (large)}$$

**Flat minimum:** Low curvature, small $\lambda_{\max}$
$$f(\mathbf{w} + \boldsymbol{\epsilon}) - f(\mathbf{w}) \approx \frac{1}{2}\boldsymbol{\epsilon}^T \mathbf{H} \boldsymbol{\epsilon} \text{ (small)}$$

```
Sharp vs Flat Minima:

Sharp:                    Flat:
  │                         │
  │╲      ╱                 │  ╲___╱
  │ ╲    ╱                  │
  │  ╲  ╱                   │
  │   ●                     │   ●
──┴────────                ──┴────────

High curvature            Low curvature
Small perturbation        Robust to
→ large loss change       perturbations
```

### 4.3 Flatness and Generalization

**Hypothesis:** Flat minima generalize better

**Intuition:**

- Training and test distributions differ slightly
- Flat minima are robust to this distribution shift
- Sharp minima are sensitive to small changes

---

## 5. Mode Connectivity

### 5.1 Definition

Two minima are **mode connected** if there exists a path between them with low loss.

### 5.2 Linear Mode Connectivity

$$f((1-t)\mathbf{w}_1 + t\mathbf{w}_2) \approx (1-t)f(\mathbf{w}_1) + tf(\mathbf{w}_2)$$

If loss is approximately constant along the line, minima are linearly connected.

### 5.3 Non-linear Paths

Often minima are connected by curved (non-linear) low-loss paths:

```
Loss landscape view:

    High loss
       ╱╲
      ╱  ╲
     ╱    ╲
●───╱──────╲───●
w₁  Linear   w₂
    path (high barrier)

    ●─────────●
   w₁  curved w₂
       path (low loss)
```

---

## 6. The Loss Surface of Neural Networks

### 6.1 Key Observations

1. **Many global minima:** Overparameterized networks have many solutions
2. **Saddle points dominate:** Most critical points are saddles
3. **Connected minima:** Good minima often form connected regions
4. **Lottery ticket hypothesis:** Sparse subnetworks can achieve good performance

### 6.2 Empirical Findings

| Property         | Finding                     |
| ---------------- | --------------------------- |
| Bad local minima | Rare for large networks     |
| Saddle points    | Abundant, but SGD escapes   |
| Initialization   | Critical for final solution |
| Width            | Wider = smoother landscape  |

### 6.3 The Role of Overparameterization

More parameters → more directions → easier to find descent paths

**Neural Tangent Kernel (NTK) regime:** Very wide networks behave like kernel methods with convex optimization.

---

## 7. Hessian Analysis

### 7.1 Hessian Spectrum

The eigenvalue distribution of the Hessian reveals landscape structure:

```
Typical Hessian spectrum of neural networks:

Number of
eigenvalues
    │
    │█
    │██
    │███
    │████
    │█████
    └──────────────────→ eigenvalue
    0     bulk    outliers
          (small)  (large)
```

### 7.2 Key Findings

- **Bulk:** Most eigenvalues are small (near zero)
- **Outliers:** Few large positive eigenvalues
- **Negative eigenvalues:** Present early in training (saddle points)

### 7.3 Implications

- Low effective dimension of the loss surface
- Only a few directions matter for optimization
- Justifies low-rank approximations (K-FAC, etc.)

---

## 8. Gradient Flow and Implicit Bias

### 8.1 Gradient Flow

Continuous-time limit of gradient descent:

$$\frac{d\mathbf{w}}{dt} = -\nabla f(\mathbf{w})$$

### 8.2 Implicit Regularization

Gradient flow/descent implicitly regularizes:

- **Linear regression:** Finds minimum norm solution
- **Matrix factorization:** Prefers low-rank solutions
- **Classification:** Maximizes margin (in certain limits)

### 8.3 Example: Linear Networks

For deep linear network $f(\mathbf{x}) = \mathbf{W}_L \cdots \mathbf{W}_1 \mathbf{x}$:

Gradient descent on $\mathbf{W}_1, \ldots, \mathbf{W}_L$ implicitly prefers low-rank solutions for the product $\mathbf{W} = \mathbf{W}_L \cdots \mathbf{W}_1$.

---

## 9. Visualization Techniques

### 9.1 1D/2D Slices

Project high-dimensional loss onto 1D or 2D subspace:

$$f(\alpha, \beta) = \mathcal{L}(\mathbf{w}^* + \alpha \mathbf{d}_1 + \beta \mathbf{d}_2)$$

Common choices for directions $\mathbf{d}_1, \mathbf{d}_2$:

- Random directions
- Principal Hessian eigenvectors
- Filter-normalized directions

### 9.2 Filter Normalization

Scale directions by filter norms to account for scale invariance:

$$\mathbf{d}'_{i,j} = \frac{\|\mathbf{w}^*_{i,j}\|}{\|\mathbf{d}_{i,j}\|} \mathbf{d}_{i,j}$$

### 9.3 Interpolation Plots

Plot loss along line between two solutions:

$$f(t) = \mathcal{L}((1-t)\mathbf{w}_1 + t\mathbf{w}_2), \quad t \in [0, 1]$$

---

## 10. Practical Implications

### 10.1 For Training

| Landscape Feature | Training Strategy       |
| ----------------- | ----------------------- |
| Saddle points     | Use momentum, noise     |
| Ill-conditioning  | Use adaptive methods    |
| Flat regions      | Large learning rates OK |
| Sharp minima      | Small learning rates    |

### 10.2 For Generalization

| Finding Flat Minima          | Technique                     |
| ---------------------------- | ----------------------------- |
| Large batch + low LR         | May find sharp minima         |
| Small batch                  | Noise helps find flat         |
| Sharpness-aware minimization | Explicitly minimize sharpness |
| Weight averaging             | Average over training         |

### 10.3 Sharpness-Aware Minimization (SAM)

$$\min_{\mathbf{w}} \max_{\|\boldsymbol{\epsilon}\| \leq \rho} f(\mathbf{w} + \boldsymbol{\epsilon})$$

Seeks parameters that are robust to perturbations.

---

## 11. Summary

| Concept           | Key Point                                                  |
| ----------------- | ---------------------------------------------------------- |
| Critical points   | Gradient = 0; classify by Hessian                          |
| Condition number  | $\kappa = \lambda_{\max}/\lambda_{\min}$; affects GD speed |
| Saddle points     | Prevalent in high-D; SGD escapes                           |
| Flatness          | May correlate with generalization                          |
| Mode connectivity | Good minima often connected                                |
| Hessian spectrum  | Most eigenvalues small; few outliers                       |

**Key insight:** Understanding the loss landscape helps explain why deep learning works and guides algorithm design.

---

## Exercises

1. **Critical Point Classification**: For $f(x,y) = x^3 - 3xy^2$, find all critical points and classify each as local min, max, or saddle point using the Hessian.

2. **Condition Number Impact**: Create a 2D quadratic with condition number $\kappa = 100$. Visualize gradient descent paths and count iterations to convergence vs $\kappa = 1$.

3. **Saddle Point Dynamics**: For $f(x,y) = x^2 - y^2$, starting from $(0.1, 0.1)$, simulate gradient descent. Does it escape the saddle? How does noise help?

4. **Flatness Measure**: Propose and implement a measure of minima flatness. Test it on a small neural network trained with different batch sizes.

5. **Mode Connectivity**: Train two neural networks with different random seeds. Interpolate between their weights and plot the loss along the path. Are they linearly connected?

---

## References

1. Goodfellow et al. - "Qualitatively Characterizing Neural Network Optimization Problems"
2. Li et al. - "Visualizing the Loss Landscape of Neural Nets"
3. Keskar et al. - "On Large-Batch Training for Deep Learning"
4. Draxler et al. - "Essentially No Barriers in Neural Network Energy Landscape"
