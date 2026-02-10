# Second-Order Optimization Methods

> **Navigation**: [← 02-Gradient-Descent](../02-Gradient-Descent/) | [Optimization](../) | [04-Constrained-Optimization →](../04-Constrained-Optimization/)

**Files in this section:**
- [examples.ipynb](examples.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Second-order optimization methods use curvature information (Hessian) to achieve faster convergence than first-order methods. While computationally expensive for deep learning, they are essential for convex optimization, hyperparameter tuning, and understanding optimization landscapes.

## Prerequisites

- Gradient descent
- Matrix calculus (Hessians)
- Linear algebra (matrix inverses, eigenvalues)

## Learning Objectives

1. Understand Newton's method
2. Apply quasi-Newton methods (BFGS, L-BFGS)
3. Compare first vs second-order methods
4. Use second-order methods in ML contexts

---

## 1. Newton's Method

### 1.1 The Update Rule

$$\mathbf{w}_{t+1} = \mathbf{w}_t - [\nabla^2 f(\mathbf{w}_t)]^{-1} \nabla f(\mathbf{w}_t)$$

or equivalently:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{H}^{-1} \mathbf{g}$$

where $\mathbf{H} = \nabla^2 f$ (Hessian), $\mathbf{g} = \nabla f$ (gradient).

### 1.2 Intuition: Quadratic Approximation

Newton's method minimizes the local quadratic approximation:

$$f(\mathbf{w}) \approx f(\mathbf{w}_t) + \mathbf{g}^T(\mathbf{w} - \mathbf{w}_t) + \frac{1}{2}(\mathbf{w} - \mathbf{w}_t)^T \mathbf{H} (\mathbf{w} - \mathbf{w}_t)$$

Setting gradient of approximation to zero:
$$\mathbf{g} + \mathbf{H}(\mathbf{w} - \mathbf{w}_t) = 0$$
$$\mathbf{w} = \mathbf{w}_t - \mathbf{H}^{-1}\mathbf{g}$$

```
First-order (GD):              Second-order (Newton):

f(x)                           f(x)
│╲                             │╲
│ ╲←tangent line               │ ╲  ╱ quadratic approx
│  ╲                           │  ╲╱
│   ╲                          │   ●──jumps to minimum
│    ●                         │
└─────── x                     └─────── x

Steps along gradient           Steps to minimum of quadratic
```

### 1.3 Convergence

**Quadratic convergence** near optimum:

$$\|\mathbf{w}_{t+1} - \mathbf{w}^*\| \leq C\|\mathbf{w}_t - \mathbf{w}^*\|^2$$

| Method               | Convergence | Rate            |
| -------------------- | ----------- | --------------- |
| GD (strongly convex) | Linear      | $O(\rho^t)$     |
| Newton               | Quadratic   | $O(\rho^{2^t})$ |

**Example:** If error = 0.1, after one Newton step: error ≈ 0.01

---

## 2. Issues with Pure Newton

### 2.1 Non-Positive-Definite Hessian

At saddle points or maxima, $\mathbf{H}$ may not be positive definite:

- Newton step may go uphill
- May converge to saddle/maximum

**Solution:** Modified Newton methods, trust regions

### 2.2 Computational Cost

| Operation       | Cost             |
| --------------- | ---------------- |
| Compute Hessian | $O(n \cdot d^2)$ |
| Invert Hessian  | $O(d^3)$         |
| Store Hessian   | $O(d^2)$         |

For $d = 10^6$ parameters (small NN): impossible!

### 2.3 Damped Newton

Add a step size:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha_t \mathbf{H}^{-1} \mathbf{g}$$

Use line search to find good $\alpha_t$.

---

## 3. Gauss-Newton Method

### 3.1 For Nonlinear Least Squares

Problem: $\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{r}(\mathbf{w})\|^2$

where $\mathbf{r}(\mathbf{w})$ is the residual vector.

### 3.2 The Approximation

Hessian of least squares:
$$\mathbf{H} = \mathbf{J}^T\mathbf{J} + \sum_i r_i \nabla^2 r_i$$

Gauss-Newton ignores second term:
$$\mathbf{H} \approx \mathbf{J}^T\mathbf{J}$$

where $\mathbf{J} = \nabla \mathbf{r}$ (Jacobian of residuals).

### 3.3 Update Rule

$$\mathbf{w}_{t+1} = \mathbf{w}_t - (\mathbf{J}^T\mathbf{J})^{-1}\mathbf{J}^T\mathbf{r}$$

**Advantages:**

- Only need Jacobian (first derivatives)
- $\mathbf{J}^T\mathbf{J}$ is always positive semidefinite
- Works well when residuals are small

---

## 4. Levenberg-Marquardt

### 4.1 Damped Gauss-Newton

Blend Gauss-Newton with gradient descent:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - (\mathbf{J}^T\mathbf{J} + \lambda\mathbf{I})^{-1}\mathbf{J}^T\mathbf{r}$$

| $\lambda$            | Behavior         |
| -------------------- | ---------------- |
| $\lambda \to 0$      | Gauss-Newton     |
| $\lambda \to \infty$ | Gradient descent |

### 4.2 Adaptive $\lambda$

- If step reduces loss: decrease $\lambda$ (more Newton-like)
- If step increases loss: increase $\lambda$ (more GD-like)

### 4.3 Applications

- Neural network training (historically)
- Curve fitting
- Camera calibration
- Any nonlinear least squares

---

## 5. Quasi-Newton Methods

### 5.1 Key Idea

Approximate the Hessian (or its inverse) from gradient information only.

Build $\mathbf{B}_t \approx \mathbf{H}$ or $\mathbf{H}_t \approx \mathbf{H}^{-1}$ using:

- Gradient differences: $\mathbf{y}_t = \mathbf{g}_{t+1} - \mathbf{g}_t$
- Step differences: $\mathbf{s}_t = \mathbf{w}_{t+1} - \mathbf{w}_t$

### 5.2 Secant Condition

A good approximation should satisfy:

$$\mathbf{B}_{t+1}\mathbf{s}_t = \mathbf{y}_t$$

or equivalently:

$$\mathbf{H}_{t+1}\mathbf{y}_t = \mathbf{s}_t$$

---

## 6. BFGS Algorithm

### 6.1 Inverse Hessian Update

$$\mathbf{H}_{t+1} = \left(\mathbf{I} - \rho_t \mathbf{s}_t \mathbf{y}_t^T\right) \mathbf{H}_t \left(\mathbf{I} - \rho_t \mathbf{y}_t \mathbf{s}_t^T\right) + \rho_t \mathbf{s}_t \mathbf{s}_t^T$$

where $\rho_t = \frac{1}{\mathbf{y}_t^T \mathbf{s}_t}$

### 6.2 BFGS Algorithm

```
Initialize: w₀, H₀ = I
For t = 0, 1, 2, ...:
    1. Compute gradient g_t = ∇f(w_t)
    2. Compute direction p_t = -H_t g_t
    3. Line search: α_t = argmin_α f(w_t + α p_t)
    4. Update: w_{t+1} = w_t + α_t p_t
    5. Compute s_t = w_{t+1} - w_t
    6. Compute y_t = g_{t+1} - g_t
    7. Update H_{t+1} using BFGS formula
```

### 6.3 Properties

- Superlinear convergence (faster than linear, slower than quadratic)
- $\mathbf{H}_t$ stays positive definite if $\mathbf{y}_t^T\mathbf{s}_t > 0$
- Storage: $O(d^2)$
- Per iteration: $O(d^2)$

---

## 7. L-BFGS (Limited-memory BFGS)

### 7.1 Key Insight

Don't store full $\mathbf{H}$; store only last $m$ pairs $\{(\mathbf{s}_i, \mathbf{y}_i)\}$.

Compute $\mathbf{H}_t \mathbf{g}_t$ directly using two-loop recursion.

### 7.2 Two-Loop Recursion

```
Algorithm: Compute H_t g (the search direction)

q = g
for i = t-1, t-2, ..., t-m:
    α_i = ρ_i s_i^T q
    q = q - α_i y_i

r = H₀ q  (typically H₀ = γI where γ = s_{t-1}^T y_{t-1} / y_{t-1}^T y_{t-1})

for i = t-m, t-m+1, ..., t-1:
    β = ρ_i y_i^T r
    r = r + s_i (α_i - β)

return r  (this is H_t g)
```

### 7.3 Memory and Computation

| Aspect        | Full BFGS | L-BFGS (m=10) |
| ------------- | --------- | ------------- |
| Storage       | $O(d^2)$  | $O(md)$       |
| Per iteration | $O(d^2)$  | $O(md)$       |

**For $d = 10^6$:** Full BFGS: 8TB storage; L-BFGS: 80MB

### 7.4 Typical Settings

- $m = 5$ to $20$ (often 10)
- Standard optimizer for large-scale smooth optimization
- Used in scikit-learn, scipy, etc.

---

## 8. Natural Gradient

### 8.1 Motivation

Standard gradient doesn't account for the geometry of parameter space.

For probabilistic models, use **Fisher Information Matrix**:

$$\mathbf{F} = \mathbb{E}\left[\nabla \log p(x|\theta) \nabla \log p(x|\theta)^T\right]$$

### 8.2 Natural Gradient Update

$$\theta_{t+1} = \theta_t - \eta \mathbf{F}^{-1} \nabla \mathcal{L}(\theta_t)$$

### 8.3 Properties

- Invariant to reparameterization
- Faster convergence for certain problems
- Related to second-order methods (Fisher ≈ expected Hessian for certain losses)

### 8.4 Approximations in Deep Learning

- **K-FAC:** Kronecker-factored approximate curvature
- **ADAM:** Can be viewed as diagonal natural gradient approximation

---

## 9. Comparison of Methods

| Method       | Order     | Storage  | Per-Step | Convergence    |
| ------------ | --------- | -------- | -------- | -------------- |
| GD           | 1st       | $O(d)$   | $O(d)$   | Linear         |
| Newton       | 2nd       | $O(d^2)$ | $O(d^3)$ | Quadratic      |
| Gauss-Newton | ~2nd      | $O(d^2)$ | $O(d^3)$ | Quadratic (LS) |
| BFGS         | Quasi-2nd | $O(d^2)$ | $O(d^2)$ | Superlinear    |
| L-BFGS       | Quasi-2nd | $O(md)$  | $O(md)$  | Superlinear    |

```
When to use what:

d < 100?     ─Yes─▶  Newton or BFGS
    │
   No
    │
    ▼
d < 10000?   ─Yes─▶  L-BFGS
    │
   No
    │
    ▼
Deep learning? ─Yes─▶ Adam, SGD+Momentum
    │
   No
    │
    ▼
Consider: L-BFGS, Conjugate Gradient, or specialized
```

---

## 10. Second-Order in Deep Learning

### 10.1 Why Not Widely Used

- $O(d^2)$ or $O(d^3)$ costs prohibitive for millions of parameters
- Stochastic gradients complicate Hessian estimation
- SGD noise may help generalization

### 10.2 Approximations Used

| Method         | Approximation                     |
| -------------- | --------------------------------- |
| Diagonal       | Only diagonal of $\mathbf{H}$     |
| Block-diagonal | Block structure (e.g., per layer) |
| K-FAC          | Kronecker factorization           |
| Shampoo        | Block-diagonal with Kronecker     |

### 10.3 When to Use

- **Hyperparameter optimization** (small d)
- **Final fine-tuning** (L-BFGS on full batch)
- **Research:** Understanding loss landscapes

---

## 11. Trust Region Methods

### 11.1 Idea

Instead of line search, restrict step to a trust region:

$$\min_{\mathbf{p}} m(\mathbf{p}) = f + \mathbf{g}^T\mathbf{p} + \frac{1}{2}\mathbf{p}^T\mathbf{H}\mathbf{p}$$
$$\text{s.t.} \|\mathbf{p}\| \leq \Delta$$

### 11.2 Adaptively Update $\Delta$

Compare actual vs predicted reduction:

$$\rho = \frac{f(\mathbf{w}) - f(\mathbf{w} + \mathbf{p})}{m(\mathbf{0}) - m(\mathbf{p})}$$

- $\rho > 0.75$: Expand trust region
- $\rho < 0.25$: Shrink trust region
- $\rho < 0$: Reject step

---

## 12. Summary

| Concept          | Key Point                                             |
| ---------------- | ----------------------------------------------------- |
| Newton           | Uses Hessian, quadratic convergence                   |
| Gauss-Newton     | For least squares, approx Hessian                     |
| BFGS             | Quasi-Newton, builds $\mathbf{H}^{-1}$ from gradients |
| L-BFGS           | Limited memory, scales to large problems              |
| Natural gradient | Uses Fisher information                               |

**For ML practitioners:**

- Use L-BFGS for convex problems with moderate $d$
- Use Adam/SGD for deep learning
- Consider K-FAC for research

---

## Exercises

1. **Newton's Method**: Apply Newton's method to minimize $f(x) = x^4 - 3x^2 + 2$. Starting from $x_0 = 2$, compute the first 3 iterations.

2. **Hessian Computation**: For logistic regression loss, derive the Hessian matrix and show it is positive semidefinite.

3. **BFGS Update**: Given $s = [1, 0]^T$, $y = [2, 1]^T$, and $H_0 = I$, compute $H_1$ using the BFGS update formula.

4. **L-BFGS Memory**: Explain why L-BFGS is preferred over BFGS for problems with $d > 10000$ parameters. What is the memory complexity of each?

5. **Gauss-Newton**: For a nonlinear least squares problem $\min_w \|r(w)\|^2$ where $r(w) = [w^2 - 1, w - 2]^T$, derive and apply one Gauss-Newton step from $w_0 = 0$.

---

## References

1. Nocedal & Wright - "Numerical Optimization"
2. Boyd & Vandenberghe - "Convex Optimization"
3. Martens - "Deep Learning via Hessian-free Optimization"
