# Numerical Optimization

> **Navigation**: [← 02-Numerical-Linear-Algebra](../02-Numerical-Linear-Algebra/) | [Numerical Methods](../) | [04-Interpolation-and-Approximation →](../04-Interpolation-and-Approximation/)

**Files in this section:**
- [examples.ipynb](examples.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Overview

Numerical optimization provides algorithms to find minima of functions when analytical solutions are unavailable. This is the computational backbone of machine learning, powering everything from linear regression to deep neural networks.

## Prerequisites

- Calculus (gradients, Hessians)
- Linear algebra (matrix operations)
- Convexity concepts

## Learning Objectives

- Understand gradient-based optimization algorithms
- Implement and compare first-order methods
- Apply second-order methods appropriately
- Handle constraints in optimization

---

## 1. Problem Formulation

### Unconstrained Optimization

$$\min_{x \in \mathbb{R}^n} f(x)$$

### Constrained Optimization

$$\min_{x} f(x) \quad \text{subject to} \quad g_i(x) \leq 0, \quad h_j(x) = 0$$

### Optimality Conditions

**First-order necessary condition**:
$$\nabla f(x^*) = 0$$

**Second-order sufficient condition**:
$$\nabla^2 f(x^*) \succ 0 \quad \text{(positive definite)}$$

---

## 2. Gradient Descent

### Basic Algorithm

$$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$$

where $\alpha_k$ is the **learning rate** (step size).

### Convergence Analysis

For $L$-smooth, $\mu$-strongly convex $f$:

$$f(x_k) - f(x^*) \leq \left(1 - \frac{\mu}{L}\right)^k (f(x_0) - f(x^*))$$

**Condition number**: $\kappa = L/\mu$
**Optimal learning rate**: $\alpha = 2/(L + \mu)$

### Learning Rate Selection

| Strategy     | Formula                                               | Notes                |
| ------------ | ----------------------------------------------------- | -------------------- |
| Constant     | $\alpha_k = \alpha$                                   | Simple but sensitive |
| Decay        | $\alpha_k = \alpha_0 / (1 + kt)$                      | Slower convergence   |
| Line search  | $\alpha_k = \arg\min_\alpha f(x_k - \alpha \nabla f)$ | Expensive per step   |
| Backtracking | Armijo condition                                      | Good balance         |

### Backtracking Line Search (Armijo)

```
α = 1
while f(x - α∇f) > f(x) - c·α·||∇f||²:
    α = ρ·α  # typically ρ = 0.5, c = 0.1
```

---

## 3. Momentum Methods

### Classical Momentum

$$v_{k+1} = \beta v_k + \nabla f(x_k)$$
$$x_{k+1} = x_k - \alpha v_{k+1}$$

- $\beta$: momentum coefficient (typically 0.9)
- Accelerates convergence in consistent gradient directions

### Nesterov Accelerated Gradient (NAG)

$$v_{k+1} = \beta v_k + \nabla f(x_k - \alpha \beta v_k)$$
$$x_{k+1} = x_k - \alpha v_{k+1}$$

"Look-ahead" gradient improves convergence rate.

**Optimal rate for convex functions**:
$$f(x_k) - f(x^*) = O(1/k^2)$$

vs. $O(1/k)$ for standard gradient descent.

---

## 4. Adaptive Learning Rate Methods

### AdaGrad

Adapts learning rate based on accumulated squared gradients:

$$G_k = G_{k-1} + g_k \odot g_k$$
$$x_{k+1} = x_k - \frac{\alpha}{\sqrt{G_k + \epsilon}} \odot g_k$$

**Pros**: Good for sparse gradients
**Cons**: Learning rate keeps decreasing

### RMSprop

Uses exponential moving average:

$$v_k = \beta v_{k-1} + (1-\beta) g_k^2$$
$$x_{k+1} = x_k - \frac{\alpha}{\sqrt{v_k + \epsilon}} g_k$$

- $\beta$: decay rate (typically 0.9)

### Adam (Adaptive Moment Estimation)

Combines momentum and adaptive learning rates:

$$m_k = \beta_1 m_{k-1} + (1-\beta_1) g_k$$
$$v_k = \beta_2 v_{k-1} + (1-\beta_2) g_k^2$$

Bias correction:
$$\hat{m}_k = \frac{m_k}{1 - \beta_1^k}, \quad \hat{v}_k = \frac{v_k}{1 - \beta_2^k}$$

Update:
$$x_{k+1} = x_k - \frac{\alpha}{\sqrt{\hat{v}_k} + \epsilon} \hat{m}_k$$

**Default hyperparameters**: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### AdamW (Adam with Decoupled Weight Decay)

$$x_{k+1} = x_k - \alpha\left(\frac{\hat{m}_k}{\sqrt{\hat{v}_k} + \epsilon} + \lambda x_k\right)$$

Weight decay is separate from gradient update.

---

## 5. Second-Order Methods

### Newton's Method

$$x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

**Pros**: Quadratic convergence near optimum
**Cons**:

- $O(n^3)$ per iteration (matrix inverse)
- Requires positive definite Hessian

### Damped Newton

$$x_{k+1} = x_k - \alpha_k [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$$

Use line search to ensure descent.

### Gauss-Newton (for least squares)

For $f(x) = \frac{1}{2}\|r(x)\|^2$:

$$H \approx J^T J$$

$$x_{k+1} = x_k - (J^T J)^{-1} J^T r(x_k)$$

Avoids computing second derivatives.

### Levenberg-Marquardt

$$x_{k+1} = x_k - (J^T J + \lambda I)^{-1} J^T r(x_k)$$

- $\lambda$ large: gradient descent
- $\lambda$ small: Gauss-Newton
- Adaptive $\lambda$ based on progress

---

## 6. Quasi-Newton Methods

Approximate the Hessian or its inverse using gradient information.

### BFGS (Broyden-Fletcher-Goldfarb-Shanno)

Maintain approximate inverse Hessian $H_k \approx [\nabla^2 f(x_k)]^{-1}$:

$$H_{k+1} = \left(I - \rho_k s_k y_k^T\right) H_k \left(I - \rho_k y_k s_k^T\right) + \rho_k s_k s_k^T$$

where:

- $s_k = x_{k+1} - x_k$
- $y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$
- $\rho_k = 1/(y_k^T s_k)$

**Update rule**:
$$x_{k+1} = x_k - \alpha_k H_k \nabla f(x_k)$$

### L-BFGS (Limited memory BFGS)

Store only last $m$ pairs $(s_k, y_k)$ instead of full $n \times n$ matrix.

**Memory**: $O(mn)$ instead of $O(n^2)$

Two-loop recursion computes $H_k \nabla f$ implicitly.

---

## 7. Stochastic Optimization

### Stochastic Gradient Descent (SGD)

For $f(x) = \frac{1}{N}\sum_{i=1}^N f_i(x)$:

$$x_{k+1} = x_k - \alpha_k \nabla f_{i_k}(x_k)$$

where $i_k$ is randomly sampled.

**Variance**: $\mathbb{E}[\|\nabla f_i - \nabla f\|^2]$

### Mini-batch SGD

Average over batch $B_k$:
$$g_k = \frac{1}{|B_k|} \sum_{i \in B_k} \nabla f_i(x_k)$$

**Variance reduction**: $\text{Var}(g_k) = \text{Var}(\nabla f_i) / |B_k|$

### Learning Rate Schedules

| Schedule    | Formula                                                  | Use Case             |
| ----------- | -------------------------------------------------------- | -------------------- |
| Step decay  | $\alpha_t = \alpha_0 \cdot \gamma^{\lfloor t/s \rfloor}$ | Standard training    |
| Exponential | $\alpha_t = \alpha_0 \cdot e^{-\lambda t}$               | Smooth decay         |
| Cosine      | $\alpha_t = \frac{\alpha_0}{2}(1 + \cos(\pi t/T))$       | Fine-tuning          |
| Warmup      | Linear increase then decay                               | Large batch training |

### Gradient Clipping

Prevent exploding gradients:

$$g_k \leftarrow \min\left(1, \frac{c}{\|g_k\|}\right) g_k$$

---

## 8. Constrained Optimization

### Lagrangian Method

For $\min f(x)$ s.t. $g(x) \leq 0$, $h(x) = 0$:

$$L(x, \lambda, \mu) = f(x) + \lambda^T g(x) + \mu^T h(x)$$

**KKT conditions**:

1. $\nabla_x L = 0$ (stationarity)
2. $g(x) \leq 0$, $h(x) = 0$ (primal feasibility)
3. $\lambda \geq 0$ (dual feasibility)
4. $\lambda_i g_i(x) = 0$ (complementary slackness)

### Projected Gradient Descent

$$x_{k+1} = \text{Proj}_C(x_k - \alpha_k \nabla f(x_k))$$

Projection onto constraint set $C$.

### Penalty Methods

Transform constrained to unconstrained:
$$\min_x f(x) + \frac{\rho}{2}\|h(x)\|^2 + \frac{\rho}{2}\|\max(0, g(x))\|^2$$

Increase $\rho$ as optimization progresses.

### Augmented Lagrangian

$$L_\rho(x, \mu) = f(x) + \mu^T h(x) + \frac{\rho}{2}\|h(x)\|^2$$

Alternating minimization over $x$ and dual update for $\mu$.

---

## 9. Convex Optimization Techniques

### Proximal Gradient Method

For $f(x) = g(x) + h(x)$ where $g$ is smooth, $h$ is not:

$$x_{k+1} = \text{prox}_{\alpha h}(x_k - \alpha \nabla g(x_k))$$

**Proximal operator**:
$$\text{prox}_h(y) = \arg\min_x \left(h(x) + \frac{1}{2}\|x-y\|^2\right)$$

### Common Proximal Operators

| $h(x)$                       | $\text{prox}_{\alpha h}(y)$        |
| ---------------------------- | ---------------------------------- |
| $\lambda\|x\|_1$             | soft-threshold$(y, \alpha\lambda)$ |
| $\mathbb{I}_{C}(x)$          | $\text{Proj}_C(y)$                 |
| $\frac{\lambda}{2}\|x\|_2^2$ | $\frac{y}{1+\alpha\lambda}$        |

### ADMM (Alternating Direction Method of Multipliers)

For $\min f(x) + g(z)$ s.t. $Ax + Bz = c$:

$$x_{k+1} = \arg\min_x \left(f(x) + \frac{\rho}{2}\|Ax + Bz_k - c + u_k\|^2\right)$$
$$z_{k+1} = \arg\min_z \left(g(z) + \frac{\rho}{2}\|Ax_{k+1} + Bz - c + u_k\|^2\right)$$
$$u_{k+1} = u_k + Ax_{k+1} + Bz_{k+1} - c$$

---

## 10. Convergence Rates Summary

| Method   | Smooth Convex   | Strongly Convex       | Non-convex  |
| -------- | --------------- | --------------------- | ----------- |
| GD       | $O(1/k)$        | $O(\rho^k)$, $\rho<1$ | Local min   |
| Nesterov | $O(1/k^2)$      | $O(\rho^k)$, faster   | -           |
| Newton   | -               | Quadratic (local)     | Local       |
| SGD      | $O(1/\sqrt{k})$ | $O(1/k)$              | Local       |
| Adam     | Similar to SGD  | -                     | Widely used |

---

## 11. Practical Considerations

### Hyperparameter Tuning

**Learning rate**: Most important hyperparameter

- Start with standard values (0.001 for Adam)
- Use learning rate finder (Leslie Smith)
- Grid/random search

**Batch size**: Trade-off

- Larger: more stable gradients, more memory
- Smaller: implicit regularization, faster iterations

### Debugging Optimization

1. **Monitor loss curve**: Should decrease
2. **Check gradient norms**: Not exploding/vanishing
3. **Validate on held-out data**: Detect overfitting
4. **Visualize parameter updates**: Ensure reasonable scale

### Common Issues

| Problem                | Symptom                  | Solution                     |
| ---------------------- | ------------------------ | ---------------------------- |
| Learning rate too high | Loss oscillates/diverges | Reduce α                     |
| Learning rate too low  | Very slow progress       | Increase α                   |
| Bad initialization     | Training stuck           | Better init (Xavier, He)     |
| Saddle points          | Plateau in loss          | Momentum, noise              |
| Vanishing gradients    | No parameter updates     | Batch norm, skip connections |

---

## Summary

### Method Selection Guide

```
Start with Adam
↓
Not converging? → Try SGD + momentum + scheduler
↓
Small dataset? → Consider L-BFGS
↓
Constrained? → Projected GD or ADMM
↓
Non-smooth regularizer? → Proximal methods
```

### Key Takeaways

1. **Adam** is a robust default for most deep learning
2. **SGD + momentum** often achieves better final performance
3. **Learning rate scheduling** is crucial for good results
4. **Second-order methods** shine for small, well-conditioned problems
5. **Convex optimization** has strong theoretical guarantees

---

## Exercises

### Exercise 1: Line Search Methods
Implement backtracking line search with Armijo condition. Compare with exact line search for quadratic functions.

### Exercise 2: Gradient Descent Variants
Compare vanilla gradient descent, momentum, and Nesterov accelerated gradient on ill-conditioned quadratics.

### Exercise 3: L-BFGS Implementation
Implement L-BFGS with m=5 memory. Compare convergence with gradient descent on a logistic regression problem.

### Exercise 4: Convergence Rate Analysis
Empirically verify the linear convergence rate of gradient descent on strongly convex functions.

### Exercise 5: Trust Region Methods
Implement a basic trust region method. Compare with line search on the Rosenbrock function.

---

## References

1. Nocedal & Wright - "Numerical Optimization"
2. Boyd & Vandenberghe - "Convex Optimization"
3. Goodfellow et al. - "Deep Learning" (Ch. 8)
4. Ruder - "An Overview of Gradient Descent Optimization Algorithms"
