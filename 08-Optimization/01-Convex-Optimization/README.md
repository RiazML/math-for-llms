# Convex Optimization

## Introduction

Convex optimization is the backbone of machine learning. Most ML problems are framed as optimization problems, and convexity provides guarantees that local minima are global minima. Understanding convexity is essential for analyzing convergence, designing algorithms, and understanding when neural networks work well despite non-convexity.

## Prerequisites

- Linear algebra (matrices, eigenvalues)
- Multivariable calculus (gradients, Hessians)
- Basic optimization concepts

## Learning Objectives

1. Understand convex sets and functions
2. Recognize and verify convexity
3. Apply convex optimization to ML
4. Understand duality and KKT conditions
5. Analyze convergence of optimization algorithms

---

## 1. Convex Sets

### 1.1 Definition

A set $C$ is **convex** if for any $x, y \in C$ and $\theta \in [0, 1]$:

$$\theta x + (1 - \theta) y \in C$$

The line segment between any two points lies entirely in the set.

```
Convex Sets:                    Non-Convex Sets:

  ╱───────╲                        ╱─────╲
 │         │                      │       ╲
 │    ●────●                     ●    ╲    │
 │         │                      ╲    ╱──●
  ╲───────╱                        ╲──╱

  ●─────────●                     ●       ●
    (circle)                   (crescent)
                               line crosses outside
```

### 1.2 Examples of Convex Sets

| Set                             | Convex? | Reason                     |
| ------------------------------- | ------- | -------------------------- |
| Hyperplane: $\{x: a^Tx = b\}$   | ✓       | Affine sets are convex     |
| Halfspace: $\{x: a^Tx \leq b\}$ | ✓       | Linear inequality          |
| Ball: $\{x: \|x - c\| \leq r\}$ | ✓       | By definition              |
| Polyhedron: $\{x: Ax \leq b\}$  | ✓       | Intersection of halfspaces |
| Positive semidefinite cone      | ✓       | $\{X: X \succeq 0\}$       |

### 1.3 Operations Preserving Convexity

- **Intersection:** If $C_1, C_2$ convex, then $C_1 \cap C_2$ convex
- **Affine transformation:** $f(C) = \{Ax + b : x \in C\}$ is convex
- **Projection:** Projection of convex set onto subspace is convex

---

## 2. Convex Functions

### 2.1 Definition

A function $f: \mathbb{R}^n \to \mathbb{R}$ is **convex** if its domain is convex and:

$$f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$$

for all $x, y$ in domain and $\theta \in [0, 1]$.

```
Convex Function:                Non-Convex Function:

f(x)                            f(x)
 │      ╱                        │    ╱╲
 │     ╱                         │   ╱  ╲
 │    ●─────●  secant line       │  ●    ●─── secant
 │   ╱      above curve          │ ╱ ╲  ╱    crosses
 │  ╱                            │╱   ╲╱
 │ ╱                             │
 └─────────── x                  └─────────── x
```

**Key property:** Chord lies above the function

### 2.2 First-Order Condition

If $f$ is differentiable, $f$ is convex iff:

$$f(y) \geq f(x) + \nabla f(x)^T(y - x)$$

**Interpretation:** Function lies above all tangent lines/planes

### 2.3 Second-Order Condition

If $f$ is twice differentiable, $f$ is convex iff:

$$\nabla^2 f(x) \succeq 0$$

The Hessian is positive semidefinite everywhere.

### 2.4 Strictly Convex

Strict inequality (< becomes strict, ≥ becomes >):

- Unique global minimum
- $\nabla^2 f(x) \succ 0$ (positive definite) implies strictly convex

---

## 3. Examples of Convex Functions

### 3.1 Common Convex Functions

| Function                        | Domain         | Convex?    |
| ------------------------------- | -------------- | ---------- |
| $f(x) = ax + b$                 | $\mathbb{R}$   | ✓ (affine) |
| $f(x) = x^2$                    | $\mathbb{R}$   | ✓          |
| $f(x) = e^{ax}$                 | $\mathbb{R}$   | ✓          |
| $f(x) = -\log x$                | $x > 0$        | ✓          |
| $f(x) = x\log x$                | $x > 0$        | ✓          |
| $f(x) = \|x\|_p$ for $p \geq 1$ | $\mathbb{R}^n$ | ✓          |
| $f(x) = \max_i x_i$             | $\mathbb{R}^n$ | ✓          |

### 3.2 ML Loss Functions

| Loss          | Formula                    | Convex?  |
| ------------- | -------------------------- | -------- |
| MSE           | $\|y - X\mathbf{w}\|^2$    | ✓ (in w) |
| Cross-entropy | $-\sum y_i\log(\hat{y}_i)$ | ✓        |
| Hinge         | $\max(0, 1-y\cdot f(x))$   | ✓        |
| Log loss      | $\log(1 + e^{-yf(x)})$     | ✓        |

### 3.3 Non-Convex Examples

| Function/Problem     | Why Non-Convex                          |
| -------------------- | --------------------------------------- |
| $f(x) = x^3$         | Inflection point                        |
| $f(x) = \sin(x)$     | Multiple minima                         |
| Neural networks      | Composition with non-linear activations |
| Matrix factorization | Product of matrices                     |

---

## 4. Operations Preserving Convexity

### 4.1 Sum/Average

If $f_1, f_2$ convex, then $f_1 + f_2$ is convex.

$$\text{MSE} + \lambda \|\mathbf{w}\|^2 \text{ is convex}$$

### 4.2 Scalar Multiplication

If $f$ convex and $\alpha \geq 0$, then $\alpha f$ is convex.

### 4.3 Pointwise Maximum

If $f_1, \ldots, f_m$ convex:

$$f(x) = \max_i f_i(x) \text{ is convex}$$

This explains why hinge loss and max pooling are convex!

### 4.4 Composition Rules

| $g$                     | $h$     | $f = g \circ h$ |
| ----------------------- | ------- | --------------- |
| Convex, non-decreasing  | Convex  | Convex          |
| Convex, non-increasing  | Concave | Convex          |
| Concave, non-decreasing | Concave | Concave         |

Example: $e^{x^2}$ is convex ($e^t$ convex increasing, $x^2$ convex)

---

## 5. Convex Optimization Problem

### 5.1 Standard Form

$$
\begin{aligned}
\min_{x} \quad & f(x) \\
\text{s.t.} \quad & g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}
$$

where $f, g_i$ convex, $h_j$ affine.

### 5.2 Key Property

**Any local minimum is a global minimum!**

```
Convex:                        Non-Convex:

f(x)                           f(x)
 │╲                            │╲    ╱╲
 │ ╲                           │ ╲  ╱  ╲    local min
 │  ╲                          │  ╲╱    ╲   (not global)
 │   ●  global min             │   ●     ●
 │    ╲╱                       │        ╱
 └─────────── x                └─────────── x
```

### 5.3 ML as Convex Optimization

**Regularized Linear Regression:**
$$\min_{\mathbf{w}} \|y - X\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|^2$$

**SVM:**
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \max(0, 1-y_i(\mathbf{w}^Tx_i + b))$$

**Logistic Regression:**
$$\min_{\mathbf{w}} \sum_i \log(1 + e^{-y_i \mathbf{w}^T x_i}) + \lambda\|\mathbf{w}\|^2$$

---

## 6. Lagrangian Duality

### 6.1 Lagrangian

$$L(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

where $\lambda_i \geq 0$ (dual variables for inequalities), $\nu_j$ (dual for equalities).

### 6.2 Dual Function

$$d(\lambda, \nu) = \inf_x L(x, \lambda, \nu)$$

Always concave (even if primal not convex).

### 6.3 Weak and Strong Duality

$$d^* = \max_{\lambda \geq 0, \nu} d(\lambda, \nu) \leq p^* = \min_x f(x)$$

**Weak duality:** Always $d^* \leq p^*$

**Strong duality:** $d^* = p^*$ (holds under constraint qualifications like Slater's condition)

### 6.4 Duality Gap

$$p^* - d^* \geq 0$$

Zero gap means we can solve dual instead of primal.

---

## 7. KKT Conditions

### 7.1 The Conditions

For convex problems with strong duality, $(x^*, \lambda^*, \nu^*)$ is optimal iff:

1. **Primal feasibility:** $g_i(x^*) \leq 0$, $h_j(x^*) = 0$
2. **Dual feasibility:** $\lambda_i^* \geq 0$
3. **Complementary slackness:** $\lambda_i^* g_i(x^*) = 0$
4. **Stationarity:** $\nabla f(x^*) + \sum_i \lambda_i^* \nabla g_i(x^*) + \sum_j \nu_j^* \nabla h_j(x^*) = 0$

### 7.2 Complementary Slackness Intuition

$$\lambda_i^* g_i(x^*) = 0$$

Either constraint is inactive ($g_i(x^*) < 0$) or its multiplier is zero ($\lambda_i^* = 0$).

Only binding constraints "matter" at optimum.

---

## 8. ML Applications of Convexity

### 8.1 Why Convexity Matters in ML

| Advantage            | Explanation                           |
| -------------------- | ------------------------------------- |
| Global optimum       | Gradient descent finds best solution  |
| Efficient algorithms | Polynomial time complexity            |
| Theoretical analysis | Convergence guarantees                |
| Regularization       | Adds convex term, preserves convexity |

### 8.2 SVM Dual Problem

Primal: $\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{w}\|^2$ s.t. $y_i(\mathbf{w}^Tx_i + b) \geq 1$

Dual: $\max_{\alpha} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j}\alpha_i\alpha_j y_i y_j x_i^T x_j$

s.t. $\alpha_i \geq 0$, $\sum_i \alpha_i y_i = 0$

**Advantage:** Kernel trick works in dual!

### 8.3 Neural Networks: Non-Convex but Effective

- Loss landscape is highly non-convex
- Multiple local minima
- Yet SGD finds good solutions!
- Research: "loss surfaces are flat" near minima

---

## 9. Checking Convexity

### 9.1 Verification Methods

1. **Definition:** Show $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$
2. **Second derivative:** Show $\nabla^2 f(x) \succeq 0$
3. **Composition rules:** Build from known convex functions
4. **Restriction to line:** $g(t) = f(x + tv)$ convex for all $x, v$

### 9.2 Algorithm

```
Is f convex?
│
├── Compute Hessian H(x)
│   │
│   ├── H ≽ 0 for all x? → Convex ✓
│   │
│   └── H ≻ 0 for all x? → Strictly Convex ✓
│
└── Can't compute Hessian?
    │
    └── Use composition rules or definition
```

---

## 10. Important Convex Problems

### 10.1 Linear Programming (LP)

$$\min_x c^Tx \text{ s.t. } Ax \leq b$$

Objective and constraints both linear.

### 10.2 Quadratic Programming (QP)

$$\min_x \frac{1}{2}x^TPx + q^Tx \text{ s.t. } Ax \leq b$$

$P \succeq 0$ for convexity.

### 10.3 Semidefinite Programming (SDP)

$$\min_X \text{tr}(CX) \text{ s.t. } \text{tr}(A_iX) = b_i, X \succeq 0$$

Optimization over positive semidefinite matrices.

---

## 11. Summary

| Concept         | Key Point                         |
| --------------- | --------------------------------- |
| Convex set      | Line segment stays in set         |
| Convex function | Chord above curve                 |
| First-order     | Above all tangent planes          |
| Second-order    | Hessian ≽ 0                       |
| Global optimum  | Local = global for convex         |
| Duality         | Alternative formulation           |
| KKT             | Necessary & sufficient conditions |

**For ML:**

- Convex losses → guaranteed convergence
- Regularization → maintains convexity
- Deep learning → non-convex but works empirically

---

## References

1. Boyd & Vandenberghe - "Convex Optimization"
2. Nocedal & Wright - "Numerical Optimization"
3. Boyd - "Convex Optimization" course (Stanford)
