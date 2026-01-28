# Optimization Theory and Constrained Optimization

## Introduction

Optimization is the heart of machine learning. Every training algorithm seeks to minimize (or maximize) an objective function. This section covers the theoretical foundations of optimization, including constrained optimization using Lagrange multipliers and KKT conditions.

## Prerequisites

- Gradients and partial derivatives
- Jacobians and Hessians
- Linear algebra

## Learning Objectives

1. Understand optimization problem formulations
2. Master Lagrange multipliers for equality constraints
3. Learn KKT conditions for inequality constraints
4. Apply optimization theory to ML problems

---

## 1. Optimization Problem Formulations

### Unconstrained Optimization

$$\min_{\mathbf{x}} f(\mathbf{x})$$

Find $\mathbf{x}^*$ such that $f(\mathbf{x}^*)$ is minimal.

### Constrained Optimization

**With equality constraints:**
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g_i(\mathbf{x}) = 0, \; i = 1, \ldots, m$$

**With inequality constraints:**
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } h_j(\mathbf{x}) \leq 0, \; j = 1, \ldots, p$$

---

## 2. Necessary Conditions for Optimality

### First-Order Necessary Condition

At a local minimum $\mathbf{x}^*$ of $f$:

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

Points where $\nabla f = 0$ are called **critical points** or **stationary points**.

### Second-Order Necessary Condition

At a local minimum $\mathbf{x}^*$:

$$\mathbf{H}(\mathbf{x}^*) \succeq 0 \quad \text{(positive semi-definite)}$$

### Second-Order Sufficient Condition

If $\nabla f(\mathbf{x}^*) = 0$ and $\mathbf{H}(\mathbf{x}^*) \succ 0$, then $\mathbf{x}^*$ is a **strict local minimum**.

```
Classification of Critical Points:
│
├── ∇f = 0 (necessary)
│
├── Check Hessian H:
│   │
│   ├── All eigenvalues > 0 → Local minimum
│   │
│   ├── All eigenvalues < 0 → Local maximum
│   │
│   ├── Mixed signs → Saddle point
│   │
│   └── Some zero eigenvalues → Inconclusive
```

---

## 3. Lagrange Multipliers (Equality Constraints)

### The Method

For the problem:
$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t. } g(\mathbf{x}) = 0$$

**Lagrangian:**
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$$

**Optimal conditions:**
$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f + \lambda \nabla g = \mathbf{0}$$
$$\nabla_{\lambda} \mathcal{L} = g(\mathbf{x}) = 0$$

### Geometric Intuition

At the optimum, $\nabla f$ is parallel to $\nabla g$ (constraint gradient).

```
Level curves of f(x,y)
           ╱│╲
          ╱ │ ╲
         ╱  │  ╲
        ╱   │   ╲    Constraint g(x,y) = 0
       ╱    │    ╲        │
      ╱     │     ╲       │
     ╱      *──────●──────│──
    ╱       │ ∇f  ╱       │
   ╱        │↓   ╱        │
  ╱         │   ╱         │
            │  ╱ ∇g
            │ ↓

At optimum: ∇f = -λ∇g (parallel)
```

### Multiple Equality Constraints

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x})$$

---

## 4. KKT Conditions (Inequality Constraints)

### Problem Setup

$$\min_{\mathbf{x}} f(\mathbf{x})$$
$$\text{s.t. } g_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$
$$\phantom{\text{s.t. }} h_j(\mathbf{x}) \leq 0, \quad j = 1, \ldots, p$$

### The Lagrangian

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \mu_j h_j(\mathbf{x})$$

### KKT Conditions (Necessary for Optimality)

1. **Stationarity:** $\nabla_{\mathbf{x}} \mathcal{L} = 0$
2. **Primal feasibility:** $g_i(\mathbf{x}) = 0$, $h_j(\mathbf{x}) \leq 0$
3. **Dual feasibility:** $\mu_j \geq 0$
4. **Complementary slackness:** $\mu_j h_j(\mathbf{x}) = 0$

### Complementary Slackness Intuition

```
For each inequality constraint h_j(x) ≤ 0:

Either:
├── h_j(x) < 0 (constraint inactive)
│   └── Then μ_j = 0 (multiplier zero)
│
└── h_j(x) = 0 (constraint active/tight)
    └── Then μ_j ≥ 0 (multiplier non-negative)
```

---

## 5. Convex Optimization

### Convex Function Definition

$f$ is convex if for all $\mathbf{x}, \mathbf{y}$ and $t \in [0, 1]$:

$$f(t\mathbf{x} + (1-t)\mathbf{y}) \leq tf(\mathbf{x}) + (1-t)f(\mathbf{y})$$

### Second-Order Condition

$f$ is convex $\Leftrightarrow$ $\mathbf{H}(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$

### Why Convexity Matters

```
Convex Function:          Non-Convex Function:

    ╲     ╱                    ╱╲
     ╲   ╱                    ╱  ╲    ╱╲
      ╲ ╱                    ╱    ╲  ╱  ╲
       *                    *      ╲╱    ╲
   global min          local min    saddle  global min

Convex: local min = global min
Non-convex: many local minima, saddle points
```

### Convex Optimization Properties

- Local minimum is global minimum
- KKT conditions are sufficient (not just necessary)
- Efficient algorithms exist (polynomial time)

---

## 6. Common ML Optimization Problems

### Linear Regression (Unconstrained)

$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

**Solution:** $\nabla_{\mathbf{w}} = 2\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y}) = 0$
$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### Ridge Regression (Constrained)

$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 \quad \text{s.t. } \|\mathbf{w}\|^2 \leq t$$

**Equivalent Lagrangian form:**
$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2$$

**Solution:** $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

### SVM Dual Problem

**Primal:**
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$$

**Dual (using Lagrange multipliers):**
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$
$$\text{s.t. } \alpha_i \geq 0, \sum_i \alpha_i y_i = 0$$

### PCA (Constrained Maximization)

$$\max_{\mathbf{v}} \mathbf{v}^T\mathbf{C}\mathbf{v} \quad \text{s.t. } \|\mathbf{v}\|^2 = 1$$

**Lagrangian:** $\mathcal{L} = \mathbf{v}^T\mathbf{C}\mathbf{v} - \lambda(\mathbf{v}^T\mathbf{v} - 1)$

**Optimal condition:** $\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$ (eigenvalue problem!)

---

## 7. Duality

### Lagrangian Dual

**Dual function:**
$$g(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})$$

**Dual problem:**
$$\max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} g(\boldsymbol{\lambda}, \boldsymbol{\mu}) \quad \text{s.t. } \boldsymbol{\mu} \geq 0$$

### Weak Duality

$$g(\boldsymbol{\lambda}, \boldsymbol{\mu}) \leq p^*$$

The dual optimal is always a lower bound on primal optimal.

### Strong Duality

Under certain conditions (e.g., Slater's condition for convex problems):

$$d^* = p^*$$

Dual and primal have the same optimal value.

---

## 8. Algorithms Overview

### First-Order Methods

| Method           | Update Rule                                            | Convergence     |
| ---------------- | ------------------------------------------------------ | --------------- |
| Gradient Descent | $\mathbf{x} \leftarrow \mathbf{x} - \eta\nabla f$      | $O(1/k)$        |
| SGD              | $\mathbf{x} \leftarrow \mathbf{x} - \eta\nabla f_i$    | $O(1/\sqrt{k})$ |
| Momentum         | $\mathbf{v} \leftarrow \beta\mathbf{v} - \eta\nabla f$ | Accelerated     |
| Adam             | Adaptive learning rates                                | Popular in DL   |

### Second-Order Methods

| Method              | Update Rule                                                  | Convergence  |
| ------------------- | ------------------------------------------------------------ | ------------ |
| Newton's            | $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{H}^{-1}\nabla f$ | Quadratic    |
| Quasi-Newton (BFGS) | Approximate Hessian                                          | Super-linear |
| L-BFGS              | Memory-efficient BFGS                                        | Large-scale  |

---

## 9. Summary Tables

### Optimization Problem Types

| Type          | Constraints             | Solution Method      |
| ------------- | ----------------------- | -------------------- |
| Unconstrained | None                    | $\nabla f = 0$       |
| Equality      | $g(\mathbf{x}) = 0$     | Lagrange multipliers |
| Inequality    | $h(\mathbf{x}) \leq 0$  | KKT conditions       |
| Convex        | Convex $f$, constraints | Efficient algorithms |

### Key Conditions

| Condition          | Formula                                                            | Meaning                  |
| ------------------ | ------------------------------------------------------------------ | ------------------------ |
| Stationarity       | $\nabla f + \sum \lambda_i \nabla g_i + \sum \mu_j \nabla h_j = 0$ | Gradient balance         |
| Primal feasibility | $g_i = 0$, $h_j \leq 0$                                            | Constraints satisfied    |
| Dual feasibility   | $\mu_j \geq 0$                                                     | Non-negative multipliers |
| Complementarity    | $\mu_j h_j = 0$                                                    | Tight or zero            |

### ML Applications

```
ML Problem          │ Optimization Formulation
────────────────────┼────────────────────────────────
Linear Regression   │ min ||Xw - y||²
Logistic Regression │ min -Σ[y log p + (1-y)log(1-p)]
Ridge/Lasso         │ min loss + λ·regularizer
SVM                 │ min ½||w||² s.t. margin ≥ 1
PCA                 │ max variance s.t. ||v|| = 1
```

---

## 10. Practical Tips

### Checking Optimality

1. Compute gradient - should be near zero
2. Check Hessian eigenvalues for min/max/saddle
3. Verify constraint satisfaction

### Debugging

- Plot loss curve - should decrease
- Check gradient numerically
- Monitor constraint violations

### Convexity Verification

- For $f(x)$: check $f'' \geq 0$
- For $f(\mathbf{x})$: check $\mathbf{H} \succeq 0$
- Many ML losses are convex (MSE, cross-entropy)

---

## Exercises

1. Use Lagrange multipliers to find the maximum of $f(x,y) = xy$ on the unit circle
2. Derive the dual problem for soft-margin SVM
3. Show that MSE loss is convex
4. Apply KKT conditions to $\ell_1$-regularized regression (Lasso)
5. Prove that for convex problems, local minimum = global minimum

---

## References

1. Boyd & Vandenberghe - "Convex Optimization"
2. Nocedal & Wright - "Numerical Optimization"
3. Bertsekas - "Nonlinear Programming"
