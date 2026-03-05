# Optimization Theory and Constrained Optimization

> **Navigation**: [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/) | [02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/) | [03-Chain-Rule-and-Backpropagation](../03-Chain-Rule-and-Backpropagation/)

## Overview

Optimization is the **heart of machine learning**. Every training algorithm seeks to minimize (or maximize) an objective function. This section covers the theoretical foundations of optimization, including constrained optimization using Lagrange multipliers and KKT conditions.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION IN ML                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Unconstrained                    Constrained                           │
│  ─────────────                    ───────────                           │
│                                                                          │
│  min L(θ)                         min L(θ)                              │
│   θ                                θ                                    │
│                                   s.t. constraints                      │
│                                                                          │
│  Examples:                        Examples:                             │
│  • Neural network                 • SVM (margin = 1)                    │
│  • Linear regression              • Ridge (||w|| ≤ t)                   │
│  • Logistic regression            • PCA (||v|| = 1)                     │
│                                   • Trust region                        │
│                                                                          │
│  Method: ∇L = 0                   Method: Lagrange / KKT                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/)
- [02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/)
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

```
Constrained vs Unconstrained:

Unconstrained                 Equality Constraint           Inequality Constraint
─────────────                 ───────────────────           ─────────────────────

   ╲     ╱                         │                          ┌─────────────┐
    ╲   ╱                          │  g(x)=0                  │  h(x) ≤ 0   │
     ╲ ╱                           │                          │  (feasible  │
      ● ← minimum                  ●─┼─ ← min on curve         │   region)   │
     ╱ ╲                           │                          │      ●      │
    ╱   ╲                          │                          │  ← min     │
                                                              └─────────────┘

Free to move                  Constrained to curve          Constrained to region
anywhere                      (lower-dimensional)           (might be on boundary)
```

---

## 2. Necessary Conditions for Optimality

### First-Order Necessary Condition

At a local minimum $\mathbf{x}^*$ of $f$:

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

Points where $\nabla f = 0$ are called **critical points** or **stationary points**.

> **⚠️ Warning**: Not all critical points are minima! Could be maximum or saddle point.

### Second-Order Necessary Condition

At a local minimum $\mathbf{x}^*$:

$$\mathbf{H}(\mathbf{x}^*) \succeq 0 \quad \text{(positive semi-definite)}$$

### Second-Order Sufficient Condition

If $\nabla f(\mathbf{x}^*) = 0$ **AND** $\mathbf{H}(\mathbf{x}^*) \succ 0$, then $\mathbf{x}^*$ is a **strict local minimum**.

```
Classification of Critical Points:

                    ∇f = 0 ?
                       │
            Yes ───────┴─────── No
             │                   │
             ▼                   ▼
      Check Hessian H        Not a critical
             │                  point
    ┌────────┼────────┐
    │        │        │
    ▼        ▼        ▼
 All λ>0  Mixed λ  All λ<0
    │        │        │
    ▼        ▼        ▼
 MINIMUM  SADDLE   MAXIMUM
```

---

## 3. Lagrange Multipliers (Equality Constraints)

### The Problem

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g(\mathbf{x}) = 0$$

### The Method

**Lagrangian:**
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$$

**Optimal conditions:**
$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f + \lambda \nabla g = \mathbf{0}$$
$$\nabla_{\lambda} \mathcal{L} = g(\mathbf{x}) = 0$$

> **💡 Intuition**: At the optimum, the gradient of $f$ must be parallel to the gradient of $g$ (the constraint).

### Geometric Intuition

```
Lagrange Multiplier Geometry:

Level curves of f(x,y)           ∇f
           ╱│╲                    ↓
          ╱ │ ╲                 ╱
         ╱  │  ╲    Constraint g(x,y) = 0
        ╱   │   ╲      │
       ╱  ∇f│    ╲     │
      ╱    ↓│     ╲    │
     ╱──────●───────╲──┼── ← Optimal point
    ╱       │↑       ╲ │
   ╱        │∇g       ╲│
            │          ╲

At optimum:
• ∇f is parallel to ∇g
• ∇f = -λ∇g for some λ
• Can't improve f while staying on constraint!
```

### Multiple Equality Constraints

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x})$$

### Example: Maximum on a Circle

Find max of $f(x,y) = x + y$ on the circle $x^2 + y^2 = 1$.

$$\mathcal{L} = x + y + \lambda(x^2 + y^2 - 1)$$

Setting gradients to zero:
$$\frac{\partial \mathcal{L}}{\partial x} = 1 + 2\lambda x = 0$$
$$\frac{\partial \mathcal{L}}{\partial y} = 1 + 2\lambda y = 0$$
$$x^2 + y^2 = 1$$

Solution: $x = y = \frac{1}{\sqrt{2}}$, $\lambda = -\frac{1}{\sqrt{2}}$

---

## 4. KKT Conditions (Inequality Constraints)

### Problem Setup

$$\min_{\mathbf{x}} f(\mathbf{x})$$
$$\text{s.t. } g_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$
$$\phantom{\text{s.t. }} h_j(\mathbf{x}) \leq 0, \quad j = 1, \ldots, p$$

### The Lagrangian

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) + \sum_{j=1}^p \mu_j h_j(\mathbf{x})$$

### KKT Conditions (Karush-Kuhn-Tucker)

```
┌───────────────────────────────────────────────────────────────┐
│                    KKT CONDITIONS                             │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  1. STATIONARITY:                                             │
│     ∇f + Σλᵢ∇gᵢ + Σμⱼ∇hⱼ = 0                                  │
│                                                                │
│  2. PRIMAL FEASIBILITY:                                       │
│     gᵢ(x) = 0    for all i                                    │
│     hⱼ(x) ≤ 0    for all j                                    │
│                                                                │
│  3. DUAL FEASIBILITY:                                         │
│     μⱼ ≥ 0       for all j                                    │
│                                                                │
│  4. COMPLEMENTARY SLACKNESS:                                  │
│     μⱼhⱼ(x) = 0  for all j                                    │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Complementary Slackness Intuition

```
For each inequality constraint hⱼ(x) ≤ 0:

CASE 1: Constraint INACTIVE            CASE 2: Constraint ACTIVE
────────────────────────               ──────────────────────────

  hⱼ(x) < 0                            hⱼ(x) = 0
  (strictly inside feasible)           (on the boundary)
            │                                   │
            ▼                                   ▼
  Then μⱼ = 0                          Then μⱼ ≥ 0
  (constraint doesn't matter)          (constraint is "pushing")

Either hⱼ = 0 OR μⱼ = 0 (or both)
Product: μⱼhⱼ = 0 always!
```

---

## 5. Convex Optimization

### Convex Function Definition

$f$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $t \in [0, 1]$:

$$f(t\mathbf{x} + (1-t)\mathbf{y}) \leq tf(\mathbf{x}) + (1-t)f(\mathbf{y})$$

> **💡 Intuition**: The line segment between any two points on the graph lies ABOVE the graph.

### Second-Order Condition

$f$ is convex $\Leftrightarrow$ $\mathbf{H}(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$

### Why Convexity Matters

```
Convex Function:                    Non-Convex Function:

        ╲       ╱                         ╱╲     ╱╲
         ╲     ╱                         ╱  ╲   ╱  ╲
          ╲   ╱                         ╱    ╲ ╱    ╲
           ╲ ╱                         ╱   local    ╲
            ●  ← GLOBAL minimum       ●──────●────── ●
                                   local  saddle  global
                                    min           min

✓ Local minimum = Global minimum    ✗ Local ≠ Global (hard!)
✓ Gradient descent converges        ✗ Can get stuck
✓ Efficient algorithms exist        ✗ NP-hard in general
```

### Convex Optimization Properties

1. **Local minimum is global minimum**
2. **KKT conditions are sufficient** (not just necessary)
3. **Efficient algorithms exist** (polynomial time)

> **🔑 Good News**: Many ML problems have convex subproblems (linear regression, logistic regression, SVM)!

---

## 6. Common ML Optimization Problems

### Linear Regression (Unconstrained, Convex)

$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

**Solution:** Set $\nabla_{\mathbf{w}} = 2\mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y}) = 0$
$$\mathbf{w}^* = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

### Ridge Regression (Constrained → Regularized)

**Constrained form:**
$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 \quad \text{s.t. } \|\mathbf{w}\|^2 \leq t$$

**Equivalent Lagrangian form:**
$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2 + \lambda \|\mathbf{w}\|^2$$

**Solution:** $\mathbf{w}^* = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{y}$

```
Constrained ↔ Regularized:

Constrained:                    Regularized:
─────────────                   ─────────────

min loss                        min loss + λ·penalty
s.t. ||w||² ≤ t                 

     ╱────────╲                      ╱────────╲
    ╱  ● min   ╲                    ╱  contours╲
   │   in       │                  │  of loss + │
   │  circle    │                  │  λ||w||²   │
    ╲          ╱                    ╲     ●    ╱
     ╲────────╱                      ╲────────╱
      ||w||≤t                       unconstrained

Same solution for corresponding λ and t!
```

### SVM (Constrained → Dual Problem)

**Primal:**
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$$

**Dual (using Lagrange multipliers):**
$$\max_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$
$$\text{s.t. } \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

> **💡 Key Insight**: The dual only depends on inner products $\mathbf{x}_i^T\mathbf{x}_j$ — this enables the kernel trick!

### PCA (Constrained Maximization)

$$\max_{\mathbf{v}} \mathbf{v}^T\mathbf{C}\mathbf{v} \quad \text{s.t. } \|\mathbf{v}\|^2 = 1$$

**Lagrangian:** $\mathcal{L} = \mathbf{v}^T\mathbf{C}\mathbf{v} - \lambda(\mathbf{v}^T\mathbf{v} - 1)$

**Optimal condition:** 
$$\nabla_{\mathbf{v}} \mathcal{L} = 2\mathbf{C}\mathbf{v} - 2\lambda\mathbf{v} = 0$$
$$\mathbf{C}\mathbf{v} = \lambda\mathbf{v}$$

> **🔑 PCA = Eigenvalue problem!** Principal components are eigenvectors of the covariance matrix.

---

## 7. Duality

### Lagrangian Dual

**Dual function:**
$$g(\boldsymbol{\lambda}, \boldsymbol{\mu}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, \boldsymbol{\mu})$$

**Dual problem:**
$$\max_{\boldsymbol{\lambda}, \boldsymbol{\mu}} g(\boldsymbol{\lambda}, \boldsymbol{\mu}) \quad \text{s.t. } \boldsymbol{\mu} \geq 0$$

### Weak Duality (Always True)

$$d^* \leq p^*$$

The dual optimal is always a **lower bound** on primal optimal.

### Strong Duality (Under Certain Conditions)

$$d^* = p^*$$

Holds for convex problems satisfying **Slater's condition** (strictly feasible point exists).

```
Duality:

         p* (primal optimal)
            │
    ────────┼────────────
            │
            │   Duality gap
            │   (= 0 if strong duality)
            │
    ────────┼────────────
            │
         d* (dual optimal)

Weak duality: d* ≤ p* always
Strong duality: d* = p* for convex + Slater
```

---

## 8. Algorithms Overview

### First-Order Methods

| Method | Update Rule | Convergence |
|--------|-------------|-------------|
| Gradient Descent | $\mathbf{x} \leftarrow \mathbf{x} - \eta\nabla f$ | $O(1/k)$ |
| SGD | $\mathbf{x} \leftarrow \mathbf{x} - \eta\nabla f_i$ | $O(1/\sqrt{k})$ |
| Momentum | $\mathbf{v} \leftarrow \beta\mathbf{v} - \eta\nabla f$ | Accelerated |
| Adam | Adaptive learning rates | Popular in DL |

### Second-Order Methods

| Method | Update Rule | Convergence |
|--------|-------------|-------------|
| Newton's | $\mathbf{x} \leftarrow \mathbf{x} - \mathbf{H}^{-1}\nabla f$ | Quadratic |
| Quasi-Newton (BFGS) | Approximate Hessian | Super-linear |
| L-BFGS | Memory-efficient BFGS | Large-scale |

```
First vs Second Order:

First Order (use ∇f):            Second Order (use ∇f and H):
─────────────────────            ─────────────────────────────

 │                                │
 │  ●───→───→───→●                │  ●─────────→●
 │  Many small steps              │  Fewer smart steps
 │                                │
 Cost: O(n) per step              Cost: O(n³) per step
 Works for large n                Hard for large n

Trade-off: computation vs convergence speed
```

---

## 9. Summary Tables

### Optimization Problem Types

| Type | Constraints | Solution Method |
|------|-------------|-----------------|
| Unconstrained | None | $\nabla f = 0$ |
| Equality | $g(\mathbf{x}) = 0$ | Lagrange multipliers |
| Inequality | $h(\mathbf{x}) \leq 0$ | KKT conditions |
| Convex | Convex $f$, constraints | Efficient algorithms |

### KKT Conditions Summary

```
┌──────────────────────────────────────────────────────────────┐
│                    KKT REFERENCE                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Condition           │  Formula           │  Meaning          │
│  ──────────────────  │  ─────────────────  │  ──────────────── │
│  Stationarity        │  ∇ₓℒ = 0           │  Gradient balance │
│  Primal feasibility  │  g=0, h≤0          │  Constraints met  │
│  Dual feasibility    │  μ ≥ 0             │  Non-neg. mult.   │
│  Complementarity     │  μh = 0            │  Active or zero   │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### ML Applications Summary

```
┌───────────────────────────────────────────────────────────────┐
│                    ML OPTIMIZATION PROBLEMS                   │
├───────────────────────┬───────────────────────────────────────┤
│  ML Problem           │  Optimization Formulation             │
├───────────────────────┼───────────────────────────────────────┤
│  Linear Regression    │  min ||Xw - y||²                      │
│  Logistic Regression  │  min -Σ[y log p + (1-y)log(1-p)]      │
│  Ridge                │  min loss + λ||w||²                   │
│  Lasso                │  min loss + λ||w||₁                   │
│  SVM                  │  min ½||w||²  s.t. margin ≥ 1         │
│  PCA                  │  max variance  s.t. ||v|| = 1         │
│  Neural Networks      │  min L(θ) (non-convex!)               │
└───────────────────────┴───────────────────────────────────────┘
```

---

## 10. Practical Tips

### Checking Optimality

1. Compute gradient — should be near zero
2. Check Hessian eigenvalues for min/max/saddle
3. Verify constraint satisfaction
4. For constrained: check KKT conditions

### Debugging Optimization

```
Debugging Checklist:

□ Plot loss curve — should decrease
□ Check gradient numerically
□ Monitor constraint violations
□ Try different learning rates
□ Check for numerical issues (nan, overflow)
□ Verify convexity if assuming it
```

### Convexity Verification

- For $f(x)$: check $f'' \geq 0$
- For $f(\mathbf{x})$: check $\mathbf{H} \succeq 0$ (all eigenvalues ≥ 0)

**Common convex functions in ML:**
- MSE loss: $\|\mathbf{y} - \mathbf{\hat{y}}\|^2$
- Cross-entropy: $-\sum y_i \log \hat{y}_i$
- L2 regularization: $\|\mathbf{w}\|^2$
- Log-sum-exp: $\log(\sum e^{x_i})$

---

## Exercises

1. Use Lagrange multipliers to find the maximum of $f(x,y) = xy$ on the unit circle
2. Derive the dual problem for soft-margin SVM
3. Show that MSE loss is convex (compute Hessian)
4. Apply KKT conditions to $\ell_1$-regularized regression (Lasso)
5. Prove that for convex problems, local minimum = global minimum

---

## References

1. Boyd & Vandenberghe - "Convex Optimization"
2. Nocedal & Wright - "Numerical Optimization"
3. Bertsekas - "Nonlinear Programming"

---

> **Return to**: [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/) | [Section Overview](../)
