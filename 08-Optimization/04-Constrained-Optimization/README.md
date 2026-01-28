# Constrained Optimization

## Introduction

Many real-world optimization problems have constraints. In machine learning, constraints arise in SVMs, constrained regression, fairness requirements, and resource allocation. Understanding constrained optimization is essential for formulating and solving these problems.

## Prerequisites

- Multivariate calculus (gradients, Hessians)
- Convex optimization basics
- Linear algebra

## Learning Objectives

1. Understand equality and inequality constraints
2. Apply Lagrange multipliers
3. Understand KKT conditions
4. Use penalty and barrier methods
5. Apply projected gradient descent

---

## 1. Problem Formulation

### 1.1 General Form

$$\min_{\mathbf{x}} f(\mathbf{x})$$
$$\text{subject to } g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m$$
$$\phantom{\text{subject to }} h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p$$

| Component                | Description                                    |
| ------------------------ | ---------------------------------------------- |
| $f(\mathbf{x})$          | Objective function                             |
| $g_i(\mathbf{x}) \leq 0$ | Inequality constraints                         |
| $h_j(\mathbf{x}) = 0$    | Equality constraints                           |
| Feasible region          | Set of $\mathbf{x}$ satisfying all constraints |

### 1.2 Example: SVM

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2$$
$$\text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1, \quad \forall i$$

---

## 2. Equality Constraints: Lagrange Multipliers

### 2.1 The Method

For $\min f(\mathbf{x})$ subject to $h(\mathbf{x}) = 0$:

**Lagrangian:**
$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda h(\mathbf{x})$$

**Optimality conditions:**
$$\nabla_{\mathbf{x}} \mathcal{L} = \nabla f + \lambda \nabla h = 0$$
$$\nabla_\lambda \mathcal{L} = h(\mathbf{x}) = 0$$

### 2.2 Geometric Intuition

At optimum, gradient of $f$ is parallel to gradient of $h$:

```
Level curves of f
        ╲    ╱
         ╲  ╱
    ──────●────── Constraint h(x) = 0
         ╱╲
        ╱  ╲

At optimal point ●:
∇f ∥ ∇h (tangent to constraint)
```

### 2.3 Example

Minimize $f(x, y) = x^2 + y^2$ subject to $x + y = 1$.

$$\mathcal{L} = x^2 + y^2 + \lambda(x + y - 1)$$

$$\frac{\partial \mathcal{L}}{\partial x} = 2x + \lambda = 0 \Rightarrow x = -\frac{\lambda}{2}$$
$$\frac{\partial \mathcal{L}}{\partial y} = 2y + \lambda = 0 \Rightarrow y = -\frac{\lambda}{2}$$
$$x + y = 1 \Rightarrow -\lambda = 1 \Rightarrow \lambda = -1$$

**Solution:** $x = y = \frac{1}{2}$, $f^* = \frac{1}{2}$

---

## 3. Inequality Constraints: KKT Conditions

### 3.1 The KKT Conditions

For $\min f(\mathbf{x})$ s.t. $g_i(\mathbf{x}) \leq 0$:

**Lagrangian:**
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}) = f(\mathbf{x}) + \sum_i \mu_i g_i(\mathbf{x})$$

**KKT conditions:**

| Condition               | Expression                               |
| ----------------------- | ---------------------------------------- |
| Stationarity            | $\nabla f + \sum_i \mu_i \nabla g_i = 0$ |
| Primal feasibility      | $g_i(\mathbf{x}) \leq 0$                 |
| Dual feasibility        | $\mu_i \geq 0$                           |
| Complementary slackness | $\mu_i g_i(\mathbf{x}) = 0$              |

### 3.2 Complementary Slackness

Either the constraint is **active** ($g_i = 0$) or the multiplier is zero ($\mu_i = 0$):

```
Case 1: Inactive constraint     Case 2: Active constraint

   constraint                      constraint
      │                               │
      │  ●←optimum                    ●←optimum (on boundary)
      │                               │
   ─────────                       ─────────
   g_i < 0, μ_i = 0               g_i = 0, μ_i ≥ 0
```

### 3.3 Example

Minimize $f(x) = x^2$ subject to $x \geq 1$ (equivalently $g(x) = 1 - x \leq 0$).

**Lagrangian:** $\mathcal{L} = x^2 + \mu(1 - x)$

**KKT conditions:**

- Stationarity: $2x - \mu = 0$
- Primal: $x \geq 1$
- Dual: $\mu \geq 0$
- Complementary: $\mu(1 - x) = 0$

**Case 1:** $\mu = 0$ → $x = 0$, violates $x \geq 1$. ✗

**Case 2:** $x = 1$ → $\mu = 2 \geq 0$. ✓

**Solution:** $x^* = 1$, $f^* = 1$

---

## 4. Lagrangian Duality

### 4.1 Dual Problem

**Primal:** $\min_{\mathbf{x}} \max_{\boldsymbol{\mu} \geq 0} \mathcal{L}(\mathbf{x}, \boldsymbol{\mu})$

**Dual:** $\max_{\boldsymbol{\mu} \geq 0} \min_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\mu})$

The dual function:
$$g(\boldsymbol{\mu}) = \min_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\mu})$$

### 4.2 Weak and Strong Duality

**Weak duality** (always holds):
$$g(\boldsymbol{\mu}) \leq f(\mathbf{x}^*) \quad \forall \boldsymbol{\mu} \geq 0$$

**Strong duality** (holds for convex problems + constraint qualification):
$$g(\boldsymbol{\mu}^*) = f(\mathbf{x}^*)$$

### 4.3 Why Duality Matters

| Benefit        | Description                                 |
| -------------- | ------------------------------------------- |
| Lower bounds   | Dual provides lower bound on primal         |
| Easier problem | Sometimes dual is easier to solve           |
| Insights       | Dual variables have economic interpretation |
| SVM            | Dual formulation enables kernel trick       |

---

## 5. Penalty Methods

### 5.1 Basic Idea

Convert constrained problem to sequence of unconstrained problems:

$$\min_{\mathbf{x}} f(\mathbf{x}) + \rho P(\mathbf{x})$$

where $P(\mathbf{x})$ penalizes constraint violations.

### 5.2 Quadratic Penalty

For $g(\mathbf{x}) \leq 0$:

$$P(\mathbf{x}) = \max(0, g(\mathbf{x}))^2$$

For $h(\mathbf{x}) = 0$:

$$P(\mathbf{x}) = h(\mathbf{x})^2$$

```
Penalty function for g(x) ≤ 0:

P(x)│
    │      ╱
    │     ╱
    │    ╱
    │   ╱
    └──────────── x
       │
    g(x)=0

Quadratic penalty grows as x violates constraint
```

### 5.3 Algorithm

```
for ρ = ρ₁, ρ₂, ... (increasing):
    Solve: min f(x) + ρ × P(x)
    if constraint satisfaction < ε:
        return x
```

### 5.4 Trade-offs

| Large $\rho$                   | Small $\rho$            |
| ------------------------------ | ----------------------- |
| Better constraint satisfaction | Faster convergence      |
| Ill-conditioned problem        | May violate constraints |

---

## 6. Barrier Methods (Interior Point)

### 6.1 Log Barrier

For $g(\mathbf{x}) \leq 0$, add barrier inside feasible region:

$$\min_{\mathbf{x}} f(\mathbf{x}) - \frac{1}{t} \sum_i \log(-g_i(\mathbf{x}))$$

```
Barrier function -log(-g):

   │
  ∞│        │
   │       ╱
   │      ╱
   │    ╱
   │__╱
   └──────────── x
      │
   g(x)=0

Barrier → ∞ as x approaches boundary
```

### 6.2 Interior Point Algorithm

```
Start with x inside feasible region
for t = t₁, t₂, ... (increasing):
    Solve: min f(x) - (1/t) × Σ log(-g_i(x))
    # Central path: solutions for different t
```

### 6.3 Central Path

As $t \to \infty$, solution approaches optimal solution while staying interior.

| $t$      | Behavior                         |
| -------- | -------------------------------- |
| Small    | Stay well inside feasible region |
| Large    | Approach boundary/optimal        |
| $\infty$ | Reach constrained optimum        |

---

## 7. Projected Gradient Descent

### 7.1 The Algorithm

For $\min_{\mathbf{x} \in \mathcal{C}} f(\mathbf{x})$:

$$\mathbf{x}_{t+1} = \Pi_{\mathcal{C}}(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t))$$

where $\Pi_{\mathcal{C}}$ is projection onto constraint set $\mathcal{C}$.

```
Projected Gradient Step:

  constraint
     region
    ┌─────┐
    │  ←←←●  x_t - η∇f  (gradient step)
    │   ↙
    │  ●───────────────● projected point
    │       projection
    └─────┘
```

### 7.2 Projection onto Common Sets

| Set                                        | Projection                                                                     |
| ------------------------------------------ | ------------------------------------------------------------------------------ |
| Box $[l, u]$                               | $\text{clip}(x, l, u)$                                                         |
| Ball $\|x\| \leq r$                        | $r \cdot x / \max(\|x\|, r)$                                                   |
| Simplex $\sum x_i = 1, x \geq 0$           | Sort and threshold                                                             |
| Half-space $\mathbf{a}^T\mathbf{x} \leq b$ | $\mathbf{x} - \max(0, \mathbf{a}^T\mathbf{x} - b) \mathbf{a}/\|\mathbf{a}\|^2$ |

### 7.3 Convergence

For convex $f$ and convex $\mathcal{C}$:

- Same convergence rates as unconstrained GD
- Requires efficient projection

---

## 8. Proximal Methods

### 8.1 Proximal Operator

$$\text{prox}_{\lambda g}(\mathbf{v}) = \arg\min_{\mathbf{x}} \left( g(\mathbf{x}) + \frac{1}{2\lambda}\|\mathbf{x} - \mathbf{v}\|^2 \right)$$

### 8.2 Proximal Gradient Descent

For $\min f(\mathbf{x}) + g(\mathbf{x})$ (smooth $f$, possibly non-smooth $g$):

$$\mathbf{x}_{t+1} = \text{prox}_{\eta g}(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t))$$

### 8.3 Common Proximal Operators

| $g(\mathbf{x})$               | $\text{prox}_{\lambda g}(\mathbf{v})$ |
| ----------------------------- | ------------------------------------- |
| $\|\mathbf{x}\|_1$            | Soft thresholding                     |
| $I_{\mathcal{C}}(\mathbf{x})$ | Projection $\Pi_{\mathcal{C}}$        |
| $\frac{1}{2}\|\mathbf{x}\|^2$ | $\frac{\mathbf{v}}{1 + \lambda}$      |

### 8.4 LASSO with Proximal GD

$$\min_{\mathbf{w}} \frac{1}{2}\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2 + \lambda\|\mathbf{w}\|_1$$

Update: $\mathbf{w} \leftarrow \text{soft}_{\eta\lambda}(\mathbf{w} - \eta \mathbf{X}^T(\mathbf{X}\mathbf{w} - \mathbf{y}))$

---

## 9. ADMM (Alternating Direction Method of Multipliers)

### 9.1 Problem Form

$$\min_{\mathbf{x}, \mathbf{z}} f(\mathbf{x}) + g(\mathbf{z})$$
$$\text{s.t. } \mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} = \mathbf{c}$$

### 9.2 Augmented Lagrangian

$$\mathcal{L}_\rho(\mathbf{x}, \mathbf{z}, \mathbf{y}) = f(\mathbf{x}) + g(\mathbf{z}) + \mathbf{y}^T(\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}) + \frac{\rho}{2}\|\mathbf{A}\mathbf{x} + \mathbf{B}\mathbf{z} - \mathbf{c}\|^2$$

### 9.3 ADMM Updates

```
repeat:
    x ← argmin_x L_ρ(x, z, y)  # x-update
    z ← argmin_z L_ρ(x, z, y)  # z-update
    y ← y + ρ(Ax + Bz - c)     # dual update
until convergence
```

### 9.4 Applications

- LASSO and other $\ell_1$ problems
- Distributed optimization
- Consensus problems
- Image processing

---

## 10. ML Applications

### 10.1 SVM (Support Vector Machine)

**Primal:**
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i$$
$$\text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

**Dual (enables kernel trick):**
$$\max_{\boldsymbol{\alpha}} \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T\mathbf{x}_j$$
$$\text{s.t. } 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0$$

### 10.2 Constrained Regression

Ridge with constraint:
$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$$
$$\text{s.t. } \|\mathbf{w}\|^2 \leq t$$

Equivalent to Lagrangian form with appropriate $\lambda$.

### 10.3 Fairness Constraints

$$\min_{\theta} \mathcal{L}(\theta)$$
$$\text{s.t. } |P(\hat{Y}=1|A=0) - P(\hat{Y}=1|A=1)| \leq \epsilon$$

---

## 11. Summary

| Method               | Best For                |
| -------------------- | ----------------------- |
| Lagrange multipliers | Equality constraints    |
| KKT                  | General constrained     |
| Penalty methods      | Simple constraints      |
| Barrier (IP)         | Inequality constraints  |
| Projected GD         | Convex set constraints  |
| Proximal             | Non-smooth regularizers |
| ADMM                 | Decomposable problems   |

**Key insights:**

- Dual variables = shadow prices of constraints
- KKT = necessary conditions for optimality
- Convexity ensures KKT are sufficient
- Choose method based on constraint structure

---

## References

1. Boyd & Vandenberghe - "Convex Optimization"
2. Nocedal & Wright - "Numerical Optimization"
3. Bertsekas - "Nonlinear Programming"
