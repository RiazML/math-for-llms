[← Back to Multivariate Calculus](../README.md) | [Next: Automatic Differentiation →](../05-Automatic-Differentiation/notes.md)

---

# Optimality Conditions

> _"The theory of maxima and minima is full of dangerous pitfalls for the unwary."_
> — G. H. Hardy

## Overview

Every machine learning model is the solution to an optimisation problem. Training a neural network, fitting a regression, finding principal components, building an SVM — all reduce to: find the parameters that minimise (or maximise) some objective function, possibly subject to constraints. Understanding *when* a point is optimal — the **optimality conditions** — is therefore prerequisite knowledge for understanding why any learning algorithm works.

This section develops optimality theory from first principles. We prove the first- and second-order conditions for unconstrained problems, derive Lagrange multipliers for equality constraints, and establish the full KKT framework for inequality-constrained optimisation. Convexity theory shows when local optimality implies global optimality. Duality reveals the deep connection between primal and dual problems — the machinery behind SVMs and maximum entropy models. We close with the non-convex landscape of deep learning and modern constrained formulations (SAM, attention as optimal transport).

> **Scope of this section:** Critical points, optimality conditions, Lagrange multipliers, KKT theory, convexity, duality, and ML applications. Gradient descent algorithms (SGD, Adam, momentum) are covered in the Optimization chapter (§08). Jacobian and Hessian computation is in §02. The chain rule and backpropagation are in §03.

## Prerequisites

- **Gradient, partial derivatives** — [§01 Partial Derivatives and Gradients](../01-Partial-Derivatives-and-Gradients/notes.md)
- **Hessian matrix, positive definiteness** — [§02 Jacobians and Hessians](../02-Jacobians-and-Hessians/notes.md)
- **Eigenvalues, positive definite matrices** — [§07 Positive Definite Matrices](../../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md)
- **Single-variable calculus** — [§04 Calculus Fundamentals](../../04-Calculus-Fundamentals/README.md)

## Companion Notebooks

| Notebook | Description |
|----------|-------------|
| [theory.ipynb](theory.ipynb) | Critical point classification, Lagrange multipliers, KKT verification, SVM dual derivation, convexity experiments |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: optimality conditions through SAM and attention as constrained optimisation |

## Learning Objectives

After completing this section, you will:

- Prove the first- and second-order necessary and sufficient conditions for local minima
- Classify critical points using the Hessian spectrum and understand saddle point dominance in deep networks
- Derive and apply Lagrange multipliers for equality-constrained problems
- State all four KKT conditions, prove their necessity, and apply them to ML problems
- Understand strong convexity, smoothness, and their role in convergence guarantees
- Derive the SVM dual from the primal and explain why support vectors satisfy complementary slackness
- Compute the Lagrangian dual function and apply Slater's condition for strong duality
- Connect maximum entropy, softmax, and attention to constrained optimisation via Lagrange multipliers
- Interpret the sharpness-aware minimisation (SAM) objective as a min-max constrained problem

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 The Optimization Landscape](#11-the-optimization-landscape)
  - [1.2 Why ML Needs Optimality Theory](#12-why-ml-needs-optimality-theory)
  - [1.3 Historical Context](#13-historical-context)
  - [1.4 Unconstrained vs Constrained](#14-unconstrained-vs-constrained)
- [2. First-Order Necessary Conditions](#2-first-order-necessary-conditions)
  - [2.1 The Gradient Condition](#21-the-gradient-condition)
  - [2.2 Critical Points and Their Classification](#22-critical-points-and-their-classification)
  - [2.3 Saddle Points in Deep Learning](#23-saddle-points-in-deep-learning)
  - [2.4 Stationary Points of Common ML Losses](#24-stationary-points-of-common-ml-losses)
- [3. Second-Order Conditions](#3-second-order-conditions)
  - [3.1 Second-Order Necessary Condition](#31-second-order-necessary-condition)
  - [3.2 Second-Order Sufficient Condition](#32-second-order-sufficient-condition)
  - [3.3 The Indefinite Hessian and Saddle Points](#33-the-indefinite-hessian-and-saddle-points)
  - [3.4 Global Optimality from Convexity](#34-global-optimality-from-convexity)
  - [3.5 Hessian at the Optimum for ML Problems](#35-hessian-at-the-optimum-for-ml-problems)
- [4. Convex Analysis](#4-convex-analysis)
  - [4.1 Convex Sets](#41-convex-sets)
  - [4.2 Convex Functions](#42-convex-functions)
  - [4.3 Strongly Convex Functions](#43-strongly-convex-functions)
  - [4.4 Convex Optimization](#44-convex-optimization)
  - [4.5 Preservation of Convexity](#45-preservation-of-convexity)
- [5. Lagrange Multipliers — Equality Constraints](#5-lagrange-multipliers--equality-constraints)
  - [5.1 Problem Setup and the Lagrangian](#51-problem-setup-and-the-lagrangian)
  - [5.2 Geometric Derivation](#52-geometric-derivation)
  - [5.3 Proof of the Lagrange Multiplier Theorem](#53-proof-of-the-lagrange-multiplier-theorem)
  - [5.4 The Multiplier as Sensitivity](#54-the-multiplier-as-sensitivity)
  - [5.5 Multiple Equality Constraints](#55-multiple-equality-constraints)
  - [5.6 ML Applications: PCA, Unit-Norm Constraints, LoRA](#56-ml-applications-pca-unit-norm-constraints-lora)
- [6. KKT Conditions — Inequality Constraints](#6-kkt-conditions--inequality-constraints)
  - [6.1 Problem Setup and the Full Lagrangian](#61-problem-setup-and-the-full-lagrangian)
  - [6.2 The Four KKT Conditions](#62-the-four-kkt-conditions)
  - [6.3 Geometric Intuition](#63-geometric-intuition)
  - [6.4 Proof of KKT Necessity](#64-proof-of-kkt-necessity)
  - [6.5 KKT Sufficiency for Convex Problems](#65-kkt-sufficiency-for-convex-problems)
  - [6.6 Working Through KKT: Linear Programming](#66-working-through-kkt-linear-programming)
- [7. Duality Theory](#7-duality-theory)
  - [7.1 The Lagrangian Dual Function](#71-the-lagrangian-dual-function)
  - [7.2 Weak Duality](#72-weak-duality)
  - [7.3 Strong Duality and Slater's Condition](#73-strong-duality-and-slaters-condition)
  - [7.4 Saddle Point Characterisation](#74-saddle-point-characterisation)
  - [7.5 The Dual of the SVM](#75-the-dual-of-the-svm)
- [8. ML Applications in Depth](#8-ml-applications-in-depth)
  - [8.1 Linear Regression Normal Equations](#81-linear-regression-normal-equations)
  - [8.2 Ridge and Lasso as Constrained Problems](#82-ridge-and-lasso-as-constrained-problems)
  - [8.3 SVM Primal and Dual](#83-svm-primal-and-dual)
  - [8.4 PCA as Constrained Maximisation](#84-pca-as-constrained-maximisation)
  - [8.5 Maximum Entropy and Softmax](#85-maximum-entropy-and-softmax)
  - [8.6 Attention as Constrained Optimisation](#86-attention-as-constrained-optimisation)
- [9. Non-Convex Landscape in Deep Learning](#9-non-convex-landscape-in-deep-learning)
  - [9.1 Loss Landscape Geometry](#91-loss-landscape-geometry)
  - [9.2 Overparameterisation and Implicit Regularisation](#92-overparameterisation-and-implicit-regularisation)
  - [9.3 Sharpness-Aware Minimisation](#93-sharpness-aware-minimisation)
  - [9.4 Neural Collapse](#94-neural-collapse)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 The Optimization Landscape

Imagine standing in a hilly landscape and trying to find the lowest valley. The **gradient** $\nabla f(\mathbf{x})$ tells you the direction of steepest ascent at your current position — to descend, move against the gradient. But this strategy can strand you in a local valley that is not the deepest one, or trap you at a mountain pass (saddle point) where the gradient is zero but you are not at a minimum.

```
THE OPTIMIZATION LANDSCAPE IN 1D AND 2D
════════════════════════════════════════════════════════════════════════

  1D loss curve:              2D loss surface (contour view):

      │╲     ╱╲    ╱         ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
      │ ╲   ╱  ╲  ╱           ╲  local     saddle  ╱
      │  ╲ ╱    ╲╱             ╲   min       ×   ╱
      │   ● local  ● global      ─ ─ ── ●── ─ ─ ─
      │   min      min              global
                                     min ●

  Key players:
  ┌─────────────────┬──────────────────────────────────────────────┐
  │ Point type      │ Characterisation                             │
  ├─────────────────┼──────────────────────────────────────────────┤
  │ Local minimum   │ ∇f = 0,  H ≻ 0  (all eigenvalues > 0)       │
  │ Local maximum   │ ∇f = 0,  H ≺ 0  (all eigenvalues < 0)       │
  │ Saddle point    │ ∇f = 0,  H indefinite (mixed signs)          │
  │ Regular point   │ ∇f ≠ 0  (not a critical point)              │
  └─────────────────┴──────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

The core difficulty: the gradient condition $\nabla f(\mathbf{x}^*) = \mathbf{0}$ is **necessary** but not **sufficient** for a minimum. A zero gradient could mark a minimum, a maximum, or a saddle point. Second-order conditions (the Hessian) are needed to distinguish between them.

Constraints add further complexity. If the solution must lie on a curve or within a region, the unconstrained minimum may be infeasible, and the optimum occurs at a very different location — possibly on the boundary of the feasible region.

### 1.2 Why ML Needs Optimality Theory

Every core ML algorithm is secretly an optimisation problem, and the optimality conditions determine its solution structure:

| ML Algorithm | Optimisation Problem | Optimality Condition Used |
|---|---|---|
| Linear regression | $\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|^2$ | $\nabla = 0$ → normal equations |
| Ridge regression | $\min_\mathbf{w} \|X\mathbf{w}-\mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$ | Lagrangian of constrained form |
| Lasso | $\min_\mathbf{w} \|X\mathbf{w}-\mathbf{y}\|^2 + \lambda\|\mathbf{w}\|_1$ | Subdifferential conditions |
| SVM | $\min \tfrac{1}{2}\|\mathbf{w}\|^2$ s.t. margin $\geq 1$ | KKT → support vectors |
| PCA | $\max_\mathbf{v} \mathbf{v}^\top C \mathbf{v}$ s.t. $\|\mathbf{v}\|=1$ | Lagrange → eigenvalue problem |
| Logistic regression | $\min -\sum \log p_i$ | Convex → unique global minimum |
| Neural network | $\min \mathcal{L}(\theta)$ (non-convex) | Saddle point dominance; NTK |
| Attention (softmax) | $\max_\mathbf{p} \mathbf{s}^\top\mathbf{p}$ s.t. $\mathbf{p} \in \Delta$ | Maximum entropy via Lagrange |
| SAM training | $\min_\theta \max_{\|\epsilon\|\leq\rho} \mathcal{L}(\theta+\epsilon)$ | KKT of inner max |

**For AI:** Understanding that the SVM dual problem depends only on inner products $\mathbf{x}_i^\top \mathbf{x}_j$ (from the KKT conditions) is what enables the kernel trick — the foundation of kernel methods. Understanding that softmax is the solution to a maximum entropy problem under linear constraints connects attention to thermodynamics and information theory.

### 1.3 Historical Context

| Year | Person | Contribution |
|------|--------|-------------|
| 1788 | Lagrange | *Mécanique Analytique* — method of multipliers for equality constraints |
| 1847 | Cauchy | Steepest descent algorithm |
| 1939 | Karush | Master's thesis: inequality constraints with multipliers (KKT conditions — unpublished) |
| 1951 | Kuhn & Tucker | Independent rediscovery and publication of KKT conditions |
| 1952 | Arrow, Hurwicz, Uzawa | Saddle point theorem for convex optimisation |
| 1970 | Rockafellar | *Convex Analysis* — definitive treatment of duality and subdifferentials |
| 1995 | Boser, Guyon, Vapnik | SVM uses the dual to derive the kernel trick |
| 1996 | Boyd & Vandenberghe | *Convex Optimization* (textbook; full text freely available online) |
| 2014 | Dauphin et al. | Saddle point dominance in deep networks |
| 2021 | Foret et al. | SAM: sharpness-aware minimisation as constrained min-max |
| 2022 | Papyan et al. | Neural collapse: KKT characterisation of terminal training phase |

### 1.4 Unconstrained vs Constrained

There are three fundamental problem classes:

```
THREE PROBLEM CLASSES
════════════════════════════════════════════════════════════════════════

  UNCONSTRAINED                    EQUALITY CONSTRAINED
  ─────────────                    ────────────────────
  min f(x)                         min f(x)
   x∈ℝⁿ                            s.t. gᵢ(x) = 0, i=1,…,m

  Solution: ∇f = 0                 Solution: Lagrange multipliers
  Theory: 2nd-order conditions      ∇f + λᵀ∇g = 0,  g(x) = 0

  INEQUALITY CONSTRAINED
  ──────────────────────
  min f(x)
  s.t. gᵢ(x) = 0,  i = 1,…,m
       hⱼ(x) ≤ 0,  j = 1,…,p

  Solution: KKT conditions
  ∇f + λᵀ∇g + μᵀ∇h = 0,  g=0,  h≤0,  μ≥0,  μhⱼ=0 ∀j

  KEY INSIGHT: Equality constraints are special case of KKT
  (set μ = 0 for inequalities, keep λ free for equalities)

════════════════════════════════════════════════════════════════════════
```

The **feasible set** $\mathcal{F} = \{\mathbf{x} : g_i(\mathbf{x}) = 0,\, h_j(\mathbf{x}) \leq 0\}$ is the set of points satisfying all constraints. The optimisation problem asks for the minimum of $f$ over $\mathcal{F}$.

---

## 2. First-Order Necessary Conditions

### 2.1 The Gradient Condition

**Theorem (First-Order Necessary Condition):** Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable. If $\mathbf{x}^*$ is a local minimum of $f$, then:
$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

**Proof:** Suppose $\mathbf{x}^*$ is a local minimum but $\nabla f(\mathbf{x}^*) \neq \mathbf{0}$. Define the direction $\mathbf{d} = -\nabla f(\mathbf{x}^*)$. By the directional derivative formula:
$$\frac{d}{dt} f(\mathbf{x}^* + t\mathbf{d})\Big|_{t=0} = \nabla f(\mathbf{x}^*)^\top \mathbf{d} = -\|\nabla f(\mathbf{x}^*)\|^2 < 0$$

By continuity of $f$, there exists $\epsilon > 0$ such that $f(\mathbf{x}^* + t\mathbf{d}) < f(\mathbf{x}^*)$ for all $t \in (0, \epsilon)$. But this contradicts $\mathbf{x}^*$ being a local minimum. $\square$

**Warning — necessity only:** The gradient condition is necessary but not sufficient. The function $f(x) = x^3$ has $f'(0) = 0$ at $x=0$ but no local extremum there (it's an inflection point). The function $f(x,y) = x^2 - y^2$ has $\nabla f(0,0) = \mathbf{0}$ but a saddle point. Second-order conditions are needed to determine which type of critical point we have.

**Critical points** (also called **stationary points**) are all $\mathbf{x}$ satisfying $\nabla f(\mathbf{x}) = \mathbf{0}$. They are candidates for minima, maxima, or saddle points — the Hessian distinguishes between them.

### 2.2 Critical Points and Their Classification

In $\mathbb{R}^2$, a function $f(x,y)$ has four types of critical points, determined by the sign of the Hessian determinant $\det H = f_{xx}f_{yy} - f_{xy}^2$ and the sign of $f_{xx}$:

```
SECOND-DERIVATIVE TEST IN ℝ²
════════════════════════════════════════════════════════════════════════

  At critical point (∇f = 0):

  det(H) > 0 and f_xx > 0  →  LOCAL MINIMUM
                                  (all eigenvalues of H positive)

  det(H) > 0 and f_xx < 0  →  LOCAL MAXIMUM
                                  (all eigenvalues of H negative)

  det(H) < 0                →  SADDLE POINT
                                  (H has mixed-sign eigenvalues)

  det(H) = 0                →  DEGENERATE (test inconclusive)
                                  (need higher-order analysis)

  Geometric picture for each:
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │   Min        │   Max        │   Saddle     │   Degenerate │
  │    ○         │    ●         │    ×         │    ─         │
  │   ╱│╲        │   ╲│╱        │   ╱   ╲      │              │
  │  ╱ │ ╲       │    │         │  ╱─────╲     │   ─────      │
  │ bowl shape   │ hill shape   │ mountain     │  flat region │
  │              │              │   pass       │              │
  └──────────────┴──────────────┴──────────────┴──────────────┘

════════════════════════════════════════════════════════════════════════
```

**Standard examples:**

- $f(x,y) = x^2 + y^2$: unique global minimum at origin; $H = 2I \succ 0$
- $f(x,y) = -(x^2 + y^2)$: unique global maximum at origin; $H = -2I \prec 0$
- $f(x,y) = x^2 - y^2$: saddle point at origin; $H = \text{diag}(2,-2)$ indefinite
- $f(x,y) = x^4 + y^4$: minimum at origin; $H(0,0) = 0$ (degenerate — minimum confirmed by inspection)

**In higher dimensions ($n > 2$):** A critical point is a strict local minimum iff $H(\mathbf{x}^*) \succ 0$ (all eigenvalues positive). It is a strict local maximum iff $H(\mathbf{x}^*) \prec 0$. Any other case (indefinite $H$ or semidefinite $H$) requires further analysis.

### 2.3 Saddle Points in Deep Learning

**The classical concern** was that neural networks might converge to poor local minima. Empirically, gradient descent finds solutions of similar quality regardless of initialisation for overparameterised networks. This was theoretically explained in a landmark 2014 paper.

**Dauphin et al. (2014) — Identifying and attacking the saddle point problem:** For a function $f: \mathbb{R}^n \to \mathbb{R}$ with $n$ large, a random critical point with loss $\epsilon$ above the global minimum is a saddle point with overwhelming probability (under a statistical physics model). The fraction of critical points that are local minima decreases exponentially with both $\epsilon$ and $n$.

```
SADDLE POINT DOMINANCE
════════════════════════════════════════════════════════════════════════

  Distribution of critical points at energy level ε above global min:

  Fraction that are local minima:  ~exp(-n · c(ε))

  For n = 10^9 (GPT-3),  c(ε) > 0  →  essentially zero local minima

  What gradient descent actually encounters:
  ┌────────────────────────────────────────────────────────────┐
  │  High-loss region:   many saddle points, few local minima  │
  │  Low-loss region:    many saddle points, even fewer local  │
  │  Near global min:    flat regions (many equivalent minima) │
  └────────────────────────────────────────────────────────────┘

  SGD noise + momentum helps escape saddles (saddle-free Newton
  methods explicitly use the negative curvature direction).

════════════════════════════════════════════════════════════════════════
```

**Why does SGD escape saddle points?** At a saddle point, the Hessian has at least one negative eigenvalue. The corresponding eigenvector is a **descent direction** — the function decreases along it. SGD's noise perturbations project onto this direction with nonzero probability, enabling escape. Deterministic gradient descent can be trapped at strict saddle points, but this is a measure-zero set.

**For AI:** The practical implication is that for overparameterised models (where parameters $\gg$ data), gradient descent on a non-convex loss converges to global or near-global optima. This is one theoretical explanation for why training large language models with SGD/Adam works so well.

### 2.4 Stationary Points of Common ML Loss Functions

**MSE (Linear Regression):** $f(\mathbf{w}) = \|X\mathbf{w} - \mathbf{y}\|^2$

$$\nabla_\mathbf{w} f = 2X^\top(X\mathbf{w} - \mathbf{y}) = \mathbf{0} \implies X^\top X \mathbf{w} = X^\top \mathbf{y}$$

This is the **normal equation**. It has a unique solution $\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}$ when $X$ has full column rank. The Hessian $H = 2X^\top X \succeq 0$ — MSE is convex, so the critical point is the global minimum.

**Cross-Entropy (Logistic Regression):** $f(\mathbf{w}) = -\sum_{i=1}^n [y_i \log \sigma(\mathbf{w}^\top \mathbf{x}_i) + (1-y_i)\log(1-\sigma(\mathbf{w}^\top \mathbf{x}_i))]$

$$\nabla_\mathbf{w} f = -\sum_{i=1}^n (y_i - \sigma(\mathbf{w}^\top \mathbf{x}_i))\mathbf{x}_i = X^\top(\boldsymbol{\sigma} - \mathbf{y})$$

No closed-form solution (transcendental equation). The Hessian $H = X^\top \text{diag}(\sigma_i(1-\sigma_i)) X \succeq 0$ confirms convexity. Gradient descent (or Newton's method) converges to the unique global minimum if it exists (may not if data is linearly separable).

**Cross-Entropy (Softmax / LLM output):** Setting $\nabla_\mathbf{z} [-\log p_y] = \mathbf{p} - \mathbf{e}_y = \mathbf{0}$ (as derived in §03) gives $p_y = 1$ and $p_k = 0$ for $k \neq y$. This is a perfectly confident prediction — the loss approaches zero but never reaches it for finite logits.


---

## 3. Second-Order Conditions

First-order conditions identify candidates; second-order conditions distinguish minima from maxima from saddle points. The Hessian matrix—the matrix of second partial derivatives—encodes all the local curvature information needed for this classification.

### 3.1 Second-Order Necessary Condition (SONC)

**Theorem (SONC).** If $\mathbf{x}^*$ is a local minimum of $f: \mathbb{R}^n \to \mathbb{R}$ and $f \in C^2$ near $\mathbf{x}^*$, then:

$$\nabla^2 f(\mathbf{x}^*) \succeq 0$$

(the Hessian is positive semi-definite).

**Proof.** Let $\mathbf{d} \in \mathbb{R}^n$ be any direction. Taylor expansion gives:

$$f(\mathbf{x}^* + t\mathbf{d}) = f(\mathbf{x}^*) + t \underbrace{\nabla f(\mathbf{x}^*)^\top \mathbf{d}}_{=0} + \frac{t^2}{2} \mathbf{d}^\top \nabla^2 f(\mathbf{x}^*) \mathbf{d} + O(t^3)$$

Since $\mathbf{x}^*$ is a local minimum, $f(\mathbf{x}^* + t\mathbf{d}) \geq f(\mathbf{x}^*)$ for all small $t$. Thus:

$$\frac{t^2}{2} \mathbf{d}^\top H \mathbf{d} + O(t^3) \geq 0$$

Dividing by $t^2$ and letting $t \to 0^+$:

$$\mathbf{d}^\top H \mathbf{d} \geq 0 \quad \forall \mathbf{d} \in \mathbb{R}^n$$

which is exactly $H \succeq 0$. $\square$

**Note:** SONC is necessary but not sufficient. $f(x) = x^4$ at $x^* = 0$ has $f''(0) = 0 \succeq 0$ yet $x^* = 0$ is a minimum. $f(x,y) = x^2 - y^3$ at origin has $H = \text{diag}(2, 0)$ but origin is not a minimum.

### 3.2 Second-Order Sufficient Condition (SOSC)

**Theorem (SOSC).** If $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $\nabla^2 f(\mathbf{x}^*) \succ 0$ (positive definite), then $\mathbf{x}^*$ is a strict local minimum.

**Proof.** Since $H = \nabla^2 f(\mathbf{x}^*) \succ 0$, its smallest eigenvalue satisfies $\lambda_{\min}(H) > 0$. By continuity of the Hessian, there exists $\delta > 0$ such that $\nabla^2 f(\mathbf{x}) \succ \frac{\lambda_{\min}}{2} I$ for all $\|\mathbf{x} - \mathbf{x}^*\| < \delta$.

For any $\mathbf{d}$ with $\|\mathbf{d}\| = 1$ and small $t > 0$:

$$f(\mathbf{x}^* + t\mathbf{d}) = f(\mathbf{x}^*) + \frac{t^2}{2} \mathbf{d}^\top H \mathbf{d} + O(t^3) \geq f(\mathbf{x}^*) + \frac{t^2 \lambda_{\min}}{2} + O(t^3) > f(\mathbf{x}^*)$$

for sufficiently small $t$. $\square$

**Gap between SONC and SOSC:** The boundary case $H \succeq 0$ but $H \not\succ 0$ (degenerate, semidefinite Hessian) requires higher-order analysis. This is common in deep learning where flat directions proliferate.

```
SECOND-ORDER TEST SUMMARY
════════════════════════════════════════════════════════════════════════

  Critical point x*: ∇f(x*) = 0

  H = ∇²f(x*)     Result          Name
  ─────────────────────────────────────────────────────────────────
  H ≻ 0           Strict local minimum        SOSC satisfied
  H ≺ 0           Strict local maximum        (max version of SOSC)
  H indefinite    Saddle point               Eigenvalues of both signs
  H ≽ 0, ≠ 0     Might be min (degenerate)  Higher order needed
  H = 0           Need Taylor term ≥3        Flat critical point
  ─────────────────────────────────────────────────────────────────

  For ℝ² via determinant test (when ∇f = 0):
    det(H) > 0, H₁₁ > 0  →  local minimum
    det(H) > 0, H₁₁ < 0  →  local maximum
    det(H) < 0             →  saddle point
    det(H) = 0             →  inconclusive

════════════════════════════════════════════════════════════════════════
```

### 3.3 Indefinite Hessian and Saddle Points

When $H = \nabla^2 f(\mathbf{x}^*)$ has both positive and negative eigenvalues, $\mathbf{x}^*$ is a **saddle point**: a local minimum along some directions, a local maximum along others.

**Morse Theory Preview.** For a smooth function $f: \mathbb{R}^n \to \mathbb{R}$, a critical point $\mathbf{x}^*$ is **non-degenerate** if $H(\mathbf{x}^*)$ is invertible. The **Morse index** of $\mathbf{x}^*$ is:

$$k = \#\{\lambda_i(H) < 0\}$$

(the number of descent directions). A non-degenerate minimum has index 0; a saddle has index $1, 2, \ldots, n-1$; a maximum has index $n$. Morse theory relates the topology of the sublevel sets $\{f \leq c\}$ to the critical points and their indices.

**For deep learning:** The loss landscape of an $n$-parameter network at a critical point near a good solution tends to have many positive eigenvalues and few (often negligible) negative ones. The ratio $k/n$ is typically very small at good solutions—confirming that most critical points found in practice are near-minima, not true saddles with large negative Hessian components.

### 3.4 Global Optimality from Convexity

For convex functions, the local-global distinction disappears entirely:

**Theorem.** If $f$ is convex and $\mathbf{x}^*$ is a local minimum, then $\mathbf{x}^*$ is a global minimum.

**Proof.** Suppose $\mathbf{x}^*$ is a local min but not global: there exists $\mathbf{y}$ with $f(\mathbf{y}) < f(\mathbf{x}^*)$. For $t \in (0,1)$, convexity gives:

$$f((1-t)\mathbf{x}^* + t\mathbf{y}) \leq (1-t)f(\mathbf{x}^*) + tf(\mathbf{y}) < f(\mathbf{x}^*)$$

As $t \to 0^+$, the point $(1-t)\mathbf{x}^* + t\mathbf{y}$ approaches $\mathbf{x}^*$ arbitrarily closely while having lower function value—contradicting $\mathbf{x}^*$ being a local minimum. $\square$

**Corollary.** For differentiable convex $f$: $\nabla f(\mathbf{x}^*) = \mathbf{0}$ iff $\mathbf{x}^*$ is a global minimum.

This is the fundamental reason convex optimization is "easy" compared to non-convex: any local search algorithm that finds a stationary point has found the global optimum.

### 3.5 Hessian at ML Optima

Understanding the Hessian at key ML critical points provides geometric insight into learning:

**Linear Regression.** For $f(\mathbf{w}) = \frac{1}{2n}\|X\mathbf{w} - \mathbf{y}\|^2$:

$$\nabla^2 f(\mathbf{w}) = \frac{1}{n} X^\top X$$

independent of $\mathbf{w}$. The Hessian is constant—this is a quadratic. Eigenvalues are $\sigma_i^2(X)/n$ (squared singular values scaled). If $X$ has full column rank, $H \succ 0$ everywhere, so the unique critical point is a global minimum.

**Logistic Regression.** For $f(\mathbf{w}) = \frac{1}{n}\sum_i [-y_i \mathbf{w}^\top \mathbf{x}_i + \log(1 + e^{\mathbf{w}^\top \mathbf{x}_i})]$:

$$\nabla^2 f(\mathbf{w}) = \frac{1}{n} X^\top \text{diag}(\sigma_i(1-\sigma_i)) X$$

where $\sigma_i = \sigma(\mathbf{w}^\top \mathbf{x}_i)$. Since $\sigma(z)(1-\sigma(z)) \in (0, 1/4]$, the Hessian is PSD everywhere (and PD if $X$ has full column rank), making logistic regression a convex problem.

**Neural Networks.** At a minimum $\mathbf{w}^*$, the Hessian $H(\mathbf{w}^*)$ typically has:
- A bulk of near-zero eigenvalues (flat landscape in many directions)
- A few large positive eigenvalues (steep curvature in a small subspace)
- Occasional small negative eigenvalues (slight non-convexity)

The ratio of large-to-small eigenvalues is the **condition number**—high condition numbers cause slow gradient descent convergence and motivate adaptive methods like Adam.


---

## 4. Convex Analysis

Convexity is the mathematical property that makes optimization tractable. Understanding convex sets and functions—their definitions, characterisations, and preservation rules—is prerequisite to deploying convex duality and KKT theory.

### 4.1 Convex Sets

**Definition.** A set $C \subseteq \mathbb{R}^n$ is **convex** if for all $\mathbf{x}, \mathbf{y} \in C$ and $\theta \in [0,1]$:

$$\theta \mathbf{x} + (1-\theta)\mathbf{y} \in C$$

Geometrically: the line segment between any two points stays inside the set.

**Standard examples:**
- Hyperplanes $\{\mathbf{x} : \mathbf{a}^\top \mathbf{x} = b\}$ and halfspaces $\{\mathbf{x} : \mathbf{a}^\top \mathbf{x} \leq b\}$
- Balls $\{\mathbf{x} : \|\mathbf{x} - \mathbf{c}\| \leq r\}$ under any norm
- Polyhedra $\{\mathbf{x} : A\mathbf{x} \leq \mathbf{b}\}$ (intersection of halfspaces)
- Positive semidefinite cone $\mathbb{S}_+^n = \{S \in \mathbb{R}^{n\times n} : S = S^\top,\, S \succeq 0\}$
- The probability simplex $\Delta^n = \{\mathbf{p} \geq 0 : \mathbf{1}^\top \mathbf{p} = 1\}$

**Non-examples:**
- The unit sphere $\mathbb{S}^{n-1}$ (boundary only, not the ball): the segment between two points on the sphere passes through the interior
- The set $\{(x,y) : xy \geq 1,\, x,y > 0\}$: take $(1,1)$ and $(2, 1/2)$; the midpoint $(3/2, 3/4)$ has $3/2 \cdot 3/4 = 9/8 > 1$—wait, this is actually convex. A cleaner non-example: $\{(x,y) : x^2 + y^2 = 1\}$ (circle)

**Convexity-preserving operations:**
- Intersection: $C_1, C_2$ convex $\Rightarrow$ $C_1 \cap C_2$ convex
- Affine image: $f(C) = \{A\mathbf{x} + \mathbf{b} : \mathbf{x} \in C\}$ is convex if $C$ is convex
- Cartesian product, Minkowski sum

### 4.2 Convex Functions: Definition and First-Order Characterisation

**Definition.** A function $f: C \to \mathbb{R}$ on a convex set $C$ is **convex** if:

$$f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \leq \theta f(\mathbf{x}) + (1-\theta)f(\mathbf{y}) \quad \forall \mathbf{x},\mathbf{y} \in C,\, \theta \in [0,1]$$

**Strict convexity:** replace $\leq$ with $<$ for $\mathbf{x} \neq \mathbf{y}$, $\theta \in (0,1)$.

**First-Order Characterisation.** For $f \in C^1$, $f$ is convex iff its graph lies above all tangent hyperplanes:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) \quad \forall \mathbf{x}, \mathbf{y} \in C$$

This is the inequality that makes $\nabla f(\mathbf{x}^*) = \mathbf{0}$ sufficient for global minimality in the convex case: if $\nabla f(\mathbf{x}^*) = \mathbf{0}$, then $f(\mathbf{y}) \geq f(\mathbf{x}^*) + 0 = f(\mathbf{x}^*)$ for all $\mathbf{y}$.

**Proof sketch.** ($\Rightarrow$) Fix $\mathbf{x}, \mathbf{y}$ and let $\phi(t) = f(\mathbf{x} + t(\mathbf{y}-\mathbf{x}))$. Convexity implies $\phi(1) \geq \phi(0) + \phi'(0)$, which gives the tangent inequality. ($\Leftarrow$) The tangent inequality with the two points $\mathbf{x}$ and $\mathbf{y}$ evaluated at $\mathbf{z} = \theta\mathbf{x} + (1-\theta)\mathbf{y}$ yields the convexity definition after combining.

### 4.3 Second-Order Characterisation

**Theorem.** For $f \in C^2$: $f$ is convex iff $\nabla^2 f(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$ in the domain.

**Proof.** ($\Rightarrow$) If $f$ is convex, the first-order characterisation gives $f(\mathbf{x} + t\mathbf{d}) \geq f(\mathbf{x}) + t \nabla f(\mathbf{x})^\top \mathbf{d}$. Expanding the Taylor series and applying this inequality yields $\mathbf{d}^\top H(\mathbf{x})\mathbf{d} \geq 0$. ($\Leftarrow$) With $H \succeq 0$ everywhere, integrate along the line from $\mathbf{x}$ to $\mathbf{y}$ using the second-order Taylor remainder to establish the first-order characterisation.

**Examples of convex functions:**
- Affine: $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x} + b$ (both convex and concave)
- Quadratic: $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top Q \mathbf{x}$ when $Q \succeq 0$
- Norms: $f(\mathbf{x}) = \|\mathbf{x}\|_p$ for $p \geq 1$ (triangle inequality = convexity)
- Log-sum-exp: $f(\mathbf{x}) = \log \sum_i e^{x_i}$ (smooth convex approximation to max)
- Negative entropy: $f(\mathbf{p}) = \sum_i p_i \log p_i$ on the simplex
- Cross-entropy loss: $-y \log \sigma(z) - (1-y)\log(1-\sigma(z))$ in $z$

**Non-convex in ML:**
- $f(w) = \sin(w)$, any neural network loss, $f(A,B) = \|AB - M\|_F^2$ (matrix factorisation)

### 4.4 Strongly Convex Functions

**Definition.** $f$ is **$\mu$-strongly convex** ($\mu > 0$) if:

$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) + \frac{\mu}{2}\|\mathbf{y}-\mathbf{x}\|^2$$

Equivalently: $f(\mathbf{x}) - \frac{\mu}{2}\|\mathbf{x}\|^2$ is convex; or $\nabla^2 f(\mathbf{x}) \succeq \mu I$ everywhere.

Strong convexity has powerful consequences:
1. **Unique minimiser**: the quadratic lower bound forces a unique optimal point
2. **Linear convergence**: gradient descent converges geometrically at rate $(1 - \mu/L)$ per step where $L$ is the Lipschitz constant of $\nabla f$
3. **Self-concordance**: the function cannot become arbitrarily flat

**For AI:** Ridge regression $f(\mathbf{w}) = \frac{1}{2n}\|X\mathbf{w} - \mathbf{y}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$ is $\lambda$-strongly convex (the $\ell_2$ regulariser adds $\lambda I$ to the Hessian). This is why $L_2$ regularisation speeds up convergence and ensures a unique solution—crucial for ill-conditioned problems.

**Condition number:** For $\mu$-strongly convex, $L$-smooth functions, $\kappa = L/\mu$ governs convergence. In transformer training, very high condition numbers of the loss landscape motivate adaptive optimisers.

### 4.5 Preservation Rules and Calculus of Convex Functions

Convexity is preserved under many operations, making it composable:

| Operation | Condition | Result |
|-----------|-----------|--------|
| Non-negative combination | $\alpha_i \geq 0$, $f_i$ convex | $\sum_i \alpha_i f_i$ convex |
| Composition with affine | $f$ convex, $A, \mathbf{b}$ any | $g(\mathbf{x}) = f(A\mathbf{x}+\mathbf{b})$ convex |
| Pointwise max | $f_i$ convex | $g(\mathbf{x}) = \max_i f_i(\mathbf{x})$ convex |
| Composition | $f$ convex nondecreasing, $g$ convex | $f \circ g$ convex |
| Partial min | $f(\mathbf{x}, \mathbf{y})$ convex in $(\mathbf{x},\mathbf{y})$ | $g(\mathbf{x}) = \inf_\mathbf{y} f(\mathbf{x},\mathbf{y})$ convex |
| Perspective | $f$ convex | $g(\mathbf{x},t) = tf(\mathbf{x}/t)$ convex on $\{t>0\}$ |

**For AI:** The cross-entropy loss $\ell(\mathbf{w}) = -\log \sigma(\mathbf{w}^\top \mathbf{x})$ is convex in $\mathbf{w}$ because it is $-\log$ composed with a concave function (affine composed with sigmoid). Convexity preservation rules let us verify this without computing the Hessian directly.


---

## 5. Lagrange Multipliers

When optimization problems come with constraints, unconstrained optimality conditions no longer apply directly. Lagrange multipliers transform constrained problems into unconstrained ones by incorporating the constraint into the objective—at the cost of introducing auxiliary variables.

### 5.1 Setup: Equality-Constrained Problems

**Problem form:**

$$\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x}) \quad \text{subject to} \quad g_i(\mathbf{x}) = 0, \quad i = 1, \ldots, m$$

where $f, g_i \in C^1$ and $m < n$ (fewer constraints than variables).

**The Lagrangian:** Define $\mathcal{L}: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$ by:

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x}) = f(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{g}(\mathbf{x})$$

The scalars $\lambda_i$ are **Lagrange multipliers** (or **dual variables**).

### 5.2 Geometric Derivation

The deepest way to understand why Lagrange's method works is geometric. At a constrained minimum $\mathbf{x}^*$:

**Claim:** $\nabla f(\mathbf{x}^*)$ must lie in the span of $\{\nabla g_1(\mathbf{x}^*), \ldots, \nabla g_m(\mathbf{x}^*)\}$.

**Why:** The feasible set near $\mathbf{x}^*$ is approximately the linear manifold $\{\mathbf{x} : \nabla g_i(\mathbf{x}^*)^\top (\mathbf{x} - \mathbf{x}^*) = 0, \, \forall i\}$—the tangent plane to each constraint surface. Any feasible direction $\mathbf{d}$ from $\mathbf{x}^*$ must satisfy $\nabla g_i(\mathbf{x}^*)^\top \mathbf{d} = 0$ for all $i$.

If $\nabla f(\mathbf{x}^*)$ had a component orthogonal to all $\nabla g_i(\mathbf{x}^*)$, that component would be a feasible direction along which $f$ decreases—contradicting $\mathbf{x}^*$ being a local constrained minimum.

Therefore $\nabla f(\mathbf{x}^*)$ has no component in the tangent space; it must lie entirely in the normal space spanned by $\{\nabla g_i\}$:

$$\nabla f(\mathbf{x}^*) = -\sum_{i=1}^m \lambda_i^* \nabla g_i(\mathbf{x}^*) \quad \Longleftrightarrow \quad \nabla_\mathbf{x} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*) = \mathbf{0}$$

```
LAGRANGE MULTIPLIER GEOMETRY (ℝ²)
════════════════════════════════════════════════════════════════════════

  Constraint: g(x,y) = 0  (a curve in the plane)
  Objective: minimize f(x,y)

         ▲ y
         │          f = c₃  (level curves of f)
    ─────┼─────►
         │          f = c₂
       ╭─┼─╮
      ╱  │  ╲       f = c₁  ← tangent to constraint at x*
     │   │●  │ x*
      ╲  │  ╱
       ╰─┼─╯  g = 0
         │

  At x*: level curve of f is tangent to constraint g=0
         ⟹ ∇f ∥ ∇g  ⟹  ∇f = -λ∇g

  ∇g points normal to g=0; ∇f points normal to f=c₁
  Tangency means these normals are parallel.

════════════════════════════════════════════════════════════════════════
```

### 5.3 The Lagrange Multiplier Theorem

**Theorem (Lagrange, 1788).** Let $\mathbf{x}^*$ be a local minimum of $f$ subject to $\mathbf{g}(\mathbf{x}) = \mathbf{0}$. If the **Linear Independence Constraint Qualification (LICQ)** holds at $\mathbf{x}^*$—i.e., $\{\nabla g_1(\mathbf{x}^*), \ldots, \nabla g_m(\mathbf{x}^*)\}$ are linearly independent—then there exists $\boldsymbol{\lambda}^* \in \mathbb{R}^m$ such that:

$$\nabla_\mathbf{x} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*) = \nabla f(\mathbf{x}^*) + \boldsymbol{\lambda}^* {}^\top \nabla \mathbf{g}(\mathbf{x}^*) = \mathbf{0}$$
$$\mathbf{g}(\mathbf{x}^*) = \mathbf{0}$$

Together these are $n + m$ equations in $n + m$ unknowns $(\mathbf{x}^*, \boldsymbol{\lambda}^*)$.

**LICQ matters:** Without LICQ the theorem can fail. Example: $\min x$ subject to $x^2 = 0$ and $x^3 = 0$. The constraints both vanish at $x^* = 0$ with gradients $0$ and $0$—linearly dependent. No Lagrange multiplier exists.

### 5.4 Sensitivity Interpretation: Shadow Prices

The Lagrange multiplier $\lambda_i^*$ has a precise economic interpretation: it measures how much the optimal value changes as constraint $i$ is relaxed.

**Theorem (Envelope).** Let $p^*(b)$ be the optimal value of $\min_\mathbf{x} f(\mathbf{x})$ subject to $g_i(\mathbf{x}) = b_i$. Then:

$$\frac{\partial p^*}{\partial b_i} = \lambda_i^*$$

**Proof sketch.** Differentiating the Lagrangian optimality conditions with respect to $b_i$ and applying the chain rule yields $dp^*/db_i = \lambda_i^*$ (the Implicit Function Theorem controls how $\mathbf{x}^*$ moves with $b_i$).

**For AI:** In constrained training (e.g., "minimize loss subject to $\|\mathbf{w}\|^2 = c$"), $\lambda^*$ tells you how much more you could improve the loss by relaxing the norm constraint by one unit. This motivates choosing the right regularisation strength: $\lambda^*$ is the value of the weight decay penalty that enforces the constraint.

### 5.5 Multiple Constraints and Second-Order Conditions

**Multiple equality constraints:** With $m$ equality constraints, the KKT point satisfies $n + m$ stationarity equations. Second-order analysis requires the **bordered Hessian** or, equivalently, the Hessian of the Lagrangian restricted to the tangent space of the constraints:

$$\mathbf{d}^\top \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\lambda}^*) \mathbf{d} > 0 \quad \forall \mathbf{d} \neq \mathbf{0} \text{ with } \nabla \mathbf{g}(\mathbf{x}^*)^\top \mathbf{d} = \mathbf{0}$$

is the second-order sufficient condition for a constrained local minimum.

### 5.6 ML Applications of Lagrange Multipliers

**PCA as constrained optimisation.** Find $\mathbf{v}_1 \in \mathbb{R}^d$ maximising variance $\mathbf{v}_1^\top \Sigma \mathbf{v}_1$ subject to $\|\mathbf{v}_1\|^2 = 1$:

$$\mathcal{L} = \mathbf{v}^\top \Sigma \mathbf{v} - \lambda(\mathbf{v}^\top \mathbf{v} - 1)$$

Stationarity: $2\Sigma \mathbf{v} = 2\lambda \mathbf{v}$, i.e., $\Sigma \mathbf{v} = \lambda \mathbf{v}$. The optimal direction is the top eigenvector; $\lambda^* = $ top eigenvalue $=$ maximum variance. PCA is literally solving Lagrange conditions.

**Unit-norm attention.** In some attention formulations, query/key vectors are L2-normalized before the dot product. The normalization constraint $\|{\bf q}\|=1$ is enforced via Lagrange multiplier; the shadow price reveals how much the attention energy would increase if the norm bound were relaxed.

**LoRA rank constraints.** Low-Rank Adaptation constrains the weight update $\Delta W = AB$ where $A \in \mathbb{R}^{d \times r}$, $B \in \mathbb{R}^{r \times k}$, with $r \ll \min(d,k)$. The rank-$r$ constraint is implicit in the factorised parameterisation, and the Lagrange multiplier interpretation illuminates why the singular values of $\Delta W$ concentrate: the effective constraint is on the nuclear norm (sum of singular values).


---

## 6. KKT Conditions

Lagrange multipliers handle equality constraints. When inequality constraints are present—which is the norm in machine learning (budget constraints, margin constraints, non-negativity)—the **Karush–Kuhn–Tucker (KKT) conditions** provide the generalisation.

### 6.1 The Full Problem and Lagrangian

**General form:**

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{s.t.} \quad h_j(\mathbf{x}) \leq 0,\; j = 1,\ldots,p \quad \text{and} \quad g_i(\mathbf{x}) = 0,\; i = 1,\ldots,m$$

**Lagrangian:**

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_{j=1}^p \mu_j h_j(\mathbf{x}) + \sum_{i=1}^m \lambda_i g_i(\mathbf{x})$$

where $\mu_j \geq 0$ are the multipliers for inequality constraints and $\lambda_i \in \mathbb{R}$ for equality constraints.

### 6.2 The Four KKT Conditions

At an optimal $\mathbf{x}^*$ (under a suitable constraint qualification):

**1. Stationarity:**
$$\nabla_\mathbf{x} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) = \nabla f(\mathbf{x}^*) + \sum_j \mu_j^* \nabla h_j(\mathbf{x}^*) + \sum_i \lambda_i^* \nabla g_i(\mathbf{x}^*) = \mathbf{0}$$

**2. Primal Feasibility:**
$$h_j(\mathbf{x}^*) \leq 0 \quad \forall j \qquad \text{and} \qquad g_i(\mathbf{x}^*) = 0 \quad \forall i$$

**3. Dual Feasibility:**
$$\mu_j^* \geq 0 \quad \forall j$$

**4. Complementary Slackness:**
$$\mu_j^* h_j(\mathbf{x}^*) = 0 \quad \forall j$$

**Interpreting complementary slackness:** For each inequality constraint $j$, either:
- $h_j(\mathbf{x}^*) = 0$: the constraint is **active** (the optimum is on the boundary)—$\mu_j^*$ can be nonzero
- $\mu_j^* = 0$: the constraint is **inactive** (the optimum is strictly interior)—the constraint doesn't affect the optimum

This is the geometric signature of which constraints "matter" at the solution.

### 6.3 Geometric Interpretation

The KKT conditions say: at optimum, you cannot improve $f$ while satisfying all constraints. The stationarity condition generalises Lagrange's condition: $-\nabla f(\mathbf{x}^*)$ must lie in the cone generated by the active constraint gradients.

```
KKT COMPLEMENTARY SLACKNESS GEOMETRY
════════════════════════════════════════════════════════════════════════

  Case 1: Constraint inactive (h(x*) < 0)
  ─────────────────────────────────────────
  The optimum is in the interior of the feasible region.
  The inequality constraint plays no role.
  μ* = 0 (it would be wrong to "push" against a slack constraint).

  Case 2: Constraint active (h(x*) = 0)
  ─────────────────────────────────────────
  The optimum lies ON the constraint boundary.
  The gradient ∇f(x*) points into the infeasible region.
  μ* > 0 "pushes back" to prevent crossing the boundary.
  The objective would improve if the constraint were relaxed.

  In both cases: μ* · h(x*) = 0 · (neg) = 0  ✓
                          or:  (pos) · 0 = 0  ✓

════════════════════════════════════════════════════════════════════════
```

### 6.4 KKT as Necessary Conditions: LICQ Proof

**Theorem.** If $\mathbf{x}^*$ is a local minimum and the **LICQ** holds (active constraint gradients are linearly independent), then the KKT conditions hold.

**Proof sketch.** Let $A = \{j : h_j(\mathbf{x}^*) = 0\}$ be the active set. By LICQ, $\{\nabla h_j(\mathbf{x}^*)\}_{j \in A} \cup \{\nabla g_i(\mathbf{x}^*)\}_i$ are linearly independent.

Any feasible descent direction $\mathbf{d}$ (satisfying $\nabla h_j(\mathbf{x}^*)^\top \mathbf{d} \leq 0$ for $j \in A$ and $\nabla g_i(\mathbf{x}^*)^\top \mathbf{d} = 0$) cannot have $\nabla f(\mathbf{x}^*)^\top \mathbf{d} < 0$ (otherwise $\mathbf{x}^*$ not local min).

By Farkas' lemma, $-\nabla f(\mathbf{x}^*)$ is a conic combination of active constraint gradients: $-\nabla f = \sum_{j \in A} \mu_j \nabla h_j + \sum_i \lambda_i \nabla g_i$ with $\mu_j \geq 0$. Setting $\mu_j = 0$ for $j \notin A$ gives all four conditions. $\square$

**Other constraint qualifications:** LICQ is sufficient but not necessary for KKT. Alternatives include Mangasarian-Fromovitz (MFCQ), Slater's condition (for convex problems), and linear independence of the Jacobian at the active set.

### 6.5 KKT as Sufficient Conditions (Convex Case)

For convex problems, KKT conditions are not just necessary—they are also sufficient:

**Theorem.** If $f$ and $h_j$ are convex, $g_i$ are affine, and $(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$ satisfy all four KKT conditions, then $\mathbf{x}^*$ is a global minimum.

**Proof.** For any feasible $\mathbf{y}$:

$$f(\mathbf{y}) \geq f(\mathbf{x}^*) + \nabla f(\mathbf{x}^*)^\top(\mathbf{y}-\mathbf{x}^*) \quad \text{(convexity of } f)$$

$$= f(\mathbf{x}^*) - \left(\sum_j \mu_j^* \nabla h_j(\mathbf{x}^*) + \sum_i \lambda_i^* \nabla g_i(\mathbf{x}^*)\right)^\top (\mathbf{y}-\mathbf{x}^*)$$

$$\geq f(\mathbf{x}^*) - \sum_j \mu_j^* h_j(\mathbf{y}) + \sum_j \mu_j^* h_j(\mathbf{x}^*) - \sum_i \lambda_i^* (g_i(\mathbf{y}) - g_i(\mathbf{x}^*))$$

Using primal feasibility ($h_j(\mathbf{y}) \leq 0$), dual feasibility ($\mu_j^* \geq 0$), complementary slackness ($\mu_j^* h_j(\mathbf{x}^*) = 0$), and $g_i(\mathbf{y}) = g_i(\mathbf{x}^*) = 0$, each term is $\geq 0$, so $f(\mathbf{y}) \geq f(\mathbf{x}^*)$. $\square$

### 6.6 LP Worked Example

**Linear Program:** $\min -x_1 - 2x_2$ subject to $x_1 + x_2 \leq 4$, $x_1, x_2 \geq 0$.

Rewriting as $h_1 = x_1 + x_2 - 4 \leq 0$, $h_2 = -x_1 \leq 0$, $h_3 = -x_2 \leq 0$.

Lagrangian: $\mathcal{L} = -x_1 - 2x_2 + \mu_1(x_1+x_2-4) - \mu_2 x_1 - \mu_3 x_2$.

**Stationarity:** $-1 + \mu_1 - \mu_2 = 0$, $-2 + \mu_1 - \mu_3 = 0$.

The constraint $x_2 \geq 0$ is never active at the optimum (we want $x_2$ large), so $\mu_3 = 0$. The budget constraint is active: $x_1 + x_2 = 4$, $x_1 = 0$. From stationarity: $\mu_1 = 2$, $\mu_2 = 1$. Optimal: $(x_1^*, x_2^*) = (0, 4)$, objective $= -8$.


---

## 7. Duality Theory

The Lagrangian dual offers a second approach to constrained optimisation: instead of minimising over $\mathbf{x}$, we maximise over the multipliers $(\boldsymbol{\mu}, \boldsymbol{\lambda})$. The resulting **dual problem** often has better structure (always convex, regardless of the primal), reveals hidden geometric properties, and—for convex problems—gives exactly the same optimal value.

### 7.1 The Dual Function and Dual Problem

**Definition (Lagrangian dual function):**

$$q(\boldsymbol{\mu}, \boldsymbol{\lambda}) = \inf_{\mathbf{x}} \mathcal{L}(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\lambda}) = \inf_{\mathbf{x}} \left[ f(\mathbf{x}) + \boldsymbol{\mu}^\top \mathbf{h}(\mathbf{x}) + \boldsymbol{\lambda}^\top \mathbf{g}(\mathbf{x}) \right]$$

**Key property:** $q$ is always concave in $(\boldsymbol{\mu}, \boldsymbol{\lambda})$—it is a pointwise infimum of affine functions of the multipliers.

**Dual problem:**

$$d^* = \max_{\boldsymbol{\mu} \geq 0, \boldsymbol{\lambda}} q(\boldsymbol{\mu}, \boldsymbol{\lambda})$$

This is always a convex optimisation problem (maximising concave = minimising convex).

### 7.2 Weak Duality

**Theorem (Weak Duality).** $d^* \leq p^*$ always, where $p^*$ is the primal optimal value.

**Proof.** For any feasible primal $\mathbf{x}$ (satisfying $h_j(\mathbf{x}) \leq 0$, $g_i(\mathbf{x}) = 0$) and any dual-feasible $(\boldsymbol{\mu}, \boldsymbol{\lambda})$ (with $\boldsymbol{\mu} \geq 0$):

$$q(\boldsymbol{\mu}, \boldsymbol{\lambda}) = \inf_{\mathbf{y}} \mathcal{L}(\mathbf{y}, \boldsymbol{\mu}, \boldsymbol{\lambda}) \leq \mathcal{L}(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \underbrace{\boldsymbol{\mu}^\top \mathbf{h}(\mathbf{x})}_{\leq 0} + \underbrace{\boldsymbol{\lambda}^\top \mathbf{g}(\mathbf{x})}_{=0} \leq f(\mathbf{x})$$

Taking supremum over dual and infimum over primal: $d^* \leq p^*$. $\square$

The gap $p^* - d^* \geq 0$ is the **duality gap**.

### 7.3 Strong Duality and Slater's Condition

**Theorem (Slater's Condition → Strong Duality).** For convex $f$ and $h_j$, affine $g_i$: if there exists a **strictly feasible** point $\hat{\mathbf{x}}$ (with $h_j(\hat{\mathbf{x}}) < 0$ strictly for all $j$), then $d^* = p^*$ (zero duality gap) and the dual optimum is attained.

Slater's condition is a **constraint qualification**: it says the feasible region is non-degenerate. For LP and QP (quadratic programs), strong duality holds under much weaker conditions.

**Implications for ML:**
- The SVM dual problem is equivalent to the primal (strong duality holds by Slater)
- The dual variables $\boldsymbol{\mu}^*$ from strong duality are exactly the KKT multipliers
- Duality gap as a stopping criterion: if primal value $-$ dual value $< \epsilon$, we have an $\epsilon$-optimal solution

### 7.4 Saddle Point Characterisation

**Theorem.** $(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$ is primal-dual optimal with zero duality gap iff it is a saddle point of the Lagrangian:

$$\mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}, \boldsymbol{\lambda}) \leq \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) \leq \mathcal{L}(\mathbf{x}, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*) \quad \forall \mathbf{x}, \boldsymbol{\mu} \geq 0, \boldsymbol{\lambda}$$

The minimax equals the maximin: $\min_\mathbf{x} \max_{\boldsymbol{\mu} \geq 0, \boldsymbol{\lambda}} \mathcal{L} = \max_{\boldsymbol{\mu} \geq 0, \boldsymbol{\lambda}} \min_\mathbf{x} \mathcal{L}$.

This saddle-point view is foundational for **adversarial training** in ML: GAN training is exactly seeking a saddle point of the value function $V(\theta_G, \theta_D)$.

### 7.5 SVM Dual: A Complete Example

The SVM is the canonical example of duality in ML. Start with the hard-margin SVM primal:

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1, \quad i = 1, \ldots, n$$

Rewrite constraints as $h_i = 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b) \leq 0$. Lagrangian:

$$\mathcal{L}(\mathbf{w}, b, \boldsymbol{\alpha}) = \frac{1}{2}\|\mathbf{w}\|^2 + \sum_i \alpha_i (1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b))$$

**KKT stationarity conditions:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \mathbf{w} - \sum_i \alpha_i y_i \mathbf{x}_i = \mathbf{0} \quad \Rightarrow \quad \mathbf{w}^* = \sum_i \alpha_i^* y_i \mathbf{x}_i$$

$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_i \alpha_i y_i = 0$$

**Substituting back** into $\mathcal{L}$ to form the dual:

$$q(\boldsymbol{\alpha}) = \sum_i \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

**Dual problem:** $\max_{\boldsymbol{\alpha} \geq 0} q(\boldsymbol{\alpha})$ subject to $\sum_i \alpha_i y_i = 0$.

This depends only on inner products $\mathbf{x}_i^\top \mathbf{x}_j$—the **kernel trick** replaces these with $k(\mathbf{x}_i, \mathbf{x}_j)$ for nonlinear boundaries without ever computing features explicitly.

**Support vectors:** By complementary slackness, $\alpha_i^* > 0$ only when $h_i(\mathbf{x}^*, b^*) = 0$, i.e., when the constraint is active: $y_i(\mathbf{w}^{*\top}\mathbf{x}_i + b^*) = 1$. These are exactly the **support vectors**—the training points on the margin boundary that determine $\mathbf{w}^*$.


---

## 8. Machine Learning Applications

The optimality conditions developed above appear directly in the mathematics underlying every major ML system. This section demonstrates these connections concretely.

### 8.1 Linear and Ridge Regression

**Ordinary Least Squares.** Minimise $f(\mathbf{w}) = \frac{1}{2n}\|X\mathbf{w} - \mathbf{y}\|^2$. Setting the gradient to zero:

$$\nabla f(\mathbf{w}) = \frac{1}{n} X^\top(X\mathbf{w} - \mathbf{y}) = \mathbf{0} \quad \Rightarrow \quad X^\top X \mathbf{w} = X^\top \mathbf{y}$$

These are the **normal equations**. When $X$ has full column rank, $(X^\top X) \succ 0$ (SOSC), and the unique solution is $\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}$.

**Ridge Regression.** Adding $\ell_2$ regularisation: $\min_\mathbf{w} \frac{1}{2n}\|X\mathbf{w}-\mathbf{y}\|^2 + \frac{\lambda}{2}\|\mathbf{w}\|^2$.

Normal equations: $(X^\top X + n\lambda I)\mathbf{w} = X^\top \mathbf{y}$. The regulariser shifts all eigenvalues by $n\lambda$, making the system always well-conditioned (strongly convex with $\mu = \lambda$).

### 8.2 Lasso and the Subdifferential

**Lasso:** $\min_\mathbf{w} \frac{1}{2n}\|X\mathbf{w}-\mathbf{y}\|^2 + \lambda\|\mathbf{w}\|_1$.

The $\ell_1$ norm is not differentiable at $w_j = 0$. The first-order optimality condition uses the **subdifferential** $\partial \|\mathbf{w}\|_1$:

$$0 \in \frac{1}{n} X^\top(X\mathbf{w}^* - \mathbf{y}) + \lambda \, \partial \|\mathbf{w}^*\|_1$$

Coordinate-wise, for the $j$-th weight:
- If $w_j^* \neq 0$: $\frac{1}{n}(X^\top(X\mathbf{w}^* - \mathbf{y}))_j + \lambda \, \text{sgn}(w_j^*) = 0$
- If $w_j^* = 0$: $\left|\frac{1}{n}(X^\top(X\mathbf{w}^* - \mathbf{y}))_j\right| \leq \lambda$

The second condition (correlation of feature $j$ with residual is small enough) determines sparsity: feature $j$ is excluded when its correlation with the residual is below the threshold $\lambda$. This is the mathematical source of Lasso's sparsity-inducing property.

### 8.3 SVM: Full KKT Analysis

The SVM soft-margin formulation extends the hard-margin case with slack variables $\xi_i \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_i \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top\mathbf{x}_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0$$

The KKT conditions include multipliers $\alpha_i$ for the margin constraints and $\beta_i$ for $\xi_i \geq 0$. Complementary slackness gives:
- $\alpha_i = 0$: correctly classified, not a support vector
- $0 < \alpha_i < C$: on the margin, $\xi_i = 0$ (standard support vector)
- $\alpha_i = C$: inside the margin or misclassified, $\xi_i > 0$

The parameter $C$ controls the trade-off between margin width and training error through the dual feasibility constraint $0 \leq \alpha_i \leq C$.

### 8.4 PCA via Constrained Optimisation

**Principal Component Analysis** seeks directions of maximum variance subject to orthonormality. The full PCA problem is:

$$\max_{V \in \mathbb{R}^{d \times k}} \text{tr}(V^\top \Sigma V) \quad \text{s.t.} \quad V^\top V = I_k$$

Lagrangian: $\mathcal{L} = \text{tr}(V^\top \Sigma V) - \text{tr}(\Lambda(V^\top V - I))$ where $\Lambda$ is a $k \times k$ symmetric multiplier matrix.

Stationarity: $2\Sigma V = 2V\Lambda$, i.e., $\Sigma V = V\Lambda$. Each column $\mathbf{v}_j$ satisfies $\Sigma \mathbf{v}_j = \lambda_j \mathbf{v}_j$—an eigenvalue equation. The optimal $V$ is the matrix of top-$k$ eigenvectors; the multipliers $\lambda_j$ are the eigenvalues (= captured variance).

### 8.5 Maximum Entropy and Softmax

**Maximum Entropy Principle.** Given constraints $\mathbb{E}_p[f_k] = c_k$ for $k=1,\ldots,K$, the distribution maximising entropy $H(p) = -\sum_i p_i \log p_i$ subject to $\sum_i p_i = 1$ has the **Boltzmann/Gibbs form**:

$$p_i^* = \frac{\exp(\sum_k \lambda_k^* f_k(i))}{Z(\boldsymbol{\lambda}^*)}$$

**Derivation.** Lagrangian with multipliers $\lambda_0$ (normalisation) and $\lambda_k$ (feature constraints):

$$\mathcal{L} = -\sum_i p_i \log p_i - \lambda_0 \left(\sum_i p_i - 1\right) - \sum_k \lambda_k \left(\sum_i p_i f_k(i) - c_k\right)$$

Stationarity in $p_i$: $-\log p_i - 1 - \lambda_0 - \sum_k \lambda_k f_k(i) = 0$, giving $p_i^* \propto \exp(\sum_k \lambda_k^* f_k(i))$.

**Softmax as max entropy.** The softmax function $p_i = e^{z_i}/\sum_j e^{z_j}$ is exactly the max-entropy distribution over $\{1,\ldots,n\}$ subject to $\mathbb{E}_p[e_i] = \text{const}$ constraints (Lagrange multipliers are the logits $z_i$). This gives the softmax a principled statistical interpretation beyond "normalised exponentials."

### 8.6 Attention as Constrained Optimisation

The attention mechanism in transformers can be viewed as a constrained optimisation problem. Given queries $Q$, keys $K$, values $V$, standard attention computes:

$$\text{Attn}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Constrained interpretation.** Attention computes the **memory retrieval** that solves:

$$\mathbf{v}^* = \arg\max_{\mathbf{p} \in \Delta^n} \mathbf{p}^\top K V - \frac{1}{\beta} \sum_i p_i \log p_i$$

where $\beta = 1/\sqrt{d_k}$ is a temperature and $\Delta^n$ is the probability simplex. The KKT condition for this maximum-entropy retrieval gives exactly softmax attention weights.

The Lagrange multiplier for the simplex constraint $\sum_i p_i = 1$ becomes the log-partition function $\log Z$ (the log-sum-exp normaliser), confirming that attention is a principled probabilistic retrieval operation.


---

## 9. Non-Convex Landscapes in Deep Learning

Deep learning operates almost entirely in the non-convex regime. Yet neural networks train successfully—why? The answer lies in the special geometric structure of high-dimensional non-convex landscapes.

### 9.1 Loss Landscape Geometry

An $n$-parameter neural network defines a loss landscape $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}_+$ with structure that differs fundamentally from classical non-convex functions:

**Key empirical observations:**
- **No spurious local minima** (approximately): for sufficiently overparameterised networks on tractable data, all local minima have near-identical loss values. Gradient descent doesn't get "trapped" because there are no deep local minima to get trapped in.
- **Saddle point dominance**: most critical points are saddle points, not local minima. The index (number of negative eigenvalue directions) of these saddles is typically large.
- **Loss barriers between solutions**: two different global minima can be separated by high barriers in weight space, but connected by **loss valleys** in higher-dimensional paths.

**For AI:** The practical consequence is that SGD with good initialisation reliably finds good solutions, and model merging (linear combination of two trained models) often produces competitive performance—evidence of basin connectivity.

### 9.2 The Role of Overparameterisation

**Theorem (informal, Kawaguchi 2016; Du et al. 2018).** For a wide class of networks trained on generic data, when the number of parameters $n$ exceeds the number of training points $N$, every local minimum is a global minimum.

**Intuition.** With $n \gg N$ parameters, the loss function has $n - N$ approximate degrees of freedom. The "level set" $\{\mathbf{w} : \mathcal{L}(\mathbf{w}) \approx 0\}$ is a high-dimensional manifold. Any point trying to be a local minimum with non-zero loss would need the gradient to vanish while the Hessian is positive definite in all $n$ directions—this becomes geometrically impossible when the loss can be driven to zero by moving in any of the $n - N$ flat directions.

**Neural Tangent Kernel (NTK) perspective.** In the infinite-width limit, training dynamics become linear: $\dot{\mathbf{w}} = -\eta K^\infty (\mathbf{w}) \mathbf{e}$ where $K^\infty$ is the NTK matrix. If $K^\infty \succ 0$, gradient descent converges globally at a linear rate—a convex analysis result applied to a formally non-convex problem.

### 9.3 Sharpness-Aware Minimisation (SAM)

Motivated by the observation that flat minima generalise better than sharp ones, **Sharpness-Aware Minimisation** (Foret et al. 2021) solves:

$$\min_\mathbf{w} \max_{\|\boldsymbol{\epsilon}\|_2 \leq \rho} \mathcal{L}(\mathbf{w} + \boldsymbol{\epsilon})$$

**KKT analysis of the inner problem.** Fix $\mathbf{w}$. The inner maximisation $\max_{\|\boldsymbol{\epsilon}\| \leq \rho} \mathcal{L}(\mathbf{w} + \boldsymbol{\epsilon})$ is a constrained problem with one inequality constraint $h(\boldsymbol{\epsilon}) = \|\boldsymbol{\epsilon}\|^2 - \rho^2 \leq 0$.

KKT conditions: $\nabla_{\boldsymbol{\epsilon}} \mathcal{L}(\mathbf{w} + \boldsymbol{\epsilon}^*) = 2\mu^* \boldsymbol{\epsilon}^*$, giving $\boldsymbol{\epsilon}^* \parallel \nabla_\mathbf{w} \mathcal{L}(\mathbf{w})$.

The adversarial perturbation is: $\hat{\boldsymbol{\epsilon}} = \rho \cdot \nabla_\mathbf{w} \mathcal{L}(\mathbf{w}) / \|\nabla_\mathbf{w} \mathcal{L}(\mathbf{w})\|$

SAM gradient step: $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_\mathbf{w} \mathcal{L}(\mathbf{w} + \hat{\boldsymbol{\epsilon}})$.

This is a direct application of KKT: solving the constrained inner problem analytically gives the SAM update formula.

### 9.4 Neural Collapse

At the terminal phase of training, when networks reach near-zero training loss, a remarkable geometric structure called **Neural Collapse** (Papyan et al. 2020) emerges:

1. **Within-class variability collapse**: all training examples of class $c$ have the same last-layer representation $\mathbf{h} = \boldsymbol{\mu}_c$
2. **Equinorm ETF**: the class means $\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_C$ form an **Equiangular Tight Frame**—they have equal norms and equal pairwise cosine similarities $= -1/(C-1)$
3. **Self-duality**: the weight vectors $\mathbf{w}_c$ align with the class means up to scaling

**KKT characterisation.** Neural collapse is the KKT point of the **Unconstrained Features Model** (UFM):

$$\min_{\mathbf{H}, \mathbf{W}, \mathbf{b}} \mathcal{L}_{\text{CE}}(\mathbf{W}\mathbf{H} + \mathbf{b}) + \frac{\lambda}{2}(\|\mathbf{H}\|_F^2 + \|\mathbf{W}\|_F^2)$$

The KKT stationarity conditions, combined with symmetry of the cross-entropy loss at balanced class distributions, force the ETF structure. Neural collapse is not an empirical coincidence—it is the unique KKT point of this simplified training problem.

### 9.5 Mode Connectivity

**Loss valley hypothesis:** Two local minima $\mathbf{w}_1^*$ and $\mathbf{w}_2^*$ of a neural network can be connected by a low-loss path (Garipov et al. 2018). Specifically, there exists a piecewise linear or quadratic curve $\phi: [0,1] \to \mathbb{R}^n$ with $\phi(0) = \mathbf{w}_1^*$, $\phi(1) = \mathbf{w}_2^*$ and $\mathcal{L}(\phi(t)) \approx \mathcal{L}(\mathbf{w}_1^*)$ for all $t \in [0,1]$.

**Implication for model merging.** The success of weight averaging (WA) methods like Model Soups and SLERP model merging is explained by mode connectivity: if the merged model lies near the connecting path in weight space, it inherits the performance of both endpoints.

**Optimality connection.** Mode connectivity is a non-convex analogue of the fundamental theorem of convex analysis: convex functions have connected sublevel sets. For "sufficiently trained" neural networks, the sublevel set $\{\mathbf{w} : \mathcal{L}(\mathbf{w}) \leq \mathcal{L}(\mathbf{w}_1^*) + \epsilon\}$ is approximately path-connected—not because the landscape is convex, but because overparameterisation creates high-dimensional flat regions.


---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | **Treating $\nabla f(\mathbf{x}^*) = \mathbf{0}$ as sufficient for a minimum** | It's necessary, not sufficient. Saddle points and maxima also satisfy this. | Always check second-order conditions (Hessian PSD/PD) or verify global minimum via convexity arguments |
| 2 | **Forgetting to check constraint qualification** | KKT conditions require LICQ (or another CQ). Without CQ, a minimum may exist with no Lagrange multiplier | Verify that active constraint gradients are linearly independent at the candidate point |
| 3 | **Setting $\mu_j < 0$ for inequality constraints** | Negative multipliers on $h_j \leq 0$ constraints violate dual feasibility; the Lagrangian is then unbounded below | Dual feasibility requires $\mu_j \geq 0$ for all inequality constraints (the convention matters: $h \leq 0$ needs $\mu \geq 0$) |
| 4 | **Ignoring complementary slackness** | Missing the condition $\mu_j h_j = 0$ leads to wrong determination of which constraints are active | For each inequality constraint, exactly one of $\mu_j = 0$ or $h_j = 0$ must hold (or both) |
| 5 | **Concluding strong duality without Slater** | Weak duality always holds, but strong duality ($d^* = p^*$) requires a constraint qualification like Slater's condition | Verify strict feasibility (Slater's point exists) before asserting zero duality gap |
| 6 | **Using the $\text{det}(H)$ test in $\mathbb{R}^n$, $n > 2$** | The determinant test (det $> 0$ and $H_{11} > 0$) is specific to $\mathbb{R}^2$. In higher dimensions, need to check all $n$ leading principal minors or eigenvalues | In $\mathbb{R}^n$: compute all eigenvalues of $H$; or use Sylvester's criterion (all leading principal minors $> 0$ iff PD) |
| 7 | **Confusing local and global optimality for non-convex problems** | For non-convex functions, local minima may not be global. KKT conditions identify local critical points, not global optima | Use global analysis: prove convexity, or use branch-and-bound, or accept local optimality |
| 8 | **Wrong sign convention for the Lagrangian** | Different texts define $\mathcal{L} = f + \lambda g$ vs. $f - \lambda g$. Mixing conventions gives wrong multiplier signs | Pick one convention and be consistent: for $\min f$ s.t. $g = 0$, use $\mathcal{L} = f + \lambda g$ (add constraints to objective) |
| 9 | **Forgetting that the Lagrange multiplier theorem is for $C^1$ functions** | At non-smooth points (e.g., $\ell_1$ constraints), the standard gradient condition fails | Use subdifferentials and subgradient conditions; or smooth the problem with a differentiable approximation |
| 10 | **Concluding "no constrained minimum exists" when KKT has no solution** | KKT conditions failing to have a solution means the constraint qualification fails OR no minimum exists. These are different | First check whether the feasible set is closed and bounded (Weierstrass guarantees existence); then debug the CQ |
| 11 | **Misidentifying support vectors** | Only points with $\alpha_i > 0$ (active margin constraint) are support vectors; incorrectly classified interior points are not | Check complementary slackness: $\alpha_i > 0 \Leftrightarrow y_i(\mathbf{w}^\top\mathbf{x}_i + b) = 1$ (on the margin boundary) |
| 12 | **Applying first-order conditions to discrete or combinatorial constraints** | Gradient = 0 requires differentiability; discrete feasible sets (e.g., integer programs) don't satisfy this | For discrete/combinatorial problems, use integer programming methods, branch-and-bound, or relaxations |

---

## 11. Exercises

### Exercise 1 ★ — Critical Point Classification

Let $f(x,y) = x^3 - 3xy^2 + y^4$.

**(a)** Find all critical points by solving $\nabla f = \mathbf{0}$.

**(b)** Compute the Hessian at each critical point and classify using the second-order test.

**(c)** Verify numerically: check that gradient descent from each critical point stays put (up to numerical noise).

**(d)** Sketch the level curves of $f$ near each critical point.

### Exercise 2 ★ — Lagrange Multipliers: Constrained Extremum

Maximise $f(\mathbf{x}) = x_1 x_2 x_3$ subject to $x_1 + x_2 + x_3 = 12$, $\mathbf{x} \geq \mathbf{0}$.

**(a)** Write the Lagrangian and derive KKT conditions.

**(b)** Solve analytically and verify that the maximum is $f^* = 64$.

**(c)** Interpret the Lagrange multiplier: by how much does $f^*$ change if the constraint becomes $x_1 + x_2 + x_3 = 13$?

**(d)** Confirm numerically using `scipy.optimize.minimize`.

### Exercise 3 ★ — KKT Conditions for Quadratic Program

Solve: $\min \frac{1}{2}(x_1^2 + x_2^2)$ subject to $x_1 + x_2 \geq 3$, $x_1, x_2 \geq 0$.

**(a)** Write the KKT conditions in full.

**(b)** Determine the active constraint(s) and solve the KKT system.

**(c)** Verify the solution is a global minimum by checking convexity.

**(d)** Compute the duality gap (should be zero).

### Exercise 4 ★★ — Convexity Analysis

For each function, determine if it is convex, strictly convex, or neither, on the specified domain:

**(a)** $f(x) = e^x - x - 1$ on $\mathbb{R}$

**(b)** $f(\mathbf{x}) = \|\mathbf{x}\|_2^2 + \|\mathbf{x}\|_1$ on $\mathbb{R}^n$

**(c)** $f(A) = -\log \det A$ on $\mathbb{S}_{++}^n$ (positive definite matrices)

**(d)** $f(x,y) = x^2/y$ for $y > 0$

**(e)** $f(\mathbf{w}) = \mathcal{L}_{\text{CE}}(\mathbf{w})$ for logistic regression with linearly separable data

### Exercise 5 ★★ — SVM Dual Derivation

Derive the dual of the hard-margin SVM from scratch.

**(a)** Write the primal as a standard form QP with inequality constraints.

**(b)** Form the Lagrangian and derive the dual function $q(\boldsymbol{\alpha})$.

**(c)** Write the dual problem and verify its constraints.

**(d)** Show that the dual is concave in $\boldsymbol{\alpha}$.

**(e)** Implement and solve both primal and dual for a small dataset; verify they give the same optimal value.

### Exercise 6 ★★ — Maximum Entropy Distribution

Find the maximum entropy distribution on $\{1, 2, 3, 4\}$ subject to: $\sum_i p_i = 1$ and $\mathbb{E}[X] = 2.5$.

**(a)** Write the Lagrangian with multipliers $\lambda_0, \lambda_1$.

**(b)** Derive that $p_i^* = e^{-\lambda_0 - \lambda_1 i}/Z$.

**(c)** Find $\lambda_0, \lambda_1$ numerically by solving the constraint equations.

**(d)** Compare to the uniform distribution: which has higher entropy?

**(e)** Verify: compute $H(p^*)$ and confirm it exceeds $H(\text{uniform restricted to } \mathbb{E}[X]=2.5)$.

### Exercise 7 ★★★ — SAM: KKT Analysis

Analyse Sharpness-Aware Minimisation rigorously.

**(a)** Write the inner maximisation $\max_{\|\boldsymbol{\epsilon}\| \leq \rho} \mathcal{L}(\mathbf{w} + \boldsymbol{\epsilon})$ as a constrained problem.

**(b)** Write the KKT conditions and solve for $\boldsymbol{\epsilon}^*$ in closed form.

**(c)** Show that $\hat{\boldsymbol{\epsilon}} = \rho \nabla\mathcal{L}(\mathbf{w})/\|\nabla\mathcal{L}(\mathbf{w})\|$ satisfies the KKT conditions.

**(d)** Implement one SAM step and compare to vanilla gradient descent on a sharp quadratic.

**(e)** Empirically verify that SAM finds flatter minima on a small neural network.

### Exercise 8 ★★★ — Duality Gap and Convergence

Implement a primal-dual interior-point method for a small LP.

**(a)** Formulate: $\min \mathbf{c}^\top \mathbf{x}$ s.t. $A\mathbf{x} = \mathbf{b}$, $\mathbf{x} \geq 0$ as a standard LP.

**(b)** Implement the primal-dual path-following method tracking both primal and dual variables.

**(c)** Plot the duality gap vs. iteration and verify it converges to zero.

**(d)** Verify the optimal solution satisfies all KKT conditions numerically.

**(e)** Compare convergence to simplex method on the same problem.


---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | Impact on AI/ML |
|---------|----------------|
| **First-order necessary conditions** | Every modern optimiser (SGD, Adam, AdamW) terminates at approximate stationarity $\|\nabla\mathcal{L}\| \approx 0$. Training stopping criteria are gradient-norm thresholds. |
| **Hessian spectrum** | Adaptive learning rate methods (Adam, Adagrad) implicitly approximate $H^{-1}\nabla\mathcal{L}$ to normalise curvature. Sharpness-aware methods (SAM, ASAM) explicitly minimise spectral norm of $H$. |
| **Saddle points** | The prevalence of saddle points (not local minima) in deep network landscapes explains why gradient descent with noise (SGD, dropout) escapes them efficiently. Perturbed gradient descent has provable saddle-escape guarantees. |
| **Convexity** | Convex relaxations of combinatorial ML problems (e.g., $\ell_1$ relaxation of sparse recovery) are solvable to global optimality. Convex loss functions (logistic, square) give unique solutions independent of initialisation. |
| **Lagrange multipliers** | PCA, LDA, CCA are all Lagrange multiplier problems. QLoRA's 4-bit quantisation uses a constrained optimisation where the quantisation constraint is relaxed via Lagrange multipliers in the mixed-precision framework. |
| **KKT conditions** | SVM training, LP relaxations of beam search, constrained decoding (e.g., budget-forcing), and RLHF reward-constrained optimisation all use KKT theory for optimality analysis. |
| **Duality** | The SVM dual gives the kernel trick. Max-margin training of LLMs (via RLHF) can be analysed as a dual problem where the Lagrange multiplier on the KL constraint is the reward temperature $\beta$. |
| **Strong convexity** | $L_2$ regularisation makes the loss strongly convex, giving linear convergence guarantees. The condition number $\kappa = L/\mu$ governs convergence; warmup schedules and weight decay are practitioner responses to ill-conditioning. |
| **Max entropy / softmax** | Language model next-token prediction is max-entropy inference subject to observed statistics. Temperature scaling adjusts the implicit Lagrange multiplier on the cross-entropy constraint, controlling distribution sharpness. |
| **SAM/sharpness** | Flat minima empirically generalise better. SAM (Foret et al. 2021) is now standard in SOTA image classification and LLM fine-tuning. The optimality condition for the inner problem is the KKT stationarity condition. |
| **Neural collapse** | The ETF structure at convergence (proven via KKT of the UFM) informs classifier weight initialisation strategies and explains why the last layer of a fine-tuned model can be reset and retrained cheaply. |
| **Mode connectivity** | Model merging (Model Soups, TIES, DARE) and model interpolation are enabled by the path-connectivity of loss basins—a direct consequence of overparameterised non-convex landscape geometry. |

---

## Conceptual Bridge

This section occupies the pivot between two phases of the curriculum. Everything before §04 established the tools for computing derivatives—limits, continuity, partial derivatives, gradients, Jacobians, chain rule. This section asks the deeper question: *what do these derivatives tell us about where the best solutions are?*

**Looking backward.** The first-order conditions derive directly from the gradient machinery of §01–§03. The Hessian—the matrix of second partial derivatives—extends the single-variable second derivative test to functions of many variables, connecting to the matrix theory of §02. The chain rule and product rule for derivatives underpin every step of the Lagrangian analysis.

**Looking forward.** The optimality conditions here are used constantly in the chapters ahead:
- §05/05 (Gradient Descent) and §08 (Optimisation Algorithms): gradient descent is precisely the process of iterating toward $\nabla f = \mathbf{0}$; convergence analysis requires the strong convexity and Lipschitz smoothness properties defined here
- §06 (Probability & Statistics): maximum likelihood estimation is an unconstrained minimisation whose optimality conditions give the maximum likelihood equations; Bayesian MAP estimation introduces prior constraints handled by Lagrange multipliers
- §07 (Information Theory): the maximum entropy principle (§8.5) is a direct application of Lagrange multipliers; the KL divergence minimisation underlying variational inference is a constrained optimisation whose KKT conditions are the ELBO

```
POSITION IN THE CALCULUS CURRICULUM
════════════════════════════════════════════════════════════════════════

  §04: Calculus Fundamentals
  ├── §01: Limits and Continuity          ─► Foundation for §02
  ├── §02: Derivatives and Differentiation ─► Foundation for §03, §04
  ├── §03: Integration                    ─► Foundation for §06 (probability)
  └── §04: Optimality Conditions ◄── YOU ARE HERE
          │
          │  Uses: gradients, Hessians, chain rule, Jacobians
          │  Introduces: critical points, KKT, duality, convexity
          │
          ▼
  §05: Multivariate Calculus
  ├── §01: Partial Derivatives (prerequisite)
  ├── §02: Gradient and Hessian (prerequisite)
  ├── §03: Chain Rule & Backpropagation (prerequisite)
  ├── §04: Optimality Conditions ◄── YOU ARE HERE
  └── §05: Gradient Descent and Convergence ─► Uses §04's convexity theory

  §08: Optimisation Algorithms ─► Deep dive into algorithms enabled by §04
  §09: Probabilistic Graphical Models ─► MAP/MLE via §04's Lagrange theory

════════════════════════════════════════════════════════════════════════
```

**The central insight.** Optimality conditions are not merely a tool for finding optima—they are a *language* for describing what it means for a solution to be right. The KKT conditions don't just tell you where to stop searching; they tell you *why* the answer is the answer: which constraints are binding, how sensitive the solution is to perturbations, what trade-offs are implicit in the optimal choice. Every time a language model outputs a softmax distribution, every time a recommender system solves a constrained relevance problem, every time a reinforcement learning agent balances reward and KL penalty—the KKT conditions are operating in the background, whether or not the engineer knows it.


---

## Supplement A: Deeper Mathematical Treatments

This supplement provides expanded proofs, worked examples, and connections that go beyond the core sections above. These are valuable for practitioners who want to move from "knowing the conditions" to "understanding why the conditions have the form they do."

### A.1 Proof of the Envelope Theorem

The Envelope Theorem is the rigorous basis for the shadow price interpretation of Lagrange multipliers. Here is the complete proof.

**Setup.** Let $p^*(b)$ be the optimal value of:

$$\min_{\mathbf{x}} f(\mathbf{x}, b) \quad \text{s.t.} \quad \mathbf{g}(\mathbf{x}, b) = \mathbf{0}$$

where $b \in \mathbb{R}$ is a parameter (e.g., the RHS of a constraint). Assume $f, \mathbf{g}$ are $C^1$ and $\mathbf{x}^*(b)$ varies smoothly.

**Theorem.** $\frac{dp^*}{db} = \frac{\partial \mathcal{L}}{\partial b}\bigg|_{\mathbf{x}^*(b), \boldsymbol{\lambda}^*(b)}$

where $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}, b) = f(\mathbf{x}, b) + \boldsymbol{\lambda}^\top \mathbf{g}(\mathbf{x}, b)$.

**Proof.** Differentiate $p^*(b) = f(\mathbf{x}^*(b), b)$ with respect to $b$:

$$\frac{dp^*}{db} = \nabla_\mathbf{x} f(\mathbf{x}^*(b), b)^\top \frac{d\mathbf{x}^*}{db} + \frac{\partial f}{\partial b}(\mathbf{x}^*(b), b)$$

From stationarity: $\nabla_\mathbf{x} f = -\boldsymbol{\lambda}^{*\top} \nabla_\mathbf{x} \mathbf{g}$. Substituting:

$$\frac{dp^*}{db} = -\boldsymbol{\lambda}^{*\top} \nabla_\mathbf{x} \mathbf{g}^\top \frac{d\mathbf{x}^*}{db} + \frac{\partial f}{\partial b}$$

Differentiating the constraint $\mathbf{g}(\mathbf{x}^*(b), b) = \mathbf{0}$ with respect to $b$:

$$\nabla_\mathbf{x} \mathbf{g}^\top \frac{d\mathbf{x}^*}{db} + \frac{\partial \mathbf{g}}{\partial b} = \mathbf{0} \quad \Rightarrow \quad \nabla_\mathbf{x} \mathbf{g}^\top \frac{d\mathbf{x}^*}{db} = -\frac{\partial \mathbf{g}}{\partial b}$$

Substituting back:

$$\frac{dp^*}{db} = \boldsymbol{\lambda}^{*\top} \frac{\partial \mathbf{g}}{\partial b} + \frac{\partial f}{\partial b} = \frac{\partial \mathcal{L}}{\partial b}\bigg|_{\mathbf{x}^*, \boldsymbol{\lambda}^*} \quad \square$$

**Consequence for ML.** If we add a constraint $\|\mathbf{w}\|^2 = c$ (norm budget) to a training problem, the Lagrange multiplier $\lambda^*$ equals $-dp^*/dc$—the rate at which the optimal loss decreases as we increase the allowed norm. This is why weight decay $\lambda$ (which enforces a soft norm budget) and the optimal multiplier are equal at the optimum: they're the same object.

### A.2 Farkas' Lemma: The Foundation of KKT

The Farkas Lemma is the linear algebra result that underpins the entire theory of KKT conditions. It provides the criterion for when a linear system has a solution versus when a certificate of infeasibility exists.

**Lemma (Farkas, 1902).** Let $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$. Exactly one of the following holds:

1. There exists $\mathbf{x} \geq \mathbf{0}$ with $A\mathbf{x} = \mathbf{b}$
2. There exists $\mathbf{y}$ with $A^\top \mathbf{y} \geq \mathbf{0}$ and $\mathbf{b}^\top \mathbf{y} < 0$

**Geometric interpretation.** Statement 1 says $\mathbf{b}$ is in the conic hull of the columns of $A$. Statement 2 says there is a hyperplane $\mathbf{y}$ separating $\mathbf{b}$ from this cone. The lemma asserts these are mutually exclusive and exhaustive.

**Proof sketch.** If 1 holds: $\mathbf{b}^\top \mathbf{y} = (A\mathbf{x})^\top \mathbf{y} = \mathbf{x}^\top A^\top \mathbf{y} \geq 0$ since $\mathbf{x} \geq 0$ and $A^\top \mathbf{y} \geq 0$, so 2 fails. The converse (if 1 fails then 2 holds) follows from the separating hyperplane theorem for convex cones.

**How Farkas gives KKT.** At a constrained minimum $\mathbf{x}^*$, if $\mathbf{d}$ is a feasible descent direction (satisfying active constraint gradients and having $\nabla f^\top \mathbf{d} < 0$), the Farkas alternatives state that such $\mathbf{d}$ cannot exist iff $-\nabla f$ is in the cone of active constraint gradients—i.e., the KKT conditions hold.

### A.3 Fritz John Conditions: When LICQ Fails

When the LICQ fails (active constraint gradients are linearly dependent), the standard KKT multiplier theorem breaks down. The **Fritz John conditions** provide a fallback:

**Theorem (Fritz John, 1948).** At a local minimum $\mathbf{x}^*$ of $f$ subject to $\mathbf{g} = \mathbf{0}$, $\mathbf{h} \leq \mathbf{0}$, there exist $(\mu_0, \boldsymbol{\mu}, \boldsymbol{\lambda}) \neq \mathbf{0}$ with $\mu_0, \boldsymbol{\mu} \geq 0$ such that:

$$\mu_0 \nabla f(\mathbf{x}^*) + \sum_j \mu_j \nabla h_j(\mathbf{x}^*) + \sum_i \lambda_i \nabla g_i(\mathbf{x}^*) = \mathbf{0}$$

and complementary slackness $\mu_j h_j(\mathbf{x}^*) = 0$.

**The anomaly.** When $\mu_0 = 0$, the objective function disappears from the condition entirely—the conditions are "degenerate" and say nothing about the objective. This is called an **abnormal solution**.

**Practical implication.** If LICQ holds, we can always normalize to $\mu_0 = 1$, recovering the standard KKT conditions. Fritz John is thus a generalization that covers degenerate cases, but the cleanest results require LICQ (or another constraint qualification that ensures $\mu_0 > 0$).

### A.4 Second-Order KKT Conditions

For constrained problems, second-order sufficient conditions involve the Hessian of the Lagrangian restricted to the tangent cone of the active constraints.

**Setup.** At a KKT point $(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$ with active set $A = \{j : h_j(\mathbf{x}^*) = 0\}$, define:
- $L(\mathbf{x}) = \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L}(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*)$ (Hessian of Lagrangian w.r.t. $\mathbf{x}$)
- The **critical cone**: $C = \{\mathbf{d} : \nabla h_j(\mathbf{x}^*)^\top \mathbf{d} \leq 0 \text{ for } j \in A,\; \nabla g_i(\mathbf{x}^*)^\top \mathbf{d} = 0\}$

**Second-order necessary condition (SONC):** If $\mathbf{x}^*$ is a local minimum, then $\mathbf{d}^\top L \mathbf{d} \geq 0$ for all $\mathbf{d} \in C$.

**Second-order sufficient condition (SOSC):** If the KKT conditions hold and $\mathbf{d}^\top L \mathbf{d} > 0$ for all $\mathbf{d} \in C \setminus \{\mathbf{0}\}$, then $\mathbf{x}^*$ is a strict local minimum.

**For SVM:** At the SVM optimum, the Hessian of the Lagrangian in the primal $(\mathbf{w}, b)$ variables is positive definite restricted to the constraint tangent space (verified by the positive definiteness of the kernel matrix for distinct support vectors)—confirming the optimum is indeed a strict constrained minimum.


### A.5 Convex Conjugate and Fenchel Duality

The Fenchel conjugate provides a powerful alternative to Lagrangian duality, particularly useful for non-smooth problems.

**Definition (Conjugate function).** For $f: \mathbb{R}^n \to \mathbb{R} \cup \{+\infty\}$:

$$f^*(\mathbf{y}) = \sup_{\mathbf{x}} \left[ \mathbf{y}^\top \mathbf{x} - f(\mathbf{x}) \right]$$

$f^*$ is always convex (pointwise supremum of affine functions) regardless of whether $f$ is convex.

**Key examples:**

| $f(\mathbf{x})$ | Domain | $f^*(\mathbf{y})$ | Domain |
|-----------------|--------|-------------------|--------|
| $\frac{1}{2}\|\mathbf{x}\|^2$ | $\mathbb{R}^n$ | $\frac{1}{2}\|\mathbf{y}\|^2$ | $\mathbb{R}^n$ |
| $\|\mathbf{x}\|_1$ | $\mathbb{R}^n$ | $0$ if $\|\mathbf{y}\|_\infty \leq 1$ else $+\infty$ | Indicator |
| $\|\mathbf{x}\|_2$ | $\mathbb{R}^n$ | $0$ if $\|\mathbf{y}\|_2 \leq 1$ else $+\infty$ | Indicator |
| $e^x$ | $\mathbb{R}$ | $y\log y - y$ | $y > 0$ |
| $-\log x$ | $x > 0$ | $-1 - \log(-y)$ | $y < 0$ |

**Fenchel duality.** The **Fenchel dual problem** of $\min_\mathbf{x} f(A\mathbf{x}) + g(\mathbf{x})$ is:

$$\max_\mathbf{y} -f^*(\mathbf{y}) - g^*(-A^\top \mathbf{y})$$

**Lasso as Fenchel dual.** The Lasso primal $\min_\mathbf{w} \frac{1}{2}\|X\mathbf{w}-\mathbf{y}\|^2 + \lambda\|\mathbf{w}\|_1$ has a Fenchel dual that is a quadratic program with box constraints. The dual variables are the residuals $\mathbf{r} = \mathbf{y} - X\mathbf{w}$ scaled by $1/\lambda$, and the dual constraint $\|X^\top \mathbf{r}\|_\infty \leq 1$ gives the KKT condition for Lasso sparsity.

### A.6 Proximal Operators and Subdifferential Calculus

For non-smooth convex problems, gradient descent fails but **proximal methods** succeed. The proximal operator is intimately connected to KKT conditions.

**Definition.** The **proximal operator** of $g$ with parameter $t > 0$:

$$\text{prox}_{tg}(\mathbf{v}) = \arg\min_\mathbf{x} \left[ g(\mathbf{x}) + \frac{1}{2t}\|\mathbf{x} - \mathbf{v}\|^2 \right]$$

**Connection to KKT.** The fixed point of $\mathbf{v} \mapsto \text{prox}_{tg}(\mathbf{v})$ is the minimiser of $g$. The KKT condition for the proximal subproblem is $\mathbf{0} \in \partial g(\mathbf{x}^*) + \frac{1}{t}(\mathbf{x}^* - \mathbf{v})$, which at the fixed point gives $\mathbf{0} \in \partial g(\mathbf{x}^*)$—the subdifferential optimality condition.

**Proximal gradient method** for $\min f(\mathbf{x}) + g(\mathbf{x})$ (smooth $f$ + non-smooth $g$):

$$\mathbf{x}_{k+1} = \text{prox}_{\eta g}(\mathbf{x}_k - \eta \nabla f(\mathbf{x}_k))$$

**For Lasso:** $\text{prox}_{\eta\lambda\|\cdot\|_1}(\mathbf{v})_j = \text{sgn}(v_j)\max(|v_j| - \eta\lambda, 0)$ (soft thresholding). This is the closed-form solution to the proximal subproblem, and it directly implements the KKT condition: the $j$-th weight is zero when the gradient is smaller than $\lambda$ (inactive constraint), nonzero otherwise.


### A.7 Constrained Optimisation in High Dimensions: Curse and Blessing

Classical optimisation theory was developed primarily for problems in $\mathbb{R}^n$ with small $n$ ($n \leq 1000$). Modern ML operates at $n = 10^9$ to $10^{11}$. Some classical results break down; others become stronger.

**What breaks:**
- Computing the Hessian: $O(n^2)$ memory, impossible for $n = 10^9$
- Solving KKT systems: $O(n^3)$ for dense systems
- Verifying positive definiteness: $O(n^2)$ eigendecomposition
- LICQ: in high dimensions, constraints from discrete data rarely achieve full-rank Jacobians

**What gets better (blessing of overparameterisation):**
- Saddle points become easier to escape: a random direction has $O(1/n)$ probability of being a descent direction at a saddle; gradient descent with noise escapes saddles in polynomial time
- Local minima become global: as shown by Kawaguchi and Du et al., the geometry of overparameterised networks makes spurious local minima rare
- Flat minima proliferate: with $n \gg N$ parameters and $N$ data points, there is an $(n-N)$-dimensional manifold of global minima—gradient descent finds some flat point in this manifold

**Practical approximations to KKT in high-$n$ regime:**
- **Gradient norm** $\|\nabla\mathcal{L}\| \leq \epsilon$ replaces exact stationarity
- **Curvature estimation**: diagonal Hessian (Adagrad, Adam) approximates second-order information
- **Sketched KKT systems**: random projections reduce system size while preserving approximate solutions
- **Dual certificate**: in sparse recovery, checking $\|X^\top(\mathbf{y} - X\hat\mathbf{w})\|_\infty \leq \lambda$ verifies the KKT condition for Lasso without solving the primal—$O(nd)$ rather than $O(d^3)$

---

## Supplement B: Worked Examples

### B.1 Complete KKT Analysis: Portfolio Optimisation

**Problem.** Markowitz mean-variance portfolio: minimise risk $\frac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w}$ subject to expected return $\boldsymbol{\mu}^\top \mathbf{w} = r$ and budget $\mathbf{1}^\top \mathbf{w} = 1$.

**Lagrangian:**

$$\mathcal{L} = \frac{1}{2}\mathbf{w}^\top \Sigma \mathbf{w} - \lambda_1(\boldsymbol{\mu}^\top \mathbf{w} - r) - \lambda_2(\mathbf{1}^\top \mathbf{w} - 1)$$

**Stationarity:** $\Sigma \mathbf{w}^* = \lambda_1^* \boldsymbol{\mu} + \lambda_2^* \mathbf{1}$

**Solution:** $\mathbf{w}^* = \Sigma^{-1}(\lambda_1^* \boldsymbol{\mu} + \lambda_2^* \mathbf{1})$ (assuming $\Sigma \succ 0$)

**Finding multipliers:** Substitute into constraints:

$$\boldsymbol{\mu}^\top \Sigma^{-1}(\lambda_1^* \boldsymbol{\mu} + \lambda_2^* \mathbf{1}) = r$$
$$\mathbf{1}^\top \Sigma^{-1}(\lambda_1^* \boldsymbol{\mu} + \lambda_2^* \mathbf{1}) = 1$$

Define: $A = \boldsymbol{\mu}^\top \Sigma^{-1} \boldsymbol{\mu}$, $B = \boldsymbol{\mu}^\top \Sigma^{-1} \mathbf{1}$, $C = \mathbf{1}^\top \Sigma^{-1} \mathbf{1}$.

System: $\begin{pmatrix} A & B \\ B & C \end{pmatrix}\begin{pmatrix}\lambda_1^* \\ \lambda_2^*\end{pmatrix} = \begin{pmatrix} r \\ 1 \end{pmatrix}$

The **efficient frontier** is the locus of $(r, \mathbf{w}^*(r))$ as $r$ varies—it's a parabola in $(r, \text{variance})$ space, parameterised by the Lagrange multiplier $\lambda_1$.

**Sensitivity:** $d(\text{min variance})/dr = \lambda_1^*$ — a one-unit increase in required return increases minimum variance by $\lambda_1^*$ units. This is the shadow price: the cost of reaching for higher returns.

### B.2 KKT Verification via Duality: Water-Filling

The **water-filling** solution for capacity maximisation in parallel Gaussian channels is a classic KKT example.

**Problem.** Maximise total capacity $\sum_i \log(1 + p_i/\sigma_i^2)$ subject to $\sum_i p_i = P$, $p_i \geq 0$.

**Lagrangian:** $\mathcal{L} = \sum_i \log(1 + p_i/\sigma_i^2) - \lambda(\sum_i p_i - P) + \sum_i \mu_i p_i$

**KKT stationarity:** $\frac{1}{\sigma_i^2 + p_i^*} = \lambda^* - \mu_i^* \leq \lambda^*$

**Complementary slackness:** $\mu_i^* p_i^* = 0$. So either $p_i^* = 0$ (channel off) or $\mu_i^* = 0$ and $p_i^* = 1/\lambda^* - \sigma_i^2$ (channel on with water level $1/\lambda^*$).

**Solution:** $p_i^* = \max(0, 1/\lambda^* - \sigma_i^2)$ where $\lambda^*$ is chosen so $\sum_i p_i^* = P$.

The water-filling metaphor: imagine filling a container with flat water level $1/\lambda^*$; channels with floor $\sigma_i^2$ above the water level get no power (inactive constraint). This is the KKT complementary slackness condition made vivid.


### B.3 RLHF as a Constrained Optimisation Problem

**Reinforcement Learning from Human Feedback (RLHF)** can be formulated as a constrained optimisation problem whose KKT conditions provide exact theoretical insight into the resulting policy.

**Problem formulation.** Let $\pi_\theta(\mathbf{y}|\mathbf{x})$ be the model policy and $\pi_{\text{ref}}$ the reference policy. Maximize expected reward subject to a KL constraint:

$$\max_\theta \mathbb{E}_{y \sim \pi_\theta}[r(\mathbf{x}, \mathbf{y})] \quad \text{s.t.} \quad \mathbb{E}_\mathbf{x}[D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})] \leq \delta$$

**Lagrangian:** $\mathcal{L} = \mathbb{E}[r(\mathbf{x}, \mathbf{y})] - \beta \cdot \mathbb{E}[D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})]$

where $\beta \geq 0$ is the Lagrange multiplier (dual variable) for the KL constraint.

**KKT stationarity** (with respect to the policy at each $(\mathbf{x}, \mathbf{y})$):

$$r(\mathbf{x}, \mathbf{y}) - \beta(\log \pi_\theta(\mathbf{y}|\mathbf{x}) - \log \pi_{\text{ref}}(\mathbf{y}|\mathbf{x})) - \beta = 0$$

**Solving for the optimal policy:**

$$\pi_\theta^*(\mathbf{y}|\mathbf{x}) = \frac{\pi_{\text{ref}}(\mathbf{y}|\mathbf{x}) \exp(r(\mathbf{x},\mathbf{y})/\beta)}{Z(\mathbf{x})}$$

where $Z(\mathbf{x}) = \sum_\mathbf{y} \pi_{\text{ref}}(\mathbf{y}|\mathbf{x}) \exp(r(\mathbf{x},\mathbf{y})/\beta)$ is the partition function.

This is exactly the **Boltzmann distribution** from max-entropy principles! The optimal RLHF policy is the reference policy **tilted** by the reward, with temperature $\beta$. Higher $\beta$ (larger KL penalty) $\Rightarrow$ stays closer to $\pi_{\text{ref}}$; lower $\beta$ $\Rightarrow$ maximises reward more aggressively.

**Complementary slackness:** If the KL budget $\delta$ is not exhausted (the model doesn't need to move far to maximise reward), then $\beta^* = 0$ and the optimal policy maximises reward unconstrained. If the constraint binds, $\beta^* > 0$ is the shadow price of KL—how much reward per unit of KL divergence the model could gain by allowing more deviation from reference.

**Direct Preference Optimisation (DPO).** DPO (Rafailov et al. 2023) avoids sampling from $\pi_\theta$ by reparameterising the reward using the KKT condition above: $r(\mathbf{x},\mathbf{y}) = \beta \log(\pi_\theta(\mathbf{y}|\mathbf{x})/\pi_{\text{ref}}(\mathbf{y}|\mathbf{x})) + \beta \log Z(\mathbf{x})$. This transforms the constrained RL problem into a supervised binary classification problem—an elegant application of duality and KKT theory.

---

## Supplement C: Computational Methods for Finding Critical Points

### C.1 Newton's Method and KKT Systems

For smooth unconstrained problems, **Newton's method** solves $\nabla f(\mathbf{x}) = \mathbf{0}$ by iterating:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

Newton's method has **quadratic convergence** near a non-degenerate critical point (where $H \succ 0$): the error satisfies $\|\mathbf{x}_{k+1} - \mathbf{x}^*\| \leq c\|\mathbf{x}_k - \mathbf{x}^*\|^2$.

**For equality-constrained problems**, Newton's method solves the KKT system:

$$\begin{pmatrix} \nabla^2_{\mathbf{x}\mathbf{x}} \mathcal{L} & \nabla_\mathbf{x} \mathbf{g}^\top \\ \nabla_\mathbf{x} \mathbf{g} & 0 \end{pmatrix} \begin{pmatrix} \Delta \mathbf{x} \\ \Delta \boldsymbol{\lambda} \end{pmatrix} = -\begin{pmatrix} \nabla_\mathbf{x} \mathcal{L} \\ \mathbf{g}(\mathbf{x}) \end{pmatrix}$$

This **KKT matrix** (also called the augmented system or saddle-point matrix) is symmetric but indefinite—it has both positive and negative eigenvalues. Solving it efficiently is the core computational challenge in constrained optimisation.

### C.2 Interior Point Methods and the Central Path

**Interior point methods** (barrier methods) solve inequality-constrained problems by converting them to a sequence of unconstrained problems. Replace $\min f$ s.t. $h_j \leq 0$ with:

$$\min_\mathbf{x} f(\mathbf{x}) - t \sum_j \log(-h_j(\mathbf{x}))$$

for decreasing $t > 0$. The log-barrier $-t\log(-h_j)$ forces iterates to stay feasible (it $\to +\infty$ as $h_j \to 0$) and approximates the indicator $\mathbf{1}[h_j \leq 0]$ as $t \to 0$.

**Central path.** The solution $\mathbf{x}^*(t)$ as a function of $t$ traces the **central path**—a smooth curve converging to the optimal $\mathbf{x}^*$ as $t \to 0$. Along the central path, the modified KKT conditions hold:

$$\nabla f(\mathbf{x}^*(t)) + \sum_j \mu_j(t) \nabla h_j(\mathbf{x}^*(t)) = \mathbf{0}, \quad \mu_j(t) h_j(\mathbf{x}^*(t)) = -t$$

As $t \to 0$: $\mu_j(t) h_j(\mathbf{x}^*(t)) \to 0$ recovers complementary slackness. Interior point methods achieve the best known polynomial complexity for LP and convex QP: $O(\sqrt{n} \log(1/\epsilon))$ iterations.

### C.3 Automatic Differentiation and Gradient Verification

Modern ML uses **automatic differentiation (AD)** to compute exact gradients of any differentiable computation graph. The core mechanism is an exact application of the chain rule through reverse mode accumulation—backpropagation.

**Gradient checking.** A standard technique for verifying backprop correctness is comparing against **finite differences**:

$$\frac{\partial \mathcal{L}}{\partial w_j} \approx \frac{\mathcal{L}(\mathbf{w} + h\mathbf{e}_j) - \mathcal{L}(\mathbf{w} - h\mathbf{e}_j)}{2h}$$

with $h = 10^{-5}$. This centred difference approximation has error $O(h^2)$, so at $h = 10^{-5}$ the finite-difference gradient should agree with the exact gradient to about 10 significant digits.

**Why finite differences are not used in practice:**
- Cost: $2n$ forward passes for $n$ parameters, vs. 1 backward pass for AD
- Numerical precision: for very large or small gradients, floating-point rounding limits accuracy

But finite differences remain the **gold standard for correctness verification** during algorithm development—if your AD gradient doesn't match finite differences at 1e-5 relative tolerance, you have a bug.


### C.4 Constraint Satisfaction in Modern ML Systems

Modern ML increasingly uses hard and soft constraints, and the optimality conditions for handling these have real-world implementations:

**Constitutional AI and RLHF with hard constraints.** Recent LLM alignment work adds explicit constraints on outputs (harmlessness, helpfulness). The dual variable $\beta$ controlling KL deviation from a reference policy is analogous to the Lagrange multiplier on the constraint "don't deviate too far from the safe base model." The KKT condition determines the optimal trade-off.

**Quantisation as constrained optimisation.** Post-training quantisation (PTQ) and quantisation-aware training (QAT) seek weight matrices $\hat{W}$ that are close to $W$ but have values restricted to a discrete grid. This is:

$$\min_{\hat{W}} \|W - \hat{W}\|_F^2 \quad \text{s.t.} \quad \hat{W}_{ij} \in \{-q, \ldots, q\}$$

The optimal solution is the projection onto the nearest grid point (rounding)—the KKT conditions confirm that the Lagrange multiplier is the quantisation error, and the optimal $\hat{W}$ satisfies complementary slackness: either the weight is at the grid boundary (multiplier nonzero) or it's rounded to the nearest interior point (multiplier zero).

**FlashAttention and memory constraints.** FlashAttention (Dao et al. 2022) reformulates the attention computation to respect SRAM capacity constraints. The tiling strategy solves a constrained I/O minimisation problem: process tiles of $Q, K, V$ that fit in SRAM while computing exact attention. The optimal tile size is determined by KKT-like analysis of the memory hierarchy constraints.

---

## Supplement D: Visualisation Guide

### D.1 Saddle Point Visualisation

The canonical saddle point is $f(x,y) = x^2 - y^2$ at the origin. Its Hessian is $\text{diag}(2, -2)$—indefinite. Level curves are hyperbolas; the gradient flow converges to the origin along the $y$-axis but diverges along the $x$-axis.

**In high dimensions.** For an $n$-parameter saddle with Morse index $k$, random gradient descent escapes in time $O(\log(1/\epsilon)/\epsilon^2)$ steps when corrupted with Gaussian noise of variance $\epsilon^2$ (Jin et al. 2017). The escape direction is the eigenvector corresponding to the most negative eigenvalue.

### D.2 Level Curve Analysis

For any scalar $f: \mathbb{R}^2 \to \mathbb{R}$:
- **Near a local minimum**: level curves are approximately ellipses centred at $\mathbf{x}^*$; aspect ratio given by $\sqrt{\lambda_{\max}/\lambda_{\min}}$ of the Hessian
- **Near a saddle point**: level curves form hyperbolas; the asymptotes are the stable/unstable manifolds  
- **Near a local maximum**: level curves are ellipses with gradient pointing inward (decreasing $f$)

The **condition number** of the Hessian at a minimum determines the eccentricity of the ellipses. High condition number (elongated ellipses) means gradient descent zigzags slowly; Newton's method normalises this.

### D.3 Constraint Geometry

For $\min f$ s.t. $g = 0$:
- The feasible set is a manifold (curve in $\mathbb{R}^2$, surface in $\mathbb{R}^3$)
- Level curves of $f$ foliate the ambient space
- The minimum is where a level curve is **tangent** to the constraint manifold
- Tangency means $\nabla f \perp T_{\mathbf{x}^*}$ (constraint tangent space), i.e., $\nabla f \parallel \nabla g$—which is exactly Lagrange's condition

The visual clarity of this geometric argument is why Lagrange's method is so compelling: it replaces algebra with geometry.

---

## Supplement E: Key Formulas Reference

```
OPTIMALITY CONDITIONS CHEAT SHEET
════════════════════════════════════════════════════════════════════════

UNCONSTRAINED
─────────────
  FONC: ∇f(x*) = 0                    (necessary)
  SONC: H(x*) ≽ 0                     (necessary)
  SOSC: H(x*) ≻ 0 (+ FONC)            (sufficient for strict local min)
  Convex: FONC iff global min

EQUALITY CONSTRAINTS (g(x) = 0)
────────────────────────────────
  Lagrangian: L = f + λᵀg
  KKT: ∇ₓL = 0  and  g(x*) = 0
  SOSC on tangent: dᵀ∇²ₓₓL d > 0  ∀d: ∇g(x*)ᵀd = 0

INEQUALITY CONSTRAINTS (h(x) ≤ 0)
───────────────────────────────────
  Lagrangian: L = f + μᵀh + λᵀg
  KKT (4 conditions):
    (1) Stationarity:   ∇ₓL = 0
    (2) Primal feas.:   h(x*) ≤ 0, g(x*) = 0
    (3) Dual feas.:     μ* ≥ 0
    (4) Comp. slack.:   μ*ᵢ hᵢ(x*) = 0 ∀i

DUALITY
───────
  Dual function: q(μ,λ) = inf_x L(x, μ, λ)
  Weak duality:  d* ≤ p* always
  Strong duality: d* = p* under Slater's condition (convex, strictly feas.)
  Shadow price:  ∂p*/∂bᵢ = λᵢ* (sensitivity of optimal value)

SPECIFIC PROBLEMS
─────────────────
  OLS:     (XᵀX)w = Xᵀy          (normal equations = FONC)
  Ridge:   (XᵀX + nλI)w = Xᵀy   (FONC with L2 reg)
  SVM:     w* = Σᵢ αᵢ yᵢ xᵢ      (stationarity); αᵢ > 0 iff on margin
  PCA:     Σv = λv                (FONC for variance max on sphere)
  Softmax: pᵢ ∝ exp(zᵢ)          (FONC for max entropy with logit constraints)
  RLHF:   π*(y|x) ∝ πᵣₑf exp(r/β) (KKT for KL-constrained reward max)

════════════════════════════════════════════════════════════════════════
```


---

## References and Further Reading

### Primary References

1. **Boyd & Vandenberghe, "Convex Optimization" (2004)** — Chapters 4–5 (convex functions, duality), Chapter 5 (KKT conditions). The standard graduate reference. Available free online at stanford.edu/~boyd/cvxbook.
2. **Nocedal & Wright, "Numerical Optimization" (2006)** — Chapters 12–15. Rigorous treatment of constrained optimisation, KKT theory, interior point methods, and convergence analysis.
3. **Rockafellar, "Convex Analysis" (1970)** — The foundational text on subdifferentials, conjugate functions, and duality. Rigorous but dense.
4. **Bertsekas, "Nonlinear Programming" (2016)** — Chapter 3 (Lagrange multiplier methods) and Chapter 4 (duality). Excellent treatment of constraint qualifications.

### For ML/AI Connections

5. **Dauphin et al. "Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization" (2014)** — Empirical evidence for saddle point dominance in deep networks.
6. **Foret et al. "Sharpness-Aware Minimization for Efficiently Improving Generalization" (2021)** — SAM algorithm; KKT analysis of the inner maximisation.
7. **Papyan et al. "Prevalence of Neural Collapse during the Terminal Phase of Deep Learning Training" (2020)** — Neural collapse as KKT critical point of the UFM.
8. **Rafailov et al. "Direct Preference Optimization" (2023)** — DPO derivation from RLHF KKT conditions.
9. **Du et al. "Gradient Descent Finds Global Minima of Non-Convex Neural Networks" (2018)** — Overparameterisation and global optimality.
10. **Kawaguchi "Deep Learning without Poor Local Minima" (2016)** — Theoretical analysis of local minima in deep linear networks.

### Computational Tools

11. **CVXPY** (Diamond & Boyd 2016) — Python library for convex optimisation with automatic KKT verification. Handles LP, QP, SOCP, SDP with strong duality guarantees.
12. **SciPy `scipy.optimize.minimize`** — Numerical optimisation with constraint support; uses SLSQP (Sequential Least Squares Programming) which solves KKT subproblems at each step.

---

*This section is part of the §05 Multivariate Calculus chapter. For gradient descent algorithms and convergence theory using these optimality conditions, see [§05/05 Gradient Descent and Convergence](../05-Gradient-Descent/notes.md). For the probability chapter where Lagrange multipliers appear in maximum likelihood and Bayesian estimation, see §06.*


---

## Supplement F: Extended Theory

### F.1 The Morse Lemma and Local Geometry of Non-Degenerate Critical Points

**Theorem (Morse Lemma).** Near a non-degenerate critical point $\mathbf{x}^*$ (where $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $H(\mathbf{x}^*)$ is invertible), there exist local coordinates $\mathbf{u}$ in which $f$ has the form:

$$f(\mathbf{x}^* + \mathbf{u}) = f(\mathbf{x}^*) - u_1^2 - \cdots - u_k^2 + u_{k+1}^2 + \cdots + u_n^2$$

where $k$ is the Morse index (number of negative eigenvalues of $H$).

**Implication.** All non-degenerate critical points have the *same local geometry* as pure quadratics—up to a smooth change of coordinates. This is why the Hessian gives complete local information about a critical point: it completely determines the topological type of the critical point.

**For AI.** In transformer loss landscapes, measuring the Morse index of critical points found during training is computationally expensive but theoretically illuminating. Empirical studies using Lanczos approximations of $H$ suggest that good minima have small Morse index (few descent directions), while saddles found earlier in training have large index.

### F.2 Generalized Saddle Points and Minimax Problems

**Definition.** A point $(\mathbf{x}^*, \mathbf{y}^*)$ is a **saddle point** of $f(\mathbf{x}, \mathbf{y})$ if:

$$f(\mathbf{x}^*, \mathbf{y}) \leq f(\mathbf{x}^*, \mathbf{y}^*) \leq f(\mathbf{x}, \mathbf{y}^*) \quad \forall \mathbf{x}, \mathbf{y}$$

(minimum over $\mathbf{x}$, maximum over $\mathbf{y}$).

**Existence theorem.** If $f$ is convex in $\mathbf{x}$ and concave in $\mathbf{y}$, and both domains are compact convex sets, then a saddle point exists (Von Neumann minimax theorem).

**Optimality conditions for saddle points:**

$$\nabla_\mathbf{x} f(\mathbf{x}^*, \mathbf{y}^*) = \mathbf{0}, \quad \nabla_\mathbf{y} f(\mathbf{x}^*, \mathbf{y}^*) = \mathbf{0}$$

This looks identical to an unconstrained critical point, but the additional saddle structure (convex-concave) ensures the point is a minimax rather than an arbitrary stationarity point.

**GAN training as saddle point optimisation.** Generative Adversarial Networks minimise the objective:

$$\min_G \max_D V(G, D) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_\mathbf{z}}[\log(1 - D(G(\mathbf{z})))]$$

The optimal discriminator $D^*$ maximises $V$ for fixed $G$; the optimal generator $G^*$ minimises $V$ for fixed $D^*$. At the Nash equilibrium (saddle point), $G^*$ generates samples from $p_{\text{data}}$ and $D^* = 1/2$ everywhere.

The challenge: the convex-concave structure is not present in neural network GAN training (both $G$ and $D$ are non-convex), so saddle point existence is not guaranteed and gradient descent alternation can cycle or diverge.

### F.3 Proximal Point Algorithm and Augmented Lagrangian

**Proximal Point Algorithm (PPA).** For convex $f$:

$$\mathbf{x}_{k+1} = \arg\min_\mathbf{x} \left[ f(\mathbf{x}) + \frac{1}{2\eta}\|\mathbf{x} - \mathbf{x}_k\|^2 \right] = \text{prox}_{\eta f}(\mathbf{x}_k)$$

This is equivalent to finding the KKT point of the proximal subproblem: $\nabla f(\mathbf{x}_{k+1}) + \frac{1}{\eta}(\mathbf{x}_{k+1} - \mathbf{x}_k) = \mathbf{0}$.

**Augmented Lagrangian (method of multipliers).** For $\min f(\mathbf{x})$ s.t. $A\mathbf{x} = \mathbf{b}$, the augmented Lagrangian:

$$\mathcal{L}_\rho(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \boldsymbol{\lambda}^\top(A\mathbf{x}-\mathbf{b}) + \frac{\rho}{2}\|A\mathbf{x}-\mathbf{b}\|^2$$

combines the standard Lagrangian with a quadratic penalty term. The quadratic term is zero at feasibility, so the KKT conditions for the augmented Lagrangian match the original. But the augmented Lagrangian is better conditioned—the $\rho\|A\mathbf{x}-\mathbf{b}\|^2$ term adds curvature near the constraint.

**ADMM.** The Alternating Direction Method of Multipliers (ADMM) is an augmented Lagrangian method split across two blocks of variables. It is widely used for distributed training, consensus optimisation, and lasso-type problems. At convergence, the ADMM iterates satisfy the KKT conditions of the original problem.

### F.4 Implicit Function Theorem and Sensitivity Analysis

The KKT conditions define a nonlinear system $F(\mathbf{x}^*, \boldsymbol{\mu}^*, \boldsymbol{\lambda}^*; \boldsymbol{\theta}) = \mathbf{0}$ parameterised by $\boldsymbol{\theta}$ (problem data). The **Implicit Function Theorem (IFT)** tells us when and how the optimal solution changes with $\boldsymbol{\theta}$.

**Theorem (IFT).** If $F(\mathbf{z}^*; \boldsymbol{\theta}^*) = \mathbf{0}$ and the Jacobian $\nabla_\mathbf{z} F(\mathbf{z}^*; \boldsymbol{\theta}^*)$ is invertible, then there exists a smooth function $\mathbf{z}^*(\boldsymbol{\theta})$ near $\boldsymbol{\theta}^*$ with $F(\mathbf{z}^*(\boldsymbol{\theta}); \boldsymbol{\theta}) = \mathbf{0}$, and:

$$\frac{\partial \mathbf{z}^*}{\partial \boldsymbol{\theta}} = -[\nabla_\mathbf{z} F]^{-1} \nabla_{\boldsymbol{\theta}} F$$

**Application: differentiating through optimisation.** In **meta-learning** (MAML) and **bilevel optimisation**, we need $\partial \mathbf{x}^*(\boldsymbol{\theta})/\partial \boldsymbol{\theta}$—the derivative of the optimal solution with respect to parameters. The IFT gives this derivative via solving a linear system involving the KKT Jacobian.

**MAML connection.** Model-Agnostic Meta-Learning computes:

$$\frac{d\mathcal{L}_{\text{meta}}(\theta - \alpha \nabla\mathcal{L}_{\text{task}}(\theta))}{d\theta}$$

This requires backpropagating through the inner gradient step—equivalent to applying the IFT to the first-order optimality condition $\nabla\mathcal{L}_{\text{task}}(\mathbf{x}^*(\theta)) = \mathbf{0}$ of the inner problem.

### F.5 Optimality and Generalisation: The Flat Minima Hypothesis

A deep connection links optimality conditions to generalisation in deep learning. The **sharpness** of a minimum—measured by the trace or maximum eigenvalue of the Hessian—correlates empirically with test performance.

**Measures of sharpness:**
- $\text{tr}(H) = \sum_i \lambda_i(H)$: total curvature (Frobenius norm of $H^{1/2}$)
- $\lambda_{\max}(H)$: maximum curvature (sharpness in the steepest direction)
- $\Phi_\epsilon(\mathbf{w}^*) = \max_{\|\boldsymbol{\epsilon}\| \leq \rho} \mathcal{L}(\mathbf{w}^* + \boldsymbol{\epsilon}) - \mathcal{L}(\mathbf{w}^*)$: PAC-Bayes sharpness

**Why flat minima generalise better (hypothesis).** A flat minimum has many parameters with nearly zero curvature. Small perturbations to the weights (due to finite data, distribution shift, or quantisation error) cause small loss increases. A sharp minimum has high curvature: small weight perturbations cause large loss spikes, leading to poor generalisation.

**PAC-Bayes bound.** The PAC-Bayes generalisation bound for a posterior $Q$ over weights near $\mathbf{w}^*$ includes a term:

$$\text{KL}(Q \| P) \leq \sum_i \frac{(\sigma_i)^2 \lambda_i(H)}{2}$$

where $P$ is the prior. This directly penalises large Hessian eigenvalues—confirming that flat minima (small eigenvalues) admit tighter generalisation bounds.

**Implication for optimality.** The "best" solution from a generalisation perspective is not just any critical point—it is a critical point that is also a *flat* critical point: one where $\lambda_{\max}(H)$ is minimised. SAM seeks this; the KKT condition for the SAM minimax problem gives the perturbation direction that reveals the sharpest descent direction.


---

## Supplement G: Historical Perspective

### G.1 Timeline of Key Developments

```
HISTORY OF OPTIMALITY CONDITIONS
════════════════════════════════════════════════════════════════════════

  1788  Lagrange — "Mécanique Analytique": Lagrange multipliers for
        constrained mechanics. Invented to handle pendulum constraints.

  1815  Gauss — Method of least squares: unconstrained minimisation,
        normal equations as first-order conditions.

  1844  Cauchy — Gradient descent proposed as an optimisation method.
        First explicit use of the gradient as a direction.

  1902  Farkas — Farkas Lemma for linear systems of inequalities.
        The algebraic foundation of all LP and KKT theory.

  1928  Von Neumann — Minimax theorem for zero-sum games.
        Foundation of saddle-point theory and game theory.

  1939  Karush — Master's thesis: first-order conditions for
        inequality-constrained problems. (Unknown for decades.)

  1951  Kuhn & Tucker — Independent rediscovery, published in
        "Nonlinear Programming". Conditions become "KKT" in their honor
        (Karush's contribution recognised only later).

  1953  Wolfe — Duality theorem for nonlinear programming.

  1955  Charnes, Cooper, Henderson — Simplex method, LP duality.

  1963  Zoutendijk — Feasible direction methods, active set concepts.

  1979  Khachiyan — Ellipsoid method: first polynomial algorithm for LP.

  1984  Karmarkar — Interior point method for LP, O(n^3.5 L) time.
        Practical polynomial LP solver.

  1993  Cortes & Vapnik — Support Vector Machine: KKT dual is the
        modern SVM. Complementary slackness identifies support vectors.

  1996  Byrd et al. — L-BFGS-B: limited memory quasi-Newton for
        bound-constrained problems. Industry standard for large-scale
        constrained optimisation.

  2011  Duchi, Hazan, Singer — Adagrad: adaptive learning rate
        implicitly approximates inverse Hessian structure (second-order).

  2014  Dauphin et al. — Saddle point dominance in deep learning.
        First systematic empirical study of neural network landscape.

  2015  Kingma & Ba — Adam optimiser: diagonal Hessian approximation
        with momentum. Becomes default for LLM training.

  2018  Du et al. — Gradient descent finds global minima in wide nets.
        Theoretical basis for why deep learning "works."

  2021  Foret et al. — SAM: Sharpness-Aware Minimisation. KKT-derived
        update for flatter minima. New SOTA on multiple benchmarks.

  2021  Papyan et al. — Neural Collapse: KKT analysis of terminal
        training geometry. UFM framework.

  2023  Rafailov et al. — DPO: RLHF framed as KKT-constrained
        optimisation. Replaces PPO for LLM alignment.

════════════════════════════════════════════════════════════════════════
```

### G.2 The Karush Omission

The Karush–Kuhn–Tucker conditions are named for Kuhn and Tucker (1951), but Karush derived essentially the same conditions in his 1939 master's thesis at the University of Chicago under L.M. Graves. Karush never published his thesis, and his work remained unknown until the 1970s when Kuhn acknowledged the priority.

This is a cautionary tale for researchers: unpublished work does not enter the scientific record. Karush's conditions are mathematically equivalent to KKT and were derived first, but priority in science is determined by publication and dissemination.

Modern convention is to include Karush's name (KKT), and some authors use "FOOC" (first-order optimality conditions) to avoid the naming entirely.

### G.3 From Constrained Mechanics to Language Models

Lagrange invented his multiplier method to handle constraints in mechanics—specifically, the rigid-rod constraint in a pendulum. The constraint that "the rod length is fixed" is $|\mathbf{r}|^2 = L^2$, and the Lagrange multiplier is the tension in the rod.

Two centuries later, the same mathematical structure appears in:
- The tension (KL penalty) in RLHF that prevents language models from deviating too far from the base policy
- The margin constraint in SVMs, where the Lagrange multiplier is the "tension" pulling the decision boundary toward the support vectors
- The softmax temperature, which is the Lagrange multiplier for the entropy constraint in maximum entropy next-token prediction

The deep unity of Lagrange's insight—that constraints can be absorbed into an objective via auxiliary multipliers—is one of the most productive ideas in the history of mathematics.


---

## Supplement H: Problem Reduction Patterns

### H.1 Reducing Constrained Problems to Unconstrained

Many constrained problems have hidden structure that allows elimination of constraints, converting a constrained problem to unconstrained.

**Pattern 1: Substitution.** If constraint $g(\mathbf{x}) = 0$ can be solved for one variable, substitute to eliminate it.

Example: $\min f(x,y)$ s.t. $y = x^2$. Substitute: $\min_{x} f(x, x^2)$. The constraint disappears, FONC is $\frac{d}{dx}f(x, x^2) = 0$—an unconstrained problem.

**Pattern 2: Reparameterisation.** If the constraint defines a manifold $\mathcal{M}$, parameterise $\mathcal{M}$ directly.

Example: Unit sphere $\|\mathbf{x}\|=1$ in $\mathbb{R}^3$: parameterise as $\mathbf{x} = (\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta)$. Optimise over $(\theta, \phi) \in \mathbb{R}^2$ without any constraint.

For ML: **Stiefel manifold optimisation** (orthonormal matrices) uses geodesic gradient descent on the manifold, avoiding explicit constraint handling. Used in RNN training and low-rank optimisation.

**Pattern 3: Lagrangian method.** When substitution is impractical, Lagrange multipliers convert the constrained problem to a (possibly harder) unconstrained problem in augmented variables.

**Pattern 4: Penalty methods.** Replace $\min f$ s.t. $g = 0$ with $\min f + \frac{\rho}{2}|g|^2$. As $\rho \to \infty$, the penalised solution approaches the constrained solution. Less accurate than Lagrange but simpler to implement.

### H.2 Problem Recognition Guide

When you encounter an optimisation problem, use this decision tree:

```
OPTIMIZATION PROBLEM RECOGNITION
════════════════════════════════════════════════════════════════════════

  Does it have constraints?
  ├── NO → Unconstrained
  │         Is f convex?
  │         ├── YES → FONC (∇f = 0) is sufficient; any local = global
  │         └── NO → FONC necessary; check SOSC; may have local optima
  │
  └── YES → Constrained
             Are constraints equality only?
             ├── YES → Lagrange multipliers; solve n+m equations
             └── NO (has inequalities) → KKT conditions
                  Is problem convex?
                  ├── YES → KKT iff global optimum (Slater's → strong duality)
                  └── NO → KKT necessary (LICQ needed); local optimum only

════════════════════════════════════════════════════════════════════════
```

### H.3 Standard Problem Forms and Their KKT Systems

**Linear Program (LP):**
$$\min \mathbf{c}^\top\mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b},\; \mathbf{x} \geq 0$$
KKT: $\mathbf{c} - A^\top \boldsymbol{\lambda} - \boldsymbol{\mu} = \mathbf{0}$, $\mu_i x_i = 0$, $\boldsymbol{\mu} \geq 0$

**Quadratic Program (QP):**
$$\min \frac{1}{2}\mathbf{x}^\top Q\mathbf{x} + \mathbf{c}^\top \mathbf{x} \quad \text{s.t.} \quad A\mathbf{x} = \mathbf{b},\; G\mathbf{x} \leq \mathbf{h}$$
KKT: $Q\mathbf{x}^* + \mathbf{c} + A^\top\boldsymbol{\lambda}^* + G^\top\boldsymbol{\mu}^* = \mathbf{0}$, standard feasibility and complementary slackness

**Second-Order Cone Program (SOCP):**
$$\min \mathbf{c}^\top\mathbf{x} \quad \text{s.t.} \quad \|A_i \mathbf{x} + \mathbf{b}_i\| \leq \mathbf{c}_i^\top \mathbf{x} + d_i$$
Subsumes QP and LP; solvable in polynomial time; KKT conditions involve second-order cone complementarity

**Semidefinite Program (SDP):**
$$\min \langle C, X \rangle \quad \text{s.t.} \quad \langle A_i, X \rangle = b_i,\; X \succeq 0$$
KKT: matrix complementarity $XZ = 0$ where $Z$ is the dual variable; powerful for combinatorial relaxations and spectral graph problems

These standard forms are recognised and solved by CVXPY and other convex optimisation packages. Expressing your ML problem in one of these forms gives you access to provably correct, polynomial-time solvers.


---

## Supplement I: Numerical Experiments and Verification

### I.1 Verifying KKT Conditions Numerically

In practice, KKT conditions are verified numerically after solving an optimisation problem. Here is a systematic protocol:

```python
def verify_kkt(x_star, lambda_star, mu_star, f, g, h, tol=1e-6):
    """
    Verify KKT conditions for problem:
      min f(x) s.t. g(x) = 0, h(x) <= 0
    at candidate solution (x_star, lambda_star, mu_star).
    """
    # 1. Stationarity
    grad_f = numerical_gradient(f, x_star)
    grad_g = numerical_jacobian(g, x_star)  # (m, n)
    grad_h = numerical_jacobian(h, x_star)  # (p, n)
    stationarity = grad_f + lambda_star @ grad_g + mu_star @ grad_h
    cond1 = np.allclose(stationarity, 0, atol=tol)

    # 2. Primal feasibility
    eq_residual = g(x_star)
    ineq_residual = h(x_star)
    cond2 = np.allclose(eq_residual, 0, atol=tol) and np.all(ineq_residual <= tol)

    # 3. Dual feasibility
    cond3 = np.all(mu_star >= -tol)

    # 4. Complementary slackness
    cs = mu_star * h(x_star)
    cond4 = np.allclose(cs, 0, atol=tol)

    print(f"Stationarity:        {'PASS' if cond1 else 'FAIL'}")
    print(f"Primal feasibility:  {'PASS' if cond2 else 'FAIL'}")
    print(f"Dual feasibility:    {'PASS' if cond3 else 'FAIL'}")
    print(f"Complementary slack: {'PASS' if cond4 else 'FAIL'}")
    return cond1 and cond2 and cond3 and cond4
```

This systematic verification—checking all four KKT conditions numerically—is a key debugging tool when implementing constrained optimisation algorithms.

### I.2 Duality Gap Monitoring

For convex problems, monitoring the duality gap during optimisation provides both a convergence certificate and a diagnostic:

```python
def duality_gap(primal_val, dual_val):
    """
    Compute duality gap. Should be >= 0 (weak duality).
    Converges to 0 at optimality for convex problems (strong duality).
    """
    gap = primal_val - dual_val
    assert gap >= -1e-8, f"Negative duality gap: {gap}"  # weak duality violation
    return gap
```

A duality gap of $\epsilon$ certifies that both the primal and dual solutions are within $\epsilon$ of optimality—a **certificate of near-optimality** without knowing the true optimal value.

### I.3 Critical Point Classification Checklist

When you find a critical point $\mathbf{x}^*$ numerically, this checklist determines its type:

| Step | Check | How | Interpretation |
|------|-------|-----|----------------|
| 1 | Verify stationarity | $\|\nabla f(\mathbf{x}^*)\| < 10^{-6}$ | True critical point (not just near-critical) |
| 2 | Compute Hessian | $H = \nabla^2 f(\mathbf{x}^*)$ | Via AD or finite differences |
| 3 | Eigenvalue decomposition | $H = Q\Lambda Q^\top$ | Sort eigenvalues |
| 4 | Count negatives | $k = \#\{\lambda_i < -10^{-8}\}$ | Morse index |
| 5 | Count near-zero | $m = \#\{|\lambda_i| < 10^{-8}\}$ | Flat directions |
| 6 | Classify | $k=0$: min; $k=n$: max; else: saddle | Standard classification |
| 7 | Check SOSC | All $\lambda_i > 0$ strictly | Strict local min |
| 8 | Verify global (if convex) | FONC alone suffices | No need for SOSC |

This procedure is the **numerical implementation of the second-order test** and is embedded in standard optimisation post-processing.

---

## Summary: The Three Pillars of Optimality

This section has developed three interconnected pillars that together form the mathematical foundation of constrained optimisation:

**Pillar 1: First and Second-Order Conditions.** The gradient and Hessian encode all local information about a critical point. FONC ($\nabla f = 0$) is necessary; SOSC ($H \succ 0$) is sufficient. For convex problems, FONC is sufficient for global optimality—the Hessian never needs to be checked.

**Pillar 2: Lagrange Multipliers and KKT.** Constraints are not obstacles but information: each active constraint generates a Lagrange multiplier (shadow price) measuring the sensitivity of the objective to that constraint. The KKT conditions generalize FONC to handle both equality and inequality constraints, with complementary slackness providing the geometric signature of which constraints matter.

**Pillar 3: Duality.** Every constrained problem has a dual that provides a lower bound (weak duality) and—for convex problems with a Slater point—an exact alternative formulation (strong duality). The dual reveals hidden structure (SVM kernel trick), provides certificates of optimality (duality gap), and connects constrained optimisation to game theory (saddle points).

These three pillars appear throughout AI and ML: gradient-based training (Pillar 1), SVM/PCA/attention (Pillar 2), and RL/alignment (Pillar 3). Understanding them deeply is not optional for a serious ML researcher—they are the grammar of the field.


---

## Supplement J: Connections to Information Geometry

### J.1 Fisher Information and the Natural Gradient

The gradient $\nabla_\theta \mathcal{L}$ is computed with respect to the Euclidean metric on parameter space. But parameter spaces of probability distributions carry a natural Riemannian metric—the **Fisher information metric**:

$$g_{ij}(\theta) = \mathbb{E}_{x \sim p_\theta}\left[\frac{\partial \log p_\theta(x)}{\partial \theta_i} \frac{\partial \log p_\theta(x)}{\partial \theta_j}\right] = F_{ij}(\theta)$$

**Natural gradient.** The gradient of $\mathcal{L}$ in the Fisher metric is:

$$\tilde{\nabla}_\theta \mathcal{L} = F(\theta)^{-1} \nabla_\theta \mathcal{L}$$

The natural gradient step $\theta \leftarrow \theta - \eta F^{-1} \nabla\mathcal{L}$ is the KKT condition for:

$$\min_{\Delta\theta} \nabla\mathcal{L}^\top \Delta\theta + \frac{1}{2\eta}\Delta\theta^\top F \Delta\theta$$

(minimise linearised loss subject to a KL-ball constraint $\Delta\theta^\top F \Delta\theta \leq \text{const}$).

**K-FAC and Adam as natural gradient approximations.** K-FAC (Kronecker-Factored Approximate Curvature) approximates $F^{-1}$ via Kronecker products for efficiency. Adam's adaptive learning rates approximate $\text{diag}(F)^{-1}$—a diagonal approximation to the Fisher inverse. Both are principled from the viewpoint of natural gradient descent, whose KKT formulation makes the Fisher metric (= second-order constraint) explicit.

### J.2 Optimal Transport and Wasserstein Duality

The **Wasserstein distance** between distributions $\mu$ and $\nu$ over $\mathbb{R}^d$ is defined as:

$$W_2(\mu, \nu)^2 = \min_{\gamma \in \Pi(\mu,\nu)} \mathbb{E}_{(x,y) \sim \gamma}[\|x-y\|^2]$$

where $\Pi(\mu,\nu)$ is the set of joint distributions with marginals $\mu$ and $\nu$.

**Kantorovich duality.** By Lagrange duality (the constraint is linear in $\gamma$), the Wasserstein distance admits a dual:

$$W_2(\mu,\nu)^2 = \max_{\phi,\psi: \phi(x)+\psi(y) \leq \|x-y\|^2} \int \phi\, d\mu + \int \psi\, d\nu$$

The dual functions $\phi$ and $\psi$ are the Lagrange multipliers for the marginal constraints $\int \gamma(x, \cdot)\, dy = d\mu(x)$ and $\int \gamma(\cdot, y)\, dx = d\nu(y)$.

**For LLMs.** Wasserstein GANs (WGANs) use the Kantorovich dual to train discriminators with 1-Lipschitz constraint (the dual constraint simplified). The dual formulation is more stable than the Jensen-Shannon divergence of standard GANs—a direct application of duality theory to generative model training.

---

*End of §04 Optimality Conditions notes. This section is 2000+ lines covering unconstrained/constrained optimality from first principles through modern AI applications.*

*Navigation: [← Chain Rule and Backpropagation](../03-Chain-Rule-and-Backpropagation/notes.md) | [Next: Gradient Descent and Convergence →](../05-Gradient-Descent/notes.md)*


---

## Supplement K: Extended Exercise Solutions (Hints and Partial Solutions)

### Exercise 1 Hints

**(a)** Finding critical points of $f(x,y) = x^3 - 3xy^2 + y^4$:

$\partial f/\partial x = 3x^2 - 3y^2 = 3(x-y)(x+y) = 0$

$\partial f/\partial y = -6xy + 4y^3 = 2y(2y^2 - 3x) = 0$

Cases: ($y=0$ or $x = \pm y$) combined with ($y=0$ or $x = 2y^2/3$).

For $y=0$: from $\partial f/\partial x = 0$ get $x=0$. Critical point: $(0,0)$.

For $y \neq 0$, $x = y$ or $x = -y$: combine with $x = 2y^2/3$.

**(b)** At $(0,0)$: $H = \begin{pmatrix}6x & -6y \\ -6y & -6x+12y^2\end{pmatrix}\bigg|_{(0,0)} = \begin{pmatrix}0 & 0 \\ 0 & 0\end{pmatrix}$. Hessian is zero—degenerate critical point, inconclusive test. Must use higher-order analysis (origin is a saddle of $f$).

### Exercise 4 Hints

**(c)** $f(A) = -\log \det A$ on $\mathbb{S}_{++}^n$: this is convex. Key: compute the Hessian in the matrix sense. For a perturbation $\Delta$: $-\log\det(A+\Delta) \approx -\log\det A + \text{tr}(A^{-1}\Delta) + \frac{1}{2}\text{tr}(A^{-1}\Delta A^{-1}\Delta) + \ldots$ The second-order term is $\frac{1}{2}\|A^{-1/2}\Delta A^{-1/2}\|_F^2 \geq 0$.

**(d)** $f(x,y) = x^2/y$ for $y > 0$. Hessian: $H = \begin{pmatrix} 2/y & -2x/y^2 \\ -2x/y^2 & 2x^2/y^3 \end{pmatrix}$. Check: $\det H = (2/y)(2x^2/y^3) - (2x/y^2)^2 = 4x^2/y^4 - 4x^2/y^4 = 0$. PSD but not PD. Actually convex: it's a perspective function $f(x,y) = g(x,y) = x^2/y$—the perspective of the convex $g(x)=x^2$.

### Exercise 7 Extended Notes: SAM Convergence

The SAM update $\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla\mathcal{L}(\mathbf{w} + \hat{\boldsymbol{\epsilon}})$ can be viewed as gradient descent on the perturbed objective $\mathcal{L}^{\text{SAM}}(\mathbf{w}) \approx \mathcal{L}(\mathbf{w}) + \rho \|\nabla\mathcal{L}(\mathbf{w})\|$. 

The KKT condition of the inner maximisation shows that $\hat{\boldsymbol{\epsilon}} = \rho \nabla\mathcal{L}/\|\nabla\mathcal{L}\|$ is correct when the constraint $\|\boldsymbol{\epsilon}\| \leq \rho$ is active (which it always is for nonzero gradients). The multiplier $\mu^* = \|\nabla\mathcal{L}\|/(2\rho)$ is the shadow price of relaxing the perturbation budget.

This connection shows SAM is not heuristic—it is the exact solution to a well-defined KKT problem. The sharpness regularisation effect comes from the fact that at a sharp minimum, $\|\nabla\mathcal{L}(\mathbf{w} + \hat{\boldsymbol{\epsilon}})\|$ is large (large Hessian eigenvalue $\times$ gradient), so SAM takes a larger effective step away from sharp directions.


---

## Supplement L: Quick Reference Tables

### Lagrange vs KKT: Choosing the Right Tool

| Situation | Method | Key Condition |
|-----------|--------|---------------|
| No constraints | First/Second order | $\nabla f = 0$, check $H$ |
| Equality constraints only | Lagrange multipliers | $\nabla f + \lambda^\top \nabla g = 0$ |
| Inequality constraints | KKT | 4 conditions including $\mu \geq 0$, CS |
| Mixed equality + inequality | KKT | Full 4-condition system |
| Convex problem | KKT (global) | KKT $\Leftrightarrow$ global min (Slater holds) |
| Non-convex problem | KKT (local) | KKT $\Rightarrow$ local min only |
| LP/QP | Interior point or simplex | KKT system solved directly |
| Non-smooth $f$ | Subdifferential | $0 \in \partial f + \partial h$ |

### Shadow Price Interpretation Guide

| Context | Multiplier meaning |
|---------|--------------------|
| Budget constraint $\mathbf{1}^\top\mathbf{w} \leq B$ | $\lambda^*$ = value of one more unit of budget |
| Norm constraint $\|\mathbf{w}\|^2 \leq c$ | $\lambda^*$ = decrease in loss per unit norm increase |
| SVM margin $y(\mathbf{w}^\top\mathbf{x}+b) \geq 1$ | $\alpha_i^*$ = importance of sample $i$ to the decision boundary |
| KL constraint $D_{\text{KL}} \leq \delta$ in RLHF | $\beta^*$ = reward per unit KL allowed (the "temperature") |
| Power budget $\sum p_i = P$ (water-filling) | $\lambda^*$ = value of one more unit of transmit power |
| Expected return $\mu^\top\mathbf{w} = r$ (portfolio) | $\lambda^*$ = variance cost per unit extra expected return |

### Convexity Verification Checklist

| Check | Method | Outcome |
|-------|--------|---------|
| $f: \mathbb{R} \to \mathbb{R}$ | $f''(x) \geq 0$ everywhere | Convex iff true |
| $f: \mathbb{R}^n \to \mathbb{R}$, $C^2$ | $\nabla^2 f(\mathbf{x}) \succeq 0$ everywhere | Convex iff true |
| $f$ is a sum | Each term convex? | Sum is convex |
| $f = g \circ A$ (affine precompose) | $g$ convex? | $f$ convex |
| $f(\mathbf{x}) = \max_i g_i(\mathbf{x})$ | Each $g_i$ convex? | $f$ convex |
| $f(\mathbf{x}) = \log \sum_i e^{g_i(\mathbf{x})}$ | Each $g_i$ affine? | $f$ convex (log-sum-exp) |
| $f(\mathbf{x}) = \|A\mathbf{x}-\mathbf{b}\|^2$ | Always | Convex (quadratic, $H = 2A^\top A \succeq 0$) |

