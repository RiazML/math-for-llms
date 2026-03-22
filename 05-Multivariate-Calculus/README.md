[← Previous Chapter: Calculus Fundamentals](../04-Calculus-Fundamentals/README.md) | [Next Chapter: Probability Theory →](../06-Probability-Theory/README.md)

---

# Chapter 5 — Multivariate Calculus

> _"In machine learning, everything is a function of millions of parameters. Multivariate calculus is the language that tells us how each one matters."_

## Overview

This chapter extends single-variable calculus to functions of many variables — the setting where all practical machine learning lives. A neural network loss $\mathcal{L}(\theta)$ depends on billions of parameters; understanding how it changes with each one, how those changes compose through layers, and where the minimum lies requires the full machinery of multivariate calculus.

The progression moves from partial derivatives and gradients (the basic instruments of measurement in high dimensions), through Jacobians and Hessians (the matrix generalizations that capture interactions between variables), through the multivariate chain rule and backpropagation (how derivatives flow through computation graphs), through optimality conditions (the theory of when you've found a minimum), and finally to automatic differentiation (how modern frameworks compute exact derivatives of arbitrary code).

Every concept here has direct ML instantiation: the gradient is the update direction for every optimizer; the Jacobian is backpropagation through a layer; the Hessian curvature determines optimizer convergence; KKT conditions underlie support vector machines and constrained training; automatic differentiation is PyTorch/JAX.

---

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
|---|-----------|----------------|-----------------|
| 01 | [Partial Derivatives and Gradients](01-Partial-Derivatives-and-Gradients/notes.md) | Extending differentiation to many-variable functions; the gradient as the direction of steepest ascent | Partial derivatives, gradient vector, directional derivatives, gradient fields, level curves, steepest ascent/descent, gradient in ML (loss landscapes) |
| 02 | [Jacobians and Hessians](02-Jacobians-and-Hessians/notes.md) | Matrix generalizations of the derivative for vector-valued functions and second-order curvature | Jacobian matrix, Hessian matrix, symmetry (Clairaut's theorem), positive/negative definiteness, curvature of loss landscapes, Newton's method curvature |
| 03 | [Chain Rule and Backpropagation](03-Chain-Rule-and-Backpropagation/notes.md) | How derivatives compose through multivariable function chains; the mathematical engine behind neural network training | Multivariate chain rule, computation graphs, forward pass, backward pass, backpropagation algorithm, gradient flow through layers, vanishing/exploding gradients |
| 04 | [Optimality Conditions](04-Optimality-Conditions/notes.md) | First- and second-order conditions for local/global minima; constrained optimization via Lagrange multipliers and KKT theory | Critical points, first-order necessary conditions, second-order sufficient conditions, saddle points, Lagrange multipliers, KKT conditions, convex vs non-convex, SVM dual |
| 05 | [Automatic Differentiation](05-Automatic-Differentiation/notes.md) | Systematic exact differentiation of arbitrary computer programs; the foundation of PyTorch, JAX, and TensorFlow | Forward mode AD, reverse mode AD, dual numbers, Wengert lists/tape, computational complexity, higher-order AD, implementation in modern frameworks |

---

## Reading Order and Dependencies

```
01-Partial-Derivatives-and-Gradients    (foundation: what a derivative means in ℝⁿ)
        ↓
02-Jacobians-and-Hessians               (matrix form: Jacobian = stacked gradients; Hessian = gradient of gradient)
        ↓
03-Chain-Rule-and-Backpropagation       (composition: how gradients flow through function chains)
        ↓
04-Optimality-Conditions                (theory: when does ∇f = 0 guarantee a minimum?)
        ↓
05-Automatic-Differentiation            (practice: how modern frameworks compute all of the above exactly)
        ↓
08-Optimization (Chapter 8)             (algorithms: gradient descent, Adam, Newton, constrained methods)
```

---

## What Belongs Where — Canonical Homes

This table is the authoritative scoping guide. **If a topic has a canonical home, every other section must give at most a 1–2 paragraph preview with a forward/backward reference — never a full treatment.**

| Topic | Canonical Home | Preview Only In |
|-------|---------------|-----------------|
| Partial derivatives (definition, computation) | §01 | — |
| Gradient vector $\nabla f$ | §01 | §02 (as row of Jacobian), §03 (as backprop output) |
| Directional derivative, gradient as steepest ascent | §01 | §04 (gradient condition for optimality) |
| Level curves and contour plots | §01 | §04 (visualizing optimality) |
| Gradient in ML: loss w.r.t. weights | §01 | §03 (as special case of backprop) |
| Jacobian matrix $J_f$ | §02 | §03 (Jacobian of layer = weight matrix) |
| Hessian matrix $H_f$ | §02 | §04 (second-order optimality conditions) |
| Clairaut's theorem (symmetry of mixed partials) | §02 | — |
| Positive definiteness of Hessian | §02 | §04 (sufficient condition for local min) |
| Newton's method curvature | §02 | §04 (Newton step derivation) |
| Multivariate chain rule | §03 | §01 (brief forward ref only) |
| Computation graphs | §03 | §05 (AD implementation uses same graphs) |
| Backpropagation algorithm | §03 | §05 (AD is the systematic implementation) |
| Vanishing/exploding gradients | §03 | — |
| Critical points ($\nabla f = 0$) | §04 | §01 (gradient = 0 as motivation) |
| First-order necessary conditions | §04 | — |
| Second-order sufficient conditions | §04 | §02 (Hessian role previewed) |
| Saddle points, local vs global minima | §04 | §03 (gradient flow around saddle points) |
| Lagrange multipliers | §04 | — |
| KKT conditions | §04 | — |
| Convex vs non-convex landscapes | §04 | §01 (preview only) |
| Forward mode automatic differentiation | §05 | — |
| Reverse mode automatic differentiation | §05 | §03 (conceptual link to backprop) |
| Dual numbers | §05 | — |
| Wengert list / tape | §05 | — |
| PyTorch autograd / JAX grad | §05 | §03 (brief forward reference) |
| Higher-order AD (Hessian-vector products) | §05 | §02 (Hessian computation preview) |

---

## Overlap Danger Zones

These are the most commonly duplicated topics across sections.

### 1. Gradient ↔ Jacobian
- **§01** defines the gradient $\nabla f : \mathbb{R}^n \to \mathbb{R}^n$ for scalar-valued $f$.
- **§02** defines the Jacobian $J_f \in \mathbb{R}^{m \times n}$ for vector-valued $f : \mathbb{R}^n \to \mathbb{R}^m$. The gradient is the special case $m=1$ (a row of the Jacobian). §02 should backward-reference §01 rather than re-derive the gradient.

### 2. Backpropagation ↔ Chain Rule ↔ Automatic Differentiation
- **§03** derives the multivariate chain rule and applies it to neural network layers, giving the backpropagation algorithm as a direct consequence.
- **§05** shows how automatic differentiation implements backpropagation systematically using computation graphs and a tape. §05 should backward-reference §03 for the mathematical derivation and focus on the algorithmic/implementation perspective.
- Do NOT re-derive the chain rule in §05, and do NOT describe the tape/dual-number mechanism in §03.

### 3. Hessian ↔ Optimality Conditions
- **§02** defines the Hessian, proves Clairaut's theorem, and connects Hessian definiteness to curvature.
- **§04** uses the Hessian as a tool for classifying critical points (positive definite → local minimum). §04 must not re-derive the Hessian — only apply it.

### 4. Optimality Conditions ↔ Optimization Algorithms
- **§04** covers the *theory*: necessary and sufficient conditions, Lagrange multipliers, KKT. It does NOT cover specific algorithms (gradient descent, Adam, Newton's method with step rules).
- **Chapter 8 (Optimization)** covers the *algorithms*. §04 may preview that gradient descent follows $-\nabla f$ but must not describe convergence rates, learning rate schedules, or adaptive methods.

### 5. Single-variable chain rule ↔ Multivariate chain rule
- **Chapter 4, §02** (Derivatives and Differentiation) derives the single-variable chain rule $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ and applies it to backpropagation through scalar computations.
- **§03 of this chapter** owns the full multivariate chain rule $\frac{\partial z}{\partial x_i} = \sum_j \frac{\partial z}{\partial y_j}\frac{\partial y_j}{\partial x_i}$ and the general Jacobian chain rule $J_{f \circ g} = J_f \cdot J_g$.

---

## Forward and Backward Reference Format

**Forward reference** (full treatment is later):
```markdown
> **Preview: Backpropagation**
> The multivariate chain rule, applied to computation graphs, yields backpropagation:
> derivatives propagate backward layer by layer via $\delta^{(l)} = (W^{(l+1)})^\top \delta^{(l+1)} \odot \sigma'(z^{(l)})$.
>
> → _Full treatment: [Chain Rule and Backpropagation](../03-Chain-Rule-and-Backpropagation/notes.md)_
```

**Backward reference** (builds on earlier section):
```markdown
> **Recall:** The gradient $\nabla f(\mathbf{x}) \in \mathbb{R}^n$ was defined in
> [Partial Derivatives and Gradients](../01-Partial-Derivatives-and-Gradients/notes.md)
> as the vector of partial derivatives. The Hessian stacks the gradients of each
> $\partial f / \partial x_i$ to form a matrix.
```

---

## Key Cross-Chapter Dependencies

**From Chapter 4 — Calculus Fundamentals:**
- Single-variable chain rule (§02) → multivariate chain rule in §03
- Single-variable critical points (§02) → optimality conditions in §04
- Taylor series (§04) → multivariate Taylor expansion (Hessian appears as the second-order term) in §02 and §04

**From Chapter 3 — Advanced Linear Algebra:**
- Matrix–vector products → Jacobian–vector products in §02 and §03
- Positive definite matrices ([§07](../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md)) → Hessian definiteness in §02 and §04
- Eigenvalues ([§01](../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md)) → curvature analysis (eigenvalues of the Hessian) in §02

**Into Chapter 8 — Optimization:**
- §01 (gradient) → gradient descent direction
- §02 (Hessian) → Newton's method, Gauss-Newton, curvature preconditioning
- §03 (backpropagation) → efficient gradient computation for all optimizers
- §04 (KKT conditions) → constrained optimization algorithms, SVM training
- §05 (automatic differentiation) → all gradient-based learning in practice

**Into Chapter 6 — Probability Theory:**
- §01 (gradients) → score functions, Fisher information matrix
- §04 (Lagrange multipliers) → maximum entropy distributions

---

## ML Concept Map

| ML Concept | Multivariate Calculus Foundation | Section |
|-----------|----------------------------------|---------|
| Gradient descent parameter update $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$ | Gradient of scalar loss w.r.t. parameter vector | §01 |
| Backpropagation through a dense layer | Multivariate chain rule: $\delta^{(l)} = J^\top \delta^{(l+1)}$ | §03 |
| Jacobian of a layer's output w.r.t. input | Jacobian matrix of vector-valued function | §02 |
| Second-order optimizer (Newton, K-FAC) | Hessian matrix and its inverse | §02 |
| Saddle points in loss landscapes | Second-order optimality conditions | §04 |
| LoRA / low-rank adaptation | Jacobian rank structure of weight updates | §02 |
| Support vector machines (hard/soft margin) | Lagrange multipliers + KKT conditions | §04 |
| Batch normalization gradients | Chain rule through normalization layer | §03 |
| Mixed-precision training (gradient scaling) | Jacobian condition number, numerical stability | §02 |
| PyTorch `.backward()`, JAX `grad()` | Reverse-mode automatic differentiation | §05 |
| Gradient checkpointing | Memory–compute trade-off in reverse-mode AD | §05 |
| Hessian-vector products (HVP) | Forward-over-reverse AD | §05 |
| Fisher information matrix | Hessian of log-likelihood; natural gradient | §02, §04 |
| Attention score gradient | Chain rule through softmax + dot-product | §03 |

---

## Prerequisites

Before starting this chapter, ensure you are comfortable with:

- **Single-variable derivatives**: chain rule, product rule, implicit differentiation — [§02-Derivatives-and-Differentiation](../04-Calculus-Fundamentals/02-Derivatives-and-Differentiation/notes.md)
- **Taylor series**: first- and second-order approximations — [§04-Series-and-Sequences](../04-Calculus-Fundamentals/04-Series-and-Sequences/notes.md)
- **Matrix–vector multiplication**: $A\mathbf{x}$, transpose, dot products — [§02-Linear-Algebra-Basics](../02-Linear-Algebra-Basics/README.md)
- **Positive definite matrices**: definiteness, eigenvalue sign conditions — [§07-Positive-Definite-Matrices](../03-Advanced-Linear-Algebra/07-Positive-Definite-Matrices/notes.md)

---

[← Previous Chapter: Calculus Fundamentals](../04-Calculus-Fundamentals/README.md) | [Next Chapter: Probability Theory →](../06-Probability-Theory/README.md)
