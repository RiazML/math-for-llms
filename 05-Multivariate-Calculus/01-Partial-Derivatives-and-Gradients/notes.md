[← Back to Multivariate Calculus](../README.md) | [Next: Jacobians and Hessians →](../02-Jacobians-and-Hessians/notes.md)

---

# Partial Derivatives and Gradients

> _"The gradient is the generalization of the derivative to functions of several variables, and it points in the direction in which the function increases most rapidly. This single idea powers every learning algorithm in modern AI."_
> — inspired by Cauchy, 1847

## Overview

Almost every function that matters in machine learning maps a high-dimensional parameter vector to a single scalar: the loss. A neural network with a billion parameters defines a surface in a billion-plus-one-dimensional space, and training means navigating that surface downhill. To navigate, you need to know the slope — not just along one axis, but simultaneously in every direction. That is precisely what the gradient provides.

This section builds the rigorous foundation for differentiation in $\mathbb{R}^n$. We begin with partial derivatives — the idea of measuring rate of change while holding all other variables fixed — then assemble them into the gradient vector, the single most important object in optimization. We prove that the gradient points in the direction of steepest ascent, show that it is perpendicular to the level sets of the function, and derive its formula as the linear part of a first-order approximation. We then compute gradients for every standard ML expression (MSE loss, cross-entropy, quadratic forms, regularized objectives) and study how to verify gradient implementations numerically.

The geometric and algebraic perspective developed here is the prerequisite for everything that follows: Jacobians and Hessians (§02) generalize the gradient to vector-valued functions and second-order information; the chain rule (§03) tells us how gradients propagate through composed functions; optimality conditions (§04) use gradient conditions to characterize minima; automatic differentiation (§05) computes gradients of arbitrary programs exactly.

## Prerequisites

- **Single-variable derivatives** — definition as a limit, differentiation rules, chain rule: [§04/02-Derivatives-and-Differentiation](../../04-Calculus-Fundamentals/02-Derivatives-and-Differentiation/notes.md)
- **First-order Taylor approximation** — $f(x+h) \approx f(x) + f'(x)h$: [§04/04-Series-and-Sequences](../../04-Calculus-Fundamentals/04-Series-and-Sequences/notes.md)
- **Vector notation** — $\mathbf{x} \in \mathbb{R}^n$, dot product, norms: [§02-Linear-Algebra-Basics](../../02-Linear-Algebra-Basics/README.md)
- **Matrix–vector multiplication** — $A\mathbf{x}$, transpose: [§02/02-Matrix-Operations](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive: partial derivative surfaces, gradient fields, directional derivatives, loss landscape geometry, gradient checking |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from partial derivative computation to natural gradient and softmax derivations |

## Learning Objectives

After completing this section, you will be able to:

1. **Define** the partial derivative as a limit and compute it for polynomial, exponential, logarithmic, and composite functions
2. **Explain** why a function can have all partial derivatives at a point without being differentiable there
3. **Define** the gradient $\nabla_{\mathbf{x}} f \in \mathbb{R}^n$ and identify it as a column vector
4. **Prove** that the gradient points in the direction of steepest ascent, using the directional derivative formula
5. **Prove** that the gradient is perpendicular to every level set of the function
6. **Compute** gradients of standard ML expressions: $\mathbf{a}^\top \mathbf{x}$, $\mathbf{x}^\top A \mathbf{x}$, $\lVert A\mathbf{x} - \mathbf{b} \rVert^2$, softmax cross-entropy
7. **Derive** the directional derivative formula $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}$ and use it to find the directions of maximum and minimum change
8. **Implement** numerical gradient checking using centered finite differences and measure relative error
9. **Explain** gradient clipping, gradient accumulation, and gradient norm as training diagnostics
10. **Preview** how the gradient generalizes to Jacobians (§02), flows through computation graphs (§03), and is computed by automatic differentiation (§05)

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 From Slopes to Surfaces](#11-from-slopes-to-surfaces)
  - [1.2 Historical Context](#12-historical-context)
  - [1.3 Why the Gradient Is the Heart of ML](#13-why-the-gradient-is-the-heart-of-ml)
- [2. Functions of Multiple Variables](#2-functions-of-multiple-variables)
  - [2.1 Scalar Fields](#21-scalar-fields)
  - [2.2 Level Sets and Contour Geometry](#22-level-sets-and-contour-geometry)
  - [2.3 Limits and Continuity in $\mathbb{R}^n$](#23-limits-and-continuity-in-rn)
  - [2.4 Differentiability vs. Partial Differentiability](#24-differentiability-vs-partial-differentiability)
- [3. Partial Derivatives](#3-partial-derivatives)
  - [3.1 Definition as a Limit](#31-definition-as-a-limit)
  - [3.2 Computation: Treating Other Variables as Constants](#32-computation-treating-other-variables-as-constants)
  - [3.3 Higher-Order Partial Derivatives](#33-higher-order-partial-derivatives)
  - [3.4 Equality of Mixed Partials — Clairaut Preview](#34-equality-of-mixed-partials--clairaut-preview)
  - [3.5 The Total Derivative](#35-the-total-derivative)
- [4. The Gradient Vector](#4-the-gradient-vector)
  - [4.1 Definition and Notation](#41-definition-and-notation)
  - [4.2 The Gradient Points in the Direction of Steepest Ascent](#42-the-gradient-points-in-the-direction-of-steepest-ascent)
  - [4.3 Gradient Perpendicular to Level Sets](#43-gradient-perpendicular-to-level-sets)
  - [4.4 Gradient as Linear Approximation](#44-gradient-as-linear-approximation)
  - [4.5 Gradient of Standard Forms](#45-gradient-of-standard-forms)
- [5. Directional Derivatives](#5-directional-derivatives)
  - [5.1 Definition and Limit Form](#51-definition-and-limit-form)
  - [5.2 The Gradient Formula](#52-the-gradient-formula)
  - [5.3 Extremizing the Directional Derivative](#53-extremizing-the-directional-derivative)
  - [5.4 Subgradients — Preview](#54-subgradients--preview)
- [6. Gradient in Machine Learning](#6-gradient-in-machine-learning)
  - [6.1 Mean Squared Error Gradient](#61-mean-squared-error-gradient)
  - [6.2 Cross-Entropy and Logistic Regression Gradient](#62-cross-entropy-and-logistic-regression-gradient)
  - [6.3 Gradient of Regularized Objectives](#63-gradient-of-regularized-objectives)
  - [6.4 Weight Sharing and Gradient Accumulation](#64-weight-sharing-and-gradient-accumulation)
  - [6.5 Gradient Descent: The Algorithm](#65-gradient-descent-the-algorithm)
- [7. Numerical Differentiation](#7-numerical-differentiation)
  - [7.1 Finite Difference Formulas](#71-finite-difference-formulas)
  - [7.2 Optimal Step Size](#72-optimal-step-size)
  - [7.3 Gradient Checking](#73-gradient-checking)
  - [7.4 Limitations and the Case for Automatic Differentiation](#74-limitations-and-the-case-for-automatic-differentiation)
- [8. Loss Landscape Geometry](#8-loss-landscape-geometry)
  - [8.1 Loss Landscapes for Neural Networks](#81-loss-landscapes-for-neural-networks)
  - [8.2 Gradient Norm as a Training Diagnostic](#82-gradient-norm-as-a-training-diagnostic)
  - [8.3 Gradient Clipping](#83-gradient-clipping)
  - [8.4 Gradient Accumulation for Large Batch Training](#84-gradient-accumulation-for-large-batch-training)
- [9. Advanced Topics](#9-advanced-topics)
  - [9.1 Gateaux and Fréchet Derivatives](#91-gateaux-and-fréchet-derivatives)
  - [9.2 Wirtinger Derivatives](#92-wirtinger-derivatives)
  - [9.3 Gradients on Manifolds and the Natural Gradient](#93-gradients-on-manifolds-and-the-natural-gradient)
  - [9.4 Jacobian Preview](#94-jacobian-preview)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 From Slopes to Surfaces

In single-variable calculus, the derivative $f'(x)$ measures how steeply $f$ rises or falls as $x$ changes. The picture is one-dimensional: you stand on a hill, face a direction, and the derivative tells you the slope. Multivariate calculus asks: what if you can move in any direction? The surface $z = f(x, y)$ is a two-dimensional landscape — a mountain range — and from any point on it you can move north, south, east, west, or any diagonal. Different directions give different slopes.

The key insight is that knowing the slope in every possible direction would be infinitely much information, but it turns out you only need to store $n$ numbers — one slope per coordinate axis. These are the **partial derivatives**. And assembling them into a single vector — the **gradient** — encodes all directional information: given $\nabla f$, you can compute the slope in any direction via a dot product.

This reduction from "infinitely many slopes" to "one $n$-dimensional vector" is the central miracle of multivariable calculus. It is why gradient descent is computationally tractable for models with a billion parameters: at each point, we compute one gradient vector $\nabla_{\boldsymbol{\theta}} \mathcal{L} \in \mathbb{R}^n$, and we move in the direction $-\nabla_{\boldsymbol{\theta}} \mathcal{L}$.

```
GRADIENT: FROM ONE SLOPE TO A DIRECTION VECTOR
════════════════════════════════════════════════════════════════════════

  Single variable:           Multiple variables:
  f: ℝ → ℝ                   f: ℝⁿ → ℝ

  f'(x) ∈ ℝ                  ∇f(x) ∈ ℝⁿ
  (one number)                (a vector of n partial derivatives)

  Slope along x-axis          Slope along x₁-axis: ∂f/∂x₁
                              Slope along x₂-axis: ∂f/∂x₂
                               ⋮
                              Slope along xₙ-axis: ∂f/∂xₙ

  "Steepest direction"        "Steepest direction" = direction of ∇f
  is trivially ±x             requires computing ∇f

════════════════════════════════════════════════════════════════════════
```

**For AI:** A language model with $n = 7 \times 10^9$ parameters defines a loss function $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$. The gradient $\nabla_{\boldsymbol{\theta}} \mathcal{L} \in \mathbb{R}^n$ tells every parameter how much to change and in which direction. One gradient computation, via backpropagation, produces all $n$ values simultaneously — this is why reverse-mode automatic differentiation (§05) is so powerful.

### 1.2 Historical Context

The theory of partial derivatives developed over two centuries, driven largely by physics and the need to describe phenomena depending on space and time simultaneously.

| Year | Mathematician | Contribution |
|------|--------------|-------------|
| 1734 | Euler | Introduced partial derivative notation $\partial f / \partial x$ for fluid mechanics |
| 1770 | Lagrange | Developed calculus of variations; implicit use of gradients in mechanics |
| 1816 | Cauchy | Rigorous foundation of multivariate analysis; ε-δ for limits in $\mathbb{R}^n$ |
| 1847 | Cauchy | Published the **method of gradient descent** for minimizing functions |
| 1873 | Maxwell | Gradient, divergence, curl notation for electromagnetic field equations |
| 1884 | Gibbs / Heaviside | Modern vector calculus notation; $\nabla$ (nabla) operator |
| 1944 | Hestenes, Stiefel | Conjugate gradient method — gradient descent with memory |
| 1986 | Rumelhart, Hinton, Williams | Backpropagation as systematic gradient computation in neural networks |
| 2012 | Krizhevsky et al. | AlexNet — gradient descent on GPU scales to ImageNet |
| 2017 | Vaswani et al. | Attention mechanisms trained by gradient descent (Transformers) |
| 2022–2026 | LLM era | Gradient descent with AdamW, gradient clipping, and mixed precision trains 10¹² parameter models |

The notation $\nabla$ (nabla) was introduced by the Scottish mathematician Peter Guthrie Tait around 1884, borrowing the shape of the ancient Assyrian harp. Cauchy's 1847 gradient descent paper is remarkably modern — he proves convergence for smooth functions and discusses step-size selection. The 140-year gap between Cauchy's paper and backpropagation's popularization shows how long theoretical tools can wait for the right application.

### 1.3 Why the Gradient Is the Heart of ML

Every major operation in modern AI either computes or responds to a gradient:

- **Training**: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$ — gradient descent or its adaptive variants (Adam, AdamW, Lion)
- **Backpropagation**: the chain rule for computing $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ by propagating gradients backward through a computation graph
- **Attention gradient**: $\nabla_Q \mathcal{L}$ drives how query projections are updated to attend to more useful keys
- **LoRA (Hu et al., 2022)**: restricts gradient updates to a low-rank subspace $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ — the gradient of $\mathcal{L}$ w.r.t. $A$ and $B$ is what gets computed
- **Gradient clipping**: prevents exploding gradients in transformer training by rescaling $\nabla \mathcal{L}$ when $\lVert \nabla \mathcal{L} \rVert > G$
- **Mechanistic interpretability**: gradient $\frac{\partial \text{output}}{\partial \text{input}}$ is a saliency map — it tells us which input tokens matter most

Understanding the gradient rigorously — not just as a formula but as a geometric object with provable properties — is the difference between knowing how to run gradient descent and understanding why it works, when it fails, and how to fix it.

---

## 2. Functions of Multiple Variables

### 2.1 Scalar Fields: $f: \mathbb{R}^n \to \mathbb{R}$

**Definition.** A **scalar field** (or scalar-valued function of $n$ variables) is a function

$$f: \mathcal{D} \subseteq \mathbb{R}^n \to \mathbb{R}$$

that assigns to each point $\mathbf{x} = (x_1, x_2, \ldots, x_n)^\top \in \mathcal{D}$ a real number $f(\mathbf{x})$.

The domain $\mathcal{D}$ is a subset of $\mathbb{R}^n$; we usually take $\mathcal{D} = \mathbb{R}^n$ (for polynomials) or the natural domain (where the formula is defined).

**Standard examples:**

| Function | Formula | Domain | Geometric meaning |
|---|---|---|---|
| Paraboloid | $f(x,y) = x^2 + y^2$ | $\mathbb{R}^2$ | Bowl shape; unique minimum at origin |
| Saddle | $f(x,y) = x^2 - y^2$ | $\mathbb{R}^2$ | Rises in $x$, falls in $y$; saddle point at origin |
| Gaussian | $f(\mathbf{x}) = e^{-\lVert \mathbf{x} \rVert^2 / (2\sigma^2)}$ | $\mathbb{R}^n$ | Bell shaped; probability density (unnormalized) |
| MSE loss | $f(\mathbf{w}) = \frac{1}{m}\lVert X\mathbf{w} - \mathbf{y} \rVert^2$ | $\mathbb{R}^d$ | Convex paraboloid in weight space |
| Cross-entropy | $f(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^m \log p_{\mathbf{w}}(y^{(i)} \mid \mathbf{x}^{(i)})$ | $\mathbb{R}^d$ | Convex for logistic regression; non-convex for neural networks |

**Non-examples** (not scalar fields):
- $\mathbf{f}(\mathbf{x}) = A\mathbf{x} \in \mathbb{R}^m$ — this is vector-valued; its derivative is the Jacobian (§02)
- $F(\mathbf{x}) = \int_0^1 f(\mathbf{x}, t)\,dt$ — this is a functional, not a finite-dimensional function

**For AI:** Every loss function in supervised learning, reinforcement learning, and self-supervised learning is a scalar field over parameter space. Training is navigation of this surface.

### 2.2 Level Sets and Contour Geometry

**Definition.** The **level set** (or level surface) of $f: \mathbb{R}^n \to \mathbb{R}$ at value $c$ is

$$L_c(f) = \{\mathbf{x} \in \mathbb{R}^n : f(\mathbf{x}) = c\}$$

For $n = 2$, level sets are **contour lines** (curves in the plane). For $n = 3$, they are **level surfaces**. For general $n$, they are $(n-1)$-dimensional hypersurfaces.

**Geometric meaning:** Moving along a level set keeps the function value constant. You expend no potential energy. The gradient, as we will prove in §4.3, is always perpendicular to every level set through the point.

**Examples:**

- $f(x,y) = x^2 + y^2$: level sets are concentric circles $x^2 + y^2 = c$ (for $c > 0$)
- $f(x,y) = x^2 - y^2$: level sets are hyperbolas
- $f(\mathbf{w}) = \lVert X\mathbf{w} - \mathbf{y} \rVert^2$: level sets are ellipsoids in $\mathbb{R}^d$ (elongated if $X$ is ill-conditioned)

```
CONTOUR GEOMETRY FOR f(x,y) = x² + y²
════════════════════════════════════════════════════════════════════════

         y
         │
     3 ──┼──   ╭───────────╮   c = 9
         │    ╭┼───────────┼╮
     2 ──┼──  │╭─────────╮│   c = 4
         │    ││╭───────╮││
     1 ──┼──  │││   ●   │││  ← minimum (0,0)  c = 1
         │    │││       │││
     0 ──┼────┼┼┼───────┼┼┼── x
         │    │││       │││
    -1 ──┼──  │││       │││
         │    ││╰───────╯││
    -2 ──┼──  │╰─────────╯│
         │    ╰┼───────────┼╯
    -3 ──┼──   ╰───────────╯

  • Each ring = constant f value   • Minimum at center
  • ∇f points radially outward    • ∇f ⊥ each ring (proved in §4.3)

════════════════════════════════════════════════════════════════════════
```

**For AI:** The loss landscape's level sets determine how gradient descent navigates. If level sets are elongated ellipses (high condition number), standard gradient descent zigzags slowly — motivating preconditioning and adaptive methods like Adam, which effectively rescale the gradient to account for curvature.

### 2.3 Limits and Continuity in $\mathbb{R}^n$

**Definition.** We say $\lim_{\mathbf{x} \to \mathbf{a}} f(\mathbf{x}) = L$ if for every $\varepsilon > 0$ there exists $\delta > 0$ such that

$$0 < \lVert \mathbf{x} - \mathbf{a} \rVert < \delta \implies |f(\mathbf{x}) - L| < \varepsilon$$

This definition is formally identical to the single-variable case, but with Euclidean distance $\lVert \mathbf{x} - \mathbf{a} \rVert$ replacing $|x - a|$.

**Critical difference from single-variable calculus:** In $\mathbb{R}$, you can approach a point from only two directions (left or right). In $\mathbb{R}^n$, you can approach along infinitely many paths. A limit exists only if the same value $L$ is obtained along every path.

**Non-example (path-dependent limit):** Consider $f(x,y) = \frac{xy}{x^2 + y^2}$ as $(x,y) \to (0,0)$:
- Along $y = 0$: $f(x,0) = 0 \to 0$
- Along $y = x$: $f(x,x) = \frac{x^2}{2x^2} = \frac{1}{2} \to \frac{1}{2}$

Different paths give different limits, so $\lim_{(x,y)\to(0,0)} f(x,y)$ does not exist.

**Continuity:** $f$ is **continuous at $\mathbf{a}$** if $\lim_{\mathbf{x}\to\mathbf{a}} f(\mathbf{x}) = f(\mathbf{a})$. Polynomials, rational functions (away from zeros of denominator), $e^{f(\mathbf{x})}$, $\log f(\mathbf{x})$ (where $f > 0$) are all continuous on their domains.

All loss functions used in practice (MSE, cross-entropy, hinge, Huber) are continuous on their natural domains. This ensures the gradient descent trajectory is well-defined and doesn't jump discontinuously.

### 2.4 Differentiability vs. Partial Differentiability

This is a subtle but important distinction that illuminates why the gradient is defined the way it is.

**Partial differentiability does NOT imply differentiability.** The partial derivatives can exist at a point while the function fails to be differentiable (or even continuous) there.

**Non-example (Cauchy 1821):** Define

$$f(x,y) = \begin{cases} \frac{xy}{x^2 + y^2} & (x,y) \neq (0,0) \\ 0 & (x,y) = (0,0) \end{cases}$$

- $\frac{\partial f}{\partial x}(0,0) = \lim_{h \to 0} \frac{f(h,0) - f(0,0)}{h} = \lim_{h \to 0} \frac{0}{h} = 0$ ✓
- $\frac{\partial f}{\partial y}(0,0) = 0$ ✓ (by symmetry)
- But $f$ is **not continuous** at $(0,0)$ (as shown in §2.3, the limit doesn't exist)

**True differentiability** requires the existence of a linear map $L: \mathbb{R}^n \to \mathbb{R}$ such that

$$\lim_{\mathbf{h} \to \mathbf{0}} \frac{f(\mathbf{a} + \mathbf{h}) - f(\mathbf{a}) - L(\mathbf{h})}{\lVert \mathbf{h} \rVert} = 0$$

When $f$ is differentiable at $\mathbf{a}$, the linear map $L$ is given by $L(\mathbf{h}) = \nabla f(\mathbf{a})^\top \mathbf{h}$. The gradient is then the representation of this linear map in the standard basis.

**Sufficient condition:** If all partial derivatives exist and are **continuous** on a neighborhood of $\mathbf{a}$, then $f$ is differentiable at $\mathbf{a}$. All standard ML functions satisfy this condition away from a few exceptional points (like $x = 0$ for ReLU).

---

## 3. Partial Derivatives

### 3.1 Definition as a Limit

**Definition.** The **partial derivative** of $f: \mathbb{R}^n \to \mathbb{R}$ with respect to $x_i$ at the point $\mathbf{a}$ is

$$\frac{\partial f}{\partial x_i}(\mathbf{a}) = \lim_{h \to 0} \frac{f(a_1, \ldots, a_i + h, \ldots, a_n) - f(a_1, \ldots, a_i, \ldots, a_n)}{h}$$

provided this limit exists. We perturb only the $i$-th coordinate by $h$ and hold all others fixed.

**Notation variants** (all in common use):

| Notation | Context |
|---|---|
| $\frac{\partial f}{\partial x_i}$ | Standard — preferred in this repo |
| $\partial_{x_i} f$ or $f_{x_i}$ | Compact, used in PDEs |
| $D_i f$ | Operator notation |
| $f_i$ | When clear from context |

**Geometric interpretation:** $\frac{\partial f}{\partial x_i}(\mathbf{a})$ is the slope of the curve obtained by intersecting the surface $z = f(\mathbf{x})$ with the hyperplane $\{x_j = a_j : j \neq i\}$. It measures the instantaneous rate of change of $f$ when we move parallel to the $x_i$-axis.

```
GEOMETRIC MEANING OF A PARTIAL DERIVATIVE
════════════════════════════════════════════════════════════════════════

  Surface z = f(x, y)                  Cross-section at fixed y = a₂

       z                                      z
       │    ╱╲                                │
       │   ╱  ╲   surface                     │    ╱
       │  ╱    ╲                              │   ╱  slope = ∂f/∂x at (a₁,a₂)
       │ ╱      ╲                             │  ╱
       │╱────────╲──── y                      │ ╱
       ╱ fixed y = a₂                         │╱────────────────── x
      ╱                                       a₁
     x
  
  The partial ∂f/∂x is the slope of the curve when we
  slice the surface with the plane y = a₂ (constant).

════════════════════════════════════════════════════════════════════════
```

**For AI:** PyTorch's `.grad` attribute on a scalar after `.backward()` is exactly $\frac{\partial \mathcal{L}}{\partial \theta_i}$ for each parameter $\theta_i$. When you call `loss.backward()`, it computes all $n$ partial derivatives simultaneously via reverse-mode automatic differentiation.

### 3.2 Computation: Treating Other Variables as Constants

The practical rule: **hold all variables except $x_i$ constant, then differentiate as a single-variable function of $x_i$.**

All standard differentiation rules (power rule, product rule, chain rule) apply, treating the fixed variables as parameters.

**Example 1:** $f(x, y) = 3x^2y + e^{xy} - y^3$

$$\frac{\partial f}{\partial x} = 6xy + ye^{xy} \qquad \text{(treat } y \text{ as constant: } \frac{d}{dx}[3x^2 \cdot y] = 6xy, \; \frac{d}{dx}[e^{xy}] = ye^{xy} \text{)}$$

$$\frac{\partial f}{\partial y} = 3x^2 + xe^{xy} - 3y^2 \qquad \text{(treat } x \text{ as constant)}$$

**Example 2:** $f(x, y, z) = \ln(x^2 + y^2) + z\sin(x)$

$$\frac{\partial f}{\partial x} = \frac{2x}{x^2 + y^2} + z\cos(x)$$

$$\frac{\partial f}{\partial y} = \frac{2y}{x^2 + y^2}$$

$$\frac{\partial f}{\partial z} = \sin(x)$$

**Example 3 (ML):** Loss $\mathcal{L}(w_1, w_2) = (w_1 x_1 + w_2 x_2 - y)^2$ for one data point $(x_1, x_2, y)$:

$$\frac{\partial \mathcal{L}}{\partial w_1} = 2(w_1 x_1 + w_2 x_2 - y) \cdot x_1$$

$$\frac{\partial \mathcal{L}}{\partial w_2} = 2(w_1 x_1 + w_2 x_2 - y) \cdot x_2$$

This illustrates the general pattern: $\frac{\partial}{\partial w_i}[\text{prediction error}]^2 = 2 \cdot \text{error} \cdot x_i$ — the gradient of a squared loss is error times feature.

### 3.3 Higher-Order Partial Derivatives

Since $\frac{\partial f}{\partial x_i}$ is itself a function, we can differentiate again:

**Second-order partials:**

$$\frac{\partial^2 f}{\partial x_i^2} = \frac{\partial}{\partial x_i}\!\left(\frac{\partial f}{\partial x_i}\right) \qquad \text{(differentiate } x_i \text{ twice)}$$

$$\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial}{\partial x_j}\!\left(\frac{\partial f}{\partial x_i}\right) \qquad \text{(differentiate } x_i \text{ first, then } x_j \text{)}$$

The collection of all second-order partials forms the **Hessian matrix** $H_f \in \mathbb{R}^{n \times n}$:

$$(H_f)_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

> **Scope note:** The Hessian is defined here, but all properties (symmetry, definiteness, Newton's method) are treated in [§02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/notes.md).

**Example:** $f(x,y) = x^3y^2 + \sin(xy)$

$$\frac{\partial f}{\partial x} = 3x^2y^2 + y\cos(xy), \qquad \frac{\partial^2 f}{\partial x^2} = 6xy^2 - y^2\sin(xy)$$

$$\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}[3x^2y^2 + y\cos(xy)] = 6x^2y + \cos(xy) - xy\sin(xy)$$

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}[2x^3y + x\cos(xy)] = 6x^2y + \cos(xy) - xy\sin(xy)$$

Note: $\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial^2 f}{\partial x \partial y}$ — this equality is Clairaut's theorem.

### 3.4 Equality of Mixed Partials — Clairaut Preview

**Theorem (Clairaut/Schwarz).** If $f: \mathbb{R}^n \to \mathbb{R}$ has continuous second partial derivatives on an open set, then for all $i \neq j$:

$$\frac{\partial^2 f}{\partial x_j \partial x_i} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

That is, the order of partial differentiation does not matter.

**Consequence for the Hessian:** The Hessian matrix $H_f$ is symmetric: $H_f = H_f^\top$.

**Why continuity is required:** Without continuity of the mixed partials, equality can fail. The standard counterexample is:

$$f(x,y) = \begin{cases} xy\,\frac{x^2 - y^2}{x^2 + y^2} & (x,y) \neq (0,0) \\ 0 & (x,y) = (0,0) \end{cases}$$

For this function, $\frac{\partial^2 f}{\partial x \partial y}(0,0) = 1 \neq -1 = \frac{\partial^2 f}{\partial y \partial x}(0,0)$.

> **Full proof and implications:** [§02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/notes.md) — Clairaut's theorem, Hessian symmetry, positive definiteness.

**For AI:** The symmetry $H_f = H_f^\top$ is why Hessian-vector products are computationally feasible — symmetric matrices have real eigenvalues, and efficient algorithms (Pearlmutter 1994) compute $H\mathbf{v}$ without forming $H$ explicitly.

### 3.5 The Total Derivative

When all variables change simultaneously by small amounts $\delta x_1, \ldots, \delta x_n$, the total change in $f$ is:

$$\delta f \approx \sum_{i=1}^n \frac{\partial f}{\partial x_i} \delta x_i = \nabla f(\mathbf{a})^\top \boldsymbol{\delta}$$

In differential notation:

$$df = \frac{\partial f}{\partial x_1} dx_1 + \frac{\partial f}{\partial x_2} dx_2 + \cdots + \frac{\partial f}{\partial x_n} dx_n$$

This is the **total differential** of $f$ at $\mathbf{a}$. It is exact (not an approximation) in the differential sense, and gives the first-order approximation

$$f(\mathbf{a} + \boldsymbol{\delta}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top \boldsymbol{\delta}$$

when $\lVert \boldsymbol{\delta} \rVert$ is small. This first-order Taylor approximation is the foundation of gradient descent: taking a step $\boldsymbol{\delta} = -\eta \nabla f(\mathbf{a})$ decreases $f$ by approximately $\eta \lVert \nabla f(\mathbf{a}) \rVert^2$ (to first order).

**Perturbation analysis in ML:** When we change a model's weight $w_i$ by a small amount $\delta w_i$, the loss changes by approximately $\frac{\partial \mathcal{L}}{\partial w_i} \cdot \delta w_i$. This is why the gradient tells us exactly which parameters, perturbed in which direction, reduce the loss most efficiently.

---

## 4. The Gradient Vector

### 4.1 Definition and Notation

**Definition.** Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable at $\mathbf{a}$. The **gradient** of $f$ at $\mathbf{a}$ is the column vector of all partial derivatives:

$$\nabla_{\mathbf{x}} f(\mathbf{a}) = \begin{pmatrix} \frac{\partial f}{\partial x_1}(\mathbf{a}) \\ \frac{\partial f}{\partial x_2}(\mathbf{a}) \\ \vdots \\ \frac{\partial f}{\partial x_n}(\mathbf{a}) \end{pmatrix} \in \mathbb{R}^n$$

**Notation conventions** (following `docs/NOTATION_GUIDE.md`):

| Notation | Meaning |
|---|---|
| $\nabla f(\mathbf{a})$ | Gradient at $\mathbf{a}$; subscript on $\nabla$ omitted when variable is clear |
| $\nabla_{\mathbf{x}} f$ | Gradient with respect to $\mathbf{x}$ (explicit subscript) |
| $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ | Gradient of loss $\mathcal{L}$ with respect to parameters $\boldsymbol{\theta}$ |
| $\mathbf{g}$ | Common shorthand for the gradient in optimization |

**Orientation convention:** Following Nocedal & Wright (2006) and all major ML frameworks, the gradient $\nabla f(\mathbf{a}) \in \mathbb{R}^n$ is a **column vector**. This means the first-order approximation is $f(\mathbf{a} + \boldsymbol{\delta}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top \boldsymbol{\delta}$.

**Non-notation note:** Do not write $\nabla f$ as a row vector. Some texts (especially those following Jacobian conventions) write the gradient as a row, but this creates ambiguity with the Jacobian $J_f \in \mathbb{R}^{1 \times n}$. We reserve the Jacobian convention for §02.

**Examples:**

$f(x,y) = x^2 + 3xy - y^2$:
$$\nabla f = \begin{pmatrix} 2x + 3y \\ 3x - 2y \end{pmatrix}$$

$f(\mathbf{x}) = \lVert \mathbf{x} \rVert^2 = \sum_i x_i^2$:
$$\nabla f = \begin{pmatrix} 2x_1 \\ 2x_2 \\ \vdots \\ 2x_n \end{pmatrix} = 2\mathbf{x}$$

$f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x} = \sum_i a_i x_i$:
$$\nabla f = \begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix} = \mathbf{a}$$

### 4.2 The Gradient Points in the Direction of Steepest Ascent

**Theorem.** Among all unit vectors $\mathbf{u}$ with $\lVert \mathbf{u} \rVert = 1$, the directional derivative $D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a})^\top \mathbf{u}$ is maximized when $\mathbf{u} = \frac{\nabla f(\mathbf{a})}{\lVert \nabla f(\mathbf{a}) \rVert}$, giving maximum value $\lVert \nabla f(\mathbf{a}) \rVert$.

**Proof.** By the Cauchy-Schwarz inequality:

$$D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a})^\top \mathbf{u} \leq \lVert \nabla f(\mathbf{a}) \rVert \cdot \lVert \mathbf{u} \rVert = \lVert \nabla f(\mathbf{a}) \rVert$$

with equality when $\mathbf{u} \parallel \nabla f(\mathbf{a})$, i.e., $\mathbf{u} = \frac{\nabla f(\mathbf{a})}{\lVert \nabla f(\mathbf{a}) \rVert}$.

Similarly, the minimum directional derivative is $-\lVert \nabla f(\mathbf{a}) \rVert$, achieved in direction $-\frac{\nabla f(\mathbf{a})}{\lVert \nabla f(\mathbf{a}) \rVert}$.

**Corollary (gradient descent).** To decrease $f$ most rapidly, step in direction $-\nabla f(\mathbf{a})$. This is why gradient descent uses $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}$ — it moves in the direction of steepest descent.

```
DIRECTIONAL DERIVATIVE vs. DIRECTION
════════════════════════════════════════════════════════════════════════

  Given ∇f at a point, the directional derivative in direction u:

     D_u f = ∇f · u = ||∇f|| cos(θ)
                               where θ = angle between ∇f and u

  Maximum (θ=0°):    u = ∇f/||∇f||    D_u f = ||∇f||    ← steepest ascent
  Zero      (θ=90°): u ⊥ ∇f           D_u f = 0         ← along level set
  Minimum (θ=180°):  u = -∇f/||∇f||   D_u f = -||∇f||   ← steepest descent

  Gradient descent chooses u = -∇f/||∇f|| at every step.

════════════════════════════════════════════════════════════════════════
```

**For AI:** The gradient $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ is the unique direction in which perturbing $\boldsymbol{\theta}$ increases the loss most rapidly per unit norm. Gradient descent follows its negative. Adam modifies this direction by dividing by the running second moment $\sqrt{v_t} + \varepsilon$, effectively normalizing each coordinate — this is why Adam often converges faster when gradients have very different scales across parameters.

### 4.3 Gradient Perpendicular to Level Sets

**Theorem.** Let $f: \mathbb{R}^n \to \mathbb{R}$ be differentiable, and let $\mathbf{a}$ lie on the level set $L_c(f) = \{f(\mathbf{x}) = c\}$. Then $\nabla f(\mathbf{a})$ is perpendicular to $L_c(f)$ at $\mathbf{a}$ — that is, $\nabla f(\mathbf{a})$ is orthogonal to every tangent vector of $L_c(f)$ at $\mathbf{a}$.

**Proof.** Let $\boldsymbol{\gamma}: (-\varepsilon, \varepsilon) \to \mathbb{R}^n$ be any smooth curve lying in $L_c(f)$ with $\boldsymbol{\gamma}(0) = \mathbf{a}$. Then $f(\boldsymbol{\gamma}(t)) = c$ for all $t$. Differentiating:

$$\frac{d}{dt} f(\boldsymbol{\gamma}(t)) \Big|_{t=0} = 0$$

By the single-variable chain rule applied along the curve:

$$\frac{d}{dt} f(\boldsymbol{\gamma}(t)) = \nabla f(\boldsymbol{\gamma}(t))^\top \boldsymbol{\gamma}'(t)$$

At $t = 0$: $\nabla f(\mathbf{a})^\top \boldsymbol{\gamma}'(0) = 0$.

Since $\boldsymbol{\gamma}'(0)$ is any tangent vector of $L_c(f)$ at $\mathbf{a}$, we conclude $\nabla f(\mathbf{a}) \perp T_{\mathbf{a}} L_c(f)$. $\square$

**Geometric consequence:** The gradient always points directly "away" from the level set — it is the normal vector to the surface $f = c$. To travel along the level set (constant loss), you must move perpendicular to the gradient.

**For AI:** In continual learning and orthogonal gradient descent (OGD), gradients are projected to be orthogonal to previous task gradients. This ensures learning a new task doesn't increase loss on previous tasks — it literally moves along the level set of the old loss while decreasing the new one.

### 4.4 Gradient as Linear Approximation

The gradient gives the best linear (affine) approximation to $f$ near a point:

$$f(\mathbf{a} + \boldsymbol{\delta}) = f(\mathbf{a}) + \nabla f(\mathbf{a})^\top \boldsymbol{\delta} + O(\lVert \boldsymbol{\delta} \rVert^2)$$

This is the first-order multivariate Taylor expansion. The remainder term $O(\lVert \boldsymbol{\delta} \rVert^2)$ is controlled by the Hessian (§02).

**Interpretation:** The hyperplane $z = f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a})$ is the **tangent hyperplane** to the surface $z = f(\mathbf{x})$ at the point $(\mathbf{a}, f(\mathbf{a}))$.

**Error bound:** If $f$ has bounded second derivatives — specifically if $\lVert H_f(\mathbf{x}) \rVert_2 \leq L$ for all $\mathbf{x}$ — then:

$$\left| f(\mathbf{a} + \boldsymbol{\delta}) - f(\mathbf{a}) - \nabla f(\mathbf{a})^\top \boldsymbol{\delta} \right| \leq \frac{L}{2} \lVert \boldsymbol{\delta} \rVert^2$$

**For AI:** The $L$-smoothness condition $\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert \leq L \lVert \mathbf{x} - \mathbf{y} \rVert$ (equivalent to bounded Hessian) is the key assumption in gradient descent convergence proofs. It guarantees that the linear approximation is not too far off, so taking a step $-\eta \nabla f$ with $\eta \leq 1/L$ always decreases $f$.

### 4.5 Gradient of Standard Forms

These are among the most frequently used gradient computations in ML. All can be derived from first principles via the definition.

**Theorem (matrix calculus identities).** Let $\mathbf{x} \in \mathbb{R}^n$, $\mathbf{a} \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times n}$.

| $f(\mathbf{x})$ | $\nabla_{\mathbf{x}} f$ | Proof sketch |
|---|---|---|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ | $f = \sum_i a_i x_i$, so $\partial f/\partial x_j = a_j$ |
| $\mathbf{x}^\top \mathbf{a}$ | $\mathbf{a}$ | Same as above (dot product is symmetric) |
| $\lVert \mathbf{x} \rVert^2 = \mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}$ | $f = \sum_i x_i^2$, so $\partial f/\partial x_j = 2x_j$ |
| $\mathbf{x}^\top A \mathbf{x}$ | $(A + A^\top)\mathbf{x}$ | See derivation below |
| $\mathbf{x}^\top A \mathbf{x}$ ($A$ symmetric) | $2A\mathbf{x}$ | Since $A = A^\top$ |
| $\lVert A\mathbf{x} - \mathbf{b} \rVert^2$ | $2A^\top(A\mathbf{x} - \mathbf{b})$ | Chain rule + above |
| $\log(\mathbf{a}^\top \mathbf{x})$ | $\mathbf{a} / (\mathbf{a}^\top \mathbf{x})$ | Chain rule: $\partial/\partial x_j = a_j / (\mathbf{a}^\top \mathbf{x})$ |

**Proof of $\nabla(\mathbf{x}^\top A \mathbf{x}) = (A + A^\top)\mathbf{x}$:**

$$f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x} = \sum_{i,j} A_{ij} x_i x_j$$

$$\frac{\partial f}{\partial x_k} = \sum_j A_{kj} x_j + \sum_i A_{ik} x_i = (A\mathbf{x})_k + (A^\top \mathbf{x})_k$$

Therefore $\nabla f = A\mathbf{x} + A^\top \mathbf{x} = (A + A^\top)\mathbf{x}$. When $A$ is symmetric ($A = A^\top$), this simplifies to $2A\mathbf{x}$. $\square$

**Proof of $\nabla \lVert A\mathbf{x} - \mathbf{b} \rVert^2 = 2A^\top(A\mathbf{x} - \mathbf{b})$:**

Let $\mathbf{r} = A\mathbf{x} - \mathbf{b}$. Then $f = \lVert \mathbf{r} \rVert^2 = \mathbf{r}^\top \mathbf{r}$. By the chain rule:

$$\nabla_{\mathbf{x}} f = 2A^\top \mathbf{r} = 2A^\top(A\mathbf{x} - \mathbf{b})$$

For the MSE loss $\mathcal{L}(\mathbf{w}) = \frac{1}{m}\lVert X\mathbf{w} - \mathbf{y} \rVert^2$, this gives:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{2}{m} X^\top (X\mathbf{w} - \mathbf{y})$$

This is the gradient of the ordinary least squares objective — the formula that underlies every linear regression solver.

---

## 5. Directional Derivatives

### 5.1 Definition and Limit Form

The partial derivative measures change along a coordinate axis. The **directional derivative** measures change in any arbitrary direction.

**Definition.** Let $f: \mathbb{R}^n \to \mathbb{R}$, $\mathbf{a} \in \mathbb{R}^n$, and $\mathbf{u} \in \mathbb{R}^n$ with $\lVert \mathbf{u} \rVert = 1$ (unit vector). The **directional derivative** of $f$ at $\mathbf{a}$ in direction $\mathbf{u}$ is:

$$D_{\mathbf{u}} f(\mathbf{a}) = \lim_{h \to 0} \frac{f(\mathbf{a} + h\mathbf{u}) - f(\mathbf{a})}{h}$$

**Note on unit vectors:** The convention $\lVert \mathbf{u} \rVert = 1$ is important. If we allowed $\lVert \mathbf{u} \rVert \neq 1$, the directional derivative would scale with $\lVert \mathbf{u} \rVert$, making direction and magnitude conflated. With $\lVert \mathbf{u} \rVert = 1$, $D_{\mathbf{u}} f$ measures purely the rate of change per unit distance in direction $\mathbf{u}$.

**Relationship to partial derivatives:** Partial derivatives are special cases. If $\mathbf{u} = \mathbf{e}_i$ (the $i$-th standard basis vector), then:

$$D_{\mathbf{e}_i} f(\mathbf{a}) = \lim_{h \to 0} \frac{f(\mathbf{a} + h\mathbf{e}_i) - f(\mathbf{a})}{h} = \frac{\partial f}{\partial x_i}(\mathbf{a})$$

So partial derivatives are directional derivatives along coordinate axes.

**Example:** $f(x,y) = x^2 - xy + y^2$, $\mathbf{a} = (1,1)$, $\mathbf{u} = (1/\sqrt{2}, 1/\sqrt{2})$ (northeast direction).

$$D_{\mathbf{u}} f(1,1) = \lim_{h \to 0} \frac{f(1 + h/\sqrt{2}, 1 + h/\sqrt{2}) - f(1,1)}{h}$$

We can compute this via the gradient formula (§5.2): $\nabla f = (2x - y, -x + 2y)$, so $\nabla f(1,1) = (1, 1)^\top$, and:

$$D_{\mathbf{u}} f(1,1) = (1,1) \cdot (1/\sqrt{2}, 1/\sqrt{2}) = \sqrt{2}$$

### 5.2 The Gradient Formula

**Theorem.** If $f$ is differentiable at $\mathbf{a}$, then for any unit vector $\mathbf{u}$:

$$D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a})^\top \mathbf{u} = \sum_{i=1}^n \frac{\partial f}{\partial x_i}(\mathbf{a}) \cdot u_i$$

**Proof.** Define $g(t) = f(\mathbf{a} + t\mathbf{u})$ for $t \in \mathbb{R}$. Then $D_{\mathbf{u}} f(\mathbf{a}) = g'(0)$. By the single-variable chain rule applied component-wise:

$$g'(t) = \sum_{i=1}^n \frac{\partial f}{\partial x_i}(\mathbf{a} + t\mathbf{u}) \cdot u_i$$

At $t = 0$: $g'(0) = \sum_{i=1}^n \frac{\partial f}{\partial x_i}(\mathbf{a}) \cdot u_i = \nabla f(\mathbf{a})^\top \mathbf{u}$. $\square$

**Geometric meaning:** $D_{\mathbf{u}} f(\mathbf{a}) = \nabla f(\mathbf{a})^\top \mathbf{u} = \lVert \nabla f(\mathbf{a}) \rVert \cos\theta$, where $\theta$ is the angle between $\nabla f(\mathbf{a})$ and $\mathbf{u}$.

**Consequence:** The entire directional derivative function $\mathbf{u} \mapsto D_{\mathbf{u}} f$ is determined by the single vector $\nabla f(\mathbf{a})$. This is the key economy: $n$ numbers encode all directional information.

### 5.3 Extremizing the Directional Derivative

**Summary table:**

| Direction | Angle $\theta$ | $D_{\mathbf{u}} f$ | Meaning |
|---|---|---|---|
| $\mathbf{u} = \hat{\nabla} f$ | $0°$ | $\lVert \nabla f \rVert$ | Steepest ascent |
| $\mathbf{u} \perp \nabla f$ | $90°$ | $0$ | Along level set |
| $\mathbf{u} = -\hat{\nabla} f$ | $180°$ | $-\lVert \nabla f \rVert$ | Steepest descent |

where $\hat{\nabla} f = \nabla f / \lVert \nabla f \rVert$ is the unit gradient.

**Worked example:** $f(x,y) = e^{x^2+y^2}$ at $\mathbf{a} = (1,0)$. Find: (a) steepest ascent direction; (b) directional derivative along $\mathbf{u} = (0,1)$.

$$\nabla f = \begin{pmatrix} 2x e^{x^2+y^2} \\ 2y e^{x^2+y^2} \end{pmatrix}, \qquad \nabla f(1,0) = \begin{pmatrix} 2e \\ 0 \end{pmatrix}$$

(a) Steepest ascent: $\hat{\nabla} f(1,0) = (1, 0)^\top$ (eastward), rate $= 2e$.

(b) $D_{(0,1)} f(1,0) = (2e, 0) \cdot (0,1) = 0$ — the Gaussian is flat in the $y$-direction at $(1,0)$ (it's on a ridge).

### 5.4 Subgradients — Preview

At points where $f$ is not differentiable, the gradient $\nabla f$ doesn't exist, but we can generalize via **subgradients**.

**Definition.** A vector $\mathbf{g} \in \mathbb{R}^n$ is a **subgradient** of $f$ at $\mathbf{a}$ if for all $\mathbf{x}$:

$$f(\mathbf{x}) \geq f(\mathbf{a}) + \mathbf{g}^\top (\mathbf{x} - \mathbf{a})$$

The **subdifferential** $\partial f(\mathbf{a})$ is the set of all subgradients.

**Example (ReLU):** $f(x) = \max(0, x)$:
- For $x > 0$: $\partial f(x) = \{1\}$ (unique gradient)
- For $x < 0$: $\partial f(x) = \{0\}$ (unique gradient)
- For $x = 0$: $\partial f(0) = [0, 1]$ (any $g \in [0,1]$ is a subgradient)

In PyTorch, `torch.relu(x).backward()` uses $g = 0$ at $x = 0$ by convention — this is a valid subgradient choice and works fine in practice since $x = 0$ occurs with probability zero for continuous distributions.

> **Full treatment of subgradients and convex optimization:** [§04-Optimality-Conditions](../04-Optimality-Conditions/notes.md) and [Chapter 8 — Optimization](../../08-Optimization/README.md).

---

## 6. Gradient in Machine Learning

### 6.1 Mean Squared Error Gradient

**Setup:** Linear regression model $\hat{\mathbf{y}} = X\mathbf{w}$, where $X \in \mathbb{R}^{m \times d}$ is the data matrix, $\mathbf{w} \in \mathbb{R}^d$ is the weight vector, $\mathbf{y} \in \mathbb{R}^m$ is the target vector.

**Loss:** Mean squared error (MSE):

$$\mathcal{L}(\mathbf{w}) = \frac{1}{m}\lVert X\mathbf{w} - \mathbf{y} \rVert^2 = \frac{1}{m}(X\mathbf{w} - \mathbf{y})^\top(X\mathbf{w} - \mathbf{y})$$

**Gradient derivation:**

Let $\mathbf{r} = X\mathbf{w} - \mathbf{y}$ (residuals). Then:

$$\mathcal{L} = \frac{1}{m}\mathbf{r}^\top \mathbf{r}$$

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{m} \nabla_{\mathbf{w}} [\mathbf{r}^\top \mathbf{r}] = \frac{1}{m} \cdot 2X^\top \mathbf{r} = \frac{2}{m} X^\top(X\mathbf{w} - \mathbf{y})$$

**Interpretation:** Each component $\frac{\partial \mathcal{L}}{\partial w_j} = \frac{2}{m} \sum_{i=1}^m (x_{ij})(r_i)$ is the correlation between the $j$-th feature and the residuals. The gradient is zero when all features are uncorrelated with the residuals — exactly the OLS normal equations $X^\top(X\mathbf{w} - \mathbf{y}) = \mathbf{0}$.

**Gradient descent for linear regression:**

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \frac{2}{m} X^\top(X\mathbf{w}_t - \mathbf{y})$$

With learning rate $\eta < 1/\sigma_{\max}(X^\top X / m)$, this converges to the least squares solution (Boyd & Vandenberghe, 2004, §9.3).

### 6.2 Cross-Entropy and Logistic Regression Gradient

**Logistic regression:** Binary classifier $\hat{y} = \sigma(\mathbf{w}^\top \mathbf{x})$ where $\sigma(z) = 1/(1 + e^{-z})$.

**Loss** (binary cross-entropy on $m$ examples):

$$\mathcal{L}(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)} \log \sigma(\mathbf{w}^\top \mathbf{x}^{(i)}) + (1 - y^{(i)}) \log(1 - \sigma(\mathbf{w}^\top \mathbf{x}^{(i)}))\right]$$

**Gradient derivation:**

For one example, let $z = \mathbf{w}^\top \mathbf{x}$, $p = \sigma(z)$. Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$\frac{\partial}{\partial \mathbf{w}}\left[-y\log p - (1-y)\log(1-p)\right] = \left(\frac{-y}{p} + \frac{1-y}{1-p}\right) \cdot \sigma'(z) \cdot \mathbf{x}$$

$$= \frac{p - y}{p(1-p)} \cdot p(1-p) \cdot \mathbf{x} = (p - y)\mathbf{x}$$

Therefore, averaging over $m$ examples:

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{m} X^\top(\hat{\mathbf{y}} - \mathbf{y})$$

where $\hat{\mathbf{y}} = \sigma(X\mathbf{w})$ elementwise. Strikingly, this has the same form as the linear regression gradient: **error times features**. This pattern holds for all generalized linear models in the exponential family.

**Softmax (multiclass) gradient:**

For $K$-class classification with softmax probabilities $p_k(\mathbf{x}) = \frac{e^{\mathbf{w}_k^\top \mathbf{x}}}{\sum_j e^{\mathbf{w}_j^\top \mathbf{x}}}$ and one-hot label $\mathbf{y}$:

$$\nabla_{\mathbf{w}_k} \mathcal{L} = \frac{1}{m} \sum_{i=1}^m (p_k(\mathbf{x}^{(i)}) - y_k^{(i)}) \mathbf{x}^{(i)}$$

In matrix form: $\nabla_W \mathcal{L} = \frac{1}{m} X^\top(P - Y)$, where $P \in \mathbb{R}^{m \times K}$ has rows of predicted probabilities.

**For AI:** The output gradient of a transformer's logit layer $\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \mathbf{p} - \mathbf{y}$ (predicted minus true distribution) is used directly in backpropagation. Its simplicity — no need to differentiate through the softmax explicitly — is why cross-entropy is the universal choice for classification losses.

### 6.3 Gradient of Regularized Objectives

Regularization adds a penalty term to the loss. The gradient of the combined objective is the sum of gradients:

**L2 regularization (weight decay):**

$$\mathcal{L}_{\text{reg}}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \frac{\lambda}{2}\lVert \mathbf{w} \rVert^2$$

$$\nabla_{\mathbf{w}} \mathcal{L}_{\text{reg}} = \nabla_{\mathbf{w}} \mathcal{L} + \lambda \mathbf{w}$$

The update rule becomes: $\mathbf{w}_{t+1} = \mathbf{w}_t(1 - \eta\lambda) - \eta \nabla_{\mathbf{w}} \mathcal{L}(\mathbf{w}_t)$. The factor $(1 - \eta\lambda)$ shrinks the weights at every step — this is called **weight decay**, and is the standard regularization in LLM training (AdamW with $\lambda = 0.01$–$0.1$).

**L1 regularization (sparsity):**

$$\mathcal{L}_{\text{reg}}(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \lVert \mathbf{w} \rVert_1$$

The gradient of $\lVert \mathbf{w} \rVert_1 = \sum_i |w_i|$ is $\operatorname{sign}(\mathbf{w})$ (elementwise), which is a subgradient at $w_i = 0$. L1 regularization promotes sparsity — many weights become exactly zero — but is rarely used in deep learning because automatic differentiation requires smooth functions.

**Elastic net:** $\lambda_1 \lVert \mathbf{w} \rVert_1 + \lambda_2 \lVert \mathbf{w} \rVert^2$ combines both.

### 6.4 Weight Sharing and Gradient Accumulation

**Weight sharing:** When the same parameter $w$ appears multiple times in a model (e.g., a shared embedding layer used in both encoder and decoder), the gradient is the sum of gradients from each occurrence:

$$\frac{\partial \mathcal{L}}{\partial w} = \sum_{\text{occurrences}} \frac{\partial \mathcal{L}}{\partial w_{\text{copy}}}$$

This follows from the multivariable chain rule. In convolutional neural networks, all positions share the same filter weights — the filter's gradient is the sum of its gradient contributions from every spatial location.

**Gradient accumulation:** For large batch sizes that don't fit in GPU memory, we split the batch into micro-batches, accumulate gradients across them, and apply one optimizer step:

```
for micro_batch in micro_batches:
    loss = model(micro_batch) / num_micro_batches
    loss.backward()  # accumulates .grad
optimizer.step()
optimizer.zero_grad()
```

This gives the same gradient as computing on the full batch because $\nabla \frac{1}{m}\sum_i \mathcal{L}_i = \frac{1}{m}\sum_i \nabla \mathcal{L}_i$ (linearity of the gradient).

**For AI:** GPT-3 and subsequent LLMs were trained with gradient accumulation because the full batch size (millions of tokens) far exceeds GPU memory. Gradient accumulation is mathematically equivalent to large-batch training, making it essential infrastructure for frontier model training.

### 6.5 Gradient Descent: The Algorithm

**Gradient descent** is the iterative algorithm for minimizing $\mathcal{L}(\boldsymbol{\theta})$:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

where $\eta_t > 0$ is the **learning rate** (or step size) at step $t$.

**Why it works (local guarantee):** By the first-order approximation:

$$\mathcal{L}(\boldsymbol{\theta}_{t+1}) \approx \mathcal{L}(\boldsymbol{\theta}_t) - \eta_t \lVert \nabla \mathcal{L}(\boldsymbol{\theta}_t) \rVert^2 + O(\eta_t^2)$$

For small enough $\eta_t$, the second term dominates, guaranteeing descent: $\mathcal{L}(\boldsymbol{\theta}_{t+1}) < \mathcal{L}(\boldsymbol{\theta}_t)$ (unless $\nabla \mathcal{L} = \mathbf{0}$).

**Variants:**

| Variant | Update | When to use |
|---|---|---|
| Batch GD | $\nabla$ over all $m$ examples | Exact gradient; slow per step |
| Stochastic GD (SGD) | $\nabla$ over 1 example | Noisy but fast; escapes sharp minima |
| Mini-batch GD | $\nabla$ over $B$ examples | Best of both; standard in practice |
| SGD + Momentum | Adds exponential moving average of gradient | Accelerated convergence; reduces oscillation |
| Adam | Adaptive per-coordinate learning rates | Default for transformer training |

**Convergence (L-smooth, convex case):** If $\mathcal{L}$ is convex and $L$-smooth with $\eta = 1/L$:

$$\mathcal{L}(\boldsymbol{\theta}_t) - \mathcal{L}^* \leq \frac{\lVert \boldsymbol{\theta}_0 - \boldsymbol{\theta}^* \rVert^2}{2\eta t}$$

This $O(1/t)$ rate is tight for gradient descent. Strongly convex functions achieve $O((1 - \mu/L)^t)$ linear (exponential) convergence.

**For AI:** Modern LLM training uses AdamW with cosine learning rate decay, gradient clipping at $G = 1.0$, and weight decay $\lambda = 0.1$. The gradient itself is the same object as described here — the engineering around it (clipping, schedule, accumulation) addresses numerical and optimization dynamics, not the mathematical definition.

> **Full convergence theory and algorithm variants:** [Chapter 8 — Optimization](../../08-Optimization/README.md).

---

## 7. Numerical Differentiation

### 7.1 Finite Difference Formulas

When an analytical gradient is unavailable or hard to verify, we can approximate it numerically using **finite differences**.

**Forward difference:**
$$\frac{\partial f}{\partial x_i}(\mathbf{a}) \approx \frac{f(\mathbf{a} + h\mathbf{e}_i) - f(\mathbf{a})}{h}$$
Error: $O(h)$ — one term of Taylor expansion is dropped.

**Backward difference:**
$$\frac{\partial f}{\partial x_i}(\mathbf{a}) \approx \frac{f(\mathbf{a}) - f(\mathbf{a} - h\mathbf{e}_i)}{h}$$
Error: $O(h)$ — same order.

**Centered difference:**
$$\frac{\partial f}{\partial x_i}(\mathbf{a}) \approx \frac{f(\mathbf{a} + h\mathbf{e}_i) - f(\mathbf{a} - h\mathbf{e}_i)}{2h}$$
Error: $O(h^2)$ — second-order accurate (all odd-order Taylor terms cancel).

**Derivation of centered difference error:**

By Taylor expansion:
$$f(\mathbf{a} + h\mathbf{e}_i) = f(\mathbf{a}) + h\frac{\partial f}{\partial x_i} + \frac{h^2}{2}\frac{\partial^2 f}{\partial x_i^2} + O(h^3)$$
$$f(\mathbf{a} - h\mathbf{e}_i) = f(\mathbf{a}) - h\frac{\partial f}{\partial x_i} + \frac{h^2}{2}\frac{\partial^2 f}{\partial x_i^2} + O(h^3)$$

Subtracting: $f(\mathbf{a} + h\mathbf{e}_i) - f(\mathbf{a} - h\mathbf{e}_i) = 2h\frac{\partial f}{\partial x_i} + O(h^3)$, so the centered difference approximation has error $O(h^2)$.

**Computational cost:** Computing the full gradient $\nabla f \in \mathbb{R}^n$ via centered differences requires $2n$ function evaluations. For a model with $n = 10^9$ parameters, this is computationally infeasible — this is why automatic differentiation (§05) is essential.

### 7.2 Optimal Step Size

Choosing $h$ involves a fundamental trade-off:

- **Too large** $h$: Taylor truncation error $O(h^2)$ is large
- **Too small** $h$: Floating-point subtraction cancellation error dominates

For centered differences, the optimal $h$ minimizes total error $\approx h^2 C + \epsilon_\text{mach} / h$, which gives:

$$h_\text{opt} \approx \epsilon_\text{mach}^{1/3} \approx 10^{-5} \qquad (\text{for double precision } \epsilon_\text{mach} \approx 10^{-16})$$

```
NUMERICAL GRADIENT ERROR vs. STEP SIZE h
════════════════════════════════════════════════════════════════════════

  log(error)
      │
   -2 ┤
      │╲  Truncation error O(h²)
   -4 ┤ ╲
      │  ╲                 Optimal h ≈ 1e-5
   -6 ┤   ╲_____________/
      │     minimum     ╲ Roundoff error ε/h
   -8 ┤                  ╲
      │                   ╲
  -10 ┤                    ╲
      └──┬──┬──┬──┬──┬──── log(h)
        -12 -9 -6 -3  0

════════════════════════════════════════════════════════════════════════
```

**Rule of thumb:** Use $h = 10^{-5}$ for centered differences in double precision. In single precision ($\epsilon_\text{mach} \approx 10^{-7}$), use $h \approx 10^{-4}$.

### 7.3 Gradient Checking

**Gradient checking** is the practice of verifying an analytical gradient implementation by comparing it to the numerical gradient.

**Algorithm:**
1. Compute analytical gradient $\mathbf{g}_\text{anal} = \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta})$ using backpropagation
2. For each parameter $\theta_i$, compute:
   $$g_i^\text{num} = \frac{\mathcal{L}(\boldsymbol{\theta} + h\mathbf{e}_i) - \mathcal{L}(\boldsymbol{\theta} - h\mathbf{e}_i)}{2h}$$
3. Compute **relative error**:
   $$\text{relerr} = \frac{\lVert \mathbf{g}_\text{anal} - \mathbf{g}_\text{num} \rVert_2}{\lVert \mathbf{g}_\text{anal} \rVert_2 + \lVert \mathbf{g}_\text{num} \rVert_2}$$

**Thresholds:**
- $< 10^{-7}$: Great — gradient is likely correct
- $10^{-7}$ to $10^{-5}$: Acceptable for most applications
- $> 10^{-3}$: Bug in gradient — investigate component-by-component

**Practical tips:**
- Use double precision (`float64`) during gradient checks, even if training uses `float16`
- Check on a small random input, not on all training data
- Check each component separately to localize bugs
- Do NOT gradient check with dropout active (stochastic masks break the comparison)

**For AI:** Andrej Karpathy's `micrograd` tutorial uses gradient checking to verify backpropagation at every step. Gradient checking was critical during the early days of deep learning to ensure backprop implementations were correct; today, libraries like PyTorch and JAX are well-tested, but gradient checking remains essential when implementing custom layers or loss functions.

### 7.4 Limitations and the Case for Automatic Differentiation

Numerical differentiation has three fatal limitations for large models:

1. **Cost:** $O(n)$ function evaluations for an $n$-parameter model — infeasible for $n > 10^6$
2. **Accuracy:** $O(h^2)$ error is often insufficient for sensitive applications (e.g., second-order methods)
3. **No higher-order:** Computing Hessian-vector products numerically costs $O(n^2)$

Automatic differentiation (§05) resolves all three:
- **Cost:** Reverse mode computes the full gradient in $O(1)$ model evaluations (same cost as a forward pass)
- **Accuracy:** Exact to machine precision — no truncation error
- **Higher-order:** Forward-over-reverse AD computes Hessian-vector products exactly in $O(1)$ cost

> **Full treatment:** [§05-Automatic-Differentiation](../05-Automatic-Differentiation/notes.md).

---

## 8. Loss Landscape Geometry

### 8.1 Loss Landscapes for Neural Networks

A neural network's loss $\mathcal{L}(\boldsymbol{\theta})$ defines a high-dimensional surface over parameter space $\mathbb{R}^n$. Unlike convex functions (which have a unique global minimum), neural network loss landscapes are:

- **Non-convex:** Multiple local minima, saddle points, and flat regions
- **High-dimensional:** Intuitions from 2D and 3D break down; most "local minima" in high dimensions are actually saddle points with at least one negative curvature direction
- **Approximately convex near minima:** Empirically, well-trained networks sit in wide flat basins where the Hessian has many near-zero eigenvalues and few negative ones

**Critical points and their classification** (using the Hessian — see §02):

| Type | $\nabla \mathcal{L}$ | Hessian eigenvalues | Behavior |
|---|---|---|---|
| Global minimum | $= \mathbf{0}$ | All $> 0$ | Best possible parameters |
| Local minimum | $= \mathbf{0}$ | All $> 0$ | Training converges here; good generalization if in a wide basin |
| Saddle point | $= \mathbf{0}$ | Mixed signs | Gradient descent escapes with noise (SGD) |
| Flat region | $\approx \mathbf{0}$ | Many $\approx 0$ | Plateau; training stalls; warm restarts help |

**The Hessian eigenvalue view:** The curvature of the loss surface at a minimum is characterized by the Hessian eigenvalues. Wide minima (small eigenvalues) generalize better (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017) — this is the theoretical basis for SAM (Sharpness-Aware Minimization, Foret et al., 2021), which explicitly seeks flat minima.

### 8.2 Gradient Norm as a Training Diagnostic

The gradient norm $\lVert \nabla_{\boldsymbol{\theta}} \mathcal{L} \rVert$ is one of the most informative training diagnostics:

| Gradient norm behavior | Possible cause |
|---|---|
| Steadily decreasing | Normal convergence |
| Suddenly large (spike) | Batch with unusual data; learning rate too high |
| Near zero early in training | Vanishing gradients (very deep networks without skip connections) |
| Near zero late in training | Near a critical point (minimum or saddle) |
| Persistently large | Divergence; reduce learning rate |

**Vanishing gradients:** In deep networks without normalization or skip connections, the gradient norm often decays exponentially with depth. If layer $l$ has weight matrix $W^{[l]}$ with $\lVert W^{[l]} \rVert_2 < 1$, then $\lVert \nabla_{\boldsymbol{\theta}^{[1]}} \mathcal{L} \rVert \leq \prod_l \lVert W^{[l]} \rVert_2 \to 0$ exponentially. This motivated: BatchNorm (Ioffe & Szegedy, 2015), LayerNorm (Ba et al., 2016), and residual connections (He et al., 2016).

### 8.3 Gradient Clipping

**Problem:** In recurrent networks (RNNs, early transformers), gradients can explode — growing exponentially through time — making training unstable.

**Solution (Pascanu et al., 2013):** Rescale the gradient whenever its norm exceeds a threshold $G$:

$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g} & \text{if } \lVert \mathbf{g} \rVert \leq G \\ G \cdot \frac{\mathbf{g}}{\lVert \mathbf{g} \rVert} & \text{if } \lVert \mathbf{g} \rVert > G \end{cases}$$

This preserves the **direction** of the gradient while bounding the step size. It is not the same as gradient truncation (which clips each component independently and changes the direction).

**Effect:** Clipping to $G$ guarantees the update $\eta \mathbf{g}$ has norm at most $\eta G$, regardless of the curvature of $\mathcal{L}$.

**Standard values:** $G = 1.0$ for transformer training (used in GPT-2, GPT-3, Llama, and most modern LLMs). The PyTorch call is `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.

**For AI:** Gradient clipping is a near-universal practice in LLM training. Without it, a single pathological batch can send parameters into a numerically unstable region from which training never recovers. The choice $G = 1.0$ is empirical — it rarely triggers during stable training but provides a safety net during instability events.

### 8.4 Gradient Accumulation for Large Batch Training

When the effective batch size $B$ is larger than what fits in GPU memory, **gradient accumulation** simulates the large batch:

$$\nabla_{\boldsymbol{\theta}} \mathcal{L}_\text{batch} = \frac{1}{B}\sum_{i=1}^B \nabla_{\boldsymbol{\theta}} \mathcal{L}_i = \frac{1}{k}\sum_{j=1}^k \underbrace{\frac{1}{B/k}\sum_{i \in \text{micro-batch}_j} \nabla_{\boldsymbol{\theta}} \mathcal{L}_i}_{\text{micro-batch gradient}}$$

where $k$ is the number of gradient accumulation steps. The gradient is accumulated across $k$ micro-batches before a single optimizer step.

**Mathematical validity:** By linearity of the gradient, the accumulated gradient is exactly equal to the gradient computed over the full batch. This holds exactly (not approximately).

**For AI:** Llama 2 70B used a global batch size of 2048 sequences of 4096 tokens — far exceeding GPU memory for a single forward pass. Gradient accumulation, combined with data-parallel training across multiple GPUs, makes this feasible.

---

## 9. Advanced Topics

### 9.1 Gateaux and Fréchet Derivatives

In infinite-dimensional settings (function spaces), the ordinary gradient doesn't apply. Two generalizations are:

**Gateaux derivative (directional derivative in function space):**

$$D_h F(f) = \lim_{\varepsilon \to 0} \frac{F(f + \varepsilon h) - F(f)}{\varepsilon}$$

where $F$ is a functional (maps functions to scalars), $f$ is a function, and $h$ is a perturbation function. This is the infinite-dimensional analog of the directional derivative.

**Fréchet derivative:** The Gateaux derivative $D_h F(f)$ is linear and continuous in $h$, and can be written as $D_h F(f) = \langle \nabla F(f), h \rangle$ for some function $\nabla F(f)$ — this is the functional gradient.

**Applications in ML:**
- **Variational inference:** Optimizing the ELBO $\mathcal{L}[q]$ over distributions $q$ — the gradient is a functional derivative
- **Optimal transport:** Wasserstein gradient flows define how distributions evolve under gradient descent in the space of probability measures
- **Neural tangent kernel (NTK):** As network width $\to \infty$, gradient descent on $\boldsymbol{\theta}$ induces gradient descent in function space with the NTK as inner product

### 9.2 Wirtinger Derivatives

When optimizing real-valued functions of complex variables — common in signal processing, compressed sensing, and some neural network architectures — the standard gradient doesn't apply because complex functions satisfying the Cauchy-Riemann equations are conformal (angle-preserving), and most practical functions are not holomorphic.

**Wirtinger calculus** defines pseudo-derivatives:

$$\frac{\partial f}{\partial z} = \frac{1}{2}\left(\frac{\partial f}{\partial x} - i\frac{\partial f}{\partial y}\right), \qquad \frac{\partial f}{\partial \bar{z}} = \frac{1}{2}\left(\frac{\partial f}{\partial x} + i\frac{\partial f}{\partial y}\right)$$

where $z = x + iy$. For real-valued $f$, gradient descent follows $-\frac{\partial f}{\partial \bar{z}}$ (the Wirtinger gradient).

**For AI:** Complex-valued neural networks used in audio processing and MRI reconstruction use Wirtinger derivatives. PyTorch supports complex tensors and correctly computes Wirtinger gradients via `.backward()`.

### 9.3 Gradients on Manifolds and the Natural Gradient

Standard gradient descent treats parameter space as flat Euclidean space. But probability distributions form a **statistical manifold** with a non-Euclidean geometry defined by the **Fisher information matrix**:

$$F(\boldsymbol{\theta}) = \mathbb{E}_{x \sim p_{\boldsymbol{\theta}}}\!\left[\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x) \nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(x)^\top\right]$$

The **natural gradient** (Amari, 1998) accounts for this geometry:

$$\tilde{\nabla}_{\boldsymbol{\theta}} \mathcal{L} = F(\boldsymbol{\theta})^{-1} \nabla_{\boldsymbol{\theta}} \mathcal{L}$$

Natural gradient descent converges faster than standard gradient descent because it takes steps of equal "distributional size" rather than equal "parameter size."

**Connection to LLM fine-tuning:**
- **LoRA (Hu et al., 2022):** Constrains the gradient update $\Delta W = BA$ to a low-rank subspace. The intuition is that the gradient lies in a low-rank manifold, and parameterizing it as $BA$ is an efficient approximation
- **K-FAC (Martens & Grosse, 2015):** Approximates the Fisher information as a Kronecker product of smaller matrices, making natural gradient descent tractable for neural networks
- **DoRA (Liu et al., 2024):** Decomposes weight updates into magnitude and direction components, further refining the LoRA idea

**Riemannian gradient:** On a manifold with metric tensor $G(\boldsymbol{\theta})$, the Riemannian gradient is $G(\boldsymbol{\theta})^{-1}\nabla_{\boldsymbol{\theta}} \mathcal{L}$. The Fisher information $F(\boldsymbol{\theta})$ plays the role of $G$ on the statistical manifold.

### 9.4 Jacobian Preview

So far we have discussed gradients of **scalar** functions $f: \mathbb{R}^n \to \mathbb{R}$. When $f$ is **vector-valued**, $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the gradient generalizes to the **Jacobian matrix** $J_{\mathbf{f}} \in \mathbb{R}^{m \times n}$:

$$(J_{\mathbf{f}})_{ij} = \frac{\partial f_i}{\partial x_j}$$

The Jacobian is the "matrix of all first-order partial derivatives" — its $i$-th row is the gradient of the $i$-th output component.

> **Preview: Jacobians and Hessians**
>
> When $f: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian $J_{\mathbf{f}} \in \mathbb{R}^{m \times n}$ generalizes the gradient and is the fundamental object for backpropagation through vector-valued layers. The Hessian $H_f \in \mathbb{R}^{n \times n}$ generalizes the second derivative and characterizes curvature, saddle points, and Newton's method.
>
> → _Full treatment: [§02 Jacobians and Hessians](../02-Jacobians-and-Hessians/notes.md)_

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Writing $\nabla f$ as a row vector | Gradient is a column vector $\in \mathbb{R}^n$; writing as a row conflates it with the Jacobian (which is $1 \times n$ for scalar $f$) and causes dimension errors in $\nabla f^\top \delta$ | Always write $\nabla f \in \mathbb{R}^n$ as a column vector; write $\nabla f^\top$ when you need a row |
| 2 | Confusing $\frac{\partial f}{\partial x}\frac{\partial g}{\partial y}$ with $\frac{\partial^2 (fg)}{\partial x \partial y}$ | The product rule applies: $\frac{\partial^2 (fg)}{\partial x \partial y} = \frac{\partial f}{\partial x}\frac{\partial g}{\partial y} + f \frac{\partial^2 g}{\partial x \partial y} + \frac{\partial^2 f}{\partial x \partial y} g + \frac{\partial g}{\partial x}\frac{\partial f}{\partial y}$ | Apply the product rule in each differentiation step |
| 3 | Treating $\nabla(\mathbf{x}^\top A \mathbf{x}) = 2A\mathbf{x}$ when $A$ is not symmetric | This holds only when $A = A^\top$; in general, $\nabla(\mathbf{x}^\top A \mathbf{x}) = (A + A^\top)\mathbf{x}$ | Check symmetry; use the general formula from §4.5 |
| 4 | Using forward differences instead of centered differences for gradient checking | Forward difference has $O(h)$ error; centered has $O(h^2)$; at $h = 10^{-5}$ this is $10^{-5}$ vs $10^{-10}$ — a factor of $10^5$ | Always use centered differences $\frac{f(\mathbf{x}+h\mathbf{e}_i) - f(\mathbf{x}-h\mathbf{e}_i)}{2h}$ for gradient checking |
| 5 | Concluding a point is a local minimum because all partial derivatives are zero | $\nabla f = \mathbf{0}$ is necessary but not sufficient; the point could be a saddle point (e.g., $f(x,y) = x^2 - y^2$ at the origin) | Check the Hessian: if $H_f \succ 0$, it's a local minimum (§02, §04) |
| 6 | Claiming a function is differentiable because its partial derivatives exist | False — all partials can exist while $f$ is not even continuous (§2.4 counterexample) | Differentiability requires more: the linear approximation error must be $o(\lVert \boldsymbol{\delta}\rVert)$ |
| 7 | Applying gradient descent with a step $\eta > 2/L$ for $L$-smooth functions | This can cause divergence; the safe step size for gradient descent is $\eta \leq 1/L$ where $L$ is the Lipschitz constant of $\nabla f$ | Check the smoothness constant; use line search or backtracking if unknown |
| 8 | Gradient clipping individual components (element-wise) instead of the norm | Element-wise clipping changes the gradient direction and can slow convergence; norm clipping preserves direction | Use `clip_grad_norm_()` (norm clipping), not `clip_grad_value_()` unless you specifically want element-wise clipping |
| 9 | Ignoring the $\frac{1}{m}$ factor in batch gradients vs. single-sample gradients | Learning rate must be retuned when switching batch sizes; gradient magnitudes scale with $m$ | Consistently include the $\frac{1}{m}$ factor in the loss; or rescale learning rate by $B$ when changing batch size $B$ |
| 10 | Confusing the directional derivative with the gradient projected onto $\mathbf{u}$ | The directional derivative $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}$ requires $\lVert \mathbf{u} \rVert = 1$; for non-unit $\mathbf{u}$, the result scales with $\lVert \mathbf{u} \rVert$ | Normalize $\mathbf{u}$ before computing the directional derivative |

---

## 11. Exercises

**Exercise 1** ★ — Partial Derivatives

Let $f(x, y, z) = x^3 y^2 - e^{xz} + \ln(y^2 + z^2)$.

**(a)** Compute $\frac{\partial f}{\partial x}$, $\frac{\partial f}{\partial y}$, $\frac{\partial f}{\partial z}$.

**(b)** Evaluate $\nabla f(1, 1, 0)$.

**(c)** Verify that the mixed partials $\frac{\partial^2 f}{\partial x \partial y}$ and $\frac{\partial^2 f}{\partial y \partial x}$ are equal.

**(d)** For AI: The loss $\mathcal{L}(w_1, w_2) = (w_1^2 - w_2)^2 + (w_2 - 1)^2$. Find $\nabla \mathcal{L}$ and the critical points.

---

**Exercise 2** ★ — Gradient Computation and Geometry

Let $f(x,y) = x^2 + xy + y^2$.

**(a)** Compute $\nabla f$ and evaluate at $(1, -1)$, $(0, 0)$, $(2, 3)$.

**(b)** At each of the three points, find the direction of steepest ascent (unit vector).

**(c)** Find all level curves $f(x,y) = c$ for $c = 1, 3, 7$. Verify numerically that $\nabla f(1,-1)$ is perpendicular to the level curve passing through $(1,-1)$.

**(d)** Show that $f(x,y) = x^2 + xy + y^2 > 0$ for all $(x,y) \neq (0,0)$ by completing the square or using the discriminant of the associated quadratic form.

---

**Exercise 3** ★ — Directional Derivatives

Let $f(x,y) = \sin(xy) + \cos(x)$.

**(a)** Compute $\nabla f$ at $\mathbf{a} = (\pi/2, 1)$.

**(b)** Find the direction $\mathbf{u}$ that maximizes $D_{\mathbf{u}} f(\mathbf{a})$ and the direction that minimizes it. What are the maximum and minimum values?

**(c)** Find all directions $\mathbf{u}$ such that $D_{\mathbf{u}} f(\mathbf{a}) = 0$.

**(d)** Compute the directional derivative in direction $(3/5, 4/5)^\top$ and in direction $(1/\sqrt{2}, -1/\sqrt{2})^\top$.

---

**Exercise 4** ★★ — Matrix Calculus from First Principles

Prove the following from the definition $\frac{\partial f}{\partial x_k} = \lim_{h\to 0}\frac{f(\mathbf{x}+h\mathbf{e}_k)-f(\mathbf{x})}{h}$:

**(a)** $\nabla_{\mathbf{x}}(\mathbf{a}^\top \mathbf{x}) = \mathbf{a}$ for fixed $\mathbf{a} \in \mathbb{R}^n$.

**(b)** $\nabla_{\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (A + A^\top)\mathbf{x}$ for fixed $A \in \mathbb{R}^{n \times n}$.

**(c)** $\nabla_{\mathbf{x}} \lVert A\mathbf{x} - \mathbf{b} \rVert^2 = 2A^\top(A\mathbf{x} - \mathbf{b})$ for fixed $A \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$.

**(d)** Show that the gradient of the MSE loss $\frac{1}{m}\lVert X\mathbf{w} - \mathbf{y} \rVert^2$ equals zero when $X^\top X \mathbf{w} = X^\top \mathbf{y}$ — the normal equations of linear regression.

---

**Exercise 5** ★★ — MSE Gradient and Linear Regression

Implement gradient descent for linear regression from scratch.

**(a)** Generate synthetic data: $\mathbf{y} = X\mathbf{w}^* + \boldsymbol{\varepsilon}$ where $X \in \mathbb{R}^{100 \times 3}$, $\mathbf{w}^* = (2, -1, 0.5)^\top$, $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, 0.1I)$.

**(b)** Implement the MSE gradient $\nabla_{\mathbf{w}} \mathcal{L} = \frac{2}{m}X^\top(X\mathbf{w} - \mathbf{y})$.

**(c)** Run gradient descent for 1000 steps with $\eta = 0.01$. Plot the loss vs. step number.

**(d)** Compute the analytical solution $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top \mathbf{y}$ and compare with gradient descent's solution.

**(e)** How does the convergence rate change with learning rate $\eta$? Try $\eta \in \{0.001, 0.01, 0.1, 1.0\}$.

---

**Exercise 6** ★★ — Gradient Checking Implementation

**(a)** Implement centered-difference numerical gradient for an arbitrary function $f: \mathbb{R}^n \to \mathbb{R}$.

**(b)** Test on $f(\mathbf{w}) = \frac{1}{2}\lVert X\mathbf{w} - \mathbf{y} \rVert^2$ with random data. Compute the relative error between numerical and analytical gradients.

**(c)** Plot relative error vs. step size $h$ for $h \in [10^{-12}, 10^{-1}]$. Identify the optimal $h$ and the crossover between truncation error and floating-point cancellation.

**(d)** Implement a gradient checker for the logistic regression cross-entropy loss. Use it to verify the gradient formula $\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{m}X^\top(\sigma(X\mathbf{w}) - \mathbf{y})$.

---

**Exercise 7** ★★★ — Softmax Gradient Derivation

The softmax function $\operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$ takes logits $\mathbf{z} \in \mathbb{R}^K$ to a probability distribution.

**(a)** Compute the Jacobian $\frac{\partial p_k}{\partial z_j}$ for all pairs $(k,j)$. Show that it equals $p_k(\delta_{kj} - p_j)$ where $\delta_{kj}$ is the Kronecker delta.

**(b)** Let $\mathcal{L} = -\sum_k y_k \log p_k$ (cross-entropy with one-hot labels $\mathbf{y}$). Using the chain rule $\frac{\partial \mathcal{L}}{\partial z_j} = \sum_k \frac{\partial \mathcal{L}}{\partial p_k}\frac{\partial p_k}{\partial z_j}$, derive $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$.

**(c)** Verify numerically: compute $\nabla_{\mathbf{z}} \mathcal{L}$ using your formula and using centered differences. Confirm relative error $< 10^{-6}$.

**(d)** Why is the gradient $\mathbf{p} - \mathbf{y}$ so elegant? Explain its interpretation: what happens when the model is perfectly confident and correct? When it's confident and wrong?

---

**Exercise 8** ★★★ — Natural Gradient vs. Gradient Descent

Consider fitting a Gaussian model $p_{\mu,\sigma}(x) = \mathcal{N}(x;\mu,\sigma^2)$ to observed data $\{x_1,\ldots,x_m\}$ by maximizing the log-likelihood $\mathcal{L}(\mu,\sigma) = \sum_i \log p_{\mu,\sigma}(x_i)$.

**(a)** Compute the ordinary gradient $\nabla_{(\mu,\sigma)} \mathcal{L}$.

**(b)** Compute the Fisher information matrix $F(\mu,\sigma) = -\mathbb{E}\left[\nabla^2 \log p_{\mu,\sigma}(X)\right]$ and show that for a Gaussian, $F = \operatorname{diag}(1/\sigma^2, 2/\sigma^2)$.

**(c)** Compute the natural gradient $F^{-1}\nabla\mathcal{L}$ and compare it to the ordinary gradient.

**(d)** Simulate: generate 100 points from $\mathcal{N}(2, 1)$, run gradient descent and natural gradient descent from $(\mu_0,\sigma_0) = (0, 2)$. Plot trajectories in $(\mu,\sigma)$ space. Which converges faster?

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Impact |
|---|---|
| **Gradient $\nabla_{\boldsymbol{\theta}}\mathcal{L}$** | The update signal for every parameter in every neural network; computed billions of times per second during LLM training |
| **MSE gradient derivation** | Foundation of all linear models; normal equations; least squares regularization; used in LoRA target initialization |
| **Cross-entropy gradient $\mathbf{p} - \mathbf{y}$** | Elegantly simple form used in every transformer output layer; enables efficient backpropagation through softmax |
| **Gradient as steepest descent direction** | Theoretical guarantee that SGD moves downhill; motivates all descent-based optimization algorithms |
| **Gradient perpendicular to level sets** | Foundation of constrained optimization; used in gradient projection methods and orthogonal gradient descent for continual learning |
| **First-order Taylor approximation** | Guarantees gradient descent decreases loss for small steps; basis of convergence proofs for smooth functions |
| **Gradient of quadratic forms** | $\nabla(\mathbf{x}^\top A\mathbf{x}) = 2A\mathbf{x}$ is used in kernel methods, Gaussian process regression, and second-order optimization |
| **Gradient clipping** | Standard practice in all LLM training; prevents training instability from pathological batches; norm $G=1.0$ used in GPT-3, Llama, Mistral |
| **Gradient accumulation** | Enables large effective batch sizes (millions of tokens) on GPU-memory-constrained systems; essential for frontier model training |
| **Numerical gradient checking** | Debugging tool for custom layers; verifies correctness of backpropagation implementation before costly training runs |
| **Weight decay as L2 gradient** | AdamW decouples weight decay from the adaptive learning rate; the gradient of L2 regularization adds $\lambda\mathbf{w}$ to the update |
| **Gradient of log-likelihood** | Score function $\nabla_{\boldsymbol{\theta}}\log p_{\boldsymbol{\theta}}(x)$ used in REINFORCE, PPO, and all policy gradient reinforcement learning methods |
| **Natural gradient / Fisher information** | K-FAC approximates it for efficient second-order optimization; Shampoo computes Kronecker factors; understanding it motivates AdaGrad, RMSprop, Adam |
| **Functional gradients** | Wasserstein gradient flows, neural tangent kernels, variational inference — all use gradients in infinite-dimensional function spaces |

---

## Conceptual Bridge

### Looking Backward

The gradient builds directly on two prerequisites:

1. **Single-variable chain rule** ([§04/02](../../04-Calculus-Fundamentals/02-Derivatives-and-Differentiation/notes.md)): The directional derivative formula $D_{\mathbf{u}} f = \nabla f^\top \mathbf{u}$ was proved using the chain rule along a curve $\boldsymbol{\gamma}(t)$. Every property of the gradient ultimately rests on single-variable calculus applied component-by-component.

2. **First-order Taylor series** ([§04/04](../../04-Calculus-Fundamentals/04-Series-and-Sequences/notes.md)): The gradient-as-linear-approximation view $f(\mathbf{a}+\boldsymbol{\delta}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top\boldsymbol{\delta}$ is the multivariate Taylor expansion truncated at first order. The remainder is $O(\lVert\boldsymbol{\delta}\rVert^2)$, controlled by the second-order term — the Hessian.

3. **Vector norms and inner products** ([§02-Linear-Algebra-Basics](../../02-Linear-Algebra-Basics/README.md)): The Cauchy-Schwarz inequality $\nabla f^\top \mathbf{u} \leq \lVert\nabla f\rVert\lVert\mathbf{u}\rVert$ was the key step in proving the gradient points in the steepest ascent direction.

### Looking Forward

This section is the foundation for three immediate extensions:

1. **Jacobians and Hessians** (§02): When $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ is vector-valued, each output $f_i$ has a gradient $\nabla f_i \in \mathbb{R}^n$. Stacking these as rows gives the Jacobian $J_{\mathbf{f}} \in \mathbb{R}^{m \times n}$. The Hessian stacks the gradients of the gradient components to form a symmetric matrix capturing curvature.

2. **Chain Rule and Backpropagation** (§03): Backpropagation is the multivariate chain rule applied to computation graphs. If $\mathbf{z} = \mathbf{f}(\mathbf{y})$ and $\mathbf{y} = \mathbf{g}(\mathbf{x})$, then $J_{\mathbf{f}\circ\mathbf{g}} = J_{\mathbf{f}} \cdot J_{\mathbf{g}}$. Propagating gradients backward layer by layer uses exactly this matrix product structure.

3. **Optimality Conditions** (§04): The condition $\nabla f(\mathbf{a}) = \mathbf{0}$ is necessary (but not sufficient) for $\mathbf{a}$ to be a local minimum. Adding the condition $H_f(\mathbf{a}) \succ 0$ (positive definite Hessian) makes it sufficient. Lagrange multipliers extend this to constrained optimization.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  Ch. 4 — Calculus Fundamentals
  ├── §01 Limits and Continuity    (ε-δ, sequential limits)
  ├── §02 Derivatives              (chain rule, activation derivatives)
  ├── §03 Integration              (FTC, probability integrals)
  └── §04 Series and Sequences     (Taylor series, convergence)
         │
         │  builds on: derivatives, Taylor expansion, vector algebra
         ▼
  Ch. 5 — Multivariate Calculus
  ├── §01 Partial Derivatives and Gradients   ← YOU ARE HERE
  │        • Partial derivatives = derivatives holding others fixed
  │        • Gradient = vector of all partial derivatives
  │        • Directional derivative = ∇f · u
  │        • Gradient of ML losses (MSE, cross-entropy)
  │        • Numerical differentiation, gradient checking
  │
  ├── §02 Jacobians and Hessians              → matrix generalization
  │        (gradient of vector-valued f; curvature matrix)
  │
  ├── §03 Chain Rule and Backpropagation      → gradient composition
  │        (Jacobian chain rule; backprop algorithm)
  │
  ├── §04 Optimality Conditions               → when ∇f = 0 is a minimum
  │        (first/second-order conditions; KKT; Lagrange multipliers)
  │
  └── §05 Automatic Differentiation           → computing ∇f exactly
           (reverse mode; tapes; PyTorch/JAX)
         │
         ▼
  Ch. 8 — Optimization
  (gradient descent variants, Adam, Newton, convergence theory)

════════════════════════════════════════════════════════════════════════
```

---

[← Back to Multivariate Calculus](../README.md) | [Next: Jacobians and Hessians →](../02-Jacobians-and-Hessians/notes.md)

---

## Appendix A: Standard Gradient Reference Table

The following identities are used throughout the curriculum. All assume $\mathbf{x}, \mathbf{a} \in \mathbb{R}^n$, $A \in \mathbb{R}^{n \times n}$, $B \in \mathbb{R}^{m \times n}$, $\mathbf{b} \in \mathbb{R}^m$ unless noted.

### A.1 Linear Forms

| $f(\mathbf{x})$ | $\nabla_{\mathbf{x}} f$ | Notes |
|---|---|---|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ | Constant gradient — loss increases uniformly in direction $\mathbf{a}$ |
| $\mathbf{x}^\top \mathbf{a}$ | $\mathbf{a}$ | Same as above (dot product commutes) |
| $\mathbf{x}^\top \mathbf{x} = \lVert\mathbf{x}\rVert^2$ | $2\mathbf{x}$ | Gradient points radially outward |
| $\lVert\mathbf{x} - \mathbf{a}\rVert^2$ | $2(\mathbf{x} - \mathbf{a})$ | Points away from $\mathbf{a}$; zero at $\mathbf{a}$ |
| $\lVert\mathbf{x}\rVert$ | $\mathbf{x}/\lVert\mathbf{x}\rVert$ | Unit vector in direction of $\mathbf{x}$; undefined at $\mathbf{0}$ |

### A.2 Quadratic Forms

| $f(\mathbf{x})$ | $\nabla_{\mathbf{x}} f$ | Notes |
|---|---|---|
| $\mathbf{x}^\top A \mathbf{x}$ | $(A + A^\top)\mathbf{x}$ | General $A$ |
| $\mathbf{x}^\top A \mathbf{x}$ | $2A\mathbf{x}$ | When $A = A^\top$ (symmetric) |
| $\mathbf{a}^\top A \mathbf{x}$ | $A^\top \mathbf{a}$ | Bilinear: gradient w.r.t. $\mathbf{x}$ |
| $\mathbf{x}^\top A \mathbf{b}$ | $A\mathbf{b}$ | Linear in $\mathbf{x}$ |

### A.3 Least Squares and Norms

| $f(\mathbf{x})$ | $\nabla_{\mathbf{x}} f$ | Derivation |
|---|---|---|
| $\lVert B\mathbf{x} - \mathbf{b}\rVert^2$ | $2B^\top(B\mathbf{x} - \mathbf{b})$ | Chain rule: $\nabla\lVert\mathbf{r}\rVert^2 = 2B^\top\mathbf{r}$ |
| $\lVert B\mathbf{x} - \mathbf{b}\rVert^2 + \lambda\lVert\mathbf{x}\rVert^2$ | $2B^\top(B\mathbf{x}-\mathbf{b}) + 2\lambda\mathbf{x}$ | Ridge regression gradient |
| $\frac{1}{2}\mathbf{x}^\top A\mathbf{x} - \mathbf{b}^\top\mathbf{x}$ | $A\mathbf{x} - \mathbf{b}$ ($A$ symmetric) | Set to 0: $A\mathbf{x}=\mathbf{b}$ (linear system) |

### A.4 Elementwise Functions

For elementwise functions $f(\mathbf{x}) = \sum_i g(x_i)$, the gradient is elementwise:

$$(\nabla f)_i = g'(x_i)$$

| $f(\mathbf{x})$ | $\nabla_{\mathbf{x}} f$ | ML use |
|---|---|---|
| $\sum_i e^{x_i}$ | $e^{\mathbf{x}}$ (elementwise) | Partition function in softmax |
| $\sum_i \log x_i$ | $1/\mathbf{x}$ (elementwise, $x_i > 0$) | Log-likelihood |
| $\sum_i \sigma(x_i)$ | $\sigma(\mathbf{x})\odot(1-\sigma(\mathbf{x}))$ | Sigmoid gradient |
| $\sum_i \max(0, x_i)$ | $\mathbf{1}[\mathbf{x}>0]$ (indicator) | ReLU gradient (subgradient at 0) |

### A.5 Composed Functions

| $f(\mathbf{x})$ | $\nabla_{\mathbf{x}} f$ | Proof |
|---|---|---|
| $g(\mathbf{a}^\top\mathbf{x})$ | $g'(\mathbf{a}^\top\mathbf{x})\,\mathbf{a}$ | Chain rule: $\partial/\partial x_i = g'\cdot a_i$ |
| $g(\lVert\mathbf{x}\rVert^2)$ | $2g'(\lVert\mathbf{x}\rVert^2)\,\mathbf{x}$ | Chain rule on $h = \lVert\mathbf{x}\rVert^2$ |
| $\exp(\mathbf{a}^\top\mathbf{x})$ | $\exp(\mathbf{a}^\top\mathbf{x})\,\mathbf{a}$ | Special case of above with $g=\exp$ |
| $\log(\mathbf{a}^\top\mathbf{x})$ | $\mathbf{a}/(\mathbf{a}^\top\mathbf{x})$ | Special case with $g=\log$ |

---

## Appendix B: Gradient Descent Convergence for Smooth Convex Functions

We prove a basic convergence result establishing the $O(1/t)$ rate for gradient descent on smooth convex functions.

**Setting:** $f: \mathbb{R}^n \to \mathbb{R}$ is:
- **Convex:** $f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$ for all $\mathbf{x},\mathbf{y}$
- **$L$-smooth:** $\lVert\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\rVert \leq L\lVert\mathbf{x}-\mathbf{y}\rVert$ for all $\mathbf{x},\mathbf{y}$

**Gradient descent:** $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\nabla f(\mathbf{x}_t)$ with constant step $\eta = 1/L$.

**Lemma (descent lemma).** For $L$-smooth $f$ and step $\eta = 1/L$:

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t) - \frac{1}{2L}\lVert\nabla f(\mathbf{x}_t)\rVert^2$$

**Proof of lemma.** By $L$-smoothness and Taylor's theorem:

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^\top(\mathbf{x}_{t+1}-\mathbf{x}_t) + \frac{L}{2}\lVert\mathbf{x}_{t+1}-\mathbf{x}_t\rVert^2$$

Substituting $\mathbf{x}_{t+1} - \mathbf{x}_t = -\frac{1}{L}\nabla f(\mathbf{x}_t)$:

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t) - \frac{1}{L}\lVert\nabla f(\mathbf{x}_t)\rVert^2 + \frac{L}{2}\cdot\frac{1}{L^2}\lVert\nabla f(\mathbf{x}_t)\rVert^2 = f(\mathbf{x}_t) - \frac{1}{2L}\lVert\nabla f(\mathbf{x}_t)\rVert^2$$

$\square$

**Convergence proof.** Let $\mathbf{x}^*$ be a minimizer of $f$ with value $f^* = f(\mathbf{x}^*)$. By convexity:

$$f(\mathbf{x}_t) \leq f^* + \nabla f(\mathbf{x}_t)^\top(\mathbf{x}_t - \mathbf{x}^*)$$

Also:

$$\lVert\mathbf{x}_{t+1} - \mathbf{x}^*\rVert^2 = \lVert\mathbf{x}_t - \frac{1}{L}\nabla f(\mathbf{x}_t) - \mathbf{x}^*\rVert^2 = \lVert\mathbf{x}_t-\mathbf{x}^*\rVert^2 - \frac{2}{L}\nabla f(\mathbf{x}_t)^\top(\mathbf{x}_t-\mathbf{x}^*) + \frac{1}{L^2}\lVert\nabla f(\mathbf{x}_t)\rVert^2$$

Rearranging: $\nabla f(\mathbf{x}_t)^\top(\mathbf{x}_t-\mathbf{x}^*) = \frac{L}{2}(\lVert\mathbf{x}_t-\mathbf{x}^*\rVert^2 - \lVert\mathbf{x}_{t+1}-\mathbf{x}^*\rVert^2) + \frac{1}{2L}\lVert\nabla f(\mathbf{x}_t)\rVert^2$

Combining with the descent lemma and summing over $t = 0, \ldots, T-1$:

$$f(\mathbf{x}_T) - f^* \leq \frac{L\lVert\mathbf{x}_0-\mathbf{x}^*\rVert^2}{2T}$$

This establishes the $O(L/T)$ convergence rate. $\square$

**Interpretation:** After $T$ steps with step size $\eta = 1/L$, the excess loss decreases as $O(1/T)$. To achieve accuracy $\varepsilon$, we need $T = O(L/\varepsilon)$ steps. For strongly convex functions (Hessian lower-bounded by $\mu I$), the rate improves to $O((1-\mu/L)^T)$ — exponential convergence.

---

## Appendix C: Gradient of the Attention Mechanism

Computing the gradient of scaled dot-product attention is a fundamental exercise in multivariable calculus applied to transformers.

**Setup.** Let $Q, K, V \in \mathbb{R}^{n \times d}$ and define:

$$S = \frac{QK^\top}{\sqrt{d}} \in \mathbb{R}^{n \times n}, \qquad A = \operatorname{softmax}(S) \in \mathbb{R}^{n \times n}, \qquad O = AV \in \mathbb{R}^{n \times d}$$

where softmax is applied row-wise. The loss $\mathcal{L}$ is a scalar depending on $O$.

**Gradient w.r.t. $V$:**

$$\frac{\partial \mathcal{L}}{\partial V} = A^\top \frac{\partial \mathcal{L}}{\partial O}$$

Derivation: $O = AV$, so $O_{ij} = \sum_k A_{ik}V_{kj}$, giving $\frac{\partial \mathcal{L}}{\partial V_{kj}} = \sum_i \frac{\partial \mathcal{L}}{\partial O_{ij}} A_{ik} = (A^\top \frac{\partial \mathcal{L}}{\partial O})_{kj}$.

**Gradient w.r.t. $A$:**

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial O} V^\top$$

**Gradient w.r.t. $S$ (through softmax):**

Let $\mathbf{a}_i = \operatorname{softmax}(\mathbf{s}_i)$ be the $i$-th row of $A$. The Jacobian of softmax is $J_i = \operatorname{diag}(\mathbf{a}_i) - \mathbf{a}_i\mathbf{a}_i^\top$. Therefore:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{s}_i} = J_i^\top \frac{\partial \mathcal{L}}{\partial \mathbf{a}_i} = \mathbf{a}_i \odot \frac{\partial \mathcal{L}}{\partial \mathbf{a}_i} - \mathbf{a}_i \left(\mathbf{a}_i^\top \frac{\partial \mathcal{L}}{\partial \mathbf{a}_i}\right)$$

This is the "softmax gradient" formula used in FlashAttention (Dao et al., 2022) to avoid materializing the full $n \times n$ attention matrix.

**Gradient w.r.t. $Q$ and $K$:**

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial S} \cdot \frac{K}{\sqrt{d}}, \qquad \frac{\partial \mathcal{L}}{\partial K} = \left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top \cdot \frac{Q}{\sqrt{d}}$$

**For AI:** FlashAttention implements this gradient computation in a memory-efficient way by fusing the forward and backward passes and using tiling to avoid storing the full $n \times n$ attention matrix. The gradient formulas above are exactly what FlashAttention computes, just in a hardware-aware order.

---

## Appendix D: Coordinate Descent and Block Coordinate Descent

When computing the full gradient $\nabla f \in \mathbb{R}^n$ is expensive, **coordinate descent** updates one variable at a time:

$$x_i^{(t+1)} = \arg\min_{x_i} f(x_1^{(t)}, \ldots, x_{i-1}^{(t)}, x_i, x_{i+1}^{(t)}, \ldots, x_n^{(t)})$$

or equivalently, takes a gradient step only along the $i$-th coordinate:

$$x_i^{(t+1)} = x_i^{(t)} - \eta \frac{\partial f}{\partial x_i}(\mathbf{x}^{(t)})$$

**Convergence:** For smooth convex $f$, cyclic coordinate descent (cycling through $i = 1, \ldots, n$) converges at the same $O(1/t)$ rate as gradient descent — but each step costs $O(1)$ partial derivatives instead of $O(n)$.

**Block coordinate descent:** Update groups of variables simultaneously:

$$\mathbf{x}_{\mathcal{B}}^{(t+1)} = \arg\min_{\mathbf{x}_{\mathcal{B}}} f(\mathbf{x}_{\mathcal{B}}, \mathbf{x}_{\bar{\mathcal{B}}}^{(t)})$$

where $\mathcal{B}$ is a block of indices and $\bar{\mathcal{B}}$ is its complement.

**For AI (LoRA):** LoRA can be seen as block coordinate descent in parameter space. Instead of optimizing all $W \in \mathbb{R}^{d \times k}$, it freezes $W$ and optimizes the low-rank factors $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ (block $\mathcal{B} = \{A, B\}$). The gradient w.r.t. $A$ and $B$ are partial derivatives holding all other weights fixed.

---

## Appendix E: Numerical Stability in Gradient Computations

### E.1 Log-Sum-Exp and Softmax Gradients

Computing $\operatorname{softmax}(\mathbf{z})_k = e^{z_k} / \sum_j e^{z_j}$ naively overflows when $z_k$ is large.

**Stable implementation:** Subtract the maximum before exponentiating:

$$\operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k - z_{\max}}}{\sum_j e^{z_j - z_{\max}}}$$

This doesn't change the value (same numerator and denominator factor cancel) but keeps all exponentials $\leq 1$.

The gradient $\nabla_{\mathbf{z}} \mathcal{L} = \mathbf{p} - \mathbf{y}$ is inherently stable since $p_k \in [0,1]$.

### E.2 Gradient Vanishing Through Deep Networks

In a depth-$L$ feedforward network without residual connections, the gradient of the loss w.r.t. the first layer's weights passes through $L$ Jacobian matrices:

$$\nabla_{W^{[1]}} \mathcal{L} = \underbrace{J^{[L]} \cdot J^{[L-1]} \cdots J^{[2]}}_{\text{product of }L-1\text{ Jacobians}} \cdot \nabla_{W^{[1]}} \mathbf{z}^{[2]}$$

If each Jacobian has spectral norm $\sigma < 1$ (all singular values $< 1$), then $\lVert \nabla_{W^{[1]}} \mathcal{L} \rVert \leq \sigma^{L-1} \to 0$ exponentially. This is **vanishing gradients**.

**Fixes used in practice:**
- **Residual connections** (He et al., 2016): $\mathbf{x}^{[l+1]} = \mathbf{x}^{[l]} + F(\mathbf{x}^{[l]})$ — the identity path ensures gradient $\mathbf{I}$ passes through each block
- **LayerNorm** (Ba et al., 2016): Normalizes pre-activations, keeping Jacobian singular values near 1
- **Careful initialization** (He, Xavier/Glorot): Sets initial weight variance to maintain gradient magnitude across layers

### E.3 Mixed Precision and Gradient Scaling

In half-precision (fp16) training, gradients can underflow to zero (fp16 smallest positive $\approx 6 \times 10^{-8}$). **Gradient scaling** multiplies the loss by a large scale factor $s$ before backpropagation:

$$\nabla_{\boldsymbol{\theta}} (\underbrace{s \cdot \mathcal{L}}_{\text{scaled}}) = s \cdot \nabla_{\boldsymbol{\theta}} \mathcal{L}$$

After computing the scaled gradient, we divide by $s$ before the optimizer step. This keeps gradient values in the representable range of fp16. PyTorch's `torch.cuda.amp.GradScaler` manages $s$ automatically, halving it on overflow and increasing it gradually otherwise.

---

## Appendix F: Gradient Flow — Continuous-Time View

The gradient descent iteration $\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\nabla f(\boldsymbol{\theta}_t)$ can be viewed as the Euler discretization of the **gradient flow ODE**:

$$\frac{d\boldsymbol{\theta}}{dt} = -\nabla f(\boldsymbol{\theta}(t))$$

In continuous time, this is exact gradient descent — the parameter moves in the direction of steepest descent at each instant.

**Properties of gradient flow:**
1. **Energy decreases:** $\frac{d}{dt}f(\boldsymbol{\theta}(t)) = \nabla f^\top \frac{d\boldsymbol{\theta}}{dt} = -\lVert\nabla f\rVert^2 \leq 0$
2. **Fixed points:** Equilibria occur at $\nabla f = \mathbf{0}$ — critical points of $f$
3. **Convergence for convex $f$:** For $L$-smooth convex $f$: $f(\boldsymbol{\theta}(t)) - f^* \leq \frac{L\lVert\boldsymbol{\theta}(0) - \boldsymbol{\theta}^*\rVert^2}{2Lt} = \frac{\lVert\boldsymbol{\theta}(0)-\boldsymbol{\theta}^*\rVert^2}{2t}$

**For AI:** The Neural Tangent Kernel (NTK) theory (Jacot et al., 2018) analyzes infinitely wide networks by studying gradient flow in function space. The NTK $\Theta(\mathbf{x}, \mathbf{x}') = \nabla_{\boldsymbol{\theta}} f_{\boldsymbol{\theta}}(\mathbf{x})^\top \nabla_{\boldsymbol{\theta}} f_{\boldsymbol{\theta}}(\mathbf{x}')$ determines convergence speed and generalization in this regime. Understanding gradient flow is the entry point to NTK theory, which explains why overparameterized networks can generalize despite fitting the training data exactly.

---

## Appendix G: Historical Notes on Gradient Methods

**Augustin-Louis Cauchy (1847).** "Méthode générale pour la résolution des systèmes d'équations simultanées" — the first gradient descent paper. Cauchy showed that moving in the direction of steepest descent decreases a function and proposed an iterative scheme that converges to a local minimum. His motivation was solving systems of equations, not machine learning.

**Gradient descent lay dormant for nearly a century** as an optimization tool because numerical computation was impractical before computers. The method was rediscovered multiple times:

- **1944, Hestenes & Stiefel:** Conjugate gradient — stores memory of previous gradient directions to avoid zigzagging on elongated loss surfaces. Remains the gold standard for solving large linear systems.
- **1951, Robbins & Monro:** Stochastic approximation — convergence theory for gradient descent with noisy observations. The theoretical foundation for SGD.
- **1986, Rumelhart, Hinton, Williams:** "Learning representations by back-propagating errors" — showed that gradient descent on layered networks via backpropagation trains useful representations. This paper launched the first neural network renaissance.
- **2012–present:** Deep learning era — GPUs make gradient descent tractable for billions of parameters. Adam (Kingma & Ba, 2015) adapts the learning rate per-parameter. Gradient clipping (Pascanu et al., 2013) stabilizes RNN/transformer training. AdamW (Loshchilov & Hutter, 2019) decouples weight decay from adaptive learning rates.

The gradient is the same mathematical object Cauchy defined in 1847 — what changed is our ability to compute it (via backpropagation and automatic differentiation) and the scale at which we apply it (billions of parameters, trillions of tokens).

---

## Appendix H: Worked Examples — Gradient Computations in Full Detail

### H.1 Gradient of Logistic Regression Loss (Full Derivation)

**Setup:** $m$ binary examples $\{(\mathbf{x}^{(i)}, y^{(i)})\}$ with $y^{(i)} \in \{0,1\}$. Model: $p^{(i)} = \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})$ where $\sigma(z) = (1+e^{-z})^{-1}$.

**Loss:** $\mathcal{L}(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\log p^{(i)} + (1-y^{(i)})\log(1-p^{(i)})\right]$

**Step 1:** Gradient of a single term. Fix $i$; let $z^{(i)} = \mathbf{w}^\top\mathbf{x}^{(i)}$.

$$\ell^{(i)} = -y^{(i)}\log\sigma(z^{(i)}) - (1-y^{(i)})\log(1-\sigma(z^{(i)}))$$

$$\frac{\partial \ell^{(i)}}{\partial z^{(i)}} = -\frac{y^{(i)}}{\sigma(z^{(i)})} \cdot \sigma'(z^{(i)}) + \frac{1-y^{(i)}}{1-\sigma(z^{(i)})} \cdot \sigma'(z^{(i)})$$

Using $\sigma'(z) = \sigma(z)(1-\sigma(z))$:

$$= -y^{(i)}(1-\sigma(z^{(i)})) + (1-y^{(i)})\sigma(z^{(i)}) = \sigma(z^{(i)}) - y^{(i)} = p^{(i)} - y^{(i)}$$

**Step 2:** Gradient w.r.t. $\mathbf{w}$ via chain rule.

$$\frac{\partial \ell^{(i)}}{\partial \mathbf{w}} = \frac{\partial \ell^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial \mathbf{w}} = (p^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)}$$

**Step 3:** Average over examples.

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{m}\sum_{i=1}^m (p^{(i)} - y^{(i)}) \mathbf{x}^{(i)} = \frac{1}{m}X^\top(\mathbf{p} - \mathbf{y})$$

**Verification:** At the optimum $\nabla_{\mathbf{w}}\mathcal{L} = \mathbf{0}$, so $X^\top\mathbf{p} = X^\top\mathbf{y}$ — the predicted probabilities are "feature-uncorrelated" with the residuals.

### H.2 Gradient of the Negative Log-Likelihood for a Gaussian

**Setup:** $n$ iid observations $x_1,\ldots,x_n \sim \mathcal{N}(\mu, \sigma^2)$. Parameters: $\boldsymbol{\theta} = (\mu, \sigma)$.

**Negative log-likelihood:**

$$\mathcal{L}(\mu,\sigma) = \frac{n}{2}\log(2\pi) + n\log\sigma + \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**Partial derivatives:**

$$\frac{\partial \mathcal{L}}{\partial \mu} = -\frac{1}{\sigma^2}\sum_{i=1}^n(x_i-\mu) = \frac{n(\mu - \bar{x})}{\sigma^2}$$

$$\frac{\partial \mathcal{L}}{\partial \sigma} = \frac{n}{\sigma} - \frac{1}{\sigma^3}\sum_{i=1}^n(x_i-\mu)^2$$

**Setting to zero (MLE):**

From $\frac{\partial \mathcal{L}}{\partial \mu} = 0$: $\hat{\mu} = \bar{x} = \frac{1}{n}\sum_i x_i$ (sample mean).

From $\frac{\partial \mathcal{L}}{\partial \sigma} = 0$: $\hat{\sigma}^2 = \frac{1}{n}\sum_i(x_i-\bar{x})^2$ (biased sample variance).

**For AI:** Maximum likelihood estimation of neural network parameters is gradient descent on the negative log-likelihood $\mathcal{L}(\boldsymbol{\theta}) = -\sum_i \log p_{\boldsymbol{\theta}}(y^{(i)}\mid\mathbf{x}^{(i)})$ — identical structure to the Gaussian example, but with a deep network defining $p_{\boldsymbol{\theta}}$.

### H.3 Gradient of the Frobenius Norm Loss

**Setup:** Matrix regression — minimize $\mathcal{L}(W) = \frac{1}{2}\lVert XW - Y\rVert_F^2$ where $X \in \mathbb{R}^{m\times d}$, $W \in \mathbb{R}^{d\times k}$, $Y \in \mathbb{R}^{m\times k}$.

**Gradient derivation:**

$$\mathcal{L}(W) = \frac{1}{2}\operatorname{tr}((XW-Y)^\top(XW-Y))$$

$$\nabla_W \mathcal{L} = X^\top(XW - Y)$$

**Proof:** Let $R = XW - Y$. Then $\mathcal{L} = \frac{1}{2}\operatorname{tr}(R^\top R)$. A perturbation $\delta W$ gives:

$$\delta \mathcal{L} = \frac{1}{2}\operatorname{tr}((R + X\delta W)^\top(R + X\delta W)) - \frac{1}{2}\operatorname{tr}(R^\top R)$$
$$= \operatorname{tr}((X\delta W)^\top R) + O(\lVert\delta W\rVert^2) = \operatorname{tr}(\delta W^\top X^\top R) = \langle X^\top R, \delta W \rangle_F$$

So $\nabla_W \mathcal{L} = X^\top R = X^\top(XW-Y)$.

This is the matrix analog of the vector gradient $2X^\top(X\mathbf{w}-\mathbf{y})$ (without the factor of 2 because of the $\frac{1}{2}$). Applied to transformer attention: optimizing the value projection $W_V$ uses this exact gradient structure.

---

## Appendix I: Gradient Interpretation — A Physics Analogy

The mathematical definition of the gradient acquires deeper meaning through physical analogies.

**Temperature field:** Consider $T(\mathbf{x})$ as the temperature at point $\mathbf{x}$ in a room. The gradient $\nabla T(\mathbf{x})$ points in the direction of fastest temperature increase. Heat flows in direction $-\nabla T$ (Fourier's law): $\mathbf{q} = -k\nabla T$, where $\mathbf{q}$ is the heat flux vector and $k$ is thermal conductivity.

**Gradient descent = steepest descent on a potential:**

- **Elevation field:** $f(\mathbf{x})$ = altitude at position $\mathbf{x}$. $-\nabla f$ = direction of steepest downhill. Water flows in direction $-\nabla f$ (gradient flow).
- **Electric potential:** $\phi(\mathbf{x})$ = electric potential. $\mathbf{E} = -\nabla\phi$ = electric field (force per unit charge). Charges accelerate in direction $-\nabla\phi$.
- **Loss surface:** $\mathcal{L}(\boldsymbol{\theta})$ = loss at parameters $\boldsymbol{\theta}$. $-\nabla\mathcal{L}$ = direction of maximum loss decrease. Gradient descent moves parameters in direction $-\nabla\mathcal{L}$.

**The physics insight:** In all three cases, the gradient defines a **vector field** (a vector at every point in space), and dynamics follow the negative gradient. This is the fundamental structure of steepest-descent optimization — borrowed from physics, applied to machine learning.

**Potential energy landscape:** The loss $\mathcal{L}(\boldsymbol{\theta})$ plays the role of potential energy. Parameters play the role of a particle. Gradient descent is a particle rolling downhill with heavy friction (no momentum). SGD adds noise — like a particle in a bath of random thermal kicks (Langevin dynamics). Adam is like an adaptive friction coefficient that varies per dimension.

---

## Appendix J: Reading Guide and Further References

### Core References

1. **Nocedal & Wright, *Numerical Optimization* (2006), Chapters 1–2** — rigorous treatment of gradient descent, step size rules, convergence for smooth functions. The standard reference for optimization in ML.

2. **Boyd & Vandenberghe, *Convex Optimization* (2004), Chapter 9** — gradient descent for convex functions, convergence rates, extensions to non-smooth functions. Freely available online.

3. **Goodfellow, Bengio & Courville, *Deep Learning* (2016), Chapter 4** — matrix calculus for ML practitioners. Gradient and Jacobian conventions used throughout this repo follow GBC.

4. **Strang, *Introduction to Linear Algebra* (2016), Chapter 8** — accessible introduction to gradients and their geometric meaning, from a linear algebra perspective.

5. **Amari, *Information Geometry and Its Applications* (2016)** — the mathematical foundation of natural gradient methods, Fisher information, and gradient flows on statistical manifolds.

### For AI/ML Practitioners

6. **Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2015)** — introduces the Adam optimizer; Section 2 discusses the moment estimators in terms of the gradient.

7. **Pascanu, Mikolov & Bengio, "On the difficulty of training recurrent neural networks" (2013)** — introduces gradient clipping; explains exploding/vanishing gradients mathematically.

8. **Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2022)** — the key idea is restricting gradient updates to a low-rank subspace.

9. **Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)** — implements attention gradients in a hardware-aware fashion; the gradient formulas in Appendix C are what FlashAttention computes.

10. **Foret et al., "Sharpness-Aware Minimization for Efficiently Improving Generalization" (2021)** — uses gradient norm and loss landscape geometry to find flat minima; directly applies the concepts in §8.

### Online Resources

- **Andrej Karpathy's micrograd** — a 150-line automatic differentiation engine built from scratch; excellent for understanding how gradients are computed
- **Matrix Cookbook** — comprehensive reference for matrix calculus identities (freely available as PDF)
- **Chris Olah's "Neural Networks, Manifolds, and Topology"** — visual intuition for how gradients relate to neural network geometry

---

## Appendix K: Stochastic Gradient Descent — Theory and Practice

### K.1 Why Stochastic Gradients Work

Full gradient descent computes $\nabla_{\boldsymbol{\theta}}\mathcal{L} = \frac{1}{m}\sum_{i=1}^m \nabla_{\boldsymbol{\theta}}\ell^{(i)}$, which is expensive for large $m$. SGD replaces this with an unbiased estimate using a single randomly sampled example $i$:

$$\mathbf{g}_t = \nabla_{\boldsymbol{\theta}}\ell^{(i_t)}, \qquad \mathbb{E}[\mathbf{g}_t \mid \boldsymbol{\theta}_t] = \nabla_{\boldsymbol{\theta}}\mathcal{L}(\boldsymbol{\theta}_t)$$

**Why it converges:** The Robbins-Monro conditions ensure convergence for non-convex functions under SGD:
$$\sum_{t=1}^\infty \eta_t = \infty, \qquad \sum_{t=1}^\infty \eta_t^2 < \infty$$

A schedule satisfying these: $\eta_t = \eta_0 / t$ (harmonic decay). In practice, cosine decay or piecewise constant schedules are used.

**Variance of the stochastic gradient:**

$$\operatorname{Var}(\mathbf{g}_t) = \mathbb{E}[\lVert\mathbf{g}_t - \nabla\mathcal{L}\rVert^2]$$

With mini-batch size $B$: $\operatorname{Var}(\bar{\mathbf{g}}) = \operatorname{Var}(\mathbf{g}_t)/B$ — variance decreases as $1/B$. This is why larger batches have smaller gradient noise but don't always converge better (the noise helps escape sharp minima).

### K.2 Gradient Noise as Regularization

The noise in SGD gradients acts as an implicit regularizer. Formally:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta[\nabla\mathcal{L}(\boldsymbol{\theta}_t) + \boldsymbol{\epsilon}_t]$$

where $\boldsymbol{\epsilon}_t = \mathbf{g}_t - \nabla\mathcal{L}(\boldsymbol{\theta}_t)$ is the gradient noise. This is equivalent to gradient flow with a stochastic forcing term — a Langevin dynamics:

$$d\boldsymbol{\theta} = -\nabla\mathcal{L}(\boldsymbol{\theta})\,dt + \sqrt{2\eta}\,d\mathbf{W}_t$$

where $\mathbf{W}_t$ is Brownian motion. The stationary distribution of this SDE is approximately $p(\boldsymbol{\theta}) \propto \exp(-\mathcal{L}(\boldsymbol{\theta})/\eta)$ — a Gibbs distribution at "temperature" $\eta$. Lower learning rate = lower temperature = sharper distribution concentrated near minima.

**Practical implication:** SGD with large learning rate explores the loss landscape (high temperature), while small learning rate exploits (low temperature). Warmup schedules start with a small learning rate (before gradients are reliable) then increase to full rate, then decay.

### K.3 Mini-Batch Gradient and Gradient Variance Reduction

| Method | Gradient estimate | Variance | Cost per step |
|---|---|---|---|
| Full batch | $\frac{1}{m}\sum_i \nabla\ell^{(i)}$ | 0 | $O(m)$ |
| SGD | $\nabla\ell^{(i)}$ | $\sigma^2$ | $O(1)$ |
| Mini-batch | $\frac{1}{B}\sum_{i\in\mathcal{B}}\nabla\ell^{(i)}$ | $\sigma^2/B$ | $O(B)$ |
| SVRG | $\nabla\ell^{(i)} - \nabla\ell^{(i)}_\text{anchor} + \bar{\mathbf{g}}$ | $O(\text{dist from anchor})$ | $O(1)+$ anchor cost |

**For LLMs:** Gradient noise in transformer training is beneficial — it helps the model escape sharp minima and generalize better (Jiang et al., 2020). The typical LLM training batch (1M–4M tokens) is large enough to reduce gradient variance substantially while retaining some beneficial noise.

---

## Appendix L: Gradient-Based Saliency and Interpretability

### L.1 Gradient Saliency Maps

The simplest form of neural network interpretability uses the gradient of the output w.r.t. the input:

$$S_{ij} = \left|\frac{\partial \hat{y}}{\partial x_{ij}}\right|$$

where $x_{ij}$ is the $j$-th feature of the $i$-th input token (or pixel). Large $|S_{ij}|$ means small changes in $x_{ij}$ cause large changes in $\hat{y}$ — the model is "sensitive" to that feature.

**For language models:** $\frac{\partial \log p(y\mid\mathbf{x})}{\partial x_i}$ measures how much token $i$'s embedding affects the predicted probability. This gives a token-level attribution — which words most influence the model's prediction.

**Limitations:**
- Gradient saliency is **local** (valid only near the current input)
- Can be **adversarially fooled** — the gradient can be large for imperceptible perturbations
- Does not account for **saturation** (sigmoid/tanh can saturate, making gradients near zero even for important features)

### L.2 Integrated Gradients (Sundararajan et al., 2017)

To fix saturation issues, integrate the gradient along a path from a baseline $\mathbf{x}'$ to the input $\mathbf{x}$:

$$\phi_i(\mathbf{x}) = (x_i - x'_i) \int_0^1 \frac{\partial f(\mathbf{x}' + \alpha(\mathbf{x}-\mathbf{x}'))}{\partial x_i}\,d\alpha$$

This satisfies the **completeness axiom**: $\sum_i \phi_i(\mathbf{x}) = f(\mathbf{x}) - f(\mathbf{x}')$ (the attributions sum to the prediction difference from the baseline).

**Implementation:** Approximate the integral with a Riemann sum over $k$ steps. In practice, $k = 50$ interpolations between $\mathbf{x}'$ and $\mathbf{x}$, computing gradients at each.

**For AI:** Integrated gradients is the standard attribution method for language models (used in LIME, SHAP extensions, and many interpretability tools). Understanding that it's an integral of partial derivatives connects it directly to the material in this section.

---

## Appendix M: Exercises with Full Solutions

### M.1 Worked Solution — Exercise 2(c)

**Problem:** Verify numerically that $\nabla f(1,-1)$ is perpendicular to the level curve of $f(x,y) = x^2+xy+y^2$ through $(1,-1)$.

**Solution:**

Step 1: Compute $f(1,-1) = 1 - 1 + 1 = 1$. The level curve is $\{x^2+xy+y^2 = 1\}$.

Step 2: Gradient at $(1,-1)$: $\nabla f = (2x+y, x+2y)^\top = (2(1)+(-1), 1+2(-1))^\top = (1, -1)^\top$.

Step 3: Parametrize the level curve near $(1,-1)$. By implicit differentiation: $\frac{dy}{dx} = -\frac{f_x}{f_y} = -\frac{1}{-1} = 1$. So the tangent direction is $(1, 1)^\top$ (normalized: $(1/\sqrt{2}, 1/\sqrt{2})^\top$).

Step 4: Check perpendicularity: $(1, -1) \cdot (1, 1) = 1 - 1 = 0$. Perpendicular. $\checkmark$

**Numerical verification:** Sample points on the level curve near $(1,-1)$: try $(1 + \varepsilon, -1 + \varepsilon)$ for small $\varepsilon$. We have $f(1+\varepsilon, -1+\varepsilon) = (1+\varepsilon)^2 + (1+\varepsilon)(-1+\varepsilon) + (-1+\varepsilon)^2 = 1 + 2\varepsilon + \varepsilon^2 + (-1 + \varepsilon^2) + 1 - 2\varepsilon + \varepsilon^2 = 1 + 3\varepsilon^2 \approx 1$ for small $\varepsilon$. This confirms the tangent direction $(1,1)^\top$ lies along the level curve.

### M.2 Worked Solution — Key MSE Formula

**Problem:** Derive $\nabla_{\mathbf{w}} \frac{1}{2m}\lVert X\mathbf{w} - \mathbf{y}\rVert^2$ step by step.

**Solution:**

$$\mathcal{L}(\mathbf{w}) = \frac{1}{2m}(X\mathbf{w}-\mathbf{y})^\top(X\mathbf{w}-\mathbf{y}) = \frac{1}{2m}\sum_{i=1}^m\left(\sum_j X_{ij}w_j - y_i\right)^2$$

**Component-wise:** The $k$-th component of $\nabla_{\mathbf{w}}\mathcal{L}$:

$$\frac{\partial\mathcal{L}}{\partial w_k} = \frac{1}{m}\sum_{i=1}^m\left(\sum_j X_{ij}w_j - y_i\right) \cdot X_{ik} = \frac{1}{m}\sum_{i=1}^m X_{ik}(X\mathbf{w}-\mathbf{y})_i$$

In matrix form: $\frac{\partial\mathcal{L}}{\partial w_k} = \frac{1}{m}[X^\top(X\mathbf{w}-\mathbf{y})]_k$, so $\nabla_{\mathbf{w}}\mathcal{L} = \frac{1}{m}X^\top(X\mathbf{w}-\mathbf{y})$.

**Setting to zero:** $X^\top(X\mathbf{w}-\mathbf{y}) = \mathbf{0} \Rightarrow X^\top X\mathbf{w} = X^\top\mathbf{y}$ — the normal equations. When $X^\top X$ is invertible: $\hat{\mathbf{w}} = (X^\top X)^{-1}X^\top\mathbf{y}$.

---

## Appendix N: Connection Map — Gradient to Modern AI

```
GRADIENT ∇f: CONNECTIONS TO MODERN AI SYSTEMS (2026)
════════════════════════════════════════════════════════════════════════

  FOUNDATIONAL                    ML ALGORITHMS
  ──────────                      ─────────────
  ∂f/∂xᵢ (partial deriv)  ───►   Backpropagation (§03)
  ∇f ∈ ℝⁿ (gradient)      ───►   Gradient descent / Adam / AdamW
  D_u f (directional)     ───►   Projected gradient descent
  ∇f ⊥ level set          ───►   Orthogonal gradient descent (continual learning)
  Linear approx f(a+δ)    ───►   L-smoothness condition (convergence proofs)
  ‖∇f‖ = 0 at extremum    ───►   Optimality conditions → KKT (§04)

  SPECIFIC ML SYSTEMS
  ───────────────────
  ∇_w(cross-entropy)      ───►   p - y: softmax output gradient (every LLM)
  ∇_w ‖Xw-y‖²            ───►   Linear regression, LoRA initialization
  Gradient clipping       ───►   GPT-3, Llama, Mistral training stability
  Gradient accumulation   ───►   LLM pretraining (effective batch = millions of tokens)
  Fisher F = E[∇logp·∇logpᵀ]───► Natural gradient, K-FAC, second-order methods
  ∂L/∂input (saliency)    ───►   Mechanistic interpretability, activation patching
  Integrated gradients    ───►   LIME, SHAP, LLM attribution
  Functional gradient     ───►   Neural Tangent Kernel, Wasserstein gradient flows

════════════════════════════════════════════════════════════════════════
```

---

## Appendix O: Multivariate Calculus Identities Quick Reference

### O.1 Rules for Computing Partial Derivatives

| Rule | Formula | Notes |
|---|---|---|
| **Sum rule** | $\frac{\partial(f+g)}{\partial x_i} = \frac{\partial f}{\partial x_i} + \frac{\partial g}{\partial x_i}$ | Always valid |
| **Product rule** | $\frac{\partial(fg)}{\partial x_i} = \frac{\partial f}{\partial x_i}g + f\frac{\partial g}{\partial x_i}$ | Both factors can depend on $x_i$ |
| **Quotient rule** | $\frac{\partial(f/g)}{\partial x_i} = \frac{\frac{\partial f}{\partial x_i}g - f\frac{\partial g}{\partial x_i}}{g^2}$ | $g \neq 0$ required |
| **Chain rule** | $\frac{\partial}{\partial x_i}[h(f(\mathbf{x}))] = h'(f(\mathbf{x}))\frac{\partial f}{\partial x_i}$ | Composite function |
| **Mixed chain rule** | $\frac{\partial}{\partial x_i}[f(g(\mathbf{x}),h(\mathbf{x}))] = \frac{\partial f}{\partial u}\frac{\partial g}{\partial x_i} + \frac{\partial f}{\partial v}\frac{\partial h}{\partial x_i}$ | $u=g(\mathbf{x})$, $v=h(\mathbf{x})$ |

### O.2 Gradient Operator Properties

The gradient operator $\nabla$ (applied to scalar fields) is **linear**:

$$\nabla(af + bg) = a\nabla f + b\nabla g$$

**Product rule for gradients:**

$$\nabla(fg) = g\nabla f + f\nabla g$$

**Chain rule for gradients:**

$$\nabla h(f(\mathbf{x})) = h'(f(\mathbf{x}))\nabla f(\mathbf{x})$$

**For composed functions $f: \mathbb{R}^n \to \mathbb{R}$, $g: \mathbb{R}^m \to \mathbb{R}^n$:**

$$\nabla_{\mathbf{z}}[f(g(\mathbf{z}))] = J_g(\mathbf{z})^\top \nabla_{\mathbf{x}} f(g(\mathbf{z}))$$

This is the full multivariate chain rule — proved in §03.

### O.3 Gradient in Different Coordinate Systems

While we always work in Cartesian coordinates in this section, the gradient takes different forms in other systems:

**Polar coordinates** $(r, \theta)$:

$$\nabla f = \frac{\partial f}{\partial r}\hat{\mathbf{r}} + \frac{1}{r}\frac{\partial f}{\partial \theta}\hat{\boldsymbol{\theta}}$$

**Spherical coordinates** $(r, \theta, \phi)$:

$$\nabla f = \frac{\partial f}{\partial r}\hat{\mathbf{r}} + \frac{1}{r}\frac{\partial f}{\partial \theta}\hat{\boldsymbol{\theta}} + \frac{1}{r\sin\theta}\frac{\partial f}{\partial \phi}\hat{\boldsymbol{\phi}}$$

**For ML:** Riemannian optimization on manifolds (e.g., Stiefel manifold for orthogonal weight matrices, SPD manifold for covariance matrices) uses a metric-aware gradient — the Riemannian gradient — which is the projection of the Euclidean gradient onto the tangent space of the manifold.

---

## Appendix P: Common Partial Derivative Computations in ML Practice

### P.1 Attention Score Gradient (Simplified)

For a single query-key pair, the attention logit is $s = \mathbf{q}^\top\mathbf{k}/\sqrt{d}$. Gradients:

$$\frac{\partial s}{\partial \mathbf{q}} = \frac{\mathbf{k}}{\sqrt{d}}, \qquad \frac{\partial s}{\partial \mathbf{k}} = \frac{\mathbf{q}}{\sqrt{d}}$$

These are the update signals that drive query and key projections to produce higher/lower attention scores.

### P.2 Layer Normalization Gradient

LayerNorm normalizes: $\hat{\mathbf{x}} = (\mathbf{x} - \mu\mathbf{1})/\sigma$, then scales and shifts: $\mathbf{y} = \boldsymbol{\gamma}\odot\hat{\mathbf{x}} + \boldsymbol{\beta}$.

Gradients w.r.t. the learnable parameters $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$:

$$\frac{\partial\mathcal{L}}{\partial\boldsymbol{\gamma}} = \frac{\partial\mathcal{L}}{\partial\mathbf{y}}\odot\hat{\mathbf{x}}, \qquad \frac{\partial\mathcal{L}}{\partial\boldsymbol{\beta}} = \frac{\partial\mathcal{L}}{\partial\mathbf{y}}$$

The gradient w.r.t. $\mathbf{x}$ (for backpropagation through LayerNorm) involves the partial derivatives of $\hat{\mathbf{x}}$ w.r.t. $\mathbf{x}$, which requires careful treatment because $\mu$ and $\sigma$ both depend on $\mathbf{x}$ — this is a full multivariate chain rule computation covered in §03.

### P.3 Embedding Table Gradient

For a language model with vocabulary size $V$ and embedding matrix $E \in \mathbb{R}^{V\times d}$:

The gradient $\frac{\partial\mathcal{L}}{\partial E}$ is sparse — only the rows corresponding to tokens that appeared in the batch have nonzero gradients. Specifically:

$$\left(\frac{\partial\mathcal{L}}{\partial E}\right)_{v,:} = \sum_{t : x_t = v} \frac{\partial\mathcal{L}}{\partial\mathbf{e}_{x_t}}$$

where $\mathbf{e}_{x_t} \in \mathbb{R}^d$ is the embedding of token $x_t$. This sparsity is exploited by embedding optimizers that only update rows corresponding to tokens in the current batch.

### P.4 Weight Tying Gradient

When the input embedding $E$ and output projection $W_\text{out} = E^\top$ share the same weight matrix (common in transformer language models), the gradient of $\mathcal{L}$ w.r.t. $E$ receives contributions from both uses:

$$\frac{\partial\mathcal{L}}{\partial E} = \frac{\partial\mathcal{L}}{\partial E_\text{input}} + \left(\frac{\partial\mathcal{L}}{\partial W_\text{out}}\right)^\top$$

This is the weight-sharing gradient accumulation rule (§6.4) applied to the input-output embedding tying used in GPT-2 and many transformer models.

---

## Appendix Q: Visualization Gallery — Gradient Fields

The following descriptions correspond to plots generated in `theory.ipynb`.

**Plot 1 — Gradient field of a paraboloid:** $f(x,y) = x^2 + y^2$. Arrows at a grid of points show $\nabla f = (2x, 2y)^\top$. All arrows point radially outward from the minimum at the origin, with arrow length proportional to $\lVert\nabla f\rVert = 2\sqrt{x^2+y^2}$.

**Plot 2 — Gradient and level curves:** Contour lines of $f$ overlaid with gradient arrows. Visually confirms $\nabla f \perp$ level curves at every point.

**Plot 3 — Saddle point gradient field:** $f(x,y) = x^2 - y^2$. Gradient $\nabla f = (2x, -2y)^\top$ points outward in $x$ and inward in $y$. At the origin, $\nabla f = \mathbf{0}$ — a critical point that is neither a local min nor max.

**Plot 4 — Loss landscape and gradient descent trajectory:** A non-convex 2D function (e.g., Beale function or Rosenbrock) with gradient descent path from several starting points. Shows convergence to different minima depending on initialization.

**Plot 5 — Gradient norm as a function of training step:** From a simple neural network training run, shows the gradient norm $\lVert\nabla_{\boldsymbol{\theta}}\mathcal{L}\rVert$ decreasing as training progresses, with occasional spikes.

---

## Appendix R: Self-Assessment Checklist

After studying this section, verify you can:

**Mechanics:**
- [ ] Compute partial derivatives of polynomial, exponential, and logarithmic functions
- [ ] Write down the gradient vector $\nabla f$ for a given function
- [ ] Compute directional derivatives using the formula $D_{\mathbf{u}}f = \nabla f^\top\mathbf{u}$
- [ ] Find the directions of maximum/minimum directional derivative

**Theory:**
- [ ] Prove that the gradient points in the direction of steepest ascent (from Cauchy-Schwarz)
- [ ] Prove that the gradient is perpendicular to level sets (from the curve argument)
- [ ] Explain why partial differentiability does not imply differentiability (with a counterexample)
- [ ] State and apply Clairaut's theorem on equality of mixed partials

**Applications:**
- [ ] Derive the gradient of the MSE loss $\frac{1}{m}\lVert X\mathbf{w}-\mathbf{y}\rVert^2$
- [ ] Derive the gradient of the cross-entropy loss for logistic regression
- [ ] Implement gradient descent for a simple ML problem
- [ ] Implement gradient checking using centered differences
- [ ] Explain gradient clipping, gradient accumulation, and their role in LLM training

**Advanced:**
- [ ] State what the Fisher information matrix is and how it relates to the natural gradient
- [ ] Explain why gradient descent converges at rate $O(1/t)$ for $L$-smooth convex functions
- [ ] Describe how LoRA restricts gradient updates to a low-rank subspace

---

## Appendix S: Notation Summary for This Section

| Symbol | Meaning | Convention |
|---|---|---|
| $f: \mathbb{R}^n \to \mathbb{R}$ | Scalar field (loss function) | Lowercase $f$, uppercase $\mathcal{L}$ for loss |
| $\mathbf{x} \in \mathbb{R}^n$ | Input vector (column) | Bold lowercase |
| $\frac{\partial f}{\partial x_i}$ | Partial derivative w.r.t. $x_i$ | $\partial$ (curly d), not $d$ |
| $\nabla_{\mathbf{x}} f \in \mathbb{R}^n$ | Gradient (column vector) | Bold $\nabla$ with subscript; result is column |
| $\nabla_{\boldsymbol{\theta}}\mathcal{L}$ | Gradient of loss w.r.t. parameters | Always column; rows = parameter dimensions |
| $D_{\mathbf{u}} f$ | Directional derivative in direction $\mathbf{u}$ | Requires $\lVert\mathbf{u}\rVert = 1$ |
| $H_f \in \mathbb{R}^{n\times n}$ | Hessian (second-order partial derivatives) | Symmetric for $C^2$ functions; full treatment in §02 |
| $J_{\mathbf{f}} \in \mathbb{R}^{m\times n}$ | Jacobian (gradient of vector-valued $\mathbf{f}$) | Full treatment in §02 |
| $L_c(f)$ | Level set $\{f = c\}$ | $(n-1)$-dimensional hypersurface |
| $\eta$ | Learning rate | Always positive; $\eta \leq 1/L$ for guaranteed descent |
| $\boldsymbol{\theta}$ | All model parameters (bold, lowercase) | Follows `NOTATION_GUIDE.md` §7 |
| $\mathcal{L}(\boldsymbol{\theta})$ | Loss function | Calligraphic $\mathcal{L}$ reserved for loss |
| $F(\boldsymbol{\theta})$ | Fisher information matrix | $n \times n$ PSD matrix; used in natural gradient |

---

## Appendix T: Extended Worked Examples for ML Practitioners

### T.1 Gradient Tape Mental Model

Before automatic differentiation (§05), practitioners had to derive gradients by hand. The "gradient tape" mental model — tracking how each computation depends on parameters — is still valuable for debugging.

**Example: One-layer network forward pass.**

Given $\mathbf{x} \in \mathbb{R}^d$, $W \in \mathbb{R}^{k\times d}$, $\mathbf{b} \in \mathbb{R}^k$, $\mathbf{y}_\text{true} \in \mathbb{R}^k$:

```
Forward:
  z = W·x + b        (pre-activation, shape k)
  a = σ(z)           (post-activation, shape k)
  L = ½‖a - y_true‖²  (MSE loss, scalar)

Backward (right to left):
  ∂L/∂a = a - y_true                            (shape k)
  ∂L/∂z = (∂L/∂a) ⊙ σ'(z) = (a - y_true) ⊙ σ'(z)   (elementwise, shape k)
  ∂L/∂W = (∂L/∂z) · x^T                         (outer product, shape k×d)
  ∂L/∂b = ∂L/∂z                                 (shape k)
  ∂L/∂x = W^T · (∂L/∂z)                         (shape d, for next layer)
```

Every line is a gradient formula from this section:
- $\partial L/\partial \mathbf{a} = \mathbf{a} - \mathbf{y}_\text{true}$ — gradient of MSE (§6.1)
- The $\odot\sigma'(\mathbf{z})$ step — chain rule, elementwise (§3.2, §5.2)
- $\partial L/\partial W = (\partial L/\partial\mathbf{z})\mathbf{x}^\top$ — gradient of a bilinear form (§4.5)

### T.2 Softmax Temperature Gradient

In language model decoding, the temperature-scaled softmax is:

$$p_k = \frac{e^{z_k/\tau}}{\sum_j e^{z_j/\tau}}$$

**Gradient w.r.t. $\tau$:**

Let $s_k = z_k/\tau$. By the chain rule:

$$\frac{\partial p_k}{\partial \tau} = \frac{\partial p_k}{\partial s_k} \cdot \frac{\partial s_k}{\partial \tau} + \sum_{j\neq k} \frac{\partial p_k}{\partial s_j} \cdot \frac{\partial s_j}{\partial \tau}$$

$$= \sum_j \frac{\partial p_k}{\partial s_j} \cdot \left(-\frac{z_j}{\tau^2}\right) = -\frac{1}{\tau^2}\sum_j \frac{\partial p_k}{\partial s_j} z_j$$

Using the softmax Jacobian $\frac{\partial p_k}{\partial s_j} = p_k(\delta_{kj} - p_j)$:

$$\frac{\partial p_k}{\partial \tau} = -\frac{1}{\tau^2}\left[p_k z_k - p_k\sum_j p_j z_j\right] = -\frac{p_k}{\tau^2}\left[z_k - \mathbb{E}_p[z]\right]$$

**Interpretation:** Temperature gradient is negative when $z_k > \mathbb{E}_p[z]$ (the chosen token has above-average logit). Increasing $\tau$ flattens the distribution; decreasing $\tau$ sharpens it — this is reflected in the gradient sign.

### T.3 Gradient w.r.t. Embedding Lookup

For input tokens $x_1, \ldots, x_T$ (indices), the embedding layer looks up $\mathbf{e}_{x_t} = E_{x_t, :}$ (the $x_t$-th row of the embedding matrix $E \in \mathbb{R}^{V\times d}$).

For a loss $\mathcal{L}$ that depends on the embeddings $\{\mathbf{e}_{x_t}\}$, the gradient w.r.t. the embedding matrix has the sparsity structure:

$$\frac{\partial\mathcal{L}}{\partial E_{v,:}} = \sum_{t: x_t = v} \frac{\partial\mathcal{L}}{\partial \mathbf{e}_{x_t}}$$

Only rows $v$ that appear in the batch receive gradient updates. For vocabulary size $V = 128{,}000$ (Llama 3) and a batch with $B = 4096$ tokens, fewer than $4096$ out of $128{,}000$ rows have nonzero gradients. Sparse embedding optimizers (Adagrad, sparse Adam) exploit this structure for efficiency.

---

## Appendix U: Connections to Other Branches of Mathematics

### U.1 Gradient and Topology (Morse Theory)

Morse theory studies how the topology of a manifold changes as we sweep through level sets $\{f \leq c\}$. The topology changes only at **critical points** where $\nabla f = \mathbf{0}$. The number and type (index = number of negative Hessian eigenvalues) of critical points constrain the global topology.

**For ML:** The "loss landscape topology" program (Goodfellow et al., 2015; Li et al., 2018) studies how gradient descent navigates the network of critical points. The key result: most local minima of large overparameterized networks have comparable loss to the global minimum (high-dimensional Morse-Bott theory).

### U.2 Gradient and Differential Forms

In differential geometry, the gradient is the unique vector field dual to the exterior derivative (1-form) $df$:

$$df = \sum_i \frac{\partial f}{\partial x_i} dx_i$$

The gradient $\nabla f$ is obtained by "raising the index" using the metric tensor $g$:

$$\nabla f = g^{-1}(df)$$

In Euclidean space $g = I$, so $\nabla f = df$ (as a vector). On a Riemannian manifold, the metric $g$ introduces a non-trivial transformation — this is exactly the natural gradient transformation $F^{-1}\nabla\mathcal{L}$ where $F$ plays the role of the metric tensor on the statistical manifold.

### U.3 Gradient and Functional Analysis

For functionals on Hilbert spaces $\mathcal{H}$ (function spaces with inner product), the gradient (Fréchet derivative) is defined analogously: $F: \mathcal{H} \to \mathbb{R}$ is differentiable with gradient $\nabla F(f) \in \mathcal{H}$ if:

$$F(f + h) = F(f) + \langle \nabla F(f), h\rangle_{\mathcal{H}} + o(\lVert h\rVert_{\mathcal{H}})$$

**Applications:** The ELM (Energy-based Language Model) and the score function $\nabla_x \log p(x)$ (used in diffusion models, score matching, Langevin MCMC) are functional gradients. The denoising score matching objective of diffusion models (Ho et al., 2020; Song et al., 2021) trains a network to estimate $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ — the gradient of the log-density — which is used to sample via Langevin dynamics.

---

## Appendix V: Gradient Descent Variants — A Unified View

All modern optimizers can be written as preconditioned gradient descent:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta P_t^{-1} \nabla_{\boldsymbol{\theta}} \mathcal{L}_t$$

where $P_t$ is a preconditioning matrix that captures curvature information.

| Optimizer | Preconditioning $P_t$ | Key property |
|---|---|---|
| Gradient descent | $I$ | No curvature info; uniform step per parameter |
| AdaGrad | $\operatorname{diag}(\sum_s \mathbf{g}_s \odot \mathbf{g}_s)^{1/2}$ | Adapts to historical gradient magnitude per coordinate; sparse-gradient friendly |
| RMSprop | $\operatorname{diag}(\mathbb{E}[\mathbf{g}^2])^{1/2}$ | Exponential moving average of $\mathbf{g}^2$; non-monotone, avoids diminishing step |
| Adam | $\operatorname{diag}(\hat{v}_t)^{1/2}$, bias-corrected | Combines momentum (first moment) with RMSprop (second moment) |
| Natural gradient | $F(\boldsymbol{\theta})^{1/2}$ | Riemannian gradient; parameter-efficient steps on statistical manifold |
| Newton | $H_f(\boldsymbol{\theta}_t)^{1/2}$ | Exact curvature; quadratic convergence; $O(n^3)$ cost for $n$ parameters |
| K-FAC | Kronecker-factored Fisher approx. | Tractable approximation to natural gradient for neural networks |
| Shampoo | Kronecker product of Hessian factors | Full matrix preconditioning per layer; used in large-scale LLM training |

The gradient $\nabla_{\boldsymbol{\theta}}\mathcal{L}$ is computed identically in all cases — the optimizer only changes what it does with that gradient. Understanding the raw gradient (this section) is prerequisite to understanding any of these methods.

**For AI:** In 2025–2026, frontier LLM training uses AdamW with gradient clipping and weight decay. Shampoo and distributed Shampoo are beginning to appear in production (Google, DeepMind), offering faster convergence by more faithfully approximating the natural gradient. The mathematical basis for all these methods is the gradient this section defines.

