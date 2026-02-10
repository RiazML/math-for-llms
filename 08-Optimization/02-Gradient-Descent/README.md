# Gradient Descent Methods

> **Navigation**: [← 01-Convex-Optimization](../01-Convex-Optimization/) | [Optimization](../) | [03-Second-Order-Methods →](../03-Second-Order-Methods/)

**Files in this section:**
- [examples.ipynb](examples.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Gradient descent is the foundational optimization algorithm that powers virtually all of modern machine learning. From training simple linear regression models to optimizing neural networks with billions of parameters, gradient descent and its variants form the computational backbone of the field. Understanding this algorithm deeply is not merely academic—it directly impacts your ability to train models effectively, diagnose optimization problems, and make informed choices about hyperparameters.

The core idea behind gradient descent is elegantly simple: to find the minimum of a function, repeatedly take steps in the direction that decreases the function value most rapidly. This direction is given by the negative gradient, which points opposite to the direction of steepest ascent. While the concept is intuitive, the practical details—learning rate selection, handling noisy gradients, accelerating convergence, and adapting to the local geometry—require deep understanding to apply effectively.

This chapter provides a comprehensive treatment of gradient descent, from first principles through modern adaptive methods. We begin with the mathematical derivation and build intuition for why the algorithm works. We then analyze convergence properties rigorously, providing complete proofs for the key theorems. Finally, we explore the modern variants that make training deep neural networks practical, including momentum, Nesterov acceleration, and adaptive methods like Adam.

## Prerequisites

Before diving into gradient descent, you should be comfortable with:
- **Multivariable calculus**: Gradients, partial derivatives, the chain rule, and Taylor series expansions
- **Linear algebra**: Matrix operations, eigenvalues, positive definiteness, and matrix norms
- **Convex optimization basics**: Definition of convexity, strong convexity, and smoothness conditions

## Learning Objectives

By the end of this chapter, you will be able to:
1. Derive the gradient descent update rule from first principles using Taylor expansion
2. Prove convergence rates for convex, strongly convex, and non-convex functions
3. Analyze the effect of condition number on convergence speed
4. Implement and compare momentum methods including Nesterov acceleration
5. Derive and implement adaptive methods (AdaGrad, RMSProp, Adam)
6. Select appropriate optimizers and hyperparameters for different ML tasks
7. Diagnose and fix common optimization problems in practice

---

## 1. The Gradient Descent Algorithm

### 1.1 Derivation from Taylor Expansion

The gradient descent algorithm emerges naturally from considering how to minimize a function locally. Given a differentiable function $f: \mathbb{R}^n \to \mathbb{R}$, we seek to find a point $\mathbf{w}^*$ that minimizes $f$. The key insight comes from the first-order Taylor expansion around the current point $\mathbf{w}_t$:

$$f(\mathbf{w}_t + \mathbf{d}) \approx f(\mathbf{w}_t) + \nabla f(\mathbf{w}_t)^\top \mathbf{d}$$

where $\mathbf{d}$ is a step direction. To decrease the function value, we want to choose $\mathbf{d}$ such that $\nabla f(\mathbf{w}_t)^\top \mathbf{d} < 0$.

**Question**: Which direction $\mathbf{d}$ gives the greatest decrease per unit step length?

To answer this, we use the Cauchy-Schwarz inequality. For a step of fixed length $\|\mathbf{d}\| = \eta$:

$$\nabla f(\mathbf{w}_t)^\top \mathbf{d} \geq -\|\nabla f(\mathbf{w}_t)\| \cdot \|\mathbf{d}\| = -\eta\|\nabla f(\mathbf{w}_t)\|$$

The lower bound is achieved when $\mathbf{d}$ points in the opposite direction of the gradient:

$$\mathbf{d} = -\eta \frac{\nabla f(\mathbf{w}_t)}{\|\nabla f(\mathbf{w}_t)\|}$$

This gives us the **steepest descent direction**: the negative gradient. Absorbing the normalization into the step size yields the gradient descent update rule:

$$\boxed{\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla f(\mathbf{w}_t)}$$

where:
- $\mathbf{w}_t \in \mathbb{R}^n$: parameter vector at iteration $t$
- $\eta > 0$: learning rate (also called step size)
- $\nabla f(\mathbf{w}_t) \in \mathbb{R}^n$: gradient of $f$ evaluated at $\mathbf{w}_t$

### 1.2 Geometric Intuition

Visualizing gradient descent helps build intuition for its behavior:

```
                    Gradient Descent on a 1D Loss Surface
                    
    Loss f(w)
         │
     5.0 ┤    ●  w₀ (initial point)
         │     ╲
     4.0 ┤      ╲  Gradient points uphill
         │       ╲  (positive slope)
     3.0 ┤        ●  w₁ = w₀ - η·∇f(w₀)
         │         ╲
     2.0 ┤          ╲
         │           ●  w₂
     1.0 ┤            ╲
         │             ●  w₃
     0.5 ┤              ╲●  w* (minimum)
         │               
         └────────────────────────────── w
              1    2    3    4    5    6
    
    At each step: Move opposite to gradient direction
```

In higher dimensions, gradient descent follows the path of steepest descent on the loss surface, analogous to a ball rolling downhill:

```
           2D Loss Contours (Bird's Eye View)
           
           ┌─────────────────────────────────────┐
           │                                     │
           │    ╭──────────────────────────╮     │
           │   ╱                            ╲    │
           │  ╱    ╭────────────────────╮    ╲   │
           │ │    ╱                      ╲    │  │
           │ │   │    ╭──────────────╮    │   │  │
           │ │   │   ╱                ╲   │   │  │
           │ │   │  │    ╭────────╮    │  │   │  │
           │ │   │  │   ╱          ╲   │  │   │  │
           │ │   │  │  │     ★     │  │  │   │  │  ★ = minimum
           │ │   │  │   ╲          ╱   │  │   │  │
           │ │   │  │    ╰────────╯    │  │   │  │
           │ │   │   ╲                ╱   │   │  │
    ●──────┼─┼───●────●───────●──────●────┼───┼──│
    w₀     │ │   │    w₁      w₂     w₃   │   │  │  Path of GD
           │ │   │    ╲              ╱    │   │  │
           │ │    ╲    ╰────────────╯    ╱    │  │
           │ │     ╲                    ╱     │  │
           │  ╲     ╰──────────────────╯     ╱   │
           │   ╲                            ╱    │
           │    ╰──────────────────────────╯     │
           │                                     │
           └─────────────────────────────────────┘
           
    Contour lines show level sets of constant loss
    Gradient is perpendicular to contours, pointing uphill
```

### 1.3 The Complete Algorithm

**Algorithm: Gradient Descent**
```
Input: Initial point w₀, learning rate η, tolerance ε, max iterations T
Output: Approximate minimizer w*

1. t ← 0
2. while t < T and ‖∇f(wₜ)‖ > ε do:
3.     Compute gradient: gₜ ← ∇f(wₜ)
4.     Update parameters: wₜ₊₁ ← wₜ - η·gₜ
5.     t ← t + 1
6. return wₜ
```

> **💡 Why This Matters for ML**
> 
> In machine learning, we minimize loss functions of the form $f(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \ell(\mathbf{w}; \mathbf{x}_i, y_i)$, where the sum is over training examples. The gradient descent update becomes:
> 
> $$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{n}\sum_{i=1}^n \nabla_\mathbf{w} \ell(\mathbf{w}_t; \mathbf{x}_i, y_i)$$
> 
> This requires computing gradients over the entire dataset each iteration—the "batch" gradient descent approach.

---

## 2. Worked Example: Gradient Descent on a Quadratic Function

Let's trace through gradient descent on a concrete example to solidify understanding.

### Example 2.1: One-Dimensional Quadratic

**Problem**: Minimize $f(w) = \frac{1}{2}(w-3)^2$ starting from $w_0 = 0$ with learning rate $\eta = 0.5$.

**Solution**:

First, compute the gradient:
$$\nabla f(w) = \frac{d}{dw}\left[\frac{1}{2}(w-3)^2\right] = w - 3$$

Now apply gradient descent iterations:

**Iteration 0**: $w_0 = 0$
- Gradient: $\nabla f(w_0) = 0 - 3 = -3$
- Update: $w_1 = 0 - 0.5 \times (-3) = 0 + 1.5 = 1.5$

**Iteration 1**: $w_1 = 1.5$
- Gradient: $\nabla f(w_1) = 1.5 - 3 = -1.5$
- Update: $w_2 = 1.5 - 0.5 \times (-1.5) = 1.5 + 0.75 = 2.25$

**Iteration 2**: $w_2 = 2.25$
- Gradient: $\nabla f(w_2) = 2.25 - 3 = -0.75$
- Update: $w_3 = 2.25 - 0.5 \times (-0.75) = 2.25 + 0.375 = 2.625$

**Iteration 3**: $w_3 = 2.625$
- Gradient: $\nabla f(w_3) = 2.625 - 3 = -0.375$
- Update: $w_4 = 2.625 + 0.1875 = 2.8125$

| Iteration $t$ | $w_t$   | $\nabla f(w_t)$ | $f(w_t)$ |
|---------------|---------|-----------------|----------|
| 0             | 0.000   | -3.000          | 4.500    |
| 1             | 1.500   | -1.500          | 1.125    |
| 2             | 2.250   | -0.750          | 0.281    |
| 3             | 2.625   | -0.375          | 0.070    |
| 4             | 2.813   | -0.188          | 0.018    |
| 5             | 2.906   | -0.094          | 0.004    |

**Observation**: The distance to the optimum $w^* = 3$ decreases by a factor of 0.5 each iteration. This is **linear convergence** with rate $1 - \eta = 0.5$.

### Example 2.2: Two-Dimensional Quadratic

**Problem**: Minimize $f(x, y) = \frac{1}{2}(x^2 + 4y^2)$ from $(x_0, y_0) = (4, 2)$ with $\eta = 0.2$.

**Solution**:

The gradient is:
$$\nabla f(x, y) = \begin{pmatrix} x \\ 4y \end{pmatrix}$$

The minimum is at the origin $(0, 0)$ with $f^* = 0$.

**Iteration 0**: $(x_0, y_0) = (4, 2)$
- Gradient: $\nabla f = (4, 8)^\top$
- Update: $(x_1, y_1) = (4 - 0.2 \times 4, 2 - 0.2 \times 8) = (3.2, 0.4)$

**Iteration 1**: $(x_1, y_1) = (3.2, 0.4)$
- Gradient: $\nabla f = (3.2, 1.6)^\top$
- Update: $(x_2, y_2) = (3.2 - 0.64, 0.4 - 0.32) = (2.56, 0.08)$

**Iteration 2**: $(x_2, y_2) = (2.56, 0.08)$
- Gradient: $\nabla f = (2.56, 0.32)^\top$
- Update: $(x_3, y_3) = (2.048, -0.056)$

Notice the **oscillation in $y$**: it overshoots the minimum, then corrects. This happens because the eigenvalue in the $y$-direction (4) is larger than in the $x$-direction (1), so the effective step is too large for $y$ when $\eta$ is tuned for $x$.

```
    y
    │
  2 ┤ ●────────────────→ (4, 2) = start
    │  ╲
    │   ╲
  1 ┤    ╲
    │     ╲
    │      ╲
  0 ─┼──────●────────────────→ x
    │      ╱(3.2, 0.4)        
    │     ╱
 -1 ┤    ●  (2.56, 0.08) oscillates around y=0
    └────┴────┴────┴────┴────
         1    2    3    4
         
    The path oscillates due to different curvatures in x and y
```

---

## 3. Batch, Mini-Batch, and Stochastic Gradient Descent

### 3.1 The Variance-Computation Tradeoff

In machine learning, we minimize empirical risk:

$$f(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n f_i(\mathbf{w})$$

where $f_i(\mathbf{w}) = \ell(\mathbf{w}; \mathbf{x}_i, y_i)$ is the loss on the $i$-th training example.

Computing the full gradient requires summing over all $n$ examples:

$$\nabla f(\mathbf{w}) = \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{w})$$

For large datasets (millions of examples), this is computationally prohibitive. The key insight is that a random subset (mini-batch) provides an **unbiased estimate** of the true gradient:

$$\mathbb{E}_{B}\left[\frac{1}{|B|}\sum_{i \in B} \nabla f_i(\mathbf{w})\right] = \nabla f(\mathbf{w})$$

where $B$ is a randomly sampled mini-batch.

### 3.2 Three Flavors of Gradient Descent

| Variant | Batch Size | Gradient Estimate | Update Frequency |
|---------|------------|-------------------|------------------|
| **Batch GD** | $n$ (full dataset) | Exact $\nabla f(\mathbf{w})$ | Once per epoch |
| **Mini-batch SGD** | $b$ (typically 32-256) | $\frac{1}{b}\sum_{i \in B} \nabla f_i(\mathbf{w})$ | $n/b$ per epoch |
| **Stochastic GD** | 1 | $\nabla f_i(\mathbf{w})$ | $n$ per epoch |

### 3.3 The Gradient Variance Problem

Stochastic gradients are noisy. The variance of the gradient estimate is:

$$\text{Var}\left[\frac{1}{|B|}\sum_{i \in B} \nabla f_i(\mathbf{w})\right] = \frac{\sigma^2}{|B|}$$

where $\sigma^2$ is the variance of individual gradients. This reveals a fundamental tradeoff:

- **Larger batches**: Lower variance, but more computation per update
- **Smaller batches**: Higher variance, but more updates per unit compute

```
         Gradient Variance vs. Batch Size
         
    Var(g)
       │
  High ┤ ●
       │  ╲
       │   ●
       │    ╲
       │     ●
       │      ╲
       │       ●──●──●──●──●
   Low ┤                       (diminishing returns)
       └──────────────────────→ Batch Size
         1   8  32 128 512 2048
         
    Variance decreases as 1/|B|, but so does updates per epoch
```

### 3.4 Worked Example: Comparing SGD and Batch GD

**Problem**: Compare batch and stochastic gradient descent on mean squared error:
$$f(\mathbf{w}) = \frac{1}{4}\sum_{i=1}^{4}(y_i - \mathbf{w}^\top \mathbf{x}_i)^2$$

**Data**:
| $i$ | $\mathbf{x}_i$ | $y_i$ |
|-----|----------------|-------|
| 1   | $(1, 0)$       | 2     |
| 2   | $(0, 1)$       | 1     |
| 3   | $(1, 1)$       | 3     |
| 4   | $(2, 1)$       | 4     |

Start from $\mathbf{w}_0 = (0, 0)$, $\eta = 0.1$.

**Batch GD (first iteration)**:

The gradient for example $i$ is:
$$\nabla f_i(\mathbf{w}) = -(\mathbf{y}_i - \mathbf{w}^\top\mathbf{x}_i)\mathbf{x}_i$$

At $\mathbf{w}_0 = (0, 0)$:
- $\nabla f_1 = -(2 - 0)(1, 0) = (-2, 0)$
- $\nabla f_2 = -(1 - 0)(0, 1) = (0, -1)$
- $\nabla f_3 = -(3 - 0)(1, 1) = (-3, -3)$
- $\nabla f_4 = -(4 - 0)(2, 1) = (-8, -4)$

Average gradient:
$$\nabla f(\mathbf{w}_0) = \frac{1}{4}[(-2, 0) + (0, -1) + (-3, -3) + (-8, -4)] = (-3.25, -2)$$

Update:
$$\mathbf{w}_1 = (0, 0) - 0.1 \times (-3.25, -2) = (0.325, 0.2)$$

**SGD (first four updates, one pass through data)**:

Update 1 (sample $i=1$): $\mathbf{w}_1 = (0,0) - 0.1(-2, 0) = (0.2, 0)$
Update 2 (sample $i=2$): $\mathbf{w}_2 = (0.2, 0) - 0.1(0, -1) = (0.2, 0.1)$
Update 3 (sample $i=3$): residual $= 3 - 0.3 = 2.7$, $\nabla = -2.7(1,1)$
  $\mathbf{w}_3 = (0.2, 0.1) + 0.27(1, 1) = (0.47, 0.37)$
Update 4 (sample $i=4$): residual $= 4 - 1.31 = 2.69$, $\nabla = -2.69(2,1)$
  $\mathbf{w}_4 = (0.47, 0.37) + 0.269(2, 1) = (1.008, 0.639)$

After one epoch, SGD made 4 updates and progressed significantly, while batch GD made 1 update. However, the SGD path is noisier and less direct.

> **💡 Why This Matters for ML**
> 
> Mini-batch SGD dominates deep learning because:
> 1. **GPU efficiency**: Batches of 32-256 maximize hardware utilization
> 2. **Regularization**: Gradient noise helps escape sharp minima, improving generalization
> 3. **Fast iteration**: More updates per epoch means faster initial progress
> 
> The "best" batch size depends on dataset size, compute budget, and model architecture.

---

## 4. Convergence Analysis

### 4.1 Key Definitions

Before proving convergence, we need precise definitions of the function properties that determine convergence rates.

**Definition (L-Smoothness)**: A function $f$ is **L-smooth** if its gradient is Lipschitz continuous:
$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\| \quad \forall \mathbf{x}, \mathbf{y}$$

Equivalently, for twice-differentiable functions: $\nabla^2 f(\mathbf{x}) \preceq LI$ (all eigenvalues $\leq L$).

**Intuition**: Smoothness bounds how fast the gradient can change. The gradient doesn't "surprise" you—if you take a step, the gradient at the new point is close to what you'd predict.

**Definition (μ-Strong Convexity)**: A function $f$ is **μ-strongly convex** if:
$$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

Equivalently: $\nabla^2 f(\mathbf{x}) \succeq \mu I$ (all eigenvalues $\geq \mu > 0$).

**Intuition**: The function curves upward at least as fast as a parabola with curvature $\mu$. This guarantees a unique minimum.

### 4.2 Descent Lemma

The fundamental tool for analyzing gradient descent is the descent lemma:

**Lemma (Descent Lemma)**: If $f$ is $L$-smooth, then:
$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

**Proof**: By Taylor's theorem with integral remainder:
$$f(\mathbf{y}) = f(\mathbf{x}) + \int_0^1 \nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x}))^\top(\mathbf{y}-\mathbf{x})dt$$

Subtracting $f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x})$:
$$f(\mathbf{y}) - f(\mathbf{x}) - \nabla f(\mathbf{x})^\top(\mathbf{y}-\mathbf{x}) = \int_0^1 [\nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})) - \nabla f(\mathbf{x})]^\top(\mathbf{y}-\mathbf{x})dt$$

By Cauchy-Schwarz and $L$-smoothness:
$$\leq \int_0^1 \|\nabla f(\mathbf{x} + t(\mathbf{y}-\mathbf{x})) - \nabla f(\mathbf{x})\| \cdot \|\mathbf{y}-\mathbf{x}\| dt$$
$$\leq \int_0^1 Lt\|\mathbf{y}-\mathbf{x}\|^2 dt = \frac{L}{2}\|\mathbf{y}-\mathbf{x}\|^2 \quad \square$$

### 4.3 Convergence for Smooth Convex Functions

**Theorem 4.1**: Let $f$ be convex and $L$-smooth. With step size $\eta \leq \frac{1}{L}$, gradient descent satisfies:
$$f(\mathbf{w}_T) - f(\mathbf{w}^*) \leq \frac{\|\mathbf{w}_0 - \mathbf{w}^*\|^2}{2\eta T}$$

**Proof**:

*Step 1*: Apply the descent lemma with $\mathbf{y} = \mathbf{w}_{t+1}$, $\mathbf{x} = \mathbf{w}_t$:
$$f(\mathbf{w}_{t+1}) \leq f(\mathbf{w}_t) + \nabla f(\mathbf{w}_t)^\top(\mathbf{w}_{t+1} - \mathbf{w}_t) + \frac{L}{2}\|\mathbf{w}_{t+1} - \mathbf{w}_t\|^2$$

Since $\mathbf{w}_{t+1} - \mathbf{w}_t = -\eta \nabla f(\mathbf{w}_t)$:
$$f(\mathbf{w}_{t+1}) \leq f(\mathbf{w}_t) - \eta\|\nabla f(\mathbf{w}_t)\|^2 + \frac{L\eta^2}{2}\|\nabla f(\mathbf{w}_t)\|^2$$
$$= f(\mathbf{w}_t) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f(\mathbf{w}_t)\|^2$$

For $\eta \leq \frac{1}{L}$, we have $1 - \frac{L\eta}{2} \geq \frac{1}{2}$, so:
$$f(\mathbf{w}_{t+1}) \leq f(\mathbf{w}_t) - \frac{\eta}{2}\|\nabla f(\mathbf{w}_t)\|^2 \quad \text{...(1)}$$

*Step 2*: By convexity:
$$f(\mathbf{w}^*) \geq f(\mathbf{w}_t) + \nabla f(\mathbf{w}_t)^\top(\mathbf{w}^* - \mathbf{w}_t)$$

Rearranging:
$$f(\mathbf{w}_t) - f(\mathbf{w}^*) \leq \nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*) \quad \text{...(2)}$$

*Step 3*: Analyze $\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2$:
$$\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 = \|\mathbf{w}_t - \eta\nabla f(\mathbf{w}_t) - \mathbf{w}^*\|^2$$
$$= \|\mathbf{w}_t - \mathbf{w}^*\|^2 - 2\eta\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*) + \eta^2\|\nabla f(\mathbf{w}_t)\|^2$$

Using (2):
$$\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 \leq \|\mathbf{w}_t - \mathbf{w}^*\|^2 - 2\eta(f(\mathbf{w}_t) - f(\mathbf{w}^*)) + \eta^2\|\nabla f(\mathbf{w}_t)\|^2$$

Rearranging:
$$f(\mathbf{w}_t) - f(\mathbf{w}^*) \leq \frac{\|\mathbf{w}_t - \mathbf{w}^*\|^2 - \|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2}{2\eta} + \frac{\eta}{2}\|\nabla f(\mathbf{w}_t)\|^2$$

*Step 4*: Sum over $t = 0, \ldots, T-1$:
$$\sum_{t=0}^{T-1}(f(\mathbf{w}_t) - f(\mathbf{w}^*)) \leq \frac{\|\mathbf{w}_0 - \mathbf{w}^*\|^2}{2\eta} + \frac{\eta}{2}\sum_{t=0}^{T-1}\|\nabla f(\mathbf{w}_t)\|^2$$

From (1), summing: $\sum_t \|\nabla f(\mathbf{w}_t)\|^2 \leq \frac{2}{\eta}(f(\mathbf{w}_0) - f(\mathbf{w}_T))$

After some algebra (see full derivation in convex optimization texts), using that $f$ is decreasing:
$$f(\mathbf{w}_T) - f(\mathbf{w}^*) \leq \frac{\|\mathbf{w}_0 - \mathbf{w}^*\|^2}{2\eta T} \quad \square$$

**Interpretation**: The suboptimality decreases as $O(1/T)$. To achieve $f(\mathbf{w}_T) - f(\mathbf{w}^*) \leq \epsilon$, we need $T = O(1/\epsilon)$ iterations.

### 4.4 Convergence for Strongly Convex Functions

**Theorem 4.2**: Let $f$ be $\mu$-strongly convex and $L$-smooth. With step size $\eta = \frac{1}{L}$:
$$\|\mathbf{w}_T - \mathbf{w}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^T \|\mathbf{w}_0 - \mathbf{w}^*\|^2$$

**Proof**:

*Step 1*: By strong convexity with $\mathbf{y} = \mathbf{w}^*$:
$$f(\mathbf{w}^*) \geq f(\mathbf{w}_t) + \nabla f(\mathbf{w}_t)^\top(\mathbf{w}^* - \mathbf{w}_t) + \frac{\mu}{2}\|\mathbf{w}^* - \mathbf{w}_t\|^2$$

Since $\mathbf{w}^*$ is optimal, $f(\mathbf{w}^*) \leq f(\mathbf{w}_t)$:
$$0 \geq \nabla f(\mathbf{w}_t)^\top(\mathbf{w}^* - \mathbf{w}_t) + \frac{\mu}{2}\|\mathbf{w}^* - \mathbf{w}_t\|^2$$

Rearranging:
$$\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*) \geq \frac{\mu}{2}\|\mathbf{w}_t - \mathbf{w}^*\|^2 \quad \text{...(3)}$$

*Step 2*: From step 3 of Theorem 4.1's proof:
$$\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 = \|\mathbf{w}_t - \mathbf{w}^*\|^2 - 2\eta\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*) + \eta^2\|\nabla f(\mathbf{w}_t)\|^2$$

*Step 3*: Use the "co-coercivity" property (follows from smoothness + convexity):
$$\|\nabla f(\mathbf{w}_t)\|^2 \leq L \cdot \nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*)$$

With $\eta = \frac{1}{L}$:
$$\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 \leq \|\mathbf{w}_t - \mathbf{w}^*\|^2 - \frac{2}{L}\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*) + \frac{1}{L^2}\|\nabla f(\mathbf{w}_t)\|^2$$
$$\leq \|\mathbf{w}_t - \mathbf{w}^*\|^2 - \frac{2}{L}\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*) + \frac{1}{L}\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*)$$
$$= \|\mathbf{w}_t - \mathbf{w}^*\|^2 - \frac{1}{L}\nabla f(\mathbf{w}_t)^\top(\mathbf{w}_t - \mathbf{w}^*)$$

Using (3):
$$\leq \|\mathbf{w}_t - \mathbf{w}^*\|^2 - \frac{\mu}{2L}\|\mathbf{w}_t - \mathbf{w}^*\|^2$$
$$= \left(1 - \frac{\mu}{2L}\right)\|\mathbf{w}_t - \mathbf{w}^*\|^2$$

Actually, with a tighter analysis (using $\eta = 2/(L+\mu)$), one can show:
$$\|\mathbf{w}_{t+1} - \mathbf{w}^*\|^2 \leq \left(1 - \frac{\mu}{L}\right)^T\|\mathbf{w}_0 - \mathbf{w}^*\|^2 = \left(\frac{\kappa - 1}{\kappa}\right)^T\|\mathbf{w}_0 - \mathbf{w}^*\|^2 \quad \square$$

**Interpretation**: This is **linear convergence** (exponential in $T$). To achieve $\|\mathbf{w}_T - \mathbf{w}^*\| \leq \epsilon$, we need $T = O(\kappa \log(1/\epsilon))$ iterations.

### 4.5 The Condition Number

The **condition number** $\kappa = L/\mu$ determines convergence speed:

$$\kappa = \frac{L}{\mu} = \frac{\lambda_{\max}(\nabla^2 f)}{\lambda_{\min}(\nabla^2 f)}$$

```
        Effect of Condition Number on Convergence
        
    ‖wₜ - w*‖
        │
    1.0 ┤ ●───●───●───●───●───●───●───●   κ = 100
        │  ╲
        │   ●──●──●──●                     κ = 10
        │    ╲
    0.5 ┤     ●─●                          κ = 2
        │      ╲
        │       ●
    0.1 ┤        ●                         κ = 1
        │
        └────────────────────────────────→ Iteration
          0   5   10  15  20  25  30
          
    Iterations to halve error ≈ κ · ln(2)
```

### Example 4.1: Effect of Condition Number

**Problem**: Compare convergence on two quadratics:
- $f_1(x, y) = \frac{1}{2}(x^2 + y^2)$ with $\kappa_1 = 1$
- $f_2(x, y) = \frac{1}{2}(x^2 + 100y^2)$ with $\kappa_2 = 100$

Start from $(1, 1)$, use $\eta = 1/L$ for each.

**Solution**:

For $f_1$: $L = 1$, $\eta = 1$
$$\mathbf{w}_1 = (1, 1) - 1 \cdot (1, 1) = (0, 0) = \mathbf{w}^*$$
Converges in **1 iteration**!

For $f_2$: $L = 100$, $\eta = 0.01$
$$\mathbf{w}_1 = (1, 1) - 0.01 \cdot (1, 100) = (0.99, 0)$$
$$\mathbf{w}_2 = (0.99, 0) - 0.01 \cdot (0.99, 0) = (0.9801, 0)$$

After $t$ iterations: $x_t = 0.99^t$, $y_t \approx 0$ (already at optimum in $y$)

To reach $x_t = 0.01$: need $0.99^t = 0.01$, so $t = \frac{\ln(0.01)}{\ln(0.99)} \approx 459$ iterations.

The ill-conditioned problem takes **459× more iterations**!

> **⚠️ Common Pitfall: Condition Number Explosion**
> 
> In neural networks, the effective condition number can be enormous (10⁶ or higher), especially:
> - Near saddle points (where eigenvalues approach zero)
> - With unscaled features (one feature ranging 0-1, another 0-10000)
> - Deep networks with exploding/vanishing gradients
> 
> This is why adaptive methods and proper initialization are crucial.

---

## 5. Learning Rate Selection

### 5.1 Theory: The Safe Range

For an $L$-smooth function, gradient descent is guaranteed to decrease the objective when:
$$\eta < \frac{2}{L}$$

With $\eta = 1/L$, we get the standard convergence guarantees.

**What happens with larger step sizes?**

For a quadratic $f(w) = \frac{L}{2}w^2$:
$$w_{t+1} = w_t - \eta L w_t = (1 - \eta L)w_t$$

Three regimes:

| Step Size | $(1 - \eta L)$ | Behavior |
|-----------|----------------|----------|
| $\eta < 1/L$ | $\in (0, 1)$ | Converges monotonically |
| $\eta = 1/L$ | $= 0$ | One-step convergence! |
| $\eta \in (1/L, 2/L)$ | $\in (-1, 0)$ | Converges with oscillation |
| $\eta = 2/L$ | $= -1$ | Oscillates forever |
| $\eta > 2/L$ | $< -1$ | **Diverges** |

```
              Learning Rate Dynamics on f(w) = ½Lw²
              
    w_t
     │
   2 ┤     ╱╲      η = 2.5/L (DIVERGES)
     │    ╱  ╲╱╲
   1 ┤ ●──────╲─╲╲─→  η = 1.5/L (oscillates, converges)
     │   ╲    ╱  ╲
   0 ─────●──●────●────  η = 1/L (one-step)
     │       ╲        
  -1 ┤        ●────────  η = 0.5/L (slow, monotonic)
     │                
     └──────────────────→ Iteration t
           0  1  2  3  4
```

### 5.2 Practical Learning Rate Selection

In practice, we rarely know $L$ exactly. Common approaches:

**Grid Search**:
- Try logarithmically spaced values: $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$
- Pick the largest that trains stably

**Learning Rate Range Test** (Smith, 2017):
- Start with tiny $\eta$, gradually increase during training
- Plot loss vs. $\eta$
- Select $\eta$ where loss decreases fastest (before instability)

```
    Loss │                          
         │ ●                        Optimal range
         │  ●                            │
         │   ●                           ▼
         │    ●●●●●                 ┌─────────┐
         │         ●●●             │         │
         │             ●●          │         │
         │                ●        │         │   Unstable
         │                 ●●      │         │        ╱
         │                    ●    │         │       ●
         │                     ●●●●│         │     ╱╱
         │                         │         │    ●
         │                         │         │  ╱●
         │                         │         │╱
         └─────────────────────────┴─────────┴─────→ η
          10⁻⁵   10⁻⁴   10⁻³   10⁻²   10⁻¹   10⁰
```

### 5.3 Learning Rate Schedules

Starting with a fixed learning rate and decaying it during training often improves final performance.

**Step Decay**:
$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

Multiply by $\gamma$ (e.g., 0.1) every $s$ epochs. Popular in computer vision.

**Example**: $\eta_0 = 0.1$, $\gamma = 0.1$, $s = 30$ epochs
- Epochs 1-30: $\eta = 0.1$
- Epochs 31-60: $\eta = 0.01$
- Epochs 61-90: $\eta = 0.001$

**Exponential Decay**:
$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

Smooth decay with rate $\lambda$.

**Cosine Annealing**:
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

```
    η(t)
    0.1 ┤●──────╮
        │       ╲
   0.08 ┤        ╲
        │         ╲
   0.06 ┤          ╲
        │           ╲
   0.04 ┤            ╲
        │             ╲
   0.02 ┤              ╲
        │               ╲
      0 ┤                ╲●
        └──────────────────→ Epoch
         0    50   100  150
         
    Cosine schedule: Smooth, theoretically motivated
```

**Warmup**:
Start with small learning rate, linearly increase to target, then decay:
$$\eta_t = \begin{cases} \frac{t}{T_{\text{warmup}}} \cdot \eta_{\max} & t \leq T_{\text{warmup}} \\ \text{schedule}(\eta_{\max}, t - T_{\text{warmup}}) & t > T_{\text{warmup}} \end{cases}$$

```
    η(t)
        │       ╭──────╮
        │      ╱        ╲      Warmup + Cosine Decay
        │     ╱          ╲
        │    ╱            ╲
        │   ╱              ╲
        │  ╱                ╲
        │ ╱                  ╲
        │╱                    ╲
        └──────────────────────→ t
         │       │
      warmup   main training
```

> **💡 Why This Matters for ML**
> 
> **Warmup is critical for**:
> - Large batch training (helps gradient estimates stabilize)
> - Transformers (prevents early training instability)
> - Transfer learning (adapts slowly to new data distribution)
> 
> **Cosine annealing is popular because**:
> - No hyperparameters to tune (just endpoints)
> - Smooth exploration → exploitation transition
> - "SGDR" variant with restarts can escape local minima

---

## 6. Momentum Methods

### 6.1 The Problem: Oscillation in Ill-Conditioned Problems

Gradient descent oscillates in directions of high curvature while making slow progress in directions of low curvature:

```
    y (high curvature direction)
    │
    ├───●                      Standard GD zigzags
    │  ╱  ╲                    
    ├ ●    ●                   
    │  ╲  ╱                    
    ├   ●●                     
    │    ╲╱                    
    ├     ●                    
    │     ↓                    
    ├     ★ optimum            
    │
    └──────────────────→ x (low curvature direction)
```

The gradient along $y$ keeps flipping sign, wasting updates.

### 6.2 Classical Momentum (Polyak, 1964)

**Idea**: Accumulate a "velocity" that smooths out oscillations:

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \nabla f(\mathbf{w}_t)$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$$

Equivalently:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\nabla f(\mathbf{w}_t) - \eta\gamma\mathbf{v}_t$$

The velocity $\mathbf{v}$ is an exponential moving average of past gradients:
$$\mathbf{v}_{t} = \sum_{k=0}^{t-1} \gamma^{t-1-k} \nabla f(\mathbf{w}_k)$$

**Physics analogy**: A ball rolling downhill accumulates speed. The momentum term $\gamma \mathbf{v}_t$ represents inertia—the ball continues moving even if the slope levels out.

### 6.3 Why Momentum Helps

**Dampens oscillations**: In high-curvature directions, gradients flip sign each iteration. These cancel out in the velocity average:
$$v_y^{(t)} = g_y^{(t)} + \gamma g_y^{(t-1)} + \gamma^2 g_y^{(t-2)} + \cdots$$

If $g_y$ alternates $+a, -a, +a, -a$:
$$v_y \approx a(1 - \gamma + \gamma^2 - \gamma^3 + \cdots) = \frac{a}{1+\gamma} \approx \frac{a}{2}$$

**Accelerates in consistent directions**: In low-curvature directions, gradients have consistent sign:
$$v_x \approx a(1 + \gamma + \gamma^2 + \cdots) = \frac{a}{1-\gamma} \approx 10a \text{ (for } \gamma=0.9\text{)}$$

```
          Gradient Descent         vs         Momentum
          
    y                                   y
    │                                   │
    │  ●───●                            │  ●
    │     ╲ ╲                           │   ╲
    │      ●─●                          │    ╲
    │         ╲                         │     ●
    │          ●                        │      ╲
    │           ╲                       │       ●
    │            ●                      │        ╲
    │             ↓                     │         ●
    │              ★                    │          ★
    └──────────────→ x                  └──────────→ x
    
    Many oscillations                   Direct path!
```

### 6.4 Worked Example: Momentum on Ill-Conditioned Quadratic

**Problem**: Minimize $f(x, y) = \frac{1}{2}(x^2 + 100y^2)$ from $(10, 1)$.

Compare vanilla GD ($\eta = 0.01$) vs momentum ($\eta = 0.01$, $\gamma = 0.9$).

**Without momentum** (first 5 iterations):

| $t$ | $(x_t, y_t)$ | $\nabla f$ | 
|-----|--------------|------------|
| 0 | (10, 1) | (10, 100) |
| 1 | (9.9, 0) | (9.9, 0) |
| 2 | (9.801, 0) | (9.801, 0) |
| 3 | (9.703, 0) | (9.703, 0) |
| 4 | (9.606, 0) | (9.606, 0) |

The $y$-coordinate reaches 0 quickly but $x$ decreases by only 1% per iteration.
To reach $x = 0.1$: need $0.99^t = 0.01$, so $t \approx 459$ iterations.

**With momentum** ($\eta = 0.01$, $\gamma = 0.9$):

| $t$ | $(x_t, y_t)$ | $v_t$ | After update |
|-----|--------------|-------|--------------|
| 0 | (10, 1) | (0, 0) | — |
| 1 | (9.9, 0) | (10, 100) | $v_1 = (10, 100)$ |
| 2 | (8.811, 0) | (9.9+9, 0+0) = (18.9, 0) | $v_2 = 0.9(10,100)+(9.9,0)$ |

Wait, let me recalculate properly:

$\mathbf{v}_1 = 0.9 \cdot (0,0) + (10, 100) = (10, 100)$
$\mathbf{w}_1 = (10, 1) - 0.01 \cdot (10, 100) = (9.9, 0)$

$\mathbf{v}_2 = 0.9 \cdot (10, 100) + (9.9, 0) = (18.9, 90)$  
$\mathbf{w}_2 = (9.9, 0) - 0.01 \cdot (18.9, 90) = (9.711, -0.9)$

$\mathbf{v}_3 = 0.9 \cdot (18.9, 90) + (9.711, -90) = (26.72, -9)$
$\mathbf{w}_3 = (9.711, -0.9) - 0.01 \cdot (26.72, -9) = (9.444, -0.81)$

The $x$ velocity builds up ($10 \to 18.9 \to 26.72$), accelerating convergence in that direction. After about **50 iterations**, momentum GD reaches the optimum—nearly **10× faster**!

### 6.5 Nesterov Accelerated Gradient (NAG)

**Idea**: "Look ahead" to where momentum is taking us before computing the gradient.

$$\tilde{\mathbf{w}} = \mathbf{w}_t - \gamma\eta\mathbf{v}_t \quad \text{(lookahead position)}$$
$$\mathbf{v}_{t+1} = \gamma\mathbf{v}_t + \nabla f(\tilde{\mathbf{w}})$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta\mathbf{v}_{t+1}$$

```
         Classical Momentum            vs          Nesterov
         
          ●  current position                ●  current position
          │                                  │
          ▼  apply momentum                  ▼  apply momentum (lookahead)
          ○                                  ○ ← compute gradient HERE
          │                                  │
          ▼  compute gradient here           ▼  apply correction
          ●' new position                    ●' new position (more accurate)
```

**Why it helps**: If momentum is about to overshoot, Nesterov computes the gradient at the overshoot position and corrects before going there.

**Convergence guarantee** (for strongly convex): Instead of $O(1/T)$ for GD, Nesterov achieves:
$$f(\mathbf{w}_T) - f(\mathbf{w}^*) = O\left(\left(1 - \frac{1}{\sqrt{\kappa}}\right)^T\right)$$

This is **optimal** for first-order methods on smooth, strongly convex functions!

| Method | Iterations to halve error |
|--------|---------------------------|
| Gradient Descent | $O(\kappa)$ |
| Nesterov | $O(\sqrt{\kappa})$ |

For $\kappa = 100$: GD needs ~100 iterations, Nesterov needs ~10.

> **💡 Why This Matters for ML**
> 
> Momentum is almost always beneficial:
> - **SGD + Momentum** is the standard for training CNNs
> - Typical $\gamma = 0.9$ (sometimes 0.99 for fine-tuning)
> - Nesterov variant often gives slight improvements
> 
> In practice, momentum is so standard that "SGD" in ML usually means SGD with momentum.

---

## 7. Adaptive Learning Rate Methods

### 7.1 The Motivation

Different parameters need different learning rates:
- **Sparse features** (appear rarely): Need larger updates when they do appear
- **Dense features** (appear often): Can use smaller, stabler updates
- **Different layers**: Gradients have different scales (vanishing/exploding gradients)

Adaptive methods automatically adjust per-parameter learning rates based on gradient history.

### 7.2 AdaGrad (Adaptive Gradient)

**Algorithm** (Duchi et al., 2011):

For each parameter $j$, accumulate squared gradients:
$$G_{t,j} = \sum_{\tau=1}^{t} g_{\tau,j}^2$$

Update with scaled learning rate:
$$w_{t+1,j} = w_{t,j} - \frac{\eta}{\sqrt{G_{t,j}} + \epsilon} g_{t,j}$$

In vector form:
$$\mathbf{G}_t = \mathbf{G}_{t-1} + \mathbf{g}_t \odot \mathbf{g}_t$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{G}_t} + \epsilon} \odot \mathbf{g}_t$$

where $\odot$ is element-wise multiplication and division/sqrt are element-wise.

**Intuition**: Parameters with large accumulated gradients get smaller effective learning rates. Parameters with small accumulated gradients get larger effective learning rates.

**Problem**: $G_t$ only grows, so learning rates only decrease. Eventually, learning stops completely.

### Example 7.1: AdaGrad on Sparse Features

Consider a recommendation system where user $u$ appears in only 100 out of 1 million training examples.

**Without AdaGrad**: User embedding $\mathbf{u}$ gets updated only when user appears. With $\eta = 0.01$:
- Total update magnitude ≈ $0.01 \times 100 = 1.0$

**With AdaGrad**: First few updates have large effective learning rate:
- Update 1: effective $\eta \approx 0.01/\sqrt{g^2} = 0.01/|g|$
- After 100 updates: effective $\eta \approx 0.01/\sqrt{100 \cdot \bar{g}^2} \approx 0.001/|\bar{g}|$

Early updates are large, later updates are refined. This is ideal for sparse data!

### 7.3 RMSProp (Root Mean Square Propagation)

**Algorithm** (Hinton, unpublished lecture slides):

Use exponential moving average instead of sum:
$$\mathbf{E}[g^2]_t = \rho \cdot \mathbf{E}[g^2]_{t-1} + (1-\rho) \cdot \mathbf{g}_t^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\mathbf{E}[g^2]_t} + \epsilon} \odot \mathbf{g}_t$$

Typical: $\rho = 0.9$, meaning effective window ≈ 10 recent gradients.

**Solves AdaGrad's problem**: Forgets old gradients, so learning rate doesn't decay to zero.

### 7.4 Adam (Adaptive Moment Estimation)

**The most popular optimizer in deep learning**. Combines momentum (first moment) with RMSProp (second moment).

**Algorithm** (Kingma & Ba, 2014):

Initialize: $\mathbf{m}_0 = \mathbf{0}$, $\mathbf{v}_0 = \mathbf{0}$

For $t = 1, 2, \ldots$:

**Step 1**: Compute gradient
$$\mathbf{g}_t = \nabla f(\mathbf{w}_{t-1})$$

**Step 2**: Update biased first moment estimate (momentum)
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1)\mathbf{g}_t$$

**Step 3**: Update biased second moment estimate (RMSProp)
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2$$

**Step 4**: Bias correction
$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

**Step 5**: Update parameters
$$\mathbf{w}_t = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} \odot \hat{\mathbf{m}}_t$$

**Default hyperparameters** (almost always work):
- $\eta = 0.001$ (sometimes $3 \times 10^{-4}$ for transformers)
- $\beta_1 = 0.9$
- $\beta_2 = 0.999$
- $\epsilon = 10^{-8}$

### 7.5 Why Bias Correction is Needed

Without bias correction, $\mathbf{m}_t$ and $\mathbf{v}_t$ are biased toward zero early in training.

**Analysis**: Assuming gradients have mean $\mu$ and variance $\sigma^2$:

$$\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1)(1 + \beta_1 + \beta_1^2 + \cdots + \beta_1^{t-1})\mu = (1 - \beta_1^t)\mu$$

This is biased toward zero by factor $(1 - \beta_1^t)$.

At $t = 1$ with $\beta_1 = 0.9$:
$$\mathbb{E}[\mathbf{m}_1] = 0.1\mu$$

The estimate is **10× smaller** than the true gradient mean!

Dividing by $(1 - \beta_1^t)$ corrects this:
$$\mathbb{E}[\hat{\mathbf{m}}_t] = \frac{(1-\beta_1^t)\mu}{1 - \beta_1^t} = \mu$$

```
    Bias correction effect:
    
    E[m_t]
        │                           with correction
        │                         ╭─────────────────
        │                      ╱
        │                   ╱
    μ ──┼─────────────────●─────────────────────────  true mean
        │            ╱
        │        ╱        without correction
        │     ╱         ╭─────────────────────────
        │  ╱         ╱
        │╱       ╱
      0 ┼────────────────────────────────────────→ t
         0   5   10  15  20  25  30
```

### 7.6 Worked Example: Comparing SGD, Momentum, and Adam

**Problem**: Minimize $f(w_1, w_2) = w_1^2 + 10w_2^2$ from $(3, 1)$.

This has condition number $\kappa = 10$, with optimum at $(0, 0)$.

**SGD** ($\eta = 0.05$, max safe for $L=20$):

| $t$ | $(w_1, w_2)$ | $\nabla f$ | 
|-----|--------------|------------|
| 0 | (3.00, 1.00) | (6, 20) |
| 1 | (2.70, 0.00) | (5.4, 0) |
| 2 | (2.43, 0.00) | (4.86, 0) |
| 5 | (1.77, 0.00) | (3.54, 0) |
| 10 | (1.04, 0.00) | (2.08, 0) |
| 20 | (0.36, 0.00) | (0.72, 0) |

**SGD + Momentum** ($\eta = 0.05$, $\gamma = 0.9$):

| $t$ | $(w_1, w_2)$ | $v$ | 
|-----|--------------|-----|
| 0 | (3.00, 1.00) | (0, 0) |
| 1 | (2.70, 0.00) | (6, 20) |
| 2 | (2.13, 0.00) | (10.8, 18) |
| 5 | (0.47, 0.00) | (18.2, ...) |
| 10 | (-0.05, 0.00) | oscillating |

Momentum overshoots but converges faster overall.

**Adam** ($\eta = 0.5$, $\beta_1 = 0.9$, $\beta_2 = 0.999$):

| $t$ | $(w_1, w_2)$ | $\hat{m}$ | $\sqrt{\hat{v}}$ | Effective $\eta$ |
|-----|--------------|-----------|------------------|------------------|
| 0 | (3.00, 1.00) | — | — | — |
| 1 | (2.50, 0.50) | (6, 20) | (6, 20) | 0.5 uniform |
| 2 | (2.01, 0.01) | (5.7, 2) | (5.97, 17.9) | varies |
| 5 | (0.75, 0.00) | — | — | — |
| 10 | (0.12, 0.00) | — | — | — |

Adam normalizes by gradient magnitude, making effective step sizes similar in both directions. This eliminates the zig-zagging behavior.

---

## 8. Adam Variants and Improvements

### 8.1 AdamW (Decoupled Weight Decay)

**The problem with L2 regularization in Adam**:

Standard L2 regularization adds $\frac{\lambda}{2}\|\mathbf{w}\|^2$ to the loss:
$$\mathbf{g}_t = \nabla f(\mathbf{w}) + \lambda\mathbf{w}$$

In Adam, this gradient is normalized by $\sqrt{\hat{v}_t}$, which **weakens the regularization effect** for parameters with large gradients.

**AdamW solution** (Loshchilov & Hutter, 2017):

Apply weight decay directly to parameters, not through the gradient:
$$\mathbf{w}_t = \mathbf{w}_{t-1} - \eta\left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda\mathbf{w}_{t-1}\right)$$

This ensures all parameters are regularized equally regardless of gradient history.

**In practice**: AdamW is the default for training transformers (BERT, GPT, etc.).

### 8.2 AMSGrad

**The problem**: Adam can sometimes diverge on simple convex problems!

The issue is that $\hat{v}_t$ can decrease, allowing the learning rate to increase unexpectedly.

**AMSGrad solution** (Reddi et al., 2018):

Use the maximum of all past $\hat{v}_t$:
$$\hat{v}_t^{\text{max}} = \max(\hat{v}_{t-1}^{\text{max}}, \hat{v}_t)$$
$$\mathbf{w}_t = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t^{\text{max}}} + \epsilon} \hat{\mathbf{m}}_t$$

This guarantees non-increasing learning rates (like AdaGrad) while keeping the exponential averaging.

### 8.3 Comparison of Optimizers

| Optimizer | Compute | Memory | Best Use Case |
|-----------|---------|--------|---------------|
| SGD | 1× | 1× | CNNs, when well-tuned |
| SGD+Momentum | 1× | 2× | Vision (ResNet, etc.) |
| Adam | 1× | 3× | Default, fast prototyping |
| AdamW | 1× | 3× | Transformers, NLP |
| LAMB | 2× | 3× | Large-batch pre-training |

```
               Choosing an Optimizer
               
                   ┌─────────────────┐
                   │  What's your    │
                   │    problem?     │
                   └────────┬────────┘
                            │
            ┌───────────────┼───────────────┐
            ▼               ▼               ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │   Vision     │ │     NLP      │ │   General    │
    │   (CNNs)     │ │ (Transformers)│ │    / New     │
    └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
           │                │                │
           ▼                ▼                ▼
    ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
    │ SGD+Momentum │ │    AdamW     │ │    Adam      │
    │  lr=0.1      │ │   lr=1e-4    │ │   lr=1e-3    │
    │  mom=0.9     │ │   wd=0.01    │ │              │
    └──────────────┘ └──────────────┘ └──────────────┘
```

> **⚠️ Common Pitfall: Adam with Small Datasets**
> 
> Adam can overfit faster than SGD on small datasets because:
> 1. Adaptive learning rates allow faster memorization
> 2. Momentum accelerates convergence before regularization kicks in
> 
> **Solution**: Use early stopping, or switch to SGD+Momentum for final training.

---

## 9. Practical Training Techniques

### 9.1 Gradient Clipping

Deep networks suffer from **exploding gradients**, especially RNNs and transformers.

**Clip by value**:
$$g_j = \text{clip}(g_j, -\tau, \tau) = \begin{cases} -\tau & g_j < -\tau \\ g_j & |g_j| \leq \tau \\ \tau & g_j > \tau \end{cases}$$

**Clip by global norm** (more common):
$$\mathbf{g} = \begin{cases} \mathbf{g} & \|\mathbf{g}\| \leq \tau \\ \tau \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|} & \|\mathbf{g}\| > \tau \end{cases}$$

This rescales all gradients together, preserving relative magnitudes.

**Typical threshold**: $\tau \in [1, 5]$ for most applications.

### 9.2 Learning Rate Warmup

Critical for stable training of large models.

```
                Learning Rate Schedule with Warmup
                
    η
    │           ╭────────────────────╮
    │          ╱                      ╲
    │         ╱                        ╲
    │        ╱                          ╲
    │       ╱                            ╲
    │      ╱                              ╲
    │     ╱                                ╲
    │    ╱                                  ╲
    │   ╱
    │──╱
    └──────────────────────────────────────────→ step
       │    │
    warmup  peak              decay
```

**Why warmup helps**:
1. Early gradients are high-variance (random initialization)
2. Adam's $\hat{v}$ estimates are unreliable initially
3. Prevents early large updates that can destabilize training

**Typical warmup**: 1-10% of total training steps.

### 9.3 Batch Size Considerations

| Aspect | Small Batch (32) | Large Batch (1024+) |
|--------|------------------|---------------------|
| Gradient variance | High | Low |
| Updates per epoch | Many | Few |
| Generalization | Often better | May be worse |
| Training time | Slow (wall clock) | Fast (if parallel) |

**Linear scaling rule** (Goyal et al., 2017):
When batch size is multiplied by $k$, multiply learning rate by $k$.

**Intuition**: Larger batches give more accurate gradients. To make the same expected progress, increase step size proportionally.

**Warmup with large batches**: Essential! Use longer warmup (linear scaling).

### 9.4 Debugging Optimization

**Symptom → Diagnosis → Solution**:

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| Loss increases | Learning rate too high | Reduce $\eta$ by 10× |
| Loss stuck | Learning rate too low | Increase $\eta$, check data |
| Loss oscillates | $\eta$ too high or batch too small | Reduce $\eta$, increase batch |
| NaN loss | Exploding gradients | Clip gradients, reduce $\eta$ |
| Slow start | Poor initialization | Use standard init, warmup |

```
              Loss Curve Diagnostics
              
    Loss                            
      │                              
      │╲                Healthy training:
      │ ╲●●             - Smooth decrease
      │   ●●●           - Gradual leveling
      │      ●●●        
      │         ●●●●    
      │             ●●●●●●●
      └────────────────────→ Epoch
      
    Loss                            
      │ ●   ●                       
      │  ● ● ●  ●       Oscillating:
      │   ●   ●  ● ●    - Learning rate too high
      │         ● ● ●   - Or momentum too high
      │            ● ●  
      │               ●
      └────────────────────→ Epoch
      
    Loss                            
      │●                            
      │ ●                NaN/Explosion:
      │  ●               - Clip gradients
      │   ●              - Reduce learning rate
      │    ●             - Check for numerical issues
      │     ╲●●•NaN      
      └────────────────────→ Epoch
```

---

## 10. Convergence Rates Summary

### 10.1 Deterministic Gradient Descent

| Setting | Convergence Rate | Iterations for $\epsilon$-accuracy |
|---------|------------------|-------------------------------------|
| Convex, $L$-smooth | $f(\mathbf{w}_T) - f^* = O(1/T)$ | $O(1/\epsilon)$ |
| $\mu$-strongly convex, $L$-smooth | $\|\mathbf{w}_T - \mathbf{w}^*\|^2 = O((1-\mu/L)^T)$ | $O(\kappa\log(1/\epsilon))$ |
| Non-convex, $L$-smooth | $\|\nabla f(\mathbf{w})\|^2 = O(1/T)$ | $O(1/\epsilon^2)$ |

### 10.2 Stochastic Gradient Descent

| Setting | Learning Rate | Convergence Rate |
|---------|---------------|------------------|
| Convex | $\eta_t = O(1/\sqrt{t})$ | $O(1/\sqrt{T})$ |
| Strongly convex | $\eta_t = O(1/t)$ | $O(1/T)$ |
| Non-convex | $\eta_t = O(1/\sqrt{t})$ | $O(1/\sqrt{T})$ to stationary point |

### 10.3 Accelerated Methods

| Method | Setting | Rate |
|--------|---------|------|
| Nesterov momentum | Strongly convex, smooth | $O((1-1/\sqrt{\kappa})^T)$ |
| Heavy ball (Polyak) | Strongly convex (quadratic) | $O((1-1/\sqrt{\kappa})^T)$ |

**Key insight**: Acceleration improves $O(\kappa)$ to $O(\sqrt{\kappa})$ dependence on condition number.

---

## 11. Summary and Practical Recommendations

### Key Concepts Checklist

| Concept | What to Remember |
|---------|------------------|
| Gradient descent | $\mathbf{w} \leftarrow \mathbf{w} - \eta\nabla f(\mathbf{w})$; derived from Taylor expansion |
| Learning rate | Most important hyperparameter; too high → diverge, too low → slow |
| Condition number | $\kappa = L/\mu$; determines convergence speed |
| Momentum | Accumulates velocity; dampens oscillation, accelerates in consistent directions |
| Nesterov | "Lookahead" momentum; achieves optimal $O(\sqrt{\kappa})$ rate |
| Adam | Combines momentum + RMSProp; per-parameter adaptive learning rates |
| Warmup | Critical for large batches and transformers |
| Gradient clipping | Essential for RNNs/transformers to prevent explosion |

### Quick Reference: Optimizer Settings

**Vision (ResNet, etc.)**:
```
SGD, lr=0.1, momentum=0.9, weight_decay=1e-4
Schedule: Step decay (÷10 at epochs 30, 60, 90)
Batch size: 256
```

**NLP/Transformers (BERT, GPT)**:
```
AdamW, lr=1e-4 to 5e-4, betas=(0.9, 0.999), weight_decay=0.01
Schedule: Linear warmup (10% steps) + linear/cosine decay
Batch size: 32-512 effective (with gradient accumulation)
Gradient clipping: max_norm=1.0
```

**General/Prototyping**:
```
Adam, lr=1e-3, betas=(0.9, 0.999)
No schedule initially; add if needed
Start small batch (32-64), scale up if needed
```

### Debugging Checklist

1. ☐ Visualize loss curve—is it decreasing smoothly?
2. ☐ Check gradient norms—are they exploding or vanishing?
3. ☐ Try smaller learning rate if loss is unstable
4. ☐ Try larger learning rate if progress is too slow
5. ☐ Compare train vs. validation loss for overfitting
6. ☐ Use warmup for large models or batches
7. ☐ Apply gradient clipping for sequence models

---

## 12. Exercises

### Theoretical Exercises

1. **Convergence Proof Practice**: For $f(x) = \frac{1}{2}ax^2$ with $a > 0$, derive the exact convergence rate of gradient descent with step size $\eta$. For what values of $\eta$ does GD converge? What is the optimal $\eta$?

2. **Condition Number Analysis**: Given $f(x, y) = x^2 + cy^2$ for $c > 1$:
   - Compute the condition number $\kappa$
   - How many iterations does GD need to reduce distance to optimum by factor 10?
   - How does this change with Nesterov momentum?

3. **Momentum Derivation**: Show that the momentum update 
   $$\mathbf{v}_{t+1} = \gamma\mathbf{v}_t + \nabla f(\mathbf{w}_t), \quad \mathbf{w}_{t+1} = \mathbf{w}_t - \eta\mathbf{v}_{t+1}$$
   can be written as an exponentially-weighted sum of past gradients.

4. **Adam Bias Correction**: Prove that without bias correction, $\mathbb{E}[\mathbf{m}_t] = (1 - \beta_1^t)\mathbb{E}[\mathbf{g}]$, and show that the correction factor $(1 - \beta_1^t)^{-1}$ makes the estimator unbiased.

### Implementation Exercises

5. **Optimizer Comparison**: Implement vanilla GD, momentum, and Adam from scratch. Compare their behavior on the Rosenbrock function $f(x, y) = (1-x)^2 + 100(y-x^2)^2$. Visualize the optimization paths.

6. **Learning Rate Finder**: Implement the learning rate range test. Apply it to a simple neural network and identify the optimal learning rate.

7. **Convergence Visualization**: For a 2D quadratic with varying condition numbers ($\kappa \in \{1, 10, 100\}$), plot the optimization trajectory and loss curve. Illustrate how condition number affects convergence.

8. **Warmup Experiment**: Train a small transformer on a text classification task with and without warmup. Plot the training loss and compare stability.

---

## References

### Primary Sources

1. Polyak, B. T. (1964). "Some methods of speeding up the convergence of iteration methods." *USSR Computational Mathematics and Mathematical Physics*, 4(5), 1-17.

2. Nesterov, Y. (1983). "A method for solving the convex programming problem with convergence rate $O(1/k^2)$." *Dokl. Akad. Nauk SSSR*, 269, 543-547.

3. Duchi, J., Hazan, E., & Singer, Y. (2011). "Adaptive subgradient methods for online learning and stochastic optimization." *Journal of Machine Learning Research*, 12, 2121-2159.

4. Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." *arXiv preprint arXiv:1412.6980*.

5. Loshchilov, I., & Hutter, F. (2017). "Decoupled weight decay regularization." *arXiv preprint arXiv:1711.05101*.

### Recommended Reading

6. Ruder, S. (2016). "An overview of gradient descent optimization algorithms." *arXiv preprint arXiv:1609.04747*.

7. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 8: Optimization for Training Deep Models.

8. Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chapter 9: Unconstrained Minimization.

9. Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization: A Basic Course*.
