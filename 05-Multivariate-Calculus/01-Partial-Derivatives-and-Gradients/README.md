# Partial Derivatives and Gradients

> **Navigation**: [02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/) | [03-Chain-Rule-and-Backpropagation](../03-Chain-Rule-and-Backpropagation/) | [04-Optimization-Theory](../04-Optimization-Theory/)

## Overview

Multivariate calculus extends single-variable concepts to functions of multiple variables. In machine learning, **nearly all functions** (loss functions, neural networks, etc.) depend on many parameters. The gradient is the single most important concept for training ML models.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GRADIENT: THE ML WORKHORSE                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Loss Function L(θ₁, θ₂, ..., θₙ)                                      │
│                │                                                         │
│                ▼                                                         │
│         ┌──────────────┐                                                │
│         │   Gradient    │     ∇L = (∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ)        │
│         │   Compute     │                                                │
│         └──────┬───────┘                                                │
│                │                                                         │
│                ▼                                                         │
│         θ_new = θ_old - η·∇L    ←  Gradient Descent                     │
│                                                                          │
│   Every training step uses gradients!                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Single-variable derivatives
- Vector notation
- Basic linear algebra

## Learning Objectives

1. Compute partial derivatives
2. Understand and calculate gradients
3. Work with directional derivatives
4. Apply gradient concepts to ML optimization

---

## 1. Functions of Multiple Variables

### Notation

A function of $n$ variables:
$$f: \mathbb{R}^n \to \mathbb{R}$$
$$f(x_1, x_2, \ldots, x_n) = f(\mathbf{x})$$

**Examples:**
- $f(x, y) = x^2 + y^2$ (2D paraboloid)
- $f(x, y, z) = xyz$ (3D hyperplane intersection)
- Loss function: $L(\theta_1, \theta_2, \ldots, \theta_n)$ (millions of parameters!)

### Level Sets (Contours)

For $f: \mathbb{R}^2 \to \mathbb{R}$, level curves are sets where $f(x, y) = c$

```
Level curves of f(x,y) = x² + y²:

     y
     │      ╭───────╮ c=9
     │    ╭─┼───────┼─╮ c=4
     │  ╭─┼─┼───────┼─┼─╮ c=1
     │  │ │ │       │ │ │
  ───┼──┼─┼─┼───●───┼─┼─┼──→ x
     │  │ │ │  min  │ │ │
     │  ╰─┼─┼───────┼─┼─╯
     │    ╰─┼───────┼─╯
     │      ╰───────╯
     │

• Each ring = constant function value
• Minimum at center (0, 0)
• Gradient ∇f points outward (perpendicular to rings)
```

> **💡 ML Connection**: Loss landscapes have level sets too! Training navigates these contours toward the minimum.

---

## 2. Partial Derivatives

### Definition

The **partial derivative** of $f$ with respect to $x_i$:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

> **💡 Key Idea**: Treat all other variables as **constants** and differentiate normally!

```
Partial Derivative Visualization:

        z = f(x, y)
           │
           │    ╱│╲
           │   ╱ │ ╲
           │  ╱  │  ╲  ← Surface
           │ ╱   │   ╲
           │╱    │    ╲
           ┼─────┼─────┼────→ y
          ╱      │
         ╱       │
        ╱        ▼
       x      ∂f/∂y: slope along y-direction
              (holding x fixed)
```

### Notation Variants

| Notation | Meaning |
|----------|---------|
| $\frac{\partial f}{\partial x}$ | Partial derivative w.r.t. $x$ |
| $f_x$ or $\partial_x f$ | Compact notation |
| $D_x f$ | Operator notation |
| $\nabla_x f$ | Gradient notation (when $x$ is a vector) |

### Example

For $f(x, y) = x^2y + 3xy^2 - 2y$:

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \text{(treat } y \text{ as constant)}$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy - 2 \quad \text{(treat } x \text{ as constant)}$$

---

## 3. The Gradient

### Definition

The **gradient** of $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of all partial derivatives:

$$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

### Three Critical Properties

```
┌───────────────────────────────────────────────────────────────┐
│                    GRADIENT PROPERTIES                        │
├───────────────────────────────────────────────────────────────┤
│                                                                │
│  1. DIRECTION: Points toward steepest ascent                  │
│     ─────────                                                 │
│     Move in ∇f direction → fastest increase of f              │
│     Move in -∇f direction → fastest DECREASE (optimization!)  │
│                                                                │
│  2. MAGNITUDE: ||∇f|| = rate of steepest increase             │
│     ─────────                                                 │
│     Large ||∇f|| → steep slope → big changes                  │
│     Small ||∇f|| → flat region → near extremum                │
│                                                                │
│  3. GEOMETRY: ∇f ⊥ level sets                                 │
│     ────────                                                  │
│     Gradient always perpendicular to contour lines            │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

```
Gradient and Level Curves:

        ╭─────────────╮
       ╱       ↑       ╲
      │    →→ ↗↑↖ ←←    │      ← Level curves
      │      ↗ ↑ ↖      │
      │     ↗  ↑  ↖     │      Arrows show ∇f
      │    ↗   ●   ↖    │      (perpendicular to curves)
      │     ↗     ↖     │
       ╲    →→   ←←    ╱       Points toward
        ╰─────────────╯        increasing f
```

### Example

For $f(x, y) = x^2 + y^2$:

$$\nabla f = \begin{pmatrix} 2x \\ 2y \end{pmatrix}$$

At point $(1, 2)$:
$$\nabla f(1, 2) = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$$

> **💡 Interpretation**: At (1, 2), steepest ascent is in direction (2, 4). The function increases fastest when moving in that direction.

---

## 4. Directional Derivatives

### Definition

The **directional derivative** of $f$ in direction $\mathbf{u}$ (unit vector):

$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos\theta$$

where $\theta$ is the angle between $\nabla f$ and $\mathbf{u}$.

```
Directional Derivative Diagram:

                    ∇f (steepest ascent)
                     ↑
                    ╱│╲
                   ╱ │ ╲
              u   ╱  │  ╲
               ↗ ╱ θ │    ← angle θ
                ╱────┘
               ╱

D_u f = ||∇f|| × cos(θ)

• θ = 0°   → D_u f = ||∇f||  (maximum, along ∇f)
• θ = 90°  → D_u f = 0       (along level set)
• θ = 180° → D_u f = -||∇f|| (minimum, against ∇f)
```

### Key Results

| Direction | Angle θ | Directional Derivative |
|-----------|---------|----------------------|
| $\mathbf{u} \parallel \nabla f$ | 0° | $\|\nabla f\|$ (maximum) |
| $\mathbf{u} \perp \nabla f$ | 90° | 0 (along level set) |
| $\mathbf{u} \parallel -\nabla f$ | 180° | $-\|\nabla f\|$ (minimum) |

---

## 5. Gradient Descent

### The Algorithm

To **minimize** $f(\mathbf{x})$:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$$

where $\eta$ is the **learning rate**.

> **💡 Intuition**: Since $-\nabla f$ points toward steepest descent, we take steps in that direction to decrease $f$.

```
Gradient Descent Visualization:

Loss
  │╲
  │ ╲
  │  ╲  ●→ Start
  │   ╲  ╲
  │    ●  ╲  ← Following -∇f
  │     ╲  ╲
  │      ●  ╲
  │       ╲  ╲
  │        ●──● ← Minimum!
  │
  └────────────────→ θ (parameter)

Each step: θ_new = θ_old - η·∇L
```

### Learning Rate Effects

```
Learning Rate:

Too Small (η = 0.001)      Just Right (η = 0.1)       Too Large (η = 2.0)
        │                         │                          │
        ●                         ●                          ●
        ↓                          ╲                        ╱ ╲
        ●                           ╲                      ╱   ●
        ↓                            ●                    ●
        ●                             ╲                    ╲  ╱
        ↓                              ●                    ●╱
        ● (still far...)               ● ← Converged!      ╱ ╲ DIVERGING!

Slow convergence         Fast convergence           Oscillation/Divergence
```

> **⚠️ Warning**: Choosing the right learning rate is critical! Too small = slow. Too large = unstable.

---

## 6. Higher-Order Partial Derivatives

### Second Partial Derivatives

$$\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)$$

$$\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)$$

### Clairaut's Theorem (Symmetry of Mixed Partials)

If $f$ has continuous second partial derivatives:

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$

> **💡 Practical Impact**: Order of differentiation doesn't matter! This makes the Hessian matrix symmetric.

---

## 7. ML Applications

### 1. Neural Network Training

For a neural network with loss $L(\mathbf{w})$:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} L$$

The gradient $\nabla_{\mathbf{w}} L$ is computed via **backpropagation** (chain rule).

### 2. Linear Regression Gradient

For MSE loss: $L(\mathbf{w}) = \frac{1}{n}\|\mathbf{Xw} - \mathbf{y}\|^2$

$$\nabla_{\mathbf{w}} L = \frac{2}{n}\mathbf{X}^T(\mathbf{Xw} - \mathbf{y})$$

```
Linear Regression Gradient:

         Predictions         Targets
              ↓                  ↓
Error:      Xw         -        y
              ↓
          (Xw - y)    ← Residuals
              ↓
         X^T(Xw - y)  ← Gradient (weighted by features)
              ↓
    w_new = w - η·gradient
```

### 3. Logistic Regression Gradient

For cross-entropy loss with sigmoid $\sigma$:

$$\nabla_{\mathbf{w}} L = \frac{1}{n}\mathbf{X}^T(\sigma(\mathbf{Xw}) - \mathbf{y})$$

> **💡 Beautiful Property**: Same form as linear regression! (predictions - targets) weighted by features.

### 4. Softmax Gradient

For softmax classification:
$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

where $p_i = \text{softmax}(z)_i$.

---

## 8. Gradient Computation in Practice

### Numerical Gradient (Finite Differences)

Central difference approximation:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}$$

### Gradient Checking

Compare analytical gradient with numerical:
$$\text{relative error} = \frac{\|\nabla f_{\text{analytic}} - \nabla f_{\text{numerical}}\|}{\|\nabla f_{\text{analytic}}\| + \|\nabla f_{\text{numerical}}\|}$$

Should be $< 10^{-5}$ for correct implementation.

```
Gradient Checking Workflow:

┌──────────────────┐     ┌───────────────────┐
│ Analytical       │     │ Numerical         │
│ Gradient         │     │ Gradient          │
│ (backprop)       │     │ (finite diff)     │
└────────┬─────────┘     └────────┬──────────┘
         │                        │
         └──────────┬─────────────┘
                    │
           Compare: should match!
           Relative error < 10⁻⁵
```

> **⚠️ Debug Tip**: Always gradient check your custom layers before training!

---

## 9. Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Partial derivative | $\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(...,x_i+h,...) - f(...,x_i,...)}{h}$ |
| Gradient | $\nabla f = (\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n})^T$ |
| Directional derivative | $D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u}$ |
| Gradient descent | $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f$ |

### Common Gradients Cheat Sheet

```
┌──────────────────────────────────────────────────────────────┐
│                    COMMON GRADIENTS                          │
├──────────────────────────────────────────────────────────────┤
│  Function                    │  Gradient                     │
├──────────────────────────────┼───────────────────────────────┤
│  f = aᵀx                     │  ∇f = a                       │
│  f = xᵀx = ||x||²            │  ∇f = 2x                      │
│  f = xᵀAx                    │  ∇f = (A + Aᵀ)x               │
│  f = ||Ax - b||²             │  ∇f = 2Aᵀ(Ax - b)             │
│  f = log(1 + e^x)  (softplus)│  ∇f = σ(x)  (sigmoid)         │
│  f = -log(σ(x))  (BCE)       │  ∇f = σ(x) - 1                │
└──────────────────────────────┴───────────────────────────────┘
```

### Gradient Properties Summary

```
Gradient ∇f:
│
├── Direction: steepest ascent
├── Magnitude: rate of max increase
├── Perpendicular to level sets
│
└── ML Applications:
    ├── -∇f for optimization (descent)
    ├── Backpropagation computes ∇L
    └── Learning rate scales step size
```

---

## Exercises

1. Compute all partial derivatives of $f(x, y, z) = x^2y + yz^2 + xz$
2. Find $\nabla f$ for $f(x, y) = e^{xy} + \ln(x+y)$
3. Calculate the directional derivative of $f(x,y) = x^2 - y^2$ at $(1, 1)$ in direction $(3, 4)/5$
4. Implement gradient descent for $f(x, y) = (x-1)^2 + (y-2)^2$
5. Derive the gradient of MSE loss for linear regression

---

## References

1. Stewart - "Multivariable Calculus"
2. Goodfellow et al. - "Deep Learning"
3. Boyd & Vandenberghe - "Convex Optimization"

---

> **Next**: [02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/) — Second derivatives and curvature
