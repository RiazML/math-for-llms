# Partial Derivatives and Gradients

## Introduction

Multivariate calculus extends single-variable concepts to functions of multiple variables. In machine learning, nearly all functions (loss functions, neural networks, etc.) depend on many parameters. Understanding gradients, partial derivatives, and directional derivatives is essential for optimization.

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

- $f(x, y) = x^2 + y^2$ (2D)
- $f(x, y, z) = xyz$ (3D)
- Loss function: $L(\theta_1, \theta_2, \ldots, \theta_n) = L(\boldsymbol{\theta})$

### Level Sets

For $f: \mathbb{R}^2 \to \mathbb{R}$, level curves are sets where $f(x, y) = c$

```
Level curves of f(x,y) = x² + y²:
     y
     │    ╭─╮ c=4
     │  ╭─┼─╮ c=2
     │ ╭┼─┼─┼╮ c=1
─────┼─┼─┼─┼─┼────→ x
     │ ╰┼─┼─┼╯
     │  ╰─┼─╯
     │    ╰─╯
```

---

## 2. Partial Derivatives

### Definition

The **partial derivative** of $f$ with respect to $x_i$:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_n)}{h}$$

**Key idea:** Treat all other variables as constants and differentiate normally.

### Notation

| Notation                        | Meaning                       |
| ------------------------------- | ----------------------------- |
| $\frac{\partial f}{\partial x}$ | Partial derivative w.r.t. $x$ |
| $f_x$ or $f_{x}$                | Partial derivative w.r.t. $x$ |
| $\partial_x f$                  | Partial derivative w.r.t. $x$ |
| $D_x f$                         | Partial derivative w.r.t. $x$ |

### Example

For $f(x, y) = x^2y + 3xy^2 - 2y$:

$$\frac{\partial f}{\partial x} = 2xy + 3y^2$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy - 2$$

---

## 3. The Gradient

### Definition

The **gradient** of $f: \mathbb{R}^n \to \mathbb{R}$ is the vector of all partial derivatives:

$$\nabla f = \begin{pmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{pmatrix}$$

### Properties

1. **Direction of steepest ascent**: $\nabla f$ points in the direction of maximum increase
2. **Perpendicular to level sets**: $\nabla f \perp$ level curves/surfaces
3. **Magnitude**: $\|\nabla f\|$ gives the rate of maximum increase

```
Gradient and Level Curves:

     ╭────────╮
    ╱  ↑       ╲
   │   ↑  ↑     │
   │ →→ ↑ ←←    │
   │   ↑  ↑     │
    ╲  ↑       ╱
     ╰────────╯

Arrows (∇f) perpendicular to level curves
```

### Example

For $f(x, y) = x^2 + y^2$:

$$\nabla f = \begin{pmatrix} 2x \\ 2y \end{pmatrix}$$

At point $(1, 2)$:
$$\nabla f(1, 2) = \begin{pmatrix} 2 \\ 4 \end{pmatrix}$$

---

## 4. Directional Derivatives

### Definition

The **directional derivative** of $f$ in direction $\mathbf{u}$ (unit vector):

$$D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u} = \|\nabla f\| \cos\theta$$

where $\theta$ is the angle between $\nabla f$ and $\mathbf{u}$.

### Key Results

- **Maximum**: When $\mathbf{u} \parallel \nabla f$: $D_{\mathbf{u}}f = \|\nabla f\|$
- **Zero**: When $\mathbf{u} \perp \nabla f$: $D_{\mathbf{u}}f = 0$ (along level set)
- **Minimum**: When $\mathbf{u} \parallel -\nabla f$: $D_{\mathbf{u}}f = -\|\nabla f\|$

```
Directional Derivative:
                    ∇f (steepest ascent)
                    ↑
                   ╱│
                  ╱ │
              u  ╱  │  ← θ = angle
                ╱   │
               ╱────┘
          D_u f = ||∇f|| cos(θ)
```

---

## 5. Gradient Descent

### Algorithm

To minimize $f(\mathbf{x})$:

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f(\mathbf{x}_k)$$

where $\eta$ is the learning rate.

### Intuition

- $-\nabla f$ points in the direction of steepest **descent**
- Move in that direction to decrease $f$
- Learning rate controls step size

```
Gradient Descent Visualization:

     │ ↘
     │   ↘
     │     ↘    ← Following -∇f
     │       ↘
     │         ↘
     │           ●  ← Minimum
     └─────────────→
```

### Learning Rate Effects

| Learning Rate | Effect                 |
| ------------- | ---------------------- |
| Too small     | Slow convergence       |
| Too large     | Oscillation/divergence |
| Just right    | Optimal convergence    |

---

## 6. Higher-Order Partial Derivatives

### Second Partial Derivatives

$$\frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)$$

$$\frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)$$

### Clairaut's Theorem

If $f$ has continuous second partial derivatives:

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$

(Mixed partials are equal)

---

## 7. Applications in ML/AI

### 1. Neural Network Training

For a neural network with loss $L(\mathbf{w})$:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_{\mathbf{w}} L$$

The gradient $\nabla_{\mathbf{w}} L$ is computed via backpropagation.

### 2. Linear Regression Gradient

For MSE loss: $L(\mathbf{w}) = \frac{1}{n}\|\mathbf{Xw} - \mathbf{y}\|^2$

$$\nabla_{\mathbf{w}} L = \frac{2}{n}\mathbf{X}^T(\mathbf{Xw} - \mathbf{y})$$

### 3. Logistic Regression Gradient

For cross-entropy loss with sigmoid:

$$\nabla_{\mathbf{w}} L = \frac{1}{n}\mathbf{X}^T(\boldsymbol{\sigma}(\mathbf{Xw}) - \mathbf{y})$$

### 4. Softmax Gradient

For softmax classification:
$$\frac{\partial L}{\partial z_i} = p_i - y_i$$

where $p_i = \text{softmax}(z)_i$.

---

## 8. Gradient Computation in Practice

### Numerical Gradient

Finite difference approximation:

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h\mathbf{e}_i) - f(\mathbf{x} - h\mathbf{e}_i)}{2h}$$

### Gradient Checking

Compare analytical gradient with numerical:
$$\text{relative error} = \frac{\|\nabla f_{\text{analytic}} - \nabla f_{\text{numerical}}\|}{\|\nabla f_{\text{analytic}}\| + \|\nabla f_{\text{numerical}}\|}$$

Should be $< 10^{-5}$ for correct implementation.

---

## 9. Summary

### Key Formulas

| Concept                | Formula                                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------------------- |
| Partial derivative     | $\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(...,x_i+h,...) - f(...,x_i,...)}{h}$ |
| Gradient               | $\nabla f = (\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n})^T$         |
| Directional derivative | $D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u}$                                                  |
| Gradient descent       | $\mathbf{x}_{k+1} = \mathbf{x}_k - \eta \nabla f$                                              |

### Gradient Properties

```
Gradient ∇f:
│
├── Direction: steepest ascent
├── Magnitude: rate of max increase
├── Perpendicular to level sets
│
└── ML Applications:
    ├── -∇f for optimization
    ├── Backpropagation computes ∇L
    └── Gradient descent minimizes loss
```

### Common Gradients

| Function                             | Gradient                                             |
| ------------------------------------ | ---------------------------------------------------- |
| $f = \mathbf{a}^T\mathbf{x}$         | $\nabla f = \mathbf{a}$                              |
| $f = \mathbf{x}^T\mathbf{x}$         | $\nabla f = 2\mathbf{x}$                             |
| $f = \mathbf{x}^T\mathbf{Ax}$        | $\nabla f = (\mathbf{A} + \mathbf{A}^T)\mathbf{x}$   |
| $f = \|\mathbf{Ax} - \mathbf{b}\|^2$ | $\nabla f = 2\mathbf{A}^T(\mathbf{Ax} - \mathbf{b})$ |

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
