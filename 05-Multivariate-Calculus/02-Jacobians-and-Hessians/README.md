# Jacobians and Hessians

## Introduction

The Jacobian and Hessian matrices generalize derivatives to vector-valued functions and second-order derivatives, respectively. These are fundamental to understanding neural network backpropagation, optimization, and the geometry of loss landscapes.

## Prerequisites

- Partial derivatives and gradients
- Matrix operations
- Chain rule

## Learning Objectives

1. Understand and compute Jacobian matrices
2. Understand and compute Hessian matrices
3. Apply these to ML optimization
4. Understand their role in backpropagation

---

## 1. The Jacobian Matrix

### Definition

For a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$\mathbf{f}(\mathbf{x}) = \begin{pmatrix} f_1(x_1, \ldots, x_n) \\ f_2(x_1, \ldots, x_n) \\ \vdots \\ f_m(x_1, \ldots, x_n) \end{pmatrix}$$

The **Jacobian** is an $m \times n$ matrix:

$$
\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{pmatrix}
\frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\
\frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n}
\end{pmatrix}
$$

### Notation

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

Row $i$ is the gradient of $f_i$: $(\nabla f_i)^T$

### Example

For $\mathbf{f}(x, y) = \begin{pmatrix} x^2 + y \\ xy \end{pmatrix}$:

$$
\mathbf{J} = \begin{pmatrix}
2x & 1 \\
y & x
\end{pmatrix}
$$

---

## 2. Jacobian Properties

### Linear Approximation

$$\mathbf{f}(\mathbf{x} + \Delta\mathbf{x}) \approx \mathbf{f}(\mathbf{x}) + \mathbf{J}\Delta\mathbf{x}$$

This is the multivariate analog of $f(x + h) \approx f(x) + f'(x)h$.

### Chain Rule

For $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^m$ and $\mathbf{f}: \mathbb{R}^m \to \mathbb{R}^p$:

$$\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}} \cdot \mathbf{J}_{\mathbf{g}}$$

The chain rule becomes **matrix multiplication**!

```
Chain Rule for Jacobians:

Input      Layer 1     Layer 2     Output
  x    →     g(x)   →   f(g(x))   →   y
  ↑           ↑           ↑
  n           m           p

J_total = J_f (p×m) × J_g (m×n) = (p×n)
```

### Determinant

The **Jacobian determinant** measures volume change:
$$\det(\mathbf{J}) = \text{ratio of output volume to input volume}$$

---

## 3. The Hessian Matrix

### Definition

For a scalar function $f: \mathbb{R}^n \to \mathbb{R}$, the **Hessian** is the matrix of second partial derivatives:

$$
\mathbf{H} = \nabla^2 f = \begin{pmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{pmatrix}
$$

### Properties

1. **Symmetric**: $H_{ij} = H_{ji}$ (by Clairaut's theorem)
2. **Jacobian of gradient**: $\mathbf{H} = \mathbf{J}(\nabla f)$

### Example

For $f(x, y) = x^3 + 2xy^2 - y^3$:

$$\nabla f = \begin{pmatrix} 3x^2 + 2y^2 \\ 4xy - 3y^2 \end{pmatrix}$$

$$
\mathbf{H} = \begin{pmatrix}
6x & 4y \\
4y & 4x - 6y
\end{pmatrix}
$$

---

## 4. Hessian and Optimization

### Second-Order Taylor Expansion

$$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T \mathbf{H} \Delta\mathbf{x}$$

### Critical Point Classification

At a critical point where $\nabla f = 0$:

| Hessian Property                         | Point Type    |
| ---------------------------------------- | ------------- |
| $\mathbf{H} \succ 0$ (positive definite) | Local minimum |
| $\mathbf{H} \prec 0$ (negative definite) | Local maximum |
| $\mathbf{H}$ indefinite                  | Saddle point  |
| $\mathbf{H}$ singular                    | Inconclusive  |

### Positive Definiteness Tests

For 2×2 Hessian $\mathbf{H} = \begin{pmatrix} a & b \\ b & c \end{pmatrix}$:

- **Positive definite**: $a > 0$ and $ac - b^2 > 0$
- **Negative definite**: $a < 0$ and $ac - b^2 > 0$
- **Indefinite**: $ac - b^2 < 0$ (saddle point)

```
Saddle Point Visualization:

    ↗ f increases    f(x,y) = x² - y²
   ╱
──╳── ← saddle
   ╲
    ↘ f decreases
```

---

## 5. Newton's Method (Second-Order Optimization)

### Algorithm

$$\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}^{-1} \nabla f(\mathbf{x}_k)$$

### Derivation

From Taylor expansion, minimize quadratic approximation:
$$q(\Delta\mathbf{x}) = f + \nabla f^T \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^T \mathbf{H} \Delta\mathbf{x}$$

Setting $\nabla_{\Delta\mathbf{x}} q = 0$:
$$\nabla f + \mathbf{H}\Delta\mathbf{x} = 0$$
$$\Delta\mathbf{x} = -\mathbf{H}^{-1}\nabla f$$

### Comparison with Gradient Descent

| Method           | Update                      | Convergence |
| ---------------- | --------------------------- | ----------- |
| Gradient Descent | $-\eta \nabla f$            | Linear      |
| Newton's Method  | $-\mathbf{H}^{-1} \nabla f$ | Quadratic   |

---

## 6. Applications in ML

### 1. Backpropagation and Jacobians

For a neural network layer $\mathbf{y} = \mathbf{f}(\mathbf{x})$:

$$\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}^T \frac{\partial L}{\partial \mathbf{y}}$$

Backpropagation is repeated Jacobian-vector products!

### 2. Softmax Jacobian

For softmax $p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$:

$$J_{ij} = \frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

$$\mathbf{J} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

### 3. Loss Landscape Curvature

The Hessian describes the **curvature** of the loss surface:

- Large eigenvalues → steep directions (fast learning)
- Small eigenvalues → flat directions (slow learning)
- Negative eigenvalues → saddle points

### 4. Natural Gradient

Uses Fisher Information Matrix (expected Hessian of log-likelihood):
$$\mathbf{F} = \mathbb{E}[\nabla \log p \cdot (\nabla \log p)^T]$$

Natural gradient: $\mathbf{F}^{-1} \nabla L$

---

## 7. Jacobian-Vector Products (JVP) and VJP

### JVP (Forward Mode)

Compute $\mathbf{J}\mathbf{v}$ without forming $\mathbf{J}$:
$$\mathbf{J}\mathbf{v} = \lim_{\epsilon \to 0} \frac{\mathbf{f}(\mathbf{x} + \epsilon\mathbf{v}) - \mathbf{f}(\mathbf{x})}{\epsilon}$$

### VJP (Reverse Mode / Backprop)

Compute $\mathbf{J}^T\mathbf{v}$ without forming $\mathbf{J}$:

This is what backpropagation computes!

```
Forward vs Reverse Mode:

Forward (JVP):           Reverse (VJP):
Input → ... → Output     Input ← ... ← Output
  ↓ v                          v^T J^T ↓
J·v propagates forward   v^T·J propagates backward
```

---

## 8. Computing Jacobians and Hessians

### Numerical Jacobian

```python
def numerical_jacobian(f, x, h=1e-7):
    n = len(x)
    f_x = f(x)
    m = len(f_x)
    J = np.zeros((m, n))
    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += h
        J[:, j] = (f(x_plus) - f_x) / h
    return J
```

### Numerical Hessian

```python
def numerical_hessian(f, x, h=1e-5):
    n = len(x)
    H = np.zeros((n, n))
    f_x = f(x)
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            H[i,j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4*h*h)
    return H
```

---

## 9. Summary

### Key Matrices

| Matrix   | Dimension    | Definition                                                |
| -------- | ------------ | --------------------------------------------------------- |
| Jacobian | $m \times n$ | $J_{ij} = \frac{\partial f_i}{\partial x_j}$              |
| Hessian  | $n \times n$ | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ |

### Key Results

| Concept          | Formula                                                                                        |
| ---------------- | ---------------------------------------------------------------------------------------------- |
| Chain rule       | $\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}} \mathbf{J}_{\mathbf{g}}$   |
| Taylor expansion | $f(\mathbf{x}+\Delta) \approx f + \nabla f^T\Delta + \frac{1}{2}\Delta^T\mathbf{H}\Delta$      |
| Newton's method  | $\mathbf{x}_{k+1} = \mathbf{x}_k - \mathbf{H}^{-1}\nabla f$                                    |
| Backprop         | $\frac{\partial L}{\partial \mathbf{x}} = \mathbf{J}^T \frac{\partial L}{\partial \mathbf{y}}$ |

### ML Applications

```
Jacobian and Hessian in ML:
│
├── Jacobian
│   ├── Backpropagation: J^T × (∂L/∂y)
│   ├── Softmax gradient
│   └── Change of variables (normalizing flows)
│
├── Hessian
│   ├── Curvature of loss landscape
│   ├── Newton's method
│   ├── Second-order optimization
│   └── Saddle point detection
│
└── Efficient Computation
    ├── JVP: Forward mode autodiff
    └── VJP: Reverse mode (backprop)
```

---

## Exercises

1. Compute the Jacobian of $\mathbf{f}(x, y) = (x^2 - y, xy, e^x)$
2. Find the Hessian of $f(x, y) = x^4 + y^4 - 2x^2y^2$ and classify critical points
3. Derive the Jacobian of the softmax function
4. Show that the Hessian of $f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}$ is $\mathbf{A} + \mathbf{A}^T$
5. Implement Newton's method to minimize $f(x, y) = (x-1)^2 + 10(y-x^2)^2$

---

## References

1. Magnus & Neudecker - "Matrix Differential Calculus"
2. Goodfellow et al. - "Deep Learning"
3. Boyd & Vandenberghe - "Convex Optimization"
