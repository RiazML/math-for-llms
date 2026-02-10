# Interpolation and Approximation

> **Navigation**: [← 03-Numerical-Optimization](../03-Numerical-Optimization/) | [Numerical Methods](../) | [05-Numerical-Integration →](../05-Numerical-Integration/)

**Files in this section:**
- [examples.ipynb](examples.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Overview

Interpolation and approximation methods construct functions that pass through or closely approximate given data points. These techniques are foundational for function approximation in neural networks, signal processing, and numerical analysis.

## Prerequisites

- Linear algebra
- Basic calculus
- Polynomial mathematics

## Learning Objectives

- Understand polynomial interpolation and its limitations
- Implement spline interpolation methods
- Apply approximation theory to ML contexts
- Recognize connections to neural network architectures

---

## 1. Polynomial Interpolation

### Problem Statement

Given $n+1$ data points $(x_0, y_0), (x_1, y_1), \ldots, (x_n, y_n)$, find polynomial $P(x)$ of degree $\leq n$ such that:

$$P(x_i) = y_i, \quad i = 0, 1, \ldots, n$$

**Existence & Uniqueness**: The interpolating polynomial exists and is unique.

### Lagrange Interpolation

$$P(x) = \sum_{i=0}^{n} y_i L_i(x)$$

where Lagrange basis polynomials are:

$$L_i(x) = \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

**Properties**:

- $L_i(x_j) = \delta_{ij}$ (Kronecker delta)
- Each $L_i$ has degree $n$

### Newton Interpolation

$$P(x) = a_0 + a_1(x-x_0) + a_2(x-x_0)(x-x_1) + \cdots$$

Coefficients are **divided differences**:

- $a_0 = f[x_0] = y_0$
- $a_1 = f[x_0, x_1] = \frac{y_1 - y_0}{x_1 - x_0}$
- $a_k = f[x_0, \ldots, x_k]$ (recursive formula)

**Divided Difference Table**:

```
x_0   f[x_0]
              f[x_0,x_1]
x_1   f[x_1]              f[x_0,x_1,x_2]
              f[x_1,x_2]
x_2   f[x_2]
```

**Advantage**: Easy to add new points (vs. recomputing Lagrange).

---

## 2. Interpolation Error

### Error Bound

For $f \in C^{n+1}[a,b]$:

$$f(x) - P_n(x) = \frac{f^{(n+1)}(\xi)}{(n+1)!} \prod_{i=0}^{n}(x - x_i)$$

for some $\xi \in [a,b]$.

### Runge's Phenomenon

High-degree polynomial interpolation on equally spaced points can oscillate wildly near boundaries.

**Example**: Interpolating $f(x) = \frac{1}{1+25x^2}$

**Solutions**:

1. Use Chebyshev nodes
2. Use piecewise polynomials (splines)
3. Regularization

### Chebyshev Nodes

Optimal placement minimizes $\max_x |\prod(x - x_i)|$:

$$x_k = \cos\left(\frac{2k+1}{2(n+1)}\pi\right), \quad k = 0, \ldots, n$$

Clustered near endpoints.

---

## 3. Spline Interpolation

### Piecewise Linear

Connect points with straight lines.

$$S(x) = y_i + \frac{y_{i+1} - y_i}{x_{i+1} - x_i}(x - x_i), \quad x \in [x_i, x_{i+1}]$$

**Continuous but not smooth** ($C^0$ only).

### Cubic Splines

On each interval $[x_i, x_{i+1}]$, use cubic polynomial:

$$S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3$$

**Conditions** (for $n+1$ points, $n$ intervals):

1. **Interpolation**: $S_i(x_i) = y_i$, $S_i(x_{i+1}) = y_{i+1}$
2. **Continuity**: $S_i(x_{i+1}) = S_{i+1}(x_{i+1})$
3. **Smooth first derivative**: $S'_i(x_{i+1}) = S'_{i+1}(x_{i+1})$
4. **Smooth second derivative**: $S''_i(x_{i+1}) = S''_{i+1}(x_{i+1})$

This gives $4n$ unknowns and $4n - 2$ equations.

**Boundary Conditions** (need 2 more):

- **Natural**: $S''(x_0) = S''(x_n) = 0$
- **Clamped**: $S'(x_0) = f'(x_0)$, $S'(x_n) = f'(x_n)$
- **Not-a-knot**: Third derivative continuous at $x_1$, $x_{n-1}$

### Properties of Cubic Splines

1. $C^2$ continuous (smooth)
2. Minimizes $\int [S''(x)]^2 dx$ (minimum curvature)
3. Local control: changing one point affects only nearby segments
4. No Runge's phenomenon

---

## 4. B-Splines

### Definition

B-splines are basis functions for spline spaces.

**Order $k$ B-spline** defined recursively:

$$B_{i,1}(x) = \begin{cases} 1 & t_i \leq x < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$B_{i,k}(x) = \frac{x - t_i}{t_{i+k-1} - t_i} B_{i,k-1}(x) + \frac{t_{i+k} - x}{t_{i+k} - t_{i+1}} B_{i+1,k-1}(x)$$

### Properties

1. **Local support**: $B_{i,k}(x) = 0$ outside $[t_i, t_{i+k}]$
2. **Non-negative**: $B_{i,k}(x) \geq 0$
3. **Partition of unity**: $\sum_i B_{i,k}(x) = 1$
4. **Smoothness**: $C^{k-2}$ continuous

### Spline Representation

Any spline can be written as:

$$S(x) = \sum_i c_i B_{i,k}(x)$$

**Advantages**:

- Numerically stable
- Local control via coefficients $c_i$
- Efficient evaluation and differentiation

---

## 5. Radial Basis Functions (RBF)

### Definition

Interpolant:
$$s(x) = \sum_{i=1}^{n} c_i \phi(\|x - x_i\|)$$

where $\phi$ is a radial function.

### Common RBF Types

| Name         | $\phi(r)$                             | Properties             |
| ------------ | ------------------------------------- | ---------------------- |
| Gaussian     | $e^{-(\epsilon r)^2}$                 | Smooth, localized      |
| Multiquadric | $\sqrt{1 + (\epsilon r)^2}$           | Global, smooth         |
| Inverse MQ   | $\frac{1}{\sqrt{1 + (\epsilon r)^2}}$ | Localized              |
| Thin-plate   | $r^2 \log r$                          | Minimum bending energy |
| Polyharmonic | $r^{2k+1}$ or $r^{2k}\log r$          | Various smoothness     |

### Interpolation System

Finding coefficients $c_i$:

$$\Phi c = y$$

where $\Phi_{ij} = \phi(\|x_i - x_j\|)$.

**Condition**: $\Phi$ must be invertible (depends on $\phi$ choice).

### Connection to Neural Networks

RBF networks: Hidden layer computes $\phi(\|x - c_i\|)$

```
Input → RBF Layer → Linear Output
         ↓
    φ(||x - c₁||)
    φ(||x - c₂||)
         ⋮
    φ(||x - cₙ||)
```

---

## 6. Least Squares Approximation

### When interpolation is too much

Given noisy data, exact interpolation may overfit.

### Polynomial Least Squares

Find degree-$m$ polynomial minimizing:

$$\min_{a_0,\ldots,a_m} \sum_{i=0}^{n} (P(x_i) - y_i)^2$$

where $m < n$ (more data than parameters).

### Normal Equations

$$X^T X a = X^T y$$

where Vandermonde matrix:

$$
X = \begin{bmatrix} 1 & x_0 & x_0^2 & \cdots & x_0^m \\
1 & x_1 & x_1^2 & \cdots & x_1^m \\
\vdots \\
1 & x_n & x_n^2 & \cdots & x_n^m \end{bmatrix}
$$

**Warning**: Vandermonde matrices are ill-conditioned for high $m$.

### Orthogonal Polynomials

Use orthogonal basis (Legendre, Chebyshev) for better conditioning.

**Orthogonality**:
$$\langle P_i, P_j \rangle = \int_a^b P_i(x) P_j(x) w(x) dx = 0, \quad i \neq j$$

**Result**: Normal equations become diagonal!

---

## 7. Fourier Approximation

### Fourier Series

For periodic function on $[-\pi, \pi]$:

$$f(x) \approx \frac{a_0}{2} + \sum_{k=1}^{n} (a_k \cos kx + b_k \sin kx)$$

Coefficients:
$$a_k = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \cos kx \, dx$$
$$b_k = \frac{1}{\pi} \int_{-\pi}^{\pi} f(x) \sin kx \, dx$$

### Discrete Fourier Transform (DFT)

For sampled data:

$$X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i kn/N}$$

**Inverse**:
$$x_n = \frac{1}{N} \sum_{k=0}^{N-1} X_k e^{2\pi i kn/N}$$

### Fast Fourier Transform (FFT)

Compute DFT in $O(N \log N)$ instead of $O(N^2)$.

### Applications in ML

- Signal processing
- Convolutional neural networks (convolution in frequency domain)
- Positional encodings in Transformers

---

## 8. Approximation Theory

### Weierstrass Approximation Theorem

Any continuous function on $[a,b]$ can be uniformly approximated by polynomials.

### Stone-Weierstrass Theorem

Generalizes to algebras of continuous functions.

### Best Approximation

**Minimax**: $\min_{p \in P_n} \max_{x \in [a,b]} |f(x) - p(x)|$

**Chebyshev polynomials** give optimal approximation for polynomials.

### Neural Networks as Approximators

**Universal Approximation Theorem**: Neural networks with one hidden layer can approximate any continuous function (given enough neurons).

**Connections**:

- Sigmoid networks ↔ B-splines
- ReLU networks ↔ Piecewise linear
- Fourier features ↔ Trigonometric approximation

---

## 9. Multi-dimensional Interpolation

### Tensor Product

For 2D: $S(x,y) = \sum_i \sum_j c_{ij} B_i(x) B_j(y)$

**Curse of dimensionality**: $n^d$ basis functions for $d$ dimensions.

### Scattered Data

**Triangulation-based**:

1. Delaunay triangulation
2. Linear interpolation on each triangle

**RBF for scattered data**:
$$s(x) = \sum_i c_i \phi(\|x - x_i\|_2)$$

Works in any dimension naturally.

### Sparse Grids

Combat curse of dimensionality with sparse tensor products.

**Reduction**: $O(N^d) \to O(N (\log N)^{d-1})$

---

## 10. ML Connections

### Neural Network Initialization

Xavier/He initialization relates to approximation theory.

### Positional Encoding (Transformers)

$$PE(pos, 2i) = \sin(pos / 10000^{2i/d})$$
$$PE(pos, 2i+1) = \cos(pos / 10000^{2i/d})$$

This is Fourier feature approximation!

### Kernel Methods

Kernel $K(x, x') = \sum_i \phi_i(x)\phi_i(x')$

RBF kernel: $K(x,x') = \exp(-\gamma\|x-x'\|^2)$

### Attention as Interpolation

Attention weights interpolate value vectors:
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Similar to RBF interpolation with learned basis.

---

## Summary

### Method Selection

| Scenario               | Method                  |
| ---------------------- | ----------------------- |
| Few smooth data points | Polynomial              |
| Many data points       | Splines                 |
| High dimensions        | RBF                     |
| Noisy data             | Least squares           |
| Periodic data          | Fourier                 |
| Neural networks        | ReLU = piecewise linear |

### Key Insights

1. **Polynomial degree ≠ accuracy**: High degree can oscillate
2. **Splines balance smoothness and flexibility**
3. **RBF naturally extends to high dimensions**
4. **Neural networks are universal approximators**
5. **Fourier features capture periodic structure**

---

## Exercises

### Exercise 1: Polynomial Interpolation
Implement Lagrange and Newton interpolation. Compare numerical stability for high-degree polynomials.

### Exercise 2: Runge's Phenomenon
Demonstrate Runge's phenomenon with equispaced nodes. Show how Chebyshev nodes mitigate the problem.

### Exercise 3: Spline Interpolation
Implement cubic spline interpolation with natural boundary conditions. Apply to 2D curve fitting.

### Exercise 4: Least Squares Approximation
Fit polynomials of varying degrees to noisy data. Analyze the bias-variance tradeoff.

### Exercise 5: RBF Interpolation
Implement radial basis function interpolation with Gaussian kernels. Investigate the effect of shape parameter.

---

## References

1. Burden & Faires - "Numerical Analysis"
2. de Boor - "A Practical Guide to Splines"
3. Powell - "Approximation Theory and Methods"
4. Trefethen - "Approximation Theory and Approximation Practice"
