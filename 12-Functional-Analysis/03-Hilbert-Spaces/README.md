# Hilbert Spaces

## Overview

Hilbert spaces are complete inner product spaces that form the mathematical foundation for quantum mechanics, signal processing, and machine learning. They generalize Euclidean spaces to infinite dimensions while preserving the geometric intuition of angles and projections.

## Prerequisites

- Linear algebra fundamentals
- Vector spaces
- Inner products
- Normed spaces and completeness

## Learning Objectives

1. Understand inner products and induced norms
2. Master orthogonality and orthonormal bases
3. Apply projection theorems
4. Work with Fourier analysis and orthogonal expansions
5. Understand reproducing kernel Hilbert spaces (RKHS)

---

## 1. Inner Product Spaces

### Definition

An **inner product** on a vector space $V$ over $\mathbb{R}$ (or $\mathbb{C}$) is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ (or $\mathbb{C}$) satisfying:

1. **Conjugate symmetry**: $\langle x, y \rangle = \overline{\langle y, x \rangle}$
2. **Linearity in first argument**: $\langle ax + by, z \rangle = a\langle x, z \rangle + b\langle y, z \rangle$
3. **Positive definiteness**: $\langle x, x \rangle \geq 0$ with equality iff $x = 0$

### Common Inner Products

**Euclidean**: $\langle x, y \rangle = \sum_{i=1}^n x_i y_i = x^T y$

**Weighted**: $\langle x, y \rangle_W = x^T W y$ for positive definite $W$

**$L^2[a,b]$**: $\langle f, g \rangle = \int_a^b f(x) \overline{g(x)} \, dx$

**$\ell^2$** (sequences): $\langle (a_n), (b_n) \rangle = \sum_{n=1}^\infty a_n \overline{b_n}$

### Induced Norm

Every inner product induces a norm:
$$\|x\| = \sqrt{\langle x, x \rangle}$$

### Cauchy-Schwarz Inequality

$$|\langle x, y \rangle| \leq \|x\| \|y\|$$

Equality holds iff $x$ and $y$ are linearly dependent.

### Parallelogram Law

$$\|x + y\|^2 + \|x - y\|^2 = 2(\|x\|^2 + \|y\|^2)$$

A norm satisfies the parallelogram law iff it comes from an inner product.

---

## 2. Hilbert Spaces

### Definition

A **Hilbert space** is a complete inner product space:

- Every Cauchy sequence converges
- The limit is in the space

### Examples

**Finite-dimensional**: $\mathbb{R}^n$ with standard inner product

**$L^2(\Omega)$**: Square-integrable functions
$$L^2(\Omega) = \{f : \int_\Omega |f(x)|^2 dx < \infty\}$$

**$\ell^2$**: Square-summable sequences
$$\ell^2 = \{(a_n) : \sum_{n=1}^\infty |a_n|^2 < \infty\}$$

### ML Relevance

- Feature spaces in kernel methods
- Function spaces for infinite-width neural networks
- Gaussian processes live in RKHS

---

## 3. Orthogonality

### Orthogonal Vectors

$x \perp y$ iff $\langle x, y \rangle = 0$

### Orthogonal Complement

$$M^\perp = \{x \in H : \langle x, y \rangle = 0 \text{ for all } y \in M\}$$

### Properties

For closed subspace $M$:

- $H = M \oplus M^\perp$ (direct sum)
- $(M^\perp)^\perp = M$
- $\dim(M) + \dim(M^\perp) = \dim(H)$ (finite dim)

### Pythagorean Theorem

If $x \perp y$:
$$\|x + y\|^2 = \|x\|^2 + \|y\|^2$$

---

## 4. Orthonormal Systems

### Definition

A set $\{e_\alpha\}$ is **orthonormal** if:

- $\langle e_\alpha, e_\beta \rangle = 0$ for $\alpha \neq \beta$
- $\|e_\alpha\| = 1$

### Orthonormal Basis (Complete System)

An orthonormal set is a **basis** if its span is dense in $H$.

### Bessel's Inequality

For any orthonormal set $\{e_n\}$:
$$\sum_n |\langle x, e_n \rangle|^2 \leq \|x\|^2$$

### Parseval's Identity

For an orthonormal basis:
$$\|x\|^2 = \sum_n |\langle x, e_n \rangle|^2$$

### Gram-Schmidt Orthonormalization

Given linearly independent $\{v_1, ..., v_n\}$:
$$u_k = v_k - \sum_{j=1}^{k-1} \langle v_k, e_j \rangle e_j, \quad e_k = \frac{u_k}{\|u_k\|}$$

---

## 5. Projection Theorem

### Theorem (Closest Point)

Let $M$ be a closed convex subset of Hilbert space $H$. For any $x \in H$, there exists a unique $y \in M$ such that:
$$\|x - y\| = \inf_{z \in M} \|x - z\|$$

### Orthogonal Projection

For closed subspace $M$, the **orthogonal projection** $P_M: H \to M$:
$$P_M x = \arg\min_{y \in M} \|x - y\|$$

Characterized by: $x - P_M x \perp M$

### Properties

- $P_M^2 = P_M$ (idempotent)
- $P_M^* = P_M$ (self-adjoint)
- $\|P_M\| = 1$ (unless $M = \{0\}$)
- $I - P_M = P_{M^\perp}$

### ML Application: Least Squares

Find $\hat{x}$ minimizing $\|Ax - b\|^2$:

The solution satisfies: $A^T(A\hat{x} - b) = 0$

So: $\hat{x} = (A^T A)^{-1} A^T b$ (when $A^T A$ invertible)

This is projection of $b$ onto column space of $A$.

---

## 6. Riesz Representation Theorem

### Theorem

Every continuous linear functional $f: H \to \mathbb{R}$ on a Hilbert space can be represented as:
$$f(x) = \langle x, y_f \rangle$$
for a unique $y_f \in H$ with $\|f\| = \|y_f\|$.

### Consequences

- $H^* \cong H$ (Hilbert space is self-dual)
- Identifies functionals with vectors
- Enables kernel trick in ML

---

## 7. Fourier Analysis

### Fourier Series

For $L^2[0, 2\pi]$, orthonormal basis:
$$e_n(x) = \frac{1}{\sqrt{2\pi}} e^{inx}, \quad n \in \mathbb{Z}$$

Any $f \in L^2$ has expansion:
$$f = \sum_{n=-\infty}^{\infty} \hat{f}_n e_n$$

where $\hat{f}_n = \langle f, e_n \rangle$ (Fourier coefficients).

### Parseval's Theorem

$$\|f\|^2 = \sum_{n=-\infty}^{\infty} |\hat{f}_n|^2$$

### ML Applications

- Spectral analysis of signals
- Fourier features for kernel approximation
- Frequency-domain analysis of neural networks

---

## 8. Reproducing Kernel Hilbert Spaces (RKHS)

### Definition

A Hilbert space $H$ of functions $f: X \to \mathbb{R}$ is an **RKHS** if evaluation functionals are continuous:
$$|f(x)| \leq C_x \|f\|_H$$

### Reproducing Kernel

By Riesz theorem, exists unique $K_x \in H$ such that:
$$f(x) = \langle f, K_x \rangle_H$$

The **kernel** is $K(x, y) = \langle K_x, K_y \rangle_H = K_y(x)$

### Properties

- $K$ is positive semi-definite
- **Reproducing property**: $f(x) = \langle f, K(\cdot, x) \rangle$
- **Moore-Aronszajn**: Every PSD kernel defines unique RKHS

### Common Kernels and RKHS

**Linear**: $K(x, y) = x^T y$ → $H = \mathbb{R}^n$

**Polynomial**: $K(x, y) = (x^T y + c)^d$ → polynomial functions

**RBF/Gaussian**: $K(x, y) = \exp(-\|x-y\|^2/2\sigma^2)$ → smooth functions

**Laplacian**: $K(x, y) = \exp(-\|x-y\|_1/\sigma)$ → less smooth

---

## 9. Kernel Methods

### Representer Theorem

For regularized learning:
$$\min_{f \in H} \sum_{i=1}^n L(y_i, f(x_i)) + \lambda \|f\|_H^2$$

The solution has form:
$$f^*(x) = \sum_{i=1}^n \alpha_i K(x_i, x)$$

### Kernel Ridge Regression

$$f^*(x) = K_{x*}^T (K + \lambda I)^{-1} y$$

where $K_{ij} = K(x_i, x_j)$.

### Support Vector Machines

Decision function in RKHS:
$$f(x) = \sum_{i \in SV} \alpha_i y_i K(x_i, x) + b$$

### Kernel PCA

Eigendecomposition in feature space via kernel:
$$K \alpha_k = \lambda_k n \alpha_k$$

Principal components: $\phi_k(x) = \sum_i \alpha_{ki} K(x_i, x)$

---

## 10. Gradient Flows in Hilbert Space

### Gradient in Hilbert Space

For functional $J: H \to \mathbb{R}$, gradient $\nabla J(f) \in H$:
$$\langle \nabla J(f), g \rangle = \lim_{t \to 0} \frac{J(f + tg) - J(f)}{t}$$

### Gradient Flow

$$\frac{df}{dt} = -\nabla J(f)$$

### Neural Tangent Kernel

Infinite-width neural networks: training dynamics are linear in function space with kernel:
$$K_{NTK}(x, x') = \nabla_\theta f(x; \theta)^T \nabla_\theta f(x'; \theta)$$

---

## 11. Compact Operators

### Definition

Operator $T: H \to H$ is **compact** if it maps bounded sets to precompact sets.

### Spectral Theorem for Compact Self-Adjoint Operators

If $T$ is compact and self-adjoint:
$$T = \sum_{n=1}^\infty \lambda_n \langle \cdot, e_n \rangle e_n$$

where $\lambda_n \to 0$ and $\{e_n\}$ are orthonormal eigenvectors.

### Integral Operators

$$Tf(x) = \int K(x, y) f(y) dy$$

Kernel $K$ defines compact operator; eigenfunctions are principal components of kernel.

### Mercer's Theorem

For continuous PSD kernel $K$ on compact domain:
$$K(x, y) = \sum_{n=1}^\infty \lambda_n \phi_n(x) \phi_n(y)$$

where $\phi_n$ are eigenfunctions, $\lambda_n > 0$.

---

## 12. Applications in ML

### Gaussian Processes

A GP defines a distribution over functions in RKHS:

- Mean function: $m(x) = \mathbb{E}[f(x)]$
- Covariance: $K(x, x') = \mathbb{E}[(f(x)-m(x))(f(x')-m(x'))]$

### Optimal Transport

Wasserstein distance uses $L^2$ structure for displacement interpolation.

### Attention Mechanisms

Softmax attention can be viewed as kernel smoothing:
$$\text{Attention}(Q, K, V) \approx \text{kernel weighted average}$$

### Infinite-Width Networks

Neural networks at infinite width converge to GPs with specific kernels determined by architecture.

---

## Summary

| Concept               | Definition                      | ML Application        |
| --------------------- | ------------------------------- | --------------------- |
| Inner product         | $\langle x, y \rangle$          | Similarity measures   |
| Orthogonal projection | $P_M x$                         | Least squares         |
| RKHS                  | Function space with kernel      | Kernel methods        |
| Reproducing kernel    | $f(x) = \langle f, K_x \rangle$ | SVM, GP               |
| Spectral theorem      | Eigendecomposition              | Kernel PCA            |
| NTK                   | $K_{NTK}$                       | Neural network theory |

## Key Theorems

1. **Projection**: Unique closest point in closed convex set
2. **Riesz**: Functionals ↔ vectors
3. **Representer**: Solutions are kernel expansions
4. **Mercer**: Kernel eigendecomposition
5. **Parseval**: $\|f\|^2 = \sum |\langle f, e_n \rangle|^2$

## References

- Kreyszig, "Introductory Functional Analysis with Applications"
- Berlinet & Thomas-Agnan, "Reproducing Kernel Hilbert Spaces"
- Schölkopf & Smola, "Learning with Kernels"
- Cucker & Smale, "On the Mathematical Foundations of Learning"
