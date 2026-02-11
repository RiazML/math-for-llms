# Hilbert Spaces

[← Previous: Normed Spaces](../02-Normed-Spaces) | [Next: Kernel Methods →](../04-Kernel-Methods)

## Overview

Hilbert spaces are complete inner product spaces that form the mathematical foundation for quantum mechanics, signal processing, and machine learning. They generalize Euclidean spaces to infinite dimensions while preserving the geometric intuition of angles and projections.

## Why This Matters for Machine Learning

Hilbert spaces are where the "magic" of kernel methods happens. When you use an RBF kernel, you're implicitly working in an infinite-dimensional Hilbert space where your data becomes linearly separable. This isn't just a mathematical curiosity—it's why SVMs can learn complex nonlinear decision boundaries using only linear algebra.

The inner product structure of Hilbert spaces gives us something precious: a notion of angle and orthogonality. This is why attention mechanisms work. The dot product between queries and keys measures "alignment" or "relevance"—a concept that only makes sense because we're working in an inner product space. Cosine similarity, the workhorse of semantic search and recommendation systems, is fundamentally a Hilbert space concept.

The Reproducing Kernel Hilbert Space (RKHS) framework unifies much of modern ML theory. It explains why Gaussian processes work, provides the mathematical foundation for the neural tangent kernel (showing that infinite-width networks are linear in function space), and gives us the representer theorem—guaranteeing that the solution to any regularized learning problem is a finite linear combination of kernel evaluations. Understanding Hilbert spaces reveals the hidden structure connecting these seemingly disparate ideas.

## Chapter Roadmap

- **Section 1-2**: Foundations—inner products, induced norms, and the definition of Hilbert spaces with examples
- **Section 3-5**: Geometry—orthogonality, orthonormal bases, Parseval's identity, and the projection theorem
- **Section 6-7**: Duality—Riesz representation theorem and Fourier analysis as an application
- **Section 8-9**: RKHS—reproducing kernels, the Moore-Aronszajn theorem, and kernel methods
- **Section 10-11**: Operators—gradient flows, compact operators, and Mercer's theorem
- **Section 12**: Applications—Gaussian processes, attention, and infinite-width networks

## Files in This Section

| File | Description |
|------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

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

> 💡 **Insight:** The key difference between a Hilbert space and a mere normed space is the inner product—and therefore angles. This is why attention works! Without an inner product, we couldn't compute "similarity" between query and key vectors. The softmax attention score is fundamentally $\cos(\theta) \cdot \|q\| \cdot \|k\|$ (scaled dot product), measuring both alignment and magnitude.

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

> 💡 **Insight:** The Riesz representation theorem is secretly why neural networks can learn. It says every linear functional can be represented as an inner product with some vector. When a neural network learns to predict a scalar output from a vector input (like sentiment from an embedding), it's finding the Riesz representative—the vector that, when dotted with inputs, gives the right predictions.

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

> 💡 **Insight:** The neural tangent kernel (NTK) reveals that sufficiently wide neural networks behave like kernel machines! During training, the network function evolves in an RKHS defined by the NTK. This explains why wide networks generalize despite having more parameters than training examples—they're implicitly regularized by the RKHS norm, which favors smooth functions.

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

## Key Takeaways

- **Inner products give us angles and similarity**: Unlike general norms, Hilbert space norms come from inner products, enabling cosine similarity and attention mechanisms.

- **The projection theorem is the foundation of least squares**: Every least squares problem is finding the orthogonal projection onto the column space—the closest point in a subspace.

- **RKHS = functions that can be evaluated pointwise**: The defining property of RKHS is that evaluation $f \mapsto f(x)$ is continuous, which is surprisingly restrictive and powerful.

- **Kernels are infinite-dimensional inner products**: The kernel trick computes inner products in (potentially infinite-dimensional) feature space without ever constructing the features.

- **The representer theorem makes kernel methods tractable**: No matter how complex the RKHS, the optimal solution is always a finite linear combination of kernel evaluations at training points.

- **Wide neural networks are secretly kernel machines**: The NTK shows that training infinitely wide networks is equivalent to kernel regression, bridging deep learning and classical ML.

- **Fourier analysis is Hilbert space theory**: The Fourier basis is just an orthonormal basis for $L^2$, and Parseval's identity is just Pythagoras in infinite dimensions.

## Key Theorems

1. **Projection**: Unique closest point in closed convex set
2. **Riesz**: Functionals ↔ vectors
3. **Representer**: Solutions are kernel expansions
4. **Mercer**: Kernel eigendecomposition
5. **Parseval**: $\|f\|^2 = \sum |\langle f, e_n \rangle|^2$

## Exercises

1. **Cauchy-Schwarz Application**: Prove that for any vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, the cosine similarity $\cos(\theta) = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\| \|\mathbf{y}\|}$ satisfies $|\cos(\theta)| \leq 1$. How is this used in attention mechanisms?

2. **Orthogonal Projection**: Given the subspace $W = \text{span}\{(1, 1, 0), (0, 1, 1)\}$ in $\mathbb{R}^3$, compute the orthogonal projection of $\mathbf{v} = (1, 2, 3)$ onto $W$ using the Gram-Schmidt process. Verify your answer satisfies $\mathbf{v} - P_W\mathbf{v} \perp W$.

3. **Parseval's Identity**: For the orthonormal basis $\{\mathbf{e}_1, \mathbf{e}_2, \mathbf{e}_3\}$ in $\mathbb{R}^3$, verify Parseval's identity for $\mathbf{v} = (3, 4, 0)$ by showing $\|\mathbf{v}\|^2 = \sum_i |\langle \mathbf{v}, \mathbf{e}_i \rangle|^2$.

4. **RKHS and Reproducing Property**: For the linear kernel $K(x, y) = x^T y$ on $\mathbb{R}^d$, show that the RKHS is $\mathbb{R}^d$ itself. Verify the reproducing property $f(x) = \langle f, K(\cdot, x) \rangle$ for a linear function $f(\mathbf{z}) = \mathbf{w}^T \mathbf{z}$.

5. **Kernel Ridge Regression**: Derive the closed-form solution $\alpha = (K + \lambda I)^{-1} \mathbf{y}$ for kernel ridge regression by applying the representer theorem. Show that this is equivalent to the projection interpretation in Hilbert space.

## References

- Kreyszig, "Introductory Functional Analysis with Applications"
- Berlinet & Thomas-Agnan, "Reproducing Kernel Hilbert Spaces"
- Schölkopf & Smola, "Learning with Kernels"
- Cucker & Smale, "On the Mathematical Foundations of Learning"
