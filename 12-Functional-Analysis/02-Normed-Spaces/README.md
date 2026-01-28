# Normed Spaces for Machine Learning

## Overview

Normed spaces extend vector spaces with a notion of "size" or "length". They provide the foundation for analyzing convergence, continuity, and approximation in machine learning.

## 1. Definition and Axioms

### Norm

A **norm** on vector space $V$ is a function $\|\cdot\|: V \to \mathbb{R}$ satisfying:

| Axiom | Property                                                           | Meaning             |
| ----- | ------------------------------------------------------------------ | ------------------- | --------------- | ----------- |
| N1    | $\|\mathbf{v}\| \geq 0$ with $= 0$ iff $\mathbf{v} = \mathbf{0}$   | Positivity          |
| N2    | $\|c\mathbf{v}\| =                                                 | c                   | \|\mathbf{v}\|$ | Homogeneity |
| N3    | $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ | Triangle inequality |

### Normed Space

A **normed space** $(V, \|\cdot\|)$ is a vector space with a norm.

### Induced Metric

Every norm induces a metric:
$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|$$

## 2. Common Norms in ML

### $\ell^p$ Norms on $\mathbb{R}^n$

$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}, \quad p \geq 1$$

| Norm          | Formula               | Properties                    |
| ------------- | --------------------- | ----------------------------- |
| $\ell^1$      | $\sum_i \|x_i\|$      | Sparsity-inducing             |
| $\ell^2$      | $\sqrt{\sum_i x_i^2}$ | Euclidean, rotation invariant |
| $\ell^\infty$ | $\max_i \|x_i\|$      | Captures worst case           |

### $\ell^0$ "Norm"

$$\|\mathbf{x}\|_0 = |\{i : x_i \neq 0\}|$$

**Note**: Not actually a norm (not homogeneous), but counts non-zeros.

### Matrix Norms

**Frobenius norm**:
$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\text{tr}(A^T A)}$$

**Spectral (operator) norm**:
$$\|A\|_2 = \max_{\|\mathbf{x}\|_2 = 1} \|A\mathbf{x}\|_2 = \sigma_{\max}(A)$$

**Nuclear norm** (trace norm):
$$\|A\|_* = \sum_i \sigma_i(A)$$

## 3. Relationships Between Norms

### Norm Equivalence

In finite dimensions, all norms are **equivalent**:
$$\exists c_1, c_2 > 0: \quad c_1 \|\mathbf{x}\|_a \leq \|\mathbf{x}\|_b \leq c_2 \|\mathbf{x}\|_a$$

### Specific Bounds

For $\mathbf{x} \in \mathbb{R}^n$:
$$\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq \sqrt{n} \|\mathbf{x}\|_2$$
$$\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n} \|\mathbf{x}\|_\infty$$
$$\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_1 \leq n \|\mathbf{x}\|_\infty$$

### Norm Balls

The unit ball $B_p = \{\mathbf{x} : \|\mathbf{x}\|_p \leq 1\}$:

- $p = 1$: Diamond (cross-polytope)
- $p = 2$: Sphere
- $p = \infty$: Cube
- $p < 1$: Non-convex (not a norm!)

## 4. Convergence in Normed Spaces

### Sequence Convergence

$\mathbf{x}_n \to \mathbf{x}$ if $\|\mathbf{x}_n - \mathbf{x}\| \to 0$

### Cauchy Sequences

$(\mathbf{x}_n)$ is **Cauchy** if:
$$\forall \epsilon > 0, \exists N: m, n > N \Rightarrow \|\mathbf{x}_m - \mathbf{x}_n\| < \epsilon$$

### Completeness

A normed space is **complete** if every Cauchy sequence converges.

**Banach space** = Complete normed space

### Examples

| Space          | Norm         | Complete?  |
| -------------- | ------------ | ---------- |
| $\mathbb{R}^n$ | Any $\ell^p$ | ✓ (Banach) |
| $C[0,1]$       | Sup norm     | ✓ (Banach) |
| $\mathbb{Q}^n$ | $\ell^2$     | ✗          |

## 5. Continuity and Lipschitz Functions

### Continuity

$f: V \to W$ is **continuous** at $\mathbf{x}_0$ if:
$$\mathbf{x}_n \to \mathbf{x}_0 \Rightarrow f(\mathbf{x}_n) \to f(\mathbf{x}_0)$$

### Lipschitz Continuity

$f$ is **$L$-Lipschitz** if:
$$\|f(\mathbf{x}) - f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|$$

**ML Applications**:

- Lipschitz constraints for robust networks
- Spectral normalization: $\|W\|_2 \leq 1$
- WGAN: Lipschitz discriminator

### Operator Norms

For linear $T: V \to W$:
$$\|T\| = \sup_{\|\mathbf{v}\| = 1} \|T\mathbf{v}\| = \sup_{\mathbf{v} \neq 0} \frac{\|T\mathbf{v}\|}{\|\mathbf{v}\|}$$

$T$ is Lipschitz with constant $L = \|T\|$.

## 6. Bounded Linear Operators

### Definition

$T: V \to W$ is **bounded** if:
$$\exists C: \|T\mathbf{v}\| \leq C\|\mathbf{v}\| \quad \forall \mathbf{v} \in V$$

### Key Theorem

For linear operators: bounded $\Leftrightarrow$ continuous

### Space of Bounded Operators

$\mathcal{B}(V, W) = \{T: V \to W : T \text{ is bounded linear}\}$

With operator norm, $\mathcal{B}(V, W)$ is a normed space.

If $W$ is Banach, so is $\mathcal{B}(V, W)$.

## 7. Dual Spaces and Dual Norms

### Dual Space

$$V^* = \{f: V \to \mathbb{R} : f \text{ is bounded linear}\}$$

### Dual Norm

$$\|\mathbf{y}\|_* = \sup_{\|\mathbf{x}\| \leq 1} |\langle \mathbf{y}, \mathbf{x} \rangle|$$

### $\ell^p$ Duals

| Primal $\ell^p$ | Dual $\ell^q$ | Relationship                    |
| --------------- | ------------- | ------------------------------- |
| $\ell^1$        | $\ell^\infty$ | $1 + \frac{1}{\infty} = 1$      |
| $\ell^2$        | $\ell^2$      | Self-dual                       |
| $\ell^p$        | $\ell^q$      | $\frac{1}{p} + \frac{1}{q} = 1$ |

### Hölder's Inequality

$$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\|_p \|\mathbf{y}\|_q$$

where $\frac{1}{p} + \frac{1}{q} = 1$.

## 8. Regularization and Norms

### Norm-Based Regularization

$$\min_\theta L(\theta) + \lambda \|\theta\|_p$$

| Regularizer              | Effect            | Sparsity |
| ------------------------ | ----------------- | -------- |
| $\|\theta\|_2^2$ (Ridge) | Shrinks all       | No       |
| $\|\theta\|_1$ (Lasso)   | Shrinks and zeros | Yes      |
| Elastic Net              | Both              | Moderate |

### Why $\ell^1$ Induces Sparsity

At $\theta_i = 0$, the $\ell^1$ gradient is the subgradient $[-1, 1]$.
This allows solutions exactly at zero.

For $\ell^2$: gradient at $\theta_i = 0$ is $0$, so no "pull" toward zero.

### Nuclear Norm Regularization

$$\min_X \|Y - X\|_F^2 + \lambda \|X\|_*$$

Encourages low-rank solutions (matrix completion, recommender systems).

## 9. Compactness in Normed Spaces

### Compact Sets

$K \subseteq V$ is **compact** if every sequence in $K$ has a convergent subsequence.

### Finite Dimensions

In $\mathbb{R}^n$: $K$ compact $\Leftrightarrow$ $K$ closed and bounded (Heine-Borel)

### Infinite Dimensions

**NOT** true in infinite dimensions! Closed and bounded $\not\Rightarrow$ compact.

**Example**: Unit ball in $\ell^2$ is closed and bounded but not compact.

### Compactness for Optimization

Continuous function on compact set attains min/max (Weierstrass).

## 10. Function Spaces

### $L^p$ Spaces

$$L^p(\Omega) = \left\{ f: \Omega \to \mathbb{R} : \int_\Omega |f|^p < \infty \right\}$$

$$\|f\|_{L^p} = \left( \int_\Omega |f(x)|^p dx \right)^{1/p}$$

### $C[a, b]$ with Sup Norm

$$\|f\|_\infty = \sup_{x \in [a, b]} |f(x)|$$

This is a Banach space.

### Sobolev Spaces

$$W^{k,p}(\Omega) = \{f : D^\alpha f \in L^p, |\alpha| \leq k\}$$

Important for understanding neural network smoothness.

## 11. Approximation in Normed Spaces

### Best Approximation

Given closed subspace $W \subseteq V$, the **best approximation** to $\mathbf{v}$ in $W$:
$$\hat{\mathbf{w}} = \arg\min_{\mathbf{w} \in W} \|\mathbf{v} - \mathbf{w}\|$$

### Existence

In any normed space with $W$ closed: best approximation exists if $V$ is reflexive.

In Hilbert spaces: always unique (orthogonal projection).

### Approximation in ML

- Neural networks approximate functions in $L^p$
- Universal approximation theorems
- Depth vs width trade-offs

## 12. Fixed Point Theorems

### Banach Fixed Point Theorem

Let $(X, d)$ be complete metric space and $T: X \to X$ be a **contraction**:
$$d(Tx, Ty) \leq \alpha \cdot d(x, y), \quad \alpha < 1$$

Then $T$ has unique fixed point $x^* = Tx^*$.

**Iteration converges**: $x_n = T(x_{n-1}) \to x^*$

### Applications in ML

- **Iterative algorithms**: Convergence proofs
- **Bellman operator**: Reinforcement learning
- **Self-consistent equations**: Mean-field models

### Convergence Rate

$$d(x_n, x^*) \leq \frac{\alpha^n}{1 - \alpha} d(x_1, x_0)$$

Linear convergence with rate $\alpha$.

## 13. Spectral Theory Basics

### Spectrum of Operator

For $T: V \to V$, the **spectrum**:
$$\sigma(T) = \{\lambda : (T - \lambda I) \text{ not invertible}\}$$

### Spectral Radius

$$\rho(T) = \sup\{|\lambda| : \lambda \in \sigma(T)\}$$

**Key relation**: $\rho(T) \leq \|T\|$ for any operator norm.

### Power Series

If $\|T\| < 1$, then:
$$(I - T)^{-1} = \sum_{n=0}^{\infty} T^n$$

(Neumann series)

## Key Inequalities

### Minkowski's Inequality

$$\|\mathbf{x} + \mathbf{y}\|_p \leq \|\mathbf{x}\|_p + \|\mathbf{y}\|_p$$

(Triangle inequality for $\ell^p$)

### Jensen's Inequality

For convex $\phi$:
$$\phi\left(\int f d\mu\right) \leq \int \phi(f) d\mu$$

### Cauchy-Schwarz

$$|\langle \mathbf{x}, \mathbf{y} \rangle| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$$

## Summary: Norms in ML

| Concept            | ML Application                   |
| ------------------ | -------------------------------- |
| $\ell^1$ norm      | Sparse models, feature selection |
| $\ell^2$ norm      | Weight decay, Lipschitz bounds   |
| $\ell^\infty$ norm | Adversarial robustness           |
| Operator norm      | Spectral normalization           |
| Nuclear norm       | Low-rank matrix completion       |
| Lipschitz          | Stability, generalization        |
| Completeness       | Convergence guarantees           |

## Important Theorems

1. **Norm equivalence**: All norms on $\mathbb{R}^n$ are equivalent
2. **Banach fixed point**: Contractions have unique fixed points
3. **Bounded = continuous** for linear operators
4. **Dual of $\ell^p$ is $\ell^q$** where $1/p + 1/q = 1$
5. **Weierstrass**: Continuous function on compact set has min/max
