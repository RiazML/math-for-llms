# Normed Spaces for Machine Learning

[← Previous: Vector Spaces](../01-Vector-Spaces) | [Next: Hilbert Spaces →](../03-Hilbert-Spaces)

## Overview

Normed spaces extend vector spaces with a notion of "size" or "length". They provide the foundation for analyzing convergence, continuity, and approximation in machine learning.

## Why This Matters for Machine Learning

Norms are the mathematical tools that let us measure and control in machine learning. Every time you add a regularization term to a loss function, you're using a norm. The choice between L1 and L2 regularization—perhaps the most fundamental hyperparameter decision in classical ML—is a choice between norms with profoundly different geometric and optimization properties.

The Lipschitz constant, derived from the operator norm, has become central to modern deep learning. It governs the stability of neural networks: a network with bounded Lipschitz constant can't change its output too drastically for small input perturbations. This insight underlies spectral normalization, gradient clipping, and the entire field of adversarial robustness. The $\ell^\infty$ norm measures worst-case perturbations, explaining why adversarial examples are often constructed in this norm.

The Banach fixed-point theorem—a cornerstone result about complete normed spaces—explains why iterative algorithms converge. From the Bellman equation in reinforcement learning to iterative refinement in diffusion models, contractions in normed spaces guarantee we reach a unique solution. Understanding norms transforms regularization from a "trick that works" into a principled choice with geometric meaning.

## Chapter Roadmap

- **Section 1-2**: Foundations—norm axioms and the complete catalog of norms used in ML ($\ell^p$, matrix norms)
- **Section 3-4**: Geometry and limits—norm equivalence, unit balls, and convergence in normed spaces
- **Section 5-6**: Continuity—Lipschitz functions, bounded operators, and the bounded-continuous equivalence
- **Section 7-8**: Duality—dual norms, Hölder's inequality, and the regularization interpretation
- **Section 9-11**: Completeness—Banach spaces, compactness, and function spaces (Lᵖ, Sobolev)
- **Section 12-13**: Fixed points and spectra—Banach fixed-point theorem and spectral radius

## Files in This Section

| File | Description |
|------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

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

> 💡 **Insight:** Gradient clipping is Lipschitz enforcement in disguise! When you clip gradients to have norm at most $C$, you're ensuring the "update function" is $C$-Lipschitz. This prevents catastrophic updates from exploding gradients and is mathematically equivalent to constraining the operator norm of the Jacobian.

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

> 💡 **Insight:** The geometry of norm balls is key to understanding regularization. The $\ell^1$ ball has sharp corners (at the axes), while the $\ell^2$ ball is smooth. When minimizing a linear objective (like a gradient step), the optimal solution tends to land on corners of the constraint set. This is why $\ell^1$ regularization produces sparse solutions—the corners of the $\ell^1$ ball are exactly the sparse points!

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

> 💡 **Insight:** The Bellman operator in reinforcement learning is a contraction in the supremum norm with factor $\gamma$ (the discount factor). This is why value iteration converges—and why $\gamma < 1$ is essential! The convergence rate $\gamma^n$ also explains why high discount factors make learning slower: the contraction is weaker, so more iterations are needed.

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

## Key Takeaways

- **Norms quantify "size" in ways that matter**: The choice of norm (L1 vs L2 vs L∞) fundamentally changes optimization behavior and solution properties.

- **L1 sparsity is geometric**: The corners of the L1 ball lie on coordinate axes—this geometry, not magic, explains why Lasso produces sparse solutions.

- **Lipschitz constants bound sensitivity**: The operator norm of a layer's weight matrix is its Lipschitz constant. Spectral normalization constrains this to 1.

- **Adversarial robustness lives in $\ell^\infty$**: The worst-case perturbation norm matches how we measure adversarial attacks—small in $\ell^\infty$ means imperceptible to humans.

- **Completeness enables convergence guarantees**: We need Banach spaces (complete normed spaces) to ensure our iterative algorithms actually converge to something in our space.

- **The Banach fixed-point theorem is everywhere**: Value iteration, policy iteration, equilibrium computation—any time you iterate a contraction, this theorem guarantees convergence.

- **Dual norms arise naturally in optimization**: The dual of $\ell^1$ is $\ell^\infty$, explaining why sparse priors (L1 regularization) lead to bounded gradients.

## Important Theorems

1. **Norm equivalence**: All norms on $\mathbb{R}^n$ are equivalent
2. **Banach fixed point**: Contractions have unique fixed points
3. **Bounded = continuous** for linear operators
4. **Dual of $\ell^p$ is $\ell^q$** where $1/p + 1/q = 1$
5. **Weierstrass**: Continuous function on compact set has min/max

## Exercises

1. **Norm Computation**: For $\mathbf{x} = (3, -4, 0, 2)$, compute $\|\mathbf{x}\|_1$, $\|\mathbf{x}\|_2$, and $\|\mathbf{x}\|_\infty$. Verify the norm inequalities $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1$.

2. **Lipschitz Constant**: Prove that a linear transformation $T: \mathbb{R}^n \to \mathbb{R}^m$ with matrix $A$ is Lipschitz continuous with constant $L = \|A\|_2$ (spectral norm). What does this imply for the stability of a neural network layer?

3. **Sparsity and $\ell^1$**: Consider minimizing $\|\mathbf{x}\|_1$ subject to $A\mathbf{x} = \mathbf{b}$. Geometrically explain why the solution tends to be sparse (lies on a vertex of the $\ell^1$ ball).

4. **Banach Fixed Point**: Apply the Banach fixed point theorem to prove that the iteration $\mathbf{x}_{k+1} = \frac{1}{2}A\mathbf{x}_k + \mathbf{b}$ converges for any starting point when $\|A\|_2 < 2$. Find the fixed point in terms of $A$ and $\mathbf{b}$.

5. **Matrix Norms**: For the matrix $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$, compute $\|A\|_F$ (Frobenius norm) and $\|A\|_*$ (nuclear norm). Explain why nuclear norm regularization encourages low-rank solutions.
