[← Back to Advanced Linear Algebra](../README.md) | [← Linear Transformations](../04-Linear-Transformations/notes.md) | [Matrix Norms →](../06-Matrix-Norms/notes.md)

---

# Orthogonality and Orthonormality

> _"Orthogonality is the single most important idea in applied mathematics. When things are perpendicular, they don't interfere with each other — and that's the key to making computation tractable."_

## Overview

Orthogonality is the mathematical formalization of "not interfering." When two vectors are orthogonal, they carry completely independent information. When a set of basis vectors is orthonormal, coordinates can be computed by simple dot products. When a matrix is orthogonal, it preserves lengths and angles — it cannot distort or amplify errors.

This section develops the full theory of orthogonality from first principles. We begin with the geometric intuition — right angles, projections, and the remarkable simplification that occurs when bases are orthonormal. We then build the formal machinery: Gram-Schmidt orthogonalization (classical and numerically stable modified form), Householder and Givens methods for QR decomposition, orthogonal projections and least squares, and the spectral theorem for symmetric matrices.

Throughout, the AI connections are direct and structural. Orthogonal weight initialization preserves gradient norms through linear networks. RoPE encodes position via rotation matrices — orthogonal transformations that preserve the inner products needed for attention. Spectral normalization constrains weight matrices to have unit spectral norm. The QR algorithm for eigenvalues underlies every numerical eigensolver. Understanding orthogonality at depth is understanding the numerical backbone of machine learning.

## Prerequisites

- Inner products and dot products — [Chapter 2 §01: Vectors and Spaces](../../02-Linear-Algebra-Basics/01-Vectors-and-Spaces/notes.md)
- Vector spaces, subspaces, dimension — [Chapter 2 §06: Vector Spaces and Subspaces](../../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md)
- Linear transformations, kernel, image — [§04: Linear Transformations](../04-Linear-Transformations/notes.md)
- Eigenvalues and eigenvectors (used in §7) — [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive exploration: Gram-Schmidt visualization, projection geometry, QR factorization, spectral theorem, orthogonal init |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from projection computation through RoPE and orthogonal initialization |

## Learning Objectives

After completing this section, you will be able to:

- State and prove the Cauchy-Schwarz inequality and derive the angle formula
- Prove that any orthogonal set of nonzero vectors is linearly independent
- Compute coordinates in an orthonormal basis using Fourier coefficients
- Construct orthogonal complements and verify the orthogonal decomposition theorem
- Characterize orthogonal matrices via $Q^\top Q = I$ and prove they are isometries
- Implement Gram-Schmidt orthogonalization (classical and modified forms)
- Explain why modified Gram-Schmidt is numerically superior to classical
- Construct QR decompositions via Gram-Schmidt, Householder reflectors, and Givens rotations
- Solve least squares problems via QR and explain why this is numerically superior to normal equations
- State and prove the spectral theorem for real symmetric matrices
- Apply orthogonal initialization, RoPE, and spectral normalization in AI contexts

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Orthogonality Buys You](#11-what-orthogonality-buys-you)
  - [1.2 Geometric Picture](#12-geometric-picture)
  - [1.3 Why Orthogonality Matters for AI](#13-why-orthogonality-matters-for-ai)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Inner Products and the Angle Formula](#21-inner-products-and-the-angle-formula)
  - [2.2 Orthogonal and Orthonormal Vectors](#22-orthogonal-and-orthonormal-vectors)
  - [2.3 Orthogonal Sets and Linear Independence](#23-orthogonal-sets-and-linear-independence)
  - [2.4 Orthonormal Bases](#24-orthonormal-bases)
  - [2.5 Orthogonal Complements](#25-orthogonal-complements)
- [3. Orthogonal Matrices](#3-orthogonal-matrices)
  - [3.1 Definition and Characterizations](#31-definition-and-characterizations)
  - [3.2 The Orthogonal Group](#32-the-orthogonal-group)
  - [3.3 Geometric Action: Isometries](#33-geometric-action-isometries)
  - [3.4 Important Examples](#34-important-examples)
  - [3.5 Numerical Stability Properties](#35-numerical-stability-properties)
- [4. Orthogonal Projections](#4-orthogonal-projections)
  - [4.1 Projection onto a Vector](#41-projection-onto-a-vector)
  - [4.2 Projection onto a Subspace](#42-projection-onto-a-subspace)
  - [4.3 Properties of Projection Matrices](#43-properties-of-projection-matrices)
  - [4.4 The Orthogonal Decomposition Theorem](#44-the-orthogonal-decomposition-theorem)
  - [4.5 Least Squares via Orthogonal Projection](#45-least-squares-via-orthogonal-projection)
- [5. Gram-Schmidt Orthogonalization](#5-gram-schmidt-orthogonalization)
  - [5.1 Classical Gram-Schmidt](#51-classical-gram-schmidt)
  - [5.2 Modified Gram-Schmidt](#52-modified-gram-schmidt)
  - [5.3 Householder QR](#53-householder-qr)
  - [5.4 Givens Rotations](#54-givens-rotations)
  - [5.5 Numerical Comparison](#55-numerical-comparison)
- [6. QR Decomposition](#6-qr-decomposition)
  - [6.1 Definition and Existence](#61-definition-and-existence)
  - [6.2 Thin QR](#62-thin-qr)
  - [6.3 QR for Least Squares](#63-qr-for-least-squares)
  - [6.4 QR Iteration for Eigenvalues](#64-qr-iteration-for-eigenvalues)
  - [6.5 Uniqueness and Variants](#65-uniqueness-and-variants)
- [7. The Spectral Theorem for Symmetric Matrices](#7-the-spectral-theorem-for-symmetric-matrices)
  - [7.1 Statement and Proof](#71-statement-and-proof)
  - [7.2 Consequences and the Rayleigh Quotient](#72-consequences-and-the-rayleigh-quotient)
  - [7.3 Symmetric Positive Semidefinite Matrices](#73-symmetric-positive-semidefinite-matrices)
  - [7.4 Matrix Functional Calculus](#74-matrix-functional-calculus)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Orthogonal Weight Initialization](#81-orthogonal-weight-initialization)
  - [8.2 Orthogonality in Attention](#82-orthogonality-in-attention)
  - [8.3 RoPE: Rotation Matrices as Positional Encoding](#83-rope-rotation-matrices-as-positional-encoding)
  - [8.4 Orthogonal Regularization](#84-orthogonal-regularization)
  - [8.5 QR in Numerical ML](#85-qr-in-numerical-ml)
  - [8.6 Orthonormal Bases in Signal Processing](#86-orthonormal-bases-in-signal-processing)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---
## 1. Intuition

### 1.1 What Orthogonality "Buys" You

Orthogonality is the key that makes linear algebra computationally tractable. Without it, working in a basis requires solving systems of equations. With it, everything simplifies to dot products.

**Five computational miracles of orthogonality:**

**Miracle 1: Coordinates become dot products.** In a general basis $\{\mathbf{b}_1, \ldots, \mathbf{b}_n\}$, finding the coordinates of a vector $\mathbf{v} = \sum c_i \mathbf{b}_i$ requires solving an $n \times n$ linear system. In an orthonormal basis $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$, the coordinate $c_i = \langle \mathbf{v}, \mathbf{q}_i \rangle$ — just a dot product. No system to solve.

**Miracle 2: Length is preserved component-wise.** In an ONB: $\|\mathbf{v}\|^2 = \sum_i c_i^2$ (Parseval's identity). The squared norm is simply the sum of squared coordinates — no cross terms.

**Miracle 3: Orthogonal matrices have condition number 1.** The condition number of $Q$ is $\|Q\| \cdot \|Q^{-1}\| = 1 \cdot 1 = 1$. This is the best possible — orthogonal systems are maximally well-conditioned. Solving $Q\mathbf{x} = \mathbf{b}$ is trivially $\mathbf{x} = Q^\top \mathbf{b}$.

**Miracle 4: Projection is the best approximation.** The orthogonal projection of $\mathbf{b}$ onto a subspace $S$ gives the closest point in $S$ to $\mathbf{b}$. This is the geometric foundation of least squares, PCA, and low-rank approximation.

**Miracle 5: Composition of orthogonal maps is free.** Two isometries composed give another isometry. Chaining rotations, reflections, and permutations never accumulates numerical error.

```
THE ORTHOGONALITY ADVANTAGE
════════════════════════════════════════════════════════════════════════

  General basis {b₁, ..., bₙ}         Orthonormal basis {q₁, ..., qₙ}
  ─────────────────────────────        ──────────────────────────────────

  Coordinates:  solve Bc = v           Coordinates:  cᵢ = <v, qᵢ> (dot!)
  Norm:         v^T B^T B v            Norm:         Σ cᵢ² (Parseval)
  Projection:   A(A^T A)⁻¹ A^T b      Projection:   QQ^T b (just Q^T then Q)
  Solve Ax=b:   Gaussian elim.         Solve Qx=b:   x = Q^T b (trivial)
  Condition#:   can be >> 1            Condition#:   exactly 1

════════════════════════════════════════════════════════════════════════
```

### 1.2 Geometric Picture

In Euclidean space, two vectors are orthogonal if they meet at a right angle. The dot product measures "how much they point in the same direction":

$$\langle \mathbf{u}, \mathbf{v} \rangle = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

- $\theta = 0°$: $\langle\mathbf{u},\mathbf{v}\rangle = \|\mathbf{u}\|\|\mathbf{v}\|$ (parallel, maximum alignment)
- $\theta = 90°$: $\langle\mathbf{u},\mathbf{v}\rangle = 0$ (orthogonal, zero alignment)
- $\theta = 180°$: $\langle\mathbf{u},\mathbf{v}\rangle = -\|\mathbf{u}\|\|\mathbf{v}\|$ (antiparallel, maximum anti-alignment)

**The projection picture.** The orthogonal projection of $\mathbf{v}$ onto $\mathbf{u}$ is the "shadow" of $\mathbf{v}$ cast perpendicularly onto the line through $\mathbf{u}$:

$$\text{proj}_{\mathbf{u}}\mathbf{v} = \frac{\langle\mathbf{v},\mathbf{u}\rangle}{\|\mathbf{u}\|^2}\mathbf{u}$$

The error $\mathbf{v} - \text{proj}_{\mathbf{u}}\mathbf{v}$ is orthogonal to $\mathbf{u}$. This Pythagorean decomposition is the geometric core of all least-squares theory.

**Orthonormal bases as "rulers."** A set of orthonormal vectors $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$ forms a perfect coordinate system: each $\mathbf{q}_i$ is a unit "ruler" pointing in an independent direction. The coordinate of $\mathbf{v}$ along $\mathbf{q}_i$ is exactly $\langle\mathbf{v}, \mathbf{q}_i\rangle$ — how far $\mathbf{v}$ extends in the $i$-th direction.

### 1.3 Why Orthogonality Matters for AI

**PCA finds orthogonal directions of variance.** The principal components are the eigenvectors of the sample covariance matrix $C = X^\top X/(n-1)$. By the spectral theorem (§7), these eigenvectors are orthogonal. PCA is possible — and gives uncorrelated components — precisely because of orthogonality.

**Attention subspaces.** In multi-head attention, each head projects queries and keys into a $d_k$-dimensional subspace. The heads are designed to attend to different aspects of the input. The more orthogonal the subspaces, the more independent information each head extracts.

**Orthogonal weight initialization** (Saxe et al., 2013) initializes weight matrices as random orthogonal matrices (via QR decomposition of a random Gaussian matrix). This preserves the norm of activations and gradients through linear layers, enabling training of very deep networks without vanishing/exploding gradients.

**RoPE** (Su et al., 2023) encodes position $m$ by rotating the query/key vectors by angle $m\theta_i$ in each 2D subspace. The key property: the dot product between a rotated query at position $m$ and a rotated key at position $n$ depends only on $(m-n)$ — position appears as a relative rotation. This works because rotation is orthogonal: it preserves the magnitude of inner products.

**Numerical stability.** QR decomposition is the backbone of numerical linear algebra. Householder QR is used to solve least-squares problems, compute eigenvalues (QR iteration), and factor matrices. Its superiority over LU (for rectangular systems) stems from orthogonality: the Householder reflectors are orthogonal transformations that never amplify errors.

### 1.4 Historical Timeline

| Year | Person | Contribution |
| --- | --- | --- |
| 1801 | Gauss | Least squares by orthogonal projection (unpublished until 1809) |
| 1805 | Legendre | Published least squares — minimizing sum of squared residuals |
| 1883 | Jørgen Gram | Gram-Schmidt process published in Danish |
| 1907 | Erhard Schmidt | Gram-Schmidt republished and popularized |
| 1958 | Alston Householder | Householder reflections for stable QR factorization |
| 1961 | J.G.F. Francis | Practical QR algorithm for eigenvalue computation |
| 1961 | V. Kublanovskaya | Independent discovery of QR algorithm (USSR) |
| 1965 | James Wilkinson | Error analysis of Householder QR; numerical stability theory |
| 2013 | Saxe, McClelland, Ganguli | Orthogonal initialization for deep linear networks |
| 2021 | Su et al. | RoPE: rotary positional encoding via orthogonal rotation matrices |
| 2021 | Hua et al. | Transformer Quality with orthogonal attention mechanisms |

---

## 2. Formal Definitions

### 2.1 Inner Products and the Angle Formula

**Definition (Inner Product).** A real **inner product** on a vector space $V$ is a function $\langle \cdot, \cdot \rangle: V \times V \to \mathbb{R}$ satisfying:
1. **Symmetry:** $\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. **Linearity in first argument:** $\langle a\mathbf{u} + b\mathbf{v}, \mathbf{w} \rangle = a\langle\mathbf{u},\mathbf{w}\rangle + b\langle\mathbf{v},\mathbf{w}\rangle$
3. **Positive definiteness:** $\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$ with equality iff $\mathbf{v} = \mathbf{0}$

The **standard inner product** (dot product) on $\mathbb{R}^n$ is $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^\top \mathbf{v} = \sum_{i=1}^n u_i v_i$.

**The induced norm:** $\|\mathbf{v}\| = \sqrt{\langle\mathbf{v},\mathbf{v}\rangle}$ is the Euclidean length of $\mathbf{v}$.

**Theorem 2.1.1 (Cauchy-Schwarz Inequality).** For any $\mathbf{u}, \mathbf{v}$ in an inner product space:
$$|\langle\mathbf{u},\mathbf{v}\rangle| \leq \|\mathbf{u}\| \cdot \|\mathbf{v}\|$$
with equality iff $\mathbf{u}$ and $\mathbf{v}$ are linearly dependent (one is a scalar multiple of the other).

**Proof.** For any $t \in \mathbb{R}$, $\|\mathbf{u} + t\mathbf{v}\|^2 \geq 0$. Expanding:
$$\|\mathbf{u}\|^2 + 2t\langle\mathbf{u},\mathbf{v}\rangle + t^2\|\mathbf{v}\|^2 \geq 0$$
This is a quadratic in $t$ that is always non-negative, so its discriminant must be $\leq 0$:
$$4\langle\mathbf{u},\mathbf{v}\rangle^2 - 4\|\mathbf{u}\|^2\|\mathbf{v}\|^2 \leq 0$$
which gives $|\langle\mathbf{u},\mathbf{v}\rangle| \leq \|\mathbf{u}\|\|\mathbf{v}\|$. Equality holds iff $t^* = -\langle\mathbf{u},\mathbf{v}\rangle/\|\mathbf{v}\|^2$ gives $\|\mathbf{u} + t^*\mathbf{v}\| = 0$, i.e., $\mathbf{u} = -t^*\mathbf{v}$. $\square$

**The angle formula.** By Cauchy-Schwarz, $-1 \leq \langle\mathbf{u},\mathbf{v}\rangle/(\|\mathbf{u}\|\|\mathbf{v}\|) \leq 1$, so it is valid to define the angle $\theta$ between $\mathbf{u}$ and $\mathbf{v}$ by:
$$\cos\theta = \frac{\langle\mathbf{u},\mathbf{v}\rangle}{\|\mathbf{u}\|\|\mathbf{v}\|}, \quad \theta \in [0, \pi]$$

**Corollary (Triangle Inequality).** $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$.

*Proof:* $\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\langle\mathbf{u},\mathbf{v}\rangle + \|\mathbf{v}\|^2 \leq \|\mathbf{u}\|^2 + 2\|\mathbf{u}\|\|\mathbf{v}\| + \|\mathbf{v}\|^2 = (\|\mathbf{u}\|+\|\mathbf{v}\|)^2$. $\square$

### 2.2 Orthogonal and Orthonormal Vectors

**Definition.** Two vectors $\mathbf{u}, \mathbf{v}$ are **orthogonal** (written $\mathbf{u} \perp \mathbf{v}$) if $\langle\mathbf{u},\mathbf{v}\rangle = 0$.

A vector $\mathbf{v}$ is a **unit vector** if $\|\mathbf{v}\| = 1$. The **normalization** of nonzero $\mathbf{v}$ is $\hat{\mathbf{v}} = \mathbf{v}/\|\mathbf{v}\|$.

Two vectors are **orthonormal** if they are both orthogonal AND both unit vectors:
$$\langle\mathbf{q}_i, \mathbf{q}_j\rangle = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

**Key consequences of orthogonality:**

**1. Zero maps to everything:** $\mathbf{0} \perp \mathbf{v}$ for all $\mathbf{v}$ (since $\langle\mathbf{0},\mathbf{v}\rangle = 0$).

**2. Pythagorean theorem:** If $\mathbf{u} \perp \mathbf{v}$, then $\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$.
*Proof:* $\|\mathbf{u}+\mathbf{v}\|^2 = \langle\mathbf{u}+\mathbf{v},\mathbf{u}+\mathbf{v}\rangle = \|\mathbf{u}\|^2 + 2\underbrace{\langle\mathbf{u},\mathbf{v}\rangle}_{=0} + \|\mathbf{v}\|^2$. $\square$

**3. Generalized Pythagorean theorem:** For mutually orthogonal $\mathbf{v}_1, \ldots, \mathbf{v}_k$:
$$\left\|\sum_{i=1}^k \mathbf{v}_i\right\|^2 = \sum_{i=1}^k \|\mathbf{v}_i\|^2$$

**4. Linearity of orthogonality:** If $\mathbf{u} \perp \mathbf{v}$ and $\mathbf{u} \perp \mathbf{w}$, then $\mathbf{u} \perp (a\mathbf{v} + b\mathbf{w})$ for all scalars $a, b$.

### 2.3 Orthogonal Sets and Linear Independence

**Definition.** A set $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is an **orthogonal set** if $\langle\mathbf{v}_i,\mathbf{v}_j\rangle = 0$ for all $i \neq j$. It is an **orthonormal set** if additionally each $\|\mathbf{v}_i\| = 1$.

**Theorem 2.3.1.** Any orthogonal set of nonzero vectors is linearly independent.

**Proof.** Suppose $\sum_{i=1}^k c_i \mathbf{v}_i = \mathbf{0}$. Take the inner product of both sides with $\mathbf{v}_j$:
$$\left\langle \sum_i c_i \mathbf{v}_i, \mathbf{v}_j \right\rangle = c_j \langle\mathbf{v}_j,\mathbf{v}_j\rangle = c_j \|\mathbf{v}_j\|^2 = 0$$
Since $\mathbf{v}_j \neq \mathbf{0}$, we have $\|\mathbf{v}_j\|^2 > 0$, so $c_j = 0$. This holds for all $j$. $\square$

**Remark.** This theorem gives an effortless proof of linear independence for orthogonal sets — no row reduction needed. The inner product "isolates" each coefficient.

**Non-examples:**
- $\{(1,0), (1,1)\}$ is not orthogonal: $\langle(1,0),(1,1)\rangle = 1 \neq 0$.
- $\{(1,1,0)/\sqrt{2}, (0,0,1), (1,-1,0)/\sqrt{2}, (1,0,0)\}$ — the last vector breaks orthogonality.

### 2.4 Orthonormal Bases

**Definition.** An **orthonormal basis (ONB)** for $V$ is a set $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$ that is both orthonormal and a basis for $V$.

**Theorem 2.4.1 (Fourier Coefficients).** If $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$ is an ONB for $V$, then for any $\mathbf{v} \in V$:
$$\mathbf{v} = \sum_{i=1}^n \langle\mathbf{v}, \mathbf{q}_i\rangle \mathbf{q}_i$$

The coefficients $\hat{v}_i = \langle\mathbf{v}, \mathbf{q}_i\rangle$ are called **Fourier coefficients** (or coordinates of $\mathbf{v}$ in the ONB).

**Proof.** Since $\{\mathbf{q}_i\}$ is a basis, $\mathbf{v} = \sum_j c_j \mathbf{q}_j$ for some unique $c_j$. Taking the inner product with $\mathbf{q}_i$:
$$\langle\mathbf{v}, \mathbf{q}_i\rangle = \sum_j c_j \langle\mathbf{q}_j, \mathbf{q}_i\rangle = \sum_j c_j \delta_{ji} = c_i \quad \square$$

**Theorem 2.4.2 (Parseval's Identity).** Under the same conditions:
$$\|\mathbf{v}\|^2 = \sum_{i=1}^n |\langle\mathbf{v}, \mathbf{q}_i\rangle|^2 = \sum_{i=1}^n |\hat{v}_i|^2$$

**Proof.** $\|\mathbf{v}\|^2 = \langle\sum_i \hat{v}_i\mathbf{q}_i, \sum_j \hat{v}_j\mathbf{q}_j\rangle = \sum_{i,j} \hat{v}_i\hat{v}_j \langle\mathbf{q}_i,\mathbf{q}_j\rangle = \sum_i \hat{v}_i^2$. $\square$

**Remark.** Parseval's identity says: in an ONB, the squared norm equals the sum of squared coordinates. There are no cross terms — the orthonormality kills them all. This is the high-dimensional Pythagorean theorem.

**Matrix form.** Assembling the ONB into a matrix $Q = [\mathbf{q}_1 | \cdots | \mathbf{q}_n]$, the Fourier decomposition is simply $\mathbf{v} = Q(Q^\top\mathbf{v})$: first compute coordinates $\hat{\mathbf{v}} = Q^\top\mathbf{v}$, then reconstruct $\mathbf{v} = Q\hat{\mathbf{v}}$.

### 2.5 Orthogonal Complements

**Definition.** Given a subspace $S \subseteq V$, the **orthogonal complement** is:
$$S^\perp = \{\mathbf{v} \in V : \langle\mathbf{v},\mathbf{s}\rangle = 0 \text{ for all } \mathbf{s} \in S\}$$

**Theorem 2.5.1.** $S^\perp$ is a subspace of $V$.

*Proof:* If $\mathbf{u}, \mathbf{v} \in S^\perp$ and $a, b \in \mathbb{R}$, then for any $\mathbf{s} \in S$: $\langle a\mathbf{u}+b\mathbf{v}, \mathbf{s}\rangle = a\langle\mathbf{u},\mathbf{s}\rangle + b\langle\mathbf{v},\mathbf{s}\rangle = 0$. $\square$

**Theorem 2.5.2 (Orthogonal Decomposition).** If $V$ is finite-dimensional and $S$ is a subspace, then:
$$V = S \oplus S^\perp \quad (\text{direct sum})$$
Every $\mathbf{v} \in V$ decomposes uniquely as $\mathbf{v} = \mathbf{v}_S + \mathbf{v}_{S^\perp}$ with $\mathbf{v}_S \in S$ and $\mathbf{v}_{S^\perp} \in S^\perp$.

Moreover: $\dim(S) + \dim(S^\perp) = \dim(V)$.

**Proof.** Let $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ be an ONB for $S$. Define $\mathbf{v}_S = \sum_{i=1}^k \langle\mathbf{v},\mathbf{q}_i\rangle\mathbf{q}_i$ (projection onto $S$) and $\mathbf{v}_{S^\perp} = \mathbf{v} - \mathbf{v}_S$. Then $\mathbf{v} = \mathbf{v}_S + \mathbf{v}_{S^\perp}$ with $\mathbf{v}_S \in S$. For any $\mathbf{q}_j$:
$$\langle\mathbf{v}_{S^\perp}, \mathbf{q}_j\rangle = \langle\mathbf{v},\mathbf{q}_j\rangle - \langle\mathbf{v}_S,\mathbf{q}_j\rangle = \langle\mathbf{v},\mathbf{q}_j\rangle - \langle\mathbf{v},\mathbf{q}_j\rangle = 0$$
So $\mathbf{v}_{S^\perp} \perp \mathbf{q}_j$ for all $j$, hence $\mathbf{v}_{S^\perp} \in S^\perp$. Uniqueness: if $\mathbf{v} = \mathbf{w}_S + \mathbf{w}_{S^\perp}$ also, then $\mathbf{v}_S - \mathbf{w}_S = \mathbf{w}_{S^\perp} - \mathbf{v}_{S^\perp} \in S \cap S^\perp = \{\mathbf{0}\}$. $\square$

**Connection to the four fundamental subspaces.** For $A \in \mathbb{R}^{m \times n}$:
- $\operatorname{null}(A) = \operatorname{row}(A)^\perp$ in $\mathbb{R}^n$
- $\operatorname{null}(A^\top) = \operatorname{col}(A)^\perp$ in $\mathbb{R}^m$

These orthogonality relations are the deep structure behind the four fundamental subspaces (see [§04: Linear Transformations §4.5](../04-Linear-Transformations/notes.md#45-the-four-fundamental-subspaces-via-linear-maps)).

### 2.6 Orthogonality in Complex Inner Product Spaces

The theory extends to **complex vector spaces** with a Hermitian (sesquilinear) inner product:
$$\langle\mathbf{u},\mathbf{v}\rangle = \sum_{k=1}^n \overline{u_k}v_k$$

The complex analog of orthogonal matrices is **unitary matrices** $U \in \mathbb{C}^{n \times n}$ satisfying $U^*U = I$, where $U^* = \overline{U}^\top$ is the conjugate transpose.

**Unitary group:** $\mathcal{U}(n) = \{U \in \mathbb{C}^{n \times n} : U^*U = I\}$ contains all $n \times n$ unitary matrices. It reduces to $O(n)$ when restricted to real matrices.

**Special unitary group:** $\mathcal{SU}(n) = \{U \in \mathcal{U}(n) : \det(U) = 1\}$.

**Key properties of unitary matrices:**
- $|\det(U)| = 1$ (complex number of modulus 1, not just $\pm 1$)
- Eigenvalues lie on the unit circle: $|\lambda_i| = 1$
- Preserve complex inner products: $\langle U\mathbf{u}, U\mathbf{v}\rangle = \langle\mathbf{u},\mathbf{v}\rangle$
- The DFT matrix $F_n/\sqrt{n}$ is unitary (not orthogonal, since its entries are complex)

**Hermitian matrices** ($A = A^*$) are the complex analogs of real symmetric matrices. The spectral theorem holds: every Hermitian matrix has real eigenvalues and a unitary eigenbasis.

For AI, complex-valued neural networks and quantum computing naturally use unitary matrices. The DFT, crucial for signal processing and some positional encoding schemes, is a unitary transform in $\mathbb{C}^n$.

### 2.7 Abstract Inner Product Spaces: Axioms

We have been working with the Euclidean inner product $\langle\mathbf{u},\mathbf{v}\rangle = \mathbf{u}^\top\mathbf{v}$. More generally, an **inner product** on a real vector space $V$ is a bilinear map $\langle\cdot,\cdot\rangle: V \times V \to \mathbb{R}$ satisfying:

| Axiom | Expression | Meaning |
|-------|-----------|---------|
| Symmetry | $\langle\mathbf{u},\mathbf{v}\rangle = \langle\mathbf{v},\mathbf{u}\rangle$ | Order doesn't matter |
| Linearity | $\langle a\mathbf{u}+b\mathbf{v},\mathbf{w}\rangle = a\langle\mathbf{u},\mathbf{w}\rangle + b\langle\mathbf{v},\mathbf{w}\rangle$ | Linear in first argument |
| Positive definiteness | $\langle\mathbf{v},\mathbf{v}\rangle \geq 0$ and $= 0 \Rightarrow \mathbf{v} = \mathbf{0}$ | Self-inner product is positive |

Any positive definite symmetric matrix $M \succ 0$ defines an inner product:
$$\langle\mathbf{u},\mathbf{v}\rangle_M = \mathbf{u}^\top M\mathbf{v}$$

**Example (precision-weighted inner product):** In Bayesian statistics, the **Mahalanobis distance** between $\mathbf{x}$ and $\boldsymbol{\mu}$ is:
$$d_M(\mathbf{x},\boldsymbol{\mu}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

This is the distance induced by the inner product $\langle\mathbf{u},\mathbf{v}\rangle_{\Sigma^{-1}} = \mathbf{u}^\top\Sigma^{-1}\mathbf{v}$. In this metric, vectors along high-variance directions of $\Sigma$ are "shorter" (the precision $\Sigma^{-1}$ down-weights those directions).

**Orthogonality is metric-dependent.** Vectors that are orthogonal in the Euclidean metric may not be orthogonal in the Mahalanobis metric, and vice versa. The choice of inner product determines what "perpendicular" means.

**For ML:** The natural gradient method (Amari 1998) uses the Fisher information matrix $F$ to define the inner product, giving rise to a geometry where parameter updates are $\Delta\theta = F^{-1}\nabla\mathcal{L}$. This is gradient descent in the Riemannian metric $\langle\Delta\theta_1, \Delta\theta_2\rangle_F = \Delta\theta_1^\top F \Delta\theta_2$.

---



## 3. Orthogonal Matrices

### 3.1 Definition and Characterizations

**Definition.** A square matrix $Q \in \mathbb{R}^{n \times n}$ is **orthogonal** if:
$$Q^\top Q = QQ^\top = I_n$$

Equivalently, $Q^{-1} = Q^\top$ — the transpose is the inverse.

**Characterization via columns.** $Q$ is orthogonal if and only if its columns $\mathbf{q}_1, \ldots, \mathbf{q}_n$ form an orthonormal basis for $\mathbb{R}^n$:
$$\mathbf{q}_i^\top \mathbf{q}_j = \delta_{ij} = \begin{cases} 1 & i = j \\ 0 & i \neq j \end{cases}$$

The condition $Q^\top Q = I$ encodes exactly these $n^2$ inner products.

**Characterization via rows.** By the symmetry $QQ^\top = I$, the rows of $Q$ also form an orthonormal basis. This is a non-trivial statement: a matrix whose columns are orthonormal must also have orthonormal rows (and vice versa).

**Characterization via norms (isometry property).** $Q$ is orthogonal if and only if it preserves the Euclidean norm:
$$\|Q\mathbf{x}\|_2 = \|\mathbf{x}\|_2 \quad \text{for all } \mathbf{x} \in \mathbb{R}^n$$

*Proof:* $\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^\top(Q\mathbf{x}) = \mathbf{x}^\top Q^\top Q \mathbf{x} = \mathbf{x}^\top I \mathbf{x} = \|\mathbf{x}\|^2$. $\square$

**Characterization via inner products.** $Q$ preserves inner products:
$$\langle Q\mathbf{x}, Q\mathbf{y}\rangle = \langle\mathbf{x},\mathbf{y}\rangle \quad \text{for all } \mathbf{x}, \mathbf{y}$$

*Proof:* $(Q\mathbf{x})^\top(Q\mathbf{y}) = \mathbf{x}^\top Q^\top Q \mathbf{y} = \mathbf{x}^\top \mathbf{y}$. $\square$

This means orthogonal matrices preserve **angles and distances** — they are isometries of Euclidean space.

### 3.2 The Orthogonal Group and Special Orthogonal Group

The set of all $n \times n$ orthogonal matrices forms the **orthogonal group**:
$$O(n) = \{Q \in \mathbb{R}^{n \times n} : Q^\top Q = I\}$$

**Group axioms:**
- *Closure:* If $Q_1, Q_2 \in O(n)$, then $(Q_1 Q_2)^\top (Q_1 Q_2) = Q_2^\top Q_1^\top Q_1 Q_2 = I$, so $Q_1 Q_2 \in O(n)$.
- *Identity:* $I \in O(n)$.
- *Inverses:* $Q^{-1} = Q^\top \in O(n)$ (since $(Q^\top)^\top Q^\top = QQ^\top = I$).
- *Associativity:* Inherited from matrix multiplication.

**Determinant.** Since $\det(Q^\top Q) = \det(Q)^2 = \det(I) = 1$, we have $\det(Q) = \pm 1$.

- $\det(Q) = +1$: **rotations** — orientation-preserving isometries
- $\det(Q) = -1$: **improper rotations** — reflections (or rotation composed with a reflection)

The **special orthogonal group** $SO(n) = \{Q \in O(n) : \det(Q) = +1\}$ consists of pure rotations.

**In 2D:**
$$SO(2) = \left\{\begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix} : \theta \in [0,2\pi)\right\}$$

This is the group of rotations of the plane.

**In 3D:** $SO(3)$ parameterizes 3D rotations. This group is the configuration space of a rigid body and appears throughout robotics, computer vision, and physics. Its double cover $SU(2)$ appears in quaternion-based rotation representations used in game engines and IMUs.

### 3.3 Condition Number and Numerical Properties

The **condition number** of a matrix measures how much it amplifies errors:
$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

For orthogonal matrices $Q$: all singular values equal 1, so $\kappa(Q) = 1/1 = 1$. **Orthogonal matrices have the best possible condition number.**

This means:
- Multiplying by $Q$ never amplifies rounding errors
- Solving $Q\mathbf{x} = \mathbf{b}$ is trivially stable: $\mathbf{x} = Q^\top\mathbf{b}$
- QR factorization (which uses orthogonal matrices) is numerically stable

**For AI:** Deep learning systems that maintain orthogonal weight matrices throughout training benefit from gradient signals that are neither exploding nor vanishing. The isometry property means the network's "signal amplification" is exactly 1 at those layers.

### 3.4 Eigenvalues of Orthogonal Matrices

**Theorem.** The eigenvalues of a real orthogonal matrix lie on the unit circle in $\mathbb{C}$: $|\lambda| = 1$.

*Proof:* If $Q\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \neq \mathbf{0}$:
$$\|\mathbf{v}\| = \|Q\mathbf{v}\| = \|\lambda\mathbf{v}\| = |\lambda|\|\mathbf{v}\|$$
Dividing by $\|\mathbf{v}\| > 0$: $|\lambda| = 1$. $\square$

For real eigenvalues: $\lambda = +1$ (rotation-like) or $\lambda = -1$ (reflection-like).
Complex eigenvalues come in conjugate pairs $e^{i\theta}, e^{-i\theta}$ corresponding to 2D rotations.

### 3.5 The Cayley Transform

The **Cayley transform** establishes a correspondence between orthogonal matrices and skew-symmetric matrices (matrices satisfying $S^\top = -S$).

**Definition.** For a skew-symmetric matrix $S$ (such that $I + S$ is invertible):
$$Q = (I - S)(I + S)^{-1}$$

**Claim:** $Q$ is orthogonal.

*Proof:* Using $(I + S)^\top = I - S$ (since $S^\top = -S$):
$$Q^\top Q = ((I+S)^{-1})^\top (I-S)^\top (I-S)(I+S)^{-1}$$
$$= (I-S)^{-1}(I+S)(I-S)(I+S)^{-1}$$

Since $S$ is skew-symmetric, $I-S$ and $I+S$ commute (they differ only by sign of $S$, which is a normal matrix in this context), so:
$$(I+S)(I-S) = I - S^2 = (I-S)(I+S)$$
Hence $Q^\top Q = (I-S)^{-1}(I-S)(I+S)(I+S)^{-1} = I$. $\square$

**Inverse Cayley transform.** Given orthogonal $Q$ without eigenvalue $-1$:
$$S = (I - Q)(I + Q)^{-1}$$

**Why this matters:**
- Skew-symmetric matrices form a vector space (easy to parameterize)
- The Cayley transform maps this space bijectively to "most" orthogonal matrices
- This gives a smooth parameterization of $O(n)$ near the identity — useful for Riemannian optimization

**Example in 2D:** $S = \begin{pmatrix}0 & -t \\ t & 0\end{pmatrix}$ (skew-symmetric) gives:
$$Q = \frac{1}{1+t^2}\begin{pmatrix}1-t^2 & -2t \\ 2t & 1-t^2\end{pmatrix}$$

With $t = \tan(\theta/2)$ (the Weierstrass substitution), this gives the rotation matrix $R_\theta$. The Cayley transform thus rationalizes the trigonometric functions appearing in rotation matrices.

**Applications in neural networks:** Neural networks that maintain orthogonal weight matrices throughout training can parameterize them via their skew-symmetric "log" and exponentiate via the matrix exponential or Cayley transform. This enables gradient descent on $O(n)$ using Lie algebra tools.

---



## 4. Orthogonal Projections

### 4.1 The Projection Formula

**Definition.** The **orthogonal projection** of $\mathbf{v}$ onto a subspace $S$ is the unique vector $\mathbf{v}_S \in S$ such that $\mathbf{v} - \mathbf{v}_S \perp S$.

**Formula (1D case).** Projection onto the line spanned by unit vector $\hat{\mathbf{u}}$:
$$\operatorname{proj}_{\hat{\mathbf{u}}}(\mathbf{v}) = \langle\mathbf{v},\hat{\mathbf{u}}\rangle\,\hat{\mathbf{u}} = \hat{\mathbf{u}}\hat{\mathbf{u}}^\top\mathbf{v}$$

The matrix $P = \hat{\mathbf{u}}\hat{\mathbf{u}}^\top$ is the **rank-1 projection matrix**.

**Formula (general case).** If $S = \operatorname{col}(A)$ for a matrix $A$ with linearly independent columns:
$$\operatorname{proj}_S(\mathbf{v}) = A(A^\top A)^{-1}A^\top\mathbf{v}$$

The matrix $P = A(A^\top A)^{-1}A^\top$ is the **projection matrix** onto $\operatorname{col}(A)$.

**When $A$ has orthonormal columns** (i.e., $A = Q$ with $Q^\top Q = I$), this simplifies dramatically:
$$P = QQ^\top$$

This is why orthonormal bases make projections computationally trivial.

### 4.2 Projection Matrices: Properties

A matrix $P$ is a **projection** if and only if it is **idempotent**: $P^2 = P$.

A projection is an **orthogonal projection** if and only if it is also **symmetric**: $P^\top = P$.

**Verification:** For $P = QQ^\top$ with $Q^\top Q = I$:
- Idempotent: $P^2 = QQ^\top QQ^\top = Q(Q^\top Q)Q^\top = QIQ^\top = QQ^\top = P$ ✓
- Symmetric: $P^\top = (QQ^\top)^\top = QQ^\top = P$ ✓

**Eigenvalues of projections.** $P$ is a projection if and only if its eigenvalues are 0 and 1.

*Proof:* If $P\mathbf{v} = \lambda\mathbf{v}$, then $\lambda\mathbf{v} = P\mathbf{v} = P^2\mathbf{v} = \lambda P\mathbf{v} = \lambda^2\mathbf{v}$, so $\lambda^2 = \lambda$, giving $\lambda = 0$ or $\lambda = 1$. $\square$

Geometrically: eigenvectors with $\lambda = 1$ lie in the range of $P$ (they are unchanged), while eigenvectors with $\lambda = 0$ lie in the null space (they are annihilated).

### 4.3 Best Approximation Theorem

**Theorem (Best Approximation).** The projection $\mathbf{v}_S = \operatorname{proj}_S(\mathbf{v})$ is the closest point in $S$ to $\mathbf{v}$:
$$\|\mathbf{v} - \mathbf{v}_S\|^2 \leq \|\mathbf{v} - \mathbf{s}\|^2 \quad \text{for all } \mathbf{s} \in S$$

with equality only when $\mathbf{s} = \mathbf{v}_S$.

*Proof.* For any $\mathbf{s} \in S$:
$$\|\mathbf{v} - \mathbf{s}\|^2 = \|(\mathbf{v}-\mathbf{v}_S) + (\mathbf{v}_S - \mathbf{s})\|^2 = \|\mathbf{v}-\mathbf{v}_S\|^2 + 2\langle\mathbf{v}-\mathbf{v}_S, \mathbf{v}_S-\mathbf{s}\rangle + \|\mathbf{v}_S-\mathbf{s}\|^2$$

Since $\mathbf{v}-\mathbf{v}_S \perp S$ and $\mathbf{v}_S - \mathbf{s} \in S$, the cross term vanishes:
$$= \|\mathbf{v}-\mathbf{v}_S\|^2 + \|\mathbf{v}_S-\mathbf{s}\|^2 \geq \|\mathbf{v}-\mathbf{v}_S\|^2 \quad \square$$

**This theorem is everywhere in machine learning:**
- **Least squares:** The normal equations give the projection of $\mathbf{b}$ onto $\operatorname{col}(A)$
- **PCA:** Principal components are the projection onto the subspace of maximum variance (→ [§03: PCA](../03-PCA-and-Low-Rank-Approximations/notes.md))
- **Attention:** Softmax attention can be viewed as computing a weighted projection of value vectors
- **Linear regression:** The fitted values $\hat{\mathbf{y}} = H\mathbf{y}$ where $H = X(X^\top X)^{-1}X^\top$ is the hat matrix

### 4.4 Projection in Coordinate Systems

Given an ONB $\{\mathbf{q}_1, \ldots, \mathbf{q}_n\}$ for $\mathbb{R}^n$, the projection onto the subspace $S = \operatorname{span}(\mathbf{q}_1, \ldots, \mathbf{q}_k)$ (for $k \leq n$) is:
$$P_S = \sum_{i=1}^k \mathbf{q}_i\mathbf{q}_i^\top$$

This is a sum of rank-1 projectors, one per basis direction. The decomposition:
$$I = P_S + P_{S^\perp} = \sum_{i=1}^k \mathbf{q}_i\mathbf{q}_i^\top + \sum_{i=k+1}^n \mathbf{q}_i\mathbf{q}_i^\top$$

is the **resolution of the identity** — every vector is split into its $S$ and $S^\perp$ components.

---

## 5. Gram-Schmidt Orthogonalization

### 5.1 The Algorithm: Classical Gram-Schmidt

**Problem.** Given linearly independent vectors $\mathbf{a}_1, \ldots, \mathbf{a}_k$, construct an orthonormal basis for $\operatorname{span}(\mathbf{a}_1, \ldots, \mathbf{a}_k)$.

**Classical Gram-Schmidt (CGS):**

**Initialization:** $\mathbf{q}_1 = \mathbf{a}_1 / \|\mathbf{a}_1\|$

**Iteration:** For $j = 2, \ldots, k$:

$$\mathbf{u}_j = \mathbf{a}_j - \sum_{i=1}^{j-1}\langle\mathbf{a}_j, \mathbf{q}_i\rangle\,\mathbf{q}_i \qquad \mathbf{q}_j = \frac{\mathbf{u}_j}{\|\mathbf{u}_j\|}$$

**What it does geometrically:** At step $j$, we subtract the projection of $\mathbf{a}_j$ onto the span of the already-processed vectors, leaving the component orthogonal to everything before it. Normalizing gives the $j$-th orthonormal vector.

**Inductive proof of correctness.** After step $j$, $\{\mathbf{q}_1, \ldots, \mathbf{q}_j\}$ is an ONB for $\operatorname{span}(\mathbf{a}_1, \ldots, \mathbf{a}_j)$.

*Base:* $j=1$: $\mathbf{q}_1 = \mathbf{a}_1/\|\mathbf{a}_1\|$ is a unit vector spanning the same subspace. ✓

*Step:* Assume true for $j-1$. Then $\mathbf{u}_j = \mathbf{a}_j - P_{j-1}\mathbf{a}_j$ where $P_{j-1}$ projects onto $\operatorname{span}(\mathbf{q}_1,\ldots,\mathbf{q}_{j-1})$.

For any $\ell < j$: $\langle\mathbf{u}_j, \mathbf{q}_\ell\rangle = \langle\mathbf{a}_j,\mathbf{q}_\ell\rangle - \langle\mathbf{a}_j,\mathbf{q}_\ell\rangle = 0$ ✓

Since $\mathbf{a}_1,\ldots,\mathbf{a}_j$ are independent, $\mathbf{u}_j \neq \mathbf{0}$ (otherwise $\mathbf{a}_j \in \operatorname{span}(\mathbf{a}_1,\ldots,\mathbf{a}_{j-1})$). So $\mathbf{q}_j = \mathbf{u}_j/\|\mathbf{u}_j\|$ is a valid unit vector. $\square$

### 5.2 Modified Gram-Schmidt

**The numerical problem with CGS.** In floating-point arithmetic, CGS can catastrophically lose orthogonality. This happens when $\mathbf{a}_j$ is nearly in the span of $\{\mathbf{a}_1,\ldots,\mathbf{a}_{j-1}\}$ — rounding errors in the projection subtraction can accumulate, producing a $\mathbf{u}_j$ that isn't truly orthogonal to earlier vectors.

**Modified Gram-Schmidt (MGS)** reorders the computation to orthogonalize against already-computed $\mathbf{q}_i$ sequentially rather than all at once:

```
for j = 1 to k:
    v = a_j
    for i = 1 to j-1:
        v = v - ⟨v, q_i⟩ q_i    # orthogonalize against q_i using CURRENT v
    q_j = v / ‖v‖
```

**Why MGS is more stable.** In CGS, all projections use the original $\mathbf{a}_j$. In MGS, after each orthogonalization step, the updated $\mathbf{v}$ is used — so numerical errors from the first projection are themselves projected away in subsequent steps.

**Theorem (Björck 1967).** MGS is backward stable: the computed $\hat{\mathbf{q}}_i$ satisfy $\hat{\mathbf{q}}_i^\top \hat{\mathbf{q}}_j = \delta_{ij} + O(\epsilon_{\text{mach}}\kappa(A))$ where $\kappa(A)$ is the condition number of the input matrix. CGS satisfies only $O(\epsilon_{\text{mach}}\kappa(A)^2)$.

**Practical advice:** Use MGS for explicit basis construction; use Householder for QR factorization (even more stable).

### 5.3 Gram-Schmidt and QR Factorization

Gram-Schmidt applied to the columns of $A = [\mathbf{a}_1|\cdots|\mathbf{a}_n]$ produces a **QR factorization**:
$$A = QR$$

where $Q = [\mathbf{q}_1|\cdots|\mathbf{q}_n]$ has orthonormal columns and $R$ is upper triangular with:
$$R_{ij} = \begin{cases}\langle\mathbf{a}_j, \mathbf{q}_i\rangle & i < j \\ \|\mathbf{u}_j\| & i = j \\ 0 & i > j\end{cases}$$

The diagonal entries $R_{jj} = \|\mathbf{u}_j\|$ are positive (since $\mathbf{u}_j \neq \mathbf{0}$ for independent columns).

This gives a **unique QR factorization** of $A$ with positive diagonal entries of $R$.

**For AI:** Gram-Schmidt implicitly occurs in layer normalization variants, in computing orthogonal weight bases for LoRA, and in the derivation of stable training algorithms.

---

## 6. Householder Reflectors and Givens Rotations

### 6.1 Householder Reflectors

**Motivation.** While Gram-Schmidt produces QR by building $Q$ column by column, Householder applies successive orthogonal transformations to reduce $A$ to upper triangular form $R$ directly — achieving greater numerical stability.

**Definition.** The **Householder reflector** determined by unit vector $\mathbf{n}$ (the normal to the mirror) is:
$$H = I - 2\mathbf{n}\mathbf{n}^\top$$

**Properties:**
- *Symmetric:* $H^\top = H$ ✓
- *Orthogonal:* $H^\top H = H^2 = (I - 2\mathbf{n}\mathbf{n}^\top)^2 = I - 4\mathbf{n}\mathbf{n}^\top + 4\mathbf{n}\mathbf{n}^\top\mathbf{n}\mathbf{n}^\top = I - 4\mathbf{n}\mathbf{n}^\top + 4\mathbf{n}\mathbf{n}^\top = I$ ✓
- *Involution:* $H^2 = I$, so $H = H^{-1}$ ✓
- *Determinant:* $\det(H) = -1$ (it's a reflection, not a rotation)

**Action:** $H$ reflects any vector across the hyperplane with normal $\mathbf{n}$:
- Vectors in $\mathbf{n}^\perp$ are unchanged: $H\mathbf{v} = \mathbf{v}$ if $\langle\mathbf{v},\mathbf{n}\rangle = 0$
- The vector $\mathbf{n}$ is negated: $H\mathbf{n} = \mathbf{n} - 2\mathbf{n} = -\mathbf{n}$

**Zeroing a column.** To zero out all components of $\mathbf{a}$ below the first entry: choose:
$$\mathbf{n} = \frac{\mathbf{a} + \|\mathbf{a}\|\mathbf{e}_1}{\|\mathbf{a} + \|\mathbf{a}\|\mathbf{e}_1\|} \quad \Rightarrow \quad H\mathbf{a} = -\|\mathbf{a}\|\mathbf{e}_1$$

(The $\pm$ sign choice avoids catastrophic cancellation when $a_1 > 0$.)

### 6.2 QR via Householder

To factor $A \in \mathbb{R}^{m \times n}$:

1. Find $H_1$ to zero the first column below row 1: $H_1 A$ has $\ast$ in position $(1,1)$ and zeros below
2. Find $H_2$ (acting on the $(n-1) \times (n-1)$ submatrix) to zero the second column below row 2
3. Continue: $H_{n} \cdots H_2 H_1 A = R$ (upper triangular)

Then $Q = H_1 H_2 \cdots H_n$ (product of reflectors, hence orthogonal), and $A = QR$.

**Complexity:** $O(mn^2 - n^3/3)$ flops — same asymptotic complexity as Gram-Schmidt but with better numerical constants.

**Stability:** Householder QR is backward stable, meaning the computed factors satisfy $\hat{Q}\hat{R} = A + E$ where $\|E\| = O(\epsilon_{\text{mach}}\|A\|)$.

### 6.3 Givens Rotations

**Definition.** The **Givens rotation** $G(i,j,\theta) \in \mathbb{R}^{n \times n}$ rotates in the $(i,j)$-plane:
$$G(i,j,\theta)_{kl} = \begin{cases}
\cos\theta & k=l=i \text{ or } k=l=j \\
-\sin\theta & k=i, l=j \\
\sin\theta & k=j, l=i \\
\delta_{kl} & \text{otherwise}
\end{cases}$$

**Purpose:** Choose $\theta$ to zero a specific element $a_{ji}$ by rotating in the $(i,j)$-plane.

**When to use:** Givens rotations are preferred when $A$ is sparse (each rotation affects only two rows/columns), for parallel computation, and in eigenvalue algorithms like QR iteration.

---

## 7. QR Decomposition: Theory and Applications

### 7.1 Existence and Uniqueness

**Theorem (QR Decomposition).** Every matrix $A \in \mathbb{R}^{m \times n}$ with $m \geq n$ admits a factorization:
$$A = QR$$

where $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns ($Q^\top Q = I_n$) and $R \in \mathbb{R}^{n \times n}$ is upper triangular.

This is the **thin (reduced) QR decomposition**. There is also a **full QR decomposition** where $Q \in \mathbb{R}^{m \times m}$ is a full orthogonal matrix and $R \in \mathbb{R}^{m \times n}$ is upper trapezoidal.

**Uniqueness (when $A$ has full column rank).** If we require $R_{ii} > 0$ for all $i$, the thin QR decomposition is unique.

*Proof sketch:* Two decompositions $Q_1 R_1 = Q_2 R_2$ give $Q_2^\top Q_1 = R_2 R_1^{-1}$. The left side is orthogonal; the right side is upper triangular. An upper triangular orthogonal matrix must be diagonal with $\pm 1$ entries. With the sign convention $R_{ii} > 0$, this diagonal must be $I$. $\square$

**When $A$ has rank $r < n$:** The decomposition still exists but $R$ has $n - r$ zero diagonal entries. The columns of $Q$ corresponding to zero $R_{ii}$ can be chosen as any completion to an orthonormal set.

### 7.2 QR for Least Squares

**Problem.** Given $A \in \mathbb{R}^{m \times n}$ ($m > n$) and $\mathbf{b} \in \mathbb{R}^m$, solve:
$$\min_{\mathbf{x} \in \mathbb{R}^n} \|A\mathbf{x} - \mathbf{b}\|_2^2$$

**Solution via QR.** Let $A = QR$ (thin QR, $Q^\top Q = I$):
$$\|A\mathbf{x} - \mathbf{b}\|^2 = \|QR\mathbf{x} - \mathbf{b}\|^2 = \|R\mathbf{x} - Q^\top\mathbf{b}\|^2 + \|(I - QQ^\top)\mathbf{b}\|^2$$

The second term is independent of $\mathbf{x}$, so the minimum occurs at:
$$R\mathbf{x} = Q^\top\mathbf{b}$$

This is an upper triangular system, solved by **back substitution** in $O(n^2)$ operations.

**Comparison with normal equations.** The normal equations $(A^\top A)\mathbf{x} = A^\top\mathbf{b}$ are equivalent but form $A^\top A$, which squares the condition number: $\kappa(A^\top A) = \kappa(A)^2$. QR avoids this squaring and is numerically preferred whenever $\kappa(A)$ is moderate.

### 7.3 QR Algorithm for Eigenvalues

**Preview.** The QR algorithm — the standard method for computing eigenvalues of symmetric matrices — works by iterating:
$$A_0 = A, \quad A_{k+1} = R_k Q_k \quad \text{where } A_k = Q_k R_k$$

Under mild conditions, $A_k \to$ upper triangular (Schur form) and the diagonal entries converge to eigenvalues. This connects orthogonal matrices directly to spectral theory.

> **Forward Reference:** The full QR algorithm with shifts is covered in [§08: Matrix Decompositions](../08-Matrix-Decompositions/notes.md). The eigenvalue theory it computes is developed in [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md).

---

## 8. The Spectral Theorem for Symmetric Matrices

### 8.1 Statement and Proof

**Theorem (Spectral Theorem).** Every real symmetric matrix $A = A^\top \in \mathbb{R}^{n \times n}$ has:

1. All eigenvalues are **real**: $\lambda_i \in \mathbb{R}$
2. Eigenvectors for distinct eigenvalues are **orthogonal**
3. There exists an orthonormal basis of eigenvectors: $A = Q\Lambda Q^\top$ where $Q \in O(n)$ and $\Lambda = \operatorname{diag}(\lambda_1, \ldots, \lambda_n)$

**Proof of (1):** Suppose $A\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \in \mathbb{C}^n$, $\mathbf{v} \neq \mathbf{0}$. Take the Hermitian inner product:
$$\bar{\mathbf{v}}^\top A\mathbf{v} = \lambda\bar{\mathbf{v}}^\top\mathbf{v} = \lambda\|\mathbf{v}\|^2$$
$$\overline{\bar{\mathbf{v}}^\top A\mathbf{v}} = \bar{\mathbf{v}}^\top A^\top\mathbf{v} = \bar{\mathbf{v}}^\top A\mathbf{v} \quad (\text{since } A = A^\top)$$
So $\lambda\|\mathbf{v}\|^2$ is real, hence $\lambda \in \mathbb{R}$. $\square$

**Proof of (2):** If $A\mathbf{u} = \lambda\mathbf{u}$ and $A\mathbf{v} = \mu\mathbf{v}$ with $\lambda \neq \mu$:
$$\lambda\langle\mathbf{u},\mathbf{v}\rangle = \langle A\mathbf{u},\mathbf{v}\rangle = \langle\mathbf{u},A^\top\mathbf{v}\rangle = \langle\mathbf{u},A\mathbf{v}\rangle = \mu\langle\mathbf{u},\mathbf{v}\rangle$$
$(\lambda - \mu)\langle\mathbf{u},\mathbf{v}\rangle = 0$. Since $\lambda \neq \mu$: $\langle\mathbf{u},\mathbf{v}\rangle = 0$. $\square$

**Proof of (3):** By induction. The base case ($n=1$) is trivial. For the inductive step: let $\mathbf{q}_1$ be a unit eigenvector for eigenvalue $\lambda_1$. Let $W = \mathbf{q}_1^\perp$ (the orthogonal complement). Since $A$ is symmetric, $W$ is $A$-invariant (if $\langle\mathbf{w},\mathbf{q}_1\rangle = 0$, then $\langle A\mathbf{w},\mathbf{q}_1\rangle = \langle\mathbf{w},A\mathbf{q}_1\rangle = \lambda_1\langle\mathbf{w},\mathbf{q}_1\rangle = 0$). Apply the inductive hypothesis to the $(n-1)\times(n-1)$ symmetric restriction $A|_W$. $\square$

### 8.2 The Rayleigh Quotient

**Definition.** The **Rayleigh quotient** of a symmetric matrix $A$ at vector $\mathbf{x} \neq \mathbf{0}$ is:
$$R_A(\mathbf{x}) = \frac{\mathbf{x}^\top A\mathbf{x}}{\mathbf{x}^\top\mathbf{x}}$$

**Key properties:**
- $R_A(\mathbf{q}_i) = \lambda_i$ for eigenvectors $\mathbf{q}_i$
- $\lambda_{\min} \leq R_A(\mathbf{x}) \leq \lambda_{\max}$ for all $\mathbf{x}$
- $\max_{\mathbf{x}} R_A(\mathbf{x}) = \lambda_{\max}$, achieved at the top eigenvector
- $\min_{\mathbf{x}} R_A(\mathbf{x}) = \lambda_{\min}$, achieved at the bottom eigenvector

*Proof of bounds:* Expand $\mathbf{x} = \sum_i c_i \mathbf{q}_i$ in the eigenbasis:
$$R_A(\mathbf{x}) = \frac{\sum_i c_i^2 \lambda_i}{\sum_i c_i^2}$$

This is a **weighted average** of eigenvalues, hence bounded by $[\lambda_{\min}, \lambda_{\max}]$. $\square$

**Courant-Fischer Min-Max Theorem.** The $k$-th largest eigenvalue satisfies:
$$\lambda_k = \max_{\dim S = k}\, \min_{\mathbf{x} \in S \setminus \{0\}} R_A(\mathbf{x}) = \min_{\dim S = n-k+1}\, \max_{\mathbf{x} \in S \setminus \{0\}} R_A(\mathbf{x})$$

This variational characterization is fundamental to perturbation theory and numerical linear algebra.

### 8.3 Quadratic Forms and Symmetric Matrices

A **quadratic form** is $f(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} = \sum_{i,j} A_{ij} x_i x_j$ for symmetric $A$.

**Positive definiteness via the spectral theorem:** $A \succ 0$ (positive definite) $\Leftrightarrow$ all eigenvalues $> 0$.

*Proof:* In the eigenbasis, $\mathbf{x}^\top A\mathbf{x} = \sum_i \lambda_i c_i^2$. This is positive for all $\mathbf{x} \neq \mathbf{0}$ iff all $\lambda_i > 0$. $\square$

**For AI:** The Hessian $\nabla^2\mathcal{L}$ of a loss function is symmetric. Its eigenvalues tell us the curvature in each direction — large eigenvalues are "sharp" directions (fast loss change), small eigenvalues are "flat" directions. The Rayleigh quotient appears in sharpness-aware minimization (SAM), gradient clipping analysis, and understanding training dynamics.

---

## 9. Orthogonality in Function Spaces

### 9.1 The L² Inner Product

Orthogonality extends naturally from finite-dimensional vector spaces to infinite-dimensional **function spaces**. The space $L^2([a,b])$ of square-integrable functions carries the inner product:
$$\langle f, g\rangle = \int_a^b f(x)g(x)\,dx$$

**Orthogonality of functions:** $f \perp g$ if $\int_a^b f(x)g(x)\,dx = 0$.

**Example:** On $[-\pi, \pi]$, the functions $\{\sin(nx), \cos(mx)\}$ form a mutually orthogonal family:
$$\int_{-\pi}^{\pi} \sin(nx)\cos(mx)\,dx = 0 \quad \text{for all } n, m$$
$$\int_{-\pi}^{\pi} \sin(nx)\sin(mx)\,dx = \pi\,\delta_{nm}$$
$$\int_{-\pi}^{\pi} \cos(nx)\cos(mx)\,dx = \pi\,\delta_{nm} \quad (n, m \geq 1)$$

This is the foundation of **Fourier series** — a basis expansion for $L^2$ functions.

### 9.2 Fourier Series as Orthonormal Expansion

**Orthonormal basis for $L^2([-\pi,\pi])$:**
$$\left\{\frac{1}{\sqrt{2\pi}},\, \frac{\cos(x)}{\sqrt{\pi}},\, \frac{\sin(x)}{\sqrt{\pi}},\, \frac{\cos(2x)}{\sqrt{\pi}},\, \frac{\sin(2x)}{\sqrt{\pi}},\, \ldots\right\}$$

The **Fourier series** of $f$ is its expansion in this ONB:
$$f(x) = \frac{a_0}{2} + \sum_{n=1}^\infty \left[a_n\cos(nx) + b_n\sin(nx)\right]$$

where the **Fourier coefficients** are orthogonal projections:
$$a_n = \frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\cos(nx)\,dx, \qquad b_n = \frac{1}{\pi}\int_{-\pi}^{\pi}f(x)\sin(nx)\,dx$$

**Parseval's theorem for Fourier series:**
$$\frac{1}{\pi}\int_{-\pi}^{\pi}|f(x)|^2\,dx = \frac{a_0^2}{2} + \sum_{n=1}^\infty(a_n^2 + b_n^2)$$

This is the infinite-dimensional analog of $\|\mathbf{v}\|^2 = \sum_i \hat{v}_i^2$.

### 9.3 Discrete Fourier Transform as a Unitary Matrix

The **Discrete Fourier Transform (DFT)** maps $\mathbf{x} \in \mathbb{C}^n$ to $\hat{\mathbf{x}} \in \mathbb{C}^n$:
$$\hat{x}_k = \sum_{j=0}^{n-1} x_j\, \omega^{jk}, \qquad \omega = e^{-2\pi i/n}$$

In matrix form: $\hat{\mathbf{x}} = F_n\mathbf{x}$ where $(F_n)_{jk} = \omega^{jk}$.

**Key fact:** $F_n / \sqrt{n}$ is **unitary** (the complex analog of orthogonal):
$$\frac{1}{n}F_n^* F_n = I \quad \Leftrightarrow \quad \frac{F_n}{\sqrt{n}} \text{ is unitary}$$

This is because the DFT basis vectors $\mathbf{f}_k = (1, \omega^k, \omega^{2k}, \ldots, \omega^{(n-1)k})/\sqrt{n}$ are orthonormal in $\mathbb{C}^n$:
$$\langle\mathbf{f}_j, \mathbf{f}_k\rangle = \frac{1}{n}\sum_{\ell=0}^{n-1}\omega^{\ell(k-j)} = \delta_{jk}$$

(The last equality is the geometric series identity for roots of unity.)

**Parseval's theorem for DFT:** $\|\hat{\mathbf{x}}\|^2 = n\|\mathbf{x}\|^2$.

**For AI:** The DFT/FFT appears in:
- **Signal processing:** Feature extraction from audio (MFCCs, spectrograms)
- **Efficient convolutions:** $f * g = \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g))$ — used in efficient CNN implementations
- **Positional encodings:** Sinusoidal positional embeddings in the original Transformer are Fourier features
- **SSMs:** State space models (Mamba, S4) use structured orthogonal/unitary matrices

---

## 10. Applications in Machine Learning

### 10.1 Orthogonal Weight Initialization

**The problem with random initialization.** If weight matrices have singular values spread over a wide range, gradients either explode or vanish during backpropagation through many layers.

**Saxe et al. (2013)** showed that initializing with orthogonal matrices preserves gradient norms through linear networks:

**Theorem.** For a deep linear network with orthogonal weight matrices $W_1, \ldots, W_L$ (each $\in \mathbb{R}^{n \times n}$), the Jacobian of the full network is $W_L \cdots W_1$, which is also orthogonal (product of orthogonal matrices). Hence $\|\partial\mathcal{L}/\partial\mathbf{x}\| = \|\partial\mathcal{L}/\partial\mathbf{y}\|$ — gradients neither explode nor vanish.

**Practical implementation:**

1. Generate a random $n \times n$ matrix with i.i.d. standard normal entries: $M \sim \mathcal{N}(0, I)$
2. Compute its QR decomposition: $M = QR$
3. Use $Q$ (or $Q \cdot \operatorname{sign}(\operatorname{diag}(R))$ for uniform distribution over $O(n)$) as the initial weight matrix

This is implemented as `torch.nn.init.orthogonal_()` in PyTorch.

**Why it works beyond linear networks:** Even in nonlinear networks, orthogonal initialization places the initial weights in a "good" part of weight space where gradients are well-conditioned at initialization. Empirically, networks with orthogonal initialization often converge faster on deep architectures.

### 10.2 Rotary Position Embeddings (RoPE)

**Standard self-attention** lacks inherent position information. The original Transformer adds sinusoidal absolute position embeddings. **RoPE** (Su et al. 2021, used in LLaMA, GPT-NeoX, Falcon) takes a different approach: it encodes position by rotating query and key vectors.

**Construction (2D case).** For position $m$, the rotation matrix is:
$$R_m = \begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix}$$

For a $d$-dimensional embedding, pair up dimensions and apply $R_m$ to each pair using $d/2$ different frequencies $\theta_i = 10000^{-2i/d}$.

**Key property.** The inner product of rotated queries and keys depends only on the relative position $m - n$:
$$\langle R_m \mathbf{q}, R_n \mathbf{k}\rangle = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{m-n}\mathbf{k}$$

since $R_m^\top R_n = R_{n-m}$ (rotation matrices satisfy $R_a^\top R_b = R_{b-a}$, because the inverse of a rotation is a rotation in the opposite direction).

**Orthogonality is central:** The key equation $R_m^\top R_n = R_{m-n}$ uses the fact that $R_m$ is orthogonal ($R_m^\top = R_m^{-1}$), so $(R_m^\top)(R_n) = R_m^{-1}R_n = R_{n-m}$.

### 10.3 Spectral Normalization

**Spectral normalization** (Miyato et al. 2018) stabilizes GAN training by constraining weight matrices to have spectral norm (= largest singular value) equal to 1.

**Implementation:** At each forward pass, divide the weight matrix by its spectral norm: $\hat{W} = W/\sigma_1(W)$.

**Connection to orthogonality:** A matrix with $\sigma_1 = \sigma_2 = \cdots = 1$ is exactly an orthogonal matrix. Spectral normalization enforces $\sigma_1 = 1$ while letting other singular values vary freely — it's a "one-sided" orthogonality constraint.

**Why it helps GANs:** The discriminator $D$ satisfies a Lipschitz condition $\|D(\mathbf{x}) - D(\mathbf{y})\| \leq \|\mathbf{x} - \mathbf{y}\|$ when all weight layers are spectrally normalized. This relates to the Wasserstein GAN objective, which requires a 1-Lipschitz discriminator.

### 10.4 QR in Optimization: Orthogonal Gradients

Several modern optimization techniques use orthogonality:

**Shampoo optimizer** (Gupta et al. 2018): Maintains a Kronecker-factored preconditioner and periodically computes its orthogonal "square root" via eigendecomposition, orthogonalizing the gradient update directions.

**Muon optimizer** (2024): Orthogonalizes gradient matrices via Newton-Schulz iterations (a matrix-valued version of Newton's method for computing polar decomposition), then applies the orthogonalized gradient as an update. This ensures each weight matrix sees updates with balanced singular values, preventing the accumulation of rank deficiency.

**Gradient orthogonality in continual learning:** Methods like Gradient Episodic Memory (GEM) and Orthogonal Gradient Descent (OGD) project gradients for new tasks onto the orthogonal complement of the gradient space for previous tasks — preserving past performance while learning new tasks.

---

---

## 10b. Bessel's Inequality and Completeness

### 10b.1 Bessel's Inequality

When we expand a vector $\mathbf{v}$ in terms of an orthonormal set $\{\mathbf{q}_1, \ldots, \mathbf{q}_k\}$ that is not necessarily a complete basis for the entire space, we get **Bessel's inequality**:

$$\sum_{i=1}^k |\langle\mathbf{v}, \mathbf{q}_i\rangle|^2 \leq \|\mathbf{v}\|^2$$

**Proof.** Let $\mathbf{v}_k = \sum_{i=1}^k \langle\mathbf{v},\mathbf{q}_i\rangle\mathbf{q}_i$ be the projection of $\mathbf{v}$ onto $\operatorname{span}(\mathbf{q}_1,\ldots,\mathbf{q}_k)$. Then:

$$\|\mathbf{v} - \mathbf{v}_k\|^2 = \|\mathbf{v}\|^2 - \|\mathbf{v}_k\|^2 = \|\mathbf{v}\|^2 - \sum_{i=1}^k |\langle\mathbf{v},\mathbf{q}_i\rangle|^2 \geq 0$$

The inequality follows immediately. $\square$

**Equality** holds if and only if $\mathbf{v} \in \operatorname{span}(\mathbf{q}_1,\ldots,\mathbf{q}_k)$ — i.e., when $\mathbf{v}$ itself lies in the subspace spanned by the chosen orthonormal vectors.

**Parseval's identity** is the limit case: when $\{\mathbf{q}_i\}$ is a complete ONB:
$$\sum_{i=1}^\infty |\langle\mathbf{v}, \mathbf{q}_i\rangle|^2 = \|\mathbf{v}\|^2$$

### 10b.2 Completeness of Orthonormal Sets

An orthonormal set $\mathcal{B} = \{\mathbf{q}_i\}$ in a Hilbert space $\mathcal{H}$ is **complete** (or **maximal**) if it satisfies any of the following equivalent conditions:

1. $\mathcal{B}$ is an ONB: every $\mathbf{v} \in \mathcal{H}$ can be written $\mathbf{v} = \sum_i \langle\mathbf{v},\mathbf{q}_i\rangle\mathbf{q}_i$
2. Parseval's identity holds: $\|\mathbf{v}\|^2 = \sum_i |\langle\mathbf{v},\mathbf{q}_i\rangle|^2$ for all $\mathbf{v}$
3. $\langle\mathbf{v},\mathbf{q}_i\rangle = 0$ for all $i$ implies $\mathbf{v} = \mathbf{0}$ (no non-zero vector is orthogonal to all $\mathbf{q}_i$)

In finite dimensions, any ONB is automatically complete (by dimension counting). In infinite dimensions, completeness is a non-trivial requirement — this is why Hilbert space theory requires careful treatment.

**For ML context:** In kernel methods, the RKHS is an infinite-dimensional Hilbert space. A kernel function $k(\mathbf{x},\mathbf{y})$ defines an inner product, and the question of whether the kernel features span the whole space (completeness/universality) determines the expressivity of the corresponding kernel machine.

---

## 10c. Detailed Worked Examples

### 10c.1 Projection in Least-Squares Regression

**Setup.** We observe $m = 4$ data points $(x_i, y_i)$ and want to fit a line $y = \beta_0 + \beta_1 x$:

$$x = (1, 2, 3, 4)^\top, \quad y = (2.1, 3.9, 5.8, 8.2)^\top$$

**Build the design matrix:**
$$A = \begin{pmatrix}1 & 1 \\ 1 & 2 \\ 1 & 3 \\ 1 & 4\end{pmatrix}$$

**Gram-Schmidt on the columns of $A$:**

Column 1: $\mathbf{a}_1 = (1,1,1,1)^\top$

$$\mathbf{q}_1 = \frac{\mathbf{a}_1}{\|\mathbf{a}_1\|} = \frac{1}{2}(1,1,1,1)^\top$$

Column 2: $\mathbf{a}_2 = (1,2,3,4)^\top$. Project out $\mathbf{q}_1$:

$$\langle\mathbf{a}_2, \mathbf{q}_1\rangle = \frac{1}{2}(1+2+3+4) = 5$$

$$\mathbf{u}_2 = \mathbf{a}_2 - 5\mathbf{q}_1 = (1,2,3,4)^\top - \frac{5}{2}(1,1,1,1)^\top = (-\tfrac{3}{2}, -\tfrac{1}{2}, \tfrac{1}{2}, \tfrac{3}{2})^\top$$

$$\|\mathbf{u}_2\| = \sqrt{\tfrac{9}{4}+\tfrac{1}{4}+\tfrac{1}{4}+\tfrac{9}{4}} = \sqrt{5}, \quad \mathbf{q}_2 = \frac{1}{\sqrt{5}}(-\tfrac{3}{2}, -\tfrac{1}{2}, \tfrac{1}{2}, \tfrac{3}{2})^\top$$

**The projection (fitted values):**
$$\hat{\mathbf{y}} = QQ^\top\mathbf{y} \quad \text{where } Q = [\mathbf{q}_1|\mathbf{q}_2]$$

This is the projection of $\mathbf{y}$ onto $\operatorname{col}(A)$ = the space of all affine functions of $x$. The residual $\mathbf{y} - \hat{\mathbf{y}}$ is orthogonal to every affine function.

**Verification of the normal equations:** The least-squares solution $\hat{\boldsymbol{\beta}}$ satisfies $A^\top A\hat{\boldsymbol{\beta}} = A^\top\mathbf{y}$:

$$A^\top A = \begin{pmatrix}4 & 10 \\ 10 & 30\end{pmatrix}, \quad A^\top\mathbf{y} = \begin{pmatrix}20 \\ 56.8\end{pmatrix}$$

Solving: $\hat{\boldsymbol{\beta}} \approx (0.1, 2.04)^\top$, so $\hat{y} \approx 0.1 + 2.04x$ — a good linear fit.

### 10c.2 Full Worked Example: Gram-Schmidt in $\mathbb{R}^3$

**Input vectors:**
$$\mathbf{a}_1 = \begin{pmatrix}1\\1\\0\end{pmatrix}, \quad \mathbf{a}_2 = \begin{pmatrix}1\\0\\1\end{pmatrix}, \quad \mathbf{a}_3 = \begin{pmatrix}0\\1\\1\end{pmatrix}$$

**Step 1:** $\mathbf{q}_1 = \mathbf{a}_1/\|\mathbf{a}_1\| = (1/\sqrt{2})(1,1,0)^\top$

**Step 2:** Remove the component of $\mathbf{a}_2$ along $\mathbf{q}_1$:

$$\langle\mathbf{a}_2, \mathbf{q}_1\rangle = \frac{1}{\sqrt{2}}(1 \cdot 1 + 1 \cdot 0 + 0 \cdot 1) = \frac{1}{\sqrt{2}}$$

$$\mathbf{u}_2 = \mathbf{a}_2 - \frac{1}{\sqrt{2}}\mathbf{q}_1 = \begin{pmatrix}1\\0\\1\end{pmatrix} - \frac{1}{2}\begin{pmatrix}1\\1\\0\end{pmatrix} = \begin{pmatrix}1/2\\-1/2\\1\end{pmatrix}$$

$$\|\mathbf{u}_2\| = \sqrt{1/4 + 1/4 + 1} = \sqrt{3/2}, \quad \mathbf{q}_2 = \sqrt{\frac{2}{3}}\begin{pmatrix}1/2\\-1/2\\1\end{pmatrix} = \frac{1}{\sqrt{6}}\begin{pmatrix}1\\-1\\2\end{pmatrix}$$

**Step 3:** Remove components of $\mathbf{a}_3$ along $\mathbf{q}_1$ and $\mathbf{q}_2$:

$$\langle\mathbf{a}_3, \mathbf{q}_1\rangle = \frac{1}{\sqrt{2}}(0+1+0) = \frac{1}{\sqrt{2}}$$

$$\langle\mathbf{a}_3, \mathbf{q}_2\rangle = \frac{1}{\sqrt{6}}(0-1+2) = \frac{1}{\sqrt{6}}$$

$$\mathbf{u}_3 = \begin{pmatrix}0\\1\\1\end{pmatrix} - \frac{1}{\sqrt{2}}\cdot\frac{1}{\sqrt{2}}\begin{pmatrix}1\\1\\0\end{pmatrix} - \frac{1}{\sqrt{6}}\cdot\frac{1}{\sqrt{6}}\begin{pmatrix}1\\-1\\2\end{pmatrix}$$

$$= \begin{pmatrix}0\\1\\1\end{pmatrix} - \frac{1}{2}\begin{pmatrix}1\\1\\0\end{pmatrix} - \frac{1}{6}\begin{pmatrix}1\\-1\\2\end{pmatrix} = \begin{pmatrix}-2/3\\2/3\\2/3\end{pmatrix}$$

$$\|\mathbf{u}_3\| = \sqrt{4/9+4/9+4/9} = 2/\sqrt{3}, \quad \mathbf{q}_3 = \frac{1}{\sqrt{3}}\begin{pmatrix}-1\\1\\1\end{pmatrix}$$

**Verification:** $\langle\mathbf{q}_1,\mathbf{q}_2\rangle = (1/\sqrt{2})(1/\sqrt{6})(1-1+0) = 0$ ✓
$\langle\mathbf{q}_1,\mathbf{q}_3\rangle = (1/\sqrt{2})(1/\sqrt{3})(-1+1+0) = 0$ ✓
$\langle\mathbf{q}_2,\mathbf{q}_3\rangle = (1/\sqrt{6})(1/\sqrt{3})(-1-1+2) = 0$ ✓

**QR factorization:** The matrix $R$ is upper triangular with:
$$R = \begin{pmatrix}\|\mathbf{a}_1\| & \langle\mathbf{a}_2,\mathbf{q}_1\rangle & \langle\mathbf{a}_3,\mathbf{q}_1\rangle \\ 0 & \|\mathbf{u}_2\| & \langle\mathbf{a}_3,\mathbf{q}_2\rangle \\ 0 & 0 & \|\mathbf{u}_3\|\end{pmatrix} = \begin{pmatrix}\sqrt{2} & 1/\sqrt{2} & 1/\sqrt{2} \\ 0 & \sqrt{3/2} & 1/\sqrt{6} \\ 0 & 0 & 2/\sqrt{3}\end{pmatrix}$$

### 10c.3 Spectral Decomposition Example

Let $A = \begin{pmatrix}3 & 1 \\ 1 & 3\end{pmatrix}$ (symmetric).

**Eigenvalues:** $\det(A - \lambda I) = (3-\lambda)^2 - 1 = 0 \Rightarrow \lambda = 2$ or $\lambda = 4$.

**Eigenvectors:**
- $\lambda_1 = 2$: $(A - 2I)\mathbf{v} = 0 \Rightarrow \begin{pmatrix}1&1\\1&1\end{pmatrix}\mathbf{v} = 0 \Rightarrow \mathbf{v}_1 = (1,-1)^\top/\sqrt{2}$
- $\lambda_2 = 4$: $(A - 4I)\mathbf{v} = 0 \Rightarrow \begin{pmatrix}-1&1\\1&-1\end{pmatrix}\mathbf{v} = 0 \Rightarrow \mathbf{v}_2 = (1,1)^\top/\sqrt{2}$

**Verify orthogonality:** $\mathbf{v}_1^\top\mathbf{v}_2 = (1)(1) + (-1)(1) = 0$ ✓

**Spectral decomposition:**
$$A = \lambda_1\mathbf{v}_1\mathbf{v}_1^\top + \lambda_2\mathbf{v}_2\mathbf{v}_2^\top = 2\begin{pmatrix}1/2 & -1/2 \\ -1/2 & 1/2\end{pmatrix} + 4\begin{pmatrix}1/2 & 1/2 \\ 1/2 & 1/2\end{pmatrix}$$

$$= \begin{pmatrix}1 & -1 \\ -1 & 1\end{pmatrix} + \begin{pmatrix}2 & 2 \\ 2 & 2\end{pmatrix} = \begin{pmatrix}3 & 1 \\ 1 & 3\end{pmatrix} = A \checkmark$$

**Rayleigh quotient at $\mathbf{x} = (1,0)^\top$:**
$$R_A(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} = (1,0)\begin{pmatrix}3 & 1 \\ 1 & 3\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix} = 3$$

Indeed $\lambda_{\min} = 2 \leq 3 \leq 4 = \lambda_{\max}$.

### 10c.4 Householder Reflector: Concrete Computation

**Problem:** Find a Householder reflector $H$ such that $H\mathbf{a} = \|\mathbf{a}\|\mathbf{e}_1$ where $\mathbf{a} = (4, 3)^\top$.

**Step 1:** $\|\mathbf{a}\| = \sqrt{16+9} = 5$.

**Step 2:** Since $a_1 = 4 > 0$, use $+$ sign to avoid cancellation:
$$\mathbf{n}_{\text{unnorm}} = \mathbf{a} + \|\mathbf{a}\|\mathbf{e}_1 = \begin{pmatrix}4\\3\end{pmatrix} + \begin{pmatrix}5\\0\end{pmatrix} = \begin{pmatrix}9\\3\end{pmatrix}$$

**Step 3:** Normalize: $\|\mathbf{n}_{\text{unnorm}}\| = \sqrt{81+9} = \sqrt{90} = 3\sqrt{10}$

$$\mathbf{n} = \frac{1}{3\sqrt{10}}\begin{pmatrix}9\\3\end{pmatrix} = \frac{1}{\sqrt{10}}\begin{pmatrix}3\\1\end{pmatrix}$$

**Step 4:** Build the reflector:
$$H = I - 2\mathbf{n}\mathbf{n}^\top = \begin{pmatrix}1 & 0 \\ 0 & 1\end{pmatrix} - \frac{2}{10}\begin{pmatrix}9 & 3 \\ 3 & 1\end{pmatrix} = \begin{pmatrix}1-9/5 & -3/5 \\ -3/5 & 1-1/5\end{pmatrix} = \begin{pmatrix}-4/5 & -3/5 \\ -3/5 & 4/5\end{pmatrix}$$

**Verify:** $H\mathbf{a} = \begin{pmatrix}-4/5 & -3/5 \\ -3/5 & 4/5\end{pmatrix}\begin{pmatrix}4\\3\end{pmatrix} = \begin{pmatrix}-16/5-9/5\\-12/5+12/5\end{pmatrix} = \begin{pmatrix}-5\\0\end{pmatrix} = -5\mathbf{e}_1$

This gives $-\|\mathbf{a}\|\mathbf{e}_1$ (a valid Householder result; the sign depends on the convention). $\det(H) = -1$ as expected. ✓

---

## 10d. Orthogonality and the Geometry of Neural Networks

### 10d.1 The Geometry of Weight Matrices

A single linear layer $\mathbf{y} = W\mathbf{x}$ transforms the input geometry. Understanding this transformation requires decomposing $W$ via its **singular value decomposition** (preview):

$$W = U\Sigma V^\top \qquad \text{(covered fully in §02: SVD)}$$

- $V^\top$: rotate the input space (first orthogonal transformation)
- $\Sigma$: scale along the new axes by singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$
- $U$: rotate the output space (second orthogonal transformation)

**For orthogonal $W$:** All singular values are 1, so $\Sigma = I$. The layer is a pure rotation — it doesn't compress or expand the input distribution. Distances and angles are perfectly preserved.

**Implication for gradient flow.** The gradient of the loss with respect to the input is:

$$\frac{\partial\mathcal{L}}{\partial\mathbf{x}} = W^\top\frac{\partial\mathcal{L}}{\partial\mathbf{y}}$$

For orthogonal $W$: $\|W^\top\mathbf{g}\| = \|\mathbf{g}\|$ — gradients are neither amplified nor attenuated. This is the formal reason orthogonal initialization prevents gradient vanishing/explosion in deep linear networks.

### 10d.2 Attention Heads as Orthogonal Projections

In a Transformer's multi-head attention with $H$ heads:

$$\text{head}_h = \text{Attn}(XW_h^Q, XW_h^K, XW_h^V)$$

where $W_h^Q, W_h^K, W_h^V \in \mathbb{R}^{d \times d_k}$ are the query/key/value projection matrices.

**Mechanistic interpretability perspective** (Elhage et al. 2021): Each attention head computes a **projection** of the residual stream onto a low-dimensional subspace, adds information from that subspace back, and (in some sense) different heads should attend to different "directions" of information.

**Mathematical idealization:** If we require $W_h^V (W_{h'}^V)^\top = 0$ for $h \neq h'$ (the value matrices project to orthogonal subspaces), then different heads genuinely operate on independent information. In practice, heads aren't exactly orthogonal, but studying the **overlap** $\|W_h^V (W_{h'}^V)^\top\|_F$ quantifies how much heads interfere with each other.

**QK circuit analysis.** The effective attention pattern for head $h$ involves $W_h^Q (W_h^K)^\top \in \mathbb{R}^{d \times d}$ (or rather its action on the residual stream). This matrix's singular values determine how "sharp" vs "diffuse" the attention pattern is.

### 10d.3 Layer Normalization and Orthogonal Complements

**Layer normalization** (Ba et al. 2016) operates on a vector $\mathbf{x} \in \mathbb{R}^d$:

$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu\mathbf{1}}{\sigma} \cdot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

where $\mu = \frac{1}{d}\sum_i x_i$ and $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$.

**Orthogonal decomposition view.** The centering step $\mathbf{x} \mapsto \mathbf{x} - \mu\mathbf{1}$ is an **orthogonal projection** onto the subspace $\mathbf{1}^\perp = \{\mathbf{v} \in \mathbb{R}^d : \sum_i v_i = 0\}$:

$$P_{\mathbf{1}^\perp} = I - \frac{\mathbf{1}\mathbf{1}^\top}{d}$$

This is a rank-$(d-1)$ orthogonal projector. Layer normalization first projects out the "mean direction" $\mathbf{1}/\sqrt{d}$, then rescales to unit norm on the remaining $(d-1)$-dimensional subspace.

**Consequence for representational geometry:** After layer normalization, the model's internal representations $\mathbf{h}$ always satisfy $\mathbf{h} \perp \mathbf{1}$. The model can only represent information in the orthogonal complement of the constant vector — this is why all-zero inputs and constant inputs are indistinguishable after layer norm.

### 10d.4 Orthogonal Regularization in Weight Matrices

Several training techniques maintain (near-)orthogonality in weight matrices throughout training:

**Orthogonal regularization:** Add a penalty $\lambda\|W^\top W - I\|_F^2$ to the loss. This penalizes deviation from orthogonality without enforcing it exactly.

**Spectral regularization:** Penalize $\lambda \sum_i (\sigma_i - 1)^2$ — penalizes singular values deviating from 1. Stronger than spectral normalization, which only clips the largest singular value.

**Riemannian optimization on the Stiefel manifold:** The set of $n \times k$ matrices with orthonormal columns, $\text{St}(n,k) = \{Q \in \mathbb{R}^{n \times k} : Q^\top Q = I_k\}$, is a smooth Riemannian manifold. Gradient descent on $\text{St}(n,k)$ uses the Riemannian gradient (projection of Euclidean gradient onto the tangent space) and retracts to the manifold via QR decomposition or Cayley transform after each step.

**Applications:** Orthogonal RNNs, rotation-equivariant neural networks, invertible neural networks, and normalizing flows all benefit from exact or approximate orthogonality constraints.

---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|---------------|-----|
| 1 | Confusing "orthogonal" and "orthonormal" | Orthogonal means $\langle\mathbf{u},\mathbf{v}\rangle = 0$; orthonormal additionally requires unit length. An orthogonal matrix $Q$ has orthonormal columns, not just orthogonal ones. | Check: orthogonal ↔ zero inner products; orthonormal ↔ zero inner products AND $\|\mathbf{q}_i\| = 1$ |
| 2 | Assuming $QQ^\top = I$ implies $Q^\top Q = I$ without checking square | For non-square $Q \in \mathbb{R}^{m \times n}$ ($m > n$), $Q^\top Q = I_n$ but $QQ^\top \neq I_m$ (it's a rank-$n$ projection). Only square orthogonal matrices satisfy both. | Specify dimensions: "thin Q" ($Q^\top Q = I_n$) vs "full Q" ($QQ^\top = I_m = Q^\top Q$) |
| 3 | Thinking Gram-Schmidt preserves span ordering | CGS does: $\operatorname{span}(\mathbf{q}_1,\ldots,\mathbf{q}_k) = \operatorname{span}(\mathbf{a}_1,\ldots,\mathbf{a}_k)$. But with pivoting or reordering, this property is lost. | Track column ordering explicitly when pivoting QR is used |
| 4 | Using CGS when numerical stability matters | Classical Gram-Schmidt squares the error: $O(\epsilon_\text{mach}\kappa(A)^2)$ vs MGS's $O(\epsilon_\text{mach}\kappa(A))$. For ill-conditioned $A$, CGS can produce non-orthogonal "orthonormal" vectors. | Use Modified Gram-Schmidt (MGS) or Householder QR for production code |
| 5 | Projecting onto a non-orthogonal basis | The formula $P = QQ^\top$ assumes $Q^\top Q = I$. If the columns are merely linearly independent (not orthonormal), the correct formula is $P = Q(Q^\top Q)^{-1}Q^\top$, which involves the $(Q^\top Q)$ inverse. | Always check if $Q^\top Q = I$ before using the simplified projection formula |
| 6 | Thinking $P^2 = P$ is sufficient for orthogonal projection | Idempotence $P^2 = P$ only guarantees $P$ is *some* projection. Orthogonal projections additionally require $P^\top = P$ (symmetry). Oblique projections satisfy idempotence but not symmetry. | An orthogonal projection must be both idempotent ($P^2 = P$) and symmetric ($P^\top = P$) |
| 7 | Claiming eigenvectors are always orthogonal | Eigenvectors of different eigenvalues are orthogonal only for **symmetric** matrices. For general matrices, eigenvectors can be arbitrary non-orthogonal vectors (or even non-real). | The spectral theorem applies to symmetric/Hermitian matrices; general matrices need Schur decomposition |
| 8 | Forgetting the sign convention in Householder | When computing $\mathbf{n} = (\mathbf{a} \pm \|\mathbf{a}\|\mathbf{e}_1)/\|\cdots\|$, the sign should be chosen as $+\operatorname{sign}(a_1)$ to avoid cancellation when $a_1 > 0$. | Always choose the sign of $\pm\|\mathbf{a}\|\mathbf{e}_1$ to be opposite the sign of $a_1$: $\mathbf{n} = (\mathbf{a} + \operatorname{sign}(a_1)\|\mathbf{a}\|\mathbf{e}_1)/\|\cdots\|$ |
| 9 | Misinterpreting Parseval's identity as an approximation | $\|\mathbf{v}\|^2 = \sum_i \hat{v}_i^2$ is an equality when $\{\mathbf{q}_i\}$ is a complete orthonormal basis for the whole space. If it's only a basis for a subspace, you get Bessel's inequality: $\sum_i \hat{v}_i^2 \leq \|\mathbf{v}\|^2$. | Distinguish: complete ONB → Parseval's equality; partial ONB → Bessel's inequality |
| 10 | Assuming QR is always better than Cholesky for normal equations | QR avoids squaring the condition number but costs more flops ($O(mn^2)$ vs $O(mn^2/2 + n^3/6)$ for Cholesky). For well-conditioned problems with large $m$, Cholesky on $A^\top A$ may be faster. | Choose based on condition number: if $\kappa(A) \gg 1$, use QR; if $\kappa(A) \approx 1$, Cholesky is acceptable and faster |
| 11 | Using $\det(Q) = 1$ as the definition of orthogonal | $\det(Q) = \pm 1$ is a **consequence** of orthogonality, not the definition. A matrix can have determinant 1 without being orthogonal (e.g., any upper triangular matrix with diagonal product 1). | Definition is $Q^\top Q = I$. Determinant $= \pm 1$ follows from this. |
| 12 | Thinking orthogonal initialization prevents gradient problems throughout training | Orthogonal initialization helps at initialization. As training proceeds, weights deviate from orthogonality. Maintaining orthogonality throughout training requires explicit regularization (e.g., spectral normalization, orthogonal regularization loss). | Use orthogonal init as a starting point; for persistent orthogonality through training, use explicit constraints |

---

## 12. Exercises

**Exercise 1 ★ — Projection and Decomposition**

Given $\mathbf{v} = (3, 1, 2)^\top$ and the subspace $S = \operatorname{span}\{(1, 1, 0)^\top, (0, 1, 1)^\top\}$:

**(a)** Apply Gram-Schmidt to the spanning vectors to obtain an ONB $\{\mathbf{q}_1, \mathbf{q}_2\}$ for $S$.

**(b)** Compute the projection $\mathbf{v}_S = \operatorname{proj}_S(\mathbf{v})$ using the ONB.

**(c)** Verify: $\mathbf{v} - \mathbf{v}_S \perp \mathbf{q}_1$ and $\mathbf{v} - \mathbf{v}_S \perp \mathbf{q}_2$.

**(d)** Compute $\|\mathbf{v}\|^2$ and verify $\|\mathbf{v}\|^2 = \|\mathbf{v}_S\|^2 + \|\mathbf{v} - \mathbf{v}_S\|^2$ (Pythagorean theorem).

---

**Exercise 2 ★ — Gram-Schmidt and QR**

Let $A = \begin{pmatrix}1 & 1 & 0 \\ 1 & 0 & 1 \\ 0 & 1 & 1\end{pmatrix}$.

**(a)** Apply Gram-Schmidt to the columns of $A$ to produce $Q$ and $R$ such that $A = QR$.

**(b)** Verify that $Q^\top Q = I$ and $QR = A$ numerically.

**(c)** Solve the least-squares problem $\min_\mathbf{x}\|A\mathbf{x} - \mathbf{b}\|$ for $\mathbf{b} = (1,2,3)^\top$ by solving $R\mathbf{x} = Q^\top\mathbf{b}$ via back substitution.

**(d)** Verify the solution against `np.linalg.lstsq`.

---

**Exercise 3 ★ — Orthogonal Matrices**

**(a)** Verify that the $2\times 2$ rotation matrix $R_\theta = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}$ is orthogonal for $\theta = \pi/4$.

**(b)** Show that the composition of two rotations $R_\theta R_\phi = R_{\theta+\phi}$ by matrix multiplication.

**(c)** Construct a Householder reflector $H$ such that $H\mathbf{a} = \|\mathbf{a}\|\mathbf{e}_1$ for $\mathbf{a} = (3, 4)^\top$. Verify $H^\top H = I$ and $H\mathbf{a} = (5, 0)^\top$.

**(d)** What is $\det(H)$? Explain geometrically.

---

**Exercise 4 ★★ — Modified Gram-Schmidt**

**(a)** Implement both Classical and Modified Gram-Schmidt.

**(b)** Test on the Hilbert matrix $H_{ij} = 1/(i+j-1)$ for $n = 10$. Compute the maximum off-diagonal entry of $Q^\top Q - I$ for each method.

**(c)** Plot orthogonality error ($\|Q^\top Q - I\|_F$) vs matrix size $n$ for both methods. What do you observe?

**(d)** Explain why MGS is more stable using the error analysis from §5.2.

---

**Exercise 5 ★★ — Least Squares Stability**

Given the Vandermonde system fitting a degree-$d$ polynomial through $m$ points:

**(a)** Build the design matrix $A \in \mathbb{R}^{m \times (d+1)}$ with $A_{ij} = x_i^{j-1}$ for $m = 20$, $d = 10$.

**(b)** Solve the least-squares problem via: (i) normal equations $(A^\top A)\hat{\mathbf{c}} = A^\top\mathbf{y}$, (ii) QR factorization $A = QR$ then $R\hat{\mathbf{c}} = Q^\top\mathbf{y}$.

**(c)** Compute the condition number $\kappa(A)$ and $\kappa(A^\top A)$. What is their ratio?

**(d)** Perturb $\mathbf{y}$ by small noise and compare the residuals from both methods. Which is more stable?

---

**Exercise 6 ★★ — Spectral Theorem**

Let $A = \begin{pmatrix}4 & 2 \\ 2 & 1\end{pmatrix}$.

**(a)** Find all eigenvalues and eigenvectors of $A$ analytically.

**(b)** Verify that eigenvectors are orthogonal. Normalize them to form an orthogonal matrix $Q$.

**(c)** Verify the spectral decomposition: $A = Q\Lambda Q^\top$ where $\Lambda = \operatorname{diag}(\lambda_1, \lambda_2)$.

**(d)** Compute the Rayleigh quotient $R_A(\mathbf{x})$ for $\mathbf{x} = (1, 0)^\top$, $(0, 1)^\top$, $(1, 1)^\top/\sqrt{2}$. Verify each is in $[\lambda_{\min}, \lambda_{\max}]$.

**(e)** Is $A$ positive definite? Verify using eigenvalues and by direct computation of $\mathbf{x}^\top A\mathbf{x}$.

---

**Exercise 7 ★★★ — Orthogonal Weight Initialization**

**(a)** Generate 100 random matrices $W_1, \ldots, W_L \in \mathbb{R}^{64 \times 64}$ with i.i.d. Gaussian entries (standard init). Compute the spectral norm of their product $\|W_L \cdots W_1\|_2$ for $L = 1, 5, 10, 20$.

**(b)** Repeat with each $W_k$ initialized as a random orthogonal matrix (via QR of a Gaussian matrix). Plot $\|W_L \cdots W_1\|_2$ vs $L$ for both initializations.

**(c)** Generate synthetic "gradient" vectors $\mathbf{g} \in \mathbb{R}^{64}$ and propagate backward through the chains. Compare gradient norms: $\|(W_L \cdots W_1)^\top\mathbf{g}\|$ vs $\|\mathbf{g}\|$.

**(d)** Explain the result using the isometry property of orthogonal matrices.

---

**Exercise 8 ★★★ — Rayleigh Quotient Iteration**

**Rayleigh Quotient Iteration** is a method for finding eigenvectors with cubic convergence:

$$\mathbf{x}^{(k+1)} = \frac{(A - \rho_k I)^{-1}\mathbf{x}^{(k)}}{\|(A - \rho_k I)^{-1}\mathbf{x}^{(k)}\|}, \qquad \rho_k = R_A(\mathbf{x}^{(k)})$$

**(a)** Implement Rayleigh Quotient Iteration for a symmetric matrix.

**(b)** Test on $A = \begin{pmatrix}3 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 1\end{pmatrix}$ with random initial $\mathbf{x}^{(0)}$. Track $|\rho_k - \lambda_\star|$ over iterations.

**(c)** Plot convergence on a log scale and estimate the convergence rate (should be cubic: $|\rho_{k+1} - \lambda_\star| \approx C|\rho_k - \lambda_\star|^3$).

**(d)** Compare convergence speed vs power iteration and inverse iteration with a fixed shift.

**(e)** Explain why Rayleigh quotient (not a fixed shift) is needed for cubic convergence.

---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | AI Application | Impact |
|---------|---------------|--------|
| **Orthogonal initialization** | `torch.nn.init.orthogonal_()` | Prevents gradient explosion/vanishing at initialization; standard practice for deep linear networks (Saxe et al.) and RNNs |
| **QR decomposition** | Muon optimizer; orthogonal gradient updates | Orthogonalizing gradient matrices via QR/Newton-Schulz stabilizes LLM training; avoids rank collapse of weight matrices |
| **RoPE embeddings** | LLaMA, Mistral, GPT-NeoX, Falcon, Gemma | Rotary position encoding uses orthogonal rotation matrices to encode relative position in self-attention; rotation composition gives relative position property |
| **Spectral normalization** | GANs, discriminator regularization | Constrains $\sigma_1(W) = 1$ for 1-Lipschitz discriminator; stabilizes Wasserstein GAN training |
| **Gram-Schmidt / QR** | Numerically stable least squares | Used in regression, feature selection, basis pursuit; avoiding normal equations prevents condition number squaring |
| **Orthogonal complement** | Continual learning (OGD, GEM) | New task gradients are projected onto the orthogonal complement of old task gradient subspaces; prevents catastrophic forgetting |
| **Spectral theorem** | Hessian analysis, SAM optimizer | Eigendecomposition of the loss Hessian reveals sharp/flat directions; Sharpness-Aware Minimization (SAM) seeks minima with small $\lambda_\max(\nabla^2\mathcal{L})$ |
| **Orthonormal bases** | Attention head analysis | Queries, keys, values in each attention head span subspaces; orthogonality between heads corresponds to specialization; mechanistic interpretability studies this |
| **DFT (unitary matrix)** | Audio models, efficient convolutions | Wav2Vec, Whisper process spectrograms (DFT magnitude); convolutional layers in frequency domain use FFT = matrix multiply by unitary $F_n$ |
| **Householder reflectors** | QR in neural network training | Used in computing QR factorizations during optimizer steps (Shampoo); Householder product form compresses orthogonal matrices for efficient multiplication |
| **Rayleigh quotient** | Eigenvalue estimation in training | Stochastic Lanczos quadrature (Ghorbani et al. 2019) approximates the Hessian spectrum via Rayleigh quotients on random probes |
| **Gram matrix** | Gram-matrix style loss, NTK | Neural Tangent Kernel $\Theta = J J^\top$ (where $J$ is the Jacobian) is a Gram matrix; its eigenstructure determines training dynamics |

---

## 14. Advanced Topics

### 14.1 Polar Decomposition

Every matrix $A \in \mathbb{R}^{m \times n}$ ($m \geq n$) has a **polar decomposition**:
$$A = QS$$

where $Q \in \mathbb{R}^{m \times n}$ has orthonormal columns ($Q^\top Q = I_n$) and $S \in \mathbb{R}^{n \times n}$ is symmetric positive semidefinite.

**Uniqueness:** If $A$ has full column rank, $Q$ and $S$ are unique, with $S = (A^\top A)^{1/2}$ (the matrix square root) and $Q = A(A^\top A)^{-1/2}$.

**Geometric meaning:** $S$ captures the "stretching" and $Q$ captures the "rotation" — this is the matrix analog of writing a complex number as $re^{i\theta}$.

**Relation to SVD:** If $A = U\Sigma V^\top$ (thin SVD), then:
$$A = (UV^\top)(V\Sigma V^\top) = Q \cdot S$$

So $Q = UV^\top$ (the "nearest orthogonal matrix" to $A$) and $S = V\Sigma V^\top$.

**For AI:** The Muon optimizer computes $Q = UV^\top$ from the SVD of the gradient matrix $G$, then applies $Q$ as the weight update — this is exactly the orthogonal factor of the polar decomposition of $G$.

### 14.2 Orthogonality in Hilbert Spaces

All finite-dimensional results extend to **Hilbert spaces** — complete inner product spaces, possibly infinite-dimensional.

**Orthogonal Projection Theorem (Hilbert spaces).** If $\mathcal{H}$ is a Hilbert space and $\mathcal{M} \subseteq \mathcal{H}$ is a closed subspace, then every $\mathbf{v} \in \mathcal{H}$ has a unique decomposition $\mathbf{v} = \mathbf{v}_\mathcal{M} + \mathbf{v}_{\mathcal{M}^\perp}$ with $\mathbf{v}_\mathcal{M} \in \mathcal{M}$ and $\mathbf{v}_{\mathcal{M}^\perp} \in \mathcal{M}^\perp$.

**Completeness is essential:** Without it, the projection may not exist. (The "closed" hypothesis ensures that sequences of projections converge.)

**Examples of Hilbert spaces in ML:**
- $L^2(\mathbb{R})$: the space of square-integrable functions (kernel methods, RKHS)
- $\ell^2$: square-summable sequences (infinite-dimensional limits of neural networks)
- Reproducing Kernel Hilbert Spaces (RKHS): the function space for kernel machines, Gaussian processes

### 14.3 Gram Matrices and the Kernel Trick

Given vectors $\mathbf{x}_1, \ldots, \mathbf{x}_m \in \mathbb{R}^d$, the **Gram matrix** is:
$$G_{ij} = \langle\mathbf{x}_i, \mathbf{x}_j\rangle = \mathbf{x}_i^\top\mathbf{x}_j$$

In matrix form: $G = XX^\top$ where $X = [\mathbf{x}_1|\cdots|\mathbf{x}_m]^\top \in \mathbb{R}^{m \times d}$.

**Properties:** $G$ is symmetric positive semidefinite. Its rank equals $\operatorname{rank}(X)$.

**Kernel trick:** Replace $\langle\mathbf{x}_i,\mathbf{x}_j\rangle$ with $k(\mathbf{x}_i,\mathbf{x}_j)$ where $k$ is a positive definite kernel. The resulting kernel matrix $K_{ij} = k(\mathbf{x}_i,\mathbf{x}_j)$ is a Gram matrix in a (possibly infinite-dimensional) RKHS.

**For attention:** The attention matrix $\text{Attn}_{ij} \propto \exp(\mathbf{q}_i^\top\mathbf{k}_j / \sqrt{d_k})$ is a kernelized Gram matrix between queries and keys. Its spectral properties determine how information flows between tokens.

---

## Conceptual Bridge

### Looking Backward: What Made This Possible

The theory developed in this section builds directly on foundations from earlier chapters:

- **Vectors and inner products** ([§01: Vectors and Spaces](../../02-Linear-Algebra-Basics/01-Vectors-and-Spaces/notes.md)): The inner product $\langle\mathbf{u},\mathbf{v}\rangle$ is the primitive notion from which orthogonality, angles, and projections all derive.
- **Linear transformations** ([§04: Linear Transformations](../04-Linear-Transformations/notes.md)): Orthogonal matrices are the isometric linear maps — those that preserve both angles and distances. The kernel and image of projection operators are orthogonal complements.
- **Matrix rank** ([§02-LA: Matrix Rank](../../02-Linear-Algebra-Basics/05-Matrix-Rank/notes.md)): The rank-nullity theorem and the four fundamental subspaces are unified by orthogonality: $\operatorname{null}(A) = \operatorname{row}(A)^\perp$.
- **Eigenvalues** ([§01: Eigenvalues](../01-Eigenvalues-and-Eigenvectors/notes.md)): The spectral theorem tells us that symmetric matrices are diagonalizable with an orthonormal eigenbasis — the connection between orthogonality and spectral theory.

### Looking Forward: What This Enables

Orthogonality and orthonormality are not endpoints but foundations:

- **Singular Value Decomposition** ([§02: SVD](../02-Singular-Value-Decomposition/notes.md)): The SVD $A = U\Sigma V^\top$ is built from two orthogonal matrices. The right singular vectors $V$ form an ONB for the row space; the left singular vectors $U$ form an ONB for the column space. Without the theory developed here, SVD cannot be understood.
- **Matrix Decompositions** ([§08: Matrix Decompositions](../08-Matrix-Decompositions/notes.md)): LU, QR, Cholesky, Schur — QR is the central algorithm among these, and the QR algorithm for eigenvalues iterates orthogonal similarity transformations.
- **Matrix Norms** ([§06: Matrix Norms](../06-Matrix-Norms/notes.md)): The spectral norm $\|A\|_2 = \sigma_1(A)$ is defined via orthogonal invariance. The condition number $\kappa(Q) = 1$ for orthogonal $Q$.
- **Optimization** ([Chapter 08](../../08-Optimization/README.md)): Gradient methods, second-order methods, and their variants all navigate spaces equipped with inner products. The geometry of the loss landscape is described using the spectral theory developed here.

```
CURRICULUM POSITION: ORTHOGONALITY AND ORTHONORMALITY
═══════════════════════════════════════════════════════════════════════════

  FOUNDATIONS                    THIS SECTION                  FORWARD
  ───────────                    ────────────                  ───────
  ┌─────────────────┐            ┌─────────────────────────┐   ┌──────────────────┐
  │ Inner Products  │──────────▶ │ Orthogonality (§05)     │──▶│ SVD (§02)        │
  │ Vectors/Spaces  │            │ ├─ Gram-Schmidt          │   │ low-rank approx  │
  └─────────────────┘            │ ├─ QR Decomposition      │   └──────────────────┘
                                 │ ├─ Orthogonal Matrices   │
  ┌─────────────────┐            │ ├─ Projections           │   ┌──────────────────┐
  │ Eigenvalues     │──────────▶ │ ├─ Spectral Theorem      │──▶│ Matrix Norms     │
  │ (§01)           │            │ └─ ML Applications       │   │ (§06)            │
  └─────────────────┘            └─────────────────────────┘   └──────────────────┘
                                          │
  ┌─────────────────┐                     │                     ┌──────────────────┐
  │ Linear Transf.  │──────────▶──────────┘                  ──▶│ Decompositions   │
  │ (§04)           │                                            │ (§08) QR alg.    │
  └─────────────────┘                                            └──────────────────┘

  AI CONNECTIONS:
  ┌──────────────────────────────────────────────────────────────────┐
  │  RoPE embeddings → Gram-Schmidt  → Muon optimizer  → OGD/GEM    │
  │  Orthogonal init → Spectral norm → Rayleigh quotient → SAM      │
  │  QR least squares → DFT (unitary) → Kernel Gram matrix → NTK    │
  └──────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════
```

---

[← Back to Advanced Linear Algebra](../README.md) | [Next: Matrix Norms →](../06-Matrix-Norms/notes.md)

---

## Appendix A: Key Inequalities in Inner Product Spaces

### A.1 Cauchy-Schwarz in Different Forms

The Cauchy-Schwarz inequality $|\langle\mathbf{u},\mathbf{v}\rangle| \leq \|\mathbf{u}\|\|\mathbf{v}\|$ has many equivalent forms:

**Discrete form:**
$$\left(\sum_{k=1}^n a_k b_k\right)^2 \leq \left(\sum_{k=1}^n a_k^2\right)\left(\sum_{k=1}^n b_k^2\right)$$

**Integral form:**
$$\left(\int_a^b f(x)g(x)\,dx\right)^2 \leq \left(\int_a^b f(x)^2\,dx\right)\left(\int_a^b g(x)^2\,dx\right)$$

**Probability form:** For random variables $X, Y$ with $\mathbb{E}[X^2], \mathbb{E}[Y^2] < \infty$:
$$|\mathbb{E}[XY]|^2 \leq \mathbb{E}[X^2]\mathbb{E}[Y^2]$$

The probability form directly implies: the correlation coefficient satisfies $|\text{Corr}(X,Y)| \leq 1$.

**Second proof (via AM-GM).** For any $t > 0$:
$$0 \leq \|t\mathbf{u} - \frac{1}{t}\mathbf{v}\|^2 = t^2\|\mathbf{u}\|^2 - 2\langle\mathbf{u},\mathbf{v}\rangle + \frac{1}{t^2}\|\mathbf{v}\|^2$$

Minimizing over $t > 0$ gives $t^2 = \|\mathbf{v}\|/\|\mathbf{u}\|$ and:
$$0 \leq 2\|\mathbf{u}\|\|\mathbf{v}\| - 2\langle\mathbf{u},\mathbf{v}\rangle \quad \Rightarrow \quad \langle\mathbf{u},\mathbf{v}\rangle \leq \|\mathbf{u}\|\|\mathbf{v}\|$$

Applying to $-\mathbf{v}$ gives the lower bound. $\square$

### A.2 Triangle Inequality and Its Uses

The **triangle inequality** for norms follows from Cauchy-Schwarz:
$$\|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + 2\langle\mathbf{u},\mathbf{v}\rangle + \|\mathbf{v}\|^2 \leq \|\mathbf{u}\|^2 + 2\|\mathbf{u}\|\|\mathbf{v}\| + \|\mathbf{v}\|^2 = (\|\mathbf{u}\|+\|\mathbf{v}\|)^2$$

**Parallelogram law:** For any inner product space:
$$\|\mathbf{u}+\mathbf{v}\|^2 + \|\mathbf{u}-\mathbf{v}\|^2 = 2(\|\mathbf{u}\|^2 + \|\mathbf{v}\|^2)$$

*Proof:* Expand both norms and add. The cross terms cancel. $\square$

This law **characterizes** inner product spaces: a norm satisfying the parallelogram law comes from an inner product via the **polarization identity**:
$$\langle\mathbf{u},\mathbf{v}\rangle = \frac{1}{4}\left(\|\mathbf{u}+\mathbf{v}\|^2 - \|\mathbf{u}-\mathbf{v}\|^2\right)$$

**Practical importance:** The $\ell^1$ and $\ell^\infty$ norms do NOT satisfy the parallelogram law, confirming they do not arise from inner products. This is why optimization in $\ell^1$ (LASSO) has different geometry from $\ell^2$ (ridge regression) — there is no meaningful notion of "orthogonality" in $\ell^1$.

### A.3 Weyl's Inequality for Eigenvalue Perturbation

When a symmetric matrix $A$ is perturbed to $A + E$ (with $E$ symmetric), Weyl's inequality bounds the eigenvalue shifts:

$$|\lambda_k(A+E) - \lambda_k(A)| \leq \|E\|_2 = \sigma_1(E)$$

This means: the eigenvalues of a symmetric matrix are **Lipschitz** functions of the matrix entries, with Lipschitz constant 1 in the spectral norm.

**Consequence for ML:** If the Hessian $\nabla^2\mathcal{L}(\theta)$ changes slowly along a training trajectory (small $\|H(t+1) - H(t)\|_2$), its eigenvalues also change slowly. This justifies quasi-Newton methods that assume the curvature is approximately constant over several steps.

---

## Appendix B: Numerical Orthogonalization Recipes

### B.1 When to Use Each Method

| Method | Complexity | Stability | When to Use |
|--------|-----------|----------|-------------|
| Classical Gram-Schmidt (CGS) | $O(mnk)$ | Poor: $O(\epsilon\kappa^2)$ | Never in production; educational use only |
| Modified Gram-Schmidt (MGS) | $O(mnk)$ | Good: $O(\epsilon\kappa)$ | When explicit $Q$ needed and $\kappa(A)$ moderate |
| Householder QR | $O(mn^2 - n^3/3)$ | Excellent: $O(\epsilon)$ | General QR factorization; recommended default |
| Givens QR | $O(mn^2)$ | Excellent | Sparse matrices; parallel processing |
| Randomized QR | $O(mn\log k + nk^2)$ | Good | Large matrices; approximate low-rank QR |

### B.2 Loss of Orthogonality: Quantifying the Damage

In IEEE double precision ($\epsilon_\text{mach} \approx 2.2 \times 10^{-16}$), the computed $\hat{Q}$ from CGS on an $n \times n$ matrix satisfies:

$$\|\hat{Q}^\top\hat{Q} - I\|_F \lesssim n \cdot \epsilon_\text{mach} \cdot \kappa(A)^2$$

For a well-conditioned matrix ($\kappa = 10$): error $\approx n \cdot 10^{-14}$ — acceptable.

For an ill-conditioned matrix ($\kappa = 10^8$): error $\approx n \cdot 1$ — completely non-orthogonal!

MGS improves this to $\lesssim n \cdot \epsilon_\text{mach} \cdot \kappa(A)$, and Householder to $\lesssim n \cdot \epsilon_\text{mach}$ regardless of $\kappa$.

### B.3 Reorthogonalization

When MGS produces inadequate orthogonality (common for nearly-dependent input vectors), apply **reorthogonalization**: after completing Gram-Schmidt once, apply it again to the already-computed $Q$. This reduces the error from $O(\epsilon\kappa)$ to $O(\epsilon^2\kappa^2)$, which is typically negligible.

```python
def mgs_with_reorth(A, num_reorth=1):
    """Modified Gram-Schmidt with optional reorthogonalization."""
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        for reorth in range(1 + num_reorth):
            for i in range(j):
                R[i, j] = Q[:, i] @ v  if reorth == 0 else Q[:, i] @ v
                v -= (Q[:, i] @ v) * Q[:, i]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R
```

### B.4 Block Gram-Schmidt for Parallel Computation

For very large matrices on parallel hardware, **block Gram-Schmidt** processes columns in blocks of size $b$:

1. Orthogonalize within each block using MGS or Householder
2. Orthogonalize each new block against all previous blocks using a BLAS-3 matrix operation

The key advantage: step 2 uses matrix-matrix products ($O(nb^2)$ flops, BLAS Level 3) rather than matrix-vector products ($O(nb)$ flops, BLAS Level 1/2). On modern hardware with SIMD and caching, this gives 10-100× speedup for large $n$.

---

## Appendix C: Orthogonality in Machine Learning Libraries

### C.1 PyTorch Orthogonal Utilities

```python
import torch
import torch.nn as nn

# Orthogonal initialization
linear = nn.Linear(64, 64)
nn.init.orthogonal_(linear.weight)  # U from QR decomposition

# Spectral normalization
conv = nn.utils.spectral_norm(nn.Conv2d(64, 64, 3))
# Divides weight by its spectral norm at each forward pass

# Orthogonal regularization
def orthogonal_reg(model, lambda_=1e-4):
    loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            W = param
            loss += lambda_ * torch.norm(W.T @ W - torch.eye(W.shape[1]))
    return loss
```

### C.2 NumPy/SciPy QR and Orthogonal Matrix Tools

```python
import numpy as np
from scipy.linalg import qr, qr_multiply, polar

# Full QR decomposition
Q, R = np.linalg.qr(A)              # 'reduced' (default) or 'complete'
Q, R, P = qr(A, pivoting=True)      # QR with column pivoting

# Random orthogonal matrix (Haar measure)
M = np.random.randn(n, n)
Q, R = np.linalg.qr(M)
Q *= np.sign(np.diag(R))            # Fix signs for uniform distribution

# Polar decomposition: A = Q @ S
Q, S = polar(A)                     # scipy.linalg.polar

# Check orthogonality
I_approx = Q.T @ Q
print(f"Max off-diagonal: {np.max(np.abs(I_approx - np.eye(n))):.2e}")
```

### C.3 Efficient Householder Representation

For large $n$, storing the full $Q$ matrix explicitly requires $O(n^2)$ space. The **compact WY representation** stores $Q = I - WY^\top$ where $W, Y \in \mathbb{R}^{n \times k}$, requiring only $O(nk)$ space and allowing matrix-vector products in $O(nk)$ time:

```python
def apply_householder_sequence(v_list, tau_list, x):
    """Apply Q = H_1 H_2 ... H_k to x using the compact representation."""
    for v, tau in zip(v_list, tau_list):
        x = x - tau * v * (v @ x)  # Apply H = I - tau * v * v^T
    return x
```

This is how LAPACK and NumPy store $Q$ internally — only the Householder vectors are saved, not the explicit matrix.


---

## Appendix D: The Geometry of Projection — Visual Summary

```
ORTHOGONAL DECOMPOSITION THEOREM
════════════════════════════════════════════════════════════════════════

  The Euclidean space ℝⁿ splits into complementary orthogonal subspaces:

  For any subspace S ⊆ ℝⁿ:

           ℝⁿ = S ⊕ S⊥

           ┌──────────────────────────────────────────┐
           │                  ℝⁿ                      │
           │         ┌────────────────┐               │
           │         │       S        │               │
           │         │    v_S = Pv   │               │
           │         │      •        │               │
           │         │    /         │               │
           │         │  /           │               │
           │  v•     │/  ← v_S      │               │
           │   \     └────────────────┘               │
           │    \          |  orthogonal               │
           │     \         |  complement               │
           │      \        |  S⊥                      │
           │       •────── + ───────────────────────── │
           │    v - v_S    |  (v - v_S ⊥ every s ∈ S) │
           └──────────────────────────────────────────┘

  Properties of projection matrix P = QQ⊤ (Q has ONB for S):

    Symmetric:    P⊤ = P         (orthogonal, not oblique)
    Idempotent:   P² = P         (projecting twice = projecting once)
    Range:        col(P) = S     (maps into S)
    Null space:   null(P) = S⊥  (annihilates complement)

════════════════════════════════════════════════════════════════════════
```

```
GRAM-SCHMIDT PROCESS — STEP BY STEP
════════════════════════════════════════════════════════════════════════

  Input: linearly independent vectors a₁, a₂, a₃ (not orthogonal)
  Output: orthonormal basis q₁, q₂, q₃ for same subspace

  STEP 1:  q₁ = a₁ / ‖a₁‖
           ────────────────────────────────────
           a₁  ──────────────────────────▶ q₁ (unit vector)

  STEP 2:  u₂ = a₂ - ⟨a₂,q₁⟩q₁   (subtract projection onto q₁)
           q₂ = u₂ / ‖u₂‖
           ────────────────────────────────────
           a₂  ──→  remove q₁ component  ──→ u₂ ──→ q₂

                     a₂
                    /
                   / ↑ ⟨a₂,q₁⟩q₁ (this part removed)
                  /
           q₁ ──────────── q₁ direction
                  u₂ (perpendicular remainder) ──▶ q₂

  STEP 3:  u₃ = a₃ - ⟨a₃,q₁⟩q₁ - ⟨a₃,q₂⟩q₂
           q₃ = u₃ / ‖u₃‖
           ────────────────────────────────────
           Remove projections onto BOTH q₁ and q₂

  RESULT: {q₁, q₂, q₃} is an ONB with same span as {a₁, a₂, a₃}

  QR FACTORIZATION: A = [a₁|a₂|a₃] = [q₁|q₂|q₃] · R
  where R is upper triangular with positive diagonal.

════════════════════════════════════════════════════════════════════════
```

```
SPECTRAL THEOREM: SYMMETRIC ↔ ORTHOGONAL EIGENBASIS
════════════════════════════════════════════════════════════════════════

  A = Aᵀ  (symmetric)
       ↕
  A has orthonormal eigenbasis {q₁, ..., qₙ}
       ↕
  A = QΛQᵀ  where  Q = [q₁|...|qₙ] ∈ O(n),  Λ = diag(λ₁,...,λₙ)
       ↕
  Quadratic form:  xᵀAx = Σᵢ λᵢ(xᵀqᵢ)² = weighted sum of squared projections

  INTERPRETATION:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Qᵀ: Express x in eigenbasis  →  Λ: Scale each component  →   │
  │  Q: Return to standard basis                                    │
  │                                                                  │
  │  λᵢ > 0: stretch in qᵢ direction  (positive definite: all λᵢ>0)│
  │  λᵢ = 0: collapse in qᵢ direction  (singular matrix)           │
  │  λᵢ < 0: reflection in qᵢ direction  (indefinite: mixed signs) │
  └─────────────────────────────────────────────────────────────────┘

  AI CONNECTION: Hessian ∇²ℒ is symmetric. Its eigenvectors are the
  "principal curvature directions" of the loss surface. Eigenvalues
  tell us how curved the loss is in each direction.

════════════════════════════════════════════════════════════════════════
```

```
ORTHOGONAL MATRICES: STRUCTURE AND CLASSIFICATION
════════════════════════════════════════════════════════════════════════

  QᵀQ = I  (definition)   ⟹   det(Q) = ±1

  det(Q) = +1:  ROTATIONS (SO(n))      det(Q) = -1:  REFLECTIONS
  ────────────────────────────         ──────────────────────────
  Preserve orientation                 Reverse orientation
  Examples in 2D: R_θ for all θ        Example: diag(1,...,1,-1)
  Compose to give rotations            Compose to give rotations

  O(n) = SO(n) ∪ {reflections}   (two connected components)

  EIGENVALUES: always on unit circle ∈ ℂ
  ┌─────────────────────────────────────────────────────────────────┐
  │   Real eigenvalues: ±1 only                                     │
  │   Complex eigenvalues: e^{iθ}, e^{-iθ} pairs → 2D rotations    │
  │                                                                  │
  │   So(2n) block structure:                                        │
  │   ┌──────────┬───────┐                                          │
  │   │ R_{θ₁}  │       │  each block is a 2×2 rotation           │
  │   ├──────────┤ R_{θ₂}│  with its own angle θᵢ                 │
  │   │          ├───────┤                                          │
  │   │          │  ...  │                                          │
  │   └──────────┴───────┘                                          │
  └─────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

---

## Appendix E: Summary of Key Formulas

### Inner Products and Orthogonality

| Formula | Expression | Notes |
|---------|-----------|-------|
| Inner product | $\langle\mathbf{u},\mathbf{v}\rangle = \mathbf{u}^\top\mathbf{v} = \sum_i u_i v_i$ | Euclidean; generalizes to $\langle\mathbf{u},\mathbf{v}\rangle_M = \mathbf{u}^\top M\mathbf{v}$ |
| Norm | $\|\mathbf{v}\| = \sqrt{\langle\mathbf{v},\mathbf{v}\rangle}$ | Induced by inner product |
| Angle | $\cos\theta = \langle\mathbf{u},\mathbf{v}\rangle / (\|\mathbf{u}\|\|\mathbf{v}\|)$ | Requires both nonzero |
| Cauchy-Schwarz | $|\langle\mathbf{u},\mathbf{v}\rangle| \leq \|\mathbf{u}\|\|\mathbf{v}\|$ | Equality iff parallel |
| Pythagorean theorem | $\mathbf{u} \perp \mathbf{v} \Rightarrow \|\mathbf{u}+\mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$ | Converse also holds |
| Parseval's identity | $\|\mathbf{v}\|^2 = \sum_{i=1}^n |\langle\mathbf{v},\mathbf{q}_i\rangle|^2$ | For complete ONB $\{\mathbf{q}_i\}$ |
| Bessel's inequality | $\sum_{i=1}^k |\langle\mathbf{v},\mathbf{q}_i\rangle|^2 \leq \|\mathbf{v}\|^2$ | For partial ONS |

### Projections

| Formula | Expression | Notes |
|---------|-----------|-------|
| Proj onto unit vector | $\operatorname{proj}_{\hat{\mathbf{u}}}(\mathbf{v}) = \langle\mathbf{v},\hat{\mathbf{u}}\rangle\hat{\mathbf{u}}$ | 1D case |
| Proj onto subspace (ONB $Q$) | $P_S\mathbf{v} = QQ^\top\mathbf{v}$ | Requires $Q^\top Q = I$ |
| Proj onto col$(A)$ (general) | $P_S\mathbf{v} = A(A^\top A)^{-1}A^\top\mathbf{v}$ | General basis |
| Projection matrix properties | $P^2 = P$, $P^\top = P$ | Idempotent + symmetric |
| Complementary projector | $P_{S^\perp} = I - P_S$ | Also a projector |
| Best approx property | $\|\mathbf{v} - P_S\mathbf{v}\| \leq \|\mathbf{v} - \mathbf{s}\|$ for all $\mathbf{s} \in S$ | Fundamental theorem |

### Gram-Schmidt and QR

| Formula | Expression | Notes |
|---------|-----------|-------|
| GS step | $\mathbf{u}_j = \mathbf{a}_j - \sum_{i<j}\langle\mathbf{a}_j,\mathbf{q}_i\rangle\mathbf{q}_i$ | Remove prior directions |
| QR factorization | $A = QR$ | $Q$ orthonormal cols, $R$ upper triangular |
| Least squares via QR | $R\mathbf{x} = Q^\top\mathbf{b}$ | Solve upper triangular system |
| Householder | $H = I - 2\mathbf{n}\mathbf{n}^\top$ | Reflection across $\mathbf{n}^\perp$ |
| Cayley transform | $Q = (I-S)(I+S)^{-1}$ | From skew-symmetric $S$ |

### Spectral Theorem

| Formula | Expression | Notes |
|---------|-----------|-------|
| Spectral decomposition | $A = Q\Lambda Q^\top = \sum_i \lambda_i\mathbf{q}_i\mathbf{q}_i^\top$ | Symmetric $A$ only |
| Rayleigh quotient | $R_A(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x}/\|\mathbf{x}\|^2$ | Bounded by $[\lambda_\min,\lambda_\max]$ |
| Positive definiteness | $A \succ 0 \Leftrightarrow$ all $\lambda_i > 0$ | Via spectral theorem |
| Polar decomposition | $A = QS$, $Q^\top Q = I$, $S \succeq 0$ | Extends to non-square |


---

## Appendix F: Connections to Other Chapters

### F.1 From Eigenvalues (§01) to Orthogonality

The spectral theorem shows that symmetric matrices are the matrices that are *diagonalizable by orthogonal transformations*. Non-symmetric matrices may still be diagonalizable, but with a non-orthogonal change of basis — making computations numerically less stable.

The **QR algorithm** for computing eigenvalues (§01) is built entirely on orthogonal matrix iterations:

```
A₀ = A
for k = 0, 1, 2, ...:
    A_k = Q_k R_k    (QR factorization of current iterate)
    A_{k+1} = R_k Q_k  (swap factors)
```

Each iterate is orthogonally similar to the previous: $A_{k+1} = Q_k^\top A_k Q_k$ (since $R_k = Q_k^\top A_k$ and $A_{k+1} = R_k Q_k = Q_k^\top A_k Q_k$). The iterates converge to the Schur form — upper triangular, with eigenvalues on the diagonal. The entire algorithm is numerically stable precisely because it uses only orthogonal transformations.

### F.2 From SVD (§02) to Orthogonality

The SVD $A = U\Sigma V^\top$ can be viewed as the "orthogonality structure" of $A$:

- $V = [\mathbf{v}_1|\cdots|\mathbf{v}_n]$: ONB for the row space, mapping inputs to "pure directions"
- $U = [\mathbf{u}_1|\cdots|\mathbf{u}_m]$: ONB for the column space, the natural output directions
- $\Sigma$: the non-orthogonal part (pure scaling)

The polar decomposition $A = (U V^\top)(V \Sigma V^\top)$ isolates the orthogonal factor $Q = UV^\top$ from the symmetric factor $S = V\Sigma V^\top$. This factorization requires all the tools developed in this section.

### F.3 Orthogonality Enables Low-Rank Approximation (§02, §03)

**Eckart-Young-Mirsky theorem:** The best rank-$k$ approximation to $A$ in both spectral and Frobenius norms is:
$$A_k = \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^\top$$

This result requires:
1. The SVD (which produces orthogonal $U, V$)
2. The best approximation theorem (projection gives the nearest element in a subspace)
3. Parseval's identity (to express $\|A - A_k\|_F^2 = \sum_{i>k}\sigma_i^2$)

All three are developed in this section. The Eckart-Young theorem is the mathematical foundation for PCA, LoRA, and any other low-rank approximation method.

### F.4 Orthogonality and Matrix Norms (§06)

**Unitarily invariant norms:** A matrix norm $\|A\|$ is **unitarily invariant** if $\|UAV\| = \|A\|$ for all unitary $U, V$. The three most important matrix norms are all unitarily invariant:

- **Spectral norm:** $\|A\|_2 = \sigma_1(A)$ (largest singular value)
- **Frobenius norm:** $\|A\|_F = \sqrt{\sum_{i,j}A_{ij}^2} = \sqrt{\sum_i\sigma_i^2}$ (sum of squared singular values)
- **Nuclear norm:** $\|A\|_* = \sum_i\sigma_i$ (sum of singular values)

These norms are "orthogonality-aware" — they don't change when the matrix is multiplied by orthogonal matrices. This property is what allows the Eckart-Young theorem to work: the error in low-rank approximation depends only on the discarded singular values, not on the choice of orthogonal bases.

### F.5 Orthogonality in Optimization (Chapter 08)

Modern optimization methods for training large neural networks increasingly exploit orthogonal structure:

**Sharpness-Aware Minimization (SAM):** The sharpness of a minimum is $\lambda_\text{max}(\nabla^2\mathcal{L})$. SAM perturbs parameters in the direction of the *eigenvector of the Hessian corresponding to $\lambda_\text{max}$* — computed approximately using power iteration. Power iteration produces a sequence of normalized vectors converging to the top eigenvector, each step involving a matrix-vector product with the Hessian and a re-normalization (orthogonalization against zero-dimensional subspace = normalization).

**SOAP optimizer (2024):** Maintains a factored approximation of the Adam preconditioner in which the "rotation" and "scaling" factors are tracked separately. The rotation factor is updated via QR decomposition to maintain orthogonality, while the scaling factor tracks the second moments in the rotating basis.

**K-FAC and its variants:** The Kronecker-Factored Approximate Curvature approximates the Fisher information matrix as a Kronecker product $F \approx A \otimes G$. Computing the update requires inverting these factors, which can be done via their eigendecompositions — the eigenvectors are orthonormal matrices.

---

## Appendix G: Historical Notes and References

### Timeline: Development of Orthogonality Theory

| Year | Mathematician | Contribution |
|------|---------------|-------------|
| 1801 | Gauss | Least squares method; orthogonal projection as normal equations |
| 1844 | Grassmann | Abstract vector spaces and subspaces; inner product generalization |
| 1850s | Cauchy, Schwarz | Cauchy-Schwarz inequality (Cauchy for sums 1821, Schwarz integral form 1888) |
| 1897 | Gram | Gram-Schmidt process (Gram's formulation for $L^2$ functions) |
| 1902 | Schmidt | Finite-dimensional Gram-Schmidt for $\ell^2$ |
| 1907 | Parseval | Parseval's theorem for Fourier series |
| 1908 | Hilbert | Abstract Hilbert spaces; orthogonal projection theorem |
| 1915–1927 | Weyl, von Neumann | Spectral theorem in infinite dimensions |
| 1954 | Householder | Householder reflectors for numerically stable QR |
| 1965 | Francis | QR algorithm for eigenvalues (with implicit shifts) |
| 1967 | Björck | Numerical analysis of Gram-Schmidt stability |
| 1983 | Saxe, McClelland, Ganguli | Orthogonal initialization for deep linear networks |
| 2014 | Mishkin, Matas | LSUV initialization using SVD/QR |
| 2018 | Miyato et al. | Spectral normalization for GANs |
| 2021 | Su et al. | RoPE: Rotary Position Embedding using orthogonal rotations |
| 2024 | Jordan et al. | Muon optimizer: gradient orthogonalization via Newton-Schulz |

### Key References

1. **Golub & Van Loan**, *Matrix Computations* (4th ed., 2013) — Definitive reference on numerical linear algebra including Gram-Schmidt, Householder, and QR algorithms
2. **Axler**, *Linear Algebra Done Right* (3rd ed., 2015) — Elegant proof-based treatment of inner product spaces and spectral theorem
3. **Trefethen & Bau**, *Numerical Linear Algebra* (1997) — Excellent coverage of QR, projections, and stability from computational perspective
4. **Saxe et al.** (2013), "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks" — Theoretical foundation for orthogonal initialization
5. **Miyato et al.** (2018), "Spectral Normalization for GANs" — Seminal paper on spectral normalization in deep learning
6. **Su et al.** (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding" — RoPE using orthogonal rotation matrices
7. **Jordan et al.** (2024), "Muon: An optimizer for hidden layers in neural networks" — Newton-Schulz orthogonalization for gradients


---

## Appendix H: Proofs of Selected Theorems

### H.1 Complete Proof: Cauchy-Schwarz Inequality

**Theorem.** For any inner product space and vectors $\mathbf{u}, \mathbf{v}$:
$$|\langle\mathbf{u},\mathbf{v}\rangle|^2 \leq \langle\mathbf{u},\mathbf{u}\rangle\langle\mathbf{v},\mathbf{v}\rangle$$

*Proof.* If $\mathbf{v} = \mathbf{0}$, both sides are 0 and the inequality holds trivially. Assume $\mathbf{v} \neq \mathbf{0}$.

For any scalar $t \in \mathbb{R}$:
$$0 \leq \langle\mathbf{u} - t\mathbf{v},\mathbf{u} - t\mathbf{v}\rangle = \langle\mathbf{u},\mathbf{u}\rangle - 2t\langle\mathbf{u},\mathbf{v}\rangle + t^2\langle\mathbf{v},\mathbf{v}\rangle$$

This is a quadratic in $t$ that is always $\geq 0$. Its discriminant must therefore be $\leq 0$:
$$\Delta = 4\langle\mathbf{u},\mathbf{v}\rangle^2 - 4\langle\mathbf{u},\mathbf{u}\rangle\langle\mathbf{v},\mathbf{v}\rangle \leq 0$$
$$\langle\mathbf{u},\mathbf{v}\rangle^2 \leq \langle\mathbf{u},\mathbf{u}\rangle\langle\mathbf{v},\mathbf{v}\rangle \quad \square$$

**Equality condition.** Equality holds iff $\Delta = 0$ iff the quadratic in $t$ has a zero iff there exists $t_0$ with $\mathbf{u} - t_0\mathbf{v} = \mathbf{0}$ iff $\mathbf{u}$ and $\mathbf{v}$ are linearly dependent (one is a scalar multiple of the other).

**Remark.** This proof works for any abstract inner product space, finite or infinite dimensional. The key was that $\langle\cdot,\cdot\rangle$ is positive definite.

### H.2 Complete Proof: Best Approximation Uniqueness

**Theorem.** Let $S$ be a closed subspace of a Hilbert space $\mathcal{H}$. For any $\mathbf{v} \in \mathcal{H}$, the projection $\mathbf{v}_S$ satisfying $\mathbf{v} - \mathbf{v}_S \perp S$ is unique.

*Proof of uniqueness.* Suppose both $\mathbf{w}_1, \mathbf{w}_2 \in S$ satisfy $\mathbf{v} - \mathbf{w}_i \perp S$. Then $(\mathbf{v} - \mathbf{w}_1) - (\mathbf{v} - \mathbf{w}_2) = \mathbf{w}_2 - \mathbf{w}_1 \in S$ (since $S$ is a subspace). But also $\mathbf{w}_2 - \mathbf{w}_1 \perp S$ (as a difference of vectors each perpendicular to $S$). So $\mathbf{w}_2 - \mathbf{w}_1 \in S \cap S^\perp = \{\mathbf{0}\}$, giving $\mathbf{w}_1 = \mathbf{w}_2$. $\square$

### H.3 Complete Proof: Gram-Schmidt Produces ONB

**Theorem.** If $\mathbf{a}_1, \ldots, \mathbf{a}_k$ are linearly independent vectors in an inner product space $V$, the Gram-Schmidt process produces an orthonormal set $\mathbf{q}_1, \ldots, \mathbf{q}_k$ with $\operatorname{span}(\mathbf{q}_1,\ldots,\mathbf{q}_j) = \operatorname{span}(\mathbf{a}_1,\ldots,\mathbf{a}_j)$ for each $j = 1,\ldots,k$.

*Proof by induction on $j$.*

**Base case ($j=1$):** $\mathbf{q}_1 = \mathbf{a}_1/\|\mathbf{a}_1\|$ is a unit vector (since $\mathbf{a}_1 \neq \mathbf{0}$ by independence). $\operatorname{span}(\mathbf{q}_1) = \operatorname{span}(\mathbf{a}_1)$. ✓

**Inductive step:** Assume the claim holds for $j-1$. Define $S_{j-1} = \operatorname{span}(\mathbf{a}_1,\ldots,\mathbf{a}_{j-1}) = \operatorname{span}(\mathbf{q}_1,\ldots,\mathbf{q}_{j-1})$.

**(i) $\mathbf{u}_j \neq \mathbf{0}$:** Suppose for contradiction $\mathbf{u}_j = \mathbf{0}$. Then $\mathbf{a}_j = \sum_{i<j}\langle\mathbf{a}_j,\mathbf{q}_i\rangle\mathbf{q}_i \in S_{j-1} = \operatorname{span}(\mathbf{a}_1,\ldots,\mathbf{a}_{j-1})$, contradicting linear independence.

**(ii) $\mathbf{q}_j \perp \mathbf{q}_\ell$ for $\ell < j$:** For any $\ell < j$:
$$\langle\mathbf{u}_j, \mathbf{q}_\ell\rangle = \langle\mathbf{a}_j, \mathbf{q}_\ell\rangle - \sum_{i<j}\langle\mathbf{a}_j,\mathbf{q}_i\rangle\langle\mathbf{q}_i,\mathbf{q}_\ell\rangle = \langle\mathbf{a}_j,\mathbf{q}_\ell\rangle - \langle\mathbf{a}_j,\mathbf{q}_\ell\rangle\cdot 1 = 0$$

(using the inductive hypothesis $\langle\mathbf{q}_i,\mathbf{q}_\ell\rangle = \delta_{i\ell}$). Since $\mathbf{q}_j = \mathbf{u}_j/\|\mathbf{u}_j\|$, we also have $\langle\mathbf{q}_j,\mathbf{q}_\ell\rangle = 0$. ✓

**(iii) Same span:** $\mathbf{q}_j = \mathbf{u}_j/\|\mathbf{u}_j\|$ is in $\operatorname{span}(\mathbf{a}_j, \mathbf{q}_1,\ldots,\mathbf{q}_{j-1}) = \operatorname{span}(\mathbf{a}_1,\ldots,\mathbf{a}_j)$. Conversely, $\mathbf{a}_j = \mathbf{u}_j + \sum_{i<j}\langle\mathbf{a}_j,\mathbf{q}_i\rangle\mathbf{q}_i = \|\mathbf{u}_j\|\mathbf{q}_j + \sum_{i<j}\langle\mathbf{a}_j,\mathbf{q}_i\rangle\mathbf{q}_i \in \operatorname{span}(\mathbf{q}_1,\ldots,\mathbf{q}_j)$. ✓

By induction, the claim holds for all $j \leq k$. $\square$

---

*This section is part of the [03-Advanced-Linear-Algebra](../README.md) chapter.*


---

## Appendix I: The Stiefel Manifold and Optimization on Orthogonal Groups

### I.1 The Stiefel Manifold $\text{St}(n, k)$

The set of $n \times k$ matrices with orthonormal columns is the **Stiefel manifold**:
$$\text{St}(n, k) = \{Q \in \mathbb{R}^{n \times k} : Q^\top Q = I_k\}$$

Special cases:
- $\text{St}(n, n) = O(n)$ — the full orthogonal group
- $\text{St}(n, 1) = S^{n-1}$ — the unit sphere in $\mathbb{R}^n$
- $\text{St}(n, k)$ for $1 < k < n$ — Stiefel manifolds proper

**Dimension:** $\dim(\text{St}(n,k)) = nk - k(k+1)/2$

*Intuition:* We have $nk$ free parameters (entries of a $n \times k$ matrix), subject to $k(k+1)/2$ constraints ($k$ normalization constraints + $k(k-1)/2$ orthogonality constraints between distinct columns).

**Tangent space at $Q \in \text{St}(n,k)$:** The tangent space consists of matrices $\dot{Q}$ satisfying $Q^\top\dot{Q} + \dot{Q}^\top Q = 0$, i.e., $Q^\top\dot{Q}$ is skew-symmetric:
$$T_Q\text{St}(n,k) = \{\dot{Q} \in \mathbb{R}^{n \times k} : Q^\top\dot{Q} + \dot{Q}^\top Q = 0\}$$

### I.2 Riemannian Gradient Descent on the Stiefel Manifold

To minimize $f(Q)$ subject to $Q \in \text{St}(n,k)$:

**Step 1: Euclidean gradient.** Compute $\nabla_Q f$ in $\mathbb{R}^{n \times k}$.

**Step 2: Project to tangent space.** The Riemannian gradient is the projection of $\nabla_Q f$ onto $T_Q\text{St}(n,k)$:
$$\text{grad}_f(Q) = \nabla_Q f - Q\,\text{sym}(Q^\top\nabla_Q f)$$

where $\text{sym}(M) = (M + M^\top)/2$ is the symmetric part.

**Step 3: Update and retract.** Move in the direction of the (negative) Riemannian gradient, then retract back to the manifold:

*Retraction via QR:* $Q_{\text{new}} = \text{qr}(Q - \eta\,\text{grad}_f(Q))$ (take the $Q$ factor from QR decomposition)

*Retraction via Cayley:* $Q_{\text{new}} = (I + \eta W/2)^{-1}(I - \eta W/2)Q$ for a suitable skew-symmetric $W$

### I.3 Applications in Modern AI

**Orthogonal RNNs (Henaff et al. 2016, Arjovsky et al. 2016).** Vanilla RNNs suffer from vanishing/exploding gradients because the recurrence matrix $W$ can have singular values $\neq 1$. **Unitary/Orthogonal RNNs** constrain $W \in O(n)$, ensuring stable gradient flow. Training optimizes over $O(n)$ using Stiefel manifold optimization or parameterization via Cayley transform.

**Rotation-Equivariant Networks.** Networks for 3D point cloud processing (SE(3)-equivariant networks, e3nn) explicitly parameterize their weight tensors using representations of $SO(3)$. The convolution filters transform predictably under 3D rotations of the input, making the network geometry-aware.

**LoRA with Orthogonal Factors.** Standard LoRA approximates a weight update as $\Delta W = BA$ for low-rank matrices $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$. **GLoRA, DoRA, and OLoRA** variants constrain $B$ or $A$ to have orthonormal columns (lying on the Stiefel manifold), improving gradient flow during fine-tuning and reducing the rank collapse problem.

### I.4 The Polar Decomposition as Nearest Orthogonal Matrix

**Problem:** Given $A \in \mathbb{R}^{n \times n}$ (not necessarily orthogonal), find the nearest orthogonal matrix:
$$Q^* = \arg\min_{Q \in O(n)} \|A - Q\|_F$$

**Solution:** The polar decomposition $A = QS$ gives $Q^* = UV^\top$ where $A = U\Sigma V^\top$ (SVD).

*Proof:* For any $P \in O(n)$:
$$\|A - P\|_F^2 = \|U\Sigma V^\top - P\|_F^2 = \|\Sigma - U^\top P V\|_F^2$$

(using unitary invariance of Frobenius norm). Let $R = U^\top P V \in O(n)$. Then:
$$\|\Sigma - R\|_F^2 = \|\Sigma\|_F^2 - 2\langle\Sigma, R\rangle_F + \|R\|_F^2 = \sum_i\sigma_i^2 - 2\text{tr}(\Sigma R) + n$$

This is minimized when $\text{tr}(\Sigma R)$ is maximized. Since $\Sigma$ is diagonal with nonneg entries and $R$ is orthogonal, the Hadamard-type inequality gives $\text{tr}(\Sigma R) \leq \sum_i\sigma_i$, with equality at $R = I$ (i.e., $P = UV^\top$). $\square$

**Muon optimizer interpretation:** The Muon optimizer applies the update $W \leftarrow W - \eta Q_G$ where $Q_G = UV^\top$ is the nearest orthogonal matrix to the gradient $G$. This can be computed efficiently via Newton-Schulz iterations:
$$X_0 = G/\|G\|, \quad X_{k+1} = \frac{3}{2}X_k - \frac{1}{2}X_k X_k^\top X_k$$

which converges quadratically to the orthogonal factor of the polar decomposition of $G$.

---

## Appendix J: Quick Reference — Orthogonality Vocabulary

| Term | Definition | Key Formula |
|------|-----------|-------------|
| Orthogonal vectors | $\langle\mathbf{u},\mathbf{v}\rangle = 0$ | $\mathbf{u}^\top\mathbf{v} = 0$ |
| Orthonormal vectors | Orthogonal + unit length | $\mathbf{q}_i^\top\mathbf{q}_j = \delta_{ij}$ |
| Orthonormal basis (ONB) | ONS that spans the space | $\mathbf{v} = \sum_i\langle\mathbf{v},\mathbf{q}_i\rangle\mathbf{q}_i$ |
| Orthogonal matrix | Square, orthonormal columns | $Q^\top Q = QQ^\top = I$ |
| Orthogonal group | All $n \times n$ orthogonal matrices | $O(n)$, closed under $\cdot$, $\top$, $\cdot^{-1}$ |
| Special orthogonal group | Rotations (det = +1) | $SO(n) \subset O(n)$ |
| Orthogonal projection | $\mathbf{v}_S = $ closest point in $S$ | $P_S = QQ^\top$, $P^2=P$, $P^\top=P$ |
| Orthogonal complement | Vectors $\perp$ all of $S$ | $S^\perp = \{\mathbf{v}: \langle\mathbf{v},\mathbf{s}\rangle=0\,\forall\mathbf{s}\}$ |
| Gram-Schmidt | Orthogonalization algorithm | Build ONB from any basis |
| Householder reflector | Reflection across a hyperplane | $H = I - 2\mathbf{n}\mathbf{n}^\top$ |
| QR decomposition | $A = QR$ factorization | $Q$: orthonormal cols, $R$: upper triangular |
| Rayleigh quotient | Quotient for eigenvalue bounds | $R_A(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x}/\|\mathbf{x}\|^2$ |
| Spectral theorem | Symmetric $A$ has ONB of eigenvectors | $A = Q\Lambda Q^\top$ |
| Isometry | Norm-preserving linear map | $\|Q\mathbf{x}\| = \|\mathbf{x}\|$ |
| Stiefel manifold | Matrices with orthonormal columns | $\text{St}(n,k) = \{Q: Q^\top Q = I_k\}$ |
| Polar decomposition | Orthogonal × symmetric factorization | $A = QS$ |
| Unitary matrix | Complex analog of orthogonal | $U^*U = I$, entries $\in \mathbb{C}$ |


---

## Appendix K: Common Numerical Experiments

### K.1 Verifying Gram-Schmidt Stability

The following experiment quantifies the loss of orthogonality in CGS vs MGS:

```python
import numpy as np

def cgs(A):
    m, n = A.shape
    Q = np.zeros_like(A)
    R = np.zeros((n, n))
    for j in range(n):
        v = A[:, j].copy()
        R[:j, j] = Q[:, :j].T @ v
        v -= Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R

def mgs(A):
    m, n = A.shape
    Q = A.astype(float).copy()
    R = np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] /= R[j, j]
        R[j, j+1:] = Q[:, j] @ Q[:, j+1:]
        Q[:, j+1:] -= np.outer(Q[:, j], R[j, j+1:])
    return Q, R

# Test on Hilbert matrix (ill-conditioned)
n = 12
H = 1.0 / (np.arange(1, n+1)[:, None] + np.arange(1, n+1)[None, :] - 1)

Q_cgs, _ = cgs(H)
Q_mgs, _ = mgs(H)

err_cgs = np.max(np.abs(Q_cgs.T @ Q_cgs - np.eye(n)))
err_mgs = np.max(np.abs(Q_mgs.T @ Q_mgs - np.eye(n)))

print(f"CGS orthogonality error: {err_cgs:.2e}")   # ~1e-4 (catastrophic)
print(f"MGS orthogonality error: {err_mgs:.2e}")   # ~1e-8 (acceptable)
print(f"kappa(H): {np.linalg.cond(H):.2e}")        # ~10^16 for n=12
```

### K.2 Orthogonal Initialization Impact on Gradient Norms

```python
import numpy as np

def grad_norm_experiment(n=64, L=20, trials=100):
    """Compare gradient norms: Gaussian init vs Orthogonal init."""
    results = {'gaussian': [], 'orthogonal': []}
    
    for _ in range(trials):
        # Gaussian initialization
        Ws_gauss = [np.random.randn(n, n) / np.sqrt(n) for _ in range(L)]
        prod_gauss = Ws_gauss[0]
        for W in Ws_gauss[1:]:
            prod_gauss = prod_gauss @ W
        results['gaussian'].append(np.linalg.norm(prod_gauss, 2))
        
        # Orthogonal initialization
        Ws_orth = []
        for _ in range(L):
            M = np.random.randn(n, n)
            Q, _ = np.linalg.qr(M)
            Ws_orth.append(Q)
        prod_orth = Ws_orth[0]
        for W in Ws_orth[1:]:
            prod_orth = prod_orth @ W
        results['orthogonal'].append(np.linalg.norm(prod_orth, 2))
    
    print(f"Gaussian: mean={np.mean(results['gaussian']):.3e}, "
          f"std={np.std(results['gaussian']):.3e}")
    print(f"Orthogonal: mean={np.mean(results['orthogonal']):.3e}, "
          f"std={np.std(results['orthogonal']):.3e}")
    # Expected: Gaussian shows extreme variance (very large or very small)
    #           Orthogonal shows norm ≈ 1.0 with tiny variance

grad_norm_experiment()
```

### K.3 Rayleigh Quotient Iteration Convergence

```python
import numpy as np

def rayleigh_quotient_iteration(A, x0, n_iter=20):
    """Track convergence of Rayleigh Quotient Iteration."""
    x = x0 / np.linalg.norm(x0)
    lam_true = np.sort(np.linalg.eigvalsh(A))[-1]  # largest eigenvalue
    
    history = []
    for _ in range(n_iter):
        rho = x @ A @ x        # Rayleigh quotient
        history.append(abs(rho - lam_true))
        try:
            y = np.linalg.solve(A - rho * np.eye(len(A)), x)
            x = y / np.linalg.norm(y)
        except np.linalg.LinAlgError:
            break  # Converged (A - rho*I is singular)
    return history

A = np.array([[3., 1., 0.], [1., 2., 1.], [0., 1., 1.]])
x0 = np.random.randn(3)
errors = rayleigh_quotient_iteration(A, x0)
print("Errors per iteration:")
for i, e in enumerate(errors[:10]):
    print(f"  iter {i}: {e:.2e}")
# Expected: errors should decrease cubically (each step ~= previous^3)
```

