[← Back to Advanced Linear Algebra](../README.md) | [← Matrix Norms](../06-Matrix-Norms/notes.md) | [Next: Matrix Decompositions →](../08-Matrix-Decompositions/notes.md)

---

# Positive Definite Matrices

> _"Positive definiteness is the matrix condition that makes everything work: optimization has a unique minimum, Gaussians are proper distributions, and Cholesky gives you a square root. It is the difference between a bowl and a saddle."_

## Overview

Positive definite matrices are the matrices that behave like positive numbers. Just as a positive scalar $c > 0$ makes the equation $cx = b$ uniquely solvable and the function $cx^2$ have a unique minimum, a positive definite matrix $A \succ 0$ makes the system $A\mathbf{x} = \mathbf{b}$ uniquely solvable, the quadratic form $\mathbf{x}^\top A \mathbf{x}$ have a unique minimum at zero, and the decomposition $A = LL^\top$ (Cholesky) exist and be unique. Positive definiteness is the structural condition that separates "well-posed" from "ill-posed" in a precise algebraic sense.

This section develops the full theory of positive definite and positive semidefinite matrices. We begin with quadratic forms and the geometric picture — level sets that form ellipsoids, energy functions with unique bowls — then build the complete toolkit: four equivalent characterizations (eigenvalue, determinantal, pivot, Cholesky), the Cholesky and LDLᵀ decompositions as the canonical section topic, matrix square roots, the Schur complement and its role in Gaussian conditioning, log-determinants, Gram matrices, and the PSD cone with semidefinite programming.

The AI connections are dense and load-bearing. Every covariance matrix in a multivariate Gaussian must be positive semidefinite — this is not a convenience but a mathematical necessity. The Fisher information matrix is PSD by construction and drives natural gradient methods (K-FAC). The Hessian at a local minimum is PSD. Cholesky factorization is the standard algorithm for sampling from multivariate Gaussians and the reparameterization trick in VAEs. Log-determinants appear in every Gaussian log-likelihood and normalizing flow objective. The attention mechanism computes a scaled Gram matrix. Understanding positive definiteness is understanding the algebraic backbone of probabilistic machine learning.

## Prerequisites

- Eigenvalues, eigenvectors, spectral theorem for symmetric matrices — [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md)
- Singular value decomposition — [§02: SVD](../02-Singular-Value-Decomposition/notes.md)
- Inner products, orthogonality, orthogonal matrices — [§05: Orthogonality](../05-Orthogonality-and-Orthonormality/notes.md)
- Matrix norms, condition numbers — [§06: Matrix Norms](../06-Matrix-Norms/notes.md)
- Determinants — [Chapter 2 §04: Determinants](../../02-Linear-Algebra-Basics/04-Determinants/notes.md)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive exploration: Cholesky algorithms, log-det computation, Gram matrices, GPR, VAE reparameterization, SDP |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from PD verification through Cholesky-based sampling for VAEs |

## Learning Objectives

After completing this section, you will be able to:

- State all four equivalent characterizations of positive definiteness and prove the equivalences
- Distinguish PD ($A \succ 0$), PSD ($A \succeq 0$), negative definite, and indefinite with examples
- Apply Sylvester's criterion to certify positive definiteness without computing eigenvalues
- Implement the Cholesky algorithm from scratch and analyze its numerical stability
- Factor $A = LDL^\top$ and explain when to prefer LDLᵀ over standard Cholesky
- Compute the matrix square root $A^{1/2}$ and explain its uniqueness
- Define the Schur complement and use it to test positive definiteness of block matrices
- Apply the matrix inversion lemma via Schur complements
- Compute $\log\det A$ via Cholesky and derive the gradient $\nabla_A \log\det A = A^{-1}$
- Construct Gram matrices and prove they are PSD; state Mercer's theorem
- Describe the PSD cone $\mathbb{S}_+^n$ as a convex set and formulate SDPs
- Apply Cholesky reparameterization to sample from multivariate Gaussians in VAEs
- Connect the Fisher information matrix, natural gradient, and K-FAC to PD structure

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Positive Definiteness Captures](#11-what-positive-definiteness-captures)
  - [1.2 Geometric Picture: Ellipsoids and Level Sets](#12-geometric-picture-ellipsoids-and-level-sets)
  - [1.3 Why PD Matrices Matter for AI](#13-why-pd-matrices-matter-for-ai)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Quadratic Forms](#21-quadratic-forms)
  - [2.2 The Four Cases: PD, PSD, ND, Indefinite](#22-the-four-cases-pd-psd-nd-indefinite)
  - [2.3 Immediate Consequences](#23-immediate-consequences)
  - [2.4 The Loewner Order](#24-the-loewner-order)
- [3. Eigenvalue and Determinantal Characterizations](#3-eigenvalue-and-determinantal-characterizations)
  - [3.1 Spectral Characterization](#31-spectral-characterization)
  - [3.2 Sylvester's Criterion](#32-sylvesters-criterion)
  - [3.3 Pivot Characterization](#33-pivot-characterization)
  - [3.4 Certifying PSD vs PD](#34-certifying-psd-vs-pd)
- [4. Cholesky Decomposition](#4-cholesky-decomposition)
  - [4.1 The Theorem: Existence and Uniqueness](#41-the-theorem-existence-and-uniqueness)
  - [4.2 The Algorithm: Constructing L](#42-the-algorithm-constructing-l)
  - [4.3 Complexity and Numerical Properties](#43-complexity-and-numerical-properties)
  - [4.4 LDLᵀ Factorization](#44-ldlt-factorization)
  - [4.5 Modified Cholesky for Near-PD Matrices](#45-modified-cholesky-for-near-pd-matrices)
- [5. Matrix Square Root and Whitening](#5-matrix-square-root-and-whitening)
  - [5.1 The PSD Square Root](#51-the-psd-square-root)
  - [5.2 Computing Square Roots](#52-computing-square-roots)
  - [5.3 Square Root vs Cholesky](#53-square-root-vs-cholesky)
  - [5.4 Whitening Transforms](#54-whitening-transforms)
- [6. Schur Complement](#6-schur-complement)
  - [6.1 Definition for Block Matrices](#61-definition-for-block-matrices)
  - [6.2 Schur Complement and Positive Definiteness](#62-schur-complement-and-positive-definiteness)
  - [6.3 Matrix Inversion Lemma](#63-matrix-inversion-lemma)
  - [6.4 Gaussian Conditioning via Schur Complement](#64-gaussian-conditioning-via-schur-complement)
- [7. Log-Determinant](#7-log-determinant)
  - [7.1 Definition and Properties](#71-definition-and-properties)
  - [7.2 Log-Det as a Concave Function](#72-log-det-as-a-concave-function)
  - [7.3 Gradient and Matrix Calculus](#73-gradient-and-matrix-calculus)
  - [7.4 Log-Det in Machine Learning](#74-log-det-in-machine-learning)
- [8. Gram Matrices and Kernel Connections](#8-gram-matrices-and-kernel-connections)
  - [8.1 The Gram Matrix Construction](#81-the-gram-matrix-construction)
  - [8.2 Every PSD Matrix is a Gram Matrix](#82-every-psd-matrix-is-a-gram-matrix)
  - [8.3 Kernel Matrices and Mercer's Theorem](#83-kernel-matrices-and-mercers-theorem)
  - [8.4 Attention Scores as Gram Matrices](#84-attention-scores-as-gram-matrices)
- [9. The PSD Cone and Semidefinite Programming](#9-the-psd-cone-and-semidefinite-programming)
  - [9.1 The Cone of PSD Matrices](#91-the-cone-of-psd-matrices)
  - [9.2 Semidefinite Programming](#92-semidefinite-programming)
  - [9.3 SDP in Machine Learning](#93-sdp-in-machine-learning)
- [10. Applications in Machine Learning](#10-applications-in-machine-learning)
  - [10.1 Multivariate Gaussians and Covariance Matrices](#101-multivariate-gaussians-and-covariance-matrices)
  - [10.2 Fisher Information Matrix and Natural Gradient](#102-fisher-information-matrix-and-natural-gradient)
  - [10.3 Gaussian Process Regression](#103-gaussian-process-regression)
  - [10.4 Hessian and Loss Landscape Sharpness](#104-hessian-and-loss-landscape-sharpness)
  - [10.5 Reparameterization Trick in VAEs](#105-reparameterization-trick-in-vaes)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Why This Matters for AI (2026 Perspective)](#13-why-this-matters-for-ai-2026-perspective)
- [14. Conceptual Bridge](#14-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Positive Definiteness Captures

The most useful single-sentence definition of a positive definite matrix is: **a matrix that behaves like a positive number**. To understand what this means, compare scalars and matrices in parallel.

For a scalar $a > 0$:
- The equation $ax = b$ has a unique solution $x = b/a$
- The function $f(x) = ax^2$ has a unique minimum at $x = 0$
- The scalar has a real square root: $\sqrt{a} > 0$
- The product $ax \cdot x = ax^2 \geq 0$, with equality only at $x = 0$

For a positive definite matrix $A \succ 0$:
- The system $A\mathbf{x} = \mathbf{b}$ has a unique solution $\mathbf{x} = A^{-1}\mathbf{b}$
- The function $f(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ has a unique minimum at $\mathbf{x} = \mathbf{0}$
- The matrix has a unique positive definite square root $A^{1/2} \succ 0$
- The quadratic form $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$

The quadratic form $Q(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x}$ is the central object. Think of it as a machine that takes a vector $\mathbf{x}$ and returns a scalar measuring its "energy" with respect to $A$. When $A$ is the identity, $Q(\mathbf{x}) = \|\mathbf{x}\|^2$, the familiar squared Euclidean length. When $A$ is diagonal with positive entries $d_1, \ldots, d_n$, then $Q(\mathbf{x}) = d_1 x_1^2 + \cdots + d_n x_n^2$ — a weighted sum of squared coordinates. Positive definiteness requires this energy to be positive for every nonzero direction.

The formal definition requires $A$ to be symmetric. The condition $\mathbf{x}^\top A \mathbf{x} > 0$ for $\mathbf{x} \neq \mathbf{0}$ is called **strict positive definiteness**. The relaxed condition $\mathbf{x}^\top A \mathbf{x} \geq 0$ is **positive semidefiniteness** — it allows zero energy in some directions (those in the null space of $A$).

**For AI:** Every loss function $\mathcal{L}(\boldsymbol{\theta})$ that has a strict local minimum at $\boldsymbol{\theta}^*$ satisfies $\nabla^2 \mathcal{L}(\boldsymbol{\theta}^*) \succ 0$. The Hessian being positive definite at a critical point is exactly the second-order sufficient condition for a local minimum. This is the mathematical content of "the loss landscape is a bowl."

### 1.2 Geometric Picture: Ellipsoids and Level Sets

The level sets of the quadratic form $\mathbf{x}^\top A \mathbf{x} = c$ reveal the geometry of $A$ directly.

**Case 1: $A = I$ (identity).** The level set $\mathbf{x}^\top I \mathbf{x} = c$ is $\|\mathbf{x}\|^2 = c$ — a sphere of radius $\sqrt{c}$.

**Case 2: $A$ diagonal, $A = \text{diag}(\lambda_1, \lambda_2)$ with $\lambda_1 > \lambda_2 > 0$.** The level set $\lambda_1 x_1^2 + \lambda_2 x_2^2 = 1$ is an ellipse with semi-axes $1/\sqrt{\lambda_1}$ and $1/\sqrt{\lambda_2}$. The larger eigenvalue compresses that direction; the smaller eigenvalue stretches it.

**Case 3: $A$ general symmetric PD.** The level set $\mathbf{x}^\top A \mathbf{x} = 1$ is an ellipsoid whose axes are aligned with the eigenvectors of $A$ and whose semi-axis lengths are $1/\sqrt{\lambda_i}$. This follows from the spectral decomposition $A = Q\Lambda Q^\top$: substituting $\mathbf{y} = Q^\top \mathbf{x}$ gives $\mathbf{y}^\top \Lambda \mathbf{y} = 1$, a coordinate-aligned ellipsoid.

**Case 4: $A$ indefinite (some positive, some negative eigenvalues).** The level set is a hyperboloid — unbounded in the negative-eigenvalue directions. There is no minimum energy.

**Case 5: $A$ positive semidefinite (some zero eigenvalues).** The level set is an "infinite cylinder" — flat in the null space directions. Energy is zero for all vectors in the null space.

```
QUADRATIC FORM GEOMETRY
════════════════════════════════════════════════════════════════════════

  2D quadratic form  Q(x₁,x₂) = x⊤Ax,  level set Q = 1

  PD (both λ > 0):     PSD (one λ = 0):    Indefinite:
                        
     ┌───────┐           ─────────          ╱       ╲
     │  ╱─╲  │           ─────────          ─────────
     │ ╱   ╲ │           ─────────          ╲       ╱
     └───────┘           ─────────          ─────────
     ellipse             parallel lines     hyperbola
     
  Semi-axes ∝ 1/√λᵢ
  Axes aligned with eigenvectors of A

════════════════════════════════════════════════════════════════════════
```

This geometric picture has a direct machine learning reading. The covariance matrix $\Sigma$ of a multivariate Gaussian defines a metric on feature space: $(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$ is the squared Mahalanobis distance, measuring how many standard deviations $\mathbf{x}$ is from $\boldsymbol{\mu}$ in each principal direction. The level sets of this distance are the ellipsoidal contours of the Gaussian density.

### 1.3 Why PD Matrices Matter for AI

Positive definiteness is not a technical curiosity — it is a pervasive structural condition in machine learning systems:

**Covariance matrices.** Any valid probability distribution must have a non-negative variance. For a multivariate random variable $\mathbf{x} \in \mathbb{R}^n$, the covariance matrix $\Sigma = \mathbb{E}[(\mathbf{x}-\boldsymbol{\mu})(\mathbf{x}-\boldsymbol{\mu})^\top]$ is always PSD. For a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with a proper density, we need $\Sigma \succ 0$ (so $\det \Sigma \neq 0$ and the density is normalized). Every time a VAE encoder outputs a covariance or diagonal variance, it must produce a PSD (or PD) parameterization.

**Fisher information matrix.** The Fisher information $F(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \sim p(\cdot|\boldsymbol{\theta})}[\nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\boldsymbol{\theta})\, \nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\boldsymbol{\theta})^\top]$ is PSD by construction (it is a covariance matrix of score functions). The natural gradient $\tilde{\nabla} \mathcal{L} = F^{-1} \nabla \mathcal{L}$ uses the Fisher as a Riemannian metric on parameter space. K-FAC (Kronecker-Factored Approximate Curvature) approximates $F$ with a Kronecker-product PSD matrix for scalable second-order optimization.

**Gram matrices and kernels.** The inner product matrix $G_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle$ for any set of vectors is always PSD. Kernel methods rely on this: a function $k(\mathbf{x},\mathbf{z})$ is a valid kernel if and only if its Gram matrix is PSD for any set of inputs (Mercer's theorem). The scaled attention score matrix $QK^\top / \sqrt{d_k}$ in transformers is a (non-symmetric) Gram matrix.

**Cholesky in numerical computation.** Cholesky decomposition ($A = LL^\top$) is the fastest algorithm for solving $A\mathbf{x} = \mathbf{b}$ when $A$ is known to be PD — twice as fast as LU. It is the standard backend for Gaussian process inference, multivariate normal sampling, and Bayesian linear regression. NumPy, SciPy, PyTorch all use Cholesky internally whenever PD structure is detected.

**Hessian and second-order methods.** At a local minimum $\boldsymbol{\theta}^*$, the Hessian $\nabla^2 \mathcal{L}(\boldsymbol{\theta}^*) \succeq 0$. Sharpness-Aware Minimization (SAM) seeks parameters where $\lambda_{\max}(\nabla^2 \mathcal{L})$ is small, empirically improving generalization. The Gauss-Newton approximation $H \approx J^\top J$ is always PSD and is the basis for K-FAC and natural gradient methods.

### 1.4 Historical Timeline

| Year | Contributor | Development |
|------|-------------|-------------|
| 1801 | Gauss | Quadratic forms in celestial mechanics; method of least squares |
| 1847 | Sylvester | Named "positive definite" forms; leading minor criterion |
| 1852 | Sylvester | Sylvester's law of inertia (signature invariance under congruence) |
| 1878 | Frobenius | Systematic theory of bilinear and quadratic forms |
| 1910 | Cholesky | Cholesky decomposition for geodetic calculations (unpublished until 1924) |
| 1934 | Loewner | Matrix ordering (Loewner order) $A \succeq B$ defined rigorously |
| 1936 | Mercer | Mercer's theorem linking PSD functions to kernel expansions |
| 1955 | Bellman | Positive definite matrices in dynamic programming and control |
| 1990 | Vandenberghe & Boyd | Semidefinite programming (SDP) as efficient optimization |
| 1998 | Schölkopf & Smola | Kernel methods: SVMs via PSD kernel matrices |
| 2013 | Kingma & Welling | VAE reparameterization trick using Cholesky sampling |
| 2015 | Martens & Grosse | K-FAC: Kronecker-factored PSD approximation to Fisher |
| 2024–2026 | Multiple groups | PSD structure in diffusion model covariances, normalizing flows, structured covariance estimation |

---

## 2. Formal Definitions

### 2.1 Quadratic Forms

**Definition 2.1 (Quadratic Form).** Let $A \in \mathbb{R}^{n \times n}$ be a symmetric matrix and $\mathbf{x} \in \mathbb{R}^n$. The associated **quadratic form** is the function $Q_A : \mathbb{R}^n \to \mathbb{R}$ defined by

$$Q_A(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x} = \sum_{i=1}^n \sum_{j=1}^n A_{ij} x_i x_j.$$

Note: every quadratic form can be written with a symmetric matrix. If $A$ is not symmetric, replacing it with $(A + A^\top)/2$ gives the same quadratic form, since $\mathbf{x}^\top A \mathbf{x} = \mathbf{x}^\top \frac{A+A^\top}{2} \mathbf{x}$ for all $\mathbf{x}$. **We always assume $A$ is symmetric.**

**Expanding in 2D.** For $A = \begin{pmatrix} a & b \\ b & d \end{pmatrix}$ and $\mathbf{x} = (x_1, x_2)^\top$:

$$Q_A(\mathbf{x}) = ax_1^2 + 2bx_1x_2 + dx_2^2.$$

The diagonal entries $a, d$ give the pure squared terms; the off-diagonal $b$ gives the cross term.

**Completing the square.** For the 1D quadratic $q(x) = ax^2 + 2bx + c = a(x + b/a)^2 + (c - b^2/a)$, the minimum is $c - b^2/a$, achieved at $x = -b/a$. The matrix analogue is the Schur complement (§6) — the quadratic form $\mathbf{x}^\top A \mathbf{x} + 2\mathbf{b}^\top \mathbf{x} + c$ is minimized at $\mathbf{x}^* = -A^{-1}\mathbf{b}$ (when $A \succ 0$) with minimum value $c - \mathbf{b}^\top A^{-1} \mathbf{b}$.

**Connection to bilinear forms.** The quadratic form $Q_A(\mathbf{x})$ is the diagonal of the bilinear form $B_A(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top A \mathbf{y}$, i.e., $Q_A(\mathbf{x}) = B_A(\mathbf{x}, \mathbf{x})$. The bilinear form encodes the inner product structure induced by $A$.

### 2.2 The Four Cases: PD, PSD, ND, Indefinite

**Definition 2.2.** Let $A \in \mathbb{R}^{n \times n}$ be symmetric. Then $A$ is:

| Name | Symbol | Condition | Colloquial |
|------|--------|-----------|-----------|
| **Positive definite** | $A \succ 0$ | $\mathbf{x}^\top A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ | "bowl" |
| **Positive semidefinite** | $A \succeq 0$ | $\mathbf{x}^\top A \mathbf{x} \geq 0$ for all $\mathbf{x}$ | "flat bowl" |
| **Negative definite** | $A \prec 0$ | $\mathbf{x}^\top A \mathbf{x} < 0$ for all $\mathbf{x} \neq \mathbf{0}$ | "dome" |
| **Negative semidefinite** | $A \preceq 0$ | $\mathbf{x}^\top A \mathbf{x} \leq 0$ for all $\mathbf{x}$ | "flat dome" |
| **Indefinite** | — | $Q_A$ takes both positive and negative values | "saddle" |

Note: $A \prec 0 \Leftrightarrow -A \succ 0$. The interesting cases for applications are PD and PSD.

**Standard examples:**

$A = I_n$: $\mathbf{x}^\top I \mathbf{x} = \|\mathbf{x}\|^2 > 0$ for $\mathbf{x} \neq \mathbf{0}$. $I \succ 0$. ✓

$A = \begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}$: $Q = 2x_1^2 + 2x_1x_2 + 2x_2^2 = (x_1+x_2)^2 + x_1^2 + x_2^2 \geq x_1^2 + x_2^2 > 0$ for $\mathbf{x} \neq \mathbf{0}$. $\succ 0$. ✓

$A = \begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}$: $Q = x_1^2 \geq 0$, and $Q(\mathbf{e}_2) = 0$. $\succeq 0$ but not $\succ 0$. (PSD, not PD.)

$A = \begin{pmatrix} 1 & 2 \\ 2 & 1 \end{pmatrix}$: $Q(1,-1) = 1 - 4 + 1 = -2 < 0$, $Q(1,0) = 1 > 0$. Indefinite.

**Non-examples (common mistakes):**

$A = \begin{pmatrix} 2 & 3 \\ 3 & 1 \end{pmatrix}$: looks "positive" but $\det A = 2 - 9 = -7 < 0$, so indefinite.

$A = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}$: $Q \equiv 0$, so PSD but NOT PD.

$A = \begin{pmatrix} -1 & 0 \\ 0 & 2 \end{pmatrix}$: $Q(\mathbf{e}_1) = -1$, indefinite.

### 2.3 Immediate Consequences

If $A \succ 0$, then many properties follow immediately from the definition:

**Proposition 2.3.** Let $A \succ 0$ (symmetric). Then:

1. **Positive diagonal entries.** $A_{ii} > 0$ for all $i$. *Proof:* Take $\mathbf{x} = \mathbf{e}_i$: $\mathbf{e}_i^\top A \mathbf{e}_i = A_{ii} > 0$.

2. **Positive trace.** $\text{tr}(A) = \sum_i A_{ii} > 0$.

3. **Positive determinant.** $\det A > 0$. *(Follows from all eigenvalues positive, §3.1.)*

4. **Invertibility.** $A$ is invertible, and $A^{-1} \succ 0$. *Proof:* If $A\mathbf{x} = \mathbf{0}$ then $\mathbf{x}^\top A \mathbf{x} = 0$, so $\mathbf{x} = \mathbf{0}$. For $A^{-1}$: $(A^{-1})$ is symmetric, and $\mathbf{y}^\top A^{-1} \mathbf{y} = (A^{-1}\mathbf{y})^\top A (A^{-1}\mathbf{y}) > 0$ for $\mathbf{y} \neq \mathbf{0}$.

5. **Closure under positive scaling.** $\alpha A \succ 0$ for $\alpha > 0$.

6. **Closure under addition.** If $A, B \succ 0$ then $A + B \succ 0$. *Proof:* $\mathbf{x}^\top(A+B)\mathbf{x} = \mathbf{x}^\top A\mathbf{x} + \mathbf{x}^\top B\mathbf{x} > 0$.

7. **Tikhonov regularization is PD.** If $A \succeq 0$ and $\lambda > 0$, then $A + \lambda I \succ 0$. *Proof:* $\mathbf{x}^\top(A + \lambda I)\mathbf{x} \geq \lambda \|\mathbf{x}\|^2 > 0$.

8. **Congruence preserves PD.** If $A \succ 0$ and $B$ is invertible, then $B^\top A B \succ 0$. *Proof:* $\mathbf{x}^\top B^\top A B \mathbf{x} = (B\mathbf{x})^\top A (B\mathbf{x}) > 0$ since $B\mathbf{x} \neq \mathbf{0}$ when $\mathbf{x} \neq \mathbf{0}$.

**For AI:** Property 4 ensures that covariance matrices can be inverted for precision computations. Property 7 is exactly why adding $\lambda I$ (weight decay, Tikhonov regularization, jitter in Gaussian processes) to a PSD matrix makes it PD and invertible.

### 2.4 The Loewner Order

**Definition 2.4 (Loewner Order).** For symmetric matrices $A, B \in \mathbb{R}^{n \times n}$, define the **Loewner ordering**:

$$A \succeq B \quad \Longleftrightarrow \quad A - B \succeq 0.$$

Similarly, $A \succ B \Leftrightarrow A - B \succ 0$. This ordering is:

- **Reflexive:** $A \succeq A$.
- **Antisymmetric:** $A \succeq B$ and $B \succeq A$ implies $A = B$.
- **Transitive:** $A \succeq B$ and $B \succeq C$ implies $A \succeq C$.

However, it is **NOT** a total order — not every pair of symmetric matrices is comparable. For example, $\begin{pmatrix}1&0\\0&0\end{pmatrix}$ and $\begin{pmatrix}0&0\\0&1\end{pmatrix}$ are neither $\succeq$ nor $\preceq$ each other.

**Properties of the Loewner order:**

If $A \succeq B$ then:
- $\text{tr}(A) \geq \text{tr}(B)$ (trace is monotone)
- $\lambda_i(A) \geq \lambda_i(B)$ for all $i$ when eigenvalues are sorted in decreasing order (Weyl's theorem)
- $C^\top A C \succeq C^\top B C$ for any $C$ (congruence is monotone)
- If additionally $A, B \succ 0$: $B^{-1} \succeq A^{-1}$ (inversion reverses the order!)

The last property is surprising: if $A \succeq B \succ 0$, then $A^{-1} \preceq B^{-1}$. Intuition: a larger matrix "moves vectors more," so its inverse "moves them less."

**For AI:** In Bayesian inference, the posterior covariance is $\Sigma_{\text{post}} = (\Sigma_{\text{prior}}^{-1} + X^\top X / \sigma^2)^{-1}$. Since we added $X^\top X / \sigma^2 \succeq 0$ to the prior precision, the posterior precision is larger, so by order-reversal, $\Sigma_{\text{post}} \preceq \Sigma_{\text{prior}}$. Observing data never increases uncertainty — the Loewner order makes this rigorous.


---

## 3. Eigenvalue and Determinantal Characterizations

### 3.1 Spectral Characterization

The most computationally transparent characterization of positive definiteness uses eigenvalues.

**Theorem 3.1 (Spectral Characterization).** Let $A \in \mathbb{R}^{n \times n}$ be symmetric with eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$. Then:

$$A \succ 0 \quad \Longleftrightarrow \quad \lambda_i > 0 \text{ for all } i = 1, \ldots, n.$$
$$A \succeq 0 \quad \Longleftrightarrow \quad \lambda_i \geq 0 \text{ for all } i = 1, \ldots, n.$$

**Proof.** By the spectral theorem (→ [§01: Eigenvalues](../01-Eigenvalues-and-Eigenvectors/notes.md)), $A = Q\Lambda Q^\top$ where $Q$ is orthogonal and $\Lambda = \text{diag}(\lambda_1,\ldots,\lambda_n)$. For any $\mathbf{x} \neq \mathbf{0}$, let $\mathbf{y} = Q^\top \mathbf{x} \neq \mathbf{0}$ (since $Q$ is invertible):

$$\mathbf{x}^\top A \mathbf{x} = \mathbf{y}^\top \Lambda \mathbf{y} = \sum_{i=1}^n \lambda_i y_i^2.$$

If all $\lambda_i > 0$: the sum is positive for $\mathbf{y} \neq \mathbf{0}$. ✓

If some $\lambda_k \leq 0$: take $\mathbf{y} = \mathbf{e}_k$ (so $\mathbf{x} = Q\mathbf{e}_k = \mathbf{q}_k$, the $k$-th eigenvector). Then $\mathbf{x}^\top A \mathbf{x} = \lambda_k \leq 0$. ✗

The characterization for $\succeq 0$ follows by the same argument with $\geq$ replacing $>$. $\square$

**Immediate consequences:**
- $\text{tr}(A) = \sum \lambda_i > 0$ for PD
- $\det A = \prod \lambda_i > 0$ for PD
- $\|A\|_2 = \lambda_1$ and $\|A^{-1}\|_2 = 1/\lambda_n$ for PD (→ [§06: Norms](../06-Matrix-Norms/notes.md))
- The condition number $\kappa(A) = \lambda_1/\lambda_n$ measures how "nearly singular" $A$ is

**For AI:** In Gaussian process regression, the covariance kernel matrix $K$ must be PSD. Verifying $K \succeq 0$ by computing eigenvalues is expensive; checking a Cholesky decomposition (§4) is faster and also provides the factorization needed for inference.

> **Note on the spectral theorem:** This proof uses the spectral theorem for symmetric matrices — that every symmetric real matrix is orthogonally diagonalizable. For the full proof and discussion of spectral theory, see [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md).

### 3.2 Sylvester's Criterion

Sylvester's criterion provides a purely determinantal test that avoids eigenvalue computation entirely, making it practical for hand calculations and symbolic proofs.

**Definition 3.2 (Leading Principal Minors).** The **$k$-th leading principal minor** of $A \in \mathbb{R}^{n \times n}$ is:

$$\Delta_k = \det(A[1:k, 1:k]) = \det\begin{pmatrix}A_{11} & \cdots & A_{1k} \\ \vdots & \ddots & \vdots \\ A_{k1} & \cdots & A_{kk}\end{pmatrix}.$$

So $\Delta_1 = A_{11}$, $\Delta_2 = A_{11}A_{22} - A_{12}^2$ (for symmetric $A$), and $\Delta_n = \det A$.

**Theorem 3.3 (Sylvester's Criterion).** A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive definite if and only if all leading principal minors are positive:

$$A \succ 0 \quad \Longleftrightarrow \quad \Delta_k > 0 \text{ for all } k = 1, 2, \ldots, n.$$

**Proof sketch.** The key insight is that the leading principal submatrix $A_k = A[1:k,1:k]$ is itself symmetric, and $A \succ 0$ implies $A_k \succ 0$ for every $k$ (restriction to the first $k$ standard basis vectors). The converse uses induction: if $\Delta_1, \ldots, \Delta_{n-1} > 0$ and $\Delta_n > 0$, then by the inductive hypothesis all principal leading submatrices are PD, and the Cholesky factorization can be completed (§4 gives the constructive proof). The determinant of a PD matrix equals the product of its Cholesky diagonal entries squared, all positive, so $\Delta_n > 0$. $\square$

**Working example.** Test $A = \begin{pmatrix}4 & 2 & 1\\2 & 3 & 0\\1 & 0 & 2\end{pmatrix}$:

- $\Delta_1 = 4 > 0$ ✓
- $\Delta_2 = 4 \cdot 3 - 2^2 = 12 - 4 = 8 > 0$ ✓
- $\Delta_3 = \det A = 4(6-0) - 2(4-0) + 1(0-3) = 24 - 8 - 3 = 13 > 0$ ✓

So $A \succ 0$.

**Caution for PSD.** Sylvester's criterion does NOT characterize $A \succeq 0$. A matrix can have all leading principal minors $\geq 0$ but still not be PSD (a counterexample is $A = \begin{pmatrix}0 & 0 \\ 0 & -1\end{pmatrix}$: $\Delta_1 = 0 \geq 0$, $\Delta_2 = 0 \geq 0$, but $Q(\mathbf{e}_2) = -1 < 0$). For PSD testing, use all principal minors (not just leading ones), or use eigenvalues.

**For AI:** Sylvester's criterion is used in symbolic proofs and in optimization algorithms that maintain PD structure (e.g., proving that a block covariance matrix built by a GP model is PD, by verifying the diagonal blocks and Schur complements are positive).

### 3.3 Pivot Characterization

The connection between Gaussian elimination and positive definiteness is deep and computationally significant.

**Theorem 3.4 (Pivot Characterization).** A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive definite if and only if Gaussian elimination without row exchanges produces all positive pivots.

The pivots of $A$ are $d_1 = A_{11}$, $d_2 = \Delta_2/\Delta_1$, $\ldots$, $d_k = \Delta_k/\Delta_{k-1}$. So $A \succ 0 \Leftrightarrow$ all pivots $d_k > 0 \Leftrightarrow$ all $\Delta_k > 0$ (Sylvester). The LDLᵀ factorization makes this explicit: $A = LDL^\top$ where $D = \text{diag}(d_1, \ldots, d_n)$ and $A \succ 0 \Leftrightarrow$ all diagonal entries of $D$ are positive (§4.4).

The pivot characterization is how Cholesky "discovers" positive definiteness: the Cholesky algorithm fails (tries to take the square root of a non-positive number) exactly when a pivot is non-positive. This makes Cholesky the standard computational test for positive definiteness: try to factor; failure reveals non-PD structure.

```
PIVOT CHARACTERIZATION SUMMARY
════════════════════════════════════════════════════════════════════════

  A ∈ ℝⁿˣⁿ symmetric.  The following are equivalent:

  (i)   A ≻ 0  (quadratic form positive for all x ≠ 0)
  (ii)  All eigenvalues λᵢ > 0                          [Spectral]
  (iii) All leading principal minors Δₖ > 0              [Sylvester]
  (iv)  All Gaussian elimination pivots dₖ > 0           [Pivot]
  (v)   Cholesky factorization A = LL⊤ exists with L     [Cholesky]
        lower triangular, Lᵢᵢ > 0

  Each characterization provides a different computational test:
  (ii)  O(n³): eigenvalue decomposition  ← most expensive
  (iii) O(n⁴): compute n determinants  ← symbolic/manual only
  (iv)  O(n³): Gaussian elimination
  (v)   O(n³/3): Cholesky             ← fastest in practice

════════════════════════════════════════════════════════════════════════
```

### 3.4 Certifying PSD vs PD

In numerical practice, the PD/PSD boundary is a source of subtle bugs. Here we describe how to handle the boundary cases.

**Near-PSD matrices.** A matrix that is theoretically PSD may have small negative eigenvalues due to floating-point errors. For example, if $A$ is constructed as $X^\top X$ but $X$ is nearly rank-deficient, $A$ might have eigenvalues like $-10^{-15}$ due to rounding. The standard fix is to add a small jitter: $A_\epsilon = A + \epsilon I$ for $\epsilon \approx 10^{-6}$ or $10^{-8}$.

**Numerical rank.** For a PSD matrix, the rank equals the number of positive eigenvalues. In finite precision, eigenvalues below $\epsilon \cdot \|A\|_2$ (machine epsilon times spectral norm) are treated as zero. `numpy.linalg.matrix_rank` uses this threshold internally.

**Testing PSD in practice:**
1. Try `np.linalg.cholesky(A)` — succeeds iff $A \succ 0$ (strictly PD)
2. For PSD test: check `np.all(np.linalg.eigvalsh(A) >= -tol)` with `tol = 1e-10`
3. For rank-$r$ PSD: compute `np.linalg.matrix_rank(A)` and verify $r < n$

**The zero-eigenvalue case.** If $A \succeq 0$ and $\text{rank}(A) = r < n$, the Cholesky factorization of the full $n \times n$ matrix does not exist. However, the factorization $A = LL^\top$ where $L$ is $n \times r$ (a "thin" Cholesky) does exist and can be computed via pivoted Cholesky. This is used in sparse GP approximations (Nyström approximation, inducing points).

**For AI:** PyTorch's `torch.linalg.cholesky` raises `torch.linalg.LinAlgError` if the matrix is not PD. In practice, diagonal jitter (`A + 1e-6 * torch.eye(n)`) is the standard workaround in GP implementations and VAE covariance parameterizations.

---

## 4. Cholesky Decomposition

Cholesky decomposition is the most important computational tool in this section. Unlike the spectral characterization (eigenvalue decomposition, $O(n^3)$ with a large constant) or LU (general, $O(n^3)$ with pivoting), Cholesky is:

- **Twice as fast as LU** for symmetric positive definite systems ($\approx n^3/3$ vs $\approx 2n^3/3$ operations)
- **Numerically backward-stable** without pivoting (which LU requires for stability)
- **Reveals the square root:** $L$ satisfies $LL^\top = A$, i.e., $L$ is a "matrix square root"
- **The canonical test** for positive definiteness: it exists iff $A \succ 0$

### 4.1 The Theorem: Existence and Uniqueness

**Theorem 4.1 (Cholesky Factorization).** Let $A \in \mathbb{R}^{n \times n}$ be symmetric. Then:

$$A \succ 0 \quad \Longleftrightarrow \quad \exists! \text{ lower triangular } L \text{ with positive diagonal such that } A = LL^\top.$$

This $L$ is called the **Cholesky factor** of $A$. The factorization $A = LL^\top$ is the **Cholesky decomposition**.

**Proof of existence (constructive).** We prove by induction on $n$. 

*Base case ($n=1$):* $A = (a_{11})$ with $a_{11} > 0$ (since $A \succ 0$). Take $L = (\sqrt{a_{11}})$.

*Inductive step:* Write $A$ in block form:

$$A = \begin{pmatrix} a_{11} & \mathbf{a}^\top \\ \mathbf{a} & B \end{pmatrix}$$

where $a_{11} > 0$ (Proposition 2.3), $\mathbf{a} \in \mathbb{R}^{n-1}$, and $B \in \mathbb{R}^{(n-1)\times(n-1)}$.

Set $\ell_{11} = \sqrt{a_{11}}$, $\boldsymbol{\ell} = \mathbf{a}/\ell_{11}$. Then $B - \boldsymbol{\ell}\boldsymbol{\ell}^\top = B - \mathbf{a}\mathbf{a}^\top/a_{11}$ is the Schur complement $S = B - \mathbf{a} a_{11}^{-1} \mathbf{a}^\top$.

Since $A \succ 0$, its Schur complement $S \succ 0$ (Theorem 6.2). By the inductive hypothesis, $S = L_{22}L_{22}^\top$ for unique lower triangular $L_{22}$ with positive diagonal.

Then:

$$L = \begin{pmatrix}\ell_{11} & \mathbf{0}^\top \\ \boldsymbol{\ell} & L_{22}\end{pmatrix}, \quad LL^\top = \begin{pmatrix}\ell_{11}^2 & \ell_{11}\boldsymbol{\ell}^\top \\ \ell_{11}\boldsymbol{\ell} & \boldsymbol{\ell}\boldsymbol{\ell}^\top + L_{22}L_{22}^\top\end{pmatrix} = \begin{pmatrix}a_{11} & \mathbf{a}^\top \\ \mathbf{a} & B\end{pmatrix} = A.$$

**Proof of uniqueness.** Suppose $A = L_1 L_1^\top = L_2 L_2^\top$ with $L_1, L_2$ lower triangular and positive diagonal. Then $L_2^{-1}L_1 = L_2^\top (L_1^\top)^{-1}$. The left side is lower triangular; the right side is upper triangular. Both sides equal the same matrix, which must be diagonal with positive entries (since $L_1, L_2$ have positive diagonals). Comparing: $L_2^{-1}L_1 = D$ (diagonal positive), so $L_1 = L_2 D$. But $LL^\top = L_2 D D L_2^\top$, so $D^2 = I$, so $D = I$, so $L_1 = L_2$. $\square$

**Converse.** If $A = LL^\top$ with $L$ lower triangular and positive diagonal, then for $\mathbf{x} \neq \mathbf{0}$: $\mathbf{x}^\top A \mathbf{x} = \mathbf{x}^\top L L^\top \mathbf{x} = \|L^\top \mathbf{x}\|^2 > 0$ (since $L^\top$ is invertible by positive diagonal). So $A \succ 0$. $\square$

### 4.2 The Algorithm: Constructing L

The Cholesky algorithm computes $L$ column by column. From $A = LL^\top$, expanding element by element:

$$A_{ij} = \sum_{k=1}^{\min(i,j)} L_{ik} L_{jk}.$$

For the diagonal entry $(i,i)$:

$$A_{ii} = \sum_{k=1}^{i} L_{ik}^2 = L_{ii}^2 + \sum_{k=1}^{i-1} L_{ik}^2 \implies L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}.$$

For off-diagonal entry $(i,j)$ with $i > j$:

$$A_{ij} = \sum_{k=1}^{j} L_{ik}L_{jk} = L_{ij}L_{jj} + \sum_{k=1}^{j-1} L_{ik}L_{jk} \implies L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1} L_{ik}L_{jk}\right).$$

**Algorithm 4.2 (Cholesky, column-wise):**

```
Input:  Symmetric A ∈ ℝⁿˣⁿ
Output: Lower triangular L with A = LL⊤, or FAILURE

for j = 1 to n:
    s = A[j,j] - sum(L[j,k]² for k = 1..j-1)
    if s ≤ 0:
        FAILURE (A is not positive definite)
    L[j,j] = sqrt(s)
    for i = j+1 to n:
        L[i,j] = (A[i,j] - sum(L[i,k]*L[j,k] for k=1..j-1)) / L[j,j]
    for i = 1 to j-1:
        L[i,j] = 0       (upper triangle is zero)
```

**Worked example.** Compute the Cholesky factor of $A = \begin{pmatrix}4 & 2 & 0\\2 & 5 & 1\\0 & 1 & 3\end{pmatrix}$:

Column 1: $L_{11} = \sqrt{4} = 2$, $L_{21} = 2/2 = 1$, $L_{31} = 0/2 = 0$.

Column 2: $L_{22} = \sqrt{5 - 1^2} = \sqrt{4} = 2$, $L_{32} = (1 - 0\cdot1)/2 = 1/2$.

Column 3: $L_{33} = \sqrt{3 - 0^2 - (1/2)^2} = \sqrt{3 - 1/4} = \sqrt{11/4} = \sqrt{11}/2$.

$$L = \begin{pmatrix}2 & 0 & 0 \\ 1 & 2 & 0 \\ 0 & 1/2 & \sqrt{11}/2\end{pmatrix}.$$

Verify: $LL^\top = \begin{pmatrix}4 & 2 & 0\\2 & 5 & 1\\0 & 1 & 3\end{pmatrix} = A$. ✓

### 4.3 Complexity and Numerical Properties

**Flop count.** The Cholesky algorithm requires approximately $n^3/3$ floating-point operations — precisely half the $2n^3/3$ required by LU decomposition. For large $n$, this factor-of-2 speedup is significant.

**Memory.** Only the lower triangle of $L$ needs to be stored: $n(n+1)/2$ entries vs $n^2$ for general LU. In-place variants overwrite the lower triangle of $A$ with $L$.

**Backward stability.** Cholesky is numerically backward-stable without pivoting: the computed $\hat{L}$ satisfies $\hat{L}\hat{L}^\top = A + E$ where $\|E\|_2 \leq c_n \epsilon_{\text{mach}} \|A\|_2$ for a modest constant $c_n$. This is stronger than LU without pivoting (which can be unstable). The reason: each step computes $\sqrt{\text{positive}}$, which cannot amplify errors.

**Solving $A\mathbf{x} = \mathbf{b}$:** Factor $A = LL^\top$, then solve $L\mathbf{y} = \mathbf{b}$ by forward substitution and $L^\top\mathbf{x} = \mathbf{y}$ by backward substitution. Total cost: $n^3/3 + 2n^2$ (factorization + two triangular solves).

**Failure = non-PD.** If at any step the quantity under the square root is non-positive, $A$ is not positive definite. This makes Cholesky the standard computational test for PD structure.

**For AI:** JAX's `jax.scipy.linalg.cholesky`, PyTorch's `torch.linalg.cholesky`, and SciPy's `scipy.linalg.cholesky` are all efficient LAPACK-backed implementations. In Gaussian process models, the Cholesky solve `L \ b` (forward substitution) appears in every prediction and log-marginal-likelihood computation.

### 4.4 LDLᵀ Factorization

The LDLᵀ decomposition is a variant of Cholesky that avoids square roots — useful when the matrix is PSD but not strictly PD, or when square root computation is expensive.

**Theorem 4.3 (LDLᵀ Factorization).** Every symmetric $A \in \mathbb{R}^{n \times n}$ that admits Gaussian elimination without row swaps can be uniquely factored as:

$$A = LDL^\top$$

where $L$ is unit lower triangular ($L_{ii} = 1$) and $D = \text{diag}(d_1, \ldots, d_n)$.

Moreover, $A \succ 0 \Leftrightarrow$ all diagonal entries $d_i > 0$.

**Relation to Cholesky.** If $A \succ 0$, the LDLᵀ and Cholesky factorizations are related by:

$$A = LDL^\top = L\sqrt{D}\sqrt{D}L^\top = (L\sqrt{D})(L\sqrt{D})^\top.$$

So the Cholesky factor is $\hat{L} = L\sqrt{D}$ where $\sqrt{D} = \text{diag}(\sqrt{d_1},\ldots,\sqrt{d_n})$. The LDLᵀ decomposition computes the $d_i$ directly without taking square roots.

**Algorithm 4.4 (LDLᵀ, column-wise):**

```
for j = 1 to n:
    v[1..j-1] = D[1..j-1] * L[j,1..j-1]   (element-wise)
    D[j] = A[j,j] - dot(L[j,1..j-1], v[1..j-1])
    for i = j+1 to n:
        L[i,j] = (A[i,j] - dot(L[i,1..j-1], v[1..j-1])) / D[j]
    L[j,j] = 1
```

**Worked example.** Factor $A = \begin{pmatrix}4 & 2\\2 & 5\end{pmatrix}$:

$d_1 = 4$, $L_{21} = 2/4 = 1/2$, $d_2 = 5 - (1/2)^2 \cdot 4 = 5 - 1 = 4$.

$$L = \begin{pmatrix}1 & 0 \\ 1/2 & 1\end{pmatrix}, \quad D = \begin{pmatrix}4 & 0 \\ 0 & 4\end{pmatrix}.$$

Verify: $LDL^\top = \begin{pmatrix}1&0\\1/2&1\end{pmatrix}\begin{pmatrix}4&0\\0&4\end{pmatrix}\begin{pmatrix}1&1/2\\0&1\end{pmatrix} = \begin{pmatrix}4&2\\2&5\end{pmatrix}$. ✓

**When to prefer LDLᵀ over Cholesky:**
- When $A$ is indefinite but you still want a factorization (use $BKLT$ pivoted LDLᵀ with $2\times 2$ pivots)
- When square root computation is numerically undesirable
- In interval arithmetic or symbolic computation

**Symmetric indefinite systems.** The Bunch-Kaufman (BK) algorithm extends LDLᵀ to indefinite matrices using $1 \times 1$ and $2 \times 2$ pivots. The LAPACK routine `dsytrf` implements this.

### 4.5 Modified Cholesky for Near-PD Matrices

In optimization (especially trust-region methods and quasi-Newton methods), we frequently need to factor a matrix that is "nearly" PD — the Hessian approximation may have small negative eigenvalues due to finite differences or insufficient curvature.

**The problem.** Standard Cholesky fails (negative pivot) when $A$ is not strictly PD. Rather than failing, we want a PD matrix $A + E$ close to $A$ that can be Cholesky-factored.

**Gill-Murray-Wright (GMW) modification.** At each pivot step $j$, if the computed pivot $d_j < \delta$ for a small threshold $\delta > 0$, set $d_j \leftarrow \delta + |d_j|$ (adding a correction). After factorization, $A + E = LL^\top$ where $E = L_{\text{computed}}L_{\text{computed}}^\top - A$ is a small diagonal correction.

**Simple diagonal jitter.** The most widely used approach in machine learning is to add $\epsilon I$ before attempting Cholesky:

```python
def robust_cholesky(A, jitter=1e-6):
    try:
        return np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(A + jitter * np.eye(len(A)))
```

This is the standard "GP jitter" pattern. The added $\epsilon I$ corresponds to assuming a small observation noise or numerical regularization.

**For AI:** PyTorch GPyTorch (Gaussian Process library) wraps all kernel matrix Cholesky calls with diagonal jitter and a retry mechanism. The JAX implementation uses `jax.scipy.linalg.cholesky` and adds jitter via `A += diag_jitter * jnp.eye(n)`. Understanding why jitter works — it adds $\epsilon$ to each eigenvalue, making all eigenvalues at least $\epsilon > 0$ — is exactly positive definiteness via the spectral characterization.


---

## 5. Matrix Square Root and Whitening

### 5.1 The PSD Square Root

For a scalar $a \geq 0$, there is a unique non-negative square root $\sqrt{a} \geq 0$. For a PSD matrix, the same uniqueness holds — but only in the class of PSD matrices.

**Theorem 5.1 (PSD Square Root).** Let $A \in \mathbb{R}^{n \times n}$ be symmetric and positive semidefinite. Then there exists a unique symmetric positive semidefinite matrix $A^{1/2}$ such that:

$$(A^{1/2})^2 = A^{1/2} A^{1/2} = A.$$

If $A \succ 0$, then $A^{1/2} \succ 0$.

**Proof.** By the spectral theorem, $A = Q\Lambda Q^\top$ where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ with all $\lambda_i \geq 0$. Define:

$$A^{1/2} = Q \Lambda^{1/2} Q^\top, \quad \Lambda^{1/2} = \text{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n}).$$

Then $(A^{1/2})^2 = Q\Lambda^{1/2}Q^\top Q\Lambda^{1/2}Q^\top = Q\Lambda Q^\top = A$. This $A^{1/2}$ is symmetric PSD.

**Uniqueness:** Suppose $B^2 = A$ with $B$ symmetric PSD. Then $B$ and $A$ commute (since $BA = B \cdot B^2 = B^3 = B^2 \cdot B = A B$), so they share eigenvectors. On each shared eigenspace with eigenvalue $\lambda$, $B$ must equal $\sqrt{\lambda}$ (the unique non-negative root). So $B = Q\Lambda^{1/2}Q^\top = A^{1/2}$. $\square$

**Warning: Non-uniqueness without PSD constraint.** The equation $B^2 = A$ for $A \succ 0$ has $2^n$ solutions in the class of symmetric matrices (one can independently choose $\pm\sqrt{\lambda_i}$ for each eigenvalue). The PSD square root $A^{1/2}$ is the unique solution with all non-negative eigenvalues.

### 5.2 Computing Square Roots

**Via eigendecomposition:**

$A = Q\Lambda Q^\top \implies A^{1/2} = Q\Lambda^{1/2}Q^\top.$

This requires the full eigendecomposition, costing $O(n^3)$ with a large constant. For $A \succ 0$:

```python
eigvals, Q = np.linalg.eigh(A)   # symmetric eigendecomposition
A_sqrt = Q @ np.diag(np.sqrt(eigvals)) @ Q.T
```

**Via Cholesky (the "left Cholesky factor").** Note: the Cholesky factor $L$ satisfies $LL^\top = A$ but $L \neq A^{1/2}$ in general (unless $A$ is diagonal). The Cholesky factor is a square root of $A$ in the sense $A = LL^\top$, but it is not symmetric and not equal to the eigendecomposition-based $A^{1/2}$.

**Via Newton's method (for matrix functions):** The iteration $X_{k+1} = \frac{1}{2}(X_k + X_k^{-1}A)$ starting from $X_0 = I$ converges to $A^{1/2}$ for $A \succ 0$. This is the matrix analogue of Heron's method for scalar square roots. Convergence is quadratic once close enough.

**Via Padé approximants:** For $A$ close to $I$, write $A = I + E$ and use a matrix series $A^{1/2} = (I+E)^{1/2} = I + \frac{1}{2}E - \frac{1}{8}E^2 + \cdots$, truncated at some order.

### 5.3 Square Root vs Cholesky

The two "square root" operations serve different purposes:

| Property | Cholesky factor $L$ | PSD square root $A^{1/2}$ |
|----------|---------------------|--------------------------|
| Satisfies | $A = LL^\top$ | $A = (A^{1/2})^2$ |
| Shape | Lower triangular | Symmetric |
| Uniqueness | Unique (positive diagonal) | Unique (PSD) |
| Computation | $O(n^3/3)$ | $O(n^3)$ (full eigen) |
| Invertibility | $L^{-1}$ is lower triangular | $(A^{1/2})^{-1} = A^{-1/2}$ |
| Use case | Solving systems, sampling | Mahalanobis, whitening |

**When to use Cholesky:** solving $A\mathbf{x}=\mathbf{b}$, computing $\log\det A$, sampling from $\mathcal{N}(\mathbf{0},A)$.

**When to use $A^{1/2}$:** defining the Mahalanobis inner product $\langle\mathbf{x},\mathbf{y}\rangle_A = \mathbf{x}^\top A^{-1}\mathbf{y} = \|A^{-1/2}\mathbf{x}\|^2$; whitening transforms; matrix differential equations.

### 5.4 Whitening Transforms

**Definition 5.2 (Whitening).** Given data $\mathbf{x}$ with covariance $\Sigma \succ 0$, the **whitening transform** maps $\mathbf{x} \mapsto \mathbf{z} = \Sigma^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$. The transformed variable $\mathbf{z}$ has covariance $I$ (identity) — it is "white."

**ZCA whitening.** The ZCA (Zero-phase Component Analysis) whitening uses the symmetric square root: $W_{\text{ZCA}} = \Sigma^{-1/2}$. The transformed data $W_{\text{ZCA}}\mathbf{x}$ is as close to the original $\mathbf{x}$ as possible while being white. This minimizes the change in "appearance" of the data.

**Cholesky whitening.** Using $W_{\text{Chol}} = L^{-1}$ (where $\Sigma = LL^\top$) also produces white data but rotates it differently. This form is computationally cheaper.

**Mahalanobis distance.** For $A = \Sigma^{-1}$, the quadratic form $\mathbf{x}^\top \Sigma^{-1} \mathbf{x}$ is the squared Mahalanobis distance from the origin. Geometrically, it measures the distance in "standard deviations units," accounting for the correlation structure of $\Sigma$. In $n$ dimensions, $(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}) = \|(\mathbf{x}-\boldsymbol{\mu})\|_{\Sigma^{-1}}^2$.

**For AI:** Batch normalization can be viewed as approximate ZCA whitening per mini-batch. More precisely, it normalizes each feature to zero mean and unit variance (diagonal whitening), avoiding the expensive full whitening. The full Mahalanobis distance appears in metric learning (e.g., Siamese networks learning $A$ such that $\mathbf{x}^\top A \mathbf{x}$ is a meaningful distance) and in attention mechanisms where different layers may learn non-isotropic attention kernels.

---

## 6. Schur Complement

The Schur complement is the "matrix analogue of completing the square" for block matrices. It appears everywhere in probability (Gaussian conditioning), linear algebra (block matrix inversion), and optimization (constraint elimination).

### 6.1 Definition for Block Matrices

**Definition 6.1 (Schur Complement).** Let

$$M = \begin{pmatrix}A & B \\ C & D\end{pmatrix}$$

be a block matrix with $A \in \mathbb{R}^{p \times p}$ invertible. The **Schur complement** of $A$ in $M$ is:

$$S = D - CA^{-1}B \in \mathbb{R}^{q \times q}.$$

Similarly, if $D \in \mathbb{R}^{q \times q}$ is invertible, the Schur complement of $D$ is $A - BD^{-1}C$.

**Origin: block Gaussian elimination.** The Schur complement arises naturally when eliminating the $(2,1)$ block:

$$M = \begin{pmatrix}A & B \\ C & D\end{pmatrix} = \begin{pmatrix}I & 0 \\ CA^{-1} & I\end{pmatrix}\begin{pmatrix}A & B \\ 0 & D-CA^{-1}B\end{pmatrix}.$$

The $(2,2)$ block in the upper triangular factor is exactly $S = D - CA^{-1}B$.

**Determinant formula.** A key consequence of the block LU:

$$\det M = \det A \cdot \det(D - CA^{-1}B) = \det A \cdot \det S.$$

Similarly, $\det M = \det D \cdot \det(A - BD^{-1}C)$ when $D$ is invertible.

**Block matrix inverse.** Using the Schur complement:

$$M^{-1} = \begin{pmatrix}A^{-1} + A^{-1}B S^{-1} C A^{-1} & -A^{-1}B S^{-1} \\ -S^{-1}CA^{-1} & S^{-1}\end{pmatrix}$$

when both $A$ and $S = D - CA^{-1}B$ are invertible.

### 6.2 Schur Complement and Positive Definiteness

The Schur complement provides an elegant characterization of block PD matrices.

**Theorem 6.2 (Schur PD Criterion).** Let $M = \begin{pmatrix}A & B \\ B^\top & D\end{pmatrix}$ be symmetric (so $C = B^\top$). Then:

$$M \succ 0 \quad \Longleftrightarrow \quad A \succ 0 \text{ and } S = D - B^\top A^{-1} B \succ 0.$$

**Proof.** We use the block Cholesky:

$$M = \begin{pmatrix}A & B \\ B^\top & D\end{pmatrix} = \begin{pmatrix}I & 0 \\ B^\top A^{-1} & I\end{pmatrix}\begin{pmatrix}A & 0 \\ 0 & S\end{pmatrix}\begin{pmatrix}I & A^{-1}B \\ 0 & I\end{pmatrix}.$$

The middle factor is block diagonal: $\begin{pmatrix}A & 0 \\ 0 & S\end{pmatrix}$. For any $\mathbf{v} = (\mathbf{x}, \mathbf{y})^\top$:

$$\mathbf{v}^\top M \mathbf{v} = (\mathbf{x} + A^{-1}B\mathbf{y})^\top A (\mathbf{x} + A^{-1}B\mathbf{y}) + \mathbf{y}^\top S \mathbf{y}.$$

($\Rightarrow$): If $M \succ 0$, taking $\mathbf{y} = \mathbf{0}$ shows $A \succ 0$; taking $\mathbf{x} = -A^{-1}B\mathbf{y}$ shows $\mathbf{y}^\top S\mathbf{y} > 0$ for $\mathbf{y} \neq 0$, so $S \succ 0$.

($\Leftarrow$): If $A \succ 0$ and $S \succ 0$, then both terms are non-negative, and at least one is positive for $(\mathbf{x},\mathbf{y}) \neq (\mathbf{0},\mathbf{0})$, so $M \succ 0$. $\square$

**Corollary.** $M \succeq 0 \Leftrightarrow A \succeq 0$ and $S = D - B^\top A^{-1}B \succeq 0$ (when $A$ is invertible; otherwise use the rank condition).

```
SCHUR COMPLEMENT AND BLOCK PD
════════════════════════════════════════════════════════════════════════

  M = ( A    B  )  symmetric
      ( B⊤   D  )
      
  M ≻ 0  ⟺  A ≻ 0  AND  S = D - B⊤A⁻¹B ≻ 0

  Intuition: "completing the square" in block form
  
  v⊤Mv = (x + A⁻¹By)⊤A(x + A⁻¹By) + y⊤Sy
          ─────────────────────────────   ─────
          ≥ 0 (since A ≻ 0)              ≥ 0 (since S ≻ 0)

════════════════════════════════════════════════════════════════════════
```

### 6.3 Matrix Inversion Lemma

The **Woodbury matrix identity** (also called the matrix inversion lemma or Sherman-Morrison-Woodbury formula) is:

$$(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}$$

where $A \in \mathbb{R}^{n \times n}$, $C \in \mathbb{R}^{k \times k}$, $U \in \mathbb{R}^{n \times k}$, $V \in \mathbb{R}^{k \times n}$. This is valuable when $k \ll n$ (low-rank update): instead of inverting an $n \times n$ matrix, invert a $k \times k$ matrix.

**Derivation via Schur complement.** Consider the block system:

$$M = \begin{pmatrix}A & U \\ -V & C^{-1}\end{pmatrix}.$$

The Schur complement of $C^{-1}$ is $A - U(-V)^{-1}(-V) = A + UCV$ (with some sign manipulation). The Schur complement of $A$ is $C^{-1} + VA^{-1}U$. Using the block inverse formula gives the Woodbury identity.

**Special case (rank-1 update):** If $U = \mathbf{u}$, $V = \mathbf{v}^\top$, $C = c$ (scalar):

$$(A + c\mathbf{u}\mathbf{v}^\top)^{-1} = A^{-1} - \frac{c A^{-1}\mathbf{u}\mathbf{v}^\top A^{-1}}{1 + c\mathbf{v}^\top A^{-1}\mathbf{u}}.$$

This is the Sherman-Morrison formula for rank-1 updates.

**For AI:** The Woodbury identity is used in:
- **Gaussian process prediction:** posterior covariance $(K + \sigma^2 I)^{-1}$ where $K = X^\top X$ and $\sigma^2 I$ is noise; Woodbury allows using $n \times n$ or $p \times p$ (whichever is smaller)
- **LoRA / low-rank adaptation:** the effective weight $W_0 + BA$ is a rank-$r$ update; Woodbury allows efficient inversion without materializing the full matrix
- **Kalman filter update step:** $(P^{-1} + H^\top R^{-1}H)^{-1}$ uses Woodbury to avoid inverting large state covariances

### 6.4 Gaussian Conditioning via Schur Complement

The Schur complement is the algebraic engine behind the conditional distribution of multivariate Gaussians.

**Setup.** Let $\begin{pmatrix}\mathbf{x}_1 \\ \mathbf{x}_2\end{pmatrix} \sim \mathcal{N}\!\left(\begin{pmatrix}\boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2\end{pmatrix}, \begin{pmatrix}\Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{pmatrix}\right)$.

**Conditional distribution.** The conditional $\mathbf{x}_1 | \mathbf{x}_2 = \mathbf{a}$ is Gaussian:

$$\mathbf{x}_1 | \mathbf{x}_2 = \mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu}_{1|2},\, \Sigma_{1|2})$$

where:

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{a} - \boldsymbol{\mu}_2)$$
$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}.$$

The conditional covariance $\Sigma_{1|2}$ is exactly the **Schur complement** of $\Sigma_{22}$ in $\Sigma$. Theorem 6.2 guarantees $\Sigma_{1|2} \succ 0$ whenever $\Sigma \succ 0$.

**Derivation.** Complete the square in the joint density. The exponent:

$$(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$$

factored using the block inverse of $\Sigma^{-1}$ (the "precision matrix") gives the Schur complement form.

**For AI:** In Gaussian process regression, the predictive distribution at new points $\mathbf{x}_*$ given training observations $\mathbf{y}$ uses exactly this formula:

$$\boldsymbol{\mu}_* = K_{*n}(K_{nn} + \sigma^2 I)^{-1}\mathbf{y}, \quad \Sigma_* = K_{**} - K_{*n}(K_{nn} + \sigma^2 I)^{-1}K_{n*}.$$

The second term is the Schur complement. Computing it via Cholesky: $L = \text{chol}(K_{nn} + \sigma^2 I)$, then $\Sigma_* = K_{**} - (L^{-1}K_{n*})^\top(L^{-1}K_{n*})$.


---

## 7. Log-Determinant

### 7.1 Definition and Properties

For a positive definite matrix $A \succ 0$, the **log-determinant** is:

$$\log\det A = \log \prod_{i=1}^n \lambda_i = \sum_{i=1}^n \log \lambda_i.$$

Since $A \succ 0$ implies $\lambda_i > 0$ for all $i$, this is well-defined. The log-det is defined only on the interior of the PSD cone (i.e., on PD matrices).

**Key computation via Cholesky:**

$$\log\det A = \log\det(LL^\top) = \log(\det L)^2 = 2\log\det L = 2\sum_{i=1}^n \log L_{ii}.$$

Since $L_{ii} > 0$ (Cholesky diagonal is positive), $\log L_{ii}$ is well-defined. This is the standard computational formula: factor $A = LL^\top$, then sum the logs of the diagonal entries.

```python
L = np.linalg.cholesky(A)
log_det_A = 2 * np.sum(np.log(np.diag(L)))
```

This is numerically more stable than `np.log(np.linalg.det(A))` for large matrices, because `det` can underflow or overflow.

**Properties:**

- $\log\det(AB) = \log\det A + \log\det B$ (for any invertible $A, B$)
- $\log\det(A^{-1}) = -\log\det A$
- $\log\det(\alpha A) = n\log\alpha + \log\det A$ for scalar $\alpha > 0$
- $\log\det(A) = \text{tr}(\log A)$ where $\log A$ is the matrix logarithm (eigendecomposition-based)
- As $A \to \partial \mathbb{S}_+^n$ (boundary, a singular matrix): $\log\det A \to -\infty$

### 7.2 Log-Det as a Concave Function

**Theorem 7.1.** The function $f: \mathbb{S}_{++}^n \to \mathbb{R}$ defined by $f(A) = \log\det A$ is **strictly concave** on the cone of PD matrices.

**Proof.** We need to show $f(\lambda A + (1-\lambda)B) \geq \lambda f(A) + (1-\lambda) f(B)$ for $A, B \succ 0$ and $\lambda \in (0,1)$, with equality iff $A = B$.

Fix $A \succ 0$ and let $C = A^{-1/2}BA^{-1/2} \succ 0$. The eigenvalues of $C$ are $\mu_1 \geq \cdots \geq \mu_n > 0$.

$$f(\lambda A + (1-\lambda)B) = \log\det(\lambda A + (1-\lambda)B).$$

Factoring: $\lambda A + (1-\lambda)B = A^{1/2}(\lambda I + (1-\lambda)C)A^{1/2}$, so:

$$f(\lambda A + (1-\lambda)B) = \log\det A + \sum_{i=1}^n \log(\lambda + (1-\lambda)\mu_i).$$

Since $g(t) = \log t$ is strictly concave: $\log(\lambda + (1-\lambda)\mu_i) \geq \lambda\log 1 + (1-\lambda)\log\mu_i = (1-\lambda)\log\mu_i$.

Summing: $f(\lambda A + (1-\lambda)B) \geq \log\det A + (1-\lambda)\sum_i \log\mu_i = \lambda f(A) + (1-\lambda)f(B)$.

Equality holds iff all $\mu_i = 1$, i.e., $C = I$, i.e., $B = A$. $\square$

**Consequences of concavity:**
- The log-det has no local maxima except at the global maximum (over any convex domain)
- Gradient methods for maximizing $\log\det A$ over a convex set converge to the global maximum
- The function is differentiable at every PD matrix (the gradient is $A^{-1}$, see §7.3)

### 7.3 Gradient and Matrix Calculus

**Theorem 7.2 (Log-Det Gradient).** Let $f(A) = \log\det A$ for $A \succ 0$. Then:

$$\frac{\partial \log\det A}{\partial A} = A^{-\top} = A^{-1} \quad (\text{since } A \text{ is symmetric}).$$

More precisely, in the matrix calculus convention where $\partial f/\partial A_{ij}$ means the $(i,j)$ entry of the gradient matrix:

$$\left(\frac{\partial \log\det A}{\partial A}\right)_{ij} = (A^{-1})_{ji}.$$

**Proof using Jacobi's formula.** Jacobi's formula states $d(\det A) = \text{tr}(\text{adj}(A)\, dA) = \det(A)\,\text{tr}(A^{-1}\,dA)$. Therefore:

$$d(\log\det A) = \frac{d(\det A)}{\det A} = \text{tr}(A^{-1}\,dA).$$

Since $d(\log\det A) = \langle \nabla_A \log\det A,\, dA\rangle_F = \text{tr}((\nabla_A \log\det A)^\top dA)$, we read off $\nabla_A \log\det A = A^{-\top} = A^{-1}$ for symmetric $A$.

**Second derivative (Hessian operator):** For the function $A \mapsto \log\det A$:

$$d^2(\log\det A)[H, H] = -\text{tr}(A^{-1}H A^{-1}H) = -\text{tr}((A^{-1/2}HA^{-1/2})^2) \leq 0.$$

The negative sign confirms concavity.

**Other useful log-det derivatives:**

- $\frac{\partial}{\partial \mathbf{a}} \log\det(A + \mathbf{a}\mathbf{a}^\top) = 2(A + \mathbf{a}\mathbf{a}^\top)^{-1}\mathbf{a}$ (rank-1 update)
- $\frac{\partial}{\partial t} \log\det(A + tB) = \text{tr}((A+tB)^{-1}B)$

### 7.4 Log-Det in Machine Learning

**Multivariate Gaussian log-likelihood.** For $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$:

$$\log p(\mathbf{x}) = -\frac{n}{2}\log(2\pi) - \frac{1}{2}\log\det\Sigma - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}).$$

The term $-\frac{1}{2}\log\det\Sigma$ penalizes large (spread-out) distributions. When fitting $\Sigma$ to data, maximizing the log-likelihood requires differentiating through $\log\det\Sigma$, using $\partial\log\det\Sigma/\partial\Sigma = \Sigma^{-1}$.

**Gaussian process marginal likelihood.** For GP regression with kernel matrix $K$ and noise $\sigma^2$:

$$\log p(\mathbf{y}) = -\frac{1}{2}\mathbf{y}^\top(K+\sigma^2 I)^{-1}\mathbf{y} - \frac{1}{2}\log\det(K+\sigma^2I) - \frac{n}{2}\log(2\pi).$$

Both terms require Cholesky: $L = \text{chol}(K + \sigma^2 I)$, then $\log\det = 2\sum\log L_{ii}$ and the quadratic form via triangular solves. Gradient with respect to kernel hyperparameters uses $\partial\log p/\partial\theta = \text{tr}(\alpha\alpha^\top - (K+\sigma^2I)^{-1})\partial K/\partial\theta$ where $\alpha = (K+\sigma^2I)^{-1}\mathbf{y}$.

**Normalizing flows.** A normalizing flow defines a bijective mapping $f: \mathbf{z} \mapsto \mathbf{x}$ where $\mathbf{z} \sim \mathcal{N}(0,I)$. The log-likelihood of data $\mathbf{x}$ is:

$$\log p(\mathbf{x}) = \log p_z(f^{-1}(\mathbf{x})) + \log|\det J_{f^{-1}}|$$

where $J_{f^{-1}}$ is the Jacobian. Efficiently computing $\log|\det J|$ is the central computational challenge of normalizing flows. Architectures like RealNVP (triangular Jacobian, $\log|\det J| = \sum\log|J_{ii}|$) and FFJORD (stochastic trace estimator) are designed specifically to make this computation tractable.

**Log-det estimators for large matrices.** When $K$ is very large (e.g., a kernel matrix for millions of training points), exact Cholesky is intractable. Randomized log-det estimators use the identity $\log\det K = \text{tr}(\log K)$ and the stochastic trace estimator $\text{tr}(\log K) \approx \frac{1}{m}\sum_{i=1}^m \mathbf{z}_i^\top (\log K)\mathbf{z}_i$ with random $\mathbf{z}_i \sim \mathcal{N}(0,I)$, computed via Lanczos iterations. This is used in scalable GP libraries (GPyTorch, 2018).

---

## 8. Gram Matrices and Kernel Connections

### 8.1 The Gram Matrix Construction

**Definition 8.1 (Gram Matrix).** Let $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^d$ be a collection of vectors. Their **Gram matrix** is:

$$G \in \mathbb{R}^{n \times n}, \quad G_{ij} = \langle \mathbf{x}_i, \mathbf{x}_j \rangle = \mathbf{x}_i^\top \mathbf{x}_j.$$

In matrix form: if $X = [\mathbf{x}_1|\cdots|\mathbf{x}_n]^\top \in \mathbb{R}^{n \times d}$ (rows are data points), then $G = XX^\top$.

**Theorem 8.2.** Every Gram matrix is positive semidefinite. Moreover, $G \succ 0$ iff $\mathbf{x}_1, \ldots, \mathbf{x}_n$ are linearly independent.

**Proof.** For any $\mathbf{c} \in \mathbb{R}^n$:

$$\mathbf{c}^\top G \mathbf{c} = \sum_{i,j} c_i G_{ij} c_j = \sum_{i,j} c_i \mathbf{x}_i^\top \mathbf{x}_j c_j = \left\|\sum_i c_i \mathbf{x}_i\right\|^2 \geq 0.$$

So $G \succeq 0$. Equality holds iff $\sum_i c_i\mathbf{x}_i = \mathbf{0}$, i.e., iff the vectors are linearly dependent. So $G \succ 0$ iff they are linearly independent. $\square$

**Corollary.** $\text{rank}(G) = \text{rank}(X)$, the number of linearly independent data vectors.

**Examples:**
- $X = I_n$ (standard basis): $G = I_n \succ 0$.
- $n > d$: $\text{rank}(G) \leq d < n$, so $G \succeq 0$ but $G \not\succ 0$.
- $n = d$, $X$ invertible: $G \succ 0$.

### 8.2 Every PSD Matrix is a Gram Matrix

**Theorem 8.3.** $G \succeq 0$ if and only if $G$ is the Gram matrix of some set of vectors in some inner product space.

**Proof.** ($\Leftarrow$): proved above. ($\Rightarrow$): If $G \succeq 0$, let $L$ be the Cholesky-like factor: $G = LL^\top$ (where $L$ may be rectangular, $n \times r$, $r = \text{rank}(G)$). Take $\mathbf{x}_i^\top = L[i,:]$ (the $i$-th row of $L$). Then $G_{ij} = L[i,:] \cdot L[j,:]^\top = \mathbf{x}_i^\top \mathbf{x}_j$. $\square$

This is a deep result: the class of PSD matrices and the class of Gram matrices are exactly the same. Any PSD matrix can be "explained" as a matrix of inner products between some set of vectors.

**Feature maps.** If $\phi: \mathcal{X} \to \mathbb{R}^d$ is a feature map, then the Gram matrix $G_{ij} = \phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$ is PSD. Kernel methods replace $\phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)$ with $k(\mathbf{x}_i, \mathbf{x}_j)$ directly, avoiding explicit feature computation.

### 8.3 Kernel Matrices and Mercer's Theorem

A **positive definite kernel** is a function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ such that for every finite set $\{x_1,\ldots,x_n\} \subset \mathcal{X}$, the Gram matrix $K_{ij} = k(x_i, x_j)$ is PSD.

**Mercer's Theorem (informal statement).** A continuous, symmetric function $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a positive definite kernel (i.e., produces PSD Gram matrices for any finite set) if and only if there exists a Hilbert space $\mathcal{H}$ and a feature map $\phi: \mathcal{X} \to \mathcal{H}$ such that:

$$k(x, z) = \langle \phi(x), \phi(z) \rangle_{\mathcal{H}}.$$

This is the mathematical foundation of the **kernel trick**: instead of explicitly computing $\phi(x)$ (which may be infinite-dimensional), we evaluate $k(x,z)$ directly.

> **Forward reference:** The full treatment of Mercer's theorem, reproducing kernel Hilbert spaces (RKHS), and the kernel trick appears in [Chapter 12: Functional Analysis](../../12-Functional-Analysis/01-Hilbert-Spaces/notes.md).

**Standard PD kernels:**
- Linear kernel: $k(\mathbf{x},\mathbf{z}) = \mathbf{x}^\top\mathbf{z}$ (standard Gram matrix)
- RBF/Gaussian kernel: $k(\mathbf{x},\mathbf{z}) = \exp(-\|\mathbf{x}-\mathbf{z}\|^2 / 2\ell^2)$
- Polynomial kernel: $k(\mathbf{x},\mathbf{z}) = (\mathbf{x}^\top\mathbf{z} + c)^d$ for $c \geq 0$, $d \in \mathbb{N}$
- Matérn kernel: used in GP regression with controllable smoothness

**Verifying kernel validity.** For a proposed kernel $k$, the standard test is:
1. Compute the $n \times n$ Gram matrix $K$ for a test set
2. Check $K \succeq 0$ (e.g., via `np.linalg.eigvalsh(K) >= -1e-10`)

### 8.4 Attention Scores as Gram Matrices

The scaled dot-product attention mechanism in transformers computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q, K \in \mathbb{R}^{n \times d_k}$ are query and key matrices.

The score matrix $S = QK^\top / \sqrt{d_k} \in \mathbb{R}^{n \times n}$ is a **scaled Gram matrix** (but with different row-spaces for $Q$ and $K$, so not necessarily PSD). In the special case of self-attention with tied weights $Q = K$, $S$ is proportional to $QQ^\top / \sqrt{d_k} \succeq 0$.

**Why the $1/\sqrt{d_k}$ scaling.** If $Q$ and $K$ have independent entries from $\mathcal{N}(0,1)$, then $Q_{ij}K_{ij}$ has variance 1 and $(QK^\top)_{ij} = \sum_{k=1}^{d_k} Q_{ik}K_{jk}$ has variance $d_k$. The $1/\sqrt{d_k}$ rescaling brings the variance back to 1, preventing the softmax from saturating into near-one-hot distributions.

**Attention as kernel regression.** The attention output for query $\mathbf{q}$ is:

$$\text{output} = \sum_{j=1}^n \frac{\exp(\mathbf{q}^\top\mathbf{k}_j/\sqrt{d_k})}{\sum_l \exp(\mathbf{q}^\top\mathbf{k}_l/\sqrt{d_k})} \mathbf{v}_j.$$

This is a Nadaraya-Watson kernel regression with an exponential kernel $k(\mathbf{q},\mathbf{k}_j) = \exp(\mathbf{q}^\top\mathbf{k}_j/\sqrt{d_k})$. The attention weights are the normalized kernel similarities, and the output is a kernel-weighted average of values. The exponential of the dot product is related to the RBF kernel (by the Johnson-Lindenstrauss / random Fourier features perspective used in Performer / FAVOR+).


---

## 9. The PSD Cone and Semidefinite Programming

### 9.1 The Cone of PSD Matrices

The set of all $n \times n$ symmetric positive semidefinite matrices is denoted $\mathbb{S}_+^n$ (or $\mathbb{S}_{\geq 0}^n$). It lives inside the vector space $\mathbb{S}^n$ of $n \times n$ real symmetric matrices (which has dimension $n(n+1)/2$).

**Theorem 9.1.** $\mathbb{S}_+^n$ is a **proper convex cone**:

1. **Convex:** If $A, B \succeq 0$ and $\lambda \in [0,1]$, then $\lambda A + (1-\lambda)B \succeq 0$.
2. **Cone:** If $A \succeq 0$ and $t \geq 0$, then $tA \succeq 0$.
3. **Pointed:** $A \succeq 0$ and $A \preceq 0$ implies $A = 0$ (no lines through the origin in the cone).
4. **Closed:** $\mathbb{S}_+^n$ is a closed subset of $\mathbb{S}^n$ (limits of PSD sequences are PSD).
5. **Full-dimensional:** The interior of $\mathbb{S}_+^n$ is $\mathbb{S}_{++}^n$ (the PD matrices), which is non-empty.

**Proof of convexity:** For any $\mathbf{x}$: $\mathbf{x}^\top(\lambda A + (1-\lambda)B)\mathbf{x} = \lambda\mathbf{x}^\top A\mathbf{x} + (1-\lambda)\mathbf{x}^\top B\mathbf{x} \geq 0$. $\square$

**The boundary.** The boundary $\partial\mathbb{S}_+^n = \mathbb{S}_+^n \setminus \mathbb{S}_{++}^n$ consists of PSD matrices with at least one zero eigenvalue. These are rank-deficient PSD matrices. The boundary has lower dimension: the set of rank-$r$ PSD matrices is a manifold of dimension $rn - r(r-1)/2$.

**Low-dimensional picture.** For $n=2$: $\mathbb{S}^2$ is 3-dimensional (coordinates $A_{11}, A_{12}, A_{22}$). The PSD cone $\mathbb{S}_+^2$ is the set where $A_{11} \geq 0$, $A_{22} \geq 0$, and $A_{11}A_{22} \geq A_{12}^2$ — the interior of an ice cream cone in 3D.

**Dual cone.** The dual of $\mathbb{S}_+^n$ with respect to the Frobenius inner product $\langle A, B\rangle_F = \text{tr}(AB)$ is:

$$(\mathbb{S}_+^n)^* = \{B \in \mathbb{S}^n : \text{tr}(AB) \geq 0 \text{ for all } A \succeq 0\} = \mathbb{S}_+^n.$$

The PSD cone is **self-dual**: $(\mathbb{S}_+^n)^* = \mathbb{S}_+^n$. This is analogous to the non-negative reals being self-dual.

### 9.2 Semidefinite Programming

**Semidefinite programming (SDP)** is the optimization of a linear objective over the intersection of the PSD cone with an affine set:

**Standard form SDP (primal):**

$$\min_{X \in \mathbb{S}^n} \quad \langle C, X\rangle_F = \text{tr}(CX)$$
$$\text{subject to} \quad \langle A_i, X\rangle_F = b_i, \quad i = 1,\ldots,m$$
$$\quad\quad\quad\quad X \succeq 0.$$

Here $C, A_1, \ldots, A_m \in \mathbb{S}^n$ and $b \in \mathbb{R}^m$ are the problem data; $X \in \mathbb{S}^n$ is the decision variable.

**Dual SDP:**

$$\max_{\mathbf{y} \in \mathbb{R}^m} \quad \mathbf{b}^\top\mathbf{y}$$
$$\text{subject to} \quad C - \sum_{i=1}^m y_i A_i \succeq 0.$$

**Duality.** Weak duality always holds: $\text{primal value} \geq \text{dual value}$. Strong duality (primal = dual) holds under Slater's condition: if the primal is strictly feasible (some $X \succ 0$ satisfies all constraints), then strong duality holds and the dual optimum is attained.

**Relation to other optimization problems.** SDP generalizes:
- **Linear programming (LP):** LP is SDP with diagonal matrices $C, A_i$
- **SOCP (second-order cone programming):** SOCP is a special SDP
- **Quadratically constrained QP:** Many QCQPs can be lifted to SDPs via semidefinite relaxation

**Algorithms.** Interior-point methods (barrier methods) solve SDPs in polynomial time: $O(m^{1.5} n^3)$ per iteration for an $m$-constraint, $n \times n$ SDP. Standard solvers: SCS, MOSEK, CVXPY (modelling layer).

### 9.3 SDP in Machine Learning

**Max-cut relaxation (Goemans-Williamson, 1995).** The max-cut problem on a graph $G=(V,E)$ with edge weights $w_{ij}$ is NP-hard. The Goemans-Williamson SDP relaxation gives a $0.878$-approximation:

$$\max_{X \succeq 0} \quad \frac{1}{2}\sum_{ij} w_{ij}(1 - X_{ij})$$
$$\text{s.t.} \quad X_{ii} = 1, \quad i=1,\ldots,n.$$

The solution $X^*$ satisfies $X^* = VV^\top$ for some unit vectors $\mathbf{v}_1,\ldots,\mathbf{v}_n$; random hyperplane rounding recovers a near-optimal cut.

**Metric learning.** Learning a Mahalanobis distance $d_A(\mathbf{x},\mathbf{z}) = (\mathbf{x}-\mathbf{z})^\top A (\mathbf{x}-\mathbf{z})^{1/2}$ requires $A \succeq 0$. Methods like ITML (Information-Theoretic Metric Learning) and SDML formulate this as an SDP over PSD matrices with constraints that similar pairs are close and dissimilar pairs are far.

**Covariance estimation.** In high dimensions ($p > n$), the sample covariance $\hat{\Sigma} = \frac{1}{n}X^\top X$ may be singular. Regularized covariance estimation (graphical lasso, precision matrix estimation) adds a sparsity constraint or penalty:

$$\min_{\Sigma \succ 0} \left[ \text{tr}(\hat{S}\Sigma^{-1}) - \log\det\Sigma + \lambda\|\Sigma^{-1}\|_1 \right]$$

where $\hat{S}$ is the sample covariance and $\|\cdot\|_1$ is the element-wise $\ell_1$ norm. This is not an SDP directly, but the PSD constraint $\Sigma \succ 0$ is the core structural requirement.

**Fairness constraints.** In algorithmic fairness, PSD constraints arise as necessary conditions for disparate impact compliance. Certified defenses against adversarial examples (semidefinite relaxations of neural network verification) are large-scale SDPs; solvers like $\alpha$-CROWN reformulate them via Lagrangian relaxation.

---

## 10. Applications in Machine Learning

### 10.1 Multivariate Gaussians and Covariance Matrices

**The multivariate Gaussian.** A random vector $\mathbf{x} \in \mathbb{R}^n$ has the distribution $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ if:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\det\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right).$$

For this density to be a valid (normalized, integrable) probability distribution, $\Sigma$ must be symmetric positive definite. The three requirements:

1. **Symmetry:** $\Sigma = \Sigma^\top$ (covariance is symmetric by definition)
2. **Positive definiteness:** $\Sigma \succ 0$ ensures $\det\Sigma \neq 0$ (normalizing constant finite) and $\Sigma^{-1}$ exists (the exponent is a proper quadratic form)
3. **If $\Sigma \succeq 0$ but singular:** The distribution becomes degenerate — supported on an affine subspace, not all of $\mathbb{R}^n$

**Parameterizing covariances in neural networks.** A neural network that outputs a covariance matrix must parameterize it to be PSD. Standard approaches:

- **Diagonal:** $\Sigma = \text{diag}(\exp(\boldsymbol{s}))$ where $\boldsymbol{s}$ is a learned vector. Automatically PD.
- **Cholesky lower triangular:** $\Sigma = LL^\top$ where $L$ has positive diagonal (enforced via softplus on diagonal entries). This is the most expressive parameterization.
- **Low-rank + diagonal:** $\Sigma = FF^\top + D$ where $F \in \mathbb{R}^{n \times k}$ ($k \ll n$) and $D$ diagonal positive. Woodbury allows efficient inversion.

**Sampling from $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$:** Given $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$:

$$\mathbf{x} = \boldsymbol{\mu} + L\boldsymbol{\epsilon}$$

where $L = \text{chol}(\Sigma)$. This is the fundamental sampling algorithm: the Cholesky factor maps isotropic noise to correlated noise.

### 10.2 Fisher Information Matrix and Natural Gradient

**Definition 10.1 (Fisher Information Matrix).** For a statistical model $p(\mathbf{x}|\boldsymbol{\theta})$ with parameter $\boldsymbol{\theta} \in \mathbb{R}^d$, the Fisher information matrix is:

$$F(\boldsymbol{\theta}) = \mathbb{E}_{\mathbf{x} \sim p(\cdot|\boldsymbol{\theta})}\!\left[\nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\boldsymbol{\theta})\, \nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\boldsymbol{\theta})^\top\right].$$

**PSD proof.** $F$ is a covariance matrix of the score function $\nabla_{\boldsymbol{\theta}} \log p(\mathbf{x}|\boldsymbol{\theta})$: it is $\mathbb{E}[\mathbf{s}\mathbf{s}^\top]$ where $\mathbf{s}$ is a random vector. Any matrix of the form $\mathbb{E}[\mathbf{s}\mathbf{s}^\top]$ is PSD (it is the expected outer product). In fact, $F \succ 0$ for regular statistical models (identifiable, full rank).

**Natural gradient.** Ordinary gradient descent $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\nabla_{\boldsymbol{\theta}}\mathcal{L}$ uses the Euclidean metric on parameter space. The natural gradient uses the Fisher metric:

$$\tilde{\nabla}\mathcal{L} = F(\boldsymbol{\theta})^{-1}\nabla_{\boldsymbol{\theta}}\mathcal{L}.$$

The natural gradient is invariant to reparameterization of the model — it measures the steepest descent direction with respect to the KL divergence geometry (information geometry).

**K-FAC (Kronecker-Factored Approximate Curvature).** Martens & Grosse (2015) approximate the Fisher information matrix for neural networks as a block-diagonal matrix, where each block factorizes as a Kronecker product:

$$F \approx \hat{F} = \text{block-diag}(A_1 \otimes G_1, A_2 \otimes G_2, \ldots)$$

where $A_l = \mathbb{E}[\mathbf{a}_{l-1}\mathbf{a}_{l-1}^\top]$ (input activation covariance) and $G_l = \mathbb{E}[\delta_l\delta_l^\top]$ (pre-activation gradient covariance) for layer $l$. Both $A_l$ and $G_l$ are PSD (covariance matrices). The Kronecker product $A_l \otimes G_l$ is PSD, and its inverse is $(A_l \otimes G_l)^{-1} = A_l^{-1} \otimes G_l^{-1}$, computable cheaply via Cholesky of each factor separately.

### 10.3 Gaussian Process Regression

Gaussian process regression is the Bayesian non-parametric regression method that uses PD kernel matrices as the core computational object.

**Model.** Place a GP prior: $f \sim \mathcal{GP}(0, k)$ where $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a PD kernel. Observe noisy outputs $\mathbf{y} = f(\mathbf{X}) + \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$.

**Prediction.** The predictive distribution at new points $X_*$ is:

$$f_* | X_*, \mathbf{X}, \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)$$

where:
$$\boldsymbol{\mu}_* = K_{*n}(K_{nn} + \sigma^2I)^{-1}\mathbf{y}$$
$$\Sigma_* = K_{**} - K_{*n}(K_{nn} + \sigma^2I)^{-1}K_{n*}$$

and $K_{nn} = k(\mathbf{X}, \mathbf{X})$, $K_{*n} = k(X_*, \mathbf{X})$, $K_{**} = k(X_*, X_*)$.

**Computational core: Cholesky.** Factor $K_{nn} + \sigma^2 I = LL^\top$. Then:
- $\boldsymbol{\alpha} = L^\top \backslash (L \backslash \mathbf{y})$ (two triangular solves)
- $\boldsymbol{\mu}_* = K_{*n}\boldsymbol{\alpha}$ (matrix-vector product)
- $V = L \backslash K_{n*}$ (triangular solve, $n \times n_*$ system)
- $\Sigma_* = K_{**} - V^\top V$ (Schur complement form)
- $\log p(\mathbf{y}) = -\frac{1}{2}\mathbf{y}^\top\boldsymbol{\alpha} - \sum_i\log L_{ii} - \frac{n}{2}\log(2\pi)$

The entire GP regression computation is $O(n^3)$ via Cholesky — the classic bottleneck for large datasets, motivating sparse GP approximations (inducing points, Nyström, SVGP).

### 10.4 Hessian and Loss Landscape Sharpness

**Second-order characterization of minima.** At a critical point $\nabla\mathcal{L}(\boldsymbol{\theta}^*) = 0$ of a smooth loss $\mathcal{L}$:

- $\nabla^2\mathcal{L}(\boldsymbol{\theta}^*) \succ 0$: strict local minimum (the loss bowl is strictly convex)
- $\nabla^2\mathcal{L}(\boldsymbol{\theta}^*) \succeq 0$: local minimum or saddle (Hessian is PSD)
- $\nabla^2\mathcal{L}(\boldsymbol{\theta}^*)$ indefinite: saddle point

**Sharpness.** The **sharpness** of a minimum is $\lambda_{\max}(\nabla^2\mathcal{L}(\boldsymbol{\theta}^*))$ — the largest eigenvalue of the Hessian. Flat minima (small sharpness) generalize better than sharp minima: a small perturbation $\boldsymbol{\theta}^* + \boldsymbol{\delta}$ with $\|\boldsymbol{\delta}\| \leq \rho$ changes the loss by at most $\rho^2 \lambda_{\max}(\nabla^2\mathcal{L})$ (by Taylor expansion and the PSD bound $\boldsymbol{\delta}^\top H\boldsymbol{\delta} \leq \lambda_{\max}\|\boldsymbol{\delta}\|^2$).

**SAM (Sharpness-Aware Minimization).** Foret et al. (2021) propose:

$$\min_{\boldsymbol{\theta}} \max_{\|\boldsymbol{\delta}\|\leq\rho} \mathcal{L}(\boldsymbol{\theta}+\boldsymbol{\delta}).$$

The inner max finds the worst-case perturbation (solved approximately as $\boldsymbol{\delta}^* = \rho\, \nabla\mathcal{L}/\|\nabla\mathcal{L}\|$). SAM is a first-order approximation to minimizing sharpness and is reported to improve generalization on image and language tasks.

**Gauss-Newton and PSD Hessian approximations.** The true Hessian $\nabla^2\mathcal{L}$ may be indefinite during training (early stages, overparameterized models). The Gauss-Newton matrix $G = J^\top J$ (where $J$ is the Jacobian of the predictions) is always PSD and is often a better preconditioner. K-FAC uses the Gauss-Newton approximation.

### 10.5 Reparameterization Trick in VAEs

**The variational autoencoder (VAE)** requires sampling from a distribution $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \Sigma_\phi(\mathbf{x}))$ where $\boldsymbol{\mu}_\phi$ and $\Sigma_\phi$ are outputs of an encoder neural network, and differentiating through the sampling process.

**The reparameterization trick.** Instead of sampling $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ directly (which blocks gradient flow), write:

$$\mathbf{z} = \boldsymbol{\mu} + L\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

where $L = \text{chol}(\Sigma)$. Now $\mathbf{z}$ is a deterministic function of $(\boldsymbol{\mu}, L, \boldsymbol{\epsilon})$, and gradients can flow through $\boldsymbol{\mu}$ and $L$.

**Diagonal VAE (standard).** Most VAE implementations use diagonal covariance: $\Sigma = \text{diag}(\exp(\mathbf{s}))$, so $L = \text{diag}(\exp(\mathbf{s}/2))$ and $\mathbf{z} = \boldsymbol{\mu} + \exp(\mathbf{s}/2) \odot \boldsymbol{\epsilon}$.

**Full covariance VAE.** For a full covariance Cholesky parameterization, the encoder outputs:
1. Mean $\boldsymbol{\mu} \in \mathbb{R}^d$
2. Lower triangular $L \in \mathbb{R}^{d \times d}$ with positive diagonal (e.g., $L_{ii} = \text{softplus}(\tilde{L}_{ii})$, off-diagonal unconstrained)

Then $\Sigma = LL^\top$ and $\mathbf{z} = \boldsymbol{\mu} + L\boldsymbol{\epsilon}$. The gradient through $L$ back to the encoder parameters flows via:

$$\frac{\partial \mathbf{z}}{\partial L} = \frac{\partial(L\boldsymbol{\epsilon})}{\partial L} = \boldsymbol{\epsilon} \otimes I_d$$

(the outer product of the sampled noise $\boldsymbol{\epsilon}$ with the identity).

**For normalizing flows.** More expressive VAE variants use normalizing flows for the posterior $q_\phi(\mathbf{z}|\mathbf{x})$. The flow is a sequence of invertible maps; the log-det Jacobian of each map must be computed efficiently. Triangular Jacobians (e.g., masked autoregressive flow / IAF) achieve $O(d)$ log-det computation.


---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---------|----------------|-----|
| 1 | Checking only diagonal entries to test PD | Positive diagonal is necessary but not sufficient. $A = \begin{pmatrix}1 & 2 \\ 2 & 1\end{pmatrix}$ has positive diagonal but is indefinite ($\det = -3$). | Use Sylvester's criterion (all leading minors > 0) or attempt Cholesky. |
| 2 | Concluding $A \succ 0$ from $\det A > 0$ alone | $\det > 0$ is necessary but not sufficient. $\begin{pmatrix}-1 & 0 \\ 0 & -2\end{pmatrix}$ has $\det = 2 > 0$ but is negative definite. | Check all leading principal minors, not just the full determinant. |
| 3 | Assuming $A^\top A \succ 0$ for any $A$ | $A^\top A \succeq 0$ always, but $A^\top A \succ 0$ iff $A$ has full column rank. If $A$ has a null vector ($A\mathbf{v} = 0$), then $\mathbf{v}^\top A^\top A \mathbf{v} = 0$. | Verify $\text{rank}(A) = \text{number of columns}$. |
| 4 | Confusing the Cholesky factor $L$ with the PSD square root $A^{1/2}$ | $L$ is lower triangular; $A^{1/2}$ is symmetric. Both satisfy "squared = $A$" but in different senses ($LL^\top = A$ vs $(A^{1/2})^2 = A$). They are equal only when $A$ is diagonal. | Use $L$ for solving/sampling; use $A^{1/2}$ for Mahalanobis/whitening. |
| 5 | Using `np.log(np.linalg.det(A))` for large $A$ | `np.linalg.det` can overflow/underflow for large $n$ (products of many numbers). | Use Cholesky: `2 * np.sum(np.log(np.diag(np.linalg.cholesky(A))))`. |
| 6 | Forgetting Sylvester's criterion does NOT characterize PSD | Sylvester requires all leading principal minors > 0, which gives PD not PSD. For PSD, you need all principal minors (not just leading ones) $\geq 0$ — a much larger set. | For PSD testing, use eigenvalues or attempt pivoted Cholesky. |
| 7 | Inverting the Loewner order incorrectly | Students often think $A \succeq B$ implies $A^{-1} \succeq B^{-1}$. The correct fact is $A \succeq B \succ 0 \Rightarrow B^{-1} \succeq A^{-1}$ (inversion reverses the order). | Remember: larger matrix → smaller inverse (as for positive scalars: $a \geq b > 0 \Rightarrow 1/a \leq 1/b$). |
| 8 | Adding jitter $\epsilon I$ without checking scale | If $\|A\|_2 \approx 10^6$ and you add $\epsilon = 10^{-6}$, the relative jitter is $10^{-12}$, effectively zero numerically. | Set jitter proportional to the scale: $\epsilon = \delta \cdot \text{tr}(A)/n$ for some small $\delta$. |
| 9 | Assuming the kernel matrix stays PD after operations | Sum, product, and pointwise operations on PD kernels may not preserve PD structure. E.g., $k_1(\mathbf{x},\mathbf{z}) - k_2(\mathbf{x},\mathbf{z})$ may not be PD. | Verify the Gram matrix or use closure properties: sum/product/exponentiation of PD kernels are PD. |
| 10 | Computing Schur complement with singular $A$ | The Schur complement $D - CA^{-1}B$ requires $A$ invertible. If $A$ is only PSD (rank-deficient), $A^{-1}$ does not exist. | Use the Moore-Penrose pseudoinverse: $S = D - CA^\dagger B$, though the PD characterization no longer holds as stated. |
| 11 | Treating "positive definite" as a non-symmetric matrix property | PD is only defined for symmetric matrices. An arbitrary matrix $A$ with all positive eigenvalues may not have a positive quadratic form (complex eigenvalues, non-symmetric). | Always symmetrize: replace $A$ with $(A + A^\top)/2$ before testing PD. |
| 12 | Thinking PD constraint is automatically satisfied by a NN output | A neural network outputting $n(n+1)/2$ numbers does not automatically produce a valid lower triangular $L$ with positive diagonal. | Enforce: use softplus on the diagonal entries; leave off-diagonal unconstrained. Then $\Sigma = LL^\top$ is guaranteed PD. |

---

## 12. Exercises

**Exercise 1 ★ — Verify PD Axioms for $A^\top A + \lambda I$**

Let $A \in \mathbb{R}^{m \times n}$ with $m \geq n$ and $\lambda > 0$. Define $B = A^\top A + \lambda I$.

(a) Prove $B \succ 0$ using the definition $\mathbf{x}^\top B\mathbf{x} > 0$ for $\mathbf{x} \neq \mathbf{0}$.

(b) What is the smallest eigenvalue of $B$? Express it in terms of the singular values of $A$ and $\lambda$.

(c) What happens as $\lambda \to 0$? Under what condition on $A$ does $A^\top A$ itself become PD?

(d) Show $B^{-1} \preceq \frac{1}{\lambda}I$ using the Loewner order.

(e) **AI connection:** Identify $B = A^\top A + \lambda I$ as the normal equations matrix for ridge regression. Why does $\lambda > 0$ guarantee $B \succ 0$ regardless of the rank of $A$?

---

**Exercise 2 ★ — Implement Cholesky from Scratch**

(a) Implement the Cholesky algorithm (Algorithm 4.2) in pure NumPy without using `np.linalg.cholesky`. Your function should return the lower triangular $L$ satisfying $A = LL^\top$, or raise a `ValueError` if $A$ is not PD.

(b) Test on $A = \begin{pmatrix}4 & 12 & -16 \\ 12 & 37 & -43 \\ -16 & -43 & 98\end{pmatrix}$ and verify $LL^\top = A$.

(c) Compare the result with `np.linalg.cholesky` — they should agree to machine precision.

(d) Test what happens when you apply your function to an indefinite matrix $A = \begin{pmatrix}1 & 3 \\ 3 & 1\end{pmatrix}$.

(e) **Efficiency:** What is the flop count of your implementation for $n \times n$ input? Compare to LU factorization.

---

**Exercise 3 ★ — Sylvester's Criterion**

Let $A(\alpha) = \begin{pmatrix}1 & \alpha & 0 \\ \alpha & 2 & \alpha \\ 0 & \alpha & 4\end{pmatrix}$.

(a) Compute the three leading principal minors $\Delta_1, \Delta_2, \Delta_3$ as functions of $\alpha$.

(b) Use Sylvester's criterion to find all values of $\alpha \in \mathbb{R}$ for which $A(\alpha) \succ 0$.

(c) At the boundary values of $\alpha$ (where $A$ transitions from PD to indefinite), what is the nullspace of $A$? Interpret this geometrically.

(d) Verify your answer computationally by plotting the smallest eigenvalue of $A(\alpha)$ as a function of $\alpha$.

---

**Exercise 4 ★★ — LDLᵀ Factorization**

(a) Implement the LDLᵀ algorithm (Algorithm 4.4) in NumPy. Your function returns unit lower triangular $L$ and diagonal $D$ such that $A = LDL^\top$.

(b) Test on $A = \begin{pmatrix}4 & 2 & 1 \\ 2 & 5 & 3 \\ 1 & 3 & 6\end{pmatrix}$. Verify $LDL^\top = A$.

(c) Using your factorization, solve $A\mathbf{x} = \mathbf{b}$ for $\mathbf{b} = (1, 2, 3)^\top$ by: (i) solve $L\mathbf{y} = \mathbf{b}$, (ii) solve $D\mathbf{w} = \mathbf{y}$, (iii) solve $L^\top\mathbf{x} = \mathbf{w}$.

(d) Explain why LDLᵀ avoids the numerical issues of Cholesky when $A$ is indefinite.

(e) **AI connection:** Why do second-order optimizers (like KFAC) prefer working with $D$ (the pivots) rather than $\sqrt{D}$ (the Cholesky diagonal)?

---

**Exercise 5 ★★ — Schur Complement and Block PD Test**

Let $M = \begin{pmatrix}A & B \\ B^\top & D\end{pmatrix}$ where:
$$A = \begin{pmatrix}4 & 1 \\ 1 & 3\end{pmatrix}, \quad B = \begin{pmatrix}1 \\ 2\end{pmatrix}, \quad D = \begin{pmatrix}5\end{pmatrix}.$$

(a) Compute the Schur complement $S = D - B^\top A^{-1}B$.

(b) Use Theorem 6.2 to determine whether $M \succ 0$.

(c) Compute $\det M$ using $\det M = \det A \cdot \det S$ and verify numerically.

(d) Use the block inverse formula to compute $M^{-1}$. Verify $MM^{-1} = I$.

(e) **Gaussian conditioning:** Interpret $M$ as a joint covariance matrix $\Sigma = \begin{pmatrix}\Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{pmatrix}$. What is the conditional covariance $\Sigma_{1|2}$? Interpret: does conditioning reduce uncertainty?

---

**Exercise 6 ★★ — Log-Det via Cholesky and Gradient Verification**

(a) Compute $\log\det A$ for $A = \begin{pmatrix}3 & 1 & 0 \\ 1 & 4 & 2 \\ 0 & 2 & 5\end{pmatrix}$ using: (i) direct eigenvalues, (ii) Cholesky diagonal formula, (iii) `np.log(np.linalg.det(A))`. Verify all three agree.

(b) Implement a function `log_det(A)` using Cholesky. Compare speed vs eigenvalue method for $n = 100, 500, 1000$ random PD matrices.

(c) Numerically verify the gradient formula $\nabla_A \log\det A = A^{-1}$ using finite differences: $\frac{f(A + hE_{ij}) - f(A)}{h} \approx (A^{-1})_{ij}$.

(d) For the Gaussian log-likelihood: given 50 samples from $\mathcal{N}(\mathbf{0}, \Sigma_{\text{true}})$ where $\Sigma_{\text{true}} = A$ from part (a), compute the log-likelihood $\ell(\Sigma) = -\frac{n}{2}\log\det\Sigma - \frac{1}{2}\text{tr}(\Sigma^{-1}S)$ where $S = \frac{1}{n}\sum_i\mathbf{x}_i\mathbf{x}_i^\top$ is the sample covariance.

---

**Exercise 7 ★★★ — Gram Matrix / Kernel Matrix and Mercer Check**

(a) Given 5 points $\mathbf{x}_1, \ldots, \mathbf{x}_5 \in \mathbb{R}^3$ (design your own interesting configuration), compute the Gram matrix $G = XX^\top$.

(b) Verify $G \succeq 0$ by checking eigenvalues. Explain why $\text{rank}(G) \leq 3$.

(c) Implement and test three kernels on these points: (i) linear $k(\mathbf{x},\mathbf{z}) = \mathbf{x}^\top\mathbf{z}$, (ii) RBF $k(\mathbf{x},\mathbf{z}) = \exp(-\|\mathbf{x}-\mathbf{z}\|^2/2)$, (iii) polynomial $k(\mathbf{x},\mathbf{z}) = (\mathbf{x}^\top\mathbf{z}+1)^2$. For each, compute the $5 \times 5$ kernel matrix and verify PSD.

(d) Now try the "not-a-kernel" $k(\mathbf{x},\mathbf{z}) = \exp(-\|\mathbf{x}-\mathbf{z}\|)$ (Laplacian kernel IS a valid kernel) vs $k(\mathbf{x},\mathbf{z}) = \cos(\|\mathbf{x}-\mathbf{z}\|)$ (NOT always a valid kernel). Determine empirically whether each produces a PSD Gram matrix.

(e) **AI connection:** Implement a simple kernel regression: given training inputs $\mathbf{X}$ and outputs $\mathbf{y}$, compute predictions at test points $X_*$ using $\hat{\mathbf{y}}_* = K_{*n}(K_{nn} + 0.1 I)^{-1}\mathbf{y}$ (Nadaraya-Watson/GPR predictive mean).

---

**Exercise 8 ★★★ — Cholesky Reparameterization for VAE Sampling**

(a) Implement a function `sample_mvn(mu, L, n_samples)` that samples $n$ vectors from $\mathcal{N}(\boldsymbol{\mu}, LL^\top)$ using the reparameterization trick $\mathbf{z} = \boldsymbol{\mu} + L\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},I)$.

(b) Let $\boldsymbol{\mu} = (1, 2)^\top$ and $\Sigma = \begin{pmatrix}4 & 2 \\ 2 & 3\end{pmatrix}$. Compute $L = \text{chol}(\Sigma)$. Sample 1000 points, compute the sample mean and covariance, and verify they match $\boldsymbol{\mu}$ and $\Sigma$ (up to sampling noise).

(c) Simulate a VAE encoder: the encoder takes a scalar input $x$ and outputs $\boldsymbol{\mu}_\phi(x) = Wx + b$ and a lower triangular $L_\phi(x)$ (a diagonal $L$ for simplicity). Sample $\mathbf{z}$ and pass it through a simple decoder $\hat{x} = c^\top\mathbf{z}$. Compute the ELBO:

$$\text{ELBO} = \mathbb{E}_{\mathbf{z}}[\log p(x|\mathbf{z})] - D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}, \Sigma) \| \mathcal{N}(\mathbf{0}, I)).$$

where $D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu}, \Sigma) \| \mathcal{N}(\mathbf{0},I)) = \frac{1}{2}[\text{tr}(\Sigma) + \boldsymbol{\mu}^\top\boldsymbol{\mu} - d - \log\det\Sigma]$.

(d) Verify the KL formula using both the analytical expression and Monte Carlo estimation from samples.

(e) **Gradient check:** Confirm that the reparameterization trick allows gradients to flow through $\boldsymbol{\mu}$ and $L$ (not $\boldsymbol{\epsilon}$) by computing $\partial\text{ELBO}/\partial\boldsymbol{\mu}$ analytically and comparing with finite differences.

---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | Concrete AI/LLM Role | Where / Method |
|---------|---------------------|----------------|
| Positive definite covariance | Valid multivariate Gaussian distributions | VAEs (Kingma & Welling, 2013); diffusion models (DDPM, score matching) |
| Cholesky factorization | Reparameterization trick for differentiable sampling | All variational inference; NUTS-HMC sampler in Pyro, NumPyro |
| LDLᵀ factorization | Stable Hessian factorization in trust-region optimizers | L-BFGS, trust-region Newton; scipy.optimize |
| Schur complement | Conditional Gaussian inference; GP prediction | GPyTorch, GPflow; Bayesian neural networks |
| Matrix inversion lemma (Woodbury) | Low-rank updates without full matrix inversion | LoRA weight merging; Kalman filter; GP with inducing points |
| Log-determinant | Normalizing constant in Gaussian log-likelihood | GP hyperparameter optimization; normalizing flows (RealNVP, Glow) |
| $\nabla\log\det A = A^{-1}$ | Gradient of log-likelihood w.r.t. covariance | Fitting GP kernel parameters; structured covariance learning |
| Gram matrix / kernel matrix | Kernel methods, self-attention similarity | SVM training; transformer attention; Performer / FAVOR+ |
| Fisher information matrix | Riemannian metric on parameter space | K-FAC optimizer (Google Brain, 2015); natural gradient variational inference |
| PSD Hessian at minima | Second-order optimality condition | SAM (flatness-seeking optimizer); loss landscape analysis |
| Sharpness $\lambda_{\max}(H)$ | Generalization proxy; flat minima theory | SAM, ASAM, Stein Self-Repulsive (2024) |
| PSD cone / SDP | Constrained optimization; certified robustness | $\alpha$-CROWN, $\beta$-CROWN neural network verification; fairness constraints |
| Mercer's theorem | Theoretical foundation for kernel machines | SVMs; kernel PCA; connections to infinite-width NTK theory |
| Cholesky reparameterization | Parameterizing full covariance in deep models | Deep covariance estimation; Matrix-variate distributions in meta-learning |
| Log-det stochastic estimator | Scalable GP training on millions of points | GPyTorch (Gardner et al., 2018); SVGP; stochastic trace estimation |

---

## 14. Conceptual Bridge

### Looking Back

This section builds on a foundation laid across the previous six sections of Advanced Linear Algebra. From eigenvalues and eigenvectors (§01), we borrow the spectral theorem — the fact that symmetric matrices diagonalize orthogonally — which provides the cleanest proof that positive definiteness is equivalent to positive eigenvalues. From SVD (§02), we inherit the language of singular values and the connection between the spectral norm, condition number, and the stability of Cholesky. The orthogonality theory from §05 underpins the proof that $Q\Lambda Q^\top$ is symmetric (the spectral theorem) and that the PSD square root is well-defined and unique. Matrix norms from §06 give precise bounds on how a PD matrix behaves numerically: the condition number $\kappa(A) = \lambda_{\max}/\lambda_{\min}$ measures how far $A$ is from singular, and ill-conditioned PD matrices lead to numerical difficulties in Cholesky.

Most fundamentally, positive definite matrices are the matrices that have all four of the desirable properties we want from a symmetric matrix: uniquely solvable linear systems (invertible), well-defined quadratic energy (positive), computable square root (Cholesky), and monotone response to matrix operations (Loewner order). Understanding this quadrant of symmetry × positivity is the analytical key to all probabilistic machine learning.

### Looking Forward

Positive definite matrices are the entry point to the next major development: §08 (Matrix Decompositions) uses the Cholesky factorization developed here as one member of a family that includes LU (general matrices) and QR (orthogonal factorization). The Cholesky decomposition studied here in full generality is previewed (only briefly) in §08 — by the time students reach §08, the Cholesky algorithm should feel familiar.

The deeper application of positive definiteness comes later in the curriculum. In Chapter 6 (Probability Theory), covariance matrices $\Sigma \succeq 0$ appear in every multivariate distribution. In Chapter 8 (Optimization), the second-order conditions for minima require $\nabla^2\mathcal{L} \succeq 0$, and convexity of a function is equivalent to its Hessian being PSD everywhere. In Chapter 12 (Functional Analysis), Mercer's theorem and RKHS theory extend PSD kernels to infinite-dimensional inner product spaces. The PSD cone and semidefinite programming previewed in §9 appear extensively in Chapter 8 (Optimization) and in the robotics/control applications in later chapters.

### Curriculum Position

```
POSITIVE DEFINITE MATRICES IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  Chapter 2: Linear Algebra Basics
  ├── §01 Vectors and Spaces        → inner products, quadratic forms
  ├── §04 Determinants              → Sylvester's criterion
  └── §06 Vector Spaces             → subspaces, null space
  
  Chapter 3: Advanced Linear Algebra
  ├── §01 Eigenvalues         → spectral characterization of PD
  ├── §02 SVD                 → singular values, condition number
  ├── §05 Orthogonality       → spectral theorem, square root
  ├── §06 Matrix Norms        → condition number, jitter scale
  ├── §07 Positive Definite   ◄══ YOU ARE HERE
  │     │   Cholesky, LDLᵀ, Schur, log-det, Gram, SDP
  └── §08 Matrix Decompositions  → LU, QR, brief Cholesky recap
  
  Downstream:
  ├── Ch. 6 Probability: Gaussian covariance Σ ≻ 0
  ├── Ch. 8 Optimization: Hessian ≻ 0 at local minima; SDP
  ├── Ch. 12 Functional Analysis: Mercer, RKHS
  └── Ch. 13 ML Math: K-FAC, VAE, GP, normalizing flows

════════════════════════════════════════════════════════════════════════
```

The central insight to carry forward: **positive definiteness is not a restriction but a characterization**. A matrix that fails to be PD is not "almost OK" — it is categorically different, with no unique minimum, no valid probability interpretation, no Cholesky factorization. The condition $A \succ 0$ is the boundary between well-posedness and ill-posedness in a wide range of mathematical problems, and recognizing that boundary — and the tools to work with it (Cholesky, Schur, log-det) — is an essential skill for the working ML practitioner.


---

## Appendix A: Closure Properties and Algebraic Operations

### A.1 Preserving Positive Definiteness Under Operations

Understanding which operations preserve PD/PSD structure is practically important — it tells you when a composed matrix is guaranteed to be PD without recomputing.

**Direct results:**

| Operation | Inputs | Result | Condition |
|-----------|--------|--------|-----------|
| $\alpha A$ | $A \succ 0$, $\alpha > 0$ | $\succ 0$ | Always |
| $A + B$ | $A, B \succ 0$ | $\succ 0$ | Always |
| $A + B$ | $A \succ 0$, $B \succeq 0$ | $\succ 0$ | Always |
| $B^\top A B$ | $A \succ 0$, $B$ invertible | $\succ 0$ | Always |
| $B^\top A B$ | $A \succ 0$, $B$ full column rank | $\succ 0$ | Always |
| $A^{-1}$ | $A \succ 0$ | $\succ 0$ | Always |
| $A^k$ | $A \succ 0$, $k \in \mathbb{Z}$ | $\succ 0$ | Always |
| $A^t$ | $A \succ 0$, $t \in \mathbb{R}$ | $\succ 0$ | Matrix power via exp(t log A) |
| $AB$ | $A, B \succ 0$ | **NOT necessarily** $\succeq 0$ | Only if $AB = BA$ |
| $A \otimes B$ | $A, B \succ 0$ | $\succ 0$ | Always (Kronecker) |
| $A \otimes B$ | $A, B \succeq 0$ | $\succeq 0$ | Always (Kronecker) |
| $A \odot B$ (Hadamard) | $A, B \succeq 0$ | $\succeq 0$ | Schur product theorem |

**The Schur product theorem.** The Hadamard (element-wise) product of two PSD matrices is PSD. This is non-obvious! It is used in kernel methods: if $k_1$ and $k_2$ are valid kernels, their pointwise product $k_1(\mathbf{x},\mathbf{z}) \cdot k_2(\mathbf{x},\mathbf{z})$ is also a valid kernel.

**Proof of Schur product theorem.** If $A, B \succeq 0$, write $A = \sum_i \mathbf{u}_i\mathbf{u}_i^\top$ and $B = \sum_j \mathbf{v}_j\mathbf{v}_j^\top$ (Gram representations). Then $(A \odot B)_{kl} = A_{kl}B_{kl}$, which can be expressed as the Gram matrix of vectors $\mathbf{u}_i \otimes \mathbf{v}_j$ (Kronecker products of the generating vectors). Gram matrices are PSD. $\square$

**Why $AB$ is not always PSD.** Consider $A = \begin{pmatrix}1 & 0 \\ 0 & 2\end{pmatrix}$ and $B = \begin{pmatrix}2 & 1 \\ 1 & 1\end{pmatrix}$. Both are PD, but $AB = \begin{pmatrix}2 & 1 \\ 2 & 2\end{pmatrix}$, which is not symmetric (hence not PSD). The product of symmetric matrices is symmetric iff they commute.

**For AI:** Kronecker product structure is central to K-FAC: $F \approx A_l \otimes G_l$ is PSD since $A_l, G_l \succeq 0$. The Schur product theorem ensures that combining PD kernels (feature interactions) produces valid kernels.

### A.2 Principal Submatrix Inheritance

**Theorem A.1.** If $A \succ 0$, every principal submatrix is $\succ 0$. If $A \succeq 0$, every principal submatrix is $\succeq 0$.

A **principal submatrix** of $A$ is obtained by deleting rows and columns with the same index set: $A_S = A[S, S]$ for $S \subseteq \{1,\ldots,n\}$.

**Proof.** For any $\mathbf{y} \in \mathbb{R}^{|S|}$, embed in $\mathbb{R}^n$ via $\mathbf{x}_i = y_i$ for $i \in S$ and $\mathbf{x}_i = 0$ for $i \notin S$. Then $\mathbf{y}^\top A_S \mathbf{y} = \mathbf{x}^\top A \mathbf{x} > 0$ for $\mathbf{y} \neq 0$ (since $\mathbf{x} \neq 0$). $\square$

**Consequence.** Every diagonal entry $A_{ii} > 0$ for $A \succ 0$ (take $S = \{i\}$). Every $2 \times 2$ principal submatrix is PD, so $A_{ii}A_{jj} > A_{ij}^2$ for all $i \neq j$ (Cauchy-Schwarz-type bound for matrix entries).

### A.3 Indefinite Matrices and Saddle Points

For completeness: an **indefinite** symmetric matrix $A$ has both positive and negative eigenvalues. The corresponding quadratic form takes positive values in some directions and negative values in others. The level sets are hyperboloids.

**Connection to saddle points in optimization.** A critical point $\nabla\mathcal{L}(\boldsymbol{\theta}^*) = 0$ with indefinite Hessian $\nabla^2\mathcal{L}(\boldsymbol{\theta}^*)$ is a saddle point — not a local minimum. In neural network optimization:

- Gradient descent near a saddle point may slow down dramatically (gradient is small, Hessian is indefinite)
- The negative eigenvalue directions are "escape directions" — following them reduces the loss
- Stochastic gradient descent (SGD) with noise "escapes" saddles; deterministic gradient descent can get stuck

**Sylvester's law of inertia.** For a symmetric matrix $A$, the **signature** $(p, q, r)$ (number of positive, negative, zero eigenvalues) is invariant under congruence transformations $A \mapsto B^\top A B$ (for invertible $B$). This means no invertible change of basis can turn a saddle into a minimum — the number of descent directions is a geometric invariant of the quadratic form.

---

## Appendix B: Numerical Methods and Implementation

### B.1 Cholesky for Linear System Solving

A primary application of Cholesky is solving $A\mathbf{x} = \mathbf{b}$ for PD $A$.

**Algorithm:**

1. **Factor:** $A = LL^\top$ (Cholesky, $O(n^3/3)$)
2. **Forward solve:** $L\mathbf{y} = \mathbf{b}$ (substitute from top, $O(n^2)$)
3. **Backward solve:** $L^\top\mathbf{x} = \mathbf{y}$ (substitute from bottom, $O(n^2)$)

For multiple right-hand sides $B = [\mathbf{b}_1|\cdots|\mathbf{b}_k]$: factor once, solve $k$ times at cost $O(n^3/3 + kn^2)$ total.

**Numerical example:** Solve $\begin{pmatrix}4 & 2 \\ 2 & 3\end{pmatrix}\mathbf{x} = \begin{pmatrix}8 \\ 7\end{pmatrix}$.

Cholesky: $L = \begin{pmatrix}2 & 0 \\ 1 & \sqrt{2}\end{pmatrix}$ (since $L_{11}=2$, $L_{21}=1$, $L_{22}=\sqrt{3-1}=\sqrt{2}$).

Forward: $2y_1 = 8 \Rightarrow y_1 = 4$; $y_1 + \sqrt{2}y_2 = 7 \Rightarrow y_2 = (7-4)/\sqrt{2} = 3/\sqrt{2}$.

Backward: $\sqrt{2}x_2 = 3/\sqrt{2} \Rightarrow x_2 = 3/2$; $2x_1 + x_2 = 4 \Rightarrow x_1 = (4 - 3/2)/2 = 5/4$.

Verify: $\begin{pmatrix}4&2\\2&3\end{pmatrix}\begin{pmatrix}5/4\\3/2\end{pmatrix} = \begin{pmatrix}5+3\\5/2+9/2\end{pmatrix} = \begin{pmatrix}8\\7\end{pmatrix}$. ✓

### B.2 Cholesky Update and Downdate

When $A$ changes by a rank-1 update $A \leftarrow A + \mathbf{v}\mathbf{v}^\top$ (or downdate $A \leftarrow A - \mathbf{v}\mathbf{v}^\top$), it is wasteful to recompute the full Cholesky factorization. **Rank-1 Cholesky update** algorithms (LINPACK's `dchud`) update $L$ in $O(n^2)$ time.

**Algorithm (rank-1 update, Gill-Golub-Murray-Saunders style):**

Given $A = LL^\top$, compute $L'$ such that $L'L'^\top = A + \mathbf{v}\mathbf{v}^\top$:

```
x = L \ v           (forward solve: O(n²))
for k = 1 to n:
    r = sqrt(L[k,k]² + x[k]²)
    c = r / L[k,k]
    s = x[k] / L[k,k]
    L[k,k] = r
    L[k+1:n, k] = (L[k+1:n,k] + s*x[k+1:n]) / c
    x[k+1:n]    = c*x[k+1:n] - s*L[k+1:n,k]
```

This uses Givens rotations (→ [§05: Orthogonality](../05-Orthogonality-and-Orthonormality/notes.md)) to update $L$ in place.

**For AI:** Online learning algorithms that add one data point at a time use rank-1 Cholesky updates to maintain the Cholesky factorization of the Gram matrix. Sequential Bayesian updating (online GP regression) uses rank-1 Cholesky updates at cost $O(n^2)$ per update instead of $O(n^3)$ full refactorization.

### B.3 Condition Number and Numerical Stability of Cholesky

**Condition number.** For $A \succ 0$, the condition number $\kappa(A) = \lambda_{\max}/\lambda_{\min}$. The relative error in solving $A\mathbf{x} = \mathbf{b}$ satisfies:

$$\frac{\|\hat{\mathbf{x}} - \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \cdot \epsilon_{\text{mach}}.$$

For Cholesky specifically: the backward error bound on the Cholesky factor is $\|A - \hat{L}\hat{L}^\top\| \leq c_n\epsilon_{\text{mach}}\|A\|$. This means Cholesky is **unconditionally backward stable** for PD matrices. In contrast, LU without pivoting can have backward errors proportional to $\|L\|\|U\|$, which can be exponentially large.

**Ill-conditioning warning.** For large $\kappa(A)$:
- The log-det computation $2\sum\log L_{ii}$ involves some very small $L_{ii}$ (near-zero for near-singular dimensions) — those log terms dominate and may lose digits
- Linear solves $A^{-1}\mathbf{b}$ lose approximately $\log_{10}\kappa(A)$ digits of accuracy
- Cholesky itself remains stable; the error comes from the ill-conditioning of the problem, not the algorithm

**Practical rule:** If $\kappa(A) \approx 10^k$ and your floating-point arithmetic has $\approx 15$ decimal digits (double precision), you lose $k$ digits in the solution. For $k \geq 12$, the solution has no reliable digits.

### B.4 Sparse Cholesky

For sparse PD matrices (many zero entries, arising in graph-structured models, FEM discretizations, and spatial statistics), general Cholesky is wasteful because it may "fill in" zeros and produce dense $L$.

**The fill-in problem.** Even if $A$ is sparse, $L$ may be dense. The fill-in pattern depends on the sparsity structure of $A$ and the ordering of variables.

**Reordering algorithms.** The **approximate minimum degree (AMD)** and **nested dissection** algorithms reorder the variables (permute $A \leftarrow PAP^\top$ for a permutation matrix $P$) to minimize fill-in. The standard library for sparse Cholesky is SuiteSparse (CHOLMOD).

**For AI:** In Gaussian Markov random fields (GMRFs), the precision matrix (inverse covariance) is sparse. Sparse Cholesky enables efficient inference in spatial models, graph neural networks with Gaussian priors, and structured prediction. GPyTorch's KeOps and PyTorch Sparse backends exploit sparsity for large-scale GP approximations.

---

## Appendix C: Positive Definiteness in Complex Vector Spaces

### C.1 Hermitian Positive Definite Matrices

Over $\mathbb{C}$, positive definiteness generalizes to **Hermitian positive definite (HPD)** matrices.

**Definition.** $A \in \mathbb{C}^{n \times n}$ is Hermitian positive definite if:
- $A = A^*$ (where $A^* = \bar{A}^\top$ is the conjugate transpose)
- $\mathbf{x}^* A \mathbf{x} > 0$ for all $\mathbf{x} \in \mathbb{C}^n \setminus \{0\}$

(Note: $\mathbf{x}^* A \mathbf{x}$ is always real for Hermitian $A$, so the positivity condition makes sense.)

**Cholesky for HPD.** Every HPD matrix $A$ has a unique Cholesky factorization $A = LL^*$ where $L$ is lower triangular with positive real diagonal entries.

**For AI:** Complex-valued neural networks (used in signal processing, quantum computing simulation) use HPD covariance matrices. Radar signal processing, MRI reconstruction, and quantum chemistry all work in $\mathbb{C}^n$ with HPD matrices.


---

## Appendix D: Advanced Theory

### D.1 The Polar Decomposition and Positive Definite Factors

Every invertible matrix $A \in \mathbb{R}^{n \times n}$ has a unique **polar decomposition**:

$$A = QP$$

where $Q$ is orthogonal ($Q^\top Q = I$) and $P = \sqrt{A^\top A} \succ 0$ is the unique PD symmetric positive definite "right polar factor." This is the matrix analogue of the polar form $z = e^{i\theta}|z|$ for complex numbers.

**Existence and uniqueness.** Using SVD $A = U\Sigma V^\top$:
$$A = U\Sigma V^\top = (UV^\top)(V\Sigma V^\top) = QP$$
where $Q = UV^\top$ (orthogonal) and $P = V\Sigma V^\top \succ 0$ (PSD square root of $A^\top A$).

**Uniqueness of $P$:** $P = (A^\top A)^{1/2}$, which is the unique PSD square root (Theorem 5.1). Given $P$, $Q = AP^{-1}$ is uniquely determined.

**For AI:** The polar decomposition appears in:
- **Weight matrix orthogonalization:** in Muon optimizer (Kosson et al., 2024), gradient updates are "orthogonalized" using the polar factor $Q = AP^{-1}$, which projects the weight update onto the manifold of orthogonal matrices, maintaining stable singular value spectrum during training
- **Group equivariance:** equivariant neural networks (E(n)-equivariant GNNs) use orthogonal matrices as symmetry group elements; the polar decomposition provides a differentiable projection onto $O(n)$

**Computing the polar factor.** Newton iteration: $X_{k+1} = \frac{1}{2}(X_k + X_k^{-\top})$ converges to $Q$ starting from $X_0 = A$. Quadratic convergence once close to $Q$. Each step costs one matrix inversion.

### D.2 Positive Definite Functions and Bochner's Theorem

A function $k: \mathbb{R}^d \to \mathbb{R}$ is a **positive definite function** if for every $n$ and every set of points $\mathbf{x}_1,\ldots,\mathbf{x}_n \in \mathbb{R}^d$, the matrix $K_{ij} = k(\mathbf{x}_i - \mathbf{x}_j)$ is PSD.

**Bochner's theorem.** A continuous stationary kernel $k(\mathbf{x}-\mathbf{z})$ is a positive definite function on $\mathbb{R}^d$ if and only if it is the Fourier transform of a non-negative measure $\mu$ (spectral density):

$$k(\boldsymbol{\tau}) = \int_{\mathbb{R}^d} e^{i\boldsymbol{\omega}^\top\boldsymbol{\tau}} d\mu(\boldsymbol{\omega}).$$

**Implications for kernel design:**
- **RBF kernel:** $k(\boldsymbol{\tau}) = \exp(-\|\boldsymbol{\tau}\|^2/2\ell^2)$ is PD (its Fourier transform is a Gaussian, which is non-negative)
- **Matérn kernels:** PD for all valid smoothness parameters $\nu > 0$
- **Checking validity:** A kernel is valid iff its Fourier transform is non-negative — this is the ultimate test

**Random Fourier features (Rahimi & Recht, 2007).** Bochner's theorem gives a randomized approximation to stationary PD kernels:

$$k(\mathbf{x}-\mathbf{z}) \approx z(\mathbf{x})^\top z(\mathbf{z}), \quad z(\mathbf{x}) = \sqrt{2/D}[\cos(\boldsymbol{\omega}_1^\top\mathbf{x}+b_1),\ldots,\cos(\boldsymbol{\omega}_D^\top\mathbf{x}+b_D)]$$

where $\boldsymbol{\omega}_i \sim \mu$ (the spectral distribution) and $b_i \sim \text{Uniform}[0,2\pi]$. This approximates the PD kernel as a finite inner product, enabling scalable kernel methods without forming the full $n \times n$ kernel matrix. Used in Performer transformer attention (FAVOR+, Choromanski et al., 2020).

### D.3 The Determinant and Volume

**Geometric interpretation of $\det A$ for $A \succ 0$.**

For $A \succ 0$, the quadratic form $\mathbf{x}^\top A^{-1} \mathbf{x} \leq 1$ defines an ellipsoid $\mathcal{E}_A$ in $\mathbb{R}^n$. Its volume is:

$$\text{Vol}(\mathcal{E}_A) = \frac{\pi^{n/2}}{\Gamma(n/2+1)} \cdot (\det A)^{1/2}.$$

So $\det A$ measures the "volume" of the ellipsoid associated with $A$. Larger $\det A$ means a more "spread out" distribution.

**In the Gaussian context:** The normalizing constant of $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ is $(2\pi)^{n/2}(\det\Sigma)^{1/2}$. This is the volume of the "effective support" ellipsoid scaled by $(2\pi)^{n/2}$.

**D-optimal experimental design.** In Bayesian experimental design, the **D-optimal criterion** selects experiments $\mathbf{x}_1,\ldots,\mathbf{x}_n$ to maximize $\det(X^\top X + \sigma^{-2}I)$ — the determinant of the posterior precision matrix, which equals $1/\det(\Sigma_{\text{post}})$. Maximizing $\det(\text{posterior precision})$ minimizes the volume of the posterior confidence ellipsoid. This is an SDP with $X$ as the optimization variable.

**Maximum entropy Gaussians.** Among all distributions with mean $\boldsymbol{\mu}$ and covariance $\Sigma$, the maximum entropy distribution is $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$, with entropy $H = \frac{n}{2}(1+\log(2\pi)) + \frac{1}{2}\log\det\Sigma$. Maximizing entropy subject to a trace constraint $\text{tr}(\Sigma) \leq c$ gives $\Sigma = (c/n)I$ (isotropic Gaussian) — the most "uninformed" Gaussian with bounded total variance.

### D.4 Matrix Exponential and the PD Manifold

The set of PD matrices $\mathbb{S}_{++}^n$ is an **open smooth manifold** (in fact, a Riemannian manifold with the affine-invariant metric). This means concepts from differential geometry apply.

**The matrix exponential.** For any symmetric $S \in \mathbb{S}^n$, $e^S = Q e^\Lambda Q^\top \in \mathbb{S}_{++}^n$ (positive definite, since exponentials of real numbers are positive). The map $\exp: \mathbb{S}^n \to \mathbb{S}_{++}^n$ is a bijection — PD matrices are exactly exponentials of symmetric matrices.

**Log-Euclidean metric.** A simple Riemannian metric on $\mathbb{S}_{++}^n$ (used in diffusion tensor imaging) defines the "distance" between $A, B \succ 0$ as:

$$d_{\log}(A, B) = \|\log A - \log B\|_F.$$

This treats PD matrices as if they lived in a flat space via the log map. Computations (means, geodesics, interpolations) become Euclidean operations on $\log A$ and $\log B$.

**Affine-invariant metric (Riemannian).** The more geometrically natural metric:

$$d_{\text{AI}}(A, B) = \|\log(A^{-1/2}B A^{-1/2})\|_F = \left(\sum_{i=1}^n \log^2\lambda_i(A^{-1}B)\right)^{1/2}$$

is invariant under congruence $A \mapsto C^\top A C$ (affine-invariant), making it suitable for geometric statistics.

**For AI:** The SPD (symmetric positive definite) manifold appears in:
- **Diffusion tensor MRI:** each voxel has a $3 \times 3$ diffusion tensor $\in \mathbb{S}_{++}^3$; Riemannian means avoid non-PD artifacts
- **Covariance-based classifiers:** classifying neural activation covariance matrices using Riemannian geometry (SPDNet, 2017)
- **Transformer positional encodings with structured covariances:** attention with SPD constraints on the key-query metric tensors

### D.5 Inequalities for Positive Definite Matrices

Several fundamental inequalities apply specifically to PD matrices:

**Hadamard's inequality.** For $A \succ 0$:
$$\det A \leq \prod_{i=1}^n A_{ii}.$$
Equality iff $A$ is diagonal. Interpretation: the volume of the ellipsoid defined by $A$ is at most the product of the diagonal entries (the volumes along coordinate axes).

**Minkowski's inequality.** For $A, B \succ 0$:
$$(\det(A+B))^{1/n} \geq (\det A)^{1/n} + (\det B)^{1/n}.$$
This is the matrix analogue of $\sqrt[n]{a+b} \geq \sqrt[n]{a} + \sqrt[n]{b}$ for scalars (Minkowski). It follows from the concavity of $A \mapsto (\det A)^{1/n}$ on $\mathbb{S}_{++}^n$.

**Fan-Ky inequality.** For $A \succ 0$ and $k \leq n$:
$$\sum_{i=1}^k \log\lambda_i(A+B) \geq \sum_{i=1}^k \log\lambda_i(A) + \sum_{i=1}^k \log\lambda_i(B).$$
(The sum of the $k$ largest log-eigenvalues of $A+B$ is at least the sum from $A$ plus the sum from $B$.)

**Anderson's inequality.** For $A, B, C \succ 0$ with $C = A + B$:
$$C^{-1} \preceq A^{-1} + B^{-1}.$$
Interpretation: combining two experiments (adding their information matrices) gives more information than either alone, but the combined precision matrix is "sandwiched" by the sum of individual inverse covariances.

**For AI:** Hadamard's inequality provides a bound on the log-det: $\log\det A \leq \sum_i \log A_{ii} = \log\prod A_{ii}$, useful when diagonal entries are cheaply computed. Minkowski's inequality is used in information-geometric proofs about combining Gaussian priors in Bayesian learning.


---

## Appendix E: Extended Examples and Case Studies

### E.1 Case Study: Covariance Estimation in High Dimensions

One of the most practically important applications of PSD matrix theory is estimating covariance matrices from data when the number of features $p$ is comparable to or larger than the number of samples $n$.

**The problem.** Given $n$ samples $\mathbf{x}_1,\ldots,\mathbf{x}_n \in \mathbb{R}^p$ (zero mean for simplicity), the sample covariance is:

$$\hat{S} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}_i\mathbf{x}_i^\top = \frac{1}{n}X^\top X.$$

$\hat{S} \succeq 0$ always (it's a Gram matrix). But:
- If $n < p$: $\text{rank}(\hat{S}) \leq n < p$, so $\hat{S}$ is singular (not PD)
- Even for $n \geq p$: $\hat{S}$ can be ill-conditioned (extreme ratio of largest to smallest eigenvalue)

**The Marchenko-Pastur law.** For $n, p \to \infty$ with $n/p \to \gamma > 1$ and $\mathbf{x}_i \sim \mathcal{N}(0, I_p)$, the empirical eigenvalue distribution of $\hat{S}$ converges to the Marchenko-Pastur distribution:

$$f(\lambda) = \frac{\sqrt{(\lambda_+ - \lambda)(\lambda - \lambda_-)}}{2\pi\gamma\lambda}$$

for $\lambda \in [\lambda_-, \lambda_+]$, where $\lambda_\pm = (1 \pm 1/\sqrt{\gamma})^2$. This means even under the identity covariance, sample eigenvalues are spread over a wide range — the smallest eigenvalues of $\hat{S}$ are much smaller than 1 and the largest are much larger than 1.

**Ledoit-Wolf shrinkage.** The Ledoit-Wolf estimator regularizes the sample covariance toward a multiple of identity:

$$\hat{\Sigma}_{\text{LW}} = \alpha\hat{S} + (1-\alpha)\mu I$$

where $\mu = \text{tr}(\hat{S})/p$ (the average eigenvalue) and $\alpha$ is chosen to minimize mean squared error under the Frobenius norm. The result is always PD (since $\mu I \succ 0$ and $\hat{S} \succeq 0$).

**Graphical lasso.** Assuming the precision matrix (inverse covariance) $\Omega = \Sigma^{-1}$ is sparse (many zeros encode conditional independence), the graphical lasso estimates:

$$\hat{\Omega} = \arg\min_{\Omega \succ 0} \left[\text{tr}(\hat{S}\Omega) - \log\det\Omega + \lambda\|\Omega\|_1\right]$$

where $\|\cdot\|_1$ is the element-wise $\ell_1$ norm. The constraint $\Omega \succ 0$ is the PD cone constraint. The objective $\text{tr}(\hat{S}\Omega) - \log\det\Omega$ is the negative Gaussian log-likelihood (up to constants). Optimized by block coordinate descent (glasso algorithm) using the Schur complement for each block update.

**For LLMs:** Large language models trained with structured sparse attention patterns (local + global attention, as in Longformer and BigBird) can be understood as learning sparse precision matrices of the attention distributions.

### E.2 Case Study: Gaussian Process Regression, Step by Step

Let's trace through a complete GP regression computation to see all the PD machinery in action.

**Setup.** Training data: $(x_1, y_1) = (0, 0.5)$, $(x_2, y_2) = (1, 1.2)$, $(x_3, y_3) = (2, 0.8)$. Test point: $x_* = 1.5$. Kernel: $k(x,z) = \exp(-(x-z)^2/2)$ (RBF with length-scale 1). Noise: $\sigma^2 = 0.1$.

**Step 1: Compute the kernel matrix.**

$$K_{ij} = \exp(-(x_i - x_j)^2/2).$$

$$K = \begin{pmatrix}1 & e^{-0.5} & e^{-2} \\ e^{-0.5} & 1 & e^{-0.5} \\ e^{-2} & e^{-0.5} & 1\end{pmatrix} \approx \begin{pmatrix}1 & 0.607 & 0.135 \\ 0.607 & 1 & 0.607 \\ 0.135 & 0.607 & 1\end{pmatrix}.$$

**Step 2: Add noise and Cholesky factor.**

$$K + \sigma^2 I = K + 0.1I \approx \begin{pmatrix}1.1 & 0.607 & 0.135 \\ 0.607 & 1.1 & 0.607 \\ 0.135 & 0.607 & 1.1\end{pmatrix}.$$

Compute $L = \text{chol}(K + 0.1I)$. (Details in theory.ipynb.)

**Step 3: Predictive mean.**

$\mathbf{k}_* = [k(x_*, x_i)]_i = [\exp(-2.25/2), \exp(-0.25/2), \exp(-0.25/2)] \approx [0.325, 0.882, 0.882]$.

$$\boldsymbol{\alpha} = (K + 0.1I)^{-1}\mathbf{y} = L^{-\top}L^{-1}\mathbf{y} \quad (\text{via Cholesky solve})$$
$$\mu_* = \mathbf{k}_*^\top\boldsymbol{\alpha}.$$

**Step 4: Predictive variance.**

$$\sigma_*^2 = k(x_*, x_*) - \mathbf{k}_*^\top(K+0.1I)^{-1}\mathbf{k}_* = 1 - \mathbf{v}^\top\mathbf{v}, \quad \mathbf{v} = L^{-1}\mathbf{k}_*.$$

The variance is the Schur complement of $(K + 0.1I)$ evaluated at $x_*$.

**Interpretation.** The Cholesky factorization appears in every step: solving for $\boldsymbol{\alpha}$, computing the predictive variance, and computing the log-marginal likelihood for hyperparameter optimization. The PD condition on $K + 0.1I$ guarantees $\sigma_*^2 \geq 0$ (the Schur complement is non-negative).

### E.3 Case Study: The Multivariate Normal in Deep Learning

**Parameterizing structured covariances.** In deep generative models (VAEs, normalizing flows, diffusion models), we frequently need to parameterize a multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with $\Sigma$ learnable. The challenge: $\Sigma$ must be PD, and the parameterization must be differentiable.

**Approach 1: Diagonal.** $\Sigma = \text{diag}(\sigma_1^2,\ldots,\sigma_d^2)$ with $\sigma_i = \text{softplus}(s_i)$. Free parameters: $d$. Correlation structure: none. Used in standard VAE (Kingma & Welling, 2013).

**Approach 2: Lower triangular Cholesky.** $\Sigma = LL^\top$ where $L$ is lower triangular:
- Diagonal: $L_{ii} = \text{softplus}(\tilde{L}_{ii})$ ensures positivity
- Off-diagonal: $L_{ij}$ for $i > j$ unconstrained (any real number)
Free parameters: $d(d+1)/2$. Full correlation structure. Used in full-covariance VAEs, normalizing flows.

**Approach 3: Low-rank + diagonal.** $\Sigma = FF^\top + D$ where $F \in \mathbb{R}^{d \times r}$ ($r \ll d$) and $D = \text{diag}(d_1,\ldots,d_d)$ with $d_i > 0$. Free parameters: $dr + d$. Captures $r$ "directions of correlation." Used in structured VBs, mean-field approximations.

**The KL divergence between Gaussians.** For $q = \mathcal{N}(\boldsymbol{\mu}_q, \Sigma_q)$ and $p = \mathcal{N}(\boldsymbol{\mu}_p, \Sigma_p)$:

$$D_{\text{KL}}(q\|p) = \frac{1}{2}\left[\text{tr}(\Sigma_p^{-1}\Sigma_q) + (\boldsymbol{\mu}_p-\boldsymbol{\mu}_q)^\top\Sigma_p^{-1}(\boldsymbol{\mu}_p-\boldsymbol{\mu}_q) - d + \log\frac{\det\Sigma_p}{\det\Sigma_q}\right].$$

For $p = \mathcal{N}(\mathbf{0}, I)$ (standard VAE prior):

$$D_{\text{KL}}(q\|\mathcal{N}(\mathbf{0},I)) = \frac{1}{2}\left[\text{tr}(\Sigma_q) + \boldsymbol{\mu}_q^\top\boldsymbol{\mu}_q - d - \log\det\Sigma_q\right].$$

For diagonal $\Sigma_q = \text{diag}(\sigma_i^2)$:

$$D_{\text{KL}} = \frac{1}{2}\sum_{i=1}^d \left[\sigma_i^2 + \mu_{q,i}^2 - 1 - \log\sigma_i^2\right].$$

For Cholesky $\Sigma_q = LL^\top$: $\log\det\Sigma_q = 2\sum_i \log L_{ii}$ and $\text{tr}(\Sigma_q) = \|L\|_F^2$.

All these computations — $\log\det\Sigma_q$, $\text{tr}(\Sigma_q)$, $\Sigma_q^{-1}$ — exploit the Cholesky and log-det tools developed in §4 and §7.

### E.4 The Normal Equations and Ridge Regression

**Least squares.** Given $X \in \mathbb{R}^{n \times p}$ (data) and $\mathbf{y} \in \mathbb{R}^n$ (targets), ordinary least squares (OLS) minimizes:

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|X\boldsymbol{\beta} - \mathbf{y}\|^2.$$

The normal equations: $X^\top X \hat{\boldsymbol{\beta}} = X^\top\mathbf{y}$. The matrix $X^\top X \succeq 0$ is always PSD (Gram matrix). If $X$ has full column rank ($n \geq p$, rank $p$), then $X^\top X \succ 0$ and the unique solution is $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1}X^\top\mathbf{y}$.

**Ridge regression.** Add $\ell_2$ regularization: minimize $\|X\boldsymbol{\beta}-\mathbf{y}\|^2 + \lambda\|\boldsymbol{\beta}\|^2$.

Normal equations: $(X^\top X + \lambda I)\hat{\boldsymbol{\beta}} = X^\top\mathbf{y}$.

The matrix $A = X^\top X + \lambda I \succ 0$ for any $\lambda > 0$ (by Proposition 2.3.7). This is why ridge regression is always numerically stable: the PD matrix $A$ has all positive pivots, and Cholesky never fails.

**Condition number improvement.** $\kappa(X^\top X + \lambda I) = (\sigma_1^2 + \lambda)/(\sigma_p^2 + \lambda)$ where $\sigma_1 \geq \cdots \geq \sigma_p \geq 0$ are the singular values of $X$. For small $\lambda$, $\kappa \approx \sigma_1^2/\sigma_p^2 = \kappa(X)^2$. As $\lambda \to \infty$, $\kappa \to 1$ (well-conditioned). Choosing $\lambda$ balances bias (regularization) vs numerical stability.

**Kernel ridge regression.** For nonlinear regression using a PD kernel $k$, form the kernel matrix $K = k(\mathbf{X},\mathbf{X})$ and solve $(K + \lambda I)\boldsymbol{\alpha} = \mathbf{y}$. Predictions: $\hat{f}(\mathbf{x}_*) = \mathbf{k}_*^\top\boldsymbol{\alpha}$ where $\mathbf{k}_* = k(\mathbf{x}_*, \mathbf{X})$. This is equivalent to GP regression with the kernel $k$ and noise variance $\lambda$. The PD condition on $K + \lambda I$ is exactly the Cholesky solvability condition.


---

## Appendix F: Quick Reference

### F.1 Four Equivalent Characterizations of $A \succ 0$

```
POSITIVE DEFINITENESS: FOUR EQUIVALENT TESTS
════════════════════════════════════════════════════════════════════════

  Let A ∈ ℝⁿˣⁿ be symmetric. The following are equivalent:

  DEFINITION   ∀x≠0: x⊤Ax > 0            (quadratic form positive)
  SPECTRAL     ∀i: λᵢ(A) > 0              (all eigenvalues positive)
  SYLVESTER    ∀k: Δₖ = det(A[1:k,1:k]) > 0 (leading minors positive)
  CHOLESKY     ∃! lower triangular L with positive diagonal: A = LL⊤

  COST:
  Definition  O(n) per test vector            ← manual/symbolic
  Spectral    O(n³) eigendecomposition        ← most informative
  Sylvester   O(n⁴) for all n determinants    ← manual/symbolic
  Cholesky    O(n³/3) factorization           ← fastest in practice

════════════════════════════════════════════════════════════════════════
```

### F.2 Key Formulas

| Formula | Notes |
|---------|-------|
| $A = LL^\top$ | Cholesky factorization (A PD ↔ exists) |
| $A = LDL^\top$ | LDLᵀ: unit lower triangular + diagonal |
| $A = QA^{1/2}$ | Polar: orthogonal × PD |
| $\log\det A = 2\sum_i\log L_{ii}$ | Log-det via Cholesky |
| $\nabla_A\log\det A = A^{-1}$ | Matrix calculus: gradient of log-det |
| $S = D - CA^{-1}B$ | Schur complement of $A$ in block matrix |
| $M \succ 0 \Leftrightarrow A \succ 0,\, S \succ 0$ | Schur PD criterion |
| $(A+UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}$ | Woodbury identity |
| $\mathbf{x} = \boldsymbol{\mu} + L\boldsymbol{\epsilon}$, $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$ | Reparameterization trick |
| $D_{\text{KL}}(\mathcal{N}(\boldsymbol{\mu},\Sigma)\|\mathcal{N}(0,I)) = \frac{1}{2}[\text{tr}\Sigma + \|\boldsymbol{\mu}\|^2 - d - \log\det\Sigma]$ | VAE KL term |

### F.3 Notation Summary

Following the NOTATION_GUIDE.md conventions:

| Symbol | Meaning |
|--------|---------|
| $A \succ 0$ | $A$ is positive definite |
| $A \succeq 0$ | $A$ is positive semidefinite |
| $A \prec 0$ | $A$ is negative definite |
| $\mathbb{S}^n$ | Space of $n \times n$ symmetric matrices |
| $\mathbb{S}_{++}^n$ | Cone of $n \times n$ PD matrices (interior) |
| $\mathbb{S}_+^n$ | Cone of $n \times n$ PSD matrices |
| $L$ | Cholesky factor (lower triangular, positive diagonal) |
| $A^{1/2}$ | Symmetric PSD square root |
| $A^{-1/2}$ | Inverse PSD square root |
| $\Sigma$ | Covariance matrix ($\succeq 0$) |
| $\Lambda$ | Precision matrix ($= \Sigma^{-1}$, $\succ 0$) |
| $F$ | Fisher information matrix ($\succeq 0$) |
| $S$ | Schur complement |
| $G$ | Gram matrix ($= XX^\top$, $\succeq 0$) |
| $K$ | Kernel matrix ($\succeq 0$) |

### F.4 Python Cheatsheet

```python
import numpy as np
import scipy.linalg as la

# === Cholesky factorization ===
L = np.linalg.cholesky(A)          # A = L @ L.T
# Or via scipy (can also compute upper factor):
L = la.cholesky(A, lower=True)

# === LDL^T factorization ===
L_ldl, D_ldl, _ = la.ldl(A)       # A = L @ D @ L.T

# === Log-determinant (numerically stable) ===
L = np.linalg.cholesky(A)
log_det = 2 * np.sum(np.log(np.diag(L)))

# === Solve A x = b via Cholesky ===
L = np.linalg.cholesky(A)
x = la.cho_solve((L, True), b)     # uses LAPACK dpotrs

# === PSD square root ===
eigvals, Q = np.linalg.eigh(A)     # symmetric eigendecomp
A_sqrt = Q @ np.diag(np.sqrt(eigvals)) @ Q.T

# === Woodbury: (A + U C V)^{-1} ===
# via np.linalg.solve on the smaller system

# === Reparameterization trick ===
L = np.linalg.cholesky(Sigma)      # Sigma = L @ L.T
eps = np.random.randn(n_samples, d)
z = mu + (L @ eps.T).T             # z ~ N(mu, Sigma)

# === Gram matrix ===
G = X @ X.T                        # G = X X^T, PSD

# === Check PD ===
try:
    np.linalg.cholesky(A)
    is_pd = True
except np.linalg.LinAlgError:
    is_pd = False

# === Check PSD ===
eigvals = np.linalg.eigvalsh(A)
is_psd = np.all(eigvals >= -1e-10)
```


---

## Appendix G: Proofs and Extensions

### G.1 Complete Proof of the Cholesky Existence Theorem

We present the full inductive proof more carefully, making each step explicit.

**Theorem (Full Cholesky Existence).** $A \in \mathbb{R}^{n \times n}$ symmetric. $A \succ 0$ if and only if $A = LL^\top$ for a unique lower triangular $L$ with positive diagonal.

**Proof by strong induction on $n$.**

*Base case $n=1$:* $A = (a)$ with $a > 0$ (since $A \succ 0$ iff quadratic form $ax^2 > 0$ for $x \neq 0$ iff $a > 0$). Take $L = (\sqrt{a})$. Unique since $\ell > 0$ and $\ell^2 = a$ gives $\ell = \sqrt{a}$.

*Inductive step:* Assume the result holds for all $(n-1) \times (n-1)$ PD matrices. Let $A \in \mathbb{R}^{n \times n}$ be PD. Write:

$$A = \begin{pmatrix}a_{11} & \mathbf{a}^\top \\ \mathbf{a} & \hat{A}\end{pmatrix}$$

where $a_{11} > 0$ (Proposition 2.3, item 1), $\mathbf{a} \in \mathbb{R}^{n-1}$, and $\hat{A} \in \mathbb{R}^{(n-1)\times(n-1)}$.

Define $\ell_{11} = \sqrt{a_{11}} > 0$ and $\boldsymbol{\ell} = \mathbf{a}/\ell_{11} \in \mathbb{R}^{n-1}$.

Compute the Schur complement: $\tilde{A} = \hat{A} - \boldsymbol{\ell}\boldsymbol{\ell}^\top = \hat{A} - \mathbf{a}a_{11}^{-1}\mathbf{a}^\top$.

**Claim:** $\tilde{A} \succ 0$.

*Proof of claim:* For any $\mathbf{y} \in \mathbb{R}^{n-1}$ with $\mathbf{y} \neq \mathbf{0}$, set $\mathbf{x} = \begin{pmatrix}-a_{11}^{-1}\mathbf{a}^\top\mathbf{y} \\ \mathbf{y}\end{pmatrix} \in \mathbb{R}^n$. Note $\mathbf{x} \neq \mathbf{0}$ (the second block is $\mathbf{y} \neq \mathbf{0}$). Then:

$$\mathbf{x}^\top A \mathbf{x} = a_{11}(a_{11}^{-1}\mathbf{a}^\top\mathbf{y})^2 - 2\mathbf{a}^\top\mathbf{y} \cdot a_{11}^{-1}\mathbf{a}^\top\mathbf{y} + \mathbf{y}^\top\hat{A}\mathbf{y} = \mathbf{y}^\top(\hat{A} - \mathbf{a}a_{11}^{-1}\mathbf{a}^\top)\mathbf{y} = \mathbf{y}^\top\tilde{A}\mathbf{y}.$$

Since $A \succ 0$: $\mathbf{x}^\top A\mathbf{x} > 0$, so $\mathbf{y}^\top\tilde{A}\mathbf{y} > 0$. Since $\mathbf{y} \neq 0$ was arbitrary, $\tilde{A} \succ 0$. $\square$ (claim)

By the inductive hypothesis, $\tilde{A} = L_{22}L_{22}^\top$ for a unique lower triangular $L_{22} \in \mathbb{R}^{(n-1)\times(n-1)}$ with positive diagonal.

Set $L = \begin{pmatrix}\ell_{11} & \mathbf{0}^\top \\ \boldsymbol{\ell} & L_{22}\end{pmatrix}$. Then $L$ is lower triangular with positive diagonal $(\ell_{11}, (L_{22})_{11}, \ldots, (L_{22})_{n-1,n-1})$ and:

$$LL^\top = \begin{pmatrix}\ell_{11} & \mathbf{0}^\top \\ \boldsymbol{\ell} & L_{22}\end{pmatrix}\begin{pmatrix}\ell_{11} & \boldsymbol{\ell}^\top \\ \mathbf{0} & L_{22}^\top\end{pmatrix} = \begin{pmatrix}\ell_{11}^2 & \ell_{11}\boldsymbol{\ell}^\top \\ \ell_{11}\boldsymbol{\ell} & \boldsymbol{\ell}\boldsymbol{\ell}^\top + L_{22}L_{22}^\top\end{pmatrix} = \begin{pmatrix}a_{11} & \mathbf{a}^\top \\ \mathbf{a} & \hat{A}\end{pmatrix} = A.$$

*Uniqueness.* Suppose $A = L_1L_1^\top = L_2L_2^\top$ with $L_1, L_2$ lower triangular and positive diagonal. Then $M := L_2^{-1}L_1$ (product of lower triangular matrices, lower triangular) satisfies $MM^\top = I$. An orthogonal lower triangular matrix must be diagonal. Since $L_1, L_2$ have positive diagonal, $M$ has positive diagonal. $MM^\top = I$ for diagonal $M$ gives $M = I$, so $L_1 = L_2$. $\square$

### G.2 Proof That the Fisher Information Is PSD

**Theorem.** For a regular statistical model $p(\mathbf{x}|\boldsymbol{\theta})$, the Fisher information matrix $F(\boldsymbol{\theta}) = \mathbb{E}[\mathbf{s}(\mathbf{x};\boldsymbol{\theta})\mathbf{s}(\mathbf{x};\boldsymbol{\theta})^\top]$ where $\mathbf{s} = \nabla_{\boldsymbol{\theta}}\log p$ is the score function is PSD.

**Proof.** For any $\mathbf{v} \in \mathbb{R}^d$:

$$\mathbf{v}^\top F\mathbf{v} = \mathbb{E}[\mathbf{v}^\top\mathbf{s}\mathbf{s}^\top\mathbf{v}] = \mathbb{E}[(\mathbf{v}^\top\mathbf{s})^2] \geq 0.$$

(The expectation of a squared real random variable is non-negative.) $\square$

**When is $F \succ 0$?** $F$ is PD iff $\mathbb{E}[(\mathbf{v}^\top\mathbf{s})^2] > 0$ for all $\mathbf{v} \neq 0$. This holds iff the score function $\mathbf{v}^\top\nabla_{\boldsymbol{\theta}}\log p(\mathbf{x}|\boldsymbol{\theta}) \neq 0$ with positive probability for every $\mathbf{v} \neq 0$ — the model is "identifiable" in all directions. Singular $F$ indicates structural non-identifiability (two parameters produce identical distributions).

### G.3 Derivatives and the Matrix-Valued Chain Rule

For completeness, we derive the key matrix calculus formulas used in §7.3.

**Jacobi's formula.** For differentiable $A(t)$:
$$\frac{d}{dt}\det A(t) = \det A(t) \cdot \text{tr}(A(t)^{-1}\dot{A}(t)).$$

*Proof:* Using the adjugate matrix (cofactor expansion), $\det A = \sum_j A_{ij}(\text{adj}A)_{ji}$. Differentiating with respect to $A_{ij}$ gives $(\text{adj}A)_{ji} = (\det A)(A^{-1})_{ji}$ (Cramer's rule). By the chain rule: $d(\det A)/dt = \sum_{ij}(\text{adj}A)_{ji}\dot{A}_{ij} = \det(A)\sum_{ij}(A^{-1})_{ji}\dot{A}_{ij} = \det(A)\,\text{tr}(A^{-1}\dot{A})$.

**Log-det gradient.** $d\log\det A = d(\det A)/\det A = \text{tr}(A^{-1}dA)$. Since $\text{tr}(A^{-1}dA) = \langle A^{-\top}, dA\rangle_F$, the gradient of $\log\det$ at $A$ is $A^{-\top} = A^{-1}$ (for symmetric $A$).

**Trace-inverse gradient.** For $f(A) = \text{tr}(A^{-1}B)$: $df = \text{tr}(-A^{-1}dA\,A^{-1}B) = -\text{tr}(A^{-1}BA^{-1}dA) = -\langle (A^{-1}BA^{-1})^\top, dA\rangle_F$. So $\nabla_A\text{tr}(A^{-1}B) = -(A^{-1}BA^{-1})^\top = -A^{-\top}B^\top A^{-\top}$.

**For AI — GP hyperparameter gradients:** The gradient of the GP log-marginal-likelihood with respect to a kernel hyperparameter $\theta$ is:

$$\frac{\partial\log p(\mathbf{y}|\theta)}{\partial\theta} = \frac{1}{2}\text{tr}\!\left[(\boldsymbol{\alpha}\boldsymbol{\alpha}^\top - (K+\sigma^2I)^{-1})\frac{\partial K}{\partial\theta}\right]$$

where $\boldsymbol{\alpha} = (K+\sigma^2I)^{-1}\mathbf{y}$. This uses $\nabla_K\log\det K = K^{-1}$ and $\nabla_K\text{tr}(K^{-1}S) = -(K^{-1}SK^{-1})^\top$. Efficiently computed via Cholesky: form $V = L^{-1}\partial K/\partial\theta$ (triangular solve), then $\text{tr}(K^{-1}\partial K/\partial\theta) = \text{tr}(V^\top V) = \|V\|_F^2$.


---

## Appendix H: Summary and Further Reading

### H.1 Core Theorems Summary

| Theorem | Statement | Reference |
|---------|-----------|-----------|
| Spectral characterization | $A \succ 0 \Leftrightarrow$ all eigenvalues $> 0$ | §3.1 |
| Sylvester's criterion | $A \succ 0 \Leftrightarrow$ all leading principal minors $> 0$ | §3.2 |
| Cholesky existence | $A \succ 0 \Leftrightarrow \exists!$ lower triangular $L$ (pos. diag.) with $A = LL^\top$ | §4.1 |
| LDLᵀ | Symmetric $A \to A = LDL^\top$; $A \succ 0 \Leftrightarrow$ all $d_i > 0$ | §4.4 |
| PSD square root | $A \succeq 0 \Rightarrow \exists!$ symmetric PSD $A^{1/2}$ with $(A^{1/2})^2 = A$ | §5.1 |
| Schur PD criterion | $M = \begin{pmatrix}A&B\\B^\top&D\end{pmatrix} \succ 0 \Leftrightarrow A \succ 0, D-B^\top A^{-1}B \succ 0$ | §6.2 |
| Woodbury identity | $(A+UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1}+VA^{-1}U)^{-1}VA^{-1}$ | §6.3 |
| Log-det concavity | $f(A) = \log\det A$ is strictly concave on $\mathbb{S}_{++}^n$ | §7.2 |
| Log-det gradient | $\nabla_A\log\det A = A^{-1}$ | §7.3 |
| Gram matrix PSD | $G = XX^\top \succeq 0$ always; PD iff rows of $X$ are linearly independent | §8.1 |
| Schur product | Hadamard product of PSD matrices is PSD | Appendix A |
| Hadamard inequality | $\det A \leq \prod_i A_{ii}$ for $A \succ 0$ | Appendix D |
| Log-det Cholesky | $\log\det A = 2\sum_i\log L_{ii}$ | §7.1 |

### H.2 Further Reading

1. **Textbooks:**
   - Golub & Van Loan, *Matrix Computations* (4th ed., 2013) — Chapters 4, 7: definitive reference on Cholesky, LDLᵀ algorithms
   - Higham, *Accuracy and Stability of Numerical Algorithms* (2002) — backward stability proofs for Cholesky, modified Cholesky
   - Boyd & Vandenberghe, *Convex Optimization* (2004) — Chapter 4: SDP, PSD cone; freely available online
   - Bernstein, *Matrix Mathematics* (2nd ed., 2009) — comprehensive collection of PD matrix identities

2. **Research papers:**
   - Cholesky (1924, posthumous publication via Benoit): original algorithm for geodetic calculations
   - Gill, Murray & Wright (1981): modified Cholesky for optimization
   - Vandenberghe & Boyd (1996): semidefinite programming tutorial
   - Martens & Grosse (2015): K-FAC — Kronecker-factored natural gradient
   - Gardner et al. (2018): GPyTorch — scalable GP with stochastic log-det estimation
   - Foret et al. (2021): SAM — sharpness-aware minimization

3. **Online resources:**
   - Petersen & Pedersen, *The Matrix Cookbook* — practical matrix calculus identities
   - NumPy/SciPy docs: `np.linalg.cholesky`, `scipy.linalg.cholesky`, `scipy.linalg.ldl`
   - GPyTorch documentation: scalable GP inference with Cholesky
   - CVXPY documentation: SDP examples and semidefinite programming

4. **Next sections:**
   - [§08: Matrix Decompositions](../08-Matrix-Decompositions/notes.md) — LU, QR, SVD algorithms (Cholesky briefly revisited in algorithmic context)
   - [Chapter 6: Probability Theory](../../06-Probability-Theory/) — multivariate Gaussians, covariance matrices
   - [Chapter 8: Optimization](../../08-Optimization/) — second-order methods, convexity, SDP
   - [Chapter 12: Functional Analysis](../../12-Functional-Analysis/) — RKHS, Mercer's theorem, kernel methods


---

*End of §07: Positive Definite Matrices. Next: [§08: Matrix Decompositions](../08-Matrix-Decompositions/notes.md)*

---

## Appendix I: Worked Examples — Complete Solutions

### I.1 Full 3×3 Cholesky Example with Verification

Compute $L = \text{chol}(A)$ for $A = \begin{pmatrix}9 & 3 & -3 \\ 3 & 17 & -1 \\ -3 & -1 & 12\end{pmatrix}$.

**Step 1:** $L_{11} = \sqrt{9} = 3$.

**Step 2:** $L_{21} = A_{21}/L_{11} = 3/3 = 1$. $L_{31} = A_{31}/L_{11} = -3/3 = -1$.

**Step 3:** $L_{22} = \sqrt{A_{22} - L_{21}^2} = \sqrt{17 - 1} = \sqrt{16} = 4$.

**Step 4:** $L_{32} = (A_{32} - L_{31}L_{21})/L_{22} = (-1 - (-1)(1))/4 = 0/4 = 0$.

**Step 5:** $L_{33} = \sqrt{A_{33} - L_{31}^2 - L_{32}^2} = \sqrt{12 - 1 - 0} = \sqrt{11}$.

$$L = \begin{pmatrix}3 & 0 & 0 \\ 1 & 4 & 0 \\ -1 & 0 & \sqrt{11}\end{pmatrix}.$$

**Verification:**
$$LL^\top = \begin{pmatrix}3&0&0\\1&4&0\\-1&0&\sqrt{11}\end{pmatrix}\begin{pmatrix}3&1&-1\\0&4&0\\0&0&\sqrt{11}\end{pmatrix} = \begin{pmatrix}9&3&-3\\3&17&-1\\-3&-1&12\end{pmatrix} = A. \checkmark$$

Also: $\log\det A = 2(\log 3 + \log 4 + \log\sqrt{11}) = 2(1.099 + 1.386 + 1.199) = 2(3.684) = 7.368$.

### I.2 Schur Complement and Conditional Gaussian

Let $\Sigma = \begin{pmatrix}4 & 2 \\ 2 & 5\end{pmatrix}$ be the covariance of $(X_1, X_2)^\top \sim \mathcal{N}(\mathbf{0}, \Sigma)$.

**Schur complement of $\Sigma_{11}$:**
$$S = \Sigma_{22} - \Sigma_{21}\Sigma_{11}^{-1}\Sigma_{12} = 5 - 2 \cdot \frac{1}{4} \cdot 2 = 5 - 1 = 4.$$

**Conditional distribution $X_1 | X_2 = a$:**
$$\mu_{1|2} = 0 + \Sigma_{12}\Sigma_{22}^{-1}(a - 0) = \frac{2}{5}a.$$
$$\sigma_{1|2}^2 = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = 4 - 2 \cdot \frac{1}{5} \cdot 2 = 4 - \frac{4}{5} = \frac{16}{5}.$$

Note: the unconditional variance of $X_1$ is $\sigma_1^2 = 4$. The conditional variance $16/5 = 3.2 < 4$ — observing $X_2$ reduces uncertainty about $X_1$ (as guaranteed by the Loewner order, §2.4). The correlation $\rho = 2/\sqrt{4 \cdot 5} = 2/\sqrt{20} = 1/\sqrt{5} \approx 0.447$ explains the moderate uncertainty reduction.

