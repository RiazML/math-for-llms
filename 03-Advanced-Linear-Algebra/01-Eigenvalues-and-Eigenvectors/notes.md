[← Back to Advanced Linear Algebra](../README.md) | [Next: Singular Value Decomposition →](../02-Singular-Value-Decomposition/notes.md)

---

# Eigenvalues and Eigenvectors

> _"The eigenvalues of a matrix are not just numbers — they are the heartbeat of the linear map, the frequencies at which it resonates, the rates at which it remembers and forgets."_

## Overview

Eigenvalues and eigenvectors are the most powerful diagnostic tool in linear algebra. Given a square matrix $A$, an eigenvector is a non-zero vector $\mathbf{v}$ that the matrix maps to a scalar multiple of itself: $A\mathbf{v} = \lambda\mathbf{v}$. The scalar $\lambda$ is the corresponding eigenvalue. Every other vector gets rotated when $A$ is applied; eigenvectors are the unique directions that survive without rotation — they are the natural axes of the linear map, the directions in which the map acts most simply.

The power of this concept is hard to overstate. The long-run behaviour of any linear dynamical system — whether a Markov chain, a recurrent neural network, or gradient descent on a quadratic loss — is entirely determined by the eigenvalue spectrum. The eigenvector for the largest eigenvalue is where the system ends up; the ratio of the top two eigenvalues controls the convergence rate; a single eigenvalue outside the unit disk triggers instability. Understanding eigenvalues is understanding the future of any linear system.

For AI practitioners in 2026, eigenvalues appear everywhere: PCA decomposes the data covariance matrix into eigenvectors (principal components) and eigenvalues (variances); the Hessian eigenspectrum determines gradient descent convergence and reveals flat vs sharp minima; attention matrices are row-stochastic with Perron-Frobenius eigenvalue 1; LoRA adapts the dominant eigenspace of weight matrices; spectral normalisation bounds the largest singular value to stabilise GAN training; graph neural networks convolve in the eigenspace of the graph Laplacian. This section develops all of these connections from first principles.

## Prerequisites

- Vector spaces, subspaces, span, and linear independence (Section 02-06)
- Matrix multiplication, transpose, and invertibility (Section 02-02)
- Determinants and their connection to invertibility (Section 02-04)
- Systems of linear equations and row reduction / RREF (Section 02-03)
- Matrix rank and null spaces (Section 02-05)
- Basic familiarity with complex numbers (for complex eigenvalues)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive demonstrations: geometric visualisation, power iteration, QR algorithm, spectral theorem, PCA, Hessian spectrum, graph Laplacian, spectral normalisation |
| [exercises.ipynb](exercises.ipynb) | 8 graded problems: eigenpair verification, characteristic polynomial, diagonalisation, spectral theorem, matrix powers, Rayleigh quotient, PCA from scratch, power iteration |

## Learning Objectives

After completing this section, you will:

- State the eigenvalue equation $A\mathbf{v} = \lambda\mathbf{v}$ and explain its geometric meaning
- Compute eigenvalues by solving the characteristic equation $\det(\lambda I - A) = 0$
- Find eigenvectors by solving the null space of $(A - \lambda I)$
- Distinguish algebraic and geometric multiplicity; identify defective matrices
- Diagonalise a matrix as $A = V\Lambda V^{-1}$ and use it to compute matrix powers
- State and apply the spectral theorem for symmetric matrices ($A = Q\Lambda Q^\top$)
- Characterise positive (semi)definite matrices via their eigenvalue spectrum
- Apply the Rayleigh quotient and Courant-Fischer theorem
- Compute the dominant eigenvector via power iteration
- Connect eigenvalues to PCA, gradient descent convergence, RNN stability, and attention structure
- Explain the Hessian eigenspectrum and its role in neural network training dynamics
- Use Gershgorin circles and Weyl inequalities to bound eigenvalues without computing them

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Are Eigenvalues and Eigenvectors?](#11-what-are-eigenvalues-and-eigenvectors)
  - [1.2 The Geometric Picture](#12-the-geometric-picture)
  - [1.3 Why Eigenvalues Are Central to AI](#13-why-eigenvalues-are-central-to-ai)
  - [1.4 Eigenvalues as Long-Run Behaviour](#14-eigenvalues-as-long-run-behaviour)
  - [1.5 Historical Timeline](#15-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The Eigenvalue Equation](#21-the-eigenvalue-equation)
  - [2.2 The Characteristic Polynomial](#22-the-characteristic-polynomial)
  - [2.3 Algebraic and Geometric Multiplicity](#23-algebraic-and-geometric-multiplicity)
  - [2.4 Eigenspaces](#24-eigenspaces)
  - [2.5 The Spectrum](#25-the-spectrum)
- [3. Computing Eigenvalues and Eigenvectors](#3-computing-eigenvalues-and-eigenvectors)
  - [3.1 The Two-Step Process](#31-the-two-step-process)
  - [3.2 The 2×2 Case](#32-the-22-case)
  - [3.3 The 3×3 and Higher Cases](#33-the-33-and-higher-cases)
  - [3.4 Worked Example — 3×3 Defective Matrix](#34-worked-example--33-defective-matrix)
  - [3.5 Special Cases for the Characteristic Polynomial](#35-special-cases-for-the-characteristic-polynomial)
  - [3.6 Power Iteration](#36-power-iteration)
  - [3.7 The QR Algorithm](#37-the-qr-algorithm)
  - [3.8 Lanczos Algorithm for Large Sparse Matrices](#38-lanczos-algorithm-for-large-sparse-matrices)
- [4. Properties of Eigenvalues](#4-properties-of-eigenvalues)
  - [4.1 Trace and Determinant Relations](#41-trace-and-determinant-relations)
  - [4.2 Eigenvalues Under Matrix Operations](#42-eigenvalues-under-matrix-operations)
  - [4.3 Eigenvalues of Special Matrices](#43-eigenvalues-of-special-matrices)
  - [4.4 The Rayleigh Quotient](#44-the-rayleigh-quotient)
  - [4.5 The Courant-Fischer Min-Max Theorem](#45-the-courant-fischer-min-max-theorem)
  - [4.6 Gershgorin Circle Theorem](#46-gershgorin-circle-theorem)
- [5. Diagonalisation](#5-diagonalisation)
  - [5.1 The Diagonalisation Theorem](#51-the-diagonalisation-theorem)
  - [5.2 When Is A Diagonalisable?](#52-when-is-a-diagonalisable)
  - [5.3 Matrix Powers via Diagonalisation](#53-matrix-powers-via-diagonalisation)
  - [5.4 Matrix Functions via Diagonalisation](#54-matrix-functions-via-diagonalisation)
  - [5.5 Change of Basis Interpretation](#55-change-of-basis-interpretation)
  - [5.6 Simultaneous Diagonalisation](#56-simultaneous-diagonalisation)
- [6. The Spectral Theorem](#6-the-spectral-theorem)
  - [6.1 Statement and Significance](#61-statement-and-significance)
  - [6.2 Proof Sketch](#62-proof-sketch)
  - [6.3 The Spectral Decomposition](#63-the-spectral-decomposition)
  - [6.4 Positive (Semi)definite Matrices via the Spectral Theorem](#64-positive-semidefinite-matrices-via-the-spectral-theorem)
  - [6.5 The Complex Spectral Theorem](#65-the-complex-spectral-theorem)
- [7. Jordan Normal Form](#7-jordan-normal-form)
  - [7.1 Motivation — Defective Matrices](#71-motivation--defective-matrices)
  - [7.2 Jordan Blocks](#72-jordan-blocks)
  - [7.3 The Jordan Normal Form](#73-the-jordan-normal-form)
  - [7.4 Powers of Jordan Blocks](#74-powers-of-jordan-blocks)
- [8. Singular Value Decomposition](#8-singular-value-decomposition)
  - [8.1 SVD as Generalisation of Eigendecomposition](#81-svd-as-generalisation-of-eigendecomposition)
  - [8.2 Connection to Eigenvalues](#82-connection-to-eigenvalues)
  - [8.3 Thin and Truncated SVD](#83-thin-and-truncated-svd)
  - [8.4 Geometric Interpretation](#84-geometric-interpretation)
  - [8.5 SVD and the Four Fundamental Subspaces](#85-svd-and-the-four-fundamental-subspaces)
- [9. Eigendecomposition in AI Applications](#9-eigendecomposition-in-ai-applications)
  - [9.1 Principal Component Analysis (PCA)](#91-principal-component-analysis-pca)
  - [9.2 Eigenvalues and Gradient Descent](#92-eigenvalues-and-gradient-descent)
  - [9.3 Neural Network Hessian Spectrum](#93-neural-network-hessian-spectrum)
  - [9.4 Attention Eigenstructure](#94-attention-eigenstructure)
  - [9.5 Graph Laplacian and GNNs](#95-graph-laplacian-and-gnns)
  - [9.6 Spectral Analysis of Weight Matrices](#96-spectral-analysis-of-weight-matrices)
  - [9.7 Eigenvalues in Recurrent Networks](#97-eigenvalues-in-recurrent-networks)
  - [9.8 Diffusion Models and Score Functions](#98-diffusion-models-and-score-functions)
- [10. Generalised Eigenproblems](#10-generalised-eigenproblems)
- [11. Eigenvalues and Stability Analysis](#11-eigenvalues-and-stability-analysis)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)
- [14. Why This Matters for AI (2026 Perspective)](#14-why-this-matters-for-ai-2026-perspective)
- [15. Conceptual Bridge](#15-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Are Eigenvalues and Eigenvectors?

A linear map $A: \mathbb{R}^n \to \mathbb{R}^n$ generally rotates and stretches every vector it acts on. Apply it to any random vector $\mathbf{x}$ and the result $A\mathbf{x}$ points in a completely different direction. But for certain special non-zero vectors $\mathbf{v}$, something remarkable happens: $A\mathbf{v}$ points in exactly the same direction as $\mathbf{v}$. The map stretches or shrinks $\mathbf{v}$, possibly flips it, but does not rotate it. These special vectors are called **eigenvectors** of $A$, and the stretching factor is the corresponding **eigenvalue** $\lambda$:

$$A\mathbf{v} = \lambda\mathbf{v}$$

The eigenvalue $\lambda$ encodes everything about how $A$ acts on $\mathbf{v}$: if $\lambda > 1$ the vector is stretched; if $0 < \lambda < 1$ it is shrunk; if $\lambda < 0$ it is reversed in direction and possibly scaled; if $\lambda = 0$ it is collapsed to the zero vector; if $\lambda = 1$ it is left completely unchanged. The eigenvector itself gives the direction; the eigenvalue gives the magnitude and sign of the action.

The word *eigen* is German for "own" or "characteristic." Eigenvectors are the characteristic directions that belong intrinsically to the linear map — they are independent of the coordinate system you choose to describe the map. No matter how you rotate your axes, the eigenvectors of $A$ remain the same geometric objects. They are the natural language in which the map speaks most simply.

**For AI:** The eigenvectors of the sample covariance matrix $C = \tilde{X}^\top \tilde{X}/(n-1)$ are the principal components — the directions of maximum variance in the data. The eigenvalues of the Hessian $\nabla^2 \mathcal{L}$ are the curvatures of the loss landscape in each eigendirection; large eigenvalues correspond to sharp, fast-converging directions, small eigenvalues to flat, slow-converging ones. The eigenvalues of a recurrent weight matrix $W$ determine whether the RNN's hidden state grows, decays, or oscillates as time steps accumulate.

### 1.2 The Geometric Picture

The clearest way to see eigenvalues geometrically is to watch what $A$ does to every unit vector simultaneously. Imagine the unit circle $\{\mathbf{x} : \|\mathbf{x}\| = 1\}$ in $\mathbb{R}^2$. Apply $A$ to every point on this circle. The result is an ellipse. The eigenvectors of $A$ are the directions that map to points on the same ray through the origin — vectors that get stretched along their own line.

For a symmetric matrix $A = A^\top$, the eigenvectors are orthogonal to each other. The ellipse's axes align exactly with the eigenvectors, and the lengths of the semi-axes equal the absolute eigenvalues. The map acts as a pure coordinate-wise scaling in the eigenbasis: no coupling, no rotation, just independent stretching of orthogonal directions. This is the content of the Spectral Theorem (Section 6), the most important structural result in applied linear algebra.

For a non-symmetric matrix, the picture is more complex. Eigenvectors may not be orthogonal. They may be complex even for a real matrix (a 2D rotation matrix, for instance, has no real eigenvectors — it rotates every real vector). But the eigenvalues still control the long-run behaviour: repeated application of $A$ amplifies the direction of the largest $|\lambda_i|$ and suppresses all others.

```
GEOMETRIC ACTION OF A 2×2 MATRIX
════════════════════════════════════════════════════════════════════════

  Unit circle                    After applying A
       ·                              ·  ·
     ·   ·          A             ·        ·
    ·     ·    ─────────►      ·              ·
     ·   ·                    (ellipse, axes = eigenvectors,
       ·                       semi-axes = |eigenvalues|)

  Most vectors get ROTATED.     Eigenvectors v₁, v₂ stay on
                                the same ray — only stretched.

════════════════════════════════════════════════════════════════════════
```

### 1.3 Why Eigenvalues Are Central to AI

Eigenvalues are not a piece of abstract machinery that happens to appear in ML — they are the fundamental language of how linear systems evolve, converge, compress, and fail. Here are the core connections:

**PCA and covariance structure.** Given a centred data matrix $\tilde{X} \in \mathbb{R}^{n \times d}$, the sample covariance is $C = \tilde{X}^\top \tilde{X}/(n-1) \in \mathbb{R}^{d \times d}$, a symmetric positive semidefinite matrix. Its eigendecomposition $C = Q\Lambda Q^\top$ gives eigenvectors $\mathbf{q}_1, \ldots, \mathbf{q}_d$ (the principal components) and eigenvalues $\lambda_1 \geq \cdots \geq \lambda_d \geq 0$ (the variances in each direction). Keeping only the top-$k$ eigenvectors gives the optimal rank-$k$ approximation to the data — the subspace capturing maximum variance.

**Gradient descent convergence.** For a quadratic loss $\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^\top H \boldsymbol{\theta} - \mathbf{b}^\top \boldsymbol{\theta}$, gradient descent with step size $\eta$ converges iff $|1 - \eta\lambda_i| < 1$ for all eigenvalues $\lambda_i$ of the Hessian $H$. The convergence rate in each eigendirection is $(1 - \eta\lambda_i)^t$. The condition number $\kappa(H) = \lambda_{\max}/\lambda_{\min}$ determines how much slower the slowest direction converges relative to the fastest — large $\kappa$ means gradient descent zigzags and stalls.

**Vanishing and exploding gradients.** In a deep linear network with the same weight matrix $W$ at each of $L$ layers, the gradient magnitude scales as $\|W^L\| \approx \rho(W)^L$ where $\rho(W) = \max_i |\lambda_i|$ is the spectral radius. If $\rho(W) > 1$: gradients explode. If $\rho(W) < 1$: gradients vanish. The eigenspectrum of $W$ is the fundamental quantity controlling gradient flow through depth.

**Attention and Perron-Frobenius.** The attention weight matrix $S = \text{softmax}(QK^\top/\sqrt{d_k})$ is row-stochastic. By the Perron-Frobenius theorem, its dominant eigenvalue is exactly 1, with all others having $|\lambda_i| \leq 1$. The structure of the remaining eigenvalues characterises the type of attention pattern — induction heads, name-mover heads, and copying heads each have distinct eigenvalue signatures.

**Spectral normalisation and stability.** Dividing a weight matrix $W$ by its largest singular value $\sigma_{\max}(W)$ (spectral norm) enforces a Lipschitz constraint of 1 on that layer. This stabilises GAN training and normalising flow invertibility. The largest singular value is itself the square root of the largest eigenvalue of $W^\top W$.

### 1.4 Eigenvalues as Long-Run Behaviour

The deepest reason eigenvalues matter is that they completely determine the long-run behaviour of any linear dynamical system. If $A\mathbf{v}_i = \lambda_i \mathbf{v}_i$, then repeated application gives:

$$A^t \mathbf{v}_i = \lambda_i^t \mathbf{v}_i$$

Each eigenvector mode grows or decays at rate $\lambda_i^t$. For any initial vector decomposed in the eigenbasis, $\mathbf{x}_0 = \sum_i c_i \mathbf{v}_i$:

$$A^t \mathbf{x}_0 = \sum_{i=1}^n c_i \lambda_i^t \mathbf{v}_i$$

The mode with the largest $|\lambda_i|$ — the **dominant eigenvalue** $\lambda_1$ — eventually dominates everything: as $t \to \infty$, $A^t \mathbf{x}_0 \approx c_1 \lambda_1^t \mathbf{v}_1$. This is why **power iteration** works: repeatedly apply $A$ and renormalise, and the iterates converge to the eigenvector for $\lambda_1$ at rate $|\lambda_2/\lambda_1|$ per step. It is also why PageRank works — the steady-state page importance vector is the dominant eigenvector of the web link matrix.

**For AI:** Training dynamics near a loss minimum are approximately a linear system with $A = I - \eta H$. The eigenvalues of $A$ are $1 - \eta\lambda_i(H)$. The long-run behaviour of training error is therefore $\sum_i c_i (1-\eta\lambda_i)^t \mathbf{v}_i$ — each Hessian eigendirection decays geometrically at its own rate. The slowest-converging direction is the one with the smallest Hessian eigenvalue (smallest curvature = flattest direction).

### 1.5 Historical Timeline

| Year | Who | What |
|---|---|---|
| 1750s | Euler | Special exponential solutions to ODEs; implicit eigenvalue reasoning for linear differential equations |
| 1762 | Lagrange | Reduction of quadratic forms to "principal axes"; normal modes of mechanical systems |
| 1829 | Cauchy | Rigorous proof that real symmetric matrices have real eigenvalues; foundation of spectral theory |
| 1856 | Hermite | Generalisation to complex Hermitian matrices; eigenvalues real for $A = A^*$ |
| 1866 | Kronecker | Formulated the characteristic polynomial $\det(\lambda I - A)$; eigenvalue problem as polynomial root-finding |
| 1870 | Jordan | Jordan normal form; canonical classification of all square matrices up to similarity |
| 1904–10 | Hilbert | Spectral theorem for infinite-dimensional operators; eigenvalues of integral operators; birth of functional analysis |
| 1925–27 | Heisenberg, von Neumann | Quantum mechanics as eigenvalue problems; observable quantities = eigenvalues of Hermitian operators |
| 1950 | Lanczos | Krylov subspace method for extreme eigenvalues of large sparse symmetric matrices |
| 1955 | Wigner | Random matrix theory; semicircle law for eigenvalue distributions of large random symmetric matrices |
| 1961–62 | Francis; Kublanovskaya | QR algorithm — the practical standard for computing all eigenvalues; one of the top 10 algorithms of the 20th century |
| 1965 | Golub-Reinsch | Practical SVD algorithm; singular values as generalised eigenvalues for rectangular matrices |
| 1989 | LeCun et al. | Hessian eigenspectrum of neural networks analysed; curvature-aware learning rate scheduling |
| 2010 | Martens | K-FAC — Kronecker-factored approximate curvature; eigendecomposition of layer-wise Hessian approximations |
| 2019–26 | Martin & Mahoney | WeightWatcher — heavy-tailed eigenvalue distributions as quality metrics for trained neural networks |
| 2017–26 | Transformer era | Spectral analysis of attention matrices; NTK eigenvalues; spectral normalisation in GANs; LoRA via dominant eigenspace |

---

## 2. Formal Definitions

### 2.1 The Eigenvalue Equation

**Definition.** Let $A \in \mathbb{R}^{n \times n}$ be a square matrix. A scalar $\lambda \in \mathbb{C}$ is an **eigenvalue** of $A$ and a non-zero vector $\mathbf{v} \in \mathbb{C}^n \setminus \{\mathbf{0}\}$ is the corresponding **eigenvector** if:

$$A\mathbf{v} = \lambda\mathbf{v}$$

Three equivalent ways to state this:

1. **Geometric**: $A$ maps $\mathbf{v}$ to a scalar multiple of itself; direction is preserved, only magnitude (and possibly sign) changes.
2. **Algebraic**: $(A - \lambda I)\mathbf{v} = \mathbf{0}$ has a non-trivial solution $\mathbf{v} \neq \mathbf{0}$.
3. **Subspace**: $\mathbf{v} \in \text{null}(A - \lambda I)$ and $\text{null}(A - \lambda I) \neq \{\mathbf{0}\}$.

The requirement $\mathbf{v} \neq \mathbf{0}$ is essential: the zero vector satisfies $A\mathbf{0} = \lambda\mathbf{0}$ for every $\lambda$, so it carries no information. Any non-zero scalar multiple of an eigenvector is also an eigenvector: $A(\alpha\mathbf{v}) = \alpha A\mathbf{v} = \alpha\lambda\mathbf{v} = \lambda(\alpha\mathbf{v})$. Eigenvectors are therefore not unique — they define a direction, not a specific vector. The set of all eigenvectors for a given $\lambda$, together with $\mathbf{0}$, forms a subspace called the **eigenspace** (Section 2.4).

**Non-examples of eigenvalues:**
- The zero matrix $A = 0$ has every non-zero vector as an eigenvector with $\lambda = 0$, but no non-zero eigenvalue.
- A rotation matrix in $\mathbb{R}^2$ by angle $\theta \neq 0, \pi$ has no real eigenvalues — it rotates every real vector off its own direction.
- Non-square matrices have no eigenvalues (the equation $A\mathbf{v} = \lambda\mathbf{v}$ requires $A$ to map $\mathbb{R}^n \to \mathbb{R}^n$).

### 2.2 The Characteristic Polynomial

For $\lambda$ to be an eigenvalue, $(A - \lambda I)\mathbf{v} = \mathbf{0}$ must have a non-trivial solution. This happens precisely when $A - \lambda I$ is singular, i.e., when its determinant is zero. This gives the **characteristic equation**:

$$\det(\lambda I - A) = 0$$

The function $p(\lambda) = \det(\lambda I - A)$ is a degree-$n$ polynomial in $\lambda$ called the **characteristic polynomial** of $A$. Its roots are exactly the eigenvalues of $A$. By convention we use $\det(\lambda I - A)$ (rather than $\det(A - \lambda I)$) so that the leading coefficient is $+1$ (monic polynomial):

$$p(\lambda) = \lambda^n - \text{tr}(A)\,\lambda^{n-1} + \cdots + (-1)^n \det(A)$$

Two key coefficients appear regardless of the matrix:
- **Coefficient of $\lambda^{n-1}$**: $-\text{tr}(A) = -\sum_i \lambda_i$ — the sum of eigenvalues equals the trace.
- **Constant term** $p(0) = (-1)^n \det(A) = (-1)^n \prod_i \lambda_i$ — the product of eigenvalues equals the determinant (up to sign).

By the **Fundamental Theorem of Algebra**, $p(\lambda)$ has exactly $n$ roots over $\mathbb{C}$ (counting multiplicity). Every $n \times n$ real matrix therefore has exactly $n$ eigenvalues in $\mathbb{C}$, though some may be complex and some may be repeated.

**For a $2 \times 2$ matrix** $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$p(\lambda) = \lambda^2 - (a+d)\lambda + (ad - bc) = \lambda^2 - \text{tr}(A)\,\lambda + \det(A)$$

$$\lambda_{1,2} = \frac{\text{tr}(A) \pm \sqrt{\text{tr}(A)^2 - 4\det(A)}}{2}$$

The discriminant $\Delta = \text{tr}(A)^2 - 4\det(A)$ determines the nature of the eigenvalues: $\Delta > 0$ gives two distinct real eigenvalues; $\Delta = 0$ gives one repeated real eigenvalue; $\Delta < 0$ gives two complex conjugate eigenvalues.

### 2.3 Algebraic and Geometric Multiplicity

When an eigenvalue $\lambda_i$ appears as a repeated root of $p(\lambda)$, we distinguish two notions of multiplicity:

**Algebraic multiplicity** $m_a(\lambda_i)$: the number of times $(\lambda - \lambda_i)$ divides $p(\lambda)$; the multiplicity of $\lambda_i$ as a root of the characteristic polynomial.

**Geometric multiplicity** $m_g(\lambda_i)$: the dimension of the eigenspace $\text{null}(A - \lambda_i I)$; the number of linearly independent eigenvectors for $\lambda_i$.

These two numbers satisfy the fundamental inequality:

$$1 \leq m_g(\lambda_i) \leq m_a(\lambda_i)$$

The lower bound holds because every eigenvalue has at least one eigenvector. The upper bound requires proof but follows from the fact that the eigenspace cannot be larger than the multiplicity of the root. When $m_g(\lambda_i) = m_a(\lambda_i)$ for every eigenvalue, the matrix is **diagonalisable** — it has $n$ linearly independent eigenvectors. When $m_g(\lambda_i) < m_a(\lambda_i)$ for some eigenvalue, the matrix is **defective** and cannot be diagonalised; it requires the Jordan Normal Form (Section 7).

**Example.** Consider $A = \begin{pmatrix} 2 & 1 \\ 0 & 2 \end{pmatrix}$. The characteristic polynomial is $(\lambda - 2)^2$, so $\lambda = 2$ has $m_a = 2$. But $A - 2I = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$ has null space spanned only by $\mathbf{e}_1 = (1,0)^\top$, so $m_g = 1 < m_a = 2$. This matrix is defective.

### 2.4 Eigenspaces

For eigenvalue $\lambda$, the **eigenspace** is:

$$E(\lambda) = \text{null}(A - \lambda I) = \{\mathbf{v} \in \mathbb{R}^n : A\mathbf{v} = \lambda\mathbf{v}\}$$

$E(\lambda)$ is always a subspace (it is the null space of $A - \lambda I$). It contains the zero vector and is closed under addition and scalar multiplication. Its dimension equals the geometric multiplicity: $\dim E(\lambda) = m_g(\lambda) = n - \text{rank}(A - \lambda I)$.

Two key properties make eigenspaces structurally special:

**Invariance**: $E(\lambda)$ is invariant under $A$. If $\mathbf{v} \in E(\lambda)$ then $A\mathbf{v} = \lambda\mathbf{v} \in E(\lambda)$. Applying $A$ keeps you inside the eigenspace — eigenvectors stay eigenvectors.

**Independence across eigenvalues**: Eigenvectors for different eigenvalues are linearly independent. If $A\mathbf{v} = \lambda_1\mathbf{v}$ and $A\mathbf{w} = \lambda_2\mathbf{w}$ with $\lambda_1 \neq \lambda_2$, then $\mathbf{v}$ and $\mathbf{w}$ cannot be parallel. Proof: suppose $\mathbf{v} = \alpha\mathbf{w}$ for some scalar $\alpha \neq 0$. Then $\lambda_1\mathbf{v} = A\mathbf{v} = \alpha A\mathbf{w} = \alpha\lambda_2\mathbf{w} = \lambda_2\mathbf{v}$, giving $(\lambda_1 - \lambda_2)\mathbf{v} = \mathbf{0}$. Since $\lambda_1 \neq \lambda_2$ and $\mathbf{v} \neq \mathbf{0}$, contradiction. More generally: eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ for distinct eigenvalues $\lambda_1, \ldots, \lambda_k$ are always linearly independent.

**For AI:** In PCA, the eigenspaces of the covariance matrix are the principal component subspaces. The first eigenspace (for $\lambda_1 = $ largest variance) gives the direction of greatest spread. In spectral clustering, the eigenspaces of the graph Laplacian partition the graph. In the analysis of attention heads, the eigenspace for eigenvalue 1 is the fixed-point subspace of the attention map.

### 2.5 The Spectrum

The **spectrum** of $A$ is the set of all distinct eigenvalues: $\sigma(A) = \{\lambda \in \mathbb{C} : \det(\lambda I - A) = 0\}$.

The **spectral radius** is $\rho(A) = \max\{|\lambda| : \lambda \in \sigma(A)\}$ — the largest absolute eigenvalue. It controls the long-run growth rate of $A^t$:

- $\rho(A) < 1$: $A^t \to 0$ as $t \to \infty$ (all trajectories of $\mathbf{x}_{t+1} = A\mathbf{x}_t$ converge to $\mathbf{0}$)
- $\rho(A) = 1$: trajectories are bounded (marginal stability, provided Jordan blocks for $|\lambda|=1$ eigenvalues have size 1)
- $\rho(A) > 1$: some trajectories grow without bound (unstable)

For real matrices, complex eigenvalues always come in **conjugate pairs**: if $\lambda = a + bi \in \sigma(A)$ then $\bar{\lambda} = a - bi \in \sigma(A)$, with corresponding conjugate eigenvectors. This means real matrices of odd dimension always have at least one real eigenvalue.

**For AI:** The spectral radius of the recurrent weight matrix $W$ in an RNN determines stability: $\rho(W) < 1$ causes vanishing gradients; $\rho(W) > 1$ causes explosion; $\rho(W) = 1$ is the critical point targeted by orthogonal RNN designs. The spectral radius of the attention matrix $S$ is always 1 (Perron-Frobenius). The spectral radius of $I - \eta H$ (where $H$ is the Hessian) determines gradient descent convergence.

---

## 3. Computing Eigenvalues and Eigenvectors

### 3.1 The Two-Step Process

Computing eigenvalues and eigenvectors follows a universal two-step recipe:

**Step 1 — Find eigenvalues:** Solve the characteristic equation $\det(\lambda I - A) = 0$. This gives a degree-$n$ polynomial whose roots are the eigenvalues $\lambda_1, \ldots, \lambda_n$ (with algebraic multiplicity).

**Step 2 — Find eigenvectors:** For each eigenvalue $\lambda_i$, solve the homogeneous linear system $(A - \lambda_i I)\mathbf{v} = \mathbf{0}$. Row-reduce $[A - \lambda_i I \mid \mathbf{0}]$ to RREF. The free variables give the null space basis vectors — these are the eigenvectors for $\lambda_i$.

This procedure is exact for small matrices. For $n \geq 5$, the Abel-Ruffini theorem guarantees no algebraic formula for roots of degree-5 polynomials, so numerical methods (Section 3.7) are mandatory in practice.

### 3.2 The 2×2 Case

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$, the characteristic polynomial is $p(\lambda) = \lambda^2 - (a+d)\lambda + (ad-bc)$, giving:

$$\lambda_{1,2} = \frac{(a+d) \pm \sqrt{(a-d)^2 + 4bc}}{2}$$

**Worked example.** $A = \begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$.

- $p(\lambda) = \lambda^2 - 6\lambda + 8 = (\lambda-4)(\lambda-2)$; eigenvalues $\lambda_1 = 4$, $\lambda_2 = 2$.
- For $\lambda_1 = 4$: $(A - 4I) = \begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & -1 \\ 0 & 0 \end{pmatrix}$; eigenvector $\mathbf{v}_1 = (1, 1)^\top$.
- For $\lambda_2 = 2$: $(A - 2I) = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 1 \\ 0 & 0 \end{pmatrix}$; eigenvector $\mathbf{v}_2 = (1, -1)^\top$.
- Check: $\lambda_1 + \lambda_2 = 6 = \text{tr}(A)$ ✓; $\lambda_1 \cdot \lambda_2 = 8 = \det(A)$ ✓; $\mathbf{v}_1 \perp \mathbf{v}_2$ ✓ (symmetric matrix).

### 3.3 The 3×3 and Higher Cases

For a $3 \times 3$ matrix $A$, expand $\det(\lambda I - A)$ by cofactor expansion along any row or column:

$$p(\lambda) = \lambda^3 - \text{tr}(A)\,\lambda^2 + \frac{1}{2}[(\text{tr}(A))^2 - \text{tr}(A^2)]\,\lambda - \det(A)$$

The coefficient of $\lambda$ equals the sum of the three $2 \times 2$ principal minors $M_{11} + M_{22} + M_{33}$ (where $M_{ii}$ is the determinant of $A$ with row $i$ and column $i$ deleted). Factor the cubic if possible; otherwise use the cubic formula or numerical root-finding.

For $n \geq 4$: expand the determinant directly; factor by guessing integer roots; use the rational root theorem. For $n \geq 5$: numerical methods are mandatory (Section 3.7).

**Key shortcut — triangular and diagonal matrices.** If $A$ is upper triangular, lower triangular, or diagonal, then $\det(\lambda I - A) = \prod_{i=1}^n (\lambda - A_{ii})$. The eigenvalues are exactly the diagonal entries. No expansion needed.

**Key shortcut — block diagonal matrices.** If $A = \text{diag}(B_1, B_2, \ldots, B_k)$, then $p(\lambda) = \prod_j p_{B_j}(\lambda)$. The eigenvalues of $A$ are the union of eigenvalues of each block.

### 3.4 Worked Example — 3×3 Defective Matrix

Let $A = \begin{pmatrix} 2 & 1 & 0 \\ 0 & 2 & 1 \\ 0 & 0 & 3 \end{pmatrix}$ (upper triangular).

**Step 1:** $p(\lambda) = (\lambda-2)^2(\lambda-3)$. Eigenvalues: $\lambda_1 = 2$ (algebraic multiplicity 2), $\lambda_2 = 3$ (multiplicity 1).

**Step 2a — eigenvectors for $\lambda_1 = 2$:**

$$A - 2I = \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 1 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

Free variable: $x_1$; forced: $x_2 = 0$, $x_3 = 0$. Eigenvector: $\mathbf{v}_1 = (1,0,0)^\top$. Geometric multiplicity $m_g(\lambda_1) = 1 < m_a(\lambda_1) = 2$. **Defective eigenvalue.** The matrix cannot be diagonalised.

**Step 2b — eigenvectors for $\lambda_2 = 3$:**

$$A - 3I = \begin{pmatrix} -1 & 1 & 0 \\ 0 & -1 & 1 \\ 0 & 0 & 0 \end{pmatrix} \xrightarrow{\text{RREF}} \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & -1 \\ 0 & 0 & 0 \end{pmatrix}$$

Free variable: $x_3$; $x_1 = x_3$, $x_2 = x_3$. Eigenvector: $\mathbf{v}_2 = (1,1,1)^\top$. Geometric multiplicity $m_g(\lambda_2) = 1 = m_a(\lambda_2)$. Non-defective.

**Conclusion:** This matrix has only 2 linearly independent eigenvectors for 3 eigenvalues (counting multiplicity). Diagonalisation fails; Section 7 covers the Jordan form needed here.

### 3.5 Eigenvalues of Special Matrices

A table of shortcuts that apply throughout the course:

| Matrix type | Condition | Eigenvalue property |
|---|---|---|
| Triangular (upper/lower) | $A_{ij} = 0$ for $i > j$ (or $i < j$) | Eigenvalues = diagonal entries $A_{ii}$ |
| Diagonal | $A = \text{diag}(d_1,\ldots,d_n)$ | Eigenvalues $= d_i$; eigenvectors $= \mathbf{e}_i$ |
| Symmetric | $A = A^\top$ | All $\lambda_i \in \mathbb{R}$; eigenvectors orthogonal |
| Skew-symmetric | $A = -A^\top$ | All $\lambda_i \in i\mathbb{R}$ (purely imaginary or 0) |
| Orthogonal | $Q^\top Q = I$ | All $|\lambda_i| = 1$ |
| Projection | $P^2 = P$, $P^\top = P$ | $\lambda_i \in \{0, 1\}$ |
| Nilpotent | $A^k = 0$ for some $k$ | All $\lambda_i = 0$ |
| Row-stochastic | All entries $\geq 0$, row sums $= 1$ | $\lambda_1 = 1$; $|\lambda_i| \leq 1$ (Perron-Frobenius) |
| PSD | $A \succeq 0$ | All $\lambda_i \geq 0$ |
| PD | $A \succ 0$ | All $\lambda_i > 0$ |

### 3.6 Power Iteration

Power iteration computes the **dominant eigenvector** — the eigenvector corresponding to the eigenvalue with largest absolute value — at cost $O(n^2)$ per step (or $O(\text{nnz})$ for sparse matrices).

**Algorithm:** Given initial vector $\mathbf{x}_0 \in \mathbb{R}^n$ (random, not orthogonal to $\mathbf{v}_1$):

```
For k = 1, 2, 3, …:
    y_k = A x_{k-1}              (apply A — one matrix-vector multiply)
    x_k = y_k / ‖y_k‖            (normalise to unit length)
    λ_k = x_kᵀ A x_k             (Rayleigh quotient estimate)
    stop when |λ_k − λ_{k-1}| < tolerance
```

**Convergence:** The iterates $\mathbf{x}_k \to \mathbf{v}_1$ at geometric rate $|\lambda_2/\lambda_1|$ per step. If the spectral gap $|\lambda_1| - |\lambda_2|$ is large, convergence is rapid; if the gap is small, convergence is slow.

**Requirements:** $\lambda_1$ must be strictly larger in absolute value than all other eigenvalues; $\mathbf{x}_0$ must have a non-zero component in the $\mathbf{v}_1$ direction (almost surely true for a random initialisation).

**For AI:** Power iteration underlies PageRank (dominant eigenvector of the link transition matrix), spectral initialisation of word embeddings, and efficient estimation of the spectral norm $\sigma_{\max}(W) = \sqrt{\lambda_{\max}(W^\top W)}$ for spectral normalisation in GANs and normalising flows. One step of power iteration per gradient update costs only $O(mn)$ and gives a running estimate of $\sigma_{\max}$.

### 3.7 The QR Algorithm

The QR algorithm is the practical standard for computing all eigenvalues of a dense matrix. It is one of the top 10 algorithms of the 20th century (Dongarra and Sullivan, 2000) and underlies `numpy.linalg.eig`, `scipy.linalg.eig`, and LAPACK's `dgeev`/`dsyev`.

**Basic QR iteration:**

```
A₀ = A
For k = 1, 2, 3, …:
    Aₖ₋₁ = Qₖ Rₖ     (QR factorisation of current matrix)
    Aₖ   = Rₖ Qₖ      (reverse-multiply; Aₖ is similar to Aₖ₋₁)
```

Each $A_k = Q_k^\top A_{k-1} Q_k$ is orthogonally similar to $A$ and thus has the same eigenvalues. The sequence $A_k$ converges to an upper triangular (Schur) form whose diagonal entries are the eigenvalues.

**Practical QR algorithm:** Reduce $A$ to Hessenberg form first (zero below the first sub-diagonal) using Householder reflections — $O(n^3)$ once. QR steps on a Hessenberg matrix cost $O(n^2)$ each. With Wilkinson shifts (shifting by an estimate of the bottom eigenvalue), convergence is typically cubic. Total cost: $O(n^3)$.

**For AI:** The QR algorithm is called every time you run `np.linalg.eig(A)` or `np.linalg.eigh(A)` (the symmetric version). Computing the full eigendecomposition of the Hessian approximation in K-FAC uses block-wise QR. Eigenvalue decomposition of the attention weight matrix $W_O W_V$ for mechanistic interpretability uses `np.linalg.eig`.

### 3.8 Lanczos Algorithm for Large Sparse Matrices

For large sparse symmetric matrices (e.g., graph Laplacians with $n = 10^6$ nodes), computing all eigenvalues via QR is $O(n^3)$ — completely infeasible. The Lanczos algorithm computes the $k$ extreme eigenvalues (largest or smallest) at cost $O(k \cdot \text{nnz})$ where $\text{nnz}$ is the number of non-zeros.

**Core idea:** Build a Krylov subspace $\mathcal{K}_k(A, \mathbf{b}) = \text{span}\{\mathbf{b}, A\mathbf{b}, A^2\mathbf{b}, \ldots, A^{k-1}\mathbf{b}\}$. The eigenvalues of the $k \times k$ tridiagonal projection $T_k$ (called Ritz values) converge to the extreme eigenvalues of $A$ as $k$ grows. Typically $k \ll n$.

**Algorithm (symmetric Lanczos):**

```
β₀ = 0, q₀ = 0, q₁ = b/‖b‖
For j = 1, 2, …, k:
    z     = A qⱼ
    αⱼ    = qⱼᵀ z
    z     ← z − αⱼ qⱼ − βⱼ₋₁ qⱼ₋₁
    βⱼ    = ‖z‖;  qⱼ₊₁ = z / βⱼ
T_k = tridiag(α₁,…,αₖ; β₁,…,βₖ₋₁)
```

The Ritz values (eigenvalues of $T_k$) converge to extreme eigenvalues of $A$ at exponential rate. In practice, re-orthogonalisation is needed to combat floating-point accumulation of rounding errors.

**For AI:** Lanczos is used to compute top-$r$ eigenvectors of large graph Laplacians for spectral clustering and graph neural network positional encodings; to estimate the Hessian eigenspectrum of large neural networks (PyHessian, HessianFlow); to compute dominant eigenvectors of large kernel matrices (Nyström approximation); and to analyse the spectral structure of attention weight matrices in large transformers.

---

## 4. Properties of Eigenvalues

### 4.1 Trace and Determinant Relations

Two fundamental identities connect eigenvalues to matrix invariants computable without solving the characteristic equation:

$$\sum_{i=1}^n \lambda_i = \text{tr}(A) \qquad \prod_{i=1}^n \lambda_i = \det(A)$$

**Proof (trace).** The characteristic polynomial is $p(\lambda) = \prod_{i=1}^n (\lambda - \lambda_i)$. Expanding: the coefficient of $\lambda^{n-1}$ is $-\sum_i \lambda_i$. Expanding $\det(\lambda I - A)$ directly: the coefficient of $\lambda^{n-1}$ comes only from the product of diagonal terms $(\lambda - A_{11})(\lambda - A_{22})\cdots(\lambda - A_{nn})$, giving $-\text{tr}(A)$ for the coefficient of $\lambda^{n-1}$. Hence $\sum_i \lambda_i = \text{tr}(A)$.

**Proof (determinant).** $p(0) = \det(-A) = (-1)^n \det(A)$. Also $p(0) = \prod_i (0 - \lambda_i) = (-1)^n \prod_i \lambda_i$. Hence $\det(A) = \prod_i \lambda_i$.

These provide quick sanity checks for eigenvalue computations and theoretical tools:

**For AI:** $\text{tr}(\nabla^2 \mathcal{L}) = \sum_i \lambda_i$ is the total curvature of the loss surface; it can be estimated cheaply via Hutchinson's trace estimator (random vector $\mathbf{z}$: $\text{tr}(H) \approx \mathbf{z}^\top H \mathbf{z}$ for $\mathbf{z} \sim \mathcal{N}(0,I)$). $\log\det(C) = \sum_i \log\lambda_i$ appears in Gaussian log-likelihoods and normalising flow log-densities. $\text{tr}(A^\top A) = \sum_i \sigma_i^2 = \|A\|_F^2$ for symmetric PSD matrices where $\sigma_i = \lambda_i$.

### 4.2 Eigenvalues Under Matrix Operations

A powerful feature of eigenvalues: many matrix operations transform eigenvalues in simple, predictable ways — without changing the eigenvectors.

| Operation | Eigenvalues | Eigenvectors | Proof sketch |
|---|---|---|---|
| $\alpha A$ | $\alpha\lambda_i$ | same $\mathbf{v}_i$ | $(\alpha A)\mathbf{v} = \alpha(A\mathbf{v}) = \alpha\lambda\mathbf{v}$ |
| $A^k$ | $\lambda_i^k$ | same $\mathbf{v}_i$ | Induction: $A^k\mathbf{v} = A^{k-1}(\lambda\mathbf{v}) = \lambda^k\mathbf{v}$ |
| $A^{-1}$ (if exists) | $1/\lambda_i$ | same $\mathbf{v}_i$ | $A^{-1}\mathbf{v} = A^{-1}(A\mathbf{v}/\lambda) = \mathbf{v}/\lambda$ |
| $p(A) = \sum c_k A^k$ | $p(\lambda_i)$ | same $\mathbf{v}_i$ | Linearity + $A^k$ rule |
| $A^\top$ | same $\lambda_i$ | different (left eigenvectors of $A$) | $\det(\lambda I - A^\top) = \det(\lambda I - A)$ |
| $P^{-1}AP$ | same $\lambda_i$ | $P^{-1}\mathbf{v}_i$ | $\det(\lambda I - P^{-1}AP) = \det(\lambda I - A)$ |
| $A + \mu I$ | $\lambda_i + \mu$ | same $\mathbf{v}_i$ | $(A+\mu I)\mathbf{v} = \lambda\mathbf{v} + \mu\mathbf{v} = (\lambda+\mu)\mathbf{v}$ |

**For AI:** The eigenvalues of $I - \eta H$ are $1 - \eta\lambda_i(H)$ — this is the key formula for gradient descent convergence. The eigenvalues of $e^{At}$ are $e^{\lambda_i t}$ — this governs neural ODE dynamics. Preconditioning replaces $H$ with $P^{-1}H$ to give better-conditioned eigenvalues.

### 4.3 Eigenvalues of Special Matrices

**Symmetric matrices ($A = A^\top$):** All eigenvalues are real; eigenvectors for distinct eigenvalues are orthogonal; always diagonalisable with an orthonormal eigenbasis. (Full proof in Section 6.)

**Positive semidefinite ($A \succeq 0$):** All $\lambda_i \geq 0$. Equivalent characterisations: $\mathbf{x}^\top A\mathbf{x} \geq 0$ for all $\mathbf{x}$; $A = B^\top B$ for some $B$; all principal minors non-negative.

**Orthogonal ($Q^\top Q = I$):** All $|\lambda_i| = 1$. Over $\mathbb{R}$: eigenvalues are $\pm 1$. Over $\mathbb{C}$: eigenvalues lie on the unit circle. Proof: $\|Q\mathbf{v}\| = \|\mathbf{v}\|$ (orthogonal maps preserve length) and $\|Q\mathbf{v}\| = |\lambda|\|\mathbf{v}\|$ so $|\lambda| = 1$.

**Projection ($P^2 = P$, $P^\top = P$):** Eigenvalues $\in \{0, 1\}$. The 1-eigenspace is $\text{col}(P)$; the 0-eigenspace is $\text{null}(P)$. Proof: $P\mathbf{v} = \lambda\mathbf{v}$ implies $P^2\mathbf{v} = \lambda P\mathbf{v} = \lambda^2\mathbf{v}$; but $P^2 = P$ so $\lambda^2 = \lambda$; hence $\lambda \in \{0,1\}$.

**Row-stochastic ($A\mathbf{1} = \mathbf{1}$, $A_{ij} \geq 0$):** By Perron-Frobenius, the dominant eigenvalue is $\lambda_1 = 1$ with eigenvector $\mathbf{1}$; all other eigenvalues satisfy $|\lambda_i| \leq 1$. The attention weight matrix $S = \text{softmax}(QK^\top/\sqrt{d_k})$ is row-stochastic, so its largest eigenvalue is always 1.

### 4.4 The Rayleigh Quotient

For a symmetric matrix $A \in \mathbb{R}^{n \times n}$ and any non-zero $\mathbf{x} \in \mathbb{R}^n$, the **Rayleigh quotient** is:

$$R(A, \mathbf{x}) = \frac{\mathbf{x}^\top A \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}$$

It is bounded between the smallest and largest eigenvalues:

$$\lambda_{\min}(A) \leq R(A, \mathbf{x}) \leq \lambda_{\max}(A)$$

**Proof.** Write $\mathbf{x} = \sum_i c_i \mathbf{q}_i$ in the orthonormal eigenbasis $\{\mathbf{q}_i\}$. Then $\mathbf{x}^\top A \mathbf{x} = \sum_i \lambda_i c_i^2$ and $\mathbf{x}^\top \mathbf{x} = \sum_i c_i^2$, giving $R = \frac{\sum_i \lambda_i c_i^2}{\sum_i c_i^2}$ — a weighted average of eigenvalues with non-negative weights $c_i^2/\sum_j c_j^2$ summing to 1. A weighted average is bounded by the min and max of its terms. Equality at $\mathbf{x} = \mathbf{q}_1$ gives $R = \lambda_{\max}$; at $\mathbf{x} = \mathbf{q}_n$ gives $R = \lambda_{\min}$.

This gives the **variational characterisation** of extreme eigenvalues:

$$\lambda_{\max}(A) = \max_{\mathbf{x} \neq \mathbf{0}} R(A,\mathbf{x}) = \max_{\|\mathbf{x}\|=1} \mathbf{x}^\top A \mathbf{x}$$
$$\lambda_{\min}(A) = \min_{\mathbf{x} \neq \mathbf{0}} R(A,\mathbf{x}) = \min_{\|\mathbf{x}\|=1} \mathbf{x}^\top A \mathbf{x}$$

**For AI:** The PCA objective — find the direction of maximum variance — is exactly $\max_{\|\mathbf{w}\|=1} \mathbf{w}^\top C \mathbf{w} = \lambda_{\max}(C)$, achieved at $\mathbf{w} = \mathbf{q}_1$. The curvature of the loss in direction $\mathbf{d}$ is $R(\nabla^2 \mathcal{L}, \mathbf{d})$; the maximum curvature is $\lambda_{\max}(H)$; the minimum is $\lambda_{\min}(H)$. Rayleigh quotient iteration refines an eigenvector estimate $\mathbf{x}_k$ by shifting: solve $(A - R_k I)\mathbf{y} = \mathbf{x}_k$, renormalise — this gives cubic convergence to the nearest eigenvector.

### 4.5 The Courant-Fischer Min-Max Theorem

The Rayleigh quotient gives $\lambda_{\max}$ and $\lambda_{\min}$. The **Courant-Fischer theorem** generalises this to all eigenvalues. For symmetric $A$ with eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$:

$$\lambda_k = \max_{\dim(W)=k} \min_{\mathbf{x} \in W,\, \mathbf{x} \neq \mathbf{0}} R(A,\mathbf{x})$$

Equivalently, $\lambda_k$ is the best minimum Rayleigh quotient you can guarantee if you choose a $k$-dimensional subspace $W$ optimally.

**Weyl's inequalities** give eigenvalue stability under perturbation: for symmetric $A$ and symmetric perturbation $E$:

$$\lambda_k(A) + \lambda_n(E) \leq \lambda_k(A+E) \leq \lambda_k(A) + \lambda_1(E)$$

In particular, $|\lambda_k(A+E) - \lambda_k(A)| \leq \|E\|_2 = \lambda_{\max}(|E|)$. Eigenvalues of symmetric matrices are Lipschitz in the matrix entries with constant 1 (in spectral norm).

**For AI:** Weyl's inequalities bound how much eigenvalues shift when model weights are perturbed (e.g., by quantisation noise or gradient updates). If the perturbation $E$ has $\|E\|_2 \leq \epsilon$, then no eigenvalue moves by more than $\epsilon$. This is why spectral properties of trained models are relatively robust to small weight perturbations.

### 4.6 Gershgorin Circle Theorem

The **Gershgorin Circle Theorem** localises eigenvalues using only the matrix entries — no eigenvalue computation required.

**Theorem.** For $A \in \mathbb{C}^{n \times n}$, every eigenvalue lies in at least one Gershgorin disc:

$$D_i = \left\{\lambda \in \mathbb{C} : |\lambda - A_{ii}| \leq R_i\right\}, \quad R_i = \sum_{j \neq i} |A_{ij}|$$

Each disc $D_i$ is centred at the diagonal entry $A_{ii}$ with radius equal to the sum of absolute values of the off-diagonal entries in row $i$. All eigenvalues: $\sigma(A) \subseteq \bigcup_{i=1}^n D_i$.

**Proof.** Let $\lambda$ be an eigenvalue with eigenvector $\mathbf{v}$. Choose $i$ such that $|v_i| = \|\mathbf{v}\|_\infty$ (the largest component). From $(A - \lambda I)\mathbf{v} = \mathbf{0}$ at row $i$: $(A_{ii} - \lambda)v_i = -\sum_{j \neq i} A_{ij}v_j$. Taking absolute values and dividing by $|v_i| > 0$: $|A_{ii} - \lambda| \leq \sum_{j \neq i} |A_{ij}| \cdot |v_j/v_i| \leq \sum_{j \neq i}|A_{ij}| = R_i$. So $\lambda \in D_i$.

**Corollary (diagonal dominance):** If $|A_{ii}| > R_i$ for all $i$, then all Gershgorin discs exclude the origin, so $0 \notin \sigma(A)$ — the matrix is invertible.

**For AI:** Gershgorin circles give fast bounds on the eigenvalue range of attention matrices, Hessians, or weight matrices without computing eigenvalues. If all discs lie in the right half-plane ($\text{Re}(\lambda) > 0$), the matrix is positive definite. Useful for verifying that a discretised ODE system is stable before running the forward pass.

---

## 5. Diagonalisation

### 5.1 The Diagonalisation Theorem

**Definition.** A matrix $A \in \mathbb{R}^{n \times n}$ is **diagonalisable** if there exists an invertible matrix $V$ and a diagonal matrix $\Lambda$ such that:

$$A = V\Lambda V^{-1}$$

where $V = [\mathbf{v}_1 \mid \mathbf{v}_2 \mid \cdots \mid \mathbf{v}_n]$ has the eigenvectors as columns and $\Lambda = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$ has the corresponding eigenvalues on the diagonal.

**Proof ($\Rightarrow$).** If $A\mathbf{v}_i = \lambda_i\mathbf{v}_i$ for $i=1,\ldots,n$ and the $\mathbf{v}_i$ are linearly independent, then $AV = V\Lambda$ (column $i$ of $AV$ is $A\mathbf{v}_i = \lambda_i\mathbf{v}_i$, which equals column $i$ of $V\Lambda$). Since $V$ is invertible (independent columns): $A = V\Lambda V^{-1}$.

**Proof ($\Leftarrow$).** If $A = V\Lambda V^{-1}$ then $AV = V\Lambda$; column $j$ gives $A\mathbf{v}_j = \lambda_j\mathbf{v}_j$; the columns of $V$ are eigenvectors. Since $V$ is invertible, they are linearly independent.

**Key insight:** Diagonalisation is a change of basis. In the eigenbasis (coordinates given by $V^{-1}\mathbf{x}$), the matrix $A$ acts as pure coordinate-wise scaling by $\lambda_i$ — no mixing between different directions. This is why diagonalisable matrices are "simple": in their natural basis, they do nothing but stretch.

### 5.2 When Is A Diagonalisable?

**Sufficient condition:** If $A$ has $n$ distinct eigenvalues, it is diagonalisable. Proof: eigenvectors for distinct eigenvalues are linearly independent (Section 2.4); $n$ independent vectors in $\mathbb{R}^n$ form a basis; $V$ is invertible.

**Necessary and sufficient:** $A$ is diagonalisable if and only if $m_g(\lambda_i) = m_a(\lambda_i)$ for every eigenvalue $\lambda_i$. Equivalently: the eigenspaces together span all of $\mathbb{R}^n$.

**Always diagonalisable:**
- Symmetric matrices $A = A^\top$ (Spectral Theorem, Section 6)
- Any matrix with $n$ distinct eigenvalues
- Normal matrices satisfying $AA^\top = A^\top A$

**Not necessarily diagonalisable:**
- Matrices with repeated eigenvalues where $m_g < m_a$ (defective)
- The $2 \times 2$ matrix $\begin{pmatrix} 2 & 1 \\ 0 & 2\end{pmatrix}$ has $\lambda = 2$ with $m_a = 2$, $m_g = 1$ — defective

**Non-examples of diagonalisability over $\mathbb{R}$:**
- Any real matrix with complex (non-real) eigenvalues (e.g., rotation matrices) cannot be diagonalised over $\mathbb{R}$, though it can over $\mathbb{C}$

### 5.3 Matrix Powers via Diagonalisation

If $A = V\Lambda V^{-1}$, then matrix powers become trivial:

$$A^k = V\Lambda^k V^{-1}, \qquad \text{where } \Lambda^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$$

**Proof.** $A^2 = (V\Lambda V^{-1})(V\Lambda V^{-1}) = V\Lambda(V^{-1}V)\Lambda V^{-1} = V\Lambda^2 V^{-1}$. By induction: $A^k = V\Lambda^k V^{-1}$.

**Computational gain.** Without diagonalisation: computing $A^k$ by repeated matrix multiplication costs $O(n^3 \log k)$ (via fast exponentiation). With diagonalisation: once $V$ and $\Lambda$ are known, $A^k \mathbf{x} = V\Lambda^k V^{-1}\mathbf{x}$ costs $O(n^2)$ per query (three matrix-vector products) for any $k$.

**Worked example — Fibonacci numbers.** The Fibonacci recurrence $F_{k+1} = F_k + F_{k-1}$ is a linear system: $\begin{pmatrix}F_{k+1}\\F_k\end{pmatrix} = \begin{pmatrix}1&1\\1&0\end{pmatrix}^k \begin{pmatrix}1\\0\end{pmatrix}$. Diagonalising $A = \begin{pmatrix}1&1\\1&0\end{pmatrix}$ gives eigenvalues $\phi = (1+\sqrt{5})/2 \approx 1.618$ (golden ratio) and $\hat\phi = (1-\sqrt{5})/2 \approx -0.618$. Therefore $F_k = (\phi^k - \hat\phi^k)/\sqrt{5}$ — Binet's formula. The dominant eigenvalue $\phi$ controls asymptotic growth: $F_k \approx \phi^k/\sqrt{5}$.

**For AI:** Matrix powers appear in RNN unrolling ($\mathbf{h}_t = W^t \mathbf{h}_0 + \ldots$); Markov chain mixing ($\pi_t = A^t \pi_0$); and computing the influence of past inputs on current hidden states. The eigendecomposition reveals which "modes" persist long-term (large $|\lambda_i|$) and which decay quickly (small $|\lambda_i|$).

### 5.4 Matrix Functions via Diagonalisation

For a diagonalisable matrix $A = V\Lambda V^{-1}$ and any function $f: \mathbb{C} \to \mathbb{C}$, define:

$$f(A) = V\, f(\Lambda)\, V^{-1} = V\, \text{diag}(f(\lambda_1), \ldots, f(\lambda_n))\, V^{-1}$$

This is the natural extension of scalar functions to matrices: apply $f$ to each eigenvalue.

**Matrix exponential:** $\exp(A) = V\,\text{diag}(e^{\lambda_1},\ldots,e^{\lambda_n})\,V^{-1}$. The solution to $d\mathbf{x}/dt = A\mathbf{x}$ is $\mathbf{x}(t) = \exp(At)\mathbf{x}_0$. For AI: neural ODEs use $\exp(At)$ to propagate hidden states through continuous time; the stability of the ODE is governed by whether $\text{Re}(\lambda_i) < 0$ for all $i$.

**Matrix square root:** $A^{1/2} = V\,\text{diag}(\sqrt{\lambda_1},\ldots,\sqrt{\lambda_n})\,V^{-1}$ — requires $\lambda_i \geq 0$ (valid for PSD matrices). Used in preconditioning: the natural gradient update involves $F^{-1/2}$ where $F$ is the Fisher information matrix.

**Log-determinant:** $\log\det(A) = \text{tr}(\log A) = \sum_i \log\lambda_i$. Appears in: Gaussian log-likelihoods $-\frac{1}{2}\log\det(\Sigma) - \frac{1}{2}(\mathbf{x}-\mu)^\top\Sigma^{-1}(\mathbf{x}-\mu)$; normalising flow log-densities; variational inference ELBO terms. Efficient estimation via stochastic trace estimators when $n$ is large.

**Cayley-Hamilton theorem:** Every matrix satisfies its own characteristic polynomial: $p(A) = 0$. This means $A^n$ can be written as a linear combination of $I, A, \ldots, A^{n-1}$ — matrix polynomials can always be reduced to degree at most $n-1$.

### 5.5 Change of Basis Interpretation

Diagonalisation $A = V\Lambda V^{-1}$ encodes a three-step computation:

```
DIAGONALISATION AS CHANGE OF BASIS
════════════════════════════════════════════════════════════════════════

  Input x  ──► V⁻¹ x  ──►  Λ(V⁻¹x)  ──►  V Λ V⁻¹ x  =  Ax
           (express in    (scale each    (convert back
            eigenbasis)    coordinate     to standard
                           by λᵢ)        basis)

  In the eigenbasis, A acts as pure scaling — no coupling!

════════════════════════════════════════════════════════════════════════
```

This is the fundamental reason diagonalisable matrices are tractable: in the right coordinate system, they decouple completely. The $i$-th eigenvector direction is scaled by $\lambda_i$ and is independent of all other directions. No information leaks between eigenvector components.

**For AI:** PCA is precisely this change of basis. The eigenvectors $\mathbf{q}_1, \ldots, \mathbf{q}_d$ of the covariance matrix $C$ form a new coordinate system (the principal component basis). In this basis, $C$ is diagonal ($C = Q\Lambda Q^\top$, so $Q^\top C Q = \Lambda$): the features are uncorrelated. Each principal component direction is independent, scaled by its variance $\lambda_i$. Running PCA = expressing your data in the eigenbasis of its covariance.

### 5.6 Simultaneous Diagonalisation

Two symmetric matrices $A$ and $B$ are **simultaneously diagonalisable** — they share the same orthonormal eigenbasis — if and only if $AB = BA$ (they commute).

**Proof ($\Rightarrow$).** If $A = Q\Lambda_A Q^\top$ and $B = Q\Lambda_B Q^\top$ with the same $Q$: $AB = Q\Lambda_A Q^\top Q\Lambda_B Q^\top = Q\Lambda_A\Lambda_B Q^\top = Q\Lambda_B\Lambda_A Q^\top = BA$.

**Proof ($\Leftarrow$).** If $AB = BA$ and $A\mathbf{v} = \lambda\mathbf{v}$: $A(B\mathbf{v}) = B(A\mathbf{v}) = \lambda(B\mathbf{v})$, so $B\mathbf{v}$ is also in the $\lambda$-eigenspace of $A$. $B$ maps each eigenspace of $A$ into itself. On each eigenspace (a symmetric subspace for $B$), $B$ can be diagonalised. Choose an orthonormal basis for each eigenspace that also diagonalises $B$. Together these form a simultaneous eigenbasis.

**For AI:** Simultaneous diagonalisation appears in the analysis of optimisers. If the Hessian $H$ and the Fisher information matrix $F$ commute, they share an eigenbasis — in this basis, the natural gradient $F^{-1}\nabla\mathcal{L}$ is a simple eigenvalue-rescaled version of the gradient. K-FAC (Kronecker-Factored Approximate Curvature) builds a structured approximation to $F$ that is easier to diagonalise, enabling efficient natural gradient updates. Attention heads whose key-query weight matrices commute have aligned eigenspaces, simplifying their circuit analysis.

---

## 6. The Spectral Theorem

### 6.1 Statement and Significance

The Spectral Theorem is the crown jewel of applied linear algebra. It guarantees that symmetric matrices have a perfect, complete structure — one that makes them tractable both theoretically and computationally.

**Real Spectral Theorem.** For any symmetric matrix $A = A^\top \in \mathbb{R}^{n \times n}$:

1. All eigenvalues are real: $\lambda_i \in \mathbb{R}$.
2. Eigenvectors for distinct eigenvalues are orthogonal: $\lambda_i \neq \lambda_j \Rightarrow \mathbf{q}_i \perp \mathbf{q}_j$.
3. There always exist $n$ orthonormal eigenvectors, regardless of multiplicities.
4. $A$ is orthogonally diagonalisable: $A = Q\Lambda Q^\top$ where $Q$ is orthogonal ($Q^\top Q = QQ^\top = I$) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ is real diagonal.

The significance cannot be overstated. For a general matrix, diagonalisation may fail (defective case) or require a non-orthogonal $V$ (so $V^{-1} \neq V^\top$, making computations expensive). For symmetric matrices: diagonalisation always works, and $V^{-1} = V^\top$ (free, numerically stable). The eigenbasis is orthonormal — the most natural and well-conditioned basis possible.

**Why symmetry matters for AI:** Covariance matrices, Gram matrices, Hessians at local minima, graph Laplacians, and kernel matrices are all symmetric PSD. The Spectral Theorem applies to all of them, guaranteeing real non-negative eigenvalues and an orthonormal eigenbasis. This is why PCA, spectral clustering, and natural gradient methods are well-defined and numerically stable.

### 6.2 Proof Sketch

**All eigenvalues real.** Let $A\mathbf{v} = \lambda\mathbf{v}$ with $\mathbf{v} \in \mathbb{C}^n$. Take the complex inner product: $\bar{\mathbf{v}}^\top A\mathbf{v} = \lambda \bar{\mathbf{v}}^\top\mathbf{v} = \lambda\|\mathbf{v}\|^2$. Also, $\bar{\mathbf{v}}^\top A\mathbf{v} = (A\bar{\mathbf{v}})^\top\mathbf{v} = \overline{A\mathbf{v}}^\top\mathbf{v} = \bar{\lambda}\bar{\mathbf{v}}^\top\mathbf{v} = \bar{\lambda}\|\mathbf{v}\|^2$ (using $A = A^\top$ real). So $\lambda = \bar{\lambda}$, meaning $\lambda$ is real. $\square$

**Eigenvectors for distinct eigenvalues are orthogonal.** Let $A\mathbf{v}_1 = \lambda_1\mathbf{v}_1$ and $A\mathbf{v}_2 = \lambda_2\mathbf{v}_2$ with $\lambda_1 \neq \lambda_2$. Then:
$$\lambda_1\langle\mathbf{v}_1,\mathbf{v}_2\rangle = \langle A\mathbf{v}_1,\mathbf{v}_2\rangle = \langle\mathbf{v}_1,A\mathbf{v}_2\rangle = \lambda_2\langle\mathbf{v}_1,\mathbf{v}_2\rangle$$
(using $\langle A\mathbf{u},\mathbf{v}\rangle = \langle\mathbf{u},A\mathbf{v}\rangle$ — self-adjointness of $A$). Since $\lambda_1 \neq \lambda_2$: $\langle\mathbf{v}_1,\mathbf{v}_2\rangle = 0$. $\square$

**Existence of orthonormal eigenbasis.** By induction on $n$. $A$ has at least one real eigenvalue $\lambda_1$ (proved above). Let $\mathbf{q}_1$ be a unit eigenvector. Consider the restriction of $A$ to $\mathbf{q}_1^\perp$ (the orthogonal complement). This restriction is again symmetric (since $A$ maps $\mathbf{q}_1^\perp$ to itself: if $\mathbf{x} \perp \mathbf{q}_1$ then $\langle A\mathbf{x}, \mathbf{q}_1\rangle = \langle\mathbf{x}, A\mathbf{q}_1\rangle = \lambda_1\langle\mathbf{x},\mathbf{q}_1\rangle = 0$). Apply induction to this $(n-1)$-dimensional restriction. $\square$

### 6.3 The Spectral Decomposition

From $A = Q\Lambda Q^\top$ with columns $\mathbf{q}_1,\ldots,\mathbf{q}_n$:

$$A = \sum_{i=1}^n \lambda_i\, \mathbf{q}_i \mathbf{q}_i^\top$$

Each term $\lambda_i\mathbf{q}_i\mathbf{q}_i^\top$ is a rank-1 symmetric matrix — a scaled orthogonal projection onto $\text{span}\{\mathbf{q}_i\}$. The full matrix $A$ is a weighted sum of $n$ rank-1 orthogonal projections, with weights equal to the eigenvalues.

The projectors $P_i = \mathbf{q}_i\mathbf{q}_i^\top$ satisfy: $P_i^2 = P_i$ (projection); $P_i^\top = P_i$ (symmetric); $P_iP_j = 0$ for $i \neq j$ (orthogonality); $\sum_i P_i = I$ (completeness). These are the building blocks of the spectral decomposition.

**For repeated eigenvalues:** If $\lambda$ has multiplicity $k$ with orthonormal eigenvectors $\mathbf{q}_1,\ldots,\mathbf{q}_k$, the contribution is $\lambda \sum_{j=1}^k \mathbf{q}_j\mathbf{q}_j^\top = \lambda P_{E(\lambda)}$ — the eigenvalue times the orthogonal projection onto the eigenspace.

**For AI:** The spectral decomposition of the covariance matrix $C = \sum_i \lambda_i\mathbf{q}_i\mathbf{q}_i^\top$ reveals PCA directly: each term $\lambda_i\mathbf{q}_i\mathbf{q}_i^\top$ is one principal component scaled by its variance. Keeping the top-$k$ terms gives the best rank-$k$ approximation to $C$ (Eckart-Young). The spectral decomposition of the Hessian $H = \sum_i \lambda_i\mathbf{q}_i\mathbf{q}_i^\top$ decomposes the loss curvature into independent directions: gradient descent along $\mathbf{q}_i$ converges at rate $(1-\eta\lambda_i)^t$, independent of all other directions.

### 6.4 Positive (Semi)definite Matrices via the Spectral Theorem

A symmetric matrix $A$ is **positive semidefinite** (PSD, written $A \succeq 0$) iff all eigenvalues are non-negative; **positive definite** (PD, $A \succ 0$) iff all eigenvalues are strictly positive.

**Proof.** Using the spectral decomposition: $\mathbf{x}^\top A\mathbf{x} = \mathbf{x}^\top\!\left(\sum_i \lambda_i\mathbf{q}_i\mathbf{q}_i^\top\right)\!\mathbf{x} = \sum_i \lambda_i(\mathbf{q}_i^\top\mathbf{x})^2$. This is a sum of $\lambda_i$ weighted by non-negative squares $(\mathbf{q}_i^\top\mathbf{x})^2$. The sum is $\geq 0$ for all $\mathbf{x}$ iff all $\lambda_i \geq 0$; strictly $> 0$ for all $\mathbf{x} \neq \mathbf{0}$ iff all $\lambda_i > 0$.

**Equivalent characterisations of PSD:**
- $A = B^\top B$ for some matrix $B$ (Gram matrix form)
- All principal minors are non-negative
- Cholesky decomposition $A = LL^\top$ exists (with positive diagonal $L$ for PD)

**Condition number:** For PD matrix $A$: $\kappa_2(A) = \lambda_{\max}/\lambda_{\min}$. Measures how elongated the ellipsoid $\{\mathbf{x}: \mathbf{x}^\top A\mathbf{x} = 1\}$ is. Large $\kappa_2$: ill-conditioned; gradient descent converges slowly.

**For AI:** Covariance matrices and Gram matrices are always PSD (they are of the form $X^\top X$). The Hessian at a local minimum is PSD; at a saddle point, it has both positive and negative eigenvalues. Checking PSD via eigenvalues is the standard diagnostic: if `np.linalg.eigvalsh(H).min() < 0`, the critical point is not a local minimum.

### 6.5 The Complex Spectral Theorem

For complex **Hermitian** matrices $A = A^* = \bar{A}^\top$ (conjugate transpose), the same theorem holds over $\mathbb{C}$: all eigenvalues are real; eigenvectors for distinct eigenvalues are orthogonal under the complex inner product; $A$ is unitarily diagonalisable: $A = U\Lambda U^*$ where $U$ is unitary ($U^*U = I$).

**For AI:** Quantum computing represents observables as Hermitian operators; measurement outcomes are eigenvalues (always real). Complex-valued neural networks and unitary RNNs use Hermitian or unitary weight matrices to ensure stable, norm-preserving dynamics. The Fourier transform is a unitary operator whose eigenvalues are complex roots of unity — the eigenfunctions are pure sinusoids.

---

## 7. Jordan Normal Form

### 7.1 Motivation — Defective Matrices

Not every matrix is diagonalisable. When $m_g(\lambda_i) < m_a(\lambda_i)$ for some eigenvalue, we lack enough eigenvectors to form a basis. The Jordan Normal Form (JNF) is the canonical replacement for diagonalisation — it works for every square matrix over $\mathbb{C}$, whether diagonalisable or not.

The JNF tells us exactly what the "obstruction" to diagonalisation is: instead of a pure scaling $\lambda$ in each coordinate, we get a scaling $\lambda$ plus a "coupling" to the next coordinate — a **Jordan block**.

### 7.2 Jordan Blocks

A **Jordan block** of size $k$ for eigenvalue $\lambda$ is:

$$J_k(\lambda) = \begin{pmatrix} \lambda & 1 & 0 & \cdots & 0 \\ 0 & \lambda & 1 & \cdots & 0 \\ \vdots & & \ddots & \ddots & \vdots \\ 0 & 0 & \cdots & \lambda & 1 \\ 0 & 0 & \cdots & 0 & \lambda \end{pmatrix} \in \mathbb{R}^{k \times k}$$

Diagonal: all $\lambda$. Superdiagonal: all 1. Everything else: 0.

- $J_1(\lambda) = [\lambda]$: ordinary eigenvalue, non-defective.
- $J_k(\lambda)$ for $k > 1$: exactly one eigenvector $\mathbf{e}_1 = (1,0,\ldots,0)^\top$; remaining $k-1$ vectors are **generalised eigenvectors** satisfying $(A-\lambda I)\mathbf{v}_j = \mathbf{v}_{j-1}$ for $j = 2,\ldots,k$.

### 7.3 The Jordan Normal Form

**Theorem.** For any $A \in \mathbb{C}^{n \times n}$, there exists an invertible $P$ such that:

$$A = PJP^{-1}, \qquad J = \begin{pmatrix} J_{k_1}(\lambda_{i_1}) & & \\ & J_{k_2}(\lambda_{i_2}) & \\ & & \ddots \end{pmatrix}$$

$J$ is block diagonal with Jordan blocks; $\sum_j k_j = n$. For each eigenvalue $\lambda$: the number of Jordan blocks equals $m_g(\lambda)$; the sizes are determined by the rank sequence of $(A-\lambda I)^k$.

The JNF is the unique canonical form for matrices up to similarity: two matrices are similar ($B = PAP^{-1}$ for some invertible $P$) iff they have the same Jordan form (up to reordering blocks).

### 7.4 Powers of Jordan Blocks

$$J_k(\lambda)^t = \begin{pmatrix} \binom{t}{0}\lambda^t & \binom{t}{1}\lambda^{t-1} & \binom{t}{2}\lambda^{t-2} & \cdots \\ 0 & \binom{t}{0}\lambda^t & \binom{t}{1}\lambda^{t-1} & \cdots \\ \vdots & & \ddots & \\ 0 & \cdots & 0 & \binom{t}{0}\lambda^t \end{pmatrix}$$

The entries grow as $\binom{t}{j}\lambda^{t-j} \sim t^j \lambda^{t-j}/j!$ for fixed $j$.

- $|\lambda| < 1$: all entries $\to 0$ despite polynomial factor $t^{k-1}$ — stable convergence.
- $|\lambda| = 1$, $k > 1$: entries grow polynomially as $t^{k-1}$ — **marginal instability**.
- $|\lambda| > 1$: exponential growth — unstable.

**For AI:** An RNN with $\rho(W) = 1$ exactly (at the stability boundary) is dangerous if $W$ has a Jordan block of size $> 1$ for the unit eigenvalue — the hidden state grows polynomially even though no eigenvalue exceeds 1. LSTM forget gates aim to keep the effective spectral radius exactly 1 while avoiding non-trivial Jordan blocks. Gradient clipping and spectral normalisation are practical safeguards against this polynomial growth.

---

## 8. Singular Value Decomposition

### 8.1 SVD as Generalisation of Eigendecomposition

Eigendecomposition $A = V\Lambda V^{-1}$ requires $A$ to be square and diagonalisable. The **Singular Value Decomposition** (SVD) generalises to any matrix $A \in \mathbb{R}^{m \times n}$:

$$A = U\Sigma V^\top$$

where $U \in \mathbb{R}^{m \times m}$ is orthogonal (left singular vectors), $\Sigma \in \mathbb{R}^{m \times n}$ is "diagonal" with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$ (singular values), and $V \in \mathbb{R}^{n \times n}$ is orthogonal (right singular vectors).

SVD always exists and is unique (when singular values are distinct). It applies to rectangular matrices, rank-deficient matrices, and matrices with repeated or zero singular values — no assumptions needed.

### 8.2 Connection to Eigenvalues

SVD and eigendecomposition are deeply linked via the symmetric matrices $A^\top A$ and $AA^\top$:

- $A^\top A \in \mathbb{R}^{n \times n}$ is symmetric PSD; its eigenvalues are $\sigma_i^2$; its orthonormal eigenvectors are the **right singular vectors** $\mathbf{v}_i$ (columns of $V$): $A^\top A\,\mathbf{v}_i = \sigma_i^2\,\mathbf{v}_i$.
- $AA^\top \in \mathbb{R}^{m \times m}$ is symmetric PSD; its non-zero eigenvalues are the same $\sigma_i^2$; its orthonormal eigenvectors are the **left singular vectors** $\mathbf{u}_i$ (columns of $U$): $AA^\top\,\mathbf{u}_i = \sigma_i^2\,\mathbf{u}_i$.
- The connection: $A\mathbf{v}_i = \sigma_i\mathbf{u}_i$ and $A^\top\mathbf{u}_i = \sigma_i\mathbf{v}_i$.

For a square symmetric PSD matrix: singular values = eigenvalues; $U = V = Q$ (the orthogonal eigenvector matrix); SVD = eigendecomposition.

### 8.3 Thin and Truncated SVD

**Full SVD:** $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$, $V \in \mathbb{R}^{n \times n}$. Includes singular vectors for zero singular values (null space directions).

**Thin (economy) SVD** for $m \geq n$: $U_n \in \mathbb{R}^{m \times n}$, $\Sigma_n \in \mathbb{R}^{n \times n}$ (square), $V \in \mathbb{R}^{n \times n}$. Equivalent for computing $A$ but smaller: `np.linalg.svd(A, full_matrices=False)`.

**Truncated rank-$r$ SVD:** Keep only the top $r$ singular triplets:

$$A_r = \sum_{i=1}^r \sigma_i\,\mathbf{u}_i\mathbf{v}_i^\top = U_r\Sigma_r V_r^\top$$

**Eckart-Young Theorem:** $A_r$ is the best rank-$r$ approximation to $A$ in both spectral norm and Frobenius norm:
$$\|A - A_r\|_2 = \sigma_{r+1}, \qquad \|A - A_r\|_F = \sqrt{\sigma_{r+1}^2 + \cdots + \sigma_{\min(m,n)}^2}$$

No other rank-$r$ matrix is closer to $A$. This is the mathematical foundation for LoRA, PCA, image compression, and recommender systems.

### 8.4 Geometric Interpretation

The SVD $A = U\Sigma V^\top$ decomposes any linear map into three pure operations:

```
SVD GEOMETRIC DECOMPOSITION
════════════════════════════════════════════════════════════════════════

  x  ──► Vᵀx  ──────► Σ(Vᵀx)  ──────► U Σ Vᵀ x  =  Ax
     (rotate/       (scale each      (rotate/reflect
      reflect        coordinate       into output
      input          by σᵢ)          space)
      basis)

  Unit sphere ──► (same orientation) ──► ellipsoid
                                         semi-axes: σ₁u₁, σ₂u₂, …

════════════════════════════════════════════════════════════════════════
```

The unit sphere in $\mathbb{R}^n$ maps to an ellipsoid in $\mathbb{R}^m$ with semi-axes $\sigma_i\mathbf{u}_i$. The singular values are the lengths of the ellipsoid axes; the right singular vectors $\mathbf{v}_i$ are the input directions that map to those axes; the left singular vectors $\mathbf{u}_i$ are the output directions.

### 8.5 SVD and the Four Fundamental Subspaces

SVD reveals all four fundamental subspaces of $A \in \mathbb{R}^{m \times n}$ with rank $r$ simultaneously:

| Subspace | Basis | Space |
|---|---|---|
| Row space $\text{row}(A)$ | $\mathbf{v}_1,\ldots,\mathbf{v}_r$ (columns of $V_r$) | $\mathbb{R}^n$ |
| Null space $\text{null}(A)$ | $\mathbf{v}_{r+1},\ldots,\mathbf{v}_n$ | $\mathbb{R}^n$ |
| Column space $\text{col}(A)$ | $\mathbf{u}_1,\ldots,\mathbf{u}_r$ (columns of $U_r$) | $\mathbb{R}^m$ |
| Left null space $\text{null}(A^\top)$ | $\mathbf{u}_{r+1},\ldots,\mathbf{u}_m$ | $\mathbb{R}^m$ |

The orthogonality of $U$ and $V$ gives explicit orthogonal decompositions: $\mathbb{R}^n = \text{row}(A) \oplus \text{null}(A)$ and $\mathbb{R}^m = \text{col}(A) \oplus \text{null}(A^\top)$, with orthonormal bases for each subspace directly from $V$ and $U$.

**For AI:** LoRA represents weight updates as $\Delta W = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$. The SVD of $\Delta W$ has at most $r$ non-zero singular values — it lives in an $r$-dimensional subspace of weight space. The Eckart-Young theorem justifies this: if the true weight update is approximately low-rank (as empirically observed), LoRA captures most of its structure with far fewer parameters.

---

## 9. Eigendecomposition in AI Applications

### 9.1 Principal Component Analysis (PCA)

PCA is eigendecomposition of the sample covariance matrix, period. Given centred data $\tilde{X} \in \mathbb{R}^{n \times d}$ (subtract column means):

$$C = \frac{\tilde{X}^\top \tilde{X}}{n-1} \in \mathbb{R}^{d \times d}, \qquad C = Q\Lambda Q^\top, \quad \lambda_1 \geq \cdots \geq \lambda_d \geq 0$$

The eigenvectors $\mathbf{q}_1, \ldots, \mathbf{q}_d$ are the **principal components** (directions of decreasing variance). The eigenvalues $\lambda_i$ are the **principal variances**. The projection onto the top-$k$ components is:

$$X_\text{pca} = \tilde{X} Q_k \in \mathbb{R}^{n \times k}, \quad Q_k = [\mathbf{q}_1 \mid \cdots \mid \mathbf{q}_k]$$

**Fraction of variance explained** by the top-$k$ components: $\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i$. A scree plot of $\lambda_i$ vs $i$ shows how quickly variance falls off — a sharp "elbow" indicates a natural low-rank structure.

**Equivalently via SVD:** $\tilde{X} = U\Sigma V^\top$; principal components = columns of $V$; principal variances = $\sigma_i^2/(n-1)$.

**For AI:** PCA of word embedding matrices (e.g., GloVe, word2vec) reveals semantic axes — the top principal components correspond to major semantic dimensions (sentiment, formality, etc.). Activation PCA in transformer residual streams (Cunningham et al. 2023) identifies sparse, interpretable directions corresponding to concepts. PCA of weight matrices diagnoses training pathologies: a sudden drop in the effective rank of weight matrices signals representational collapse.

### 9.2 Eigenvalues and Gradient Descent

For a quadratic loss $\mathcal{L}(\boldsymbol{\theta}) = \frac{1}{2}\boldsymbol{\theta}^\top H\boldsymbol{\theta} - \mathbf{b}^\top\boldsymbol{\theta}$ (the local approximation near any critical point), gradient descent with step size $\eta$ gives:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta(H\boldsymbol{\theta}_t - \mathbf{b}) = (I - \eta H)\boldsymbol{\theta}_t + \eta\mathbf{b}$$

The error $\mathbf{e}_t = \boldsymbol{\theta}_t - \boldsymbol{\theta}^*$ satisfies $\mathbf{e}_{t+1} = (I-\eta H)\mathbf{e}_t$. In the eigenbasis of $H$ (writing $\mathbf{e}_t = \sum_i c_i^{(t)}\mathbf{q}_i$):

$$c_i^{(t)} = (1 - \eta\lambda_i)^t\, c_i^{(0)}$$

Each eigendirection decays independently. **Convergence condition:** $|1 - \eta\lambda_i| < 1$ for all $i$ $\Leftrightarrow$ $0 < \eta < 2/\lambda_{\max}$.

**Optimal step size:** $\eta^* = 2/(\lambda_{\max} + \lambda_{\min})$, giving convergence rate $\rho = (\kappa - 1)/(\kappa + 1)$ where $\kappa = \lambda_{\max}/\lambda_{\min}$ is the condition number.

**Ill-conditioning:** Large $\kappa$ means slow convergence. Gradient descent zigzags: it overshoots in high-curvature directions (large $\lambda$) while barely moving in low-curvature ones (small $\lambda$). Preconditioning (multiply gradient by $H^{-1}$ — Newton's method, or an approximation — Adam, K-FAC) aligns step sizes to eigenvalues and achieves $\kappa = 1$ in the ideal case.

### 9.3 Neural Network Hessian Spectrum

The Hessian $\nabla^2\mathcal{L}$ of a neural network loss at a critical point has a universal spectral structure observed empirically across architectures and datasets:

```
TYPICAL NEURAL NETWORK HESSIAN EIGENSPECTRUM
════════════════════════════════════════════════════════════════════════

  Density
    │
    │████
    │████████
    │███████████                           •  •
    │████████████████                   •        •
    └──────────────────────────────────────────────── λ
    0         small bulk           large outliers

  Bulk: ≈ Marchenko-Pastur distribution (random matrix theory)
  Outliers: O(1)–O(100) eigenvalues, well-separated from bulk
  Ratio κ = λ_max / λ_min can be 10⁶ or larger

════════════════════════════════════════════════════════════════════════
```

- **Bulk eigenvalues:** Dense cluster near 0; approximately Marchenko-Pastur distributed; correspond to weakly-learned or unlearned features.
- **Outlier eigenvalues:** A small number ($O(\text{layers})$) of large eigenvalues, well-separated from the bulk; correspond to the most-curved loss directions; dominate training dynamics.
- **Evolution during training:** Outliers grow during early training as the model fits data; bulk shifts toward zero; the outlier/bulk gap widens.

**2026 tools:** PyHessian (Yao et al. 2020) estimates the top eigenvalues and eigenvectors of the Hessian using randomised Lanczos; HessianFlow provides gradient-free eigenvalue density estimation. Both are practical for networks with $10^8$ parameters.

### 9.4 Attention Eigenstructure

The scaled dot-product attention weight matrix $S = \text{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{n \times n}$ is row-stochastic: all entries are non-negative and each row sums to 1. By the **Perron-Frobenius theorem**: the dominant eigenvalue is $\lambda_1 = 1$; all other eigenvalues satisfy $|\lambda_i| \leq 1$; the left eigenvector for $\lambda_1$ is $\mathbf{1}^\top/n$ (the uniform distribution).

**Softmax temperature effects on the spectrum:**
- High temperature ($\tau \to \infty$): $S \to \mathbf{1}\mathbf{1}^\top/n$ (rank-1 uniform); eigenvalues $= 1, 0, 0, \ldots, 0$. Completely diffuse attention.
- Low temperature ($\tau \to 0$): $S \to$ permutation matrix; eigenvalues = roots of unity on unit circle. Sharp, deterministic attention.
- Intermediate: eigenvalues lie on a spectrum between these extremes; their distribution characterises head behaviour.

**Mechanistic interpretability:** Induction heads (heads that copy from the previous occurrence of a token) have attention matrices approximating a shifted identity — eigenvalues cluster near roots of unity. Name-mover heads have matrices with a few dominant eigenvalues corresponding to "this token attends to its referent." The eigenvalue structure is a fingerprint of the head's function.

### 9.5 Graph Laplacian and GNNs

For an undirected graph $G$ with $n$ nodes, adjacency matrix $A$ (symmetric, $A_{ij} = 1$ if edge $(i,j)$ exists), and degree matrix $D = \text{diag}(d_1,\ldots,d_n)$ where $d_i = \sum_j A_{ij}$, the **graph Laplacian** is:

$$L = D - A$$

$L$ is symmetric PSD with eigenvalues $0 = \lambda_0 \leq \lambda_1 \leq \cdots \leq \lambda_{n-1}$.

**Key eigenvalue facts:**
- $\lambda_0 = 0$ always, with eigenvector $\mathbf{1} = (1,\ldots,1)^\top$ (constant on all nodes).
- The number of zero eigenvalues equals the number of connected components.
- $\lambda_1 > 0$ iff the graph is connected; $\lambda_1$ is the **Fiedler value** (algebraic connectivity): larger $\lambda_1$ means better connectivity and faster mixing.

**Spectral clustering:** The bottom-$k$ eigenvectors $\mathbf{q}_0,\ldots,\mathbf{q}_{k-1}$ of $L$ embed each node $i$ into $\mathbb{R}^k$ via its row in $Q_k = [\mathbf{q}_0\mid\cdots\mid\mathbf{q}_{k-1}]$. Running $k$-means on these embeddings clusters the graph into $k$ groups — nodes in the same cluster are well-connected. This is **spectral clustering**.

**Spectral GNNs:** Graph convolution is defined spectrally as $f(L) = Qf(\Lambda)Q^\top$ — a function of $L$'s eigenvalues applied in the eigenbasis. Polynomial filters $h(L) = \sum_{k=0}^K c_k L^k$ (ChebConv) approximate any spectral filter without computing the full eigendecomposition. This is the mathematical foundation of graph neural networks.

**For AI (2026):** Graph transformers use Laplacian eigenvectors as positional encodings — the eigenvectors encode the multi-scale geometric structure of the graph, serving as a canonical "position" for each node regardless of graph isomorphism. Standard in molecular property prediction (OGB benchmarks) and code analysis.

### 9.6 Spectral Analysis of Weight Matrices

**WeightWatcher** (Martin & Mahoney, 2019–2026) analyses the eigenvalue (or singular value) spectra of weight matrices to predict model quality and generalisation — without any test data.

**Heavy-tailed self-regularisation:** Well-trained models develop weight matrices whose eigenvalue distributions follow a power law: $\rho(\lambda) \sim \lambda^{-\alpha}$ with $\alpha \in [2, 4]$ for the tail. This deviates from the **Marchenko-Pastur distribution** (expected for random Gaussian matrices) — the deviation indicates that the model has learned non-random structure.

**Alpha metric:** Fit a power law to the tail of the singular value distribution of each layer; $\alpha \approx 2$–$4$ indicates a well-trained layer; $\alpha > 6$ or a purely MP distribution indicates an undertrained layer.

**Stable rank:** $\text{sr}(W) = \|W\|_F^2 / \|W\|_2^2 = \sum_i \sigma_i^2 / \sigma_1^2$. Measures effective dimensionality. Higher stable rank = more uniform singular value distribution = richer representations. Decreasing stable rank during fine-tuning signals representational collapse.

**For AI (2026):** WeightWatcher deployed at scale to compare LLM checkpoints, guide layer-wise learning rate selection in fine-tuning (layers with lower $\alpha$ need more adaptation), and identify which layers to include in LoRA adaptation without any downstream evaluation.

### 9.7 Eigenvalues in Recurrent Networks

The linearised RNN update $\mathbf{h}_t \approx W\mathbf{h}_{t-1} + U\mathbf{x}_t$ is a linear dynamical system. After $T$ steps from $\mathbf{h}_0$:

$$\mathbf{h}_T \approx W^T\mathbf{h}_0 + \sum_{t=0}^{T-1} W^{T-1-t} U\mathbf{x}_t$$

The gradient of the loss with respect to $\mathbf{h}_0$ involves $(W^\top)^T$. By eigendecomposition: $\|W^T\| \approx \rho(W)^T$:

- $\rho(W) < 1$: gradients **vanish** exponentially; long-range dependencies are forgotten.
- $\rho(W) > 1$: gradients **explode**; training diverges without clipping.
- $\rho(W) = 1$: **marginal stability**; gradients preserved in magnitude, but Jordan blocks cause polynomial growth.

**Orthogonal RNNs:** Constrain $W$ to be orthogonal ($W^\top W = I$) so $\rho(W) = 1$ with all eigenvalues on the unit circle and no Jordan blocks. Used in unitary RNNs (uRNN, EURNN) for long-range sequence modelling.

**LSTM/GRU gating:** The forget gate $\mathbf{f}_t = \sigma(W_f\mathbf{h}_{t-1} + U_f\mathbf{x}_t)$ dynamically controls the effective spectral radius of the recurrent dynamics: when $\mathbf{f}_t \approx 1$, the eigenvalue is $\approx 1$ (perfect memory); when $\mathbf{f}_t \approx 0$, the eigenvalue is $\approx 0$ (full reset). Gating allows the effective spectral radius to be context-dependent.

### 9.8 Diffusion Models and Score Functions

In a diffusion model, the forward process gradually adds Gaussian noise: $q(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)I)$. The conditional covariance of $\mathbf{x}_t$ given $\mathbf{x}_0$ is $\bar\alpha_t \Sigma_\text{data} + (1-\bar\alpha_t)I$.

**Spectral perspective:** Each eigenvalue $\lambda_i$ of the data covariance $\Sigma_\text{data}$ gets noise level $1-\bar\alpha_t$ added. High-variance data directions (large $\lambda_i$) survive longer before being drowned by noise; low-variance directions are masked quickly. The score function $\nabla_\mathbf{x}\log p_t(\mathbf{x})$ at noise level $t$ effectively operates on the "surviving" eigenspectrum.

**Optimal noise scheduling:** The EDM framework (Karras et al. 2022) uses a continuous noise level $\sigma$ parameterisation that sweeps uniformly across the log-eigenvalue spectrum of the data covariance — ensuring the network receives a balanced training signal at each noise level rather than spending too long on high-noise or low-noise regimes.

**Flow matching (2022–2026):** Constructs a straight-line path between noise distribution $\mathcal{N}(0,I)$ and data distribution in the eigenbasis of the data covariance. Eigenvalue-aware interpolation reduces the number of function evaluations needed for accurate generation. Now the dominant paradigm for image and video generation at scale.

---

## 10. Generalised Eigenproblems

### 10.1 The Generalised Eigenvalue Problem

The standard eigenvalue problem $A\mathbf{v} = \lambda\mathbf{v}$ measures eigenvalues relative to the identity. The **generalised eigenvalue problem** measures them relative to a symmetric positive definite matrix $B$:

$$A\mathbf{v} = \lambda B\mathbf{v}$$

For symmetric $A$ and SPD $B$: generalised eigenvalues are all real; generalised eigenvectors are $B$-orthogonal ($\mathbf{v}_i^\top B\mathbf{v}_j = \delta_{ij}$); analogous to the standard spectral theorem.

**Reduction to standard form.** Cholesky-decompose $B = LL^\top$. Substituting $\mathbf{w} = L^\top\mathbf{v}$: $(L^{-1}AL^{-\top})\mathbf{w} = \lambda\mathbf{w}$. The transformed matrix $\tilde{A} = L^{-1}AL^{-\top}$ is symmetric, so the standard spectral theorem applies. This is the numerically preferred approach (implemented in `scipy.linalg.eigh(A, B)`).

**Variational characterisation.** The $k$-th generalised eigenvalue satisfies $\lambda_k = \min/\max$ of the generalised Rayleigh quotient $R_B(A,\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} / \mathbf{x}^\top B\mathbf{x}$ over $B$-orthogonal subspaces — exactly as in Courant-Fischer.

### 10.2 AI Applications of Generalised Eigenproblem

**Linear Discriminant Analysis (LDA).** Find projection $\mathbf{w}$ maximising between-class variance relative to within-class variance: $\max_\mathbf{w} \mathbf{w}^\top S_B\mathbf{w} / \mathbf{w}^\top S_W\mathbf{w}$ where $S_B$ is the between-class scatter matrix and $S_W$ is the within-class scatter. This is a generalised Rayleigh quotient; the optimal $\mathbf{w}$ is the top generalised eigenvector solving $S_B\mathbf{w} = \lambda S_W\mathbf{w}$.

**Natural gradient.** The natural gradient update $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta F^{-1}\nabla\mathcal{L}$ requires solving $F\delta\boldsymbol{\theta} = \nabla\mathcal{L}$ where $F$ is the Fisher information matrix. This is equivalent to finding the update direction that minimises loss while accounting for the geometry of the parameter manifold. K-FAC (Kronecker-Factored Approximate Curvature) approximates $F$ as a Kronecker product and diagonalises it layer-wise, enabling efficient natural gradient updates for neural networks.

**Canonical Correlation Analysis (CCA).** Find directions in two feature spaces that are maximally correlated. Reduces to a generalised eigenproblem involving cross-covariance matrices. Used in multimodal representation learning (aligning image and text embeddings) and multi-view learning.

### 10.3 Spectral Graph Positional Encodings for Transformers

Standard transformers use sinusoidal positional encodings (fixed functions of position index). For graph-structured data (molecules, code, knowledge graphs), position is not a linear index — it is a graph-theoretic notion.

**Solution:** Use the bottom-$k$ eigenvectors of the graph Laplacian $L$ as positional encodings. Node $i$ gets encoding $\text{PE}(i) = [\mathbf{q}_1(i), \ldots, \mathbf{q}_k(i)] \in \mathbb{R}^k$ (its row in the eigenvector matrix). These encodings: (1) are permutation-equivariant — relabelling nodes relabels the encoding consistently; (2) capture multi-scale graph structure — low eigenvectors encode global structure, high eigenvectors encode local; (3) are provably more expressive than message-passing GNNs for distinguishing graph structures.

**2026 status:** Laplacian eigenvector positional encodings are standard in graph transformers for molecular property prediction (PCQM4Mv2 leaderboard), protein structure analysis, and code understanding graphs.

---

## 11. Eigenvalues and Stability Analysis

### 11.1 Discrete Dynamical Systems

The system $\mathbf{x}_{t+1} = A\mathbf{x}_t$ with initial condition $\mathbf{x}_0$ has solution (for diagonalisable $A = V\Lambda V^{-1}$):

$$\mathbf{x}_t = A^t\mathbf{x}_0 = V\Lambda^t V^{-1}\mathbf{x}_0 = \sum_{i=1}^n c_i\lambda_i^t\mathbf{v}_i$$

where $c_i = (\mathbf{v}_i^*)^\top \mathbf{x}_0$ are the initial projections onto each mode.

**Stability classification:**

| Condition | Behaviour | AI example |
|---|---|---|
| $\rho(A) < 1$ | $\mathbf{x}_t \to \mathbf{0}$ (asymptotically stable) | GD on strongly convex loss |
| $\rho(A) = 1$, Jordan size 1 | $\mathbf{x}_t$ bounded (marginally stable) | GD at critical step size |
| $\rho(A) = 1$, Jordan size $> 1$ | $\mathbf{x}_t$ grows polynomially | RNN at spectral radius = 1 with Jordan block |
| $\rho(A) > 1$ | $\mathbf{x}_t$ grows exponentially (unstable) | Exploding gradients, diverging GD |

### 11.2 Continuous Dynamical Systems and Neural ODEs

The system $d\mathbf{x}/dt = A\mathbf{x}$ has solution $\mathbf{x}(t) = e^{At}\mathbf{x}_0 = \sum_i c_i e^{\lambda_i t}\mathbf{v}_i$.

**Stability:** Asymptotically stable iff $\text{Re}(\lambda_i) < 0$ for all $i$; marginally stable iff $\text{Re}(\lambda_i) \leq 0$ with $\text{Re}(\lambda_i) = 0$ eigenvalues having Jordan blocks of size 1; unstable if any $\text{Re}(\lambda_i) > 0$.

**Neural ODEs** (Chen et al. 2018) parameterise $d\mathbf{h}/dt = f_\theta(\mathbf{h}, t)$ and use an ODE solver for the forward pass. Stability of the learned dynamics is controlled by the Jacobian $\partial f_\theta/\partial\mathbf{h}$ eigenvalues. Contractive neural ODEs enforce $\text{Re}(\lambda_i) < 0$ everywhere, guaranteeing convergence to a fixed point — used for stable generative models and physics-informed networks.

### 11.3 Spectral Normalisation

**Spectral normalisation** (Miyato et al. 2018) divides each weight matrix by its spectral norm (largest singular value) at every update step:

$$\hat{W} = W / \sigma_{\max}(W), \qquad \sigma_{\max}(W) = \|W\|_2$$

This enforces a Lipschitz constant $\leq 1$ for each linear layer. Applying it to all layers makes the entire network 1-Lipschitz: $\|f(\mathbf{x}) - f(\mathbf{y})\| \leq \|\mathbf{x} - \mathbf{y}\|$.

**Efficient implementation:** Estimate $\sigma_{\max}$ via one step of power iteration per training step — cost $O(mn)$ for an $m \times n$ matrix. Keep a running estimate $\hat{\mathbf{u}}, \hat{\mathbf{v}}$ and update: $\hat{\mathbf{v}} \leftarrow W^\top\hat{\mathbf{u}}/\|W^\top\hat{\mathbf{u}}\|$, $\hat{\mathbf{u}} \leftarrow W\hat{\mathbf{v}}/\|W\hat{\mathbf{v}}\|$; then $\sigma \approx \hat{\mathbf{u}}^\top W\hat{\mathbf{v}}$.

**Used in:** GAN discriminators (stable training; eliminates mode collapse); normalising flows (invertibility guarantee); safe RL (bounding policy Lipschitz constant); diffusion model discriminators (2026 standard).

### 11.4 Gradient Flow and Spectral Bias

The **gradient flow** (continuous-time limit of gradient descent) satisfies $d\boldsymbol{\theta}/dt = -\nabla\mathcal{L}(\boldsymbol{\theta})$. Near a quadratic minimum with Hessian $H = Q\Lambda Q^\top$, each eigendirection decays independently: $d(Q^\top\mathbf{e})/dt = -\Lambda(Q^\top\mathbf{e})$, so each component $c_i(t) = c_i(0)e^{-\lambda_i t}$.

**Spectral bias (frequency principle):** In the infinite-width Neural Tangent Kernel (NTK) regime, the gradient flow dynamics are governed by the NTK matrix $\Theta$. Large NTK eigenvalues correspond to low-frequency target function components — these are learned first and fastest. Small eigenvalues correspond to high-frequency components — learned slowly or not at all. This explains the empirical observation that neural networks learn coarse structure before fine details, and are biased toward smooth functions regardless of architecture.

---

## 12. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using $\det(A - \lambda I) = 0$ vs $\det(\lambda I - A) = 0$ and getting sign errors in the characteristic polynomial | Both define the same eigenvalues, but the polynomial differs by $(-1)^n$; using the wrong one gives wrong coefficients for non-eigenvalue computations | Use $p(\lambda) = \det(\lambda I - A)$ consistently (monic convention); eigenvalues are roots of either |
| 2 | Concluding a matrix is diagonalisable just because it has $n$ eigenvalues | Repeated eigenvalues can have fewer than $n$ independent eigenvectors (defective case) | Check $m_g(\lambda_i) = m_a(\lambda_i)$ for each repeated eigenvalue; compute $\text{rank}(A - \lambda_i I)$ to verify |
| 3 | Applying eigendecomposition to non-square matrices | The equation $A\mathbf{v} = \lambda\mathbf{v}$ requires $A: \mathbb{R}^n \to \mathbb{R}^n$ (same domain and codomain) | Use SVD for rectangular matrices; eigenvalues only for square matrices |
| 4 | Forgetting that real matrices can have complex eigenvalues | A real matrix with negative discriminant ($\Delta < 0$) has complex conjugate eigenvalue pairs; they are still valid eigenvalues | Always check the discriminant; complex eigenvalues are meaningful (e.g., rotation matrices) |
| 5 | Claiming all eigenvalues of $A^\top$ differ from those of $A$ | $A$ and $A^\top$ always have identical eigenvalues (same characteristic polynomial) but different eigenvectors | The eigenvectors of $A^\top$ are the left eigenvectors of $A$; eigenvalues are the same |
| 6 | Confusing algebraic and geometric multiplicity | AM counts polynomial roots; GM counts independent eigenvectors; they can differ for defective matrices | Always compute GM via $\dim\,\text{null}(A - \lambda I) = n - \text{rank}(A - \lambda I)$; don't assume AM = GM |
| 7 | Thinking eigenvectors are unique | Eigenvectors define a direction, not a specific vector; any non-zero scalar multiple is also an eigenvector | Speak of eigenspaces (subspaces), not single eigenvectors; normalise when uniqueness is needed |
| 8 | Applying the spectral theorem to non-symmetric matrices | Only symmetric (or Hermitian) matrices are guaranteed real eigenvalues and orthogonal eigenvectors | Check $A = A^\top$ first; for general $A$, use Schur decomposition or Jordan form |
| 9 | Setting step size based on $\lambda_{\max}$ but using the wrong norm | Gradient descent convergence requires $\eta < 2/\lambda_{\max}(H)$ where $H$ is the Hessian, not the weight matrix | Compute the Hessian or its spectral norm; do not confuse weight matrix eigenvalues with Hessian eigenvalues |
| 10 | Concluding $\rho(A) < 1$ from $\text{tr}(A) < n$ | Trace = sum of eigenvalues; small trace is compatible with some large and some negative eigenvalues | Compute $\rho(A) = \max_i|\lambda_i|$ explicitly; small trace does not imply small spectral radius |

---

## 13. Exercises

**Exercise 1 ★ — Eigenpair Verification and 2×2 Computation**

(a) Verify that $\mathbf{v} = (1, 2)^\top$ is an eigenvector of $A = \begin{pmatrix}3 & 1\\2 & 2\end{pmatrix}$ and find the eigenvalue. Check using the trace and determinant.

(b) Compute both eigenvalues and eigenvectors of $A$ using the characteristic polynomial. Verify $\lambda_1 + \lambda_2 = \text{tr}(A)$ and $\lambda_1\lambda_2 = \det(A)$.

(c) Is $A$ diagonalisable? Construct $V$ and $\Lambda$. Verify $A = V\Lambda V^{-1}$ numerically.

**Exercise 2 ★ — Characteristic Polynomial and Eigenspaces**

(a) Compute the characteristic polynomial of $B = \begin{pmatrix}4&1&0\\0&4&1\\0&0&2\end{pmatrix}$.

(b) Find all eigenvalues and their algebraic multiplicities.

(c) For each eigenvalue, compute a basis for the eigenspace. Identify defective eigenvalues.

(d) What is $\text{tr}(B)$ and $\det(B)$? Verify against the eigenvalues.

**Exercise 3 ★ — Diagonalisation**

Let $C = \begin{pmatrix}5&-1\\3&1\end{pmatrix}$.

(a) Find eigenvalues and eigenvectors. Construct $V = [\mathbf{v}_1|\mathbf{v}_2]$ and $\Lambda$.

(b) Compute $V^{-1}$ and verify $C = V\Lambda V^{-1}$.

(c) Use diagonalisation to compute $C^{10}$ efficiently. Compare against `np.linalg.matrix_power(C, 10)`.

(d) Compute $e^C = Ve^{\Lambda}V^{-1}$. Compare against `scipy.linalg.expm(C)`.

**Exercise 4 ★★ — Spectral Theorem**

Let $S = \begin{pmatrix}4&2\\2&1\end{pmatrix}$.

(a) Verify $S$ is symmetric and find its eigenvalues. Are they real and non-negative?

(b) Find orthonormal eigenvectors. Construct the orthogonal matrix $Q$ and verify $Q^\top Q = I$.

(c) Verify the spectral decomposition: $S = \lambda_1\mathbf{q}_1\mathbf{q}_1^\top + \lambda_2\mathbf{q}_2\mathbf{q}_2^\top$.

(d) Is $S$ PSD? PD? Compute the condition number $\kappa_2(S) = \lambda_{\max}/\lambda_{\min}$.

**Exercise 5 ★★ — Matrix Powers and Recurrence Relations**

The Fibonacci-like recurrence $a_{n+2} = 3a_{n+1} - 2a_n$ with $a_0 = 1$, $a_1 = 2$.

(a) Write the recurrence as $\mathbf{x}_{n+1} = A\mathbf{x}_n$ where $\mathbf{x}_n = (a_{n+1}, a_n)^\top$.

(b) Find the eigenvalues and eigenvectors of $A$. Diagonalise $A$.

(c) Use $A^n = V\Lambda^n V^{-1}$ to derive a closed-form formula for $a_n$.

(d) Verify for $n = 0, 1, 2, 3, 4$ numerically.

**Exercise 6 ★★ — Rayleigh Quotient and PSD**

Let $M = \begin{pmatrix}3&1&0\\1&3&1\\0&1&3\end{pmatrix}$.

(a) Compute the eigenvalues of $M$ using `np.linalg.eigvalsh`. Verify $M$ is PD.

(b) Compute $R(M, \mathbf{x})$ for $\mathbf{x} = (1,0,0)^\top$, $(0,1,0)^\top$, $(1,1,1)^\top/\sqrt{3}$, and a random unit vector. Verify $\lambda_{\min} \leq R \leq \lambda_{\max}$.

(c) Find the direction $\mathbf{x}$ that maximises $R(M,\mathbf{x})$ — confirm it is the top eigenvector.

(d) What is the condition number $\kappa_2(M)$? How does it affect the convergence rate of gradient descent on a quadratic loss with Hessian $M$?

**Exercise 7 ★★★ — PCA from Scratch**

Using the provided dataset $X \in \mathbb{R}^{100 \times 5}$:

(a) Centre the data. Compute the sample covariance matrix $C = \tilde{X}^\top\tilde{X}/99$.

(b) Compute the eigendecomposition of $C$. Sort eigenvalues in descending order.

(c) Project the data onto the top 2 principal components. Plot the 2D embedding.

(d) Compute the fraction of variance explained by 1, 2, 3 components. How many components explain 95% of variance?

(e) Verify that SVD of $\tilde{X}$ gives the same principal components as the eigendecomposition of $C$. Explain the connection: principal components = right singular vectors; principal variances = $\sigma_i^2/99$.

**Exercise 8 ★★★ — Power Iteration and PageRank**

(a) Implement power iteration for the dominant eigenpair of a random $5 \times 5$ symmetric positive definite matrix $A$. Track convergence of $|\lambda_k - \lambda_{\text{true}}|$ vs iteration $k$.

(b) Verify the convergence rate is approximately $|\lambda_2/\lambda_1|$ per iteration (compare theoretical to empirical).

(c) Build a small web graph (6 nodes) with a link matrix $L$. Construct the row-stochastic PageRank transition matrix $G = 0.85 L + 0.15 \cdot \mathbf{1}\mathbf{1}^\top/n$.

(d) Run power iteration on $G$ to find the PageRank vector (dominant eigenvector). Compare to `np.linalg.eig(G.T)`.

(e) **For attention:** Construct a $4 \times 4$ attention weight matrix $S$ (softmax of random logits). Compute its eigenvalues. Verify the dominant eigenvalue is 1. Plot the eigenvalue spectrum.

---

## 14. Why This Matters for AI (2026 Perspective)

| Concept | Where It Appears in AI/LLMs | Concrete Method |
|---|---|---|
| Eigendecomposition of covariance | Dimensionality reduction, embedding analysis | PCA, Kernel PCA |
| Spectral radius $\rho(W)$ | RNN stability, gradient flow control | Orthogonal RNNs, spectral normalisation |
| Hessian eigenspectrum | Loss landscape analysis, learning rate selection | PyHessian, K-FAC, SAM |
| Perron-Frobenius eigenvalue = 1 | Attention matrix structure, Markov chain mixing | Attention head analysis, mechanistic interpretability |
| Low-rank eigenspace adaptation | Parameter-efficient fine-tuning | LoRA, DoRA, GaLore |
| Graph Laplacian eigenvectors | Graph transformer positional encodings | Lim et al. 2023, GRIT |
| SVD / singular values | Weight compression, stable rank, generalisation | WeightWatcher, model pruning |
| QR algorithm | All eigenvalue computations in numpy/scipy | `np.linalg.eigh`, `scipy.linalg.eig` |
| Power iteration | Dominant eigenvalue estimation, PageRank, spectral norm | Spectral normalisation (Miyato 2018) |
| Condition number $\kappa(H)$ | Gradient descent convergence rate | Preconditioning, Adam, AdaGrad |
| Rayleigh quotient | PCA objective, curvature measurement | PCA, Fisher LDA, natural gradient |
| Heavy-tailed eigenvalue distribution | Model quality without test data | WeightWatcher alpha metric |
| NTK eigenvalues | Which features are learned first (spectral bias) | NTK theory, frequency principle |

---

## 15. Conceptual Bridge

### Where This Section Sits

Eigenvalues and eigenvectors are the payoff for everything in Chapters 1 and 2. Vector spaces (Section 02-06) established the abstract setting — subspaces, span, dimension. Matrix rank (Section 02-05) measured the effective dimensionality of a linear map. Determinants (Section 02-04) provided the characteristic polynomial via $\det(\lambda I - A) = 0$. Row reduction (Section 02-03) is the tool for finding null spaces, hence eigenvectors. All of this machinery was built so that we could ask and answer the deepest question about a linear map: *what are its natural axes, and how does it scale along them?*

### What This Section Unlocks

Eigenvalues open three major directions forward in this curriculum. First, **Singular Value Decomposition** (Section 03-02) is the direct generalisation: eigenvalues of $A^\top A$ are the squared singular values; the eigenvectors of $A^\top A$ and $AA^\top$ are the right and left singular vectors. SVD extends eigendecomposition to rectangular matrices and gives the four fundamental subspaces with optimal orthonormal bases. Second, **PCA** (Section 03-03) is eigendecomposition of the covariance matrix in practice — the full pipeline from raw data to compressed, interpretable representations. Third, **Positive Definite Matrices** (Section 03-07) uses the spectral theorem to characterise the geometry of quadratic forms — every optimisation problem in Chapter 8 involves a positive definite Hessian.

### The Larger Arc

Stepping back further: eigenvalues connect linear algebra to analysis, probability, and AI in ways that become clearer as the curriculum progresses. The Hessian eigenspectrum (Chapter 8, Optimization) controls whether gradient descent converges, how fast, and to what kind of minimum. The graph Laplacian eigenvalues (Chapter 11, Graph Theory) determine the topology of learned graph representations. The eigenvalues of the Fisher information matrix (Chapter 13, ML-Specific Math) define the natural gradient, the theoretically optimal update direction. The NTK eigenvalues (Chapter 14, Transformers) determine which target function components are learned first. Every time you see a covariance matrix, a Gram matrix, a weight matrix, or a Hessian — you are dealing with a system whose behaviour is ultimately governed by eigenvalues.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  02-06 Vector Spaces          02-05 Matrix Rank
  (subspaces, basis)           (null space, column space)
          │                              │
          └──────────────┬───────────────┘
                         ▼
              03-01 EIGENVALUES & EIGENVECTORS   ◄── YOU ARE HERE
              (characteristic polynomial,
               diagonalisation, spectral theorem,
               Jordan form, SVD connection)
                         │
          ┌──────────────┼───────────────────────┐
          ▼              ▼                        ▼
    03-02 SVD      03-03 PCA              03-07 Positive
    (singular      (covariance             Definite
     values,        eigenstructure,        Matrices
     Eckart-Young)  dimension reduction)   (Cholesky,
          │              │                 curvature)
          └──────────────┼────────────────────────┘
                         ▼
          Ch. 08 Optimization  ·  Ch. 11 Graph Theory
          Ch. 13 ML-Specific Math  ·  Ch. 14 Transformers

════════════════════════════════════════════════════════════════════════
```

[← Back to Advanced Linear Algebra](../README.md) | [Next: Singular Value Decomposition →](../02-Singular-Value-Decomposition/notes.md)
