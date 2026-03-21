[← Back to Advanced Linear Algebra](../README.md) | [Next: Principal Component Analysis →](../03-Principal-Component-Analysis/notes.md)

---

# Singular Value Decomposition

> _"Every matrix tells a story of rotation, scaling, and rotation again — the SVD reads that story in full."_

## Overview

The Singular Value Decomposition (SVD) is the most universally applicable matrix factorisation in all of applied mathematics. Given any matrix $A \in \mathbb{R}^{m \times n}$ — rectangular, rank-deficient, ill-conditioned, whatever — the SVD expresses it as $A = U\Sigma V^\top$: a rotation in the input space ($V^\top$), a coordinate-wise scaling ($\Sigma$), and a rotation in the output space ($U$). The scaling factors $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ are the **singular values**; they completely determine the linear map's geometry. Unlike the eigendecomposition, which requires a square matrix and may involve complex numbers, the SVD always exists, always produces real non-negative singular values, and works for matrices of any shape.

For AI practitioners in 2026, the SVD is inescapable. Low-Rank Adaptation (LoRA) fine-tunes billion-parameter language models by constraining weight updates to a low-rank subspace — the dominant singular subspace of the gradient matrix. DeepSeek's Multi-Head Latent Attention (MLA) compresses key-value caches by projecting them through a low-rank bottleneck. The Eckart-Young theorem guarantees that the rank-$k$ SVD truncation is the best possible rank-$k$ approximation, which is why SVD underlies recommender systems, image compression, latent semantic analysis, and the pseudoinverse. WeightWatcher diagnoses model quality by examining the singular value spectrum of weight matrices: healthy trained networks have heavy-tailed spectra; undertrained or overtrained networks do not. Every time you compute a spectral norm, solve a least-squares problem, or talk about the "effective rank" of a weight matrix, you are using the SVD.

This section develops the SVD from first principles, proves the Eckart-Young theorem, connects SVD to eigenvalues and the four fundamental subspaces, and builds every major AI application from scratch.

## Prerequisites

- Eigenvalues and eigenvectors, spectral theorem for symmetric matrices (Section 03-01)
- Matrix rank, null space, column space (Section 02-05)
- Vector spaces and orthogonality (Section 02-06)
- Matrix transpose, invertibility, determinants (Section 02-02 through 02-04)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive demos: geometric action, Eckart-Young compression, pseudoinverse, randomised SVD, LoRA, image compression, weight matrix analysis |
| [exercises.ipynb](exercises.ipynb) | 8 graded problems: SVD by hand, four subspaces, low-rank approximation, pseudoinverse, condition number, randomised SVD, LoRA adapter, image compression |

## Learning Objectives

After completing this section, you will:

- State the SVD theorem and explain its geometric meaning as three successive linear maps
- Compute the SVD of small matrices by finding the eigendecomposition of $A^\top A$ and $AA^\top$
- Distinguish full SVD, thin SVD, and truncated SVD; know when to use each
- Identify the four fundamental subspaces of $A$ from its SVD
- Prove and apply the Eckart-Young theorem for optimal low-rank approximation
- Compute the Moore-Penrose pseudoinverse $A^+$ and use it to solve least-squares problems
- Express the spectral norm, Frobenius norm, and nuclear norm in terms of singular values
- Implement randomised SVD for large matrices
- Connect SVD to LoRA, MLA, spectral normalisation, recommender systems, and LSA
- Interpret the singular value spectrum of neural network weight matrices

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What SVD Is](#11-what-svd-is)
  - [1.2 The Geometric Picture](#12-the-geometric-picture)
  - [1.3 Why SVD Is the Most Important Decomposition in AI](#13-why-svd-is-the-most-important-decomposition-in-ai)
  - [1.4 SVD vs Eigendecomposition](#14-svd-vs-eigendecomposition)
  - [1.5 Historical Timeline](#15-historical-timeline)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The SVD Theorem](#21-the-svd-theorem)
  - [2.2 Singular Values](#22-singular-values)
  - [2.3 Left and Right Singular Vectors](#23-left-and-right-singular-vectors)
  - [2.4 Full, Thin, and Truncated SVD](#24-full-thin-and-truncated-svd)
  - [2.5 The Four Fundamental Subspaces](#25-the-four-fundamental-subspaces)
- [3. Computing the SVD](#3-computing-the-svd)
  - [3.1 Connection to Eigendecomposition](#31-connection-to-eigendecomposition)
  - [3.2 Golub-Reinsch Algorithm](#32-golub-reinsch-algorithm)
  - [3.3 Randomised SVD](#33-randomised-svd)
  - [3.4 Lanczos Bidiagonalisation](#34-lanczos-bidiagonalisation)
  - [3.5 Numerical Considerations](#35-numerical-considerations)
- [4. Properties of the SVD](#4-properties-of-the-svd)
  - [4.1 Norms via Singular Values](#41-norms-via-singular-values)
  - [4.2 The Eckart-Young Theorem](#42-the-eckart-young-theorem)
  - [4.3 The Moore-Penrose Pseudoinverse](#43-the-moore-penrose-pseudoinverse)
  - [4.4 SVD and Rank](#44-svd-and-rank)
  - [4.5 Condition Number](#45-condition-number)
  - [4.6 The Procrustes Problem](#46-the-procrustes-problem)
- [5. Outer Product Form and Rank-1 Decomposition](#5-outer-product-form-and-rank-1-decomposition)
  - [5.1 Rank-1 Outer Product Decomposition](#51-rank-1-outer-product-decomposition)
  - [5.2 Incremental and Streaming SVD](#52-incremental-and-streaming-svd)
  - [5.3 Symmetric Matrices: SVD Equals Eigendecomposition](#53-symmetric-matrices-svd-equals-eigendecomposition)
- [6. SVD in Machine Learning and AI](#6-svd-in-machine-learning-and-ai)
  - [6.1 Low-Rank Approximation and LoRA](#61-low-rank-approximation-and-lora)
  - [6.2 Recommender Systems](#62-recommender-systems)
  - [6.3 Latent Semantic Analysis and Word Vectors](#63-latent-semantic-analysis-and-word-vectors)
  - [6.4 Image Compression](#64-image-compression)
  - [6.5 Pseudoinverse and Least Squares](#65-pseudoinverse-and-least-squares)
  - [6.6 Spectral Methods for Graphs](#66-spectral-methods-for-graphs)
  - [6.7 Weight Matrix Analysis (WeightWatcher)](#67-weight-matrix-analysis-weightwatcher)
  - [6.8 Attention and SVD](#68-attention-and-svd)
- [7. Generalised SVD and Extensions](#7-generalised-svd-and-extensions)
  - [7.1 Generalised SVD (GSVD)](#71-generalised-svd-gsvd)
  - [7.2 Tensor SVD and Tucker Decomposition](#72-tensor-svd-and-tucker-decomposition)
  - [7.3 Truncated and Randomised Variants in Practice](#73-truncated-and-randomised-variants-in-practice)
- [8. Common Mistakes](#8-common-mistakes)
- [9. Exercises](#9-exercises)
- [10. Why This Matters for AI (2026 Perspective)](#10-why-this-matters-for-ai-2026-perspective)
- [11. Conceptual Bridge](#11-conceptual-bridge)

---

## 1. Intuition

### 1.1 What SVD Is

A matrix $A \in \mathbb{R}^{m \times n}$ represents a linear map from $\mathbb{R}^n$ to $\mathbb{R}^m$. That map can be arbitrarily complex — it can rotate, stretch, shrink, and project. The SVD breaks this complexity into three primitives that are each individually simple:

$$A = U \Sigma V^\top$$

- **$V^\top$: rotation/reflection in the input space $\mathbb{R}^n$**. $V$ is orthogonal ($V^\top V = I_n$), so $V^\top$ is a rigid transformation that does not change lengths or angles — it just changes the orientation of the coordinate axes.
- **$\Sigma$: scaling along coordinate axes**. $\Sigma$ is an $m \times n$ diagonal-like matrix with non-negative entries $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$ on the diagonal. It stretches or shrinks each axis independently. If $m \neq n$, it also changes the dimension.
- **$U$: rotation/reflection in the output space $\mathbb{R}^m$**. $U$ is orthogonal ($UU^\top = I_m$), another rigid transformation.

The sequence $V^\top \to \Sigma \to U$ is the "rotate → scale → rotate" story of the matrix. The first rotation aligns the input coordinates to the special axes (the **right singular vectors** $\mathbf{v}_i$, columns of $V$). Then each axis is scaled by $\sigma_i$. Then the result is rotated to the output coordinate system via $U$, whose columns $\mathbf{u}_i$ are the **left singular vectors**.

This decomposition exists for every real (or complex) matrix, regardless of shape, rank, or conditioning. The singular values $\sigma_i$ are unique (though the vectors $\mathbf{u}_i, \mathbf{v}_i$ have sign ambiguity and are non-unique when singular values repeat). This universality is what distinguishes the SVD from the eigendecomposition, which only applies to square matrices and can fail (defective matrices) or become complex.

**For AI:** When you apply a linear layer $W \in \mathbb{R}^{d_{out} \times d_{in}}$ to an input $\mathbf{x}$, you are implicitly performing $W\mathbf{x} = U\Sigma V^\top \mathbf{x}$. The right singular vectors $\mathbf{v}_i$ are the "input features" the layer responds to; the singular values $\sigma_i$ are how strongly each feature is amplified; the left singular vectors $\mathbf{u}_i$ are the "output features" it writes to. LoRA exploits this by constraining $\Delta W = BA$ where $B \in \mathbb{R}^{d_{out} \times r}$, $A \in \mathbb{R}^{r \times d_{in}}$ — a rank-$r$ update that modifies only the dominant $r$ singular directions.

### 1.2 The Geometric Picture

The clearest geometric picture of the SVD comes from watching what $A$ does to the unit sphere $S^{n-1} = \{\mathbf{x} \in \mathbb{R}^n : \|\mathbf{x}\| = 1\}$. Apply $A$ to every point on the unit sphere. The result is an ellipsoid in $\mathbb{R}^m$. This is not obvious — it is a theorem — but it follows directly from the SVD:

$$A(S^{n-1}) = U\Sigma V^\top(S^{n-1}) = U\Sigma(S^{n-1}) = U(\text{axis-aligned ellipsoid}) = \text{ellipsoid}$$

Step by step:
1. $V^\top$ is orthogonal, so $V^\top(S^{n-1}) = S^{n-1}$ (rigid transformation preserves the sphere).
2. $\Sigma$ scales each axis by $\sigma_i$, turning the sphere into an axis-aligned ellipsoid with semi-axes $\sigma_1, \sigma_2, \ldots$.
3. $U$ is orthogonal, so it rotates/reflects the ellipsoid without changing its shape.

The **singular values** $\sigma_i$ are the semi-axis lengths of the output ellipsoid. The **right singular vectors** $\mathbf{v}_i$ are the directions in the input space that map to the semi-axes of the ellipsoid. The **left singular vectors** $\mathbf{u}_i$ are the directions of the ellipsoid's semi-axes in the output space.

```
GEOMETRIC ACTION OF A ∈ ℝ^(m×n)
════════════════════════════════════════════════════════════════════════

  Input space ℝⁿ               Output space ℝᵐ

      v₂                            u₂
       ↑                             ↑
       │  Unit sphere                │
       │    ·                        │   σ₂·u₂
       │  ·   ·       V^T          · │ ·
       │ ·     ·  ──────────►    ·   │   ·       A = U Σ V^T
       │  ·   ·                ·     │     ·
       │    ·               ─────────────────── u₁ →
  ─────┼────────► v₁       ·         σ₁      ·
       │                     ·               ·
       │                       ·           ·
       │                         · · · · ·
       │                        (ellipse)
       │                  semi-axes = singular values

  Right singular vectors v_i    Left singular vectors u_i
  = natural input axes          = natural output axes

════════════════════════════════════════════════════════════════════════
```

For a $2 \times 2$ matrix with $\sigma_1 > \sigma_2 > 0$: the unit circle maps to an ellipse with semi-axes $\sigma_1$ and $\sigma_2$. The direction $\mathbf{v}_1$ is the input direction that gets stretched the most (by $\sigma_1$); $\mathbf{v}_2$ is the direction stretched least (by $\sigma_2$). If $\sigma_2 = 0$, the map collapses one direction to zero — the matrix is rank-1 and maps every vector onto a line.

### 1.3 Why SVD Is the Most Important Decomposition in AI

The SVD is not just an abstract factorisation — it is the foundational tool for understanding, compressing, and controlling linear maps in neural networks. Here are the core connections:

**Low-rank approximation (LoRA, DoRA, MLA).** Every weight matrix in a neural network is a linear map. After training, weight update matrices $\Delta W$ tend to have low effective rank — most of the "learning signal" lives in a few singular directions. LoRA exploits this: instead of fine-tuning all of $W$, it learns $\Delta W = BA$ with $r \ll d$. The SVD is used to initialise $B$ and $A$ from the dominant singular subspace of $\Delta W$, or to analyse which singular directions are being adapted. DoRA (Weight-Decomposed Low-Rank Adaptation) further decomposes $W = \|W\|_c \cdot W/\|W\|_c$ and adapts direction (via low-rank) and magnitude (via scalar) separately. DeepSeek's MLA uses low-rank projections of key-value matrices to compress the KV cache, reducing memory from $O(n \cdot d_{kv})$ to $O(n \cdot r)$ with $r \ll d_{kv}$.

**Spectral normalisation.** The spectral norm of a weight matrix $\|W\|_2 = \sigma_{\max}(W)$ is the largest singular value. Spectral normalisation (Miyato et al. 2018) divides each weight matrix by its spectral norm, enforcing a Lipschitz-1 constraint on each layer. This stabilises GAN training. The spectral norm is computed via power iteration (equivalent to computing the top singular vector pair) and is $O(mn)$ per forward pass rather than $O(mn\min(m,n))$ for the full SVD.

**Pseudoinverse and least squares.** Every linear regression problem $\min_\mathbf{x} \|A\mathbf{x} - \mathbf{b}\|_2$ has solution $\hat{\mathbf{x}} = A^+ \mathbf{b}$ where $A^+ = V\Sigma^+U^\top$ is the Moore-Penrose pseudoinverse. The SVD provides the most numerically stable way to compute this. Ridge regression adds a $\lambda I$ regulariser, which in SVD form becomes shrinkage: each singular component is shrunk by factor $\sigma_i^2/(\sigma_i^2 + \lambda)$.

**Matrix compression and efficiency.** Images, attention matrices, embedding tables, and gradient matrices all have approximate low-rank structure. The Eckart-Young theorem guarantees the SVD gives the best rank-$k$ approximation. Randomised SVD (Halko et al. 2011) computes this in $O(mnk)$ time rather than $O(mn\min(m,n))$, making it practical for million-dimensional problems.

**Understanding model geometry.** The singular value spectrum of a weight matrix reveals its "information geometry": a flat spectrum (all $\sigma_i$ equal) means the layer treats all directions equally; a steep spectrum means the layer has discovered dominant directions. WeightWatcher (Martin and Mahoney, 2019–2026) quantifies model quality by fitting power-law distributions to singular value spectra: well-trained models have $\sigma_i \sim i^{-\alpha}$ with $\alpha \in [2, 4]$; undertrained models have flat spectra; overtrained models have spike/bulk structure.

### 1.4 SVD vs Eigendecomposition

Students often confuse SVD and eigendecomposition. Here is a precise comparison:

| Property | Eigendecomposition $A = V\Lambda V^{-1}$ | SVD $A = U\Sigma V^\top$ |
|---|---|---|
| **Applicable to** | Square matrices only | Any $m \times n$ matrix |
| **Eigenvalues/singular values** | Can be complex, can be negative | Always real, always $\geq 0$ |
| **$V$ is orthogonal?** | Only for symmetric $A$ | Always |
| **Same input/output basis?** | Yes ($V$ used twice) | No ($U$ for output, $V$ for input) |
| **Existence** | Not always (defective over $\mathbb{C}$) | Always exists |
| **For symmetric PSD $A$** | $A = Q\Lambda Q^\top$, $\lambda_i \geq 0$ | Coincides: $U=V=Q$, $\sigma_i = \lambda_i$ |
| **Computation cost** | $O(n^3)$ | $O(mn\min(m,n))$ |
| **Geometric meaning** | Fixed points of the map | Extreme stretching directions |

The key insight: **singular values and eigenvalues are generally different quantities**. For a matrix $A$, the eigenvalues of $A$ are not the same as the singular values of $A$ (unless $A$ is symmetric PSD). The singular values of $A$ are the square roots of the eigenvalues of $A^\top A$ (which is always symmetric PSD). For example, the rotation matrix $R_\theta$ has eigenvalues $e^{\pm i\theta}$ (complex, magnitude 1) but singular values $\sigma_1 = \sigma_2 = 1$ (all equal — a rotation is an isometry).

### 1.5 Historical Timeline

| Year | Who | What |
|---|---|---|
| 1873 | Beltrami, Jordan | Independent discovery of SVD for bilinear forms; Beltrami published first, Jordan one month later |
| 1889 | Sylvester | Generalised to arbitrary rectangular real matrices |
| 1907 | Schmidt | Infinite-dimensional version (integral operators); modern functional-analytic foundation |
| 1936 | Eckart & Young | Proved the best low-rank approximation theorem; connected SVD to statistics |
| 1965 | Golub & Reinsch | Practical stable algorithm: bidiagonalisation + implicit QR shift; still in use today |
| 1976 | Lanczos bidiag. | Krylov subspace method for large sparse matrices; precursor to ARPACK |
| 1992 | Berry et al. | Latent Semantic Analysis (LSA) — SVD of term-document matrices for information retrieval |
| 2000 | Cai, Candès | Matrix completion theory; nuclear norm minimisation as convex relaxation of low-rank |
| 2009 | Koren et al. | Winning Netflix Prize solution used SVD-based matrix factorisation |
| 2011 | Halko, Martinsson, Tropp | Randomised SVD — near-linear-time algorithms for large matrices |
| 2017–26 | LoRA era | Hu et al. (LoRA 2021), Liu et al. (DoRA 2024), DeepSeek (MLA 2024) — low-rank SVD in LLMs |
| 2019–26 | Martin & Mahoney | WeightWatcher — power-law singular value spectra as model quality metrics |

---

## 2. Formal Definitions

### 2.1 The SVD Theorem

**Theorem (Singular Value Decomposition).** Let $A \in \mathbb{R}^{m \times n}$ with $\text{rank}(A) = r$. Then there exist orthogonal matrices $U \in \mathbb{R}^{m \times m}$, $V \in \mathbb{R}^{n \times n}$, and a matrix $\Sigma \in \mathbb{R}^{m \times n}$ of the form

$$\Sigma = \begin{pmatrix} \sigma_1 & & & \\ & \sigma_2 & & \\ & & \ddots & \\ & & & \sigma_{\min(m,n)} \end{pmatrix}_{m \times n}$$

with $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0 = \sigma_{r+1} = \cdots = \sigma_{\min(m,n)}$, such that

$$\boxed{A = U\Sigma V^\top}$$

The $\sigma_i$ are called the **singular values** of $A$. The columns $\mathbf{u}_1, \ldots, \mathbf{u}_m$ of $U$ are the **left singular vectors**. The columns $\mathbf{v}_1, \ldots, \mathbf{v}_n$ of $V$ are the **right singular vectors**.

**Proof sketch (constructive).** Since $A^\top A \in \mathbb{R}^{n \times n}$ is symmetric positive semidefinite, by the Spectral Theorem it has an eigendecomposition $A^\top A = V\Lambda V^\top$ with $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$, $\lambda_i \geq 0$. Define $\sigma_i = \sqrt{\lambda_i} \geq 0$. For each $\sigma_i > 0$, define $\mathbf{u}_i = A\mathbf{v}_i / \sigma_i$. One can verify:
- $\mathbf{u}_i^\top \mathbf{u}_j = (A\mathbf{v}_i/\sigma_i)^\top(A\mathbf{v}_j/\sigma_j) = \mathbf{v}_i^\top A^\top A \mathbf{v}_j / (\sigma_i\sigma_j) = \lambda_j \mathbf{v}_i^\top\mathbf{v}_j/(\sigma_i\sigma_j) = \delta_{ij}$ (orthonormal).
- Extend $\{\mathbf{u}_1,\ldots,\mathbf{u}_r\}$ to an orthonormal basis of $\mathbb{R}^m$ to complete $U$.
- Then $A\mathbf{v}_i = \sigma_i \mathbf{u}_i$ for $i \leq r$ and $A\mathbf{v}_i = \mathbf{0}$ for $i > r$, which compactly gives $A = U\Sigma V^\top$. $\square$

**Uniqueness.** The singular values are unique (they are square roots of eigenvalues of $A^\top A$, which are unique). The singular vectors are unique up to:
- Sign flips: $(\mathbf{u}_i, \mathbf{v}_i) \to (-\mathbf{u}_i, -\mathbf{v}_i)$ leaves $A = U\Sigma V^\top$ unchanged.
- Rotation within repeated singular value blocks: if $\sigma_i = \sigma_{i+1} = \cdots = \sigma_{i+k}$, any orthogonal rotation within the corresponding subspaces is valid.

### 2.2 Singular Values

The singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_{\min(m,n)} \geq 0$ carry all the magnitude information about $A$. Key properties:

**Relation to eigenvalues of $A^\top A$ and $AA^\top$:**
$$\sigma_i^2 = \lambda_i(A^\top A) = \lambda_i(AA^\top) \quad (i = 1, \ldots, r)$$

The nonzero eigenvalues of $A^\top A$ and $AA^\top$ are identical (even though the matrices have different sizes), and they equal the squared singular values of $A$.

**Geometric meaning:** $\sigma_1 = \max_{\|\mathbf{x}\|=1} \|A\mathbf{x}\|$ (the maximum stretching factor). More generally, $\sigma_i$ is the maximum stretching factor among all directions orthogonal to $\mathbf{v}_1, \ldots, \mathbf{v}_{i-1}$.

**Rank:** $\text{rank}(A) = r$ = number of positive singular values.

**Examples:**
- Identity $I_n$: all $n$ singular values equal 1.
- Scaling matrix $\text{diag}(3, 1, 0)$: singular values are 3, 1, 0; rank = 2.
- Rotation matrix $R_\theta$: all singular values equal 1 (isometry).
- Projection matrix onto a $k$-dimensional subspace: $k$ singular values equal 1, rest equal 0.
- All-ones matrix $\mathbf{1}\mathbf{1}^\top \in \mathbb{R}^{n\times n}$: $\sigma_1 = n$, $\sigma_2 = \cdots = \sigma_n = 0$; rank = 1.

### 2.3 Left and Right Singular Vectors

The left and right singular vectors satisfy a pair of eigenvalue-like equations:

$$A\mathbf{v}_i = \sigma_i \mathbf{u}_i \qquad (i = 1, \ldots, r)$$
$$A^\top \mathbf{u}_i = \sigma_i \mathbf{v}_i \qquad (i = 1, \ldots, r)$$
$$A\mathbf{v}_i = \mathbf{0} \quad (i > r), \qquad A^\top \mathbf{u}_i = \mathbf{0} \quad (i > r)$$

Combining: $A^\top A \mathbf{v}_i = \sigma_i^2 \mathbf{v}_i$ and $AA^\top \mathbf{u}_i = \sigma_i^2 \mathbf{u}_i$. So $\mathbf{v}_i$ are eigenvectors of $A^\top A$ and $\mathbf{u}_i$ are eigenvectors of $AA^\top$, both with eigenvalue $\sigma_i^2$.

The relationships can be rewritten as a single symmetric eigenproblem by introducing the augmented matrix:

$$\begin{pmatrix} 0 & A \\ A^\top & 0 \end{pmatrix} \begin{pmatrix} \mathbf{u}_i \\ \mathbf{v}_i \end{pmatrix} = \sigma_i \begin{pmatrix} \mathbf{u}_i \\ \mathbf{v}_i \end{pmatrix}$$

This $(m+n) \times (m+n)$ symmetric matrix has eigenvalues $\pm\sigma_1, \pm\sigma_2, \ldots, \pm\sigma_r, 0, \ldots, 0$. Some SVD algorithms exploit this formulation.

### 2.4 Full, Thin, and Truncated SVD

For an $m \times n$ matrix with $m > n$ and rank $r \leq n$, there are three common variants:

```
FULL SVD vs THIN SVD vs TRUNCATED SVD  (m > n, rank r)
════════════════════════════════════════════════════════════════

FULL:
  A    =    U      ×     Σ      ×    V^T
(m×n)    (m×m)       (m×n)       (n×n)
         [all m      [n nonzero   [all n
          columns]    rows]        columns]

THIN (economy):
  A    =    Û      ×    Σ̂      ×    V^T
(m×n)    (m×n)       (n×n)       (n×n)
         [first n   [square      [all n
          columns]   diagonal]    columns]

TRUNCATED (rank-k):
  A_k  =   Û_k    ×    Σ̂_k    ×   V_k^T
(m×n)    (m×k)       (k×k)       (k×n)
         [first k   [top-k       [first k
          columns]   diagonal]    columns]

════════════════════════════════════════════════════════════════
```

**Full SVD** is the complete factorisation; $U$ and $V$ are square orthogonal matrices. Storage: $O(m^2 + mn + n^2)$.

**Thin SVD** (also called economy or compact SVD): keep only the first $n$ columns of $U$ (for $m > n$). Since $\Sigma$ has zeros in rows $n+1$ through $m$, these columns of $U$ do not contribute. Storage: $O(mn + n^2)$.

**Truncated SVD** (also called low-rank SVD): keep only the top $k$ terms ($k \leq r$). This gives the best rank-$k$ approximation $A_k$ (Eckart-Young theorem, Section 4.2). Storage: $O((m+n)k)$. This is what LoRA exploits.

**For AI:** In practice, `np.linalg.svd(A, full_matrices=False)` computes the thin SVD. `scipy.sparse.linalg.svds(A, k=k)` and `sklearn.utils.extmath.randomized_svd(A, n_components=k)` compute truncated SVDs efficiently.

### 2.5 The Four Fundamental Subspaces

The SVD provides the cleanest way to identify the four fundamental subspaces of $A$ (Strang's framework):

| Subspace | Definition | From SVD | Dimension |
|---|---|---|---|
| **Column space** (range) | $\{\text{col}(A)\} = \{A\mathbf{x}\}$ | span$\{\mathbf{u}_1,\ldots,\mathbf{u}_r\}$ | $r$ |
| **Left null space** | $\{\mathbf{y}: A^\top\mathbf{y}=\mathbf{0}\}$ | span$\{\mathbf{u}_{r+1},\ldots,\mathbf{u}_m\}$ | $m-r$ |
| **Row space** | $\{\text{col}(A^\top)\}$ | span$\{\mathbf{v}_1,\ldots,\mathbf{v}_r\}$ | $r$ |
| **Null space** (kernel) | $\{\mathbf{x}: A\mathbf{x}=\mathbf{0}\}$ | span$\{\mathbf{v}_{r+1},\ldots,\mathbf{v}_n\}$ | $n-r$ |

The column space is spanned by the first $r$ left singular vectors; the null space by the last $n-r$ right singular vectors. These pairs are **orthogonal complements**: column space $\perp$ left null space in $\mathbb{R}^m$; row space $\perp$ null space in $\mathbb{R}^n$.

**For AI:** In a linear layer $W \in \mathbb{R}^{d_{out} \times d_{in}}$:
- The row space is the subspace of inputs that "activate" the layer (non-null inputs).
- The null space is the subspace of inputs that are completely ignored.
- The column space is the set of reachable outputs.
- LoRA constrains $\Delta W$ to have column space within a rank-$r$ subspace.

---

## 3. Computing the SVD

### 3.1 Connection to Eigendecomposition

The most conceptually direct way to compute the SVD exploits the relationship $A^\top A = V\Sigma^\top\Sigma V^\top$ and $AA^\top = U\Sigma\Sigma^\top U^\top$:

**Algorithm (eigendecomposition-based SVD):**
1. Form $M = A^\top A \in \mathbb{R}^{n \times n}$ (symmetric PSD).
2. Compute eigendecomposition $M = V\Lambda V^\top$ with $\Lambda = \text{diag}(\lambda_1,\ldots,\lambda_n)$, $\lambda_i \geq 0$.
3. Set $\sigma_i = \sqrt{\lambda_i}$ and discard near-zero singular values (numerical rank).
4. For each $\sigma_i > 0$: $\mathbf{u}_i = A\mathbf{v}_i / \sigma_i$.
5. Extend $\{\mathbf{u}_i\}$ to an orthonormal basis of $\mathbb{R}^m$ (e.g., by Gram-Schmidt or QR).

This works, but it has a critical flaw: **forming $A^\top A$ squares the condition number**. If $\kappa(A) = \sigma_1/\sigma_r$, then $\kappa(A^\top A) = \kappa(A)^2$. For an ill-conditioned matrix (say $\kappa(A) = 10^7$), forming $A^\top A$ destroys the last 7 decimal digits of information in double precision. The Golub-Reinsch algorithm avoids this.

**For small, well-conditioned matrices** this approach is perfectly fine and is what `np.linalg.eigh(A.T @ A)` computes. For production SVD, use `np.linalg.svd` which implements Golub-Reinsch.

### 3.2 Golub-Reinsch Algorithm

The Golub-Reinsch algorithm (1965) is the standard method implemented in LAPACK and numpy. It proceeds in two phases:

**Phase 1: Bidiagonalisation.** Using Householder reflectors, reduce $A$ to upper bidiagonal form:
$$A = U_1 B V_1^\top$$
where $B \in \mathbb{R}^{n \times n}$ is upper bidiagonal (nonzeros only on main diagonal and superdiagonal) and $U_1, V_1$ are products of Householder matrices. This costs $O(mn^2 - n^3/3)$ operations.

**Phase 2: Diagonalisation of $B$.** Apply implicit QR iterations with Wilkinson shifts to $B$, converging to diagonal form $B = U_2 \Sigma V_2^\top$. Combining: $A = (U_1 U_2) \Sigma (V_1 V_2)^\top = U\Sigma V^\top$.

The key advantage: bidiagonalisation works directly on $A$ without forming $A^\top A$, so the condition number is not squared. The algorithm is backward stable with error $O(\epsilon_{\text{mach}} \|A\|)$.

**Cost:** $O(mn^2 + n^3)$ for $m \geq n$. This is too expensive for large sparse matrices.

### 3.3 Randomised SVD

For large matrices where only the top-$k$ singular values/vectors are needed, the randomised SVD (Halko, Martinsson, Tropp 2011) offers a dramatic speedup:

**Algorithm:**
1. **Sketch:** Draw $\Omega \in \mathbb{R}^{n \times (k+p)}$ with i.i.d. Gaussian entries ($p$ = oversampling, typically 5–10).
2. **Range sketch:** Compute $Y = A\Omega \in \mathbb{R}^{m \times (k+p)}$.
3. **Orthogonalise:** QR decompose $Y = QR$; $Q \in \mathbb{R}^{m \times (k+p)}$ captures the range of $A$.
4. **Project:** Compute $B = Q^\top A \in \mathbb{R}^{(k+p) \times n}$ (small matrix).
5. **SVD of small matrix:** Compute $B = \hat{U}\Sigma V^\top$ exactly.
6. **Recover:** $U = Q\hat{U}$. Return $(U[:, :k], \Sigma[:k,:k], V[:, :k])$.

**Cost:** $O(mnk)$ — linear in the matrix size! Compare to $O(mn\min(m,n))$ for full SVD.

**Error bound:** With oversampling $p \geq 5$, the approximation error is:
$$\|A - A_k^{\text{rand}}\|_2 \leq \left(1 + 11\sqrt{(k+p)n}\right) \sigma_{k+1}(A)$$
with high probability. In practice the error is much closer to $\sigma_{k+1}$.

**Power iteration improvement:** Replace $Y = A\Omega$ with $Y = (AA^\top)^q A\Omega$ for small $q$ (1–3 power iteration steps). This dramatically sharpens the approximation for matrices with slowly decaying singular values, at cost $O(qmnk)$.

**For AI:** `sklearn.utils.extmath.randomized_svd(A, n_components=k, n_iter=2)` implements this. It's what `TruncatedSVD` uses for large sparse document matrices. PyTorch's `torch.svd_lowrank` uses randomised SVD for LoRA-related computations.

### 3.4 Lanczos Bidiagonalisation

For very large sparse matrices (e.g., $A \in \mathbb{R}^{10^6 \times 10^6}$ with $O(n)$ nonzeros), Lanczos bidiagonalisation builds a Krylov subspace for the SVD problem:

**Algorithm:** Starting from a unit vector $\mathbf{v}_1$, alternate between multiplying by $A$ and $A^\top$:
$$\beta_{j+1}\mathbf{u}_{j+1} = A\mathbf{v}_j - \alpha_j\mathbf{u}_j$$
$$\gamma_{j+1}\mathbf{v}_{j+1} = A^\top\mathbf{u}_{j+1} - \beta_{j+1}\mathbf{v}_j$$

After $k$ steps, this builds a bidiagonal matrix $B_k$ whose singular values approximate the largest (and sometimes smallest) singular values of $A$. Only matrix-vector products $A\mathbf{x}$ and $A^\top\mathbf{y}$ are needed — no explicit matrix storage.

This is implemented in `scipy.sparse.linalg.svds` and is what makes SVD feasible for web-scale matrices (e.g., the Netflix prize data had $\sim 10^8$ entries).

### 3.5 Numerical Considerations

**Do not form $A^\top A$ for ill-conditioned problems.** As noted in Section 3.1, this squares the condition number. Use `np.linalg.svd(A)` directly, not `np.linalg.eigh(A.T @ A)`.

**Numerical rank.** In finite precision, all singular values are slightly nonzero even if the true rank is less than $\min(m,n)$. The numerical rank is the number of singular values greater than a threshold $\tau = \epsilon_{\text{mach}} \cdot \sigma_1 \cdot \max(m, n)$ (where $\epsilon_{\text{mach}} \approx 2.2 \times 10^{-16}$ for float64). `np.linalg.matrix_rank(A)` uses exactly this threshold.

**One-sided Jacobi SVD** achieves machine accuracy even for matrices with a condition number of $10^{15}$ by working directly on the columns of $A$ without any intermediate matrix products. Used in high-accuracy requirements (e.g., geodesy, orbit determination).

**Mixed precision.** In training, gradients are often computed in float16/bfloat16 but SVD is performed in float32 for stability. This matters for gradient-based LoRA rank selection.

```
NUMERICAL SVD PITFALLS
════════════════════════════════════════════════════════════════════════

  A has condition kappa:    A^T A has condition kappa^2

  kappa(A) = 1e7  =>  kappa(A^T A) = 1e14   (only 2 correct digits
                                               in last eigenvalue!)

  SAFE:   U, s, Vt = np.linalg.svd(A)         ← O(mn min(m,n))
  RISKY:  eigs = np.linalg.eigh(A.T @ A)      ← avoidable precision loss

════════════════════════════════════════════════════════════════════════
```

---

## 4. Properties of the SVD

### 4.1 Norms via Singular Values

The three most important matrix norms all have elegant expressions in terms of singular values:

| Norm | Definition | SVD Expression | Use in AI |
|---|---|---|---|
| **Spectral norm** $\|A\|_2$ | $\max_{\|\mathbf{x}\|=1}\|A\mathbf{x}\|$ | $\sigma_1$ (largest SV) | Lipschitz constant; GAN stability |
| **Frobenius norm** $\|A\|_F$ | $\sqrt{\sum_{i,j}A_{ij}^2}$ | $\sqrt{\sum_i \sigma_i^2}$ | Weight decay regularisation |
| **Nuclear norm** $\|A\|_*$ | $\sum_i \sigma_i$ | $\sum_i \sigma_i$ | Low-rank regularisation; matrix completion |

**Spectral norm $\|A\|_2 = \sigma_1$:** This is the operator norm — the maximum factor by which $A$ can amplify a vector. For a neural network layer $W$, $\|W\|_2 = \sigma_1(W)$ is the Lipschitz constant. Spectral normalisation $W \to W/\sigma_1(W)$ enforces $\|W\|_2 = 1$.

**Frobenius norm $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$:** This is the matrix version of the Euclidean norm. Note $\|A\|_F^2 = \text{tr}(A^\top A) = \sum_i \sigma_i^2$. L2 weight decay minimises $\|W\|_F^2$, which in SVD terms means penalising all singular values equally.

**Nuclear norm $\|A\|_* = \sum_i \sigma_i$:** This is the convex envelope (tightest convex relaxation) of the rank function. Minimising $\|A\|_*$ promotes low-rank solutions. Nuclear norm regularisation is used in matrix completion, recommender systems, and multi-task learning.

**Norm inequalities:** For any $A \in \mathbb{R}^{m \times n}$:
$$\|A\|_2 \leq \|A\|_F \leq \sqrt{r} \|A\|_2 \leq \sqrt{\min(m,n)}\|A\|_2$$
$$\|A\|_2 \leq \|A\|_* \leq \sqrt{r}\|A\|_F$$

### 4.2 The Eckart-Young Theorem

This is one of the most important theorems in applied mathematics — it justifies every low-rank approximation method in machine learning.

**Theorem (Eckart-Young, 1936).** Let $A = U\Sigma V^\top$ be the SVD with singular values $\sigma_1 \geq \cdots \geq \sigma_r > 0$. Define the rank-$k$ truncation:
$$A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = U_k \Sigma_k V_k^\top$$

Then for both the spectral norm and the Frobenius norm:
$$A_k = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_2 = \arg\min_{\text{rank}(B) \leq k} \|A - B\|_F$$

with optimal errors:
$$\|A - A_k\|_2 = \sigma_{k+1} \qquad \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

**Proof sketch (spectral norm).** For any rank-$k$ matrix $B$, there exists a unit vector $\mathbf{x}$ in the $(k+1)$-dimensional span of $\{\mathbf{v}_1,\ldots,\mathbf{v}_{k+1}\}$ that lies in the null space of $B$ (by dimension counting: $k+1 > k = n - \dim(\text{null}(B))$). For this $\mathbf{x}$:
$$\|A - B\|_2 \geq \|(A-B)\mathbf{x}\|_2 = \|A\mathbf{x}\|_2 \geq \sigma_{k+1}$$
since $\mathbf{x}$ is in the span of the top-$k+1$ right singular vectors. The bound is achieved by $A_k$ since $(A - A_k)\mathbf{v}_i = \mathbf{0}$ for $i \leq k$ and $= \sigma_i\mathbf{u}_i$ for $i > k$. $\square$

**Practical significance:** The Eckart-Young theorem is the mathematical foundation of:
- **PCA:** The rank-$k$ SVD of the centred data matrix gives the best rank-$k$ reconstruction (Section 6.1).
- **LoRA:** The low-rank decomposition $\Delta W = BA$ approximates the full gradient update $\Delta W$ with minimum Frobenius error.
- **Image compression:** A rank-$k$ image approximation discards all but the top $k$ singular components; Eckart-Young guarantees this is optimal.
- **Recommender systems:** Matrix factorisation finds the best low-rank model of the user-item rating matrix.

### 4.3 The Moore-Penrose Pseudoinverse

The **Moore-Penrose pseudoinverse** $A^+ \in \mathbb{R}^{n \times m}$ is the unique matrix satisfying the four Moore-Penrose conditions:
1. $AA^+A = A$
2. $A^+AA^+ = A^+$
3. $(AA^+)^\top = AA^+$
4. $(A^+A)^\top = A^+A$

The SVD gives the explicit formula:
$$\boxed{A^+ = V\Sigma^+U^\top}$$
where $\Sigma^+ \in \mathbb{R}^{n \times m}$ replaces each nonzero $\sigma_i$ with $1/\sigma_i$ and leaves zeros as zeros.

**Geometric meaning:** $AA^+$ is the orthogonal projection onto the column space of $A$; $A^+A$ is the orthogonal projection onto the row space of $A$.

**Solving linear systems:**
- **Overdetermined ($m > n$, full column rank):** The least-squares solution $\hat{\mathbf{x}} = \arg\min\|A\mathbf{x}-\mathbf{b}\|_2$ is $\hat{\mathbf{x}} = A^+\mathbf{b} = (A^\top A)^{-1}A^\top\mathbf{b}$ (normal equations). The minimum residual is $\|(I - AA^+)\mathbf{b}\|_2$.
- **Underdetermined ($m < n$, full row rank):** The minimum-norm solution $\hat{\mathbf{x}} = \arg\min\|\mathbf{x}\|_2$ s.t. $A\mathbf{x}=\mathbf{b}$ is $\hat{\mathbf{x}} = A^+\mathbf{b} = A^\top(AA^\top)^{-1}\mathbf{b}$.
- **Rank-deficient:** $A^+\mathbf{b}$ gives the minimum-norm least-squares solution.

**Ridge regression / Tikhonov regularisation:** Adding $\lambda\|\mathbf{x}\|_2^2$ to the objective gives:
$$\hat{\mathbf{x}}_\lambda = (A^\top A + \lambda I)^{-1}A^\top\mathbf{b} = V(\Sigma^\top\Sigma + \lambda I)^{-1}\Sigma^\top U^\top \mathbf{b} = \sum_{i=1}^r \frac{\sigma_i}{\sigma_i^2 + \lambda} (\mathbf{u}_i^\top \mathbf{b}) \mathbf{v}_i$$

Each singular component is attenuated by factor $\sigma_i^2/(\sigma_i^2 + \lambda)$ — components with $\sigma_i \gg \sqrt{\lambda}$ are kept, components with $\sigma_i \ll \sqrt{\lambda}$ are suppressed. This is **spectral shrinkage** or Tikhonov regularisation.

### 4.4 SVD and Rank

The SVD provides the most numerically stable definition of rank:

**Exact rank:** $\text{rank}(A)$ = number of positive singular values.

**Numerical rank:** In finite precision, use $\text{rank}_\tau(A) = |\{i : \sigma_i > \tau\}|$ where typically $\tau = \epsilon_{\text{mach}} \cdot \sigma_1 \cdot \max(m,n)$.

**Stable rank:** The stable rank $\text{sr}(A) = \|A\|_F^2 / \|A\|_2^2 = \sum_i \sigma_i^2 / \sigma_1^2$ is a smooth, noise-robust proxy for rank. It lies in $[1, r]$ and equals 1 iff $A$ is rank-1. For LoRA analysis, if $\text{sr}(\Delta W) \ll \min(m,n)$, the update is intrinsically low-rank and can be well-approximated with small $r$.

**Effective rank (Roy & Vetterli):** $\text{er}(A) = \exp(H(p))$ where $p_i = \sigma_i^2/\|A\|_F^2$ and $H(p) = -\sum p_i \log p_i$ is the entropy of the normalised squared singular values. This is a smooth measure that is maximised (= $r$) for uniform singular value spectra and minimised (= 1) for rank-1 matrices.

### 4.5 Condition Number

The **condition number** of a matrix $A$ with respect to the 2-norm is:
$$\kappa(A) = \|A\|_2 \cdot \|A^+\|_2 = \frac{\sigma_1}{\sigma_r}$$

(where $\sigma_r$ is the smallest positive singular value). For a square invertible matrix, $\kappa(A) = \sigma_1/\sigma_n$.

**Meaning:** If $A\mathbf{x} = \mathbf{b}$ and we perturb $\mathbf{b}$ by $\delta\mathbf{b}$, the relative perturbation in $\mathbf{x}$ satisfies:
$$\frac{\|\delta\mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \cdot \frac{\|\delta\mathbf{b}\|}{\|\mathbf{b}\|}$$

A well-conditioned matrix has $\kappa(A) \approx 1$; a nearly singular matrix has $\kappa(A) \gg 1$.

**For AI:** The condition number of the Gram matrix $X^\top X$ (or the Hessian $H = \nabla^2\mathcal{L}$) governs gradient descent convergence: $\kappa(H) = \lambda_{\max}/\lambda_{\min} = \sigma_{\max}^2/\sigma_{\min}^2$ for symmetric PSD matrices. Feature normalisation (standardisation, batch norm, layer norm) reduces $\kappa(X^\top X)$, accelerating convergence.

**Preconditioning:** For linear systems $A\mathbf{x} = \mathbf{b}$, a preconditioner $P \approx A$ transforms the system to $P^{-1}A\mathbf{x} = P^{-1}\mathbf{b}$ with smaller $\kappa(P^{-1}A)$. K-FAC (Kronecker-Factored Approximate Curvature) is a natural gradient method that preconditioning by the Fisher information matrix — equivalent to using a block-diagonal approximation to the Hessian.

### 4.6 The Procrustes Problem

The **orthogonal Procrustes problem** asks: given matrices $A, B \in \mathbb{R}^{m \times n}$, find the orthogonal matrix $R \in \mathbb{R}^{m \times m}$ ($R^\top R = I$) minimising $\|RA - B\|_F$. The solution is:

**Solution:** Compute $BA^\top = U\Sigma V^\top$ (SVD). Then $\hat{R} = UV^\top$.

**Proof:** $\|RA - B\|_F^2 = \|RA\|_F^2 - 2\langle RA, B\rangle_F + \|B\|_F^2$. Since $R$ is orthogonal, $\|RA\|_F = \|A\|_F$ is constant. Minimising over $R$ means maximising $\text{tr}(R^\top B A^\top) = \text{tr}(R^\top U\Sigma V^\top)$. Setting $W = V^\top R^\top U$, this is $\text{tr}(W\Sigma) = \sum_i W_{ii}\sigma_i \leq \sum_i\sigma_i$ with equality when $W = I$, i.e., $R = UV^\top$. $\square$

**For AI:**
- **Shape alignment:** Procrustes is used in protein structure alignment, morphometric analysis.
- **Domain adaptation:** Aligning word vector spaces across languages (cross-lingual transfer) uses Procrustes.
- **RetNet / linear attention:** Some position encoding methods use orthogonal transformations computed via Procrustes.
- **Gradient alignment:** The Procrustes problem appears in task arithmetic — finding the optimal rotation to align task vectors.

---

## 5. Outer Product Form and Rank-1 Decomposition

### 5.1 Rank-1 Outer Product Decomposition

The SVD has a beautiful outer product representation: every matrix is a weighted sum of rank-1 matrices:

$$A = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$$

Each term $\sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ is a rank-1 matrix (outer product of two vectors). The singular values $\sigma_i$ act as importance weights: the first term contributes the most to $A$, the last the least. The truncated sum $A_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i\mathbf{v}_i^\top$ is the best rank-$k$ approximation (Eckart-Young).

```
RANK-1 DECOMPOSITION
════════════════════════════════════════════════════════════════════════

  A  =  σ₁ · u₁v₁ᵀ  +  σ₂ · u₂v₂ᵀ  +  ···  +  σᵣ · uᵣvᵣᵀ
        ─────────────   ─────────────          ─────────────
         rank-1 term      rank-1 term            rank-1 term
         (dominant)       (less important)       (least important)

  Truncating at rank k: discard σ_{k+1},...,σ_r terms
  Error: ‖A - A_k‖_F = √(σ_{k+1}² + ··· + σ_r²)

════════════════════════════════════════════════════════════════════════
```

**For AI — matrix as a sum of "features":** Each rank-1 term $\sigma_i \mathbf{u}_i\mathbf{v}_i^\top$ is an "atom" — a pattern in the output ($\mathbf{u}_i$) correlated with a pattern in the input ($\mathbf{v}_i$), with strength $\sigma_i$. In a trained weight matrix $W$:
- The top singular direction ($\sigma_1, \mathbf{u}_1, \mathbf{v}_1$) is the most amplified input-output feature pair.
- Lower singular directions correspond to less important, often noisier associations.
- LoRA's $\Delta W = BA$ with $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$ is equivalent to adding $r$ rank-1 terms to $W$.

### 5.2 Incremental and Streaming SVD

When data arrives in a stream and the SVD must be updated without recomputing from scratch, **incremental SVD** methods are used. Brand's method (2002) updates the SVD $A = U\Sigma V^\top$ to $(A, \mathbf{c}) = [A | \mathbf{c}]$ in $O(rn)$ time (rather than recomputing from scratch in $O(mn^2)$).

**Algorithm sketch (rank-1 update):**
1. Project $\mathbf{c}$ onto current bases: $\mathbf{p} = U^\top\mathbf{c}$, $\mathbf{q} = \mathbf{c} - U\mathbf{p}$.
2. Form augmented bidiagonal $(2r+1) \times (r+1)$ matrix.
3. Compute small SVD and update.

This is used in online recommender systems (user ratings arrive one at a time), PCA on streaming data, and online learning scenarios.

### 5.3 Symmetric Matrices: SVD Equals Eigendecomposition

For symmetric matrices, the SVD and eigendecomposition coincide (with a sign adjustment for indefinite matrices):

**Case 1: Symmetric PSD ($A = A^\top$, all $\lambda_i \geq 0$).** The spectral theorem gives $A = Q\Lambda Q^\top$. This is already an SVD with $U = V = Q$ and $\Sigma = \Lambda$. Singular values equal eigenvalues.

**Case 2: Symmetric but indefinite ($A = A^\top$, mixed signs).** Say $A = Q\Lambda Q^\top$ with some $\lambda_i < 0$. Then $|\lambda_i|$ are the singular values. The sign of $\lambda_i$ is absorbed into $U$: if $\lambda_i < 0$, then $\mathbf{u}_i = -\mathbf{q}_i$ (flip the sign of the left singular vector). So $U \neq V$ for indefinite symmetric matrices.

**Consequence:** For a symmetric matrix, $\sigma_i = |\lambda_i|$. The singular values are the absolute values of the eigenvalues. If you compute `np.linalg.eigvalsh(A)` and `np.linalg.svd(A, compute_uv=False)`, the latter will equal the absolute values of the former, sorted in descending order.

---

## 6. SVD in Machine Learning and AI

### 6.1 Low-Rank Approximation and LoRA

**LoRA (Low-Rank Adaptation, Hu et al. 2021)** is the most widely used parameter-efficient fine-tuning method for large language models. Its mathematical foundation is the Eckart-Young theorem.

The key observation: when fine-tuning a pre-trained model on a new task, the weight updates $\Delta W = W_{\text{fine}} - W_{\text{pre}}$ have low intrinsic rank. Rather than storing and updating the full $\Delta W \in \mathbb{R}^{d \times k}$ (which has $dk$ parameters), LoRA parameterises:

$$\Delta W = BA, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}, \quad r \ll \min(d, k)$$

Only $A$ and $B$ are trained (with $A$ initialised from a Gaussian and $B$ initialised to zero so $\Delta W_0 = 0$). The forward pass becomes $W_0\mathbf{x} + BA\mathbf{x}/\sqrt{r}$ (the $1/\sqrt{r}$ scaling stabilises training).

**SVD connection:** The best rank-$r$ initialisation for $(B, A)$ comes from the SVD of $\Delta W$: $B = U_r\Sigma_r^{1/2}$, $A = \Sigma_r^{1/2}V_r^\top$. In practice, LoRA uses random initialisation (since $\Delta W$ is unknown before training), but SVD is used post-hoc to analyse which singular directions were actually modified.

**DoRA (Liu et al. 2024)** further decomposes: $W = m \cdot W/\|W\|_c$ where $m = \|W\|_c$ is the column-wise norm. It then applies LoRA to the directional component, achieving better performance per parameter.

**MLA (DeepSeek 2024)** compresses the KV cache by factoring the projection matrices: $W_K = W_C^{DK} W_C^{UK}$ where $W_C^{DK} \in \mathbb{R}^{d_h \times c}$ compresses to a latent dimension $c \ll d_h$. This is exactly the rank-$c$ SVD structure applied to the key/value projections.

**Parameters saved by LoRA:** Instead of $dk$ parameters for $\Delta W$, LoRA uses $(d + k)r$ parameters. For $d = k = 4096$ and $r = 8$: full is $16.7M$ parameters, LoRA is $65K$ — a $256\times$ reduction.

### 6.2 Recommender Systems

Recommender systems model the user-item interaction matrix $R \in \mathbb{R}^{n_u \times n_i}$ where $R_{ij}$ is user $i$'s rating of item $j$ (or 0 if unrated). The goal is to predict missing entries.

**SVD-based collaborative filtering:** Approximate $R \approx U\Sigma V^\top$ (truncated to rank $k$), where:
- $U \in \mathbb{R}^{n_u \times k}$: user latent factors (each user's preferences in $k$-dimensional space)
- $V \in \mathbb{R}^{n_i \times k}$: item latent factors (each item's properties in $k$-dimensional space)
- $\sigma_i$: importance of the $i$-th latent dimension

Predicted rating: $\hat{R}_{ij} = \sum_{f=1}^k \sigma_f U_{if} V_{jf}$.

**Challenge:** $R$ is sparse (mostly unobserved). Direct SVD fills missing values with 0, which is wrong. **Simon Funk's SGD-SVD** (2006, won one of the Netflix Prize progress checks) instead minimises $\sum_{(i,j)\text{ observed}} (R_{ij} - U_i^\top V_j)^2 + \lambda(\|U\|_F^2 + \|V\|_F^2)$ via stochastic gradient descent.

**Non-negative Matrix Factorisation (NMF):** Constrains $U, V \geq 0$ elementwise. Gives "parts-based" representations (pixels, topics, users). NMF is a constrained SVD with non-negativity.

### 6.3 Latent Semantic Analysis and Word Vectors

**Latent Semantic Analysis (LSA, Deerwester et al. 1990)** applies truncated SVD to the term-document matrix $X \in \mathbb{R}^{|V| \times |D|}$ where $X_{ij}$ = tf-idf weight of term $i$ in document $j$.

The rank-$k$ SVD $X \approx U_k\Sigma_k V_k^\top$ gives:
- $U_k \in \mathbb{R}^{|V| \times k}$: word embeddings in the $k$-dimensional latent semantic space
- $V_k \in \mathbb{R}^{|D| \times k}$: document embeddings
- $\Sigma_k$: importance of each latent dimension

Words that co-occur in similar documents get nearby embeddings. This is one of the first distributed word representations — a precursor to word2vec and GloVe.

**Connection to GloVe:** GloVe (Pennington et al. 2014) can be interpreted as a shifted version of LSA. The GloVe model minimises the Frobenius norm between the log pointwise mutual information (PPMI) matrix and its low-rank factorisation — structurally an SVD with a different weighting function.

**Modern transformer word embeddings** are not directly SVD, but the embedding table $E \in \mathbb{R}^{|V| \times d}$ can be analysed via SVD. Well-trained embeddings typically have a steep singular value spectrum (dominant directions = common syntactic/semantic features; tail = noise).

### 6.4 Image Compression

The rank-$k$ SVD of a grayscale image $I \in \mathbb{R}^{m \times n}$ gives:
$$I \approx I_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i\mathbf{v}_i^\top$$

Storage: $k(m + n + 1)$ numbers vs $mn$ for the full image. Compression ratio: $mn / [k(m+n+1)]$.

For a $512 \times 512$ image with $k = 50$: original stores $262K$ numbers, compressed stores $51K$ — a $5\times$ compression. JPEG achieves better compression ratios than SVD at the same quality because it exploits block structure and human visual system properties, but SVD compression is mathematically optimal in the $\ell^2$ sense.

**Colour images:** Apply SVD independently to each channel (R, G, B), or decompose the matrix of RGB triples. In practice, the colour channels are correlated and the singular value spectra of different channels are similar.

### 6.5 Pseudoinverse and Least Squares

**Overdetermined systems (linear regression):** Given $X \in \mathbb{R}^{n \times d}$ (data, $n \gg d$) and $\mathbf{y} \in \mathbb{R}^n$, the least-squares problem $\min_{\boldsymbol{\beta}} \|X\boldsymbol{\beta} - \mathbf{y}\|_2$ has solution:
$$\hat{\boldsymbol{\beta}} = X^+\mathbf{y} = V\Sigma^+U^\top\mathbf{y} = \sum_{i=1}^r \frac{\mathbf{u}_i^\top\mathbf{y}}{\sigma_i}\mathbf{v}_i$$

Each component $(\mathbf{u}_i^\top\mathbf{y})/\sigma_i$ is the projection of $\mathbf{y}$ onto the $i$-th left singular vector, divided by the singular value. If $\sigma_i$ is very small (ill-conditioned), this component is amplified enormously — catastrophic if $\mathbf{u}_i^\top\mathbf{y}$ contains noise.

**Ridge regression shrinkage:** The regularised solution is:
$$\hat{\boldsymbol{\beta}}_\lambda = \sum_{i=1}^r \frac{\sigma_i}{\sigma_i^2 + \lambda} (\mathbf{u}_i^\top\mathbf{y})\mathbf{v}_i$$

For $\sigma_i \gg \sqrt{\lambda}$: factor $\to \sigma_i/\sigma_i^2 = 1/\sigma_i$ (same as unregularised). For $\sigma_i \ll \sqrt{\lambda}$: factor $\to \sigma_i/\lambda \approx 0$ (component is suppressed). The threshold $\sqrt{\lambda}$ acts as an effective singular value cutoff.

**For AI:** The normal equations $(X^\top X)\hat{\boldsymbol{\beta}} = X^\top\mathbf{y}$ are equivalent but numerically inferior (condition number squared). Always use `np.linalg.lstsq(X, y)` which uses the SVD internally.

### 6.6 Spectral Methods for Graphs

For a bipartite graph with vertices $U, V$ and edge weight matrix $A \in \mathbb{R}^{|U| \times |V|}$ (biadjacency matrix), the SVD $A = U\Sigma V^\top$ provides a spectral embedding:
- Left singular vectors $U$: embed the $U$-side vertices
- Right singular vectors $V$: embed the $V$-side vertices
- Singular values: weight each dimension

This is used in **spectral co-clustering** (Dhillon 2001): simultaneously cluster rows and columns of a data matrix by finding the top-$k$ singular vectors and applying $k$-means to the embedded points.

**Connection to symmetric spectral clustering:** For a bipartite graph, the SVD of the normalised biadjacency matrix is equivalent to the eigendecomposition of the symmetric Laplacian of the full bipartite graph. The $k$-th singular value of $A$ corresponds to the $(k+1)$-th eigenvalue of the graph Laplacian.

### 6.7 Weight Matrix Analysis (WeightWatcher)

**WeightWatcher** (Martin and Mahoney, 2019) is a tool for diagnosing model quality without test data by analysing the singular value spectra of weight matrices.

**Key finding:** Well-trained neural networks have weight matrices with **heavy-tailed singular value spectra** that follow power laws:
$$\rho(\sigma) \sim \sigma^{-(\alpha+1)}, \quad \sigma_i \sim i^{-1/\alpha}$$
with $\alpha \in [2, 4]$ for high-quality models. This is predicted by **random matrix theory** (Marchenko-Pastur distribution) for random matrices, with the "learned" part manifesting as the heavy tail.

**Diagnostic metrics:**
- **$\hat{\alpha}$** (power-law exponent): fit to the tail of the singular value distribution. $\hat{\alpha} \in [2, 4]$ indicates well-trained; $\hat{\alpha} > 6$ indicates undertrained; $\hat{\alpha} < 2$ indicates overtrained.
- **Weighted alpha**: $\sum_l \hat{\alpha}_l \log(\sigma_{1,l})$ averaged across layers, weighted by layer size.
- **Stable rank**: measures effective rank per layer.

**For AI:** WeightWatcher can predict model quality (test accuracy) from just the weight matrices, without any data. It has been used to select the best checkpoint during training, identify layers that need more training, and detect fine-tuning quality.

### 6.8 Attention and SVD

The attention weight matrix $A = \text{softmax}(QK^\top/\sqrt{d_k}) \in \mathbb{R}^{n \times n}$ (for sequence length $n$) has interesting SVD structure in trained transformers:

**Low-rank structure of attention:** Trained attention matrices are approximately low-rank — most of the attention pattern is captured by a few singular vectors. This has been observed empirically across multiple model families. The effective rank is roughly $O(\log n)$ for induction heads and $O(1)$ for copying heads.

**SVD-based attention compression:** Several works approximate the attention computation using truncated SVD of $Q$ and $K$: write $Q = U_Q\Sigma_Q V_Q^\top$, $K = U_K\Sigma_K V_K^\top$, then $QK^\top = U_Q\Sigma_Q(V_Q^\top V_K)\Sigma_K U_K^\top$. If the middle factor $V_Q^\top V_K$ has low rank, the computation is cheaper.

**FlashAttention** does not directly use SVD but exploits the block structure of attention matrices for I/O-efficient computation. The theoretical analysis of FlashAttention's approximation quality involves the singular value decay of the attention matrix.

**MLA's KV compression:** DeepSeek's MLA projects $K, V$ through a low-rank bottleneck $W^{DKV} \in \mathbb{R}^{d \times c}$ ($c \ll d$). At inference, only the $c$-dimensional latent $\mathbf{c}^{KV} = W^{DKV}\mathbf{h}$ is cached per token, reducing KV cache by factor $d/c$. This is exactly a rank-$c$ SVD-style projection of the key-value information.

---

## 7. Generalised SVD and Extensions

### 7.1 Generalised SVD (GSVD)

The **Generalised SVD** (GSVD) factorises a pair of matrices $(A, B)$ with the same number of columns:

$$A = U C X^\top, \quad B = V S X^\top$$

where $U$, $V$ are orthogonal, $X$ is non-singular, and $C$, $S$ are diagonal with $C_{ii}^2 + S_{ii}^2 = 1$.

**Applications:**
- **Regularised least squares:** Solve $\min\|A\mathbf{x}-\mathbf{b}\|_2^2 + \lambda^2\|B\mathbf{x}\|_2^2$ via GSVD.
- **Generalised eigenproblems:** $Av = \lambda Bv$ is solved via GSVD of $(A, B)$.
- **Fisher LDA:** The generalised eigenproblem $S_B\mathbf{v} = \lambda S_W\mathbf{v}$ (between/within scatter) is a GSVD problem.
- **CCA:** Canonical Correlation Analysis decomposes the cross-covariance matrix and uses a GSVD-like structure.

Available in SciPy via `scipy.linalg.svd` with the `lapack_driver='gesdd'` option, or directly through LAPACK's `dggsvd` routine.

### 7.2 Tensor SVD and Tucker Decomposition

For tensors (multi-dimensional arrays) $\mathcal{A} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_d}$, several SVD-like decompositions exist:

**Tucker Decomposition:** $\mathcal{A} \approx \mathcal{G} \times_1 U_1 \times_2 U_2 \cdots \times_d U_d$ where $\mathcal{G} \in \mathbb{R}^{r_1 \times r_2 \times \cdots \times r_d}$ is the core tensor and $U_k \in \mathbb{R}^{n_k \times r_k}$ are orthogonal factor matrices. The **Higher-Order SVD (HOSVD)** computes each $U_k$ as the SVD of the mode-$k$ unfolding of $\mathcal{A}$.

**CP Decomposition (CANDECOMP/PARAFAC):** $\mathcal{A} \approx \sum_{r=1}^R \mathbf{a}_r^{(1)} \otimes \mathbf{a}_r^{(2)} \otimes \cdots \otimes \mathbf{a}_r^{(d)}$ — a sum of rank-1 tensors. Unlike matrix SVD, tensor rank is NP-hard to compute in general, and the best rank-$R$ approximation may not exist (unlike Eckart-Young for matrices).

**For AI:** Tucker decomposition is used to compress convolutional layers (a 4D weight tensor). MobileNet-style depthwise separable convolutions are an approximate Tucker decomposition. Tensor train (TT) decomposition is used for compressing embedding tables and attention weight matrices in extreme low-resource settings.

### 7.3 Truncated and Randomised Variants in Practice

| Method | Function | When to Use | Cost |
|---|---|---|---|
| Full SVD | `np.linalg.svd(A)` | Small matrices, need all singular values | $O(mn\min(m,n))$ |
| Thin SVD | `np.linalg.svd(A, full_matrices=False)` | Standard case | $O(mn\min(m,n))$ |
| Truncated SVD | `scipy.sparse.linalg.svds(A, k=k)` | Sparse/large, need top-$k$ | $O(mnk)$ |
| Randomised SVD | `sklearn.utils.extmath.randomized_svd` | Dense large, approximate top-$k$ | $O(mnk)$ |
| Incremental SVD | `sklearn.decomposition.IncrementalPCA` | Streaming data | $O(nk^2)$ per batch |
| PyTorch low-rank | `torch.svd_lowrank(A, q=k)` | GPU tensors | $O(mnk)$ |

For LoRA initialisation from a checkpoint, the typical workflow is:
1. Load the fine-tuned weight matrix $W_{\text{fine}}$.
2. Compute $\Delta W = W_{\text{fine}} - W_0$.
3. Run `U, s, Vh = np.linalg.svd(delta_W, full_matrices=False)`.
4. Set $B = U[:, :r] \cdot \text{diag}(\sqrt{s[:r]})$, $A = \text{diag}(\sqrt{s[:r]}) \cdot Vh[:r, :]$.
5. Confirm $\|BA - \Delta W\|_F / \|\Delta W\|_F$ is small.

---

## 8. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Treating singular values as eigenvalues | $\sigma_i \neq \lambda_i$ in general; rotation matrix has eigenvalues $e^{\pm i\theta}$ but all $\sigma_i = 1$ | Remember: $\sigma_i = \sqrt{\lambda_i(A^\top A)}$; only equal for symmetric PSD matrices |
| 2 | Forming $A^\top A$ to compute the SVD | Squares the condition number ($\kappa(A^\top A) = \kappa(A)^2$), destroying small singular values | Use `np.linalg.svd(A)` directly; never `np.linalg.eigh(A.T @ A)` for the SVD |
| 3 | Assuming $U = V$ for non-symmetric matrices | $U$ and $V$ are different orthogonal matrices (different sizes for rectangular $A$); only $U = V$ if $A$ is symmetric PSD | Check dimensions: $U \in \mathbb{R}^{m\times m}$, $V \in \mathbb{R}^{n\times n}$ for an $m\times n$ matrix |
| 4 | Confusing sign conventions in $U, V$ | $(\mathbf{u}_i, \mathbf{v}_i) \to (-\mathbf{u}_i, -\mathbf{v}_i)$ leaves $A = U\Sigma V^\top$ unchanged; numpy may return different signs each run | Always check $A \approx U\Sigma V^\top$ numerically; compare absolute cosine similarity between vectors |
| 5 | Using full SVD when thin SVD suffices | Full SVD computes $U \in \mathbb{R}^{m\times m}$ including $m - r$ unnecessary columns that don't contribute | Use `np.linalg.svd(A, full_matrices=False)` for the thin SVD |
| 6 | Claiming rank from nonzero singular values in finite precision | Every matrix in float64 has all $\sigma_i > 0$ due to rounding | Use `np.linalg.matrix_rank(A)` with appropriate tolerance $\tau = \epsilon_{\text{mach}} \cdot \sigma_1 \cdot \max(m,n)$ |
| 7 | Applying LoRA to all layers equally | Not all layers have equally low-rank updates; key/query projections often need higher rank than MLP layers | Use `loftq`-style analysis to allocate rank $r_l$ per layer based on $\hat{\alpha}$ of each layer's update |
| 8 | Confusing nuclear norm with Frobenius norm | $\|A\|_* = \sum\sigma_i$ vs $\|A\|_F = \sqrt{\sum\sigma_i^2}$; minimising nuclear norm promotes sparsity in singular values (low rank), not in the matrix entries | Choose norm based on goal: low rank → nuclear norm; small magnitude → Frobenius norm |
| 9 | Thinking pseudoinverse solves all linear systems stably | $A^+$ amplifies small singular value components; for noisy $\mathbf{b}$, the component $(\mathbf{u}_i^\top\mathbf{b})/\sigma_i$ is huge when $\sigma_i \approx 0$ | Use ridge regression ($\lambda > 0$) or truncate singular values below a threshold |
| 10 | Using `scipy.linalg.svd` instead of `numpy.linalg.svd` interchangeably | `scipy.linalg.svd` returns `(U, s, Vh)` while `numpy.linalg.svd` returns `(U, s, Vh)` — same convention, but scipy offers more options; be consistent | Pick one and note that `Vh` is already $V^\top$, not $V$ |

---

## 9. Exercises

**Exercise 1 ★ — SVD by Hand (2×2)**

Compute the SVD of $A = \begin{pmatrix}3 & 1\\1 & 3\end{pmatrix}$ by hand:
(a) Form $A^\top A$ and find its eigenvalues and eigenvectors.
(b) Determine $\sigma_1, \sigma_2$ and $\mathbf{v}_1, \mathbf{v}_2$.
(c) Compute $\mathbf{u}_i = A\mathbf{v}_i/\sigma_i$ for each $i$.
(d) Write out $U$, $\Sigma$, $V$ and verify $U\Sigma V^\top = A$.
(e) What are the semi-axes of the ellipse $A$ maps the unit circle to?

**Exercise 2 ★ — Four Fundamental Subspaces via SVD**

Given $A = \begin{pmatrix}1&2&3\\4&5&6\\7&8&9\end{pmatrix}$ (rank 2):
(a) Compute the thin SVD.
(b) Identify a basis for the column space, row space, null space, and left null space from the SVD.
(c) Verify your null space basis by showing $A\mathbf{v} = \mathbf{0}$.
(d) Verify the column space basis by expressing $A$'s columns as linear combinations.

**Exercise 3 ★★ — Eckart-Young Reconstruction Error**

For the $5\times5$ matrix $A$ with singular values $[10, 5, 3, 1, 0.1]$:
(a) Implement `truncated_svd(A, k)` that returns the rank-$k$ approximation $A_k$.
(b) Compute the exact spectral norm error $\|A - A_k\|_2$ for $k = 1, 2, 3, 4$.
(c) Compute the Frobenius norm error $\|A - A_k\|_F$ for each $k$.
(d) Verify the Eckart-Young bound: $\|A - A_k\|_2 = \sigma_{k+1}$.
(e) What is the minimum rank needed to retain 95% of the Frobenius norm of $A$?

**Exercise 4 ★★ — Moore-Penrose Pseudoinverse**

Implement `pseudoinverse(A, tol=1e-10)` using the SVD:
(a) Compute the thin SVD $A = U\Sigma V^\top$.
(b) Invert the nonzero singular values (use `tol` to determine which are zero).
(c) Return $A^+ = V\Sigma^+U^\top$.
(d) Verify the four Moore-Penrose conditions for a random $4\times3$ matrix.
(e) Use your pseudoinverse to solve the overdetermined system $A\mathbf{x} = \mathbf{b}$ and verify it gives the minimum-norm least-squares solution.

**Exercise 5 ★★ — Condition Number and Least Squares Stability**

(a) Generate an ill-conditioned matrix $A \in \mathbb{R}^{10\times5}$ with singular values $[100, 50, 10, 1, 0.001]$ using random $U, V$.
(b) Set $\mathbf{b} = A\mathbf{x}^* + 0.01\boldsymbol{\epsilon}$ (add small noise $\boldsymbol{\epsilon}$).
(c) Solve using `np.linalg.lstsq` (SVD-based) and via normal equations $(A^\top A)^{-1}A^\top\mathbf{b}$.
(d) Compare errors $\|\hat{\mathbf{x}} - \mathbf{x}^*\|$ for both methods.
(e) Repeat with ridge regression for $\lambda \in [10^{-6}, 10^2]$ and plot the trade-off between fit and regularisation.

**Exercise 6 ★★★ — Randomised SVD**

Implement `randomized_svd(A, k, n_oversampling=5, n_iter=2)`:
(a) Generate Gaussian sketch matrix $\Omega \in \mathbb{R}^{n\times(k+p)}$.
(b) Compute the range sketch $Y = (AA^\top)^q A\Omega$ (with `n_iter=q` power iterations).
(c) Orthogonalise $Y$ via QR.
(d) Project and compute small SVD.
(e) Compare your implementation against `np.linalg.svd` and `sklearn.utils.extmath.randomized_svd` on a $500\times300$ matrix. Report relative Frobenius error vs rank.

**Exercise 7 ★★★ — LoRA-Style Low-Rank Adapter**

Simulate LoRA fine-tuning:
(a) Create a "pre-trained" weight matrix $W_0 \in \mathbb{R}^{64\times32}$ (random Gaussian).
(b) Create a "true fine-tuned" update $\Delta W^*$ with rank 4 (use SVD to construct).
(c) Implement `lora_decompose(delta_W, r)` that finds the rank-$r$ factorisation $B, A$ minimising $\|BA - \Delta W\|_F$.
(d) Compute the relative approximation error for $r = 1, 2, 4, 8, 16$.
(e) Show that $r = 4$ recovers $\Delta W^*$ almost exactly (since it's rank-4) while $r < 4$ introduces error. Plot the Eckart-Young error bound.

**Exercise 8 ★★★ — Image Compression via SVD**

(a) Load or generate a grayscale "image" matrix $I \in \mathbb{R}^{128\times128}$ (use a structured synthetic image with edges and gradients).
(b) Implement `svd_compress(I, k)` that returns the rank-$k$ approximation.
(c) Compute PSNR (peak signal-to-noise ratio) vs compression ratio for $k = 1, 5, 10, 20, 50, 100$.
(d) Plot the singular value spectrum and identify the "knee" where adding more singular values gives diminishing returns.
(e) Compare storage: full image vs rank-$k$ representation. What rank achieves $10\times$ compression?

---

## 10. Why This Matters for AI (2026 Perspective)

| Concept | Impact on AI/LLMs in 2026 | Specific Methods |
|---|---|---|
| **Eckart-Young theorem** | Every low-rank method is justified by this theorem | LoRA, DoRA, MLA, spectral pruning |
| **Truncated SVD** | Foundation of parameter-efficient fine-tuning; compress $B \cdot 256$ to $B \cdot 1$ parameters | LoRA (Hu 2021), AdaLoRA (Zhang 2023), QLoRA |
| **Randomised SVD** | Makes SVD practical at web scale; $O(mnk)$ vs $O(mn^2)$ | TruncatedSVD in sklearn, `torch.svd_lowrank` |
| **Spectral norm** | Lipschitz control of each layer; GAN stability; gradient norm bounding | Spectral normalisation (Miyato 2018), SN-GAN |
| **Nuclear norm** | Convex relaxation of rank; used in matrix completion and multi-task learning | Matrix completion, inductive matrix completion |
| **Pseudoinverse** | Numerically stable least squares; basis of linear probing | Linear regression, probing classifiers, OLS |
| **Condition number** | Explains ill-conditioning in training; motivates normalisation | BatchNorm, LayerNorm, feature scaling, K-FAC |
| **SVD of weight matrices** | Quality metric for trained models without test data | WeightWatcher, model selection, layer diagnosis |
| **SVD of attention matrices** | Explains attention head specialisation; guides compression | Attention pruning, low-rank attention, MLA |
| **Procrustes problem** | Cross-lingual transfer; domain adaptation; task arithmetic | MUSE (cross-lingual), task arithmetic, LoRA merge |
| **Singular value spectrum** | Power-law tail indicates implicit self-regularisation in SGD | WeightWatcher, alpha power law metric |
| **MLA / KV compression** | Low-rank projection of K/V matrices reduces inference cost | DeepSeek-V2/V3 MLA, GQA, MQA |

---

## 11. Conceptual Bridge

**Where we came from:** Section 03-01 (Eigenvalues and Eigenvectors) gave us the tools to decompose square matrices into invariant directions and scaling factors. But eigendecomposition is limited: it requires square matrices, may involve complex numbers, and fails for defective matrices. The SVD generalises this to any matrix by abandoning the requirement that input and output live in the same space. Instead of eigenvectors — directions that map to themselves — we have singular vectors: optimal input and output directions paired by the matrix's stretching action.

**Where we are going:** Section 03-03 (Principal Component Analysis) is, in its essence, the SVD of the centred data matrix. PCA finds the principal axes of data variation by computing the SVD of $\tilde{X} \in \mathbb{R}^{n \times d}$; the right singular vectors are the principal components, and the singular values (scaled) are the standard deviations in each principal direction. Everything you have learned about the SVD — Eckart-Young, pseudoinverse, truncation — applies directly to PCA.

**The SVD as a unifying lens:** The SVD connects to every major topic in the rest of this curriculum:
- **Matrix norms** (Section 03-06): spectral, Frobenius, and nuclear norms are all functions of singular values.
- **Positive definite matrices** (Section 03-07): $A$ is PSD iff all singular values equal eigenvalues and are non-negative.
- **Optimisation** (Chapter 08): the gradient of $\|A\|_*$ with respect to $A$ involves the singular vectors; nuclear norm minimisation is a semidefinite program solvable via SVD.
- **Probability** (Chapter 06): the covariance matrix $\Sigma = X^\top X / n$ has SVD directly related to the SVD of $X$; Mahalanobis distance uses $\Sigma^{-1/2} = V\Lambda^{-1/2}V^\top$.

```
CONCEPTUAL MAP — SVD IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  Eigenvalues &      Singular Value       Principal Component
  Eigenvectors  ──►  Decomposition   ──►  Analysis
  (03-01)            (03-02) ◄HERE        (03-03)
                          │
           ┌──────────────┼──────────────────────────────┐
           │              │                              │
           ▼              ▼                              ▼
       Matrix          Least Squares              AI Applications
       Norms           & Pseudoinverse            ─────────────────
       (03-06)         (everywhere)               LoRA / DoRA / MLA
                                                  WeightWatcher
                                                  Recommender Sys.
                                                  Image Compression
                                                  Spectral Norm GAN

════════════════════════════════════════════════════════════════════════
```

**The single most important insight about SVD:** Every matrix is the composition of three simple operations — rotate, scale, rotate. The singular values tell you how much the matrix stretches space in its principal directions; the singular vectors tell you which directions those are. Understanding this decomposition is understanding every linear operation in every neural network, every data compression, and every least-squares problem in machine learning.


---

[← Back to Advanced Linear Algebra](../README.md) | [← Eigenvalues](../01-Eigenvalues-and-Eigenvectors/notes.md) | [PCA →](../03-Principal-Component-Analysis/notes.md)
