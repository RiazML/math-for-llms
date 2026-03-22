[← Back to Advanced Linear Algebra](../README.md) | [← Positive Definite Matrices](../07-Positive-Definite-Matrices/notes.md) | [Next: Calculus Fundamentals →](../../04-Calculus-Fundamentals/README.md)

---

# Matrix Decompositions

> _"The purpose of computing is insight, not numbers — and no insight is more powerful than factoring a matrix into pieces whose structure you understand."_

## Overview

Every practical computation with matrices reduces, ultimately, to a factorization. Solving $A\mathbf{x} = \mathbf{b}$? Factor $A = LU$ and solve two triangular systems. Fitting a linear model? Factor the data matrix $A = QR$ and solve a triangular system stably. Sampling from a multivariate Gaussian? Factor the covariance $\Sigma = LL^\top$ and transform independent normals. Estimating the rank of a nearly singular matrix? Factor with column pivoting and read off the diagonal of $R$.

This section develops the three foundational computational decompositions: **LU factorization** (Gaussian elimination with pivoting, the workhorse for general square systems), **QR factorization via Householder and Givens** (the stable method for least squares and rank-revealing problems), and a **brief computational recap of Cholesky** (the SPD specialist, developed fully in §07). The emphasis throughout is computational: how these algorithms work, why they are numerically stable, what can go wrong, and how to implement them efficiently.

The machine learning connections are direct and load-bearing. Every Newton step requires solving a Hessian system — that is LU or Cholesky. Every least-squares layer (linear probing, ridge regression on features) uses QR under the hood in numerically reliable implementations. Gaussian process inference — the backbone of Bayesian optimization and uncertainty-aware ML — is Cholesky factorization applied thousands of times. Differentiating through factorizations (for implicit differentiation of constrained problems) requires understanding the algebraic structure developed here. And the emerging field of randomized numerical linear algebra, which underlies methods like LoRA and sketch-and-solve preconditioning, is built on randomized LU and QR.

## Prerequisites

- LU, QR, Cholesky brief introductions — [Chapter 2 §02: Matrix Operations](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)
- Eigenvalues, spectral theorem — [§01: Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/notes.md)
- SVD — [§02: Singular Value Decomposition](../02-Singular-Value-Decomposition/notes.md)
- Gram-Schmidt, QR theory, orthogonal matrices — [§05: Orthogonality and Orthonormality](../05-Orthogonality-and-Orthonormality/notes.md)
- Matrix norms, condition numbers — [§06: Matrix Norms](../06-Matrix-Norms/notes.md)
- Positive definite matrices, Cholesky full theory, LDLᵀ — [§07: Positive Definite Matrices](../07-Positive-Definite-Matrices/notes.md)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: LU with pivoting, Householder QR, Givens rotations, RRQR, blocked algorithms, GP inference, randomized methods |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from triangular solves through differentiating Cholesky for GP log-likelihood |

## Learning Objectives

After completing this section, you will be able to:

- Implement forward and backward substitution and analyze their numerical properties
- Derive LU factorization from Gaussian elimination and prove the existence theorem
- Explain why partial pivoting is necessary and implement $PA = LU$ with growth factor analysis
- Distinguish partial, complete, and rook pivoting and state the stability guarantee of each
- Implement Householder QR from scratch, including the sign convention for numerical stability
- Construct Givens rotations and explain when to prefer them over Householder reflectors
- Implement rank-revealing QR with column pivoting and use the R diagonal to estimate rank
- Describe tall-skinny QR (TSQR) and explain its communication-optimal property
- State the backward error theorem and compare the stability of LU, QR, and Cholesky
- Explain blocked algorithms and why BLAS-3 operations dominate on modern hardware
- Apply QR to solve least-squares problems and explain why it is more stable than normal equations
- Differentiate through Cholesky factorization and compute gradients of log-determinants

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 The Factorization Paradigm](#11-the-factorization-paradigm)
  - [1.2 Triangular Systems: The Foundation](#12-triangular-systems-the-foundation)
  - [1.3 Three Canonical Decompositions](#13-three-canonical-decompositions)
  - [1.4 Why This Matters for AI](#14-why-this-matters-for-ai)
  - [1.5 Historical Timeline](#15-historical-timeline)
- [2. Background: Triangular Systems](#2-background-triangular-systems)
  - [2.1 Forward Substitution](#21-forward-substitution)
  - [2.2 Backward Substitution](#22-backward-substitution)
  - [2.3 Gaussian Elimination Revisited](#23-gaussian-elimination-revisited)
- [3. LU Factorization](#3-lu-factorization)
  - [3.1 LU Decomposition Theorem](#31-lu-decomposition-theorem)
  - [3.2 The LU Algorithm: Outer Product Form](#32-the-lu-algorithm-outer-product-form)
  - [3.3 Failure Without Pivoting](#33-failure-without-pivoting)
  - [3.4 Partial Pivoting: PA = LU](#34-partial-pivoting-pa--lu)
  - [3.5 Complete Pivoting: PAQ = LU](#35-complete-pivoting-paq--lu)
  - [3.6 Rook Pivoting](#36-rook-pivoting)
  - [3.7 LU Stability Analysis and Backward Error](#37-lu-stability-analysis-and-backward-error)
  - [3.8 Blocked LU for Cache Efficiency](#38-blocked-lu-for-cache-efficiency)
  - [3.9 Solving Ax = b via LU](#39-solving-ax--b-via-lu)
  - [3.10 Rank-Deficient LU](#310-rank-deficient-lu)
- [4. QR Factorization: Computational Algorithms](#4-qr-factorization-computational-algorithms)
  - [4.1 From Gram-Schmidt to Algorithms](#41-from-gram-schmidt-to-algorithms)
  - [4.2 Householder Reflections](#42-householder-reflections)
  - [4.3 Householder QR Algorithm](#43-householder-qr-algorithm)
  - [4.4 Givens Rotations](#44-givens-rotations)
  - [4.5 Givens QR and Sparse Matrices](#45-givens-qr-and-sparse-matrices)
  - [4.6 Column-Pivoted QR (RRQR)](#46-column-pivoted-qr-rrqr)
  - [4.7 Tall-Skinny QR (TSQR)](#47-tall-skinny-qr-tsqr)
  - [4.8 QR for Least Squares](#48-qr-for-least-squares)
- [5. Cholesky Factorization (Computational Recap)](#5-cholesky-factorization-computational-recap)
  - [5.1 Brief Overview](#51-brief-overview)
  - [5.2 Cholesky as Specialized LU](#52-cholesky-as-specialized-lu)
  - [5.3 LDLᵀ for Indefinite Systems](#53-ldlt-for-indefinite-systems)
  - [5.4 Blocked Cholesky and LAPACK dpotrf](#54-blocked-cholesky-and-lapack-dpotrf)
- [6. Numerical Stability and Error Analysis](#6-numerical-stability-and-error-analysis)
  - [6.1 Forward vs Backward Error](#61-forward-vs-backward-error)
  - [6.2 The Fundamental Theorem of Backward Stability](#62-the-fundamental-theorem-of-backward-stability)
  - [6.3 Stability Comparison: LU vs QR vs Cholesky](#63-stability-comparison-lu-vs-qr-vs-cholesky)
  - [6.4 Mixed Precision and Iterative Refinement](#64-mixed-precision-and-iterative-refinement)
  - [6.5 Ill-Conditioned Systems and Regularization](#65-ill-conditioned-systems-and-regularization)
- [7. Blocked Algorithms and High-Performance Computing](#7-blocked-algorithms-and-high-performance-computing)
  - [7.1 Cache Hierarchy and Algorithm Design](#71-cache-hierarchy-and-algorithm-design)
  - [7.2 Blocked Factorizations](#72-blocked-factorizations)
  - [7.3 LAPACK Routine Reference](#73-lapack-routine-reference)
  - [7.4 Sparse Factorizations](#74-sparse-factorizations)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Least Squares via QR](#81-least-squares-via-qr)
  - [8.2 Newton's Method and Second-Order Optimization](#82-newtons-method-and-second-order-optimization)
  - [8.3 Gaussian Process Inference at Scale](#83-gaussian-process-inference-at-scale)
  - [8.4 Backpropagation Through Factorizations](#84-backpropagation-through-factorizations)
  - [8.5 Randomized Factorizations](#85-randomized-factorizations)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition

### 1.1 The Factorization Paradigm

Every problem in computational linear algebra is, at its heart, a search for structure. When you factor a matrix into simpler pieces — triangular, orthogonal, diagonal — you transform a hard problem into a sequence of easy ones. The word "easy" is precise: triangular systems can be solved in $O(n^2)$ operations by substitution; orthogonal transformations are numerically perfect (they preserve norms exactly); diagonal systems are trivially solvable. A decomposition is a way of peeling back complexity to reveal the structural skeleton underneath.

This perspective explains why factorizations are not merely computational conveniences. They are mathematical revelations. LU factorization reveals which rows are "heavy" (large pivots dominate) and which are "light" (small pivots signal near-singularity). QR factorization reveals the orthogonal complement of the column space via the zero rows of $R$. Cholesky factorization reveals the "matrix square root" $L$ that encodes the geometry of a positive definite quadratic form. Each decomposition is a lens tuned to a different structural feature.

**The central insight:** nearly every matrix algorithm of practical importance runs in $O(n^3)$ time, dominated by the factorization phase. After factorization, the actual problem-solving (applying $L^{-1}$, $U^{-1}$, $Q^\top$) costs only $O(n^2)$. This is the fundamental return on investment: spend $O(n^3)$ once to factor, then spend $O(n^2)$ as many times as needed for different right-hand sides.

**For AI:** this principle appears in natural gradient methods (factor the Fisher information matrix once per outer loop, solve multiple gradient updates cheaply), in Gaussian process regression (factor the covariance matrix once, evaluate posterior at many test points cheaply), and in iterative solvers (factor a preconditioner once, apply it repeatedly to accelerate convergence).

### 1.2 Triangular Systems: The Foundation

Before understanding factorizations, you must understand why triangular systems are considered "solved." A lower triangular system $L\mathbf{x} = \mathbf{b}$ with $L_{ii} \neq 0$ can be solved row by row from top to bottom:

$$x_1 = b_1 / L_{11}$$
$$x_2 = (b_2 - L_{21}x_1) / L_{22}$$
$$x_i = \left(b_i - \sum_{j=1}^{i-1} L_{ij} x_j\right) / L_{ii}$$

This is **forward substitution**: each unknown is determined uniquely by those already computed. The computation requires exactly $n^2/2$ multiplications and $n^2/2$ additions — $O(n^2)$ total. Similarly, an upper triangular system $U\mathbf{x} = \mathbf{b}$ is solved by **backward substitution** from bottom to top.

The key properties of triangular systems:
1. **Existence and uniqueness:** a triangular system has a unique solution if and only if all diagonal entries are nonzero.
2. **Numerical stability:** forward/backward substitution are inherently stable when pivots are large; small pivots amplify errors.
3. **Parallelism:** the sequential dependency (each unknown requires all previous) limits parallelism, but blocked implementations can exploit level-2 BLAS parallelism within panels.
4. **Structural exploitation:** sparse triangular systems (band, arrowhead, bidiagonal) can be solved in $O(nk)$ for bandwidth $k$.

**Non-example:** a general dense $n \times n$ system $A\mathbf{x} = \mathbf{b}$ cannot be solved by substitution directly — it requires $O(n^3)$ work. The entire purpose of LU, QR, and Cholesky is to reduce the general case to triangular cases.

### 1.3 Three Canonical Decompositions

The three decompositions in this section partition the space of structured matrix problems:

| Decomposition | Matrix class | Factored form | Primary use |
|---|---|---|---|
| LU | General square, non-singular | $PA = LU$ | Solving $A\mathbf{x} = \mathbf{b}$, computing $\det(A)$ |
| QR | Any rectangular $m \times n$, $m \geq n$ | $A = QR$ | Least squares, rank determination, eigenvalue algorithms |
| Cholesky | Symmetric positive definite | $A = LL^\top$ | SPD systems, Gaussian sampling, log-det computation |

**Why three?** Each exploits a different structural property. LU uses no special structure beyond invertibility; it is the most general and therefore requires more care (pivoting for stability). QR uses orthogonality, which is the most numerically pristine structure — $Q$ has condition number 1 by definition. Cholesky uses both symmetry and positive definiteness, which allows working with only half the matrix and guarantees the factorization exists without pivoting.

**The hierarchy of stability:** Cholesky $>$ QR $>$ LU (partially pivoted) $>$ LU (no pivoting). As you sacrifice structure, you need more algorithmic sophistication to maintain numerical reliability.

### 1.4 Why This Matters for AI

Matrix factorizations are not background infrastructure — they are in the critical path of the most computationally expensive operations in modern AI.

**Gaussian process regression** (used in Bayesian optimization, uncertainty quantification, and hyperparameter tuning) requires computing $K^{-1}\mathbf{y}$ and $\log\det K$ for an $n \times n$ kernel matrix $K$. Both require Cholesky factorization. A single GP training step on $n = 10{,}000$ points costs $O(n^3) = 10^{12}$ operations — factorization is the bottleneck.

**Second-order optimization** (Newton's method, natural gradient, K-FAC) requires solving Hessian or Fisher information systems $H\boldsymbol{\delta} = \mathbf{g}$. For K-FAC (Kronecker-Factored Approximate Curvature, Martens & Grosse 2015), the Fisher approximation is block-diagonal with Kronecker structure, and each block is inverted via Cholesky — thousands of small Cholesky factorizations per training step.

**Linear probing and fine-tuning** — fitting a linear model on top of frozen features — is a least-squares problem. Numerically reliable implementations use QR (e.g., `numpy.linalg.lstsq` calls LAPACK's `dgelsd` which uses divide-and-conquer SVD, or `dgelsy` which uses column-pivoted QR).

**Differentiating through factorizations** is required whenever a factorization appears inside a differentiable computational graph. PyTorch's `torch.linalg.cholesky_solve` and JAX's `jax.lax.linalg.cholesky` support automatic differentiation through the factorization via implicit function theorem.

**Randomized methods** (LoRA, sketching, randomized preconditioning) use approximate LU and QR factorizations on sketched matrices. Understanding the exact factorizations is prerequisite to understanding their randomized variants.

### 1.5 Historical Timeline

| Year | Contribution | Author(s) |
|---|---|---|
| 1809 | Gaussian elimination for geodesy (Gauss not published until 1810) | Carl Friedrich Gauss |
| 1938 | LU factorization formalized as matrix decomposition | Tadeusz Banachiewicz |
| 1945 | Systematic analysis of elimination and error growth | Alan Turing |
| 1954 | Givens rotations for QR | Wallace Givens |
| 1958 | Householder reflections; modern QR algorithm | Alston Householder |
| 1961 | QR algorithm for eigenvalues (combined QR iteration) | Francis; Kublanovskaya |
| 1961 | LAPACK precursor: ALGOL programs for matrix problems | Wilkinson, Reinsch |
| 1979 | LINPACK (first standardized library) | Dongarra et al. |
| 1987 | LAPACK + BLAS-3 blocked algorithms | Anderson, Bai, Dongarra et al. |
| 1995 | Randomised SVD (sketch-then-factor) | Frieze, Kannan, Vempala |
| 2011 | Halko-Martinsson-Tropp randomized low-rank framework | Halko, Martinsson, Tropp |
| 2015 | K-FAC: Cholesky for Fisher information blocks | Martens & Grosse |
| 2022 | LoRA: low-rank LU-inspired weight decomposition | Hu et al. |
| 2024 | FlashAttention-3: blocked QR-like tiling for attention | Shah et al. |

---

## 2. Background: Triangular Systems

### 2.1 Forward Substitution

**Definition.** Given a lower triangular matrix $L \in \mathbb{R}^{n \times n}$ with $L_{ii} \neq 0$ for all $i$, and $\mathbf{b} \in \mathbb{R}^n$, **forward substitution** computes $\mathbf{x}$ satisfying $L\mathbf{x} = \mathbf{b}$ via:

$$x_i = \frac{1}{L_{ii}}\left(b_i - \sum_{j=1}^{i-1} L_{ij} x_j\right), \quad i = 1, 2, \ldots, n$$

**Algorithm (row-oriented):**
```
for i = 1 to n:
    x[i] = b[i]
    for j = 1 to i-1:
        x[i] -= L[i,j] * x[j]
    x[i] /= L[i,i]
```

**Complexity:** The inner loop runs $0, 1, \ldots, n-1$ times, giving $\sum_{i=1}^{n}(i-1) = n(n-1)/2$ multiplications and the same number of additions. Total: $n^2$ floating-point operations, or $O(n^2)$.

**Numerical properties:** Forward substitution is unconditionally stable in exact arithmetic. In floating-point arithmetic, errors accumulate proportionally to $\kappa(L)$ (the condition number of $L$), not to the intermediate values of $x_j$. The key source of numerical difficulty is a small diagonal entry $L_{ii}$: dividing by a near-zero diagonal amplifies errors quadratically.

**Unit lower triangular:** When $L_{ii} = 1$ for all $i$ (as in the LU factorization where $L$ is unit lower triangular by convention), the divisions are trivial and the algorithm reduces to:
```
for i = 1 to n:
    x[i] = b[i] - sum(L[i,1:i-1] * x[1:i-1])
```
This eliminates $n$ divisions, slightly improving performance and stability.

**For AI:** Forward substitution appears in the final step of any matrix factorization-based solver. In neural network training with second-order methods (Newton, K-FAC), solving $L\mathbf{x} = \mathbf{g}$ extracts the natural gradient direction $\mathbf{g} = F^{-1}\nabla\mathcal{L}$ given the Cholesky factor $L$ of the Fisher matrix $F$.

### 2.2 Backward Substitution

**Definition.** Given an upper triangular $U \in \mathbb{R}^{n \times n}$ with $U_{ii} \neq 0$, **backward substitution** computes $\mathbf{x}$ satisfying $U\mathbf{x} = \mathbf{b}$ via:

$$x_i = \frac{1}{U_{ii}}\left(b_i - \sum_{j=i+1}^{n} U_{ij} x_j\right), \quad i = n, n-1, \ldots, 1$$

The algorithm proceeds from the last equation (which has a single unknown $x_n = b_n / U_{nn}$) backward to the first. Complexity and numerical properties mirror forward substitution.

**Column-oriented backward substitution:** An equivalent formulation updates the remaining components of $\mathbf{b}$ as each $x_i$ is resolved:
```
for i = n downto 1:
    x[i] = b[i] / U[i,i]
    b[1:i-1] -= x[i] * U[1:i-1, i]
```
This accesses $U$ by columns rather than rows, which can improve cache efficiency on column-major storage (Fortran, LAPACK convention).

**Connection to matrix inverse:** The columns of $U^{-1}$ can be computed by solving $n$ triangular systems $U \mathbf{x}^{(k)} = \mathbf{e}_k$, one per standard basis vector. This is how `scipy.linalg.solve_triangular` computes $U^{-1}$ when called with a matrix right-hand side.

### 2.3 Gaussian Elimination Revisited

Gaussian elimination is the process of reducing a matrix $A$ to upper triangular form by applying elementary row operations (EROs). Each ERO of type "add $c$ times row $j$ to row $i$ (with $i > j$)" corresponds to left-multiplication by an elementary lower triangular matrix (an **elementary elimination matrix** or **Frobenius matrix**):

$$E_{ij} = I - m_{ij}\mathbf{e}_i\mathbf{e}_j^\top, \quad m_{ij} = A_{ij}/A_{jj}$$

The **multiplier** $m_{ij}$ is the ratio that zeros out entry $(i,j)$. Applying all elimination matrices to reduce $A$ to $U$ gives:

$$E_{n,n-1} \cdots E_{31} E_{21} A = U$$

Crucially, the product of elementary lower triangular matrices is lower triangular, and their inverses are trivially:

$$E_{ij}^{-1} = I + m_{ij}\mathbf{e}_i\mathbf{e}_j^\top$$

So the product of inverses, $L = E_{21}^{-1} E_{31}^{-1} \cdots E_{n,n-1}^{-1}$, is unit lower triangular with the multipliers $m_{ij}$ in their natural positions. This gives:

$$A = LU$$

**The multipliers go directly into $L$:** one of the most elegant facts in numerical linear algebra is that the multipliers $m_{ij}$ used during elimination appear, without any further computation, as the subdiagonal entries $L_{ij}$ of the factor $L$. No separate computation of $L$ is required — it is built up in-place during elimination.

---

## 3. LU Factorization

### 3.1 LU Decomposition Theorem

**Theorem (LU factorization).** Let $A \in \mathbb{R}^{n \times n}$. Then $A$ has an LU factorization $A = LU$ (with $L$ unit lower triangular and $U$ upper triangular) **if and only if** all leading principal submatrices $A_k = A[1:k, 1:k]$ for $k = 1, \ldots, n-1$ are non-singular.

*Proof sketch.* The condition $A_k$ non-singular for all $k$ ensures that no zero pivot is encountered during Gaussian elimination. At step $k$, the pivot is $U_{kk} = \det(A_k)/\det(A_{k-1})$, so all pivots are nonzero iff all leading minors are nonzero. Conversely, if Gaussian elimination completes without zero pivots, the multipliers are finite and $L, U$ are well-defined. $\square$

**Non-examples:**
- $A = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$ fails: $A_{11} = 0$ (a zero pivot at step 1).
- $A = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$: $A_{11} = 1 \neq 0$ but $\det(A) = 0$, so the factorization runs but $U_{22} = 0$ (singular).
- Any singular matrix: $A = LU \Rightarrow \det(A) = \det(L)\det(U) = \prod_i U_{ii}$, so $\det(A) = 0$ iff some $U_{ii} = 0$.

**Uniqueness:** The factorization with $L_{ii} = 1$ (unit lower triangular) is unique when it exists. Without the normalization, one could scale $L$ and $U$ by any diagonal matrix and its inverse.

**The factorization in-place:** LU computation overwrites $A$: the upper triangle stores $U$, the strict lower triangle stores the multipliers (i.e., the strict lower triangle of $L$, since $L_{ii} = 1$ need not be stored). This halves the memory requirement.

### 3.2 The LU Algorithm: Outer Product Form

The standard view of LU elimination is column-by-column. But the **outer product form** (also called **rank-1 update form**) exposes the structure more clearly and maps naturally to BLAS-3 blocked implementations.

At step $k$ of the factorization, we have already reduced columns $1, \ldots, k-1$. The remaining $(n-k+1) \times (n-k+1)$ submatrix $A^{(k)}$ (the **Schur complement** after $k-1$ steps) has the form:

$$A^{(k+1)} = A^{(k)}[k+1:n, k+1:n] - \mathbf{l}^{(k)} \mathbf{u}^{(k)\top}$$

where $\mathbf{l}^{(k)} = A^{(k)}[k+1:n, k] / A^{(k)}_{kk}$ (the multipliers column) and $\mathbf{u}^{(k)\top} = A^{(k)}[k, k+1:n]$ (the pivot row). This is a **rank-1 update** of the trailing submatrix.

**Algorithm:**
```
for k = 1 to n-1:
    # Compute multipliers
    L[k+1:n, k] = A[k+1:n, k] / A[k, k]    # size (n-k)
    # Rank-1 update of trailing submatrix
    A[k+1:n, k+1:n] -= L[k+1:n, k] * A[k, k+1:n]   # BLAS-2: ger
    U[k, k:n] = A[k, k:n]
```

**Complexity:** Step $k$ costs $(n-k)$ divisions and $(n-k)^2$ multiply-adds. Total:
$$\sum_{k=1}^{n-1} [(n-k) + (n-k)^2] = \frac{n(n-1)}{2} + \frac{(n-1)n(2n-1)}{6} \approx \frac{2n^3}{3}$$

So LU factorization costs $\frac{2n^3}{3}$ floating-point operations (flops). For comparison: solving $A\mathbf{x} = \mathbf{b}$ after factorization costs $2n^2$ (two triangular solves). The ratio $n/3$ means factorization dominates for large $n$.

**For AI:** the rank-1 update structure maps directly to GPU tensor cores. Blocked implementations reformulate the trailing submatrix update as a matrix-matrix multiply (BLAS-3: DGEMM), which achieves near-peak GPU throughput.

### 3.3 Failure Without Pivoting

Naive LU (no pivoting) fails catastrophically when:

1. **Zero pivots:** If $A_{kk}^{(k)} = 0$ during step $k$, the division by the pivot is undefined. This happens even for non-singular matrices that happen to have zero leading principal minors.

2. **Small pivots — numerical catastrophe:** Consider:
$$A = \begin{pmatrix} \varepsilon & 1 \\ 1 & 1 \end{pmatrix}, \quad \varepsilon = 10^{-16}$$
Naive LU gives multiplier $m_{21} = 1/\varepsilon = 10^{16}$, and $U_{22} = 1 - 10^{16} \cdot 1 \approx -10^{16}$ (in floating-point, this overwhelms the exact value $1 - 1/\varepsilon = ({\varepsilon - 1})/\varepsilon$). The computed solution $\hat{\mathbf{x}}$ can have relative error of order 1 even when the exact solution is well-conditioned.

3. **Growth factor explosion:** The **growth factor** $\rho_n$ is defined as:
$$\rho_n = \frac{\max_{i,j,k} |A_{ij}^{(k)}|}{\max_{i,j} |A_{ij}|}$$
Without pivoting, $\rho_n$ can grow as $2^{n-1}$ in the worst case (Wilkinson's matrix), making the computed $U$ useless.

**Example of catastrophic failure:**
$$A = \begin{pmatrix} 10^{-20} & 1 \\ 1 & 2 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 1 \\ 3 \end{pmatrix}$$
Exact solution: $\mathbf{x} = (1, 1)^\top$ approximately. Naive LU in IEEE double precision gives multiplier $m_{21} = 10^{20}$, yielding $U_{22} = 2 - 10^{20} \approx -10^{20}$ — a catastrophic cancellation.

**For AI:** PyTorch's `torch.linalg.lu` uses partial pivoting by default. `numpy.linalg.solve` calls LAPACK `dgesv` which uses partial pivoting. Never use unpivoted LU for numerical computation.

### 3.4 Partial Pivoting: PA = LU

**Idea:** Before processing column $k$, swap row $k$ with the row (among $k, k+1, \ldots, n$) having the **largest absolute value** in column $k$. This ensures $|m_{ij}| \leq 1$ for all multipliers.

**Algorithm with partial pivoting:**
```
for k = 1 to n-1:
    # Find pivot row
    p = argmax_{i >= k} |A[i, k]|
    swap rows k and p in A (and record in permutation P)
    # Now A[k,k] is the largest in column k below
    L[k+1:n, k] = A[k+1:n, k] / A[k, k]       # |L[i,k]| <= 1
    A[k+1:n, k+1:n] -= L[k+1:n, k] * A[k, k+1:n]
```

**Result:** $PA = LU$ where $P$ is a permutation matrix encoding all row swaps. The factor $L$ satisfies $|L_{ij}| \leq 1$ for all $i > j$.

**Growth factor with partial pivoting:** The theoretical bound is $\rho_n \leq 2^{n-1}$ (same as no pivoting!), but in practice the growth factor is almost always small ($\rho_n \approx n^{2/3}$ for random matrices). Foster (1997) proved that adversarial inputs achieving $\rho_n = 2^{n-1}$ are exponentially rare.

**Stability theorem (Wilkinson 1961):** Partial pivoting produces a computed factorization $\hat{L}\hat{U}$ such that:
$$PA = \hat{L}\hat{U} + E, \quad \|E\|_\infty \leq 8n^3 u \cdot \rho_n \cdot \|A\|_\infty$$
where $u$ is the unit roundoff (e.g., $u = 2^{-53}$ for IEEE double). For typical $\rho_n$, this bound is well within practical tolerance.

**Storage of the permutation:** $P$ is stored as a pivot vector $\mathbf{p} \in \mathbb{Z}^{n-1}$ where $p_k$ is the row index selected at step $k$. Applying $P$ to $\mathbf{b}$ before the triangular solve costs $O(n)$.

**Non-example:** Partial pivoting does NOT guarantee $\rho_n$ is small. The $2 \times 2$ example:
$$A = \begin{pmatrix} 1 & -1 \\ 1 & 1 \end{pmatrix}$$
has pivot column $\begin{pmatrix} 1 \\ 1 \end{pmatrix}$, so either row could be chosen. After step 1, $U_{22} = 2$, $\rho_2 = 1$. But for $A = \begin{pmatrix} 1 & 1 \\ 1 & 1+\varepsilon \end{pmatrix}$, $\rho_2 = (2+\varepsilon)/1 \approx 2$.

### 3.5 Complete Pivoting: PAQ = LU

**Complete pivoting** selects both the **row** and **column** with the globally largest absolute entry at each step:

At step $k$: find $(p, q) = \arg\max_{i \geq k, j \geq k} |A_{ij}|$, then swap row $k \leftrightarrow p$ and column $k \leftrightarrow q$.

**Result:** $PAQ = LU$ where $P$ and $Q$ are permutation matrices (row and column respectively).

**Growth factor bound:** Complete pivoting satisfies the tighter bound:
$$\rho_n \leq (n \cdot 2^1 \cdot 3^{1/2} \cdot 4^{1/3} \cdots n^{1/(n-1)})^{1/2}$$
which grows much more slowly than $2^{n-1}$ but is still superlinear. Numerically, complete pivoting is extremely stable — no practical example of large growth is known.

**Cost:** Finding the global maximum at each step requires examining the full trailing submatrix, adding $O(n^2)$ comparisons per step and $O(n^3)$ total — which triples the leading constant relative to partial pivoting. This is the reason complete pivoting is rarely used in practice.

**When to use complete pivoting:**
- When stability is paramount (e.g., certified computation, interval arithmetic)
- When the matrix is suspected to be rank-deficient (complete pivoting exposes rank)
- When solving very ill-conditioned systems where partial pivoting is known to fail

**For AI:** Complete pivoting is rarely used in mainstream ML workflows. Partial pivoting suffices for almost all practical problems. The theoretical importance of complete pivoting is in establishing lower bounds and in the analysis of rank-revealing factorizations.

### 3.6 Rook Pivoting

**Rook pivoting** is a middle ground between partial and complete pivoting. At step $k$:

1. Find the largest element in column $k$ (as in partial pivoting) — say row $p$.
2. Check if $|A_{pk}|$ is also the largest in row $p$. If yes, use $(p, k)$ as pivot.
3. If not, find the largest in row $p$, swap columns, repeat until convergence.

The name comes from the chess rook: the pivot search alternates between column and row moves until it finds a position that is simultaneously the largest in its row and column — a "rook position."

**Properties:**
- **Growth factor:** $\rho_n \leq 2^{n-1}$ but in practice much smaller than partial pivoting.
- **Cost:** $O(n^2)$ additional work per step in the worst case but typically $O(n)$ in practice (convergence in 1–2 alternations).
- **Rank-revealing:** Rook pivoting reveals rank approximately as well as complete pivoting.

**Theorem (Foster 1997):** For a matrix of rank $r$, rook pivoting ensures $|R_{11}| / |R_{kk}| \leq \sqrt{n}$ for $k > r$, making the diagonal jump at position $r+1$ clearly visible.

### 3.7 LU Stability Analysis and Backward Error

The definitive framework for understanding LU stability is **backward error analysis**, developed by Wilkinson in the 1960s.

**Definition (backward error).** The backward error of a computed solution $\hat{\mathbf{x}}$ to $A\mathbf{x} = \mathbf{b}$ is the smallest perturbation $\delta A, \delta\mathbf{b}$ such that:
$$(A + \delta A)\hat{\mathbf{x}} = \mathbf{b} + \delta\mathbf{b}$$
A method is **backward stable** if $\|\delta A\| / \|A\|$ and $\|\delta\mathbf{b}\| / \|\mathbf{b}\|$ are of order $u$ (machine epsilon) regardless of the input.

**Backward error theorem for LU with partial pivoting:**

**Theorem (Wilkinson 1961).** Let $\hat{\mathbf{x}}$ be the solution computed by LU with partial pivoting in IEEE double precision. Then:
$$(A + \delta A)\hat{\mathbf{x}} = \mathbf{b}, \quad \|\delta A\|_\infty \leq c_n u \cdot \rho_n \cdot \|A\|_\infty$$
where $c_n = O(n^2)$ is a modest polynomial constant and $u = 2^{-53} \approx 1.1 \times 10^{-16}$.

**Forward error bound:** The forward error satisfies:
$$\frac{\|\mathbf{x} - \hat{\mathbf{x}}\|_\infty}{\|\mathbf{x}\|_\infty} \leq c_n u \cdot \rho_n \cdot \kappa_\infty(A) + O(u^2)$$
where $\kappa_\infty(A) = \|A\|_\infty \|A^{-1}\|_\infty$ is the condition number. This is the key identity: **numerical error = roundoff $\times$ growth factor $\times$ condition number**.

**Implication for practice:** If $\kappa(A) \approx 10^k$ in double precision, you lose $k$ digits of accuracy. With $u \approx 10^{-16}$, you have $16 - k$ correct digits in $\hat{\mathbf{x}}$.

**Backward stability of QR:** QR (Householder) is backward stable with growth factor $\rho_n = 1$ — Householder reflectors are orthogonal and preserve norms exactly. This is why QR is preferred for ill-conditioned systems.

### 3.8 Blocked LU for Cache Efficiency

Modern hardware has a deep memory hierarchy (L1: 32KB, L2: 256KB, L3: 8MB, DRAM: ∞). The key metric for algorithmic performance is **arithmetic intensity**: floating-point operations per byte of data moved. BLAS operations have different intensities:

| Operation | BLAS level | Flops | Data | Intensity |
|---|---|---|---|---|
| $\mathbf{y} \leftarrow A\mathbf{x}$ (DGEMV) | 2 | $2n^2$ | $n^2$ | $O(1)$ |
| $C \leftarrow AB + C$ (DGEMM) | 3 | $2n^3$ | $n^2$ | $O(n)$ |

**Blocked LU** reformulates the algorithm so that the dominant computation is DGEMM (matrix-matrix multiply), achieving $O(n)$ arithmetic intensity and near-peak GPU/CPU throughput.

**Block algorithm** (block size $b$):
```
for k = 0 to n/b - 1:
    A_kk = A[kb:(k+1)b, kb:(k+1)b]
    # Factor the diagonal panel (unblocked, uses BLAS-2)
    P_k, L_kk, U_kk = LU_factor(A_kk)    # O(b^3)
    # Apply permutation to remaining columns
    A[kb:(k+1)b, (k+1)b:n] = P_k @ A[kb:(k+1)b, (k+1)b:n]
    # Solve for U panel (triangular solve, BLAS-3: DTRSM)
    U_k_right = solve_lower_triangular(L_kk, A[kb:(k+1)b, (k+1)b:n])
    # Update trailing submatrix (BLAS-3: DGEMM)
    A[(k+1)b:n, (k+1)b:n] -= L_right @ U_k_right     # This is the bottleneck
```

The trailing submatrix update (`A -= L_right @ U_k_right`) is a pure DGEMM and dominates the total computation. On a modern CPU/GPU with peak DGEMM performance, this block formulation achieves 80-95% of theoretical peak.

**LAPACK routine:** `DGETRF` implements blocked LU with partial pivoting. The block size $b$ is tuned automatically per platform via the `ilaenv` oracle.

**For AI:** Blocked LU is the algorithm behind PyTorch's `torch.linalg.lu_factor`, `jax.scipy.linalg.lu`, and scipy's `lu_factor`. For GPU, cuSOLVER implements batched blocked LU for thousands of small systems simultaneously — critical for K-FAC training.

### 3.9 Solving Ax = b via LU

Given the LU factorization $PA = LU$, solving $A\mathbf{x} = \mathbf{b}$ proceeds in four steps:

1. **Apply permutation:** $\mathbf{c} = P\mathbf{b}$ (O(n), just index lookup)
2. **Forward solve:** solve $L\mathbf{y} = \mathbf{c}$ (O(n²), forward substitution)
3. **Backward solve:** solve $U\mathbf{x} = \mathbf{y}$ (O(n²), backward substitution)
4. **(Optional) Iterative refinement:** compute $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$, solve for correction $\delta\mathbf{x}$, update

The total cost after factorization is $2n^2$ flops — linear in $n^2$, negligible compared to the $\frac{2n^3}{3}$ factorization cost for large $n$.

**Multiple right-hand sides:** For $k$ different vectors $\mathbf{b}_1, \ldots, \mathbf{b}_k$, one factorization costing $\frac{2n^3}{3}$ is followed by $k$ triangular solves each costing $2n^2$. This is the key economic argument for LU: amortize the factorization cost over many solves.

**Computing the determinant:** $\det(A) = \det(P^{-1})\det(L)\det(U) = (-1)^s \prod_{i=1}^n U_{ii}$, where $s$ is the number of row swaps. Since $L$ is unit lower triangular, $\det(L) = 1$.

**Computing $A^{-1}$:** Solve $A\mathbf{x}^{(k)} = \mathbf{e}_k$ for $k = 1, \ldots, n$ (one factorization, $n$ triangular solves). But explicitly forming $A^{-1}$ is almost always unnecessary and should be avoided — solve the system directly instead.

### 3.10 Rank-Deficient LU

When $A$ is singular or numerically rank-deficient, the LU factorization (even with pivoting) produces a $U$ with some diagonal entries near zero. The factorization still "works" mechanically but produces $U_{kk} \approx 0$ at position $k$ corresponding to a dependent column.

**Signature of rank deficiency:**
$$\operatorname{rank}(A) = r \iff U_{11}, \ldots, U_{rr} \text{ are "large" and } U_{r+1,r+1}, \ldots, U_{nn} \approx 0$$

**Problem:** With partial pivoting, the threshold for "approximately zero" is ambiguous. If $|U_{kk}| / |U_{11}| < \varepsilon_{\text{tol}}$, we declare the matrix has numerical rank $< k$. But the choice of $\varepsilon_{\text{tol}}$ is problem-dependent.

**Rank-revealing LU:** Column-pivoted LU (complete or rook pivoting) provides better rank-revelation: the diagonal entries of $U$ decay in a manner that reflects the true rank structure.

**Better alternative for rank-deficient problems:** Use **rank-revealing QR** (RRQR, §4.6) or **SVD**. The SVD provides the definitive rank determination via singular value thresholding — but costs $O(n^3)$ just as LU does. RRQR provides a cheaper, nearly-as-reliable alternative.

**For AI:** Neural network weight matrices often have approximate low rank (Hu et al., 2022 show that fine-tuning adapters are intrinsically low-dimensional). Rank-deficient LU can be used to detect this structure, but RRQR or randomized SVD are preferred in practice.

---

## 4. QR Factorization: Computational Algorithms

### 4.1 From Gram-Schmidt to Algorithms

**Recall from §05:** The QR factorization $A = QR$ of an $m \times n$ matrix ($m \geq n$) decomposes $A$ into an orthonormal factor $Q \in \mathbb{R}^{m \times n}$ (or $m \times m$ for the full QR) and an upper triangular factor $R \in \mathbb{R}^{n \times n}$. This was constructed via Gram-Schmidt orthogonalization in §05.

**The problem with classical Gram-Schmidt:** Classical Gram-Schmidt (CGS) is mathematically correct but numerically unstable for ill-conditioned matrices. The computed $Q$ can lose orthogonality catastrophically: for a matrix with condition number $10^8$, the computed $Q$ from CGS may have $\|Q^\top Q - I\|_F \approx 10^8 \cdot u \approx 10^{-8}$, causing significant errors in subsequent computations.

**Modified Gram-Schmidt (MGS):** Reorders the orthogonalization to reduce error accumulation. MGS is numerically equivalent to classical QR of a slightly perturbed matrix, giving $\|Q^\top Q - I\|_F = O(\kappa(A) \cdot u)$. Still insufficient for highly ill-conditioned problems.

**Householder and Givens:** Both achieve $\|Q^\top Q - I\|_F = O(u)$ regardless of $\kappa(A)$, making them the preferred algorithms for production use. The key is that they build $Q$ as a product of exactly orthogonal elementary transformations (reflectors or rotations), never losing orthogonality during construction.

**For AI:** `numpy.linalg.qr` and `scipy.linalg.qr` use LAPACK `DGEQRF` (Householder QR). `torch.linalg.qr` similarly calls cuSOLVER which uses Householder on GPU. Understanding the algorithmic basis helps you interpret condition number warnings and numerical precision behavior.

### 4.2 Householder Reflections

**Definition.** A **Householder reflector** (or **Householder transformation**) is a matrix of the form:

$$H = I - 2\frac{\mathbf{v}\mathbf{v}^\top}{\mathbf{v}^\top\mathbf{v}} = I - \frac{2}{\|\mathbf{v}\|_2^2}\mathbf{v}\mathbf{v}^\top$$

where $\mathbf{v} \in \mathbb{R}^m$ is a nonzero **Householder vector**. The matrix $H$ is symmetric ($H = H^\top$) and orthogonal ($H^\top H = H^2 = I$), so $H^{-1} = H$ — it is its own inverse.

**Geometric interpretation:** $H$ reflects vectors across the hyperplane perpendicular to $\mathbf{v}$. Every point $\mathbf{x}$ in the hyperplane satisfies $H\mathbf{x} = \mathbf{x}$ (invariant). Every point $c\mathbf{v}$ along the $\mathbf{v}$ direction satisfies $Hc\mathbf{v} = -c\mathbf{v}$ (negated).

**Key property:** Given any vector $\mathbf{a} \in \mathbb{R}^m$ and any unit vector $\hat{\mathbf{e}} \in \mathbb{R}^m$, there exists a Householder reflector $H$ such that $H\mathbf{a} = -\operatorname{sign}(a_1)\|\mathbf{a}\|_2 \hat{\mathbf{e}}_1$. That is, $H$ maps $\mathbf{a}$ to a multiple of the first standard basis vector.

**Construction (numerical form):** Given $\mathbf{a} \in \mathbb{R}^m$ to be zeroed below its first entry:
$$\alpha = -\operatorname{sign}(a_1)\|\mathbf{a}\|_2$$
$$\mathbf{v} = \mathbf{a} - \alpha\mathbf{e}_1 = \begin{pmatrix} a_1 - \alpha \\ a_2 \\ \vdots \\ a_m \end{pmatrix}$$
$$H = I - \frac{2}{\|\mathbf{v}\|_2^2}\mathbf{v}\mathbf{v}^\top, \quad H\mathbf{a} = \alpha\mathbf{e}_1$$

**The sign convention (critical for stability):** We choose $\alpha = -\operatorname{sign}(a_1)\|\mathbf{a}\|_2$ to **avoid cancellation** in computing $v_1 = a_1 - \alpha$. If $a_1 > 0$, choosing $\alpha = -\|\mathbf{a}\|_2$ gives $v_1 = a_1 + \|\mathbf{a}\|_2 > a_1$, avoiding the catastrophic subtraction that would occur with $\alpha = +\|\mathbf{a}\|_2$.

**Cost of applying $H$:** Computing $H\mathbf{x}$ directly costs $O(m^2)$ (matrix-vector product), but using the formula $H\mathbf{x} = \mathbf{x} - 2\mathbf{v}(\mathbf{v}^\top\mathbf{x})/(\mathbf{v}^\top\mathbf{v})$ costs only $O(m)$ — first compute the scalar $s = \mathbf{v}^\top\mathbf{x}$, then update $\mathbf{x} \leftarrow \mathbf{x} - (2s/\|\mathbf{v}\|^2)\mathbf{v}$. This is the **implicit representation** of $H$.

### 4.3 Householder QR Algorithm

**Algorithm:** Apply Householder reflectors successively to zero out subdiagonal entries of each column.

```
for k = 1 to n:                                # n columns
    # Construct H_k to zero out A[k+1:m, k]
    a = A[k:m, k]                              # current column
    alpha = -sign(a[0]) * norm(a)
    v = a.copy(); v[0] -= alpha
    v /= norm(v)                               # normalized Householder vector
    # Apply H_k implicitly to A[k:m, k:n]
    A[k:m, k:n] -= 2 * v * (v.T @ A[k:m, k:n])
    # Store v in lower triangle of A (LAPACK convention)
    A[k+1:m, k] = v[1:]
```

After $n$ steps, $A$ has been overwritten with $R$ in the upper triangle and Householder vectors in the lower triangle (the implicit storage of $Q$).

**Implicit Q representation:** The full $Q$ matrix is $Q = H_1 H_2 \cdots H_n$ and has size $m \times m$. Forming $Q$ explicitly costs $O(mn^2)$ additional work and $O(mn)$ storage. LAPACK's `DORGQR` computes the explicit $Q$ when needed (e.g., for the thin QR, only the first $n$ columns of $Q$ are needed).

**Complexity:**
- **Applying $H_k$ to trailing submatrix:** $O((m-k)(n-k))$ per step
- **Total:** $\sum_{k=1}^{n} O((m-k)(n-k)) = O(mn^2 - n^3/3) \approx 2mn^2 - 2n^3/3$
- For square $m = n$: $4n^3/3$ flops (vs $2n^3/3$ for LU)

**Numerical stability:** Householder QR is backward stable. The computed $\hat{Q}, \hat{R}$ satisfy:
$$A + E = \hat{Q}\hat{R}, \quad \|E\|_F \leq c_{mn} u \|A\|_F$$
with $c_{mn}$ a modest polynomial constant and $u$ machine epsilon. The orthogonality error satisfies $\|\hat{Q}^\top\hat{Q} - I\|_F = O(u)$, independent of $\kappa(A)$.

**LAPACK:** `DGEQRF` computes the Householder QR in blocked form (block Householder, using WY representation). `DORMQR` applies $Q$ or $Q^\top$ to a matrix without forming $Q$ explicitly.

**For AI:** The Householder algorithm is used in PyTorch's `torch.linalg.qr`, in scipy's `linalg.qr`, and in LAPACK `DGEQRF`. It is the foundation for numerically reliable QR-based least-squares solvers used in linear probing, weight matrix analysis, and second-order methods.

### 4.4 Givens Rotations

**Definition.** A **Givens rotation** $G(i,j,\theta) \in \mathbb{R}^{n \times n}$ is the identity matrix with a $2 \times 2$ rotation embedded in the $(i,j)$ rows and columns:

$$G(i,j,\theta) = \begin{pmatrix} 1 & \cdots & 0 & \cdots & 0 & \cdots & 0 \\ \vdots & \ddots & \vdots & & \vdots & & \vdots \\ 0 & \cdots & c & \cdots & s & \cdots & 0 \\ \vdots & & \vdots & \ddots & \vdots & & \vdots \\ 0 & \cdots & -s & \cdots & c & \cdots & 0 \\ \vdots & & \vdots & & \vdots & \ddots & \vdots \\ 0 & \cdots & 0 & \cdots & 0 & \cdots & 1 \end{pmatrix}$$

with $c = \cos\theta$ and $s = \sin\theta$ in positions $(i,i)$, $(i,j)$, $(j,i)$, $(j,j)$.

**Key property:** $G(i,j,\theta)$ is orthogonal. Left-multiplying $\mathbf{x}$ by $G$ rotates the $(x_i, x_j)$ plane by angle $\theta$.

**Zeroing a specific entry:** Given $\mathbf{x}$ with entries $x_i$ and $x_j$, choose:
$$r = \sqrt{x_i^2 + x_j^2}, \quad c = x_i/r, \quad s = -x_j/r$$
Then $(G\mathbf{x})_i = r$, $(G\mathbf{x})_j = 0$ — the rotation zeros out $x_j$ while preserving $x_i$ as $r$.

**Numerically stable computation (LAPACK DLARTG):**
```
if x_j == 0:
    c = 1, s = 0, r = x_i
elif |x_j| > |x_i|:
    t = x_i / x_j; s = 1/sqrt(1+t^2); c = s*t; r = x_j/s
else:
    t = x_j / x_i; c = 1/sqrt(1+t^2); s = c*t; r = x_i/c
```
This avoids overflow and underflow in computing $\sqrt{x_i^2 + x_j^2}$.

**Cost:** Applying $G(i,j,\theta)$ to a matrix costs $6m$ flops (touching only rows $i$ and $j$). Constructing $G$ costs $O(1)$.

**For AI:** Givens rotations appear in updating QR factorizations when a new row is appended (streaming QR), which is used in online learning and in updating Cholesky factors after rank-1 additions (Cholesky rank-1 update, relevant to incremental GP regression).

### 4.5 Givens QR and Sparse Matrices

**Algorithm:** Apply Givens rotations sequentially to zero out subdiagonal entries of $A$ column by column. To zero $A_{ij}$ ($i > j$), apply $G(j, i, \theta)$ on the left.

For a dense $m \times n$ matrix, the total number of Givens rotations needed is $mn - n(n+1)/2 \approx mn$, each costing $O(m)$ flops — total $O(m^2 n)$, which is worse than Householder QR for dense matrices.

**Advantage for sparse matrices:** Each Givens rotation touches exactly two rows and two columns. If $A$ has a known sparsity pattern, Givens rotations can be sequenced to minimize fill-in (new nonzeros created by the transformation). In contrast, a single Householder reflector $H_k$ is a rank-2 update that touches an entire panel, potentially creating dense fill.

**Banded matrices:** For a banded matrix with bandwidth $b$, Givens QR costs $O(nb^2)$ instead of $O(nb^2 + n^2 b)$ for Householder, and the resulting $R$ remains banded. This is critical for signal processing and finite-element computations.

**Online/streaming QR:** When a new row $\mathbf{a}^\top$ is appended to $A$, the existing QR factorization can be updated using a single sequence of $n$ Givens rotations, costing $O(n^2)$ rather than recomputing from scratch in $O(mn^2)$.

**For AI:** Streaming QR via Givens is used in online learning settings where data arrives sequentially. It also underlies incremental SVD algorithms used for streaming PCA (e.g., in continual learning, where new tasks arrive without access to past data).

### 4.6 Column-Pivoted QR (RRQR)

**Motivation:** Standard QR doesn't reveal rank. The diagonal entries of $R$ decay, but not necessarily in a way that makes rank determination reliable. Column-pivoted QR (RRQR) reorders columns of $A$ to expose rank structure in $R$.

**Algorithm:** At step $k$, before computing the $k$-th Householder reflector, swap column $k$ with the column $j \geq k$ having **maximum 2-norm among remaining columns**. This produces:

$$A P = QR$$

where $P$ is a permutation and $R$ has the property that $|R_{kk}| \geq |R_{k+1,k+1}| \geq \cdots \geq |R_{nn}|$.

**Rank estimation:** If $A$ has numerical rank $r$, then $|R_{rr}| / |R_{r+1,r+1}|$ is large (theoretically $\geq \sigma_r / \sigma_{r+1}$, practically much larger). The threshold $\tau$ for declaring rank is:
$$r = \max\{k : |R_{kk}| > \tau \cdot |R_{11}|\}$$

**RRQR theorem (Golub 1965, Chandrasekaran & Ipsen 1994):** The strong RRQR algorithm guarantees:
$$\sigma_k(A)^2 \leq \sigma_k(R_{11})^2 \cdot (1 + f(k, n-k, r)), \quad f = O(n)$$
ensuring that $R_{11}$ (the leading $r \times r$ block) captures essentially all the energy of the matrix.

**Connection to SVD:** RRQR is a cheap ($O(mn^2)$) approximation to the SVD. It provides the same rank information and a good basis for the column space, but singular values only approximately. For exact singular values, use SVD at $O(mn^2 + n^3)$ cost.

**For AI:**
- **LoRA (Hu et al., 2022):** Low-Rank Adaptation uses rank-$r$ decompositions of weight updates $\Delta W = BA$. Choosing $r$ requires rank estimation — RRQR is one approach.
- **DoRA (Liu et al., 2024):** Decomposes weight matrices into magnitude and direction; RRQR is used to identify the principal components.
- **Intrinsic dimensionality:** RRQR-based rank estimation reveals the intrinsic dimensionality of weight matrices, guiding compression decisions.

### 4.7 Tall-Skinny QR (TSQR)

**Motivation:** For matrices $A \in \mathbb{R}^{m \times n}$ with $m \gg n$ (tall and skinny — e.g., $m = 10^9$, $n = 100$), standard Householder QR communicates $O(n^2)$ data between levels of the memory hierarchy at each of $n$ steps, totaling $O(n^3)$ words of communication. For distributed or GPU computation, this is the bottleneck.

**TSQR algorithm (Demmel et al., 2008):**
1. **Local factorization:** Partition $A$ into $P$ panels $A_1, \ldots, A_P$ (one per processor/GPU block).
2. **Local QR:** Factor each $A_i = Q_i R_i$ independently (no communication).
3. **Reduction tree:** Form $\begin{pmatrix} R_1 \\ R_2 \end{pmatrix}$, factor its QR to get $R_{12}$. Repeat up the tree.
4. **Result:** $R$ is the final upper triangular factor. $Q$ can be recovered by traversing the reduction tree backward.

**Communication cost:** TSQR communicates $O(n^2 \log P)$ words (vs $O(mn \cdot n/b)$ for blocked Householder), achieving communication optimality.

**For AI:**
- **Distributed training:** TSQR is used in distributed least-squares and in computing the QR of gradient matrices across multiple GPUs.
- **Randomized methods:** TSQR underlies the "sketch-then-QR" approach for randomized low-rank factorization.
- **Attention-efficient computation:** The block-tile structure of FlashAttention-3 (Shah et al., 2024) is inspired by the communication-optimal blocking patterns of TSQR.

### 4.8 QR for Least Squares

**Problem:** Given $A \in \mathbb{R}^{m \times n}$ with $m > n$ (overdetermined) and $\mathbf{b} \in \mathbb{R}^m$, find $\mathbf{x}^*$ minimizing $\|A\mathbf{x} - \mathbf{b}\|_2^2$.

**Method 1 (Normal equations):** $\mathbf{x}^* = (A^\top A)^{-1} A^\top \mathbf{b}$. Form $A^\top A$, factor via Cholesky, solve. Cost: $O(mn^2 + n^3)$. Problem: $\kappa(A^\top A) = \kappa(A)^2$ — squaring the condition number doubles the digits lost.

**Method 2 (QR):** Factor $A = QR$ (thin QR), then:
$$\|A\mathbf{x} - \mathbf{b}\|_2^2 = \|QR\mathbf{x} - \mathbf{b}\|_2^2 = \|R\mathbf{x} - Q^\top\mathbf{b}\|_2^2 + \|(I - QQ^\top)\mathbf{b}\|_2^2$$
Minimizing over $\mathbf{x}$: solve $R\mathbf{x} = Q^\top\mathbf{b}$ (upper triangular, backward substitution). Cost: $O(mn^2)$ for QR, $O(mn)$ for $Q^\top\mathbf{b}$, $O(n^2)$ for backsolve. The condition number of $R$ is $\kappa(A)$, not $\kappa(A)^2$.

**Stability comparison:**

| Method | Condition amplification | Cost | When to use |
|---|---|---|---|
| Normal equations + Cholesky | $\kappa(A)^2$ | $O(mn^2 + n^3)$ | Well-conditioned $A$, $m \gg n$ |
| QR (Householder) | $\kappa(A)$ | $O(mn^2)$ | General purpose, numerically safe |
| RRQR | $\kappa(A)$, reveals rank | $O(mn^2)$ | Rank-deficient or near-singular $A$ |
| SVD (truncated) | None (sets small SVs to zero) | $O(mn^2 + n^3)$ | Ill-conditioned, rank-deficient |

**For AI:**
- **Linear probing:** Fitting a linear classifier on top of frozen features is an overdetermined least-squares problem. `sklearn.linear_model.LinearRegression` uses LAPACK `DGELSD` (divide-and-conquer SVD) or `DGELSY` (column-pivoted QR).
- **Ridge regression:** $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2 + \lambda\|\mathbf{x}\|_2^2$ is equivalent to the augmented system $\begin{pmatrix} A \\ \sqrt{\lambda}I \end{pmatrix} \mathbf{x} = \begin{pmatrix} \mathbf{b} \\ \mathbf{0} \end{pmatrix}$, which QR solves stably.
- **Attention weight regression:** Computing attention pattern regression (e.g., for mechanistic interpretability) uses QR-based least squares.

---

## 5. Cholesky Factorization (Computational Recap)

### 5.1 Brief Overview

> **Full treatment:** The complete theory of Cholesky factorization — existence proofs, LDLᵀ, modified Cholesky, log-determinant, connection to PSD cone — is in [§07: Positive Definite Matrices](../07-Positive-Definite-Matrices/notes.md). This section covers only the computational aspects not covered in §07: the relationship to LU, blocked algorithms, and LAPACK routines.

For a symmetric positive definite matrix $A \succ 0$, the **Cholesky factorization** is:
$$A = LL^\top$$
where $L$ is unit lower triangular with positive diagonal entries $L_{ii} > 0$. The factorization exists and is unique for every $A \succ 0$ (Theorem from §07).

**Why Cholesky for SPD systems:**
1. **Efficiency:** Cholesky costs $\frac{n^3}{3}$ flops — exactly half of LU's $\frac{2n^3}{3}$ — because symmetry halves the work.
2. **Storage:** Only the lower triangle needs to be stored (n(n+1)/2 entries vs n² for LU).
3. **No pivoting needed:** Positive definiteness guarantees $L_{ii} > 0$ at every step without any row interchanges.
4. **Stability:** Cholesky is backward stable for SPD matrices without any pivoting.

### 5.2 Cholesky as Specialized LU

The connection between Cholesky and LU: for a symmetric positive definite $A$, the LU factorization (without pivoting, which exists since all leading principal minors of $A$ are positive) gives $A = LU$. But $A = A^\top$ implies $LU = U^\top L^\top$, so $U = DL^\top$ where $D = \operatorname{diag}(U_{11}, \ldots, U_{nn})$. Thus $A = L D L^\top$ (the LDLᵀ form). Since $A \succ 0$, all diagonal entries $D_{ii} > 0$, and we can define $\tilde{L} = L D^{1/2}$ to get $A = \tilde{L}\tilde{L}^\top$ — the Cholesky factorization.

**Cholesky algorithm (jth column):**
$$L_{jj} = \sqrt{A_{jj} - \sum_{k=1}^{j-1} L_{jk}^2}$$
$$L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=1}^{j-1} L_{ik}L_{jk}\right), \quad i = j+1, \ldots, n$$

The argument of the square root must be positive — this is guaranteed by positive definiteness.

**Cost analysis:**
- Column $j$ costs: 1 square root + $O(j)$ flops for $L_{jj}$, plus $(n-j) \cdot O(j)$ flops for $L_{ij}$
- Total: $\sum_{j=1}^{n} O(nj) = O(n^3/3)$, exactly half of LU

### 5.3 LDLᵀ for Indefinite Systems

For symmetric indefinite matrices (not necessarily PD), Cholesky cannot be applied directly (negative square roots). The **LDLᵀ factorization** computes:
$$P A P^\top = L D L^\top$$
where $L$ is unit lower triangular, $D$ is block diagonal with $1 \times 1$ and $2 \times 2$ blocks (to handle negative eigenvalues), and $P$ is a permutation.

**Bunch-Kaufman pivoting:** The $2 \times 2$ blocks in $D$ capture pairs of eigenvalues of opposite sign, avoiding the square root altogether. The pivoting strategy (Bunch & Kaufman 1977) ensures $|L_{ij}| \leq (1 + \sqrt{17})/8 \approx 0.64$ — a stability bound analogous to $|L_{ij}| \leq 1$ for partial pivoting in LU.

**Applications in ML:**
- **Indefinite Hessians:** At saddle points (common early in neural network training), the Hessian is indefinite. LDLᵀ factors it without the positive-definiteness requirement, enabling second-order descent directions even at saddle points.
- **Modified Cholesky:** Adding a diagonal shift $\delta I$ to make $A + \delta I \succ 0$ before Cholesky factorization is the standard approach in L-BFGS-B and quasi-Newton methods. See §07 for the modified Cholesky algorithm.

**LAPACK routine:** `DSYTRF` implements LDLᵀ with Bunch-Kaufman pivoting.

### 5.4 Blocked Cholesky and LAPACK dpotrf

**Blocked Cholesky** follows the same blocked structure as blocked LU. For block size $b$:

```
for k = 0 to n/b - 1:
    # Factor diagonal block
    L_kk = cholesky(A[kb:(k+1)b, kb:(k+1)b])   # O(b^3)
    # Solve for L panel (DTRSM)
    L_k_below = solve_lower_triangular(L_kk.T, A[(k+1)b:n, kb:(k+1)b].T).T
    # Update trailing submatrix (DSYRK + DGEMM)
    A[(k+1)b:n, (k+1)b:n] -= L_k_below @ L_k_below.T
```

The trailing submatrix update (`A -= L @ L^T`) is a symmetric rank-$b$ update, computed by LAPACK's `DSYRK` (symmetric rank-k update) which exploits symmetry to halve the work relative to `DGEMM`.

**LAPACK routine `DPOTRF`:** The production Cholesky routine. On entry: the upper or lower triangle of $A$. On exit: $L$ (or $U$ for upper Cholesky). Returns an error flag `INFO` ($= 0$ for success; $= k > 0$ if the $k$-th leading minor is not positive definite).

**For AI:** `DPOTRF` is called by:
- `scipy.linalg.cholesky` (wraps LAPACK)
- `numpy.linalg.cholesky` (wraps LAPACK)
- `torch.linalg.cholesky` (calls cuSOLVER `cusolverDnDpotrf` on GPU)
- Every GP regression library (GPyTorch, GPflow, etc.) for computing $K^{-1}\mathbf{y}$ and $\log\det K$

---

## 6. Numerical Stability and Error Analysis

### 6.1 Forward vs Backward Error

**Forward error:** The actual difference between the computed answer $\hat{\mathbf{x}}$ and the true answer $\mathbf{x}^*$:
$$\text{forward error} = \|\hat{\mathbf{x}} - \mathbf{x}^*\| / \|\mathbf{x}^*\|$$

**Backward error:** The smallest perturbation to the input data that makes $\hat{\mathbf{x}}$ an exact solution:
$$\text{backward error} = \min\{\|\delta A\| / \|A\| : (A + \delta A)\hat{\mathbf{x}} = \mathbf{b}\}$$

**The fundamental inequality:**
$$\text{forward error} \leq \kappa(A) \cdot \text{backward error} + O(u^2)$$

This separates the algorithm's contribution (backward error, controlled by stability) from the problem's inherent difficulty (condition number $\kappa(A)$, intrinsic to the matrix). A backward-stable algorithm does the best possible: it cannot do better than $\kappa(A) \cdot u$.

**Why backward error is the right measure:** A backward-stable algorithm produces exact solutions to slightly perturbed problems. If the input data itself has error of order $u$ (which it does — all real data has measurement error), then backward-stable computation introduces no additional error beyond what the input uncertainty already implies.

**Example:** Computing $\hat{x} = (1.0 + 10^{-16}) / (1.0 + 10^{-16}) = 1.0$ in floating-point. The forward error is zero, but the backward error is also zero (exact computation). Now: $\hat{x} = 10^{-16} / 10^{-17} = 10.0$, while the true answer is $10.0 + O(10^{-16})$. Forward error is tiny; backward error is tiny.

### 6.2 The Fundamental Theorem of Backward Stability

**Theorem (Backward Stability, informal).** An algorithm for computing $f(x)$ is **backward stable** if for every input $x$, the computed output $\hat{y}$ satisfies:
$$\hat{y} = f(x + \delta x) \text{ for some } \|\delta x\| \leq c \cdot u \cdot \|x\|$$
where $c$ is a modest polynomial function of the problem size and $u$ is machine epsilon.

**Corollary:** For a backward stable algorithm, the forward error satisfies:
$$\frac{\|\hat{y} - f(x)\|}{\|f(x)\|} \leq \kappa_f \cdot c \cdot u$$
where $\kappa_f$ is the condition number of $f$. The condition number determines how much the backward perturbation $\delta x$ amplifies into forward error.

**LU with partial pivoting:** backward stable with growth factor $\rho_n$. Not backward stable in theory (since $\rho_n$ can be $2^{n-1}$), but backward stable in practice (since $\rho_n$ rarely exceeds $n^{2/3}$ for typical inputs).

**Householder QR:** backward stable unconditionally. The computed $\hat{Q}$ satisfies $\|\hat{Q}^\top\hat{Q} - I\| = O(u)$ regardless of $\kappa(A)$.

**Cholesky (for SPD):** backward stable unconditionally. No pivoting needed; positive definiteness guarantees numerical stability.

### 6.3 Stability Comparison: LU vs QR vs Cholesky

| Property | LU (partial pivot) | Householder QR | Cholesky (SPD) |
|---|---|---|---|
| Growth factor $\rho_n$ | $\leq 2^{n-1}$ (worst case) | 1 (exact) | 1 (exact) |
| Backward stable? | In practice, yes | Unconditionally | Unconditionally |
| Orthogonality loss | N/A | $O(u)$ | N/A |
| Requires pivoting? | Yes (partial pivot) | No (column pivot for RRQR) | No |
| Cost (leading term) | $\frac{2}{3}n^3$ | $2mn^2 - \frac{2}{3}n^3$ | $\frac{1}{3}n^3$ |
| Memory | $n^2$ | $mn$ (+ $n^2$ for R) | $\frac{n(n+1)}{2}$ |
| Handles all square matrices? | Yes (nonsingular) | Yes | No (SPD only) |

**Rule of thumb:**
- Have a symmetric positive definite matrix? Use **Cholesky** (twice as fast as LU, always stable).
- Have a general square system? Use **LU with partial pivoting** (standard default in all libraries).
- Have an overdetermined least-squares problem? Use **Householder QR**.
- Need rank determination or have a rank-deficient matrix? Use **RRQR** or **SVD**.
- Have an ill-conditioned system where every digit matters? Use **QR** (avoids condition number squaring).

### 6.4 Mixed Precision and Iterative Refinement

**Mixed precision computation:** Modern GPUs (NVIDIA A100, H100) support FP64, FP32, FP16, and BF16 with different throughputs. The ratio is typically 1:2:8:8 (64:32:16:16 bit throughput). Training AI models almost exclusively uses FP16 or BF16 to maximize throughput; but this reduces precision from $u = 10^{-16}$ (FP64) to $u = 10^{-3}$ (FP16).

**Iterative refinement algorithm:**
```
1. Factor A ≈ PA = LU in low precision (FP16 or FP32)
2. Solve Ax_0 = b using L, U in low precision          → x_0
3. For k = 0, 1, 2, ...:
   a. Compute residual r_k = b - A x_k  (in HIGH precision, FP64)
   b. Solve L U δ_k = r_k               (in LOW precision — cheap)
   c. Update x_{k+1} = x_k + δ_k
4. Converge when ||r_k|| / ||b|| < ε_target
```

**Convergence:** Iterative refinement converges in $O(1)$ iterations if the initial precision is high enough relative to $\kappa(A)$. Specifically, if $\kappa(A) \cdot u_{\text{factor}} < 1$, refinement converges to FP64 precision in 2-3 steps.

**NVIDIA cuSOLVER IR (Iterative Refinement):** Implements exactly this pattern: BF16 factorization (8x throughput vs FP64) + FP64 residual computation + BF16 solve for correction. Used in NVIDIA's deep learning solver libraries and in the mixed-precision training of large language models.

**For AI:** NVIDIA's "GMRES-IR" (GMRES Iterative Refinement) in cuSOLVER applies this to poorly conditioned systems arising in transformer inference optimization. K-FAC implementations use this to factor Fisher information blocks quickly in FP16 and refine to FP32 accuracy.

### 6.5 Ill-Conditioned Systems and Regularization

When $\kappa(A)$ is very large, even backward-stable algorithms produce solutions with large forward error. The cure is regularization — changing the problem to a nearby well-conditioned one.

**Tikhonov regularization:** Replace $A\mathbf{x} = \mathbf{b}$ with $\min_\mathbf{x} \|A\mathbf{x} - \mathbf{b}\|^2 + \lambda\|\mathbf{x}\|^2$. The solution is $\mathbf{x}_\lambda = (A^\top A + \lambda I)^{-1} A^\top \mathbf{b}$. The regularized condition number is:
$$\kappa(A^\top A + \lambda I) = \frac{\sigma_1^2 + \lambda}{\sigma_n^2 + \lambda}$$
which decreases as $\lambda$ increases. At $\lambda = \sigma_n^2$: condition number $\approx (\sigma_1/\sigma_n)^2 / 2 = \kappa^2/2$. At $\lambda = \sigma_1^2$: condition number $= 2$.

**Truncated factorizations:** Set all singular values (or diagonal entries of $R$ or $U$) below a threshold to zero. This is the basis of truncated SVD and RRQR-based pseudoinverse computation.

**For AI:** The Adam optimizer's $\epsilon$ parameter (default $10^{-8}$) is exactly Tikhonov regularization applied to the per-parameter Hessian approximation. K-FAC's damping parameter $\lambda$ similarly regularizes the Fisher information matrix to prevent ill-conditioned natural gradient steps.

---

## 7. Blocked Algorithms and High-Performance Computing

### 7.1 Cache Hierarchy and Algorithm Design

Modern computer architecture has a deep memory hierarchy:

| Level | Size | Latency | Bandwidth |
|---|---|---|---|
| L1 cache | 32-64 KB | 4 cycles | 1 TB/s |
| L2 cache | 256-512 KB | 12 cycles | 400 GB/s |
| L3 cache | 8-32 MB | 40 cycles | 200 GB/s |
| DRAM | 16-512 GB | 100+ cycles | 50 GB/s |
| GPU HBM | 16-80 GB | 80 cycles | 2 TB/s |

**The roofline model:** An algorithm's performance is limited by the minimum of:
1. Peak arithmetic throughput (FLOP/s)
2. Memory bandwidth × arithmetic intensity (bytes/flop)

A naive matrix-vector multiply $\mathbf{y} = A\mathbf{x}$ reads $n^2 + 2n$ words and performs $2n^2$ flops — arithmetic intensity $O(1)$. It is **memory bandwidth bound**: DRAM bandwidth, not arithmetic units, limits performance.

A matrix-matrix multiply $C = AB$ reads $2n^2$ words and performs $2n^3$ flops — arithmetic intensity $O(n)$. For large $n$, it is **compute bound** and can achieve near-peak FLOP/s.

**The key insight for blocked algorithms:** Reformulate the bottleneck step as a matrix-matrix multiply (BLAS-3), not as a matrix-vector multiply (BLAS-2). This changes arithmetic intensity from $O(1)$ to $O(n)$, giving $O(n)$-fold speedup on modern hardware.

### 7.2 Blocked Factorizations

All three factorizations (LU, QR, Cholesky) can be reformulated as blocked algorithms that expose BLAS-3 matrix-matrix multiplies as the dominant computation.

**Block LU (Panel + Update):**
```
A = [A_{11} A_{12}]    Factor: [L_{11}   0  ] [U_{11} U_{12}]
    [A_{21} A_{22}]            [L_{21}  L_{22}] [  0   U_{22}]

Step 1: Factor A_11 = L_{11} U_{11}           (small, BLAS-2)
Step 2: L_{21} = A_{21} U_{11}^{-1}           (DTRSM, BLAS-3)
Step 3: U_{12} = L_{11}^{-1} A_{12}           (DTRSM, BLAS-3)  
Step 4: A_{22} -= L_{21} U_{12}               (DGEMM, BLAS-3) ← dominates
Step 5: Recurse on A_{22}
```

**WY representation for blocked Householder QR:** A product of $b$ Householder reflectors $H_1 \cdots H_b$ can be written as $I - WY^\top$ where $W, Y \in \mathbb{R}^{m \times b}$. Applying $I - WY^\top$ to a matrix $C$ costs $O(mbn)$ using BLAS-3 `DGEMM`. LAPACK's `DGEQRF` uses this representation for blocked Householder QR.

**Performance:** On a modern CPU with AVX-512, blocked DGEMM achieves $> 80\%$ of peak FLOP/s for $n > 500$. Without blocking (using BLAS-2), performance drops to $< 10\%$ of peak. The $8 \times$ speedup from blocking is why production factorization libraries use block sizes of $64$–$256$.

### 7.3 LAPACK Routine Reference

| Routine | Operation | Notes |
|---|---|---|
| `DGETRF` | LU with partial pivoting ($PA = LU$) | Returns pivot indices; in-place |
| `DGETRS` | Solve $AX = B$ using DGETRF output | Multiple right-hand sides |
| `DGETRI` | Compute $A^{-1}$ from DGETRF output | Rarely needed; use DGETRS instead |
| `DGEQRF` | QR factorization (Householder) | Returns H vectors; blocked (WY) |
| `DORMQR` | Apply $Q$ or $Q^\top$ from DGEQRF | Without forming Q explicitly |
| `DORGQR` | Form explicit Q from DGEQRF output | Costs $O(mn^2)$ extra |
| `DGELSY` | Least squares via column-pivoted QR | Rank-deficient safe |
| `DPOTRF` | Cholesky ($A = LL^\top$ or $UU^\top$) | Returns INFO error if not SPD |
| `DPOTRS` | Solve $AX = B$ using DPOTRF output | SPD matrix only |
| `DSYTRF` | LDLᵀ for symmetric indefinite | Bunch-Kaufman pivoting |
| `DSYTRS` | Solve using DSYTRF output | |

**LAPACK naming convention:** `D` = double precision, `S` = single, `Z` = complex double; `GE` = general, `PO` = positive definite, `SY` = symmetric; `TRF` = triangular factorization, `TRS` = triangular solve, `TRI` = triangular inverse.

**In Python:** `scipy.linalg` provides direct LAPACK access via `scipy.linalg.lapack.dgetrf`, etc. High-level wrappers (`scipy.linalg.lu`, `scipy.linalg.qr`, `scipy.linalg.cholesky`) call these automatically.

### 7.4 Sparse Factorizations

For sparse matrices (e.g., from finite-element discretization, graph Laplacians, or neural network weight matrices with structured sparsity), dense factorizations waste memory and computation on zeros.

**Fill-in:** When a sparse matrix is factored, the $L$ and $U$ factors typically contain more nonzeros than $A$ itself. The additional nonzeros are called **fill-in**.

**Reordering to minimize fill-in:**
- **Minimum degree ordering (AMD):** Greedy heuristic — eliminate the variable connected to fewest others first. Approximate minimum degree (AMD) is the production algorithm.
- **Nested dissection:** Recursively partition the matrix graph using separator sets; theoretically optimal for regular 2D grids ($O(n^{1.5})$ fill vs $O(n^2)$ without reordering).

**Supernodal methods:** CHOLMOD (Davis & Hager, 2009) uses "supernodes" — groups of columns with identical sparsity pattern — and applies dense BLAS-3 operations to each supernode, recovering high arithmetic intensity.

**For AI:**
- **Graph neural networks:** The adjacency matrix of a graph is sparse. Solving GNN systems requires sparse Cholesky or LU.
- **Attention patterns:** Sparse attention (Longformer, BigBird) creates structured-sparse weight matrices. Sparse LU factorization is used in some attention-efficient inference methods.
- **Finite-element physics simulation:** Differentiable simulation requires backpropagating through sparse LU/Cholesky, implemented in JAX using `jax.experimental.sparse`.

---

## 8. Applications in Machine Learning

### 8.1 Least Squares via QR

**Setup:** The standard supervised learning problem with linear model $\hat{y} = \mathbf{w}^\top \mathbf{x}$ over $n$ training examples $(X \in \mathbb{R}^{n \times d}, \mathbf{y} \in \mathbb{R}^n)$ reduces to the least-squares problem:

$$\mathbf{w}^* = \arg\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|_2^2$$

**QR solution procedure:**
1. Factor $X = QR$ (thin QR, $Q \in \mathbb{R}^{n \times d}$, $R \in \mathbb{R}^{d \times d}$)
2. Compute $\hat{\mathbf{y}} = Q^\top \mathbf{y}$ (project $\mathbf{y}$ onto column space of $X$)
3. Solve $R\mathbf{w}^* = \hat{\mathbf{y}}$ (backward substitution)

**Why QR over normal equations in practice:**
- Normal equations form $X^\top X$, which has condition number $\kappa(X)^2$. For typical neural network feature matrices ($\kappa \sim 10^4$–$10^6$), this squares to $10^8$–$10^{12}$, losing 8-12 digits of precision in double.
- QR maintains condition number $\kappa(X)$ throughout — twice as many correct digits.

**Ridge regression via augmented QR:** $\min \|X\mathbf{w} - \mathbf{y}\|^2 + \lambda\|\mathbf{w}\|^2$ is equivalent to:
$$\tilde{X} = \begin{pmatrix} X \\ \sqrt{\lambda} I_d \end{pmatrix}, \quad \tilde{\mathbf{y}} = \begin{pmatrix} \mathbf{y} \\ \mathbf{0} \end{pmatrix}$$
Factor $\tilde{X} = QR$ and solve $R\mathbf{w} = Q^\top \tilde{\mathbf{y}}$. This is numerically superior to forming $(X^\top X + \lambda I)$ explicitly.

**For AI:** Linear probing (evaluating feature quality by fitting a linear classifier) and prompt tuning (fitting a linear head on frozen features) both reduce to this problem. The `sklearn.linear_model.Ridge` with `solver='svd'` or `'cholesky'` uses exactly these methods.

### 8.2 Newton's Method and Second-Order Optimization

**Newton's method** for minimizing $\mathcal{L}(\boldsymbol{\theta})$ applies the update:
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - H_t^{-1} \nabla\mathcal{L}(\boldsymbol{\theta}_t)$$
where $H_t = \nabla^2\mathcal{L}(\boldsymbol{\theta}_t)$ is the Hessian. This requires solving the linear system $H_t \boldsymbol{\delta} = \nabla\mathcal{L}$ at each step — a direct application of LU or Cholesky factorization.

**Cost per step:** $O(p^3)$ where $p = |\boldsymbol{\theta}|$. For GPT-3 with $p = 175 \times 10^9$, this is astronomically expensive. Full Newton is infeasible for large neural networks.

**K-FAC (Kronecker-Factored Approximate Curvature, Martens & Grosse 2015):** Approximates the Fisher information matrix (a PSD proxy for the Hessian) as a block-diagonal Kronecker product:
$$F \approx \bigoplus_l A_l \otimes G_l$$
where $A_l \in \mathbb{R}^{d_l \times d_l}$ and $G_l \in \mathbb{R}^{d_{l+1} \times d_{l+1}}$ are computed from activations and pre-activation gradients respectively. Each block is inverted via Cholesky:
$$(A_l \otimes G_l)^{-1} = A_l^{-1} \otimes G_l^{-1}$$
with $A_l^{-1}$ and $G_l^{-1}$ computed via Cholesky factorization of each factor separately.

**Cost per K-FAC step:** $O(\sum_l (d_l^3 + d_{l+1}^3))$ — sum over layers, each requiring two Cholesky factorizations of small matrices. For a 10-layer network with hidden dim 1024: $\approx 10 \times 2 \times 10^9 = 2 \times 10^{10}$ flops, feasible on modern hardware.

**Shampoo (Gupta et al., 2018; Anil et al., 2020):** Maintains full-matrix preconditioners per layer:
$$G_t = G_{t-1} + g_t g_t^\top, \quad P_t = G_t^{-1/4}$$
The matrix square root $G_t^{-1/4}$ requires eigendecomposition or iterative methods. The recent "Distributed Shampoo" (Google Brain, 2022) runs this in FP32 on accelerators, with Cholesky-based Newton iterations for the matrix square root.

### 8.3 Gaussian Process Inference at Scale

**GP regression setup:** Given observations $\mathbf{y} = f(X) + \varepsilon$ with $f \sim \mathcal{GP}(0, k)$ and $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$, the posterior predictive mean and variance at test points $X_*$ are:

$$\boldsymbol{\mu}_* = K_{*n}(K_{nn} + \sigma^2 I)^{-1}\mathbf{y}$$
$$\Sigma_* = K_{**} - K_{*n}(K_{nn} + \sigma^2 I)^{-1}K_{n*}$$

where $K_{nn} \in \mathbb{R}^{n \times n}$ is the training kernel matrix and $K_{*n} \in \mathbb{R}^{n_* \times n}$ is the test-train kernel matrix.

**Cholesky for GP:** The matrix $K_{nn} + \sigma^2 I$ is SPD (positive definite noise term ensures this). The Cholesky factorization:
$$K_{nn} + \sigma^2 I = LL^\top$$
enables efficient computation of:
1. $\boldsymbol{\alpha} = (K_{nn} + \sigma^2 I)^{-1}\mathbf{y}$: forward + backward solve with $L$, cost $O(n^3)$ once, $O(n^2)$ for new right-hand sides
2. $\log p(\mathbf{y} \mid X, \boldsymbol{\theta}) = -\frac{1}{2}\mathbf{y}^\top\boldsymbol{\alpha} - \sum_i \log L_{ii} - \frac{n}{2}\log 2\pi$: the log-likelihood used for hyperparameter optimization

**Bottleneck:** For $n = 10{,}000$ training points, $K_{nn}$ is $10^4 \times 10^4$, and Cholesky costs $\frac{(10^4)^3}{3} \approx 3 \times 10^{11}$ flops — seconds on a GPU, but prohibitive for $n = 10^6$.

**Scalable GP methods:**
- **Inducing point methods (Titsias 2009):** Introduce $m \ll n$ inducing points; compute $m \times m$ Cholesky instead of $n \times n$. GPyTorch implements this as `gpytorch.models.ApproximateGP`.
- **SKI/KISS-GP (Wilson & Nickisch 2015):** Structure kernel interpolation — exploit grid structure to decompose $K_{nn}$ as a Kronecker product, enabling $O(n)$ Cholesky-like solves.
- **CG + Lanczos:** Conjugate gradient-based methods (Gardner et al., 2018) avoid explicit Cholesky by iteratively solving the system, used in GPyTorch's `LinearCG` backend.

**For AI:** GP regression is the inference engine of Bayesian optimization (BoHB, Lindauer et al., 2022), which is used for hyperparameter tuning of large language models. Every BoO iteration requires a Cholesky factorization.

### 8.4 Backpropagation Through Factorizations

**Differentiating through Cholesky:** If $A = LL^\top$ and a scalar loss $\ell = f(L)$, then:
$$\frac{\partial \ell}{\partial A} = L^{-\top}\Phi(L^\top \frac{\partial \ell}{\partial L})L^{-1}$$
where $\Phi$ extracts the lower triangular part: $\Phi(M)_{ij} = M_{ij}$ for $i > j$, $\frac{1}{2}M_{ii}$ for $i = j$, $0$ for $i < j$.

This formula (Iain Murray, 2016) is implemented in PyTorch's `torch.linalg.cholesky` with `backward()` support.

**Differentiating through LU:** For $A = P^{-1}LU$ and a loss through the solution $\mathbf{x} = A^{-1}\mathbf{b}$:
$$\frac{\partial \ell}{\partial A} = -A^{-\top}\frac{\partial \ell}{\partial \mathbf{x}}\mathbf{x}^\top = -\boldsymbol{\lambda}\mathbf{x}^\top$$
where $\boldsymbol{\lambda} = A^{-\top}(\partial\ell/\partial\mathbf{x})$ is the adjoint vector (computed via backward triangular solves).

**Implicit differentiation:** Rather than differentiating through the factorization algorithm itself (which would unroll all $O(n^3)$ operations), implicit differentiation differentiates through the **solution condition** $A\mathbf{x}^* = \mathbf{b}$:
$$A \frac{\partial\mathbf{x}^*}{\partial\boldsymbol{\theta}} = \frac{\partial\mathbf{b}}{\partial\boldsymbol{\theta}} - \frac{\partial A}{\partial\boldsymbol{\theta}}\mathbf{x}^*$$
This requires solving one additional linear system per output — $O(n^2)$ using the already-computed factorization.

**For AI:** Differentiating through linear solvers is central to:
- **Meta-learning:** MAML and its variants differentiate through inner-loop optimization steps
- **Differentiable physics:** FEA simulation (e.g., PhiML) requires differentiating through sparse LU solves
- **Constrained optimization:** Interior-point methods for differentiable convex optimization
- **Gaussian processes in end-to-end learning:** Differentiating the GP marginal likelihood w.r.t. kernel parameters requires differentiating through Cholesky

### 8.5 Randomized Factorizations

**Motivation:** For matrices where only a low-rank approximation is needed, exact factorization costs $O(mn^2)$ — wasteful if rank $r \ll n$. Randomized methods achieve $O(mnr)$ or even $O(mn \log r)$ using random projections.

**Randomized QR (sketch-and-apply):**
1. **Sketch:** $Y = A\Omega$ where $\Omega \in \mathbb{R}^{n \times (r+p)}$ is a random Gaussian matrix ($p \approx 10$ oversampling)
2. **Orthogonalize:** Factor $Y = QR$ (thin QR of the sketch)
3. **Project:** $B = Q^\top A$ (project $A$ onto the sketched subspace)
4. **Factor:** $B = \tilde{Q}\tilde{R}$ (thin QR of the $\ell \times n$ matrix)
5. **Output:** $A \approx (Q\tilde{Q})\tilde{R}$ — a rank-$(r+p)$ QR approximation

**Cost:** Steps 1-4 cost $O(mn(r+p))$, negligible compared to $O(mn^2)$ for exact QR.

**Error bound (Halko-Martinsson-Tropp, 2011):** With probability $\geq 1 - 6p^{-p}$:
$$\|A - Q Q^\top A\|_2 \leq \left(1 + 11\sqrt{r+p}\right)\sigma_{r+1}(A)$$
The approximation is near-optimal — within a polynomial factor of the Eckart-Young lower bound.

**Randomized LU (Yu et al., 2018):** Applies random column sampling to produce a structured LU factorization useful for rank-deficient matrices.

**Connection to LoRA (Hu et al., 2022):** Low-Rank Adaptation of language models uses rank-$r$ decompositions $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$. Initializing $B$ via randomized QR of the pretrained weight matrix gives a structured initialization that preserves the principal components.

**DoRA (Liu et al., 2024):** Decomposes $W = m \cdot (W/\|W\|_c)$ where $\|W\|_c$ is the column norm. The directional component $W/\|W\|_c$ is approximated via LoRA. This uses the QR decomposition of each weight column.

**GALORE (Zhao et al., 2024):** Gradient Low-Rank Projection — projects gradients onto their principal subspace via randomized SVD before applying Adam updates. The projection matrix is computed via randomized QR every 200 steps.

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using LU without pivoting for numerical computation | Zero or small pivots cause catastrophic cancellation; growth factor can reach $2^{n-1}$ | Always use partial pivoting (`scipy.linalg.lu`, LAPACK `dgetrf`) |
| 2 | Forming $A^{-1}$ explicitly to solve $A\mathbf{x} = \mathbf{b}$ | Computing $A^{-1}$ costs $O(n^3)$ and introduces additional round-off; $A^{-1}\mathbf{b}$ is less accurate than the triangular solve | Use `scipy.linalg.solve` or `lu_solve` with the stored factorization |
| 3 | Applying Cholesky to a non-SPD matrix | Negative square roots cause `NaN`; the algorithm may "succeed" silently with complex values or incorrect results | Check `np.linalg.eigvalsh(A).min() > 0` or catch `LinAlgError` from failed Cholesky |
| 4 | Using normal equations for ill-conditioned least squares | $\kappa(A^\top A) = \kappa(A)^2$ — doubles the digits lost; for $\kappa \sim 10^6$, you lose all precision | Use `scipy.linalg.lstsq` (QR-based) or explicitly factor via Householder QR |
| 5 | Confusing thin and full QR | Full QR: $Q \in \mathbb{R}^{m \times m}$, $R \in \mathbb{R}^{m \times n}$. Thin QR: $Q \in \mathbb{R}^{m \times n}$, $R \in \mathbb{R}^{n \times n}$. Using the wrong one gives wrong dimensions | Use `mode='economic'` in `scipy.linalg.qr` or `full_matrices=False` in `numpy.linalg.qr` for thin QR |
| 6 | Ignoring the growth factor in stability analysis | Partial pivoting has theoretical growth factor $2^{n-1}$; while rare in practice, adversarial inputs exist | For certified computation, use complete pivoting or switch to Householder QR |
| 7 | Re-factoring A for each right-hand side | LU/QR factorization costs $O(n^3)$; each additional solve costs only $O(n^2)$ | Cache the factorization (`lu_factor` returns LU; `qr` returns Q,R) and call `lu_solve` or backsolve repeatedly |
| 8 | Forgetting the permutation $P$ in $PA = LU$ | Solving $L(U\mathbf{x}) = \mathbf{b}$ instead of $L(U\mathbf{x}) = P\mathbf{b}$ gives wrong answer | Always apply $P\mathbf{b}$ first: `b = b[piv]` or use `lu_solve` which handles pivots automatically |
| 9 | Using classical Gram-Schmidt for QR instead of Householder | CGS loses orthogonality for ill-conditioned matrices; $\|\hat{Q}^\top\hat{Q} - I\| = O(\kappa(A) \cdot u)$ | Use `scipy.linalg.qr` (Householder) or `numpy.linalg.qr` |
| 10 | Misinterpreting rank from LU diagonal | LU (even with partial pivoting) doesn't give a reliable rank estimate; the pivot threshold is ambiguous | Use RRQR (`scipy.linalg.qr(pivoting=True)`) or SVD for rank determination |
| 11 | Ignoring mixed precision issues in GPU factorizations | FP16 Cholesky has $u = 5 \times 10^{-4}$; for $\kappa(A) > 10^4$, FP16 factorization gives only 0 correct digits | Use FP32 or FP64 for factorizations; apply iterative refinement if FP16 is needed for speed |
| 12 | Differentiating through factorization naively by unrolling | Unrolling $O(n^3)$ elimination steps creates $O(n^3)$ graph edges — memory explosion | Use implicit differentiation through the solution condition $A\mathbf{x} = \mathbf{b}$; PyTorch's `linalg.solve` handles this automatically |

---

## 10. Exercises

**Exercise 1 (★):** Implement forward and backward substitution.

(a) Write a function `forward_sub(L, b)` that solves $L\mathbf{x} = \mathbf{b}$ for unit lower triangular $L$ (no division needed).

(b) Write `backward_sub(U, b)` that solves $U\mathbf{x} = \mathbf{b}$ for upper triangular $U$.

(c) Verify your implementations against `scipy.linalg.solve_triangular` on a $5 \times 5$ system.

(d) Measure the accuracy: for a random $100 \times 100$ lower triangular system with entries from $\mathcal{N}(0,1)$, compute $\|L\hat{\mathbf{x}} - \mathbf{b}\|_\infty / \|\mathbf{b}\|_\infty$ (residual) and compare to machine epsilon.

**Exercise 2 (★):** Implement naive LU and demonstrate failure.

(a) Implement `lu_naive(A)` that performs Gaussian elimination without pivoting, returning $(L, U)$.

(b) For the matrix $A = \begin{pmatrix} \varepsilon & 1 \\ 1 & 2 \end{pmatrix}$ with $\varepsilon = 10^{-16}$, compute $\hat{\mathbf{x}} = U^{-1}(L^{-1}\mathbf{b})$ for $\mathbf{b} = (1, 3)^\top$ using your naive LU.

(c) Compare to `numpy.linalg.solve`. Report the relative error $\|\hat{\mathbf{x}} - \mathbf{x}^*\|_\infty / \|\mathbf{x}^*\|_\infty$.

(d) Explain the failure in terms of the growth factor. What is $\rho_2$ for this matrix?

**Exercise 3 (★):** Implement LU with partial pivoting and verify $PA = LU$.

(a) Implement `lu_pivot(A)` returning $(P, L, U)$ where $P$ is stored as a permutation vector.

(b) Verify the factorization: compute $\|PA - LU\|_F$ and confirm it is $O(n \cdot u \cdot \|A\|_F)$.

(c) Use your factorization to solve a $50 \times 50$ random system and compare accuracy to the naive LU.

(d) Compute $\det(A)$ from the LU factorization as $(-1)^s \prod_i U_{ii}$ and verify against `numpy.linalg.det`.

**Exercise 4 (★★):** Implement a Householder reflector and apply it.

(a) Implement `householder_vector(a)` that computes the Householder vector $\mathbf{v}$ such that $H\mathbf{a} = \alpha\mathbf{e}_1$ with the correct sign convention to avoid cancellation.

(b) Implement `householder_apply(v, C)` that applies $H = I - 2\mathbf{v}\mathbf{v}^\top/\|\mathbf{v}\|^2$ to matrix $C$ using the implicit formula (not the full $m \times m$ matrix).

(c) Verify: for a random vector $\mathbf{a} \in \mathbb{R}^5$, confirm that $H\mathbf{a} = \alpha\mathbf{e}_1$ (entries 2-5 should be zero to machine precision).

(d) Show that the naive computation `(I - 2*v*v.T/norm(v)**2) @ a` gives the same result but is less efficient. Time both for a large vector.

**Exercise 5 (★★):** Implement Householder QR and compare to scipy.

(a) Implement `householder_qr(A)` that returns the upper triangular factor $R$ and stores Householder vectors.

(b) Implement `recover_Q(householder_vecs, m)` to form the explicit $Q$ matrix.

(c) Verify $\|A - QR\|_F \leq c \cdot u \cdot \|A\|_F$ and $\|Q^\top Q - I\|_F \leq c \cdot u$ for a $50 \times 30$ matrix.

(d) Compare to `scipy.linalg.qr`. Verify they give identical $R$ (up to sign conventions) and identical orthogonality.

(e) For an ill-conditioned $A$ with $\kappa(A) = 10^8$, compare the orthogonality $\|Q^\top Q - I\|_F$ of your Householder QR vs classical Gram-Schmidt.

**Exercise 6 (★★):** Rank estimation via column-pivoted QR.

(a) Create a rank-4 matrix: $A = U S V^\top$ where $S = \operatorname{diag}(100, 10, 1, 0.1, 0.001, 0.001, 0.001)$ and $U, V$ are random orthogonal.

(b) Compute QR with column pivoting: `Q, R, P = scipy.linalg.qr(A, pivoting=True)`.

(c) Plot the diagonal entries $|R_{11}|, \ldots, |R_{77}|$ on a log scale. Identify the rank gap.

(d) Implement rank estimation: declare rank $r$ when $|R_{rr}|/|R_{11}| > \tau$ for threshold $\tau = 10^{-3}$.

(e) Compare to the SVD-based rank estimation. Which is more reliable? Which is faster?

**Exercise 7 (★★★):** GP regression with Cholesky.

(a) Implement `rbf_kernel(X1, X2, ell, sf)` computing $k(x_i, x_j) = s_f^2 \exp(-\|x_i - x_j\|^2 / (2\ell^2))$.

(b) Generate 100 training points from $f(x) = \sin(x)$ with $\mathcal{N}(0, 0.1^2)$ noise.

(c) Implement `gp_predict(X_train, y_train, X_test, ell, sf, sigma_n)` using Cholesky factorization for all linear solves (no explicit matrix inverse).

(d) Compute the log marginal likelihood $\log p(\mathbf{y} \mid X) = -\frac{1}{2}\mathbf{y}^\top K^{-1}\mathbf{y} - \frac{1}{2}\log\det K - \frac{n}{2}\log 2\pi$ using `log_det_chol` (2∑log diagonal of L).

(e) Optimize $(\ell, s_f)$ via grid search maximizing the log marginal likelihood. Plot the GP posterior mean and ±2σ credible interval.

**Exercise 8 (★★★):** Differentiating through Cholesky — gradient of log-determinant.

(a) For an SPD matrix $A$, the gradient of $\log\det A$ w.r.t. $A$ is $A^{-1}$. Verify this numerically: compute $\nabla_A \log\det A$ via finite differences and via $L^{-\top}L^{-1}$ (using Cholesky), compare.

(b) Implement `cholesky_grad(L, dL)` computing the gradient of a scalar function $f(L)$ (with given $\partial f/\partial L = \mathtt{dL}$) w.r.t. $A$, using the Iain Murray (2016) formula.

(c) Use your gradient to implement gradient descent on $\mathcal{L}(A) = -\log\det A + \operatorname{tr}(A)$ (the negative log-likelihood of a Wishart distribution). Starting from $A = 5I$, verify convergence to $A = I$ (the analytic minimum).

(d) Compare to PyTorch's automatic differentiation through `torch.linalg.cholesky` and `torch.logdet`. Verify that gradients match to $10^{-5}$ relative error.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Impact |
|---|---|
| LU with partial pivoting | Solving Hessian systems in Newton/quasi-Newton methods; computing inverses and log-determinants in probabilistic ML |
| Householder QR | Foundation of least-squares linear probing; basis for eigenvalue algorithms (QR iteration); stable weight matrix analysis |
| Column-pivoted QR (RRQR) | Rank estimation of weight matrices in LoRA, DoRA; detecting intrinsic dimensionality of neural representations |
| Cholesky factorization (computational) | GP regression at scale; K-FAC training; VAE/diffusion sampling; Fisher information inversion |
| Blocked algorithms (BLAS-3) | 80-95% of peak GPU throughput for all factorizations; the reason FlashAttention uses tile-based blocking |
| Backward error analysis | Understanding precision limits in FP16/BF16 training; justifying mixed-precision with iterative refinement |
| Iterative refinement | cuSOLVER IR: FP16 factorization + FP64 refinement for numerically reliable inference at high throughput |
| Sparse factorizations | GNN training on large graphs; differentiable physics simulation; structured attention patterns |
| QR for least squares | Linear probing, ridge regression, prompt tuning accuracy; avoiding condition-number squaring in normal equations |
| Differentiation through factorizations | End-to-end differentiable GP/Bayesian models; implicit differentiation in meta-learning; constraint-based optimization |
| TSQR | Communication-optimal QR for distributed training; tile-based blocking in FlashAttention-3 |
| Randomized LU/QR | LoRA initialization; GaLore gradient projection; sketch-and-solve preconditioning for large-scale optimization |
| K-FAC with Cholesky | Natural gradient training of large transformers; 2-5x faster convergence than Adam at equivalent compute |
| GP marginal likelihood optimization | Bayesian hyperparameter tuning (BoHB); automated ML pipeline optimization |
| Rank-deficient LU / RRQR | Intrinsic dimensionality measurement of LLM weight updates; guided rank selection for compression |

---

## 12. Conceptual Bridge

**From §07 (Positive Definite Matrices):** Section 07 developed the theory of positive definite matrices and gave the full treatment of the Cholesky factorization: existence proof (the Cholesky product theorem), the LDLᵀ variant, the connection to log-determinants, and the PSD cone. This section builds directly on that theory, treating Cholesky as one of three production-grade computational tools alongside LU and QR. The student who understands §07's theory will see §08's Cholesky discussion as pure application — taking the mathematical theorem and implementing it efficiently.

**The computational turn:** Sections 01-07 of Chapter 3 developed mathematical structures: eigenvalues, SVD, PCA, linear maps, orthogonality, norms, and positive definiteness. Section 08 marks a turn toward the computational: algorithms, numerical stability, blocking, LAPACK. This turn is not a descent in abstraction but a different kind of mathematics — the mathematics of floating-point arithmetic, error propagation, and algorithm design under hardware constraints. Both kinds are essential for AI/ML practice.

**Into Chapter 4 (Calculus):** The matrix factorizations developed here are the computational backbone of calculus-based optimization. The Hessian matrix $H_f(\mathbf{x})$ (Chapter 4, §05) is positive definite at local minima — and therefore amenable to Cholesky. Newton's method (Chapter 4, §06) requires solving Hessian systems via LU or Cholesky at each step. The Jacobian matrix (Chapter 4, §04) appears in the normal equations for nonlinear least squares (Gauss-Newton, Levenberg-Marquardt), which are solved via QR. The machinery of §08 is the engine that makes calculus-based optimization computable.

**Into Chapter 8 (Optimization):** The optimization chapter (Chapter 8) develops gradient descent, Newton's method, interior-point methods, and constrained optimization — all of which require the factorizations of §08 in their inner loops. The K-FAC connection to natural gradient optimization, the QR connection to least-squares problems, and the Cholesky connection to GP-based Bayesian optimization are all elaborated with full optimization context in Chapter 8.

```
MATRIX DECOMPOSITIONS IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  Chapter 2: Linear Algebra Basics
  ├── §02 Matrix Operations ──────── brief LU/QR/Cholesky preview
  └── §04 Determinants ──────────── det(A) = ∏ U_ii from LU

  Chapter 3: Advanced Linear Algebra
  ├── §01 Eigenvalues ────────────── QR iteration uses Householder QR
  ├── §02 SVD ────────────────────── thin QR as step in SVD algorithm
  ├── §05 Orthogonality ──────────── QR theory (Gram-Schmidt)
  ├── §07 Positive Definite ──────── Cholesky THEORY (full treatment)
  └── §08 Matrix Decompositions ◄── YOU ARE HERE
       ├── LU (canonical home)      PA = LU with partial pivoting
       ├── QR algorithms            Householder, Givens, RRQR, TSQR
       └── Cholesky (computational) Blocked dpotrf, LDLᵀ for indef.

  Chapter 4: Calculus Fundamentals
  ├── §04 Jacobian ───────────────── Gauss-Newton needs QR
  └── §05 Hessian ────────────────── Cholesky at local minima

  Chapter 8: Optimization
  ├── Newton's method ────────────── LU/Cholesky for Hessian systems
  ├── K-FAC ──────────────────────── Kronecker-Cholesky per layer
  └── Interior point methods ─────── Cholesky at each Newton step

  ML Applications (load-bearing)
  ├── GP regression ──────────────── Cholesky (O(n³) bottleneck)
  ├── K-FAC / Shampoo ────────────── blocked Cholesky per layer
  ├── LoRA / GaLore ──────────────── randomized QR for rank selection
  ├── Linear probing ─────────────── Householder QR least squares
  └── Bayesian optimization ──────── GP + Cholesky every step

════════════════════════════════════════════════════════════════════════
```

The three factorizations of this section — LU, QR, and Cholesky — are not abstract mathematical objects. They are the algorithms that run inside every call to `numpy.linalg.solve`, `scipy.linalg.qr`, `torch.linalg.cholesky`, and LAPACK's `dgesv`, `dgeqrf`, `dpotrf`. Understanding them at this level — theorem, algorithm, numerical stability, and AI application — completes the computational linear algebra foundation needed for everything that follows.

[← Back to Advanced Linear Algebra](../README.md) | [← Positive Definite Matrices](../07-Positive-Definite-Matrices/notes.md) | [Next: Calculus Fundamentals →](../../04-Calculus-Fundamentals/README.md)

---

## Appendix A: Complete Algorithms and Pseudocode

### A.1 LU with Partial Pivoting — Full Pseudocode

The following pseudocode gives the complete in-place LU factorization with partial pivoting. The matrix $A$ is overwritten: the strict lower triangle stores the multipliers $m_{ij}$ (the subdiagonal entries of $L$, since $L_{ii} = 1$ is implicit), and the upper triangle plus diagonal stores $U$.

```
Algorithm LU_PARTIAL_PIVOT(A):
    Input:  A ∈ ℝ^{n×n}
    Output: Modified A (lower: L multipliers, upper+diag: U), pivot vector piv

    piv = [1, 2, ..., n]   # permutation record

    for k = 1 to n-1:
        # Step 1: Find pivot in column k (rows k to n)
        max_val = |A[k,k]|
        max_row = k
        for i = k+1 to n:
            if |A[i,k]| > max_val:
                max_val = |A[i,k]|
                max_row = i

        # Step 2: Swap rows k and max_row
        if max_row ≠ k:
            swap A[k, :] ↔ A[max_row, :]
            swap piv[k] ↔ piv[max_row]

        # Step 3: Check for (near-)singular pivot
        if |A[k,k]| < ε:
            warn("Near-singular pivot at step k")

        # Step 4: Compute multipliers and update
        for i = k+1 to n:
            A[i,k] = A[i,k] / A[k,k]          # multiplier m_{ik}
            for j = k+1 to n:
                A[i,j] -= A[i,k] * A[k,j]     # rank-1 update

    return A, piv

Solve Ax = b using LU_PARTIAL_PIVOT:
    Compute A, piv = LU_PARTIAL_PIVOT(A)
    Apply permutation: b = b[piv]
    Forward solve Lx = b (L has unit diagonal, ignore A[k,k]):
        for i = 1 to n:
            for j = 1 to i-1:
                b[i] -= A[i,j] * b[j]
    Backward solve Ux = b:
        for i = n downto 1:
            for j = i+1 to n:
                b[i] -= A[i,j] * b[j]
            b[i] /= A[i,i]
    return b
```

**Implementation notes:**
- The inner loop `for j = k+1 to n: A[i,j] -= A[i,k] * A[k,j]` is a **SAXPY** operation (Scalar A times X Plus Y) — BLAS level 1, vectorizable.
- The full step 4 over all $i$ simultaneously is a **rank-1 update** $A[k+1:n, k+1:n] -= \mathbf{m} \mathbf{u}^\top$ — BLAS level 2 `DGER`.
- Blocked version batches $b$ steps into a panel and uses **DGEMM** for the trailing update.

### A.2 Householder QR — Full Pseudocode

```
Algorithm HOUSEHOLDER_QR(A):
    Input:  A ∈ ℝ^{m×n}, m ≥ n
    Output: Modified A (upper triangle: R; lower triangle: Householder vectors),
            beta vector (scaling factors)

    beta = zeros(n)

    for k = 1 to n:
        # Extract column k from row k to m
        x = A[k:m, k]                        # length (m-k+1)

        # Compute Householder vector
        sigma = norm(x)
        if sigma == 0: continue               # column already zero

        # Sign convention: ensure v[0] and x[0] have opposite signs
        alpha = -sign(x[0]) * sigma           # target value
        v = copy(x)
        v[0] -= alpha
        beta_k = 2 / (v^T v)                  # = 2 / ||v||^2

        # Apply H_k to trailing submatrix A[k:m, k:n]
        # H_k A = A - (beta_k * v)(v^T A) = A - beta_k * v * (v^T A)
        w = beta_k * (v^T @ A[k:m, k:n])     # row vector, length (n-k+1)
        A[k:m, k:n] -= outer(v, w)

        # Store Householder vector in lower triangle
        A[k+1:m, k] = v[1:]                  # store v[1:] (v[0] = 1 implicit)
        beta[k] = beta_k

        # A[k,k] now equals alpha = -sign(x[0]) * norm(x)

    return A, beta

Recover Q (thin, first n columns):
    Q = I_{m×n}
    for k = n downto 1:
        # Reconstruct v from stored lower triangle
        v = [1; A[k+1:m, k]]                 # prepend implicit 1
        beta_k = beta[k]
        # Apply H_k to Q[k:m, k:n]
        w = beta_k * (v^T @ Q[k:m, k:n])
        Q[k:m, k:n] -= outer(v, w)
    return Q
```

### A.3 Givens QR for Banded Matrices

For a matrix $A$ with upper and lower bandwidth $k_u$ and $k_\ell$ respectively:

```
Algorithm GIVENS_QR_BANDED(A, ku, kl):
    for j = 1 to n:              # column
        for i = j+kl downto j+1: # eliminate subdiagonal entries in bandwidth
            # Compute Givens rotation to zero A[i,j] using A[i-1,j]
            c, s, r = compute_givens(A[i-1,j], A[i,j])
            A[i,j] = 0
            A[i-1,j] = r
            # Apply G to remaining columns in bandwidth
            for jj = j+1 to min(j+ku+kl, n):
                temp = c*A[i-1,jj] + s*A[i,jj]
                A[i,jj] = -s*A[i-1,jj] + c*A[i,jj]
                A[i-1,jj] = temp
    return R (upper banded), Q (product of Givens rotations)
```

**Complexity:** $O(nk_\ell(k_\ell + k_u))$ — linear in $n$ for fixed bandwidth. This is the algorithm used in tridiagonal eigensolvers.

---

## Appendix B: Deeper Theoretical Results

### B.1 The Fundamental Theorem of Existence for LU

**Theorem (Existence of LU, precise version).** Let $A \in \mathbb{R}^{n \times n}$. The following are equivalent:
1. $A$ has an LU factorization with $L$ unit lower triangular.
2. All leading principal minors $\det(A_k) \neq 0$ for $k = 1, \ldots, n-1$.
3. Gaussian elimination completes without encountering a zero pivot.

*Proof of (1) ⟺ (2):* We use induction. Base case $n = 1$: trivial. Inductive step: write $A = \begin{pmatrix} A_{n-1} & \mathbf{b} \\ \mathbf{c}^\top & d \end{pmatrix}$. If $A_{n-1}$ has an LU factorization $A_{n-1} = L_{n-1}U_{n-1}$ (by induction, iff all leading minors of $A_{n-1}$ are nonzero), then:

$$A = \begin{pmatrix} L_{n-1} & \mathbf{0} \\ \mathbf{c}^\top U_{n-1}^{-1} & 1 \end{pmatrix} \begin{pmatrix} U_{n-1} & L_{n-1}^{-1}\mathbf{b} \\ \mathbf{0}^\top & s \end{pmatrix}$$

where $s = d - \mathbf{c}^\top A_{n-1}^{-1}\mathbf{b}$ is the **Schur complement** (see §07). This gives a valid LU factorization with all blocks well-defined. $\square$

**Corollary:** The Schur complement $s = d - \mathbf{c}^\top A_{n-1}^{-1}\mathbf{b}$ is the last pivot. It equals $\det(A)/\det(A_{n-1})$. This shows that the pivot sequence in Gaussian elimination directly encodes the leading minors via their ratios.

### B.2 The Backward Error Theorem (Precise)

**Theorem (Wilkinson, 1963).** Let $\hat{L}, \hat{U}$ be the matrices computed by Gaussian elimination with partial pivoting, applied to $PA$ (after row interchanges). Then there exists a matrix $E$ such that:

$$PA + E = \hat{L}\hat{U}$$

$$|E_{ij}| \leq n \cdot u \cdot g_n \cdot \max_{i,j}|A_{ij}| \quad \text{for all } i,j$$

where $u$ is unit roundoff and $g_n = \max_{i,j,k}|A_{ij}^{(k)}|/\max_{i,j}|A_{ij}|$ is the **growth factor**.

**Corollary (Solution accuracy).** The computed solution $\hat{\mathbf{x}}$ to $A\mathbf{x} = \mathbf{b}$ satisfies:
$$(A + \delta A)\hat{\mathbf{x}} = \mathbf{b}, \quad \|\delta A\|_\infty \leq c(n) \cdot u \cdot g_n \cdot \|A\|_\infty$$

The forward error bound then follows from the perturbation lemma:
$$\frac{\|\hat{\mathbf{x}} - \mathbf{x}^*\|_\infty}{\|\mathbf{x}^*\|_\infty} \leq \kappa_\infty(A) \cdot \frac{c(n) u g_n \|A\|_\infty}{\|A\|_\infty} + O(u^2)$$

The fundamental lesson: **error = condition number × backward error**. The algorithm controls the backward error; the problem controls the condition number.

### B.3 Optimality of Householder QR

**Theorem.** Among all algorithms for computing QR that use orthogonal similarity transformations, Householder QR minimizes the number of arithmetic operations asymptotically.

*Sketch:* Each Householder reflector zeros $m-k$ entries in column $k$ using $2(m-k)$ flops (inner product + SAXPY). Any other elementary orthogonal transformation (e.g., a sequence of Givens rotations) requires at least as many flops per zero created. Householder maximizes the work per transformation.

**Why Givens is sometimes preferred:** Despite higher total flop count, Givens rotations are preferred for:
- Sparse matrices (avoid creating fill-in in the reflection direction)
- Parallel computation (rotations on disjoint row pairs can be parallelized)
- Streaming data (new rows can be incorporated one at a time)

### B.4 Perturbation Theory for Triangular Factorizations

**Theorem (Stewart, 1977).** If $A = LU$ (LU without pivoting, assuming it exists), then:
$$\frac{\|L + \delta L\|_F}{\|L\|_F}, \frac{\|U + \delta U\|_F}{\|U\|_F} = O\left(\kappa(L)\kappa(U) \cdot \frac{\|\delta A\|_F}{\|A\|_F}\right)$$

The condition numbers $\kappa(L)$ and $\kappa(U)$ can independently be much larger than $\kappa(A)$, explaining why small perturbations to $A$ can cause large perturbations to $L$ and $U$ separately.

**For QR:** The Householder factors $Q_k$ have $\kappa(Q_k) = 1$ exactly (orthogonal matrices). Therefore:
$$\frac{\|\delta R\|_F}{\|R\|_F} = O\left(\kappa(R) \cdot \frac{\|\delta A\|_F}{\|A\|_F}\right) = O\left(\kappa(A) \cdot \frac{\|\delta A\|_F}{\|A\|_F}\right)$$
The QR factors are perturbed proportionally to $\kappa(A)$, not $\kappa(A)^2$ as in the normal equations.

---

## Appendix C: Connections to Other Decompositions

### C.1 LU and the Spectral Decomposition

The LU factorization of $A$ is not directly related to the eigendecomposition $A = Q\Lambda Q^{-1}$, but the two share a common ancestor: **block triangularization**. The Schur decomposition $A = UTU^*$ (with $T$ upper triangular, $U$ unitary) is a complex-field generalization where $T$ is upper triangular with eigenvalues on the diagonal — the "LU" of the spectral world.

The **QR algorithm** for eigenvalues alternates QR factorizations and similarity transformations to drive $A$ toward upper triangular (Schur) form. Each QR step is one Householder QR followed by one matrix-matrix product. After convergence, the diagonal of $T$ gives the eigenvalues. This is the most important use of Householder QR in all of numerical linear algebra.

### C.2 Cholesky and the SVD

For a symmetric positive definite matrix, the Cholesky factorization $A = LL^\top$ and the SVD $A = U\Sigma U^\top$ (symmetric PD, so $U$ has orthonormal columns and $\Sigma = \Lambda$ has positive diagonal) are related by:

$$L = U\Lambda^{1/2} R^\top$$

where $R$ is the upper triangular factor of a QR decomposition of $U$. Explicitly: $L = QR$ where $Q = U$ and $R = \Lambda^{1/2} V^\top$ for some orthogonal $V$ — but this is not the standard Cholesky. The key point is that **Cholesky is a non-orthogonal factorization** while SVD is orthogonal; they capture different aspects of the same structure.

**Practical consequence:** The Cholesky factor $L$ is not related to the eigenvectors of $A$ in a simple way. Computing the matrix square root $A^{1/2}$ (which requires eigenvectors) is more expensive than Cholesky.

### C.3 QR and the Gram-Schmidt-Cholesky Identity

The Gram-Schmidt process applied to the columns of $A$ produces the thin QR $A = QR$. But the normal equations give $(A^\top A) = R^\top R$ — a Cholesky factorization of the Gram matrix! This identity $A^\top A = R^\top R$ explains why:
- QR of $A$ and Cholesky of $A^\top A$ produce the same $R$
- Computing QR of $A$ is equivalent to computing Cholesky of $A^\top A$ in exact arithmetic
- In floating-point, QR is more stable (works with $\kappa(A)$) while Cholesky of $A^\top A$ works with $\kappa(A)^2$

### C.4 CUR and Interpolative Decompositions

Beyond LU, QR, and Cholesky, recent methods have developed **structure-revealing factorizations** that select actual columns/rows of $A$:

**CUR decomposition:** $A \approx CUR$ where $C$ contains a subset of columns of $A$, $R$ contains a subset of rows, and $U$ is a small "bridge" matrix. Unlike LU/QR, $C$ and $R$ are actual data columns — interpretable and memory-efficient.

**Interpolative decomposition (ID):** $A \approx A_{:,J} B$ where $J$ is a subset of column indices and $B$ is a well-conditioned matrix. The ID is computed via column-pivoted QR: the pivot columns from RRQR form $J$.

**For AI:** CUR and ID are used in interpretability research (identifying the "canonical" data points or features that the model attends to), in structured pruning (removing weight columns corresponding to small pivots), and in efficient attention (selecting key "skeleton" positions for linear-complexity attention).

---

## Appendix D: Implementation Reference

### D.1 Python/NumPy/SciPy API

```python
import numpy as np
import scipy.linalg as la

# === LU Factorization ===
# scipy.linalg.lu (returns P, L, U as separate matrices)
P, L, U = la.lu(A)
# verify: A ≈ P @ L @ U

# scipy.linalg.lu_factor (returns compact (LU, piv) for solving)
lu, piv = la.lu_factor(A)
x = la.lu_solve((lu, piv), b)     # solves Ax = b

# === QR Factorization ===
# Full QR (Q is m×m)
Q, R = la.qr(A)                   # default: full
# Thin/economy QR (Q is m×n for m×n matrix A)
Q, R = la.qr(A, mode='economic')

# Column-pivoted QR (rank-revealing)
Q, R, P = la.qr(A, pivoting=True)
# verify: A[:, P] ≈ Q @ R  (P is permutation array)
rank_est = np.sum(np.abs(np.diag(R)) > 1e-10 * np.abs(R[0,0]))

# Apply Q^T without forming Q
# (use the raw tau output for LAPACK-level efficiency)

# === Cholesky ===
L = la.cholesky(A, lower=True)    # A = L L^T
x = la.cho_solve((L, True), b)    # solves Ax = b

# === Triangular solves ===
x = la.solve_triangular(L, b, lower=True)   # solves Lx = b
x = la.solve_triangular(U, b, lower=False)  # solves Ux = b
```

### D.2 PyTorch API

```python
import torch

A = torch.randn(n, n, dtype=torch.float64)
A = A @ A.T + torch.eye(n) * 0.1  # make SPD

# LU
LU, pivots, info = torch.linalg.lu_factor(A)
x = torch.linalg.lu_solve(LU, pivots, b)

# QR
Q, R = torch.linalg.qr(A, mode='reduced')  # thin QR

# Cholesky
L = torch.linalg.cholesky(A)
x = torch.linalg.cholesky_solve(b.unsqueeze(-1), L).squeeze(-1)

# All support autograd:
A.requires_grad_(True)
L = torch.linalg.cholesky(A)
loss = torch.logdet(A)
loss.backward()   # computes grad via implicit differentiation
```

### D.3 JAX API

```python
import jax.numpy as jnp
import jax.scipy.linalg as jla

# LU (no grad through lu_factor yet in standard JAX)
lu, piv, permutation = jla.lu(A)

# QR
Q, R = jnp.linalg.qr(A, mode='reduced')

# Cholesky (supports JIT, vmap, grad)
L = jnp.linalg.cholesky(A)
# Solve via triangular
y = jla.solve_triangular(L, b, lower=True)
x = jla.solve_triangular(L.T, y, lower=False)

# Differentiating log-det through Cholesky:
def log_det_spd(A):
    L = jnp.linalg.cholesky(A)
    return 2 * jnp.sum(jnp.log(jnp.diag(L)))

grad_A = jax.grad(log_det_spd)(A)  # should equal A^{-1}
```

### D.4 Performance Benchmarks (n = 1000, FP64)

| Operation | NumPy (CPU) | PyTorch (CPU) | PyTorch (A100 GPU) |
|---|---|---|---|
| LU factorization | 85 ms | 78 ms | 3 ms |
| QR factorization | 210 ms | 195 ms | 8 ms |
| Cholesky | 35 ms | 32 ms | 1.5 ms |
| Triangular solve | 12 ms | 10 ms | 0.5 ms |
| Matrix multiply (DGEMM) | 40 ms | 38 ms | 0.8 ms |

*Approximate values on typical hardware (2024). GPU speedups increase for larger $n$ as arithmetic intensity grows.*

---

## Appendix E: Error Accumulation in Practice

### E.1 Condition Number Estimation

Computing the exact condition number $\kappa(A) = \|A\|_2 \|A^{-1}\|_2 = \sigma_1/\sigma_n$ requires computing all singular values — $O(n^3)$ work. LAPACK provides **cheap condition number estimators** via **inverse iteration**:

**LAPACK `DGECON`:** Given the LU factorization of $A$ (already computed for solving), estimates $\kappa_1(A) = \|A\|_1 \|A^{-1}\|_1$ in $O(n^2)$ via a sequence of triangular solves. The estimate is typically within a factor of 10 of the true condition number.

**Rule of thumb:** If `DGECON` returns $\kappa \approx 10^k$, you lose approximately $k$ digits of precision. For FP64 ($u \approx 10^{-16}$), you have $16-k$ correct digits. For FP32 ($u \approx 10^{-7}$), you have $7-k$ correct digits.

**For AI:** PyTorch's `torch.linalg.cond` computes the exact condition number via SVD. For quick checks without full SVD: `torch.linalg.matrix_norm(A) * torch.linalg.matrix_norm(torch.linalg.inv(A))`.

### E.2 Diagnosis of Numerical Failures

| Symptom | Likely cause | Diagnosis | Fix |
|---|---|---|---|
| `LinAlgError: Singular matrix` | Zero pivot encountered | $A$ is singular or nearly so | Check `np.linalg.matrix_rank(A)` |
| `nan` in output | Zero or tiny pivot in LU | No pivoting, tiny diagonal | Use partial pivoting |
| Solution with large residual $\|A\hat{x}-b\| \gg u\|A\|\|\hat{x}\|$ | High condition number | Compute $\kappa(A)$ | Regularize or use iterative refinement |
| `LinAlgError` from `cholesky` | Matrix not SPD | Eigenvalues not all positive | Check `np.linalg.eigvalsh(A).min()` |
| Q not orthogonal: $\|Q^TQ - I\| \gg u$ | Classical Gram-Schmidt used | Compute $\|Q^TQ-I\|_F$ | Switch to Householder QR |
| Slow factorization | Block size mismatch | Profile with BLAS calls | Tune block size; use BLAS-3 routine |

### E.3 Iterative Refinement Implementation

```python
def iterative_refinement(A, b, max_iter=3, tol=1e-14):
    """
    Solve Ax = b with iterative refinement.
    Factor in FP64, residuals in FP64, corrections in FP32.
    """
    import scipy.linalg as la
    
    # Factor in FP32 (simulating low-precision)
    A32 = A.astype(np.float32)
    lu, piv = la.lu_factor(A32.astype(np.float64))  # use FP64 here for demo
    
    # Initial solve
    x = la.lu_solve((lu, piv), b)
    
    for k in range(max_iter):
        # Residual in FP64 (high precision)
        r = b - A @ x          # FP64 residual
        r_norm = np.linalg.norm(r) / np.linalg.norm(b)
        print(f"  Iter {k}: relative residual = {r_norm:.3e}")
        
        if r_norm < tol:
            break
        
        # Correction solve (in FP32 precision)
        d = la.lu_solve((lu, piv), r)
        x = x + d
    
    return x
```

---

## Appendix F: The Geometry of Factorizations

### F.1 LU as a Basis Change

The LU factorization $PA = LU$ can be interpreted geometrically. The unit lower triangular matrix $L$ represents a **shearing transformation** — it maps the standard basis $\{\mathbf{e}_i\}$ to the columns of $L$, which form a "lower triangular" basis. The upper triangular $U$ then represents the coordinates of the rows of $A$ in this new basis.

Concretely: each step of Gaussian elimination adds multiples of row $k$ to lower rows, which is a shear in the "row space" of $A$. The cumulative effect is a change to a triangular basis — the essence of LU.

### F.2 QR as a Rotation to Triangular Form

The Householder QR factorization $A = QR$ is geometrically a **rotation of the column space of $A$ to align with the coordinate axes**. Each Householder reflector $H_k$ reflects the $k$-th "residual" column to align with $\mathbf{e}_k$, zeroing out all entries below the diagonal.

After $n$ reflections, the matrix $Q = H_1 H_2 \cdots H_n$ is a composition of orthogonal reflections — itself orthogonal — that has rotated $A$ to upper triangular form $R$.

**Geometric insight:** The columns of $Q$ form an orthonormal basis for the column space of $A$. The entries of $R$ give the coordinates of the original columns of $A$ in this orthonormal basis. This is the Gram-Schmidt process, done via reflections rather than projections.

### F.3 Cholesky as the Matrix Square Root Factorization

For $A \succ 0$, the Cholesky factor $L$ satisfies $A = LL^\top$ — making $L$ a "matrix square root" of $A$ in a specific sense. Unlike the symmetric square root $A^{1/2} = Q\Lambda^{1/2}Q^\top$ (which is symmetric), the Cholesky factor $L$ is lower triangular.

**Geometric interpretation:** The linear map $T : \mathbb{R}^n \to \mathbb{R}^n$ defined by $T\mathbf{x} = L\mathbf{x}$ transforms the unit ball $\{\mathbf{x} : \|\mathbf{x}\| \leq 1\}$ into the ellipsoid $\{\mathbf{y} : \mathbf{y}^\top A^{-1} \mathbf{y} \leq 1\}$ — the level set of the quadratic form $q(\mathbf{y}) = \mathbf{y}^\top A^{-1}\mathbf{y}$. This is why Cholesky factorization is the foundation of sampling from $\mathcal{N}(\boldsymbol{\mu}, A)$: if $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$, then $L\mathbf{z} + \boldsymbol{\mu} \sim \mathcal{N}(\boldsymbol{\mu}, LL^\top) = \mathcal{N}(\boldsymbol{\mu}, A)$.

**Uniqueness:** The Cholesky factor $L$ with positive diagonal entries is unique for every $A \succ 0$. The symmetric square root $A^{1/2}$ is also unique and PSD, but it is NOT the same as $L$ (they differ by an orthogonal factor).

### F.4 Triangular Factorizations and the Flag Manifold

The space of all $n \times n$ invertible matrices $GL_n(\mathbb{R})$ acts on the set of factorizations as follows:
- **LU factorizations** correspond to the decomposition $GL_n = B_- \cdot B_+$ where $B_-$ (lower triangular) and $B_+$ (upper triangular) are Borel subgroups. The "generic" element of $GL_n$ has a unique LU (the open Bruhat cell).
- **QR factorizations** correspond to $GL_n = O(n) \cdot B_+$ where $O(n)$ is the orthogonal group. This decomposition is always unique (no exceptional cases, unlike LU).
- The **flag manifold** $Fl(n) = GL_n / B_+$ parameterizes complete flags $V_1 \subset V_2 \subset \cdots \subset V_n = \mathbb{R}^n$ and is the natural geometric setting for both LU and QR.

This connection to Lie group theory underlies the deep relationship between matrix factorizations and the geometry of symmetric spaces — a topic developed in Chapter 12 (Functional Analysis).

---

## Appendix G: Randomized Algorithms — Deeper Theory

### G.1 The Randomized SVD via QR

The **Halko-Martinsson-Tropp (2011)** algorithm computes a rank-$r$ approximation of $A \in \mathbb{R}^{m \times n}$ in $O(mnr)$ operations:

```
Algorithm RANDOMIZED_SVD(A, r, p):
    # Oversampling: target rank r + p (p ≈ 10)
    Ω = randn(n, r+p)              # Gaussian sketch matrix
    Y = A @ Ω                       # Sketch: Y ∈ ℝ^{m×(r+p)}
    Q, _ = qr(Y)                   # Orthonormalize: Q ∈ ℝ^{m×(r+p)}
    B = Q.T @ A                    # Project: B ∈ ℝ^{(r+p)×n}
    Û, Σ, Vt = svd(B, full_matrices=False)  # SVD of small matrix
    U = Q @ Û                      # Lift back to full space
    return U[:, :r], Σ[:r], Vt[:r, :]
```

**Error bound:** With probability $\geq 1 - \delta$:
$$\|A - U\Sigma V^\top\|_2 \leq \left(1 + \sqrt{\frac{r}{p-1}} + \frac{e\sqrt{r+p}}{p}\sqrt{\min(m,n) - r}\right)\sigma_{r+1}(A)$$

For $p = 10$: error is $\approx 2\sigma_{r+1}(A)$ with probability $> 99\%$.

**Power iteration improvement:** For matrices with slowly decaying singular values, applying $Y = (AA^\top)^q A\Omega$ (with $q = 1$ or $2$ power iterations) dramatically improves accuracy:
$$\|A - \hat{A}_r\|_2 \leq \left(\frac{\sigma_{r+1}(A)}{\sigma_r(A)}\right)^{2q}\!\!\cdot \text{(base error)}$$

**For AI — GaLore (Zhao et al., 2024):** GaLore uses randomized SVD to project gradient matrices $G_t \in \mathbb{R}^{m \times n}$ onto their principal $r$-dimensional subspace every $T = 200$ steps. The projection matrix $P_t \in \mathbb{R}^{n \times r}$ is computed via randomized SVD of the gradient, then Adam is applied to the projected gradients $G_t P_t \in \mathbb{R}^{m \times r}$. This reduces optimizer memory by $n/r$ while maintaining training quality.

### G.2 Structured Random Matrices

Instead of dense Gaussian $\Omega$, structured random matrices enable faster sketching:

| Matrix | Sketch cost | Randomness | Error bound |
|---|---|---|---|
| Dense Gaussian | $O(mnr)$ | Full i.i.d. | Optimal |
| SRFT (subsampled random Fourier) | $O(mn\log r)$ | Hadamard + sampling | Near-optimal |
| Sparse embedding | $O(nnz(A) \cdot r)$ | Hash functions | Near-optimal for sparse $A$ |
| CountSketch | $O(nnz(A))$ | Hash functions | Suboptimal but fast |

**SRFT (Johnson-Lindenstrauss):** $\Omega = \sqrt{n/r} \cdot DFS$ where $D$ is a random diagonal $\pm 1$ matrix, $F$ is the DFT, and $S$ samples $r$ rows uniformly. Applying $\Omega^\top$ costs $O(n\log n)$ via FFT.

---

## Appendix H: Software Ecosystem

### H.1 BLAS and LAPACK Hierarchy

```
User Code (Python/Julia/MATLAB)
    ↓
SciPy / NumPy / PyTorch / JAX (Python wrappers)
    ↓
LAPACK (high-level: LU, QR, Cholesky, eigensolvers)
    ↓
BLAS Level 3 (DGEMM, DSYRK, DTRSM — blocked, cache-optimal)
BLAS Level 2 (DGEMV, DGER — matrix-vector)
BLAS Level 1 (DDOT, DAXPY, DNRM2 — vector)
    ↓
Hardware-optimized BLAS (MKL, OpenBLAS, cuBLAS, BLIS)
    ↓
CPU/GPU Tensor Cores / AVX-512 / AMX tiles
```

**Key implementations:**
- **Intel MKL:** Optimized for Intel CPUs; 2-3x faster than reference BLAS for blocked operations
- **OpenBLAS:** Open source, near-MKL performance on x86; default in many Linux distributions
- **BLIS (BLAS-like Library Instantiation Software):** Framework for portable high-performance BLAS; used by many modern CPU vendors
- **cuBLAS:** NVIDIA GPU BLAS; powers all PyTorch/JAX GPU linear algebra
- **cuSOLVER:** NVIDIA GPU LAPACK; powers `torch.linalg` on NVIDIA hardware
- **rocBLAS / rocSOLVER:** AMD GPU equivalents

### H.2 Choosing the Right Routine

```
Need to solve Ax = b?
├── A is SPD? → DPOTRF (Cholesky) + DPOTRS
├── A is symmetric indefinite? → DSYTRF (LDLᵀ) + DSYTRS
├── A is general square? → DGESV (= DGETRF + DGETRS, partial pivot)
└── A is triangular? → DTRTRS (triangular solve directly)

Need to solve min ||Ax - b||?
├── A is full-rank, well-conditioned? → DGELS (QR-based least squares)
├── A may be rank-deficient? → DGELSY (column-pivoted QR)
└── Need minimum-norm solution? → DGELSD (divide-and-conquer SVD)

Need eigenvalues?
├── A is symmetric? → DSYEV (or DSYEVD for large n)
└── A is general? → DGEEV (QR iteration)

Need SVD?
├── Full SVD? → DGESVD or DGESDD (divide-and-conquer, faster)
└── Truncated SVD? → Use randomized SVD (not in LAPACK)
```

CHUNK_APPENDIX

---

## Appendix I: Practical Decision Guide for Practitioners

### I.1 Which Factorization to Use? — Decision Tree

The choice among LU, QR, and Cholesky depends on matrix structure, problem type, and numerical requirements. The following decision framework covers the vast majority of practical cases.

```
MATRIX FACTORIZATION SELECTION GUIDE
════════════════════════════════════════════════════════════════════════

 Is A symmetric?
 ├── YES: Is A positive definite? (all eigenvalues > 0)
 │   ├── YES: → CHOLESKY (dpotrf)
 │   │         Cost: n³/3 flops. Fastest, most stable for SPD.
 │   │         Use for: Gaussians, GP regression, K-FAC, Fisher
 │   └── NO:  → LDLᵀ (dsytrf, Bunch-Kaufman pivoting)
 │              Cost: n³/3 flops. Handles indefinite (saddle pts).
 │              Use for: indefinite Hessians, Newton at saddle pts
 └── NO:  Is A square?
     ├── YES: → LU with partial pivoting (dgetrf)
     │         Cost: 2n³/3 flops. General purpose.
     │         Use for: Solving Ax=b, computing det(A)
     └── NO:  Is A overdetermined (m > n)? → LEAST SQUARES
         ├── Well-conditioned & fast? → Normal equations + Cholesky
         │   Cost: mn² + n³/3. Squares condition number.
         ├── General / numerically safe? → Householder QR (dgelsy)
         │   Cost: 2mn² - 2n³/3. Standard choice.
         └── May be rank-deficient? → Column-pivoted QR or SVD
             RRQR: O(mn²), good rank estimate
             SVD:  O(mn²+n³), exact singular values

════════════════════════════════════════════════════════════════════════
```

### I.2 Numerical Precision Requirements

| Scenario | Precision needed | Recommended approach |
|---|---|---|
| Single FP32 forward pass | $u \approx 10^{-7}$ | Standard FP32 factorization |
| Training with FP16/BF16 | $u \approx 10^{-4}$ to $10^{-3}$ | Mixed precision + refinement |
| Scientific computing | $u \approx 10^{-16}$ (FP64) | FP64 throughout |
| Certified computation | Exact or interval | Symbolic or interval arithmetic |
| Ill-conditioned system ($\kappa > 10^{12}$) | All digits lost in FP64 | Regularize or use extended precision |

### I.3 Memory and Flop Budgets

For a matrix of size $n \times n$:

| Factorization | Memory (dense) | Flops | Triangular solve flops |
|---|---|---|---|
| LU | $n^2$ (in-place) | $\frac{2}{3}n^3$ | $2n^2$ |
| QR (Householder) | $mn$ + $n^2$ | $2mn^2 - \frac{2}{3}n^3$ | $2mn$ for $Q^\top b$, $n^2$ for backsolve |
| Cholesky | $\frac{n(n+1)}{2}$ | $\frac{1}{3}n^3$ | $n^2$ |
| LDLᵀ | $\frac{n(n+1)}{2}$ | $\frac{1}{3}n^3$ | $n^2$ |

**Memory for large $n$:**
- $n = 10{,}000$: LU requires $10^8$ doubles = 800 MB (fits in GPU HBM)
- $n = 100{,}000$: LU requires $10^{10}$ doubles = 80 GB (exceeds most GPUs)
- $n > 10^5$: must use iterative solvers (CG, GMRES) or sparse factorizations

### I.4 Parallelism Characteristics

| Factorization | CPU parallelism | GPU parallelism | Distributed |
|---|---|---|---|
| Blocked LU | Panel: serial; trailing: DGEMM (parallel) | cuSOLVER DGETRF; good for $n > 2000$ | ScaLAPACK PDGETRF |
| Householder QR | WY representation + DGEMM | cuSOLVER DGEQRF | ScaLAPACK PDGEQRF |
| Cholesky | Panel: serial; trailing: DSYRK (parallel) | cuSOLVER DPOTRF; excellent | ScaLAPACK PDPOTRF |
| Sparse LU | AMD reordering; supernodal parallel | cuSOLVER batch for many small systems | MUMPS, SuperLU_DIST |

**Batch factorizations (small systems):** For thousands of small systems (common in K-FAC, GP mini-batches), GPU batch factorizations (`cublasDgetrfBatched`) outperform sequential large-matrix factorizations by 10-100x.

### I.5 Common Library Calls and Their Underlying Algorithms

| High-level call | Underlying algorithm | Notes |
|---|---|---|
| `np.linalg.solve(A, b)` | LAPACK DGESV (= DGETRF + DGETRS) | Partial pivoting, FP64 |
| `np.linalg.lstsq(A, b)` | LAPACK DGELSD (divide-and-conquer SVD) | Rank-deficient safe |
| `np.linalg.qr(A)` | LAPACK DGEQRF (Householder) | Full or thin via mode |
| `scipy.linalg.lu(A)` | LAPACK DGETRF | Returns P, L, U separately |
| `scipy.linalg.cholesky(A)` | LAPACK DPOTRF | Raises if not SPD |
| `scipy.linalg.lu_solve((lu,piv), b)` | LAPACK DGETRS | Amortized over many b |
| `torch.linalg.solve(A, b)` | cuSOLVER DGESV (GPU) / MKL (CPU) | Autograd supported |
| `torch.linalg.cholesky(A)` | cuSOLVER DPOTRF (GPU) | Autograd via implicit diff |
| `jax.numpy.linalg.solve(A, b)` | XLA:HLO → cuSOLVER / LAPACK | JIT, vmap, grad |
| `sklearn.linear_model.Ridge(solver='cholesky')` | LAPACK DPOTRS | Normal equations + Cholesky |

---

## Appendix J: Extended Example — GP Hyperparameter Optimization

This appendix walks through a complete GP regression pipeline showing how LU, QR, and Cholesky all appear in different roles within the same ML workflow.

### J.1 Problem Setup

We have $n = 200$ observations $\{(x^{(i)}, y^{(i)})\}_{i=1}^n$ from $f(x) = \sin(3x)\exp(-x/5) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.01)$. We fit a GP with RBF kernel $k(x,x') = s_f^2\exp(-\|x-x'\|^2/(2\ell^2))$ and noise $\sigma_n^2$.

### J.2 Where Each Factorization Appears

**Cholesky — computing the GP posterior:**

The kernel matrix $K = K_{nn} + \sigma_n^2 I \succ 0$ is factored via Cholesky:
$$L = \operatorname{chol}(K), \quad \boldsymbol{\alpha} = L^{-\top}(L^{-1}\mathbf{y})$$
$$\log p(\mathbf{y}) = -\frac{1}{2}\mathbf{y}^\top\boldsymbol{\alpha} - \sum_i\log L_{ii} - \frac{n}{2}\log 2\pi$$

The log-determinant $\sum_i \log L_{ii}$ is computed from the Cholesky diagonal at zero additional cost.

**LU (implicit, via matrix-vector products) — CG solver for large $n$:**

For $n = 10^4$, direct Cholesky costs $10^{12}$ flops. Instead, conjugate gradient (CG) is used to solve $(K_{nn} + \sigma_n^2 I)\boldsymbol{\alpha} = \mathbf{y}$ iteratively. Each CG iteration requires one matrix-vector product $K_{nn}\mathbf{v}$, plus the application of a **preconditioner**.

The preconditioner is a low-rank + diagonal approximation $(U\Lambda U^\top + \sigma_n^2 I)^{-1}$, where $U\Lambda U^\top$ is a rank-$r$ approximation of $K_{nn}$ computed via randomized SVD (which uses QR internally). The preconditioned CG converges in $O(r)$ iterations instead of $O(\kappa(K))$.

**QR — selecting inducing points:**

For the sparse GP with $m \ll n$ inducing points, the inducing point locations are selected via column-pivoted QR on the feature matrix:
$$\Phi = \begin{pmatrix} \phi(x^{(1)})^\top \\ \vdots \\ \phi(x^{(n)})^\top \end{pmatrix} \in \mathbb{R}^{n \times r}$$
The pivot columns of `scipy.linalg.qr(Φ.T, pivoting=True)` identify the most "important" training points — those that span the feature space. These become the inducing points.

**Cholesky — computing the evidence lower bound (ELBO) for inducing point GP:**

The sparse GP ELBO requires computing the Cholesky of the $m \times m$ inducing point kernel matrix $K_{mm}$ plus a correction term — again $O(m^3)$ with $m \ll n$.

### J.3 Hyperparameter Optimization

The hyperparameters $\theta = (\ell, s_f, \sigma_n)$ are optimized by maximizing the log marginal likelihood $\log p(\mathbf{y} \mid X, \theta)$ via gradient-based optimization (L-BFGS or Adam). Each gradient evaluation requires:

1. A new Cholesky factorization of $K(\theta)$ — $O(n^3)$
2. Computing $\partial\log p / \partial\theta_k = \frac{1}{2}\operatorname{tr}\left[({\boldsymbol{\alpha}}\boldsymbol{\alpha}^\top - K^{-1})\frac{\partial K}{\partial\theta_k}\right]$
3. Each trace computation uses: $\operatorname{tr}(A B) = \sum_{i,j} A_{ij}B_{ji} = \operatorname{vec}(A)^\top\operatorname{vec}(B)$ — $O(n^2)$ per hyperparameter

Total cost per hyperparameter gradient: $O(n^3 + pn^2)$ for $p$ hyperparameters. For $n = 1000$, $p = 3$: about $3 \times 10^9$ flops — feasible in seconds on modern hardware.

This complete pipeline illustrates that a single ML workflow (GP regression with hyperparameter optimization) relies on Cholesky (3+ times, for different matrices), randomized QR (for inducing points), and CG with preconditioner (which involves implicit LU-like operations) — all in service of a Bayesian inference computation.

CHUNK_APPENDIXI

---

## Appendix K: Matrix Decompositions in the 2026 AI Landscape

### K.1 Inference-Time Decompositions

As language models move toward longer context (1M+ tokens), the attention computation's quadratic cost $O(n^2 d)$ has become a primary bottleneck. Several recent methods use matrix decompositions at inference time:

**PagedAttention (Kwon et al., 2023):** Manages KV cache memory using block-based paging inspired by virtual memory. The KV cache is a large matrix $K, V \in \mathbb{R}^{n \times d}$; PagedAttention tiles this into fixed-size "pages" and maintains an LU-like decomposition of the free-list.

**MLA (Multi-head Latent Attention, DeepSeek, 2024):** Compresses the KV cache via low-rank projection: $K = K_c U_{K}^\top$ where $K_c$ is a compressed latent and $U_K$ is a fixed projection matrix. The decomposition $K = QR$ (QR of the projection matrix) is used to initialize and orthogonalize the projection basis.

**Speculative decoding with draft models:** Uses LU-based factored attention matrices to maintain a draft distribution for speculative sampling.

### K.2 Training-Time Decompositions

**GaLore (Zhao et al., 2024):** Gradient Low-Rank Projection. For a weight matrix $W \in \mathbb{R}^{m \times n}$, the gradient $G_t$ is projected onto a rank-$r$ subspace via:
$$\tilde{G}_t = G_t P_t, \quad P_t \in \mathbb{R}^{n \times r}$$
where $P_t$ comes from the right singular vectors of $G_t$ (randomized SVD). Adam is applied to $\tilde{G}_t$ (memory: $mr$ instead of $mn$), then the update is lifted back: $\Delta W = \tilde{\Delta W}_t P_t^\top$. The SVD is recomputed every $T = 200$ steps.

**Fira (Chen et al., 2024):** Extends GaLore with a closed-form gradient approximation that avoids the SVD step entirely, using a QR-based approximation to the projection.

**SOAP (Vyas et al., 2024):** Combines Shampoo's Kronecker-structured preconditioner with Adam's update rule. Each layer's preconditioner is maintained as $G_l \in \mathbb{R}^{d_l \times d_l}$ with iterative Cholesky-based Newton steps for the matrix square root.

### K.3 Model Compression via Decompositions

**SliceGPT (Ashkboos et al., 2024):** Compresses transformer weights by applying QR-based slice operations: a structured pruning where the weight matrix is decomposed as $W = Q\tilde{W}$ and the orthogonal factor $Q$ is absorbed into the layer normalization.

**ASVD (Yuan et al., 2023):** Activation-aware SVD for LLM compression. Instead of SVD of $W$ directly, computes SVD of $WX$ (weight matrix multiplied by typical activation matrix), finding a better rank-$r$ approximation for the actual inference distribution. Uses randomized QR for efficiency.

**QuIP (Chee et al., 2023) and QuIP# (Tseng et al., 2024):** Incoherence-based quantization. A random orthogonal matrix $Q$ (Hadamard-based) is applied to $W$ before quantization: $\hat{W} = Q \cdot \text{round}(Q^\top W / s)$. The orthogonal preprocessing ensures that no single weight has disproportionate magnitude, enabling aggressive quantization. The QR connection: $Q$ is chosen as a randomized Hadamard transform, equivalent to the sketch matrix in randomized QR.

### K.4 Numerical Linear Algebra for AI Accelerators

**Tensor Cores and Mixed Precision:** NVIDIA A100/H100 Tensor Cores perform $D = AB + C$ where $A, B$ are FP16/BF16 and $C, D$ are FP32, achieving $312$ TFLOPS (FP16) vs $19.5$ TFLOPS (FP64). All modern factorization libraries exploit this:
- **cuSOLVER RF** (Iterative Refinement): FP16 factor + FP32 residual + FP32 correction — full FP32 accuracy at near-FP16 throughput
- **cuBLAS Tensor Core GEMM**: All blocked factorizations (LU, QR, Cholesky trailing updates) use TC-GEMM for the dominant computation

**Intel AMX (Advanced Matrix Extensions):** New Intel architecture extensions (Sapphire Rapids, 2023) add hardware matrix tile registers and tile-based GEMM instructions. BLIS and MKL exploit AMX for $2\times$ speedup over AVX-512 on blocked factorizations.

**AMD MI300X:** 192 GB HBM3 + 5.3 TB/s bandwidth enables $n = 65{,}000$ Cholesky factorization in GPU memory — sufficient for large GP regression without blocking across devices.

---

## Appendix L: Summary of Key Results

For quick reference, here are the key theorems and facts from this section:

**LU Factorization:**
- Exists iff all leading principal minors $\det(A_k) \neq 0$ for $k = 1, \ldots, n-1$.
- With partial pivoting: $PA = LU$; $|L_{ij}| \leq 1$; cost $\frac{2}{3}n^3$ flops.
- Backward error: $\|E\|_\infty \leq c_n u \rho_n \|A\|_\infty$ where $\rho_n$ is the growth factor.
- Growth factor: $\rho_n \leq 2^{n-1}$ (theoretical), $\approx n^{2/3}$ (typical).
- Determinant: $\det(A) = (-1)^s \prod_i U_{ii}$ where $s$ = number of row swaps.

**QR Factorization:**
- Always exists for any $A \in \mathbb{R}^{m \times n}$ with $m \geq n$.
- Householder QR: backward stable with $\|\hat{Q}^\top\hat{Q} - I\|_F = O(u)$, independent of $\kappa(A)$.
- Cost: $2mn^2 - \frac{2}{3}n^3$ flops (vs $\frac{2}{3}n^3$ for LU when $m = n$).
- QR for least squares: $\kappa$ loss $= \kappa(A)$ (vs $\kappa(A)^2$ for normal equations).
- RRQR: diagonal of $R$ estimates singular values; $|R_{kk}|$ reveals rank.

**Cholesky Factorization:**
- Exists iff $A \succ 0$ (equivalently: all eigenvalues positive).
- Cost: $\frac{1}{3}n^3$ flops (half of LU); unconditionally backward stable without pivoting.
- Log-determinant: $\log\det A = 2\sum_i\log L_{ii}$ — exact, $O(n)$ after Cholesky.
- Gradient: $\nabla_A\log\det A = A^{-1}$ — computed in $O(n^2)$ via Cholesky solve.

**Backward Stability:**
- Forward error $\leq \kappa(A) \times$ backward error (fundamental inequality).
- Backward stability hierarchy: Cholesky = Householder QR $\succ$ LU with partial pivot $\succ$ LU without pivot.
- Iterative refinement: recovers FP64 accuracy from FP16 factorization in $O(1)$ iterations if $\kappa \cdot u_{16} < 1$.

CHUNK_FINAL

---

## Appendix M: Exercises — Further Problems

### M.1 Theoretical Extensions

**Problem M.1.** Show that for any invertible $A \in \mathbb{R}^{n \times n}$, there exists a permutation matrix $P$ such that $PA$ has an LU factorization. (*Hint:* show that there always exists a row ordering such that all leading principal minors of $PA$ are nonzero.)

**Problem M.2.** Prove the identity $\det(A) = \det(L)\det(U) = \prod_{i=1}^n U_{ii}$ using the LU factorization. Extend to show that $\det(PA) = (-1)^s \det(A)$ where $s$ is the number of transpositions in $P$.

**Problem M.3.** For $A \in \mathbb{R}^{n \times n}$ with Cholesky $A = LL^\top$, prove that the Schur complement of $A_{11}$ in $A$ equals $(L_{22}L_{22}^\top)$ where $L_{22}$ is the lower-right $(n-1) \times (n-1)$ block of $L$. Conclude that the Schur complement of any SPD matrix is SPD.

**Problem M.4.** Show that the Householder reflector $H = I - 2\mathbf{v}\mathbf{v}^\top/\|\mathbf{v}\|^2$ satisfies:
(a) $H = H^\top$ (symmetric)
(b) $H^\top H = I$ (orthogonal)
(c) $\det(H) = -1$ (orientation-reversing)
(d) $H^2 = I$ (involutory — its own inverse)

**Problem M.5.** Prove that QR with column pivoting finds the best rank-$r$ approximation in the following sense: if $\operatorname{rank}(A) = r$ and $P$ is chosen so that $|R_{11}| \geq |R_{22}| \geq \cdots$, then the leading $r \times r$ block $R_{11}$ of $R$ satisfies $\sigma_r(R_{11}) \geq \sigma_r(A) / \sqrt{1 + r(n-r)}$.

### M.2 Computational Challenges

**Challenge M.6 (Batched Cholesky).** Implement batched Cholesky factorization for $k = 1000$ matrices of size $n = 50 \times 50$. Compare three implementations:
- Python loop over `scipy.linalg.cholesky`
- NumPy vectorized via `np.vectorize`
- `torch.linalg.cholesky` on CPU with batch dimension

Measure time and throughput (matrices/second). Extrapolate to the K-FAC setting where $k = 10{,}000$ and $n = 128$.

**Challenge M.7 (Iterative Refinement).** Construct a matrix $A$ with condition number $\kappa \approx 10^{12}$ (near the FP64 limit). Solve $A\mathbf{x} = \mathbf{b}$ using:
(a) Direct LU (FP64): measure relative error
(b) Simulated FP32 LU + FP64 residual refinement (3 steps): measure relative error
Compare and explain the improvement using backward error theory.

**Challenge M.8 (Rank-revealing QR in practice).** Download a real dataset (e.g., the UCI Breast Cancer dataset, $n = 569$, $d = 30$ features). Apply RRQR to the feature matrix $X$ and identify the effective rank (threshold: $|R_{kk}|/|R_{11}| < 10^{-3}$). Verify by comparing to the number of significant singular values of $X$.

---

## References

1. Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations* (4th ed.). Johns Hopkins University Press. — The definitive reference for all material in this section.

2. Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. SIAM. — Exceptionally clear treatment of Householder QR and backward error analysis.

3. Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM. — Comprehensive analysis of numerical stability; source for all error bounds cited here.

4. Demmel, J. W. (1997). *Applied Numerical Linear Algebra*. SIAM. — Excellent treatment of LU pivoting strategies and condition estimation.

5. Householder, A. S. (1958). Unitary triangularization of a nonsymmetric matrix. *Journal of the ACM*, 5(4), 339–342. — Original Householder reflector paper.

6. Wilkinson, J. H. (1961). Error analysis of direct methods of matrix inversion. *Journal of the ACM*, 8(3), 281–330. — Foundational backward error analysis for LU.

7. Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review*, 53(2), 217–288. — Randomized QR/SVD framework.

8. Martens, J. & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored approximate curvature. *ICML 2015*. — K-FAC: Cholesky for Fisher information blocks.

9. Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*. — Low-rank decomposition for fine-tuning.

10. Zhao, J. et al. (2024). GaLore: Memory-efficient LLM training by gradient low-rank projection. *ICML 2024*. — Randomized SVD/QR for gradient compression.

11. Bunch, J. R. & Kaufman, L. (1977). Some stable methods for calculating inertia and solving symmetric linear systems. *Mathematics of Computation*, 31(137), 163–179. — LDLᵀ with Bunch-Kaufman pivoting.

12. Anderson, E. et al. (1999). *LAPACK Users' Guide* (3rd ed.). SIAM. — LAPACK routine reference and algorithmic details.

13. Gardner, J. R. et al. (2018). GPyTorch: Blackbox matrix-matrix Gaussian process inference with GPU acceleration. *NeurIPS 2018*. — CG-based GP inference for large-scale regression.

14. Demmel, J., Grigori, L., Hoemmen, M., & Langou, J. (2008). Communication-optimal parallel and sequential QR and LU factorizations. *SIAM Journal on Scientific Computing*, 34(1), A206–A239. — TSQR and communication-avoiding algorithms.

15. Shah, J. et al. (2024). FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. *arXiv:2407.08608*. — Block-tiled attention exploiting BLAS-3 structure.


---

## Appendix M: Exercises — Further Problems

### M.1 Theoretical Extensions

**Problem M.1.** Show that for any invertible A there exists a permutation P such that PA has an LU factorization. (Hint: show that there always exists a row ordering such that all leading principal minors of PA are nonzero.)

**Problem M.2.** Prove the identity det(A) = prod_i U_ii using the LU factorization. Extend to show that det(PA) = (-1)^s det(A) where s is the number of transpositions in P.

**Problem M.3.** For A with Cholesky A = LL^T, prove that the Schur complement of A_11 in A equals L_22 L_22^T. Conclude that the Schur complement of any SPD matrix is SPD.

**Problem M.4.** Show that the Householder reflector H = I - 2vv^T/||v||^2 is symmetric, orthogonal, has det(H) = -1, and H^2 = I.

### M.2 References

1. Golub, G. H. and Van Loan, C. F. (2013). Matrix Computations (4th ed.). Johns Hopkins University Press.
2. Trefethen, L. N. and Bau, D. (1997). Numerical Linear Algebra. SIAM.
3. Higham, N. J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.
4. Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011). Finding structure with randomness. SIAM Review, 53(2), 217-288.
5. Martens, J. and Grosse, R. (2015). Optimizing neural networks with K-FAC. ICML 2015.
6. Hu, E. J. et al. (2022). LoRA: Low-rank adaptation of large language models. ICLR 2022.
7. Zhao, J. et al. (2024). GaLore: Memory-efficient LLM training by gradient low-rank projection. ICML 2024.
8. Demmel, J. et al. (2008). Communication-optimal parallel and sequential QR and LU factorizations. SIAM JSC, 34(1).
