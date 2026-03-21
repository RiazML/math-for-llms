[← Previous Chapter: Linear Algebra Basics](../02-Linear-Algebra-Basics/README.md) | [Next Chapter: Calculus Fundamentals →](../04-Calculus-Fundamentals/README.md)

---

# Chapter 3 — Advanced Linear Algebra

> _"The eigenvalues of the weight matrices, the singular values of the attention projections, the Cholesky factor of the Fisher information — modern AI is, at its core, computational linear algebra."_

## Overview

This chapter builds on the foundations of Chapter 2 to develop the deeper algebraic structure that drives modern machine learning. The progression moves from spectral theory (eigenvalues, SVD) through geometric structure (PCA, orthogonality) through analytic tools (norms, positive definiteness) to computational algorithms (LU, QR, Cholesky).

---

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
|---|-----------|----------------|-----------------|
| 01 | [Eigenvalues and Eigenvectors](01-Eigenvalues-and-Eigenvectors/notes.md) | Spectral theory, diagonalisation, spectral theorem, Jordan form | Eigenvalues, eigenvectors, characteristic polynomial, spectral theorem, matrix functions |
| 02 | [Singular Value Decomposition](02-Singular-Value-Decomposition/notes.md) | The universal matrix factorisation; low-rank approximation | SVD, singular values/vectors, Eckart-Young, pseudo-inverse, four fundamental subspaces |
| 03 | [Principal Component Analysis](03-Principal-Component-Analysis/notes.md) | Optimal linear dimensionality reduction via SVD | PCA, explained variance, whitening, kernel PCA, probabilistic PCA |
| 04 | [Linear Transformations](04-Linear-Transformations/notes.md) | Maps between vector spaces; kernels, images, change of basis | Linear maps, kernel, image, rank-nullity, matrix representation, change of basis |
| 05 | [Orthogonality and Orthonormality](05-Orthogonality-and-Orthonormality/notes.md) | Orthogonal bases, projections, QR via Gram-Schmidt | Gram-Schmidt, QR decomposition, orthogonal projections, orthonormal bases |
| 06 | [Matrix Norms](06-Matrix-Norms/notes.md) | Measuring matrix size; conditioning; spectral norm in AI | Frobenius, spectral, nuclear, operator norms; condition number; spectral normalisation |
| 07 | [Positive Definite Matrices](07-Positive-Definite-Matrices/notes.md) | SPD matrices; Cholesky; log-det; curvature in optimisation | Positive definiteness, Cholesky decomposition, LDLᵀ, Schur complement, log-det |
| 08 | [Matrix Decompositions](08-Matrix-Decompositions/notes.md) | Computational decompositions: LU, QR, Cholesky | LU (Gaussian elimination), QR (Householder, Givens), Cholesky (SPD systems) |

---

## Reading Order and Dependencies

```
01-Eigenvalues-and-Eigenvectors   (spectral theory — start here)
        ↓
02-Singular-Value-Decomposition   (universal factorisation; uses eigenvalues of AᵀA)
        ↓
03-Principal-Component-Analysis   (dimensionality reduction; uses SVD)
        ↓
04-Linear-Transformations         (abstract map theory; uses rank, image, kernel)
        ↓
05-Orthogonality-and-Orthonormality  (orthogonal bases; QR decomposition)
        ↓
06-Matrix-Norms                   (measuring matrices; condition number)
        ↓
07-Positive-Definite-Matrices     (SPD theory; Cholesky; curvature)
        ↓
08-Matrix-Decompositions          (LU, QR, Cholesky as computational algorithms)
        ↓
04-Calculus-Fundamentals          (next chapter)
```

---

## What Belongs Where (Canonical Homes)

| Topic | Canonical Home | Previewed In |
|-------|---------------|-------------|
| Eigenvalues, eigenvectors, diagonalisation | §01 | §04 (char. poly) from ch.2 |
| Spectral theorem, Jordan form | §01 | — |
| SVD, singular values, Eckart-Young | §02 | §02 ch.2 (brief preview) |
| Pseudo-inverse via SVD | §02 | §02 ch.2 (brief preview) |
| PCA, explained variance, whitening | §03 | — |
| Kernel PCA, probabilistic PCA | §03 | — |
| Linear maps, kernel, image | §04 | §06 ch.2 (abstract spaces) |
| Change of basis | §04 | §01 ch.2 (coordinates) |
| Gram-Schmidt | §05 | §09 ch.2 (inner products) |
| QR decomposition (theory) | §05 | §08 this ch. (algorithms) |
| Frobenius, spectral, nuclear norms | §06 | — |
| Condition number | §06 | §02 ch.2 (inverse, conditioning) |
| Positive definiteness, SPD matrices | §07 | — |
| Cholesky decomposition (full) | §07 | §08 this ch. (brief overview) |
| Log-determinant | §07 | §04 ch.2 (det preview) |
| LU decomposition (algorithm) | §08 | §02 ch.2 (brief preview) |
| QR decomposition (Householder/Givens) | §08 | §05 this ch. (theory) |

---

## Key Cross-Chapter Dependencies

**From Chapter 2 (Linear Algebra Basics):**
- §01 here assumes: characteristic polynomial from [Determinants §5](../02-Linear-Algebra-Basics/04-Determinants/notes.md)
- §02 here assumes: four fundamental subspaces from [Vector Spaces §7](../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md)
- §04 here assumes: abstract vector space axioms from [Vector Spaces §2](../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md)

**Into Chapter 4 (Calculus):**
- Jacobian matrices (§04 here) appear throughout multivariable calculus
- Hessian positive definiteness (§07 here) drives second-order optimisation
- Matrix norms (§06 here) measure gradient/weight magnitudes

---

## Prerequisites

Before starting this chapter, you should be comfortable with:
- Vectors, matrices, matrix multiply, inverse ([Chapter 2 §01–§02](../02-Linear-Algebra-Basics/README.md))
- Systems of equations, rank, null space ([Chapter 2 §03–§05](../02-Linear-Algebra-Basics/README.md))
- Vector spaces, subspaces, four fundamental subspaces ([Chapter 2 §06](../02-Linear-Algebra-Basics/06-Vector-Spaces-Subspaces/notes.md))

---

[← Previous Chapter: Linear Algebra Basics](../02-Linear-Algebra-Basics/README.md) | [Next Chapter: Calculus Fundamentals →](../04-Calculus-Fundamentals/README.md)
