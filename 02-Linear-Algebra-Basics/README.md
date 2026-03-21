[← Back to Curriculum Home](../README.md) | [Next Chapter: Advanced Linear Algebra →](../03-Advanced-Linear-Algebra/README.md)

---

# Chapter 2 — Linear Algebra Basics

> _"Linear algebra is the language of data. Every modern neural network is a composition of linear maps, and every training algorithm is an optimisation over a high-dimensional linear space."_

## Overview

This chapter builds the computational and conceptual foundation of linear algebra needed for machine learning and modern AI systems. It moves in a deliberate progression: from concrete geometric objects (vectors) through computational procedures (matrix operations, solving systems) through structural properties (determinants, rank) to abstract formalization (vector spaces and subspaces).

Each subsection is **self-contained** but **designed to be read in order**. Concepts introduced concretely in earlier sections are given rigorous abstract treatment in later sections — this is intentional. The progression mirrors how practicing ML engineers actually encounter linear algebra: first operationally, then structurally.

---

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
|---|-----------|----------------|-----------------|
| 01 | [Vectors and Spaces](01-Vectors-and-Spaces/notes.md) | Concrete geometry of vectors in $\mathbb{R}^n$, norms, inner products, orthogonality, projections | Vectors, norms, dot products, orthogonal projections, coordinate geometry |
| 02 | [Matrix Operations](02-Matrix-Operations/notes.md) | Matrix arithmetic, multiplication, inverse, pseudo-inverse; decomposition overview | Matrix multiply, transpose, trace, inverse, Moore-Penrose pseudo-inverse |
| 03 | [Systems of Equations](03-Systems-of-Equations/notes.md) | Solving $Ax = b$, Gaussian elimination, least squares, iterative methods | Row reduction, RREF, existence/uniqueness, least squares, normal equations |
| 04 | [Determinants](04-Determinants/notes.md) | Determinant as volume-scaling; properties, computation, characteristic polynomial | Determinant definition, cofactor expansion, properties, characteristic polynomial, log-det |
| 05 | [Matrix Rank](05-Matrix-Rank/notes.md) | Rank as dimension of the image; rank-nullity, low-rank structure in AI | Rank definition, rank-nullity theorem, low-rank approximation, effective rank |
| 06 | [Vector Spaces and Subspaces](06-Vector-Spaces-Subspaces/notes.md) | Axiomatic vector spaces, subspaces, four fundamental subspaces, inner product spaces | Vector space axioms, subspace criteria, four fundamental subspaces, Gram-Schmidt |

---

## Reading Order and Dependencies

```
01-Vectors-and-Spaces         (concrete geometry — start here)
        ↓
02-Matrix-Operations          (computational rules for linear maps)
        ↓
03-Systems-of-Equations       (solving Ax = b; uses rank informally)
        ↓
04-Determinants               (structure via volume; introduces char. polynomial)
        ↓
05-Matrix-Rank                (structure via dimension; rank-nullity formally)
        ↓
06-Vector-Spaces-Subspaces    (rigorous axiomatics; four fundamental subspaces)
        ↓
03-Advanced-Linear-Algebra    (eigenvalues, SVD, decompositions — next chapter)
```

---

## How the Subsections Relate

**01 vs 06:** Subsection 01 treats vectors concretely in $\mathbb{R}^n$ with geometric intuition. Subsection 06 treats vector spaces axiomatically — the same concepts (span, basis, orthogonality) reappear at a higher level of abstraction. Reading 01 first gives the intuition that makes 06 meaningful.

**03 vs 05:** Subsection 03 uses rank informally (to characterize system solutions). Subsection 05 is the canonical home for rank theory — definitions, proofs, and properties. Cross-references connect them cleanly.

**04 vs 03-Advanced-Linear-Algebra:** Section 5 of Subsection 04 introduces the characteristic polynomial — this is a determinant concept. The full eigenvalue theory (algorithms, spectral theorem, diagonalization) lives in [03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors](../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md).

**Decompositions (LU, QR, SVD, Cholesky, Eigendecomposition):** Brief previews appear in Subsection 02. Full treatments are in [03-Advanced-Linear-Algebra](../03-Advanced-Linear-Algebra/README.md).

---

## What Belongs Where (Canonical Homes)

| Topic | Canonical Home | Preview In |
|-------|---------------|-----------|
| Vectors, norms, dot products | §01 | — |
| Matrix arithmetic, multiply, inverse | §02 | — |
| Row reduction, Gaussian elimination | §03 | — |
| Least squares, normal equations | §03 | §02 (pseudo-inverse) |
| Determinant, cofactors, log-det | §04 | §02 (brief preview) |
| Characteristic polynomial | §04 | — |
| Rank, rank-nullity, null space | §05 | §03 (used informally) |
| Low-rank approximation | §05 | §02 (SVD preview) |
| Vector space axioms, subspace criteria | §06 | §01 (concrete cases) |
| Four fundamental subspaces | §06 | §03 (applied), §05 (rank view) |
| Inner product spaces (abstract) | §06 | §01 (concrete $\mathbb{R}^n$) |
| Eigenvalues, eigenvectors | 03-Advanced §01 | §04 (char. polynomial) |
| SVD | 03-Advanced §02 | §02 (preview) |
| LU, QR, Cholesky | 03-Advanced §08 | §02 (preview) |

---

## Prerequisites

Before starting this chapter, you should be comfortable with:
- High-school algebra and coordinate geometry
- Summation notation $\sum_{i=1}^n$
- Basic set notation and function notation

These are covered in [Chapter 1 — Mathematical Foundations](../01-Mathematical-Foundations/README.md).

---

## After This Chapter

This chapter prepares you for:
- **[03-Advanced-Linear-Algebra](../03-Advanced-Linear-Algebra/README.md)** — Eigenvalues, SVD, matrix decompositions, spectral theory
- **[05-Calculus-and-Analysis](../05-Calculus-and-Analysis/README.md)** — Multivariable calculus uses vector space language throughout
- **[08-Optimization](../08-Optimization/README.md)** — Gradient descent operates in the vector spaces built here

---

[← Back to Curriculum Home](../README.md) | [Next Chapter: Advanced Linear Algebra →](../03-Advanced-Linear-Algebra/README.md)
