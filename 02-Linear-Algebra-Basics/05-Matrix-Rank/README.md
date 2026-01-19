# Matrix Rank

## Overview

The **rank** of a matrix is one of the most important concepts in linear algebra. It tells us about the "effective dimensionality" of a matrix—how much independent information it contains.

## Learning Objectives

- Understand what matrix rank represents
- Compute rank using various methods
- Connect rank to linear independence and solutions
- Apply rank concepts to ML problems

---

## 1. What is Matrix Rank?

### 1.1 Definition

The **rank** of a matrix $A$ is the maximum number of linearly independent rows (or columns).

$$\text{rank}(A) = \text{dim}(\text{column space}) = \text{dim}(\text{row space})$$

```
Rank tells us:
┌────────────────────────────────────────────────────────────┐
│                                                            │
│  "How many dimensions does the output span?"               │
│                                                            │
│  Full rank (rank = min(m,n)): Uses all available space     │
│  Rank deficient: Some dimensions are "wasted"              │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 1.2 Examples

```
Full Rank (rank = 2):              Rank Deficient (rank = 1):

A = [1  0]  ← independent         B = [1  2]  ← row 2 = 2 × row 1
    [0  1]  ← independent             [2  4]

Columns span all of R²            Columns span only a line in R²
```

### 1.3 Key Properties

| Property               | Formula                                                     |
| ---------------------- | ----------------------------------------------------------- |
| Row rank = Column rank | Always equal                                                |
| Upper bound            | $\text{rank}(A) \leq \min(m, n)$                            |
| Zero matrix            | $\text{rank}(0) = 0$                                        |
| Identity               | $\text{rank}(I_n) = n$                                      |
| Product bound          | $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ |
| Sum bound              | $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$   |

---

## 2. Computing Rank

### 2.1 Row Echelon Form Method

Rank = number of non-zero rows in row echelon form (number of pivots).

```
Example:
           Row reduction
A = [1  2  3]  ────────→  [1  2  3]
    [2  4  6]             [0  0  0]  ← zero row
    [1  1  1]             [0  0  1]

Pivots: columns 1 and 3
rank(A) = 2
```

### 2.2 SVD Method

Rank = number of non-zero singular values.

$$A = U\Sigma V^T$$

$$\text{rank}(A) = \text{number of non-zero } \sigma_i$$

```
Singular values: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0 = σᵣ₊₁ = ... = σₙ
                 └─────────────────────┘
                        r = rank
```

### 2.3 Determinant Method (Square Matrices)

For square matrices:

$$\text{rank}(A) = n \iff \det(A) \neq 0$$

Check determinants of all $k \times k$ submatrices to find the largest non-zero one.

---

## 3. Rank and Linear Systems

### 3.1 Rank-Nullity Theorem

For $A \in \mathbb{R}^{m \times n}$:

$$\text{rank}(A) + \text{nullity}(A) = n$$

Where **nullity** = dimension of null space = number of free variables.

```
┌──────────────────────────────────────────────────────────┐
│                    n columns                              │
│     ┌──────────────────────────────────────────┐         │
│     │                                          │         │
│     │  rank(A)  │        nullity(A)           │         │
│     │  columns  │    (null space dimension)   │         │
│     │   used    │       free variables        │         │
│     │           │                              │         │
│     └──────────────────────────────────────────┘         │
│                                                           │
│     Pivot columns + Free columns = Total columns          │
└──────────────────────────────────────────────────────────┘
```

### 3.2 Solution Existence

For $Ax = b$:

| Condition                              | Meaning                             |
| -------------------------------------- | ----------------------------------- |
| $\text{rank}(A) = \text{rank}([A\|b])$ | Solution exists                     |
| $\text{rank}(A) < \text{rank}([A\|b])$ | No solution                         |
| $\text{rank}(A) = n$                   | At most one solution                |
| $\text{rank}(A) < n$                   | If solution exists, infinitely many |

### 3.3 Full Rank Cases

```
Full Column Rank (rank = n):        Full Row Rank (rank = m):
- Columns are independent           - Rows are independent
- Ax = b has at most 1 solution    - Ax = b always has a solution
- Nullity = 0                       - Surjective transformation
- AᵀA is invertible                - AAᵀ is invertible
- Unique least squares solution     - Minimum norm solution exists

Full Rank (square, rank = n = m):
- A is invertible
- Ax = b has exactly one solution
- det(A) ≠ 0
```

---

## 4. Geometric Interpretation

### 4.1 Column Space Dimension

Rank = dimension of the image (range) of the linear transformation.

```
A: Rⁿ → Rᵐ

                n-dimensional input
                      │
                      ▼
              ┌───────────────┐
              │       A       │
              └───────────────┘
                      │
                      ▼
           rank(A)-dimensional output
           (inside m-dimensional space)
```

### 4.2 Rank Deficiency

```
Full Rank Transformation:          Rank-Deficient Transformation:

    Input                              Input
   ╱     ╲                            ╱     ╲
  ●───────●                          ●───────●
   ╲     ╱                            ╲     ╱
    ╲   ╱                              ╲   ╱
     ╲ ╱                                ╲ ╱
      ●                                  ●
                                         │
    Output                               ▼
   ╱     ╲                          ────●────  (collapsed to line)
  ●───────●
   ╲     ╱                         Output (lower dimension)
```

---

## 5. Machine Learning Applications

### 5.1 Feature Redundancy

```
Data Matrix X (n samples × d features):

If rank(X) < d:
- Some features are linear combinations of others
- Redundant information
- May cause issues with matrix inversion (XᵀX singular)

Solution: Remove redundant features or use regularization
```

### 5.2 PCA and Effective Rank

In PCA, the "effective rank" considers how many principal components capture most variance:

$$\text{effective rank} \approx \frac{\left(\sum_i \sigma_i\right)^2}{\sum_i \sigma_i^2}$$

```
Singular Values:         Interpretation:

σ₁ ████████████████     High effective rank:
σ₂ ███████████████      All singular values similar
σ₃ ██████████████       → Data uses all dimensions
σ₄ █████████████
σ₅ ████████████

σ₁ ████████████████     Low effective rank:
σ₂ ███                  Few dominant singular values
σ₃ █                    → Data mostly in low-dim subspace
σ₄ ▪
σ₅ ▪
```

### 5.3 Low-Rank Approximation

Many ML techniques exploit low-rank structure:

```
Matrix Factorization (Recommender Systems):
R ≈ UVᵀ  where U (users × k), V (items × k), k << min(users, items)

       Users × Items          Users × k    k × Items
      ┌───────────────┐     ┌─────────┐  ┌─────────┐
      │               │  ≈  │         │  │         │
      │      R        │     │    U    │  │   Vᵀ    │
      │               │     │         │  │         │
      └───────────────┘     └─────────┘  └─────────┘

      Rank k approximation uses k * (users + items) instead of users * items
```

### 5.4 Neural Network Weight Matrices

Low-rank weight matrices reduce parameters:

$$W \approx UV^T$$

```
Original: W (m × n) → mn parameters
Low-rank: U (m × r), V (n × r) → r(m + n) parameters

If r << min(m,n): significant compression
```

### 5.5 Regularization and Rank

- **Nuclear norm regularization**: Encourages low-rank solutions
- **Ridge regression**: Handles rank-deficient $X^TX$ by adding $\lambda I$

---

## 6. Numerical Rank

### 6.1 Numerical Challenges

In practice, exact rank is problematic due to floating-point errors.

```
Theoretical:                    Numerical:
σ₁ = 10                        σ₁ = 10.0
σ₂ = 5                         σ₂ = 5.0
σ₃ = 0        ← rank = 2      σ₃ = 1e-15   ← Is this zero?
σ₄ = 0                         σ₄ = 1e-16
```

### 6.2 Tolerance-Based Rank

NumPy uses a tolerance:

$$\text{numerical rank} = \#\{i : \sigma_i > \epsilon \cdot \sigma_{\max}\}$$

Default: $\epsilon = \max(m, n) \cdot \sigma_{\max} \cdot \text{machine epsilon}$

### 6.3 Condition Number Connection

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$$

Large condition number suggests near rank-deficiency.

```
Condition Number:

κ ≈ 1:        Well-conditioned, clear rank
κ ≈ 10⁶:     Potentially numerically rank-deficient
κ = ∞:        Exactly singular (rank-deficient)
```

---

## 7. Special Rank Results

### 7.1 Rank of Product

$$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$

Multiplication can only decrease rank!

### 7.2 Rank of Sum

$$|\text{rank}(A) - \text{rank}(B)| \leq \text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$$

### 7.3 Rank of Transpose

$$\text{rank}(A) = \text{rank}(A^T)$$

### 7.4 Rank After Outer Product

$$\text{rank}(uv^T) = 1 \text{ (if } u, v \neq 0\text{)}$$

Outer products always have rank 1.

### 7.5 Gram Matrix Rank

$$\text{rank}(A^T A) = \text{rank}(A A^T) = \text{rank}(A)$$

---

## 8. Summary

### Key Formulas

| Concept      | Formula                                                     |
| ------------ | ----------------------------------------------------------- |
| Rank-Nullity | $\text{rank}(A) + \text{nullity}(A) = n$                    |
| Upper bound  | $\text{rank}(A) \leq \min(m, n)$                            |
| Product      | $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ |
| Transpose    | $\text{rank}(A) = \text{rank}(A^T)$                         |
| Gram matrix  | $\text{rank}(A^T A) = \text{rank}(A)$                       |

### Interpretation Guide

```
rank(A) = r means:

1. The column space of A is r-dimensional
2. The row space of A is r-dimensional
3. A has r linearly independent columns/rows
4. A has r non-zero singular values
5. Ax = b has solutions only if b is in r-dimensional subspace
6. The transformation A collapses (n-r) dimensions
```

### ML Connections

| Concept         | ML Application                                 |
| --------------- | ---------------------------------------------- |
| Full rank       | Feature independence, invertible $X^TX$        |
| Low rank        | Dimensionality reduction, matrix factorization |
| Rank deficiency | Collinearity, need for regularization          |
| Effective rank  | Number of significant components in PCA        |

---

## Further Reading

- [3Blue1Brown: Rank](https://www.youtube.com/watch?v=uQhTuRlWMxw)
- [Gilbert Strang: Column Space and Nullspace](https://www.youtube.com/watch?v=8o5Cmfpeo6g)
- [Low-Rank Matrix Approximation (Wikipedia)](https://en.wikipedia.org/wiki/Low-rank_approximation)

---

## Navigation

← [Previous: Determinants](../04-Determinants/README.md) | [Next: Vector Spaces and Subspaces →](../06-Vector-Spaces-Subspaces/README.md)

[Back to Main](../../README.md)
