# Singular Value Decomposition (SVD)

## Introduction

The Singular Value Decomposition is one of the most powerful and widely-used matrix factorizations in mathematics and machine learning. Unlike eigendecomposition (which only works for square matrices), SVD works for **any** matrix and reveals its fundamental structure.

## Prerequisites

- Eigenvalues and eigenvectors
- Matrix multiplication
- Orthogonal matrices
- Matrix transpose

## Learning Objectives

1. Understand the SVD factorization geometrically
2. Compute SVD components
3. Apply low-rank approximation
4. Use SVD for dimensionality reduction
5. Understand SVD's role in ML applications

---

## 1. Definition

For any matrix $A \in \mathbb{R}^{m \times n}$, the SVD is:

$$A = U \Sigma V^T$$

where:

- $U \in \mathbb{R}^{m \times m}$ - orthogonal matrix (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ - diagonal matrix (singular values)
- $V \in \mathbb{R}^{n \times n}$ - orthogonal matrix (right singular vectors)

```
SVD Structure:

A         =    U      ×     Σ      ×    Vᵀ
(m×n)       (m×m)       (m×n)       (n×n)

[       ]   [| | |]   [σ₁ 0  0 ]   [--- v₁ᵀ ---]
[   A   ] = [u₁u₂u₃]  [0  σ₂ 0 ]   [--- v₂ᵀ ---]
[       ]   [| | |]   [0  0  σ₃]   [--- v₃ᵀ ---]
```

### Properties

- $U^T U = I$ and $V^T V = I$ (orthogonal)
- Singular values: $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$
- $r = \text{rank}(A)$ equals number of non-zero singular values

---

## 2. Geometric Interpretation

The SVD reveals that any linear transformation can be decomposed into:

1. **Rotation** (by $V^T$)
2. **Scaling** (by $\Sigma$)
3. **Rotation** (by $U$)

```
    Original      →    Rotate      →    Scale       →    Rotate

       ●               ●                 ●                 ●
      /|\             /|\                |                /|
     / | \    Vᵀ     / | \      Σ       |       U       / |
    ●──●──●   →    ●──●──●      →    ●──●──●    →    ●──●──●
                                     (stretched)      (final)
```

**Key Insight**: SVD finds the orthonormal bases for both domain and codomain that diagonalize the transformation.

---

## 3. Computing SVD

### Relationship to Eigendecomposition

The singular values and vectors are related to eigendecomposition:

**Right singular vectors** $V$: eigenvectors of $A^T A$
$$A^T A = V \Sigma^T \Sigma V^T = V \text{diag}(\sigma_i^2) V^T$$

**Left singular vectors** $U$: eigenvectors of $A A^T$
$$A A^T = U \Sigma \Sigma^T U^T = U \text{diag}(\sigma_i^2) U^T$$

**Singular values**: $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are eigenvalues of $A^T A$ (or $A A^T$)

### Alternative: $u_i = \frac{1}{\sigma_i} A v_i$

Once you have $V$ and $\Sigma$, compute $U$ directly.

---

## 4. Compact SVD Forms

### Full SVD

$$A = U_{m \times m} \Sigma_{m \times n} V_{n \times n}^T$$

### Thin SVD (Economy SVD)

For $m \geq n$:
$$A = U_{m \times n} \Sigma_{n \times n} V_{n \times n}^T$$

```
Full SVD vs Thin SVD (m > n):

Full:                    Thin (economy):
A = U × Σ × Vᵀ          A = Û × Σ̂ × Vᵀ
(m×n) (m×m)(m×n)(n×n)   (m×n) (m×n)(n×n)(n×n)

    [σ₁ 0  0]              [σ₁ 0  0]
Σ = [0  σ₂ 0]          Σ̂ = [0  σ₂ 0]
    [0  0  σ₃]             [0  0  σ₃]
    [0  0  0 ]             (no zero rows)
```

### Truncated SVD

Keep only top $k$ singular values:
$$A_k = U_k \Sigma_k V_k^T$$

where $U_k$ is $m \times k$, $\Sigma_k$ is $k \times k$, $V_k$ is $n \times k$.

---

## 5. Low-Rank Approximation

### Outer Product Form

$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

Each term $\sigma_i u_i v_i^T$ is a rank-1 matrix.

```
A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + σ₃u₃v₃ᵀ + ...
    \_____/   \_____/   \_____/
    rank-1    rank-1    rank-1
    (largest) (second)  (third)
```

### Eckart-Young Theorem

The best rank-$k$ approximation (in Frobenius or spectral norm) is:

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T = U_k \Sigma_k V_k^T$$

**Error bound**:
$$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$
$$\|A - A_k\|_2 = \sigma_{k+1}$$

---

## 6. Important Properties

### Matrix Norms via SVD

| Norm           | Formula                            | Definition                     |
| -------------- | ---------------------------------- | ------------------------------ |
| Spectral norm  | $\|A\|_2 = \sigma_1$               | Largest singular value         |
| Frobenius norm | $\|A\|_F = \sqrt{\sum \sigma_i^2}$ | Sum of squared singular values |
| Nuclear norm   | $\|A\|_* = \sum \sigma_i$          | Sum of singular values         |

### Rank and Condition Number

- $\text{rank}(A) = $ number of non-zero singular values
- $\text{cond}(A) = \sigma_1 / \sigma_r$ (condition number)

### Pseudoinverse

The Moore-Penrose pseudoinverse:
$$A^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ inverts non-zero singular values:
$$\Sigma^+_{ii} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

---

## 7. SVD vs Eigendecomposition

| Property      | Eigendecomposition    | SVD                       |
| ------------- | --------------------- | ------------------------- |
| Applies to    | Square matrices only  | Any matrix                |
| Factors       | $A = PDP^{-1}$        | $A = U\Sigma V^T$         |
| May not exist | For some matrices     | Always exists             |
| Values        | Can be complex        | Always real, non-negative |
| Vectors       | May not be orthogonal | Always orthogonal         |

---

## 8. Applications in ML/AI

### 1. Principal Component Analysis (PCA)

For centered data matrix $X$:

- Covariance: $C = \frac{1}{n}X^TX$
- Right singular vectors of $X$ are eigenvectors of $X^TX$
- Principal components = columns of $V$
- Variance explained = $\sigma_i^2 / (n-1)$

### 2. Image Compression

```
Original Image     →    Low-Rank Approximation
   (m × n)              k components

Full rank: mn values    Compressed: k(m+n+1) values
e.g., 1000×1000        For k=50: 50(1000+1000+1) ≈ 100k
= 1,000,000            90% compression!
```

### 3. Recommender Systems (Matrix Completion)

```
User-Item Rating Matrix:

Users  Items →
  ↓    [5  ?  3  ?]
       [?  4  ?  2]
       [3  ?  5  ?]

SVD reveals latent factors:
- Users mapped to latent space
- Items mapped to same latent space
- Predict missing: user_vector · item_vector
```

### 4. Latent Semantic Analysis (LSA)

For term-document matrix $A$:

- Truncated SVD finds latent topics
- Similar documents have similar projections
- Used in document retrieval and topic modeling

### 5. Noise Reduction

High singular values = signal, low = noise:
$$A_{\text{denoised}} = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

### 6. Solving Least Squares

For overdetermined system $Ax \approx b$:
$$x = A^+b = V\Sigma^+U^Tb$$

More numerically stable than normal equations.

---

## 9. Numerical Computation

### Algorithm Overview

1. **Golub-Kahan bidiagonalization**: Reduce $A$ to bidiagonal form
2. **QR iteration**: Compute singular values of bidiagonal matrix
3. **Accumulate transformations**: Get $U$ and $V$

### Computational Complexity

| Operation               | Complexity            |
| ----------------------- | --------------------- |
| Full SVD                | $O(\min(mn^2, m^2n))$ |
| Truncated SVD (k terms) | $O(mnk)$              |
| Randomized SVD          | $O(mn\log k)$         |

### Randomized SVD

For very large matrices, use randomized algorithms:

1. Random projection to reduce dimension
2. Compute SVD of smaller matrix
3. Recover approximate singular vectors

---

## 10. Summary

### Key Formulas

| Concept                     | Formula                                   |
| --------------------------- | ----------------------------------------- |
| SVD                         | $A = U\Sigma V^T$                         |
| Singular values from $A^TA$ | $\sigma_i = \sqrt{\lambda_i(A^TA)}$       |
| Low-rank approximation      | $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$   |
| Pseudoinverse               | $A^+ = V\Sigma^+U^T$                      |
| Spectral norm               | $\|A\|_2 = \sigma_1$                      |
| Frobenius norm              | $\|A\|_F = \sqrt{\sum \sigma_i^2}$        |
| Rank                        | $\text{rank}(A) = $ # non-zero $\sigma_i$ |
| Condition number            | $\kappa(A) = \sigma_1/\sigma_r$           |

### Quick Reference

```
Computing SVD:
1. Compute A^T A (or use algorithms directly)
2. Find eigenvalues λᵢ of A^T A
3. σᵢ = √λᵢ (singular values)
4. vᵢ = eigenvectors of A^T A (right singular vectors)
5. uᵢ = Avᵢ/σᵢ (left singular vectors)

Low-rank approximation:
- Keep top k singular values
- Error = √(σ²_{k+1} + σ²_{k+2} + ...)
```

---

## Exercises

1. Compute the SVD of $A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$
2. Find the best rank-1 approximation of $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$
3. Compute the condition number of $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$
4. Find the pseudoinverse using SVD
5. How many components needed to capture 90% of the variance?

---

## References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Trefethen & Bau - "Numerical Linear Algebra"
3. Halko et al. - "Finding Structure with Randomness" (Randomized SVD)
