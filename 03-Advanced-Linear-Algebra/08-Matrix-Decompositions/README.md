# Matrix Decompositions

## Introduction

Matrix decompositions (or factorizations) are the workhorses of numerical linear algebra. By expressing a matrix as a product of simpler matrices with special structure, we can solve systems faster, understand matrix properties, and implement efficient algorithms for ML.

## Prerequisites

- Matrix multiplication
- Triangular matrices
- Positive definite matrices
- Orthogonal matrices

## Learning Objectives

1. Understand and compute LU decomposition
2. Master QR decomposition methods
3. Apply Cholesky decomposition
4. Recognize when to use each decomposition
5. Connect decompositions to ML applications

---

## 1. Overview of Decompositions

```
Matrix A
    │
    ├─── LU Decomposition: A = LU or A = PLU
    │    └── General square matrices, solving linear systems
    │
    ├─── QR Decomposition: A = QR
    │    └── Any matrix, least squares, eigenvalue algorithms
    │
    ├─── Cholesky: A = LL^T
    │    └── Symmetric positive definite, fast and stable
    │
    ├─── Eigendecomposition: A = QΛQ^(-1)
    │    └── Diagonalizable matrices, spectral analysis
    │
    └─── SVD: A = UΣV^T
         └── Any matrix, most general decomposition
```

---

## 2. LU Decomposition

### Definition

For a square matrix $A$, LU decomposition finds:
$$A = LU$$

where:

- $L$ = Lower triangular (1s on diagonal)
- $U$ = Upper triangular

### The Algorithm (Gaussian Elimination)

```
         [a₁₁  a₁₂  a₁₃]     [1   0   0] [u₁₁  u₁₂  u₁₃]
    A =  [a₂₁  a₂₂  a₂₃]  =  [l₂₁ 1   0] [0    u₂₂  u₂₃]
         [a₃₁  a₃₂  a₃₃]     [l₃₁ l₃₂ 1] [0    0    u₃₃]
```

**Steps:**

1. First row of U is first row of A
2. First column of L: $l_{i1} = a_{i1}/u_{11}$
3. Update remaining submatrix
4. Repeat recursively

### LU with Pivoting (PLU)

$$PA = LU$$

where $P$ is a permutation matrix (row swaps for numerical stability).

### Solving Ax = b with LU

1. Factor: $A = LU$
2. Forward solve: $L\mathbf{y} = \mathbf{b}$
3. Back solve: $U\mathbf{x} = \mathbf{y}$

**Complexity**: $O(n^3)$ to factor, $O(n^2)$ per solve

---

## 3. QR Decomposition

### Definition

For any $m \times n$ matrix $A$ with $m \geq n$:
$$A = QR$$

where:

- $Q$ = $m \times n$ orthonormal columns ($Q^TQ = I$)
- $R$ = $n \times n$ upper triangular

### Full vs Reduced QR

```
Full QR:                    Reduced QR:
A(m×n) = Q(m×m) R(m×n)     A(m×n) = Q(m×n) R(n×n)

[a a a]   [q q q q] [r r r]    [a a a]   [q q q] [r r r]
[a a a] = [q q q q] [0 r r]    [a a a] = [q q q] [0 r r]
[a a a]   [q q q q] [0 0 r]    [a a a]   [q q q] [0 0 r]
[a a a]   [q q q q] [0 0 0]    [a a a]   [q q q]
```

### Method 1: Gram-Schmidt

Apply Gram-Schmidt to columns of $A$:

$$\mathbf{q}_j = \frac{\mathbf{a}_j - \sum_{i<j} (\mathbf{q}_i^T\mathbf{a}_j)\mathbf{q}_i}{\|\mathbf{a}_j - \sum_{i<j} (\mathbf{q}_i^T\mathbf{a}_j)\mathbf{q}_i\|}$$

Then $R_{ij} = \mathbf{q}_i^T \mathbf{a}_j$ for $i \leq j$.

### Method 2: Householder Reflections

More stable than Gram-Schmidt. Uses reflection matrices:
$$H = I - 2\mathbf{u}\mathbf{u}^T$$

where $\|\mathbf{u}\| = 1$.

Properties of Householder:

- Orthogonal: $H^TH = I$
- Symmetric: $H = H^T$
- Self-inverse: $H^2 = I$

### Method 3: Givens Rotations

Rotates in a 2D plane to zero out one element:
$$G = \begin{bmatrix} c & -s \\ s & c \end{bmatrix}$$

where $c^2 + s^2 = 1$.

---

## 4. Cholesky Decomposition

### Definition

For symmetric positive definite $A$:
$$A = LL^T$$

where $L$ is lower triangular with positive diagonal.

### Algorithm

$$l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$$

$$l_{ij} = \frac{1}{l_{jj}}\left(a_{ij} - \sum_{k=1}^{j-1} l_{ik}l_{jk}\right) \quad \text{for } i > j$$

### Why Cholesky?

| Aspect       | Cholesky        | LU               |
| ------------ | --------------- | ---------------- |
| Requirements | Symmetric PD    | Square           |
| Operations   | $\frac{n^3}{6}$ | $\frac{2n^3}{3}$ |
| Pivoting     | Not needed      | Often needed     |
| Storage      | Half the matrix | Full matrix      |

### Variant: LDL^T Decomposition

$$A = LDL^T$$

where $D$ is diagonal. Avoids square roots.

---

## 5. Comparison Table

| Decomposition | Exists For             | Main Use                   | Complexity |
| ------------- | ---------------------- | -------------------------- | ---------- |
| LU            | Square (with pivoting) | Linear systems             | $O(n^3)$   |
| QR            | Any (m ≥ n)            | Least squares, eigenvalues | $O(mn^2)$  |
| Cholesky      | Symmetric PD           | Fast linear systems        | $O(n^3/6)$ |
| Eigendecomp   | Square, diagonalizable | Spectral analysis          | $O(n^3)$   |
| SVD           | Any                    | General purpose            | $O(mn^2)$  |

---

## 6. Applications in ML/AI

### Linear Regression (Normal Equations)

Solve $(X^TX)\mathbf{w} = X^T\mathbf{y}$

**Options:**

1. Cholesky if $X^TX$ is PD (add regularization)
2. QR: $X = QR$, then $R\mathbf{w} = Q^T\mathbf{y}$

### Ridge Regression

$(X^TX + \lambda I)\mathbf{w} = X^T\mathbf{y}$

Use Cholesky (always PD for $\lambda > 0$).

### Gaussian Processes

Sampling from $\mathcal{N}(\mu, \Sigma)$:

1. Cholesky: $\Sigma = LL^T$
2. Sample $\mathbf{z} \sim \mathcal{N}(0, I)$
3. Return $\mu + L\mathbf{z}$

### PCA via QR

Power iteration for eigenvalues uses QR:

```python
for _ in range(iterations):
    Q, R = qr(A @ Q_old)
```

### Matrix Inversion (Avoid!)

Instead of computing $A^{-1}$, solve $A\mathbf{x} = \mathbf{b}$:

1. Factor $A$ once
2. Solve for each $\mathbf{b}$ efficiently

### Determinant Computation

From LU: $\det(A) = \prod_i u_{ii}$
From Cholesky: $\det(A) = \prod_i l_{ii}^2$

---

## 7. Numerical Stability

### Condition Number

$$\kappa(A) = \|A\| \|A^{-1}\|$$

Large $\kappa$ → numerically unstable!

### Stability Comparison

| Method            | Stability         |
| ----------------- | ----------------- |
| LU (no pivot)     | Unstable          |
| LU (pivoted)      | Stable            |
| QR (Gram-Schmidt) | Moderately stable |
| QR (Householder)  | Very stable       |
| QR (Givens)       | Very stable       |
| Cholesky          | Stable for PD     |

---

## 8. When to Use What

```
Solving Ax = b:
│
├── A is symmetric positive definite?
│   └── YES → Cholesky (fastest)
│   └── NO → Continue
│
├── A is square?
│   └── YES → LU with pivoting
│   └── NO → Continue
│
├── A is rectangular (overdetermined)?
│   └── YES → QR for least squares
│
└── Need most general solution?
    └── SVD
```

### Quick Reference

| Problem                    | Best Method    |
| -------------------------- | -------------- |
| Solve $Ax = b$ (A square)  | LU or Cholesky |
| Least squares $\|Ax - b\|$ | QR             |
| Low-rank approximation     | SVD            |
| Sample from Gaussian       | Cholesky       |
| Eigenvalues                | QR algorithm   |

---

## 9. Summary

### Key Formulas

| Decomposition | Formula     | Key Property         |
| ------------- | ----------- | -------------------- |
| LU            | $A = LU$    | $L$ lower, $U$ upper |
| PLU           | $PA = LU$   | With pivoting        |
| QR            | $A = QR$    | $Q$ orthogonal       |
| Cholesky      | $A = LL^T$  | $A$ must be SPD      |
| LDL^T         | $A = LDL^T$ | No square roots      |

### Complexity

```
Operation count for n×n matrix:

Cholesky:     ~n³/6
LU:           ~2n³/3
QR (H'holder): ~2n³/3
SVD:          ~4n³ to 11n³
```

### ML Applications Summary

```
ML Application          → Best Decomposition
─────────────────────────────────────────
Linear regression      → QR or Cholesky
Ridge regression       → Cholesky
Gaussian sampling      → Cholesky
PCA                    → SVD or Eigendecomp
Matrix completion      → SVD
Solving dense systems  → LU
Sparse systems         → Specialized methods
```

---

## Exercises

1. Compute LU decomposition of $\begin{bmatrix} 2 & 1 \\ 4 & 5 \end{bmatrix}$
2. Use QR to solve least squares: $\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}x \approx \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$
3. Compare Cholesky and LU for solving a PD system
4. Implement Householder QR for a 3×2 matrix
5. Use Cholesky to sample from a 2D Gaussian

---

## References

1. Trefethen & Bau - "Numerical Linear Algebra"
2. Golub & Van Loan - "Matrix Computations"
3. Press et al. - "Numerical Recipes"
