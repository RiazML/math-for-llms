# Matrix Decompositions

## Introduction

Matrix decompositions (factorizations) are the workhorses of numerical linear algebra. By expressing a matrix as a product of simpler matrices with special structure, we gain:

- **Computational efficiency**: Solve systems faster with factored matrices
- **Numerical stability**: Well-designed decompositions minimize error accumulation
- **Theoretical insight**: Understand matrix properties through their factors
- **Algorithm design**: Many ML algorithms are built on decompositions

This section covers the three most practical decompositions: **LU** (general systems), **QR** (least squares), and **Cholesky** (symmetric positive definite). Each has its sweet spot—knowing when to use which is a crucial skill.

## Prerequisites

- Matrix multiplication and inverses
- Triangular matrices and substitution
- Positive definite matrices
- Orthogonal/unitary matrices
- Basic concepts of numerical stability

## Learning Objectives

1. Understand and manually compute LU, QR, and Cholesky decompositions
2. Know when each decomposition exists and when to use it
3. Apply decompositions to solve linear systems efficiently
4. Use QR for least squares and Cholesky for sampling
5. Understand numerical stability and condition numbers
6. Connect decompositions to ML applications (regression, GP, optimization)

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

- $L$ = Lower triangular matrix (ones on diagonal in Doolittle form)
- $U$ = Upper triangular matrix

This is essentially Gaussian elimination stored in matrix form.

### The Algorithm (Gaussian Elimination)

```
         [a₁₁  a₁₂  a₁₃]     [1   0   0] [u₁₁  u₁₂  u₁₃]
    A =  [a₂₁  a₂₂  a₂₃]  =  [l₂₁ 1   0] [0    u₂₂  u₂₃]
         [a₃₁  a₃₂  a₃₃]     [l₃₁ l₃₂ 1] [0    0    u₃₃]
```

**Steps:**

1. First row of $U$ is first row of $A$
2. First column of $L$: $l_{i1} = a_{i1}/u_{11}$
3. Update remaining submatrix
4. Repeat recursively

### LU with Pivoting (PLU)

Without pivoting, LU can fail (divide by zero) or be numerically unstable. **Partial pivoting** swaps rows:

$$PA = LU$$

where $P$ is a permutation matrix recording the row swaps.

**When is pivoting needed?**
- Small or zero diagonal elements during elimination
- Better numerical stability in general

### Existence Conditions

LU decomposition (without pivoting) exists when all leading principal minors are nonzero. With pivoting, it exists for any invertible matrix.

### Solving Ax = b with LU

1. **Factor** (once): $PA = LU$ — $O(n^3)$
2. **Forward solve**: $L\mathbf{y} = P\mathbf{b}$ — $O(n^2)$
3. **Back solve**: $U\mathbf{x} = \mathbf{y}$ — $O(n^2)$

**Key advantage**: Factor once, solve for multiple right-hand sides efficiently.

### Determinant from LU

$$\det(A) = \det(P)^{-1} \cdot \det(L) \cdot \det(U) = (-1)^s \cdot \prod_{i=1}^n u_{ii}$$

where $s$ is the number of row swaps.

---

## 3. QR Decomposition

### Definition

For any $m \times n$ matrix $A$ with $m \geq n$:
$$A = QR$$

where:

- $Q$ = $m \times n$ matrix with orthonormal columns ($Q^TQ = I_n$)
- $R$ = $n \times n$ upper triangular matrix

QR decomposition always exists (unlike LU which may need pivoting).

### Full vs Reduced QR

```
Full QR:                    Reduced QR:
A(m×n) = Q(m×m) R(m×n)     A(m×n) = Q(m×n) R(n×n)

[a a a]   [q q q q] [r r r]    [a a a]   [q q q] [r r r]
[a a a] = [q q q q] [0 r r]    [a a a] = [q q q] [0 r r]
[a a a]   [q q q q] [0 0 r]    [a a a]   [q q q] [0 0 r]
[a a a]   [q q q q] [0 0 0]    [a a a]   [q q q]

(Q is orthogonal)          (Q has orthonormal columns)
```

The reduced form is more common in practice—it's what `numpy.linalg.qr` returns by default.

### Method 1: Gram-Schmidt Orthogonalization

Apply Gram-Schmidt to columns of $A$:

$$\mathbf{q}_j = \frac{\mathbf{a}_j - \sum_{i<j} (\mathbf{q}_i^T\mathbf{a}_j)\mathbf{q}_i}{\|\mathbf{a}_j - \sum_{i<j} (\mathbf{q}_i^T\mathbf{a}_j)\mathbf{q}_i\|}$$

Then $R_{ij} = \mathbf{q}_i^T \mathbf{a}_j$ for $i \leq j$.

**Modified Gram-Schmidt** is more stable: compute projections as you go, updating the remaining vectors.

### Method 2: Householder Reflections

More numerically stable than Gram-Schmidt. Uses reflection matrices:
$$H = I - 2\mathbf{u}\mathbf{u}^T$$

where $\|\mathbf{u}\| = 1$.

**Properties of Householder reflections**:

- Orthogonal: $H^TH = HH^T = I$
- Symmetric: $H = H^T$
- Self-inverse (involution): $H^2 = I$
- Reflects vectors across the hyperplane perpendicular to $\mathbf{u}$

Each Householder transformation zeros out elements below the diagonal in one column.

### Method 3: Givens Rotations

Rotates in a 2D plane to zero out one element at a time:
$$G = \begin{bmatrix} c & -s \\ s & c \end{bmatrix}$$

where $c = \cos\theta$ and $s = \sin\theta$ ($c^2 + s^2 = 1$).

**When to use**: Sparse matrices where you only need to zero out a few elements.

### Solving Least Squares with QR

For overdetermined system $A\mathbf{x} \approx \mathbf{b}$ (minimize $\|A\mathbf{x} - \mathbf{b}\|_2$):

1. Compute $A = QR$ (reduced)
2. Solve $R\mathbf{x} = Q^T\mathbf{b}$

This is more stable than solving normal equations $(A^TA)\mathbf{x} = A^T\mathbf{b}$.

---

## 4. Cholesky Decomposition

### Definition

For **symmetric positive definite** matrix $A$:
$$A = LL^T$$

where $L$ is **lower triangular** with **positive diagonal entries**.

Alternatively: $A = R^TR$ where $R = L^T$ is upper triangular.

### Algorithm

$$l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$$

$$l_{ij} = \frac{1}{l_{jj}}\left(a_{ij} - \sum_{k=1}^{j-1} l_{ik}l_{jk}\right) \quad \text{for } i > j$$

Compute column by column, left to right.

### Existence and Uniqueness

- **Exists** iff $A$ is symmetric positive definite
- **Unique** when diagonal of $L$ is constrained to be positive
- If algorithm encounters negative under square root → matrix is not PD

### Computational Advantages

| Aspect       | Cholesky        | LU               | QR |
| ------------ | --------------- | ---------------- | --- |
| Requirements | Symmetric PD    | Square           | Any |
| FLOPs | $\frac{n^3}{6}$ | $\frac{2n^3}{3}$ | $\frac{2n^3}{3}$ |
| Pivoting     | Never needed      | Often needed     | Never needed |
| Storage      | Half the matrix | Full matrix      | $Q$ and $R$ |
| Stability | Excellent for SPD | Needs pivoting | Excellent |

### Variant: LDLᵀ Decomposition

$$A = LDL^T$$

where $L$ is unit lower triangular (ones on diagonal) and $D$ is diagonal.

**Advantages**:
- Avoids square roots (faster, works over integers)
- Easier to detect non-positive-definiteness (check $D$ diagonal)
- Same storage requirement as Cholesky

**Relationship**: If $A = LDL^T$, then $A = (L\sqrt{D})(L\sqrt{D})^T$ is Cholesky.

### Solving Systems with Cholesky

For $A\mathbf{x} = \mathbf{b}$ where $A$ is SPD:

1. **Cholesky**: $A = LL^T$ — $O(n^3/6)$
2. **Forward substitution**: $L\mathbf{y} = \mathbf{b}$ — $O(n^2/2)$
3. **Back substitution**: $L^T\mathbf{x} = \mathbf{y}$ — $O(n^2/2)$

### Determinant from Cholesky

$$\det(A) = \det(LL^T) = (\det L)^2 = \left(\prod_{i=1}^n l_{ii}\right)^2$$

### Log-Determinant (Important for ML)

For Gaussian likelihoods, we need $\log|\Sigma|$:

$$\log\det(A) = 2\sum_{i=1}^n \log(l_{ii})$$

This avoids overflow/underflow issues with large matrices.

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

1. **Cholesky** if $X^TX$ is well-conditioned (with regularization)
2. **QR on X**: $X = QR$, then $R\mathbf{w} = Q^T\mathbf{y}$ (more stable, recommended)
3. **SVD**: Most stable but slowest

```python
# QR approach (recommended)
Q, R = np.linalg.qr(X)
w = np.linalg.solve(R, Q.T @ y)
```

### Ridge Regression

$(X^TX + \lambda I)\mathbf{w} = X^T\mathbf{y}$

Use **Cholesky** (always PD for $\lambda > 0$):

```python
A = X.T @ X + lambda_ * np.eye(n_features)
L = np.linalg.cholesky(A)
w = solve_triangular(L.T, solve_triangular(L, X.T @ y, lower=True))
```

### Gaussian Processes

Sampling from $\mathcal{N}(\mu, \Sigma)$:

1. **Cholesky**: $\Sigma = LL^T$
2. Sample $\mathbf{z} \sim \mathcal{N}(0, I)$
3. Return $\mu + L\mathbf{z}$

Computing GP log-likelihood requires $\log|\Sigma|$:

$$\log|\Sigma| = 2\sum_i \log(L_{ii})$$

### PCA via Decompositions

**Power iteration** for eigenvalues uses QR:

```python
for _ in range(iterations):
    Q, R = qr(A @ Q_old)
```

The **QR algorithm** for eigenvalues repeatedly applies QR:
$$A_0 = A, \quad A_k = Q_kR_k, \quad A_{k+1} = R_kQ_k$$

### Matrix Inversion (Avoid When Possible!)

Instead of computing $A^{-1}$ explicitly, solve $A\mathbf{x} = \mathbf{b}$:

1. Factor $A$ once (LU, Cholesky, or QR depending on structure)
2. Solve for each right-hand side efficiently

**When you must compute $A^{-1}$**: Solve $AX = I$ column by column.

### Neural Network Applications

- **Batch normalization**: Involves inverse square root of covariance
- **Natural gradient**: Inverts Fisher information (use Cholesky)
- **Second-order optimization**: Newton's method needs $H^{-1}g$ (use LU)
- **Weight orthogonalization**: QR decomposition of weight matrices

---

## 7. Numerical Stability

### Condition Number

The condition number measures sensitivity to perturbations:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

**Interpretation**:
- $\kappa \approx 1$: Well-conditioned, stable
- $\kappa \gg 1$: Ill-conditioned, small errors amplify
- $\kappa = \infty$: Singular matrix

**Rule of thumb**: If $\kappa \approx 10^k$, you lose about $k$ digits of accuracy.

### Error Bounds

For solving $A\mathbf{x} = \mathbf{b}$:

$$\frac{\|\Delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \cdot \left(\frac{\|\Delta A\|}{\|A\|} + \frac{\|\Delta \mathbf{b}\|}{\|\mathbf{b}\|}\right)$$

### Stability Comparison

| Method            | Stability         | Notes |
| ----------------- | ----------------- | ----- |
| LU (no pivoting)  | Unstable          | May fail or give wrong answer |
| LU (partial pivoting) | Stable       | Standard choice |
| LU (complete pivoting) | Very stable | Rarely needed |
| QR (Classical Gram-Schmidt) | Moderately stable | Loses orthogonality |
| QR (Modified Gram-Schmidt) | Stable | Better, but not perfect |
| QR (Householder)  | Very stable       | Standard choice |
| QR (Givens)       | Very stable       | Good for sparse |
| Cholesky          | Stable for SPD    | Excellent when applicable |

### When to Worry

- **Regression with collinear features**: $X^TX$ is ill-conditioned → use ridge regression
- **Covariance matrices from finite samples**: May not be exactly PD → add regularization
- **Iterative refinement**: Can recover lost digits

### Improving Conditioning

1. **Regularization**: Add $\lambda I$ to make matrix better conditioned
2. **Preconditioning**: Transform $A\mathbf{x} = \mathbf{b}$ to $M^{-1}A\mathbf{x} = M^{-1}\mathbf{b}$ where $M^{-1}A$ is better conditioned
3. **Scaling**: Equilibrate row/column norms

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

### Key Decompositions at a Glance

| Decomposition | Formula     | Requirements | Unique? | Key Property |
| ------------- | ----------- | ------------ | ------- | ------------ |
| LU            | $A = LU$    | Square, nonsingular minors | Yes (with normalization) | $L$ lower, $U$ upper |
| PLU           | $PA = LU$   | Invertible | Yes | With pivoting |
| QR            | $A = QR$    | Any ($m \geq n$) | Yes (with sign convention) | $Q$ orthonormal |
| Cholesky      | $A = LL^T$  | Symmetric PD | Yes | $L$ lower with $L_{ii} > 0$ |
| LDL^T         | $A = LDL^T$ | Symmetric, nonzero pivots | Yes | No square roots |

### Complexity Comparison

```
FLOP count for n×n matrix:

Factorization:
  Cholesky:     n³/3       ← Fastest for SPD
  LU:           2n³/3
  QR (H'holder): 2n³/3
  
Solve (per RHS):
  Triangular:   n²
  Full system:  n³ (without factorization)
```

### Decision Tree

```
Solving Ax = b:
│
├── Is A symmetric positive definite?
│   └── YES → Cholesky (fastest, most stable)
│
├── Is A square?
│   └── YES → LU with partial pivoting
│
├── Is A rectangular (overdetermined)?
│   └── YES → QR for least squares
│
├── Is A structured (sparse, banded)?
│   └── YES → Specialized methods (not covered here)
│
└── Need most general/stable solution?
    └── SVD (next topic)
```

### Quick Reference for ML

| Problem                    | Best Method    | Why |
| -------------------------- | -------------- | --- |
| Solve $Ax = b$ (A square)  | LU or Cholesky | Efficient |
| Least squares $\|Ax - b\|$ | QR | Stable |
| Low-rank approximation     | SVD | Optimal |
| Sample from Gaussian       | Cholesky | Fast |
| Eigenvalue problems        | QR algorithm (iterative) | Standard |
| Ridge regression           | Cholesky | Always SPD |
| GP log-likelihood          | Cholesky | Need $\log\|$ and solves |

### Python Quick Reference

```python
import numpy as np
from scipy.linalg import lu, lu_factor, lu_solve, cholesky, solve_triangular

# LU
P, L, U = lu(A)                        # Full decomposition
lu_piv = lu_factor(A)                  # Compact form
x = lu_solve(lu_piv, b)                # Solve

# QR
Q, R = np.linalg.qr(A)                 # Reduced QR
Q, R = np.linalg.qr(A, mode='complete') # Full QR

# Cholesky  
L = np.linalg.cholesky(A)              # Lower triangular
x = solve_triangular(L.T, solve_triangular(L, b, lower=True))
```

---

## Exercises

1. Compute LU decomposition of $\begin{bmatrix} 2 & 1 \\ 4 & 5 \end{bmatrix}$ manually
2. Use QR to solve least squares: $\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}x \approx \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$
3. Compare Cholesky and LU for solving a PD system (verify Cholesky is faster)
4. Implement Householder QR for a 3×2 matrix
5. Use Cholesky to sample 1000 points from a 2D Gaussian and verify sample statistics
6. Compute the condition number of a nearly singular matrix and observe error amplification
7. Find the LDL^T decomposition of a 3×3 SPD matrix
8. Use QR decomposition for polynomial fitting (degree 2)

---

## References

1. Trefethen & Bau - "Numerical Linear Algebra" (The gold standard)
2. Golub & Van Loan - "Matrix Computations" (Comprehensive reference)
3. Press et al. - "Numerical Recipes" (Practical implementations)
4. Strang, G. - "Linear Algebra and Its Applications" (Intuitive explanations)

---

## Further Reading

- **Sparse decompositions**: Fill-reducing orderings, incomplete factorizations
- **Block decompositions**: For large structured matrices
- **Randomized methods**: Randomized QR, randomized SVD for massive matrices
- **GPU implementations**: cuSOLVER, MAGMA for parallel decompositions
