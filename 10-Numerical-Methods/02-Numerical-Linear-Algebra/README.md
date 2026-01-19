# Numerical Linear Algebra

## Overview

Numerical linear algebra provides efficient algorithms for matrix operations that underpin virtually all machine learning computations. Understanding these methods is crucial for implementing efficient and numerically stable ML algorithms.

## Prerequisites

- Linear algebra fundamentals (matrices, vectors)
- Basic calculus
- Floating-point arithmetic concepts

## Learning Objectives

- Understand matrix decomposition algorithms
- Implement numerically stable solvers
- Analyze computational complexity of linear algebra operations
- Apply sparse matrix techniques for large-scale ML

---

## 1. LU Decomposition

**LU Decomposition** factors a matrix into lower and upper triangular matrices:

$$A = LU$$

where:

- $L$ is lower triangular with ones on diagonal
- $U$ is upper triangular

### Algorithm (Gaussian Elimination)

```
For k = 1 to n-1:
    For i = k+1 to n:
        L[i,k] = A[i,k] / A[k,k]
        For j = k to n:
            A[i,j] = A[i,j] - L[i,k] * A[k,j]
```

### With Pivoting (PLU)

$$PA = LU$$

Pivoting improves numerical stability by choosing the largest pivot element.

### Applications in ML

- Solving linear systems efficiently
- Matrix inversion
- Computing determinants

### Complexity

- $O(n^3)$ for dense matrices
- Solving after factorization: $O(n^2)$

---

## 2. Cholesky Decomposition

For **symmetric positive definite** matrices:

$$A = LL^T$$

where $L$ is lower triangular.

### Algorithm

```
For i = 1 to n:
    L[i,i] = sqrt(A[i,i] - sum(L[i,k]² for k=1 to i-1))
    For j = i+1 to n:
        L[j,i] = (A[j,i] - sum(L[j,k]*L[i,k] for k=1 to i-1)) / L[i,i]
```

### Advantages over LU

- **Half the operations**: ~$n^3/6$ vs $n^3/3$
- **No pivoting needed**: Always stable for SPD matrices
- **Guaranteed to exist**: For positive definite matrices

### Applications in ML

- **Covariance matrices**: Always positive semi-definite
- **Gaussian Processes**: Inverting kernel matrices
- **Optimization**: Newton's method with Hessian

### Handling Singular Matrices

Add regularization:
$$A + \epsilon I$$

---

## 3. QR Decomposition

$$A = QR$$

where:

- $Q$ is orthogonal ($Q^TQ = I$)
- $R$ is upper triangular

### Methods

#### Gram-Schmidt Process

```
For j = 1 to n:
    v_j = a_j
    For i = 1 to j-1:
        R[i,j] = q_i · a_j
        v_j = v_j - R[i,j] * q_i
    R[j,j] = ||v_j||
    q_j = v_j / R[j,j]
```

**Modified Gram-Schmidt** is more stable:

```
For j = 1 to n:
    v_j = a_j
    For i = 1 to j-1:
        R[i,j] = q_i · v_j  # Use current v_j, not a_j
        v_j = v_j - R[i,j] * q_i
    R[j,j] = ||v_j||
    q_j = v_j / R[j,j]
```

#### Householder Reflections

Most numerically stable, uses $O(2mn^2 - 2n^3/3)$ operations.

#### Givens Rotations

Good for sparse matrices, zeroes one element at a time.

### Applications in ML

- **Least squares**: Solving overdetermined systems
- **Eigenvalue algorithms**: QR iteration
- **Orthogonalization**: Creating orthonormal bases

---

## 4. Singular Value Decomposition (SVD)

$$A = U\Sigma V^T$$

where:

- $U$ is $m \times m$ orthogonal (left singular vectors)
- $\Sigma$ is $m \times n$ diagonal (singular values)
- $V$ is $n \times n$ orthogonal (right singular vectors)

### Properties

**Singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$:

- $\sigma_1 = \|A\|_2$ (spectral norm)
- $\sigma_r$ = smallest non-zero singular value
- $r$ = rank of $A$

### Relationship to Eigendecomposition

$$A^TA = V\Sigma^T\Sigma V^T$$
$$AA^T = U\Sigma\Sigma^T U^T$$

### Thin/Economy SVD

For $m > n$:
$$A = U_{m \times n} \Sigma_{n \times n} V^T_{n \times n}$$

### Truncated SVD

Keep only $k$ largest singular values:
$$A_k = U_k \Sigma_k V_k^T$$

Best rank-$k$ approximation (Eckart-Young theorem).

### Applications in ML

1. **PCA**: Principal components are right singular vectors
2. **Matrix completion**: Netflix problem
3. **Latent semantic analysis**: Document-term matrices
4. **Image compression**: Low-rank approximation
5. **Pseudoinverse**: $A^+ = V\Sigma^+U^T$

### Complexity

- Full SVD: $O(\min(mn^2, m^2n))$
- Truncated: Can be much faster

---

## 5. Eigendecomposition

For square matrix $A$:

$$A = V\Lambda V^{-1}$$

For symmetric $A$:
$$A = Q\Lambda Q^T$$

where $Q$ is orthogonal.

### Power Iteration

Find dominant eigenvalue/eigenvector:

```
v = random unit vector
For k = 1 to max_iter:
    w = A @ v
    v = w / ||w||
    λ = v · (A @ v)
```

Convergence rate: $|\lambda_2/\lambda_1|^k$

### Inverse Iteration

Find eigenvalue closest to $\mu$:

```
For k = 1 to max_iter:
    Solve (A - μI)w = v
    v = w / ||w||
```

### QR Algorithm

Find all eigenvalues:

```
A_0 = A
For k = 0, 1, 2, ...:
    Q_k, R_k = QR(A_k)
    A_{k+1} = R_k @ Q_k
```

$A_k$ converges to upper triangular with eigenvalues on diagonal.

### Applications in ML

- **PCA**: Covariance matrix eigenvectors
- **Spectral clustering**: Graph Laplacian eigenvectors
- **PageRank**: Dominant eigenvector
- **Stability analysis**: Hessian eigenvalues

---

## 6. Linear System Solvers

### Direct Methods

| Method   | When to Use                 | Complexity |
| -------- | --------------------------- | ---------- |
| LU       | General matrices            | $O(n^3)$   |
| Cholesky | Symmetric positive definite | $O(n^3/3)$ |
| QR       | Overdetermined systems      | $O(mn^2)$  |

### Iterative Methods

For large sparse systems where direct methods are too expensive.

#### Jacobi Method

$$x^{(k+1)}_i = \frac{1}{a_{ii}}\left(b_i - \sum_{j \neq i} a_{ij} x^{(k)}_j\right)$$

Matrix form: $x^{(k+1)} = D^{-1}(b - (L+U)x^{(k)})$

#### Gauss-Seidel

Use updated values immediately:
$$x^{(k+1)}_i = \frac{1}{a_{ii}}\left(b_i - \sum_{j < i} a_{ij} x^{(k+1)}_j - \sum_{j > i} a_{ij} x^{(k)}_j\right)$$

#### Conjugate Gradient

For symmetric positive definite $A$:

```
r_0 = b - A @ x_0
p_0 = r_0
For k = 0, 1, 2, ...:
    α_k = (r_k · r_k) / (p_k · A @ p_k)
    x_{k+1} = x_k + α_k * p_k
    r_{k+1} = r_k - α_k * A @ p_k
    β_k = (r_{k+1} · r_{k+1}) / (r_k · r_k)
    p_{k+1} = r_{k+1} + β_k * p_k
```

Converges in at most $n$ iterations (in exact arithmetic).

### Preconditioning

Transform $Ax = b$ to $M^{-1}Ax = M^{-1}b$ where $M \approx A$.

Good preconditioners:

- Diagonal (Jacobi)
- Incomplete LU (ILU)
- Incomplete Cholesky

---

## 7. Sparse Matrix Operations

### Storage Formats

#### Compressed Sparse Row (CSR)

```
data:    non-zero values
indices: column indices
indptr:  row pointers
```

Example:

```
[1 0 2]     data = [1, 2, 3, 4]
[0 3 0] →  indices = [0, 2, 1, 1]
[0 4 0]     indptr = [0, 2, 3, 4]
```

#### Compressed Sparse Column (CSC)

Same but columns instead of rows.

#### COO (Coordinate)

```
row:  [0, 0, 1, 2]
col:  [0, 2, 1, 1]
data: [1, 2, 3, 4]
```

### Sparse Operations Complexity

| Operation     | Dense    | Sparse (nnz)     |
| ------------- | -------- | ---------------- |
| Matrix-vector | $O(n^2)$ | $O(nnz)$         |
| Matrix-matrix | $O(n^3)$ | $O(nnz \cdot n)$ |
| Transpose     | $O(n^2)$ | $O(nnz)$         |

### When to Use Sparse

Rule of thumb: Use sparse when density < 10-20%

$$\text{density} = \frac{\text{nnz}}{m \times n}$$

---

## 8. Numerical Stability Considerations

### Backward Stability

An algorithm is backward stable if:
$$\tilde{f}(x) = f(x + \delta x), \quad |\delta x| \leq \epsilon |x|$$

### Condition Number

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

Error amplification:
$$\frac{\|\delta x\|}{\|x\|} \leq \kappa(A) \frac{\|\delta b\|}{\|b\|}$$

### Numerical Rank

Practical rank considering tolerance:
$$\text{rank}_\epsilon(A) = \#\{i : \sigma_i > \epsilon \cdot \sigma_1\}$$

### Regularization for Ill-Conditioned Systems

**Tikhonov regularization**:
$$(A^TA + \lambda I)x = A^Tb$$

**Truncated SVD**:
Use only singular values > threshold.

---

## 9. Matrix Operations Complexity

### Standard Algorithms

| Operation              | Complexity            |
| ---------------------- | --------------------- |
| Matrix-vector multiply | $O(n^2)$              |
| Matrix-matrix multiply | $O(n^3)$              |
| LU decomposition       | $O(n^3)$              |
| Matrix inverse         | $O(n^3)$              |
| Eigendecomposition     | $O(n^3)$              |
| SVD                    | $O(\min(mn^2, m^2n))$ |

### Fast Algorithms

**Strassen's Algorithm**: Matrix multiply in $O(n^{2.807})$

- Rarely used in practice due to numerical issues

**Randomized Algorithms**: Approximate SVD in $O(mn\log k)$ for rank-$k$

---

## 10. GPU Acceleration

### cuBLAS Operations

| Level | Operations    | Examples        |
| ----- | ------------- | --------------- |
| 1     | Vector-vector | dot, axpy, norm |
| 2     | Matrix-vector | gemv, trsv      |
| 3     | Matrix-matrix | gemm, trsm      |

### Batched Operations

Process many small matrices in parallel:

```python
# Batch of 1000 32×32 matrices
A = np.random.randn(1000, 32, 32)
np.linalg.inv(A)  # Batched inversion
```

### Memory Hierarchy

```
GPU Register → Shared Memory → L1/L2 Cache → Global Memory
    (fast)                                      (slow)
```

Tiled algorithms maximize cache usage.

---

## Summary

### Decomposition Selection Guide

| Problem         | Decomposition | Conditions     |
| --------------- | ------------- | -------------- |
| Solve $Ax = b$  | LU            | General square |
| Solve $Ax = b$  | Cholesky      | Symmetric PD   |
| Least squares   | QR            | Overdetermined |
| Low-rank approx | SVD           | Any matrix     |
| Eigenvalues     | Eigendecomp   | Square         |

### ML Applications

| Technique          | Used In                   |
| ------------------ | ------------------------- |
| SVD                | PCA, matrix factorization |
| Cholesky           | Gaussian processes        |
| QR                 | Linear regression         |
| Conjugate gradient | Large-scale optimization  |
| Sparse operations  | Graph neural networks     |

### Best Practices

1. **Use library implementations**: NumPy/SciPy are highly optimized
2. **Choose appropriate decomposition**: Match problem structure
3. **Exploit structure**: Sparse, symmetric, positive definite
4. **Monitor condition number**: Regularize if needed
5. **Consider iterative methods**: For very large systems

---

## References

1. Trefethen & Bau - "Numerical Linear Algebra"
2. Golub & Van Loan - "Matrix Computations"
3. Higham - "Accuracy and Stability of Numerical Algorithms"
4. Saad - "Iterative Methods for Sparse Linear Systems"
