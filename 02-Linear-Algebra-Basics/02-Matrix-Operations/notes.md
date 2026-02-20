# Matrix Operations

## Overview

Matrices are the fundamental data structure for organizing and transforming data in machine learning. Every neural network layer, every image, and every dataset can be represented as a matrix. Understanding matrix operations is essential for:

- **Neural Networks**: Weight matrices transform inputs through layers
- **Computer Vision**: Images are stored and processed as matrices
- **NLP**: Word embeddings and attention matrices are core to transformers
- **Optimization**: Hessian and Jacobian matrices guide training
- **Data Processing**: Covariance matrices capture feature relationships

## Learning Objectives

After completing this section, you will be able to:

- Understand matrix representations, notations, and terminology
- Master basic matrix arithmetic operations (addition, multiplication, transpose)
- Recognize and work with special matrix types (symmetric, orthogonal, positive definite)
- Apply element-wise operations and broadcasting
- Implement matrix operations for ML applications
- Understand computational complexity and optimization strategies
- Debug common matrix dimension errors

## Prerequisites

- [Number Systems](../../01-Mathematical-Foundations/01-Number-Systems/README.md)
- [Vectors and Spaces](../01-Vectors-and-Spaces/README.md)

---

## 1. Matrix Fundamentals

### 1.1 What is a Matrix?

A **matrix** is a rectangular array of numbers arranged in rows and columns.

$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Notation:**

- $A \in \mathbb{R}^{m \times n}$ - matrix with $m$ rows and $n$ columns
- $a_{ij}$ or $A_{ij}$ - element at row $i$, column $j$
- $[A]_{ij}$ - alternative notation for element access

### 1.2 Matrix as Data Structure

```
Matrix A (3×4):
                    Columns (features)
                    j = 1   j = 2   j = 3   j = 4
              ┌─────────────────────────────────────┐
Rows      i=1 │  a₁₁     a₁₂     a₁₃     a₁₄      │
(samples) i=2 │  a₂₁     a₂₂     a₂₃     a₂₄      │
          i=3 │  a₃₁     a₃₂     a₃₃     a₃₄      │
              └─────────────────────────────────────┘

In ML: Rows = samples/observations
       Columns = features/variables
```

### 1.3 Special Matrices

```
Square Matrix (n×n)          Diagonal Matrix              Identity Matrix (I)
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│ a₁₁  a₁₂  a₁₃  │          │ d₁   0    0    │          │  1    0    0   │
│ a₂₁  a₂₂  a₂₃  │          │  0   d₂   0    │          │  0    1    0   │
│ a₃₁  a₃₂  a₃₃  │          │  0    0   d₃   │          │  0    0    1   │
└─────────────────┘          └─────────────────┘          └─────────────────┘

Zero Matrix (0)              Symmetric (A = Aᵀ)          Upper Triangular
┌─────────────────┐          ┌─────────────────┐          ┌─────────────────┐
│  0    0    0   │          │  1    2    3    │          │ a₁₁  a₁₂  a₁₃  │
│  0    0    0   │          │  2    4    5    │          │  0   a₂₂  a₂₃  │
│  0    0    0   │          │  3    5    6    │          │  0    0   a₃₃  │
└─────────────────┘          └─────────────────┘          └─────────────────┘
```

### 1.4 Detailed Matrix Types Reference

#### Identity Matrix ($I$ or $I_n$)

The multiplicative identity: $AI = IA = A$ for any compatible matrix $A$.

$$I_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Properties:**
- $I^n = I$ for any power
- $\det(I) = 1$
- All eigenvalues are 1
- $I^{-1} = I$

**ML Use:** Regularization terms often use $\lambda I$.

#### Diagonal Matrix

Only non-zero elements are on the main diagonal.

$$D = \text{diag}(d_1, d_2, \ldots, d_n)$$

**Properties:**
- $D^n = \text{diag}(d_1^n, d_2^n, \ldots, d_n^n)$
- $D^{-1} = \text{diag}(1/d_1, 1/d_2, \ldots, 1/d_n)$ if all $d_i \neq 0$
- Multiplication is $O(n)$ instead of $O(n^2)$

**ML Use:** Scaling transformations, variance matrices.

#### Triangular Matrices

**Upper Triangular (U):** All entries below main diagonal are zero.

$$U = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}$$

**Lower Triangular (L):** All entries above main diagonal are zero.

$$L = \begin{bmatrix} \ell_{11} & 0 & 0 \\ \ell_{21} & \ell_{22} & 0 \\ \ell_{31} & \ell_{32} & \ell_{33} \end{bmatrix}$$

**Properties:**
- $\det(U) = \prod_i u_{ii}$ (product of diagonal)
- Product of triangular matrices is triangular
- Inverse of triangular is triangular
- Solving $Ux = b$ is $O(n^2)$ via back-substitution

**ML Use:** LU decomposition, Cholesky decomposition.

#### Symmetric Matrix

A matrix equal to its transpose: $A = A^T$

$$A = \begin{bmatrix} a & b & c \\ b & d & e \\ c & e & f \end{bmatrix}$$

**Properties:**
- All eigenvalues are real
- Eigenvectors for distinct eigenvalues are orthogonal
- Can be diagonalized by orthogonal matrix: $A = QDQ^T$
- Only $n(n+1)/2$ unique elements (upper or lower half)

**ML Use:**
- Covariance matrices: $\Sigma = \text{Cov}(X)$
- Gram matrices: $K = X^T X$
- Hessian matrices: $H = \nabla^2 f$
- Kernel matrices in SVMs

#### Skew-Symmetric (Anti-Symmetric) Matrix

A matrix where $A = -A^T$

$$A = \begin{bmatrix} 0 & a & b \\ -a & 0 & c \\ -b & -c & 0 \end{bmatrix}$$

**Properties:**
- Diagonal elements must be zero
- All eigenvalues are purely imaginary or zero
- $e^A$ is an orthogonal matrix (rotation)

**ML Use:** Rotation representations, Lie algebra.

#### Orthogonal Matrix

A square matrix where $Q^T Q = QQ^T = I$, equivalently $Q^{-1} = Q^T$.

**Properties:**
- Columns (and rows) are orthonormal vectors
- $\det(Q) = \pm 1$
- Preserves vector lengths: $\|Qx\| = \|x\|$
- Preserves angles between vectors
- Eigenvalues have magnitude 1

**ML Use:**
- PCA transformation matrices
- Orthogonal weight initialization
- Householder reflections
- Gram-Schmidt orthogonalization

#### Positive Definite Matrix

A symmetric matrix where $x^T A x > 0$ for all non-zero vectors $x$.

**Properties:**
- All eigenvalues are positive
- All leading principal minors are positive
- Unique Cholesky decomposition: $A = LL^T$
- Convex quadratic: $f(x) = x^T A x$ is strictly convex

**Positive Semi-Definite (PSD):** $x^T A x \geq 0$ (eigenvalues $\geq 0$)

**ML Use:**
- Covariance matrices are PSD
- Kernel matrices must be PSD
- Hessian is PD at strict local minimum
- Regularized matrices: $X^T X + \lambda I$

#### Sparse Matrix

A matrix with mostly zero elements.

```
Sparse Matrix (90% zeros):
┌─────────────────────┐
│ 0   3   0   0   0  │
│ 0   0   0   7   0  │
│ 2   0   0   0   0  │
│ 0   0   4   0   0  │
│ 0   0   0   0   1  │
└─────────────────────┘
```

**Storage Formats:**
- COO: Coordinate list (row, col, value)
- CSR: Compressed Sparse Row
- CSC: Compressed Sparse Column

**ML Use:** Adjacency matrices, one-hot encodings, attention masks.

#### Stochastic Matrix

A matrix where each row sums to 1 (row-stochastic) or each column sums to 1 (column-stochastic).

**Properties:**
- Represents probability transitions
- Largest eigenvalue is 1
- Stationary distribution is eigenvector for eigenvalue 1

**ML Use:** Markov chains, PageRank, attention weights.

---

## 2. Basic Matrix Operations

### 2.1 Matrix Addition and Subtraction

Two matrices can be added/subtracted only if they have the **same dimensions**.

$$C = A + B \text{ where } c_{ij} = a_{ij} + b_{ij}$$

```
Addition Example:

  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │  1   2  │  +  │  5   6  │  =  │  6   8  │
  │  3   4  │     │  7   8  │     │ 10  12  │
  └─────────┘     └─────────┘     └─────────┘
      A      +        B       =       C
```

**Properties:**

- Commutative: $A + B = B + A$
- Associative: $(A + B) + C = A + (B + C)$
- Identity: $A + 0 = A$
- Inverse: $A + (-A) = 0$

### 2.2 Scalar Multiplication

$$B = cA \text{ where } b_{ij} = c \cdot a_{ij}$$

```
Scalar Multiplication:

       ┌─────────┐         ┌─────────┐
  2 ×  │  1   2  │    =    │  2   4  │
       │  3   4  │         │  6   8  │
       └─────────┘         └─────────┘
```

### 2.3 Matrix Multiplication

For $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times p}$, the product $C = AB$ has dimensions $m \times p$.

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$$

**Dimension Rule:** $(m \times \mathbf{n}) \cdot (\mathbf{n} \times p) = (m \times p)$

```
Matrix Multiplication Visualization:

        B (2×3)                          C = AB (2×3)
        ┌─────────────┐                  ┌─────────────┐
        │ b₁₁ b₁₂ b₁₃│                  │ c₁₁ c₁₂ c₁₃│
        │ b₂₁ b₂₂ b₂₃│                  │ c₂₁ c₂₂ c₂₃│
        └─────────────┘                  └─────────────┘
              ↓↓↓                              ↓
    A (2×2)   columns                   c₁₂ = row₁(A) · col₂(B)
    ┌───────┐ of B                         = a₁₁b₁₂ + a₁₂b₂₂
    │ a₁₁ a₁₂│ → rows
    │ a₂₁ a₂₂│   of A
    └───────┘
```

**Example Calculation:**

$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

### 2.4 Matrix Multiplication Properties

**⚠️ NOT Commutative:** $AB \neq BA$ (in general)

```
Example where AB ≠ BA:

A = [1 2]    B = [0 1]
    [0 0]        [0 0]

AB = [0 1]   BA = [0 0]
     [0 0]        [0 0]
```

**Other Properties:**

- Associative: $(AB)C = A(BC)$
- Distributive: $A(B + C) = AB + AC$
- Scalar: $(cA)B = c(AB) = A(cB)$
- Identity: $AI = IA = A$
- Transpose: $(AB)^T = B^T A^T$

### 2.5 Matrix Multiplication Perspectives

There are four equivalent ways to understand matrix multiplication:

**1. Row-Column Dot Products** (Standard Definition)

$$c_{ij} = \sum_k a_{ik} b_{kj} = \text{row}_i(A) \cdot \text{col}_j(B)$$

**2. Linear Combination of Columns**

Each column of $C$ is a linear combination of columns of $A$:

$$C[:, j] = A \cdot B[:, j] = \sum_k b_{kj} \cdot A[:, k]$$

**3. Linear Combination of Rows**

Each row of $C$ is a linear combination of rows of $B$:

$$C[i, :] = A[i, :] \cdot B = \sum_k a_{ik} \cdot B[k, :]$$

**4. Sum of Outer Products**

$$C = AB = \sum_{k=1}^{n} A[:, k] \otimes B[k, :]$$

```
Matrix Multiplication as Outer Products:

A = [a₁ | a₂]   B = [b₁ᵀ]     AB = a₁b₁ᵀ + a₂b₂ᵀ
                    [b₂ᵀ]

where aₖ are columns of A and bₖᵀ are rows of B
```

### 2.6 Matrix-Vector Multiplication

Special case when $B$ is a column vector:

$$y = Ax \text{ where } y_i = \sum_j a_{ij} x_j$$

**Two Interpretations:**

1. **Dot products:** $y_i = \text{row}_i(A) \cdot x$
2. **Linear combination:** $y = x_1 \cdot \text{col}_1(A) + x_2 \cdot \text{col}_2(A) + \cdots$

```
     [a₁₁  a₁₂  a₁₃]   [x₁]     [a₁₁x₁ + a₁₂x₂ + a₁₃x₃]
y =  [a₂₁  a₂₂  a₂₃] × [x₂]  =  [a₂₁x₁ + a₂₂x₂ + a₂₃x₃]
     [a₃₁  a₃₂  a₃₃]   [x₃]     [a₃₁x₁ + a₃₂x₂ + a₃₃x₃]
```

### 2.7 Block Matrix Multiplication

Matrices can be partitioned into blocks and multiplied using the same rules:

$$\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} \begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix} = \begin{bmatrix} A_{11}B_{11} + A_{12}B_{21} & A_{11}B_{12} + A_{12}B_{22} \\ A_{21}B_{11} + A_{22}B_{21} & A_{21}B_{12} + A_{22}B_{22} \end{bmatrix}$$

**Strassen's Algorithm** uses clever block multiplication to achieve $O(n^{2.807})$ instead of $O(n^3)$.

---

## 3. Matrix Transpose

### 3.1 Definition

The transpose of $A \in \mathbb{R}^{m \times n}$ is $A^T \in \mathbb{R}^{n \times m}$ where $(A^T)_{ij} = A_{ji}$.

```
Original A (2×3)              Transpose Aᵀ (3×2)
┌─────────────────┐           ┌───────────┐
│  1    2    3    │   Aᵀ     │  1    4   │
│  4    5    6    │  ───→    │  2    5   │
└─────────────────┘           │  3    6   │
                              └───────────┘
Rows become columns, columns become rows
```

### 3.2 Transpose Properties

| Property         | Formula                 |
| ---------------- | ----------------------- |
| Double transpose | $(A^T)^T = A$           |
| Sum              | $(A + B)^T = A^T + B^T$ |
| Scalar           | $(cA)^T = cA^T$         |
| Product          | $(AB)^T = B^T A^T$      |

### 3.3 Symmetric Matrices

A matrix is **symmetric** if $A = A^T$ (only for square matrices).

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 5 \\ 3 & 5 & 6 \end{bmatrix} = A^T$$

**ML Applications:**

- Covariance matrices are symmetric
- Gram matrices ($X^T X$) are symmetric
- Distance matrices are symmetric

### 3.4 Symmetric and Skew-Symmetric Decomposition

Any square matrix can be uniquely decomposed:

$$A = S + K$$

where:
- $S = \frac{A + A^T}{2}$ is symmetric ($S = S^T$)
- $K = \frac{A - A^T}{2}$ is skew-symmetric ($K = -K^T$)

**Example:**
$$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

$$S = \frac{1}{2}\begin{bmatrix} 2 & 5 \\ 5 & 8 \end{bmatrix} = \begin{bmatrix} 1 & 2.5 \\ 2.5 & 4 \end{bmatrix}$$

$$K = \frac{1}{2}\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix} = \begin{bmatrix} 0 & -0.5 \\ 0.5 & 0 \end{bmatrix}$$

### 3.5 Conjugate Transpose (Hermitian Transpose)

For complex matrices, the conjugate transpose is denoted $A^*$ or $A^H$:

$$(A^*)_{ij} = \overline{A_{ji}}$$

A matrix is **Hermitian** if $A = A^*$ (complex analog of symmetric).

---

## 4. Matrix Norms and Distance

### 4.1 Frobenius Norm

The most common matrix norm, treating the matrix as a vector:

$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\text{trace}(A^T A)}$$

**Properties:**
- Submultiplicative: $\|AB\|_F \leq \|A\|_F \|B\|_F$
- Unitarily invariant: $\|UAV\|_F = \|A\|_F$ for orthogonal $U$, $V$
- Equals L2 norm of singular values: $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$

### 4.2 Induced (Operator) Norms

$$\|A\|_p = \max_{x \neq 0} \frac{\|Ax\|_p}{\|x\|_p}$$

**Common Cases:**
- **Spectral norm** ($p=2$): $\|A\|_2 = \sigma_{\max}(A)$ (largest singular value)
- **Max row sum** ($p=\infty$): $\|A\|_\infty = \max_i \sum_j |a_{ij}|$
- **Max column sum** ($p=1$): $\|A\|_1 = \max_j \sum_i |a_{ij}|$

### 4.3 Nuclear Norm (Trace Norm)

$$\|A\|_* = \sum_i \sigma_i = \text{trace}(\sqrt{A^T A})$$

**ML Use:** Promotes low-rank solutions in matrix completion, collaborative filtering.

### 4.4 Matrix Distance

Distance between matrices:
$$d(A, B) = \|A - B\|_F$$

Used for:
- Measuring reconstruction error
- Weight updates in training
- Matrix approximation quality

---

## 5. Element-wise Operations

### 5.1 Hadamard Product (Element-wise Multiplication)

$$C = A \odot B \text{ where } c_{ij} = a_{ij} \cdot b_{ij}$$

```
Hadamard Product:

  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │  1   2  │  ⊙  │  5   6  │  =  │  5  12  │
  │  3   4  │     │  7   8  │     │ 21  32  │
  └─────────┘     └─────────┘     └─────────┘
```

**ML Use:** Attention mechanisms, gating in LSTMs/GRUs, dropout masks.

**Properties:**
- Commutative: $A \odot B = B \odot A$
- Associative: $(A \odot B) \odot C = A \odot (B \odot C)$
- Identity: $A \odot \mathbf{1} = A$ (where $\mathbf{1}$ is all-ones matrix)
- Distributive over addition: $A \odot (B + C) = A \odot B + A \odot C$

**Hadamard Identity:**
$$\sum_{i,j} (A \odot B)_{ij} = \text{trace}(A^T B)$$

### 5.2 Element-wise Division

$$C = A \oslash B \text{ where } c_{ij} = a_{ij} / b_{ij}$$

**ML Use:** Normalizing by feature-wise statistics, batch normalization.

### 5.3 Element-wise Functions

Apply a scalar function to each element:

$$f(A)_{ij} = f(a_{ij})$$

**Common examples:**
- $\exp(A)$ - element-wise exponential (softmax numerator)
- $\log(A)$ - element-wise log (cross-entropy)
- $\sqrt{A}$ - element-wise square root
- $\text{ReLU}(A) = \max(0, A)$ - activation function

**⚠️ Note:** Element-wise $\exp(A)$ is different from matrix exponential $e^A$!

### 5.4 Broadcasting

When dimensions don't match exactly, smaller arrays are "broadcast" to match.

```
Broadcasting Examples:

Matrix + Scalar:               Matrix + Vector (row):
┌─────────┐                    ┌─────────┐   ┌─────┐
│  1   2  │  + 10  =          │  1   2  │ + │3  4│  =
│  3   4  │                    │  5   6  │   └─────┘
└─────────┘                    └─────────┘
┌─────────┐                    ┌─────────┐
│ 11  12  │                    │  4   6  │
│ 13  14  │                    │  8  10  │
└─────────┘                    └─────────┘
```

**Broadcasting Rules:**
1. Compare dimensions from right to left
2. Dimensions must be equal OR one of them must be 1
3. Missing dimensions are treated as 1

```
Shape Compatibility Examples:

(3, 4)  + (4,)    → (3, 4)  ✓  Vector broadcast to each row
(3, 4)  + (3, 1)  → (3, 4)  ✓  Column broadcast to each column
(3, 4)  + (1, 4)  → (3, 4)  ✓  Row broadcast to each row
(3, 4)  + (3,)    → ERROR   ✗  Incompatible (3 ≠ 4)
(2,3,4) + (3, 4)  → (2,3,4) ✓  Broadcast over batch dimension
```

**ML Use:**
- Adding bias vector to all samples
- Feature-wise normalization
- Attention score masking

---

## 6. Matrix Views and Slicing

### 6.1 Row and Column Vectors

```
Matrix A:                 Row Vector (1×n)       Column Vector (m×1)
┌─────────────┐           ┌─────────────┐        ┌─────┐
│ a₁₁ a₁₂ a₁₃│  Row 1 →  │ a₁₁ a₁₂ a₁₃│        │ a₁₁ │
│ a₂₁ a₂₂ a₂₃│                                   │ a₂₁ │  ← Col 1
│ a₃₁ a₃₂ a₃₃│                                   │ a₃₁ │
└─────────────┘                                   └─────┘
```

### 6.2 Submatrices

```
Extracting a 2×2 submatrix from rows 1-2, cols 2-3:

┌─────────────────────┐
│ a₁₁  [a₁₂  a₁₃] a₁₄│      ┌───────────┐
│ a₂₁  [a₂₂  a₂₃] a₂₄│  →   │ a₁₂  a₁₃ │
│ a₃₁   a₃₂  a₃₃  a₃₄│      │ a₂₂  a₂₃ │
└─────────────────────┘      └───────────┘
```

### 6.3 NumPy Indexing Reference

```python
# Basic indexing
A[i, j]          # Single element
A[i, :]          # Row i (1D array)
A[:, j]          # Column j (1D array)
A[i:j, k:l]      # Submatrix (rows i to j-1, cols k to l-1)

# Fancy indexing
A[[0, 2, 4], :]  # Rows 0, 2, 4
A[:, [1, 3]]     # Columns 1, 3
A[A > 0]         # All positive elements (flattened)

# Boolean masking
mask = A > 0
A[mask] = 0      # Set all positive to 0

# Slicing with step
A[::2, :]        # Every other row
A[::-1, :]       # Rows in reverse order
```

### 6.4 Diagonal Operations

```python
# Extract main diagonal
np.diag(A)                    # → 1D array of diagonal

# Create diagonal matrix from vector
np.diag([1, 2, 3])            # → 3×3 diagonal matrix

# Extract k-th diagonal
np.diag(A, k=1)               # First super-diagonal
np.diag(A, k=-1)              # First sub-diagonal

# Sum of diagonal = trace
np.trace(A)                   # Equivalent to np.sum(np.diag(A))
```

### 6.5 Reshaping and Flattening

```python
# Reshape (total elements must match)
A.reshape(m, n)               # New shape
A.reshape(-1)                 # Flatten to 1D
A.reshape(-1, 1)              # Column vector
A.reshape(1, -1)              # Row vector

# Flatten (always creates copy)
A.flatten()                   # C-order (row-major)
A.flatten('F')                # Fortran-order (column-major)

# Ravel (view when possible)
A.ravel()                     # More memory-efficient than flatten
```

---

## 7. Trace and Determinant Preview

### 7.1 Trace

The trace is the sum of diagonal elements:

$$\text{trace}(A) = \sum_{i=1}^{n} a_{ii}$$

**Properties:**
- $\text{trace}(A + B) = \text{trace}(A) + \text{trace}(B)$
- $\text{trace}(cA) = c \cdot \text{trace}(A)$
- $\text{trace}(AB) = \text{trace}(BA)$ (cyclic property)
- $\text{trace}(A^T) = \text{trace}(A)$
- $\text{trace}(A) = \sum_i \lambda_i$ (sum of eigenvalues)

**Generalized Cyclic Property:**
$$\text{trace}(ABC) = \text{trace}(BCA) = \text{trace}(CAB)$$

**ML Applications:**
- Frobenius norm: $\|A\|_F^2 = \text{trace}(A^T A)$
- Matrix derivatives often involve trace
- KL divergence between Gaussians uses trace

### 7.2 Determinant (Preview)

The determinant is a scalar measuring "volume scaling" of a linear transformation.

For $2 \times 2$:
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

For $3 \times 3$ (Sarrus' rule):
$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = aei + bfg + cdh - ceg - bdi - afh$$

**Key Properties:**
- $\det(I) = 1$
- $\det(AB) = \det(A) \cdot \det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = 1/\det(A)$
- $\det(cA) = c^n \det(A)$ for $n \times n$ matrix
- $\det(A) = \prod_i \lambda_i$ (product of eigenvalues)

**Geometric Interpretation:**
- $|\det(A)|$ = volume scaling factor
- $\det(A) < 0$ = orientation reversal

---

## 8. Machine Learning Applications

### 8.1 Data Representation

```
Dataset X (n samples × d features):

              Feature 1  Feature 2  ...  Feature d
            ┌──────────────────────────────────────┐
Sample 1    │   x₁₁       x₁₂      ...    x₁ₐ     │
Sample 2    │   x₂₁       x₂₂      ...    x₂ₐ     │
   ⋮        │    ⋮         ⋮       ⋱      ⋮       │
Sample n    │   xₙ₁       xₙ₂      ...    xₙₐ     │
            └──────────────────────────────────────┘

Example: MNIST digit images
- n = 60,000 training samples
- d = 784 features (28×28 pixels)
- X ∈ ℝ^(60000×784)
```

### 8.2 Linear Layer Transformation

A neural network linear layer: $y = Wx + b$

```
Input      ×      Weights      +    Bias      =    Output
 x              W                   b               y

[x₁]         [w₁₁ w₁₂ w₁₃]       [b₁]          [y₁]
[x₂]    ×    [w₂₁ w₂₂ w₂₃]   +   [b₂]    =     [y₂]
[x₃]

(3×1)        (2×3)               (2×1)         (2×1)
```

### 8.3 Batch Processing

```
Processing multiple samples at once:

Batch X (32 samples × 784 features)
                    ↓
          Linear Layer (784 → 256)
              W: 256 × 784
              b: 256 × 1
                    ↓
            Y = XWᵀ + b
        (32 × 256) output
```

### 8.4 Covariance Matrix

$$\Sigma = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$

```
Covariance Matrix Properties:
- Symmetric: Σ = Σᵀ
- Size: d × d (number of features)
- Diagonal: variances of each feature
- Off-diagonal: covariances between features
```

### 8.5 Image as Matrix

```
Grayscale Image (Height × Width):
┌─────────────────────────────┐
│ 0.1  0.2  0.3  ...  0.8    │  ← Row of pixels
│ 0.2  0.4  0.5  ...  0.7    │
│  ⋮    ⋮    ⋮   ⋱    ⋮      │
│ 0.9  0.8  0.6  ...  0.1    │
└─────────────────────────────┘

RGB Image: 3 matrices (H × W × 3)
- Red channel: H × W matrix
- Green channel: H × W matrix
- Blue channel: H × W matrix
```

### 8.6 Attention Mechanism

The attention mechanism in transformers uses extensive matrix operations:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```
Query Q: (seq_len, d_k)
Key K:   (seq_len, d_k)
Value V: (seq_len, d_v)

Step 1: QKᵀ → (seq_len, seq_len)  # Attention scores
Step 2: softmax(scores / √d_k)    # Normalize
Step 3: scores × V → (seq_len, d_v)  # Weighted values
```

### 8.7 Gram Matrix

The Gram matrix captures style information in neural style transfer:

$$G = F^T F$$

where $F$ is the feature map matrix (channels × spatial locations).

**Properties:**
- Always symmetric and positive semi-definite
- Size: (channels × channels)
- Captures feature correlations

### 8.8 Kernel Matrix

In kernel methods (SVM, Gaussian Processes):

$$K_{ij} = k(x_i, x_j)$$

where $k$ is the kernel function (e.g., RBF, polynomial).

**Properties:**
- Symmetric: $K = K^T$
- Positive semi-definite
- Size: (n_samples × n_samples)

---

## 9. Computational Considerations

### 9.1 Time Complexity

| Operation                           | Complexity   | Notes                           |
| ----------------------------------- | ------------ | ------------------------------- |
| Addition (m×n)                      | $O(mn)$      |                                 |
| Scalar multiplication               | $O(mn)$      |                                 |
| Element-wise operations             | $O(mn)$      |                                 |
| Matrix-vector (m×n) × (n×1)         | $O(mn)$      |                                 |
| Matrix multiplication (m×n) × (n×p) | $O(mnp)$     | Naïve algorithm                 |
| Strassen's algorithm                | $O(n^{2.81})$| For large square matrices       |
| Matrix inversion                    | $O(n^3)$     | Gauss-Jordan                    |
| Determinant                         | $O(n^3)$     | LU decomposition                |
| Eigenvalue decomposition            | $O(n^3)$     |                                 |

### 9.2 Memory Layout

```
Row-major (C, NumPy default):        Column-major (Fortran):
┌─────────┐                          ┌─────────┐
│ 1  2  3 │ Memory: [1,2,3,4,5,6]   │ 1  2  3 │ Memory: [1,4,2,5,3,6]
│ 4  5  6 │                          │ 4  5  6 │
└─────────┘                          └─────────┘

Consequence: Row access is fast in C-order, column access is fast in Fortran-order.
```

### 9.3 Optimization Tips

1. **Matrix multiplication order matters:**
   - $(AB)C$ vs $A(BC)$ can have very different costs
   - Example: $A(100\times2), B(2\times100), C(100\times2)$
     - $(AB)C$: $(100\times100) \times (100\times2)$ = 2,000,000 ops
     - $A(BC)$: $(100\times2) \times (2\times2)$ = 800 ops

2. **Avoid explicit loops** - use vectorized operations
3. **Pre-allocate** memory for large matrices
4. **Use appropriate data types** - float32 vs float64
5. **Leverage symmetry** - only compute upper/lower triangle
6. **Use sparse matrices** when density < 10%
7. **Consider GPU acceleration** for large matrices

### 9.4 Numerical Stability

**Common Issues:**
- **Catastrophic cancellation**: Subtracting similar values
- **Overflow/underflow**: Very large/small numbers
- **Ill-conditioned matrices**: Small perturbations cause large output changes

**Best Practices:**
```python
# Use log-sum-exp for numerical stability
# Instead of: np.log(np.sum(np.exp(x)))
# Use:
from scipy.special import logsumexp
result = logsumexp(x)

# Use condition number to check stability
cond = np.linalg.cond(A)
if cond > 1e10:
    print("Warning: Ill-conditioned matrix")
```

---

## 10. Common Patterns and Best Practices

### 10.1 Dimension Debugging

```python
def debug_shapes(A, B, operation="multiply"):
    """Helper to debug matrix dimension errors."""
    print(f"A shape: {A.shape}")
    print(f"B shape: {B.shape}")
    
    if operation == "multiply":
        if A.shape[1] != B.shape[0]:
            print(f"ERROR: A columns ({A.shape[1]}) != B rows ({B.shape[0]})")
            print(f"Result would be: ({A.shape[0]}, {B.shape[1]})")
    elif operation == "add":
        if A.shape != B.shape:
            print(f"ERROR: Shapes don't match for addition")
```

### 10.2 Common Mistakes

| Mistake | Description | Fix |
|---------|-------------|-----|
| `A * B` instead of `A @ B` | Element-wise vs matrix multiply | Use `@` for matrix multiplication |
| Transposition order | $(AB)^T \neq A^T B^T$ | Remember: $(AB)^T = B^T A^T$ |
| Broadcasting confusion | Unexpected dimension expansion | Check shapes explicitly |
| In-place modification | Changing matrix during iteration | Use `.copy()` when needed |
| Integer overflow | Large matrix products | Use `dtype=np.float64` |

### 10.3 Efficient Patterns

```python
# Compute X^T X efficiently (symmetric result)
XTX = X.T @ X  # NumPy optimizes for symmetric output

# Batch matrix-vector multiplication
# Instead of: [W @ x for x in batch]
result = batch @ W.T  # (batch_size, out_features)

# Compute trace of product without full multiplication
# Instead of: np.trace(A @ B)
trace_AB = np.sum(A * B.T)  # O(n²) instead of O(n³)

# Outer product
outer = u[:, np.newaxis] @ v[np.newaxis, :]
# Or: outer = np.outer(u, v)
```

---

## 11. Summary

### Key Formulas

| Operation    | Formula           | Dimensions             | NumPy               |
| ------------ | ----------------- | ---------------------- | ------------------- |
| Addition     | $C = A + B$       | Same as inputs         | `A + B`             |
| Scalar mult  | $B = cA$          | Same as $A$            | `c * A`             |
| Matrix mult  | $C = AB$          | $(m\times n)(n\times p) = (m\times p)$ | `A @ B` |
| Transpose    | $B = A^T$         | $(m\times n)^T = (n\times m)$ | `A.T`        |
| Hadamard     | $C = A \odot B$   | Same as inputs         | `A * B`             |
| Outer product| $\mathbf{u}\mathbf{v}^T$ | $(m,) \times (n,) = (m,n)$ | `np.outer(u,v)` |
| Frobenius    | $\|A\|_F$         | Scalar                 | `np.linalg.norm(A)` |
| Trace        | $\text{tr}(A)$    | Scalar                 | `np.trace(A)`       |

### Matrix Type Checklist

| Type | Test | NumPy Check |
|------|------|-------------|
| Symmetric | $A = A^T$ | `np.allclose(A, A.T)` |
| Orthogonal | $A^T A = I$ | `np.allclose(A.T @ A, np.eye(n))` |
| Positive Definite | All eigenvalues > 0 | `np.all(np.linalg.eigvalsh(A) > 0)` |
| Diagonal | Off-diagonal = 0 | `np.allclose(A, np.diag(np.diag(A)))` |
| Upper Triangular | Lower = 0 | `np.allclose(A, np.triu(A))` |

### Mental Checklist

✅ Check dimensions before multiplication
✅ Remember: $AB \neq BA$ in general
✅ $(AB)^T = B^T A^T$ (reverse order!)
✅ Symmetric matrices have real eigenvalues
✅ Orthogonal matrices preserve lengths
✅ Positive definite matrices are invertible
✅ Use broadcasting for efficiency
✅ Consider numerical stability

---

## 12. Practice Exercises

1. **Dimension Analysis**: Given $A \in \mathbb{R}^{3 \times 4}$, $B \in \mathbb{R}^{4 \times 2}$, $C \in \mathbb{R}^{2 \times 3}$, which products are valid?

2. **Symmetry**: Prove that $A^T A$ is always symmetric for any matrix $A$.

3. **Trace Trick**: Show that $\text{trace}(ABC) = \text{trace}(CAB)$.

4. **Block Multiplication**: Verify block matrix multiplication for 2×2 blocks.

5. **Neural Network**: Implement a forward pass through a 2-layer network.

See [exercises.ipynb](exercises.ipynb) for solutions.

---

## 13. Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
- [Deep Learning Book: Linear Algebra](https://www.deeplearningbook.org/contents/linear_algebra.html)
- [Stanford CS229 Linear Algebra Review](https://cs229.stanford.edu/section/cs229-linalg.pdf)

---

## 14. Connections to Other Topics

| Topic | Connection |
|-------|------------|
| [Vectors and Spaces](../01-Vectors-and-Spaces/README.md) | Matrices as collections of vectors |
| [Systems of Equations](../03-Systems-of-Equations/README.md) | Matrix form $Ax = b$ |
| [Determinants](../04-Determinants/README.md) | Matrix invertibility, volume scaling |
| [Eigenvalues](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/README.md) | Matrix decomposition |
| [SVD](../../03-Advanced-Linear-Algebra/02-Singular-Value-Decomposition/README.md) | Low-rank approximation |
| [Gradients](../../05-Multivariate-Calculus/01-Partial-Derivatives-and-Gradients/README.md) | Jacobian and Hessian matrices |
| [Optimization](../../08-Optimization/02-Gradient-Descent/README.md) | Weight updates via matrices |

---

## Navigation

← [Previous: Vectors and Spaces](../01-Vectors-and-Spaces/README.md) | [Next: Systems of Equations →](../03-Systems-of-Equations/README.md)

[Back to Main](../../README.md)
