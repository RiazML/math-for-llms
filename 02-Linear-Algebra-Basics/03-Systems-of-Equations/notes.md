# Systems of Linear Equations

## Overview

Solving systems of linear equations is one of the most fundamental applications of linear algebra. Every machine learning model that has a closed-form solution relies on solving such systems. From linear regression to neural network equilibrium analysis, the ability to efficiently and accurately solve $Ax = b$ underpins countless algorithms.

## Learning Objectives

- Understand different forms of linear systems and their representations
- Master Gaussian elimination, LU decomposition, and other solution methods
- Learn when systems have unique, infinite, or no solutions
- Understand numerical stability and condition numbers
- Apply these concepts to ML problems like linear/ridge regression
- Implement iterative refinement and block system solvers

---

## 1. Introduction to Linear Systems

### 1.1 What is a Linear System?

A **system of linear equations** is a collection of $m$ equations involving $n$ unknowns:

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

**Key Properties:**
- Each equation is linear (no products of unknowns, no powers)
- The system can be consistent (has solution) or inconsistent (no solution)
- Solutions can be unique, infinite, or non-existent

### 1.2 Matrix Form

This system can be written compactly as:

$$Ax = b$$

Where:

- $A \in \mathbb{R}^{m \times n}$ is the **coefficient matrix**
- $x \in \mathbb{R}^n$ is the **unknown vector** (what we solve for)
- $b \in \mathbb{R}^m$ is the **right-hand side** (or target vector)

```
Ax = b

┌─────────────────┐   ┌─────┐     ┌─────┐
│ a₁₁  a₁₂  a₁₃  │   │ x₁  │     │ b₁  │
│ a₂₁  a₂₂  a₂₃  │ × │ x₂  │  =  │ b₂  │
│ a₃₁  a₃₂  a₃₃  │   │ x₃  │     │ b₃  │
└─────────────────┘   └─────┘     └─────┘
    Coefficients     Unknowns   Right-hand side
      (m × n)         (n × 1)     (m × 1)
```

### 1.3 The Augmented Matrix

The **augmented matrix** combines $A$ and $b$ into a single matrix:

$$[A | b] = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} & | & b_1 \\ a_{21} & a_{22} & \cdots & a_{2n} & | & b_2 \\ \vdots & \vdots & \ddots & \vdots & | & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m \end{bmatrix}$$

This representation is convenient for Gaussian elimination and determining solution types.

### 1.4 Geometric Interpretation

**2D Example:** Each equation represents a line in the plane

```
System:                              Geometric View:
x + y = 3                                y
2x - y = 0                               │
                                    4    │    x + y = 3
Solution: x=1, y=2                  3    │   ╱
                                    2  ──●──╱────────  2x - y = 0
                                    1  ╱ │╱
                                    0──╱─┼────────── x
                                      ╱  1  2  3
                                    The point (1, 2) satisfies both equations
```

**3D Example:** Each equation represents a plane in space

```
Three planes can:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  UNIQUE SOLUTION          INFINITE SOLUTIONS    NO SOLUTION  │
│  (Point intersection)     (Line/plane)          (Parallel)   │
│                                                              │
│       ╱│╲                   ═══════════          ═══════     │
│      ╱ │ ╲                     /                 ═══════     │
│     ╱  │  ╲                   /                  ═══════     │
│    ╱   ●   ╲                 /                               │
│   ╱    │    ╲               /                   Parallel     │
│  Three planes          Line along which        planes never  │
│  meet at one point     planes intersect        intersect     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.5 Column Space Interpretation

The system $Ax = b$ asks: **Can $b$ be expressed as a linear combination of the columns of $A$?**

$$x_1 \begin{bmatrix} a_{11} \\ a_{21} \\ \vdots \end{bmatrix} + x_2 \begin{bmatrix} a_{12} \\ a_{22} \\ \vdots \end{bmatrix} + \cdots + x_n \begin{bmatrix} a_{1n} \\ a_{2n} \\ \vdots \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ \vdots \end{bmatrix}$$

- If $b$ is in the column space of $A$, a solution exists
- If $b$ is not in the column space, no solution exists

---

## 2. Types of Solutions

### 2.1 Solution Categories

The nature of solutions depends on the relationship between the ranks of $A$ and $[A|b]$:

```
┌─────────────────────────────────────────────────────────────────┐
│                      SOLUTION TYPES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. UNIQUE SOLUTION       2. NO SOLUTION       3. INFINITE      │
│     (Consistent,             (Inconsistent)       SOLUTIONS     │
│      Independent)                                (Dependent)    │
│                                                                 │
│      ╲   ╱                    │  ╱                   ╱          │
│       ╲ ╱                     │ ╱                   ╱           │
│        ●                      │╱                   ╱            │
│       ╱ ╲                   ╱ │               ═══════           │
│      ╱   ╲                 ╱  │               overlapping       │
│   Lines meet            Parallel lines        lines             │
│   at one point                                                  │
│                                                                 │
│  rank(A) = rank([A|b])   rank(A) < rank([A|b])   rank(A) =      │
│     = n (# unknowns)                              rank([A|b])   │
│                                                   < n           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 The Rank Theorem for Solution Existence

**Theorem (Rouché–Capelli):** A system $Ax = b$ is consistent (has solutions) if and only if:
$$\text{rank}(A) = \text{rank}([A|b])$$

**Proof Intuition:**
- $\text{rank}(A)$ counts independent equations about $x$
- $\text{rank}([A|b])$ counts independent equations in the augmented system
- If $\text{rank}([A|b]) > \text{rank}(A)$, the extra "equation" involves only constants (contradiction like $0 = 1$)

### 2.3 Determining Solution Type Algorithmically

```python
def classify_system(A, b):
    """Determine the type of solution for Ax = b."""
    import numpy as np
    
    augmented = np.column_stack([A, b])
    rank_A = np.linalg.matrix_rank(A)
    rank_aug = np.linalg.matrix_rank(augmented)
    n = A.shape[1]  # Number of unknowns
    
    if rank_A < rank_aug:
        return "no_solution"  # Inconsistent
    elif rank_A == n:
        return "unique"       # Full rank
    else:
        return "infinite"     # Underdetermined
```

### 2.4 Solution Examples

**Unique Solution:**
```
x + 2y = 5
3x + 4y = 11

A = [[1, 2],     b = [5, 11]
     [3, 4]]
     
rank(A) = rank([A|b]) = 2 = n
Solution: x = 1, y = 2
```

**No Solution:**
```
x + y = 2
x + y = 3    ← Contradiction!

A = [[1, 1],     b = [2, 3]
     [1, 1]]
     
rank(A) = 1, rank([A|b]) = 2
The rows are parallel lines that never meet.
```

**Infinite Solutions:**
```
x + y = 2
2x + 2y = 4   ← Same equation!

A = [[1, 1],     b = [2, 4]
     [2, 2]]
     
rank(A) = rank([A|b]) = 1 < 2 = n
Parameterized: x = 2 - t, y = t for any t ∈ ℝ
```

### 2.5 Degrees of Freedom

When rank$(A)$ = rank$([A|b]) = r < n$, the solution space has dimension $n - r$:

$$\text{Degrees of freedom} = n - \text{rank}(A)$$

This equals the number of **free variables** that can be chosen arbitrarily.

---

## 3. Gaussian Elimination

### 3.1 Elementary Row Operations

Three operations that transform a system without changing its solutions:

```
┌──────────────────────────────────────────────────────────────────┐
│                   ELEMENTARY ROW OPERATIONS                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. SWAP (Row Interchange)                                       │
│     R_i ↔ R_j                                                    │
│     Exchange two rows                                            │
│                                                                  │
│  2. SCALE (Row Scaling)                                          │
│     R_i → c · R_i    where c ≠ 0                                 │
│     Multiply a row by a non-zero constant                        │
│                                                                  │
│  3. ADD (Row Addition)                                           │
│     R_i → R_i + c · R_j                                          │
│     Add a multiple of one row to another                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Why These Preserve Solutions:**
- Swapping: Reorders equations (doesn't change them)
- Scaling: Multiplying both sides of an equation by $c \neq 0$
- Addition: Combining valid equations yields valid equations

### 3.2 Row Echelon Form (REF)

A matrix is in **row echelon form** if:

1. All zero rows are at the bottom
2. The leading entry (pivot) of each row is strictly to the right of the pivot above
3. All entries below each pivot are zero

```
Row Echelon Form Examples:

┌─────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
│ ■  *  *  *  *  │     │ ■  *  *  *  *  *  │     │ 0  ■  *  *  *  *   │
│ 0  ■  *  *  *  │     │ 0  ■  *  *  *  *  │     │ 0  0  0  ■  *  *   │
│ 0  0  ■  *  *  │     │ 0  0  0  ■  *  *  │     │ 0  0  0  0  ■  *   │
│ 0  0  0  ■  *  │     │ 0  0  0  0  0  0  │     │ 0  0  0  0  0  0   │
└─────────────────┘     └───────────────────┘     └─────────────────────┘
     Full rank              Rank = 3                Pivots at cols 2,4,5

■ = pivot (leading non-zero entry, ≠ 0)
* = any value (may be zero)
0 = must be zero
```

### 3.3 The Gaussian Elimination Algorithm

**Step-by-Step Process:**

1. Start with the augmented matrix $[A|b]$
2. For each column $k$ from 1 to $n$:
   a. Find the pivot (largest absolute value in column $k$, row $k$ and below)
   b. Swap to bring pivot to diagonal position
   c. Eliminate all entries below the pivot
3. Result: Row Echelon Form
4. Use back substitution to find solution

### 3.4 Detailed Example

**Problem:** Solve
$$\begin{cases} x + 2y + z = 9 \\ 2x - y + 3z = 8 \\ 3x + y - z = 3 \end{cases}$$

```
Step 1: Form augmented matrix
┌────────────────────────┐
│  1    2    1   │   9   │
│  2   -1    3   │   8   │
│  3    1   -1   │   3   │
└────────────────────────┘

Step 2: Eliminate column 1 (below pivot)
R₂ → R₂ - 2·R₁:  [2,-1,3,8] - 2·[1,2,1,9] = [0,-5,1,-10]
R₃ → R₃ - 3·R₁:  [3,1,-1,3] - 3·[1,2,1,9] = [0,-5,-4,-24]
┌────────────────────────┐
│  1    2    1   │   9   │
│  0   -5    1   │ -10   │
│  0   -5   -4   │ -24   │
└────────────────────────┘

Step 3: Eliminate column 2 (below pivot)
R₃ → R₃ - R₂:  [0,-5,-4,-24] - [0,-5,1,-10] = [0,0,-5,-14]
┌────────────────────────┐
│  1    2    1   │   9   │  ← REF achieved!
│  0   -5    1   │ -10   │
│  0    0   -5   │ -14   │
└────────────────────────┘

Step 4: Back substitution
From row 3:  -5z = -14       →  z = 14/5 = 2.8
From row 2:  -5y + z = -10   →  y = (10 + 14/5)/5 = 12.8/5 = 2.56
From row 1:  x + 2y + z = 9  →  x = 9 - 2(2.56) - 2.8 = 1.08

Solution: x ≈ 1.08, y ≈ 2.56, z = 2.8
```

### 3.5 Reduced Row Echelon Form (RREF)

RREF adds two more requirements:

1. Each pivot equals 1
2. Each pivot is the only non-zero entry in its column

```
Converting REF to RREF:

REF:                              RREF:
┌──────────────────┐              ┌──────────────────┐
│  2    4    6   8 │              │  1    0    0   a │
│  0    3    6   9 │     →        │  0    1    0   b │
│  0    0    5  10 │    RREF      │  0    0    1   c │
└──────────────────┘              └──────────────────┘

In RREF: Each row directly gives x_i = value
```

### 3.6 Implementation

```python
def gaussian_elimination(A, b):
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    
    Returns:
        x: Solution vector
    """
    import numpy as np
    
    n = A.shape[0]
    # Create augmented matrix
    aug = np.column_stack([A.astype(float), b.astype(float)])
    
    # Forward elimination
    for k in range(n):
        # Partial pivoting: find largest pivot
        max_row = k + np.argmax(np.abs(aug[k:, k]))
        aug[[k, max_row]] = aug[[max_row, k]]
        
        if np.abs(aug[k, k]) < 1e-10:
            raise ValueError("Matrix is singular")
        
        # Eliminate below pivot
        for i in range(k + 1, n):
            factor = aug[i, k] / aug[k, k]
            aug[i, k:] -= factor * aug[k, k:]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i, -1] - np.dot(aug[i, i+1:n], x[i+1:n])) / aug[i, i]
    
    return x
```

### 3.7 Complexity Analysis

| Operation | Flops |
|-----------|-------|
| Forward elimination | $\frac{2n^3}{3}$ |
| Back substitution | $n^2$ |
| **Total** | $O(n^3)$ |

For large systems, this becomes expensive, motivating iterative methods.

---

## 4. Triangular Systems and Substitution

### 4.1 Upper Triangular Systems (Back Substitution)

An **upper triangular** system has the form:

$$\begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}$$

**Solution (Back Substitution):**
$$x_n = \frac{b_n}{u_{nn}}, \quad x_i = \frac{b_i - \sum_{j=i+1}^{n} u_{ij}x_j}{u_{ii}}$$

```python
def back_substitution(U, b):
    """Solve Ux = b where U is upper triangular."""
    n = U.shape[0]
    x = np.zeros(n)
    
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= U[i, j] * x[j]
        x[i] /= U[i, i]
    
    return x
```

### 4.2 Lower Triangular Systems (Forward Substitution)

A **lower triangular** system has the form:

$$\begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \\ b_3 \end{bmatrix}$$

**Solution (Forward Substitution):**
$$x_1 = \frac{b_1}{l_{11}}, \quad x_i = \frac{b_i - \sum_{j=1}^{i-1} l_{ij}x_j}{l_{ii}}$$

```python
def forward_substitution(L, b):
    """Solve Lx = b where L is lower triangular."""
    n = L.shape[0]
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i, i]
    
    return x
```

### 4.3 Complexity

Both substitution algorithms are $O(n^2)$, much faster than full elimination.

---

## 5. Special Cases

### 5.1 Homogeneous Systems

When $b = \mathbf{0}$: $Ax = \mathbf{0}$

```
Properties of Homogeneous Systems:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  • Always has the trivial solution: x = 0                    │
│                                                              │
│  • Non-trivial solutions exist ⟺ rank(A) < n                │
│                                                              │
│  • Solutions form a vector space: the NULL SPACE of A        │
│                                                              │
│  • Dimension of solution space = n - rank(A)                 │
│                                                              │
│  • If A is square and invertible, only trivial solution      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Example:** Find all solutions to $\begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} x = 0$

Row 2 = 2 × Row 1, so rank = 1. Null space has dimension 3 - 1 = 2.

Free variables: $x_2, x_3$. General solution:
$$x = t_1 \begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} + t_2 \begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}$$

### 5.2 Overdetermined Systems (m > n)

More equations than unknowns — typically **no exact solution**.

```
Overdetermined System Structure:

Tall matrix A:          More constraints
┌─────────┐             than degrees of
│ ■ ■ ■   │             freedom
│ ■ ■ ■   │
│ ■ ■ ■   │  m > n      Example:
│ ■ ■ ■   │             Fitting a line
│ ■ ■ ■   │             through 100 points
└─────────┘             (2 unknowns, 100 constraints)
```

**Least Squares Solution:** Find $\hat{x}$ that minimizes $\|Ax - b\|^2$:
$$\hat{x} = (A^T A)^{-1} A^T b$$

This is the **normal equations** approach.

### 5.3 Underdetermined Systems (m < n)

Fewer equations than unknowns — **infinite solutions** (if consistent).

```
Underdetermined System Structure:

Wide matrix A:          Fewer constraints
┌─────────────────┐     than unknowns
│ ■ ■ ■ ■ ■ ■ ■   │
│ ■ ■ ■ ■ ■ ■ ■   │  m < n      Free variables
└─────────────────┘              parameterize solutions
```

**Minimum Norm Solution:** Among all solutions, find the one with smallest $\|x\|$:
$$\hat{x} = A^T (A A^T)^{-1} b$$

This uses the **pseudoinverse**.

---

## 6. Direct Methods for Solving

### 6.1 Using the Matrix Inverse

If $A$ is square and invertible:

$$Ax = b \implies x = A^{-1}b$$

**⚠️ Warning:** This is almost never the right approach!

```
Why NOT to compute A⁻¹:

1. EXPENSIVE: Computing A⁻¹ costs O(n³), same as solving directly
2. WASTEFUL: Solving A⁻¹b requires another O(n²) multiplications  
3. UNSTABLE: More numerical error than direct solve
4. MEMORY: Need to store all n² entries of A⁻¹

Instead: Use np.linalg.solve(A, b) which is:
- Faster (optimized LU with pivoting)
- More stable (doesn't accumulate error)
- Memory efficient
```

### 6.2 LU Decomposition

Factor $A = LU$ (or $PA = LU$ with pivoting) where:

- $L$ = lower triangular with 1s on diagonal
- $U$ = upper triangular  
- $P$ = permutation matrix (for pivoting)

```
LU Decomposition Visualization:

    A           =           L          ×          U
┌───────────┐       ┌───────────┐       ┌───────────┐
│ a₁₁ a₁₂ a₁₃│       │  1   0   0 │       │ u₁₁ u₁₂ u₁₃│
│ a₂₁ a₂₂ a₂₃│   =   │ l₂₁  1   0 │   ×   │  0  u₂₂ u₂₃│
│ a₃₁ a₃₂ a₃₃│       │ l₃₁ l₃₂  1 │       │  0   0  u₃₃│
└───────────┘       └───────────┘       └───────────┘
   Original         Lower triangular    Upper triangular
```

**Solving $Ax = b$ with LU:**

1. Factor: $A = LU$ (done once, $O(n^3)$)
2. Solve $Ly = b$ (forward substitution, $O(n^2)$)
3. Solve $Ux = y$ (back substitution, $O(n^2)$)

**Key Advantage:** For multiple right-hand sides, factor once, solve many times!

```python
def lu_decomposition(A):
    """Compute LU decomposition (without pivoting)."""
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for k in range(n - 1):
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    
    return L, U

def solve_lu(L, U, b):
    """Solve Ax = b using LU factors."""
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    return x
```

**With SciPy:**
```python
from scipy.linalg import lu_factor, lu_solve

# Factor once
lu, piv = lu_factor(A)

# Solve multiple right-hand sides
x1 = lu_solve((lu, piv), b1)
x2 = lu_solve((lu, piv), b2)
x3 = lu_solve((lu, piv), b3)
```

### 6.3 Cholesky Decomposition

For **symmetric positive definite** (SPD) matrices: $A = LL^T$

```
Cholesky Decomposition:

    A           =           L          ×          Lᵀ
┌───────────┐       ┌───────────┐       ┌───────────┐
│ a₁₁ a₁₂ a₁₃│       │ l₁₁  0   0 │       │ l₁₁ l₂₁ l₃₁│
│ a₂₁ a₂₂ a₂₃│   =   │ l₂₁ l₂₂  0 │   ×   │  0  l₂₂ l₃₂│
│ a₃₁ a₃₂ a₃₃│       │ l₃₁ l₃₂ l₃₃│       │  0   0  l₃₃│
└───────────┘       └───────────┘       └───────────┘
   SPD Matrix        Lower triangular    (Its transpose)
```

**When to Use Cholesky:**
- Covariance matrices (always SPD)
- Kernel/Gram matrices  
- Normal equations matrix $X^TX$ (with regularization)
- Precision matrices in Gaussian models

**Advantages over LU:**
- **2× faster** (exploits symmetry)
- **Half the storage** (only store $L$)
- **Numerically stable** (no pivoting needed)
- **Detects non-SPD** (fails if not positive definite)

```python
import numpy as np
from scipy.linalg import cholesky, cho_factor, cho_solve

# Direct Cholesky
A = np.array([[4., 2.], [2., 3.]])  # SPD matrix
L = cholesky(A, lower=True)
# Verify: L @ L.T ≈ A

# Efficient solving
c, low = cho_factor(A)
x = cho_solve((c, low), b)
```

### 6.4 QR Decomposition for Least Squares

For overdetermined systems, QR is more stable than normal equations:

$$A = QR$$

where $Q$ is orthogonal ($Q^TQ = I$) and $R$ is upper triangular.

**Solving least squares:**
$$Ax \approx b \implies Rx = Q^Tb$$

```python
# More stable than (AᵀA)⁻¹Aᵀb
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ b)

# Or use lstsq directly
x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
```

### 6.5 Method Comparison

| Method | Requirements | Complexity | Best For |
|--------|-------------|------------|----------|
| Gaussian Elimination | None | $O(n^3)$ | General systems |
| LU Decomposition | Square matrix | $O(n^3)$ | Multiple RHS |
| Cholesky | SPD matrix | $O(n^3/3)$ | Covariance systems |
| QR | Any matrix | $O(mn^2)$ | Least squares |
| SVD | Any matrix | $O(mn^2)$ | Rank-deficient |

---

## 7. Machine Learning Applications

### 7.1 Linear Regression (Normal Equations)

The least squares solution minimizes $\|X\mathbf{w} - \mathbf{y}\|^2$:

$$\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$$

```
Linear Regression as a Linear System:

Data: n samples with d features
      (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)

Design Matrix X:          Target y:           Weights w:
┌─────────────────┐       ┌─────┐             ┌─────┐
│  1   x₁₁  x₁₂  │       │ y₁  │             │ w₀  │  (bias)
│  1   x₂₁  x₂₂  │       │ y₂  │             │ w₁  │  
│  ⋮    ⋮    ⋮   │       │  ⋮  │             │ w₂  │
│  1   xₙ₁  xₙ₂  │       │ yₙ  │             └─────┘
└─────────────────┘       └─────┘
    (n × (d+1))            (n × 1)            ((d+1) × 1)

Normal Equations: (XᵀX)w = Xᵀy
```

**Implementation:**
```python
def linear_regression_normal(X, y):
    """Solve linear regression via normal equations."""
    # Add bias column
    X_bias = np.column_stack([np.ones(len(X)), X])
    
    # Normal equations
    XtX = X_bias.T @ X_bias
    Xty = X_bias.T @ y
    
    # Solve (more stable than explicit inverse)
    w = np.linalg.solve(XtX, Xty)
    return w
```

### 7.2 Ridge Regression (L2 Regularization)

Adding regularization handles ill-conditioned $X^T X$:

$$\hat{\mathbf{w}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$$

```
Why Regularization Helps:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  Problem: XᵀX might be:                                      │
│    • Singular (det = 0, no inverse)                          │
│    • Nearly singular (ill-conditioned)                       │
│    • Have very small eigenvalues → unstable solution         │
│                                                              │
│  Solution: Add λI to the diagonal                            │
│    • XᵀX + λI is always invertible for λ > 0                 │
│    • Shifts all eigenvalues by λ                             │
│    • Condition number: κ(XᵀX + λI) < κ(XᵀX)                  │
│    • Larger λ → more regularization → smaller weights        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

```python
def ridge_regression(X, y, lambda_reg):
    """Solve ridge regression."""
    n, d = X.shape
    
    # Normal equations with regularization
    XtX = X.T @ X + lambda_reg * np.eye(d)
    Xty = X.T @ y
    
    return np.linalg.solve(XtX, Xty)
```

### 7.3 Weighted Least Squares

When observations have different reliabilities:

$$\min_{\mathbf{w}} \sum_{i=1}^{n} w_i (y_i - \mathbf{x}_i^T \mathbf{w})^2$$

**Solution:** $(X^T W X)\mathbf{w} = X^T W \mathbf{y}$ where $W = \text{diag}(w_1, \ldots, w_n)$

```python
def weighted_least_squares(X, y, weights):
    """Solve weighted least squares."""
    W = np.diag(weights)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    return np.linalg.solve(XtWX, XtWy)
```

### 7.4 Polynomial Regression

Fitting polynomials is linear in the coefficients:

$$y = c_0 + c_1 x + c_2 x^2 + \cdots + c_d x^d$$

```
Design Matrix for Polynomial Regression:

       X                    =        Vandermonde Matrix
┌───────────────────────┐
│  1   x₁   x₁²  ...  x₁ᵈ │
│  1   x₂   x₂²  ...  x₂ᵈ │
│  ⋮    ⋮    ⋮    ⋱    ⋮   │
│  1   xₙ   xₙ²  ...  xₙᵈ │
└───────────────────────┘
```

### 7.5 Kernel Methods (Dual Formulation)

In kernel methods, we solve systems involving the kernel matrix $K$:

$$K \alpha = y$$

where $K_{ij} = k(x_i, x_j)$ is the kernel function.

**Kernel Ridge Regression:**
$$\alpha = (K + \lambda I)^{-1} y$$

### 7.6 Neural Network Fixed Points

Some network analyses find equilibrium states:

$$W\mathbf{x} + \mathbf{b} = \mathbf{x}$$

Rearranging: $(I - W)\mathbf{x} = -\mathbf{b}$

This finds **fixed points** where network output equals input.

---

## 8. Numerical Considerations

### 8.1 Condition Number

The **condition number** $\kappa(A)$ measures sensitivity to perturbations:

$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}}$$

where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values.

```
Condition Number Interpretation:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  κ(A) ≈ 1:        WELL-CONDITIONED                           │
│                   • Small input changes → small output changes│
│                   • Solution is reliable                      │
│                                                              │
│  κ(A) ~ 10³:      MODERATELY ILL-CONDITIONED                 │
│                   • Lose ~3 digits of accuracy               │
│                   • Double precision still okay              │
│                                                              │
│  κ(A) ~ 10⁶:      SEVERELY ILL-CONDITIONED                   │
│                   • Lose ~6 digits of accuracy               │
│                   • Use regularization                       │
│                                                              │
│  κ(A) → ∞:        SINGULAR                                   │
│                   • No unique solution                       │
│                   • Determinant ≈ 0                          │
│                                                              │
│  Rule: Lose approximately log₁₀(κ) digits of accuracy        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Computing Condition Number:**
```python
import numpy as np

A = np.array([[1, 2], [1.001, 2]])
cond = np.linalg.cond(A)
print(f"Condition number: {cond:.2e}")  # Very large = ill-conditioned
```

### 8.2 Pivoting Strategies

**Why Pivot?**

Without pivoting, small pivots can amplify rounding errors catastrophically:

```
Without pivoting:                        With partial pivoting:
┌────────────────┐                       ┌────────────────┐
│ 0.0001    1    │                       │    1        1  │ ← Swap rows
│    1      1    │                       │ 0.0001      1  │
└────────────────┘                       └────────────────┘

Small pivot 0.0001 causes massive        Pivot on 1 keeps errors
error amplification                      under control
```

**Pivoting Types:**

| Type | Strategy | Stability |
|------|----------|-----------|
| No pivoting | Use diagonal elements | Unstable |
| Partial pivoting | Largest in column | Usually stable |
| Complete pivoting | Largest in submatrix | Most stable, expensive |

### 8.3 Iterative Refinement

Improve an approximate solution by iteratively correcting residuals:

```
Iterative Refinement Algorithm:
────────────────────────────────
Input: A, b, approximate solution x₀
Output: Refined solution x

1. Compute residual in higher precision:
   r = b - Ax    (use extended precision if possible)

2. Solve for correction:
   Az = r        (same decomposition as original)

3. Update solution:
   x ← x + z

4. Repeat until convergence:
   ‖r‖ < tolerance
```

```python
def iterative_refinement(A, b, x0, n_iter=5):
    """Refine an approximate solution."""
    x = x0.copy()
    
    for _ in range(n_iter):
        r = b - A @ x          # Residual
        z = np.linalg.solve(A, r)  # Correction
        x = x + z
        
        if np.linalg.norm(r) < 1e-14:
            break
    
    return x
```

### 8.4 Detecting Numerical Issues

```python
def diagnose_system(A, b):
    """Diagnose potential numerical issues."""
    n = A.shape[0]
    
    # Condition number
    cond = np.linalg.cond(A)
    print(f"Condition number: {cond:.2e}")
    
    if cond > 1e10:
        print("⚠️ Matrix is ill-conditioned!")
    
    # Rank
    rank = np.linalg.matrix_rank(A)
    print(f"Rank: {rank} (expected: {n})")
    
    if rank < n:
        print("⚠️ Matrix is rank-deficient!")
    
    # Solution
    try:
        x = np.linalg.solve(A, b)
        residual = np.linalg.norm(b - A @ x)
        print(f"Residual norm: {residual:.2e}")
        
        if residual > 1e-10:
            print("⚠️ Large residual - solution may be inaccurate")
    except np.linalg.LinAlgError:
        print("❌ Matrix is singular - no solution")
```

### 8.5 Numerical Stability Guidelines

```
┌──────────────────────────────────────────────────────────────┐
│               BEST PRACTICES FOR NUMERICAL STABILITY         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  DO:                                                         │
│  ✓ Use np.linalg.solve() instead of computing inverse       │
│  ✓ Use Cholesky for SPD matrices                            │
│  ✓ Check condition number before solving                    │
│  ✓ Use regularization for ill-conditioned systems           │
│  ✓ Prefer QR over normal equations for least squares        │
│                                                              │
│  DON'T:                                                      │
│  ✗ Compute matrix inverse explicitly                        │
│  ✗ Ignore warnings about singular matrices                  │
│  ✗ Assume small residual means accurate solution            │
│  ✗ Use Gaussian elimination without pivoting                │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. Iterative Methods

### 9.1 When to Use Iterative Methods

Direct methods (LU, Cholesky) have $O(n^3)$ complexity, which becomes prohibitive for large systems. Iterative methods are preferred when:

- Matrix is **large** (n > 10,000)
- Matrix is **sparse** (most entries are zero)
- Approximate solution is sufficient
- A good initial guess is available

### 9.2 Jacobi Method

Split $A = D + R$ where $D$ is diagonal:

$$x^{(k+1)} = D^{-1}(b - Rx^{(k)})$$

```python
def jacobi_iteration(A, b, x0, max_iter=100, tol=1e-10):
    """Solve Ax = b using Jacobi iteration."""
    n = A.shape[0]
    D = np.diag(np.diag(A))
    R = A - D
    D_inv = np.diag(1.0 / np.diag(A))
    
    x = x0.copy()
    for _ in range(max_iter):
        x_new = D_inv @ (b - R @ x)
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return x
```

### 9.3 Gauss-Seidel Method

Use updated values immediately:

$$x_i^{(k+1)} = \frac{1}{a_{ii}}\left(b_i - \sum_{j<i} a_{ij}x_j^{(k+1)} - \sum_{j>i} a_{ij}x_j^{(k)}\right)$$

Typically converges faster than Jacobi.

### 9.4 Conjugate Gradient Method

For **symmetric positive definite** matrices, CG is the gold standard:

- Converges in at most $n$ iterations (exact arithmetic)
- With preconditioning, much faster in practice
- Memory efficient: only stores a few vectors

```python
def conjugate_gradient(A, b, x0, max_iter=1000, tol=1e-10):
    """Solve Ax = b using Conjugate Gradient."""
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    
    for _ in range(max_iter):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        
        if np.sqrt(rs_new) < tol:
            break
            
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    
    return x
```

---

## 10. Block Systems and Schur Complement

### 10.1 Block Matrix Structure

Large systems often have natural block structure:

$$\begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix} = \begin{bmatrix} b_1 \\ b_2 \end{bmatrix}$$

### 10.2 The Schur Complement

The **Schur complement** of $A_{11}$ is:

$$S = A_{22} - A_{21} A_{11}^{-1} A_{12}$$

**Block Elimination Process:**

1. Solve: $S x_2 = b_2 - A_{21} A_{11}^{-1} b_1$
2. Back-solve: $A_{11} x_1 = b_1 - A_{12} x_2$

```python
def solve_block_system(A11, A12, A21, A22, b1, b2):
    """Solve 2×2 block system using Schur complement."""
    # Precompute A11 factorization
    A11_inv_A12 = np.linalg.solve(A11, A12)
    A11_inv_b1 = np.linalg.solve(A11, b1)
    
    # Form Schur complement
    S = A22 - A21 @ A11_inv_A12
    
    # Solve reduced system
    x2 = np.linalg.solve(S, b2 - A21 @ A11_inv_b1)
    
    # Back-substitute
    x1 = np.linalg.solve(A11, b1 - A12 @ x2)
    
    return x1, x2
```

### 10.3 Applications

**Constrained Optimization (KKT Systems):**
$$\begin{bmatrix} H & A^T \\ A & 0 \end{bmatrix} \begin{bmatrix} x \\ \lambda \end{bmatrix} = \begin{bmatrix} -g \\ b \end{bmatrix}$$

**Two-block ML Problems:**
- Training/validation splits
- Multi-task learning
- Hierarchical models

---

## 11. Summary

### Key Concepts

| Concept | Description |
|---------|-------------|
| $Ax = b$ | Matrix form of linear system |
| Augmented matrix | $[A \| b]$ combining coefficients and RHS |
| Row echelon form | Triangular structure after elimination |
| Rank | Number of pivots; determines solution type |
| Condition number | $\kappa(A)$ measures numerical stability |
| LU decomposition | $A = LU$ for efficient solving |
| Cholesky | $A = LL^T$ for SPD matrices |

### Solution Decision Tree

```
                    Is rank(A) = rank([A|b])?
                           /         \
                         No           Yes
                         |             |
                    No solution    rank(A) = n?
                                    /       \
                                  No         Yes
                                  |           |
                            Infinite      Unique
                            solutions    solution
```

### Method Selection Guide

```
┌──────────────────────────────────────────────────────────────┐
│                      WHICH METHOD TO USE?                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Small dense system (n < 1000)?                              │
│    └─ Yes: LU decomposition (or Cholesky if SPD)             │
│                                                              │
│  Multiple right-hand sides with same A?                      │
│    └─ Yes: Factor once, solve many times                     │
│                                                              │
│  Overdetermined (m > n)?                                     │
│    └─ Use QR decomposition for stability                     │
│                                                              │
│  Large and sparse?                                           │
│    └─ Conjugate Gradient (if SPD)                            │
│    └─ GMRES (if non-symmetric)                               │
│                                                              │
│  Ill-conditioned?                                            │
│    └─ Add regularization                                     │
│    └─ Use SVD-based pseudoinverse                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### ML Takeaways

1. **Linear regression** = solving an overdetermined system
2. **Regularization** (ridge, Tikhonov) improves conditioning
3. **Never explicitly invert** matrices — use solve() instead
4. **Numerical stability** matters in practice
5. **Sparse methods** essential for large-scale ML
6. **Condition number** predicts solution quality

---

## 12. Common Pitfalls and Debugging

### Pitfall 1: Computing Inverse Explicitly

```python
# ❌ BAD
x = np.linalg.inv(A) @ b

# ✅ GOOD  
x = np.linalg.solve(A, b)
```

### Pitfall 2: Ignoring Numerical Warnings

```python
# Don't suppress warnings!
import warnings
warnings.filterwarnings('error')  # Treat warnings as errors

try:
    x = np.linalg.solve(A, b)
except np.linalg.LinAlgError:
    print("Matrix is singular!")
except RuntimeWarning:
    print("Numerical issues detected")
```

### Pitfall 3: Assuming Residual = Error

A small residual $\|Ax - b\|$ doesn't guarantee $x$ is accurate:
- Ill-conditioned matrices can have large errors with small residuals
- Always check condition number!

### Pitfall 4: Not Checking Solution

```python
def solve_and_verify(A, b, tol=1e-10):
    """Solve and verify the solution."""
    x = np.linalg.solve(A, b)
    residual = np.linalg.norm(A @ x - b)
    
    if residual > tol:
        print(f"Warning: Large residual {residual:.2e}")
        print(f"Condition number: {np.linalg.cond(A):.2e}")
    
    return x
```

---

## Further Reading

- [Gilbert Strang: Linear Algebra Lecture 2 - Elimination](https://www.youtube.com/watch?v=QVKj3LADCnA)
- [3Blue1Brown: Inverse Matrices](https://www.youtube.com/watch?v=uQhTuRlWMxw)
- [Numerical Linear Algebra (Trefethen & Bau)](https://people.maths.ox.ac.uk/trefethen/text.html)
- [LAPACK Users' Guide](https://www.netlib.org/lapack/lug/)

---

## Navigation

← [Previous: Matrix Operations](../02-Matrix-Operations/README.md) | [Next: Determinants →](../04-Determinants/README.md)

[Back to Linear Algebra Basics](../README.md) | [Back to Main](../../README.md)
