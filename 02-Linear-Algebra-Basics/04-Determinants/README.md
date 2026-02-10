# Determinants

## Overview

The determinant is a scalar value that encodes important properties of a square matrix. It tells us about linear independence, invertibility, and how transformations affect volume. This fundamental concept appears throughout machine learning—from covariance matrix analysis to normalizing flows.

## Learning Objectives

- Understand the geometric meaning of determinants as signed volumes
- Compute determinants using multiple methods (cofactor, LU, Cholesky)
- Connect determinants to matrix properties (rank, eigenvalues, invertibility)
- Apply determinants in ML contexts (Gaussian PDFs, Jacobians, PCA)
- Handle numerical stability issues with log-determinants

## Prerequisites

- Matrix multiplication and transpose
- Systems of linear equations
- Understanding of linear independence
- Basic understanding of eigenvalues (helpful but not required)

---

## 1. What is a Determinant?

### 1.1 Definition

The **determinant** of a square matrix $A$ is a scalar, denoted $\det(A)$ or $|A|$.

**Formal Definition:**
The determinant is the unique function $\det: M_{n \times n} \to \mathbb{R}$ satisfying:
1. $\det(I) = 1$ (identity)
2. Multilinear in rows/columns
3. Alternating (swapping two rows negates the determinant)

**2×2 Matrix:**
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**3×3 Matrix (Sarrus' Rule):**
$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = aei + bfg + cdh - ceg - bdi - afh$$

### 1.2 Intuitive Understanding

The determinant answers the question: **"How much does this matrix stretch or compress space?"**

```
Think of it as:
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  det(A) = "Size change factor" × "Orientation indicator"    │
│                                                              │
│  • |det(A)| = how much volumes get scaled                   │
│  • sign(det(A)) = whether orientation flips                 │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 1.3 Geometric Interpretation

```
The determinant measures SIGNED VOLUME:

2D: Area of parallelogram formed by column vectors
3D: Volume of parallelepiped formed by column vectors
nD: n-dimensional hypervolume

Original unit square:        After transformation A:
┌─────────────┐              ╱╲
│             │              ╱  ╲
│      1      │  →  A  →    ╱    ╲    Area = |det(A)|
│             │            ╱      ╲
└─────────────┘            ────────

If det(A) > 0: Orientation preserved (counterclockwise stays counterclockwise)
If det(A) < 0: Orientation flipped (reflection included)
If det(A) = 0: Collapsed to lower dimension (singular matrix)
```

**Visual Example - 2D Transformation:**
```
Column vectors of A:         Parallelogram formed:
                              
A = [a₁ a₂]                        a₂ = (a₁₂, a₂₂)
    [b₁ b₂]                       ↗
                              ╱    
                             ╱      ╲
                            ╱        ╲
                    Origin ●──────────→ a₁ = (a₁₁, a₂₁)
                              
Area = |det(A)| = |a₁₁·a₂₂ - a₁₂·a₂₁|
```

### 1.4 Key Property: Invertibility

$$\text{A is invertible} \iff \det(A) \neq 0$$

```
┌──────────────────────────────────────────────────────────────────────┐
│                      INVERTIBILITY CRITERION                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  det(A) ≠ 0:                         det(A) = 0:                    │
│  ─────────────                       ──────────────                  │
│  • Columns are linearly              • Columns are linearly          │
│    independent                         dependent                     │
│  • Full rank (rank = n)              • Rank deficient (rank < n)    │
│  • A⁻¹ exists                        • A⁻¹ does NOT exist           │
│  • Unique solution to Ax=b           • No unique solution           │
│  • Eigenvalues all non-zero          • At least one eigenvalue = 0  │
│  • Null space = {0}                  • Non-trivial null space       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 1.5 Why Determinants Matter in ML

| Application | Role of Determinant |
|-------------|---------------------|
| Gaussian PDF | Normalization constant includes $\|\Sigma\|^{-1/2}$ |
| Covariance Analysis | Generalized variance = $\det(\Sigma)$ |
| Normalizing Flows | Jacobian determinant for density transformation |
| Feature Selection | Detect redundant/collinear features |
| Optimization | Hessian determinant indicates saddle points |
| Change of Variables | Volume element transformation in integrals |

---

## 2. Computing Determinants

### 2.1 2×2 Determinant

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

```
Visual Mnemonic:
┌───────────┐
│  a    b   │
│   ╲  ╱    │  Main diagonal - Anti-diagonal
│    ╳      │  = (a × d) - (b × c)
│   ╱  ╲    │
│  c    d   │
└───────────┘
```

**Example 1 - Invertible Matrix:**
$$\det\begin{bmatrix} 3 & 2 \\ 1 & 4 \end{bmatrix} = (3)(4) - (2)(1) = 12 - 2 = 10 \neq 0 \quad \text{✓ Invertible}$$

**Example 2 - Singular Matrix:**
$$\det\begin{bmatrix} 2 & 4 \\ 1 & 2 \end{bmatrix} = (2)(2) - (4)(1) = 4 - 4 = 0 \quad \text{✗ Singular}$$

Notice: The second row is half the first row (linearly dependent).

### 2.2 3×3 Determinant (Sarrus' Rule)

For 3×3 matrices only, use Sarrus' diagonal trick:

```
Copy first two columns to the right:

│ a  b  c │ a  b
│ d  e  f │ d  e
│ g  h  i │ g  h

Add diagonals going down-right:      Subtract diagonals going down-left:
    ↘   ↘   ↘                            ↙   ↙   ↙
    aei + bfg + cdh               -      ceg + afh + bdi

det = (aei + bfg + cdh) - (ceg + afh + bdi)
```

**Example:**
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 5 & 6 & 0 \end{bmatrix}$$

$$\det(A) = (1 \cdot 1 \cdot 0 + 2 \cdot 4 \cdot 5 + 3 \cdot 0 \cdot 6) - (3 \cdot 1 \cdot 5 + 1 \cdot 4 \cdot 6 + 2 \cdot 0 \cdot 0)$$
$$= (0 + 40 + 0) - (15 + 24 + 0) = 40 - 39 = 1$$

### 2.3 Cofactor Expansion (Laplace Expansion)

For any n×n matrix, expand along any row $i$ or column $j$:

**Row Expansion:**
$$\det(A) = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

**Column Expansion:**
$$\det(A) = \sum_{i=1}^{n} (-1)^{i+j} a_{ij} M_{ij}$$

Where:
- $M_{ij}$ is the **minor**: determinant of $(n-1) \times (n-1)$ submatrix with row $i$ and column $j$ removed
- $C_{ij} = (-1)^{i+j} M_{ij}$ is the **cofactor**

```
Sign pattern (checkerboard):

│  +   -   +   -  │
│  -   +   -   +  │
│  +   -   +   -  │
│  -   +   -   +  │

Position (i,j) has sign (-1)^(i+j)
```

**Example - Expansion along First Row:**
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

$$\det(A) = (+1) \cdot 1 \cdot \det\begin{bmatrix} 5 & 6 \\ 8 & 9 \end{bmatrix} + (-1) \cdot 2 \cdot \det\begin{bmatrix} 4 & 6 \\ 7 & 9 \end{bmatrix} + (+1) \cdot 3 \cdot \det\begin{bmatrix} 4 & 5 \\ 7 & 8 \end{bmatrix}$$

$$= 1(45-48) - 2(36-42) + 3(32-35)$$
$$= 1(-3) - 2(-6) + 3(-3)$$
$$= -3 + 12 - 9 = 0$$

**Strategy:** Expand along the row/column with the most zeros to minimize computation!

### 2.4 Row Reduction Method (Gaussian Elimination)

Reduce matrix to upper triangular form, then multiply diagonal:

$$\det(A) = (\text{sign from swaps}) \times \prod_{i} u_{ii}$$

```
Algorithm:
1. Apply row operations to get upper triangular form
2. Track sign changes from row swaps
3. det = (±1) × product of diagonal entries

A → U (upper triangular)
    ┌─────────────┐
    │ u₁₁  *   *  │
    │  0  u₂₂  *  │  det(U) = u₁₁ × u₂₂ × u₃₃
    │  0   0  u₃₃ │
    └─────────────┘
```

**Row Operation Effects:**
| Operation | Effect on det |
|-----------|---------------|
| $R_i \leftrightarrow R_j$ (swap) | $\det \to -\det$ |
| $R_i \to c \cdot R_i$ (scale) | $\det \to c \cdot \det$ |
| $R_i \to R_i + c \cdot R_j$ (add) | $\det \to \det$ (unchanged) |

**Example:**
$$A = \begin{bmatrix} 2 & 1 \\ 4 & 3 \end{bmatrix}$$

Step 1: $R_2 \to R_2 - 2R_1$
$$\begin{bmatrix} 2 & 1 \\ 0 & 1 \end{bmatrix}$$

$\det(A) = 2 \times 1 = 2$ ✓

Verification: $ad - bc = 2(3) - 1(4) = 2$ ✓

### 2.5 LU Decomposition

For large matrices, use LU factorization:

$$A = LU \implies \det(A) = \det(L) \cdot \det(U) = 1 \cdot \prod_{i} u_{ii}$$

```
L (lower triangular, 1s on diagonal):    U (upper triangular):
┌─────────────┐                          ┌─────────────┐
│  1          │                          │ u₁₁  *   *  │
│  *   1      │  det(L) = 1              │      u₂₂ *  │  det(U) = ∏ uᵢᵢ
│  *   *   1  │                          │          u₃₃│
└─────────────┘                          └─────────────┘
```

**With Partial Pivoting (PA = LU):**
$$\det(A) = \det(P^{-1}) \cdot \det(L) \cdot \det(U) = (-1)^s \cdot \prod_{i} u_{ii}$$

Where $s$ = number of row swaps in $P$.

### 2.6 Complexity Comparison

| Method | Complexity | When to Use |
|--------|------------|-------------|
| 2×2 formula | $O(1)$ | 2×2 matrices |
| 3×3 Sarrus | $O(1)$ | 3×3 matrices only |
| Cofactor expansion | $O(n!)$ | Small matrices (n ≤ 4) |
| Row reduction | $O(n^3)$ | General case |
| LU decomposition | $O(n^3)$ | When you also need LU |
| Triangular matrix | $O(n)$ | Already triangular |

```python
import numpy as np

# NumPy uses LAPACK's LU decomposition
A = np.array([[1, 2], [3, 4]])
det_A = np.linalg.det(A)  # Returns -2.0
```

---

## 3. Properties of Determinants

### 3.1 Fundamental Properties

| Property | Formula | Proof Sketch |
|----------|---------|--------------|
| Identity | $\det(I) = 1$ | Product of 1s on diagonal |
| Transpose | $\det(A^T) = \det(A)$ | Rows and columns are symmetric |
| Product | $\det(AB) = \det(A)\det(B)$ | Composition of volume scaling |
| Inverse | $\det(A^{-1}) = 1/\det(A)$ | From $AA^{-1}=I$ and product rule |
| Scalar | $\det(cA) = c^n \det(A)$ | Each of $n$ rows scaled by $c$ |
| Power | $\det(A^k) = (\det(A))^k$ | Repeated product rule |

### 3.2 Proof: Product Rule

**Claim:** $\det(AB) = \det(A)\det(B)$

**Intuitive Proof:**
- $\det(A)$ = volume scaling factor of transformation $A$
- $\det(B)$ = volume scaling factor of transformation $B$
- $AB$ = first apply $B$, then apply $A$
- Total volume scaling = product of scaling factors

**Formal Proof Outline:**
1. Both $\det(AB)$ and $\det(A)\det(B)$ are multilinear, alternating functions in the columns of $B$
2. Both equal $\det(A)$ when $B = I$
3. By uniqueness of determinant, they're equal

### 3.3 Row and Column Operations

```
┌─────────────────────────────────────────────────────────────────────┐
│                 ROW OPERATION EFFECTS ON DETERMINANT                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Operation                        Effect on det(A)                  │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Swap two rows:                   det → -det (sign flip)            │
│      R_i ↔ R_j                    Odd permutation                   │
│                                                                     │
│  Multiply row by scalar c:        det → c × det                     │
│      R_i → c·R_i                  Scales one dimension              │
│                                                                     │
│  Add multiple of one row          det → det (unchanged)             │
│  to another:                      Shear transformation              │
│      R_i → R_i + c·R_j            doesn't change volume             │
│                                                                     │
│  Same rules apply to columns!                                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**Why a Row Swap Changes Sign:**
- Swapping rows is a reflection
- Reflections flip orientation
- Therefore $\det \to -\det$

**Why Adding Rows Preserves Determinant:**
$$\begin{vmatrix} a + cb & b \\ c + cd & d \end{vmatrix} = (a+cb)d - b(c+cd) = ad + cbd - bc - bcd = ad - bc$$

### 3.4 Special Matrix Determinants

| Matrix Type | Determinant | Why |
|-------------|-------------|-----|
| **Diagonal** | $\prod_i d_{ii}$ | Only diagonal contributes |
| **Upper/Lower Triangular** | $\prod_i a_{ii}$ | Cofactor expansion along row/column of zeros |
| **Orthogonal** ($Q^TQ = I$) | $\det(Q) = \pm 1$ | From $\det(Q^TQ) = \det(Q)^2 = 1$ |
| **Rotation** | $\det(R) = +1$ | Preserves orientation |
| **Reflection** | $\det(R) = -1$ | Flips orientation |
| **Symmetric** | Real eigenvalues → real det | — |
| **Skew-symmetric** (odd $n$) | $\det(A) = 0$ | $\det(A) = \det(-A^T) = (-1)^n\det(A)$ |
| **Nilpotent** ($A^k = 0$) | $\det(A) = 0$ | All eigenvalues are 0 |
| **Idempotent** ($A^2 = A$) | $\det(A) = 0$ or $1$ | Eigenvalues are 0 or 1 |

**Example - Triangular Matrix:**
$$\det\begin{bmatrix} 3 & 1 & 4 \\ 0 & 2 & 7 \\ 0 & 0 & 5 \end{bmatrix} = 3 \times 2 \times 5 = 30$$

### 3.5 Block Matrix Determinants

**Block Diagonal:**
$$\det\begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix} = \det(A) \cdot \det(B)$$

**Block Triangular:**
$$\det\begin{bmatrix} A & C \\ 0 & B \end{bmatrix} = \det(A) \cdot \det(B)$$

**General 2×2 Block (Schur Complement):**
If $A$ is invertible:
$$\det\begin{bmatrix} A & B \\ C & D \end{bmatrix} = \det(A) \cdot \det(D - CA^{-1}B)$$

Where $D - CA^{-1}B$ is the **Schur complement** of $A$.

**Example:**
$$\det\begin{bmatrix} 2 & 0 & 1 \\ 0 & 3 & 0 \\ 0 & 0 & 4 \end{bmatrix} = 2 \times 3 \times 4 = 24 \quad \text{(diagonal)}$$

### 3.6 Determinant and Eigenvalues

**Fundamental Relationship:**
$$\det(A) = \prod_{i=1}^n \lambda_i$$

The determinant equals the product of all eigenvalues (counting multiplicities).

**Proof:**
- Characteristic polynomial: $p(\lambda) = \det(A - \lambda I)$
- Roots are eigenvalues: $p(\lambda) = (-1)^n(\lambda - \lambda_1)(\lambda - \lambda_2)\cdots(\lambda - \lambda_n)$
- Setting $\lambda = 0$: $\det(A) = (-1)^n(-\lambda_1)(-\lambda_2)\cdots(-\lambda_n) = \lambda_1\lambda_2\cdots\lambda_n$

**Corollary:** Matrix is singular $\iff$ at least one eigenvalue is zero.

### 3.7 Trace-Determinant Relationship

For a 2×2 matrix:
$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$$

The characteristic polynomial is:
$$\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0$$

So:
- $\lambda_1 + \lambda_2 = \text{tr}(A) = a + d$
- $\lambda_1 \cdot \lambda_2 = \det(A) = ad - bc$

This generalizes: trace = sum of eigenvalues, det = product of eigenvalues.

### 3.8 Determinant Inequalities

**Hadamard's Inequality:**
For a matrix with column vectors $\mathbf{a}_1, \ldots, \mathbf{a}_n$:
$$|\det(A)| \leq \prod_{j=1}^n \|\mathbf{a}_j\|$$

Geometric meaning: The parallelepiped volume is at most the product of edge lengths (equality when columns are orthogonal).

**For Positive Definite Matrices:**
$$\det(A) \leq \prod_{i=1}^n a_{ii}$$

The determinant is bounded by the product of diagonal elements.

---

## 4. Geometric Applications

### 4.1 Area of a Parallelogram (2D)

Two vectors $\vec{u} = (u_1, u_2)$ and $\vec{v} = (v_1, v_2)$ form a parallelogram.

$$\text{Area} = \left| \det\begin{bmatrix} u_1 & v_1 \\ u_2 & v_2 \end{bmatrix} \right| = |u_1 v_2 - u_2 v_1|$$

```
              v = (v₁, v₂)
             ↗
            ╱╲
           ╱  ╲
          ╱    ╲
         ╱      ╲
        ●────────→ u = (u₁, u₂)
      Origin
      
Area = |u₁v₂ - u₂v₁|
     = |det([u|v])|
```

**Example:**
$$\vec{u} = (3, 0), \quad \vec{v} = (1, 2)$$
$$\text{Area} = |3 \cdot 2 - 0 \cdot 1| = |6| = 6$$

**Special Cases:**
- If $\vec{u}$ and $\vec{v}$ are parallel: Area = 0 (degenerate parallelogram)
- If $\vec{u}$ and $\vec{v}$ are perpendicular: Area = $\|\vec{u}\| \cdot \|\vec{v}\|$

### 4.2 Area of a Triangle

For a triangle with vertices at $(x_1, y_1)$, $(x_2, y_2)$, $(x_3, y_3)$:

$$\text{Area} = \frac{1}{2} \left| \det\begin{bmatrix} x_1 & y_1 & 1 \\ x_2 & y_2 & 1 \\ x_3 & y_3 & 1 \end{bmatrix} \right|$$

Or equivalently:
$$\text{Area} = \frac{1}{2} |x_1(y_2 - y_3) + x_2(y_3 - y_1) + x_3(y_1 - y_2)|$$

**Example:**
Triangle with vertices $(0, 0)$, $(4, 0)$, $(2, 3)$:
$$\text{Area} = \frac{1}{2} |0(0-3) + 4(3-0) + 2(0-0)| = \frac{1}{2} |12| = 6$$

### 4.3 Volume of a Parallelepiped (3D)

Three vectors $\vec{u}$, $\vec{v}$, $\vec{w}$ form a parallelepiped:

$$\text{Volume} = \left| \det\begin{bmatrix} u_1 & v_1 & w_1 \\ u_2 & v_2 & w_2 \\ u_3 & v_3 & w_3 \end{bmatrix} \right|$$

This equals the **scalar triple product**: $|\vec{u} \cdot (\vec{v} \times \vec{w})|$

```
         w
        ↗╲
       ╱  ╲
      ╱    ╲────────╲
     ╱      ╲        ╲
    ●────────→ u      ╲
     ╲        v       ╱
      ╲      ↗       ╱
       ╲    ╱       ╱
        ╲  ╱───────╱
         ╲╱

Volume = |det([u|v|w])|
       = |u · (v × w)|
```

**Example:**
$$\vec{u} = (1, 0, 0), \quad \vec{v} = (0, 2, 0), \quad \vec{w} = (0, 0, 3)$$
$$\text{Volume} = \left| \det\begin{bmatrix} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 3 \end{bmatrix} \right| = |1 \cdot 2 \cdot 3| = 6$$

### 4.4 Volume of a Tetrahedron

For a tetrahedron with vertices $\mathbf{p}_0, \mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3$:

$$\text{Volume} = \frac{1}{6} \left| \det\begin{bmatrix} \mathbf{p}_1 - \mathbf{p}_0 \\ \mathbf{p}_2 - \mathbf{p}_0 \\ \mathbf{p}_3 - \mathbf{p}_0 \end{bmatrix} \right|$$

### 4.5 Cross Product via Determinant

The cross product in 3D can be computed symbolically:

$$\vec{u} \times \vec{v} = \det\begin{bmatrix} \hat{i} & \hat{j} & \hat{k} \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{bmatrix}$$

Expanding:
$$= \hat{i}(u_2 v_3 - u_3 v_2) - \hat{j}(u_1 v_3 - u_3 v_1) + \hat{k}(u_1 v_2 - u_2 v_1)$$

**Example:**
$$\vec{u} = (1, 2, 3), \quad \vec{v} = (4, 5, 6)$$
$$\vec{u} \times \vec{v} = \det\begin{bmatrix} \hat{i} & \hat{j} & \hat{k} \\ 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$
$$= \hat{i}(12-15) - \hat{j}(6-12) + \hat{k}(5-8) = (-3, 6, -3)$$

### 4.6 Collinearity and Coplanarity Tests

**Three Points Collinear (2D):**
Points $(x_1, y_1)$, $(x_2, y_2)$, $(x_3, y_3)$ are collinear iff:
$$\det\begin{bmatrix} x_1 & y_1 & 1 \\ x_2 & y_2 & 1 \\ x_3 & y_3 & 1 \end{bmatrix} = 0$$

**Four Points Coplanar (3D):**
Points $\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3, \mathbf{p}_4$ are coplanar iff:
$$\det\begin{bmatrix} x_1 & y_1 & z_1 & 1 \\ x_2 & y_2 & z_2 & 1 \\ x_3 & y_3 & z_3 & 1 \\ x_4 & y_4 & z_4 & 1 \end{bmatrix} = 0$$

### 4.7 Signed Area and Orientation

The **signed** determinant (not absolute value) tells us about orientation:

```
Signed Area/Volume:

det > 0: Vectors form right-handed system
         (counterclockwise in 2D)
         
det < 0: Vectors form left-handed system
         (clockwise in 2D)
         
det = 0: Vectors are linearly dependent
         (collinear in 2D, coplanar in 3D)
```

**Application - Point Inside Triangle:**
Point $P$ is inside triangle $ABC$ iff signs of three signed areas are all the same:
- sign(Area($PAB$)) = sign(Area($PBC$)) = sign(Area($PCA$))

---

## 5. Determinants and Linear Transformations

### 5.1 Transformation Scaling Factor

The **absolute value** of the determinant gives the scaling factor for areas/volumes:

$$\text{Area}(T(S)) = |\det(T)| \cdot \text{Area}(S)$$

```
Example: Scaling transformation

T = [2  0]  →  det(T) = 4
    [0  2]

Original unit square:           After transformation:
┌─────┐                        ┌───────────┐
│     │  Area = 1              │           │
│     │                        │           │  Area = 4
└─────┘                        │           │
                               └───────────┘
```

### 5.2 Common Transformations

| Transformation | Matrix | Determinant | Effect |
|----------------|--------|-------------|--------|
| Identity | $I$ | 1 | No change |
| Scaling by $k$ | $kI$ | $k^n$ | Volume × $k^n$ |
| Rotation by $\theta$ | $\begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{bmatrix}$ | 1 | Preserves area |
| Reflection | $\begin{bmatrix}1 & 0 \\ 0 & -1\end{bmatrix}$ | -1 | Flips orientation |
| Shear | $\begin{bmatrix}1 & k \\ 0 & 1\end{bmatrix}$ | 1 | Preserves area |
| Projection | Rank < n | 0 | Collapses dimension |

### 5.3 Orientation

$$\det(A) > 0 \implies \text{preserves orientation (no reflection)}$$
$$\det(A) < 0 \implies \text{reverses orientation (includes reflection)}$$

```
det > 0 (preserves):             det < 0 (flips):

  ↑ y                              ↑ y
  │    2                           │  1
  │   ↗                            │   ↖
  │  1                             │     2
  ├────→ x                         ├────→ x

Counterclockwise                 Clockwise
order preserved                  order reversed

Like rotating paper              Like flipping paper
```

### 5.4 Composition of Transformations

When composing transformations:
$$\det(AB) = \det(A) \cdot \det(B)$$

**Interpretation:**
- If $A$ scales by factor 2 and $B$ scales by factor 3
- Then $AB$ scales by factor $2 \times 3 = 6$

**Example:**
```
Rotate 90° then scale by 2:

R = [0 -1]  det(R) = 1 (rotation preserves area)
    [1  0]

S = [2  0]  det(S) = 4 (scales area by 4)
    [0  2]

SR = [0 -2]  det(SR) = 4 (combined effect)
     [2  0]
```

### 5.5 Inverse Transformation

$$\det(A^{-1}) = \frac{1}{\det(A)}$$

If $A$ expands volumes by factor $k$, then $A^{-1}$ compresses by factor $1/k$.

---

## 6. Cramer's Rule

### 6.1 Statement

For a system $Ax = b$ with $\det(A) \neq 0$ (unique solution exists):

$$x_i = \frac{\det(A_i)}{\det(A)}$$

Where $A_i$ is matrix $A$ with column $i$ replaced by vector $b$.

```
For Ax = b:

        ┌─────────────────────┐
        │                     │
x₁ =    │ det(A with col 1    │  ∕  det(A)
        │ replaced by b)      │
        │                     │
        └─────────────────────┘
```

### 6.2 Detailed Example

$$\begin{cases} 2x + y = 5 \\ x + 3y = 6 \end{cases}$$

**Step 1:** Write in matrix form
$$A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$$

**Step 2:** Compute $\det(A)$
$$\det(A) = 2(3) - 1(1) = 6 - 1 = 5$$

**Step 3:** Form $A_1$ (replace column 1 with $b$)
$$A_1 = \begin{bmatrix} 5 & 1 \\ 6 & 3 \end{bmatrix}, \quad \det(A_1) = 15 - 6 = 9$$

**Step 4:** Form $A_2$ (replace column 2 with $b$)
$$A_2 = \begin{bmatrix} 2 & 5 \\ 1 & 6 \end{bmatrix}, \quad \det(A_2) = 12 - 5 = 7$$

**Step 5:** Apply Cramer's Rule
$$x = \frac{\det(A_1)}{\det(A)} = \frac{9}{5} = 1.8$$
$$y = \frac{\det(A_2)}{\det(A)} = \frac{7}{5} = 1.4$$

**Verification:** $2(1.8) + 1.4 = 3.6 + 1.4 = 5$ ✓

### 6.3 3×3 Example

$$\begin{cases} x + 2y + 3z = 6 \\ 2x + 5y + 2z = 4 \\ 6x - 3y + z = 2 \end{cases}$$

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 2 \\ 6 & -3 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 6 \\ 4 \\ 2 \end{bmatrix}$$

Compute: $\det(A) = -45$ (using cofactor expansion or LU)

Then:
$$x = \frac{\det(A_1)}{\det(A)}, \quad y = \frac{\det(A_2)}{\det(A)}, \quad z = \frac{\det(A_3)}{\det(A)}$$

### 6.4 When to Use Cramer's Rule

| Situation | Recommendation |
|-----------|---------------|
| 2×2 or 3×3 systems | Cramer's rule is fine |
| Symbolic solutions | Cramer's rule gives closed-form |
| Large systems (n > 4) | Use LU decomposition instead |
| Multiple right-hand sides | Use LU with forward/back substitution |
| Numerical computation | Cramer's is O(n!) vs O(n³) for LU |

**Complexity:**
- Cramer's Rule: $O(n! \cdot n)$ determinants
- LU Decomposition: $O(n^3)$

For $n = 10$: Cramer's needs millions of operations, LU needs ~1000.

### 6.5 Theoretical Importance

Despite computational inefficiency, Cramer's Rule proves:
1. If $\det(A) \neq 0$, a unique solution exists
2. The solution varies continuously with the entries of $A$ and $b$
3. **Adjugate formula for inverse:**
$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

Where $\text{adj}(A)$ is the **adjugate** (transpose of cofactor matrix).

---

## 7. Machine Learning Applications

### 7.1 Covariance Matrix Determinant (Generalized Variance)

The determinant of a covariance matrix $\Sigma$ is called the **generalized variance**:

$$\text{Generalized Variance} = \det(\Sigma)$$

**Interpretation:**
- Measures the "total spread" of multivariate data
- Proportional to the volume of the uncertainty ellipsoid
- Higher determinant = more overall variability

```
Large det(Σ):                    Small det(Σ):
High overall variance            Low overall variance
Data spread out broadly          Data concentrated

    ╭─────────────╮                    ╭───╮
   ╱               ╲                  │   │
  │                 │                 │   │
   ╲               ╱                  │   │
    ╰─────────────╯                    ╰───╯

det(Σ) = λ₁ × λ₂ × ... × λₙ
       = Product of eigenvalues (principal variances)
```

**Example:**
$$\Sigma = \begin{bmatrix} 4 & 2 \\ 2 & 3 \end{bmatrix}$$
$$\det(\Sigma) = 4 \cdot 3 - 2 \cdot 2 = 8$$

Eigenvalues: $\lambda_1 \approx 5.236$, $\lambda_2 \approx 1.528$
Check: $5.236 \times 1.528 \approx 8$ ✓

### 7.2 Multivariate Gaussian PDF

The probability density function of a multivariate Gaussian:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**Role of Determinant:**
- $|\Sigma|^{1/2}$ in denominator normalizes the PDF
- Larger $\det(\Sigma)$ → flatter, more spread distribution
- Smaller $\det(\Sigma)$ → peaked, concentrated distribution

```
Normalization factor breakdown:

1/[(2π)^(d/2) × |Σ|^(1/2)]

• (2π)^(d/2): scales with dimension
• |Σ|^(1/2): scales with covariance "volume"

Together they ensure: ∫ p(x) dx = 1
```

**Log-Likelihood:**
$$\log p(\mathbf{x}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\log|\Sigma| - \frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$$

### 7.3 Maximum Likelihood Estimation

When fitting a Gaussian to data, we maximize the log-likelihood:

$$\mathcal{L} = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log|\Sigma| - \frac{1}{2}\sum_{i=1}^n (\mathbf{x}_i - \boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}_i - \boldsymbol{\mu})$$

The $\log|\Sigma|$ term penalizes large variances (prevents overfitting).

### 7.4 Model Selection and Information Criteria

**Bayesian Information Criterion (BIC):**
$$\text{BIC} = -2\log L + k\log n$$

Where log-likelihood $L$ includes determinant terms.

**Akaike Information Criterion (AIC):**
$$\text{AIC} = -2\log L + 2k$$

For Gaussian models, the determinant affects model comparison.

### 7.5 Linear Discriminant Analysis (LDA)

LDA uses determinants to compare within-class and between-class scatter:

$$J = \frac{|\mathbf{S}_B|}{|\mathbf{S}_W|}$$

- $\mathbf{S}_B$: Between-class scatter matrix
- $\mathbf{S}_W$: Within-class scatter matrix
- Maximize $J$ for best class separation

### 7.6 PCA and Total Variance

In Principal Component Analysis:

$$\det(\Sigma) = \prod_{i=1}^d \lambda_i = \text{Product of eigenvalues}$$

**Interpretation:**
- Each $\lambda_i$ is variance along principal component $i$
- $\det(\Sigma)$ = "volume" of variance in all directions
- If any $\lambda_i = 0$: data lies in lower-dimensional subspace

**Proportion of Variance:**
$$\frac{\lambda_1 + \cdots + \lambda_k}{\lambda_1 + \cdots + \lambda_d} = \text{Variance explained by first k PCs}$$

### 7.7 Jacobian Determinant (Change of Variables)

**Fundamental Theorem:** When transforming a random variable through a function $g$:

$$p_Y(y) = p_X(g^{-1}(y)) \left| \det\left(\frac{\partial g^{-1}}{\partial y}\right) \right|$$

```
X ~ p_X                    Y = g(X) ~ p_Y

    p_X(x)                     p_Y(y)
       ↑                          ↑
       │  ╭──╮                    │    ╭─╮
       │ ╱    ╲                   │   ╱   ╲
       │╱      ╲                  │  ╱     ╲
       └────────→ x               └────────→ y
       
       ↓     g     ↓
       
       |det(J)|: Accounts for how g stretches/compresses space
```

**Applications in Deep Learning:**

1. **Normalizing Flows:**
   Transform simple distributions (Gaussian) to complex ones while tracking probability:
   $$\log p_K(z_K) = \log p_0(z_0) - \sum_{k=1}^K \log|\det J_{f_k}|$$

2. **Variational Autoencoders (VAEs):**
   The reparameterization trick involves Jacobian determinants.

3. **Generative Models:**
   Flow-based models (RealNVP, Glow) use efficient Jacobian determinants.

### 7.8 Efficient Jacobian Computation in Flows

For tractable normalizing flows, we need efficient $\log|\det J|$ computation:

**Triangular Jacobian:**
$$\log|\det J| = \sum_i \log|J_{ii}|$$

This is why flows use:
- Coupling layers (block triangular)
- Autoregressive flows (triangular)

**Example - Affine Coupling Layer:**
```
Split: x = [x_a, x_b]
Transform: y_b = x_b ⊙ exp(s(x_a)) + t(x_a)

Jacobian is triangular:
J = [I  0]
    [*  diag(exp(s(x_a)))]

log|det J| = sum(s(x_a))  ← Very efficient!
```

### 7.9 Regularization and Determinant

In some regularization schemes:

**Log-Determinant Regularization:**
$$\mathcal{L} = \text{loss} - \lambda \log\det(\Sigma)$$

Encourages non-degenerate covariance estimates.

**Relationship to Trace:**
For covariance regularization:
- $\text{tr}(\Sigma)$ = sum of variances
- $\det(\Sigma)$ = product of variances
- These provide different regularization effects

### 7.10 Condition Number and Numerical Stability

**Condition Number:**
$$\kappa(A) = \|A\| \cdot \|A^{-1}\| = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{|\lambda_{\max}|}{|\lambda_{\min}|}$$

**Relationship to Determinant:**
- Small $\det(A)$ doesn't always mean ill-conditioned
- If $\det(A)$ is tiny relative to matrix size: likely ill-conditioned
- Use condition number for numerical stability checks, not determinant

```
Example of misleading determinant:

A = [10^10    0  ]  det(A) = 10^10  (large, but...)
    [  0   10^-10]
    
Condition number κ(A) = 10^20  (extremely ill-conditioned!)
```

---

## 8. Computational Considerations

### 8.1 Complexity Comparison

| Method | Complexity | Memory | Best For |
|--------|------------|--------|----------|
| 2×2 formula | $O(1)$ | $O(1)$ | 2×2 matrices |
| Sarrus' rule | $O(1)$ | $O(1)$ | 3×3 matrices only |
| Cofactor expansion | $O(n!)$ | $O(n^2)$ | Symbolic, small matrices |
| Row reduction | $O(n^3)$ | $O(n^2)$ | General numerical |
| LU decomposition | $O(n^3)$ | $O(n^2)$ | When you also need LU |
| Cholesky (SPD) | $O(n^3/3)$ | $O(n^2)$ | Symmetric positive definite |
| Triangular | $O(n)$ | $O(1)$ | Already triangular |

### 8.2 Numerical Issues

**Problem 1: Overflow**
Determinants can be astronomically large.

```python
import numpy as np

# 100×100 matrix with entries ~1
A = np.random.randn(100, 100) + 2
det_A = np.linalg.det(A)  # Could be 10^50 or higher!
```

**Problem 2: Underflow**
Determinants can be incredibly small.

```python
# 100×100 matrix with small diagonal
A = np.diag(np.full(100, 0.1))
det_A = np.linalg.det(A)  # = 0.1^100 ≈ 10^-100 (underflows!)
```

**Problem 3: Near-Singular Matrices**
```python
# Nearly singular: det ≈ 0 but not exactly
A = np.array([[1, 1], [1, 1.00000001]])
det_A = np.linalg.det(A)  # ≈ 10^-8, very sensitive to perturbations
```

### 8.3 Log-Determinant: The Solution

For positive definite matrices, use the **log-determinant**:

$$\log\det(\Sigma) = \sum_{i=1}^n \log(\lambda_i) = 2\sum_{i=1}^n \log(L_{ii})$$

Where $L$ is the Cholesky factor ($\Sigma = LL^T$).

**Benefits:**
1. Avoids overflow/underflow
2. Products become sums
3. More numerically stable

```python
import numpy as np

def log_det_cholesky(Sigma):
    """Stable log-determinant via Cholesky decomposition."""
    L = np.linalg.cholesky(Sigma)
    return 2 * np.sum(np.log(np.diag(L)))

# Example
Sigma = np.array([[4.0, 2.0], [2.0, 3.0]])
log_det = log_det_cholesky(Sigma)  # = log(8) ≈ 2.079
```

**Using scipy:**
```python
from scipy.linalg import cho_factor, cho_solve

c, lower = cho_factor(Sigma)
log_det = 2 * np.sum(np.log(np.diag(c)))
```

### 8.4 Sign and Log-Determinant

For general matrices (not necessarily positive definite), use `slogdet`:

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
sign, logdet = np.linalg.slogdet(A)
# sign = -1.0 (negative determinant)
# logdet = log(2) ≈ 0.693

# Reconstruct: det(A) = sign * exp(logdet)
det_A = sign * np.exp(logdet)  # = -2.0
```

This separates the sign from the magnitude, handling negative determinants.

### 8.5 Numerical Rank vs Determinant

**Don't use determinant to check numerical rank!**

```
Problem: Determinant scales poorly with matrix size and entries

Example:
A = [0.01  0  ]   det(A) = 0.0001  (small, but full rank!)
    [ 0   0.01]

B = [1  1]        det(B) = 0.0001  (small because nearly singular)
    [1  1.0001]

Both have det ≈ 0.0001 but for completely different reasons!
```

**Better approaches:**
1. **Singular Value Decomposition (SVD):** Check smallest singular values
2. **Condition Number:** $\kappa = \sigma_{\max}/\sigma_{\min}$
3. **Rank-Revealing QR:** Identifies numerical rank

```python
import numpy as np

def numerical_rank(A, tol=1e-10):
    """Determine numerical rank via SVD."""
    _, s, _ = np.linalg.svd(A)
    return np.sum(s > tol)

# Check if matrix is numerically singular
def is_singular(A, tol=1e-10):
    return numerical_rank(A, tol) < min(A.shape)
```

### 8.6 When to Compute Determinants (and When Not To)

**DO compute determinants for:**
- Multivariate Gaussian probabilities (use log-det)
- Jacobian in change of variables
- Checking if solution exists (det ≠ 0)
- Small matrices (2×2, 3×3) in performance-critical code
- Geometric applications (areas, volumes)

**DON'T compute determinants for:**
- Solving linear systems (use LU decomposition)
- Matrix inversion (use direct methods)
- Numerical rank determination (use SVD)
- Large matrices if only checking singularity (use condition number)

### 8.7 Implementation Notes

**NumPy:**
```python
# Standard determinant
det = np.linalg.det(A)

# Log-determinant with sign
sign, logdet = np.linalg.slogdet(A)

# Reconstruct when needed
det = sign * np.exp(logdet)
```

**SciPy (more options):**
```python
from scipy import linalg

# Using LU decomposition
lu, piv = linalg.lu_factor(A)
det = np.prod(np.diag(lu)) * ((-1) ** np.sum(piv != np.arange(len(piv))))

# For positive definite: Cholesky
L = linalg.cholesky(Sigma, lower=True)
log_det = 2 * np.sum(np.log(np.diag(L)))
```

**PyTorch (for ML):**
```python
import torch

A = torch.randn(3, 3)
det = torch.det(A)

# Log-determinant (differentiable!)
logdet = torch.logdet(A)  # For positive definite
sign, logdet = torch.slogdet(A)  # General case
```

### 8.8 Determinant in Backpropagation

For normalizing flows and other models, we need gradients of log-determinants:

$$\frac{\partial}{\partial A} \log\det(A) = A^{-T}$$

This is used in training flow-based models where log $|\det J|$ appears in the loss.

---

## 9. The Adjugate Matrix

### 9.1 Definition

The **adjugate** (or **classical adjoint**) of a matrix $A$ is:

$$\text{adj}(A) = C^T$$

Where $C$ is the **cofactor matrix**: $C_{ij} = (-1)^{i+j} M_{ij}$

### 9.2 Properties

**Fundamental Identity:**
$$A \cdot \text{adj}(A) = \text{adj}(A) \cdot A = \det(A) \cdot I$$

**Inverse Formula:**
$$A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$$

**Example (2×2):**
$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \implies \text{adj}(A) = \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

$$A^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

### 9.3 Intuition

The adjugate provides the inverse "without division":
- $A \cdot \text{adj}(A) = \det(A) \cdot I$
- Divide by $\det(A)$ to get the identity
- Hence $\text{adj}(A)/\det(A) = A^{-1}$

---

## 10. Advanced Topics

### 10.1 Characteristic Polynomial

The **characteristic polynomial** of $A$ is:

$$p(\lambda) = \det(\lambda I - A)$$

Its roots are the eigenvalues of $A$.

**For 2×2:**
$$p(\lambda) = \lambda^2 - \text{tr}(A)\lambda + \det(A)$$

**For 3×3:**
$$p(\lambda) = \lambda^3 - \text{tr}(A)\lambda^2 + \frac{1}{2}(\text{tr}(A)^2 - \text{tr}(A^2))\lambda - \det(A)$$

### 10.2 Cayley-Hamilton Theorem

Every matrix satisfies its own characteristic polynomial:

$$p(A) = A^n - c_{n-1}A^{n-1} - \cdots - c_1 A - c_0 I = 0$$

**Consequence:**
$$A^{-1} = -\frac{1}{c_0}(A^{n-1} - c_{n-1}A^{n-2} - \cdots - c_1 I)$$

(where $c_0 = (-1)^n\det(A)$)

### 10.3 Determinant of Block Matrices

**For general 2×2 blocks:**

If $CD = DC$ (blocks commute):
$$\det\begin{bmatrix} A & B \\ C & D \end{bmatrix} = \det(AD - BC)$$

**Vandermonde Determinant:**
$$\det\begin{bmatrix} 1 & x_1 & x_1^2 & \cdots & x_1^{n-1} \\ 1 & x_2 & x_2^2 & \cdots & x_2^{n-1} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_n & x_n^2 & \cdots & x_n^{n-1} \end{bmatrix} = \prod_{1 \leq i < j \leq n} (x_j - x_i)$$

### 10.4 Permanent vs Determinant

The **permanent** is similar to determinant but without alternating signs:

$$\text{perm}(A) = \sum_{\sigma} \prod_{i=1}^n a_{i,\sigma(i)}$$

- Computing permanent is #P-complete (much harder than determinant)
- Used in quantum computing and combinatorics
- Permanent of a 0-1 matrix counts perfect matchings

### 10.5 Matrix Functions and Determinant

For matrix exponential:
$$\det(e^A) = e^{\text{tr}(A)}$$

For matrix logarithm (when it exists):
$$\text{tr}(\log A) = \log\det(A)$$

---

## 11. Summary

### 11.1 Key Formulas

| Concept | Formula |
|---------|---------|
| 2×2 determinant | $\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$ |
| Invertibility | $\det(A) \neq 0 \iff A$ is invertible |
| Product rule | $\det(AB) = \det(A)\det(B)$ |
| Transpose | $\det(A^T) = \det(A)$ |
| Inverse | $\det(A^{-1}) = 1/\det(A)$ |
| Scalar | $\det(cA) = c^n\det(A)$ |
| Eigenvalue product | $\det(A) = \prod_{i=1}^n \lambda_i$ |
| Triangular | $\det(A) = \prod_{i=1}^n a_{ii}$ |
| Adjugate inverse | $A^{-1} = \text{adj}(A)/\det(A)$ |

### 11.2 Geometric Interpretation Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    DETERMINANT MEANINGS                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  det(A) = 0:  Singular matrix                               │
│               - Columns are linearly dependent              │
│               - Maps to lower dimension                     │
│               - No inverse exists                           │
│               - At least one eigenvalue is zero             │
│                                                             │
│  det(A) > 0:  Invertible, preserves orientation             │
│               - Right-handed → Right-handed                 │
│               - Counterclockwise → Counterclockwise         │
│                                                             │
│  det(A) < 0:  Invertible, reverses orientation              │
│               - Right-handed → Left-handed                  │
│               - Includes a reflection                       │
│                                                             │
│  |det(A)|:    Volume scaling factor                         │
│               - How much the transformation scales volumes  │
│               - Area in 2D, volume in 3D, hypervolume in nD │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 11.3 When to Use Determinants

**✅ Good Uses:**
- Check invertibility: Is $\det(A) \neq 0$?
- Compute volume change under transformation
- Jacobian in change of variables (normalizing flows)
- Multivariate Gaussian normalization (use log-det)
- Small matrix inverse via adjugate (2×2, 3×3)
- Geometric calculations (areas, collinearity tests)

**❌ Avoid:**
- Solving linear systems (use LU decomposition)
- Large matrix computations (use specialized methods)
- Numerical rank checking (use SVD)
- Direct determinant computation for ML (use log-det)

### 11.4 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    DETERMINANT CHEAT SHEET                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  COMPUTATION:                                               │
│  • 2×2: ad - bc                                             │
│  • 3×3: Sarrus' rule or cofactor expansion                  │
│  • n×n: LU decomposition, multiply diagonal                 │
│  • Positive definite: Cholesky, log(det) = 2Σlog(L_ii)     │
│                                                             │
│  PROPERTIES:                                                │
│  • det(AB) = det(A)det(B)                                   │
│  • det(A^T) = det(A)                                        │
│  • det(A^{-1}) = 1/det(A)                                   │
│  • det(cA) = c^n det(A)                                     │
│  • det(A) = ∏λᵢ                                             │
│                                                             │
│  ROW OPERATIONS:                                            │
│  • Swap rows: det → -det                                    │
│  • Scale row by c: det → c·det                              │
│  • Add rows: det → det (unchanged)                          │
│                                                             │
│  SPECIAL MATRICES:                                          │
│  • Triangular: det = ∏ diagonal                             │
│  • Orthogonal: det = ±1                                     │
│  • Rotation: det = +1                                       │
│  • Reflection: det = -1                                     │
│                                                             │
│  NUMERICAL:                                                 │
│  • Use np.linalg.slogdet() for stability                    │
│  • Use Cholesky for positive definite matrices              │
│  • Don't use det for checking numerical rank                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 12. Practice Problems

### Conceptual Questions

1. If $\det(A) = 3$ and $\det(B) = 4$, what is $\det(AB)$? $\det(A^2B)$?

2. A matrix $A$ has eigenvalues 2, 3, and 5. What is $\det(A)$? What is $\det(A^{-1})$?

3. If swapping two rows of $A$ gives $B$, how are $\det(A)$ and $\det(B)$ related?

4. Why is the determinant of a rotation matrix always 1?

5. If $A$ is $3 \times 3$ and $\det(A) = 2$, what is $\det(3A)$?

### Computational Problems

1. Compute by hand:
   $$\det\begin{bmatrix} 2 & 1 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{bmatrix}$$

2. Use cofactor expansion to find:
   $$\det\begin{bmatrix} 1 & 0 & 2 \\ 3 & 1 & 0 \\ 0 & 2 & 1 \end{bmatrix}$$

3. Find the area of the parallelogram with vertices at $(0,0)$, $(3,1)$, $(1,4)$, $(4,5)$.

4. Solve using Cramer's rule:
   $$\begin{cases} 2x + 3y = 8 \\ 4x - y = 2 \end{cases}$$

### ML/Applied Problems

1. A covariance matrix has eigenvalues 4 and 1. What is the generalized variance?

2. In a normalizing flow, a transformation has Jacobian with diagonal entries $[e^{s_1}, e^{s_2}, e^{s_3}]$. Express $\log|\det J|$ in terms of $s_1, s_2, s_3$.

3. Why do flow-based models prefer triangular Jacobians?

---

## 13. Further Reading

### Textbooks
- **Gilbert Strang, "Linear Algebra and Its Applications"** - Excellent geometric intuition
- **David C. Lay, "Linear Algebra and Its Applications"** - Clear introductory treatment
- **Carl D. Meyer, "Matrix Analysis and Applied Linear Algebra"** - Advanced topics

### Online Resources
- [3Blue1Brown: The Determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk) - Visual geometric intuition
- [Gilbert Strang: Properties of Determinants](https://www.youtube.com/watch?v=srxexLishgY) - MIT OCW lecture
- [Matrix Cookbook: Determinant Identities](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf) - Reference

### ML-Specific
- [Normalizing Flows Tutorial](https://arxiv.org/abs/1908.09257) - Jacobian determinants in deep learning
- [A Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) - Change of variables

---

## Navigation

← [Previous: Systems of Equations](../03-Systems-of-Equations/README.md) | [Next: Matrix Rank →](../05-Matrix-Rank/README.md)

[Back to Linear Algebra Basics](../README.md) | [Back to Main](../../README.md)
