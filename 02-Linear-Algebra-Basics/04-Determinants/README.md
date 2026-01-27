# Determinants

## Overview

The determinant is a scalar value that encodes important properties of a square matrix. It tells us about linear independence, invertibility, and how transformations affect volume.

## Learning Objectives

- Understand the geometric meaning of determinants
- Compute determinants efficiently
- Connect determinants to matrix properties
- Apply determinants in ML contexts

---

## 1. What is a Determinant?

### 1.1 Definition

The **determinant** of a square matrix $A$ is a scalar, denoted $\det(A)$ or $|A|$.

**2×2 Matrix:**
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**3×3 Matrix (Sarrus' Rule):**
$$\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = aei + bfg + cdh - ceg - bdi - afh$$

### 1.2 Geometric Interpretation

```
The determinant measures SIGNED VOLUME:

2D: Area of parallelogram formed by column vectors
3D: Volume of parallelepiped formed by column vectors

Original unit square:        After transformation A:
┌─────────────┐              ┌─────────────┐
│             │              │      /\     │
│      1      │  →  A  →    │     /  \    │  Area = |det(A)|
│             │              │    /    \   │
└─────────────┘              └─────────────┘

If det(A) > 0: Orientation preserved
If det(A) < 0: Orientation flipped (reflection)
If det(A) = 0: Collapsed to lower dimension (singular)
```

### 1.3 Key Property: Invertibility

$$\text{A is invertible} \iff \det(A) \neq 0$$

```
Invertibility Decision:

det(A) ≠ 0:                    det(A) = 0:
- Columns are linearly         - Columns are linearly
  independent                    dependent
- Full rank                    - Rank deficient
- A⁻¹ exists                   - A⁻¹ does NOT exist
- Unique solution to Ax=b      - No unique solution
```

---

## 2. Computing Determinants

### 2.1 2×2 Determinant

$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

```
Visual:
┌───────────┐
│  a    b   │
│     ╲  ╱  │  = ad - bc
│      ╲╱   │
│  c    d   │  (main diagonal minus anti-diagonal)
└───────────┘
```

**Example:**
$$\det\begin{bmatrix} 3 & 2 \\ 1 & 4 \end{bmatrix} = (3)(4) - (2)(1) = 12 - 2 = 10$$

### 2.2 3×3 Determinant (Cofactor Expansion)

Expand along any row or column:

$$\det(A) = \sum_j (-1)^{i+j} a_{ij} M_{ij}$$

Where $M_{ij}$ is the **minor** (determinant of submatrix with row $i$ and column $j$ removed).

```
Expansion along first row:

det│ a  b  c │     │e f│       │d f│       │d e│
   │ d  e  f │ = a│h i│ - b │g i│ + c │g h│
   │ g  h  i │
```

**Example:**
$$\det\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$$

Expand along row 1:
$$= 1 \cdot \det\begin{bmatrix} 5 & 6 \\ 8 & 9 \end{bmatrix} - 2 \cdot \det\begin{bmatrix} 4 & 6 \\ 7 & 9 \end{bmatrix} + 3 \cdot \det\begin{bmatrix} 4 & 5 \\ 7 & 8 \end{bmatrix}$$
$$= 1(45-48) - 2(36-42) + 3(32-35)$$
$$= 1(-3) - 2(-6) + 3(-3)$$
$$= -3 + 12 - 9 = 0$$

The determinant is 0, so this matrix is **singular** (not invertible).

### 2.3 Efficient Computation: LU Decomposition

For large matrices, use row reduction:

$$\det(A) = \det(L) \cdot \det(U) = 1 \cdot \prod_{i} u_{ii}$$

Determinant of a triangular matrix = product of diagonal elements.

```
If A = LU:
            ┌─────────────┐
det(A) =    │ u₁₁         │
            │  ╲  u₂₂     │  = u₁₁ × u₂₂ × u₃₃
            │   ╲  ╲ u₃₃  │
            └─────────────┘
```

---

## 3. Properties of Determinants

### 3.1 Fundamental Properties

| Property  | Formula                     | Notes                   |
| --------- | --------------------------- | ----------------------- |
| Identity  | $\det(I) = 1$               |                         |
| Transpose | $\det(A^T) = \det(A)$       |                         |
| Product   | $\det(AB) = \det(A)\det(B)$ | Very important!         |
| Inverse   | $\det(A^{-1}) = 1/\det(A)$  | If $A$ is invertible    |
| Scalar    | $\det(cA) = c^n \det(A)$    | For $n \times n$ matrix |
| Power     | $\det(A^k) = (\det(A))^k$   |                         |

### 3.2 Row Operations and Determinants

```
Operation                           Effect on Determinant
─────────────────────────────────────────────────────────
Swap two rows                       det → -det (sign flip)
Multiply row by c                   det → c × det
Add multiple of one row to another  det → det (unchanged)
```

### 3.3 Special Matrix Determinants

```
Matrix Type              Determinant
──────────────────────────────────────────────────
Diagonal                 Product of diagonal elements
Triangular               Product of diagonal elements
Orthogonal (Q)           det(Q) = ±1
Rotation                 det(R) = +1
Reflection               det(R) = -1
Rank-deficient           det = 0
Block diagonal           Product of block determinants
```

---

## 4. Geometric Applications

### 4.1 Area of Parallelogram

Two vectors $\vec{u} = (u_1, u_2)$ and $\vec{v} = (v_1, v_2)$ form a parallelogram.

$$\text{Area} = \left| \det\begin{bmatrix} u_1 & v_1 \\ u_2 & v_2 \end{bmatrix} \right|$$

```
              ╱╲
             ╱  ╲ v
            ╱    ╲
           ╱      ╲
          ●────────●
           u

Area = |u₁v₂ - u₂v₁|
```

### 4.2 Volume of Parallelepiped

Three vectors form a parallelepiped:

$$\text{Volume} = \left| \det\begin{bmatrix} u_1 & v_1 & w_1 \\ u_2 & v_2 & w_2 \\ u_3 & v_3 & w_3 \end{bmatrix} \right|$$

### 4.3 Cross Product Connection

The cross product in 3D can be computed using determinants:

$$\vec{u} \times \vec{v} = \det\begin{bmatrix} \hat{i} & \hat{j} & \hat{k} \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{bmatrix}$$

---

## 5. Determinants and Linear Transformations

### 5.1 Transformation Scaling Factor

The determinant tells us how a linear transformation scales areas/volumes:

```
Original Area A₀ = 1 (unit square)

After transformation T:
Area = |det(T)| × A₀

Example: T = [2  0]  → det(T) = 4
             [0  2]
Unit square becomes 2×2 square with area 4.
```

### 5.2 Orientation

$$\det(A) > 0 \implies \text{preserves orientation (no reflection)}$$
$$\det(A) < 0 \implies \text{reverses orientation (includes reflection)}$$

```
det > 0:                    det < 0:
(preserves orientation)     (flips orientation)

  ↑ y                         ↑ y
  │  2                        │  1
  │ ↗                         │  ↖
  │1                          │    2
──┼────→ x                  ──┼────→ x

Same handedness             Opposite handedness
```

---

## 6. Cramer's Rule

### 6.1 Formula

For a system $Ax = b$ with $\det(A) \neq 0$:

$$x_i = \frac{\det(A_i)}{\det(A)}$$

Where $A_i$ is matrix $A$ with column $i$ replaced by $b$.

### 6.2 Example

$$\begin{cases} 2x + y = 5 \\ x + 3y = 6 \end{cases}$$

$$A = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}, \quad b = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$$

$$x = \frac{\det\begin{bmatrix} 5 & 1 \\ 6 & 3 \end{bmatrix}}{\det\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}} = \frac{15-6}{6-1} = \frac{9}{5}$$

$$y = \frac{\det\begin{bmatrix} 2 & 5 \\ 1 & 6 \end{bmatrix}}{\det\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}} = \frac{12-5}{5} = \frac{7}{5}$$

> **Note:** Cramer's rule is elegant but computationally expensive for large systems. Use LU decomposition instead!

---

## 7. Machine Learning Applications

### 7.1 Covariance Matrix Determinant

The determinant of a covariance matrix $\Sigma$ is called the **generalized variance**:

$$\det(\Sigma) = \text{Volume of uncertainty ellipsoid}$$

```
Large det(Σ):                Small det(Σ):
High overall variance        Low overall variance
Data spread out              Data concentrated

    ╭─────────╮                 ╭───╮
   ╱           ╲               │   │
  │             │              │   │
   ╲           ╱               │   │
    ╰─────────╯                 ╰───╯
```

### 7.2 Gaussian PDF Normalization

The multivariate Gaussian PDF uses the determinant:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

The normalization constant includes $|\Sigma|^{-1/2}$.

### 7.3 Model Selection (BIC uses log-likelihood)

In Bayesian model selection:
$$\text{Log-likelihood} \propto -\frac{n}{2}\log|\Sigma|$$

### 7.4 PCA and Explained Variance

Product of eigenvalues equals determinant:
$$\det(A) = \prod_{i=1}^n \lambda_i$$

In PCA, this relates to total variance explained.

### 7.5 Jacobian Determinant (Change of Variables)

When transforming variables in probability:

$$p_Y(y) = p_X(g^{-1}(y)) \left| \det\left(\frac{\partial g^{-1}}{\partial y}\right) \right|$$

Used in:

- Normalizing flows
- Variational autoencoders
- Change of variables in integrals

---

## 8. Computational Considerations

### 8.1 Complexity

| Method                   | Complexity | Notes                  |
| ------------------------ | ---------- | ---------------------- |
| Cofactor expansion       | $O(n!)$    | Never use for $n > 4$  |
| LU decomposition         | $O(n^3)$   | Standard method        |
| Specialized (triangular) | $O(n)$     | Just multiply diagonal |

### 8.2 Numerical Issues

```
Potential Problems:

1. Overflow: det can be astronomically large
   Solution: Use log-determinant

2. Underflow: det can be tiny
   Solution: Use log-determinant

3. Near-singular: det ≈ 0 but not exactly
   Solution: Check condition number instead
```

### 8.3 Log-Determinant

For numerical stability with positive definite matrices:
$$\log\det(\Sigma) = \sum_i \log(\lambda_i) = 2 \sum_i \log(L_{ii})$$

where $L$ is the Cholesky factor.

---

## 9. Summary

### Key Formulas

| Concept            | Formula                            |
| ------------------ | ---------------------------------- |
| 2×2 determinant    | $ad - bc$                          |
| Invertibility      | $\det(A) \neq 0 \iff A$ invertible |
| Product rule       | $\det(AB) = \det(A)\det(B)$        |
| Transpose          | $\det(A^T) = \det(A)$              |
| Eigenvalue product | $\det(A) = \prod \lambda_i$        |

### Interpretation Summary

```
det(A) = 0  → Singular, no inverse, dependent columns
det(A) > 0  → Invertible, preserves orientation
det(A) < 0  → Invertible, flips orientation
|det(A)|    → Scale factor for volumes
```

### When to Use Determinants

✅ **Good uses:**

- Check invertibility (is det ≠ 0?)
- Compute volume change
- Jacobian in change of variables
- Multivariate Gaussian normalization

❌ **Avoid:**

- Solving linear systems (use LU instead)
- Large matrix determinants (use log-det)
- Numerical rank checking (use SVD)

---

## Further Reading

- [3Blue1Brown: The Determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk)
- [Gilbert Strang: Properties of Determinants](https://www.youtube.com/watch?v=srxexLishgY)
- [Matrix Cookbook: Determinant Identities](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)

---

## Navigation

← [Previous: Systems of Equations](../03-Systems-of-Equations/README.md) | [Next: Matrix Rank →](../05-Matrix-Rank/README.md)

[Back to Main](../../README.md)
