# Matrix Operations

## Overview

Matrices are the fundamental data structure for organizing and transforming data in machine learning. Every neural network layer, every image, and every dataset can be represented as a matrix.

## Learning Objectives

- Understand matrix representations and notations
- Master basic matrix arithmetic operations
- Learn special matrix types and their properties
- Apply matrix operations to ML problems

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

---

## 4. Element-wise Operations

### 4.1 Hadamard Product (Element-wise Multiplication)

$$C = A \odot B \text{ where } c_{ij} = a_{ij} \cdot b_{ij}$$

```
Hadamard Product:

  ┌─────────┐     ┌─────────┐     ┌─────────┐
  │  1   2  │  ⊙  │  5   6  │  =  │  5  12  │
  │  3   4  │     │  7   8  │     │ 21  32  │
  └─────────┘     └─────────┘     └─────────┘
```

**ML Use:** Attention mechanisms, gating in LSTMs/GRUs.

### 4.2 Broadcasting

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

---

## 5. Matrix Views and Slicing

### 5.1 Row and Column Vectors

```
Matrix A:                 Row Vector (1×n)       Column Vector (m×1)
┌─────────────┐           ┌─────────────┐        ┌─────┐
│ a₁₁ a₁₂ a₁₃│  Row 1 →  │ a₁₁ a₁₂ a₁₃│        │ a₁₁ │
│ a₂₁ a₂₂ a₂₃│                                   │ a₂₁ │  ← Col 1
│ a₃₁ a₃₂ a₃₃│                                   │ a₃₁ │
└─────────────┘                                   └─────┘
```

### 5.2 Submatrices

```
Extracting a 2×2 submatrix from rows 1-2, cols 2-3:

┌─────────────────────┐
│ a₁₁  [a₁₂  a₁₃] a₁₄│      ┌───────────┐
│ a₂₁  [a₂₂  a₂₃] a₂₄│  →   │ a₁₂  a₁₃ │
│ a₃₁   a₃₂  a₃₃  a₃₄│      │ a₂₂  a₂₃ │
└─────────────────────┘      └───────────┘
```

---

## 6. Machine Learning Applications

### 6.1 Data Representation

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

### 6.2 Linear Layer Transformation

A neural network linear layer: $y = Wx + b$

```
Input      ×      Weights      +    Bias      =    Output
 x              W                   b               y

[x₁]         [w₁₁ w₁₂ w₁₃]       [b₁]          [y₁]
[x₂]    ×    [w₂₁ w₂₂ w₂₃]   +   [b₂]    =     [y₂]
[x₃]

(3×1)        (2×3)               (2×1)         (2×1)
```

### 6.3 Batch Processing

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

### 6.4 Covariance Matrix

$$\Sigma = \frac{1}{n-1}(X - \bar{X})^T(X - \bar{X})$$

```
Covariance Matrix Properties:
- Symmetric: Σ = Σᵀ
- Size: d × d (number of features)
- Diagonal: variances of each feature
- Off-diagonal: covariances between features
```

### 6.5 Image as Matrix

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

---

## 7. Computational Considerations

### 7.1 Time Complexity

| Operation                           | Complexity |
| ----------------------------------- | ---------- |
| Addition (m×n)                      | $O(mn)$    |
| Scalar multiplication               | $O(mn)$    |
| Matrix multiplication (m×n) × (n×p) | $O(mnp)$   |
| Element-wise operations             | $O(mn)$    |

### 7.2 Memory Layout

```
Row-major (C, NumPy default):        Column-major (Fortran):
┌─────────┐                          ┌─────────┐
│ 1  2  3 │ Memory: [1,2,3,4,5,6]   │ 1  2  3 │ Memory: [1,4,2,5,3,6]
│ 4  5  6 │                          │ 4  5  6 │
└─────────┘                          └─────────┘
```

### 7.3 Optimization Tips

1. **Matrix multiplication order matters:**
   - $(AB)C$ vs $A(BC)$ can have very different costs
   - Example: $A(100×2), B(2×100), C(100×2)$
     - $(AB)C$: $(100×100) \times (100×2)$ = expensive
     - $A(BC)$: $(100×2) \times (2×2)$ = cheap

2. **Avoid explicit loops** - use vectorized operations
3. **Pre-allocate** memory for large matrices
4. **Use appropriate data types** - float32 vs float64

---

## 8. Summary

### Key Formulas

| Operation   | Formula         | Dimensions           |
| ----------- | --------------- | -------------------- |
| Addition    | $C = A + B$     | Same as inputs       |
| Scalar mult | $B = cA$        | Same as $A$          |
| Matrix mult | $C = AB$        | $(m×n)(n×p) = (m×p)$ |
| Transpose   | $B = A^T$       | $(m×n)^T = (n×m)$    |
| Hadamard    | $C = A \odot B$ | Same as inputs       |

### Mental Checklist

✅ Check dimensions before multiplication
✅ Remember: $AB \neq BA$ in general
✅ $(AB)^T = B^T A^T$ (reverse order!)
✅ Symmetric matrices have real eigenvalues
✅ Use broadcasting for efficiency

---

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)

---

## Navigation

← [Previous: Vectors and Spaces](../01-Vectors-and-Spaces/README.md) | [Next: Systems of Equations →](../03-Systems-of-Equations/README.md)

[Back to Main](../../README.md)
