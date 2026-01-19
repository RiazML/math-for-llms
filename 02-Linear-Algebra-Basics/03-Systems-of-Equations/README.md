# Systems of Linear Equations

## Overview

Solving systems of linear equations is one of the most fundamental applications of linear algebra. Every machine learning model that has a closed-form solution relies on solving such systems.

## Learning Objectives

- Understand different forms of linear systems
- Master Gaussian elimination and row reduction
- Learn when systems have unique, infinite, or no solutions
- Apply these concepts to ML problems like linear regression

---

## 1. Introduction to Linear Systems

### 1.1 What is a Linear System?

A **system of linear equations** is a collection of equations that can be written as:

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

### 1.2 Matrix Form

This system can be written compactly as:

$$Ax = b$$

Where:

- $A \in \mathbb{R}^{m \times n}$ is the **coefficient matrix**
- $x \in \mathbb{R}^n$ is the **unknown vector**
- $b \in \mathbb{R}^m$ is the **right-hand side** (or target vector)

```
Ax = b

┌─────────────────┐   ┌─────┐     ┌─────┐
│ a₁₁  a₁₂  a₁₃  │   │ x₁  │     │ b₁  │
│ a₂₁  a₂₂  a₂₃  │ × │ x₂  │  =  │ b₂  │
│ a₃₁  a₃₂  a₃₃  │   │ x₃  │     │ b₃  │
└─────────────────┘   └─────┘     └─────┘
    Coefficients     Unknowns   Right-hand side
```

### 1.3 Geometric Interpretation

**2D Example:** Each equation represents a line

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

**3D Example:** Each equation represents a plane

```
Three planes can:
- Intersect at a point (unique solution)
- Intersect along a line (infinite solutions)
- Have no common intersection (no solution)
```

---

## 2. Types of Solutions

### 2.1 Solution Categories

```
┌─────────────────────────────────────────────────────────────┐
│                    SOLUTION TYPES                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. UNIQUE SOLUTION       2. NO SOLUTION      3. INFINITE   │
│     (Consistent,             (Inconsistent)      SOLUTIONS  │
│      Independent)                                (Dependent)│
│                                                             │
│      ╲   ╱                    │  ╱                   ╱      │
│       ╲ ╱                     │ ╱                   ╱       │
│        ●                      │╱                   ╱        │
│       ╱ ╲                   ╱ │               ═══════       │
│      ╱   ╲                 ╱  │               overlapping   │
│                           Parallel lines      lines         │
│                                                             │
│  rank(A) = rank([A|b])   rank(A) < rank([A|b])   rank(A) =  │
│     = n (# unknowns)                              rank([A|b])│
│                                                   < n       │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Determining Solution Type

The **augmented matrix** $[A | b]$ combines $A$ and $b$:

$$[A | b] = \begin{bmatrix} a_{11} & a_{12} & | & b_1 \\ a_{21} & a_{22} & | & b_2 \end{bmatrix}$$

**Rules:**

- If $\text{rank}(A) = \text{rank}([A|b]) = n$: **Unique solution**
- If $\text{rank}(A) = \text{rank}([A|b]) < n$: **Infinite solutions**
- If $\text{rank}(A) < \text{rank}([A|b])$: **No solution**

---

## 3. Gaussian Elimination

### 3.1 Elementary Row Operations

Three allowed operations that don't change the solution:

```
1. SWAP: Exchange two rows
   R₁ ↔ R₂

2. SCALE: Multiply a row by a non-zero constant
   R₁ → c·R₁  (c ≠ 0)

3. ADD: Add a multiple of one row to another
   R₂ → R₂ + c·R₁
```

### 3.2 Row Echelon Form (REF)

A matrix is in **row echelon form** if:

1. All zero rows are at the bottom
2. The leading entry (pivot) of each row is to the right of the pivot above it
3. All entries below a pivot are zero

```
Row Echelon Form Examples:

┌─────────────┐     ┌───────────────┐     ┌─────────────────┐
│ ■  *  *  *  │     │ ■  *  *  *  * │     │ 0  ■  *  *  *  │
│ 0  ■  *  *  │     │ 0  ■  *  *  * │     │ 0  0  0  ■  *  │
│ 0  0  ■  *  │     │ 0  0  0  ■  * │     │ 0  0  0  0  ■  │
│ 0  0  0  ■  │     │ 0  0  0  0  0 │     │ 0  0  0  0  0  │
└─────────────┘     └───────────────┘     └─────────────────┘
  Full rank          Rank = 3 < 4          Pivots in cols 2,4,5

■ = pivot (non-zero)
* = any value
```

### 3.3 Gaussian Elimination Algorithm

**Example:** Solve the system:
$$\begin{cases} x + 2y + z = 9 \\ 2x - y + 3z = 8 \\ 3x + y - z = 3 \end{cases}$$

```
Step 1: Form augmented matrix
┌───────────────────┐
│  1   2   1  │  9  │
│  2  -1   3  │  8  │
│  3   1  -1  │  3  │
└───────────────────┘

Step 2: Eliminate below first pivot
R₂ → R₂ - 2·R₁
R₃ → R₃ - 3·R₁
┌───────────────────┐
│  1   2   1  │  9  │
│  0  -5   1  │ -10 │
│  0  -5  -4  │ -24 │
└───────────────────┘

Step 3: Eliminate below second pivot
R₃ → R₃ - R₂
┌───────────────────┐
│  1   2   1  │  9  │  (REF achieved)
│  0  -5   1  │ -10 │
│  0   0  -5  │ -14 │
└───────────────────┘

Step 4: Back substitution
-5z = -14  →  z = 14/5
-5y + z = -10  →  y = (10 + 14/5)/5 = 64/25
x + 2y + z = 9  →  x = 9 - 128/25 - 14/5 = 9 - 128/25 - 70/25 = 27/25
```

### 3.4 Reduced Row Echelon Form (RREF)

Additional requirements:

- Each pivot equals 1
- Pivots are the only non-zero entries in their columns

```
RREF Example:
┌───────────────┐          ┌───────────────┐
│  1   2   3  4 │          │  1   0   0  a │
│  0  -2   1  3 │   →      │  0   1   0  b │
│  0   0   5  2 │   RREF   │  0   0   1  c │
└───────────────┘          └───────────────┘
```

---

## 4. Special Cases

### 4.1 Homogeneous Systems

When $b = 0$: $Ax = 0$

```
Properties:
- Always has at least the trivial solution x = 0
- Non-trivial solutions exist if and only if rank(A) < n
- Solutions form a vector space (null space of A)
```

### 4.2 Overdetermined Systems (m > n)

More equations than unknowns - typically no exact solution.

```
Example: Fitting a line to 3 non-collinear points

Points: (1, 2), (2, 3), (3, 7)
System: Find a, b such that ax + b = y

┌───────────┐   ┌───┐     ┌───┐
│  1    1   │   │ a │     │ 2 │
│  2    1   │ × │ b │  =  │ 3 │
│  3    1   │           │ 7 │
└───────────┘   └───┘     └───┘

No exact solution! → Use least squares
```

### 4.3 Underdetermined Systems (m < n)

Fewer equations than unknowns - infinite solutions possible.

```
Example: x + y + z = 6

Infinite solutions parameterized by free variables:
x = 6 - y - z  (y and z are free variables)
```

---

## 5. Direct Methods for Solving

### 5.1 Using the Inverse

If $A$ is square and invertible:

$$Ax = b \implies x = A^{-1}b$$

```
Computational note:
- Computing A⁻¹ explicitly is expensive: O(n³)
- Solving Ax = b directly is often more efficient
- Never compute A⁻¹ just to multiply by b!
```

### 5.2 LU Decomposition

Factor $A = LU$ where:

- $L$ = lower triangular matrix
- $U$ = upper triangular matrix

```
Solving Ax = b with LU:

A = LU
Ax = b
LUx = b

Step 1: Solve Ly = b (forward substitution)
Step 2: Solve Ux = y (back substitution)

Advantage: If same A, different b values,
           the factorization can be reused!
```

### 5.3 Cholesky Decomposition

For **symmetric positive definite** matrices: $A = LL^T$

```
When to use:
- Covariance matrices
- Kernel matrices
- Any SPD matrix

Advantages:
- 2× faster than LU
- Numerically stable
- Requires half the storage
```

---

## 6. Machine Learning Applications

### 6.1 Linear Regression (Normal Equations)

The least squares solution minimizes $\|Ax - b\|^2$:

$$\hat{x} = (A^T A)^{-1} A^T b$$

```
Linear Regression Setup:

Data: (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)
Model: y = w₀ + w₁x

Design matrix X:        Target y:        Weights w:
┌───────────┐           ┌─────┐          ┌─────┐
│  1   x₁   │           │ y₁  │          │ w₀  │
│  1   x₂   │           │ y₂  │          │ w₁  │
│  ⋮    ⋮   │           │  ⋮  │          └─────┘
│  1   xₙ   │           │ yₙ  │
└───────────┘           └─────┘

Normal equations: (XᵀX)w = Xᵀy
Solution: w = (XᵀX)⁻¹Xᵀy
```

### 6.2 Ridge Regression

Adding regularization to handle ill-conditioned $A^T A$:

$$\hat{x} = (A^T A + \lambda I)^{-1} A^T b$$

```
Why regularization helps:

AᵀA might be:
- Singular (no inverse exists)
- Nearly singular (numerically unstable)

Adding λI:
- Always invertible for λ > 0
- Improves condition number
- Shrinks weights toward zero
```

### 6.3 Solving for Neural Network Equilibrium

Some network analyses require solving:

$$Wx + b = x$$

Rearranging: $(I - W)x = -b$

This finds **fixed points** of the network.

---

## 7. Numerical Considerations

### 7.1 Condition Number

The **condition number** $\kappa(A)$ measures sensitivity to perturbations:

$$\kappa(A) = \|A\| \|A^{-1}\|$$

```
Condition Number Interpretation:

κ(A) ≈ 1:     Well-conditioned, stable
κ(A) ~ 10³:   Some accuracy loss
κ(A) ~ 10⁶:   Significant accuracy loss
κ(A) → ∞:     Singular, cannot solve

Rule of thumb: Lose log₁₀(κ) digits of accuracy
```

### 7.2 Pivoting Strategies

**Partial Pivoting:** Select the largest element in current column as pivot.

```
Without pivoting:           With partial pivoting:
┌────────────┐              ┌────────────┐
│ 0.001  1  │              │   1     1  │ ← Swap rows
│   1    1  │              │ 0.001   1  │
└────────────┘              └────────────┘

Pivoting prevents amplification of rounding errors
```

### 7.3 When to Use Each Method

| Method           | Best For                    | Time Complexity |
| ---------------- | --------------------------- | --------------- |
| Gaussian Elim    | General matrices            | $O(n^3)$        |
| LU Decomposition | Multiple right-hand sides   | $O(n^3)$ once   |
| Cholesky         | Symmetric positive definite | $O(n^3/3)$      |
| Iterative (CG)   | Large sparse systems        | $O(kn^2)$       |
| QR               | Overdetermined systems      | $O(mn^2)$       |

---

## 8. Summary

### Key Concepts

| Concept          | Description                                |
| ---------------- | ------------------------------------------ |
| $Ax = b$         | Matrix form of linear system               |
| Row echelon form | Triangular form after elimination          |
| Rank             | Number of pivots; determines solution type |
| Condition number | Measures numerical stability               |

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

### ML Takeaways

1. **Linear regression** = solving an overdetermined system
2. **Regularization** improves conditioning
3. **Never explicitly invert** matrices for solving systems
4. **Numerical stability** matters in practice

---

## Further Reading

- [Gilbert Strang: Linear Algebra Lecture 2](https://www.youtube.com/watch?v=QVKj3LADCnA)
- [3Blue1Brown: Inverse Matrices](https://www.youtube.com/watch?v=uQhTuRlWMxw)
- [Numerical Linear Algebra (Trefethen)](https://people.maths.ox.ac.uk/trefethen/text.html)

---

## Navigation

← [Previous: Matrix Operations](../02-Matrix-Operations/README.md) | [Next: Determinants →](../04-Determinants/README.md)

[Back to Main](../../README.md)
