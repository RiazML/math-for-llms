# Matrix Rank

## Overview

The **rank** of a matrix is one of the most fundamental concepts in linear algebra, serving as a critical measure of the "information content" or "effective dimensionality" of a matrix. Rank reveals how much independent information a matrix contains, connecting matrix structure to linear independence, transformations, and solution spaces.

In machine learning and AI, understanding rank is essential for:
- **Dimensionality reduction**: PCA relies on identifying low-rank structure
- **Feature engineering**: Detecting redundant features via rank analysis
- **Matrix completion**: Netflix-style recommender systems exploit low-rank assumptions
- **Model compression**: Low-rank approximations reduce neural network parameters
- **Numerical stability**: Rank-deficient matrices cause computational issues

## Learning Objectives

By the end of this section, you will be able to:

- **Define and interpret** matrix rank in multiple equivalent ways
- **Compute rank** using row reduction, SVD, and determinant methods
- **Apply the rank-nullity theorem** to connect rank with null space
- **Analyze linear systems** using rank conditions for solvability
- **Understand geometric meaning** of rank as transformation dimensionality
- **Handle numerical rank** issues in floating-point computation
- **Apply rank concepts** in PCA, matrix factorization, and neural networks

## Prerequisites

Before studying matrix rank, ensure familiarity with:

- **Matrix operations**: Multiplication, transpose, inversion
- **Vector spaces**: Linear independence, span, basis concepts
- **Linear systems**: Gaussian elimination, solution characterization
- **Determinants**: Computing and interpreting determinants
- **SVD basics**: Singular value decomposition (for advanced methods)

---

## 1. What is Matrix Rank?

### 1.1 Definition

The **rank** of a matrix $A \in \mathbb{R}^{m \times n}$ is the maximum number of linearly independent rows (or equivalently, columns).

$$\text{rank}(A) = \dim(\text{column space of } A) = \dim(\text{row space of } A)$$

This fundamental equality—that row rank equals column rank—is one of the most important results in linear algebra.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         WHAT RANK TELLS US                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  "How many dimensions does the output of this transformation span?"      │
│                                                                          │
│  Full rank (rank = min(m,n)):                                            │
│    → Matrix uses all available dimensions                                │
│    → No redundancy in rows/columns                                       │
│    → Maximum information content                                         │
│                                                                          │
│  Rank deficient (rank < min(m,n)):                                       │
│    → Some dimensions are "wasted" or redundant                           │
│    → Linear dependencies exist among rows/columns                        │
│    → Transformation collapses some dimensions                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Equivalent Definitions

The rank of a matrix can be characterized in many equivalent ways:

| Definition | Description |
|------------|-------------|
| **Row rank** | Maximum number of linearly independent rows |
| **Column rank** | Maximum number of linearly independent columns |
| **Pivot count** | Number of pivots in row echelon form |
| **SVD count** | Number of non-zero singular values |
| **Image dimension** | Dimension of the range/image of $A$ as a linear map |
| **Largest submatrix** | Size of largest invertible square submatrix |

### 1.3 Illustrative Examples

**Example 1: Full Rank Matrix**
```
A = [1  0]    Rank = 2 (full rank for 2×2)
    [0  1]

Columns: [1, 0] and [0, 1] are linearly independent
Rows: [1, 0] and [0, 1] are linearly independent
→ Both columns span all of R²
```

**Example 2: Rank-Deficient Matrix**
```
B = [1  2]    Rank = 1
    [2  4]

Row 2 = 2 × Row 1 → only 1 independent row
Column 2 = 2 × Column 1 → only 1 independent column
→ Columns span only a line through origin in R²
```

**Example 3: Rectangular Matrix**
```
C = [1  0  2]    3×3 matrix
    [0  1  1]
    [1  1  3]

Row 3 = Row 1 + Row 2 → Rank = 2
Although 3×3, only 2 independent rows/columns
```

**Example 4: Wide vs Tall Matrices**
```
Wide (2×4):                    Tall (4×2):
D = [1  0  0  0]               E = [1  0]
    [0  1  0  0]                   [0  1]
                                   [0  0]
max possible rank = 2              [0  0]
actual rank = 2 (full row rank)
                               max possible rank = 2
                               actual rank = 2 (full column rank)
```

### 1.4 Key Properties

| Property | Formula | Notes |
|----------|---------|-------|
| **Equality** | row rank = column rank | Fundamental theorem |
| **Upper bound** | $\text{rank}(A) \leq \min(m, n)$ | Limited by smallest dimension |
| **Zero matrix** | $\text{rank}(\mathbf{0}) = 0$ | No independent vectors |
| **Identity** | $\text{rank}(I_n) = n$ | Full rank |
| **Product bound** | $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ | Multiplication reduces rank |
| **Sum bound** | $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$ | Addition increases bound |
| **Transpose** | $\text{rank}(A) = \text{rank}(A^T)$ | Transpose preserves rank |

---

## 2. Computing Rank

### 2.1 Row Echelon Form Method

The most intuitive method: reduce to row echelon form and count pivots.

**Algorithm:**
1. Apply Gaussian elimination to get row echelon form
2. Count non-zero rows (equivalently, count pivot positions)
3. This count equals the rank

```
Example: Find rank of A

A = [1   2   3]        Row operations        [1   2   3]
    [2   4   6]    ─────────────────────►    [0   0   0]
    [1   1   1]                              [0  -1  -2]

Further reduction:
    [1   2   3]
    [0  -1  -2]    ← Pivot in column 2
    [0   0   0]    ← Zero row

Pivots: positions (1,1) and (2,2)
rank(A) = 2
```

**Advantages:**
- Intuitive and algorithmic
- Works for any matrix
- Reveals pivot columns (basis for column space)

**Disadvantages:**
- Numerically unstable without pivoting
- May accumulate rounding errors
- Doesn't provide singular values

### 2.2 Singular Value Decomposition (SVD) Method

The most robust numerical method: count non-zero singular values.

$$A = U\Sigma V^T$$

$$\text{rank}(A) = \#\{i : \sigma_i > 0\}$$

```
Singular value spectrum:

σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0 = σᵣ₊₁ = ... = σₙ
└──────────────────────┘   └───────────────┘
    r = rank(A)              zero singular values
```

**Example:**
```python
import numpy as np

A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9]])

U, S, Vt = np.linalg.svd(A)
print(f"Singular values: {S}")
# Output: [11.22... , 0, 0]  → rank = 1
```

**Advantages:**
- Numerically stable
- Provides additional information (singular values)
- Natural handling of near-rank-deficiency
- Optimal for low-rank approximation

**Disadvantages:**
- More computationally expensive: $O(\min(m,n) \cdot m \cdot n)$
- Requires tolerance for numerical rank

### 2.3 Determinant Method (Square Matrices)

For square matrices, check if determinant is non-zero.

$$\text{rank}(A) = n \iff \det(A) \neq 0$$

For general rank, find the largest $k$ such that some $k \times k$ submatrix has non-zero determinant.

```
Example: A = [1  2  3]
             [2  4  6]
             [0  1  2]

3×3 determinant:
det(A) = 1(4·2 - 6·1) - 2(2·2 - 6·0) + 3(2·1 - 4·0)
       = 1(8-6) - 2(4-0) + 3(2-0)
       = 2 - 8 + 6 = 0

So rank < 3. Check 2×2 submatrices:
det([1 2; 0 1]) = 1 ≠ 0

Therefore rank = 2
```

**Practical use:** Good for small matrices; impractical for finding exact rank of large matrices.

### 2.4 Comparison of Methods

| Method | Complexity | Stability | Best For |
|--------|------------|-----------|----------|
| Row Echelon | $O(mn^2)$ | Moderate | Teaching, small matrices |
| SVD | $O(mn \min(m,n))$ | High | Production, numerical rank |
| Determinant | $O(n!)$ or $O(n³)$ | Low | Theoretical analysis |
| QR Decomposition | $O(mn^2)$ | High | Column rank, least squares |

---

## 3. Rank and Linear Systems

### 3.1 The Rank-Nullity Theorem

**Theorem:** For any matrix $A \in \mathbb{R}^{m \times n}$:

$$\text{rank}(A) + \text{nullity}(A) = n$$

where **nullity** = $\dim(\ker(A))$ = dimension of the null space.

```
┌────────────────────────────────────────────────────────────────────────┐
│                         n = number of columns                           │
│                                                                         │
│    ┌────────────────────────┬──────────────────────────────────┐       │
│    │                        │                                   │       │
│    │      rank(A)           │          nullity(A)               │       │
│    │   (pivot columns)      │      (free variables)             │       │
│    │                        │                                   │       │
│    │   Basis for            │       Basis for                   │       │
│    │   column space         │       null space                  │       │
│    │                        │                                   │       │
│    └────────────────────────┴──────────────────────────────────┘       │
│                                                                         │
│    rank + nullity = n (always!)                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**Interpretation:**
- **Pivot columns** correspond to the rank (independent columns)
- **Free columns** correspond to the nullity (null space dimensions)
- Every column is either a pivot column or a free column

**Example:**
```
A = [1  2  0  1]    4 columns
    [0  0  1  2]
    [0  0  0  0]

Pivot columns: 1, 3 → rank = 2
Free columns: 2, 4 → nullity = 2
Check: 2 + 2 = 4 ✓
```

### 3.2 Solvability Conditions

For the linear system $Ax = b$:

| Condition | Interpretation | Result |
|-----------|----------------|--------|
| $\text{rank}(A) = \text{rank}([A \mid b])$ | $b$ is in column space | Solution exists |
| $\text{rank}(A) < \text{rank}([A \mid b])$ | $b$ not in column space | No solution |
| $\text{rank}(A) = n$ | Full column rank | At most one solution |
| $\text{rank}(A) < n$ | Rank deficient | Infinitely many (if any) |

**Solution Classification:**

```
                        ┌─────────────────────────────────────┐
                        │         Does solution exist?         │
                        │   rank(A) = rank([A|b]) ?           │
                        └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                   YES                              NO
                    │                               │
            ┌───────┴───────┐               ┌───────┴───────┐
            │ How many?     │               │ No solution   │
            │ rank(A) = n?  │               │ (inconsistent)│
            └───────────────┘               └───────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
       YES                      NO
        │                       │
 ┌──────┴──────┐         ┌──────┴──────┐
 │ Exactly one │         │ Infinitely  │
 │  solution   │         │    many     │
 └─────────────┘         └─────────────┘
```

### 3.3 Full Rank Cases

**Full Column Rank** ($\text{rank}(A) = n$, where $A$ is $m \times n$):
- All columns are linearly independent
- $Ax = b$ has **at most one** solution
- Null space is trivial: $\ker(A) = \{0\}$
- $A^TA$ is invertible (important for least squares)
- Unique least squares solution: $x = (A^TA)^{-1}A^Tb$

**Full Row Rank** ($\text{rank}(A) = m$):
- All rows are linearly independent
- $Ax = b$ **always has** at least one solution (for any $b$)
- Transformation is surjective (onto)
- $AA^T$ is invertible
- Minimum norm solution via pseudoinverse

**Full Rank Square Matrix** ($\text{rank}(A) = n = m$):
- Matrix is **invertible**
- $\det(A) \neq 0$
- Unique solution $x = A^{-1}b$ for any $b$
- All eigenvalues are non-zero

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      RANK AND MATRIX PROPERTIES                          │
├────────────────────┬────────────────────┬───────────────────────────────┤
│   Matrix Type      │   Rank Condition   │   Key Properties              │
├────────────────────┼────────────────────┼───────────────────────────────┤
│ Tall matrix (m>n)  │ rank = n (full col)│ At most 1 solution, A^T A inv │
│ Wide matrix (m<n)  │ rank = m (full row)│ Always solvable, infinitely   │
│ Square (m=n)       │ rank = n (full)    │ Invertible, unique solution   │
│ Any               │ rank < min(m,n)    │ Rank deficient, singularity   │
└────────────────────┴────────────────────┴───────────────────────────────┘
```

---

## 4. Geometric Interpretation

### 4.1 Column Space Dimension

The rank equals the dimension of the **image** (range) of the linear transformation $T_A: \mathbb{R}^n \to \mathbb{R}^m$ defined by $T_A(x) = Ax$.

```
Linear Transformation Perspective:

         n-dimensional          Linear Map A           m-dimensional
           domain              ═══════════════►           codomain
              │                                              │
              │                                              │
              ▼                                              ▼
         ┌────────┐                                    ┌────────────┐
         │        │           Transformation           │            │
         │   Rⁿ   │  ────────────────────────────►    │     Rᵐ     │
         │        │              A·x                   │            │
         └────────┘                                    └────────────┘
                                                              │
                                                              ▼
                                                       ┌────────────┐
                                                       │   Image    │
                                                       │ dim = rank │
                                                       └────────────┘

rank(A) = dimension of the image in Rᵐ
```

### 4.2 Visualizing Rank Deficiency

**Full Rank Transformation (preserves dimension):**
```
2D Input → 2D Output (rank = 2)

     ●───────●                      ●───────●
    /|       |                     /|       |
   ● │       ●        ────►       ● │       ●
   │ ●───────●                    │ ●───────●
   │/                             │/
   ●                              ●

  Square → Square (no collapse)
```

**Rank-1 Transformation (projects to line):**
```
2D Input → 1D Output (rank = 1)

     ●───────●                        ●
    /|       |                       /
   ● │       ●        ────►         ●─────────────●
   │ ●───────●                               
   │/                               
   ●                                

  Square → Line (dimension collapsed)
```

**Rank-2 Transformation in 3D (projects to plane):**
```
3D Input → 2D Output (rank = 2)

      ●─────────●                    
     /│        /│                       ●───────●
    ●─────────● │      ────►           /|       |
    │ ●───────│─●                     ● │       ●
    │/        │/                      │ ●───────●
    ●─────────●                       │/
                                      ●
  Cube → Flat square (one dimension lost)
```

### 4.3 Null Space Interpretation

The null space $\ker(A)$ represents the directions that get "mapped to zero."

$$\dim(\ker(A)) = n - \text{rank}(A) = \text{nullity}(A)$$

```
              Input Space (Rⁿ)
         ┌──────────────────────┐
         │                      │
         │   ┌──────────────┐   │
         │   │  Null Space  │   │   Vectors here
         │   │   ker(A)     │   │   map to zero
         │   └──────────────┘   │
         │                      │
         │   Remaining (n - nullity)   │   These map onto
         │   dimensions         │       the image
         │                      │
         └──────────────────────┘
```

---

## 5. Machine Learning Applications

### 5.1 Feature Redundancy Detection

In machine learning, a data matrix $X$ (n samples × d features) with $\text{rank}(X) < d$ indicates **redundant features**:

```
Data Matrix X (n samples × d features):

┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  rank(X) = d  (full column rank)                                         │
│    → All features are independent                                        │
│    → No feature can be written as linear combination of others           │
│    → XᵀX is invertible (good for OLS regression)                         │
│                                                                          │
│  rank(X) < d  (column rank deficient)                                    │
│    → Some features are linearly dependent                                │
│    → Multicollinearity problem                                           │
│    → XᵀX is singular (OLS fails without regularization)                  │
│    → Solutions: Remove features, use PCA, add regularization             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Practical Example:**
```python
import numpy as np

# Feature matrix with redundancy
X = np.array([
    [1, 2, 3],    # Feature 3 = Feature 1 + Feature 2
    [2, 3, 5],
    [3, 4, 7],
    [4, 5, 9]
])

print(f"Shape: {X.shape}")         # (4, 3)
print(f"Rank: {np.linalg.matrix_rank(X)}")  # 2 (not 3!)

# XᵀX will be singular
XtX = X.T @ X
print(f"det(XᵀX): {np.linalg.det(XtX):.2e}")  # ~0
```

### 5.2 Principal Component Analysis (PCA)

PCA exploits the **effective rank** of data to perform dimensionality reduction.

**Key insight:** Real-world data often has rank much lower than the number of features.

$$\text{effective rank} \approx \frac{\left(\sum_i \sigma_i\right)^2}{\sum_i \sigma_i^2}$$

```
Singular Value Spectrum Analysis:

             High Effective Rank          Low Effective Rank
             (data uses all dims)         (data is low-dimensional)

Variance:    ████████████████             ████████████████
             ███████████████              ███
             ██████████████               █
             █████████████                ▪
             ████████████                 ▪

Implication: Need many components         Few components capture most info
```

**Selecting Number of Components:**
```
Cumulative explained variance:

100% ─┬─────────────────●●●●●●●
      │              ●●●
90%  ─┤           ●●
      │         ●●
80%  ─┤       ●
      │     ●
70%  ─┤   ●                            Choose k where
      │  ●                              cumulative variance
60%  ─┤ ●                               reaches target
      │●                                (e.g., 95%)
50%  ─┼─●──┬──┬──┬──┬──┬──┬──┬──►
      │ 1  2  3  4  5  6  7  8  k
           └──┘
            └─ k = 3 captures ~90%
```

### 5.3 Low-Rank Matrix Factorization

Many ML applications exploit low-rank structure:

**Recommender Systems (Collaborative Filtering):**
```
User-Item Rating Matrix R:

                Items (movies)
           ┌─────────────────────────┐
           │ 5  ?  3  ?  ?  1  ?  4  │
  Users    │ ?  4  ?  5  ?  ?  3  ?  │
           │ 4  ?  ?  ?  2  ?  ?  5  │
           │ ?  ?  4  ?  ?  3  ?  ?  │
           └─────────────────────────┘

Assumption: R has low rank (k << min(users, items))
            because users have similar taste patterns

Factorization: R ≈ U V^T

           Users × k          k × Items
          ┌─────────┐        ┌─────────────┐
   Users  │         │    k   │             │  Items
          │    U    │   ──►  │     Vᵀ      │
          │         │        │             │
          └─────────┘        └─────────────┘

Benefits:
- Fill in missing values (predict unknown ratings)
- Reduce storage: k(m + n) instead of m × n
- Discover latent factors (genres, styles)
```

**Image Compression via SVD:**
```
Original Image A (m × n pixels):

A = U Σ Vᵀ = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + ... + σᵣuᵣvᵣᵀ

Approximation with k << r terms:
A_k = σ₁u₁v₁ᵀ + ... + σₖuₖvₖᵀ

Storage comparison:
- Original: m × n values
- Compressed: k × (m + n + 1) values

Compression ratio = mn / (k(m + n + 1))

For 1000×1000 image with k=50:
Ratio = 1,000,000 / (50 × 2001) ≈ 10× compression
```

### 5.4 Neural Network Compression

Large weight matrices can be approximated with low-rank factorizations:

```
Original Layer:
y = Wx + b    where W is m × n

Low-Rank Approximation:
W ≈ UV^T      where U is m × k, V is n × k

Factored Layer:
y = U(V^T x) + b

┌─────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER COMPARISON                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Original:  mn parameters                                                │
│  Factored:  k(m + n) parameters                                          │
│                                                                          │
│  If k << min(m,n): significant compression!                              │
│                                                                          │
│  Example: m=1000, n=500, k=50                                            │
│    Original: 500,000 parameters                                          │
│    Factored: 50(1000+500) = 75,000 parameters                            │
│    Compression: 6.67×                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Applications:**
- Mobile deployment of large models
- Reducing memory bandwidth
- Faster inference
- LoRA (Low-Rank Adaptation) for fine-tuning large language models

### 5.5 Regularization and Rank

**Ridge Regression handles rank deficiency:**
```
Problem: (XᵀX) singular when rank(X) < d

OLS Solution (fails):
β = (XᵀX)⁻¹ Xᵀy    ← Can't invert singular matrix

Ridge Solution (works):
β = (XᵀX + λI)⁻¹ Xᵀy    ← Always invertible for λ > 0

The regularization term λI makes the matrix full rank
```

**Nuclear Norm Regularization (encourages low rank):**
$$\min_M \|A - M\|_F^2 + \lambda \|M\|_*$$

where $\|M\|_* = \sum_i \sigma_i$ is the nuclear norm (sum of singular values).

---

## 6. Numerical Rank

### 6.1 The Numerical Challenge

In floating-point arithmetic, the concept of "exact rank" breaks down:

```
Theoretical Matrix:              Numerical Computation:

σ₁ = 10                         σ₁ = 9.9999999999
σ₂ = 5                          σ₂ = 5.0000000001
σ₃ = 0     ← "true" rank = 2    σ₃ = 2.3 × 10⁻¹⁵   ← Is this zero?
σ₄ = 0                          σ₄ = 1.1 × 10⁻¹⁶

Question: Is σ₃ = 2.3 × 10⁻¹⁵ genuinely zero (roundoff error)
          or a small but real singular value?
```

**Sources of numerical rank issues:**
1. **Roundoff errors** in matrix entries
2. **Computation errors** during decomposition
3. **Physical measurements** with noise
4. **Model approximations** in real applications

### 6.2 Tolerance-Based Rank

NumPy's `matrix_rank` uses a tolerance:

$$\text{numerical rank} = \#\{i : \sigma_i > \text{tol}\}$$

**Default tolerance:**
$$\text{tol} = \sigma_{\max} \cdot \max(m, n) \cdot \epsilon$$

where $\epsilon \approx 2.2 \times 10^{-16}$ is machine epsilon.

```python
import numpy as np

# Nearly rank-2 matrix
A = np.array([[1, 2, 3],
              [2, 4, 6],
              [3, 6, 9.0001]])  # Slight perturbation

print(f"Default rank: {np.linalg.matrix_rank(A)}")  # Likely 3

# With explicit tolerance
print(f"Looser tolerance: {np.linalg.matrix_rank(A, tol=0.001)}")  # Likely 2
```

### 6.3 Condition Number Connection

The **condition number** measures sensitivity to perturbations:

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_r}$$

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONDITION NUMBER INTERPRETATION                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  κ ≈ 1:           Well-conditioned, clear rank                          │
│                   Numerical rank = true rank with high confidence        │
│                                                                          │
│  κ ≈ 10³ - 10⁶:   Moderately ill-conditioned                            │
│                   Rank determination may be sensitive to tolerance       │
│                                                                          │
│  κ ≈ 10¹⁰+:       Severely ill-conditioned                              │
│                   Numerically singular, rank uncertain                   │
│                                                                          │
│  κ = ∞:           Exactly singular (rank < min(m,n))                    │
│                   Some σᵢ = 0 exactly                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.4 Best Practices for Numerical Rank

1. **Always use SVD** for numerical rank computation
2. **Examine singular value spectrum** before deciding rank
3. **Use problem-appropriate tolerance** based on data precision
4. **Consider effective rank** for dimensionality reduction
5. **Document tolerance choices** in analysis

```python
def robust_rank_analysis(A):
    """Analyze matrix rank robustly."""
    U, S, Vt = np.linalg.svd(A)
    
    # Various rank measures
    ranks = {}
    for tol in [1e-10, 1e-8, 1e-6, 1e-4]:
        ranks[tol] = np.sum(S > tol * S[0])
    
    # Effective rank
    total = np.sum(S)
    eff_rank = (total ** 2) / np.sum(S ** 2)
    
    return S, ranks, eff_rank
```

---

## 7. Special Rank Results

### 7.1 Rank of Matrix Product

$$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$

**Proof sketch:** Range of $AB$ is contained in range of $A$, and has dimension at most rank$(B)$.

**When is equality achieved?**
- When $\text{rank}(A) = \text{rank}(AB)$ (columns of $B$ don't "waste" any of $A$'s rank)
- When matrices are "compatible" in a rank-preserving way

**Example of strict inequality:**
```
A = [1  0]     rank(A) = 1
    [0  0]

B = [0  1]     rank(B) = 1
    [0  1]

AB = [0  1]    rank(AB) = 1 = min(1, 1) ✓
     [0  0]

But:
C = [1  0]     rank(C) = 1
    [0  0]

D = [0  0]     rank(D) = 1
    [1  0]

CD = [0  0]    rank(CD) = 0 < min(1, 1) ✗
     [0  0]
```

### 7.2 Sylvester's Rank Inequality

$$\text{rank}(A) + \text{rank}(B) - n \leq \text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$

where $A$ is $m \times n$ and $B$ is $n \times p$.

### 7.3 Rank of Sum

$$|\text{rank}(A) - \text{rank}(B)| \leq \text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$$

**Upper bound:** Union of column spaces has dimension at most sum of dimensions.

**Lower bound:** Rank can't decrease by more than the rank of what's being subtracted.

### 7.4 Rank of Transpose

$$\text{rank}(A) = \text{rank}(A^T)$$

**Proof:** Row rank = Column rank, and transposing swaps rows and columns.

### 7.5 Rank of Outer Product

For vectors $u \in \mathbb{R}^m$ and $v \in \mathbb{R}^n$ (both non-zero):

$$\text{rank}(uv^T) = 1$$

**Proof:** All columns of $uv^T$ are scalar multiples of $u$.

**Application:** Any rank-$r$ matrix can be written as sum of $r$ rank-1 matrices:
$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

### 7.6 Gram Matrix Rank

$$\text{rank}(A^T A) = \text{rank}(A A^T) = \text{rank}(A)$$

**Proof:** $A^T A x = 0 \iff Ax = 0$ (since $\|Ax\|^2 = x^T A^T A x$).

**Implication:** Normal equations $(A^T A)x = A^T b$ have full rank iff $A$ has full column rank.

---

## 8. Proofs and Theoretical Foundations

### 8.1 Row Rank Equals Column Rank

**Theorem:** For any matrix $A$, row rank = column rank.

**Proof:**
Let $r$ = column rank of $A$. Then the column space has a basis of $r$ vectors.
Express each column of $A$ as a linear combination of these basis vectors.
This gives a factorization $A = CR$ where:
- $C$ is $m \times r$ (the basis vectors)
- $R$ is $r \times n$ (the coefficients)

Row space of $A$ = Row space of $R$ (since $C$ has full column rank).
Row rank of $R \leq r$ (since $R$ has only $r$ rows).
Therefore row rank of $A \leq r$ = column rank.

Applying the same argument to $A^T$ gives column rank $\leq$ row rank.
Thus they are equal. ∎

### 8.2 Rank-Nullity Theorem Proof

**Theorem:** For $A \in \mathbb{R}^{m \times n}$: $\text{rank}(A) + \text{nullity}(A) = n$.

**Proof:**
Let $r = \text{rank}(A)$ and let $\{v_1, \ldots, v_{n-r}\}$ be a basis for $\ker(A)$.
Extend to a basis $\{v_1, \ldots, v_{n-r}, w_1, \ldots, w_r\}$ of $\mathbb{R}^n$.

Claim: $\{Aw_1, \ldots, Aw_r\}$ is a basis for the range of $A$.

Linear independence: If $\sum c_i Aw_i = 0$, then $\sum c_i w_i \in \ker(A)$.
So $\sum c_i w_i = \sum d_j v_j$ for some $d_j$.
By basis independence, all $c_i = 0$.

Spanning: Any $Ax$ equals $A(\sum a_j v_j + \sum b_i w_i) = \sum b_i Aw_i$.

Therefore dim(range) = $r$, confirming $r + (n-r) = n$. ∎

---

## 9. Computational Examples

### 9.1 Python Implementation

```python
import numpy as np
from scipy import linalg

def analyze_rank(A, name="Matrix"):
    """Comprehensive rank analysis."""
    m, n = A.shape
    
    # SVD-based analysis
    U, S, Vt = np.linalg.svd(A)
    
    # Numerical rank at various tolerances
    ranks = {tol: np.sum(S > tol * S[0]) for tol in [1e-10, 1e-6, 1e-2]}
    
    # Null space dimension
    nullity = n - ranks[1e-10]
    
    # Condition number
    cond = S[0] / S[-1] if S[-1] > 0 else np.inf
    
    print(f"\n{name} ({m}×{n}):")
    print(f"  Singular values: {S}")
    print(f"  Numerical rank: {ranks}")
    print(f"  Nullity: {nullity}")
    print(f"  Condition number: {cond:.2e}")
    print(f"  Full column rank: {ranks[1e-10] == n}")
    print(f"  Full row rank: {ranks[1e-10] == m}")
    
    return S, ranks, nullity

# Examples
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.eye(3)
C = np.random.randn(5, 3) @ np.random.randn(3, 4)  # Rank ≤ 3

analyze_rank(A, "Rank-2 matrix")
analyze_rank(B, "Identity")
analyze_rank(C, "Low-rank random")
```

### 9.2 Applications

```python
# Feature redundancy check
def check_feature_redundancy(X):
    """Check if feature matrix has redundant features."""
    n_samples, n_features = X.shape
    rank = np.linalg.matrix_rank(X)
    
    if rank < n_features:
        print(f"Warning: Features are linearly dependent!")
        print(f"  Matrix rank: {rank}")
        print(f"  Number of features: {n_features}")
        print(f"  Redundant dimensions: {n_features - rank}")
        return True
    return False

# Low-rank approximation
def low_rank_approx(A, k):
    """Compute best rank-k approximation."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    A_k = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.linalg.norm(A - A_k, 'fro') / np.linalg.norm(A, 'fro')
    return A_k, error
```

---

## 10. Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| **Definition** | $\text{rank}(A) = \dim(\text{col space}) = \dim(\text{row space})$ |
| **Rank-Nullity** | $\text{rank}(A) + \text{nullity}(A) = n$ |
| **Upper bound** | $\text{rank}(A) \leq \min(m, n)$ |
| **Product** | $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ |
| **Transpose** | $\text{rank}(A) = \text{rank}(A^T)$ |
| **Gram matrix** | $\text{rank}(A^T A) = \text{rank}(A)$ |
| **Outer product** | $\text{rank}(uv^T) = 1$ |

### Interpretation Guide

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 WHAT RANK TELLS YOU                                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  rank(A) = r means:                                                      │
│                                                                          │
│  1. Column space of A is r-dimensional                                   │
│  2. Row space of A is r-dimensional                                      │
│  3. A has exactly r linearly independent columns (and rows)              │
│  4. A has exactly r non-zero singular values                             │
│  5. Image of transformation A is r-dimensional subspace                  │
│  6. Null space has dimension (n - r)                                     │
│  7. Ax = b has solutions only if b ∈ r-dim subspace                      │
│  8. A = sum of r rank-1 matrices (minimal such decomposition)            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Machine Learning Connections

| Concept | ML Application |
|---------|----------------|
| **Full rank** | Feature independence, invertible $X^TX$ |
| **Low rank** | Dimensionality reduction, matrix factorization |
| **Rank deficiency** | Multicollinearity, need for regularization |
| **Effective rank** | Number of significant principal components |
| **Nuclear norm** | Regularizer encouraging low-rank solutions |

### Quick Reference

```
System Ax = b:
┌────────────────────┬─────────────────────┬─────────────────────────┐
│ Condition          │ rank(A) vs rank[A|b]│ Solutions               │
├────────────────────┼─────────────────────┼─────────────────────────┤
│ rank(A) = rank[A|b]│ Equal               │ Exists                  │
│ rank(A) < rank[A|b]│ Unequal             │ None                    │
│ rank(A) = n        │ Full column rank    │ Unique (if exists)      │
│ rank(A) < n        │ Rank deficient      │ Infinitely many         │
└────────────────────┴─────────────────────┴─────────────────────────┘
```

---

## 11. Practice Problems

See the accompanying Jupyter notebooks:
- **[examples.ipynb](examples.ipynb)**: Worked examples with visualizations
- **[exercises.ipynb](exercises.ipynb)**: Practice problems with solutions

Key exercises include:
1. Computing rank via multiple methods
2. Verifying rank-nullity theorem
3. Analyzing linear system solvability
4. Low-rank matrix approximation
5. PCA and effective rank
6. Feature redundancy detection

---

## 12. Further Reading

### Textbooks
- Strang, G. "Linear Algebra and Learning from Data" - Chapters on rank and SVD
- Trefethen, L. & Bau, D. "Numerical Linear Algebra" - Numerical rank computation

### Online Resources
- [3Blue1Brown: Rank](https://www.youtube.com/watch?v=uQhTuRlWMxw)
- [Gilbert Strang: Column Space and Nullspace](https://www.youtube.com/watch?v=8o5Cmfpeo6g)
- [Low-Rank Matrix Approximation (Wikipedia)](https://en.wikipedia.org/wiki/Low-rank_approximation)

### Papers
- Eckart-Young Theorem: Optimal low-rank approximation via SVD
- Nuclear norm minimization for matrix completion

---

## Navigation

[← Previous: Determinants](../04-Determinants/README.md) | [Next: Vector Spaces and Subspaces →](../06-Vector-Spaces-Subspaces/README.md)

[↑ Back to Linear Algebra Basics](../README.md) | [↑↑ Back to Main](../../README.md)
