# Singular Value Decomposition (SVD)

## Introduction

The Singular Value Decomposition is one of the most powerful and widely-used matrix factorizations in mathematics and machine learning. Unlike eigendecomposition (which only works for square matrices), SVD works for **any** matrix and reveals its fundamental structure.

**Why SVD is Essential for ML:**
- Works for any rectangular matrix (not just square)
- Always exists and is numerically stable
- Reveals intrinsic dimensionality of data
- Foundation for PCA, matrix completion, and compression
- Key to understanding neural network weight matrices

**Key Insight**: SVD answers the question "What is the best low-dimensional approximation to my data?" - a fundamental question in machine learning.

## Prerequisites

- Eigenvalues and eigenvectors
- Matrix multiplication and transpose
- Orthogonal matrices and their properties
- Vector norms and inner products
- Basic understanding of linear transformations

## Learning Objectives

1. Understand the SVD factorization geometrically and algebraically
2. Compute SVD components from scratch and using NumPy
3. Apply low-rank approximation (Eckart-Young theorem)
4. Use SVD for dimensionality reduction and data compression
5. Implement SVD-based ML applications (PCA, recommender systems, LSA)
6. Analyze matrix properties via singular values (rank, condition number, norms)

---

## 1. Definition and Fundamental Theorem

### The SVD Decomposition

For any matrix $A \in \mathbb{R}^{m \times n}$, there exists a factorization:

$$A = U \Sigma V^T$$

where:

| Component | Dimensions | Properties | Meaning |
|-----------|------------|------------|---------|
| $U$ | $m \times m$ | Orthogonal ($U^TU = I$) | Left singular vectors |
| $\Sigma$ | $m \times n$ | Diagonal, non-negative | Singular values |
| $V$ | $n \times n$ | Orthogonal ($V^TV = I$) | Right singular vectors |

### Visual Structure

```
SVD Structure for m > n:

A         =    U      ×     Σ      ×    Vᵀ
(m×n)       (m×m)       (m×n)       (n×n)

[       ]   [| | | |]   [σ₁ 0  0 ]   [--- v₁ᵀ ---]
[   A   ] = [u₁u₂u₃u₄]  [0  σ₂ 0 ]   [--- v₂ᵀ ---]
[       ]   [| | | |]   [0  0  σ₃]   [--- v₃ᵀ ---]
[       ]   [| | | |]   [0  0  0 ]

SVD Structure for m < n:

A         =    U      ×     Σ         ×    Vᵀ
(m×n)       (m×m)       (m×n)           (n×n)

[       ]   [| | |]   [σ₁ 0  0  0]   [--- v₁ᵀ ---]
[   A   ] = [u₁u₂u₃]  [0  σ₂ 0  0]   [--- v₂ᵀ ---]
                      [0  0  σ₃ 0]   [--- v₃ᵀ ---]
                                     [--- v₄ᵀ ---]
```

### Fundamental Properties

**Singular Value Ordering:**
$$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > \sigma_{r+1} = \cdots = 0$$

where $r = \text{rank}(A)$

**Orthonormality:**
- Columns of $U$ are orthonormal: $u_i \cdot u_j = \delta_{ij}$
- Columns of $V$ are orthonormal: $v_i \cdot v_j = \delta_{ij}$

**Key Relationships:**
$$Av_i = \sigma_i u_i \quad \text{(for } i = 1, \ldots, r)$$
$$A^T u_i = \sigma_i v_i \quad \text{(for } i = 1, \ldots, r)$$

**Existence Theorem**: SVD exists for ANY matrix (real or complex, any dimensions). This universality is what makes SVD so powerful.

---

## 2. Geometric Interpretation

### SVD as Three Transformations

The SVD reveals that **any linear transformation is a composition of:**

1. **Rotation/Reflection** (by $V^T$): Align input with principal directions
2. **Scaling** (by $\Sigma$): Stretch/compress along each axis
3. **Rotation/Reflection** (by $U$): Rotate to output space

```
Step-by-Step Transformation:

    Original      →    Rotate Vᵀ    →    Scale Σ     →    Rotate U
    unit circle        (to axes)        (to ellipse)      (to final)

       ●                   ●               ●                 ●
      /|\                 /|\             /                 /|
     / | \      Vᵀ       / | \     Σ     /       U        / |
    ●──●──●    →→→     ●──●──●   →→→   ●·····●  →→→    ●··●·●
                                        stretched        rotated
```

### Finding Principal Directions

The SVD finds the **optimal orthonormal bases** for both input and output spaces:

**Input space ($V$):**
- $v_1, v_2, \ldots, v_n$ are the principal directions in the domain
- These are the directions that get mapped to orthogonal directions in the codomain
- $v_1$ is the direction that gets stretched the most (by $\sigma_1$)

**Output space ($U$):**
- $u_1, u_2, \ldots, u_m$ are the principal directions in the codomain
- $u_i = A v_i / \sigma_i$ (for non-zero singular values)

### Understanding Singular Values Geometrically

Singular values $\sigma_i$ are the **semi-axes of the transformation ellipsoid**:

```
Unit sphere → Ellipsoid under transformation A

         Input: Unit Sphere              Output: Ellipsoid
              
              ○                              ⬭
           ╱     ╲                       ╱       ╲
         ╱    r=1  ╲      A × ●        ╱   σ₁      ╲
        ○───────────○    ─────→      ⬭─────────────⬭
         ╲         ╱                   ╲    σ₂    ╱
           ╲     ╱                       ╲       ╱
              ○                              ⬭

σ₁ = maximum stretch factor = max_∥x∥=1 ∥Ax∥
σₘᵢₙ = minimum stretch factor = min_∥x∥=1 ∥Ax∥
```

**Key Insight**: $\sigma_1 = \max_{\|x\|=1} \|Ax\|$ is the spectral norm - the maximum "amplification" of any input vector.

### The Four Fundamental Subspaces

SVD naturally reveals the four fundamental subspaces of $A$:

| Subspace | Definition | From SVD |
|----------|------------|----------|
| Column space $\mathcal{C}(A)$ | Image of $A$ | $\text{span}(u_1, \ldots, u_r)$ |
| Null space $\mathcal{N}(A)$ | Kernel of $A$ | $\text{span}(v_{r+1}, \ldots, v_n)$ |
| Row space $\mathcal{C}(A^T)$ | Image of $A^T$ | $\text{span}(v_1, \ldots, v_r)$ |
| Left null space $\mathcal{N}(A^T)$ | Kernel of $A^T$ | $\text{span}(u_{r+1}, \ldots, u_m)$ |

---

## 3. Computing SVD

### Relationship to Eigendecomposition

The SVD components are intimately connected to eigenvalue problems:

**Right Singular Vectors ($V$)**: Eigenvectors of $A^T A$
$$A^T A = V \Sigma^T \Sigma V^T = V \cdot \text{diag}(\sigma_1^2, \ldots, \sigma_n^2) \cdot V^T$$

**Left Singular Vectors ($U$)**: Eigenvectors of $A A^T$
$$A A^T = U \Sigma \Sigma^T U^T = U \cdot \text{diag}(\sigma_1^2, \ldots, \sigma_m^2) \cdot U^T$$

**Singular Values**: Square roots of eigenvalues
$$\sigma_i = \sqrt{\lambda_i(A^T A)} = \sqrt{\lambda_i(A A^T)}$$

### Step-by-Step Computation

```
Algorithm: Computing SVD from Eigendecomposition

1. Form A^T A (or A A^T if m < n for efficiency)

2. Compute eigendecomposition: A^T A = V D V^T
   - λ₁, λ₂, ..., λₙ are eigenvalues (sorted descending)
   - v₁, v₂, ..., vₙ are eigenvectors (columns of V)

3. Singular values: σᵢ = √λᵢ

4. Left singular vectors: uᵢ = (1/σᵢ) A vᵢ  (for σᵢ > 0)

Example:
A = [3  1]     A^T A = [10  6]
    [1  3]             [6  10]

eigenvalues: λ = 16, 4  →  σ = 4, 2
eigenvectors: v₁ = [1/√2, 1/√2]^T,  v₂ = [1/√2, -1/√2]^T

u₁ = A v₁ / σ₁ = [3+1, 1+3]^T / (4√2) = [1/√2, 1/√2]^T  ✓
```

### Numerical Methods

In practice, SVD is NOT computed via eigendecomposition (numerically unstable):

**1. Golub-Kahan Bidiagonalization**
- Reduce $A$ to bidiagonal form using Householder reflections
- More numerically stable than forming $A^T A$

**2. Divide and Conquer**
- For large matrices, divide into smaller subproblems
- Used by LAPACK's `dgesdd`

**3. Randomized SVD**
- For very large matrices where only top-$k$ singular values needed
- Random projection to reduce dimension first

### NumPy Implementation

```python
import numpy as np

A = np.array([[3, 1], [1, 3]])

# Full SVD (default)
U, s, Vt = np.linalg.svd(A, full_matrices=True)

# Thin/Economy SVD (for rectangular matrices)
U, s, Vt = np.linalg.svd(A, full_matrices=False)

# Note: NumPy returns s as 1D array of singular values
# To reconstruct A, need to form Σ matrix:
S = np.diag(s)
A_reconstructed = U @ S @ Vt
```

---

## 4. SVD Variants: Full, Thin, and Truncated

### Full SVD

The complete factorization with all singular vectors:

$$A_{m \times n} = U_{m \times m} \cdot \Sigma_{m \times n} \cdot V_{n \times n}^T$$

```
Full SVD (m = 4, n = 3, rank = 2):

U (4×4)           Σ (4×3)           Vᵀ (3×3)
[u₁ u₂ | u₃ u₄]   [σ₁ 0  0 ]       [--- v₁ᵀ ---]
                  [0  σ₂ 0 ]       [--- v₂ᵀ ---]
                  [0  0  0 ]       [--- v₃ᵀ ---]
                  [0  0  0 ]

  ↑  ↑    ↑  ↑      ↑  ↑  ↑          ↑    ↑    ↑
 C(A)   N(Aᵀ)     signal null      C(Aᵀ) null
```

### Thin (Economy) SVD

Omit zero rows/columns of $\Sigma$ and corresponding vectors:

$$A_{m \times n} = U_{m \times k} \cdot \Sigma_{k \times k} \cdot V_{n \times k}^T$$

where $k = \min(m, n)$

```
Thin SVD (m = 4, n = 3):

U (4×3)           Σ (3×3)           Vᵀ (3×3)
[u₁ u₂ u₃]        [σ₁ 0  0 ]       [--- v₁ᵀ ---]
                  [0  σ₂ 0 ]       [--- v₂ᵀ ---]
                  [0  0  σ₃]       [--- v₃ᵀ ---]

Storage: m×k + k + k×n vs m×m + m×n + n×n (full)
For 1000×100: 100,100 + 100 + 100×100 vs 1,000,000 + 100,000 + 10,000
```

### Compact SVD

Keep only non-zero singular values (rank $r$):

$$A_{m \times n} = U_{m \times r} \cdot \Sigma_{r \times r} \cdot V_{n \times r}^T$$

```
Compact SVD (m = 4, n = 3, rank = 2):

U (4×2)           Σ (2×2)           Vᵀ (2×3)
[u₁ u₂]           [σ₁ 0 ]          [--- v₁ᵀ ---]
                  [0  σ₂]          [--- v₂ᵀ ---]

This exactly reconstructs A with minimum storage.
```

### Truncated SVD

Keep only top $k$ singular values (where $k < r$):

$$A_k = U_k \Sigma_k V_k^T = \sum_{i=1}^{k} \sigma_i u_i v_i^T$$

```
Truncated SVD (rank-k approximation):

Original (rank r)  →  Truncated (rank k < r)

Storage comparison:
- Original A: m × n values
- Truncated: k(m + n + 1) values

Example: 1000×1000 matrix, k = 50
- Original: 1,000,000 values
- Truncated: 50 × 2001 = 100,050 values (90% compression!)
```

### When to Use Each Form

| Form | When to Use | Storage |
|------|-------------|---------|
| Full | Need null space vectors | $O(m^2 + mn + n^2)$ |
| Thin | General computation | $O(mn\min(m,n))$ |
| Compact | Exact reconstruction | $O(r(m+n+1))$ |
| Truncated | Approximation/compression | $O(k(m+n+1))$ |

---

## 5. Low-Rank Approximation

### Outer Product Form

SVD expresses any matrix as a sum of rank-1 matrices:

$$A = \sum_{i=1}^{r} \sigma_i u_i v_i^T$$

Each term $\sigma_i u_i v_i^T$ is a **rank-1 layer** with decreasing importance:

```
A = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + σ₃u₃v₃ᵀ + ...
    \_____/   \_____/   \_____/
    rank-1    rank-1    rank-1
    (most       ↓         ↓
    important)  ↓         ↓
                ↓     (increasingly
              less      small)
             important
```

### Visualization of Layered Decomposition

```
Original Matrix = Layer 1 + Layer 2 + Layer 3 + ...

[9  12  15]   [6  8  10]   [3  4  5]   [0  0  0]
[12 16  20] = [8  10.7 13.3] + [4  5.3  6.7] + [0  0  0]
[15 20  25]   [10 13.3 16.7]   [5  6.7  8.3]   [0  0  0]

 100% energy    σ₁² = 90%     σ₂² = 10%     σ₃² = 0%

Keeping just Layer 1 gives 90% of the "information"!
```

### The Eckart-Young-Mirsky Theorem

**Theorem**: The best rank-$k$ approximation to $A$ (in Frobenius or spectral norm) is:

$$A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T = U_k \Sigma_k V_k^T$$

**Error bounds:**

| Norm | Error Formula | Meaning |
|------|---------------|---------|
| Frobenius | $\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$ | Total residual energy |
| Spectral | $\|A - A_k\|_2 = \sigma_{k+1}$ | Maximum distortion |

**Optimality**: No other rank-$k$ matrix achieves a smaller error!

### Energy/Variance Interpretation

Think of $\sigma_i^2$ as "energy" captured by component $i$:

$$\text{Energy fraction in top-}k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2}$$

```
Example: Selecting k components

Singular values: σ = [10, 5, 2, 1, 0.5]
Squared:        σ² = [100, 25, 4, 1, 0.25]
Total energy = 130.25

Cumulative energy:
k=1: 100/130.25 = 76.8%
k=2: 125/130.25 = 95.9%
k=3: 129/130.25 = 99.0%

→ k=2 captures 96% of information with 60% fewer parameters
```

### Choosing the Right Rank k

**Methods for selecting $k$:**

1. **Elbow Method**: Plot singular values, look for "elbow" where decline levels off

2. **Variance Explained**: Choose $k$ such that cumulative variance > threshold (e.g., 90%)

3. **Gap Method**: Look for big gap between $\sigma_k$ and $\sigma_{k+1}$

4. **Noise Level**: Keep components with $\sigma_i$ above noise floor

```
Scree Plot (singular values):

σᵢ |
   |  ●
   |     ●
   |        ●  ← elbow: signal above, noise below
   |           ● ● ● ● ● ● ●
   +----------------------------- i
        1  2  3  4  5  6  7  8

Choose k = 3 (first 3 components are signal)
```

---

## 6. Matrix Properties via SVD

### Matrix Norms from Singular Values

SVD provides elegant formulas for the most important matrix norms:

| Norm | Formula | Geometric Meaning |
|------|---------|-------------------|
| Spectral (2-norm) | $\|A\|_2 = \sigma_1$ | Maximum stretch factor |
| Frobenius | $\|A\|_F = \sqrt{\sum_i \sigma_i^2}$ | Total "energy" or size |
| Nuclear (trace) | $\|A\|_* = \sum_i \sigma_i$ | Sum of all stretch factors |

**Why These Matter in ML:**
- **Spectral norm**: Controls Lipschitz constant of neural network layers
- **Frobenius norm**: Used in weight decay regularization
- **Nuclear norm**: Convex relaxation for rank minimization (matrix completion)

### Condition Number

The **condition number** measures sensitivity to perturbations:

$$\kappa(A) = \frac{\sigma_1}{\sigma_r} = \frac{\text{largest singular value}}{\text{smallest non-zero singular value}}$$

**Interpretation:**

| Condition Number | Matrix Quality | Impact |
|------------------|----------------|--------|
| $\kappa \approx 1$ | Well-conditioned | Numerically stable |
| $\kappa \sim 10^3$ | Moderately ill-conditioned | Some precision loss |
| $\kappa > 10^{10}$ | Severely ill-conditioned | Results unreliable |

```
Effect on Linear System Ax = b:

Well-conditioned (κ ≈ 1):          Ill-conditioned (κ ≫ 1):

      b                                    b
       ↘                                    ↘
   ●→→→→●  small change in b          ●→→→→→→→→●  small Δb
       ↗   = small change in x                 ↗   = LARGE Δx
      x                                       x

Relative error amplification: up to κ times!
∥Δx∥/∥x∥ ≤ κ(A) · ∥Δb∥/∥b∥
```

**ML Implications:**
- Poor conditioning in Hessian → slow optimization
- Gradient descent step size limited by largest eigenvalue
- Batch normalization improves conditioning

### Rank from SVD

The **rank** equals the number of non-zero singular values:

$$\text{rank}(A) = |\{i : \sigma_i > 0\}|$$

**Numerical rank** (accounting for floating-point error):
$$\text{rank}_\epsilon(A) = |\{i : \sigma_i > \epsilon \cdot \sigma_1\}|$$

Typical threshold: $\epsilon \approx 10^{-10}$ to $10^{-14}$

### The Moore-Penrose Pseudoinverse

For any matrix $A$, the pseudoinverse $A^+$ is uniquely defined:

$$A^+ = V \Sigma^+ U^T$$

where $\Sigma^+$ inverts non-zero diagonal entries:

$$(\Sigma^+)_{ii} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > \epsilon \\ 0 & \text{otherwise} \end{cases}$$

**Four Defining Properties (Moore-Penrose conditions):**
1. $A A^+ A = A$
2. $A^+ A A^+ = A^+$
3. $(A A^+)^T = A A^+$
4. $(A^+ A)^T = A^+ A$

**Applications:**

| Problem | Solution via Pseudoinverse |
|---------|---------------------------|
| Least squares $\min \|Ax - b\|$ | $x = A^+ b$ |
| Minimum norm $\min \|x\|$ s.t. $Ax = b$ | $x = A^+ b$ |
| General inverse | Works for any matrix |

```python
# NumPy pseudoinverse
import numpy as np
A_pinv = np.linalg.pinv(A)

# Solving least squares
x = A_pinv @ b  # Equivalent to np.linalg.lstsq(A, b)
```

---

## 7. SVD vs Eigendecomposition

### Comparison Table

| Property | Eigendecomposition | SVD |
|----------|-------------------|-----|
| Applies to | Square matrices only | **Any matrix** (m × n) |
| Existence | May not exist | **Always exists** |
| Factors | $A = PDP^{-1}$ | $A = U\Sigma V^T$ |
| Values | May be complex | **Always real, non-negative** |
| Vectors | May not be orthogonal | **Always orthogonal** |
| Uniqueness | Eigenvectors not unique | Singular vectors unique (up to sign) |

### When to Use Each

**Use Eigendecomposition when:**
- Matrix is square and you need actual eigenvalues (e.g., stability analysis)
- Matrix is symmetric/Hermitian (then eigen = SVD)
- Studying dynamical systems $x_{t+1} = Ax_t$

**Use SVD when:**
- Matrix is rectangular
- Need low-rank approximation
- Need orthonormal bases for column/row spaces
- Numerical stability is important
- Computing pseudoinverse or solving least squares

### Special Case: Symmetric Matrices

For symmetric $A = A^T$:

$$\text{Eigendecomposition: } A = Q \Lambda Q^T$$
$$\text{SVD: } A = U \Sigma V^T$$

**Relationship:**
- $U = V$ (or $U = V$ with possible sign flips)
- $|\lambda_i| = \sigma_i$ (singular values are absolute eigenvalues)
- If $A$ is positive semi-definite: $\lambda_i = \sigma_i$

```
Symmetric case:

A = QΛQᵀ = Q |Λ| sign(Λ) Qᵀ
          ↓   ↓      ↓    ↓
          U   Σ      ·    Vᵀ

Where sign(Λ) is diagonal matrix of ±1
```

---

## 8. Applications in Machine Learning

SVD is foundational to many ML techniques. Here we explore the major applications.

### 8.1 Principal Component Analysis (PCA)

**The Connection:**
For centered data matrix $X$ (n samples × d features):

$$X = U \Sigma V^T$$

- **Principal components**: Columns of $V$ (right singular vectors)
- **Scores/projections**: $U \Sigma$ (data in PC coordinates)
- **Variance explained**: $\frac{\sigma_i^2}{n-1}$ is variance along $i$-th PC

```
PCA via SVD:

Data X         →    SVD        →    Interpretation
(n×d)               U Σ Vᵀ

[sample 1]      [  |  ][ ]      - Columns of V = principal directions
[sample 2]   =  [  |  ][ ] Vᵀ   - Σ² / (n-1) = variances
[   ...  ]      [  |  ][σ]      - X Vₖ = projection to k-dim
[sample n]      [  U  ][Σ]

Dimensionality reduction: X_reduced = X @ V[:, :k]
```

**Why SVD for PCA?**
1. Numerically stable (don't form $X^TX$ explicitly)
2. Directly gives orthonormal PCs
3. Works for n > d or n < d

### 8.2 Image Compression

Store image as low-rank approximation:

```
Original Image (m×n)    →    Truncated SVD (rank k)

Storage: m × n values        Storage: k(m + n + 1) values

Example: 1000 × 1000 image
Original: 1,000,000 values
k=50: 50 × 2001 = 100,050 values (90% compression!)
k=100: 100 × 2001 = 200,100 values (80% compression)

Quality degrades gracefully as k decreases.
```

**The Algorithm:**
```python
def compress_image(image, k):
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
```

### 8.3 Recommender Systems (Collaborative Filtering)

**Matrix Factorization for Recommendations:**

```
User-Item Rating Matrix (with missing values):

        Items →
Users  [5  ?  3  ?  4]
  ↓    [?  4  ?  2  ?]
       [3  ?  5  ?  3]
       [?  3  ?  4  ?]

SVD reveals latent factors:
R ≈ U Σ Vᵀ = (User preferences) × (Item characteristics)

Each user → k-dimensional vector of preferences
Each item → k-dimensional vector of characteristics
Predicted rating = dot product
```

**Handling Missing Values:**
- Can't apply SVD directly to incomplete matrices
- Solutions: Alternating Least Squares (ALS), gradient descent
- Netflix Prize: SVD-based methods were key ingredients

### 8.4 Latent Semantic Analysis (LSA)

**Document-Term Matrix Analysis:**

```
Term-Document Matrix A:           Truncated SVD (k topics):

        docs →                    A ≈ Uₖ Σₖ Vₖᵀ
terms  [2  0  1  0]
  ↓    [1  3  0  0]               Uₖ: term-topic relationships
       [0  0  2  1]               Vₖ: document-topic relationships
       [0  1  1  2]               Σₖ: topic strengths

Similar documents → close in topic space
Similar terms → close in topic space (captures synonymy)
```

**Applications:**
- Document similarity (information retrieval)
- Topic modeling (before LDA became popular)
- Word embeddings (precursor to Word2Vec)

### 8.5 Noise Reduction / Denoising

**Principle**: Signal lies in low-rank subspace; noise fills full space.

```
Noisy data = True signal + Noise

Singular values:
σ₁, σ₂, σ₃     (large - signal)
σ₄, σ₅, ...    (small - noise)

Denoised = Σᵢ₌₁ᵏ σᵢ uᵢ vᵢᵀ  (keep only top k components)
```

**Works when:**
- True signal is approximately low-rank
- Noise is "spread out" (not aligned with signal)

### 8.6 Solving Least Squares Problems

For overdetermined system $Ax \approx b$:

$$x^* = A^+ b = V \Sigma^+ U^T b$$

**Why SVD-based solution is better:**

| Method | Formula | Issue |
|--------|---------|-------|
| Normal equations | $x = (A^TA)^{-1}A^Tb$ | Squares condition number |
| QR decomposition | $x = R^{-1}Q^Tb$ | Good for full-rank A |
| SVD | $x = V\Sigma^+U^Tb$ | Handles rank deficiency |

**SVD advantage**: Works for any matrix, handles rank deficiency gracefully.

### 8.7 Weight Matrices in Neural Networks

**Low-Rank Factorization:**

Replace $W_{m×n}$ with $U_{m×k} V_{k×n}^T$:

- Reduces parameters: $mn → k(m+n)$
- Regularizes model (implicit rank constraint)
- Used in LoRA (Low-Rank Adaptation) for fine-tuning LLMs

```
Original:        Low-Rank:
W: m×n           W ≈ U Vᵀ
params: mn       params: k(m+n)

Example: m=n=1000, k=10
Original: 1,000,000 params
Low-rank: 20,000 params (50× reduction!)
```

### 8.8 Data Whitening / Decorrelation

Transform data to have identity covariance:

$$X_{\text{white}} = U \Sigma^{-1} U^T X = X (\Sigma V^T)^{-1}$$

- Removes correlations between features
- Normalizes variance in all directions
- Preprocessing for some ML algorithms

---

## 9. Numerical Computation

### Standard Algorithms

**1. Golub-Kahan Bidiagonalization**
- Transform $A$ to bidiagonal form $B$ using Householder reflections
- $A = Q_1 B Q_2^T$ where $Q_1, Q_2$ are orthogonal
- More stable than forming $A^TA$ explicitly

**2. QR Iteration on Bidiagonal Matrix**
- Apply implicit QR steps to compute singular values
- Converges quadratically to diagonal form

```
SVD Algorithm Pipeline:

A (m×n)  →  Bidiagonalization  →  B  →  QR iteration  →  Σ
             (Householder)                              + U, V
```

### Computational Complexity

| Operation | Complexity | Typical Use |
|-----------|------------|-------------|
| Full SVD | $O(\min(mn^2, m^2n))$ | Small/medium matrices |
| Thin SVD | $O(\min(mn^2, m^2n))$ | General purpose |
| Truncated SVD (top k) | $O(mnk)$ | Large matrices, few components |
| Randomized SVD | $O(mn \log k + k^2(m+n))$ | Very large matrices |

### Randomized SVD

For very large matrices where only top-$k$ components needed:

```
Randomized SVD Algorithm:

1. Generate random matrix Ω (n × (k+p)), p = oversampling
2. Compute Y = A Ω (captures range of A)
3. Orthonormalize: Q = orth(Y) via QR
4. Form B = Qᵀ A (small matrix)
5. Compute SVD of B: B = Ũ Σ Ṽᵀ
6. Recover: U = Q Ũ

Complexity: O(mn log k) vs O(mn k) for deterministic
```

**When to use:**
- Matrix too large to fit in memory
- Only need top $k \ll \min(m,n)$ components
- Can tolerate small approximation error

### Numerical Stability Considerations

**Good Practices:**
1. Never form $A^TA$ explicitly (squares condition number)
2. Use thin SVD when full $U$ not needed
3. Set threshold for "zero" singular values: $\sigma < \epsilon \cdot \sigma_1$
4. Standard double precision: $\epsilon \approx 10^{-15}$

```python
# Stable rank computation
def numerical_rank(A, tol=1e-10):
    _, s, _ = np.linalg.svd(A)
    return np.sum(s > tol * s[0])

# Stable pseudoinverse
def stable_pinv(A, rcond=1e-10):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_inv = np.where(s > rcond * s[0], 1/s, 0)
    return Vt.T @ np.diag(s_inv) @ U.T
```

---

## 10. Summary and Quick Reference

### Key Formulas

| Concept | Formula | NumPy |
|---------|---------|-------|
| SVD decomposition | $A = U\Sigma V^T$ | `np.linalg.svd(A)` |
| Singular values | $\sigma_i = \sqrt{\lambda_i(A^TA)}$ | `s` from SVD |
| Low-rank approx | $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ | `U[:,:k] @ diag(s[:k]) @ Vt[:k,:]` |
| Pseudoinverse | $A^+ = V\Sigma^+U^T$ | `np.linalg.pinv(A)` |
| Spectral norm | $\|A\|_2 = \sigma_1$ | `np.linalg.norm(A, 2)` |
| Frobenius norm | $\|A\|_F = \sqrt{\sum \sigma_i^2}$ | `np.linalg.norm(A, 'fro')` |
| Nuclear norm | $\|A\|_* = \sum \sigma_i$ | `np.linalg.norm(A, 'nuc')` |
| Rank | # non-zero $\sigma_i$ | `np.linalg.matrix_rank(A)` |
| Condition number | $\kappa = \sigma_1/\sigma_r$ | `np.linalg.cond(A)` |

### Algorithm Summary

```
Computing SVD:
1. Compute eigendecomposition of A^T A (or use direct algorithms)
2. σᵢ = √λᵢ (singular values, sorted descending)
3. vᵢ = eigenvectors of A^T A (right singular vectors)
4. uᵢ = Avᵢ/σᵢ (left singular vectors)

Low-rank Approximation:
1. Compute SVD: A = U Σ Vᵀ
2. Keep top k components: Aₖ = Uₖ Σₖ Vₖᵀ
3. Error = √(σ²_{k+1} + σ²_{k+2} + ...)
```

### Application Cheat Sheet

| Application | Key Idea | SVD Role |
|-------------|----------|----------|
| PCA | Find max-variance directions | PCs = right singular vectors |
| Image compression | Low-rank approximation | Keep top k terms |
| Recommender systems | Matrix factorization | User/item latent factors |
| LSA | Topic discovery | Term/doc topic embeddings |
| Noise reduction | Filter low-energy components | Truncate small singular values |
| Least squares | Minimum-norm solution | Pseudoinverse via SVD |

### Common Pitfalls

1. **Forming $A^TA$**: Squares condition number, loses precision
2. **Ignoring numerical rank**: Zero singular values may be small but non-zero
3. **Wrong matrix orientation**: SVD of $X$ vs $X^T$ for PCA depends on convention
4. **Full vs thin SVD**: Know when you need full orthogonal bases

---

## Exercises

### Basic
1. Compute the SVD of $A = \begin{bmatrix} 3 & 0 \\ 0 & 2 \\ 0 & 0 \end{bmatrix}$ by hand
2. Find the singular values of $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ from $A^TA$
3. Compute the best rank-1 approximation of a 3×3 matrix

### Intermediate
4. Implement pseudoinverse using SVD and verify the four Moore-Penrose conditions
5. Compute the condition number and explain its effect on solving $Ax = b$
6. Calculate spectral, Frobenius, and nuclear norms from singular values

### Advanced
7. Implement PCA via SVD and compute variance explained by each component
8. Create a simple image compression demo using truncated SVD
9. Build a basic LSA system for document similarity
10. Implement noise reduction by singular value thresholding

See `exercises.ipynb` for solutions and detailed implementations.

---

## References

**Textbooks:**
1. Strang, G. - "Linear Algebra and Its Applications" (Chapter on SVD)
2. Trefethen & Bau - "Numerical Linear Algebra" (SVD and applications)
3. Golub & Van Loan - "Matrix Computations" (Algorithms and stability)

**Papers:**
4. Halko, Martinsson, Tropp - "Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions" (2011)
5. Eckart & Young - "The Approximation of One Matrix by Another of Lower Rank" (1936)

**Online Resources:**
6. NumPy documentation: `numpy.linalg.svd`
7. scikit-learn: `TruncatedSVD` for large sparse matrices
