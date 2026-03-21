[← Back to Advanced Linear Algebra](../README.md) | [← PCA](../03-Principal-Component-Analysis/notes.md) | [Orthogonality →](../05-Orthogonality-and-Orthonormality/notes.md)

---

# Linear Transformations

## Introduction

Linear transformations are the mathematical foundation of neural networks, computer graphics, signal processing, and countless ML applications. Understanding them reveals why matrices are so central to machine learning - every matrix multiplication is a linear transformation!

A linear transformation is a function between vector spaces that preserves the fundamental operations of vector addition and scalar multiplication. This seemingly simple property has profound consequences: it means any linear transformation can be completely described by a matrix, and the composition of transformations corresponds to matrix multiplication.

```
The Power of Linearity:

1. PREDICTABLE: Output of sum = sum of outputs
2. REPRESENTABLE: Entire transformation stored in one matrix
3. COMPOSABLE: Chain transformations by multiplying matrices
4. INVERTIBLE: (Sometimes) Can reverse the transformation
```

## Prerequisites

- Matrix operations (multiplication, transpose, inverse)
- Vector spaces (span, basis, dimension)
- Eigenvalues and eigenvectors
- Determinants

## Learning Objectives

1. Understand linear transformations geometrically and algebraically
2. Connect transformations to matrix representations
3. Identify and construct common transformation types
4. Compose, invert, and analyze transformations
5. Apply transformation concepts in ML/AI contexts
6. Master homogeneous coordinates for affine transformations

---

## 1. Definition

### Formal Definition

A function $T: V \rightarrow W$ between vector spaces is a **linear transformation** if:

1. **Additivity**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity**: $T(c\mathbf{v}) = cT(\mathbf{v})$

Or equivalently (combining both):
$$T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})$$

### Immediate Consequences

From these two properties, we can derive:

1. **Zero maps to zero**: $T(\mathbf{0}) = \mathbf{0}$
   - Proof: $T(\mathbf{0}) = T(0 \cdot \mathbf{v}) = 0 \cdot T(\mathbf{v}) = \mathbf{0}$

2. **Negatives are preserved**: $T(-\mathbf{v}) = -T(\mathbf{v})$
   - Proof: $T(-\mathbf{v}) = T((-1)\mathbf{v}) = (-1)T(\mathbf{v}) = -T(\mathbf{v})$

3. **General linear combinations**: $T(\sum_{i=1}^n c_i\mathbf{v}_i) = \sum_{i=1}^n c_iT(\mathbf{v}_i)$

### Key Insight: Matrix Representation

Every linear transformation can be represented by a **matrix**:
$$T(\mathbf{x}) = A\mathbf{x}$$

And every matrix defines a linear transformation!

```
Linear Transformation = Matrix Multiplication

T: ℝⁿ → ℝᵐ     ↔     A ∈ ℝᵐˣⁿ

Input x ∈ ℝⁿ   →   Output Ax ∈ ℝᵐ
   (n-dim)              (m-dim)

The columns of A are where the standard basis vectors go:
A = [T(e₁) | T(e₂) | ... | T(eₙ)]
```

### Finding the Matrix

To find the matrix of a transformation:
1. Apply $T$ to each standard basis vector $e_i$
2. The outputs become the columns of $A$

**Example**: Find the matrix for $T(x, y) = (2x + y, x - y)$

$$T(e_1) = T(1, 0) = (2, 1)$$
$$T(e_2) = T(0, 1) = (1, -1)$$

$$A = \begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix}$$

---

## 2. Examples of Linear Transformations

### What IS Linear

| Transformation | Matrix                                                                              | Effect                 |
| -------------- | ----------------------------------------------------------------------------------- | ---------------------- |
| Scaling        | $\begin{bmatrix} k & 0 \\ 0 & k \end{bmatrix}$                                      | Uniform scaling by $k$ |
| Rotation       | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | Rotate by $\theta$     |
| Reflection     | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$                                     | Reflect across x-axis  |
| Shear          | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$                                      | Horizontal shear       |
| Projection     | $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$                                      | Project onto x-axis    |

### Additional Linear Examples

| Transformation | Matrix | Effect |
| --- | --- | --- |
| Identity | $I$ | No change |
| Zero | $\mathbf{0}$ | Everything maps to zero |
| Differentiation | (infinite-dimensional) | $\frac{d}{dx}(f + g) = \frac{df}{dx} + \frac{dg}{dx}$ |
| Integration | (infinite-dimensional) | $\int (f + g) = \int f + \int g$ |
| Matrix trace | $\text{tr}: \mathbb{R}^{n \times n} \to \mathbb{R}$ | $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$ |

### What is NOT Linear

| Transformation                        | Why Not Linear                  |
| ------------------------------------- | ------------------------------- |
| Translation $\mathbf{x} + \mathbf{b}$ | $T(\mathbf{0}) \neq \mathbf{0}$ |
| $T(x) = x^2$                          | $T(2x) = 4x^2 \neq 2T(x)$       |
| $T(x) = \|x\|$                        | $T(-x) \neq -T(x)$              |
| ReLU $\max(0, x)$                     | $\text{ReLU}(-1) \neq -\text{ReLU}(1)$ |
| Sigmoid $\sigma(x)$                   | $\sigma(0) \neq 0$              |
| Polynomial $ax^2 + bx + c$            | Violates both properties        |

### Detailed Counterexamples

**Translation**: $T(\mathbf{x}) = \mathbf{x} + \mathbf{b}$
- $T(\mathbf{0}) = \mathbf{b} \neq \mathbf{0}$ ✗ (unless $\mathbf{b} = \mathbf{0}$)

**Square function**: $T(x) = x^2$
- Homogeneity: $T(cx) = (cx)^2 = c^2x^2 \neq c \cdot x^2 = cT(x)$ ✗

**Absolute value**: $T(x) = |x|$
- Additivity: $T(1 + (-1)) = T(0) = 0$, but $T(1) + T(-1) = 1 + 1 = 2$ ✗

**ReLU**: $T(x) = \max(0, x)$
- Homogeneity: $T(-2 \cdot 1) = T(-2) = 0$, but $-2 \cdot T(1) = -2$ ✗

**Note**: Non-linear functions like ReLU are crucial in neural networks precisely because they add non-linearity!

---

## 3. Geometric Transformations in 2D

### Scaling

$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

- **Uniform scaling** ($s_x = s_y = k$): Scales equally in all directions
- **Non-uniform scaling**: Different scaling along x and y
- **Determinant**: $\det(S) = s_x \cdot s_y$ (area scaling factor)

```
Uniform scaling (k=2):        Non-uniform scaling (sx=2, sy=0.5):

Before:  ■            After: □□□□        Before:  ■     After: ▭▭
         ■                   □□□□                 ■            ▭▭
         ■■                  □□□□□□□□             ■■           ▭▭▭▭
```

### Rotation

$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Properties:
- **Orthogonal**: $R^T R = I$, so $R^{-1} = R^T$
- **Preserves lengths and angles**
- **Determinant**: $\det(R) = 1$ (area preserved)
- **Composition**: $R_{\alpha} R_{\beta} = R_{\alpha + \beta}$

```
Rotation by 90°:

Before:        After:
    │              ──
    │         ──
    └──       │
              │
              
θ = 90°: R = [0  -1]     e₁ = (1,0) → (0,1)
             [1   0]     e₂ = (0,1) → (-1,0)
```

### Reflection

| Axis/Line | Matrix | Determinant |
| --- | --- | --- |
| x-axis | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$ | $-1$ |
| y-axis | $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$ | $-1$ |
| Line $y = x$ | $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$ | $-1$ |
| Line $y = -x$ | $\begin{bmatrix} 0 & -1 \\ -1 & 0 \end{bmatrix}$ | $-1$ |
| Origin | $\begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}$ | $1$ |

Note: Reflections always have $\det = -1$ (flip orientation), except point reflection through origin.

### Reflection Across Arbitrary Line Through Origin

For line at angle $\theta$ to x-axis:
$$M_\theta = \begin{bmatrix} \cos 2\theta & \sin 2\theta \\ \sin 2\theta & -\cos 2\theta \end{bmatrix}$$

### Shear

Horizontal: $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$
Vertical: $\begin{bmatrix} 1 & 0 \\ k & 1 \end{bmatrix}$

Properties:
- **Determinant**: $\det = 1$ (area preserved!)
- **Parallel lines remain parallel**
- **One axis fixed, other axis slants**

```
Before:     After (horizontal shear, k=0.5):
  ■■            ▪▪
  ■■           ▪▪▪
  ■■          ▪▪▪▪
  
y-coordinate unchanged, x shifts by k*y
```

### Projection

Onto x-axis: $P_x = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$

Onto line $y = mx$: $P = \frac{1}{1+m^2}\begin{bmatrix} 1 & m \\ m & m^2 \end{bmatrix}$

General projection onto unit vector $\mathbf{u}$:
$$P = \mathbf{u}\mathbf{u}^T$$

Properties:
- **Idempotent**: $P^2 = P$ (projecting twice = projecting once)
- **Singular**: $\det(P) = 0$ (dimension lost)
- **Symmetric**: $P = P^T$

```
Before:        After (project onto x-axis):
    •
   /           • (all points projected to x-axis)
  •
```

---

## 4. Properties

### Kernel (Null Space)

$$\text{ker}(T) = \{\mathbf{v} : T(\mathbf{v}) = \mathbf{0}\}$$

The set of vectors that map to zero.

**Interpretation**: The kernel captures what information is "lost" by the transformation.

**Examples**:
- Rotation: $\ker(R) = \{\mathbf{0}\}$ (only zero maps to zero)
- Projection onto x-axis: $\ker(P) = \text{y-axis}$ (y-information lost)

### Image (Range)

$$\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$$

The set of all possible outputs.

**Interpretation**: The image is the "reachable" part of the output space.

**Examples**:
- Rotation: $\text{Im}(R) = \mathbb{R}^2$ (can reach anywhere)
- Projection onto x-axis: $\text{Im}(P) = \text{x-axis}$ (confined to x-axis)

### Rank-Nullity Theorem

For $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$:

$$\dim(\text{ker}(T)) + \dim(\text{Im}(T)) = n$$

Or: $\text{nullity} + \text{rank} = \text{domain dimension}$

```
Domain ℝⁿ  ─────T─────>  Image ⊆ ℝᵐ
  │                         │
  │                         │
  └─ Kernel                 └─ dim = rank
     dim = nullity
     
nullity + rank = n (total input dimensions accounted for)
```

**Example**: For a 3×5 matrix with rank 2:
- Nullity = 5 - 2 = 3
- 3D kernel is "collapsed" to zero
- 2D of original 5D space survives

### One-to-One, Onto, Invertibility

| Property | Definition | Equivalent Conditions |
| --- | --- | --- |
| **Injective** (1-1) | $T(u) = T(v) \Rightarrow u = v$ | $\ker(T) = \{\mathbf{0}\}$, nullity = 0 |
| **Surjective** (onto) | $\text{Im}(T) = W$ | rank = dim($W$) |
| **Bijective** | Both 1-1 and onto | Invertible, $\det(A) \neq 0$ |

For square matrices ($m = n$):
- Injective ⟺ Surjective ⟺ Bijective
- Invertible iff $\det(A) \neq 0$

```
Injective (1-1):               Not injective:
   •  →  •                        •  →
   •  →  •                        •  →  • (multiple inputs, same output)
   •  →  •                        •  →
   
Surjective (onto):             Not surjective:
   All outputs reachable          Some outputs unreachable
```

---

## 5. Composition of Transformations

### Matrix Multiplication = Composition

If $T_1(\mathbf{x}) = A\mathbf{x}$ and $T_2(\mathbf{x}) = B\mathbf{x}$:

$$(T_2 \circ T_1)(\mathbf{x}) = T_2(T_1(\mathbf{x})) = B(A\mathbf{x}) = (BA)\mathbf{x}$$

**Important**: Order matters! $BA \neq AB$ in general.

```
Composition order:

x ──T₁──> T₁(x) ──T₂──> T₂(T₁(x))
     A           B

Combined: (T₂ ∘ T₁)(x) = (BA)x

Note: Matrix on LEFT is applied LAST
      BA means "first A, then B"
```

### Example: Rotate then Scale

```
1. Rotate 90°:           2. Then scale by 2:

   R = [0  -1]              S = [2  0]
       [1   0]                  [0  2]

Combined: SR = [2  0][0  -1] = [0  -2]
               [0  2][1   0]   [2   0]
```

### When Does Order Matter?

| Operations | Commute? | Example |
| --- | --- | --- |
| Uniform scaling + Rotation | Yes | $kI \cdot R = R \cdot kI$ |
| Non-uniform scaling + Rotation | No | Different stretching directions |
| Two rotations | Yes | $R_\alpha R_\beta = R_\beta R_\alpha = R_{\alpha+\beta}$ |
| Rotation + Reflection | No | Try it! |
| Two reflections | Generally No | (can give rotation) |
| Shear + Rotation | No | Order matters |

### Properties of Composition

1. **Associativity**: $(AB)C = A(BC)$
2. **Determinant**: $\det(AB) = \det(A) \det(B)$
3. **Inverse**: $(AB)^{-1} = B^{-1}A^{-1}$
4. **Rank**: $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$

### Neural Network Layers

Each layer is a linear transformation (plus nonlinearity):

$$\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)$$

Without $\sigma$: composition of linear = linear (no benefit of depth!)

```
Why Nonlinearity is Essential:

With nonlinearity:          Without nonlinearity:
Layer 1: σ(W₁x + b₁)        Layer 1: W₁x + b₁
Layer 2: σ(W₂h₁ + b₂)       Layer 2: W₂(W₁x + b₁) + b₂
                                    = W₂W₁x + W₂b₁ + b₂
                                    = W'x + b'  (single layer!)

100 linear layers = 1 linear layer
100 nonlinear layers = powerful function approximator
```

---

## 6. Change of Basis

### Same Transformation, Different Matrix

A linear transformation has different matrix representations in different bases.

If $A$ is the matrix in standard basis and $P$ is the change-of-basis matrix (columns are new basis vectors):

$$A' = P^{-1}AP$$

The matrix $A'$ represents the same transformation in the new basis.

```
Standard Basis View:          New Basis View:
   
   Ax = y                     P⁻¹AP(P⁻¹x) = P⁻¹y
                              A'x' = y'
   
Coordinate in       Coordinate in
standard basis      new basis
```

### Why Change Basis?

1. **Simplification**: Some bases make the transformation simpler
2. **Diagonalization**: In eigenvector basis, matrix becomes diagonal
3. **Numerical stability**: Orthonormal bases avoid amplifying errors
4. **Interpretation**: Natural bases reveal structure

### Diagonalization as Change of Basis

$$A = PDP^{-1}$$

where:
- $P$ = matrix whose columns are eigenvectors
- $D$ = diagonal matrix of eigenvalues

In the eigenvector basis (columns of $P$), the transformation is just scaling along each axis!

```
Standard basis:           Eigenvector basis:

   Ax = complex            Dy = simple scaling
   ↗                       ↑
  /\                       |
 /  \                      |
 
Original space:           Eigenvector space:
vectors rotate and        each eigenvector just
stretch together          scales independently
```

### Example

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

Eigenvalues: $\lambda_1 = 5$, $\lambda_2 = 2$

Eigenvectors: $v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$, $v_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$

$$A = \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix} \begin{bmatrix} 5 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 & 1 \\ 1 & -2 \end{bmatrix}^{-1}$$

### Matrix Powers via Diagonalization

$$A^n = PD^nP^{-1}$$

Since $D^n = \text{diag}(\lambda_1^n, \lambda_2^n, \ldots)$, this is easy to compute!

**Application**: Computing $A^{100}$ is easy with diagonalization:
$$A^{100} = P \begin{bmatrix} \lambda_1^{100} & 0 \\ 0 & \lambda_2^{100} \end{bmatrix} P^{-1}$$

---

## 7. Important Classes of Transformations

### Orthogonal Transformations

A matrix $Q$ is **orthogonal** if:
$$Q^T Q = I \quad \text{(equivalently } Q^{-1} = Q^T \text{)}$$

**Properties**:
- Preserves lengths: $\|Qx\| = \|x\|$
- Preserves angles: $\langle Qx, Qy \rangle = \langle x, y \rangle$
- Includes rotations and reflections
- Eigenvalues have $|\lambda| = 1$
- $\det(Q) = \pm 1$ (+1 for rotations, -1 for reflections)

**Examples**:
- Rotation matrices: $R_\theta$
- Reflection matrices
- Permutation matrices
- Householder reflections (used in QR decomposition)

```
Orthogonal transformation:

Before: Unit circle       After: Unit circle (same!)
          ●                         ●
         ╱│╲                       ╱│╲
        ╱ │ ╲                     ╱ │ ╲
       ●──┼──●                   ●──┼──●
        ╲ │ ╱                     ╲ │ ╱
         ╲│╱                       ╲│╱
          ●                         ●
          
Shape unchanged, only rotated/reflected
```

### Symmetric Transformations

A matrix $A$ is **symmetric** if:
$$A = A^T$$

**Properties**:
- All eigenvalues are **real**
- Eigenvectors are **orthogonal** (can choose orthonormal set)
- Always diagonalizable: $A = Q\Lambda Q^T$ with $Q$ orthogonal
- Important for quadratic forms: $x^T A x$

**Examples**:
- Covariance matrices
- Hessian matrices (second derivatives)
- Graph Laplacian matrices
- Distance matrices

### Positive Definite Transformations

A symmetric matrix $A$ is **positive definite** if:
$$\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq \mathbf{0}$$

**Equivalent Conditions**:
- All eigenvalues are positive: $\lambda_i > 0$
- All leading principal minors are positive
- $A = B^T B$ for some invertible $B$ (Cholesky)
- All pivots in Gaussian elimination are positive

**Properties**:
- Always invertible
- Has unique square root: $A^{1/2}$
- Defines an ellipsoid: $\{x : x^T A x = 1\}$

**Examples**:
- Covariance matrices (positive semi-definite)
- Gram matrices $A = X^T X$
- Kernel matrices in machine learning

**Variants**:
| Type | Condition | Eigenvalues |
| --- | --- | --- |
| Positive definite | $x^T A x > 0$ for $x \neq 0$ | All $> 0$ |
| Positive semi-definite | $x^T A x \geq 0$ | All $\geq 0$ |
| Negative definite | $x^T A x < 0$ for $x \neq 0$ | All $< 0$ |
| Indefinite | Neither | Mixed signs |

### Normal Matrices

A matrix $A$ is **normal** if:
$$A^*A = AA^*$$

(where $A^* = \bar{A}^T$ is the conjugate transpose)

Includes: symmetric, orthogonal, unitary, skew-symmetric

**Property**: Normal matrices are unitarily diagonalizable.

### Nilpotent Matrices

A matrix $N$ is **nilpotent** if:
$$N^k = 0 \text{ for some } k$$

**Property**: All eigenvalues are zero.

---

## 8. Applications in ML/AI

### 1. Neural Network Layers

```
Input        Linear Transform    Nonlinearity     Output
x (d)   →      Wx + b       →      σ(·)      →     h (m)
              (d → m)

Each layer: Stretch, rotate, translate, then "bend"
```

**Fully Connected Layer**: $h = \sigma(Wx + b)$
- $W \in \mathbb{R}^{m \times d}$: transforms d-dim input to m-dim space
- Rank of $W$ determines effective dimensions preserved

**Understanding Weight Matrices**:
- High rank: preserves information
- Low rank: compresses (like dimensionality reduction)
- Orthogonal initialization: preserves gradient magnitude

### 2. Attention Mechanism (Transformers)

Query, Key, Value transformations:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Three linear transformations of the input!

```
Input X ──┬── W_Q ──> Q (query)
          │
          ├── W_K ──> K (key)
          │
          └── W_V ──> V (value)

Attention weights = softmax(QK^T / √d_k)
Output = Attention × V
```

**Multi-Head Attention**: Multiple parallel transformations for different "aspects" of the input.

### 3. Convolutional Layers

Convolution is a linear operation! Can be expressed as matrix multiplication:

$$y = Ax$$

where $A$ is a **Toeplitz/circulant matrix** with special structure.

**Properties**:
- Weight sharing: same weights applied everywhere
- Translation equivariance: shift input → shift output
- Sparse: mostly zeros (local connections)

### 4. Batch Normalization

Whitening transformation (decorrelation + scaling):
$$\hat{x} = \frac{x - \mu}{\sigma}$$

Followed by learned affine transformation:
$$y = \gamma \hat{x} + \beta$$

**Effect**: Centers and normalizes activations, then learns optimal scale/shift.

### 5. Word Embeddings

Words mapped to vectors via learned transformation:
$$\text{word index} \rightarrow W_{\text{embed}} \rightarrow \mathbb{R}^d$$

The embedding matrix $W_{\text{embed}} \in \mathbb{R}^{d \times V}$ (V = vocabulary size) is a linear transformation from one-hot to dense representation.

Linear relationships emerge: 
$$\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}$$

### 6. PCA as Linear Transformation

**PCA** projects data onto principal components:
$$Z = XW$$

where $W$ contains the top eigenvectors of the covariance matrix.

This is a **rotation** (orthogonal transformation) to align with directions of maximum variance.

### 7. Data Augmentation

Random transformations (rotation, scaling, shearing) for augmentation:
$$x_{\text{augmented}} = T_{\text{random}}(x)$$

**Common Augmentations** (for images):
- Random rotation: $R_\theta$
- Random scaling: $S$
- Random shearing
- Random horizontal flip

### 8. Graph Neural Networks

Message passing involves linear transformations:
$$h_v^{(k+1)} = \sigma\left(W^{(k)} \sum_{u \in N(v)} h_u^{(k)}\right)$$

The weight matrix $W$ transforms aggregated neighbor features.

---

## 9. Homogeneous Coordinates

### The Problem with Translation

Translation $\mathbf{x} + \mathbf{b}$ is **not** linear in standard coordinates:
- $T(\mathbf{0}) = \mathbf{b} \neq \mathbf{0}$ ✗

This breaks our nice matrix framework!

### Solution: Homogeneous Coordinates

**Idea**: Embed 2D points in 3D by adding coordinate 1.

Standard: $(x, y)$ → Homogeneous: $(x, y, 1)$

Now translation becomes linear matrix multiplication!

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} = \begin{bmatrix} x + t_x \\ y + t_y \\ 1 \end{bmatrix}$$

### Standard Transformation Matrices in Homogeneous Coordinates

| Transformation | Matrix |
| --- | --- |
| **Translation** $(t_x, t_y)$ | $\begin{bmatrix} 1 & 0 & t_x \\ 0 & 1 & t_y \\ 0 & 0 & 1 \end{bmatrix}$ |
| **Scaling** $(s_x, s_y)$ | $\begin{bmatrix} s_x & 0 & 0 \\ 0 & s_y & 0 \\ 0 & 0 & 1 \end{bmatrix}$ |
| **Rotation** $\theta$ | $\begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$ |
| **Shear** $k$ | $\begin{bmatrix} 1 & k & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$ |

### Affine Transformations

General affine transformation = linear + translation:
$$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$$

In homogeneous coordinates:
$$\begin{bmatrix} y \\ 1 \end{bmatrix} = \begin{bmatrix} A & b \\ 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ 1 \end{bmatrix}$$

**Properties of Affine Transformations**:
- Preserve parallelism (parallel lines stay parallel)
- Preserve ratios of distances along lines
- Map lines to lines
- Do NOT preserve angles (except in special cases)

### Transformations Around a Point

To scale/rotate around point $\mathbf{p}$ (not origin):

1. Translate so $\mathbf{p}$ → origin: $T_{-p}$
2. Apply transformation: $M$
3. Translate back: $T_{p}$

Combined: $T_p \cdot M \cdot T_{-p}$

**Example**: Scale by 2 centered at $(3, 3)$:

$$M = \begin{bmatrix} 1 & 0 & 3 \\ 0 & 1 & 3 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 2 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & -3 \\ 0 & 1 & -3 \\ 0 & 0 & 1 \end{bmatrix}$$

### 3D Homogeneous Coordinates

For 3D graphics, use 4D homogeneous coordinates:

$$\begin{bmatrix} x' \\ y' \\ z' \\ 1 \end{bmatrix} = \begin{bmatrix} & & & t_x \\ & R_{3\times3} & & t_y \\ & & & t_z \\ 0 & 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

This is fundamental in:
- Computer graphics (OpenGL, DirectX)
- Robotics (pose transformations)
- Computer vision (camera projections)

### Perspective Projection

Homogeneous coordinates naturally handle perspective:

$$\begin{bmatrix} x' \\ y' \\ w \end{bmatrix} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1/d \end{bmatrix} \begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

Convert back: $(x'/w, y'/w)$ — farther objects appear smaller!

---

## 10. Determinant as Transformation Property

The determinant tells you about the transformation:

| $\det(A)$ | Meaning |
| --- | --- |
| $> 0$ | Preserves orientation, scales area by $\det(A)$ |
| $< 0$ | Flips orientation, scales area by $\|\det(A)\|$ |
| $= 1$ | Area-preserving (rotation, shear) |
| $= -1$ | Area-preserving but flips (reflection) |
| $= 0$ | Collapses dimension (singular, not invertible) |

```
det > 0:              det < 0:              det = 0:
   □ → □               □ → □                 □ → ─
Area scaled         Area scaled           Dimension lost
Orientation kept    Orientation flipped   (projects to lower dim)
```

---

## 11. Summary

### Key Concepts

| Concept               | Definition                   |
| --------------------- | ---------------------------- |
| Linear transformation | $T(au + bv) = aT(u) + bT(v)$ |
| Matrix representation | $T(x) = Ax$                  |
| Composition           | $(T_2 \circ T_1)(x) = (BA)x$ |
| Kernel                | $\{v : T(v) = 0\}$           |
| Image                 | $\{T(v) : v \in V\}$         |
| Invertible            | $\det(A) \neq 0$             |
| Orthogonal            | $Q^TQ = I$, preserves lengths |
| Symmetric             | $A = A^T$, real eigenvalues  |
| Positive definite     | $x^TAx > 0$, all $\lambda > 0$ |

### Common 2D Transformations

| Type        | Matrix                                                                              | $\det$ |
| ----------- | ----------------------------------------------------------------------------------- | --- |
| Scale       | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$                                  | $s_x s_y$ |
| Rotate      | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ | $1$ |
| Reflect (x) | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$                                     | $-1$ |
| Shear       | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$                                      | $1$ |
| Project     | $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$                                      | $0$ |

### Neural Network Insight

```
Deep learning = Composition of transformations

Input → [Linear → Nonlinear] → [Linear → Nonlinear] → ... → Output
         Layer 1                  Layer 2

Without nonlinearities: Just one big linear transformation
With nonlinearities: Can approximate any function!

Each Layer:
1. Wx: Rotate, stretch, project (linear transformation)
2. +b: Translate (shift, via homogeneous coordinates conceptually)
3. σ(·): Bend/warp (nonlinearity)

The power comes from composing many simple transformations!
```

### Quick Reference

```
Is T linear?
├── Check: T(0) = 0?
├── Check: T(u + v) = T(u) + T(v)?
└── Check: T(cv) = cT(v)?

Properties of A:
├── det(A) ≠ 0 → invertible
├── A^T A = I → orthogonal (preserves lengths)
├── A = A^T → symmetric (real eigenvalues)
└── x^T A x > 0 → positive definite

Composition:
├── (T₂ ∘ T₁)(x) = (BA)x  [B is applied last]
├── det(AB) = det(A)det(B)
└── (AB)⁻¹ = B⁻¹A⁻¹
```

---

## Exercises

1. Verify that $T(x, y) = (2x + y, x - y)$ is a linear transformation
2. Find the matrix for rotation by 45°  
3. Compose a rotation by 90° followed by reflection across y-axis
4. Find the kernel and image of projection onto x-axis
5. Show that ReLU is not a linear transformation
6. Find the matrix for reflection across the line $y = 2x$
7. Prove that the composition of two linear transformations is linear
8. Show that $T(x) = x + b$ (translation) is not linear for $b \neq 0$
9. Find eigenvalues and eigenvectors of a 2D rotation matrix
10. Use homogeneous coordinates to rotate around point $(1, 1)$

---

## Related Topics

- [Eigenvalues and Eigenvectors](../01-Eigenvalues-and-Eigenvectors/README.md)
- [Singular Value Decomposition](../02-Singular-Value-Decomposition/README.md)
- [Principal Component Analysis](../03-Principal-Component-Analysis/README.md)
- [Matrix Decompositions](../08-Matrix-Decompositions/README.md)

---

## References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Axler, S. - "Linear Algebra Done Right"
3. Goodfellow et al. - "Deep Learning" (Chapter 2)
4. 3Blue1Brown - "Essence of Linear Algebra" (YouTube series)
5. Boyd & Vandenberghe - "Introduction to Applied Linear Algebra"

---

[← Back to Advanced Linear Algebra](../README.md) | [← PCA](../03-Principal-Component-Analysis/notes.md) | [Orthogonality →](../05-Orthogonality-and-Orthonormality/notes.md)
