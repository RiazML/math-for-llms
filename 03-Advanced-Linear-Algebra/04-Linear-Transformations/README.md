# Linear Transformations

## Introduction

Linear transformations are the mathematical foundation of neural networks, computer graphics, signal processing, and countless ML applications. Understanding them reveals why matrices are so central to machine learning - every matrix multiplication is a linear transformation!

## Prerequisites

- Matrix operations
- Vector spaces
- Eigenvalues and eigenvectors
- Basis and dimension

## Learning Objectives

1. Understand linear transformations geometrically
2. Connect transformations to matrices
3. Identify common transformation types
4. Compose and invert transformations
5. Apply transformations in ML contexts

---

## 1. Definition

### Formal Definition

A function $T: V \rightarrow W$ between vector spaces is a **linear transformation** if:

1. **Additivity**: $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. **Homogeneity**: $T(c\mathbf{v}) = cT(\mathbf{v})$

Or equivalently (combining both):
$$T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})$$

### Key Insight

Every linear transformation can be represented by a **matrix**:
$$T(\mathbf{x}) = A\mathbf{x}$$

And every matrix defines a linear transformation!

```
Linear Transformation = Matrix Multiplication

T: ℝⁿ → ℝᵐ     ↔     A ∈ ℝᵐˣⁿ

Input x ∈ ℝⁿ   →   Output Ax ∈ ℝᵐ
   (n-dim)              (m-dim)
```

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

### What is NOT Linear

| Transformation                        | Why Not Linear                  |
| ------------------------------------- | ------------------------------- | --- | ------------------ |
| Translation $\mathbf{x} + \mathbf{b}$ | $T(\mathbf{0}) \neq \mathbf{0}$ |
| $T(x) = x^2$                          | $T(2x) = 4x^2 \neq 2T(x)$       |
| $T(x) =                               | x                               | $   | $T(-x) \neq -T(x)$ |
| ReLU $\max(0, x)$                     | Piecewise, not smooth at 0      |

**Note**: Non-linear functions like ReLU are crucial in neural networks precisely because they add non-linearity!

---

## 3. Geometric Transformations in 2D

### Scaling

$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

```
Uniform scaling (k=2):        Non-uniform scaling (sx=2, sy=0.5):

Before:  ■            After: □□□□        Before:  ■     After: ▭▭
         ■                   □□□□                 ■            ▭▭
         ■■                  □□□□□□□□             ■■           ▭▭▭▭
```

### Rotation

$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

```
Rotation by 90°:

Before:        After:
    │              ──
    │         ──
    └──       │
              │
```

### Reflection

Across x-axis: $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$

Across y-axis: $\begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$

Across line $y = x$: $\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$

### Shear

Horizontal: $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$

```
Before:     After (k=0.5):
  ■■            ▪▪
  ■■           ▪▪▪
  ■■          ▪▪▪▪
```

### Projection

Onto x-axis: $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$

```
Before:        After:
    •
   /           • (all points projected to x-axis)
  •
```

---

## 4. Properties

### Kernel (Null Space)

$$\text{ker}(T) = \{\mathbf{v} : T(\mathbf{v}) = \mathbf{0}\}$$

The set of vectors that map to zero.

### Image (Range)

$$\text{Im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$$

The set of all possible outputs.

### Rank-Nullity Theorem

For $T: \mathbb{R}^n \rightarrow \mathbb{R}^m$:

$$\dim(\text{ker}(T)) + \dim(\text{Im}(T)) = n$$

Or: $\text{nullity} + \text{rank} = \text{domain dimension}$

### Invertibility

$T$ is invertible if and only if:

- $\text{ker}(T) = \{\mathbf{0}\}$ (injective/one-to-one)
- $\text{Im}(T) = W$ (surjective/onto)
- $\det(A) \neq 0$ (for square matrices)

---

## 5. Composition of Transformations

### Matrix Multiplication = Composition

If $T_1(\mathbf{x}) = A\mathbf{x}$ and $T_2(\mathbf{x}) = B\mathbf{x}$:

$$(T_2 \circ T_1)(\mathbf{x}) = T_2(T_1(\mathbf{x})) = B(A\mathbf{x}) = (BA)\mathbf{x}$$

**Important**: Order matters! $BA \neq AB$ in general.

### Example: Rotate then Scale

```
1. Rotate 90°:           2. Then scale by 2:

   R = [0  -1]              S = [2  0]
       [1   0]                  [0  2]

Combined: SR = [2  0][0  -1] = [0  -2]
               [0  2][1   0]   [2   0]
```

### Neural Network Layers

Each layer is a linear transformation (plus nonlinearity):

$$\mathbf{h}_1 = \sigma(W_1 \mathbf{x} + \mathbf{b}_1)$$
$$\mathbf{h}_2 = \sigma(W_2 \mathbf{h}_1 + \mathbf{b}_2)$$

Without $\sigma$: composition of linear = linear (no benefit of depth!)

---

## 6. Change of Basis

### Same Transformation, Different Matrix

A linear transformation has different matrix representations in different bases.

If $A$ is the matrix in standard basis and $P$ is the change-of-basis matrix:

$$A' = P^{-1}AP$$

The matrix in the new basis.

### Diagonalization as Change of Basis

$$A = PDP^{-1}$$

In the eigenvector basis (columns of $P$), the transformation is just scaling along each axis!

```
Standard basis:           Eigenvector basis:

   Ax = complex            Dy = simple scaling
   ↗                       ↑
  /\                       |
 /  \                      |
```

---

## 7. Important Classes of Transformations

### Orthogonal Transformations

$$A^T A = I$$

- Preserve lengths: $\|Ax\| = \|x\|$
- Preserve angles
- Include rotations and reflections
- Eigenvalues have $|\lambda| = 1$

### Symmetric Transformations

$$A = A^T$$

- Real eigenvalues
- Orthogonal eigenvectors
- Diagonalizable by orthogonal matrix

### Positive Definite Transformations

$$\mathbf{x}^T A \mathbf{x} > 0 \text{ for all } \mathbf{x} \neq \mathbf{0}$$

- All positive eigenvalues
- Covariance matrices are positive semi-definite
- Important in optimization

---

## 8. Applications in ML/AI

### 1. Neural Network Layers

```
Input        Linear Transform    Nonlinearity     Output
x (d)   →      Wx + b       →      σ(·)      →     h (m)
              (d → m)

Each layer: Stretch, rotate, translate, then "bend"
```

### 2. Attention Mechanism

Query, Key, Value transformations:
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

Three linear transformations of the input!

### 3. Batch Normalization

Whitening transformation (decorrelation + scaling):
$$\hat{x} = \frac{x - \mu}{\sigma}$$

### 4. Word Embeddings

Words mapped to vectors via learned transformation:
$$\text{word} \rightarrow \mathbb{R}^d$$

Linear relationships emerge: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$

### 5. Convolutional Layers

Convolution is a linear operation!
Can be expressed as matrix multiplication with a special (Toeplitz) structure.

### 6. Data Augmentation

Random transformations (rotation, scaling, shearing) for augmentation:
$$x_{\text{augmented}} = T_{\text{random}}(x)$$

---

## 9. Homogeneous Coordinates

### Adding Translation

Translation $\mathbf{x} + \mathbf{b}$ is not linear in standard coordinates.

Solution: Use **homogeneous coordinates**:

$$\begin{bmatrix} x' \\ y' \\ 1 \end{bmatrix} = \begin{bmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Now translation is "linear" in augmented space!

### Affine Transformations

Linear transformation + translation:
$$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$$

Includes: rotation, scaling, shearing, translation, and compositions.

---

## 10. Summary

### Key Concepts

| Concept               | Definition                   |
| --------------------- | ---------------------------- |
| Linear transformation | $T(au + bv) = aT(u) + bT(v)$ |
| Matrix representation | $T(x) = Ax$                  |
| Composition           | $(T_2 \circ T_1)(x) = (BA)x$ |
| Kernel                | $\{v : T(v) = 0\}$           |
| Image                 | $\{T(v) : v \in V\}$         |
| Invertible            | $\det(A) \neq 0$             |

### Common 2D Transformations

| Type        | Matrix                                                                              |
| ----------- | ----------------------------------------------------------------------------------- |
| Scale       | $\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$                                  |
| Rotate      | $\begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$ |
| Reflect (x) | $\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$                                     |
| Shear       | $\begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$                                      |
| Project     | $\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$                                      |

### Neural Network Insight

```
Deep learning = Composition of transformations

Input → [Linear → Nonlinear] → [Linear → Nonlinear] → ... → Output
         Layer 1                  Layer 2

Without nonlinearities: Just one big linear transformation
With nonlinearities: Can approximate any function!
```

---

## Exercises

1. Verify that $T(x, y) = (2x + y, x - y)$ is a linear transformation
2. Find the matrix for rotation by 45°
3. Compose a rotation by 90° followed by reflection across y-axis
4. Find the kernel and image of projection onto x-axis
5. Show that ReLU is not a linear transformation

---

## References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Axler, S. - "Linear Algebra Done Right"
3. Goodfellow et al. - "Deep Learning" (Chapter 2)
