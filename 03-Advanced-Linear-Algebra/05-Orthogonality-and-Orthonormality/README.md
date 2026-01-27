# Orthogonality and Orthonormality

## Introduction

Orthogonality is one of the most powerful concepts in linear algebra. Orthogonal vectors and matrices enable simpler computations, stable numerical algorithms, and elegant theoretical results. In ML, orthogonality appears in PCA, neural network weight initialization, and attention mechanisms.

## Prerequisites

- Vectors and dot products
- Matrix multiplication
- Eigenvalues and eigenvectors
- Linear transformations

## Learning Objectives

1. Understand orthogonality geometrically and algebraically
2. Master the Gram-Schmidt process
3. Apply orthogonal projections
4. Recognize and use orthogonal matrices
5. Apply orthogonality concepts in ML contexts

---

## 1. Orthogonal Vectors

### Definition

Two vectors $\mathbf{u}$ and $\mathbf{v}$ are **orthogonal** (perpendicular) if:

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = 0$$

### Geometric Interpretation

```
Orthogonal (90°):        Not orthogonal:

    ↑ v                      ↑ v
    │                       /
    │                      /
    └────→ u              └────→ u

u · v = 0                u · v ≠ 0
```

### Properties

1. **Zero vector** is orthogonal to every vector
2. **Pythagorean theorem**: If $\mathbf{u} \perp \mathbf{v}$:
   $$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$$

---

## 2. Orthonormal Vectors

### Definition

Vectors are **orthonormal** if they are:

1. **Orthogonal**: $\mathbf{u}_i \cdot \mathbf{u}_j = 0$ for $i \neq j$
2. **Normalized**: $\|\mathbf{u}_i\| = 1$ for all $i$

In matrix form:
$$\mathbf{u}_i^T \mathbf{u}_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### Standard Basis

The standard basis vectors form an orthonormal set:

$$
\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}
$$

---

## 3. Orthogonal Matrices

### Definition

A square matrix $Q$ is **orthogonal** if:

$$Q^T Q = Q Q^T = I$$

Equivalently: $Q^{-1} = Q^T$

### Key Property

The columns of $Q$ form an orthonormal basis!

$$Q = \begin{bmatrix} | & | & & | \\ \mathbf{q}_1 & \mathbf{q}_2 & \cdots & \mathbf{q}_n \\ | & | & & | \end{bmatrix}$$

with $\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$

### Properties of Orthogonal Matrices

| Property              | Formula                                                             |
| --------------------- | ------------------------------------------------------------------- | ------- | ---- |
| Inverse = Transpose   | $Q^{-1} = Q^T$                                                      |
| Preserves length      | $\|Q\mathbf{x}\| = \|\mathbf{x}\|$                                  |
| Preserves angles      | $\angle(Q\mathbf{x}, Q\mathbf{y}) = \angle(\mathbf{x}, \mathbf{y})$ |
| Preserves dot product | $(Q\mathbf{x})^T(Q\mathbf{y}) = \mathbf{x}^T\mathbf{y}$             |
| Determinant           | $\det(Q) = \pm 1$                                                   |
| Eigenvalues           | $                                                                   | \lambda | = 1$ |
| Product               | $Q_1 Q_2$ is orthogonal                                             |

### Types

```
det(Q) = +1: Rotation (preserves orientation)
det(Q) = -1: Reflection (reverses orientation)
```

---

## 4. The Gram-Schmidt Process

### Goal

Convert any linearly independent set $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ into an orthonormal set $\{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_k\}$.

### Algorithm

**Step 1**: Start with first vector
$$\mathbf{u}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}$$

**Step 2**: Make second vector orthogonal, then normalize
$$\mathbf{w}_2 = \mathbf{v}_2 - (\mathbf{v}_2 \cdot \mathbf{u}_1)\mathbf{u}_1$$
$$\mathbf{u}_2 = \frac{\mathbf{w}_2}{\|\mathbf{w}_2\|}$$

**Step 3**: Continue for remaining vectors
$$\mathbf{w}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} (\mathbf{v}_k \cdot \mathbf{u}_j)\mathbf{u}_j$$
$$\mathbf{u}_k = \frac{\mathbf{w}_k}{\|\mathbf{w}_k\|}$$

### Visualization

```
Original:           After Gram-Schmidt:

    v₂                    u₂
   /                      |
  /                       |
 /────→ v₁               └────→ u₁

(not orthogonal)     (orthonormal)
```

### Geometric Intuition

Each step subtracts the projection onto previous vectors:

```
v₂ ──────────────────→ •
                       │
       proj_{u₁}(v₂)   │ w₂ = v₂ - proj
←──────────────────────┘
0 ─────────────────────→ u₁
```

---

## 5. Orthogonal Projections

### Projection onto a Vector

The projection of $\mathbf{v}$ onto $\mathbf{u}$:

$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u} = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|^2} \mathbf{u}$$

If $\mathbf{u}$ is unit length:
$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = (\mathbf{v} \cdot \mathbf{u}) \mathbf{u}$$

### Projection Matrix

The matrix that projects onto $\mathbf{u}$:

$$P = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}$$

For unit vector: $P = \mathbf{u}\mathbf{u}^T$

### Properties of Projection Matrices

| Property             | Meaning                             |
| -------------------- | ----------------------------------- |
| $P^2 = P$            | Projecting twice = projecting once  |
| $P^T = P$            | Symmetric                           |
| $\text{rank}(P) = 1$ | Projects onto 1D space              |
| $I - P$              | Projects onto orthogonal complement |

### Projection onto a Subspace

If columns of $A$ span a subspace, projection matrix onto that subspace:

$$P = A(A^T A)^{-1} A^T$$

If columns of $A$ are orthonormal ($Q$):
$$P = QQ^T$$

---

## 6. Orthogonal Decomposition

### Fundamental Theorem

Any vector $\mathbf{v}$ can be uniquely decomposed:

$$\mathbf{v} = \mathbf{v}_{\parallel} + \mathbf{v}_{\perp}$$

where:

- $\mathbf{v}_{\parallel}$ is in the subspace $W$
- $\mathbf{v}_{\perp}$ is in the orthogonal complement $W^{\perp}$

```
           v
          /│
         / │ v_⊥
        /  │
       •───┴──────→
   v_∥ (in subspace)
```

### Orthogonal Complement

$$W^{\perp} = \{\mathbf{v} : \mathbf{v} \cdot \mathbf{w} = 0 \text{ for all } \mathbf{w} \in W\}$$

Properties:

- $\dim(W) + \dim(W^{\perp}) = n$
- $(W^{\perp})^{\perp} = W$
- $W \cap W^{\perp} = \{\mathbf{0}\}$

---

## 7. QR Decomposition

### Definition

Any matrix $A$ with linearly independent columns can be factored:

$$A = QR$$

where:

- $Q$ has orthonormal columns
- $R$ is upper triangular with positive diagonal

### Connection to Gram-Schmidt

QR decomposition IS Gram-Schmidt in matrix form!

$$A = \begin{bmatrix} | & | \\ \mathbf{a}_1 & \mathbf{a}_2 \\ | & | \end{bmatrix} = \begin{bmatrix} | & | \\ \mathbf{q}_1 & \mathbf{q}_2 \\ | & | \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} \\ 0 & r_{22} \end{bmatrix}$$

### Applications

1. **Solving least squares**: $\mathbf{x} = R^{-1}Q^T\mathbf{b}$
2. **Computing eigenvalues**: QR algorithm
3. **Numerical stability**: Better than Gaussian elimination

---

## 8. Least Squares via Orthogonality

### The Problem

For overdetermined system $A\mathbf{x} = \mathbf{b}$, find $\hat{\mathbf{x}}$ that minimizes $\|A\mathbf{x} - \mathbf{b}\|^2$.

### Geometric Solution

$\hat{\mathbf{x}}$ is the solution where $\mathbf{b} - A\hat{\mathbf{x}}$ is orthogonal to column space of $A$:

$$A^T(b - A\hat{\mathbf{x}}) = 0$$

This gives the **normal equations**:
$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

### Using QR

If $A = QR$:
$$\hat{\mathbf{x}} = R^{-1}Q^T\mathbf{b}$$

More numerically stable!

---

## 9. Applications in ML/AI

### 1. Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance:

$$\text{Eigenvectors of } X^TX \text{ are orthogonal}$$

```
PCA Components:

    PC2 ↑
        │    /
        │   /  data
        │  /
        └──────→ PC1

PC1 ⊥ PC2 (orthogonal)
```

### 2. Orthogonal Weight Initialization

Initialize neural network weights as orthogonal matrices:

```python
# PyTorch orthogonal init
W = torch.nn.init.orthogonal_(torch.empty(n, m))
```

Benefits:

- Preserves gradient magnitudes
- Helps with vanishing/exploding gradients
- Better signal propagation

### 3. Attention in Transformers

Query-Key dot products measure similarity:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Orthogonal Q, K → Different queries attend to different things

### 4. Batch Normalization

Whitening (decorrelating) features:
$$\hat{\mathbf{x}} = \Sigma^{-1/2}(\mathbf{x} - \mu)$$

Makes features orthogonal (uncorrelated)

### 5. Word Embeddings

Analogy relations are often near-orthogonal:
$$\text{king} - \text{man} \approx \text{queen} - \text{woman}$$

### 6. Singular Value Decomposition

SVD produces orthogonal matrices:
$$A = U \Sigma V^T$$

$U$ and $V$ have orthonormal columns

---

## 10. Numerical Considerations

### Modified Gram-Schmidt

Classical Gram-Schmidt can lose orthogonality due to numerical errors. Modified version is more stable:

```python
# Classical (less stable)
for j in range(k):
    w -= (v.dot(u[j])) * u[j]

# Modified (more stable)
for j in range(k):
    w -= (w.dot(u[j])) * u[j]  # Use current w, not original v
```

### Householder Reflections

More stable than Gram-Schmidt for QR:

$$H = I - 2\mathbf{v}\mathbf{v}^T$$

where $\|\mathbf{v}\| = 1$

---

## 11. Summary

### Key Concepts

| Concept            | Definition                                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------------- |
| Orthogonal vectors | $\mathbf{u} \cdot \mathbf{v} = 0$                                                                     |
| Orthonormal        | Orthogonal + unit length                                                                              |
| Orthogonal matrix  | $Q^TQ = I$                                                                                            |
| Projection         | $\text{proj}_\mathbf{u}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|^2}\mathbf{u}$ |
| Gram-Schmidt       | Convert to orthonormal basis                                                                          |
| QR decomposition   | $A = QR$                                                                                              |

### Why Orthogonality Matters

```
Orthogonal computations are:
├── Numerically stable
├── Computationally efficient (Q⁻¹ = Qᵀ)
├── Geometrically interpretable
└── Mathematically elegant
```

### ML Applications Summary

| Application   | Orthogonality Role                     |
| ------------- | -------------------------------------- |
| PCA           | Principal components are orthogonal    |
| Weight init   | Orthogonal matrices preserve gradients |
| SVD           | $U$, $V$ are orthogonal                |
| Least squares | Error orthogonal to column space       |
| Whitening     | Decorrelate = make orthogonal          |

---

## Exercises

1. Show that $\begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$ and $\begin{bmatrix} -1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$ are orthonormal
2. Apply Gram-Schmidt to $\{(1, 1, 0), (1, 0, 1), (0, 1, 1)\}$
3. Find the projection matrix onto $\mathbf{u} = (1, 2, 2)$
4. Verify that rotation matrices are orthogonal
5. Compute the QR decomposition of $\begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$

---

## References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Trefethen & Bau - "Numerical Linear Algebra"
3. Goodfellow et al. - "Deep Learning" (weight initialization)
