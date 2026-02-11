# Orthogonality and Orthonormality

## Introduction

Orthogonality is one of the most powerful and elegant concepts in linear algebra. When vectors or matrices satisfy orthogonality conditions, computations become simpler, algorithms more stable, and theoretical results more elegant. In machine learning and AI, orthogonality appears everywhere: PCA finds orthogonal directions of variance, neural network weights are often initialized as orthogonal matrices, attention mechanisms rely on orthogonal subspaces, and SVD produces orthogonal decompositions.

This section provides a comprehensive treatment of orthogonality, from basic definitions through advanced applications in modern deep learning.

## Prerequisites

- **Vectors and dot products**: Understanding vector operations
- **Matrix multiplication**: Basic matrix algebra
- **Eigenvalues and eigenvectors**: Spectral theory basics
- **Linear transformations**: How matrices transform vectors
- **Vector spaces**: Subspaces, basis, dimension

## Learning Objectives

By the end of this section, you will be able to:

1. Define and identify orthogonal and orthonormal vectors geometrically and algebraically
2. Understand the properties and significance of orthogonal matrices
3. Master the Gram-Schmidt process and its numerically stable variants
4. Apply orthogonal projections to vectors and subspaces
5. Perform and interpret QR decomposition
6. Connect least squares to orthogonality via normal equations
7. Understand orthogonal complement and decomposition theorems
8. Apply orthogonality concepts in ML contexts (PCA, weight init, attention)
9. Recognize numerical stability issues and their solutions

---

## 1. Orthogonal Vectors

### 1.1 Definition

Two vectors $\mathbf{u}$ and $\mathbf{v}$ are **orthogonal** (perpendicular) if their dot product is zero:

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i = 0$$

We write $\mathbf{u} \perp \mathbf{v}$ to denote orthogonality.

### 1.2 Geometric Interpretation

Orthogonal vectors form a 90° angle:

```
Orthogonal (90°):        Not orthogonal:         Opposite (180°):

    ↑ v                      ↑ v                    ↑ v
    │                       /                       │
    │                      /                        │
    └────→ u              └────→ u                 ↓───→ u

u · v = 0                u · v > 0               u · v < 0
```

The dot product measures how much two vectors "point in the same direction":
- **Positive**: Acute angle (< 90°)
- **Zero**: Right angle (= 90°)
- **Negative**: Obtuse angle (> 90°)

### 1.3 Examples

**Standard basis vectors:**
```python
e1 = [1, 0, 0]
e2 = [0, 1, 0]
e3 = [0, 0, 1]

e1 · e2 = 1×0 + 0×1 + 0×0 = 0  ✓ orthogonal
e1 · e3 = 1×0 + 0×0 + 0×1 = 0  ✓ orthogonal
e2 · e3 = 0×0 + 1×0 + 0×1 = 0  ✓ orthogonal
```

**Non-trivial example:**
```python
u = [1, 2, -1]
v = [3, 0, 3]

u · v = 1×3 + 2×0 + (-1)×3 = 3 + 0 - 3 = 0  ✓ orthogonal
```

### 1.4 Properties

1. **Zero vector**: The zero vector $\mathbf{0}$ is orthogonal to every vector (since $\mathbf{0} \cdot \mathbf{v} = 0$)

2. **Self-orthogonality**: A vector is orthogonal to itself only if it's the zero vector ($\mathbf{v} \cdot \mathbf{v} = 0 \Rightarrow \mathbf{v} = \mathbf{0}$)

3. **Symmetry**: If $\mathbf{u} \perp \mathbf{v}$, then $\mathbf{v} \perp \mathbf{u}$

4. **Pythagorean theorem**: If $\mathbf{u} \perp \mathbf{v}$, then:
   $$\|\mathbf{u} + \mathbf{v}\|^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2$$
   
   This follows from expanding $(\mathbf{u} + \mathbf{v}) \cdot (\mathbf{u} + \mathbf{v}) = \|\mathbf{u}\|^2 + 2\mathbf{u} \cdot \mathbf{v} + \|\mathbf{v}\|^2$

5. **Linear combinations**: If $\mathbf{u} \perp \mathbf{v}$ and $\mathbf{u} \perp \mathbf{w}$, then $\mathbf{u} \perp (\alpha\mathbf{v} + \beta\mathbf{w})$ for any scalars $\alpha, \beta$

### 1.5 Orthogonal Sets

A set of vectors $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ is **orthogonal** if every pair is orthogonal:
$$\mathbf{v}_i \cdot \mathbf{v}_j = 0 \quad \text{for all } i \neq j$$

**Key theorem**: Any orthogonal set of nonzero vectors is linearly independent.

*Proof*: Suppose $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$. Taking the dot product with $\mathbf{v}_i$:
$$c_i \mathbf{v}_i \cdot \mathbf{v}_i = c_i \|\mathbf{v}_i\|^2 = 0$$
Since $\|\mathbf{v}_i\|^2 \neq 0$, we have $c_i = 0$ for all $i$.

---

## 2. Orthonormal Vectors

### 2.1 Definition

Vectors are **orthonormal** if they satisfy two conditions:

1. **Orthogonal**: $\mathbf{u}_i \cdot \mathbf{u}_j = 0$ for all $i \neq j$
2. **Normalized** (unit length): $\|\mathbf{u}_i\| = 1$ for all $i$

Compact notation using the Kronecker delta:
$$\mathbf{u}_i^T \mathbf{u}_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

### 2.2 Standard Basis

The simplest orthonormal set is the standard basis:

$$\mathbf{e}_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad
\mathbf{e}_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

### 2.3 Creating Orthonormal Vectors

To normalize any nonzero vector $\mathbf{v}$:
$$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

**Example (45° rotated basis in 2D):**
$$\mathbf{u}_1 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}, \quad
\mathbf{u}_2 = \begin{bmatrix} -1/\sqrt{2} \\ 1/\sqrt{2} \end{bmatrix}$$

Verification:
- $\mathbf{u}_1 \cdot \mathbf{u}_2 = \frac{1}{2}(-1) + \frac{1}{2}(1) = 0$ ✓
- $\|\mathbf{u}_1\| = \sqrt{1/2 + 1/2} = 1$ ✓
- $\|\mathbf{u}_2\| = \sqrt{1/2 + 1/2} = 1$ ✓

### 2.4 Advantages of Orthonormal Bases

Computing coordinates in an orthonormal basis is trivial:

If $\{\mathbf{u}_1, \ldots, \mathbf{u}_n\}$ is orthonormal, then any vector $\mathbf{v}$ can be written as:
$$\mathbf{v} = \sum_{i=1}^n (\mathbf{v} \cdot \mathbf{u}_i) \mathbf{u}_i$$

The coefficient for $\mathbf{u}_i$ is simply $\mathbf{v} \cdot \mathbf{u}_i$ — no matrix inversion needed!

---

## 3. Orthogonal Matrices

### 3.1 Definition

A square matrix $Q$ is **orthogonal** if:

$$Q^T Q = Q Q^T = I$$

Equivalently: $Q^{-1} = Q^T$ (the inverse is just the transpose!)

### 3.2 Column Interpretation

The columns of an orthogonal matrix form an orthonormal basis:

$$Q = \begin{bmatrix} | & | & & | \\ \mathbf{q}_1 & \mathbf{q}_2 & \cdots & \mathbf{q}_n \\ | & | & & | \end{bmatrix}$$

The condition $Q^TQ = I$ means:
- $\mathbf{q}_i^T \mathbf{q}_i = 1$ (unit length)
- $\mathbf{q}_i^T \mathbf{q}_j = 0$ for $i \neq j$ (orthogonal)

### 3.3 Properties of Orthogonal Matrices

| Property | Formula | Significance |
|----------|---------|--------------|
| Inverse = Transpose | $Q^{-1} = Q^T$ | Extremely cheap inversion |
| Preserves length | $\|Q\mathbf{x}\| = \|\mathbf{x}\|$ | Isometry |
| Preserves angles | $\angle(Q\mathbf{x}, Q\mathbf{y}) = \angle(\mathbf{x}, \mathbf{y})$ | Rigid transformation |
| Preserves dot product | $(Q\mathbf{x})^T(Q\mathbf{y}) = \mathbf{x}^T\mathbf{y}$ | Preserves inner products |
| Determinant | $\det(Q) = \pm 1$ | Area/volume preserving |
| Eigenvalues | $|\lambda| = 1$ | Complex on unit circle |
| Product | $Q_1 Q_2$ is orthogonal | Closed under multiplication |

### 3.4 Proof: Length Preservation

$$\|Q\mathbf{x}\|^2 = (Q\mathbf{x})^T(Q\mathbf{x}) = \mathbf{x}^T Q^T Q \mathbf{x} = \mathbf{x}^T I \mathbf{x} = \|\mathbf{x}\|^2$$

### 3.5 Geometric Classification

```
det(Q) = +1: ROTATION (proper orthogonal)
         - Preserves orientation
         - Example: Rotating 45° counterclockwise

det(Q) = -1: REFLECTION (improper orthogonal)
         - Reverses orientation
         - Example: Mirror flip across a line
```

### 3.6 Common Orthogonal Matrices

**2D Rotation by angle θ:**
$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**2D Reflection across x-axis:**
$$M_x = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**3D Rotation around z-axis:**
$$R_z(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Permutation matrix:**
$$P = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}$$

---

## 4. The Gram-Schmidt Process

### 4.1 Goal

Convert any linearly independent set $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k\}$ into an orthonormal set $\{\mathbf{u}_1, \mathbf{u}_2, \ldots, \mathbf{u}_k\}$ that spans the same subspace.

### 4.2 Algorithm (Classical Gram-Schmidt)

**Step 1**: Normalize first vector
$$\mathbf{u}_1 = \frac{\mathbf{v}_1}{\|\mathbf{v}_1\|}$$

**Step 2**: Subtract projection onto $\mathbf{u}_1$, then normalize
$$\mathbf{w}_2 = \mathbf{v}_2 - (\mathbf{v}_2 \cdot \mathbf{u}_1)\mathbf{u}_1$$
$$\mathbf{u}_2 = \frac{\mathbf{w}_2}{\|\mathbf{w}_2\|}$$

**General step k**: Subtract projections onto all previous, then normalize
$$\mathbf{w}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} (\mathbf{v}_k \cdot \mathbf{u}_j)\mathbf{u}_j$$
$$\mathbf{u}_k = \frac{\mathbf{w}_k}{\|\mathbf{w}_k\|}$$

### 4.3 Geometric Intuition

```
Step 1: Take v₁, make it unit length → u₁

Step 2: Take v₂, subtract its component along u₁, normalize
        
        v₂ ────────────→ •
                         │
        proj_{u₁}(v₂)    │ w₂ (perpendicular to u₁)
        ←────────────────┘
        0 ────────────→ u₁

Step 3: Take v₃, subtract components along u₁ and u₂, normalize
        Result: u₃ is perpendicular to both u₁ and u₂
```

### 4.4 Worked Example

Orthonormalize: $\mathbf{v}_1 = (1, 1, 0)$, $\mathbf{v}_2 = (1, 0, 1)$, $\mathbf{v}_3 = (0, 1, 1)$

**Step 1:**
$$\mathbf{u}_1 = \frac{(1, 1, 0)}{\sqrt{2}} = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0\right)$$

**Step 2:**
$$\mathbf{v}_2 \cdot \mathbf{u}_1 = \frac{1}{\sqrt{2}}$$
$$\mathbf{w}_2 = (1, 0, 1) - \frac{1}{\sqrt{2}} \cdot \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0\right) = \left(\frac{1}{2}, -\frac{1}{2}, 1\right)$$
$$\|\mathbf{w}_2\| = \sqrt{\frac{1}{4} + \frac{1}{4} + 1} = \sqrt{\frac{3}{2}}$$
$$\mathbf{u}_2 = \frac{1}{\sqrt{3/2}}\left(\frac{1}{2}, -\frac{1}{2}, 1\right) = \left(\frac{1}{\sqrt{6}}, -\frac{1}{\sqrt{6}}, \frac{2}{\sqrt{6}}\right)$$

**Step 3:** (Similar computation gives $\mathbf{u}_3$)

### 4.5 Modified Gram-Schmidt (More Stable)

Classical Gram-Schmidt can lose orthogonality due to floating-point errors. Modified version reorthogonalizes against the *current* intermediate result:

```python
# Classical (less stable)
def classical_gram_schmidt(V):
    Q = np.zeros_like(V)
    for j in range(V.shape[1]):
        v = V[:, j].copy()
        for i in range(j):
            v -= np.dot(V[:, j], Q[:, i]) * Q[:, i]  # Uses original V
        Q[:, j] = v / np.linalg.norm(v)
    return Q

# Modified (more stable)
def modified_gram_schmidt(V):
    Q = V.copy().astype(float)
    for j in range(V.shape[1]):
        Q[:, j] = Q[:, j] / np.linalg.norm(Q[:, j])
        for i in range(j + 1, V.shape[1]):
            Q[:, i] -= np.dot(Q[:, j], Q[:, i]) * Q[:, j]  # Uses current Q
    return Q
```

The key difference: Modified GS uses the progressively orthogonalized vectors, accumulating less error.

---

## 5. Orthogonal Projections

### 5.1 Projection onto a Vector

The **projection** of $\mathbf{v}$ onto $\mathbf{u}$ is the closest point to $\mathbf{v}$ on the line spanned by $\mathbf{u}$:

$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\mathbf{u} \cdot \mathbf{u}} \mathbf{u} = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|^2} \mathbf{u}$$

If $\mathbf{u}$ is a **unit vector**:
$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = (\mathbf{v} \cdot \mathbf{u}) \mathbf{u}$$

### 5.2 Geometric Picture

```
        v
       /│
      / │  v - proj (perpendicular to u)
     /  │
    •───┴────────→ u
    ↑
   proj_u(v)
```

The projection decomposes $\mathbf{v}$ into:
- **Parallel component**: $\mathbf{v}_\parallel = \text{proj}_{\mathbf{u}}(\mathbf{v})$
- **Perpendicular component**: $\mathbf{v}_\perp = \mathbf{v} - \mathbf{v}_\parallel$

Key property: $\mathbf{v}_\parallel \perp \mathbf{v}_\perp$

### 5.3 Projection Matrix

The matrix that projects any vector onto $\mathbf{u}$:

$$P = \frac{\mathbf{u}\mathbf{u}^T}{\mathbf{u}^T\mathbf{u}}$$

For unit vector $\mathbf{u}$: $P = \mathbf{u}\mathbf{u}^T$

**Example:** Projection onto $\mathbf{u} = (1, 2)^T$:
$$P = \frac{1}{5}\begin{bmatrix} 1 \\ 2 \end{bmatrix}\begin{bmatrix} 1 & 2 \end{bmatrix} = \frac{1}{5}\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$$

### 5.4 Properties of Projection Matrices

| Property | Meaning |
|----------|---------|
| $P^2 = P$ | **Idempotent**: Projecting twice = projecting once |
| $P^T = P$ | **Symmetric** |
| rank$(P) = 1$ | Projects onto 1D space |
| trace$(P) = 1$ | Sum of eigenvalues is 1 |
| $I - P$ | Projects onto orthogonal complement |
| Eigenvalues | Only 0 and 1 |

### 5.5 Projection onto a Subspace

If columns of $A$ span a subspace $W$, the projection matrix onto $W$:

$$P = A(A^T A)^{-1} A^T$$

If columns of $A$ are already **orthonormal** ($Q$):
$$P = QQ^T$$

**Key insight**: $P$ finds the closest point in $W$ to any vector $\mathbf{b}$.

---

## 6. Orthogonal Decomposition

### 6.1 Fundamental Theorem of Orthogonal Projections

Any vector $\mathbf{v}$ can be uniquely decomposed relative to a subspace $W$:

$$\mathbf{v} = \mathbf{v}_W + \mathbf{v}_{W^\perp}$$

where:
- $\mathbf{v}_W \in W$ (projection onto $W$)
- $\mathbf{v}_{W^\perp} \in W^\perp$ (projection onto orthogonal complement)

```
              v
             /│
            / │ v_{W⊥}
           /  │
          •───┴──────→ W
         ↑
        v_W
```

### 6.2 Orthogonal Complement

The **orthogonal complement** of $W$ is:

$$W^{\perp} = \{\mathbf{v} \in \mathbb{R}^n : \mathbf{v} \cdot \mathbf{w} = 0 \text{ for all } \mathbf{w} \in W\}$$

### 6.3 Properties

1. **Dimension formula**: $\dim(W) + \dim(W^{\perp}) = n$

2. **Double complement**: $(W^{\perp})^{\perp} = W$

3. **Intersection**: $W \cap W^{\perp} = \{\mathbf{0}\}$

4. **Direct sum**: $\mathbb{R}^n = W \oplus W^\perp$

### 6.4 Example

Let $W$ = xy-plane in $\mathbb{R}^3$ (spanned by $\mathbf{e}_1, \mathbf{e}_2$).

Then $W^\perp$ = z-axis (spanned by $\mathbf{e}_3$).

For $\mathbf{v} = (3, 4, 5)$:
- $\mathbf{v}_W = (3, 4, 0)$
- $\mathbf{v}_{W^\perp} = (0, 0, 5)$
- $\mathbf{v} = \mathbf{v}_W + \mathbf{v}_{W^\perp}$ ✓

### 6.5 Connection to Linear Systems

For any matrix $A$:
- Column space of $A$ ⊥ Left null space of $A$
- Row space of $A$ ⊥ Null space of $A$

This is the **Fundamental Theorem of Linear Algebra**.

---

## 7. QR Decomposition

### 7.1 Definition

Any matrix $A$ with linearly independent columns can be factored:

$$A = QR$$

where:
- $Q$ has orthonormal columns (same column space as $A$)
- $R$ is upper triangular with positive diagonal

### 7.2 Connection to Gram-Schmidt

QR decomposition IS Gram-Schmidt in matrix form!

If $A = [\mathbf{a}_1 | \mathbf{a}_2]$ and we apply Gram-Schmidt to get $[\mathbf{q}_1 | \mathbf{q}_2]$:

$$A = QR = \begin{bmatrix} | & | \\ \mathbf{q}_1 & \mathbf{q}_2 \\ | & | \end{bmatrix} \begin{bmatrix} r_{11} & r_{12} \\ 0 & r_{22} \end{bmatrix}$$

where:
- $r_{11} = \|\mathbf{a}_1\|$
- $r_{12} = \mathbf{q}_1^T \mathbf{a}_2$
- $r_{22} = \|\mathbf{a}_2 - r_{12}\mathbf{q}_1\|$

### 7.3 Why R is Upper Triangular

Each $\mathbf{a}_j$ is a linear combination of $\mathbf{q}_1, \ldots, \mathbf{q}_j$ (not later $\mathbf{q}$'s), so:
$$\mathbf{a}_j = r_{1j}\mathbf{q}_1 + r_{2j}\mathbf{q}_2 + \cdots + r_{jj}\mathbf{q}_j$$

The coefficients $r_{ij}$ for $i > j$ are zero.

### 7.4 Computing R

Once we have $Q$, computing $R$ is simple:
$$R = Q^T A$$

This works because $Q^T Q = I$.

### 7.5 Full vs. Thin QR

For $m \times n$ matrix with $m > n$:

**Thin QR** (reduced):
- $Q$: $m \times n$ (orthonormal columns)
- $R$: $n \times n$ (upper triangular)

**Full QR**:
- $Q$: $m \times m$ (orthogonal matrix)
- $R$: $m \times n$ (has zero rows below)

### 7.6 Applications

1. **Least squares**: Solve $A\mathbf{x} = \mathbf{b}$ via $R\mathbf{x} = Q^T\mathbf{b}$
2. **Eigenvalue computation**: QR algorithm iteratively computes eigenvalues
3. **Orthonormal basis**: Columns of $Q$ form orthonormal basis for col($A$)
4. **Matrix rank**: Number of nonzero diagonal entries in $R$

---

## 8. Least Squares via Orthogonality

### 8.1 The Problem

For an overdetermined system $A\mathbf{x} = \mathbf{b}$ (more equations than unknowns), find $\hat{\mathbf{x}}$ that minimizes:

$$\|A\mathbf{x} - \mathbf{b}\|^2$$

### 8.2 Geometric Solution

The residual $\mathbf{r} = \mathbf{b} - A\hat{\mathbf{x}}$ should be orthogonal to the column space of $A$:

```
        b
       /│
      / │ r = b - Ax̂ (residual)
     /  │
    •───┴──────── Col(A)
    ↑
    Ax̂ (projection of b onto Col(A))
```

This means: $A^T(\mathbf{b} - A\hat{\mathbf{x}}) = \mathbf{0}$

### 8.3 Normal Equations

Rearranging gives the **normal equations**:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

Solution: $\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$

### 8.4 Solution via QR (More Stable)

If $A = QR$:

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$
$$R^T Q^T Q R \hat{\mathbf{x}} = R^T Q^T \mathbf{b}$$
$$R^T R \hat{\mathbf{x}} = R^T Q^T \mathbf{b}$$
$$R \hat{\mathbf{x}} = Q^T \mathbf{b}$$

This is better because:
- R is triangular (easy to solve)
- Condition number of $R$ = $\sqrt{\text{cond}(A^T A)}$
- No need to form $A^T A$ (which squares the condition number)

### 8.5 Linear Regression Example

Fitting $y = \beta_0 + \beta_1 x$ to data points:

Design matrix: $A = \begin{bmatrix} 1 & x_1 \\ 1 & x_2 \\ \vdots & \vdots \\ 1 & x_n \end{bmatrix}$

The least squares solution via QR gives the best-fit line coefficients $[\beta_0, \beta_1]^T$.

---

## 9. Applications in Machine Learning

### 9.1 Principal Component Analysis (PCA)

PCA finds orthogonal directions of maximum variance in data.

**Key fact**: Eigenvectors of the covariance matrix $\Sigma = \frac{1}{n}X^TX$ are orthogonal.

```
PCA finds orthogonal principal components:

    PC2 ↑
        │    . . .
        │   . data .
        │  . . . .
        └──────────→ PC1

PC1 has maximum variance
PC2 ⊥ PC1, has second-most variance
```

**Why orthogonal?**: Symmetric matrices have orthogonal eigenvectors. The covariance matrix is symmetric!

### 9.2 Orthogonal Weight Initialization

Neural network weights initialized as orthogonal matrices help with:

1. **Signal preservation**: $\|W\mathbf{x}\| = \|\mathbf{x}\|$
2. **Gradient preservation**: Gradients don't vanish or explode through layers
3. **Better conditioning**: All singular values are 1

```python
# PyTorch orthogonal initialization
import torch.nn as nn

layer = nn.Linear(256, 256)
nn.init.orthogonal_(layer.weight)
```

**Signal propagation comparison:**
```
Random init:      After 50 layers: ||signal|| → 0 or ∞
Orthogonal init:  After 50 layers: ||signal|| ≈ original
```

### 9.3 Attention Mechanisms

In transformers, queries and keys use dot products:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

If query vectors are near-orthogonal:
- Different queries attend to different keys
- Reduced redundancy in attention patterns
- Better representation capacity

### 9.4 Batch Normalization and Whitening

Whitening decorrelates features, making them orthogonal:

$$\hat{\mathbf{x}} = \Sigma^{-1/2}(\mathbf{x} - \mu)$$

After whitening:
- Features have zero mean
- Features are uncorrelated (orthogonal in expectation)
- Unit variance

### 9.5 Singular Value Decomposition (SVD)

SVD produces orthogonal matrices:

$$A = U \Sigma V^T$$

- $U$: orthogonal (left singular vectors)
- $V$: orthogonal (right singular vectors)
- $\Sigma$: diagonal (singular values)

Applications:
- **Dimensionality reduction**: Keep top-$k$ singular vectors
- **Denoising**: Truncate small singular values
- **Pseudoinverse**: $A^+ = V\Sigma^+ U^T$

### 9.6 Word Embeddings

Semantic relationships often appear as near-orthogonal directions:

```
"king" - "man" ≈ "queen" - "woman"

This vector (gender direction) is ideally orthogonal 
to other semantic dimensions (age, profession, etc.)
```

### 9.7 Recurrent Neural Networks

Orthogonal recurrent weights help with:
- Long-term memory (signal doesn't decay)
- Avoiding vanishing gradients
- Better training dynamics

The Orthogonal RNN and related architectures enforce orthogonality constraints.

---

## 10. Numerical Considerations

### 10.1 Loss of Orthogonality

Classical Gram-Schmidt can produce vectors that aren't quite orthogonal due to:
- Floating-point rounding errors
- Nearly linearly dependent input
- Accumulation of errors across many vectors

### 10.2 Solutions

**Modified Gram-Schmidt (MGS):**
- Reorthogonalizes against current intermediate vectors
- More stable than classical GS
- Still not perfect for ill-conditioned problems

**Householder Reflections:**
$$H = I - 2\mathbf{v}\mathbf{v}^T \quad (\|\mathbf{v}\| = 1)$$

Properties:
- $H$ is orthogonal
- $H = H^T$ (symmetric)
- $H^2 = I$ (self-inverse)
- Used in LAPACK for QR

**Givens Rotations:**
- Rotate in 2D plane
- Good for sparse matrices
- Can parallelize easily

### 10.3 Condition Number

A matrix is well-conditioned for orthogonalization if its columns are far from linearly dependent.

Condition number: $\kappa(A) = \sigma_{\max}/\sigma_{\min}$

- $\kappa \approx 1$: Well-conditioned
- $\kappa \gg 1$: Ill-conditioned (GS may fail)

---

## 11. Advanced Topics

### 11.1 Orthogonal Procrustes Problem

Find the orthogonal matrix $Q$ that best aligns $A$ to $B$:

$$\min_Q \|A - BQ\|_F \quad \text{subject to } Q^TQ = I$$

Solution: If $B^T A = U\Sigma V^T$, then $Q = UV^T$

Applications:
- Shape alignment
- Coordinate transformation
- Word embedding alignment between languages

### 11.2 Orthogonal Regularization

Loss term encouraging orthogonality:

$$\mathcal{L}_{\text{orth}} = \|W^T W - I\|_F^2$$

Used in:
- Generator networks (StyleGAN)
- Recurrent networks
- Stable training of deep networks

### 11.3 Unitary Matrices (Complex Extension)

For complex matrices, orthogonality generalizes to **unitarity**:

$$U^* U = I \quad (\text{where } U^* = \overline{U}^T)$$

Important in:
- Quantum computing (all quantum gates are unitary)
- Complex neural networks
- Signal processing

---

## 12. Summary

### Key Concepts

| Concept | Definition | Key Property |
|---------|------------|--------------|
| Orthogonal vectors | $\mathbf{u} \cdot \mathbf{v} = 0$ | Perpendicular |
| Orthonormal | Orthogonal + unit length | $\mathbf{u}_i^T \mathbf{u}_j = \delta_{ij}$ |
| Orthogonal matrix | $Q^TQ = I$ | $Q^{-1} = Q^T$ |
| Projection | $\text{proj}_\mathbf{u}(\mathbf{v}) = \frac{\mathbf{v} \cdot \mathbf{u}}{\|\mathbf{u}\|^2}\mathbf{u}$ | Closest point |
| Gram-Schmidt | Orthonormalize a set | Produces orthonormal basis |
| QR decomposition | $A = QR$ | Q orthonormal, R triangular |
| Orthogonal complement | $W^\perp$ | Perpendicular subspace |

### Why Orthogonality Matters

```
Benefits of orthogonal computations:
├── Numerical stability (errors don't compound)
├── Computational efficiency (Q⁻¹ = Qᵀ is free)
├── Geometric interpretability (rotations, reflections)
├── Mathematical elegance (preservation properties)
├── Signal preservation (no amplification/decay)
└── Decorrelation (independent components)
```

### ML Applications Summary

| Application | Orthogonality Role |
|-------------|-------------------|
| PCA | Principal components are orthogonal eigenvectors |
| Weight init | Orthogonal matrices preserve gradient magnitude |
| SVD | U and V are orthogonal matrices |
| Least squares | Error is orthogonal to column space |
| Whitening | Decorrelate = make features orthogonal |
| Attention | Orthogonal queries/keys → diverse attention |
| RNN stability | Orthogonal recurrent weights prevent gradient issues |

---

## Exercises

### Conceptual
1. Show that the Pythagorean theorem holds for orthogonal vectors
2. Prove that orthogonal matrices preserve distance between vectors
3. Explain why modified Gram-Schmidt is more stable than classical

### Computational
4. Apply Gram-Schmidt to $\{(1, 1, 1), (0, 1, 1), (0, 0, 1)\}$
5. Find the projection matrix onto the plane spanned by $(1, 0, 1)$ and $(0, 1, 1)$
6. Compute the QR decomposition of $\begin{bmatrix} 1 & 2 \\ 1 & 1 \\ 0 & 1 \end{bmatrix}$
7. Solve the least squares problem $Ax = b$ where $A = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}$, $b = \begin{bmatrix} 2 \\ 3 \\ 5 \end{bmatrix}$

### Programming
8. Implement Gram-Schmidt (classical and modified) and compare numerical stability
9. Demonstrate signal preservation through 50 layers with orthogonal vs random weights
10. Verify that PCA directions are orthogonal on a real dataset

---

## References

1. **Strang, G.** - "Linear Algebra and Its Applications" (Chapters on orthogonality)
2. **Trefethen & Bau** - "Numerical Linear Algebra" (QR algorithms, stability)
3. **Goodfellow, Bengio, Courville** - "Deep Learning" (Weight initialization)
4. **Saxe, McClelland, Ganguli** - "Exact solutions to nonlinear dynamics of learning in deep linear neural networks" (2014) - Theory of orthogonal initialization
5. **Arjovsky, Shah, Bengio** - "Unitary Evolution Recurrent Neural Networks" (2016) - Orthogonal RNNs
6. **Vaswani et al.** - "Attention is All You Need" (2017) - Transformer attention

---

## Files in This Section

- [theory.ipynb](theory.ipynb) - Comprehensive code examples with visualizations
- [exercises.ipynb](exercises.ipynb) - Practice problems with solutions
