# Eigenvalues and Eigenvectors

## Overview

Eigenvalues and eigenvectors are among the most important concepts in linear algebra, revealing the intrinsic structure of linear transformations. When a matrix acts on an eigenvector, it only scales that vector—the direction remains unchanged. This seemingly simple property has profound implications across mathematics, physics, and machine learning.

The eigenvalue equation $A\mathbf{v} = \lambda\mathbf{v}$ asks: for which vectors does multiplication by $A$ reduce to simple scalar multiplication? The answer unlocks matrix powers, stability analysis, dimensionality reduction, and much more.

## Learning Objectives

By the end of this section, you will be able to:

- **Define** eigenvalues and eigenvectors geometrically and algebraically
- **Compute** eigenvalues via the characteristic equation
- **Find** eigenvectors by solving homogeneous systems
- **Apply** eigenvalue properties (trace, determinant, transformations)
- **Diagonalize** matrices and compute matrix powers efficiently
- **Analyze** symmetric and positive definite matrices
- **Implement** the power method for dominant eigenvalue
- **Connect** eigenvalues to PCA, Markov chains, and neural networks

## Prerequisites

- Matrix operations and multiplication
- Determinants
- Systems of linear equations
- Vector spaces and null space

---

## 1. Definition and Intuition

### 1.1 The Eigenvalue Equation

For a square matrix $A \in \mathbb{R}^{n \times n}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** with **eigenvalue** $\lambda$ if:

$$A\mathbf{v} = \lambda\mathbf{v}$$

The matrix $A$ transforms $\mathbf{v}$ by only scaling it by factor $\lambda$.

### 1.2 Geometric Interpretation

```
Regular vector transformation:        Eigenvector transformation:

       Av                                    Av = λv
      ↗                                        ↑
     /                                         |
    /    (rotated and scaled)                  | (same direction,
   v →                                     v → | only scaled by λ)

Matrix A changes both                  Eigenvector keeps its
direction and magnitude               direction, only stretched/shrunk
```

**Key Insight**: Eigenvectors are the "natural directions" of a matrix—the directions along which the transformation acts most simply.

### 1.3 Effect of Different Eigenvalues

| Eigenvalue | Effect on Eigenvector |
|------------|----------------------|
| $\lambda > 1$ | Stretched (scaled up) |
| $0 < \lambda < 1$ | Compressed (scaled down) |
| $\lambda = 1$ | Unchanged |
| $\lambda = 0$ | Collapsed to zero |
| $\lambda < 0$ | Flipped and scaled |
| $\lambda$ complex | Rotation (no real direction preserved) |

### 1.4 Eigenspace

The **eigenspace** for eigenvalue $\lambda$ is:

$$E_\lambda = \{\mathbf{v} : A\mathbf{v} = \lambda\mathbf{v}\} = \text{null}(A - \lambda I)$$

It's the set of all eigenvectors (plus zero) for that eigenvalue.

---

## 2. Finding Eigenvalues

### 2.1 The Characteristic Equation

Starting from $A\mathbf{v} = \lambda\mathbf{v}$:

$$A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For non-zero solutions, $(A - \lambda I)$ must be singular:

$$\boxed{\det(A - \lambda I) = 0}$$

This is the **characteristic equation**. The polynomial $p(\lambda) = \det(A - \lambda I)$ is the **characteristic polynomial**.

### 2.2 Example: 2×2 Matrix

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

**Step 1**: Form $A - \lambda I$

$$A - \lambda I = \begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix}$$

**Step 2**: Compute determinant

$$\det(A - \lambda I) = (4-\lambda)(3-\lambda) - (1)(2)$$
$$= \lambda^2 - 7\lambda + 12 - 2 = \lambda^2 - 7\lambda + 10$$

**Step 3**: Solve characteristic equation

$$\lambda^2 - 7\lambda + 10 = 0$$
$$(\lambda - 5)(\lambda - 2) = 0$$

**Eigenvalues**: $\lambda_1 = 5$, $\lambda_2 = 2$

### 2.3 Shortcut for 2×2 Matrices

For $A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

$$\lambda^2 - \text{tr}(A)\lambda + \det(A) = 0$$
$$\lambda^2 - (a+d)\lambda + (ad-bc) = 0$$

---

## 3. Finding Eigenvectors

For each eigenvalue $\lambda$, solve $(A - \lambda I)\mathbf{v} = \mathbf{0}$ to find the null space.

### 3.1 Continuing the Example

**For $\lambda_1 = 5$:**

$$(A - 5I)\mathbf{v} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

From row 1: $-v_1 + v_2 = 0 \Rightarrow v_1 = v_2$

**Eigenvector**: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ (or any scalar multiple)

**For $\lambda_2 = 2$:**

$$(A - 2I)\mathbf{v} = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

From row 1: $2v_1 + v_2 = 0 \Rightarrow v_2 = -2v_1$

**Eigenvector**: $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$

### 3.2 Verification

Always check: $A\mathbf{v} = \lambda\mathbf{v}$

$$A\mathbf{v}_1 = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 5\begin{bmatrix} 1 \\ 1 \end{bmatrix} = \lambda_1\mathbf{v}_1 \checkmark$$

---

## 4. Fundamental Properties

### 4.1 Trace and Determinant

For matrix $A$ with eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$:

$$\boxed{\text{tr}(A) = \sum_{i=1}^{n} \lambda_i}$$

$$\boxed{\det(A) = \prod_{i=1}^{n} \lambda_i}$$

**Consequences**:
- $\det(A) = 0$ iff at least one $\lambda_i = 0$
- Singular matrices have zero as an eigenvalue

### 4.2 Eigenvalues Under Transformations

| Transformation | Eigenvalues |
|----------------|-------------|
| $A^k$ | $\lambda^k$ |
| $A^{-1}$ | $1/\lambda$ |
| $A + cI$ | $\lambda + c$ |
| $cA$ | $c\lambda$ |
| $A^T$ | Same as $A$ |

### 4.3 Multiplicity

- **Algebraic multiplicity**: Number of times $\lambda$ is a root of characteristic polynomial
- **Geometric multiplicity**: Dimension of eigenspace = $\dim(\text{null}(A - \lambda I))$

**Key fact**: geometric multiplicity ≤ algebraic multiplicity

---

## 5. Special Matrices

### 5.1 Symmetric Matrices ($A = A^T$)

Symmetric matrices have remarkable spectral properties:

1. **All eigenvalues are real**
2. **Eigenvectors for distinct eigenvalues are orthogonal**
3. **Always diagonalizable**: $A = Q\Lambda Q^T$ where $Q$ is orthogonal

```
Symmetric matrix:
       A = QΛQᵀ
       
where:
- Q = [v₁ | v₂ | ... | vₙ] orthonormal eigenvectors
- Λ = diag(λ₁, λ₂, ..., λₙ)
- QᵀQ = QQᵀ = I
```

### 5.2 Positive Definite Matrices

A symmetric matrix $A$ is **positive definite** if:
- All eigenvalues are **positive** ($\lambda_i > 0$)
- Equivalently: $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$

**Properties**:
- $\det(A) > 0$ and all leading principal minors positive
- Cholesky decomposition exists: $A = LL^T$
- All pivots positive in Gaussian elimination

**ML importance**: Covariance matrices are positive semi-definite; Hessians at local minima are positive definite.

### 5.3 Orthogonal Matrices ($Q^TQ = I$)

- All eigenvalues satisfy $|\lambda| = 1$
- Eigenvalues are $e^{i\theta}$ (on the complex unit circle)
- Include rotations and reflections

### 5.4 Triangular Matrices

Eigenvalues are the diagonal entries (characteristic polynomial factors directly).

---

## 6. Diagonalization

### 6.1 Definition

A matrix $A$ is **diagonalizable** if:

$$A = PDP^{-1}$$

where:
- $D = \text{diag}(\lambda_1, \ldots, \lambda_n)$ has eigenvalues on diagonal
- $P = [\mathbf{v}_1 | \cdots | \mathbf{v}_n]$ has eigenvectors as columns

### 6.2 When is a Matrix Diagonalizable?

$A$ is diagonalizable iff it has $n$ linearly independent eigenvectors.

**Sufficient conditions**:
- $n$ distinct eigenvalues (always diagonalizable)
- Symmetric matrix (always diagonalizable)
- Geometric = algebraic multiplicity for each eigenvalue

### 6.3 Computing Matrix Powers

The power of diagonalization: $A^k$ becomes trivial!

$$A^k = PD^kP^{-1}$$

where $D^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$.

**Example**: Computing $A^{100}$ requires only:
1. Find eigenvalues λᵢ
2. Compute λᵢ¹⁰⁰
3. Multiply three matrices

### 6.4 Matrix Exponential

For differential equations and neural networks:

$$e^A = Pe^DP^{-1}$$

where $e^D = \text{diag}(e^{\lambda_1}, \ldots, e^{\lambda_n})$.

---

## 7. Spectral Decomposition

For a symmetric matrix $A$:

$$\boxed{A = \sum_{i=1}^{n} \lambda_i \mathbf{v}_i \mathbf{v}_i^T}$$

where $\mathbf{v}_i$ are orthonormal eigenvectors.

```
Spectral Decomposition:

A = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ + ... + λₙvₙvₙᵀ
    \_____/   \_____/       \_____/
    rank-1    rank-1        rank-1
    projection projection  projection
    matrices  matrices     matrices
```

Each term $\lambda_i \mathbf{v}_i \mathbf{v}_i^T$ is a projection onto the $i$-th eigenvector direction, scaled by $\lambda_i$.

---

## 8. Numerical Methods

### 8.1 Power Method

Finds the **dominant eigenvalue** (largest magnitude) and its eigenvector.

```
Algorithm:
1. Start with random vector v₀
2. Repeat:
   - w = A @ v_k
   - λ ≈ (vᵀAv) / (vᵀv)  [Rayleigh quotient]
   - v_{k+1} = w / ||w||   [normalize]
```

**Convergence**: Linear, rate = |λ₂/λ₁|

**Used in**: PageRank, finding largest singular value

### 8.2 Inverse Power Method

Finds the **smallest eigenvalue** by applying power method to $A^{-1}$.

### 8.3 QR Algorithm

Standard method for computing all eigenvalues:

```
A₀ = A
Repeat:
    Q_k R_k = A_k    (QR decomposition)
    A_{k+1} = R_k Q_k
```

$A_k$ converges to upper triangular (Schur form) with eigenvalues on diagonal.

---

## 9. Applications in Machine Learning

### 9.1 Principal Component Analysis (PCA)

The eigenvectors of the covariance matrix are the principal components:

$$C = \frac{1}{n-1}X^TX$$

- **Eigenvectors** = principal component directions
- **Eigenvalues** = variance along each direction
- Keep eigenvectors with largest eigenvalues for dimensionality reduction

```
PCA: Project data onto top-k eigenvectors

Original data (d dimensions)  →  Reduced data (k dimensions)
      X                              X @ V_k

where V_k = [v₁ | v₂ | ... | v_k] (top k eigenvectors)
```

### 9.2 PageRank Algorithm

The importance scores of web pages come from:

$$\pi = A\pi$$

The stationary distribution $\pi$ is the eigenvector of the web link matrix with $\lambda = 1$.

### 9.3 Spectral Clustering

Uses eigenvectors of the graph Laplacian $L = D - W$:

1. Compute eigenvalues/eigenvectors of $L$
2. Use eigenvectors for small eigenvalues as features
3. Cluster in the eigenspace

Small eigenvalues correspond to connected clusters.

### 9.4 Markov Chains

For transition matrix $P$:
- Stationary distribution: $\pi P = \pi$ (left eigenvector with $\lambda = 1$)
- Convergence rate determined by second largest eigenvalue

### 9.5 Recurrent Neural Networks

Eigenvalues of the recurrent weight matrix determine dynamics:

| |λ| | Effect |
|-----|--------|
| < 1 | Vanishing gradients (information lost) |
| = 1 | Stable (ideal) |
| > 1 | Exploding gradients (unstable) |

LSTM/GRU architectures address this by using gates.

### 9.6 Hessian Eigenvalues in Optimization

At a critical point of loss function:
- **All eigenvalues > 0**: Local minimum
- **All eigenvalues < 0**: Local maximum
- **Mixed signs**: Saddle point
- **Condition number** $\kappa = \lambda_{max}/\lambda_{min}$: affects convergence speed

---

## 10. Complex Eigenvalues

### 10.1 When Do They Occur?

Real matrices can have complex eigenvalues (always in conjugate pairs).

**Example**: Rotation by angle $\theta$

$$R_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

Eigenvalues: $e^{\pm i\theta} = \cos\theta \pm i\sin\theta$

Complex eigenvalues indicate rotational behavior—no real direction is preserved.

### 10.2 Geometric Meaning

- $|λ| = 1$: Pure rotation (no scaling)
- $|λ| < 1$: Spiraling inward
- $|λ| > 1$: Spiraling outward
- $\arg(\lambda) = \theta$: Rotation angle

---

## 11. Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Eigenvalue equation | $A\mathbf{v} = \lambda\mathbf{v}$ |
| Characteristic equation | $\det(A - \lambda I) = 0$ |
| Trace | $\text{tr}(A) = \sum \lambda_i$ |
| Determinant | $\det(A) = \prod \lambda_i$ |
| Diagonalization | $A = PDP^{-1}$ |
| Matrix powers | $A^k = PD^kP^{-1}$ |
| Spectral decomposition | $A = \sum \lambda_i \mathbf{v}_i \mathbf{v}_i^T$ |

### Quick Reference: Finding Eigenvalues/Eigenvectors

```
Step 1: Eigenvalues
   det(A - λI) = 0
   Solve for λ₁, λ₂, ..., λₙ

Step 2: Eigenvectors (for each λᵢ)
   (A - λᵢI)v = 0
   Find null space → eigenvector vᵢ

Step 3: Verify
   Check Avᵢ = λᵢvᵢ
```

### ML Quick Reference

| Application | Eigenvalue/vector Role |
|-------------|------------------------|
| PCA | Eigenvectors = principal components |
| PageRank | Dominant eigenvector = page importance |
| Spectral clustering | Graph Laplacian eigenvectors |
| RNN stability | |λ| determines gradient behavior |
| Optimization | Hessian eigenvalues = curvature info |

---

## 12. Practice Problems

See the accompanying Jupyter notebooks:
- **[theory.ipynb](theory.ipynb)**: Worked examples with visualizations
- **[exercises.ipynb](exercises.ipynb)**: Practice problems with solutions

Key exercises include:
1. Computing eigenvalues and eigenvectors
2. Verifying trace/determinant relationships
3. Matrix diagonalization
4. Symmetric matrix properties
5. Power method implementation
6. Markov chain stationary distributions
7. Positive definite checks
8. Spectral norm and condition number

---

## 13. References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Lay, D. - "Linear Algebra and Its Applications"
3. Axler, S. - "Linear Algebra Done Right"
4. Goodfellow et al. - "Deep Learning" (Chapter 2)
5. 3Blue1Brown - "Essence of Linear Algebra"

---

## Navigation

[← Previous: Linear Algebra Basics](../../02-Linear-Algebra-Basics/README.md) | [Next: Singular Value Decomposition →](../02-Singular-Value-Decomposition/README.md)

[↑ Back to Advanced Linear Algebra](../README.md) | [↑↑ Back to Main](../../README.md)
