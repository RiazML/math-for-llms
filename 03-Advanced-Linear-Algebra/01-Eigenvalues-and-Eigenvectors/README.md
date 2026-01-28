# Eigenvalues and Eigenvectors

## Introduction

Eigenvalues and eigenvectors are fundamental to understanding linear transformations. They reveal the intrinsic properties of matrices and are central to PCA, spectral clustering, Google's PageRank, quantum mechanics, and countless ML algorithms.

## Prerequisites
- Matrix operations and multiplication
- Determinants
- Systems of linear equations
- Vector spaces

## Learning Objectives
1. Understand the geometric meaning of eigenvectors
2. Compute eigenvalues and eigenvectors
3. Apply the characteristic equation
4. Understand diagonalization
5. Recognize applications in ML

---

## 1. Definition and Intuition

### Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** if:

$$A\mathbf{v} = \lambda\mathbf{v}$$

where $\lambda$ is the corresponding **eigenvalue**.

### Geometric Interpretation

```
Regular vector:                  Eigenvector:
     
     Av                              Av = λv
      ↗                                ↑
     /                                 |
    /                                  | (same direction,
   v →                             v → | scaled by λ)
                                       
Matrix A rotates and scales      Eigenvector only scaled,
most vectors                     direction preserved!
```

**Key Insight**: Eigenvectors are special directions that remain unchanged (except for scaling) when transformed by the matrix.

### Examples of λ values

| λ value | Effect on eigenvector |
|---------|----------------------|
| λ > 1 | Stretched |
| 0 < λ < 1 | Compressed |
| λ = 1 | Unchanged |
| λ = 0 | Collapsed to origin |
| λ < 0 | Flipped and scaled |
| λ complex | Rotation involved |

---

## 2. Finding Eigenvalues

### Characteristic Equation

From $A\mathbf{v} = \lambda\mathbf{v}$:

$$A\mathbf{v} - \lambda\mathbf{v} = \mathbf{0}$$
$$(A - \lambda I)\mathbf{v} = \mathbf{0}$$

For non-trivial solutions, the matrix $(A - \lambda I)$ must be singular:

$$\det(A - \lambda I) = 0$$

This is the **characteristic equation**, and the polynomial is the **characteristic polynomial**.

### Example: 2×2 Matrix

$$A = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$$

**Step 1**: Form $A - \lambda I$

$$A - \lambda I = \begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix}$$

**Step 2**: Set determinant to zero

$$\det(A - \lambda I) = (4-\lambda)(3-\lambda) - 2 = 0$$
$$\lambda^2 - 7\lambda + 10 = 0$$
$$(\lambda - 5)(\lambda - 2) = 0$$

**Eigenvalues**: $\lambda_1 = 5$, $\lambda_2 = 2$

---

## 3. Finding Eigenvectors

For each eigenvalue $\lambda$, solve $(A - \lambda I)\mathbf{v} = \mathbf{0}$

### Continuing the Example

**For $\lambda_1 = 5$:**

$$(A - 5I)\mathbf{v} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

From row 1: $-v_1 + v_2 = 0 \Rightarrow v_1 = v_2$

Eigenvector: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ (or any scalar multiple)

**For $\lambda_2 = 2$:**

$$(A - 2I)\mathbf{v} = \begin{bmatrix} 2 & 1 \\ 2 & 1 \end{bmatrix}\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}$$

From row 1: $2v_1 + v_2 = 0 \Rightarrow v_2 = -2v_1$

Eigenvector: $\mathbf{v}_2 = \begin{bmatrix} 1 \\ -2 \end{bmatrix}$

---

## 4. Properties of Eigenvalues

### Trace and Determinant

For matrix $A$ with eigenvalues $\lambda_1, \lambda_2, \ldots, \lambda_n$:

$$\text{tr}(A) = \sum_{i=1}^{n} \lambda_i$$

$$\det(A) = \prod_{i=1}^{n} \lambda_i$$

### Other Properties

| Property | Formula |
|----------|---------|
| Eigenvalues of $A^k$ | $\lambda^k$ |
| Eigenvalues of $A^{-1}$ | $1/\lambda$ |
| Eigenvalues of $A + cI$ | $\lambda + c$ |
| Eigenvalues of $cA$ | $c\lambda$ |

### Algebraic vs Geometric Multiplicity

- **Algebraic multiplicity**: How many times λ is a root of the characteristic polynomial
- **Geometric multiplicity**: Dimension of the eigenspace (number of linearly independent eigenvectors)

Always: geometric ≤ algebraic

---

## 5. Special Matrices

### Symmetric Matrices ($A = A^T$)

1. All eigenvalues are **real**
2. Eigenvectors for distinct eigenvalues are **orthogonal**
3. Always **diagonalizable**

$$A = Q\Lambda Q^T$$

where $Q$ is orthogonal ($Q^T = Q^{-1}$)

### Positive Definite Matrices

1. All eigenvalues are **positive**
2. $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$

### Orthogonal Matrices ($Q^T Q = I$)

1. All eigenvalues have $|\lambda| = 1$
2. Eigenvalues are $e^{i\theta}$ (on unit circle)

---

## 6. Diagonalization

### Definition

A matrix $A$ is **diagonalizable** if:

$$A = PDP^{-1}$$

where:
- $D$ is diagonal with eigenvalues on diagonal
- $P$ has eigenvectors as columns

### Conditions for Diagonalizability

A matrix is diagonalizable if and only if it has $n$ linearly independent eigenvectors.

### Computing Powers

If $A = PDP^{-1}$:

$$A^k = PD^kP^{-1}$$

where $D^k$ is easy to compute (just raise diagonal entries to power $k$).

```
Example: A = PDP⁻¹

A² = (PDP⁻¹)(PDP⁻¹) = PD(P⁻¹P)DP⁻¹ = PD²P⁻¹

A^k = PD^kP⁻¹ where D^k = diag(λ₁^k, λ₂^k, ..., λₙ^k)
```

---

## 7. Spectral Decomposition

For a symmetric matrix $A$:

$$A = \sum_{i=1}^{n} \lambda_i \mathbf{v}_i \mathbf{v}_i^T$$

where $\mathbf{v}_i$ are orthonormal eigenvectors.

```
Spectral Decomposition:

A = λ₁v₁v₁ᵀ + λ₂v₂v₂ᵀ + ... + λₙvₙvₙᵀ
    \_____/   \_____/       \_____/
    rank-1    rank-1        rank-1
    matrices  matrices      matrices
```

This decomposes A into a sum of rank-1 matrices, each along an eigenvector direction.

---

## 8. Applications in ML/AI

### 1. Principal Component Analysis (PCA)

Eigenvectors of covariance matrix are principal components:

$$C = \frac{1}{n}X^TX$$

Largest eigenvalues → directions of maximum variance

### 2. PageRank Algorithm

Find eigenvector of web link matrix:

$$\pi = A\pi$$

The stationary distribution $\pi$ is the eigenvector with $\lambda = 1$.

### 3. Spectral Clustering

Use eigenvectors of graph Laplacian:

$$L = D - W$$

Small eigenvalues correspond to cluster structure.

### 4. Markov Chains

Stationary distribution is eigenvector with $\lambda = 1$:

$$\pi P = \pi$$

### 5. Recurrent Neural Networks

Eigenvalues determine stability:
- $|\lambda| < 1$: vanishing gradients
- $|\lambda| > 1$: exploding gradients

### 6. Covariance Matrix Analysis

```
Eigenvalues of Σ:
- Represent variance along principal axes
- Large eigenvalue → high variance direction
- Small eigenvalue → low variance direction

For dimensionality reduction:
Keep eigenvectors with largest eigenvalues
```

---

## 9. Numerical Computation

### Power Method

Find dominant eigenvector iteratively:

```
v₀ = random vector
Repeat:
    w = Av_{k}
    v_{k+1} = w / ||w||
```

Converges to eigenvector with largest |λ|.

### QR Algorithm

Standard method for all eigenvalues:

```
A₀ = A
Repeat:
    Q_k R_k = A_k  (QR decomposition)
    A_{k+1} = R_k Q_k
```

$A_k$ converges to upper triangular with eigenvalues on diagonal.

---

## 10. Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Definition | $A\mathbf{v} = \lambda\mathbf{v}$ |
| Characteristic equation | $\det(A - \lambda I) = 0$ |
| Trace | $\text{tr}(A) = \sum \lambda_i$ |
| Determinant | $\det(A) = \prod \lambda_i$ |
| Diagonalization | $A = PDP^{-1}$ |
| Powers | $A^k = PD^kP^{-1}$ |
| Spectral decomposition | $A = \sum \lambda_i \mathbf{v}_i \mathbf{v}_i^T$ |

### Quick Reference

```
Finding eigenvalues:
1. Form A - λI
2. Solve det(A - λI) = 0 for λ

Finding eigenvectors:
1. For each λ, solve (A - λI)v = 0
2. Find null space of (A - λI)
```

---

## Exercises

1. Find eigenvalues and eigenvectors of $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$
2. Verify that $\text{tr}(A) = \lambda_1 + \lambda_2$ and $\det(A) = \lambda_1 \lambda_2$
3. Compute $A^{10}$ using diagonalization
4. Show that eigenvectors for distinct eigenvalues of a symmetric matrix are orthogonal
5. Find the dominant eigenvalue using the power method

---

## References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Lay, D. - "Linear Algebra and Its Applications"
3. Goodfellow et al. - "Deep Learning" (Chapter 2)
