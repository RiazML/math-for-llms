# Positive Definite Matrices

## Introduction

Positive definite matrices are the "good" matrices of linear algebra. They arise naturally as covariance matrices, Hessians of convex functions, and kernel matrices in ML. Understanding them unlocks deeper insights into optimization, probabilistic models, and kernel methods.

## Prerequisites

- Eigenvalues and eigenvectors
- Symmetric matrices
- Quadratic forms
- Matrix norms

## Learning Objectives

1. Define positive definiteness multiple ways
2. Recognize positive definite matrices
3. Understand the connection to convexity
4. Apply Cholesky decomposition
5. Use positive definiteness in ML contexts

---

## 1. Definition

### Quadratic Form Definition

A symmetric matrix $A$ is **positive definite (PD)** if:

$$\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq \mathbf{0}$$

**Positive semi-definite (PSD)**: $\mathbf{x}^T A \mathbf{x} \geq 0$

### Notation

- Positive definite: $A \succ 0$
- Positive semi-definite: $A \succeq 0$
- Negative definite: $A \prec 0$
- Negative semi-definite: $A \preceq 0$

### Visualization

```
Positive Definite:         Positive Semi-definite:
f(x) = x^T A x             f(x) = x^T A x

    ^                           ^
    |    /\                     |    ___/
    |   /  \                    |   /
    |  /    \                   |  /
    | /      \                  | /
 ---┼─────────→              ---┼─────────→
    0                           0

(bowl shape,               (bowl, but flat
 unique minimum)            along some direction)
```

---

## 2. Equivalent Conditions

A symmetric matrix $A$ is positive definite if and only if:

| Condition         | Description                                                          |
| ----------------- | -------------------------------------------------------------------- |
| 1. Quadratic form | $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ |
| 2. Eigenvalues    | All eigenvalues $\lambda_i > 0$                                      |
| 3. Determinants   | All leading principal minors $> 0$                                   |
| 4. Cholesky       | $A = LL^T$ exists with positive diagonal                             |
| 5. Decomposition  | $A = B^T B$ for some full-rank $B$                                   |

### Leading Principal Minors (Sylvester's Criterion)

For a 3×3 matrix:
$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$

Check:

1. $a_{11} > 0$
2. $\det\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} > 0$
3. $\det(A) > 0$

---

## 3. Properties

### Fundamental Properties

| Property              | Statement                            |
| --------------------- | ------------------------------------ |
| Invertibility         | PD matrices are always invertible    |
| Inverse               | $A^{-1}$ is also PD                  |
| Sum                   | $A + B$ is PD if both $A, B$ are PD  |
| Scalar multiplication | $cA$ is PD for $c > 0$               |
| Congruence            | $P^T A P$ is PD if $P$ is invertible |

### Spectral Properties

For symmetric PD matrix $A$:

- All eigenvalues are real and positive
- Eigenvectors are orthogonal
- $A = Q \Lambda Q^T$ where $\Lambda$ has positive diagonal

### Square Root

Every PD matrix has a unique PD square root:
$$A = (\sqrt{A})^2$$

where $\sqrt{A} = Q\sqrt{\Lambda}Q^T$

---

## 4. Cholesky Decomposition

### Definition

For PD matrix $A$:
$$A = LL^T$$

where $L$ is lower triangular with positive diagonal.

### Algorithm

$$L = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$$

Compute column by column:
$$l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$$
$$l_{ij} = \frac{1}{l_{jj}}\left(a_{ij} - \sum_{k=1}^{j-1} l_{ik}l_{jk}\right) \quad \text{for } i > j$$

### Why Cholesky?

- **Efficient**: $\frac{n^3}{3}$ operations (half of LU)
- **Stable**: No pivoting needed
- **Unique**: Only one such decomposition

### Solving Linear Systems

For $Ax = b$ where $A$ is PD:

1. Cholesky: $A = LL^T$
2. Forward solve: $Ly = b$
3. Back solve: $L^T x = y$

---

## 5. Connection to Covariance Matrices

### Covariance Matrix

For random vector $\mathbf{X}$:
$$\Sigma = \mathbb{E}[(\mathbf{X} - \mu)(\mathbf{X} - \mu)^T]$$

### Key Property

**Covariance matrices are always positive semi-definite!**

Proof: For any $\mathbf{a}$:
$$\mathbf{a}^T \Sigma \mathbf{a} = \text{Var}(\mathbf{a}^T \mathbf{X}) \geq 0$$

### When is it PD (not just PSD)?

When no linear combination of variables is constant (no perfect multicollinearity).

---

## 6. Connection to Convexity

### Hessian Test

A twice-differentiable function $f$ is:

- **Strictly convex** if $\nabla^2 f(\mathbf{x}) \succ 0$ everywhere
- **Convex** if $\nabla^2 f(\mathbf{x}) \succeq 0$ everywhere

### Quadratic Functions

For $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} - \mathbf{b}^T\mathbf{x} + c$:

$$\nabla^2 f = A$$

- $A \succ 0$: Unique global minimum at $\mathbf{x}^* = A^{-1}\mathbf{b}$
- $A \succeq 0$: Minimum may not be unique
- $A$ indefinite: Saddle point

```
f(x) = 0.5 x^T A x

A positive definite:    A indefinite:
      __                    ___
     /  \                  /   \
    |    |                /     \_
    |    |               |       |
   _/    \_              |       |

  (bowl)              (saddle)
```

---

## 7. Tests for Positive Definiteness

### Quick Tests

```python
def is_positive_definite(A):
    """Multiple ways to test positive definiteness."""

    # Test 1: Eigenvalues
    eigenvalues = np.linalg.eigvalsh(A)  # Use eigvalsh for symmetric
    if all(eigenvalues > 0):
        return True

    # Test 2: Cholesky (throws error if not PD)
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
```

### 2×2 Special Case

$$A = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$$

PD iff: $a > 0$ AND $ac - b^2 > 0$

---

## 8. Applications in ML/AI

### 1. Gaussian Distributions

Multivariate Gaussian:
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right)$$

Requires $\Sigma \succ 0$ for valid probability!

### 2. Kernel Functions

A kernel $K(\mathbf{x}, \mathbf{y})$ is valid (Mercer) if:

- Kernel matrix $K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j)$ is PSD for all datasets

Common kernels:

- Linear: $K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T\mathbf{y}$
- RBF: $K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x} - \mathbf{y}\|^2)$
- Polynomial: $K(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^T\mathbf{y} + c)^d$

### 3. Regularization

Ridge regression adds $\lambda I$ to make normal equations PD:
$$(X^TX + \lambda I)\mathbf{w} = X^T\mathbf{y}$$

$X^TX + \lambda I \succ 0$ for $\lambda > 0$!

### 4. Newton's Method

$$\mathbf{x}_{k+1} = \mathbf{x}_k - [\nabla^2 f(\mathbf{x}_k)]^{-1} \nabla f(\mathbf{x}_k)$$

Requires PD Hessian for descent direction.

### 5. Natural Gradient

Fisher information matrix is PSD:
$$F = \mathbb{E}\left[\nabla \log p(x|\theta) \nabla \log p(x|\theta)^T\right]$$

### 6. Mahalanobis Distance

$$d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$$

Valid distance requires $\Sigma \succ 0$.

---

## 9. Near-PD Matrices

### Problem

In practice, estimated covariance matrices may not be PD due to:

- Numerical errors
- More variables than samples
- Regularization needed

### Solutions

1. **Add small diagonal**: $A + \epsilon I$
2. **Project eigenvalues**: Set negative eigenvalues to small positive value
3. **Shrinkage**: $\alpha A + (1-\alpha)I$

```python
def nearest_pd(A, epsilon=1e-8):
    """Find nearest positive definite matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, epsilon)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
```

---

## 10. Summary

### Key Points

| Concept       | Meaning                      |
| ------------- | ---------------------------- |
| $A \succ 0$   | All eigenvalues positive     |
| $A \succeq 0$ | All eigenvalues non-negative |
| Cholesky      | $A = LL^T$, unique for PD    |
| Covariance    | Always PSD                   |
| Convexity     | PD Hessian → strictly convex |

### Quick Tests

```
Is A positive definite?

1. Check symmetry: A = A^T
2. Check eigenvalues: all λ > 0
3. Try Cholesky: no error
4. For 2×2: a > 0 and det > 0
```

### ML Connections

```
Positive Definiteness in ML:
│
├── Covariance matrices (PSD)
│   └── Gaussian distributions
│
├── Kernel matrices (PSD)
│   └── SVM, GP, kernel methods
│
├── Hessians (PD → convex)
│   └── Optimization, Newton's method
│
└── Regularization (makes PD)
    └── Ridge regression, stability
```

---

## Exercises

1. Show that $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ is positive definite
2. Find the Cholesky decomposition of $\begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$
3. Prove that $A^T A$ is always positive semi-definite
4. Show that adding $\lambda I$ makes any symmetric matrix PD for large enough $\lambda$
5. Verify that the RBF kernel matrix is PSD

---

## References

1. Strang, G. - "Linear Algebra and Its Applications"
2. Boyd & Vandenberghe - "Convex Optimization"
3. Bishop - "Pattern Recognition and Machine Learning"
