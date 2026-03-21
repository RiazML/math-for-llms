[← Back to Advanced Linear Algebra](../README.md) | [← Matrix Norms](../06-Matrix-Norms/notes.md) | [Matrix Decompositions →](../08-Matrix-Decompositions/notes.md)

---

# Positive Definite Matrices

## Introduction

Positive definite matrices are among the most important structures in applied mathematics and machine learning. They are the "good" matrices—matrices that guarantee well-behaved optimization problems, valid probability distributions, and stable numerical algorithms.

When you see a positive definite (PD) matrix, you can trust it:
- Its eigenvalues are positive, so it's invertible
- It defines a proper inner product and distance metric
- Quadratic forms with PD matrices have unique minima
- Cholesky decomposition provides efficient, stable factorization

In machine learning, PD matrices appear everywhere:
- **Covariance matrices** in multivariate Gaussians
- **Kernel matrices** in SVMs and Gaussian processes
- **Hessian matrices** that prove convexity
- **Precision matrices** in graphical models
- **Fisher information** in natural gradient methods

This section provides a comprehensive treatment of positive definiteness with emphasis on recognition, computation, and application.

## Prerequisites

- Eigenvalues and eigenvectors (computing and interpreting)
- Symmetric matrices and spectral theorem
- Quadratic forms ($\mathbf{x}^T A \mathbf{x}$)
- Matrix norms and condition numbers
- Basic calculus (gradients, Hessians)

## Learning Objectives

1. Define positive definiteness through multiple equivalent conditions
2. Recognize and test matrices for positive definiteness
3. Master Cholesky decomposition and its applications
4. Understand the deep connection between PD matrices and convexity
5. Apply PD matrix theory to ML problems (kernels, covariances, optimization)
6. Handle near-PD matrices and regularization techniques

---

## 1. Definition

### Quadratic Form Definition

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is **positive definite (PD)** if:

$$\mathbf{x}^T A \mathbf{x} > 0 \quad \text{for all } \mathbf{x} \neq \mathbf{0}$$

This is the fundamental definition—the quadratic form $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ is strictly positive except at the origin.

### Classification Hierarchy

| Type | Notation | Condition | Quadratic Form |
|------|----------|-----------|----------------|
| Positive definite | $A \succ 0$ | $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$ | Bowl (minimum at origin) |
| Positive semi-definite | $A \succeq 0$ | $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$ | Bowl or flat valley |
| Negative definite | $A \prec 0$ | $\mathbf{x}^T A \mathbf{x} < 0$ for all $\mathbf{x} \neq 0$ | Inverted bowl (maximum) |
| Negative semi-definite | $A \preceq 0$ | $\mathbf{x}^T A \mathbf{x} \leq 0$ for all $\mathbf{x}$ | Inverted bowl or ridge |
| Indefinite | (none) | Takes both positive and negative values | Saddle shape |

### Geometric Interpretation

The quadratic form $f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x}$ defines a surface in $(n+1)$-dimensional space.

```
Positive Definite:         Positive Semi-definite:       Indefinite:
f(x) = x^T A x             f(x) = x^T A x                f(x) = x^T A x

    ^                           ^                             /\
    |    /\                     |    ___/                    /  \__
    |   /  \                    |   /                       /      \_
    |  /    \                   |  /                    ___/        \___
    | /      \                  | /                         saddle
 ---┼─────────→              ---┼─────────→              ───┼───────────→
    0                           0                           0

(bowl shape,               (flat along null             (goes up in some
 unique minimum)            space direction)             directions, down in others)
```

### Why Symmetry Matters

The definition requires $A$ to be symmetric ($A = A^T$). For non-symmetric matrices:

$$\mathbf{x}^T A \mathbf{x} = \mathbf{x}^T \left(\frac{A + A^T}{2}\right) \mathbf{x}$$

The antisymmetric part $(A - A^T)/2$ contributes zero to the quadratic form. Thus, we analyze the symmetric part $A_s = (A + A^T)/2$.

---

## 2. Equivalent Conditions

A symmetric matrix $A \in \mathbb{R}^{n \times n}$ is positive definite if and only if any of these equivalent conditions hold:

| Condition         | Statement | Practical Use |
| ----------------- | --------- | ------------- |
| 1. Quadratic form | $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$ | Theoretical definition |
| 2. Eigenvalues    | All eigenvalues $\lambda_i > 0$ | Easy computational test |
| 3. Sylvester's criterion | All leading principal minors $> 0$ | Analytical, small matrices |
| 4. Cholesky | $A = LL^T$ exists with positive diagonal | Practical test + decomposition |
| 5. Gram representation | $A = B^T B$ for some full-rank $B$ | Constructive characterization |
| 6. Pivots | All pivots in Gaussian elimination are $> 0$ | LU-based test |

### Eigenvalue Criterion (Most Common Test)

**Theorem**: $A \succ 0$ iff all eigenvalues are positive.

**Proof**:
- ($\Rightarrow$) If $Av = \lambda v$, then $v^T A v = \lambda \|v\|^2 > 0$, so $\lambda > 0$.
- ($\Leftarrow$) Write $A = Q\Lambda Q^T$ (spectral theorem). Then $x^T A x = y^T \Lambda y = \sum_i \lambda_i y_i^2 > 0$ for $y = Q^T x \neq 0$.

**Python test**: Use `np.linalg.eigvalsh(A)` for symmetric matrices (more efficient and stable than `eig`).

```python
def is_pd_eigenvalue(A, tol=1e-10):
    eigenvalues = np.linalg.eigvalsh(A)
    return np.all(eigenvalues > tol)
```

### Sylvester's Criterion (Leading Principal Minors)

The $k$-th leading principal minor is the determinant of the top-left $k \times k$ submatrix.

For a 3×3 matrix:
$$A = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$$

Check these minors:

1. $M_1 = a_{11} > 0$
2. $M_2 = \det\begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} = a_{11}a_{22} - a_{12}a_{21} > 0$
3. $M_3 = \det(A) > 0$

**Example**: Is $A = \begin{bmatrix} 4 & 2 & 1 \\ 2 & 5 & 2 \\ 1 & 2 & 6 \end{bmatrix}$ positive definite?

- $M_1 = 4 > 0$ ✓
- $M_2 = 4(5) - 2(2) = 16 > 0$ ✓  
- $M_3 = \det(A) = 67 > 0$ ✓

All minors positive → $A$ is PD.

### Cholesky Existence Test

**Theorem**: $A \succ 0$ iff Cholesky decomposition $A = LL^T$ exists.

This provides both a test and a useful factorization:

```python
def is_pd_cholesky(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
```

### 2×2 Special Case

For $A = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$:

$A \succ 0$ iff:
1. $a > 0$ (top-left element positive)
2. $\det(A) = ac - b^2 > 0$ (determinant positive)

This is equivalent to requiring both eigenvalues positive.

---

## 3. Properties

### Algebraic Properties

| Property | Statement | Proof Sketch |
|----------|-----------|--------------|
| Invertibility | PD matrices are always invertible | All eigenvalues $> 0$, so $\det \neq 0$ |
| Inverse | $A^{-1}$ is also PD | If $\lambda > 0$ is eigenvalue of $A$, then $1/\lambda > 0$ for $A^{-1}$ |
| Sum | $A + B$ is PD if both $A, B$ are PD | $x^T(A+B)x = x^TAx + x^TBx > 0$ |
| Scalar multiplication | $cA$ is PD for $c > 0$ | Eigenvalues scale by $c$ |
| Congruence | $P^T A P$ is PD for invertible $P$ | $x^T(P^TAP)x = (Px)^TA(Px) > 0$ since $Px \neq 0$ |
| Product | $A^{1/2} B A^{1/2}$ is PD if $B$ is PD | Similarity to $B$ preserves eigenvalue signs |

### Important Inequalities

For PD matrix $A$ with eigenvalues $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$:

1. **Rayleigh quotient**:
   $$\lambda_1 \leq \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}} \leq \lambda_n$$

2. **Determinant bound**:
   $$\det(A) = \prod_{i=1}^n \lambda_i > 0$$

3. **Trace bound**:
   $$\text{tr}(A) = \sum_{i=1}^n \lambda_i = \sum_{i=1}^n a_{ii} > 0$$

### Spectral Properties

For symmetric PD matrix $A$:

- All eigenvalues are real and positive: $0 < \lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$
- Eigenvectors are orthogonal: $Q^TQ = I$
- Spectral decomposition: $A = Q \Lambda Q^T$ where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$
- Condition number: $\kappa(A) = \lambda_n / \lambda_1$ (always $\geq 1$ for PD)

### Matrix Square Root

Every PD matrix has a unique PD square root:

$$\sqrt{A} = Q\sqrt{\Lambda}Q^T = Q \, \text{diag}(\sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n}) \, Q^T$$

Properties:
- $(\sqrt{A})^2 = A$
- $\sqrt{A}$ is also symmetric and PD
- $\sqrt{A}$ commutes with $A$

### Ordering of PD Matrices (Loewner Order)

We say $A \succeq B$ (Loewner order) if $A - B \succeq 0$ (is PSD).

This creates a partial ordering on symmetric matrices:
- $A \succeq 0$ means $A$ is PSD
- $A \succ B$ means $A - B$ is PD

**Example**: $2I \succ I \succ 0$

---

## 4. Cholesky Decomposition

Cholesky decomposition is one of the most important algorithms for PD matrices—it's efficient, stable, and uniquely exists for any PD matrix.

### Definition

For PD matrix $A$:
$$A = LL^T$$

where $L$ is **lower triangular** with **positive diagonal entries**.

Alternatively: $A = U^T U$ where $U = L^T$ is upper triangular.

### Algorithm

For an $n \times n$ matrix, compute column by column:

$$L = \begin{bmatrix} l_{11} & 0 & \cdots & 0 \\ l_{21} & l_{22} & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ l_{n1} & l_{n2} & \cdots & l_{nn} \end{bmatrix}$$

**Diagonal entries**:
$$l_{jj} = \sqrt{a_{jj} - \sum_{k=1}^{j-1} l_{jk}^2}$$

**Off-diagonal entries** (for $i > j$):
$$l_{ij} = \frac{1}{l_{jj}}\left(a_{ij} - \sum_{k=1}^{j-1} l_{ik}l_{jk}\right)$$

### Worked Example

Find the Cholesky decomposition of $A = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$:

**Step 1**: $l_{11} = \sqrt{a_{11}} = \sqrt{4} = 2$

**Step 2**: $l_{21} = \frac{a_{21}}{l_{11}} = \frac{2}{2} = 1$

**Step 3**: $l_{22} = \sqrt{a_{22} - l_{21}^2} = \sqrt{5 - 1} = 2$

**Result**: $L = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}$

**Verify**: $LL^T = \begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}\begin{bmatrix} 2 & 1 \\ 0 & 2 \end{bmatrix} = \begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix} = A$ ✓

### Computational Advantages

| Aspect | Cholesky | LU Decomposition |
|--------|----------|------------------|
| Operations | $\frac{n^3}{3}$ | $\frac{2n^3}{3}$ |
| Stability | Inherently stable | May need pivoting |
| Uniqueness | Unique $L$ | Not unique |
| Storage | Only $L$ needed | Need both $L$ and $U$ |
| Applicability | PD matrices only | General matrices |

### Solving Linear Systems with Cholesky

For $A\mathbf{x} = \mathbf{b}$ where $A$ is PD:

1. **Decompose**: $A = LL^T$ (once)
2. **Forward substitution**: Solve $L\mathbf{y} = \mathbf{b}$ for $\mathbf{y}$
3. **Back substitution**: Solve $L^T\mathbf{x} = \mathbf{y}$ for $\mathbf{x}$

```python
# Efficient solution using Cholesky
L = np.linalg.cholesky(A)
y = np.linalg.solve(L, b)         # Forward
x = np.linalg.solve(L.T, y)       # Backward
```

This is twice as fast as general LU-based solving!

### Cholesky Failure ⟹ Not PD

If the algorithm encounters a negative value under a square root, the matrix is not PD:

```python
try:
    L = np.linalg.cholesky(A)
    print("A is positive definite")
except np.linalg.LinAlgError:
    print("A is NOT positive definite")
```

### Incomplete Cholesky

For sparse large matrices, **incomplete Cholesky** computes an approximate factorization by dropping small off-diagonal elements—useful as a preconditioner in iterative methods.

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

One of the most profound applications of positive definiteness is in characterizing convexity—the foundation of optimization.

### Hessian Test for Convexity

For a twice-differentiable function $f: \mathbb{R}^n \to \mathbb{R}$:

| Hessian Condition | Function Property |
|-------------------|-------------------|
| $\nabla^2 f(\mathbf{x}) \succ 0$ for all $\mathbf{x}$ | Strictly convex |
| $\nabla^2 f(\mathbf{x}) \succeq 0$ for all $\mathbf{x}$ | Convex |
| $\nabla^2 f(\mathbf{x}) \prec 0$ for all $\mathbf{x}$ | Strictly concave |
| $\nabla^2 f(\mathbf{x})$ indefinite | Saddle point exists |

### Quadratic Functions

For the quadratic function:
$$f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} - \mathbf{b}^T\mathbf{x} + c$$

The gradient and Hessian are:
$$\nabla f = A\mathbf{x} - \mathbf{b}, \quad \nabla^2 f = A$$

**Critical point**: Set $\nabla f = 0$:
$$A\mathbf{x}^* = \mathbf{b} \implies \mathbf{x}^* = A^{-1}\mathbf{b}$$

**Classification at critical point**:
- $A \succ 0$: Unique global minimum at $\mathbf{x}^* = A^{-1}\mathbf{b}$
- $A \succeq 0$ (singular): Minimum may not exist or be unique
- $A \prec 0$: Unique global maximum
- $A$ indefinite: Saddle point

### Geometric Interpretation

```
                A positive definite:              A indefinite:
                        
                         /\                            /\
                        /  \                       ___/  \___
                       /    \                     /          \
               _______/      \_______           /            \
              /        bowl          \         |   saddle     |
                                                \            /
                  unique minimum                 \_        _/
                                                   \      /
```

### Second-Order Optimality Conditions

For minimizing $f(\mathbf{x})$:

**Necessary conditions** (at minimum $\mathbf{x}^*$):
1. $\nabla f(\mathbf{x}^*) = \mathbf{0}$ (first-order)
2. $\nabla^2 f(\mathbf{x}^*) \succeq 0$ (second-order)

**Sufficient conditions** (for strict local minimum):
1. $\nabla f(\mathbf{x}^*) = \mathbf{0}$
2. $\nabla^2 f(\mathbf{x}^*) \succ 0$

### Connection to Loss Functions

In deep learning, the loss landscape $\mathcal{L}(\theta)$ typically has:
- PD Hessian near a sharp minimum (fast convergence, may overfit)
- PSD Hessian near a flat minimum (slower convergence, may generalize better)
- Indefinite Hessian at saddle points (common in high dimensions)

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

Positive definite matrices are fundamental to modern machine learning. Here we explore the key applications.

### 1. Gaussian Distributions

The multivariate Gaussian distribution requires a PD covariance matrix:

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right)$$

**Why $\Sigma \succ 0$ is required**:
- $|\Sigma| > 0$ (normalization constant is finite)
- $\Sigma^{-1}$ exists (precision matrix is well-defined)
- Exponent is always negative (probability $< 1$)
- The quadratic form defines proper Mahalanobis distance

**Precision matrix**: $\Lambda = \Sigma^{-1}$ is also PD and appears in:
- Graphical models (zeros indicate conditional independence)
- Natural parameters of the exponential family

### 2. Kernel Methods

A kernel $K(\mathbf{x}, \mathbf{y})$ is valid (Mercer kernel) if and only if the kernel matrix (Gram matrix) is PSD for any dataset:

$$K_{ij} = K(\mathbf{x}_i, \mathbf{x}_j) \succeq 0$$

**Common valid kernels**:

| Kernel | Formula | Properties |
|--------|---------|------------|
| Linear | $K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^T\mathbf{y}$ | PSD, $K = X^TX$ |
| Polynomial | $K(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^T\mathbf{y} + c)^d$ | PSD for $c \geq 0$ |
| RBF/Gaussian | $K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x} - \mathbf{y}\|^2)$ | Always PD |
| Laplacian | $K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x} - \mathbf{y}\|_1)$ | PSD |

The PSD property guarantees:
- Unique solution in kernel ridge regression
- Convergence in SVM optimization
- Valid covariance in Gaussian processes

### 3. Regularization and Ridge Regression

The normal equations in linear regression:
$$(X^TX)\mathbf{w} = X^T\mathbf{y}$$

$X^TX$ is PSD but may be singular (rank deficient). Ridge regularization adds $\lambda I$:
$$(X^TX + \lambda I)\mathbf{w} = X^T\mathbf{y}$$

For $\lambda > 0$:
- $X^TX + \lambda I \succ 0$ (guaranteed PD!)
- Unique solution: $\mathbf{w} = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$
- Better conditioning: $\kappa(X^TX + \lambda I) \leq \kappa(X^TX)$

This is why ridge regression "stabilizes" ill-conditioned problems.

### 4. Newton's Method and Second-Order Optimization

Newton's method for minimizing $f(\theta)$:
$$\theta_{k+1} = \theta_k - [\nabla^2 f(\theta_k)]^{-1} \nabla f(\theta_k)$$

**Requires PD Hessian** for:
- Descent direction: $-H^{-1}g$ must point downhill
- Well-defined update: $H$ must be invertible

When Hessian is not PD:
- **Levenberg-Marquardt**: Use $(H + \lambda I)^{-1}$ instead
- **Trust region**: Solve constrained subproblem
- **BFGS**: Maintain PD approximation to Hessian

### 5. Fisher Information and Natural Gradient

The Fisher information matrix:
$$F = \mathbb{E}\left[\nabla \log p(x|\theta) \nabla \log p(x|\theta)^T\right] = -\mathbb{E}\left[\nabla^2 \log p(x|\theta)\right]$$

Properties:
- $F \succeq 0$ (always PSD)
- $F \succ 0$ when parameters are identifiable

The natural gradient uses $F^{-1}$:
$$\theta_{k+1} = \theta_k - \eta F^{-1} \nabla \mathcal{L}(\theta_k)$$

This adapts the learning rate to the geometry of the parameter space (used in TRPO, natural policy gradient).

### 6. Mahalanobis Distance

The Mahalanobis distance accounts for correlations between features:

$$d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T\Sigma^{-1}(\mathbf{x}-\mathbf{y})}$$

**Requires $\Sigma \succ 0$** to be a valid distance metric.

Properties:
- Reduces to Euclidean distance when $\Sigma = I$
- "Stretches" space according to covariance
- Used in anomaly detection, clustering, metric learning

### 7. Positive Definite Embeddings

In metric learning and contrastive learning:
- Learn a PD matrix $M$ (or its Cholesky factor $L$)
- Define distance: $d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x}-\mathbf{y})^T M (\mathbf{x}-\mathbf{y})}$
- Equivalent to learning: $d(\mathbf{x}, \mathbf{y}) = \|L\mathbf{x} - L\mathbf{y}\|_2$

### 8. Graph Laplacian

The graph Laplacian $L = D - A$ (where $D$ is degree matrix, $A$ is adjacency):
- Always PSD: $\mathbf{x}^T L \mathbf{x} = \sum_{(i,j) \in E} (x_i - x_j)^2 \geq 0$
- Eigenvalue 0 with multiplicity = number of connected components
- Foundation of spectral clustering and GNNs

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

### Essential Tests for Positive Definiteness

| Test | Method | Best For |
|------|--------|----------|
| Eigenvalues | `all(eigvalsh(A) > 0)` | General, reliable |
| Cholesky | `cholesky(A)` succeeds | Practical, also gives factorization |
| Sylvester | All leading minors $> 0$ | Small matrices, analytical |
| 2×2 | $a_{11} > 0$ and $\det(A) > 0$ | Special case |

### Key Properties to Remember

$$\text{PD Matrix Properties}$$

| Property | Statement |
|----------|-----------|
| $A \succ 0$ | All eigenvalues positive |
| Invertible | Always (since $\det > 0$) |
| $A^{-1} \succ 0$ | Inverse also PD |
| $A + B \succ 0$ | Sum of PD is PD |
| $P^TAP \succ 0$ | Congruence preserves PD |
| $A = LL^T$ | Unique Cholesky exists |
| $\sqrt{A} \succ 0$ | Unique PD square root |

### Quick Tests in Python

```python
import numpy as np

def classify_matrix(A, tol=1e-10):
    """Classify a symmetric matrix."""
    # Ensure symmetry
    A = (A + A.T) / 2
    
    eigenvalues = np.linalg.eigvalsh(A)
    
    if all(eigenvalues > tol):
        return "Positive Definite"
    elif all(eigenvalues >= -tol):
        return "Positive Semi-Definite"
    elif all(eigenvalues < -tol):
        return "Negative Definite"
    elif all(eigenvalues <= tol):
        return "Negative Semi-Definite"
    else:
        return "Indefinite"
```

### ML Applications Summary

```
Positive Definiteness in Machine Learning:
│
├── Probability & Statistics
│   ├── Covariance matrices (must be PSD)
│   ├── Precision matrices (inverse covariance)
│   └── Fisher information (PSD)
│
├── Kernel Methods
│   ├── Gram matrices (PSD by Mercer's theorem)
│   ├── SVM (kernel matrix PSD)
│   └── Gaussian Processes (kernel = covariance)
│
├── Optimization
│   ├── Convexity (PD Hessian → strictly convex)
│   ├── Newton's method (requires PD Hessian)
│   └── Trust region methods (modify to PD)
│
├── Regularization
│   ├── Ridge regression (A + λI → PD)
│   ├── Tikhonov regularization
│   └── Condition number improvement
│
└── Distance & Similarity
    ├── Mahalanobis distance (uses Σ⁻¹)
    ├── Metric learning (learn PD matrix)
    └── Graph Laplacian (PSD)
```

### Common Pitfalls and Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| Numerical non-PD | Finite precision | Add small $\epsilon I$ |
| Singular covariance | $n < p$ (more features than samples) | Regularize or reduce dimension |
| Cholesky fails | Matrix not PD | Check eigenvalues, regularize |
| Ill-conditioned | Eigenvalues span many scales | Preconditioning, regularization |

---

## Exercises

1. Show that $A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}$ is positive definite using:
   - Eigenvalues
   - Sylvester's criterion
   - Cholesky decomposition

2. Find the Cholesky decomposition of $\begin{bmatrix} 4 & 2 \\ 2 & 5 \end{bmatrix}$ manually

3. Prove that $A^T A$ is always positive semi-definite for any matrix $A$

4. Show that adding $\lambda I$ to any symmetric matrix makes it PD for large enough $\lambda$

5. Verify that the RBF kernel matrix is PSD for any dataset

6. For $A = \begin{bmatrix} 3 & k \\ k & 3 \end{bmatrix}$, find the range of $k$ for which $A$ is PD

7. Prove that if $A \succ 0$, then $A^{-1} \succ 0$

8. Show that the trace and determinant of a PD matrix are both positive

9. Given a "nearly PD" matrix with one small negative eigenvalue, project it to the nearest PD matrix

10. Minimize $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T A \mathbf{x} - \mathbf{b}^T\mathbf{x}$ for PD matrix $A$

---

## References

1. Strang, G. - "Linear Algebra and Its Applications" (Chapter on Positive Definite Matrices)
2. Boyd & Vandenberghe - "Convex Optimization" (Chapters on Convexity and Optimization)
3. Bishop - "Pattern Recognition and Machine Learning" (Gaussian Distributions, Kernels)
4. Golub & Van Loan - "Matrix Computations" (Cholesky Decomposition, Numerical Aspects)
5. Horn & Johnson - "Matrix Analysis" (Positive Definite Matrices Theory)

---

## Further Reading

- **Semidefinite Programming (SDP)**: Optimization with PSD matrix constraints
- **Low-rank PSD Approximation**: Nyström method for kernel approximations
- **Matrix Completion**: Recovering PSD matrices from partial observations
- **Log-determinant Optimization**: Using $\log\det$ as a smooth surrogate for rank

---

[← Back to Advanced Linear Algebra](../README.md) | [← Matrix Norms](../06-Matrix-Norms/notes.md) | [Matrix Decompositions →](../08-Matrix-Decompositions/notes.md)
