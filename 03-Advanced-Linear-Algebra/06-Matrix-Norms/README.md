# Matrix Norms and Condition Numbers

## Introduction

Matrix norms measure the "size" of matrices, enabling us to quantify errors, measure distances, and analyze algorithm stability. Condition numbers tell us how sensitive a problem is to small changes. These concepts are essential for numerical linear algebra and understanding ML algorithm behavior.

## Prerequisites

- Vector norms
- Eigenvalues and singular values
- Matrix multiplication
- Linear transformations

## Learning Objectives

1. Understand different matrix norms
2. Compute and interpret condition numbers
3. Recognize ill-conditioned problems
4. Apply norms in ML contexts (regularization)
5. Analyze numerical stability

---

## 1. Vector Norms Review

### Common Vector Norms

| Norm              | Formula                                  | Geometric Meaning      |
| ----------------- | ---------------------------------------- | ---------------------- | ---------------- | ----------------- |
| $L^1$ (Manhattan) | $\|\mathbf{x}\|\_1 = \sum_i              | x_i                    | $                | Taxicab distance  |
| $L^2$ (Euclidean) | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Straight-line distance |
| $L^\infty$ (Max)  | $\|\mathbf{x}\|\_\infty = \max_i         | x_i                    | $                | Maximum component |
| $L^p$             | $\|\mathbf{x}\|\_p = \left(\sum_i        | x_i                    | ^p\right)^{1/p}$ | General $p$-norm  |

### Unit Ball Shapes

```
L¹ norm:       L² norm:       L∞ norm:

   /\             ○             ┌─┐
  /  \           / \            │ │
 /    \         │   │           │ │
 \    /         │   │           │ │
  \  /           \ /            └─┘
   \/             ○

Diamond        Circle         Square
```

---

## 2. Matrix Norms

### Induced (Operator) Norms

The induced norm measures the maximum "stretching" of a matrix:

$$\|A\|_p = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_p}{\|\mathbf{x}\|_p} = \max_{\|\mathbf{x}\|_p = 1} \|A\mathbf{x}\|_p$$

### Spectral Norm ($L^2$ Induced)

$$\|A\|_2 = \sigma_{\max}(A) = \text{largest singular value}$$

For symmetric matrices: $\|A\|_2 = |\lambda_{\max}|$

### Other Induced Norms

| Norm           | Formula                            |
| -------------- | ---------------------------------- | ------- | --- |
| $\|A\|_1$      | Maximum column sum: $\max_j \sum_i | a\_{ij} | $   |
| $\|A\|_\infty$ | Maximum row sum: $\max_i \sum_j    | a\_{ij} | $   |
| $\|A\|_2$      | Largest singular value             |

### Frobenius Norm

$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_i \sigma_i^2}$$

Treats matrix as a long vector.

### Nuclear (Trace) Norm

$$\|A\|_* = \sum_i \sigma_i = \text{sum of singular values}$$

Important for low-rank matrix completion.

---

## 3. Properties of Matrix Norms

### Basic Properties

For any matrix norm:

1. **Positive**: $\|A\| \geq 0$, with equality iff $A = 0$
2. **Homogeneous**: $\|cA\| = |c| \|A\|$
3. **Triangle inequality**: $\|A + B\| \leq \|A\| + \|B\|$

### Submultiplicativity

$$\|AB\| \leq \|A\| \|B\|$$

Essential for analyzing matrix products and powers!

### Relationships

```
||A||₂ ≤ ||A||_F ≤ √r ||A||₂

where r = rank(A)

||A||₂ ≤ √(||A||₁ ||A||_∞)
```

---

## 4. Condition Number

### Definition

The condition number measures how sensitive $A\mathbf{x} = \mathbf{b}$ is to perturbations:

$$\kappa(A) = \|A\| \|A^{-1}\|$$

Using the 2-norm (most common):

$$\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

### Interpretation

```
κ(A) ≈ 1:     Well-conditioned (stable)
κ(A) >> 1:    Ill-conditioned (unstable)
κ(A) = ∞:     Singular (A not invertible)
```

### Error Amplification

For solving $A\mathbf{x} = \mathbf{b}$:

$$\frac{\|\Delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \frac{\|\Delta \mathbf{b}\|}{\|\mathbf{b}\|}$$

A 1% error in $\mathbf{b}$ can become $\kappa(A) \cdot 1\%$ error in $\mathbf{x}$!

### Visual Interpretation

```
Well-conditioned (κ ≈ 1):    Ill-conditioned (κ >> 1):

Column space:               Column space:

    │                           ╱
    │  (nearly circular)       ╱  (elongated ellipse)
────┼────                 ────╱──────
    │                        ╱
    │                       ╱
```

---

## 5. Computing Norms and Condition Numbers

### Spectral Norm via SVD

$$\|A\|_2 = \sigma_1 \quad \text{(largest singular value)}$$

### Frobenius Norm

$$\|A\|_F = \sqrt{\sum_{i,j} a_{ij}^2} = \sqrt{\sigma_1^2 + \sigma_2^2 + \ldots}$$

### Condition Number

$$\kappa_2(A) = \frac{\sigma_1}{\sigma_n}$$

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])

# Norms
spectral = np.linalg.norm(A, 2)     # Spectral (2-norm)
frobenius = np.linalg.norm(A, 'fro') # Frobenius
l1_norm = np.linalg.norm(A, 1)       # Max column sum
inf_norm = np.linalg.norm(A, np.inf) # Max row sum

# Condition number
cond = np.linalg.cond(A)
```

---

## 6. Ill-Conditioning Examples

### Example 1: Nearly Singular

$$A = \begin{bmatrix} 1 & 1 \\ 1 & 1.0001 \end{bmatrix}$$

$$\kappa(A) \approx 40000$$

A tiny change in $A$ can wildly change the solution!

### Example 2: Hilbert Matrix

$$H_{ij} = \frac{1}{i + j - 1}$$

$$H = \begin{bmatrix} 1 & 1/2 & 1/3 \\ 1/2 & 1/3 & 1/4 \\ 1/3 & 1/4 & 1/5 \end{bmatrix}$$

Hilbert matrices are notoriously ill-conditioned:

- $\kappa(H_5) \approx 4.8 \times 10^5$
- $\kappa(H_{10}) \approx 1.6 \times 10^{13}$

### Example 3: Vandermonde Matrix

Polynomial interpolation matrices become ill-conditioned for many points.

---

## 7. Applications in ML/AI

### 1. Regularization

#### L2 Regularization (Ridge/Weight Decay)

$$\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$$

The Frobenius norm penalty prevents large weights.

#### L1 Regularization (Lasso)

$$\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_1$$

Promotes sparsity (many weights become exactly zero).

#### Nuclear Norm (Matrix Completion)

$$\min_X \|X\|_* \quad \text{subject to } X_{ij} = M_{ij} \text{ for observed entries}$$

Used in recommender systems!

### 2. Neural Network Stability

#### Spectral Normalization

Constrain $\|W\|_2 = 1$ for discriminator weights in GANs:

$$W_{\text{normalized}} = \frac{W}{\|W\|_2}$$

Improves training stability.

#### Lipschitz Constraint

A network is $L$-Lipschitz if:
$$\|f(\mathbf{x}) - f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|$$

For fully connected layer: $L = \|W\|_2$

### 3. Gradient Analysis

The condition number affects optimization:

$$\kappa(\nabla^2 f) = \frac{\lambda_{\max}}{\lambda_{\min}}$$

High condition number → slow convergence of gradient descent!

### 4. Low-Rank Approximation

The Eckart-Young theorem uses matrix norms:

$$\min_{\text{rank}(B) = k} \|A - B\|_F$$

Solution: truncated SVD

---

## 8. Norm Comparisons

### Relationships Between Norms

For an $m \times n$ matrix:

$$\|A\|_2 \leq \|A\|_F \leq \sqrt{\min(m,n)} \|A\|_2$$

$$\|A\|_2 \leq \sqrt{\|A\|_1 \|A\|_\infty}$$

### When to Use Each Norm

| Norm                  | Use Case                              |
| --------------------- | ------------------------------------- |
| Spectral ($\|A\|_2$)  | Maximum stretching, operator analysis |
| Frobenius ($\|A\|_F$) | General "size", regularization        |
| Nuclear ($\|A\|_*$)   | Low-rank promotion                    |
| $L^1$                 | Sparsity, column-wise analysis        |
| $L^\infty$            | Row-wise analysis                     |

---

## 9. Numerical Stability

### Backward vs Forward Error

**Forward error**: How far is computed answer from true answer?
$$\|\hat{\mathbf{x}} - \mathbf{x}\|$$

**Backward error**: How much do we need to perturb input to get computed answer?

### Stable Algorithms

| Algorithm                               | Condition        | Stability   |
| --------------------------------------- | ---------------- | ----------- |
| Gaussian elimination (no pivoting)      | Ill-conditioned  | Unstable    |
| Gaussian elimination (partial pivoting) | Well-conditioned | Stable      |
| QR decomposition                        | Well-conditioned | Very stable |
| SVD                                     | Well-conditioned | Most stable |

### Rule of Thumb

If $\kappa(A) \approx 10^k$, you lose about $k$ digits of accuracy!

With double precision (16 digits), $\kappa > 10^{16}$ means no reliable digits.

---

## 10. Summary

### Key Norms

| Norm                | Formula                | ML Application     |
| ------------------- | ---------------------- | ------------------ |
| Spectral $\|A\|_2$  | $\sigma_{\max}$        | Lipschitz constant |
| Frobenius $\|A\|_F$ | $\sqrt{\sum a_{ij}^2}$ | L2 regularization  |
| Nuclear $\|A\|_*$   | $\sum \sigma_i$        | Matrix completion  |

### Condition Number

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}}$$

- $\kappa \approx 1$: Well-conditioned
- $\kappa \gg 1$: Ill-conditioned (sensitive to errors)

### ML Connections

```
Regularization:
├── L2 (Frobenius) → Prevents large weights
├── L1 → Promotes sparsity
└── Nuclear → Promotes low rank

Stability:
├── Spectral normalization → Stabilizes GANs
├── Condition number → Affects convergence
└── Lipschitz bounds → Network robustness
```

---

## Exercises

1. Compute all norms ($\|A\|_1, \|A\|_2, \|A\|_\infty, \|A\|_F$) for $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$
2. Find the condition number of $A = \begin{bmatrix} 2 & 0 \\ 0 & 0.01 \end{bmatrix}$
3. Explain why the Hilbert matrix is ill-conditioned
4. Show that $\|A\|_F = \sqrt{\text{tr}(A^T A)}$
5. How does L2 regularization affect the condition number of $X^TX$?

---

## References

1. Trefethen & Bau - "Numerical Linear Algebra"
2. Golub & Van Loan - "Matrix Computations"
3. Goodfellow et al. - "Deep Learning" (regularization)
