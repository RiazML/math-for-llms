# Matrix Norms and Condition Numbers

## Introduction

Matrix norms measure the "size" of matrices, enabling us to quantify errors, measure distances, and analyze algorithm stability. The condition number tells us how sensitive a problem is to small perturbations. These concepts are fundamental to numerical linear algebra and essential for understanding machine learning algorithm behavior.

### Why Matrix Norms Matter

1. **Quantify Errors**: Measure difference between matrices (e.g., $\|A - \hat{A}\|$)
2. **Bound Operator Behavior**: How much can a linear transformation stretch vectors?
3. **Regularization**: Penalize model complexity ($\|W\|_F^2$, $\|W\|_1$, $\|W\|_*$)
4. **Stability Analysis**: Determine if algorithms are numerically stable

### Key Questions This Section Answers

- How do we measure the "size" of a matrix?
- What makes a problem numerically stable or unstable?
- Why does gradient descent converge slowly on some problems?
- How does regularization improve conditioning?

## Prerequisites

- Vector norms (L1, L2, L∞)
- Eigenvalues and singular values (SVD)
- Matrix multiplication and linear transformations
- Basic calculus (for gradient connections)

## Learning Objectives

1. Master different matrix norms and their interpretations
2. Compute and interpret condition numbers
3. Recognize and handle ill-conditioned problems
4. Apply norms for regularization in ML
5. Understand numerical stability implications

---

## 1. Vector Norms Review

### Definition

A vector norm $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}$ satisfies:

1. **Non-negativity**: $\|\mathbf{x}\| \geq 0$ with equality iff $\mathbf{x} = \mathbf{0}$
2. **Homogeneity**: $\|c\mathbf{x}\| = |c| \|\mathbf{x}\|$
3. **Triangle Inequality**: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

### Common Vector Norms

| Norm | Formula | Geometric Meaning |
|------|---------|-------------------|
| $L^1$ (Manhattan) | $\|\mathbf{x}\|_1 = \sum_i |x_i|$ | Taxicab distance |
| $L^2$ (Euclidean) | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Straight-line distance |
| $L^\infty$ (Max) | $\|\mathbf{x}\|_\infty = \max_i |x_i|$ | Maximum component |
| $L^p$ | $\|\mathbf{x}\|_p = \left(\sum_i |x_i|^p\right)^{1/p}$ | General $p$-norm |

### Example: Computing Vector Norms

For $\mathbf{x} = [3, -4, 0, 5]$:

$$\|\mathbf{x}\|_1 = |3| + |-4| + |0| + |5| = 12$$

$$\|\mathbf{x}\|_2 = \sqrt{9 + 16 + 0 + 25} = \sqrt{50} \approx 7.07$$

$$\|\mathbf{x}\|_\infty = \max(3, 4, 0, 5) = 5$$

### Unit Ball Shapes

The unit ball $\{\mathbf{x} : \|\mathbf{x}\| \leq 1\}$ has different shapes:

```
L¹ norm:       L² norm:       L⁴ norm:      L∞ norm:

   /\             ○             ○             ┌──┐
  /  \           ╱ ╲           / \            │  │
 /    \         │   │         (   )           │  │
 \    /         │   │         (   )           │  │
  \  /           ╲ ╱           \ /            │  │
   \/             ○             ○             └──┘

Diamond        Circle       Rounded       Square
                            Square
```

**ML Insight**: L1 produces "corners" where coordinates are exactly zero — this is why L1 regularization promotes sparsity!

### Norm Relationships

For any $\mathbf{x} \in \mathbb{R}^n$:

$$\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq n \|\mathbf{x}\|_\infty$$

$$\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \sqrt{n} \|\mathbf{x}\|_\infty$$

---

## 2. Matrix Norms

### Induced (Operator) Norms

The induced norm measures the maximum "stretching" of a matrix:

$$\|A\|_p = \max_{\mathbf{x} \neq \mathbf{0}} \frac{\|A\mathbf{x}\|_p}{\|\mathbf{x}\|_p} = \max_{\|\mathbf{x}\|_p = 1} \|A\mathbf{x}\|_p$$

**Interpretation**: The maximum factor by which $A$ can stretch any unit vector.

### Spectral Norm (L² Induced Norm)

$$\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$$

- Equals the largest singular value
- For symmetric matrices: $\|A\|_2 = |\lambda_{\max}(A)|$
- Most important norm for analysis

**Why Spectral Norm = σ_max**:

The SVD shows $A = U \Sigma V^T$. For unit vector $\mathbf{x}$:
$$\|A\mathbf{x}\|_2 = \|U\Sigma V^T \mathbf{x}\|_2 = \|\Sigma (V^T\mathbf{x})\|_2$$

Maximum achieved when $V^T\mathbf{x}$ aligns with $\sigma_1$, giving $\|A\|_2 = \sigma_1$.

### L¹ and L∞ Induced Norms

| Norm | Formula | Interpretation |
|------|---------|----------------|
| $\|A\|_1$ | $\max_j \sum_i |a_{ij}|$ | Maximum column sum |
| $\|A\|_\infty$ | $\max_i \sum_j |a_{ij}|$ | Maximum row sum |

**Example**: For $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$:

$$\|A\|_1 = \max(|1|+|3|, |2|+|4|) = \max(4, 6) = 6$$

$$\|A\|_\infty = \max(|1|+|2|, |3|+|4|) = \max(3, 7) = 7$$

### Frobenius Norm

$$\|A\|_F = \sqrt{\sum_{i,j} |a_{ij}|^2} = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_i \sigma_i^2}$$

**Properties**:
- Treats matrix as a long vector
- Easier to compute than spectral norm
- Used extensively in regularization

**Three Equivalent Formulas**:

1. Sum of squared elements: $\sqrt{\sum_{i,j} a_{ij}^2}$
2. Trace formulation: $\sqrt{\text{tr}(A^T A)}$
3. SVD formulation: $\sqrt{\sigma_1^2 + \sigma_2^2 + \cdots}$

### Nuclear (Trace) Norm

$$\|A\|_* = \sum_i \sigma_i = \text{sum of singular values}$$

**Properties**:
- Dual of spectral norm
- Convex surrogate for matrix rank
- Essential for low-rank matrix completion

### Norm Comparison Table

| Norm | Formula | Computation | Primary Use |
|------|---------|-------------|-------------|
| Spectral $\|A\|_2$ | $\sigma_{\max}$ | SVD or power iteration | Operator bounds, Lipschitz |
| Frobenius $\|A\|_F$ | $\sqrt{\sum a_{ij}^2}$ | Direct sum | Regularization |
| Nuclear $\|A\|_*$ | $\sum \sigma_i$ | SVD | Low-rank promotion |
| $\|A\|_1$ | Max column sum | Simple loop | Column analysis |
| $\|A\|_\infty$ | Max row sum | Simple loop | Row analysis |

---

## 3. Properties of Matrix Norms

### Basic Properties

For any matrix norm:

1. **Positive Definite**: $\|A\| \geq 0$, with equality iff $A = 0$
2. **Homogeneous**: $\|cA\| = |c| \|A\|$ for scalar $c$
3. **Triangle Inequality**: $\|A + B\| \leq \|A\| + \|B\|$

### Submultiplicativity

$$\|AB\| \leq \|A\| \|B\|$$

**Essential Property!** Enables us to bound:
- Products of matrices
- Matrix powers: $\|A^n\| \leq \|A\|^n$
- Stability of iterations

**Example**: If $\|A\| < 1$, then $A^n \to 0$ as $n \to \infty$.

### Consistency with Vector Norms

Induced norms satisfy:
$$\|A\mathbf{x}\| \leq \|A\| \|\mathbf{x}\|$$

### Relationships Between Norms

For an $m \times n$ matrix with rank $r$:

$$\|A\|_2 \leq \|A\|_F \leq \sqrt{r} \|A\|_2$$

$$\|A\|_2 \leq \sqrt{\|A\|_1 \cdot \|A\|_\infty}$$

$$\|A\|_2 \leq \|A\|_* \leq \sqrt{r} \|A\|_2$$

$$\frac{1}{\sqrt{n}}\|A\|_F \leq \|A\|_2 \leq \|A\|_F$$

---

## 4. Condition Number

### Definition

The condition number measures sensitivity of $A\mathbf{x} = \mathbf{b}$ to perturbations:

$$\kappa(A) = \|A\| \|A^{-1}\|$$

Using the spectral norm (most common):

$$\kappa_2(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

### Interpretation

| Condition Number | Classification | Meaning |
|------------------|----------------|---------|
| $\kappa \approx 1$ | Well-conditioned | Numerically stable |
| $\kappa \sim 10^3$ | Mildly ill-conditioned | Some accuracy loss |
| $\kappa \sim 10^6$ | Moderately ill-conditioned | Significant accuracy loss |
| $\kappa \sim 10^{16}$ | Severely ill-conditioned | No reliable digits |
| $\kappa = \infty$ | Singular | No unique solution |

### Error Amplification

For solving $A\mathbf{x} = \mathbf{b}$, if we perturb $\mathbf{b}$ to $\mathbf{b} + \Delta\mathbf{b}$:

$$\frac{\|\Delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \frac{\|\Delta \mathbf{b}\|}{\|\mathbf{b}\|}$$

**Rule**: A 1% error in $\mathbf{b}$ can become $\kappa(A) \cdot 1\%$ error in $\mathbf{x}$!

### Geometric Interpretation

The condition number describes how much a matrix "stretches" vectors disproportionately:

```
Well-conditioned (κ ≈ 1):       Ill-conditioned (κ >> 1):

Unit circle → nearly circular    Unit circle → thin ellipse

      ○                               ═══
     ╱ ╲                              ╱ ╲
    │   │                           ─────────
    │   │                             ╲ ╱
     ╲ ╱                              ═══
      ○
```

**Intuition**: An ill-conditioned matrix squashes some directions nearly to zero. Small perturbations in those directions get wildly amplified.

### Properties of Condition Number

1. $\kappa(A) \geq 1$ (with equality for orthogonal matrices)
2. $\kappa(A) = \kappa(A^T)$
3. $\kappa(A) = \kappa(A^{-1})$
4. $\kappa(cA) = \kappa(A)$ for $c \neq 0$
5. $\kappa(AB) \leq \kappa(A) \kappa(B)$

---

## 5. Computing Norms and Condition Numbers

### Python Implementation

```python
import numpy as np
from numpy.linalg import norm, cond, svd

A = np.array([[1, 2],
              [3, 4]])

# Matrix Norms
print(f"Spectral norm ||A||₂: {norm(A, 2):.4f}")
print(f"Frobenius norm ||A||_F: {norm(A, 'fro'):.4f}")
print(f"L1 norm ||A||₁: {norm(A, 1):.4f}")
print(f"L∞ norm ||A||_∞: {norm(A, np.inf):.4f}")

# Condition number
print(f"Condition number κ(A): {cond(A):.4f}")

# Via SVD
U, S, Vt = svd(A)
print(f"Singular values: {S}")
print(f"κ from SVD: {S[0]/S[-1]:.4f}")
print(f"Nuclear norm: {np.sum(S):.4f}")
```

### Computational Complexity

| Norm | Method | Complexity |
|------|--------|------------|
| $\|A\|_1$ | Column sums | $O(mn)$ |
| $\|A\|_\infty$ | Row sums | $O(mn)$ |
| $\|A\|_F$ | Element sum | $O(mn)$ |
| $\|A\|_2$ | SVD or power iteration | $O(mn^2)$ or $O(mn \cdot k)$ |
| $\kappa(A)$ | Full SVD | $O(mn^2)$ |

---

## 6. Ill-Conditioned Matrices

### Example 1: Nearly Singular Matrix

$$A = \begin{pmatrix} 1 & 1 \\ 1 & 1.0001 \end{pmatrix}$$

The determinant is tiny: $\det(A) = 0.0001$

$$\kappa(A) \approx 40,000$$

A 0.01% perturbation can cause 400% error in the solution!

### Example 2: Hilbert Matrix

$$H_{ij} = \frac{1}{i + j - 1}$$

$$H_3 = \begin{pmatrix} 1 & 1/2 & 1/3 \\ 1/2 & 1/3 & 1/4 \\ 1/3 & 1/4 & 1/5 \end{pmatrix}$$

| Size | Condition Number |
|------|------------------|
| $H_3$ | $5.2 \times 10^2$ |
| $H_5$ | $4.8 \times 10^5$ |
| $H_7$ | $4.8 \times 10^8$ |
| $H_{10}$ | $1.6 \times 10^{13}$ |
| $H_{12}$ | $1.8 \times 10^{16}$ |

The condition number grows exponentially! $H_{12}$ has no reliable digits in double precision.

### Example 3: Vandermonde Matrix

For polynomial interpolation at points $x_0, \ldots, x_n$:

$$V_{ij} = x_i^j$$

Becomes severely ill-conditioned for many points or clustered points.

### Recognizing Ill-Conditioning

**Signs of trouble**:
1. Nearly equal rows/columns
2. Very small eigenvalues or singular values
3. Large variation in row/column norms
4. Solution changes wildly with small input changes

---

## 7. Regularization and Conditioning

### Why Regularization Helps

L2 regularization adds $\lambda I$ to $X^T X$:

$$(X^T X + \lambda I)\mathbf{w} = X^T \mathbf{y}$$

**Effect on condition number**:

If $X^T X$ has eigenvalues $\lambda_1 \geq \cdots \geq \lambda_n > 0$, then:

$$\kappa(X^T X) = \frac{\lambda_1}{\lambda_n}$$

With regularization:

$$\kappa(X^T X + \lambda I) = \frac{\lambda_1 + \lambda}{\lambda_n + \lambda}$$

Adding $\lambda$ improves the denominator more than the numerator!

### Example: Improving Conditioning

```python
X = np.array([[1, 1, 1],
              [1, 1.001, 1],
              [1, 1, 1.001]])

XTX = X.T @ X
print(f"Original κ(X^T X) = {cond(XTX):.0f}")

for lam in [0.01, 0.1, 1.0]:
    XTX_reg = XTX + lam * np.eye(3)
    print(f"λ = {lam}: κ = {cond(XTX_reg):.2f}")
```

Output:
```
Original κ(X^T X) = 6,000,000+
λ = 0.01: κ = 600,000
λ = 0.1: κ = 60,000
λ = 1.0: κ = 10
```

### Regularization Types

| Regularization | Norm Penalized | Effect |
|----------------|----------------|--------|
| Ridge (L2) | $\|W\|_F^2$ | Improves conditioning, shrinks weights |
| Lasso (L1) | $\|W\|_1$ | Promotes sparsity |
| Elastic Net | $\alpha\|W\|_1 + (1-\alpha)\|W\|_2^2$ | Sparsity + stability |
| Nuclear | $\|W\|_*$ | Promotes low-rank |

---

## 8. Applications in Machine Learning

### 1. Weight Regularization

#### Ridge Regression (L2)

$$\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2$$

Closed form: $\mathbf{w} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$

The regularized system is always better conditioned!

#### Lasso (L1)

$$\min_\mathbf{w} \|X\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_1$$

Produces sparse solutions (many $w_i = 0$ exactly).

#### Neural Network Weight Decay

$$L = L_{\text{task}} + \lambda \sum_l \|W_l\|_F^2$$

Prevents weights from growing unboundedly.

### 2. Spectral Normalization (GANs)

Constrain each layer's weight matrix:

$$W_{\text{norm}} = \frac{W}{\|W\|_2}$$

**Effects**:
- Each layer is 1-Lipschitz
- Stabilizes GAN training
- Prevents mode collapse

```python
def spectral_norm(W):
    """Normalize by spectral norm."""
    sigma = np.linalg.norm(W, 2)
    return W / sigma
```

### 3. Lipschitz Constraints

A function $f$ is $L$-Lipschitz if:

$$\|f(\mathbf{x}) - f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|$$

For a fully connected layer $f(\mathbf{x}) = W\mathbf{x}$:

$$L = \|W\|_2$$

For a neural network: $L \leq \prod_l \|W_l\|_2$

**Applications**:
- Wasserstein GANs require 1-Lipschitz discriminators
- Certified adversarial robustness
- Stable optimization

### 4. Matrix Completion (Nuclear Norm)

For recommender systems with missing entries:

$$\min_X \|X\|_* \quad \text{s.t. } X_{ij} = M_{ij} \text{ for observed } (i,j)$$

The nuclear norm promotes low-rank solutions (users and items have few latent factors).

### 5. Low-Rank Approximation

The Eckart-Young theorem:

$$\min_{\text{rank}(B) = k} \|A - B\|_F = \sqrt{\sigma_{k+1}^2 + \cdots + \sigma_r^2}$$

Achieved by truncated SVD: $B = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$

**Applications**:
- Dimensionality reduction (PCA)
- Image compression
- Denoising

---

## 9. Condition Number and Optimization

### Effect on Gradient Descent

For quadratic $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T H \mathbf{x} - \mathbf{b}^T \mathbf{x}$:

$$\text{Convergence rate} \sim \left(\frac{\kappa - 1}{\kappa + 1}\right)^k$$

where $\kappa = \kappa(H)$.

| Condition Number | Iterations to $10^{-6}$ error |
|------------------|------------------------------|
| $\kappa = 1$ | ~14 |
| $\kappa = 10$ | ~30 |
| $\kappa = 100$ | ~300 |
| $\kappa = 1000$ | ~3000 |

**Intuition**: High condition number means:
- Eigenvalues vary widely
- Some directions require tiny learning rates
- Overall convergence is slow

### Learning Rate Constraints

Maximum stable learning rate:

$$\alpha < \frac{2}{\lambda_{\max}(H)}$$

But for fast convergence in all directions:

$$\alpha \approx \frac{2}{\lambda_{\max} + \lambda_{\min}}$$

The condition number limits the optimal learning rate!

### Preconditioning

Transform the problem to improve conditioning:

$$\min_\mathbf{y} f(P^{-1}\mathbf{y})$$

where $P$ approximates the Hessian. This is the idea behind:
- Newton's method (exact Hessian)
- Adam, RMSprop (diagonal approximation)
- Natural gradient (Fisher information)

---

## 10. Numerical Stability

### Backward vs Forward Error

**Forward error**: How far is computed solution from true solution?
$$\|\hat{\mathbf{x}} - \mathbf{x}_{\text{true}}\|$$

**Backward error**: What perturbation to input gives computed output?
$$\delta \text{ such that } (A + \delta A)\hat{\mathbf{x}} = \mathbf{b}$$

A **stable algorithm** has small backward error. Combined with low condition number, this gives small forward error.

### Stability of Algorithms

| Algorithm | Stability | Notes |
|-----------|-----------|-------|
| Gaussian elim. (no pivot) | Unstable | Can fail catastrophically |
| Gaussian elim. (partial pivot) | Stable | Standard LU decomposition |
| QR factorization | Very stable | Better conditioning |
| SVD | Most stable | Gold standard |
| Normal equations | Unstable | Squares condition number |
| QR for least squares | Stable | Preferred method |

### Rule of Thumb

If $\kappa(A) \approx 10^k$ and machine precision is $\epsilon_{\text{mach}} \approx 10^{-16}$:

- Expect to lose $\sim k$ digits of accuracy
- Reliable digits: $16 - k$

**Example**: For $\kappa = 10^8$, expect only ~8 reliable digits.

### Avoiding Numerical Issues

1. **Don't form $X^T X$ explicitly** — use QR for least squares
2. **Add regularization** — improves conditioning
3. **Use iterative methods** — for large ill-conditioned systems
4. **Scale data appropriately** — prevent large value ranges
5. **Use higher precision** — if available and necessary

---

## 11. Dual Norms

### Definition

The dual norm of $\|\cdot\|$ is:

$$\|A\|_* = \max_{\|B\| \leq 1} |\langle A, B \rangle|$$

where $\langle A, B \rangle = \text{tr}(A^T B)$.

### Dual Pairs

| Norm | Dual Norm |
|------|-----------|
| Spectral $\|A\|_2$ | Nuclear $\|A\|_*$ |
| Frobenius $\|A\|_F$ | Frobenius $\|A\|_F$ |
| $\|A\|_1$ | $\|A\|_\infty$ |

### Importance in Optimization

The dual norm appears in:
- Subdifferentials of norm regularizers
- Proximal operators
- Duality in convex optimization

---

## 12. Summary

### Essential Norms

| Norm | Formula | Key Use Case |
|------|---------|--------------|
| Spectral $\|A\|_2$ | $\sigma_{\max}$ | Lipschitz bounds, stability |
| Frobenius $\|A\|_F$ | $\sqrt{\sum a_{ij}^2}$ | L2 regularization |
| Nuclear $\|A\|_*$ | $\sum \sigma_i$ | Low-rank promotion |

### Condition Number

$$\kappa(A) = \frac{\sigma_{\max}}{\sigma_{\min}} = \|A\| \|A^{-1}\|$$

- $\kappa \approx 1$: Well-conditioned
- $\kappa \gg 1$: Ill-conditioned, expect trouble
- Rule: Lose ~$\log_{10}(\kappa)$ digits of accuracy

### Key Theorems

1. **Error Bound**: $\frac{\|\Delta \mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(A) \frac{\|\Delta \mathbf{b}\|}{\|\mathbf{b}\|}$

2. **Submultiplicativity**: $\|AB\| \leq \|A\| \|B\|$

3. **Eckart-Young**: Truncated SVD gives optimal low-rank approximation

### ML Connections

```
Regularization:
├── L2 (Frobenius) → Weight decay, improved conditioning
├── L1 → Sparsity, feature selection
└── Nuclear → Low-rank, matrix completion

Stability:
├── Spectral normalization → 1-Lipschitz layers, stable GANs
├── Condition number → Optimization convergence speed
└── Preconditioning → Adam, second-order methods

Analysis:
├── Lipschitz constant → Robustness bounds
├── Low-rank approximation → Compression, denoising
└── Numerical stability → Algorithm reliability
```

---

## Files in This Section

| File | Description |
|------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive demonstrations of all norm types |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## Further Reading

1. Golub & Van Loan - "Matrix Computations" (definitive reference)
2. Trefethen & Bau - "Numerical Linear Algebra" (accessible treatment)
3. Higham - "Accuracy and Stability of Numerical Algorithms"
4. Miyato et al. - "Spectral Normalization for GANs" (ML application)

---

## Quick Reference

```python
import numpy as np
from numpy.linalg import norm, cond, svd

A = np.array([[1, 2], [3, 4]])

# Norms
norm(A, 2)       # Spectral (σ_max)
norm(A, 'fro')   # Frobenius (√Σaᵢⱼ²)
norm(A, 1)       # Max column sum
norm(A, np.inf)  # Max row sum
np.sum(svd(A)[1]) # Nuclear (Σσᵢ)

# Condition number
cond(A)          # κ(A) = σ_max / σ_min

# SVD components
U, S, Vt = svd(A)  # A = U @ diag(S) @ Vt
```
