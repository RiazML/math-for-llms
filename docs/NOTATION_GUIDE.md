# 📖 Mathematical Notation Guide

> A comprehensive guide to the mathematical notation used throughout this repository.

---

## Table of Contents

1. [Sets and Numbers](#sets-and-numbers)
2. [Vectors](#vectors)
3. [Matrices](#matrices)
4. [Functions and Operations](#functions-and-operations)
5. [Calculus](#calculus)
6. [Probability](#probability)
7. [Linear Algebra Specific](#linear-algebra-specific)
8. [Greek Letters](#greek-letters)
9. [Information Theory](#information-theory)
10. [Graph Theory](#graph-theory)
11. [Norms and Spaces](#norms-and-spaces)
12. [Optimization](#optimization)
13. [Common Conventions in ML](#common-conventions-in-ml)
14. [Reading ML Papers: Notation Tips](#reading-ml-papers-notation-tips)

---

## Sets and Numbers

| Symbol                    | Name                     | Meaning                                             |
| ------------------------- | ------------------------ | --------------------------------------------------- | ----------- | ----------------------------- |
| $\mathbb{N}$              | Natural numbers          | $\{0, 1, 2, 3, ...\}$ or $\{1, 2, 3, ...\}$         |
| $\mathbb{Z}$              | Integers                 | $\{..., -2, -1, 0, 1, 2, ...\}$                     |
| $\mathbb{Q}$              | Rational numbers         | Fractions $\frac{p}{q}$ where $p, q \in \mathbb{Z}$ |
| $\mathbb{R}$              | Real numbers             | All numbers on the number line                      |
| $\mathbb{R}^n$            | n-dimensional real space | Vectors with $n$ real components                    |
| $\mathbb{R}^{m \times n}$ | Real matrices            | $m \times n$ matrices with real entries             |
| $\mathbb{C}$              | Complex numbers          | Numbers of form $a + bi$                            |
| $\in$                     | Element of               | $x \in S$ means $x$ is in set $S$                   |
| $\notin$                  | Not element of           | $x \notin S$ means $x$ is not in $S$                |
| $\subset$                 | Subset                   | $A \subset B$ means all of $A$ is in $B$            |
| $\subseteq$               | Subset or equal          | $A \subseteq B$ includes $A = B$                    |
| $\cup$                    | Union                    | $A \cup B$ = elements in $A$ or $B$                 |
| $\cap$                    | Intersection             | $A \cap B$ = elements in both $A$ and $B$           |
| $\emptyset$ or $\{\}$     | Empty set                | Set with no elements                                |
| $                         | S                        | $                                                   | Cardinality | Number of elements in set $S$ |

---

## Vectors

### Notation Styles

| Notation       | Meaning                 | Example                               |
| -------------- | ----------------------- | ------------------------------------- |
| $\mathbf{x}$   | Vector (bold lowercase) | $\mathbf{x} = [x_1, x_2, ..., x_n]^T$ |
| $\vec{x}$      | Vector (arrow)          | Same as above                         |
| $x_i$          | $i$-th component        | Element at position $i$               |
| $\mathbf{e}_i$ | Standard basis vector   | 1 at position $i$, 0 elsewhere        |
| $\mathbf{0}$   | Zero vector             | All components are 0                  |
| $\mathbf{1}$   | Ones vector             | All components are 1                  |

### Vector Operations

| Symbol                                   | Name             | Meaning                           |
| ---------------------------------------- | ---------------- | --------------------------------- | --- | ---------- |
| $\mathbf{x}^T$                           | Transpose        | Row vector from column vector     |
| $\mathbf{x} \cdot \mathbf{y}$            | Dot product      | $\sum_i x_i y_i$                  |
| $\mathbf{x}^T \mathbf{y}$                | Inner product    | Same as dot product               |
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | Inner product    | Alternative notation              |
| $\mathbf{x} \times \mathbf{y}$           | Cross product    | Vector perpendicular to both (3D) |
| $\|\mathbf{x}\|$                         | Euclidean norm   | $\sqrt{\sum_i x_i^2}$             |
| $\|\mathbf{x}\|_1$                       | L1 norm          | $\sum_i                           | x_i | $          |
| $\|\mathbf{x}\|_2$                       | L2 norm          | Same as Euclidean norm            |
| $\|\mathbf{x}\|_p$                       | Lp norm          | $(\sum_i                          | x_i | ^p)^{1/p}$ |
| $\|\mathbf{x}\|_\infty$                  | Infinity norm    | $\max_i                           | x_i | $          |
| $\mathbf{x} \odot \mathbf{y}$            | Hadamard product | Element-wise multiplication       |

### Visual

```
Column vector:        Row vector:
    ┌───┐
x = │ x₁│             x^T = [x₁  x₂  x₃]
    │ x₂│
    │ x₃│
    └───┘
```

---

## Matrices

### Notation

| Symbol                    | Meaning                        |
| ------------------------- | ------------------------------ |
| $A$, $B$, $M$             | Matrices (uppercase letters)   |
| $A_{ij}$ or $a_{ij}$      | Element at row $i$, column $j$ |
| $A_{i:}$                  | $i$-th row of $A$              |
| $A_{:j}$                  | $j$-th column of $A$           |
| $A^T$                     | Transpose                      |
| $A^{-1}$                  | Inverse                        |
| $A^+$                     | Pseudoinverse (Moore-Penrose)  |
| $A^*$                     | Conjugate transpose            |
| $I$ or $I_n$              | Identity matrix ($n \times n$) |
| $O$                       | Zero matrix                    |
| $\text{diag}(\mathbf{x})$ | Diagonal matrix from vector    |

### Matrix Operations

| Symbol           | Name                  | Meaning                                  |
| ---------------- | --------------------- | ---------------------------------------- | ----------- | --- |
| $AB$             | Matrix multiplication | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$        |
| $A \odot B$      | Hadamard product      | Element-wise multiplication              |
| $A \otimes B$    | Kronecker product     | Block matrix                             |
| $\text{tr}(A)$   | Trace                 | Sum of diagonal elements                 |
| $\det(A)$ or $   | A                     | $                                        | Determinant |     |
| $\text{rank}(A)$ | Rank                  | Number of linearly independent rows/cols |
| $\|A\|_F$        | Frobenius norm        | $\sqrt{\sum_{i,j} A_{ij}^2}$             |
| $\|A\|_2$        | Spectral norm         | Largest singular value                   |

### Visual

```
       Column j
          ↓
    ┌─────────────┐
    │ a₁₁ a₁₂ a₁₃│
Row i→  │ a₂₁ a₂₂ a₂₃│   A ∈ ℝ^(3×3)
    │ a₃₁ a₃₂ a₃₃│
    └─────────────┘
```

---

## Functions and Operations

### General

| Symbol            | Name             | Meaning                         |
| ----------------- | ---------------- | ------------------------------- |
| $f: X \to Y$      | Function         | Maps from $X$ to $Y$            |
| $f(x)$            | Function value   | Value of $f$ at $x$             |
| $f \circ g$       | Composition      | $(f \circ g)(x) = f(g(x))$      |
| $f^{-1}$          | Inverse function | $f^{-1}(f(x)) = x$              |
| $\arg\max_x f(x)$ | Argmax           | Value of $x$ that maximizes $f$ |
| $\arg\min_x f(x)$ | Argmin           | Value of $x$ that minimizes $f$ |
| $\max(a, b)$      | Maximum          | Larger of $a$ and $b$           |
| $\min(a, b)$      | Minimum          | Smaller of $a$ and $b$          |
| $\sup$            | Supremum         | Least upper bound               |
| $\inf$            | Infimum          | Greatest lower bound            |

### Summation and Products

| Symbol            | Meaning            | Example                                   |
| ----------------- | ------------------ | ----------------------------------------- |
| $\sum_{i=1}^{n}$  | Summation          | $\sum_{i=1}^{3} i = 1 + 2 + 3$            |
| $\prod_{i=1}^{n}$ | Product            | $\prod_{i=1}^{3} i = 1 \times 2 \times 3$ |
| $\sum_i$          | Sum over index $i$ | Short form                                |
| $\sum_{x \in S}$  | Sum over set       | Sum for all $x$ in $S$                    |

### Common Functions

| Symbol              | Name              | Definition                |
| ------------------- | ----------------- | ------------------------- | -------------- | --- |
| $\exp(x)$ or $e^x$  | Exponential       |                           |
| $\log(x)$           | Natural logarithm | Base $e$                  |
| $\log_2(x)$         | Binary logarithm  | Base 2                    |
| $\log_{10}(x)$      | Common logarithm  | Base 10                   |
| $                   | x                 | $                         | Absolute value |     |
| $\lfloor x \rfloor$ | Floor             | Largest integer $\leq x$  |
| $\lceil x \rceil$   | Ceiling           | Smallest integer $\geq x$ |
| $\text{sign}(x)$    | Sign function     | $-1$, $0$, or $1$         |

---

## Calculus

### Derivatives

| Symbol                          | Name               | Meaning                                     |
| ------------------------------- | ------------------ | ------------------------------------------- |
| $\frac{df}{dx}$                 | Derivative         | Rate of change of $f$ w.r.t. $x$            |
| $f'(x)$                         | Derivative         | Same as above                               |
| $\frac{d^2f}{dx^2}$             | Second derivative  |                                             |
| $f''(x)$                        | Second derivative  |                                             |
| $\frac{\partial f}{\partial x}$ | Partial derivative | Derivative holding other variables constant |
| $\partial_x f$                  | Partial derivative | Short notation                              |
| $f_x$                           | Partial derivative | Subscript notation                          |
| $\nabla f$                      | Gradient           | Vector of partial derivatives               |
| $\nabla^2 f$                    | Laplacian          | Sum of second partials                      |

### Gradient and Higher-Order

| Symbol                                                   | Name                     | Definition                                                                  |
| -------------------------------------------------------- | ------------------------ | --------------------------------------------------------------------------- |
| $\nabla f$                                               | Gradient                 | $[\frac{\partial f}{\partial x_1}, ..., \frac{\partial f}{\partial x_n}]^T$ |
| $\nabla_\theta f$                                        | Gradient w.r.t. $\theta$ | Specifies the variable                                                      |
| $J$ or $\frac{\partial \mathbf{f}}{\partial \mathbf{x}}$ | Jacobian                 | Matrix of first partials                                                    |
| $H$ or $\nabla^2 f$                                      | Hessian                  | Matrix of second partials                                                   |
| $\frac{\partial^2 f}{\partial x \partial y}$             | Mixed partial            |                                                                             |

### Integrals

| Symbol             | Name                |
| ------------------ | ------------------- |
| $\int f(x) dx$     | Indefinite integral |
| $\int_a^b f(x) dx$ | Definite integral   |
| $\iint$            | Double integral     |
| $\iiint$           | Triple integral     |
| $\oint$            | Contour integral    |

---

## Probability

### Basic Notation

| Symbol             | Name                 | Meaning                    |
| ------------------ | -------------------- | -------------------------- | ---------------------------- |
| $P(A)$             | Probability          | Probability of event $A$   |
| $P(A               | B)$                  | Conditional probability    | Probability of $A$ given $B$ |
| $P(A, B)$          | Joint probability    | Probability of $A$ and $B$ |
| $P(A \cap B)$      | Joint probability    | Same as above              |
| $P(A \cup B)$      | Probability of union | Either $A$ or $B$          |
| $\bar{A}$ or $A^c$ | Complement           | Not $A$                    |

### Random Variables

| Symbol                              | Name             | Meaning                          |
| ----------------------------------- | ---------------- | -------------------------------- | --------------------- |
| $X$, $Y$, $Z$                       | Random variables | Uppercase letters                |
| $x$, $y$, $z$                       | Realizations     | Specific values                  |
| $p(x)$ or $P(X=x)$                  | PMF              | Probability mass function        |
| $f(x)$                              | PDF              | Probability density function     |
| $F(x)$                              | CDF              | Cumulative distribution function |
| $X \sim \mathcal{N}(\mu, \sigma^2)$ | Distribution     | $X$ follows normal distribution  |
| $X \perp Y$                         | Independence     | $X$ and $Y$ are independent      |
| $X \perp Y                          | Z$               | Conditional independence         | Independent given $Z$ |

### Expectation and Moments

| Symbol             | Name               | Definition                                      |
| ------------------ | ------------------ | ----------------------------------------------- |
| $E[X]$             | Expectation        | $\sum_x x \cdot P(x)$ or $\int x \cdot f(x) dx$ |
| $\mathbb{E}[X]$    | Expectation        | Alternative notation                            |
| $\mu$              | Mean               | $E[X]$                                          |
| $\text{Var}(X)$    | Variance           | $E[(X - \mu)^2]$                                |
| $\sigma^2$         | Variance           | Alternative notation                            |
| $\sigma$           | Standard deviation | $\sqrt{\text{Var}(X)}$                          |
| $\text{Cov}(X, Y)$ | Covariance         | $E[(X-\mu_X)(Y-\mu_Y)]$                         |
| $\rho_{XY}$        | Correlation        | $\frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$     |
| $\Sigma$           | Covariance matrix  |                                                 |

### Common Distributions

| Distribution        | Notation                                            |
| ------------------- | --------------------------------------------------- |
| Normal/Gaussian     | $\mathcal{N}(\mu, \sigma^2)$                        |
| Multivariate Normal | $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$             |
| Uniform             | $\text{Uniform}(a, b)$ or $U(a, b)$                 |
| Bernoulli           | $\text{Bernoulli}(p)$ or $\text{Bern}(p)$           |
| Binomial            | $\text{Binomial}(n, p)$ or $B(n, p)$                |
| Poisson             | $\text{Poisson}(\lambda)$ or $\text{Pois}(\lambda)$ |
| Exponential         | $\text{Exp}(\lambda)$                               |
| Gamma               | $\text{Gamma}(\alpha, \beta)$                       |
| Beta                | $\text{Beta}(\alpha, \beta)$                        |

---

## Linear Algebra Specific

### Eigenvalues and Eigenvectors

| Symbol       | Meaning                        |
| ------------ | ------------------------------ |
| $\lambda$    | Eigenvalue                     |
| $\mathbf{v}$ | Eigenvector                    |
| $\Lambda$    | Diagonal matrix of eigenvalues |
| $V$          | Matrix of eigenvectors         |
| $\sigma_i$   | Singular value                 |
| $U, V$       | Left/right singular vectors    |

### Norms and Inner Products

| Symbol                         | Meaning                                 |
| ------------------------------ | --------------------------------------- |
| $\langle \cdot, \cdot \rangle$ | Inner product                           |
| $\|\cdot\|$                    | Norm                                    |
| $\perp$                        | Orthogonal to                           |
| $\mathbf{u} \perp \mathbf{v}$  | $\mathbf{u}$ orthogonal to $\mathbf{v}$ |

### Decompositions

| Name               | Notation               |
| ------------------ | ---------------------- |
| Eigendecomposition | $A = V \Lambda V^{-1}$ |
| SVD                | $A = U \Sigma V^T$     |
| QR                 | $A = QR$               |
| Cholesky           | $A = LL^T$             |
| LU                 | $A = LU$               |

---

## Greek Letters

### Commonly Used in ML

| Letter     | Name             | Common Use                        |
| ---------- | ---------------- | --------------------------------- |
| $\alpha$   | alpha            | Learning rate, significance level |
| $\beta$    | beta             | Momentum coefficient, parameters  |
| $\gamma$   | gamma            | Discount factor, regularization   |
| $\delta$   | delta            | Small change, error               |
| $\epsilon$ | epsilon          | Small positive number, noise      |
| $\eta$     | eta              | Learning rate                     |
| $\theta$   | theta            | Model parameters                  |
| $\lambda$  | lambda           | Eigenvalue, regularization        |
| $\mu$      | mu               | Mean                              |
| $\nu$      | nu               | Degrees of freedom                |
| $\pi$      | pi               | Policy (RL), 3.14159...           |
| $\rho$     | rho              | Correlation, density              |
| $\sigma$   | sigma            | Standard deviation, activation    |
| $\tau$     | tau              | Temperature, time constant        |
| $\phi$     | phi              | Feature function, angle           |
| $\psi$     | psi              | Activation, wave function         |
| $\omega$   | omega            | Frequency, weights                |
| $\Sigma$   | Sigma (capital)  | Covariance matrix, summation      |
| $\Omega$   | Omega (capital)  | Sample space                      |
| $\Gamma$   | Gamma (capital)  | Gamma function                    |
| $\Delta$   | Delta (capital)  | Change, difference                |
| $\Theta$   | Theta (capital)  | Parameter space                   |
| $\Lambda$  | Lambda (capital) | Diagonal matrix                   |
| $\Phi$     | Phi (capital)    | Feature matrix                    |

---

## Information Theory

| Symbol | Name | Definition |
|--------|------|------------|
| $H(X)$ | Entropy | $-\sum p(x) \log p(x)$ |
| $H(X,Y)$ | Joint entropy | $-\sum_{x,y} p(x,y) \log p(x,y)$ |
| $H(X\|Y)$ | Conditional entropy | $H(X,Y) - H(Y)$ |
| $D_{KL}(p\|\|q)$ | KL divergence | $\sum p(x) \log\frac{p(x)}{q(x)}$ |
| $H(p,q)$ | Cross-entropy | $-\sum p(x) \log q(x)$ |
| $I(X;Y)$ | Mutual information | $H(X) - H(X\|Y)$ |

> **Note:** $\log$ typically means $\ln$ (nats) or $\log_2$ (bits) depending on context.

---

## Graph Theory

| Symbol | Meaning |
|--------|----------|
| $G = (V, E)$ | Graph with vertices $V$ and edges $E$ |
| $A$ | Adjacency matrix |
| $D$ | Degree matrix |
| $L = D - A$ | Graph Laplacian |
| $\mathcal{N}(v)$ | Neighborhood of vertex $v$ |
| $\deg(v)$ | Degree of vertex $v$ |
| $d(u,v)$ | Distance between $u$ and $v$ |
| $\lambda_i$ | $i$-th eigenvalue of Laplacian |
| $K_n$ | Complete graph on $n$ vertices |
| $G'$ (or $\bar{G}$) | Complement graph |

---

## Norms and Spaces

| Symbol | Meaning |
|--------|----------|
| $\|\mathbf{x}\|_p$ | $\ell^p$ norm |
| $\|A\|_F$ | Frobenius norm |
| $\|A\|_2$ | Spectral norm (operator norm) |
| $\|A\|_*$ | Nuclear norm (trace norm) |
| $\langle x, y \rangle$ | Inner product |
| $\mathcal{H}$ | Hilbert space |
| $\mathcal{H}_K$ | Reproducing Kernel Hilbert Space |
| $K(x,x')$ | Kernel function |
| $L^p$ | Lebesgue space of $p$-integrable functions |
| $\ell^2$ | Space of square-summable sequences |
| $(V, \|\cdot\|)$ | Normed space |

---

## Optimization

| Symbol | Meaning |
|--------|----------|
| $\arg\min_x f(x)$ | Value of $x$ minimizing $f$ |
| $\arg\max_x f(x)$ | Value of $x$ maximizing $f$ |
| $\text{s.t.}$ | Subject to (constraint) |
| $\mathcal{L}(x, \lambda)$ | Lagrangian |
| $\lambda_i$ | Lagrange multiplier |
| $\mu$ | KKT dual variable |
| $\nabla^2 f$ or $H$ | Hessian matrix |
| $\succeq 0$ | Positive semi-definite |
| $\succ 0$ | Positive definite |
| $f^*$ | Optimal value |
| $x^*$ | Optimal solution |
| $\mathcal{C}$ | Constraint set |

---

## Common Conventions in ML

### Data and Models

| Symbol             | Meaning              |
| ------------------ | -------------------- |
| $\mathbf{x}$       | Input/feature vector |
| $\mathbf{y}$       | Target/output vector |
| $\hat{\mathbf{y}}$ | Predicted output     |
| $\mathbf{X}$       | Design matrix (data) |
| $\mathbf{w}$       | Weight vector        |
| $\mathbf{b}$       | Bias vector          |
| $W$                | Weight matrix        |
| $\theta$           | All parameters       |
| $n$                | Number of samples    |
| $d$ or $p$         | Number of features   |
| $m$                | Number of outputs    |
| $k$                | Number of classes    |

### Neural Networks

| Symbol            | Meaning                     |
| ----------------- | --------------------------- |
| $L$               | Loss function               |
| $\mathcal{L}$     | Loss function (script)      |
| $J$               | Cost function               |
| $a^{[l]}$         | Activation at layer $l$     |
| $z^{[l]}$         | Pre-activation at layer $l$ |
| $W^{[l]}$         | Weights at layer $l$        |
| $b^{[l]}$         | Bias at layer $l$           |
| $\sigma$          | Activation function         |
| $\nabla_\theta L$ | Gradient of loss            |

### Optimization

| Symbol    | Meaning                 |
| --------- | ----------------------- |
| $\eta$    | Learning rate           |
| $t$       | Time step / iteration   |
| $\nabla$  | Gradient operator       |
| $\alpha$  | Step size               |
| $\lambda$ | Regularization strength |

---

## Quick Reference Card

```
VECTORS & MATRICES
──────────────────
Bold lowercase = vector: x, a, v
Uppercase = matrix: A, X, W
Subscript = element: xᵢ, Aᵢⱼ
Transpose: Aᵀ
Inverse: A⁻¹
Norm: ‖x‖

CALCULUS
────────
Derivative: df/dx, f'(x)
Partial: ∂f/∂x
Gradient: ∇f
Jacobian: J
Hessian: H

PROBABILITY
───────────
Probability: P(A)
Conditional: P(A|B)
Expectation: E[X]
Variance: Var(X)
Normal: 𝒩(μ, σ²)

COMMON
──────
Sum: Σ
Product: Π
Approximately: ≈
Proportional: ∝
For all: ∀
Exists: ∃
Implies: ⟹
If and only if: ⟺
```

---

## Reading ML Papers: Notation Tips

Common notation patterns you'll encounter:

| Paper Convention | Meaning |
|-----------------|----------|
| $\theta$ | Model parameters |
| $\phi$ | Variational/encoder parameters |
| $\psi$ | Auxiliary parameters |
| $\mathcal{D}$ | Dataset |
| $\mathcal{L}$ | Loss or ELBO |
| $p_\theta(x)$ | Model distribution |
| $q_\phi(z\|x)$ | Approximate posterior |
| $\mathbb{E}_{q}[\cdot]$ | Expectation under $q$ |
| $\text{KL}(q\|\|p)$ | KL divergence |
| $\odot$ | Element-wise (Hadamard) product |
| $\otimes$ | Kronecker/tensor product |
| $[N]$ | Set $\{1, 2, \ldots, N\}$ |
| $\mathbb{1}[\cdot]$ | Indicator function |
| $\propto$ | Proportional to |
| $\sim$ | Distributed as (e.g., $x \sim \mathcal{N}(0,1)$) |

---

_Understanding notation is half the battle in reading ML papers!_ 📚
