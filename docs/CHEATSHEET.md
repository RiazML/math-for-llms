# 📋 Mathematics for AI/ML - Cheatsheet

> All essential formulas in one place. Print this out or keep it handy!

---

## Table of Contents

1. [Linear Algebra](#linear-algebra)
2. [Calculus](#calculus)
3. [Probability & Statistics](#probability--statistics)
4. [Optimization](#optimization)
5. [Information Theory](#information-theory)
6. [Numerical Methods](#numerical-methods)
7. [Graph Theory](#graph-theory)
8. [Functional Analysis & Kernels](#functional-analysis--kernels)

---

## Linear Algebra

### Vectors

| Operation   | Formula                                                                                                   | NumPy                   |
| ----------- | --------------------------------------------------------------------------------------------------------- | ----------------------- |
| Dot Product | $\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$                                                    | `np.dot(a, b)`          |
| Magnitude   | $\|\|\mathbf{a}\|\| = \sqrt{\sum_{i=1}^{n} a_i^2}$                                                        | `np.linalg.norm(a)`     |
| Unit Vector | $\hat{\mathbf{a}} = \frac{\mathbf{a}}{\|\|\mathbf{a}\|\|}$                                                | `a / np.linalg.norm(a)` |
| Angle       | $\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\|\mathbf{a}\|\| \|\|\mathbf{b}\|\|}$                  | `np.arccos(...)`        |
| Projection  | $\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\|\mathbf{b}\|\|^2}\mathbf{b}$ |                         |

### Matrices

| Operation       | Formula                           | NumPy                        |
| --------------- | --------------------------------- | ---------------------------- |
| Matrix Multiply | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ | `A @ B` or `np.matmul(A, B)` |
| Transpose       | $(A^T)_{ij} = A_{ji}$             | `A.T`                        |
| Inverse         | $AA^{-1} = I$                     | `np.linalg.inv(A)`           |
| Determinant     | $\det(A)$                         | `np.linalg.det(A)`           |
| Trace           | $\text{tr}(A) = \sum_i A_{ii}$    | `np.trace(A)`                |
| Rank            | $\text{rank}(A)$                  | `np.linalg.matrix_rank(A)`   |

### Eigendecomposition

$$A\mathbf{v} = \lambda\mathbf{v}$$

| Property                | Formula                              |
| ----------------------- | ------------------------------------ |
| Characteristic Equation | $\det(A - \lambda I) = 0$            |
| Diagonalization         | $A = PDP^{-1}$ where $D$ is diagonal |
| Matrix Power            | $A^n = PD^nP^{-1}$                   |

```python
eigenvalues, eigenvectors = np.linalg.eig(A)
```

### Singular Value Decomposition (SVD)

$$A = U\Sigma V^T$$

| Component | Description                | Shape        |
| --------- | -------------------------- | ------------ |
| $U$       | Left singular vectors      | $m \times m$ |
| $\Sigma$  | Singular values (diagonal) | $m \times n$ |
| $V^T$     | Right singular vectors     | $n \times n$ |

```python
U, S, Vt = np.linalg.svd(A)
```

### Special Matrices

| Type                   | Property                                                    |
| ---------------------- | ----------------------------------------------------------- |
| Symmetric              | $A = A^T$                                                   |
| Orthogonal             | $A^T A = I$                                                 |
| Positive Definite      | $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq 0$ |
| Positive Semi-definite | $\mathbf{x}^T A \mathbf{x} \geq 0$ for all $\mathbf{x}$     |

---

## Calculus

### Derivatives

| Function                         | Derivative                                            |
| -------------------------------- | ----------------------------------------------------- |
| $x^n$                            | $nx^{n-1}$                                            |
| $e^x$                            | $e^x$                                                 |
| $\ln(x)$                         | $\frac{1}{x}$                                         |
| $\sin(x)$                        | $\cos(x)$                                             |
| $\cos(x)$                        | $-\sin(x)$                                            |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$                              |
| $\tanh(x)$                       | $1 - \tanh^2(x)$                                      |
| $\text{ReLU}(x)$                 | $\begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$ |

### Differentiation Rules

| Rule     | Formula                                             |
| -------- | --------------------------------------------------- |
| Sum      | $(f + g)' = f' + g'$                                |
| Product  | $(fg)' = f'g + fg'$                                 |
| Quotient | $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$ |
| Chain    | $(f \circ g)' = f'(g(x)) \cdot g'(x)$               |

### Gradient

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### Jacobian Matrix

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

### Hessian Matrix

$$H = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

### Taylor Series

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

Common expansions at $a=0$:

- $e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$
- $\sin(x) = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$
- $\cos(x) = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$

---

## Probability & Statistics

### Basic Probability

| Formula                                   | Description         |
| ----------------------------------------- | ------------------- |
| $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Addition rule       |
| $P(A \cap B) = P(A)P(B\|A)$               | Multiplication rule |
| $P(A\|B) = \frac{P(B\|A)P(A)}{P(B)}$      | Bayes' theorem      |

### Expectation & Variance

| Measure     | Discrete                       | Continuous                          |
| ----------- | ------------------------------ | ----------------------------------- |
| Expectation | $E[X] = \sum_x x \cdot P(x)$   | $E[X] = \int x \cdot f(x)dx$        |
| Variance    | $\text{Var}(X) = E[(X-\mu)^2]$ | $\text{Var}(X) = E[X^2] - (E[X])^2$ |

### Common Distributions

| Distribution | PDF/PMF                                                       | Mean                | Variance              |
| ------------ | ------------------------------------------------------------- | ------------------- | --------------------- |
| Bernoulli    | $P(X=1) = p$                                                  | $p$                 | $p(1-p)$              |
| Binomial     | $\binom{n}{k}p^k(1-p)^{n-k}$                                  | $np$                | $np(1-p)$             |
| Poisson      | $\frac{\lambda^k e^{-\lambda}}{k!}$                           | $\lambda$           | $\lambda$             |
| Uniform      | $\frac{1}{b-a}$                                               | $\frac{a+b}{2}$     | $\frac{(b-a)^2}{12}$  |
| Gaussian     | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$               | $\sigma^2$            |
| Exponential  | $\lambda e^{-\lambda x}$                                      | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

### Multivariate Gaussian

$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

### Covariance & Correlation

| Measure           | Formula                                                 |
| ----------------- | ------------------------------------------------------- |
| Covariance        | $\text{Cov}(X,Y) = E[(X-\mu_X)(Y-\mu_Y)]$               |
| Correlation       | $\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$ |
| Covariance Matrix | $\Sigma_{ij} = \text{Cov}(X_i, X_j)$                    |

### Maximum Likelihood Estimation

$$\hat{\theta}_{MLE} = \arg\max_\theta \prod_{i=1}^n p(x_i|\theta) = \arg\max_\theta \sum_{i=1}^n \log p(x_i|\theta)$$

### Maximum A Posteriori

$$\hat{\theta}_{MAP} = \arg\max_\theta p(\theta|X) = \arg\max_\theta p(X|\theta)p(\theta)$$

---

## Optimization

### Gradient Descent

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

### Stochastic Gradient Descent (SGD)

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t; x_i, y_i)$$

### SGD with Momentum

$$v_{t+1} = \gamma v_t + \eta \nabla_\theta L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

### Adam Optimizer

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Default: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

### Convexity Conditions

| Condition    | Meaning                                                           |
| ------------ | ----------------------------------------------------------------- |
| $f$ convex   | $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$ |
| First-order  | $f(y) \geq f(x) + \nabla f(x)^T(y-x)$                             |
| Second-order | $\nabla^2 f(x) \succeq 0$ (positive semi-definite)                |

### Lagrangian

$$\mathcal{L}(x, \lambda) = f(x) + \sum_i \lambda_i g_i(x)$$

KKT Conditions:

1. Stationarity: $\nabla_x \mathcal{L} = 0$
2. Primal feasibility: $g_i(x) \leq 0$
3. Dual feasibility: $\lambda_i \geq 0$
4. Complementary slackness: $\lambda_i g_i(x) = 0$

---

## Information Theory

### Entropy

$$H(X) = -\sum_x p(x) \log p(x) = -E[\log p(X)]$$

### Cross-Entropy

$$H(p, q) = -\sum_x p(x) \log q(x)$$

### KL Divergence

$$D_{KL}(p || q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)$$

### Mutual Information

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

### Information Gain

$$IG(S, A) = H(S) - \sum_v \frac{|S_v|}{|S|} H(S_v)$$

---

## Numerical Methods

### Floating Point
- Machine epsilon: $\epsilon_{mach} \approx 2.2 \times 10^{-16}$ (float64)
- Condition number: $\kappa(A) = \|A\| \cdot \|A^{-1}\|$

### Numerical Differentiation
| Method | Formula | Error |
|--------|---------|-------|
| Forward | $f'(x) \approx \frac{f(x+h)-f(x)}{h}$ | $O(h)$ |
| Central | $f'(x) \approx \frac{f(x+h)-f(x-h)}{2h}$ | $O(h^2)$ |

### Numerical Integration
- Trapezoidal: $\int_a^b f(x)dx \approx \frac{h}{2}[f(a) + 2\sum f(x_i) + f(b)]$
- Simpson's: $\int_a^b f(x)dx \approx \frac{h}{3}[f(a) + 4\sum_{\text{odd}} + 2\sum_{\text{even}} + f(b)]$

---

## Graph Theory

### Basics
| Concept | Formula |
|---------|--------|
| Degree | $\deg(v) = \sum_u A_{vu}$ |
| Adjacency | $A_{ij} = 1$ if edge $(i,j)$ exists |
| Laplacian | $L = D - A$ |
| Norm. Laplacian | $\mathcal{L} = D^{-1/2}LD^{-1/2}$ |

### Spectral Properties
- Algebraic connectivity: $\lambda_2(L)$
- Number of components = multiplicity of $\lambda = 0$

### PageRank
$$\mathbf{r} = \alpha M\mathbf{r} + \frac{(1-\alpha)}{n}\mathbf{1}$$

### GNN Layer (GCN)
$$H^{(l+1)} = \sigma(\hat{A}H^{(l)}W^{(l)})$$
where $\hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$

---

## Functional Analysis & Kernels

### Norms
| Norm | Formula | Use |
|------|---------|-----|
| $\ell^1$ | $\sum|x_i|$ | Sparsity (Lasso) |
| $\ell^2$ | $\sqrt{\sum x_i^2}$ | Weight decay (Ridge) |
| $\ell^\infty$ | $\max|x_i|$ | Adversarial robustness |
| Frobenius | $\sqrt{\sum a_{ij}^2}$ | Matrix regularization |
| Nuclear | $\sum \sigma_i$ | Low-rank |

### Kernel Functions
| Kernel | Formula | Parameters |
|--------|---------|------------|
| Linear | $K(x,x') = x^Tx'$ | — |
| Polynomial | $K(x,x') = (x^Tx' + c)^d$ | $c, d$ |
| RBF/Gaussian | $K(x,x') = \exp(-\|x-x'\|^2/2\sigma^2)$ | $\sigma$ |

### Reproducing Property
$$f(x) = \langle f, K(x, \cdot)\rangle_{\mathcal{H}}$$

### Representer Theorem
$$f^* = \sum_{i=1}^n \alpha_i K(x_i, \cdot)$$

---

## Neural Network Math

### Softmax

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

### Cross-Entropy Loss

$$L = -\sum_i y_i \log(\hat{y}_i)$$

### Binary Cross-Entropy

$$L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

### MSE Loss

$$L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

### Backpropagation

For layer $l$:
$$\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$
$$\frac{\partial L}{\partial W^l} = \delta^l (a^{l-1})^T$$
$$\frac{\partial L}{\partial b^l} = \delta^l$$

### Batch Normalization

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

### Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

---

## Quick Reference: PyTorch

```python
import torch

# Tensor operations
torch.matmul(A, B)          # Matrix multiply (or A @ B)
torch.linalg.inv(A)         # Inverse
torch.linalg.eig(A)         # Eigendecomposition
torch.linalg.svd(A)         # SVD
torch.linalg.norm(x)        # Norm

# Autograd
x = torch.tensor([1.0], requires_grad=True)
y = x**2 + 3*x
y.backward()                # Compute gradients
x.grad                      # Access gradient

# Common layers
torch.nn.Linear(in, out)
torch.nn.ReLU()
torch.nn.Softmax(dim=-1)
torch.nn.CrossEntropyLoss()
torch.nn.BatchNorm1d(features)
```

---

## Quick Reference: NumPy

```python
import numpy as np

# Linear Algebra
np.dot(a, b)           # Dot product
np.matmul(A, B)        # Matrix multiplication (or A @ B)
np.linalg.inv(A)       # Inverse
np.linalg.det(A)       # Determinant
np.linalg.eig(A)       # Eigenvalues and eigenvectors
np.linalg.svd(A)       # Singular Value Decomposition
np.linalg.norm(x)      # Vector/matrix norm
np.linalg.solve(A, b)  # Solve Ax = b

# Statistics
np.mean(x)             # Mean
np.std(x)              # Standard deviation
np.var(x)              # Variance
np.cov(X)              # Covariance matrix
np.corrcoef(x, y)      # Correlation coefficient

# Random
np.random.normal(mu, sigma, size)  # Gaussian samples
np.random.uniform(low, high, size) # Uniform samples
```

---

_Keep learning, keep practicing!_ 📚
