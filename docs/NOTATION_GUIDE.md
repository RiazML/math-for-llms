# Notation Guide

> **Authority:** This document is the single source of truth for all mathematical notation
> used throughout this repository. Every section, notebook, and exercise MUST conform
> to these conventions. When conventions conflict with intuition, this guide wins.

---

## Design Principles

This guide follows the conventions of Goodfellow, Bengio & Courville (2016) *Deep Learning*,
Horn & Johnson (2013) *Matrix Analysis*, and the notation standards of NeurIPS/ICML/ICLR.
Where conventions conflict, the choice that minimises ambiguity in an ML context is adopted.

**Three rules that override everything else:**

1. A symbol's meaning must be determinable from context within three lines of its first use.
2. The same symbol never denotes two different objects in the same section.
3. Prefer established ML-paper conventions over personal preference.

---

## 1. Object Hierarchy

### 1.1 Type-to-Notation Map

| Object | Notation style | LaTeX command | Example |
| --- | --- | --- |---|
| Scalar | Italic lowercase | `a`, `\alpha` | $x \in \mathbb{R}$ |
| Vector | **Bold** lowercase | `\mathbf{x}` | $\mathbf{x} \in \mathbb{R}^n$ |
| Matrix | Uppercase italic | `A`, `W` | $A \in \mathbb{R}^{m \times n}$ |
| Tensor (order $\ge 3$) | Calligraphic | `\mathcal{X}` | $\mathcal{X} \in \mathbb{R}^{d_1 \times \cdots \times d_k}$ |
| Random variable | Uppercase italic | `X`, `Z` | $X \sim \mathcal{N}(0,1)$ |
| Set / space | Calligraphic | `\mathcal{D}`, `\mathcal{H}` | $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}$ |
| Number field | Blackboard bold | `\mathbb{R}`, `\mathbb{C}` | $\mathbb{R}^n$, $\mathbb{C}^{m \times n}$ |
| Distribution | Calligraphic | `\mathcal{N}`, `\mathcal{U}` | $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ |

> **Do:** `\mathbf{x}` for vectors.
> **Do not:** `\vec{x}` (physics convention), `\underline{x}` (typewriter convention),
> or plain `x` when $x$ is a vector — these are all errors in this codebase.

### 1.2 Indexing

| What | Notation | LaTeX | Meaning |
| --- | --- | --- |---|
| Vector component | $x_i$ | `x_i` | $i$-th scalar, 1-indexed by default |
| Matrix entry | $A_{ij}$ | `A_{ij}` | row $i$, column $j$ |
| Row $i$ | $A_{i,:}$ | `A_{i,:}` | as a row vector |
| Column $j$ | $A_{:,j}$ | `A_{:,j}` | as a column vector |
| Data sample | $\mathbf{x}^{(i)}$ | `\mathbf{x}^{(i)}` | parentheses distinguish from power |
| Layer index | $\mathbf{h}^{[l]}$ | `\mathbf{h}^{[l]}` | square brackets distinguish from data |
| Time step | $\mathbf{h}_t$ | `\mathbf{h}_t` | subscript, used in sequences |

> **Convention:** $(i)$ = data sample, $[l]$ = network layer, $_t$ = time.
> These three scopes must never share the same superscript/subscript style.

---

## 2. Number Fields and Spaces

| Symbol | LaTeX | Meaning |
| --- | --- | --- |
| $\mathbb{N}$ | `\mathbb{N}` | $\{0, 1, 2, \ldots\}$ (ISO 80000-2; 0 is included) |
| $\mathbb{Z}$ | `\mathbb{Z}` | All integers $\{\ldots,-1,0,1,\ldots\}$ |
| $\mathbb{R}$ | `\mathbb{R}` | Real line |
| $\mathbb{R}^n$ | `\mathbb{R}^n` | Euclidean $n$-space (column vectors by default) |
| $\mathbb{R}^{m \times n}$ | `\mathbb{R}^{m \times n}` | Real $m \times n$ matrices |
| $\mathbb{C}$ | `\mathbb{C}` | Complex numbers |
| $\mathbb{S}^n$ | `\mathbb{S}^n` | $n \times n$ real symmetric matrices |
| $\mathbb{S}^n_+$ | `\mathbb{S}^n_+` | Positive semidefinite (PSD) cone |
| $\mathbb{S}^n_{++}$ | `\mathbb{S}^n_{++}` | Strictly positive definite matrices |
| $[n]$ | `[n]` | Index set $\{1, 2, \ldots, n\}$ |

---

## 3. Linear Algebra

### 3.1 Vectors

| Symbol | LaTeX | Meaning |
| --- | --- | --- |
| $\mathbf{x}$ | `\mathbf{x}` | Column vector (default orientation) |
| $\mathbf{x}^\top$ | `\mathbf{x}^\top` | Transpose — use `^\top`, never `^T` in display math |
| $\mathbf{e}_i$ | `\mathbf{e}_i` | $i$-th standard basis vector |
| $\mathbf{0}$ | `\mathbf{0}` | Zero vector |
| $\mathbf{1}$ | `\mathbf{1}` | All-ones vector |
| $\langle \mathbf{x}, \mathbf{y} \rangle$ | `\langle \mathbf{x}, \mathbf{y} \rangle` | Inner product (preferred over dot) |
| $\mathbf{x} \odot \mathbf{y}$ | `\mathbf{x} \odot \mathbf{y}` | Hadamard (element-wise) product |

> **Transpose:** Use `^\top` in all display equations. Reserve `^T` for inline code
> comments only. Rationale: `^T` is ambiguous with transposing a random variable $T$.

### 3.2 Matrices

| Symbol | LaTeX | Meaning |
| --- | --- | --- |
| $I_n$ | `I_n` | $n \times n$ identity (subscript when size matters) |
| $A^\top$ | `A^\top` | Matrix transpose |
| $A^{-1}$ | `A^{-1}` | Matrix inverse (only for square, non-singular $A$) |
| $A^\dagger$ | `A^\dagger` | Moore-Penrose pseudoinverse |
| $A^*$ | `A^*` | Conjugate transpose (Hermitian adjoint) |
| $\operatorname{tr}(A)$ | `\operatorname{tr}(A)` | Trace — use `\operatorname`, not `\text` |
| $\det(A)$ | `\det(A)` | Determinant |
| $\operatorname{rank}(A)$ | `\operatorname{rank}(A)` | Rank |
| $\operatorname{diag}(\mathbf{x})$ | `\operatorname{diag}(\mathbf{x})` | Diagonal matrix from vector |
| $A \otimes B$ | `A \otimes B` | Kronecker product |
| $A \succ 0$ | `A \succ 0` | Positive definite |
| $A \succeq 0$ | `A \succeq 0` | Positive semidefinite |

### 3.3 Norms

| Symbol | LaTeX | Definition | Use case |
| --- | --- | --- | --- |
| $\lVert \mathbf{x} \rVert_2$ | `\lVert \mathbf{x} \rVert_2` | $\sqrt{\sum x_i^2}$ | Default vector norm |
| $\lVert \mathbf{x} \rVert_1$ | `\lVert \mathbf{x} \rVert_1` | $\sum \lvert x_i \rvert$ | Sparsity, L1 regularisation |
| $\lVert \mathbf{x} \rVert_p$ | `\lVert \mathbf{x} \rVert_p` | $(\sum \lvert x_i \rvert^p)^{1/p}$ | General $\ell^p$ norm |
| $\lVert \mathbf{x} \rVert_\infty$ | `\lVert \mathbf{x} \rVert_\infty` | $\max_i \lvert x_i \rvert$ | Chebyshev / max norm |
| $\lVert A \rVert_F$ | `\lVert A \rVert_F` | $\sqrt{\sum_{i,j} A_{ij}^2}$ | Frobenius norm |
| $\lVert A \rVert_2$ | `\lVert A \rVert_2` | $\sigma_{\max}(A)$ | Spectral norm (operator norm) |
| $\lVert A \rVert_*$ | `\lVert A \rVert_*` | $\sum_i \sigma_i(A)$ | Nuclear norm (trace norm) |

> **Norm delimiters:** Always use `\lVert \rVert` for norms and `\lvert \rvert` for
> absolute values — never bare `\|` or `|`. This ensures correct spacing in all renderers.

### 3.4 Decompositions

| Name | Canonical form | Conditions |
| --- | --- | --- |
| Eigendecomposition | $A = Q \Lambda Q^{-1}$ | $A$ diagonalisable |
| Spectral theorem | $A = Q \Lambda Q^\top$ | $A$ real symmetric |
| SVD | $A = U \Sigma V^\top$ | $A \in \mathbb{R}^{m \times n}$, always exists |
| QR | $A = QR$ | $Q$ orthogonal, $R$ upper triangular |
| Cholesky | $A = LL^\top$ | $A \succ 0$ |
| LU | $A = LU$ | Square, non-singular |

> **SVD convention:** $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$,
> $V \in \mathbb{R}^{n \times n}$. Singular values $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$
> are always in non-increasing order. Left singular vectors = columns of $U$;
> right singular vectors = columns of $V$.

---

## 4. Calculus and Optimisation

### 4.1 Derivatives

| Symbol | LaTeX | Meaning |
| --- | --- | --- |
| $\frac{df}{dx}$ | `\frac{df}{dx}` | Derivative of scalar $f$ w.r.t. scalar $x$ |
| $f'(x)$ | `f'(x)` | Prime notation — only for univariate functions |
| $\frac{\partial f}{\partial x_i}$ | `\frac{\partial f}{\partial x_i}` | Partial derivative |
| $\nabla_{\mathbf{x}} f$ | `\nabla_{\mathbf{x}} f` | Gradient: $\mathbf{g} \in \mathbb{R}^n$ where $g_i = \partial f/\partial x_i$ |
| $J_f(\mathbf{x})$ | `J_f(\mathbf{x})` | Jacobian of $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$; $J \in \mathbb{R}^{m \times n}$ |
| $H_f(\mathbf{x})$ | `H_f(\mathbf{x})` | Hessian: $(H)_{ij} = \partial^2 f / \partial x_i \partial x_j$ |
| $\frac{\partial \mathcal{L}}{\partial \theta}$ | `\frac{\partial \mathcal{L}}{\partial \theta}` | Gradient of loss w.r.t. parameters — standard ML form |

> **Gradient orientation:** $\nabla f \in \mathbb{R}^n$ is always a **column vector**.
> This is the convention of Nocedal & Wright (2006) and all major ML frameworks.
> The gradient points in the direction of steepest ascent.

### 4.2 Optimisation Symbols

| Symbol | LaTeX | Meaning |
| --- | --- | --- |
| $\arg\min_{\mathbf{x}} f(\mathbf{x})$ | `\arg\min_{\mathbf{x}}` | Minimiser of $f$ |
| $\arg\max_{\mathbf{x}} f(\mathbf{x})$ | `\arg\max_{\mathbf{x}}` | Maximiser of $f$ |
| $f^*$ | `f^*` | Optimal (minimum) value |
| $\mathbf{x}^*$ | `\mathbf{x}^*` | Optimal solution |
| $\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})$ | `\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda})` | Lagrangian |
| $\boldsymbol{\lambda}$ | `\boldsymbol{\lambda}` | Lagrange multipliers (vector — use bold) |
| $\text{s.t.}$ | `\text{s.t.}` | Subject to |
| $\eta$ | `\eta` | Learning rate |
| $\theta_t$ | `\theta_t` | Parameters at step $t$ |

---

## 5. Probability and Statistics

### 5.1 Core Symbols

| Symbol | LaTeX | Meaning |
| --- | --- | --- |
| $P(A)$ | `P(A)` | Probability of event $A$ |
| $p(\mathbf{x})$ | `p(\mathbf{x})` | Probability density / mass function |
| $P(A \mid B)$ | `P(A \mid B)` | Conditional probability — use `\mid`, not `\|` |
| $X \sim p$ | `X \sim p` | $X$ is distributed according to $p$ |
| $X \perp\!\!\!\perp Y$ | `X \perp\!\!\!\perp Y` | Statistical independence |
| $\mathbb{E}[X]$ | `\mathbb{E}[X]` | Expectation — use blackboard bold $\mathbb{E}$ |
| $\mathbb{E}_{\mathbf{x} \sim p}[f(\mathbf{x})]$ | `\mathbb{E}_{\mathbf{x} \sim p}[f(\mathbf{x})]` | Expectation specifying distribution |
| $\operatorname{Var}(X)$ | `\operatorname{Var}(X)` | Variance |
| $\operatorname{Cov}(X, Y)$ | `\operatorname{Cov}(X, Y)` | Covariance |
| $\Sigma$ | `\Sigma` | Covariance matrix — capital sigma, not `\sum` |

> **Conditional probability delimiter:** Always `P(A \mid B)` with `\mid`.
> Never `P(A|B)` — the bare pipe `|` has incorrect spacing in LaTeX renderers.

### 5.2 Standard Distributions

| Distribution | Notation | LaTeX |
| --- | --- | --- |
| Normal | $\mathcal{N}(\mu, \sigma^2)$ | `\mathcal{N}(\mu, \sigma^2)` |
| Multivariate Normal | $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ | `\mathcal{N}(\boldsymbol{\mu}, \Sigma)` |
| Uniform | $\mathcal{U}(a, b)$ | `\mathcal{U}(a, b)` |
| Bernoulli | $\operatorname{Bern}(p)$ | `\operatorname{Bern}(p)` |
| Categorical | $\operatorname{Cat}(\mathbf{p})$ | `\operatorname{Cat}(\mathbf{p})` |
| Dirichlet | $\operatorname{Dir}(\boldsymbol{\alpha})$ | `\operatorname{Dir}(\boldsymbol{\alpha})` |

> **Covariance matrix:** The second argument of $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$
> is the **covariance matrix**, not the precision matrix. When using precision, write
> $\mathcal{N}(\boldsymbol{\mu}, \Lambda^{-1})$ explicitly.

---

## 6. Information Theory

| Symbol | LaTeX | Definition | Unit |
| --- | --- | --- | --- |
| $H(X)$ | `H(X)` | $-\sum_x p(x) \log p(x)$ | nats (log = ln) or bits (log = log₂) |
| $H(X \mid Y)$ | `H(X \mid Y)` | $H(X,Y) - H(Y)$ | Conditional entropy |
| $D_{\mathrm{KL}}(p \| q)$ | `D_{\mathrm{KL}}(p \| q)` | $\sum_x p(x) \log \frac{p(x)}{q(x)}$ | Non-negative; zero iff $p = q$ |
| $H(p, q)$ | `H(p, q)` | $-\sum_x p(x) \log q(x)$ | Cross-entropy; $H(p,q) = H(p) + D_{\mathrm{KL}}(p\|q)$ |
| $I(X; Y)$ | `I(X; Y)` | $H(X) - H(X \mid Y)$ | Mutual information; symmetric |
| $\operatorname{PPL}$ | `\operatorname{PPL}` | $\exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log p(x_t \mid x_{<t})\right)$ | Perplexity — LLM evaluation |

> **KL divergence notation:** Write $D_{\mathrm{KL}}(p \| q)$ with `\mathrm{KL}` and
> double-bar `\|`. Read as "KL divergence **from** $q$ **to** $p$" — $p$ is the reference
> distribution. This is the convention of Kullback & Leibler (1951) and Cover & Thomas (2006).

---

## 7. Machine Learning Specifics

### 7.1 Data and Model Parameters

| Symbol | LaTeX | Meaning |
| --- | --- | --- |
| $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^n$ | `\mathcal{D}` | Dataset of $n$ examples |
| $n$ | `n` | Number of training samples |
| $d$ | `d` | Input dimension (number of features) |
| $\boldsymbol{\theta}$ | `\boldsymbol{\theta}` | All trainable parameters (vector or set) |
| $\mathbf{W}^{[l]}$ | `\mathbf{W}^{[l]}` | Weight matrix at layer $l$ |
| $\mathbf{b}^{[l]}$ | `\mathbf{b}^{[l]}` | Bias vector at layer $l$ |
| $\hat{y}$ | `\hat{y}` | Predicted output |
| $\mathcal{L}(\boldsymbol{\theta})$ | `\mathcal{L}(\boldsymbol{\theta})` | Loss function |
| $f_{\boldsymbol{\theta}}(\mathbf{x})$ | `f_{\boldsymbol{\theta}}(\mathbf{x})` | Model with parameters $\boldsymbol{\theta}$ |

### 7.2 Neural Network Layers

| Symbol | Meaning |
| --- | --- |
| $\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$ | Pre-activation at layer $l$ |
| $\mathbf{a}^{[l]} = \sigma(\mathbf{z}^{[l]})$ | Post-activation at layer $l$ |
| $\sigma$ | Generic activation function |
| $\delta^{[l]}$ | Error signal (backprop gradient) at layer $l$ |

### 7.3 Transformer-Specific

| Symbol | Meaning |
| --- | --- |
| $Q, K, V$ | Query, Key, Value matrices |
| $d_k$ | Key/query dimension (scaling: $\sqrt{d_k}$) |
| $d_{\mathrm{model}}$ | Model/embedding dimension |
| $h$ | Number of attention heads |
| $\operatorname{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$ | Softmax — always written with explicit denominator on first use |
| $\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ | Scaled dot-product attention |

---

## 8. Greek Letters — Reserved Meanings

The following assignments are fixed across the entire repository.
Do not reuse these symbols for other quantities without explicit redefinition.

| Letter | Reserved use | Context |
| --- | --- | --- |
| $\alpha$ | Learning rate (alt.), significance level | Optimisation, statistics |
| $\beta_1, \beta_2$ | Adam moment decay rates | Optimisation |
| $\gamma$ | Discount factor (RL), gradient scaling | RL, normalisation |
| $\eta$ | Learning rate (primary) | All optimisers |
| $\lambda$ | Eigenvalue; regularisation strength | Linear algebra, regularisation |
| $\mu$ | Mean; dual variable | Statistics, optimisation |
| $\sigma$ | Standard deviation; sigmoid activation | Statistics, neural networks |
| $\sigma_i$ | $i$-th singular value | SVD |
| $\tau$ | Temperature (sampling); time constant | LLM decoding, RNNs |
| $\boldsymbol{\theta}$ | All model parameters | Every ML model |
| $\phi$ | Encoder / feature-map parameters | VAE, kernels |
| $\psi$ | Auxiliary / decoder parameters | VAE |
| $\omega$ | Angular frequency; weights (secondary) | Signal processing |
| $\Sigma$ | Covariance matrix; singular value matrix | Probability, SVD |
| $\Lambda$ | Diagonal eigenvalue matrix | Spectral decomposition |
| $\Phi$ | Feature / design matrix | Kernel methods |

---

## 9. Common LaTeX Pitfalls

| Wrong | Correct | Reason |
| --- | --- | --- |
| `\|x\|` | `\lVert x \rVert` | Incorrect spacing for norms |
| `\|` inside probability | `\mid` | `P(A\|B)` renders poorly |
| `^T` for transpose | `^\top` | `T` looks like a variable name |
| `\text{tr}` | `\operatorname{tr}` | `\text` is semantic, `\operatorname` is mathematical |
| `\mathit{ReLU}` | `\operatorname{ReLU}` | Named functions use `\operatorname` |
| `...` | `\ldots` or `\cdots` | `\ldots` for baseline, `\cdots` for centred |
| `\mathbb{E}_{x}` | `\mathbb{E}_{\mathbf{x} \sim p}` | Specify distribution when not clear from context |
| `\sum` for covariance matrix | `\Sigma` | `\sum` is a summation sign |

---

*This guide is versioned with the repository. If you find an inconsistency between this
guide and any section file, the guide takes precedence — open an issue or PR to correct
the section file.*
