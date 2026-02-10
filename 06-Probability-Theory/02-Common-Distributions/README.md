# Common Probability Distributions

> **Navigation**: [01-Introduction](../01-Introduction-and-Random-Variables/) | [03-Joint-Distributions](../03-Joint-Distributions/) | [04-Expectation-and-Moments](../04-Expectation-and-Moments/)

## Overview

Probability distributions are the building blocks of probabilistic ML. Each distribution encapsulates assumptions about data generation processes. Choosing the right distribution enables better modeling, efficient inference, and interpretable results.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DISTRIBUTION FAMILY TREE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  DISCRETE                           CONTINUOUS                           │
│  ────────                           ──────────                           │
│                                                                          │
│  Bernoulli(p)                       Uniform(a,b)                         │
│      ↓                                  ↓                                │
│  Binomial(n,p) ──→ Normal             Beta(α,β) ←──┐                     │
│      ↓              (CLT)                           │                    │
│  Poisson(λ) ←────────────→ Exponential(λ)          │                    │
│      ↓                          ↓                   │                    │
│  Multinomial ←──→ Dirichlet ────┘                  Gamma(α,β)           │
│                                                       ↓                  │
│  Categorical ←──→ Dirichlet              Chi-squared ← Student's t      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- [01-Introduction-and-Random-Variables](../01-Introduction-and-Random-Variables/)
- Calculus (integration, gamma function)

## Learning Objectives

1. Master common discrete and continuous distributions
2. Understand relationships between distributions
3. Choose appropriate distributions for ML problems
4. Apply conjugate priors in Bayesian inference

---

## 1. Discrete Distributions

### Bernoulli Distribution

**Single binary trial**: success (1) with probability $p$, failure (0) with probability $1-p$.

$$P(X = x) = p^x(1-p)^{1-x}, \quad x \in \{0, 1\}$$

| Property | Value |
|----------|-------|
| Mean | $p$ |
| Variance | $p(1-p)$ |
| Mode | 1 if $p > 0.5$, else 0 |
| Support | $\{0, 1\}$ |

```
Bernoulli(p=0.7):

P(X)
1.0 │
0.8 │                 ████
0.6 │                 ████
0.4 │                 ████
0.3 │   ████          ████
0.2 │   ████          ████
0.1 │   ████          ████
    └───────────────────────
         0             1
         failure      success
```

**ML Applications:**
- Binary classification labels
- Dropout (each neuron: Bernoulli)
- Click/no-click prediction

---

### Binomial Distribution

**Number of successes in $n$ independent Bernoulli trials**.

$$P(X = k) = \binom{n}{k} p^k(1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

| Property | Value |
|----------|-------|
| Mean | $np$ |
| Variance | $np(1-p)$ |
| Mode | $\lfloor (n+1)p \rfloor$ |
| Support | $\{0, 1, \ldots, n\}$ |

```
Binomial Distributions:

n=10, p=0.2       n=10, p=0.5       n=10, p=0.8
    │ █                │                    │        █
    │ ██               │   ███              │       ██
    │ ███              │  █████             │      ███
    │ ████             │ ███████            │     ████
    │ █████            │█████████           │   █████
    └──────            └──────────          └──────────
    0  5  10           0   5   10           0   5   10
    
    Left-skewed       Symmetric            Right-skewed
```

> **💡 Intuition**: As $n \to \infty$, Binomial approaches Normal (Central Limit Theorem)

---

### Categorical Distribution

**Single draw from $K$ categories** (generalization of Bernoulli).

$$P(X = k) = p_k, \quad \sum_{k=1}^K p_k = 1$$

One-hot encoding: $\mathbf{x} = [0, \ldots, 0, 1, 0, \ldots, 0]$

**ML Applications:**
- Multi-class classification output
- Word prediction in language models
- Softmax output layer

---

### Multinomial Distribution

**Counts across $K$ categories in $n$ trials** (generalization of Binomial).

$$P(X_1 = n_1, \ldots, X_K = n_K) = \frac{n!}{n_1! \cdots n_K!} \prod_{k=1}^K p_k^{n_k}$$

where $\sum_k n_k = n$ and $\sum_k p_k = 1$.

| Property | Value |
|----------|-------|
| $E[X_k]$ | $np_k$ |
| $\text{Var}(X_k)$ | $np_k(1-p_k)$ |
| $\text{Cov}(X_i, X_j)$ | $-np_ip_j$ (negative!) |

> **⚠️ Note**: Negative covariance because more of one category means less of others!

**ML Applications:**
- Topic modeling (LDA)
- Bag-of-words document representation
- Multiclass count data

---

### Poisson Distribution

**Count of rare events** in fixed time/space.

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

| Property | Value |
|----------|-------|
| Mean | $\lambda$ |
| Variance | $\lambda$ (equals mean!) |
| Mode | $\lfloor \lambda \rfloor$ |
| Support | $\{0, 1, 2, \ldots\}$ |

```
Poisson Distributions:

λ = 1              λ = 4              λ = 10
    │█                  │                    │
    │██                 │  ██                │    ███
    │███                │ ████               │   █████
    │████               │██████              │  ███████
    │█████              │███████             │ █████████
    └──────             └──────────          └────────────
    0 1 2 3 4           0 2 4 6 8            0 5 10 15 20
```

> **💡 Connection**: As $n \to \infty$ and $p \to 0$ with $np = \lambda$, Binomial → Poisson

**ML Applications:**
- Website traffic modeling
- Rare event detection
- Count regression (Poisson regression)

---

### Geometric Distribution

**Number of trials until first success**.

$$P(X = k) = (1-p)^{k-1}p, \quad k = 1, 2, 3, \ldots$$

| Property | Value |
|----------|-------|
| Mean | $1/p$ |
| Variance | $(1-p)/p^2$ |
| Memoryless | Only discrete memoryless distribution |

**ML Applications:**
- Time until conversion
- Number of attempts until success

---

## 2. Continuous Distributions

### Uniform Distribution

**Equal probability over interval $[a, b]$**.

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

| Property | Value |
|----------|-------|
| Mean | $(a+b)/2$ |
| Variance | $(b-a)^2/12$ |
| Entropy | $\log(b-a)$ (maximum for bounded support) |

```
Uniform(0, 1):

f(x)
1.0 │ ████████████████████
    │ ████████████████████
    │ ████████████████████
0   └──────────────────────
    0                     1
```

**ML Applications:**
- Random initialization
- Parameter search bounds
- Prior for unknown parameters

---

### Normal (Gaussian) Distribution

**The most important distribution in statistics and ML**.

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| Property | Value |
|----------|-------|
| Mean | $\mu$ |
| Variance | $\sigma^2$ |
| Mode | $\mu$ (symmetric) |
| Entropy | $\frac{1}{2}\log(2\pi e\sigma^2)$ |

```
Normal Distributions with Different Parameters:

       σ=0.5                                σ=2
         │                                   │
       ╱ │ ╲                              ╱──┴──╲
      ╱  │  ╲         σ=1              ╱         ╲
     ╱   │   ╲       ╱───╲           ╱             ╲
    ╱    │    ╲    ╱       ╲       ╱                 ╲
───╱─────┼─────╲──╱─────────╲────╱─────────────────────╲───
         μ                                

• Smaller σ → taller, narrower peak
• Larger σ → shorter, wider spread
• All centered at μ
```

**Key Properties:**
1. Sum of independent normals is normal: $X_i \sim N(\mu_i, \sigma_i^2) \Rightarrow \sum X_i \sim N(\sum\mu_i, \sum\sigma_i^2)$
2. Linear transformation: $aX + b \sim N(a\mu + b, a^2\sigma^2)$
3. Central Limit Theorem: sample means → normal

> **🔑 Why Normal is everywhere in ML:**
> - CLT: sums of random effects → normal
> - Maximum entropy for fixed mean and variance
> - Makes math tractable (closed-form convolutions)
> - Gradient noise in SGD approximately normal

**ML Applications:**
- Gaussian noise models
- Weight initialization
- VAE latent space
- Gaussian processes

---

### Exponential Distribution

**Time between Poisson events** (continuous analog of Geometric).

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

| Property | Value |
|----------|-------|
| Mean | $1/\lambda$ |
| Variance | $1/\lambda^2$ |
| Memoryless | $P(X > s+t \mid X > s) = P(X > t)$ |
| Mode | 0 |

```
Exponential Distributions:

f(x)                λ = 2
2.0 │╲
    │ ╲
1.0 │  ╲   λ = 1
    │   ╲──╲
0.5 │      ╲ ╲___  λ = 0.5
    │         ╲___╲____
    └────────────────────→
    0    1    2    3    4
```

> **💡 Memoryless property**: The probability of waiting another $t$ time units doesn't depend on how long you've already waited!

**ML Applications:**
- Session duration modeling
- Inter-arrival times
- Survival analysis

---

### Gamma Distribution

**Generalization of Exponential** (sum of $\alpha$ exponentials).

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0$$

| Property | Value |
|----------|-------|
| Mean | $\alpha/\beta$ |
| Variance | $\alpha/\beta^2$ |
| Shape | $\alpha$ (shape), $\beta$ (rate) |
| Special cases | $\alpha=1$ → Exponential, $\alpha=n/2, \beta=1/2$ → Chi-squared |

```
Gamma Distributions (β=1):

f(x)
    │  α=1 (Exponential)
    │╲
    │ ╲     α=2
    │  ╲   ╱╲
    │   ╲_╱  ╲___  α=5
    │          ╱╲____
    │       __╱      ╲____
    └──────────────────────→
    0   2   4   6   8   10
```

**ML Applications:**
- Prior for precision (inverse variance)
- Bayesian inference for rates
- Wait times in queueing models

---

### Beta Distribution

**Distribution over probabilities** (values in [0, 1]).

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1$$

where $B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$

| Property | Value |
|----------|-------|
| Mean | $\alpha/(\alpha+\beta)$ |
| Variance | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| Mode | $(\alpha-1)/(\alpha+\beta-2)$ for $\alpha,\beta > 1$ |

```
Beta Distributions:

α=0.5,β=0.5 (U-shape)    α=1,β=1 (Uniform)    α=2,β=5 (Left-skewed)
      │ ╱╲                     │                    │  ╱╲
      │╱  ╲                    │─────────           │ ╱  ╲
      │    ╲                   │                    │╱    ╲
      │     ╲                  │                    │      ╲___
      ╱      ╲                 │                    │          ╲
     ╱        ╲                │                    │            ╲
    └──────────                └──────────          └──────────────
    0         1                0         1          0             1

α=5,β=2 (Right-skewed)   α=5,β=5 (Symmetric bell)
           ╱╲  │              │      ╱╲
          ╱  ╲ │              │    ╱    ╲
        _╱    ╲│              │  ╱        ╲
      _╱       │              │╱            ╲
     ╱         │              │               ╲
    └─────────────            └─────────────────
    0           1              0              1
```

> **🔑 Conjugate Prior**: Beta is the conjugate prior for Bernoulli/Binomial likelihood!

$$\text{Prior: } p \sim \text{Beta}(\alpha, \beta)$$
$$\text{Data: } k \text{ successes in } n \text{ trials}$$
$$\text{Posterior: } p \mid \text{data} \sim \text{Beta}(\alpha + k, \beta + n - k)$$

**ML Applications:**
- Bayesian A/B testing
- Success rate estimation
- Click-through rate modeling

---

### Student's t-Distribution

**Heavier tails than Normal** (robust to outliers).

$$f(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}}$$

| Property | Value |
|----------|-------|
| Mean | 0 (for $\nu > 1$) |
| Variance | $\nu/(\nu-2)$ for $\nu > 2$ |
| Degrees of freedom | $\nu$ (shape parameter) |
| Limit | As $\nu \to \infty$, t → Normal |

```
t-Distribution vs Normal:

            Normal (thin tails)
               ╱╲
              ╱  ╲
             ╱    ╲
     t-dist ╱──────╲ (heavy tails)
         _╱   ╲  ╱   ╲_
      __╱              ╲__
  ___╱                    ╲___
     │                        │
     │  More probability in   │
     │  tails = more robust   │
     ▼                        ▼
```

**ML Applications:**
- Robust regression
- Small sample inference
- Heavy-tailed noise modeling

---

## 3. Multivariate Distributions

### Multivariate Normal (Gaussian)

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

| Property | Value |
|----------|-------|
| Mean | $\boldsymbol{\mu} \in \mathbb{R}^d$ |
| Covariance | $\Sigma \in \mathbb{R}^{d \times d}$ (positive semi-definite) |
| Marginals | Each $X_i \sim N(\mu_i, \Sigma_{ii})$ |
| Conditionals | Also Gaussian! |

```
2D Multivariate Normal:

  Uncorrelated (ρ=0)      Positive correlation (ρ>0)    Negative correlation (ρ<0)
        y                        y                             y
        │     ○                  │       ╱                     │  ╲
        │   ○○○○○                │     ╱╱╱                     │    ╲╲
        │  ○○○○○○○               │   ╱╱╱╱╱                     │ ╲╲╲╲╲
        │   ○○○○○                │  ╱╱╱╱╱                      │   ╲╲╲╲
        │     ○                  │ ╱╱╱                         │     ╲╲
        └──────────x             └──────────x                  └──────────x
        
  Circular contours       Ellipse tilted up           Ellipse tilted down
```

**Key Properties:**
1. Marginals are Gaussian
2. Conditionals are Gaussian
3. Affine transformations preserve Gaussianity
4. Uncorrelated ⟺ Independent (only for Gaussians!)

**ML Applications:**
- Gaussian Mixture Models (GMM)
- Gaussian Processes
- VAE latent spaces
- Multivariate regression

---

### Dirichlet Distribution

**Distribution over probability simplices** (generalization of Beta).

$$f(\mathbf{x}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K x_k^{\alpha_k - 1}$$

where $\sum_k x_k = 1$ and $x_k \geq 0$.

| Property | Value |
|----------|-------|
| Mean | $E[x_k] = \alpha_k / \sum_j \alpha_j$ |
| Concentration | $\alpha_0 = \sum_k \alpha_k$ |
| Special case | $K=2$ → Beta distribution |

```
Dirichlet on 3-Simplex (triangle):

α = (1,1,1) - Uniform       α = (10,10,10) - Concentrated center
      ▲                              ▲
     ╱ ╲                            ╱ ╲
    ╱   ╲                          ╱   ╲
   ╱ ░░░ ╲                        ╱     ╲
  ╱ ░░░░░ ╲                      ╱  ░░░  ╲
 ╱ ░░░░░░░ ╲                    ╱   ░░░   ╲
▔▔▔▔▔▔▔▔▔▔▔▔                   ▔▔▔▔▔▔▔▔▔▔▔▔

α = (0.1,0.1,0.1) - Sparse    α = (5,1,1) - Biased to corner
      ▲                              ▲
     ╱╲╲                            ╱ ╲
    ╱  ╲                           ╱   ╲
   ╱    ╲                         ╱     ╲
  ╱░    ░╲                       ░░      ╲
 ░░      ░░                     ░░░░░     ╲
▔▔▔▔▔▔▔▔▔▔▔▔                   ▔▔▔▔▔▔▔▔▔▔▔▔
```

> **🔑 Conjugate Prior**: Dirichlet is the conjugate prior for Categorical/Multinomial!

**ML Applications:**
- Latent Dirichlet Allocation (LDA)
- Topic modeling
- Bayesian multi-class classification
- Mixture model priors

---

## 4. Conjugate Priors

A prior is **conjugate** to a likelihood if the posterior is in the same family.

```
┌────────────────────────────────────────────────────────────────┐
│                    CONJUGATE PRIOR PAIRS                       │
├────────────────────────────────────────────────────────────────┤
│  Likelihood          Prior              Posterior              │
│  ──────────          ─────              ─────────              │
│  Bernoulli/Binomial  Beta(α,β)          Beta(α+k, β+n-k)       │
│  Poisson             Gamma(α,β)         Gamma(α+Σx, β+n)       │
│  Exponential         Gamma(α,β)         Gamma(α+n, β+Σx)       │
│  Normal (μ known)    Normal(μ₀,σ₀²)     Normal(μ', σ'²)        │
│  Normal (σ² known)   Inv-Gamma(α,β)     Inv-Gamma(α',β')       │
│  Categorical/Mult.   Dirichlet(α)       Dirichlet(α+n)         │
│  Multivariate Normal Normal-Wishart     Normal-Wishart         │
└────────────────────────────────────────────────────────────────┘
```

> **💡 Why conjugacy matters**: Closed-form posteriors mean fast inference without MCMC!

### Example: Beta-Binomial

```
Prior:     Beta(2, 2)     (believe p ≈ 0.5)
           ╱╲
          ╱  ╲
         ╱    ╲

Data:     7 successes, 3 failures

Posterior: Beta(2+7, 2+3) = Beta(9, 5)
                    ╱╲
                  ╱    ╲
                ╱        ╲
               (peaks around 0.64)
```

---

## 5. Distribution Selection Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                 CHOOSING A DISTRIBUTION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  What type of data?                                             │
│  ─────────────────                                              │
│      │                                                           │
│      ├── Binary (yes/no) ────────────────→ Bernoulli            │
│      │                                                           │
│      ├── Count (0,1,2,...) ──┬── Bounded n ───→ Binomial        │
│      │                        └── Unbounded ───→ Poisson        │
│      │                                                           │
│      ├── Categories (K classes) ─────────→ Categorical          │
│      │                                                           │
│      ├── Time/Duration ──────┬── Memoryless ──→ Exponential     │
│      │                        └── Shape ──────→ Gamma/Weibull   │
│      │                                                           │
│      ├── Probability (0 to 1) ───────────→ Beta                 │
│      │                                                           │
│      ├── Real values ────────┬── Symmetric bell ─→ Normal       │
│      │                        ├── Heavy tails ────→ Student's t │
│      │                        └── Bounded ────────→ Beta (transform)│
│      │                                                           │
│      └── Probability vector ─────────────→ Dirichlet            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Summary Tables

### Quick Reference

| Distribution | Support | Mean | Variance | ML Use Case |
|-------------|---------|------|----------|-------------|
| Bernoulli($p$) | {0,1} | $p$ | $p(1-p)$ | Binary labels |
| Binomial($n,p$) | {0,...,n} | $np$ | $np(1-p)$ | Success counts |
| Categorical($\mathbf{p}$) | {1,...,K} | — | — | Multi-class |
| Poisson($\lambda$) | {0,1,2,...} | $\lambda$ | $\lambda$ | Rare events |
| Uniform($a,b$) | [a,b] | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Initialization |
| Normal($\mu,\sigma^2$) | $\mathbb{R}$ | $\mu$ | $\sigma^2$ | Everything! |
| Exponential($\lambda$) | $[0,\infty)$ | $1/\lambda$ | $1/\lambda^2$ | Wait times |
| Gamma($\alpha,\beta$) | $(0,\infty)$ | $\alpha/\beta$ | $\alpha/\beta^2$ | Rate priors |
| Beta($\alpha,\beta$) | [0,1] | $\frac{\alpha}{\alpha+\beta}$ | ... | Probability priors |
| Student's t($\nu$) | $\mathbb{R}$ | 0 | $\frac{\nu}{\nu-2}$ | Robust modeling |

### Relationship Cheat Sheet

```
┌──────────────────────────────────────────────────────────────┐
│               DISTRIBUTION RELATIONSHIPS                      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Bernoulli ──(n trials)──→ Binomial ──(n→∞, p→0)──→ Poisson  │
│      │                         │                              │
│      │ (K classes)             │ (CLT)                        │
│      ▼                         ▼                              │
│  Categorical              Normal ←──(ν→∞)── Student's t      │
│      │                                                        │
│      │ (n trials)                                             │
│      ▼                                                        │
│  Multinomial                                                  │
│                                                               │
│  Beta ──────(K=2)───────→ Dirichlet                          │
│                                                               │
│  Exponential ──(α=1)────→ Gamma                              │
│       │                                                       │
│       │ (count events)                                        │
│       ▼                                                       │
│  Poisson                                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Exercises

1. Derive the mean and variance of Binomial from Bernoulli sum
2. Show that Poisson is the limit of Binomial as $n \to \infty$, $p \to 0$, $np = \lambda$
3. Compute the posterior for Beta(1,1) prior with 5 heads, 3 tails
4. Prove that exponential distribution is memoryless
5. Generate samples from a Dirichlet and verify the mean formula

---

## References

1. Bishop - "Pattern Recognition and Machine Learning"
2. Murphy - "Machine Learning: A Probabilistic Perspective"
3. Gelman et al. - "Bayesian Data Analysis"

---

> **Next**: [03-Joint-Distributions](../03-Joint-Distributions/) — Multiple random variables together
