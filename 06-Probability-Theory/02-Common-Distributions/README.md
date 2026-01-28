# Common Probability Distributions

## Introduction

Understanding probability distributions is essential for machine learning. Different data types and phenomena are naturally modeled by specific distributions. This section covers the most important distributions used in ML.

## Prerequisites

- Basic probability theory
- Random variables
- Expectation and variance

## Learning Objectives

1. Recognize which distribution fits a given problem
2. Compute probabilities, expectations, and variances
3. Understand relationships between distributions
4. Apply distributions to ML models

---

## 1. Discrete Distributions

### 1.1 Bernoulli Distribution

**Use case:** Single binary trial (coin flip, binary classification)

$$P(X = k) = p^k (1-p)^{1-k}, \quad k \in \{0, 1\}$$

| Property | Value                              |
| -------- | ---------------------------------- |
| Mean     | $p$                                |
| Variance | $p(1-p)$                           |
| Mode     | $0$ if $p < 0.5$, $1$ if $p > 0.5$ |

### 1.2 Binomial Distribution

**Use case:** Number of successes in n independent trials

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

| Property | Value                 |
| -------- | --------------------- |
| Mean     | $np$                  |
| Variance | $np(1-p)$             |
| Support  | $\{0, 1, \ldots, n\}$ |

```
Binomial(n=10, p=0.3):

P(X=k)
0.25 │
     │    ██
0.20 │   ████
     │  █████
0.15 │  ██████
     │ ███████
0.10 │████████
     │█████████
0.05 │██████████
     └──────────────
       0 1 2 3 4 5 6 7 8 9 10
```

### 1.3 Categorical/Multinoulli Distribution

**Use case:** Single draw from K categories (multi-class classification)

$$P(X = k) = p_k, \quad \sum_{k=1}^K p_k = 1$$

Example: Rolling a biased die with probabilities $(p_1, \ldots, p_6)$

### 1.4 Multinomial Distribution

**Use case:** Counts from n trials with K outcomes

$$P(X_1=x_1, \ldots, X_K=x_K) = \frac{n!}{x_1! \cdots x_K!} p_1^{x_1} \cdots p_K^{x_K}$$

where $\sum_k x_k = n$

### 1.5 Poisson Distribution

**Use case:** Count of rare events, approximation to Binomial

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

| Property | Value                 |
| -------- | --------------------- |
| Mean     | $\lambda$             |
| Variance | $\lambda$             |
| Support  | $\{0, 1, 2, \ldots\}$ |

**Key property:** If events occur at rate $\lambda$ per unit time, counts follow Poisson($\lambda$).

### 1.6 Geometric Distribution

**Use case:** Number of trials until first success

$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, \ldots$$

| Property | Value       |
| -------- | ----------- |
| Mean     | $1/p$       |
| Variance | $(1-p)/p^2$ |

**Memoryless:** $P(X > m+n | X > m) = P(X > n)$

---

## 2. Continuous Distributions

### 2.1 Uniform Distribution

**Use case:** Random number generation, uninformative prior

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

| Property | Value         |
| -------- | ------------- |
| Mean     | $(a+b)/2$     |
| Variance | $(b-a)^2/12$  |
| CDF      | $(x-a)/(b-a)$ |

### 2.2 Normal (Gaussian) Distribution

**Use case:** Most common, Central Limit Theorem, errors

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

| Property | Value      |
| -------- | ---------- |
| Mean     | $\mu$      |
| Variance | $\sigma^2$ |
| Mode     | $\mu$      |

**Standard Normal $Z \sim N(0, 1)$:**

```
68-95-99.7 Rule:

     ┌────────────────────────┐
     │   ┌──────────────┐     │
     │   │  ┌──────┐    │     │
 f(z)│   │  │ 68%  │    │     │
     │   │  └──────┘    │     │
     │   │     95%      │     │
     │   └──────────────┘     │
     │        99.7%           │
     └────────────────────────┘
        -3  -2  -1  0  1  2  3
```

### 2.3 Exponential Distribution

**Use case:** Time between events, memoryless processes

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

| Property | Value                |
| -------- | -------------------- |
| Mean     | $1/\lambda$          |
| Variance | $1/\lambda^2$        |
| CDF      | $1 - e^{-\lambda x}$ |

**Memoryless property:** $P(X > s+t | X > s) = P(X > t)$

### 2.4 Gamma Distribution

**Use case:** Sum of exponentials, flexible shape, Bayesian priors

$$f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$$

| Property | Value            |
| -------- | ---------------- |
| Mean     | $\alpha/\beta$   |
| Variance | $\alpha/\beta^2$ |
| Support  | $x > 0$          |

Special cases:

- $\alpha = 1$: Exponential
- $\alpha = n/2, \beta = 1/2$: Chi-squared with n df

### 2.5 Beta Distribution

**Use case:** Prior for probabilities (bounded [0,1])

$$f(x) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1}$$

| Property | Value                                            |
| -------- | ------------------------------------------------ |
| Mean     | $\alpha/(\alpha+\beta)$                          |
| Variance | $\alpha\beta/[(\alpha+\beta)^2(\alpha+\beta+1)]$ |
| Support  | $[0, 1]$                                         |

```
Beta distributions for various (α, β):

    (1,1): Uniform
    (2,2): Bell-shaped symmetric
    (5,1): Left-skewed
    (1,5): Right-skewed
    (0.5,0.5): U-shaped
```

### 2.6 Student's t-Distribution

**Use case:** Heavy tails, small samples, robust regression

$$f(x) = \frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\Gamma(\nu/2)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}$$

| Property | Value                         |
| -------- | ----------------------------- |
| Mean     | $0$ (for $\nu > 1$)           |
| Variance | $\nu/(\nu-2)$ (for $\nu > 2$) |
| Limit    | $\nu \to \infty$: Normal(0,1) |

Heavy tails compared to Normal - more robust to outliers.

---

## 3. Multivariate Distributions

### 3.1 Multivariate Normal (Gaussian)

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

| Property     | Value                 |
| ------------ | --------------------- |
| Mean         | $\boldsymbol{\mu}$    |
| Covariance   | $\boldsymbol{\Sigma}$ |
| Marginals    | Normal                |
| Conditionals | Normal                |

```
2D Gaussian contours:

        ┌─────────────────┐
        │    ╭───────╮    │
        │   ╱ ╭───╮   ╲   │
        │  │ ╱     ╲  │   │
y       │  │ │  •  │  │   │
        │  │ ╲     ╱  │   │
        │   ╲ ╰───╯  ╱    │
        │    ╰──────╯     │
        └─────────────────┘
                x
• = mean, ellipses = constant probability
```

### 3.2 Dirichlet Distribution

**Use case:** Prior for probability vectors (simplex)

$$f(\mathbf{x}) = \frac{\Gamma(\sum_i \alpha_i)}{\prod_i \Gamma(\alpha_i)} \prod_{i=1}^K x_i^{\alpha_i - 1}$$

where $\sum_i x_i = 1$, $x_i \geq 0$

- Multivariate generalization of Beta
- Conjugate prior for Categorical/Multinomial

---

## 4. Distribution Relationships

```
Distribution Family Tree:

                    Bernoulli(p)
                         │
           ┌─────────────┼─────────────┐
           │             │             │
    Binomial(n,p)   Geometric(p)  Neg.Binomial
           │
           │ n→∞, np→λ
           ▼
      Poisson(λ)
           │ sum of n
           ▼
       Gamma(n,λ)
           │
           │ α=1
           ▼
    Exponential(λ)


Normal Family:

    Normal(μ,σ²) ────────────────────┐
           │                         │
           │ sum                     │ ratio
           ▼                         ▼
    Normal(nμ,nσ²)            Student's t(ν)
           │
           │ standardize
           ▼
      N(0,1) squared
           │
           ▼
       Chi-squared(1)
           │ sum of k
           ▼
       Chi-squared(k)
```

---

## 5. Conjugate Priors

**Conjugate prior:** Prior and posterior belong to same family.

| Likelihood         | Prior     | Posterior |
| ------------------ | --------- | --------- |
| Bernoulli/Binomial | Beta      | Beta      |
| Poisson            | Gamma     | Gamma     |
| Normal (known σ²)  | Normal    | Normal    |
| Normal (known μ)   | Inv-Gamma | Inv-Gamma |
| Multinomial        | Dirichlet | Dirichlet |

Example: Beta-Binomial conjugacy

- Prior: $p \sim \text{Beta}(\alpha, \beta)$
- Likelihood: $k | p \sim \text{Binomial}(n, p)$
- Posterior: $p | k \sim \text{Beta}(\alpha + k, \beta + n - k)$

---

## 6. ML Applications

### Classification

- **Binary:** Bernoulli for labels
- **Multi-class:** Categorical for labels
- **Logistic regression:** Models Bernoulli parameter

### Regression

- **Linear regression:** Normal errors
- **Robust regression:** Student's t errors
- **Count regression:** Poisson

### Generative Models

- **Gaussian Mixture Models:** Mixture of Gaussians
- **Latent Dirichlet Allocation:** Dirichlet-Multinomial

### Bayesian Inference

- **Prior selection:** Beta, Gamma, Normal
- **Posterior computation:** Use conjugacy when possible

---

## 7. Distribution Selection Guide

```
Choosing a distribution:

Data type?
├── Discrete
│   ├── Binary outcome → Bernoulli
│   ├── Count (bounded) → Binomial
│   ├── Count (unbounded) → Poisson
│   └── Multiple categories → Categorical/Multinomial
│
└── Continuous
    ├── Bounded [0,1] → Beta
    ├── Bounded [a,b] → Uniform or truncated
    ├── Positive only → Gamma, Exponential, Log-normal
    ├── Whole real line → Normal
    └── Heavy tails → Student's t
```

---

## 8. Summary Tables

### Discrete Distributions

| Distribution  | PMF                                 | Mean      | Variance    |
| ------------- | ----------------------------------- | --------- | ----------- |
| Bernoulli(p)  | $p^k(1-p)^{1-k}$                    | $p$       | $p(1-p)$    |
| Binomial(n,p) | $\binom{n}{k}p^k(1-p)^{n-k}$        | $np$      | $np(1-p)$   |
| Poisson(λ)    | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$   |
| Geometric(p)  | $(1-p)^{k-1}p$                      | $1/p$     | $(1-p)/p^2$ |

### Continuous Distributions

| Distribution   | PDF                                                           | Mean                          | Variance                                               |
| -------------- | ------------------------------------------------------------- | ----------------------------- | ------------------------------------------------------ |
| Uniform(a,b)   | $\frac{1}{b-a}$                                               | $\frac{a+b}{2}$               | $\frac{(b-a)^2}{12}$                                   |
| Normal(μ,σ²)   | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-(x-\mu)^2/2\sigma^2}$       | $\mu$                         | $\sigma^2$                                             |
| Exponential(λ) | $\lambda e^{-\lambda x}$                                      | $1/\lambda$                   | $1/\lambda^2$                                          |
| Gamma(α,β)     | $\frac{\beta^\alpha}{\Gamma(\alpha)}x^{\alpha-1}e^{-\beta x}$ | $\alpha/\beta$                | $\alpha/\beta^2$                                       |
| Beta(α,β)      | $\frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}$         | $\frac{\alpha}{\alpha+\beta}$ | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |

---

## Exercises

1. Derive the MGF of Poisson and use it to find mean and variance
2. Show that sum of independent Poissons is Poisson
3. Derive the posterior for Beta-Binomial model
4. Show that N(μ, σ²) is invariant under linear transformation
5. Compute the entropy of Bernoulli(p)

---

## References

1. Casella & Berger - "Statistical Inference"
2. Murphy - "Machine Learning: A Probabilistic Perspective"
3. Bishop - "Pattern Recognition and Machine Learning"
