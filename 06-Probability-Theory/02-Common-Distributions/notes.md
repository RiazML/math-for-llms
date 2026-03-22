[← Back to Probability Theory](../README.md) | [Next: Joint Distributions →](../03-Joint-Distributions/notes.md)

---

# §6.2 Common Distributions

> _"God does not play dice — but physicists, statisticians, and machine learning engineers do. The distributions in this chapter are the dice they use."_

## Overview

Every probabilistic model makes a choice: what distribution describes the data? A Bernoulli for a coin flip, a Gaussian for continuous measurements, a Categorical for language model token prediction, a Dirichlet for topic proportions. These are not arbitrary choices — each distribution encapsulates a specific data-generating story, a set of assumptions, and a collection of mathematical properties that make inference tractable.

This section gives the complete treatment of every named distribution that appears throughout this curriculum. For each distribution you will find: the PDF or PMF, the CDF where useful, the parameters and their interpretations, the moments (mean, variance, skewness), the moment generating function, the shape behaviour as parameters vary, the special cases and limiting forms, and the concrete ML applications where the distribution appears.

The section culminates in two unifying frameworks: the **exponential family**, which shows that Bernoulli, Gaussian, Poisson, Beta, Gamma, Dirichlet, and Categorical are all instances of a single canonical form; and **conjugate priors**, which explains why Bayesian inference is analytically tractable for precisely these distributions.

**What this section assumes:** The definitions of PDF, PMF, CDF, and the basic probability axioms from [§01](../01-Introduction-and-Random-Variables/notes.md). Bernoulli and Gaussian were introduced there; this section gives their full treatment.

**What this section defers:** Expectation derivations are stated as facts here and derived from first principles in [§04](../04-Expectation-and-Moments/notes.md). The multivariate Gaussian is previewed here and fully developed in [§03](../03-Joint-Distributions/notes.md).

---

## Prerequisites

- [§01 Introduction and Random Variables](../01-Introduction-and-Random-Variables/notes.md) — probability axioms, CDF, PDF, PMF, Bernoulli (intro)
- Integration: computing areas under curves, substitution, gamma function — [§03 Integration](../../04-Calculus-Fundamentals/03-Integration/notes.md)
- Series: power series, convergence — [§04 Series and Sequences](../../04-Calculus-Fundamentals/04-Series-and-Sequences/notes.md)

---

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive visualisations of all distributions; exponential family; conjugacy updates |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises: PMF/MGF computation, exponential family identification, conjugate Bayesian updating |

---

## Learning Objectives

After completing this section, you will:

- State the PMF or PDF, support, mean, variance, and MGF of every major named distribution
- Identify which distribution models a given data-generating process
- Explain the relationships between distributions (Binomial→Poisson, Binomial→Normal, Beta→Dirichlet)
- Write any exponential family member in canonical form and identify its sufficient statistics and log-partition function
- Compute the posterior in a conjugate Bayesian model (Beta-Binomial, Dirichlet-Categorical, Gamma-Poisson, Normal-Normal)
- Derive the softmax function as the natural parameterisation of the Categorical exponential family
- Explain how the Gaussian reparameterisation trick enables backpropagation through sampling in VAEs
- Describe the role of each distribution in at least two concrete ML architectures

---

## Table of Contents

- [1. Intuition and Overview](#1-intuition-and-overview)
  - [1.1 Why Named Distributions?](#11-why-named-distributions)
  - [1.2 How to Choose a Distribution](#12-how-to-choose-a-distribution)
  - [1.3 The Distribution Family Tree](#13-the-distribution-family-tree)
  - [1.4 Historical Timeline](#14-historical-timeline)
- [2. Discrete Distributions](#2-discrete-distributions)
  - [2.1 Bernoulli($p$)](#21-bernoullip)
  - [2.2 Binomial($n$, $p$)](#22-binomialn-p)
  - [2.3 Geometric($p$) and Negative Binomial($r$, $p$)](#23-geometricp-and-negative-binomialr-p)
  - [2.4 Poisson($\lambda$)](#24-poissonlambda)
  - [2.5 Categorical($\mathbf{p}$)](#25-categoricalmathbfp)
  - [2.6 Multinomial($n$, $\mathbf{p}$)](#26-multinomialn-mathbfp)
- [3. Continuous Distributions](#3-continuous-distributions)
  - [3.1 Uniform($a$, $b$)](#31-uniforma-b)
  - [3.2 Gaussian $\mathcal{N}(\mu, \sigma^2)$](#32-gaussian-mathcaln-mu-sigma2)
  - [3.3 Exponential($\lambda$)](#33-exponentiallambda)
  - [3.4 Gamma($\alpha$, $\beta$)](#34-gammaalpha-beta)
  - [3.5 Beta($\alpha$, $\beta$)](#35-betaalpha-beta)
  - [3.6 Dirichlet($\boldsymbol{\alpha}$)](#36-dirichletboldsymbolalpha)
  - [3.7 Student-$t$($\nu$)](#37-student-tnu)
- [4. Distribution Relationships](#4-distribution-relationships)
  - [4.1 Limiting Relationships](#41-limiting-relationships)
  - [4.2 Conjugate Pairs Table](#42-conjugate-pairs-table)
- [5. Moment Generating Functions](#5-moment-generating-functions)
  - [5.1 Definition and Uniqueness](#51-definition-and-uniqueness)
  - [5.2 MGF Product Rule](#52-mgf-product-rule)
  - [5.3 MGFs of Key Distributions](#53-mgfs-of-key-distributions)
- [6. The Exponential Family](#6-the-exponential-family)
  - [6.1 Canonical Form](#61-canonical-form)
  - [6.2 Members and Their Parameters](#62-members-and-their-parameters)
  - [6.3 The Log-Partition Function](#63-the-log-partition-function)
  - [6.4 Sufficient Statistics](#64-sufficient-statistics)
  - [6.5 ML Implications](#65-ml-implications)
- [7. Conjugate Priors in Bayesian Inference](#7-conjugate-priors-in-bayesian-inference)
  - [7.1 Conjugacy Definition](#71-conjugacy-definition)
  - [7.2 Beta-Bernoulli/Binomial](#72-beta-bernoulibinomial)
  - [7.3 Dirichlet-Categorical/Multinomial](#73-dirichlet-categoricalmultinomial)
  - [7.4 Gamma-Poisson](#74-gamma-poisson)
  - [7.5 Normal-Normal](#75-normal-normal)
- [8. ML Applications](#8-ml-applications)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition and Overview

### 1.1 Why Named Distributions?

A probability distribution is completely specified by its CDF. So why do we name particular families and memorise their formulas?

Three reasons:

**Sufficient statistics compress data.** If you observe $n$ coin flips, all the information about $p$ is contained in the count of heads — not the sequence. The Binomial distribution formalises this: its PMF depends on the data only through $\sum_i x_i$. Named distributions arise precisely when the data-generating process has this kind of compressibility.

**Tractable inference.** Computing posteriors, marginals, and predictions requires integration. For most distributions this is intractable. The named distributions are the ones for which the integration can be done in closed form — either because the MGF factors, or because they belong to the exponential family where the normalisation constant has an analytic form.

**Interpretable parameters.** A Gaussian $\mathcal{N}(\mu, \sigma^2)$ has a mean $\mu$ and a standard deviation $\sigma$ that are immediately meaningful. A Beta$(2, 5)$ prior encodes "I've seen 2 successes and 5 failures before the experiment." Named distributions give parameters human-interpretable meaning.

**For AI:** Every loss function and output layer in a neural network implicitly assumes a distribution over outputs. Cross-entropy loss assumes a Categorical distribution. Mean squared error assumes a Gaussian. Poisson regression assumes a Poisson distribution. Understanding which distribution the loss corresponds to is essential for knowing when a model is misspecified.

### 1.2 How to Choose a Distribution

```
DATA GENERATING PROCESS
════════════════════════════════════════════════════════════════════════

  Discrete outcomes?
  ├── Two outcomes (0/1)             → Bernoulli(p)  [single trial]
  ├── Count of successes in n trials → Binomial(n, p)
  ├── Trials until first success     → Geometric(p)
  ├── Trials until r-th success      → Negative Binomial(r, p)
  ├── Count of events in interval    → Poisson(λ)
  ├── One of K categories            → Categorical(p)
  └── Counts across K categories     → Multinomial(n, p)

  Continuous outcomes?
  ├── Bounded, equal weight          → Uniform(a, b)
  ├── Unbounded, symmetric bell      → Gaussian(μ, σ²)
  ├── Non-negative, time to event    → Exponential(λ) or Gamma(α, β)
  ├── Probability value in (0, 1)    → Beta(α, β)
  ├── Probability vector on simplex  → Dirichlet(α)
  └── Heavy-tailed, small samples    → Student-t(ν)

════════════════════════════════════════════════════════════════════════
```

### 1.3 The Distribution Family Tree

```
DISTRIBUTION FAMILY TREE
════════════════════════════════════════════════════════════════════════

  DISCRETE                              CONTINUOUS
  ────────────────────────────          ──────────────────────────────

  Bernoulli(p)                          Uniform(a,b)
      │ n trials                              │
      ▼                                       ▼ (generalise)
  Binomial(n,p) ─────── CLT ──────────→ Gaussian(μ,σ²)
      │                                       │
      │ n→∞, p→0, np=λ                        │ μ=0, σ²=ν (ν→∞)
      ▼                                       ▼
  Poisson(λ) ────── Poisson ──────────→ Exponential(λ)
      │             process                   │ sum of α
      │ count                                 ▼
  Multinomial(n,p) ◄──── generalise ── Gamma(α,β)
      │                                       │ α=1
  Categorical(p)                         Exponential(λ)
      │                                       │ α=k/2, β=½
  conjugate prior:                       Chi-squared(k)
      │                                       │
      ▼                                       ▼
  Dirichlet(α) ──── marginals ────────→ Beta(α,β)
                                             (K=2 case)

  Student-t(ν) = Gaussian / √(Chi²(ν)/ν)  [heavy tails; ν→∞ → Gaussian]

════════════════════════════════════════════════════════════════════════
```

### 1.4 Historical Timeline

| Year | Discovery | Contributor |
|---|---|---|
| 1713 | Bernoulli distribution, Law of Large Numbers | Jakob Bernoulli |
| 1733 | Normal approximation to Binomial | Abraham de Moivre |
| 1809 | Normal distribution as error model (least squares) | Carl Friedrich Gauss |
| 1837 | Poisson distribution as limit of rare events | Siméon-Denis Poisson |
| 1839 | Dirichlet distribution (as Bayesian prior) | Peter Gustav Lejeune Dirichlet |
| 1860 | Exponential distribution (Maxwell's speed distribution) | James Clerk Maxwell |
| 1893 | Chi-squared distribution | Karl Pearson |
| 1908 | Student-$t$ distribution (small samples) | William Sealy Gosset ("Student") |
| 1911 | Beta distribution (conjugate to Binomial) | Karl Pearson |
| 1922 | Sufficient statistics | Ronald A. Fisher |
| 1935 | Exponential family (unified framework) | Koopman, Pitman, Darmois |
| 2006 | Dirichlet-Categorical in LDA (topic models) | Blei, Ng, Jordan |
| 2013 | Gaussian VAE and reparameterisation trick | Kingma, Welling |
| 2020 | Dirichlet language model priors (BayesOPT) | various |

---

## 2. Discrete Distributions

A **discrete** random variable takes values in a countable set $\mathcal{X}$. Its distribution is fully described by the **PMF** $p_X(x) = P(X = x)$, which satisfies $\sum_{x \in \mathcal{X}} p_X(x) = 1$.

### 2.1 Bernoulli($p$)

**Story:** A single binary trial: success (1) with probability $p$, failure (0) with probability $1-p$.

**PMF:**
$$p_X(x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}, \quad p \in [0, 1]$$

Equivalently: $P(X=1) = p$ and $P(X=0) = 1-p$.

**CDF:**
$$F_X(x) = \begin{cases} 0 & x < 0 \\ 1-p & 0 \leq x < 1 \\ 1 & x \geq 1 \end{cases}$$

**Moments:**

| Quantity | Value |
|---|---|
| Mean $\mathbb{E}[X]$ | $p$ |
| Variance $\operatorname{Var}(X)$ | $p(1-p)$ |
| Mode | $\mathbf{1}[p > 0.5]$ |
| Entropy $H(X)$ | $-p\log p - (1-p)\log(1-p)$ |
| Skewness | $(1-2p)/\sqrt{p(1-p)}$ |

The variance $p(1-p)$ is maximised at $p=0.5$ (maximum uncertainty) and equals zero at $p \in \{0, 1\}$ (certainty).

**Moment Generating Function:**
$$M_X(t) = \mathbb{E}[e^{tX}] = (1-p) + pe^t = 1 - p + pe^t$$

**Log-odds / logit:** The natural parameterisation for the Bernoulli is the **logit**:
$$\eta = \log\frac{p}{1-p} \in (-\infty, \infty)$$

Inverting: $p = \sigma(\eta) = \frac{e^\eta}{1+e^\eta} = \frac{1}{1+e^{-\eta}}$ — the **sigmoid function**. This is why logistic regression parameterises the Bernoulli distribution.

**For AI:**
- **Binary classification:** labels $y \in \{0, 1\}$ are Bernoulli. The cross-entropy loss $-y\log\hat{p} - (1-y)\log(1-\hat{p})$ is the negative log-likelihood of a Bernoulli model.
- **Dropout:** each activation is independently masked with Bernoulli$(1-\text{drop\_rate})$. At inference, expectations replace samples: $h \to (1-\text{drop\_rate}) \cdot h$.
- **Stochastic depth:** transformer layers are included/skipped via Bernoulli sampling during training.

**Non-examples:** A Bernoulli is NOT appropriate when outcomes are not binary (use Categorical), when multiple trials are involved (use Binomial), or when the "probability" varies per trial (use a hierarchical model).

---

### 2.2 Binomial($n$, $p$)

**Story:** Count the number of successes in $n$ independent Bernoulli$(p)$ trials. If $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \operatorname{Bern}(p)$, then $X = \sum_{i=1}^n X_i \sim \operatorname{Binomial}(n, p)$.

**PMF:**
$$p_X(k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

The binomial coefficient $\binom{n}{k} = n!/(k!(n-k)!)$ counts the number of ways to choose which $k$ of the $n$ trials succeed.

**Moments:**

| Quantity | Value | Derivation hint |
|---|---|---|
| Mean $\mathbb{E}[X]$ | $np$ | Linearity: $\sum_{i=1}^n \mathbb{E}[X_i] = np$ |
| Variance $\operatorname{Var}(X)$ | $np(1-p)$ | Independence: $\sum_i \operatorname{Var}(X_i)$ |
| Mode | $\lfloor (n+1)p \rfloor$ | (or $\lceil (n+1)p \rceil - 1$) |
| Skewness | $(1-2p)/\sqrt{np(1-p)}$ | |

**MGF:** Since $X = X_1 + \cdots + X_n$ with independent summands:
$$M_X(t) = \prod_{i=1}^n M_{X_i}(t) = (1-p+pe^t)^n$$

**Shape behaviour:**
- $p < 0.5$: right-skewed (most counts are below the mean)
- $p = 0.5$: symmetric
- $p > 0.5$: left-skewed
- As $n$ grows: bell-shaped by the Central Limit Theorem (CLT preview → §06)

**Normal approximation (CLT preview):** For large $n$:
$$\frac{X - np}{\sqrt{np(1-p)}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

Rule of thumb: approximation is accurate when $np \geq 5$ and $n(1-p) \geq 5$.

**Poisson limit:** When $n \to \infty$ and $p \to 0$ with $np = \lambda$ fixed:
$$\binom{n}{k}p^k(1-p)^{n-k} \to \frac{\lambda^k e^{-\lambda}}{k!}$$

(Full derivation in §2.4.)

**For AI:**
- **A/B testing:** number of conversions in $n$ visits follows Binomial$(n, p)$.
- **Ensemble prediction:** number of ensemble members predicting class 1 follows Binomial$(M, p)$ where $M$ is ensemble size.
- **Batch statistics:** number of positive examples in a minibatch of size $B$ drawn from a dataset with fraction $\rho$ positive is Binomial$(B, \rho)$.

---

### 2.3 Geometric($p$) and Negative Binomial($r$, $p$)

#### Geometric($p$)

**Story:** Number of independent Bernoulli$(p)$ trials until (and including) the first success.

**PMF:**
$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

The factor $(1-p)^{k-1}$ is the probability of $k-1$ consecutive failures; $p$ is the probability of the final success.

**Moments:**

| Quantity | Value |
|---|---|
| Mean $\mathbb{E}[X]$ | $1/p$ |
| Variance $\operatorname{Var}(X)$ | $(1-p)/p^2$ |
| Mode | $1$ (always) |
| Median | $\lceil -1/\log_2(1-p) \rceil$ |

**MGF:**
$$M_X(t) = \frac{pe^t}{1-(1-p)e^t}, \quad t < -\log(1-p)$$

**Memoryless property:** For integers $s, t \geq 0$:
$$P(X > s + t \mid X > s) = P(X > t)$$

*Proof:* $P(X > k) = (1-p)^k$. Then $P(X > s+t \mid X > s) = P(X > s+t)/P(X > s) = (1-p)^{s+t}/(1-p)^s = (1-p)^t = P(X > t)$. $\square$

The Geometric distribution is the **unique discrete memoryless distribution** (just as the Exponential is the unique continuous memoryless distribution).

**Variant:** Some sources define $X$ as the number of failures before the first success, giving PMF $P(X=k) = (1-p)^k p$ for $k = 0, 1, 2, \ldots$ — this shifts everything by 1. Always check which convention a source uses.

#### Negative Binomial($r$, $p$)

**Story:** Number of trials until the $r$-th success. The sum of $r$ independent Geometric$(p)$ random variables.

**PMF:**
$$P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, r+2, \ldots$$

**Moments:** Mean $= r/p$, Variance $= r(1-p)/p^2$.

**Overdispersion:** The variance $r(1-p)/p^2 = (r/p) \cdot (1-p)/p > r/p$ exceeds the mean (for $p < 1$). This makes the Negative Binomial useful for count data with **overdispersion** — variance greater than the mean — which the Poisson cannot model.

**For AI:**
- **Sequence length modelling:** the length of a sentence until a full stop follows approximately Geometric or Negative Binomial.
- **Retry modelling:** number of API calls until success follows Geometric (with exponential backoff, the distribution changes).
- **Count regression:** Negative Binomial regression replaces Poisson regression when data exhibits overdispersion (e.g., user activity counts, bug counts per module).

---

### 2.4 Poisson($\lambda$)

**Story:** Count of independent rare events occurring in a fixed interval (time, space, area), where $\lambda$ is the average rate (events per interval).

**PMF:**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots, \quad \lambda > 0$$

**Moments:**

| Quantity | Value |
|---|---|
| Mean $\mathbb{E}[X]$ | $\lambda$ |
| Variance $\operatorname{Var}(X)$ | $\lambda$ (mean = variance!) |
| Mode | $\lfloor \lambda \rfloor$ (or $\lambda - 1$ and $\lambda$ if $\lambda$ is integer) |
| Skewness | $1/\sqrt{\lambda}$ (right-skewed; approaches 0 as $\lambda$ grows) |

The **equal mean and variance** is the defining fingerprint of Poisson data. When empirical variance significantly exceeds the mean, the data is overdispersed and the Negative Binomial is more appropriate.

**MGF:**
$$M_X(t) = e^{\lambda(e^t - 1)}$$

*Derivation:* $M_X(t) = \sum_{k=0}^\infty e^{tk} \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=0}^\infty \frac{(\lambda e^t)^k}{k!} = e^{-\lambda} \cdot e^{\lambda e^t} = e^{\lambda(e^t-1)}$.

**Additive property:** If $X \sim \operatorname{Poisson}(\lambda_1)$ and $Y \sim \operatorname{Poisson}(\lambda_2)$ independently, then $X + Y \sim \operatorname{Poisson}(\lambda_1 + \lambda_2)$.

*Proof via MGFs:* $M_{X+Y}(t) = M_X(t) M_Y(t) = e^{\lambda_1(e^t-1)} \cdot e^{\lambda_2(e^t-1)} = e^{(\lambda_1+\lambda_2)(e^t-1)}$. $\square$

**Poisson Limit Theorem (Binomial → Poisson):**

As $n \to \infty$, $p \to 0$, with $np = \lambda$ fixed:
$$\binom{n}{k} p^k (1-p)^{n-k} \to \frac{\lambda^k e^{-\lambda}}{k!}$$

*Proof sketch:*
$$\binom{n}{k}p^k(1-p)^{n-k} = \frac{n(n-1)\cdots(n-k+1)}{k!} \cdot \frac{\lambda^k}{n^k} \cdot \left(1-\frac{\lambda}{n}\right)^{n-k}$$

As $n \to \infty$: $n(n-1)\cdots(n-k+1)/n^k \to 1$, and $(1-\lambda/n)^n \to e^{-\lambda}$, $(1-\lambda/n)^{-k} \to 1$. This gives $\lambda^k e^{-\lambda}/k!$. $\square$

**Poisson Process:** A sequence of events where (1) events in disjoint intervals are independent, (2) the probability of exactly one event in a small interval $[t, t+\delta)$ is $\lambda\delta + o(\delta)$, and (3) the probability of two or more events is $o(\delta)$. The count of events in interval $[0,T]$ is $\operatorname{Poisson}(\lambda T)$.

**For AI:**
- **Poisson regression:** predicting count outcomes (clicks, API calls, bug reports). Loss is $-\log P(y \mid \lambda) = \lambda - y\log\lambda + \log(y!)$, minimised by setting $\lambda = \hat{y}$.
- **Attention pattern sparsity:** the number of tokens attended to above a threshold in sparse attention can follow approximately Poisson.
- **Dataset curation:** rare-event counts in datasets (specific entity types, low-frequency tokens) follow Poisson, motivating over-sampling strategies.

---

### 2.5 Categorical($\mathbf{p}$)

**Story:** A single draw from $K$ mutually exclusive categories, where category $k$ has probability $p_k$. The multivariate generalisation of Bernoulli.

**PMF:** Let $\mathbf{p} = (p_1, \ldots, p_K)$ with $p_k \geq 0$ and $\sum_k p_k = 1$. For outcome $X = k$:
$$P(X = k) = p_k, \quad k = 1, \ldots, K$$

In one-hot vector notation: if $\mathbf{e}_k$ is the $k$-th standard basis vector:
$$P(X = \mathbf{e}_k) = p_k$$

**Moments:** $\mathbb{E}[X_k] = p_k$, $\operatorname{Var}(X_k) = p_k(1-p_k)$, $\operatorname{Cov}(X_j, X_k) = -p_jp_k$ for $j \neq k$.

**Softmax Parameterisation:**

The probability vector $\mathbf{p}$ lives on the $(K-1)$-dimensional probability simplex $\Delta^{K-1} = \{\mathbf{p} : p_k \geq 0, \sum_k p_k = 1\}$. For unconstrained logits $\mathbf{z} \in \mathbb{R}^K$, the natural parameterisation is:
$$p_k = \operatorname{softmax}(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

This is the **exponential family natural parameterisation** of the Categorical — the logits $\mathbf{z}$ are the natural parameters.

**Temperature scaling:** The Categorical can be "sharpened" or "softened" by temperature $\tau > 0$:
$$p_k \propto e^{z_k/\tau}$$

- $\tau \to 0$: deterministic (argmax)
- $\tau = 1$: standard softmax
- $\tau \to \infty$: uniform distribution

**Entropy:** $H(X) = -\sum_k p_k \log p_k$. Maximum at uniform ($\mathbf{p} = \mathbf{1}/K$, giving $H = \log K$); minimum at deterministic ($H = 0$).

**For AI:**
- **Language model output:** every token prediction is a Categorical over vocabulary. Training minimises $-\log p_\theta(x_t \mid x_{<t})$, which is the NLL of a Categorical.
- **Gumbel-softmax trick:** the Gumbel-max trick allows differentiable sampling from a Categorical: sample $G_k \sim \operatorname{Gumbel}(0,1)$, then $X = \arg\max_k (z_k + G_k)$. The Gumbel-softmax approximation uses softmax with low temperature to make this differentiable.
- **Top-$p$ (nucleus) sampling:** truncate the Categorical to the smallest set of tokens whose cumulative probability exceeds $p$, then renormalise.

---

### 2.6 Multinomial($n$, $\mathbf{p}$)

**Story:** Counts across $K$ categories in $n$ independent Categorical draws. If each trial draws from Categorical$(\mathbf{p})$, then the vector of counts $\mathbf{X} = (X_1, \ldots, X_K)$ follows a Multinomial.

**PMF:** For non-negative integers $k_1, \ldots, k_K$ with $\sum_j k_j = n$:
$$P(X_1 = k_1, \ldots, X_K = k_K) = \frac{n!}{k_1! \cdots k_K!} \prod_{j=1}^K p_j^{k_j}$$

The multinomial coefficient $n!/(k_1!\cdots k_K!)$ counts the number of orderings.

**Moments:**
$$\mathbb{E}[X_k] = np_k, \quad \operatorname{Var}(X_k) = np_k(1-p_k), \quad \operatorname{Cov}(X_j, X_k) = -np_jp_k \; (j \neq k)$$

The **negative covariance** between categories is inevitable: if more of one category is observed, fewer of the others must be. This negative dependence structure is a fundamental constraint of fixed-total count vectors.

**Marginals:** Each marginal $X_k \sim \operatorname{Binomial}(n, p_k)$ — this is why Binomial is the two-category special case ($K=2$) of Multinomial.

**For AI:**
- **Topic models (LDA):** document word counts follow Multinomial$(|\text{doc}|, \boldsymbol{\theta})$ where $\boldsymbol{\theta}$ is the topic-word distribution.
- **Bag-of-words:** the count vector representation of a document is a realisation of the Multinomial.
- **Batch class balance:** expected class counts in a minibatch follow Multinomial$(B, \boldsymbol{\rho})$ where $\boldsymbol{\rho}$ is the class frequency vector.

---

## 3. Continuous Distributions

A **continuous** random variable has a probability density function (PDF) $f_X(x) \geq 0$ satisfying $\int_{-\infty}^\infty f_X(x)\,dx = 1$. Probabilities are computed as $P(a \leq X \leq b) = \int_a^b f_X(x)\,dx$.

### 3.1 Uniform($a$, $b$)

**Story:** All values in $[a, b]$ are equally likely. The maximum entropy distribution subject to being supported on a bounded interval.

**PDF and CDF:**
$$f_X(x) = \frac{1}{b-a}, \quad x \in [a, b] \qquad F_X(x) = \frac{x-a}{b-a}, \quad x \in [a,b]$$

**Moments:**

| Quantity | Value |
|---|---|
| Mean | $(a+b)/2$ |
| Variance | $(b-a)^2/12$ |
| Entropy | $\log(b-a)$ |

**MGF:**
$$M_X(t) = \frac{e^{tb} - e^{ta}}{t(b-a)}, \quad t \neq 0; \quad M_X(0) = 1$$

**Maximum entropy:** Among all distributions supported on $[a, b]$, the Uniform has the maximum Shannon entropy. This makes it the natural "least informative" prior when only the support is known.

**Universality:** If $X$ has continuous CDF $F_X$, then $U = F_X(X) \sim \mathcal{U}(0,1)$. Conversely, $F_X^{-1}(U) \sim F_X$ for $U \sim \mathcal{U}(0,1)$. This is the **inverse CDF sampling method** introduced in §01.

**For AI:**
- **Weight initialisation:** Kaiming uniform initialisation draws weights from $\mathcal{U}(-\sqrt{3/\text{fan\_in}}, \sqrt{3/\text{fan\_in}})$, chosen so that $\operatorname{Var}(W) = 1/\text{fan\_in}$.
- **Random search:** hyperparameter search over a bounded range uses Uniform priors.
- **Data augmentation:** random crop position, rotation angle, and colour jitter are drawn from Uniform distributions.

---

### 3.2 Gaussian $\mathcal{N}(\mu, \sigma^2)$

**Story:** The limiting distribution of the standardised sum of i.i.d. random variables (Central Limit Theorem). The maximum entropy distribution for fixed mean and variance. The distribution that makes least-squares regression optimal under Gaussian noise.

**PDF:**
$$f_X(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \quad x \in \mathbb{R}$$

**Normalisation proof sketch:**
$$\int_{-\infty}^\infty e^{-x^2/2}\,dx = \sqrt{2\pi}$$
This follows from the Gaussian integral trick: $I^2 = \int\!\int e^{-(x^2+y^2)/2}\,dx\,dy = 2\pi\int_0^\infty re^{-r^2/2}\,dr = 2\pi$, so $I = \sqrt{2\pi}$. (Full proof in Appendix A of §01.)

**Parameters:**
- $\mu \in \mathbb{R}$: **mean** (location parameter — shifts the peak)
- $\sigma^2 > 0$: **variance** (scale parameter — controls spread)
- $\sigma = \sqrt{\sigma^2}$: **standard deviation** (same units as $X$)

**Moments:**

| Quantity | Value |
|---|---|
| Mean $\mathbb{E}[X]$ | $\mu$ |
| Variance $\operatorname{Var}(X)$ | $\sigma^2$ |
| Mode | $\mu$ (unique, at the peak) |
| Median | $\mu$ (by symmetry) |
| Skewness | $0$ (perfectly symmetric) |
| Kurtosis | $3$ (excess kurtosis $= 0$) |
| Entropy | $\frac{1}{2}\log(2\pi e\sigma^2)$ |

**MGF:**
$$M_X(t) = \exp\!\left(\mu t + \frac{\sigma^2 t^2}{2}\right)$$

*Derivation:*
$$M_X(t) = \int_{-\infty}^\infty e^{tx} \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)}\,dx$$

Complete the square in the exponent: $tx - (x-\mu)^2/(2\sigma^2) = -(x-\mu-\sigma^2 t)^2/(2\sigma^2) + \mu t + \sigma^2t^2/2$. The integral of the Gaussian part equals 1, leaving $e^{\mu t + \sigma^2t^2/2}$. $\square$

**Standard Normal:** $Z \sim \mathcal{N}(0,1)$ has PDF $\phi(z) = e^{-z^2/2}/\sqrt{2\pi}$ and CDF $\Phi(z) = P(Z \leq z)$.

Any Gaussian can be standardised: $Z = (X-\mu)/\sigma \sim \mathcal{N}(0,1)$.

Key quantiles: $\Phi(1.645) \approx 0.95$, $\Phi(1.96) \approx 0.975$, $\Phi(2.576) \approx 0.995$.

**68-95-99.7 Rule:**
$$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 0.683$$
$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.954$$
$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.997$$

**Stability properties:**

1. **Linear stability:** $aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$
2. **Additive stability:** $X_1 + X_2 \sim \mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$ when $X_1 \perp\!\!\!\perp X_2$
3. **Closure under conditioning:** the conditional distribution of a Gaussian given a linear observation is Gaussian (developed in §03)

**Maximum entropy:** Among all distributions with mean $\mu$ and variance $\sigma^2$, the Gaussian maximises entropy. This is the information-theoretic justification for its ubiquity.

**Forward reference — Multivariate Gaussian:**

> **Preview: Multivariate Gaussian $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$**
> The $d$-dimensional generalisation replaces the scalar mean with $\boldsymbol{\mu} \in \mathbb{R}^d$
> and the scalar variance with a positive definite covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$:
> $$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$
> The Gaussian process (§06) extends this to infinite dimensions.
>
> → _Full treatment: [§03 Joint Distributions](../03-Joint-Distributions/notes.md)_

**For AI:**
- **Weight initialisation:** Xavier/Glorot initialisation uses $\mathcal{N}(0, 2/(n_\text{in}+n_\text{out}))$; Kaiming uses $\mathcal{N}(0, 2/n_\text{in})$.
- **VAE latent prior:** $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$ is the prior on the latent code. The encoder outputs $(\boldsymbol{\mu}_\phi, \log\sigma^2_\phi)$ and samples $\mathbf{z} = \boldsymbol{\mu}_\phi + \sigma_\phi \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$.
- **Gaussian process:** a prior over functions where any finite collection of function values follows a multivariate Gaussian (§06).
- **SGD noise:** the gradient noise in stochastic gradient descent is approximately Gaussian by the CLT, with covariance proportional to the gradient covariance matrix.
- **Diffusion models:** the forward noising process adds Gaussian noise $q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$.

---

### 3.3 Exponential($\lambda$)

**Story:** Time between consecutive events in a Poisson process with rate $\lambda$. The continuous analogue of the Geometric distribution.

**PDF and CDF:**
$$f_X(x) = \lambda e^{-\lambda x}, \quad x \geq 0 \qquad F_X(x) = 1 - e^{-\lambda x}, \quad x \geq 0$$

**Parameters:** $\lambda > 0$ is the **rate** (events per unit time). The **scale** is $\theta = 1/\lambda$ (mean inter-event time).

**Moments:**

| Quantity | Value |
|---|---|
| Mean | $1/\lambda$ |
| Variance | $1/\lambda^2$ |
| Mode | $0$ (always — the mode is at the boundary) |
| Median | $\log(2)/\lambda$ |
| Skewness | $2$ (always right-skewed, regardless of $\lambda$) |
| Entropy | $1 - \log\lambda$ |

**MGF:**
$$M_X(t) = \frac{\lambda}{\lambda - t}, \quad t < \lambda$$

**Memoryless property (continuous version):** For $s, t > 0$:
$$P(X > s + t \mid X > s) = P(X > t)$$

*Proof:* $P(X > s) = e^{-\lambda s}$. Then $P(X > s+t \mid X > s) = e^{-\lambda(s+t)}/e^{-\lambda s} = e^{-\lambda t} = P(X > t)$. $\square$

The Exponential is the **unique continuous memoryless distribution**. Knowing you have already waited $s$ units gives no information about the remaining wait.

**Relationship to Poisson:** If events arrive as a Poisson process with rate $\lambda$, then the inter-arrival times are i.i.d. Exponential$(\lambda)$. The count of events in $[0,T]$ is Poisson$(\lambda T)$.

**For AI:**
- **Regularisation via sparsity:** exponential priors on weights yield $L^1$ (Lasso) regularisation under MAP estimation.
- **Session modelling:** time between user sessions is modelled as Exponential.
- **Sampling algorithms:** the exponential distribution appears in Gillespie's algorithm (exact stochastic simulation) and in MCMC acceptance steps.

---

### 3.4 Gamma($\alpha$, $\beta$)

**Story:** The sum of $\alpha$ independent Exponential$(\beta)$ random variables. Equivalently, the time until the $\alpha$-th event in a Poisson process with rate $\beta$.

**PDF:**
$$f_X(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0, \quad \alpha > 0, \quad \beta > 0$$

Here $\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\,dt$ is the **gamma function**, satisfying $\Gamma(n) = (n-1)!$ for positive integers.

**Parameters:** $\alpha > 0$ is the **shape** (controls the number of "humps" and skewness), $\beta > 0$ is the **rate** (scale $\theta = 1/\beta$).

**Moments:**

| Quantity | Value |
|---|---|
| Mean | $\alpha/\beta$ |
| Variance | $\alpha/\beta^2$ |
| Mode | $(\alpha-1)/\beta$ for $\alpha \geq 1$; 0 otherwise |
| Skewness | $2/\sqrt{\alpha}$ (decreases as $\alpha$ grows) |

**MGF:**
$$M_X(t) = \left(\frac{\beta}{\beta - t}\right)^\alpha, \quad t < \beta$$

**Special cases:**

| Parameters | Distribution |
|---|---|
| $\alpha = 1$ | Exponential$(\beta)$ |
| $\alpha = k/2$, $\beta = 1/2$ | Chi-squared$(k)$ |
| $\alpha = n$ (integer) | Erlang$(n, \beta)$ |
| $\alpha \to \infty$ (normalised) | $\to$ Gaussian (by CLT) |

**Additive property:** If $X_1 \sim \Gamma(\alpha_1, \beta)$ and $X_2 \sim \Gamma(\alpha_2, \beta)$ independently, then $X_1 + X_2 \sim \Gamma(\alpha_1+\alpha_2, \beta)$ (same rate).

**Shape behaviour:**
- $\alpha < 1$: unbounded at 0, heavy right tail
- $\alpha = 1$: Exponential (monotone decreasing from $\lambda$)
- $\alpha > 1$: bell-shaped, mode at $(\alpha-1)/\beta$, right-skewed
- Large $\alpha$: approximately Gaussian (CLT)

**For AI:**
- **Conjugate prior for Poisson rate:** Gamma$(\alpha, \beta)$ is conjugate to Poisson. After observing $k$ events in $t$ time units, the posterior is Gamma$(\alpha+k, \beta+t)$.
- **Conjugate prior for Gaussian precision:** Gamma is conjugate to the precision ($1/\sigma^2$) of a Gaussian with known mean.
- **Variational inference:** the Gamma distribution is used as the variational posterior for non-negative latent variables.

---

### 3.5 Beta($\alpha$, $\beta$)

**Story:** A distribution over probability values in $(0,1)$. The natural prior for the unknown success probability of a Bernoulli experiment.

**PDF:**
$$f_X(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha,\beta)}, \quad x \in (0,1), \quad \alpha > 0, \quad \beta > 0$$

where the **Beta function** $B(\alpha,\beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$ is the normalising constant.

**Moments:**

| Quantity | Value |
|---|---|
| Mean | $\alpha/(\alpha+\beta)$ |
| Variance | $\frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$ |
| Mode | $(\alpha-1)/(\alpha+\beta-2)$ for $\alpha, \beta > 1$ |

The mean depends only on the ratio $\alpha/(\alpha+\beta)$. The **concentration** $\kappa = \alpha + \beta$ controls the spread: larger $\kappa$ means more concentration around the mean.

**Shape behaviour:**

| Condition | Shape |
|---|---|
| $\alpha < 1, \beta < 1$ | U-shaped (bimodal at 0 and 1) |
| $\alpha = \beta = 1$ | Uniform(0,1) |
| $\alpha = \beta > 1$ | Symmetric bell, centred at 0.5 |
| $\alpha > \beta$ | Right-skewed (mass towards 1) |
| $\alpha < \beta$ | Left-skewed (mass towards 0) |
| $\alpha, \beta \to \infty$, ratio fixed | Concentrates at $\alpha/(\alpha+\beta)$ |

**Pseudocounts interpretation:** Beta$(\alpha, \beta)$ encodes the belief arising from having seen $\alpha - 1$ successes and $\beta - 1$ failures. Beta$(1,1) = \text{Uniform}(0,1)$ encodes no prior knowledge.

**Relationship to Dirichlet:** Beta$(\alpha, \beta) = \text{Dirichlet}(\alpha, \beta)$ — the $K=2$ special case.

**For AI:**
- **RLHF preference model:** the Bradley-Terry model places a Beta prior on the probability that one response is preferred over another.
- **Proportion modelling:** the fraction of positive tokens, the click-through rate, and the fraction of attended tokens are all modelled with Beta distributions.
- **Conjugate prior for Binomial:** after $k$ successes in $n$ trials, the posterior on $p$ is Beta$(\alpha + k, \beta + n - k)$.
- **Beta-VAE:** the $\beta$-VAE regularises the KL term with a $\beta > 1$ coefficient, encouraging disentanglement of latent variables.

---

### 3.6 Dirichlet($\boldsymbol{\alpha}$)

**Story:** A distribution over probability vectors on the simplex $\Delta^{K-1} = \{\mathbf{p} : p_k \geq 0, \sum_k p_k = 1\}$. The multivariate generalisation of Beta; the natural prior for the parameter of a Categorical distribution.

**PDF:** For $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$ with $\alpha_k > 0$:
$$f(\mathbf{p}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K p_k^{\alpha_k - 1}, \quad \mathbf{p} \in \Delta^{K-1}$$

where $B(\boldsymbol{\alpha}) = \Gamma(\alpha_1)\cdots\Gamma(\alpha_K)/\Gamma(\alpha_0)$ and $\alpha_0 = \sum_k \alpha_k$.

**Moments:**

| Quantity | Value |
|---|---|
| $\mathbb{E}[p_k]$ | $\alpha_k / \alpha_0$ |
| $\operatorname{Var}(p_k)$ | $\frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0+1)}$ |
| $\operatorname{Cov}(p_j, p_k)$ | $-\frac{\alpha_j\alpha_k}{\alpha_0^2(\alpha_0+1)}$ for $j \neq k$ |

The mean probability vector is $\boldsymbol{\alpha}/\alpha_0$ — normalised concentration parameters.

**Concentration parameter $\alpha_0 = \sum_k \alpha_k$:**
- Small $\alpha_0$ (e.g., $\alpha_k = 0.1$): **sparse** — samples concentrate near corners of the simplex
- $\alpha_0 = K$ with all $\alpha_k = 1$: uniform over simplex
- Large $\alpha_0$: **concentrated** — samples cluster near the mean $\boldsymbol{\alpha}/\alpha_0$

**Symmetric Dirichlet:** When all $\alpha_k = \alpha$ (scalar), the distribution is exchangeable across categories. The parameter $\alpha$ controls concentration:
- $\alpha < 1$: sparse, near-one-hot samples
- $\alpha = 1$: uniform over simplex
- $\alpha > 1$: dense, near-uniform samples

**Marginals:** Each marginal $p_k \sim \text{Beta}(\alpha_k, \alpha_0 - \alpha_k)$.

**For AI:**
- **LDA (Latent Dirichlet Allocation):** document topic proportions $\boldsymbol{\theta}_d \sim \text{Dirichlet}(\boldsymbol{\alpha})$. A small $\alpha$ enforces document sparsity — each document covers few topics.
- **Token vocabulary priors:** Dirichlet priors on subword token distributions encode assumptions about language patterns.
- **Bayesian categorical models:** Dirichlet$(\boldsymbol{\alpha})$ is the conjugate prior to Categorical$(\mathbf{p})$. After observing counts $\mathbf{c} = (c_1, \ldots, c_K)$, the posterior is Dirichlet$(\boldsymbol{\alpha} + \mathbf{c})$.

---

### 3.7 Student-$t$($\nu$)

**Story:** The distribution of the $t$-statistic when estimating the mean of a Gaussian with unknown variance from small samples. A Gaussian with heavier tails — more robust to outliers.

**PDF:**
$$f_X(x) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}, \quad x \in \mathbb{R}$$

**Parameter:** $\nu > 0$ is the **degrees of freedom**. Controls tail heaviness.

**Moments:**

| Quantity | Value | When |
|---|---|---|
| Mean | $0$ | $\nu > 1$ |
| Variance | $\nu/(\nu-2)$ | $\nu > 2$ |
| Skewness | $0$ | $\nu > 3$ |
| Kurtosis | $6/(\nu-4)$ | $\nu > 4$ |

The moments do **not exist** for small $\nu$: variance undefined for $\nu \leq 2$, mean undefined for $\nu \leq 1$ (Cauchy distribution).

**Tail heaviness:** The tails decay as $|x|^{-(\nu+1)}$ — **polynomial** decay, much heavier than the Gaussian's $e^{-x^2/2}$ decay. For $\nu = 1$: Cauchy distribution (variance $= \infty$). For $\nu \to \infty$: $\to \mathcal{N}(0,1)$.

**Gaussian construction:** $T = Z / \sqrt{\chi^2_\nu/\nu}$ where $Z \sim \mathcal{N}(0,1) \perp\!\!\!\perp \chi^2_\nu$. This gives $T \sim t_\nu$.

**Location-scale family:** The general Student-$t$ with mean $\mu$ and scale $\sigma$ has PDF obtained by replacing $x$ with $(x-\mu)/\sigma$ and multiplying by $1/\sigma$.

**For AI:**
- **Robust regression:** Student-$t$ likelihood replaces Gaussian when outliers are present. Heavy tails assign higher probability to extreme residuals, reducing their influence on parameter estimates.
- **Bayesian neural networks:** Student-$t$ priors on weights provide robustness; as $\nu$ increases, the prior approaches Gaussian.
- **Uncertainty estimation:** predictive distributions in small-data regimes are better modelled with $t_\nu$ than Gaussian to reflect parameter uncertainty.
- **Variational inference:** the Student-$t$ arises as a scale mixture of Gaussians: $X \mid V \sim \mathcal{N}(0, V)$ and $1/V \sim \Gamma(\nu/2, \nu/2)$ gives marginally $X \sim t_\nu$.

---

## 4. Distribution Relationships

### 4.1 Limiting Relationships

**Binomial → Poisson (rare events limit):**

When $n \to \infty$, $p \to 0$, $np \to \lambda$: $\operatorname{Binomial}(n,p) \to \operatorname{Poisson}(\lambda)$.

Rule of thumb: approximation is good when $n \geq 20$ and $p \leq 0.05$.

**Binomial → Normal (Central Limit Theorem preview):**

When $n \to \infty$: $(X - np)/\sqrt{np(1-p)} \to \mathcal{N}(0,1)$.

> → _Full proof: [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md)_

**Gamma → Normal (large shape):**

When $\alpha \to \infty$ (fixed $\beta$): $(\text{Gamma}(\alpha,\beta) - \alpha/\beta)/\sqrt{\alpha}/\beta \to \mathcal{N}(0,1)$.

**Poisson → Normal (large rate):**

When $\lambda \to \infty$: $(X - \lambda)/\sqrt{\lambda} \to \mathcal{N}(0,1)$.

**Beta → Dirac (large concentration):**

When $\alpha, \beta \to \infty$ with $\alpha/(\alpha+\beta) = \mu$ fixed: Beta$(\alpha,\beta) \to \delta(\mu)$.

**Student-$t$ → Normal:**

As $\nu \to \infty$: $t_\nu \to \mathcal{N}(0,1)$. For $\nu \geq 30$, the approximation is excellent.

```
LIMITING RELATIONSHIPS
════════════════════════════════════════════════════════════════════════

  Binomial(n,p) ─── n→∞, p→0, np=λ ──→ Poisson(λ)
       │
       │  n→∞ (CLT)
       ▼
  Gaussian N(np, np(1-p))

  Gamma(α,β) ──── α→∞ ────────────→ Gaussian (CLT)
  Poisson(λ) ──── λ→∞ ────────────→ Gaussian (CLT)
  Student-t(ν) ── ν→∞ ────────────→ Gaussian N(0,1)
  Beta(α,β) ───── α,β→∞ ──────────→ Dirac(α/(α+β))

════════════════════════════════════════════════════════════════════════
```

### 4.2 Conjugate Pairs Table

| Likelihood | Prior | Posterior | Updated Parameters |
|---|---|---|---|
| Bernoulli$(p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + k, \beta + n - k)$ | $k$ successes in $n$ trials |
| Binomial$(n, p)$ | Beta$(\alpha, \beta)$ | Beta$(\alpha + k, \beta + n - k)$ | same as Bernoulli |
| Categorical$(\mathbf{p})$ | Dirichlet$(\boldsymbol{\alpha})$ | Dirichlet$(\boldsymbol{\alpha} + \mathbf{c})$ | $\mathbf{c}$ = observed counts |
| Multinomial$(n, \mathbf{p})$ | Dirichlet$(\boldsymbol{\alpha})$ | Dirichlet$(\boldsymbol{\alpha} + \mathbf{c})$ | same as Categorical |
| Poisson$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum k_i, \beta + n)$ | $n$ observations |
| Exponential$(\lambda)$ | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + n, \beta + \sum x_i)$ | $n$ observations |
| Gaussian$(μ, \sigma^2)$ (known $\sigma^2$) | $\mathcal{N}(\mu_0, \tau^2)$ | $\mathcal{N}(\mu_n, \tau_n^2)$ | Precision-weighted average |

Full derivations in §7 (Conjugate Priors).

---

## 5. Moment Generating Functions

### 5.1 Definition and Uniqueness

The **moment generating function (MGF)** of a random variable $X$ is:
$$M_X(t) = \mathbb{E}[e^{tX}], \quad t \in \mathbb{R}$$

when this expectation is finite in an open interval around $t = 0$.

**Why "moment generating":** By Taylor-expanding $e^{tX}$:
$$M_X(t) = \mathbb{E}\!\left[\sum_{k=0}^\infty \frac{(tX)^k}{k!}\right] = \sum_{k=0}^\infty \frac{\mathbb{E}[X^k]}{k!} t^k$$

Differentiating $k$ times and evaluating at $t=0$:
$$M_X^{(k)}(0) = \mathbb{E}[X^k]$$

The $k$-th derivative of the MGF at zero is the $k$-th raw moment.

**Uniqueness theorem:** If $M_X(t)$ exists in an open interval containing 0, it uniquely determines the distribution of $X$. Two random variables with the same MGF have the same distribution.

**Cumulant generating function (CGF):** $K_X(t) = \log M_X(t)$. Its derivatives at 0 give the **cumulants**: $K_X'(0) = \mathbb{E}[X]$, $K_X''(0) = \operatorname{Var}(X)$, $K_X'''(0) = \mathbb{E}[(X-\mu)^3]$ (third central moment), etc.

### 5.2 MGF Product Rule

**Theorem:** If $X \perp\!\!\!\perp Y$, then $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$.

*Proof:* $M_{X+Y}(t) = \mathbb{E}[e^{t(X+Y)}] = \mathbb{E}[e^{tX}e^{tY}] = \mathbb{E}[e^{tX}]\mathbb{E}[e^{tY}] = M_X(t)M_Y(t)$, where the third equality uses independence. $\square$

**Application:** Proving that sums of independent distributions stay in the same family:
- $\text{Bernoulli}(p)^{\oplus n} = \text{Binomial}(n,p)$: $M(t)^n = (1-p+pe^t)^n$
- $\text{Poisson}(\lambda_1) \oplus \text{Poisson}(\lambda_2) = \text{Poisson}(\lambda_1+\lambda_2)$: $e^{\lambda_1(e^t-1)} \cdot e^{\lambda_2(e^t-1)} = e^{(\lambda_1+\lambda_2)(e^t-1)}$
- $\text{Gamma}(\alpha_1,\beta) \oplus \text{Gamma}(\alpha_2,\beta) = \text{Gamma}(\alpha_1+\alpha_2,\beta)$: $(\beta/(\beta-t))^{\alpha_1+\alpha_2}$
- $\mathcal{N}(\mu_1,\sigma_1^2) \oplus \mathcal{N}(\mu_2,\sigma_2^2) = \mathcal{N}(\mu_1+\mu_2,\sigma_1^2+\sigma_2^2)$: product of exponential-of-quadratics

### 5.3 MGFs of Key Distributions

| Distribution | MGF $M_X(t)$ | Domain |
|---|---|---|
| Bernoulli$(p)$ | $1-p+pe^t$ | $\mathbb{R}$ |
| Binomial$(n,p)$ | $(1-p+pe^t)^n$ | $\mathbb{R}$ |
| Geometric$(p)$ | $pe^t/(1-(1-p)e^t)$ | $t < -\log(1-p)$ |
| Poisson$(\lambda)$ | $e^{\lambda(e^t-1)}$ | $\mathbb{R}$ |
| Uniform$(a,b)$ | $(e^{tb}-e^{ta})/(t(b-a))$ | $\mathbb{R}$ |
| Exponential$(\lambda)$ | $\lambda/(\lambda-t)$ | $t < \lambda$ |
| Gamma$(\alpha,\beta)$ | $(\beta/(\beta-t))^\alpha$ | $t < \beta$ |
| Gaussian$(\mu,\sigma^2)$ | $e^{\mu t + \sigma^2t^2/2}$ | $\mathbb{R}$ |
| Beta$(\alpha,\beta)$ | $1 + \sum_{k=1}^\infty (\prod_{r=0}^{k-1}\frac{\alpha+r}{\alpha+\beta+r})\frac{t^k}{k!}$ | $\mathbb{R}$ |

Note: the Beta MGF does not have a simple closed form; the Dirichlet and Categorical are typically characterised by their probability generating functions or characteristic functions instead.

> → _Full treatment of MGF applications and cumulants: [§04 Expectation and Moments](../04-Expectation-and-Moments/notes.md)_

---

## 6. The Exponential Family

The **exponential family** is the most important unifying framework in probability and statistics. It explains why Bernoulli, Gaussian, Poisson, Gamma, Beta, Dirichlet, and Categorical all share the same algorithmic properties.

### 6.1 Canonical Form

A distribution belongs to the exponential family if its PDF (or PMF) can be written as:
$$p(x; \boldsymbol{\eta}) = h(x) \exp\!\bigl(\boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta})\bigr)$$

where:
- $\boldsymbol{\eta} \in \mathbb{R}^k$ — **natural parameters** (the parameterisation used in optimisation)
- $T(x) \in \mathbb{R}^k$ — **sufficient statistics** (captures all information about $\boldsymbol{\eta}$ in the data)
- $A(\boldsymbol{\eta}) = \log \int h(x)\exp(\boldsymbol{\eta}^\top T(x))\,dx$ — **log-partition function** (normalisation constant in log space)
- $h(x)$ — **base measure** (does not depend on $\boldsymbol{\eta}$)

The canonical form separates the "shape" of the distribution ($h$ and $T$) from the parameterisation ($\boldsymbol{\eta}$ and $A$).

### 6.2 Members and Their Parameters

| Distribution | Natural param $\boldsymbol{\eta}$ | Sufficient stat $T(x)$ | Log-partition $A(\boldsymbol{\eta})$ | Base measure $h(x)$ |
|---|---|---|---|---|
| Bernoulli$(p)$ | $\log\frac{p}{1-p}$ | $x$ | $\log(1+e^\eta)$ | $1$ |
| Binomial$(n,p)$ | $\log\frac{p}{1-p}$ | $x$ | $n\log(1+e^\eta)$ | $\binom{n}{x}$ |
| Poisson$(\lambda)$ | $\log\lambda$ | $x$ | $e^\eta$ | $1/x!$ |
| Exponential$(\lambda)$ | $-\lambda$ | $x$ | $-\log(-\eta)$ | $1$ |
| Gaussian$(\mu,\sigma^2)$ (both unknown) | $(\mu/\sigma^2, -1/(2\sigma^2))$ | $(x, x^2)$ | $-\eta_1^2/(4\eta_2) - \frac{1}{2}\log(-2\eta_2)$ | $1/\sqrt{2\pi}$ |
| Gamma$(\alpha,\beta)$ | $(\alpha-1, -\beta)$ | $(\log x, x)$ | $\log\Gamma(\eta_1+1) - (\eta_1+1)\log(-\eta_2)$ | $1$ |
| Beta$(\alpha,\beta)$ | $(\alpha-1, \beta-1)$ | $(\log x, \log(1-x))$ | $\log B(\eta_1+1,\eta_2+1)$ | $1$ |
| Categorical$(\mathbf{p})$ | $(\log p_1/p_K, \ldots, \log p_{K-1}/p_K)$ | $(x_1,\ldots,x_{K-1})$ | $\log(1+\sum_{k=1}^{K-1}e^{\eta_k})$ | $1$ |

### 6.3 The Log-Partition Function

The log-partition function $A(\boldsymbol{\eta})$ is **convex** (always, by Hölder's inequality). Its derivatives generate the cumulants:

$$\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \mathbb{E}_{p_{\boldsymbol{\eta}}}[T(X)]$$
$$\nabla_{\boldsymbol{\eta}}^2 A(\boldsymbol{\eta}) = \operatorname{Cov}_{p_{\boldsymbol{\eta}}}[T(X)]$$

*Proof:* $A(\boldsymbol{\eta}) = \log \int h(x)e^{\boldsymbol{\eta}^\top T(x)}\,dx$. Differentiating under the integral (valid by dominated convergence):
$$\frac{\partial A}{\partial \eta_j} = \frac{\int T_j(x) h(x)e^{\boldsymbol{\eta}^\top T(x)}\,dx}{\int h(x)e^{\boldsymbol{\eta}^\top T(x)}\,dx} = \mathbb{E}[T_j(X)]$$
The second derivative similarly gives the covariance. $\square$

**Consequence:** Computing moments of any exponential family member reduces to differentiating $A(\boldsymbol{\eta})$ — a single convex function. This is an extraordinary computational shortcut.

**Example — Poisson:** $A(\eta) = e^\eta$. So $\mathbb{E}[X] = A'(\eta) = e^\eta = \lambda$ and $\operatorname{Var}(X) = A''(\eta) = e^\eta = \lambda$ — confirming the Poisson mean-equals-variance property directly from the log-partition function.

### 6.4 Sufficient Statistics

**Definition (Fisher 1922):** A statistic $T(X^{(1)}, \ldots, X^{(n)})$ is **sufficient** for $\boldsymbol{\eta}$ if the conditional distribution $p(X^{(1)}, \ldots, X^{(n)} \mid T = t; \boldsymbol{\eta})$ does not depend on $\boldsymbol{\eta}$.

Intuitively: $T$ captures all information in the data about $\boldsymbol{\eta}$.

**Fisher-Neyman factorisation theorem:** $T$ is sufficient if and only if the likelihood factors as:
$$p(\mathbf{x}; \boldsymbol{\eta}) = g(T(\mathbf{x}), \boldsymbol{\eta}) \cdot h(\mathbf{x})$$

For exponential families with $n$ i.i.d. observations:
$$p(\mathbf{x}; \boldsymbol{\eta}) = h(\mathbf{x}) \exp\!\bigl(\boldsymbol{\eta}^\top \underbrace{\sum_{i=1}^n T(x^{(i)})}_{\text{sufficient stat}} - nA(\boldsymbol{\eta})\bigr)$$

**Examples:**
- Bernoulli: $T = \sum x_i$ (total successes) is sufficient for $p$
- Gaussian (both params unknown): $(T_1, T_2) = (\sum x_i, \sum x_i^2)$ is sufficient for $(\mu, \sigma^2)$
- Poisson: $T = \sum x_i$ (total count) is sufficient for $\lambda$

**Pitman-Koopman-Darmois theorem:** The only distributions with a fixed-dimension sufficient statistic (regardless of sample size $n$) are the exponential family members.

### 6.5 ML Implications

**Softmax as categorical exponential family:**

The Categorical$(K)$ natural parameter is $\boldsymbol{\eta} = (\log p_1/p_K, \ldots, \log p_{K-1}/p_K)$. Inverting:
$$p_k = \frac{e^{\eta_k}}{\sum_{j=1}^K e^{\eta_j}} = \operatorname{softmax}(\boldsymbol{\eta})_k$$

The softmax is therefore the **canonical link function** of the Categorical exponential family. Neural networks output logits $\mathbf{z}$ which are exactly the natural parameters $\boldsymbol{\eta}$.

**Log-sum-exp = log-partition function:**

The logsumexp operation $\log\sum_k e^{z_k}$ is the log-partition function $A(\boldsymbol{\eta})$ of the Categorical distribution. The numerically stable form $\max(\mathbf{z}) + \log\sum_k e^{z_k - \max(\mathbf{z})}$ is directly derived from properties of $A$.

**Natural gradient:** The Fisher information matrix of an exponential family is $F = \nabla^2 A(\boldsymbol{\eta}) = \operatorname{Cov}[T(X)]$. The natural gradient $F^{-1}\nabla_{\boldsymbol{\eta}}\mathcal{L}$ accounts for the curvature of the distribution manifold, giving parameter-invariant updates. K-FAC approximates $F^{-1}$ for neural networks.

**MLE for exponential families:** The MLE of $\boldsymbol{\eta}$ satisfies:
$$\nabla A(\hat{\boldsymbol{\eta}}) = \frac{1}{n}\sum_{i=1}^n T(x^{(i)})$$

meaning the expected sufficient statistics under the model equal the empirical sufficient statistics — a **moment matching** condition.

---

## 7. Conjugate Priors in Bayesian Inference

Bayesian inference requires computing the posterior $p(\boldsymbol{\theta} \mid \mathbf{x}) \propto p(\mathbf{x} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})$. For most priors, this requires numerical integration. **Conjugate priors** are the special priors for which the posterior stays in the same family — making inference analytic.

### 7.1 Conjugacy Definition

**Definition:** A prior $p(\boldsymbol{\theta})$ is **conjugate** to a likelihood $p(\mathbf{x} \mid \boldsymbol{\theta})$ if the posterior $p(\boldsymbol{\theta} \mid \mathbf{x})$ is in the same family as the prior.

For exponential family likelihoods, conjugate priors always exist and take a canonical form. The general conjugate prior to $p(x; \boldsymbol{\eta}) = h(x)\exp(\boldsymbol{\eta}^\top T(x) - A(\boldsymbol{\eta}))$ is:
$$p(\boldsymbol{\eta}; \boldsymbol{\chi}, \nu) \propto \exp\!\bigl(\boldsymbol{\eta}^\top \boldsymbol{\chi} - \nu A(\boldsymbol{\eta})\bigr)$$

After observing $n$ data points $x^{(1)}, \ldots, x^{(n)}$:
$$p(\boldsymbol{\eta} \mid \mathbf{x}) \propto \exp\!\left(\boldsymbol{\eta}^\top \left(\boldsymbol{\chi} + \sum_{i=1}^n T(x^{(i)})\right) - (\nu + n)A(\boldsymbol{\eta})\right)$$

The hyperparameters update simply: $\boldsymbol{\chi} \to \boldsymbol{\chi} + \sum_i T(x^{(i)})$ and $\nu \to \nu + n$.

### 7.2 Beta-Bernoulli/Binomial

**Model:** $p \sim \text{Beta}(\alpha, \beta)$, $X_1, \ldots, X_n \mid p \overset{\text{iid}}{\sim} \operatorname{Bern}(p)$.

**Likelihood:** $p(\mathbf{x} \mid p) = p^k(1-p)^{n-k}$ where $k = \sum_i x_i$.

**Posterior:**
$$p(p \mid \mathbf{x}) \propto p^{\alpha+k-1}(1-p)^{\beta+n-k-1} \implies p \mid \mathbf{x} \sim \text{Beta}(\alpha+k, \beta+n-k)$$

**Pseudocounts interpretation:** The prior Beta$(\alpha, \beta)$ encodes $\alpha - 1$ prior successes and $\beta - 1$ prior failures. After seeing $k$ successes in $n$ trials, the posterior is Beta$(\alpha + k, \beta + n - k)$ — simply adding counts.

**Posterior mean:**
$$\mathbb{E}[p \mid \mathbf{x}] = \frac{\alpha + k}{\alpha + \beta + n}$$

This is a **shrinkage estimator** between the prior mean $\alpha/(\alpha+\beta)$ and the MLE $k/n$:
$$\mathbb{E}[p \mid \mathbf{x}] = \underbrace{\frac{\alpha+\beta}{\alpha+\beta+n}}_{\text{weight on prior}} \cdot \frac{\alpha}{\alpha+\beta} + \underbrace{\frac{n}{\alpha+\beta+n}}_{\text{weight on data}} \cdot \frac{k}{n}$$

As $n \to \infty$, the posterior concentrates at the MLE $k/n$.

### 7.3 Dirichlet-Categorical/Multinomial

**Model:** $\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$, $X_1, \ldots, X_n \mid \mathbf{p} \overset{\text{iid}}{\sim} \operatorname{Cat}(\mathbf{p})$.

**Posterior:**
$$\mathbf{p} \mid \mathbf{x} \sim \text{Dir}(\boldsymbol{\alpha} + \mathbf{c})$$

where $\mathbf{c} = (c_1, \ldots, c_K)$ are the observed category counts ($c_k = \sum_i \mathbf{1}[x^{(i)} = k]$).

**Posterior mean:** $\mathbb{E}[p_k \mid \mathbf{x}] = (\alpha_k + c_k)/(\alpha_0 + n)$.

**Add-one (Laplace) smoothing:** Setting $\boldsymbol{\alpha} = \mathbf{1}$ (uniform Dirichlet prior) gives the Laplace smoothed estimate $\hat{p}_k = (c_k + 1)/(n + K)$ — the standard technique for avoiding zero probabilities in language model unigrams.

**For LDA:** Each document $d$ has topic proportions $\boldsymbol{\theta}_d \sim \text{Dir}(\boldsymbol{\alpha})$. Each topic $k$ has word distribution $\boldsymbol{\phi}_k \sim \text{Dir}(\boldsymbol{\beta})$. Words are drawn as $w \sim \operatorname{Cat}(\boldsymbol{\phi}_{z_d})$ where $z_d \sim \operatorname{Cat}(\boldsymbol{\theta}_d)$.

### 7.4 Gamma-Poisson

**Model:** $\lambda \sim \Gamma(\alpha, \beta)$, $X_1, \ldots, X_n \mid \lambda \overset{\text{iid}}{\sim} \operatorname{Poisson}(\lambda)$.

**Posterior:**
$$\lambda \mid \mathbf{x} \sim \Gamma\!\left(\alpha + \sum_{i=1}^n x_i, \; \beta + n\right)$$

**Posterior mean:** $(\alpha + \sum x_i)/(\beta + n)$ — shrinkage between prior mean $\alpha/\beta$ and MLE $\bar{x}$.

**Interpretations:** The prior Gamma$(\alpha, \beta)$ encodes "$\alpha$ events observed over $\beta$ prior time units." After observing $\sum x_i$ events in $n$ new time units, the posterior is Gamma$(\alpha + \sum x_i, \beta + n)$.

### 7.5 Normal-Normal

**Model:** $\mu \sim \mathcal{N}(\mu_0, \tau^2)$, $X_1, \ldots, X_n \mid \mu \overset{\text{iid}}{\sim} \mathcal{N}(\mu, \sigma^2)$ with known $\sigma^2$.

**Posterior:**
$$\mu \mid \mathbf{x} \sim \mathcal{N}(\mu_n, \tau_n^2)$$

where:
$$\frac{1}{\tau_n^2} = \frac{1}{\tau^2} + \frac{n}{\sigma^2} \qquad \mu_n = \tau_n^2\!\left(\frac{\mu_0}{\tau^2} + \frac{n\bar{x}}{\sigma^2}\right)$$

**Precision-weighted average:** The posterior mean is:
$$\mu_n = \frac{\tau^{-2}}{\tau^{-2} + n\sigma^{-2}} \mu_0 + \frac{n\sigma^{-2}}{\tau^{-2} + n\sigma^{-2}} \bar{x}$$

a precision-weighted combination of prior mean and sample mean. As $n \to \infty$, $\mu_n \to \bar{x}$ and $\tau_n^2 \to 0$ (posterior concentrates at MLE).

---

## 8. ML Applications

### 8.1 Language Models: Categorical and Temperature

Every forward pass of a language model computes:
1. Logits $\mathbf{z} \in \mathbb{R}^V$ over vocabulary of size $V$
2. Token probabilities $\mathbf{p} = \operatorname{softmax}(\mathbf{z}/\tau)$ at temperature $\tau$
3. Next token $x_t \sim \operatorname{Cat}(\mathbf{p})$

The cross-entropy training loss is $-\log p_{x_t} = -z_{x_t}/\tau + \log\sum_k e^{z_k/\tau}$, which is exactly $-\eta_{x_t} + A(\boldsymbol{\eta}/\tau)$ — the NLL of a Categorical exponential family member.

**Perplexity:** $\operatorname{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^T \log p(x_t \mid x_{<t})\right) = \exp(H(p, \hat{p}))$ — the exponentiated cross-entropy, measuring the effective vocabulary size.

### 8.2 VAEs: Gaussian Reparameterisation and KL Term

The VAE ELBO is:
$$\mathcal{L}(\phi, \theta) = \mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))$$

With $q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi, \operatorname{diag}(\boldsymbol{\sigma}^2_\phi))$ and $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$:

$$D_{\mathrm{KL}} = \frac{1}{2}\sum_{j=1}^d \left(\sigma_{\phi,j}^2 + \mu_{\phi,j}^2 - 1 - \log\sigma_{\phi,j}^2\right)$$

This closed-form KL is derived from the Gaussian MGF. The **reparameterisation trick** enables backpropagation: $\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon}$, $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, I)$.

### 8.3 Dropout as Bernoulli Masking

Dropout applies a Bernoulli$(1-p_\text{drop})$ mask independently to each activation:
$$\tilde{h}_i = h_i \cdot m_i, \quad m_i \overset{\text{iid}}{\sim} \operatorname{Bern}(1-p_\text{drop})$$

At test time, the expectation $\mathbb{E}[m_i] = 1-p_\text{drop}$ is used: $\tilde{h}_i = (1-p_\text{drop}) h_i$.

This is equivalent to training an ensemble of $2^d$ networks (where $d$ is the number of units) sharing weights, with each network sampled at each step.

### 8.4 RLHF: Bradley-Terry and Beta Prior

The **Bradley-Terry model** assigns probability to human preferences:
$$P(\text{response A preferred over B} \mid r_A, r_B) = \sigma(r_A - r_B)$$

where $r_A, r_B$ are learned reward scalars and $\sigma$ is the sigmoid (Bernoulli logit link). The reward difference $r_A - r_B$ follows the Bernoulli exponential family with natural parameter $r_A - r_B$.

A Beta prior on the preference probability $p = \sigma(r_A - r_B)$ regularises the reward model toward a neutral preference ($p \approx 0.5$).

### 8.5 Diffusion Models: Gaussian Noise Schedule

The forward process adds Gaussian noise over $T$ steps:
$$q(x_t \mid x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1}, \beta_t I)$$

Using the Gaussian stability under sums (§3.2), the marginal at step $t$ is:
$$q(x_t \mid x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}\, x_0, (1-\bar{\alpha}_t) I), \quad \bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$$

The reverse process $p_\theta(x_{t-1} \mid x_t)$ is also Gaussian (for small $\beta_t$), with mean predicted by the neural network.

### 8.6 Topic Models (LDA)

Latent Dirichlet Allocation uses three distributions from this section:
1. $\boldsymbol{\theta}_d \sim \text{Dir}(\boldsymbol{\alpha})$ — topic proportions per document (Dirichlet)
2. $z_{dn} \sim \operatorname{Cat}(\boldsymbol{\theta}_d)$ — topic assignment per word (Categorical)
3. $w_{dn} \sim \operatorname{Cat}(\boldsymbol{\phi}_{z_{dn}})$ — word given topic (Categorical with Dirichlet prior $\boldsymbol{\phi}_k \sim \text{Dir}(\boldsymbol{\beta})$)

The full joint is:
$$p(\mathbf{W}, \mathbf{Z}, \boldsymbol{\Theta}, \boldsymbol{\Phi}) = \prod_k \text{Dir}(\boldsymbol{\phi}_k;\boldsymbol{\beta}) \prod_d \text{Dir}(\boldsymbol{\theta}_d;\boldsymbol{\alpha}) \prod_{dn} \operatorname{Cat}(z_{dn};\boldsymbol{\theta}_d) \operatorname{Cat}(w_{dn};\boldsymbol{\phi}_{z_{dn}})$$

Inference uses collapsed Gibbs sampling (Dirichlet-Categorical conjugacy allows marginalising $\boldsymbol{\Theta}$ and $\boldsymbol{\Phi}$).

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Confusing the Gaussian parameter $\sigma^2$ with $\sigma$ | $\mathcal{N}(\mu, \sigma^2)$ takes variance as second argument in most conventions, but NumPy's `np.random.normal(mu, sigma)` takes std dev | Always state whether you use variance or std dev; in code use `scale=sigma` (std dev) |
| 2 | Assuming Poisson when variance > mean | Poisson requires Var = Mean exactly; real count data often has overdispersion | Check the variance-to-mean ratio; use Negative Binomial for overdispersed data |
| 3 | Using Binomial for small $p$ with large $n$ | Numerical issues with $\binom{n}{k}p^k(1-p)^{n-k}$ when $n$ is large | Switch to Poisson approximation or compute in log-space |
| 4 | Interpreting Beta$(\alpha,\beta)$ parameters as probabilities | $\alpha$ and $\beta$ are pseudocount concentrations, not the mean itself | Mean is $\alpha/(\alpha+\beta)$; $\alpha=\beta=1$ is uniform, not Beta$(0.5, 0.5)$ |
| 5 | Forgetting the Dirichlet concentration parameter controls sparsity | $\alpha_k < 1$ produces sparse samples; $\alpha_k > 1$ produces dense ones | Set $\alpha_k < 1$ for topic models expecting sparse documents |
| 6 | Confusing Categorical and one-hot encoding | A Categorical sample is an integer; its one-hot encoding is a vector | In PyTorch, `Categorical.sample()` returns indices, not one-hot vectors |
| 7 | Applying Student-$t$ formulas when $\nu \leq 2$ | Variance is undefined for $\nu \leq 2$; mean undefined for $\nu \leq 1$ | Always check $\nu$ before using moment formulas; for $\nu = 1$ (Cauchy), variance is $\infty$ |
| 8 | Treating Exponential$(\lambda)$ rate and scale interchangeably | Some sources use rate $\lambda$ (mean $= 1/\lambda$); others use scale $\theta = 1/\lambda$ (mean $= \theta$) | Verify convention: SciPy `stats.expon(scale=theta)` uses scale; PyTorch `Exponential(rate=lambda)` uses rate |
| 9 | Using Normal approximation to Binomial when $np < 5$ or $n(1-p) < 5$ | CLT kicks in slowly in the tails; approximation is poor for extreme $p$ | Use exact Binomial PMF or Poisson approximation |
| 10 | Claiming exponential family membership for Student-$t$ | The Student-$t$ is NOT in the exponential family (its normalising constant depends on $\nu$ in a non-exponential way) | Student-$t$ is a scale mixture of Gaussians — handle separately |
| 11 | Confusing natural parameters with mean parameters | The Gaussian natural parameters are $(\mu/\sigma^2, -1/2\sigma^2)$, not $(\mu, \sigma^2)$ | Distinguish mean parameterisation (human-readable) from natural parameterisation (for exponential family theory) |
| 12 | Using the Beta posterior mean instead of mode for MAP estimation | Posterior mean $= (\alpha+k)/(\alpha+\beta+n)$; MAP (mode) $= (\alpha+k-1)/(\alpha+\beta+n-2)$ | For decisions, use posterior mode (MAP) or the full posterior, not mean by default |

---

## 10. Exercises

**Exercise 1 ★ — PMF and Moments**

A biased die has faces weighted so that the probability of face $k$ is proportional to $k$ for $k = 1, 2, 3, 4, 5, 6$.

(a) Find the normalising constant and write the PMF explicitly.
(b) Compute $\mathbb{E}[X]$ and $\operatorname{Var}(X)$.
(c) Find $P(X \geq 4)$.
(d) Is this distribution in the exponential family? Identify $T(x)$, $\eta$, and $A(\eta)$ if so.

**Exercise 2 ★ — Poisson Limit**

A social media post receives clicks at a rate of $\lambda = 0.8$ per minute. Model the number of clicks in a 10-minute window.

(a) Write the PMF and compute $P(X = 5)$, $P(X = 0)$, $P(X \geq 3)$.
(b) Suppose you model this as Binomial$(n=600, p)$ where each second either produces a click or not. Find $p$ and verify the Poisson limit numerically for $k = 5$.
(c) What property of the Poisson means that clicks in disjoint time windows are independent?
(d) If two different posts receive $\lambda_1 = 1.2$ and $\lambda_2 = 0.5$ clicks/minute, what is the distribution of the total clicks per minute? Prove it using MGFs.

**Exercise 3 ★ — Gaussian Properties**

Let $X \sim \mathcal{N}(3, 4)$ (mean 3, variance 4).

(a) Standardise $X$ to obtain $Z \sim \mathcal{N}(0,1)$.
(b) Compute $P(1 \leq X \leq 5)$ using $\Phi$.
(c) If $Y = 2X - 1$, find the distribution of $Y$.
(d) If $X_1, X_2 \overset{\text{iid}}{\sim} \mathcal{N}(3,4)$, find the distribution of $S = X_1 + X_2$.
(e) Verify numerically that the MGF formula gives the correct mean and variance for $X$.

**Exercise 4 ★★ — Beta-Binomial Conjugate Update**

You are estimating the click-through rate $p$ of a button. Your prior belief is Beta$(2, 8)$.

(a) Interpret this prior: what pseudocounts does it encode? What is the prior mean?
(b) You observe 15 clicks in 100 impressions. Write the posterior distribution.
(c) Compute the posterior mean and compare it to the MLE $15/100 = 0.15$.
(d) After how many additional clicks (keeping total impressions fixed at 100) would the posterior mean exceed $0.20$?
(e) Plot the prior, likelihood (rescaled), and posterior as a function of $p$ (implement in Python).

**Exercise 5 ★★ — Exponential Family Identification**

(a) Show that the Geometric$(p)$ distribution belongs to the exponential family. Identify $\eta$, $T(x)$, $A(\eta)$, and $h(x)$.
(b) For the Geometric, compute $\mathbb{E}[X]$ by differentiating $A(\eta)$.
(c) Show that the Negative Binomial$(r, p)$ distribution also belongs to the exponential family.
(d) The uniform distribution $\mathcal{U}(0, b)$ with unknown $b$ — does it belong to the exponential family? Explain.

**Exercise 6 ★★ — Dirichlet-Categorical Posterior**

A language model assigns log-probabilities to tokens. You use a symmetric Dirichlet$(0.1)$ prior over a vocabulary of size $K = 5$ (simplified).

(a) Sample 3 probability vectors from Dir$(0.1 \cdot \mathbf{1}_5)$ and 3 from Dir$(2 \cdot \mathbf{1}_5)$. Describe the visual difference.
(b) You observe the token sequence: $[A, B, A, C, A, B, A]$ (where $\{A,B,C,D,E\}$ are the 5 tokens). Compute the posterior Dirichlet.
(c) Compute the posterior mean probability for each token.
(d) Compare with the Laplace-smoothed MLE estimate. Show they are equal when $\alpha_k = 1$.

**Exercise 7 ★★★ — Softmax as Exponential Family**

(a) Derive the softmax function from the Categorical exponential family canonical form. Show that the log-partition function $A(\boldsymbol{\eta}) = \log\sum_k e^{\eta_k}$ leads to $\mathbb{E}[X_k] = e^{\eta_k}/\sum_j e^{\eta_j} = \operatorname{softmax}(\boldsymbol{\eta})_k$.
(b) Implement `log_softmax(z)` in a numerically stable way (subtract max before exponentiating). Verify it equals `log(softmax(z))` but is more numerically stable for large logits.
(c) The gradient of the cross-entropy loss $\mathcal{L} = -\sum_k y_k \log p_k$ with respect to logits $\mathbf{z}$ is $\mathbf{p} - \mathbf{y}$ where $\mathbf{p} = \operatorname{softmax}(\mathbf{z})$. Derive this result.
(d) Show that temperature scaling $p_k \propto e^{z_k/\tau}$ is equivalent to scaling the natural parameters, and explain why $\tau \to 0$ gives argmax and $\tau \to \infty$ gives uniform.

**Exercise 8 ★★★ — Gaussian VAE KL Term**

The VAE training objective requires:
$$D_{\mathrm{KL}}\!\left(\mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2)) \;\Big\|\; \mathcal{N}(\mathbf{0}, I)\right)$$

(a) Derive the closed-form expression: $\frac{1}{2}\sum_j (\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2)$ using the Gaussian MGF.
(b) Verify this equals 0 when $\mu_j = 0$ and $\sigma_j = 1$ for all $j$.
(c) Implement `kl_gaussian(mu, log_var)` where `log_var = log(sigma^2)` (the standard VAE parameterisation).
(d) Plot the per-dimension KL as a function of $\mu$ (with $\sigma = 1$) and as a function of $\sigma$ (with $\mu = 0$). What do the minima imply for the VAE encoder?

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | Impact on Modern AI |
|---|---|
| **Categorical + softmax** | Every language model output layer is a Categorical exponential family. Logits are natural parameters; cross-entropy is NLL; temperature controls entropy of the output distribution |
| **Gaussian reparameterisation** | Enables end-to-end training of VAEs (2013), diffusion model denoising (2020), and Gaussian noise models in score-based generation |
| **Dirichlet priors** | LDA (2003) topic models still used for document understanding; Dirichlet-process priors in Bayesian nonparametric methods for adaptive-capacity models |
| **Beta-Binomial conjugacy** | RLHF reward modelling uses preference probability estimates; uncertainty-aware sampling uses Beta posteriors for exploration-exploitation |
| **Exponential family unification** | Natural gradient (K-FAC, Shampoo) uses the Fisher information matrix $= \operatorname{Cov}[T(X)]$; the link between logits and natural parameters motivates output-layer design choices |
| **Student-$t$ distribution** | Robust regression and uncertainty quantification in small-data regimes; $t$-distributed stochastic embedding (t-SNE) uses $t_1$ (Cauchy) tails to separate clusters |
| **Poisson** | Language model token position counts, API call modelling, and event-based neural networks (neuromorphic AI) |
| **Gamma-Poisson conjugacy** | Bayesian A/B testing for click-through rates and conversion rates; Thompson sampling for multi-armed bandit |
| **Diffusion Gaussian schedules** | Stable Diffusion, DALL-E, Sora all use Gaussian forward processes; the closed-form $q(x_t \mid x_0)$ enables efficient training without simulating the full Markov chain |
| **Log-partition function** | The logsumexp trick (numerically stable $A(\boldsymbol{\eta})$) is fundamental to FlashAttention's online softmax computation |

---

## 12. Conceptual Bridge

This section forms the vocabulary layer of probability theory. You can now think about every probabilistic model in terms of its component distributions: a Gaussian prior, a Categorical likelihood, a Dirichlet hyperprior. Without this vocabulary, reading a VAE paper or an LDA paper is like trying to read chemistry without knowing the periodic table.

**Looking backward:** The CDF, PDF, and PMF definitions from [§01](../01-Introduction-and-Random-Variables/notes.md) gave the framework; this section fills it with concrete instances. The axioms guaranteed consistency; the named distributions give tractability. Every distribution here satisfies all the axioms of §01 — the Bernoulli is the simplest, the Dirichlet the most complex, but all obey the same rules.

**Looking forward:**
- [§03 Joint Distributions](../03-Joint-Distributions/notes.md) extends to multiple random variables — the multivariate Gaussian, joint densities, marginalisation, and the chain rule of probability.
- [§04 Expectation and Moments](../04-Expectation-and-Moments/notes.md) derives the moments stated here from first principles, introduces the full LOTUS theorem, and develops MGFs as analytical tools.
- [§05 Concentration Inequalities](../05-Concentration-Inequalities/notes.md) bounds how far the distributions here can deviate from their means.
- [§06 Stochastic Processes](../06-Stochastic-Processes/notes.md) proves the CLT — the theorem that explains why the Gaussian is the limiting distribution for so many of the relationships in §4.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  §06/01 Introduction and Random Variables
      ↓  (foundations: axioms, CDF, PDF, Bernoulli/Uniform preview)
  ► §06/02 Common Distributions ◄  ← YOU ARE HERE
      ↓  (full vocabulary: all named distributions, MGFs, exp family)
  §06/03 Joint Distributions
      ↓  (multivariate: joint PDF, marginals, multivariate Gaussian)
  §06/04 Expectation and Moments
      ↓  (derivations: LOTUS, covariance matrix, MGF applications)
  §06/05 Concentration Inequalities
      ↓  (bounds: Markov, Chebyshev, Hoeffding, PAC learning)
  §06/06 Stochastic Processes
      ↓  (CLT: proves the Gaussian limit relationships of §4)
  §06/07 Markov Chains
      (MCMC: uses conjugacy and Gaussian proposals)

════════════════════════════════════════════════════════════════════════
```

The distributions in this section are not a list to memorise — they are a language to think in. When a practitioner says "the model is overconfident," they mean the predicted Categorical is too peaked. When they say "use a stronger prior," they mean increase $\alpha_0$ in the Dirichlet. When they say "the KL term is too large," they mean the approximate Gaussian posterior is far from the standard normal prior. Every one of these statements refers to a specific distribution from this section.

[← Back to Probability Theory](../README.md) | [Next: Joint Distributions →](../03-Joint-Distributions/notes.md)

---

## Appendix A: Detailed Distribution Reference Cards

### A.1 Bernoulli Distribution — Full Reference

```
BERNOULLI(p) REFERENCE CARD
════════════════════════════════════════════════════════════════════════

  PMF:     P(X=x) = p^x (1-p)^(1-x),   x ∈ {0, 1}

  CDF:     F(x) = 0          x < 0
                  1 - p      0 ≤ x < 1
                  1          x ≥ 1

  Mean:    p
  Var:     p(1-p)              [max at p=0.5]
  Mode:    1{p > 0.5}
  Entropy: -p log p - (1-p) log(1-p)

  MGF:     M(t) = 1 - p + pe^t

  Natural param:  η = log(p/(1-p))  [logit]
  Inverse link:   p = σ(η) = 1/(1+e^{-η})  [sigmoid]

  ML role:  Binary labels, dropout, stochastic depth, RLHF preferences

════════════════════════════════════════════════════════════════════════
```

### A.2 Gaussian Distribution — Full Reference

```
GAUSSIAN N(μ, σ²) REFERENCE CARD
════════════════════════════════════════════════════════════════════════

  PDF:    f(x) = (1/σ√(2π)) exp(-(x-μ)²/2σ²)
  CDF:    F(x) = Φ((x-μ)/σ)   where Φ = standard normal CDF

  Mean:       μ
  Variance:   σ²
  Mode:       μ (unique)
  Median:     μ (by symmetry)
  Entropy:    ½ log(2πeσ²)   [maximum for fixed mean/var]

  MGF:    M(t) = exp(μt + σ²t²/2)
  CGF:    K(t) = μt + σ²t²/2   [cumulants: κ₁=μ, κ₂=σ², κ_k=0 for k≥3]

  Standard:   Z = (X - μ)/σ ~ N(0, 1)

  Key quantiles (standard normal):
    Φ(1.282) = 0.90,  Φ(1.645) = 0.95,  Φ(1.960) = 0.975
    Φ(2.326) = 0.99,  Φ(2.576) = 0.995, Φ(3.090) = 0.999

  68-95-99.7: P(|Z| ≤ 1) ≈ 0.683, P(|Z| ≤ 2) ≈ 0.954, P(|Z| ≤ 3) ≈ 0.997

  Natural params: η₁ = μ/σ², η₂ = -1/(2σ²)
  Suff stats:    T(x) = (x, x²)

  ML role:  Weight init, VAE prior/posterior, GP, diffusion noise, batch norm

════════════════════════════════════════════════════════════════════════
```

### A.3 Beta Distribution — Full Reference

```
BETA(α, β) REFERENCE CARD
════════════════════════════════════════════════════════════════════════

  PDF:    f(x) = x^(α-1)(1-x)^(β-1) / B(α,β),   x ∈ (0, 1)
          B(α,β) = Γ(α)Γ(β)/Γ(α+β)

  Mean:   α/(α+β)
  Var:    αβ / [(α+β)²(α+β+1)]
  Mode:   (α-1)/(α+β-2)   for α,β > 1

  Shape patterns:
    α<1, β<1        → U-shaped (bimodal at 0 and 1)
    α=β=1           → Uniform(0,1)
    α=β>1           → Symmetric bell
    α>β             → Skewed toward 1
    α<β             → Skewed toward 0
    α,β → ∞, ratio  → Concentrates at mode

  Conjugate prior for: Bernoulli, Binomial
  Posterior update:    Beta(α, β) + k successes, n-k failures
                       → Beta(α+k, β+n-k)

  ML role:  CTR estimation, RLHF preference priors, Beta-VAE

════════════════════════════════════════════════════════════════════════
```

---

## Appendix B: The Gamma Function

The **gamma function** $\Gamma(z)$ generalises the factorial to non-integer arguments and appears in the normalising constants of Beta, Dirichlet, Gamma, and Student-$t$ distributions.

### B.1 Definition and Key Properties

For $z > 0$:
$$\Gamma(z) = \int_0^\infty t^{z-1} e^{-t}\,dt$$

**Recurrence relation:** $\Gamma(z+1) = z\Gamma(z)$

*Proof:* Integration by parts with $u = t^z$ and $dv = e^{-t}dt$:
$$\Gamma(z+1) = \left[-t^z e^{-t}\right]_0^\infty + z\int_0^\infty t^{z-1}e^{-t}\,dt = 0 + z\Gamma(z)$$

**Factorial connection:** $\Gamma(n) = (n-1)!$ for positive integers $n$.

**Half-integer values:** $\Gamma(1/2) = \sqrt{\pi}$, $\Gamma(3/2) = \sqrt{\pi}/2$, $\Gamma(5/2) = 3\sqrt{\pi}/4$.

### B.2 The Beta Function

The **Beta function** $B(\alpha, \beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$ satisfies:
$$B(\alpha, \beta) = \int_0^1 t^{\alpha-1}(1-t)^{\beta-1}\,dt$$

This integral is precisely the normalising constant of the Beta$(\alpha, \beta)$ PDF.

**Symmetry:** $B(\alpha, \beta) = B(\beta, \alpha)$.

### B.3 Stirling's Approximation

For large $n$: $n! \approx \sqrt{2\pi n} (n/e)^n$.

In terms of Gamma: $\Gamma(n+1) \approx \sqrt{2\pi n} (n/e)^n$ for large $n$.

This is used to analyse the Poisson distribution at large $k$ and the convergence of the Binomial to the Poisson.

---

## Appendix C: Sampling from Distributions

### C.1 Inverse CDF Method

For any distribution with continuous, invertible CDF $F$:
1. Draw $U \sim \mathcal{U}(0,1)$
2. Return $X = F^{-1}(U)$

Then $X \sim F$.

**Explicit inverse CDFs:**

| Distribution | Inverse CDF $F^{-1}(u)$ |
|---|---|
| Exponential$(\lambda)$ | $-\log(1-u)/\lambda$ |
| Geometric$(p)$ | $\lceil \log(1-u)/\log(1-p) \rceil$ |
| Cauchy$(\mu, \sigma)$ | $\mu + \sigma\tan(\pi(u-1/2))$ |

Gaussian has no closed-form inverse CDF; the Box-Muller transform is used instead:
$$Z_1 = \sqrt{-2\log U_1}\cos(2\pi U_2), \quad Z_2 = \sqrt{-2\log U_1}\sin(2\pi U_2)$$

where $U_1, U_2 \sim \mathcal{U}(0,1)$ i.i.d. gives $Z_1, Z_2 \sim \mathcal{N}(0,1)$ i.i.d.

### C.2 Sampling the Dirichlet

To sample $\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$:
1. Draw $Y_k \sim \Gamma(\alpha_k, 1)$ independently for $k = 1, \ldots, K$
2. Return $p_k = Y_k / \sum_{j=1}^K Y_j$

This works because the normalised Gamma variables follow a Dirichlet distribution.

### C.3 The Gumbel-Max Trick for Categorical

To sample from Categorical$(\mathbf{p})$:
1. Compute logits $\mathbf{z} = \log\mathbf{p}$ (or use raw logits)
2. Draw $G_k \sim \text{Gumbel}(0,1) = -\log(-\log U_k)$ for $U_k \sim \mathcal{U}(0,1)$
3. Return $X = \arg\max_k (z_k + G_k)$

This gives exact categorical samples and enables the **Gumbel-softmax** differentiable approximation by replacing argmax with softmax at low temperature.

---

## Appendix D: Heavy-Tailed Distributions

### D.1 What Makes a Tail Heavy?

The Gaussian tail decays as $e^{-x^2/2}$ — **super-exponentially fast**. Distributions with tails decaying slower than any exponential are called **heavy-tailed**.

**Pareto distribution (power law):** $P(X > x) = (x_m/x)^\alpha$ for $x > x_m$. Tail decays as $x^{-\alpha}$.
- $\alpha > 2$: finite variance
- $1 < \alpha \leq 2$: finite mean, infinite variance
- $0 < \alpha \leq 1$: infinite mean

**Student-$t_\nu$:** $P(X > x) \sim x^{-\nu}$ for large $x$ — polynomial tail with exponent $\nu$.

**Cauchy ($t_1$):** Variance infinite, mean undefined. The average of $n$ i.i.d. Cauchy variables is still Cauchy — the CLT fails completely.

### D.2 Heavy Tails in ML

**Gradient noise:** Empirical studies show that SGD gradient noise often has heavier tails than Gaussian. This may explain generalization: heavy-tailed noise explores the loss landscape more broadly, escaping sharp minima.

**Neural network weight distributions:** Trained neural network weights often follow power-law distributions (Martin & Mahoney 2019, 2021). "Heavy-tailed self-regularization" is proposed as a mechanism for implicit regularization.

**Attention scores:** Without temperature scaling, softmax attention can produce near-degenerate distributions (mass concentrated on one token), which is the tail behavior of a peaked Categorical.

---

## Appendix E: Worked Derivations

### E.1 Poisson MGF Derivation

$$M_X(t) = \sum_{k=0}^\infty e^{tk} \cdot \frac{\lambda^k e^{-\lambda}}{k!} = e^{-\lambda} \sum_{k=0}^\infty \frac{(\lambda e^t)^k}{k!} = e^{-\lambda} \cdot e^{\lambda e^t} = e^{\lambda(e^t-1)}$$

Checking moments: $M'(t) = \lambda e^t M(t)$. At $t=0$: $M'(0) = \lambda \cdot 1 = \lambda = \mathbb{E}[X]$. ✓

$M''(t) = \lambda e^t M(t) + (\lambda e^t)^2 M(t)$. At $t=0$: $M''(0) = \lambda + \lambda^2 = \mathbb{E}[X^2]$.

$\operatorname{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 = (\lambda + \lambda^2) - \lambda^2 = \lambda$. ✓

### E.2 Gaussian MGF Derivation (Complete)

$$M_X(t) = \int_{-\infty}^\infty e^{tx} \cdot \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-\mu)^2/(2\sigma^2)} dx$$

Combine exponents: $tx - \frac{(x-\mu)^2}{2\sigma^2} = -\frac{1}{2\sigma^2}\left[(x-\mu)^2 - 2\sigma^2 tx\right]$

$= -\frac{1}{2\sigma^2}\left[x^2 - 2x(\mu + \sigma^2 t) + \mu^2\right]$

$= -\frac{(x - (\mu+\sigma^2 t))^2}{2\sigma^2} + \mu t + \frac{\sigma^2 t^2}{2}$

Substituting back:
$$M_X(t) = e^{\mu t + \sigma^2 t^2/2} \int_{-\infty}^\infty \frac{1}{\sigma\sqrt{2\pi}} e^{-(x-(\mu+\sigma^2 t))^2/(2\sigma^2)} dx = e^{\mu t + \sigma^2 t^2/2}$$

since the integral is 1 (it's a Gaussian PDF with mean $\mu+\sigma^2 t$). $\square$

### E.3 Beta-Bernoulli Posterior Derivation

Prior: $p(p) = \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}$. Likelihood: $L(p \mid k, n) = \binom{n}{k}p^k(1-p)^{n-k}$.

Posterior:
$$p(p \mid k, n) \propto p^{\alpha-1}(1-p)^{\beta-1} \cdot p^k(1-p)^{n-k} = p^{(\alpha+k)-1}(1-p)^{(\beta+n-k)-1}$$

This is the unnormalised Beta$(\alpha+k, \beta+n-k)$ PDF, so the posterior is:
$$p \mid (k, n) \sim \text{Beta}(\alpha+k, \beta+n-k) \quad \square$$

### E.4 Gamma-Poisson Posterior Derivation

Prior: $\lambda \sim \Gamma(\alpha, \beta)$, so $p(\lambda) \propto \lambda^{\alpha-1}e^{-\beta\lambda}$. Likelihood for $n$ observations with sum $S = \sum x_i$:
$$p(\mathbf{x} \mid \lambda) = \prod_{i=1}^n \frac{\lambda^{x_i}e^{-\lambda}}{x_i!} \propto \lambda^S e^{-n\lambda}$$

Posterior:
$$p(\lambda \mid \mathbf{x}) \propto \lambda^{\alpha-1}e^{-\beta\lambda} \cdot \lambda^S e^{-n\lambda} = \lambda^{(\alpha+S)-1}e^{-(\beta+n)\lambda}$$

This is $\Gamma(\alpha+S, \beta+n)$. $\square$

---

## Appendix F: Exponential Family — Additional Members

### F.1 Negative Binomial as Exponential Family

The Negative Binomial$(r, p)$ PMF for $k = 0, 1, 2, \ldots$:
$$P(X=k) = \binom{k+r-1}{k}(1-p)^r p^k$$

Writing $\eta = \log p$:
$$P(X=k) = \binom{k+r-1}{k}(1-p)^r \exp(\eta k)$$

Natural param: $\eta = \log p \in (-\infty, 0)$, sufficient stat: $T(x) = x$, log-partition: $A(\eta) = -r\log(1-e^\eta)$.

**Note:** The parameter $r$ (number of failures until stop) must be known; the family is parameterised only by $p$ when $r$ is fixed.

### F.2 von Mises Distribution (Circular Data)

For directional/angular data (e.g., wind direction, protein torsion angles), the von Mises distribution plays the role that the Gaussian plays for linear data:
$$f(\theta; \mu, \kappa) = \frac{e^{\kappa\cos(\theta-\mu)}}{2\pi I_0(\kappa)}, \quad \theta \in [0, 2\pi)$$

where $I_0$ is the modified Bessel function. It belongs to the exponential family with $\eta = \kappa e^{i\mu}$ (complex natural parameter).

**For AI:** Rotary Position Embedding (RoPE) in transformers embeds token positions as rotations in 2D planes. The von Mises distribution is the natural distribution for such circular position encodings.

---

## Appendix G: Numerical Stability in Practice

### G.1 Log-Sum-Exp

The log-partition function $A(\boldsymbol{\eta}) = \log\sum_k e^{\eta_k}$ is numerically unstable for large or small $\eta_k$ due to floating-point overflow/underflow.

**Stable computation:**
$$\log\sum_k e^{\eta_k} = c + \log\sum_k e^{\eta_k - c}, \quad c = \max_k \eta_k$$

Since $e^{\eta_k - c} \in (0, 1]$ for all $k$, no overflow occurs.

### G.2 Log-Space Probability Computations

For small probabilities, work in log space throughout:
- Multiply probabilities: add log-probabilities
- Normalise: subtract log-sum-exp
- Compute Bernoulli NLL: `-y * log_p - (1-y) * log(1-p)` should use `log_p = log_sigmoid(logit)` and `log(1-p) = log_sigmoid(-logit)` for numerical stability

### G.3 The Gamma Function at Large Arguments

$\log\Gamma(z)$ can be computed stably using the Lanczos approximation. For $z \gg 1$, use Stirling's log-approximation: $\log\Gamma(z+1) \approx z\log z - z + \frac{1}{2}\log(2\pi z)$.

SciPy provides `scipy.special.gammaln` for numerically stable $\log\Gamma$.

---

## Appendix H: Distribution Identification Checklist

When fitting a distribution to data, use this diagnostic checklist:

| Check | Tool | Implication |
|---|---|---|
| Are values integers ≥ 0? | — | Consider discrete (Poisson, NB, Geometric) |
| Are values in (0,1)? | — | Beta or transformed Gaussian |
| Are values on a simplex? | — | Dirichlet |
| Is variance ≈ mean? | Dispersion test | Poisson (if yes), NB (if var > mean) |
| Is distribution symmetric? | Skewness test | Gaussian, $t$, symmetric Beta |
| Are tails heavier than Gaussian? | Kurtosis, QQ-plot | Student-$t$, Laplace |
| Does a QQ-plot follow the diagonal? | `scipy.stats.probplot` | Good fit to reference distribution |
| Histogram bell-shaped? | `plt.hist` | Gaussian or Beta with $\alpha=\beta>1$ |
| Histogram exponentially decaying? | Log-scale plot | Exponential or Geometric |
| Do log-log tails follow a line? | Log-log tail plot | Power law (Pareto) |

**Sample size requirements:**

| Distribution | Minimum samples for good MLE |
|---|---|
| Bernoulli | ~30 (for $p$ far from 0/1) |
| Gaussian | ~30 (CLT kicks in) |
| Poisson | ~50 (for $\lambda < 1$); 10 for large $\lambda$ |
| Beta | ~100 (4 moment equations for $\alpha, \beta$) |
| Dirichlet | ~$100K$ (scales with $K$) |
| Student-$t$ | ~50 (for reliable $\nu$ estimate) |


---

## Appendix I: Distribution Parameter Estimation (MLE)

### I.1 Maximum Likelihood Estimation Overview

The **MLE** $\hat{\boldsymbol{\theta}}$ maximises the likelihood $L(\boldsymbol{\theta}) = \prod_{i=1}^n p(x^{(i)}; \boldsymbol{\theta})$, equivalently maximises the log-likelihood $\ell(\boldsymbol{\theta}) = \sum_i \log p(x^{(i)}; \boldsymbol{\theta})$.

For exponential family members, the MLE satisfies the moment-matching condition:
$$\nabla A(\hat{\boldsymbol{\eta}}) = \frac{1}{n}\sum_{i=1}^n T(x^{(i)})$$

The expected sufficient statistics under the model equal the empirical sufficient statistics.

### I.2 MLEs of Key Distributions

**Bernoulli$(p)$:** $\hat{p} = \frac{1}{n}\sum_i x_i = \bar{x}$ (sample proportion)

**Gaussian$(\mu, \sigma^2)$:**
$$\hat{\mu} = \bar{x} = \frac{1}{n}\sum_i x_i, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2$$

Note: $\hat{\sigma}^2_\text{MLE} = \frac{n-1}{n} S^2$ is biased (divides by $n$ not $n-1$). The unbiased estimator uses the **sample variance** $S^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$.

**Poisson$(\lambda)$:** $\hat{\lambda} = \bar{x}$ (sample mean)

**Exponential$(\lambda)$:** $\hat{\lambda} = 1/\bar{x}$ (reciprocal of sample mean)

**Categorical$(\mathbf{p})$:** $\hat{p}_k = c_k/n$ where $c_k = \sum_i \mathbf{1}[x^{(i)}=k]$ (empirical frequencies)

**Beta$(\alpha, \beta)$:** No closed form. Method of moments gives starting values:
$$\tilde{\mu} = \bar{x}, \quad \tilde{s}^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$$
$$\hat{\alpha} = \tilde{\mu}\!\left(\frac{\tilde{\mu}(1-\tilde{\mu})}{\tilde{s}^2} - 1\right), \quad \hat{\beta} = (1-\tilde{\mu})\!\left(\frac{\tilde{\mu}(1-\tilde{\mu})}{\tilde{s}^2} - 1\right)$$

Refine with Newton-Raphson using the digamma function $\psi = \Gamma'/\Gamma$.

### I.3 Bayesian Estimation vs. MLE

Under conjugate priors, the **posterior mean** often provides a better estimator than MLE, especially for small samples:

| Distribution | MLE | Posterior Mean (conjugate prior) |
|---|---|---|
| Bernoulli | $k/n$ | $(\alpha+k)/(\alpha+\beta+n)$ |
| Categorical | $c_k/n$ | $(\alpha_k+c_k)/(\alpha_0+n)$ |
| Poisson | $\bar{x}$ | $(\alpha+\sum x_i)/(\beta+n)$ |

The posterior mean shrinks the MLE toward the prior mean — a form of regularisation that prevents overfitting to small samples.

---

## Appendix J: Relationships Between Distributions — Extended

### J.1 The Exponential Family as a Unifying Framework

```
ALL MEMBERS OF THE EXPONENTIAL FAMILY
════════════════════════════════════════════════════════════════════════

  One-parameter families:
    Bernoulli(p)     — natural param: log(p/(1-p))
    Poisson(λ)       — natural param: log(λ)
    Exponential(λ)   — natural param: -λ
    Geometric(p)     — natural param: log(1-p)

  Two-parameter families:
    Gaussian(μ,σ²)   — natural params: (μ/σ², -1/2σ²)
    Gamma(α,β)       — natural params: (α-1, -β)
    Beta(α,β)        — natural params: (α-1, β-1)
    NegBin(r,p)      — natural param: log(p)  [r fixed]

  K-parameter families:
    Categorical(p)   — natural params: (log p_k/p_K)_{k<K}
    Multinomial(n,p) — same as Categorical  [n fixed]
    Dirichlet(α)     — natural params: (α_k - 1)_{k=1}^K

  NOT exponential family:
    Student-t(ν)     — tail behavior is not exponential
    Cauchy            — same
    Pareto            — support depends on parameter

════════════════════════════════════════════════════════════════════════
```

### J.2 Scale Mixtures of Gaussians

Several heavy-tailed distributions arise as **variance mixtures** of Gaussians: $X \mid V \sim \mathcal{N}(0, V)$ with random variance $V$.

| Mixing Distribution for $V$ | Marginal Distribution of $X$ |
|---|---|
| $V = \sigma^2$ (constant) | $\mathcal{N}(0, \sigma^2)$ |
| $V \sim \text{InvGamma}(\nu/2, \nu/2)$ | Student-$t_\nu$ |
| $V \sim \text{Exponential}(\lambda^2/2)$ | Laplace$(0, 1/\lambda)$ |
| $V \sim \text{Levy}(\mu, c)$ | $\alpha$-stable distribution |

**For AI:** The variance mixture representation of Student-$t$ enables efficient Gibbs sampling — alternating between sampling $X \mid V \sim \mathcal{N}$ and $V \mid X \sim \text{InvGamma}$. This technique underlies many robust Bayesian regression algorithms.

### J.3 Normalising Flows: Transforming Distributions

Any differentiable, invertible function $g$ transforms a distribution: if $X \sim p_X$ and $Y = g(X)$, then:
$$p_Y(y) = p_X(g^{-1}(y)) \cdot \left|\frac{d}{dy}g^{-1}(y)\right|$$

**Examples of distribution transformation chains:**

- Start with $U \sim \mathcal{U}(0,1)$, apply $g(u) = -\log(u)/\lambda$: get Exponential$(\lambda)$
- Start with $Z \sim \mathcal{N}(0,1)$, apply $g(z) = e^{\mu + \sigma z}$: get Log-Normal$(\mu,\sigma^2)$
- Start with $Z \sim \mathcal{N}(0,1)$, apply CDF and then categorical rounding: get Gaussian copula
- Compose $K$ invertible transforms: get a **normalising flow** with complex target distribution

> → _Full treatment: [§03 Joint Distributions](../03-Joint-Distributions/notes.md) (change-of-variables formula)_

---

## Appendix K: Quick-Reference Probability Tables

### K.1 Standard Normal Tail Probabilities

| $z$ | $P(Z > z)$ | $P(|Z| > z)$ |
|---|---|---|
| 1.00 | 0.1587 | 0.3174 |
| 1.28 | 0.1003 | 0.2005 |
| 1.64 | 0.0505 | 0.1010 |
| 1.96 | 0.0250 | 0.0500 |
| 2.00 | 0.0228 | 0.0455 |
| 2.33 | 0.0099 | 0.0197 |
| 2.58 | 0.0049 | 0.0099 |
| 3.00 | 0.0013 | 0.0027 |
| 3.29 | 0.0005 | 0.0010 |
| 3.89 | 0.0001 | 0.0002 |

### K.2 Poisson Probabilities $P(X = k)$ for $\lambda \in \{1, 2, 5\}$

| $k$ | $\lambda=1$ | $\lambda=2$ | $\lambda=5$ |
|---|---|---|---|
| 0 | 0.3679 | 0.1353 | 0.0067 |
| 1 | 0.3679 | 0.2707 | 0.0337 |
| 2 | 0.1839 | 0.2707 | 0.0842 |
| 3 | 0.0613 | 0.1804 | 0.1404 |
| 4 | 0.0153 | 0.0902 | 0.1755 |
| 5 | 0.0031 | 0.0361 | 0.1755 |
| 6 | 0.0005 | 0.0120 | 0.1462 |
| 7 | 0.0001 | 0.0034 | 0.1044 |

### K.3 Beta Distribution Moments and Shapes

| $(\alpha, \beta)$ | Mean | Std Dev | Shape Description |
|---|---|---|---|
| (1, 1) | 0.500 | 0.289 | Uniform |
| (0.5, 0.5) | 0.500 | 0.354 | U-shaped (arcsine) |
| (2, 2) | 0.500 | 0.224 | Symmetric bell |
| (2, 5) | 0.286 | 0.159 | Right-skewed |
| (5, 2) | 0.714 | 0.159 | Left-skewed |
| (10, 10) | 0.500 | 0.106 | Narrow symmetric bell |
| (1, 3) | 0.250 | 0.194 | Decreasing |
| (3, 1) | 0.750 | 0.194 | Increasing |

### K.4 Binomial Cumulative Probabilities

$P(X \leq k)$ for $\text{Binomial}(n=20, p)$:

| $k$ | $p=0.2$ | $p=0.5$ | $p=0.8$ |
|---|---|---|---|
| 2 | 0.2061 | 0.0002 | 0.0000 |
| 5 | 0.8042 | 0.0207 | 0.0000 |
| 8 | 0.9900 | 0.2517 | 0.0001 |
| 10 | 0.9994 | 0.5881 | 0.0026 |
| 12 | 1.0000 | 0.8684 | 0.0321 |
| 15 | 1.0000 | 0.9941 | 0.3704 |
| 18 | 1.0000 | 1.0000 | 0.9308 |


---

## Appendix L: Information-Theoretic Properties of Distributions

### L.1 Maximum Entropy Distributions

The **principle of maximum entropy** states: among all distributions satisfying given constraints, choose the one with maximum entropy. This gives the "least informative" distribution consistent with the known facts.

| Constraint | Maximum Entropy Distribution |
|---|---|
| Support $\{0, 1, \ldots, N-1\}$, no other info | Discrete Uniform |
| Support $(0, \infty)$, fixed mean $\mu$ | Exponential$(1/\mu)$ |
| Support $\mathbb{R}$, fixed mean $\mu$ and variance $\sigma^2$ | Gaussian$(\mu, \sigma^2)$ |
| Support $[a, b]$, no other info | Uniform$(a, b)$ |
| Support $\{0,1\}$, fixed mean $p$ | Bernoulli$(p)$ |
| Support $\Delta^{K-1}$, fixed mean $\boldsymbol{\mu}$ | Dirichlet (with matching moments) |

**Proof sketch for Gaussian:** Maximise $H[p] = -\int p(x)\log p(x)\,dx$ subject to $\int p(x)\,dx = 1$, $\int x\,p(x)\,dx = \mu$, and $\int (x-\mu)^2 p(x)\,dx = \sigma^2$. Using Lagrange multipliers (§05/04 of Chapter 5), the optimal density satisfies $\log p(x) = \lambda_0 + \lambda_1 x + \lambda_2(x-\mu)^2$, which is the Gaussian form.

### L.2 KL Divergences Between Common Distributions

**Two Gaussians:**
$$D_{\mathrm{KL}}(\mathcal{N}(\mu_1,\sigma_1^2) \| \mathcal{N}(\mu_2,\sigma_2^2)) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2+(\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$

**Gaussian from standard normal:**
$$D_{\mathrm{KL}}(\mathcal{N}(\mu,\sigma^2) \| \mathcal{N}(0,1)) = \frac{1}{2}(\sigma^2 + \mu^2 - 1 - \log\sigma^2)$$

**Two Bernoullis:**
$$D_{\mathrm{KL}}(\operatorname{Bern}(p) \| \operatorname{Bern}(q)) = p\log\frac{p}{q} + (1-p)\log\frac{1-p}{1-q}$$

**Two Categoricals:**
$$D_{\mathrm{KL}}(\operatorname{Cat}(\mathbf{p}) \| \operatorname{Cat}(\mathbf{q})) = \sum_k p_k \log\frac{p_k}{q_k}$$

**Two Poissons:**
$$D_{\mathrm{KL}}(\operatorname{Pois}(\lambda_1) \| \operatorname{Pois}(\lambda_2)) = \lambda_1\log\frac{\lambda_1}{\lambda_2} - (\lambda_1 - \lambda_2)$$

**Two Betas:**
$$D_{\mathrm{KL}}(\text{Beta}(\alpha_1,\beta_1) \| \text{Beta}(\alpha_2,\beta_2)) = \log\frac{B(\alpha_2,\beta_2)}{B(\alpha_1,\beta_1)} + (\alpha_1-\alpha_2)\psi(\alpha_1) + (\beta_1-\beta_2)\psi(\beta_1) + (\alpha_2-\alpha_1+\beta_2-\beta_1)\psi(\alpha_1+\beta_1)$$

where $\psi = \Gamma'/\Gamma$ is the digamma function.

### L.3 Entropy Ordering

For distributions with the same support and mean, entropy orders distributions by "spread":

$$H(\text{Bernoulli}(0.5)) = \log 2 \approx 0.693 \text{ nats}$$
$$H(\mathcal{N}(\mu, \sigma^2)) = \frac{1}{2}\log(2\pi e \sigma^2)$$
$$H(\text{Uniform}(a,b)) = \log(b-a)$$

For continuous distributions on $\mathbb{R}$ with fixed variance $\sigma^2$:
$$H(\text{Gaussian}) \geq H(\text{Laplace}) \geq H(\text{Logistic}) \geq \cdots$$

The Gaussian maximises entropy, making it the distribution that "knows least" given the mean and variance constraints — which is exactly why it appears in CLT results and maximum entropy arguments.

---

## Appendix M: PyTorch and SciPy API Reference

### M.1 SciPy Distributions

```python
from scipy import stats

# Discrete distributions
stats.bernoulli(p=0.3)         # Bernoulli(p)
stats.binom(n=10, p=0.3)       # Binomial(n, p)
stats.geom(p=0.3)              # Geometric(p), k=1,2,...
stats.nbinom(n=5, p=0.3)       # NegBinomial(n, p)
stats.poisson(mu=2.5)          # Poisson(lambda)

# Continuous distributions
stats.uniform(loc=0, scale=1)  # Uniform(0,1)   [loc=a, scale=b-a]
stats.norm(loc=0, scale=1)     # Gaussian(mu, sigma)  [scale=sigma, not sigma^2!]
stats.expon(scale=1/lambda_)   # Exponential(lambda)  [scale=1/rate]
stats.gamma(a=2, scale=1/beta) # Gamma(alpha, beta)   [a=shape, scale=1/rate]
stats.beta(a=2, b=5)           # Beta(alpha, beta)
stats.t(df=5)                  # Student-t(nu)

# Common methods (all distributions)
rv.pmf(k) / rv.pdf(x)  # PMF or PDF
rv.cdf(x)              # CDF
rv.ppf(q)              # Quantile (inverse CDF)
rv.sf(x)               # Survival function 1-CDF
rv.rvs(size=100)        # Random samples
rv.mean()              # E[X]
rv.var()               # Var(X)
rv.std()               # std dev
rv.entropy()           # H(X) in nats
```

### M.2 PyTorch `torch.distributions`

```python
import torch
from torch.distributions import (
    Bernoulli, Binomial, Geometric, Poisson,
    Categorical, Multinomial,
    Uniform, Normal, Exponential, Gamma, Beta, Dirichlet, StudentT
)

# Create distributions
d = Normal(loc=torch.tensor(0.0), scale=torch.tensor(1.0))

# Common methods
d.sample((5,))          # draw 5 samples
d.log_prob(x)           # log p(x) — used in loss functions
d.cdf(x)               # F(x)
d.icdf(q)              # F^{-1}(q)
d.entropy()            # H(X)
d.mean                 # E[X] (property, not method)
d.variance             # Var(X)
d.stddev               # std dev

# Reparameterised sampling (enables gradients through sampling)
d.rsample((5,))         # only for reparameterisable distributions
                        # (Normal, Gamma, Beta, Dirichlet)

# Special distributions
Categorical(logits=z)   # Categorical from logits (uses log_softmax internally)
Dirichlet(concentration=alpha)  # Dirichlet(alpha)
```

### M.3 NumPy Random Number Generation

```python
import numpy as np
rng = np.random.default_rng(seed=42)

# Discrete
rng.integers(0, 2, size=100)              # Uniform integers (Bernoulli-like)
rng.binomial(n=10, p=0.3, size=100)       # Binomial
rng.geometric(p=0.3, size=100)            # Geometric (p=success prob)
rng.poisson(lam=2.5, size=100)            # Poisson

# Continuous
rng.uniform(low=0, high=1, size=100)      # Uniform
rng.normal(loc=0, scale=1, size=100)      # Gaussian  [scale=sigma]
rng.exponential(scale=1/lambda_, size=100) # Exponential [scale=1/rate]
rng.gamma(shape=2, scale=1/beta, size=100) # Gamma
rng.beta(a=2, b=5, size=100)              # Beta

# Categorical/Dirichlet
rng.choice(K, p=probs, size=100)          # Categorical
rng.dirichlet(alpha, size=10)             # Dirichlet
rng.multinomial(n=20, pvals=probs)        # Multinomial
```


---

## Appendix N: Distribution Parameter Estimation

Quick reference for maximum likelihood estimates (MLE) and method of moments (MOM) estimators.

### Discrete Distributions

| Distribution | MLE Estimator | Method of Moments |
|---|---|---|
| Bernoulli($p$) | $\hat{p} = \bar{x}$ | Same as MLE |
| Binomial($n$,$p$) | $\hat{p} = \bar{x}/n$ | Same as MLE |
| Geometric($p$) | $\hat{p} = 1/\bar{x}$ | Same as MLE |
| Poisson($\lambda$) | $\hat{\lambda} = \bar{x}$ | Same as MLE |
| Negative Binomial($r$,$p$) | $\hat{p} = r/(r+\bar{x})$ | Match mean and variance |

### Continuous Distributions

| Distribution | MLE Estimator(s) | Notes |
|---|---|---|
| Uniform($a$,$b$) | $\hat{a} = x_{(1)}$, $\hat{b} = x_{(n)}$ | Biased; adjust for small $n$ |
| Gaussian($\mu$,$\sigma^2$) | $\hat{\mu} = \bar{x}$, $\hat{\sigma}^2 = \frac{1}{n}\sum(x_i-\bar{x})^2$ | Biased variance; unbiased uses $n-1$ |
| Exponential($\lambda$) | $\hat{\lambda} = 1/\bar{x}$ | Same as MLE |
| Gamma($\alpha$,$\beta$) | Solve $\psi(\hat{\alpha}) - \log\hat{\alpha} = \log\bar{x} - \overline{\log x}$ | $\psi$ is digamma; numerical solution needed |
| Beta($\alpha$,$\beta$) | Method of moments: $\hat{\alpha} = \bar{x}\left(\frac{\bar{x}(1-\bar{x})}{s^2}-1\right)$ | $s^2$ = sample variance |

### Bayesian Estimation with Conjugate Priors

For conjugate models, the posterior mean provides a natural estimator:

**Bernoulli with Beta prior:**
$$\hat{p}_{\text{Bayes}} = \frac{\alpha + \sum x_i}{\alpha + \beta + n}$$

This shrinks the MLE $\bar{x}$ toward the prior mean $\alpha/(\alpha+\beta)$.

**Poisson with Gamma prior:**
$$\hat{\lambda}_{\text{Bayes}} = \frac{\alpha + \sum x_i}{\beta + n}$$

**Gaussian with Gaussian prior** (known $\sigma^2$):
$$\hat{\mu}_{\text{Bayes}} = \frac{\sigma^2 \mu_0 / \sigma_0^2 + n\bar{x}}{n + \sigma^2/\sigma_0^2}$$

As $n \to \infty$, all Bayesian estimators converge to MLE — the data overwhelms the prior.

---

## Appendix O: Tail Bounds for Common Distributions

Understanding tail behaviour is essential for concentration inequalities and PAC-learning bounds.

### Gaussian Tails

The Mills ratio gives an upper bound:
$$P(X > t) \leq \frac{\phi(t)}{t} = \frac{1}{t\sqrt{2\pi}} e^{-t^2/2}, \quad t > 0$$

More precisely: $P(X > t) \sim \frac{\phi(t)}{t}$ as $t \to \infty$.

Numerically: $P(|X| > 2) \approx 0.046$, $P(|X| > 3) \approx 0.003$, $P(|X| > 6) \approx 2 \times 10^{-9}$.

### Sub-Gaussian Distributions

A random variable $X$ is **sub-Gaussian** with parameter $\sigma$ if:
$$\mathbb{E}[e^{tX}] \leq e^{\sigma^2 t^2/2} \quad \forall t$$

This implies Gaussian-like tail decay: $P(|X - \mathbb{E}[X]| \geq \epsilon) \leq 2e^{-\epsilon^2/(2\sigma^2)}$.

Examples: Gaussian($\mu, \sigma^2$) is sub-Gaussian($\sigma$); Bounded$[a,b]$ is sub-Gaussian$((b-a)/2)$.

**LLM relevance:** Token embeddings are often assumed sub-Gaussian for theoretical analysis of attention mechanisms and generalisation bounds.

### Poisson Tails (Chernoff bounds)

For $X \sim \text{Poisson}(\lambda)$:
$$P(X \geq (1+\delta)\lambda) \leq \left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\lambda \leq e^{-\lambda\delta^2/3}, \quad 0 < \delta \leq 1$$

### Exponential Tails (Memoryless)

For $X \sim \text{Exp}(\lambda)$: $P(X > t) = e^{-\lambda t}$ exactly (no approximation needed).

> **Forward reference:** Concentration inequalities (Markov, Chebyshev, Hoeffding, Bernstein) are developed fully in [§6.5 Concentration Inequalities](../05-Concentration-Inequalities/notes.md).

---

## Appendix P: Worked Example — Distribution Selection in Practice

**Problem:** A recommendation system logs how many times each user clicks on recommended items in a session. You observe counts $x_1, \ldots, x_n$ and want to model the distribution.

**Step 1: Check support.** Counts take values $\{0, 1, 2, \ldots\}$ → discrete, non-negative integer. Eliminates all continuous distributions.

**Step 2: Check bounds.** No natural upper bound → Geometric or Poisson. (If sessions had fixed length $N$, Binomial would apply.)

**Step 3: Check variance-mean relationship.**
- Poisson: $\text{Var} = \mu$
- Geometric: $\text{Var} = (1-p)/p^2 > 1/p = \mu$ (always over-dispersed)
- Negative Binomial: $\text{Var} = \mu + \mu^2/r$ (over-dispersed, extra parameter)

Compute sample mean $\hat{\mu}$ and sample variance $\hat{s}^2$. If $\hat{s}^2 \approx \hat{\mu}$, use Poisson. If $\hat{s}^2 \gg \hat{\mu}$, use Negative Binomial.

**Step 4: Fit and check.** Compute MLE, then use a chi-squared goodness-of-fit test or probability plot.

**Step 5: Consider mixture models.** If many zeros occur (zero-inflated data), consider Zero-Inflated Poisson: $P(X=0) = \pi + (1-\pi)e^{-\lambda}$, $P(X=k) = (1-\pi)\text{Pois}(k;\lambda)$ for $k \geq 1$.

**Key insight:** The choice of distribution encodes assumptions about the data-generating process. Poisson assumes events occur at a constant rate independently; Negative Binomial allows the rate itself to vary (it is a Poisson-Gamma mixture). In LLM contexts, token frequency distributions are often heavy-tailed (Zipfian), motivating log-normal or power-law models rather than Poisson.

---

_Last updated: 2026 — covers all distributions in §6.2 scope as defined in the Chapter README._
