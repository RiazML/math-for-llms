[← Back to Probability Theory](../README.md) | [Next: Expectation and Variance →](../04-Expectation-and-Moments/notes.md)

---

# §6.3 Joint Distributions

> *"Probability is not about the odds, but about the belief in the possibility of something happening."*
> — Nassim Nicholas Taleb

## Overview

Real-world phenomena are never isolated. A language model's next token depends on every preceding token. A patient's diagnosis depends on dozens of correlated symptoms. A stock price today depends on yesterday's price, the broader market, and macroeconomic variables. To reason about multiple quantities together — to model dependencies, compute conditionals, and propagate uncertainty — we need **joint distributions**.

This section develops the full theory of multivariate probability: how to define probability distributions over pairs and vectors of random variables, how to extract marginals and conditionals, and when independence permits us to factorise complexity. We give a complete treatment of the **multivariate Gaussian**, the most important distribution in machine learning, and show how the chain rule of probability underlies the autoregressive factorisation that powers every modern large language model.

The ideas here are the probabilistic backbone of Naive Bayes classifiers, Gaussian Mixture Models, Variational Autoencoders, and the attention mechanism in transformers.

## Prerequisites

- [§6.1 Introduction and Random Variables](../01-Introduction-and-Random-Variables/notes.md) — CDF, PDF, PMF, probability axioms
- [§6.2 Common Distributions](../02-Common-Distributions/notes.md) — Gaussian, Categorical, Dirichlet, exponential family
- [§5.1 Partial Derivatives and Gradients](../../05-Multivariate-Calculus/01-Partial-Derivatives-and-Gradients/notes.md) — multivariable calculus for joint densities
- [§5.2 Jacobians and Hessians](../../05-Multivariate-Calculus/02-Jacobians-and-Hessians/notes.md) — change of variables in multiple dimensions

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive exploration of joint distributions, MVN geometry, conditionals, and ML applications |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from joint PMF verification through VAE reparameterisation |

## Learning Objectives

After completing this section, you will:

- Define joint PMF, PDF, and CDF and verify they satisfy probability axioms
- Compute marginal and conditional distributions from a joint distribution
- Apply the chain rule of probability and recognise it as the foundation of autoregressive generation
- State three equivalent characterisations of independence and give counterexamples showing pairwise ≠ mutual independence
- Explain conditional independence and its role in Naive Bayes and graphical models
- Define the covariance matrix, verify it is positive semi-definite, and interpret eigenvectors geometrically
- Derive the conditional distribution of a multivariate Gaussian using the Schur complement
- Apply the change-of-variables formula with Jacobians to transform densities
- Implement Naive Bayes classification and understand when the independence assumption helps vs hurts
- Connect the chain rule factorisation $p(\mathbf{x}) = \prod_t p(x_t \mid x_{<t})$ to GPT-style language models

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 Why Joint Distributions Matter](#11-why-joint-distributions-matter)
  - [1.2 From Univariate to Multivariate](#12-from-univariate-to-multivariate)
  - [1.3 The Independence Assumption in ML](#13-the-independence-assumption-in-ml)
  - [1.4 Historical Context](#14-historical-context)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Joint PMF (Discrete Case)](#21-joint-pmf-discrete-case)
  - [2.2 Joint PDF (Continuous Case)](#22-joint-pdf-continuous-case)
  - [2.3 Joint CDF](#23-joint-cdf)
  - [2.4 Mixed Discrete-Continuous Distributions](#24-mixed-discrete-continuous-distributions)
- [3. Marginal and Conditional Distributions](#3-marginal-and-conditional-distributions)
  - [3.1 Marginalisation](#31-marginalisation)
  - [3.2 Conditional Distributions](#32-conditional-distributions)
  - [3.3 Chain Rule of Probability](#33-chain-rule-of-probability)
  - [3.4 Bayes' Theorem for Random Variables](#34-bayes-theorem-for-random-variables)
- [4. Independence and Conditional Independence](#4-independence-and-conditional-independence)
  - [4.1 Independence — Three Equivalent Characterisations](#41-independence--three-equivalent-characterisations)
  - [4.2 Pairwise vs Mutual Independence](#42-pairwise-vs-mutual-independence)
  - [4.3 Conditional Independence](#43-conditional-independence)
  - [4.4 d-Separation Preview](#44-d-separation-preview)
- [5. Covariance and Correlation](#5-covariance-and-correlation)
  - [5.1 Covariance](#51-covariance)
  - [5.2 Pearson Correlation Coefficient](#52-pearson-correlation-coefficient)
  - [5.3 Covariance Matrix](#53-covariance-matrix)
  - [5.4 Pitfalls: Anscombe, Zero Covariance, Causation](#54-pitfalls-anscombe-zero-covariance-causation)
- [6. Multivariate Gaussian Distribution](#6-multivariate-gaussian-distribution)
  - [6.1 Definition and Parameterisation](#61-definition-and-parameterisation)
  - [6.2 Geometric Interpretation](#62-geometric-interpretation)
  - [6.3 Marginals of MVN](#63-marginals-of-mvn)
  - [6.4 Conditionals of MVN — Schur Complement](#64-conditionals-of-mvn--schur-complement)
  - [6.5 Affine Transformations](#65-affine-transformations)
  - [6.6 Maximum Entropy Property](#66-maximum-entropy-property)
- [7. Transformations of Random Vectors](#7-transformations-of-random-vectors)
  - [7.1 Change of Variables and Jacobians](#71-change-of-variables-and-jacobians)
  - [7.2 Sum of Random Variables and Convolution](#72-sum-of-random-variables-and-convolution)
  - [7.3 Copulas](#73-copulas)
  - [7.4 Order Statistics](#74-order-statistics)
- [8. ML Applications](#8-ml-applications)
  - [8.1 Naive Bayes Classifier](#81-naive-bayes-classifier)
  - [8.2 Gaussian Mixture Models](#82-gaussian-mixture-models)
  - [8.3 VAE Encoder as Conditional Distribution](#83-vae-encoder-as-conditional-distribution)
  - [8.4 Attention as Joint Distribution](#84-attention-as-joint-distribution)
  - [8.5 Autoregressive Factorisation](#85-autoregressive-factorisation)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition and Motivation

### 1.1 Why Joint Distributions Matter

A univariate distribution answers one question: what is the probability that a single random variable takes a particular value? But the world is multivariate. We want to know: what is the probability that a language model generates token $A$ *and* then token $B$? What is the probability that a patient has disease $D$ *given* symptom $S$ and test result $T$? What is the probability that two neurons in a neural network fire *together*?

To answer these questions, we need **joint distributions** — probability distributions over two or more random variables simultaneously. The joint distribution is the most complete probabilistic description of a system: from it, we can derive every single-variable distribution (by marginalising), every conditional distribution (by dividing), and every notion of dependence (by comparing the joint to the product of marginals).

The central tension in probabilistic machine learning is between **expressiveness and tractability**. A fully general joint distribution over $n$ binary variables requires $2^n - 1$ parameters — exponential in $n$. A fully factored (independence) model requires only $n$ parameters but ignores all dependencies. The art of probabilistic modelling lies in choosing structured assumptions — conditional independence, low-rank covariance, mixture models — that capture the important dependencies while remaining computationally feasible.

**For AI:** Every language model's output distribution is a joint distribution over token sequences. The reason transformers are tractable is that they use the chain rule to factor $p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_1, \ldots, x_{t-1})$, converting an intractable $V^T$-dimensional joint into $T$ tractable conditionals.

### 1.2 From Univariate to Multivariate

When we have a single random variable $X$, its distribution lives on the real line $\mathbb{R}$. The CDF $F_X(x) = P(X \leq x)$ is a function of one variable, the density $f_X(x)$ integrates to 1 over $\mathbb{R}$.

When we have two random variables $X$ and $Y$, their **joint** distribution lives on the plane $\mathbb{R}^2$. The joint CDF is $F_{X,Y}(x,y) = P(X \leq x, Y \leq y)$, a function of two variables. The joint density $f_{X,Y}(x,y)$ integrates to 1 over $\mathbb{R}^2$. For $n$ variables $\mathbf{X} = (X_1, \ldots, X_n)$, the joint distribution lives on $\mathbb{R}^n$.

```
UNIVARIATE vs MULTIVARIATE PROBABILITY
════════════════════════════════════════════════════════════════════════

  Univariate:           X ∈ ℝ
  ─────────────
  PMF:  p(x) = P(X=x)                  [discrete]
  PDF:  f(x),  ∫ f(x)dx = 1            [continuous]
  CDF:  F(x) = P(X ≤ x)

  Bivariate:            (X,Y) ∈ ℝ²
  ─────────────
  PMF:  p(x,y) = P(X=x, Y=y)           [discrete]
  PDF:  f(x,y),  ∬ f(x,y)dxdy = 1      [continuous]
  CDF:  F(x,y) = P(X ≤ x, Y ≤ y)

  Multivariate:         X = (X₁,...,Xₙ) ∈ ℝⁿ
  ─────────────
  PDF:  f(x₁,...,xₙ),  ∫...∫ f(x)dx₁...dxₙ = 1
  CDF:  F(x) = P(X₁ ≤ x₁, ..., Xₙ ≤ xₙ)

════════════════════════════════════════════════════════════════════════
```

The key new concept is **dependence**: knowing the value of $X$ can change our belief about $Y$. If $X$ and $Y$ are **independent**, knowing $X$ tells us nothing about $Y$, and the joint distribution factorises as $f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)$. Independence is a very special, restrictive assumption — and it is almost never exactly true in practice, but often a useful approximation.

### 1.3 The Independence Assumption in ML

Many machine learning algorithms make explicit independence assumptions to achieve tractability:

- **Naive Bayes:** Assumes features are conditionally independent given the class label. Despite being wrong for virtually all real datasets, this assumption produces competitive classifiers because it massively reduces the number of parameters.
- **Mean-field variational inference:** Assumes the posterior over all latent variables factorises as $q(\mathbf{z}) = \prod_i q_i(z_i)$. This is the canonical tractability trick in VAEs and Bayesian neural networks.
- **i.i.d. data assumption:** Most supervised learning theory assumes training examples $(x_i, y_i)$ are drawn independently from the same distribution. This is violated for time series, graphs, and correlated datasets — and the violation matters for generalisation bounds.
- **Residual stream in transformers:** Attention heads in the same layer see the same residual stream, making their outputs dependent. Recent work on multi-head attention analyses this dependence structure.

Understanding exactly which independence assumptions are made — and when they break — is essential for diagnosing failure modes in learned models.

### 1.4 Historical Context

The study of joint distributions has a rich history tied to the development of statistics and probability theory.

- **1888 — Francis Galton** introduced correlation while studying the joint distribution of parents' and children's heights. His "regression to the mean" was originally an observation about the conditional mean.
- **1896 — Karl Pearson** formalised the correlation coefficient and the bivariate normal distribution, laying the groundwork for multivariate statistics.
- **1933 — Andrey Kolmogorov** provided the axiomatic foundation for probability theory, defining joint distributions via product measures on product spaces.
- **1959 — Abe Sklar** proved his eponymous theorem, showing any joint distribution can be decomposed into marginals and a copula — separating the dependence structure from the marginal distributions.
- **1989 — Judea Pearl** developed graphical models (Bayesian networks), providing a language for specifying conditional independence structures among many variables. His work underpins modern probabilistic AI.
- **2013 — Diederik Kingma and Max Welling** introduced VAEs, where the core idea is the joint distribution $p(\mathbf{x}, \mathbf{z}) = p(\mathbf{z}) p(\mathbf{x} \mid \mathbf{z})$ over observed data and latent variables.
- **2017 — "Attention Is All You Need"** made the chain rule factorisation $\prod_t p(x_t \mid x_{<t})$ the dominant paradigm for language modelling at scale.

---

## 2. Formal Definitions

### 2.1 Joint PMF (Discrete Case)

Let $X$ and $Y$ be discrete random variables taking values in countable sets $\mathcal{X}$ and $\mathcal{Y}$ respectively.

**Definition:** The **joint probability mass function** of $(X, Y)$ is
$$p_{X,Y}(x, y) = P(X = x, Y = y) \quad \text{for all } x \in \mathcal{X}, y \in \mathcal{Y}$$

**Axioms:** A valid joint PMF must satisfy:
1. **Non-negativity:** $p_{X,Y}(x, y) \geq 0$ for all $(x, y)$
2. **Normalisation:** $\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p_{X,Y}(x, y) = 1$

**Example 1 — Fair dice:** Roll two fair 6-sided dice. Let $X$ = value of die 1, $Y$ = value of die 2. Then $p_{X,Y}(i,j) = 1/36$ for all $i, j \in \{1,\ldots,6\}$. The joint distribution is uniform over the $6 \times 6$ grid.

**Example 2 — Dependent variables:** Let $X \sim \text{Bernoulli}(0.4)$ and $Y = X$ (perfect dependence). Then:
$$p_{X,Y}(0,0) = 0.6, \quad p_{X,Y}(1,1) = 0.4, \quad p_{X,Y}(0,1) = p_{X,Y}(1,0) = 0$$

**Example 3 — Joint table:**

| | $Y=0$ | $Y=1$ | $Y=2$ |
|---|---|---|---|
| $X=0$ | 0.1 | 0.2 | 0.1 |
| $X=1$ | 0.3 | 0.2 | 0.1 |

This is a valid joint PMF: all entries non-negative, sum = 1.0.

**Non-example 1:** $p(0,0) = 0.3, p(0,1) = 0.4, p(1,0) = 0.2, p(1,1) = 0.2$ — sum is 1.1, not valid.

**Non-example 2:** $p(0,0) = 0.5, p(1,1) = 0.5, p(0,1) = -0.1, p(1,0) = 0.1$ — negative entry, not valid.

**Extension to $n$ variables:** The joint PMF of $(X_1, \ldots, X_n)$ is
$$p(x_1, \ldots, x_n) = P(X_1 = x_1, \ldots, X_n = x_n)$$
with $\sum_{x_1} \cdots \sum_{x_n} p(x_1, \ldots, x_n) = 1$.

### 2.2 Joint PDF (Continuous Case)

Let $X$ and $Y$ be continuous random variables.

**Definition:** A function $f_{X,Y}: \mathbb{R}^2 \to \mathbb{R}_{\geq 0}$ is a **joint probability density function** for $(X,Y)$ if for every (measurable) set $A \subseteq \mathbb{R}^2$:
$$P((X,Y) \in A) = \iint_A f_{X,Y}(x,y) \, dx \, dy$$

**Axioms:**
1. $f_{X,Y}(x,y) \geq 0$ for all $(x,y)$
2. $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dx \, dy = 1$

**Important:** Unlike probabilities, densities can exceed 1 — only integrals of densities are probabilities.

**Example 1 — Uniform on unit square:** $f_{X,Y}(x,y) = 1$ for $(x,y) \in [0,1]^2$, zero elsewhere. Check: $\int_0^1 \int_0^1 1 \, dx \, dy = 1$. ✓

**Example 2 — Standard bivariate normal (uncorrelated):**
$$f_{X,Y}(x,y) = \frac{1}{2\pi} \exp\!\left(-\frac{x^2 + y^2}{2}\right)$$
This factors as $f_X(x) \cdot f_Y(y)$ — the variables are independent standard normals.

**Example 3 — Correlated bivariate normal** (correlation $\rho$):
$$f_{X,Y}(x,y) = \frac{1}{2\pi\sqrt{1-\rho^2}} \exp\!\left(-\frac{x^2 - 2\rho xy + y^2}{2(1-\rho^2)}\right)$$

**Non-example:** $f(x,y) = 2(x+y) - 1$ on $[0,1]^2$ — this can be negative (at $(0,0)$ it equals $-1$), so not a valid PDF.

**Joint PDF in $\mathbb{R}^n$:** For a random vector $\mathbf{X} = (X_1, \ldots, X_n)^\top$, the joint PDF $f_{\mathbf{X}}: \mathbb{R}^n \to \mathbb{R}_{\geq 0}$ satisfies
$$P(\mathbf{X} \in A) = \int_A f_{\mathbf{X}}(\mathbf{x}) \, d\mathbf{x}$$
where $d\mathbf{x} = dx_1 \cdots dx_n$.

### 2.3 Joint CDF

**Definition:** The **joint cumulative distribution function** of $(X_1, \ldots, X_n)$ is
$$F(x_1, \ldots, x_n) = P(X_1 \leq x_1, \ldots, X_n \leq x_n)$$

**Properties:**
1. **Monotone:** $F$ is non-decreasing in each argument
2. **Limits:** $F(x_1, \ldots, x_n) \to 1$ as all $x_i \to \infty$; $F \to 0$ if any $x_i \to -\infty$
3. **Right-continuous:** $F$ is right-continuous in each argument
4. **Recovery of the PDF:** For continuous variables, $f(x_1, \ldots, x_n) = \frac{\partial^n F}{\partial x_1 \cdots \partial x_n}$

**Computing probabilities from the CDF (bivariate):**
$$P(a < X \leq b, c < Y \leq d) = F(b,d) - F(a,d) - F(b,c) + F(a,c)$$

This inclusion-exclusion formula generalises to $n$ dimensions with $2^n$ terms.

### 2.4 Mixed Discrete-Continuous Distributions

Some important distributions are **mixed**: discrete in one component, continuous in another.

**Example — Spike-and-slab prior (used in LoRA):** Let $Z \sim \text{Bernoulli}(\pi)$ be a binary inclusion indicator and $W \mid Z=1 \sim \mathcal{N}(0, \sigma_1^2)$, $W \mid Z=0 = 0$ (point mass). The joint distribution over $(Z, W)$ is:
- $P(Z=0, W=0) = 1-\pi$ (discrete atom)
- For $w \neq 0$: $f(Z=1, W=w) = \pi \cdot \frac{1}{\sigma_1\sqrt{2\pi}} e^{-w^2/(2\sigma_1^2)}$ (continuous)

**Example — Gaussian Mixture Model (GMM):** Let $Z \in \{1, \ldots, K\}$ be a discrete cluster label with $P(Z=k) = \pi_k$, and $\mathbf{X} \mid Z=k \sim \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$. The joint $p(Z, \mathbf{X})$ is mixed. Marginalising out $Z$ gives the mixture density:
$$f(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \Sigma_k)$$

This is one of the most widely used density estimation models in unsupervised learning.

---

## 3. Marginal and Conditional Distributions

### 3.1 Marginalisation

Given a joint distribution over $(X, Y)$, we obtain the **marginal distribution** of $X$ by "integrating out" $Y$ — summing or integrating over all possible values of $Y$.

**Discrete:**
$$p_X(x) = \sum_{y \in \mathcal{Y}} p_{X,Y}(x, y)$$

**Continuous:**
$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy$$

**Intuition:** The marginal $p_X$ answers: "What is the probability that $X=x$, regardless of what $Y$ is doing?" We collapse the joint table by summing rows (to get the $X$ marginal) or columns (to get the $Y$ marginal).

**Example — From joint table:**

| | $Y=0$ | $Y=1$ | $Y=2$ | $p_X(x)$ |
|---|---|---|---|---|
| $X=0$ | 0.1 | 0.2 | 0.1 | **0.4** |
| $X=1$ | 0.3 | 0.2 | 0.1 | **0.6** |
| $p_Y(y)$ | **0.4** | **0.4** | **0.2** | **1.0** |

**Marginalisation in $n$ dimensions:** To obtain the marginal of $\mathbf{X}_A$ (a subset of variables indexed by $A$), integrate over all other variables $\mathbf{X}_{A^c}$:
$$f_{\mathbf{X}_A}(\mathbf{x}_A) = \int f_{\mathbf{X}}(\mathbf{x}_A, \mathbf{x}_{A^c}) \, d\mathbf{x}_{A^c}$$

**For AI:** Training a language model means learning the joint distribution $p(x_1, \ldots, x_T)$ over token sequences. At inference time, we often want $p(x_t \mid x_1, \ldots, x_{t-1})$ — a conditional derived from this joint. Computing the marginal $p(x_t) = \sum_{x_1,\ldots,x_{t-1}} p(x_1,\ldots,x_t)$ would require summing over $V^{t-1}$ sequences — intractable. This is why the chain rule factorisation is essential.

### 3.2 Conditional Distributions

The **conditional distribution** of $Y$ given $X = x$ captures how our belief about $Y$ changes when we learn the value of $X$.

**Discrete:**
$$p_{Y \mid X}(y \mid x) = P(Y = y \mid X = x) = \frac{p_{X,Y}(x, y)}{p_X(x)}, \quad \text{provided } p_X(x) > 0$$

**Continuous:**
$$f_{Y \mid X}(y \mid x) = \frac{f_{X,Y}(x, y)}{f_X(x)}, \quad \text{provided } f_X(x) > 0$$

**Verification:** The conditional distribution is a valid probability distribution in $y$:
$$\sum_y p_{Y \mid X}(y \mid x) = \sum_y \frac{p_{X,Y}(x,y)}{p_X(x)} = \frac{p_X(x)}{p_X(x)} = 1 \quad \checkmark$$

**Interpretation:** $f_{Y \mid X}(y \mid x)$ is a function of $y$ for fixed $x$. As $x$ varies, we get a family of distributions — the conditional distribution function is a map from values of $x$ to distributions on $\mathcal{Y}$.

**Example — Bivariate normal:** For $(X, Y)$ jointly Gaussian with correlation $\rho$:
$$Y \mid X = x \sim \mathcal{N}\!\left(\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X), \; \sigma_Y^2(1-\rho^2)\right)$$

The conditional mean is linear in $x$ — this is the linear regression formula. The conditional variance $\sigma_Y^2(1-\rho^2)$ is always smaller than the marginal variance $\sigma_Y^2$ (conditioning reduces uncertainty), with equality iff $\rho = 0$.

**For AI:** Every output of a neural network is a parameterised conditional distribution. A classifier outputs $p(y \mid \mathbf{x}; \theta)$; a language model outputs $p(x_t \mid x_{<t}; \theta)$; a diffusion model computes $p(\mathbf{x}_{t-1} \mid \mathbf{x}_t; \theta)$. All of probabilistic deep learning is the art of learning flexible conditional distributions from data.

### 3.3 Chain Rule of Probability

The **chain rule** (or product rule) expresses any joint distribution as a product of conditional distributions:

$$p(x_1, x_2) = p(x_1) \cdot p(x_2 \mid x_1)$$
$$p(x_1, x_2, x_3) = p(x_1) \cdot p(x_2 \mid x_1) \cdot p(x_3 \mid x_1, x_2)$$

In general, for any ordering of variables:
$$p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i \mid x_1, \ldots, x_{i-1})$$

**Proof:** By induction. Base case $n=2$: $p(x_1, x_2) = p(x_2 \mid x_1) p(x_1)$ is the definition of conditional probability. Inductive step: $p(x_1,\ldots,x_n) = p(x_n \mid x_1,\ldots,x_{n-1}) \cdot p(x_1,\ldots,x_{n-1})$, and apply the inductive hypothesis.

```
CHAIN RULE ORDERINGS FOR p(A, B, C)
════════════════════════════════════════════════════════════════════════

  Forward:   p(A) · p(B|A) · p(C|A,B)
  Reverse:   p(C) · p(B|C) · p(A|B,C)
  Mixed:     p(B) · p(A|B) · p(C|A,B)

  All three represent the same joint distribution p(A,B,C).
  Graphical models choose the ordering that exploits conditional
  independence to reduce the complexity of each conditional.

════════════════════════════════════════════════════════════════════════
```

**For AI — Autoregressive language models:** The chain rule directly motivates the autoregressive factorisation:
$$p(x_1, x_2, \ldots, x_T) = p(x_1) \prod_{t=2}^T p(x_t \mid x_1, \ldots, x_{t-1})$$

A GPT model learns one neural network to represent all $T$ conditional distributions simultaneously (via causal masking). Each forward pass computes all $T$ conditionals in parallel during training, while at inference time they are evaluated sequentially. The chain rule is not a modelling choice — it is an exact identity.

### 3.4 Bayes' Theorem for Random Variables

For discrete random variables:
$$p_{X \mid Y}(x \mid y) = \frac{p_{Y \mid X}(y \mid x) \cdot p_X(x)}{\sum_{x'} p_{Y \mid X}(y \mid x') \cdot p_X(x')}$$

For continuous random variables (replace sums with integrals):
$$f_{X \mid Y}(x \mid y) = \frac{f_{Y \mid X}(y \mid x) \cdot f_X(x)}{\int f_{Y \mid X}(y \mid x') \cdot f_X(x') \, dx'}$$

In Bayesian terminology:
$$\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\underbrace{p(\mathcal{D} \mid \theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}$$

**Example — Medical diagnosis:** Let $\theta$ = "patient has disease", $D$ = "test is positive".
- Prior: $p(\theta=1) = 0.01$; Sensitivity: $p(D=1 \mid \theta=1) = 0.95$; FPR: $p(D=1 \mid \theta=0) = 0.05$

$$p(\theta=1 \mid D=1) = \frac{0.95 \times 0.01}{0.95 \times 0.01 + 0.05 \times 0.99} \approx 0.161$$

Despite a 95% sensitive test, a positive result only gives 16% posterior probability — because the disease is rare. This is the **base rate fallacy**, a consequence of not properly using the joint distribution.

**Continuous Bayes — Gaussian-Gaussian:** Prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$, likelihood $X \mid \theta \sim \mathcal{N}(\theta, \sigma^2)$. The posterior is:
$$\theta \mid X = x \sim \mathcal{N}\!\left(\frac{\sigma^2 \mu_0 + \sigma_0^2 x}{\sigma^2 + \sigma_0^2}, \; \frac{\sigma^2 \sigma_0^2}{\sigma^2 + \sigma_0^2}\right)$$

The posterior mean is a weighted average of prior mean and observation — a fundamental result in Bayesian estimation.

---

## 4. Independence and Conditional Independence

### 4.1 Independence — Three Equivalent Characterisations

$X$ and $Y$ are **independent**, written $X \perp\!\!\!\perp Y$, if and only if any of the following equivalent conditions holds:

**Characterisation 1 (Factorisation):**
$$p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y) \quad \text{for all } x, y$$

**Characterisation 2 (Conditional = Marginal):**
$$p_{Y \mid X}(y \mid x) = p_Y(y) \quad \text{for all } x, y$$

**Characterisation 3 (CDF Factorisation):**
$$F_{X,Y}(x,y) = F_X(x) \cdot F_Y(y) \quad \text{for all } x, y$$

**Proof of equivalence (1) ↔ (2):** If $p_{X,Y} = p_X p_Y$, then $p_{Y \mid X}(y \mid x) = p_{X,Y}/p_X = p_Y$. Conversely, $p_{X,Y} = p_{Y \mid X} p_X = p_Y p_X$.

**Independence of functions:** If $X \perp\!\!\!\perp Y$, then $g(X) \perp\!\!\!\perp h(Y)$ for any measurable functions $g, h$.

**Caution:** Independence requires factorisation *everywhere* in the support. A single point where $p_{X,Y}(x,y) \neq p_X(x)p_Y(y)$ breaks independence.

### 4.2 Pairwise vs Mutual Independence

For three or more variables, we must distinguish between **pairwise** and **mutual** independence.

- $(X_1, X_2, X_3)$ are **pairwise independent** if $X_i \perp\!\!\!\perp X_j$ for all pairs $(i,j)$
- $(X_1, X_2, X_3)$ are **mutually independent** if $p(x_1, x_2, x_3) = p(x_1)p(x_2)p(x_3)$ for all $(x_1, x_2, x_3)$

**Mutual independence implies pairwise, but NOT vice versa.**

**Classic counterexample (Bernstein, 1946):** Let $X_1, X_2 \sim \text{Bernoulli}(1/2)$ independently, and $X_3 = X_1 \oplus X_2$ (XOR). Then all pairs are independent (pairwise), but $P(X_3=1 \mid X_1=1, X_2=1) = 0 \neq 1/2 = P(X_3=1)$ — NOT mutually independent.

**General definition:** $(X_1, \ldots, X_n)$ are mutually independent iff for every subset $S \subseteq \{1, \ldots, n\}$:
$$p\!\left(\bigcap_{i \in S} \{X_i = x_i\}\right) = \prod_{i \in S} p(X_i = x_i)$$

This requires $2^n - n - 1$ factorisation conditions — all subsets of size $\geq 2$.

### 4.3 Conditional Independence

$X$ and $Y$ are **conditionally independent given $Z$**, written $X \perp\!\!\!\perp Y \mid Z$, if:
$$p(x, y \mid z) = p(x \mid z) \cdot p(y \mid z) \quad \text{for all } x, y, z$$

Equivalently: $p(x \mid y, z) = p(x \mid z)$ — knowing $Z$, additional knowledge of $Y$ provides no information about $X$.

**Conditional independence does not imply marginal independence, and vice versa.**

**Example 1 (CI but not marginal):** $Z$ selects which coin; $X$ and $Y$ are independent flips of that coin. Given $Z$: $X \perp\!\!\!\perp Y \mid Z$. Marginally: $X$ and $Y$ are correlated (via the shared coin).

**Example 2 (Marginal but not CI):** $X, Y$ are independent fair coin flips; $Z = X \oplus Y$. Then $X \perp\!\!\!\perp Y$, but $X \not\perp\!\!\!\perp Y \mid Z=0$ (knowing the XOR makes the flips correlated).

```
CONDITIONAL INDEPENDENCE STRUCTURES
════════════════════════════════════════════════════════════════════════

  Chain:    X → Z → Y      X ⊥⊥ Y | Z  (Z blocks the path)
  Fork:     X ← Z → Y      X ⊥⊥ Y | Z  (Z explains dependence)
  Collider: X → Z ← Y      X ⊥⊥ Y      (marginally independent)
                            X ⊭⊥⊥ Y | Z (conditioning creates dependence)
                            — Berkson's paradox / selection bias

════════════════════════════════════════════════════════════════════════
```

**For AI:**
- **Naive Bayes:** $p(\mathbf{x} \mid y) = \prod_i p(x_i \mid y)$ — features are conditionally independent given class
- **Causal masking:** In transformers, position $t$ is conditionally independent of future tokens given context $x_{<t}$

### 4.4 d-Separation Preview

**Berkson's paradox** (collider bias) illustrates how conditioning can *create* dependence. Suppose acceptance to a selective university depends on both academic ability $A$ and athletic skill $S$, where $A \perp\!\!\!\perp S$ marginally. Among admitted students (conditioning on admission), $A$ and $S$ are negatively correlated — high academic ability reduces the need for athletic skill to justify admission.

This is the collider structure $A \to Z \leftarrow S$ with $Z$ conditioned on. It appears throughout ML as **selection bias**: training on filtered data creates spurious dependencies between filtered-out features.

> **Forward reference:** The systematic framework for reading off all conditional independence relationships from a directed acyclic graph (d-separation, Bayesian networks, Markov blankets) is developed in the graphical models literature; see Pearl (2000), *Causality*.

---

## 5. Covariance and Correlation

### 5.1 Covariance

**Definition:** The **covariance** of $X$ and $Y$ is:
$$\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

> **Forward reference:** The derivations of expectations using LOTUS are in [§6.4 Expectation and Variance](../04-Expectation-and-Moments/notes.md). Here we use these as defined.

**Sign interpretation:**
- $\text{Cov}(X,Y) > 0$: $X$ and $Y$ tend to move in the same direction
- $\text{Cov}(X,Y) < 0$: $X$ and $Y$ tend to move in opposite directions
- $\text{Cov}(X,Y) = 0$: **uncorrelated** — no linear relationship

**Key properties:**
1. **Symmetry:** $\text{Cov}(X,Y) = \text{Cov}(Y,X)$
2. **Bilinearity:** $\text{Cov}(aX+bZ, Y) = a\,\text{Cov}(X,Y) + b\,\text{Cov}(Z,Y)$
3. **Self-covariance:** $\text{Cov}(X,X) = \text{Var}(X)$
4. **Independence implies zero covariance:** $X \perp\!\!\!\perp Y \Rightarrow \text{Cov}(X,Y) = 0$
5. **Variance of sum:** $\text{Var}(X+Y) = \text{Var}(X) + 2\text{Cov}(X,Y) + \text{Var}(Y)$

**Caution:** Property 4 does NOT reverse. $\text{Cov}(X,Y) = 0$ does NOT imply $X \perp\!\!\!\perp Y$.

**Example — Zero covariance, strong dependence:** Let $X \sim \text{Uniform}(-1, 1)$ and $Y = X^2$. Then $\mathbb{E}[XY] = \mathbb{E}[X^3] = 0$ (odd function of symmetric distribution), so $\text{Cov}(X,Y) = 0$. But $Y$ is a deterministic function of $X$ — perfectly dependent. Covariance only measures *linear* dependence.

### 5.2 Pearson Correlation Coefficient

The **Pearson correlation coefficient** normalises covariance to $[-1, 1]$:
$$\rho(X,Y) = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\,\text{Var}(Y)}} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

**Geometric interpretation:** $\rho$ is the cosine of the angle between centred random variables viewed as vectors in $L^2$:
$$\rho(X,Y) = \cos\theta, \quad \theta = \angle(X - \mathbb{E}[X], \, Y - \mathbb{E}[Y])$$

**Properties:**
- $|\rho(X,Y)| \leq 1$ (Cauchy-Schwarz)
- $\rho = 1$: perfect positive linear relationship
- $\rho = -1$: perfect negative linear relationship
- $\rho = 0$: uncorrelated (no linear relationship — but possibly nonlinear)
- **Scale invariance:** $\rho(aX + b, cY + d) = \text{sign}(ac) \cdot \rho(X, Y)$

### 5.3 Covariance Matrix

For a random vector $\mathbf{X} = (X_1, \ldots, X_n)^\top$, the **covariance matrix** is:
$$\Sigma = \text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]$$

where $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$. Entry $(i,j)$ is $\Sigma_{ij} = \text{Cov}(X_i, X_j)$; diagonal $\Sigma_{ii} = \text{Var}(X_i)$.

**Key properties:**
1. **Symmetry:** $\Sigma = \Sigma^\top$
2. **Positive semi-definiteness:** $\mathbf{v}^\top \Sigma \mathbf{v} \geq 0$ for all $\mathbf{v} \in \mathbb{R}^n$

**Proof of PSD:** $\mathbf{v}^\top \Sigma \mathbf{v} = \mathbb{E}[(\mathbf{v}^\top(\mathbf{X}-\boldsymbol{\mu}))^2] \geq 0$ since the expectation of a squared quantity is non-negative. ∎

3. **Transformation:** $\text{Cov}(A\mathbf{X} + \mathbf{b}) = A \Sigma A^\top$

**Eigendecomposition:** $\Sigma = Q \Lambda Q^\top$ where $Q$ is orthogonal (eigenvectors) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ with $\lambda_i \geq 0$. Eigenvectors are the **principal axes** of the distribution; eigenvalues are the variances along those axes.

**For AI:**
- **PCA** finds the eigendecomposition of the sample covariance matrix
- **Adam** second moment estimate approximates the diagonal of $\mathbb{E}[g_t g_t^\top]$
- **Whitening** applies $\Sigma^{-1/2}$ to decorrelate features — the foundation of BatchNorm

### 5.4 Pitfalls: Anscombe, Zero Covariance, Causation

**Anscombe's quartet (1973):** Four datasets with nearly identical means, variances, and $\rho \approx 0.816$ yet completely different visual patterns — linear, curved, linear with outlier, vertical cluster. Lesson: always visualise. Summary statistics can be identical for very different distributions.

**Zero covariance ≠ independence:** Covariance only captures *linear* dependence. For general dependence, use **mutual information** $I(X;Y) = \mathbb{E}[\log \frac{p(X,Y)}{p(X)p(Y)}]$ — which is zero if and only if $X \perp\!\!\!\perp Y$.

**Correlation ≠ causation:** Ice cream sales and drowning rates are positively correlated (confounded by summer). In ML: spurious correlations in training data are learned as shortcuts (Geirhos et al., 2020). Causal invariance methods (IRM, DRO) aim to learn causal rather than associative features.

---

## 6. Multivariate Gaussian Distribution

### 6.1 Definition and Parameterisation

The **multivariate Gaussian (MVN)** is the most important distribution in machine learning.

**Definition:** $\mathbf{X} \in \mathbb{R}^n$ follows $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with density:
$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

where $\boldsymbol{\mu} \in \mathbb{R}^n$ is the mean vector and $\Sigma \in \mathbb{R}^{n \times n}$ is the covariance matrix (symmetric positive definite).

**Parameters:** $n$ for $\boldsymbol{\mu}$ + $n(n+1)/2$ for $\Sigma$ = $n(n+3)/2$ total.

**Natural parameterisation (precision form):** Using $\Lambda = \Sigma^{-1}$ (precision matrix) and $\boldsymbol{\eta} = \Lambda\boldsymbol{\mu}$:
$$f(\mathbf{x}) \propto \exp\!\left(-\frac{1}{2}\mathbf{x}^\top \Lambda \mathbf{x} + \boldsymbol{\eta}^\top \mathbf{x}\right)$$

**Special cases:**
- $\Sigma = \sigma^2 I$: **isotropic** — all dimensions independent, equal variance
- $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$: **diagonal** — independent, heteroscedastic
- $\Sigma$ full rank: **full covariance** — captures all pairwise correlations

### 6.2 Geometric Interpretation

The Mahalanobis distance $(\mathbf{x} - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})$ measures distance in the $\Sigma$ metric. **Level sets** (contours of constant density) are **ellipsoids** in $\mathbb{R}^n$.

The eigendecomposition $\Sigma = Q \Lambda Q^\top$ reveals:
- Eigenvectors of $\Sigma$ (columns of $Q$): **principal axes** of the ellipsoid
- Eigenvalues $\lambda_i$: **squared semi-axis lengths** — larger = more spread in that direction

```
MVN GEOMETRY (2D EXAMPLE)
════════════════════════════════════════════════════════════════════════

  Σ = [[4, 2],       Eigenvalues: λ₁ ≈ 5.24,  λ₂ ≈ 0.76
       [2, 3]]       Correlation: ρ = 2/√(4·3) ≈ 0.577

        ↑ x₂
        │    /v₁  (long axis, proportional to √λ₁)
        │   /  _.-'
        │  .-'  ellipse  ·µ
        │  `-.
        │      `-. v₂ (short axis)
        └───────────────────→ x₁

  After whitening Z = Σ^{-1/2}(X - µ): level sets become circles.

════════════════════════════════════════════════════════════════════════
```

### 6.3 Marginals of MVN

**Theorem:** Any marginal of an MVN is also Gaussian.

Let $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with block partition:
$$\mathbf{X} = \begin{pmatrix}\mathbf{X}_1 \\ \mathbf{X}_2\end{pmatrix}, \quad \boldsymbol{\mu} = \begin{pmatrix}\boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2\end{pmatrix}, \quad \Sigma = \begin{pmatrix}\Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22}\end{pmatrix}$$

Then: $\mathbf{X}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_{11})$ and $\mathbf{X}_2 \sim \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_{22})$.

**Important:** The converse is FALSE — marginals being Gaussian does not imply the joint is Gaussian. Non-Gaussian copulas can yield Gaussian marginals.

### 6.4 Conditionals of MVN — Schur Complement

**Theorem:** Under the same partition:
$$\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \Sigma_{1|2})$$

where:
$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$
$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$

The matrix $\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ is the **Schur complement** of $\Sigma_{22}$ in $\Sigma$.

**Intuition:**
- Conditional mean = prior mean + correction proportional to deviation of $\mathbf{x}_2$ from its mean
- Conditional covariance does NOT depend on $\mathbf{x}_2$ — unique to Gaussians
- $\Sigma_{1|2} \preceq \Sigma_{11}$: conditioning always reduces uncertainty

**Applications:** Gaussian process regression (predictive distribution at test points), Bayesian linear regression (posterior over weights), Kalman filter (state update step — the Kalman gain $K = \Sigma_{12}\Sigma_{22}^{-1}$).

### 6.5 Affine Transformations

**Theorem:** If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ and $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$, then:
$$\mathbf{Y} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b}, \; A\Sigma A^\top)$$

**Applications:**
- **Whitening:** $A = \Sigma^{-1/2}$ → $\mathbf{Y} \sim \mathcal{N}(\mathbf{0}, I)$
- **PCA projection:** $A = Q_k^\top$ (top $k$ eigenvectors) → $k$-dimensional projection
- **Reparameterisation trick:** Sample $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$, set $\mathbf{z} = \boldsymbol{\mu} + L\boldsymbol{\varepsilon}$ where $LL^\top = \Sigma$ — allows gradients to flow through the sampling step in VAEs

### 6.6 Maximum Entropy Property

**Theorem:** Among all distributions on $\mathbb{R}^n$ with fixed mean $\boldsymbol{\mu}$ and covariance $\Sigma$, the MVN has maximum differential entropy:
$$h(\mathcal{N}(\boldsymbol{\mu}, \Sigma)) = \frac{1}{2}\log\left((2\pi e)^n |\Sigma|\right)$$

**Implication for AI:** Gaussian priors/posteriors are the maximum-entropy (least informative) choice given mean and covariance constraints. This justifies Gaussian weight initialisation (no preferred direction), Gaussian latent priors in VAEs (maximum entropy latent space), and Laplace approximation to the Bayesian posterior.

---

## 7. Transformations of Random Vectors

### 7.1 Change of Variables and Jacobians

Given random vector $\mathbf{X}$ with PDF $f_{\mathbf{X}}$ and differentiable bijection $\mathbf{g}: \mathbb{R}^n \to \mathbb{R}^n$, the PDF of $\mathbf{Y} = \mathbf{g}(\mathbf{X})$ is:
$$f_{\mathbf{Y}}(\mathbf{y}) = f_{\mathbf{X}}(\mathbf{g}^{-1}(\mathbf{y})) \cdot \left|\det J_{\mathbf{g}^{-1}}(\mathbf{y})\right|$$

where $J_{\mathbf{g}^{-1}}$ is the Jacobian of the inverse map. The Jacobian determinant measures how much $\mathbf{g}$ stretches/compresses volume — probability mass must be conserved.

**Example 1 — Polar coordinates:** $(X_1, X_2) \sim \mathcal{N}(\mathbf{0}, I)$, transform to $(R, \Theta)$. The Jacobian is $|\det J| = r$, giving:
$$f_{R,\Theta}(r,\theta) = \frac{r}{2\pi} e^{-r^2/2}$$
So $R$ has the Rayleigh distribution, $\Theta \sim \text{Uniform}(0, 2\pi)$, and $R \perp\!\!\!\perp \Theta$.

**Example 2 — Box-Muller transform:** $U_1, U_2 \sim \text{Uniform}(0,1)$ independent. Define:
$$X_1 = \sqrt{-2\log U_1} \cos(2\pi U_2), \quad X_2 = \sqrt{-2\log U_1} \sin(2\pi U_2)$$
Then $X_1, X_2 \sim \mathcal{N}(0,1)$ independently — a practical algorithm for Gaussian sampling.

**Example 3 — Log-normal:** $X \sim \mathcal{N}(\mu, \sigma^2)$, $Y = e^X$. Then:
$$f_Y(y) = \frac{1}{y\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(\log y - \mu)^2}{2\sigma^2}\right), \quad y > 0$$

**For AI — Normalising flows:** Flows learn a chain $\mathbf{g} = \mathbf{g}_K \circ \cdots \circ \mathbf{g}_1$ from a simple base distribution to a complex target. The change-of-variables formula gives exact log-likelihoods:
$$\log f_{\mathbf{Y}}(\mathbf{y}) = \log f_{\mathbf{Z}}(\mathbf{z}) - \sum_{k=1}^K \log\left|\det J_{\mathbf{g}_k}\right|$$

Efficient flows (RealNVP, Glow) use triangular Jacobians with $O(n)$ determinant computation.

### 7.2 Sum of Random Variables and Convolution

If $X \perp\!\!\!\perp Y$ and $Z = X + Y$:
$$f_Z(z) = (f_X * f_Y)(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z-x) \, dx$$

**Stability under convolution:**
- $\mathcal{N}(\mu_1, \sigma_1^2) * \mathcal{N}(\mu_2, \sigma_2^2) = \mathcal{N}(\mu_1+\mu_2, \sigma_1^2+\sigma_2^2)$
- $\text{Poisson}(\lambda_1) * \text{Poisson}(\lambda_2) = \text{Poisson}(\lambda_1+\lambda_2)$
- $\text{Gamma}(\alpha_1, \beta) * \text{Gamma}(\alpha_2, \beta) = \text{Gamma}(\alpha_1+\alpha_2, \beta)$

> **Forward reference:** The Central Limit Theorem — repeated convolution of any distribution with finite variance converges to Gaussian — is proved in [§6.6 Stochastic Processes](../06-Stochastic-Processes/notes.md) via MGF factorisation.

### 7.3 Copulas

A **copula** separates the dependence structure from the marginals.

**Sklar's Theorem (1959):** For any joint CDF $F_{X,Y}$ with continuous marginals $F_X$ and $F_Y$, there exists a unique copula $C: [0,1]^2 \to [0,1]$ such that:
$$F_{X,Y}(x,y) = C(F_X(x), F_Y(y))$$

The copula $C$ is the joint CDF of $(F_X(X), F_Y(Y))$ — the probability-integral-transformed variables with $\text{Uniform}(0,1)$ marginals.

**Common copulas:**
- **Independence:** $C(u,v) = uv$
- **Gaussian copula:** Dependence structure of a bivariate normal with arbitrary marginals
- **Clayton:** Lower tail dependence (joint extremes at low values more likely)
- **Gumbel:** Upper tail dependence

**For AI:** Copulas model the dependence structure of generated text distributions independently of individual token marginals. In risk modelling, asset returns are often fitted with Gaussian marginals but non-Gaussian tail dependence via t-copulas.

### 7.4 Order Statistics

Given $n$ i.i.d. samples $X_1, \ldots, X_n$ from CDF $F$ with density $f$, the sorted values $X_{(1)} \leq \cdots \leq X_{(n)}$ are **order statistics**.

$$f_{X_{(k)}}(x) = \frac{n!}{(k-1)!(n-k)!} F(x)^{k-1}(1-F(x))^{n-k}f(x)$$

**Minimum:** $f_{X_{(1)}}(x) = n(1-F(x))^{n-1}f(x)$

**Maximum:** $f_{X_{(n)}}(x) = nF(x)^{n-1}f(x)$

**For AI — Top-$k$ sampling:** Language model top-$k$ sampling selects among the $k$ highest-probability tokens — analysable via the joint distribution of the $k$ largest order statistics of the logit vector. Top-$p$ (nucleus) sampling is related to the quantile of the cumulative distribution.

---

## 8. ML Applications

### 8.1 Naive Bayes Classifier

**Model:** Given class label $Y \in \{1, \ldots, K\}$ and features $\mathbf{x} = (x_1, \ldots, x_d)$:
$$p(\mathbf{x} \mid Y = k) = \prod_{i=1}^d p(x_i \mid Y = k)$$

Features are **conditionally independent** given the class — reduces parameter count from exponential to linear in $d$.

**Decision rule:**
$$\hat{y} = \arg\max_k \left[\log p(Y=k) + \sum_{i=1}^d \log p(x_i \mid Y=k)\right]$$

**Variants:** Gaussian NB (continuous features), Multinomial NB (word counts), Bernoulli NB (binary features).

**Why it works despite being wrong:** The independence assumption is almost always false, yet Naive Bayes is competitive because:
1. Classification only requires the *sign* of the log-posterior ratio to be correct — much weaker than accurate probability estimation
2. The independence assumption acts as a regulariser with limited data
3. The decision boundary is a hyperplane — equivalent to logistic regression with NB features

### 8.2 Gaussian Mixture Models

A **GMM** is a joint distribution $p(Z, \mathbf{X})$ where:
- $Z \in \{1, \ldots, K\}$ is a latent cluster indicator with $P(Z=k) = \pi_k$
- $\mathbf{X} \mid Z=k \sim \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$

The marginal mixture density: $p(\mathbf{x}) = \sum_{k=1}^K \pi_k \, \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \Sigma_k)$

**EM algorithm:**
- **E-step:** $r_{ik} = P(Z=k \mid \mathbf{x}_i) \propto \pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \Sigma_k)$ (Bayes' theorem for the joint)
- **M-step:** Update parameters by maximising the expected complete-data log-likelihood

**For AI:** GMMs model speaker diarisation, sound source separation, and codebook structure in VQ-VAEs. Mechanistic interpretability research (Elhage et al., 2022) has found that transformer residual stream activations often decompose into Gaussian clusters across different input contexts.

### 8.3 VAE Encoder as Conditional Distribution

A **VAE** defines generative joint distribution $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z}) \cdot p(\mathbf{z})$ with prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$.

The encoder approximates the intractable posterior $p_\theta(\mathbf{z} \mid \mathbf{x})$ with:
$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\sigma^2_\phi(\mathbf{x})))$$

**ELBO:**
$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \| p(\mathbf{z}))$$

**Reparameterisation trick:** Write $\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$. This moves stochasticity into $\boldsymbol{\varepsilon}$ (not a function of $\phi$), so gradients flow through $\boldsymbol{\mu}_\phi$ and $\boldsymbol{\sigma}_\phi$ — the affine transformation property of MVN applied to enable gradient-based learning.

### 8.4 Attention as Joint Distribution

For query $\mathbf{q}_t$ and keys $\mathbf{k}_1, \ldots, \mathbf{k}_T$:
$$\alpha_{ts} = \frac{\exp(\mathbf{q}_t^\top \mathbf{k}_s / \sqrt{d})}{\sum_{s'} \exp(\mathbf{q}_t^\top \mathbf{k}_{s'} / \sqrt{d})} = p(\text{attend to position } s \mid \text{query } t)$$

This defines a conditional distribution over positions given the query. The output is the conditional expectation:
$$\mathbf{o}_t = \sum_s \alpha_{ts} \mathbf{v}_s = \mathbb{E}_{s \sim p(\cdot \mid t)}[\mathbf{v}_s]$$

The attention weight matrix $\{\alpha_{ts}\}_{t,s}$ defines a joint distribution over (query, key) position pairs. Causal masking encodes $x_t \perp\!\!\!\perp x_{s'} \mid x_{<t}$ for $s' > t$ — the autoregressive conditional independence assumption.

### 8.5 Autoregressive Factorisation

The chain rule gives the autoregressive factorisation used by every modern language model:
$$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_1, \ldots, x_{t-1})$$

**Why this factorisation?**
1. **Exact:** A mathematical identity — no approximation
2. **Tractable:** Each conditional is over vocabulary $\mathcal{V}$ (~50K–150K tokens) — manageable
3. **Sample-able:** Sequential sampling from conditionals
4. **Trainable:** Cross-entropy loss on each position; all positions trained in parallel with teacher forcing

**KV cache:** Storing $(\mathbf{k}_s, \mathbf{v}_s)$ for $s < t$ avoids recomputation — a direct consequence of the sequential conditional structure. The cache grows linearly in sequence length, motivating context-length-efficient architectures (sliding window, linear attention, Mamba).

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Confusing $p(X,Y)$ with $p(X)p(Y)$ | The second assumes independence | Always verify factorisation before assuming independence |
| 2 | Marginalising over the wrong variable | Summing out $X$ gives $p_Y$, not $p_X$ | Identify which variable to sum/integrate out |
| 3 | Concluding $X \perp\!\!\!\perp Y$ from $\text{Cov}(X,Y) = 0$ | Zero covariance = no linear dependence only | Compute $p(X,Y)$ vs $p(X)p(Y)$; use mutual information |
| 4 | Confusing marginal and conditional independence | $X \perp\!\!\!\perp Y$ and $X \perp\!\!\!\perp Y \mid Z$ are independent statements | Draw the causal structure; distinguish fork, chain, collider |
| 5 | Wrong block structure in conditional MVN | Swapping $\Sigma_{11}$ and $\Sigma_{22}$ gives wrong result | Put the conditioning variable consistently in $\mathbf{x}_2$ |
| 6 | Assuming Gaussian marginals imply Gaussian joint | Non-Gaussian copulas yield Gaussian marginals | Only MVN guarantees both marginals and conditionals Gaussian |
| 7 | Using $|\det J_\mathbf{g}|$ instead of $|\det J_{\mathbf{g}^{-1}}|$ | The formula uses the Jacobian of the *inverse* | If $\mathbf{y} = \mathbf{g}(\mathbf{x})$, use $|\partial \mathbf{x}/\partial \mathbf{y}|$ |
| 8 | Confusing pairwise and mutual independence | Pairwise does not imply mutual (Bernstein XOR example) | Check all subset factorisations for mutual independence |
| 9 | Using $\rho$ as a complete measure of dependence | Anscombe's quartet: same $\rho$, very different distributions | Always visualise; use mutual information for general dependence |
| 10 | Forgetting the normalising constant in continuous Bayes | Posterior must integrate to 1 | Compute the evidence $\int p(\mathcal{D} \mid \theta)p(\theta)d\theta$ or use VI/MCMC |
| 11 | Conditioning on a collider and expecting independence | Conditioning on a collider *creates* dependence (Berkson) | Draw the causal graph; identify colliders before conditioning |
| 12 | Comparing chain rule terms from different orderings | Different orderings give different factorisations of the same joint | Fix the ordering; compare $p(x_1,\ldots,x_n)$ directly |

---

## 10. Exercises

**Exercise 1 ★ — Joint PMF Verification**
Given the joint PMF table, verify it is valid, compute all marginal distributions, and determine whether $X \perp\!\!\!\perp Y$.

| | $Y=0$ | $Y=1$ | $Y=2$ |
|---|---|---|---|
| $X=0$ | $a$ | $0.1$ | $0.15$ |
| $X=1$ | $0.2$ | $0.1$ | $b$ |

(a) Find $a, b$ such that $p_X(0) = 0.4$. (b) Compute marginals $p_X$ and $p_Y$. (c) Test independence via the factorisation condition. (d) Compute $P(X=1 \mid Y=1)$.

**Exercise 2 ★ — Continuous Marginals and Conditionals**
Let $(X,Y)$ have joint PDF $f(x,y) = c \cdot xy$ on $[0,2] \times [0,1]$, zero elsewhere.

(a) Find $c$. (b) Compute marginals $f_X$ and $f_Y$. (c) Are $X$ and $Y$ independent? (d) Compute $f_{Y \mid X}(y \mid x)$.

**Exercise 3 ★ — Chain Rule and Autoregressive Factorisation**
A simplified language model over vocabulary $\{A, B, C\}$ has: $p(A) = 0.5$, $p(B \mid A) = 0.4$, $p(C \mid A) = 0.3$, $p(A \mid B) = 0.5$.

(a) Compute $p(A, B)$ using the chain rule in forward order. (b) Compute $p(B, A)$ in reverse order. Verify $p(A,B) = p(B,A)$. (c) Implement a simple bigram language model and sample 100 two-token sequences.

**Exercise 4 ★ — Independence Testing with Simulation**
(a) Generate $10^5$ pairs $(X, Y)$ where $X \sim \mathcal{N}(0,1)$ and $Y = X^2 + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.1)$. Compute Pearson $\rho$. (b) Estimate mutual information $\hat{I}(X;Y)$ via binned histogram. (c) Explain why $\rho \approx 0$ but $\hat{I}(X;Y) \gg 0$.

**Exercise 5 ★★ — Covariance Matrix Properties**
Let $\mathbf{X} \sim \mathcal{N}(\mathbf{0}, \Sigma)$ with $\Sigma = \begin{pmatrix} 4 & 2 \\ 2 & 3 \end{pmatrix}$.

(a) Verify $\Sigma$ is PSD by computing eigenvalues. (b) Compute the Cholesky factor $L$ and sample from $\mathbf{X}$ via $\mathbf{X} = L\boldsymbol{\varepsilon}$. (c) Compute $\rho(X_1, X_2)$. (d) Compute the distribution of $Y = X_1 + X_2$ using the affine transformation property.

**Exercise 6 ★★ — MVN Conditional Distribution**
Let $\mathbf{X} = (X_1, X_2, X_3)^\top \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with $\boldsymbol{\mu} = (1, 2, 3)^\top$ and $\Sigma = \begin{pmatrix} 4 & 2 & 0 \\ 2 & 3 & 1 \\ 0 & 1 & 2 \end{pmatrix}$.

(a) Compute the conditional distribution $(X_1, X_2) \mid X_3 = 3.5$ via the Schur complement formula. (b) Verify numerically by sampling $10^4$ draws, filtering to $X_3 \in [3.4, 3.6]$, and computing empirical conditional moments.

**Exercise 7 ★★★ — Reparameterisation Trick**
(a) Show that if $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$ and $\mathbf{z} = \boldsymbol{\mu} + L\boldsymbol{\varepsilon}$, then $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, LL^\top)$. (b) For $\boldsymbol{\mu} = [1, -1]$ and $\Sigma = \begin{pmatrix}2 & 1 \\ 1 & 2\end{pmatrix}$, sample $10^4$ points and verify empirical mean and covariance. (c) Compute $\partial \mathbf{z} / \partial \boldsymbol{\mu}$ and $\partial \mathbf{z} / \partial L$. Explain why gradients exist and why this is essential for VAE training.

**Exercise 8 ★★★ — Naive Bayes vs Logistic Regression**
(a) Implement Gaussian Naive Bayes for a 2-class, 2-feature problem. Fit via MLE. (b) Implement logistic regression via gradient descent. (c) Compare decision boundaries on data with independent features. (d) Repeat with strongly correlated features. Analyse which model performs better and why, connecting to the conditional independence assumption.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI/LLM Application |
|---|---|
| **Chain rule factorisation** | Every autoregressive model (GPT, LLaMA, Gemini, Claude) uses $\prod_t p(x_t \mid x_{<t})$ — the exact chain rule identity. |
| **Conditional distributions** | The core output of every neural network: classifier outputs $p(y \mid \mathbf{x})$; LLMs output $p(x_t \mid x_{<t})$; diffusion models output $p(\mathbf{x}_{t-1} \mid \mathbf{x}_t)$. |
| **MVN conditionals (Schur complement)** | Gaussian process regression (active in Bayesian optimisation for hyperparameter search), Kalman filtering (robot navigation, sensor fusion). |
| **Reparameterisation trick** | VAEs (Kingma & Welling, 2013), continuous normalising flows, diffusion model variational bounds — all require differentiable sampling from Gaussian distributions. |
| **Conditional independence** | Naive Bayes (competitive for text classification), mean-field VI (VAE diagonal covariance), causal masking in attention (the AR factorisation independence assumption). |
| **Covariance matrix / PSD** | Adam, Shampoo, K-FAC (second-order optimisers) approximate or use the parameter covariance. BatchNorm/LayerNorm whiten activations using empirical covariance. |
| **Berkson's paradox (collider bias)** | Selection bias in training data (models trained on filtered web data), benchmark contamination, shortcut learning (Geirhos et al., 2020). |
| **Copulas** | Modelling tail dependence in financial risk; structured dependence between multi-modal outputs; beyond-correlation dependence modelling. |
| **Normalising flows / Jacobians** | RealNVP, Glow, Neural ODEs — generative models using exact change-of-variables for tractable log-likelihoods and invertible generation. |
| **GMMs** | Speaker diarisation, VQ-VAE codebook (discrete latents via k-means on Gaussian clusters), mechanistic interpretability (superposition hypothesis). |
| **Maximum entropy MVN** | Justifies Gaussian weight initialisation (Xavier/Kaiming), Gaussian latent priors in VAEs, Laplace approximation to Bayesian neural network posteriors. |

---

## 12. Conceptual Bridge

This section sits at the heart of probabilistic machine learning. We arrived here from the foundations of single-variable probability (§6.1) and named distributions (§6.2), and we have now built the full machinery for reasoning about multiple variables simultaneously. The chain rule of probability — a simple product of conditional distributions — is not merely an abstract identity: it is the *operating principle* of every language model in existence.

The multivariate Gaussian is the cornerstone of continuous multivariate modelling. Its remarkable closure properties (marginals Gaussian, conditionals Gaussian, affine transforms Gaussian) make it analytically tractable in a way no other distribution is. The Schur complement formula for conditional distributions appears in Gaussian process regression, Bayesian linear regression, the Kalman filter, and Gaussian belief propagation — it is one of the most frequently used results in applied probabilistic modelling.

Looking forward, the next section [§6.4 Expectation and Variance](../04-Expectation-and-Moments/notes.md) develops the theory of moments in depth — deriving $\mathbb{E}[X]$, $\text{Var}(X)$, and higher moments using LOTUS, the law of total expectation, and the law of total variance. Many quantities referenced here (covariance as $\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$, MVN conditional mean) will be derived properly there. Section [§6.5 Concentration Inequalities](../05-Concentration-Inequalities/notes.md) develops tail bounds that require the joint distribution structure built here. Section [§6.6 Stochastic Processes](../06-Stochastic-Processes/notes.md) proves the Central Limit Theorem via the convolution of joint distributions.

```
POSITION IN THE PROBABILITY THEORY CHAPTER
════════════════════════════════════════════════════════════════════════

  §6.1 Introduction         §6.2 Distributions      §6.3 Joint [HERE]
  ────────────────          ──────────────────       ────────────────
  Axioms, CDF, PMF,    →    Bernoulli, Gaussian, →   Joint PMF/PDF,
  PDF, single RVs           Gamma, Beta, ...         Marginals, Conditionals,
                            Exp. family               Chain rule, MVN,
                                                      Transformations

        ↓                         ↓                         ↓
  §6.4 Expectation          §6.5 Concentration       §6.6 CLT/LLN
  ─────────────────         ──────────────────       ────────────────
  E[X], Var(X), LOTUS,      Markov, Chebyshev,       LLN, CLT proof
  Total expectation/var     Hoeffding, PAC bounds     via joint dist.

                                  ↓
                      §6.7 Markov Chains
                      ──────────────────
                      Transition matrices,
                      Steady state, MCMC

════════════════════════════════════════════════════════════════════════
```

The ideas in this section — joint distributions, conditional independence, the multivariate Gaussian, the chain rule — are not just mathematical tools. They are the conceptual vocabulary of modern probabilistic AI: the language in which generative models, Bayesian inference, and information-theoretic learning are expressed. Every time a GPT model generates a token, every time a VAE encodes an image, every time a diffusion model denoises — the machinery is joint distributions.

---

[← Back to Probability Theory](../README.md) | [Next: Expectation and Variance →](../04-Expectation-and-Moments/notes.md)

---

## Appendix A: Worked Calculations — Bivariate Gaussian

This appendix provides complete, step-by-step calculations for the bivariate Gaussian that are essential for building intuition.

### A.1 Deriving the Marginals from the Joint

The bivariate normal with zero means and correlation $\rho$ has joint density:
$$f(x,y) = \frac{1}{2\pi\sqrt{1-\rho^2}} \exp\!\left(-\frac{x^2 - 2\rho xy + y^2}{2(1-\rho^2)}\right)$$

To find $f_X(x)$, integrate over $y$:
$$f_X(x) = \int_{-\infty}^\infty f(x,y)\,dy = \frac{1}{2\pi\sqrt{1-\rho^2}} \int_{-\infty}^\infty \exp\!\left(-\frac{x^2 - 2\rho xy + y^2}{2(1-\rho^2)}\right)dy$$

Complete the square in $y$. The exponent $-(x^2 - 2\rho xy + y^2)/(2(1-\rho^2))$ can be rewritten:
$$y^2 - 2\rho xy + x^2 = (y - \rho x)^2 + x^2(1-\rho^2)$$

So the exponent becomes:
$$-\frac{(y-\rho x)^2}{2(1-\rho^2)} - \frac{x^2}{2}$$

Pulling the $x$-dependent factor outside:
$$f_X(x) = \frac{e^{-x^2/2}}{2\pi\sqrt{1-\rho^2}} \int_{-\infty}^\infty \exp\!\left(-\frac{(y-\rho x)^2}{2(1-\rho^2)}\right)dy$$

The integral is $\sqrt{2\pi(1-\rho^2)}$ (Gaussian integral), so:
$$f_X(x) = \frac{e^{-x^2/2}}{2\pi\sqrt{1-\rho^2}} \cdot \sqrt{2\pi(1-\rho^2)} = \frac{e^{-x^2/2}}{\sqrt{2\pi}} = \phi(x)$$

Thus $X \sim \mathcal{N}(0,1)$, confirming the marginal is standard normal regardless of $\rho$. By symmetry, $Y \sim \mathcal{N}(0,1)$ as well.

### A.2 Deriving the Conditional from the Joint

Dividing $f(x,y)$ by $f_X(x)$:
$$f_{Y \mid X}(y \mid x) = \frac{f(x,y)}{f_X(x)} = \frac{1}{\sqrt{2\pi(1-\rho^2)}} \exp\!\left(-\frac{(y-\rho x)^2}{2(1-\rho^2)}\right)$$

This is $\mathcal{N}(\rho x, 1-\rho^2)$ — confirming the conditional mean $\mathbb{E}[Y \mid X=x] = \rho x$ and conditional variance $\text{Var}(Y \mid X=x) = 1 - \rho^2$.

**Interpretation of the conditional mean:** The slope of the regression line $\mathbb{E}[Y \mid X=x] = \rho x$ equals the correlation coefficient when both variables are standardised. For unstandardised $(X,Y)$ with means $(\mu_X, \mu_Y)$ and standard deviations $(\sigma_X, \sigma_Y)$:
$$\mathbb{E}[Y \mid X=x] = \mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X)$$

The regression coefficient $\beta = \rho \sigma_Y / \sigma_X$ is the best linear predictor of $Y$ from $X$.

### A.3 Independence Condition for Bivariate Normal

$X$ and $Y$ are independent in the bivariate normal iff $\rho = 0$.

**Proof:** If $\rho = 0$, then $f(x,y) = \frac{1}{2\pi} e^{-(x^2+y^2)/2} = f_X(x) f_Y(y)$ — direct factorisation. Conversely, if $f(x,y) = f_X(x)f_Y(y)$ for all $x,y$, then $\text{Cov}(X,Y) = 0$, which forces $\rho = 0$. ∎

This is a special property of the multivariate Gaussian: **for MVN, uncorrelated implies independent**. This does NOT hold for general distributions.

---

## Appendix B: The Multivariate Gaussian — Derivation of the Schur Complement Formula

We derive the conditional distribution $\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2$ from first principles.

**Setup:** $\mathbf{X} = (\mathbf{X}_1^\top, \mathbf{X}_2^\top)^\top \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with the block partition as in Section 6.4.

**Strategy:** Construct a new variable $\mathbf{Z} = \mathbf{X}_1 - \Sigma_{12}\Sigma_{22}^{-1}\mathbf{X}_2$ that is independent of $\mathbf{X}_2$.

**Step 1:** Compute $\text{Cov}(\mathbf{Z}, \mathbf{X}_2)$:
$$\text{Cov}(\mathbf{Z}, \mathbf{X}_2) = \text{Cov}(\mathbf{X}_1, \mathbf{X}_2) - \Sigma_{12}\Sigma_{22}^{-1}\text{Cov}(\mathbf{X}_2, \mathbf{X}_2) = \Sigma_{12} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{22} = 0$$

**Step 2:** Since $(\mathbf{Z}, \mathbf{X}_2)$ is jointly Gaussian (linear transformation) with zero cross-covariance, they are independent.

**Step 3:** Since $\mathbf{Z} \perp\!\!\!\perp \mathbf{X}_2$, conditioning on $\mathbf{X}_2 = \mathbf{x}_2$ does not change the distribution of $\mathbf{Z}$:
$$\mathbf{Z} \mid \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\mathbb{E}[\mathbf{Z}], \text{Cov}(\mathbf{Z}))$$

where $\mathbb{E}[\mathbf{Z}] = \boldsymbol{\mu}_1 - \Sigma_{12}\Sigma_{22}^{-1}\boldsymbol{\mu}_2$ and $\text{Cov}(\mathbf{Z}) = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$.

**Step 4:** Since $\mathbf{X}_1 = \mathbf{Z} + \Sigma_{12}\Sigma_{22}^{-1}\mathbf{X}_2$, conditioning on $\mathbf{X}_2 = \mathbf{x}_2$:
$$\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}\!\left(\boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2), \; \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}\right)$$

This completes the derivation. ∎

**Key insight:** The Schur complement $\Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$ is the covariance of the **residual** $\mathbf{X}_1 - \hat{\mathbf{X}}_1$ where $\hat{\mathbf{X}}_1 = \boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{X}_2 - \boldsymbol{\mu}_2)$ is the best linear predictor (BLUP) of $\mathbf{X}_1$ from $\mathbf{X}_2$.

---

## Appendix C: Independence Tests in Practice

Given $n$ samples from $(X, Y)$, how do we test independence?

### C.1 Pearson's Chi-Squared Test (Discrete)

For discrete variables $X$ and $Y$, form the contingency table with observed counts $O_{ij}$ (number of samples with $X=i, Y=j$). Under independence, expected counts are:
$$E_{ij} = \frac{(\text{row total}_i)(\text{col total}_j)}{n}$$

The test statistic:
$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \overset{H_0}{\sim} \chi^2((r-1)(c-1))$$

where $r, c$ are the numbers of rows and columns. Reject independence at level $\alpha$ if $\chi^2 > \chi^2_{(r-1)(c-1), 1-\alpha}$.

### C.2 Pearson Correlation Test (Continuous, Linear)

For continuous $(X,Y)$, test $H_0: \rho = 0$ vs $H_1: \rho \neq 0$. The test statistic:
$$t = \frac{\hat{\rho}\sqrt{n-2}}{\sqrt{1-\hat{\rho}^2}} \overset{H_0}{\sim} t_{n-2}$$

Limitations: Only detects *linear* dependence. Fails for $Y = X^2$ (zero correlation, strong dependence).

### C.3 Kolmogorov-Smirnov Test for Independence

Compare the empirical joint CDF to the product of empirical marginals. The test statistic:
$$D_n = \sup_{x,y} |F_n(x,y) - F_X^n(x) F_Y^n(y)|$$

Under independence, $D_n \overset{d}{\to} 0$ at rate $1/\sqrt{n}$.

### C.4 Mutual Information Test

Estimate $\hat{I}(X;Y)$ via binned histograms or kernel density estimation. Under independence, $2n\hat{I}(X;Y) \overset{d}{\to} \chi^2_{(r-1)(c-1)}$. More powerful than $\chi^2$ for nonlinear dependence.

**For AI model diagnostics:** Testing whether two layers' activations are conditionally independent given the input is a key step in mechanistic interpretability. High mutual information between distant layers suggests information bypasses intermediate layers — a sign of skip connections or residual pathways.

---

## Appendix D: Multivariate Change of Variables — Additional Examples

### D.1 Bivariate Transformation: Sum and Difference

Let $X, Y$ be independent $\mathcal{N}(0,1)$. Define $S = X + Y$, $D = X - Y$.

**Jacobian calculation:**
$$\frac{\partial(x,y)}{\partial(s,d)} = \begin{vmatrix} \partial x/\partial s & \partial x/\partial d \\ \partial y/\partial s & \partial y/\partial d \end{vmatrix} = \begin{vmatrix} 1/2 & 1/2 \\ 1/2 & -1/2 \end{vmatrix} = -1/2$$

So $|\det J| = 1/2$ and:
$$f_{S,D}(s,d) = f_{X,Y}\!\left(\frac{s+d}{2}, \frac{s-d}{2}\right) \cdot \frac{1}{2} = \frac{1}{2\pi} e^{-(s^2+d^2)/4} \cdot \frac{1}{2} = \frac{1}{4\pi} e^{-(s^2+d^2)/4}$$

This factors as $f_S(s) f_D(d)$ where $S, D \sim \mathcal{N}(0,2)$ independently.

**Takeaway:** The sum and difference of independent standard normals are themselves independent normals — a non-obvious result following directly from the change of variables.

### D.2 Exponential to Gamma via Convolution

For $X_1, \ldots, X_n \overset{\text{iid}}{\sim} \text{Exp}(\lambda)$, the sum $S_n = X_1 + \cdots + X_n \sim \text{Gamma}(n, \lambda)$.

**Proof by MGF:** $M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t) = \left(\frac{\lambda}{\lambda-t}\right)^n$ which is the MGF of $\text{Gamma}(n,\lambda)$.

**Connection to Poisson process:** The time until the $n$-th event in a Poisson$(\lambda)$ process is exactly $\text{Gamma}(n, \lambda)$ — a direct consequence of inter-arrival times being iid Exp$(\lambda)$.

---

## Appendix E: Copulas — Detailed Treatment

### E.1 Gaussian Copula Construction

The Gaussian copula with correlation matrix $R$ is constructed as follows:

1. Generate $\mathbf{z} = (Z_1, \ldots, Z_d)^\top \sim \mathcal{N}(\mathbf{0}, R)$
2. Apply the probability integral transform: $U_i = \Phi(Z_i)$ where $\Phi$ is the standard normal CDF
3. The resulting $(U_1, \ldots, U_d)$ has the Gaussian copula with correlation $R$

To generate samples from an arbitrary distribution with Gaussian copula:
4. Invert the marginal CDFs: $X_i = F_i^{-1}(U_i)$

**Properties:**
- Captures linear (Pearson) correlation at the copula level
- Tail dependence: the Gaussian copula has zero tail dependence — extreme events in different variables are asymptotically independent even when there is correlation in the bulk

**The Global Financial Crisis (2008):** The widespread use of the Gaussian copula for CDO pricing was criticised as a key model failure. The Gaussian copula underestimates joint tail events — the probability that many mortgages default simultaneously during a systemic crisis. The correct model should use a copula with positive tail dependence (e.g., t-copula or Clayton copula).

### E.2 Measuring Dependence via Copulas

**Kendall's tau:** A rank-based correlation measure:
$$\tau = P[(X_1-X_2)(Y_1-Y_2) > 0] - P[(X_1-X_2)(Y_1-Y_2) < 0]$$

Unlike Pearson $\rho$, Kendall's $\tau$ is invariant to monotone transformations of $X$ and $Y$ — it is purely a property of the copula.

**Spearman's rho:** $\rho_S = \rho(\text{rank}(X), \text{rank}(Y))$ — also invariant to monotone marginal transformations.

For the Gaussian copula with linear correlation $\rho$: $\tau = (2/\pi)\arcsin(\rho)$ and $\rho_S = (6/\pi)\arcsin(\rho/2)$.

---

## Appendix F: Gaussian Mixture Models — EM Algorithm Details

### F.1 Complete Data Log-Likelihood

If we observed both $\mathbf{x}_i$ and latent assignments $z_i$, the complete data log-likelihood is:
$$\ell_c(\theta) = \sum_{i=1}^n \sum_{k=1}^K \mathbf{1}[z_i=k] \left[\log\pi_k + \log\mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \Sigma_k)\right]$$

This is easy to maximise — just compute cluster-conditional MLEs.

### F.2 E-Step: Soft Assignments

Since $z_i$ is latent, replace $\mathbf{1}[z_i=k]$ by its posterior expectation:
$$r_{ik} = \mathbb{E}[\mathbf{1}[z_i=k] \mid \mathbf{x}_i, \theta] = P(z_i=k \mid \mathbf{x}_i) = \frac{\pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_j, \Sigma_j)}$$

This is Bayes' theorem applied to the joint distribution over $(z_i, \mathbf{x}_i)$.

### F.3 M-Step: Update Parameters

Maximise the expected complete data log-likelihood:
$$\pi_k^{\text{new}} = \frac{\sum_i r_{ik}}{n}, \quad \boldsymbol{\mu}_k^{\text{new}} = \frac{\sum_i r_{ik} \mathbf{x}_i}{\sum_i r_{ik}}, \quad \Sigma_k^{\text{new}} = \frac{\sum_i r_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})(\mathbf{x}_i - \boldsymbol{\mu}_k^{\text{new}})^\top}{\sum_i r_{ik}}$$

Each M-step is a weighted MLE — the soft assignments $r_{ik}$ play the role of fractional cluster memberships.

### F.4 Convergence

The EM algorithm monotonically increases the marginal log-likelihood $\ell(\theta) = \sum_i \log \sum_k \pi_k \mathcal{N}(\mathbf{x}_i; \boldsymbol{\mu}_k, \Sigma_k)$ at each iteration. It converges to a local maximum (not necessarily global). In practice:
- Random restarts help escape poor local optima
- K-means++ initialisation provides a good starting point
- The BIC/AIC criteria are used to select the number of components $K$

---

## Appendix G: The Reparameterisation Trick — Formal Derivation

### G.1 The Problem

To train a VAE, we need:
$$\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})] = \nabla_\phi \int q_\phi(\mathbf{z} \mid \mathbf{x}) f(\mathbf{z}) \, d\mathbf{z}$$

The naive approach — swapping gradient and expectation — fails because the distribution $q_\phi$ depends on $\phi$:
$$\int \nabla_\phi q_\phi(\mathbf{z} \mid \mathbf{x}) f(\mathbf{z}) \, d\mathbf{z} \neq \mathbb{E}_{q_\phi}[\nabla_\phi f(\mathbf{z})]$$

### G.2 The Reparameterisation

Write $\mathbf{z} = g_\phi(\boldsymbol{\varepsilon}, \mathbf{x})$ where $\boldsymbol{\varepsilon} \sim p(\boldsymbol{\varepsilon})$ does not depend on $\phi$. For the diagonal Gaussian encoder:
$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

Then:
$$\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})] = \mathbb{E}_{p(\boldsymbol{\varepsilon})}[f(g_\phi(\boldsymbol{\varepsilon}, \mathbf{x}))]$$

Now the expectation is over $p(\boldsymbol{\varepsilon})$ which does NOT depend on $\phi$, so:
$$\nabla_\phi \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[f(\mathbf{z})] = \mathbb{E}_{p(\boldsymbol{\varepsilon})}\!\left[\nabla_\phi f(g_\phi(\boldsymbol{\varepsilon}, \mathbf{x}))\right]$$

This gradient can be estimated by Monte Carlo: sample $\boldsymbol{\varepsilon}^{(1)}, \ldots, \boldsymbol{\varepsilon}^{(L)}$ and compute $\frac{1}{L}\sum_l \nabla_\phi f(g_\phi(\boldsymbol{\varepsilon}^{(l)}, \mathbf{x}))$. In practice, $L=1$ often suffices.

### G.3 Why the Affine Transformation Property is Key

The reparameterisation $\mathbf{z} = \boldsymbol{\mu} + L\boldsymbol{\varepsilon}$ works because:
1. $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, LL^\top)$ by the affine transformation property (Section 6.5)
2. The Jacobian of the transformation is $L$ — invertible and differentiable
3. Gradients flow through $\boldsymbol{\mu}$ and $L$ as if they were deterministic variables

This trick generalises beyond Gaussians: any distribution expressible as a deterministic transformation of a fixed noise distribution admits reparameterisation. The Gumbel-softmax trick (Jang et al., 2017) extends this to Categorical distributions.

---

## Appendix H: Common Distributions as Joint Distributions

Many standard distributions arise naturally as marginals of joint distributions over multiple variables.

### H.1 Chi-Squared as a Sum of Squared Normals

If $Z_1, \ldots, Z_k \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$, then $Q = \sum_{i=1}^k Z_i^2 \sim \chi^2(k)$.

**Proof via MGF:** $M_Q(t) = \prod_{i=1}^k M_{Z_i^2}(t)$. For $Z^2$ where $Z \sim \mathcal{N}(0,1)$:
$$M_{Z^2}(t) = \mathbb{E}[e^{tZ^2}] = \frac{1}{\sqrt{1-2t}}, \quad t < 1/2$$

So $M_Q(t) = (1-2t)^{-k/2}$ — the MGF of $\text{Gamma}(k/2, 1/2) = \chi^2(k)$.

**For AI:** The chi-squared distribution appears in:
- Wald tests for neural network weights (are weights significantly different from zero?)
- Goodness-of-fit tests for probability distributions learned by generative models
- The test for the $\chi^2$ independence test (Appendix C)

### H.2 Student-t as Ratio of Normal to Chi-Squared

If $Z \sim \mathcal{N}(0,1)$ and $V \sim \chi^2(\nu)$ independently, then:
$$T = \frac{Z}{\sqrt{V/\nu}} \sim t_\nu$$

**Intuition:** The Student-t arises when we estimate the variance from data (replacing the known $\sigma^2$ with the sample variance $\hat{s}^2$). The heavier tails reflect additional uncertainty from estimating $\sigma^2$.

**For AI:** The t-SNE visualisation algorithm uses the Student-t distribution (with $\nu=1$, i.e., Cauchy) in the low-dimensional space to model cluster structure. The heavier tails push apart nearby clusters more aggressively than a Gaussian, creating cleaner cluster separation in 2D.

### H.3 F-Distribution as Ratio of Chi-Squareds

If $V_1 \sim \chi^2(d_1)$ and $V_2 \sim \chi^2(d_2)$ independently, then:
$$F = \frac{V_1/d_1}{V_2/d_2} \sim F(d_1, d_2)$$

Used in ANOVA and comparing variances between groups.

---

## Appendix I: Expectation and Covariance — Key Identities

This appendix collects identities used throughout the section. Full derivations are in [§6.4](../04-Expectation-and-Moments/notes.md).

### I.1 Law of Total Expectation

$$\mathbb{E}[X] = \mathbb{E}_Y[\mathbb{E}[X \mid Y]]$$

**Application:** $\mathbb{E}[\mathbf{X}] = \mathbb{E}_Z[\mathbb{E}[\mathbf{X} \mid Z]] = \sum_k P(Z=k) \boldsymbol{\mu}_k$ for a GMM — the overall mean is the mixture-weighted average of component means.

### I.2 Law of Total Variance

$$\text{Var}(X) = \mathbb{E}[\text{Var}(X \mid Y)] + \text{Var}(\mathbb{E}[X \mid Y])$$

**Decomposition:** Total variance = average within-group variance + between-group variance (from ANOVA). For a GMM: $\text{Var}(\mathbf{X}) = \sum_k \pi_k \Sigma_k + \sum_k \pi_k (\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^\top$ where $\boldsymbol{\mu} = \sum_k \pi_k \boldsymbol{\mu}_k$.

### I.3 Bilinearity of Covariance for Vectors

For random vectors $\mathbf{X}$ and $\mathbf{Y}$ and matrices $A, B$:
$$\text{Cov}(A\mathbf{X}, B\mathbf{Y}) = A \, \text{Cov}(\mathbf{X}, \mathbf{Y}) \, B^\top$$

**Special case:** $\text{Cov}(A\mathbf{X}) = A \Sigma_X A^\top$ — the covariance under a linear transformation.

### I.4 Independence and Zero Covariance for MVN

For jointly Gaussian random variables: $\text{Cov}(X_i, X_j) = 0 \Leftrightarrow X_i \perp\!\!\!\perp X_j$.

This is unique to the Gaussian — in general, zero covariance is weaker than independence.

---

## Appendix J: Graphical Models — Overview

A **Bayesian network** (directed graphical model) is a directed acyclic graph (DAG) where:
- Nodes = random variables
- Edges = direct dependencies
- Each node $X_i$ has a conditional distribution $p(X_i \mid \text{parents}(X_i))$

The joint distribution factorises as:
$$p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i \mid \text{pa}(x_i))$$

This is the chain rule applied with the graphical structure choosing which conditioning variables appear in each factor.

**Conditional independence from the graph:** $X \perp\!\!\!\perp Y \mid Z$ in the joint distribution iff $X$ and $Y$ are **d-separated** by $Z$ in the DAG — all paths from $X$ to $Y$ are "blocked" by $Z$.

```
COMMON GRAPHICAL MODEL PATTERNS
════════════════════════════════════════════════════════════════════════

  Naive Bayes:              LDA:                    Hidden Markov Model:
      Y                     θ   α                   z₁ → z₂ → z₃
    ↙↓↘                   ↙↓↘    ↘                  ↓    ↓    ↓
  X₁ X₂ X₃             w₁ w₂ w₃  β                x₁   x₂   x₃
  (features cond.       (words given  (topics)       (obs given state)
   indep. given Y)       topic dist.)

  VAE:                      Kalman Filter:
   p(z)                     z_t-1 → z_t → z_t+1
     ↓                        ↓        ↓      ↓
   p(x|z)                    x_t-1   x_t   x_t+1
   (encoder   (decoder)       (hidden state dynamics)
    q(z|x) ↑)

════════════════════════════════════════════════════════════════════════
```

**For AI:** Modern large language models can be understood as extremely deep Bayesian networks where each token's probability is conditioned on all previous tokens. The attention pattern (which tokens attend to which) defines an implicit graphical structure that changes with each input.

---

## Appendix K: Numerical Stability in MVN Computations

### K.1 Cholesky Decomposition for Sampling

Never invert $\Sigma$ directly for sampling. Instead:
1. Compute the Cholesky factor $L$ where $\Sigma = LL^\top$ (stable, $O(n^3)$)
2. Sample $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$
3. Return $\mathbf{x} = \boldsymbol{\mu} + L\boldsymbol{\varepsilon}$

Direct inversion $\Sigma^{-1}$ amplifies numerical errors; Cholesky is numerically stable.

### K.2 Log-Determinant for Density Evaluation

Computing $\log|\Sigma|$ naively via `log(det(Sigma))` overflows for large $n$. Instead:
1. Compute Cholesky $\Sigma = LL^\top$
2. $\log|\Sigma| = 2\sum_i \log L_{ii}$ (sum of log-diagonal elements of $L$)

This is numerically stable because $L_{ii} > 0$ and we sum logs rather than multiply small numbers.

### K.3 Mahalanobis Distance

The Mahalanobis distance $d^2 = (\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})$ should be computed as:
1. Solve $L\mathbf{v} = (\mathbf{x} - \boldsymbol{\mu})$ via forward substitution
2. $d^2 = \|\mathbf{v}\|^2$

This avoids forming $\Sigma^{-1}$ explicitly.

### K.4 Conditioning on Nearly Degenerate $\Sigma_{22}$

When $\Sigma_{22}$ is nearly singular (nearly degenerate observations), add a small diagonal regulariser: $\Sigma_{22} + \epsilon I$ for $\epsilon \sim 10^{-6}$. This is **jitter** in Gaussian process literature. The resulting conditional covariance is $\Sigma_{11} - \Sigma_{12}(\Sigma_{22} + \epsilon I)^{-1}\Sigma_{21}$, which remains PSD.

---

## Appendix L: Summary Tables

### L.1 Key Joint Distribution Formulas

| Quantity | Discrete | Continuous |
|---|---|---|
| Joint | $p(x,y)$ | $f(x,y)$ |
| Marginal of $X$ | $\sum_y p(x,y)$ | $\int f(x,y)\,dy$ |
| Conditional $Y \mid X=x$ | $p(x,y)/p_X(x)$ | $f(x,y)/f_X(x)$ |
| Independence condition | $p(x,y) = p_X(x)p_Y(y)$ | $f(x,y) = f_X(x)f_Y(y)$ |
| Cov$(X,Y)$ | $\mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ | Same |
| Bayes | $p(x\mid y) \propto p(y\mid x)p_X(x)$ | Same |

### L.2 Multivariate Gaussian Cheat Sheet

| Property | Formula |
|---|---|
| Density | $(2\pi)^{-n/2}\|\Sigma\|^{-1/2}\exp(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}))$ |
| Marginal $\mathbf{X}_1$ | $\mathcal{N}(\boldsymbol{\mu}_1, \Sigma_{11})$ |
| Conditional $\mathbf{X}_1 \mid \mathbf{X}_2$ | $\mathcal{N}(\boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2-\boldsymbol{\mu}_2), \Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21})$ |
| Affine transform $A\mathbf{X}+\mathbf{b}$ | $\mathcal{N}(A\boldsymbol{\mu}+\mathbf{b}, A\Sigma A^\top)$ |
| Entropy | $\frac{1}{2}\log((2\pi e)^n \|\Sigma\|)$ |
| KL$(p\|q)$ for Gaussians | $\frac{1}{2}[\text{tr}(\Sigma_q^{-1}\Sigma_p) + (\boldsymbol{\mu}_q-\boldsymbol{\mu}_p)^\top\Sigma_q^{-1}(\boldsymbol{\mu}_q-\boldsymbol{\mu}_p) - n + \log\frac{\|\Sigma_q\|}{\|\Sigma_p\|}]$ |

### L.3 Independence: A Hierarchy

```
Mutual independence
       ↓ (implies)
Pairwise independence
       ↓ (implies)
Zero covariance (for general distributions)
       ↑↓ (equivalent for MVN only)
Zero covariance
```

For the multivariate Gaussian, uncorrelated = pairwise independent = mutually independent (if the full vector is jointly Gaussian).


---

## Appendix M: Detailed Exercises with Full Solutions

### M.1 Computing the Bivariate Normal Conditional — Full Worked Example

**Problem:** $(X,Y) \sim \mathcal{N}\!\left(\begin{pmatrix}1\\3\end{pmatrix}, \begin{pmatrix}4 & 2\\ 2 & 9\end{pmatrix}\right)$. Find $Y \mid X = 2$.

**Step 1:** Identify parameters. $\mu_X = 1$, $\mu_Y = 3$, $\sigma_X^2 = 4$, $\sigma_Y^2 = 9$, $\sigma_{XY} = 2$.

**Step 2:** Apply the conditional formula.
$$\mu_{Y|X=2} = \mu_Y + \sigma_{XY}\sigma_X^{-2}(x - \mu_X) = 3 + \frac{2}{4}(2 - 1) = 3.5$$

$$\sigma^2_{Y|X} = \sigma_Y^2 - \sigma_{XY}^2\sigma_X^{-2} = 9 - \frac{4}{4} = 8$$

**Answer:** $Y \mid X = 2 \sim \mathcal{N}(3.5, 8)$.

**Verification:** The conditional variance 8 < 9 = marginal variance — conditioning reduced uncertainty, as expected.

### M.2 Independence Test via Joint Table — Full Example

**Problem:** Given the joint PMF table, test independence.

| | $Y=0$ | $Y=1$ | $p_X$ |
|---|---|---|---|
| $X=0$ | 0.2 | 0.3 | 0.5 |
| $X=1$ | 0.4 | 0.1 | 0.5 |
| $p_Y$ | 0.6 | 0.4 | 1.0 |

**Independence check:** For independence, we need $p(x,y) = p_X(x) \cdot p_Y(y)$ for all $(x,y)$.

Check $(X=0, Y=0)$: $p_X(0) \cdot p_Y(0) = 0.5 \times 0.6 = 0.30 \neq 0.20$. **FAIL.**

The variables are NOT independent. In fact:
$$p(Y=0 \mid X=0) = 0.2/0.5 = 0.4 \neq 0.6 = p(Y=0)$$

Knowing $X=0$ decreases our probability estimate for $Y=0$ — the variables are negatively correlated.

**Covariance calculation:**
$$\mathbb{E}[XY] = 0 \cdot 0 \cdot 0.2 + 0 \cdot 1 \cdot 0.3 + 1 \cdot 0 \cdot 0.4 + 1 \cdot 1 \cdot 0.1 = 0.1$$
$$\mathbb{E}[X] = 0.5, \quad \mathbb{E}[Y] = 0.4$$
$$\text{Cov}(X,Y) = 0.1 - 0.5 \times 0.4 = -0.1 < 0$$

Negative covariance confirms negative association.

### M.3 Change of Variables — Log-Normal Full Derivation

**Problem:** $X \sim \mathcal{N}(\mu, \sigma^2)$. Find the density of $Y = e^X$.

**Step 1:** Find the inverse: $x = g^{-1}(y) = \log y$ for $y > 0$.

**Step 2:** Compute the Jacobian: $\frac{dx}{dy} = \frac{1}{y}$.

**Step 3:** Apply the formula:
$$f_Y(y) = f_X(g^{-1}(y)) \cdot |g^{-1}'(y)| = \frac{1}{\sigma\sqrt{2\pi}} e^{-(\log y - \mu)^2/(2\sigma^2)} \cdot \frac{1}{y}$$

**Moments of the log-normal:**
$$\mathbb{E}[Y] = e^{\mu + \sigma^2/2}, \quad \text{Var}(Y) = (e^{\sigma^2}-1)e^{2\mu+\sigma^2}$$

**For AI:** Model parameter magnitudes after training often follow approximately log-normal distributions. If $W \sim \text{Log-Normal}$, then $\log|W| \sim \mathcal{N}$ — transforming to log-scale makes analysis easier. Weight pruning thresholds are often set based on the log-normal quantiles.

---

## Appendix N: The Gaussian Process as a Joint Distribution

A **Gaussian process (GP)** is a joint distribution over function values at infinitely many input points — the continuous generalisation of the MVN.

**Definition:** $f \sim \mathcal{GP}(m(\cdot), k(\cdot, \cdot))$ if for any finite collection of inputs $\mathbf{x}_1, \ldots, \mathbf{x}_n$:
$$(f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)) \sim \mathcal{N}(\mathbf{m}, K)$$

where $m_i = m(\mathbf{x}_i)$ and $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is the kernel (covariance function).

**GP regression** uses the conditional MVN formula (Section 6.4) to make predictions. Given observations $\mathbf{y} = f(\mathbf{X}) + \boldsymbol{\varepsilon}$ with noise $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I)$, the predictive distribution at new points $\mathbf{X}_*$ is:

$$f_* \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)$$

where:
$$\boldsymbol{\mu}_* = K(\mathbf{X}_*, \mathbf{X})[K(\mathbf{X},\mathbf{X}) + \sigma^2 I]^{-1}\mathbf{y}$$
$$\Sigma_* = K(\mathbf{X}_*,\mathbf{X}_*) - K(\mathbf{X}_*,\mathbf{X})[K(\mathbf{X},\mathbf{X}) + \sigma^2 I]^{-1}K(\mathbf{X},\mathbf{X}_*)$$

This is exactly the Schur complement conditional formula applied to the GP's joint distribution over observed and unobserved function values.

**For AI:** GPs are used for Bayesian hyperparameter optimisation (BoTorch, GPyTorch) in training large models. The acquisition function (UCB, EI) is computed analytically from the GP's mean and variance — enabled by the closed-form MVN conditional.

---

## Appendix O: Information Theory and Joint Distributions

### O.1 Joint Entropy

For jointly distributed $(X,Y)$:
$$H(X,Y) = -\sum_{x,y} p(x,y) \log p(x,y) \quad \text{(discrete)}$$
$$= -\int f(x,y) \log f(x,y) \, dx\,dy \quad \text{(continuous)}$$

**Chain rule of entropy:** $H(X,Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)$

where $H(Y \mid X) = \mathbb{E}_X[H(Y \mid X=x)]$ is the **conditional entropy**.

### O.2 Mutual Information

The **mutual information** measures the information shared between $X$ and $Y$:
$$I(X;Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X,Y)$$

**Key properties:**
- $I(X;Y) \geq 0$ (non-negativity)
- $I(X;Y) = 0 \Leftrightarrow X \perp\!\!\!\perp Y$ (zero iff independent)
- $I(X;Y) = D_{\text{KL}}(p(X,Y) \| p(X)p(Y))$ — KL divergence from joint to product of marginals

**For AI:** Mutual information is used in:
- Feature selection: keep features with high $I(\text{feature}; \text{label})$
- Information bottleneck theory: compress representations to maximise $I(\mathbf{Z}; Y)$ while minimising $I(\mathbf{X}; \mathbf{Z})$
- Estimating how much a layer's representation retains information about the input vs the task

### O.3 KL Divergence Between Multivariate Gaussians

For $p = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$ and $q = \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)$:

$$D_{\text{KL}}(p \| q) = \frac{1}{2}\left[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma_2^{-1}(\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - n + \log\frac{|\Sigma_2|}{|\Sigma_1|}\right]$$

**VAE special case** ($p = q_\phi(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$, $q = \mathcal{N}(\mathbf{0},I)$):
$$D_{\text{KL}}(p \| q) = \frac{1}{2}\sum_d (\mu_d^2 + \sigma_d^2 - \log\sigma_d^2 - 1)$$

---

## Appendix P: Notation Reference for This Section

| Symbol | Meaning |
|---|---|
| $p_{X,Y}(x,y)$ | Joint PMF of discrete $(X,Y)$ |
| $f_{X,Y}(x,y)$ | Joint PDF of continuous $(X,Y)$ |
| $F_{X,Y}(x,y)$ | Joint CDF |
| $p_X(x)$, $f_X(x)$ | Marginal PMF/PDF of $X$ |
| $p_{Y\mid X}(y\mid x)$ | Conditional PMF of $Y$ given $X=x$ |
| $X \perp\!\!\!\perp Y$ | $X$ and $Y$ are (marginally) independent |
| $X \perp\!\!\!\perp Y \mid Z$ | $X$ and $Y$ are conditionally independent given $Z$ |
| $\boldsymbol{\mu}$ | Mean vector (bold lowercase) |
| $\Sigma$ | Covariance matrix (uppercase sigma) |
| $\Lambda = \Sigma^{-1}$ | Precision matrix |
| $\Sigma_{12}$ | Off-diagonal block of covariance matrix |
| $\Sigma_{1\mid 2}$ | Conditional covariance (Schur complement) |
| $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ | Multivariate Gaussian |
| $\text{Cov}(X,Y)$ | Covariance of $X$ and $Y$ |
| $\rho(X,Y)$ | Pearson correlation coefficient |
| $I(X;Y)$ | Mutual information |
| $H(X,Y)$ | Joint entropy |
| $H(Y\mid X)$ | Conditional entropy |

_Last updated: 2026 — covers §6.3 Joint Distributions as scoped by the Chapter README._

---

## Appendix Q: Proofs of Core Independence Results

### Q.1 Proof that Independence Implies Zero Covariance

**Theorem:** If $X \perp\!\!\!\perp Y$, then $\text{Cov}(X,Y) = 0$.

**Proof:**
$$\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

When $X \perp\!\!\!\perp Y$, the joint density factorises: $f(x,y) = f_X(x)f_Y(y)$. Then:
$$\mathbb{E}[XY] = \int\int xy \cdot f_X(x)f_Y(y)\,dx\,dy = \left(\int x f_X(x)\,dx\right)\left(\int y f_Y(y)\,dy\right) = \mathbb{E}[X]\mathbb{E}[Y]$$

Therefore $\text{Cov}(X,Y) = \mathbb{E}[X]\mathbb{E}[Y] - \mathbb{E}[X]\mathbb{E}[Y] = 0$. ∎

**Why the converse fails:** Zero covariance says $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$ — a single numerical condition. Independence says $p(x,y) = p_X(x)p_Y(y)$ for ALL $(x,y)$ — infinitely many conditions. A single number cannot encode the information required for independence.

### Q.2 Proof that MVN Uncorrelated Implies Independent

**Theorem:** For jointly Gaussian $(X_1, \ldots, X_n) \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$: $\text{Cov}(X_i, X_j) = 0$ for all $i \neq j$ implies $(X_1, \ldots, X_n)$ are mutually independent.

**Proof:** If $\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_n^2)$ (all off-diagonal entries zero), then:
$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

With diagonal $\Sigma$: $\Sigma^{-1} = \text{diag}(1/\sigma_1^2, \ldots, 1/\sigma_n^2)$ and $|\Sigma| = \prod_i \sigma_i^2$. So:
$$f(\mathbf{x}) = \prod_{i=1}^n \frac{1}{\sigma_i\sqrt{2\pi}} e^{-(x_i-\mu_i)^2/(2\sigma_i^2)} = \prod_{i=1}^n f_{X_i}(x_i)$$

The joint density factorises as a product of marginals — mutual independence. ∎

### Q.3 Proof of the Cauchy-Schwarz Inequality for Correlation

**Theorem:** $|\text{Cov}(X,Y)|^2 \leq \text{Var}(X)\text{Var}(Y)$, implying $|\rho(X,Y)| \leq 1$.

**Proof:** For any $t \in \mathbb{R}$:
$$0 \leq \mathbb{E}[(X - \mathbb{E}[X] + t(Y - \mathbb{E}[Y]))^2] = \text{Var}(X) + 2t\text{Cov}(X,Y) + t^2\text{Var}(Y)$$

This is a non-negative quadratic in $t$, so its discriminant is non-positive:
$$4\text{Cov}(X,Y)^2 - 4\text{Var}(X)\text{Var}(Y) \leq 0$$

Therefore $\text{Cov}(X,Y)^2 \leq \text{Var}(X)\text{Var}(Y)$, i.e. $|\rho(X,Y)| \leq 1$. ∎

**Equality:** $|\rho| = 1$ iff $Y - \mathbb{E}[Y] = t(X - \mathbb{E}[X])$ almost surely — i.e., $Y$ is a linear function of $X$.

---

## Appendix R: Continuous Bayes — Integration Examples

### R.1 Beta-Bernoulli Conjugacy via Integration

**Setup:** Prior $p \sim \text{Beta}(\alpha, \beta)$, likelihood $X \mid p \sim \text{Bernoulli}(p)$.

**Joint distribution:**
$$f(p, x) = p^x(1-p)^{1-x} \cdot \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}$$

**Marginal of $X=1$:**
$$P(X=1) = \int_0^1 p \cdot \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}\,dp = \frac{B(\alpha+1,\beta)}{B(\alpha,\beta)} = \frac{\alpha}{\alpha+\beta}$$

This is the **prior predictive** probability — integrating out the unknown $p$.

**Posterior:** Applying Bayes' theorem for continuous $p$:
$$f(p \mid X=1) = \frac{p \cdot \frac{p^{\alpha-1}(1-p)^{\beta-1}}{B(\alpha,\beta)}}{P(X=1)} = \frac{p^\alpha (1-p)^{\beta-1}}{B(\alpha+1,\beta)}$$

This is $\text{Beta}(\alpha+1, \beta)$ — one success increments $\alpha$ by 1. ✓

### R.2 Continuous Bayes for Signal in Noise

**Setup:** Observe noisy signal $Y = \theta + \varepsilon$ where $\varepsilon \sim \mathcal{N}(0, \sigma^2)$ (known) and prior $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$.

**Likelihood:** $f(y \mid \theta) = \mathcal{N}(y; \theta, \sigma^2)$

**Posterior:**
$$f(\theta \mid y) \propto f(y \mid \theta) f(\theta) \propto \exp\!\left(-\frac{(y-\theta)^2}{2\sigma^2} - \frac{(\theta-\mu_0)^2}{2\sigma_0^2}\right)$$

Completing the square in $\theta$, the exponent becomes:
$$-\frac{1}{2}\left(\frac{1}{\sigma^2} + \frac{1}{\sigma_0^2}\right)\left(\theta - \frac{y/\sigma^2 + \mu_0/\sigma_0^2}{1/\sigma^2 + 1/\sigma_0^2}\right)^2$$

So the posterior is:
$$\theta \mid y \sim \mathcal{N}\!\left(\frac{y/\sigma^2 + \mu_0/\sigma_0^2}{1/\sigma^2 + 1/\sigma_0^2},\; \frac{1}{1/\sigma^2 + 1/\sigma_0^2}\right)$$

**Interpretation:** The posterior mean is a **precision-weighted average** of the observation $y$ and the prior mean $\mu_0$. More precise information (smaller variance) gets higher weight. When $\sigma_0^2 \to \infty$ (flat prior): posterior mean $\to y$ (MLE). When $\sigma^2 \to \infty$ (very noisy observation): posterior mean $\to \mu_0$ (prior dominates).

This formula is the **Kalman filter update equation** for the 1D case.

---

## Appendix S: From Joint Distributions to Variational Inference

Variational inference (VI) approximates intractable posteriors by solving an optimisation problem over a tractable family of distributions.

**Goal:** Approximate $p(\mathbf{z} \mid \mathbf{x}) = p(\mathbf{z}, \mathbf{x}) / p(\mathbf{x})$ with $q_\phi(\mathbf{z}) \in \mathcal{Q}$.

**Evidence Lower Bound (ELBO):**
$$\log p(\mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z})}\!\left[\log \frac{p(\mathbf{z}, \mathbf{x})}{q_\phi(\mathbf{z})}\right] + D_{\text{KL}}(q_\phi(\mathbf{z}) \| p(\mathbf{z} \mid \mathbf{x}))$$

Since $D_{\text{KL}} \geq 0$, we have $\log p(\mathbf{x}) \geq \mathcal{L}(\phi)$ where:
$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\mathbf{z})}\!\left[\log \frac{p(\mathbf{z}, \mathbf{x})}{q_\phi(\mathbf{z})}\right] = \mathbb{E}_{q_\phi}[\log p(\mathbf{x} \mid \mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}) \| p(\mathbf{z}))$$

**Connection to joint distributions:** The ELBO decomposes as:
- Reconstruction: how well $q_\phi$ places mass where $p(\mathbf{x} \mid \mathbf{z})$ is large
- Regularisation: $D_{\text{KL}}(q_\phi \| p(\mathbf{z}))$ keeps $q_\phi$ close to the prior joint distribution $p(\mathbf{z})$

Maximising the ELBO is equivalent to minimising $D_{\text{KL}}(q_\phi(\mathbf{z}) \| p(\mathbf{z} \mid \mathbf{x}))$ — finding the best joint approximation in $\mathcal{Q}$.

**Mean-field VI:** The simplest tractable family: $q_\phi(\mathbf{z}) = \prod_i q_{\phi_i}(z_i)$. This assumes all latent variables are independent — a strong assumption that leads to underestimated posterior variances (VI is over-confident). VAEs use this assumption in the encoder: the output $q_\phi(\mathbf{z} \mid \mathbf{x}) = \prod_d \mathcal{N}(z_d; \mu_d, \sigma_d^2)$ is a diagonal (factored) Gaussian.

**For AI (2026):** Flow-based posteriors (normalising flows applied to the variational distribution) relax the mean-field assumption while remaining tractable. Diffusion-based VI methods (Geffner et al., 2023) use learned denoising processes as flexible approximate posteriors, achieving near-exact inference for structured joint distributions.

---

## Appendix T: Extended ML Case Studies

### T.1 Case Study: Transformer Attention as Conditional Expectation

We show in detail how the self-attention operation is a conditional expectation under a joint distribution over sequence positions.

**Setting up the joint distribution:** Define a joint distribution over "query position" $T$ and "key position" $S$ with:
$$p(S = s \mid T = t) = \alpha_{ts} = \text{softmax}\!\left(\frac{\mathbf{q}_t^\top \mathbf{k}_s}{\sqrt{d}}\right)_s$$

This is a well-defined conditional distribution: $\alpha_{ts} \geq 0$ and $\sum_s \alpha_{ts} = 1$.

**The attention output as conditional expectation:**
$$\mathbf{o}_t = \sum_s \alpha_{ts} \mathbf{v}_s = \mathbb{E}_{S \sim p(\cdot \mid T=t)}[\mathbf{v}_S]$$

This is the conditional mean of the value vector under the attention distribution — exactly the conditional expectation formula $\mathbb{E}[Y \mid X = x]$ from Section 3.2.

**Multi-head attention as mixture:**
Each head $h$ has its own attention distribution $p_h(S \mid T)$. The output is:
$$\mathbf{o}_t = W_O \left[\mathbf{o}_t^{(1)} \| \cdots \| \mathbf{o}_t^{(H)}\right]$$

where $\mathbf{o}_t^{(h)} = \mathbb{E}_{S \sim p_h(\cdot \mid T=t)}[\mathbf{v}_S^{(h)}]$. Different heads can specialise in different conditional distributions — syntax, semantics, coreference, etc.

**KV cache as sufficient statistic:** The key insight is that once $(\mathbf{k}_s, \mathbf{v}_s)$ are computed for position $s$, they fully determine $p_h(S=s \mid T=t)$ and $\mathbf{v}_S^{(h)}$ for any future query position $t > s$. The KV cache stores exactly these sufficient statistics — the conditional distributions of future positions given past keys and values.

### T.2 Case Study: VAE as Latent Variable Model

The VAE is a latent variable model with joint distribution:
$$p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z}) \cdot p(\mathbf{z})$$

**The chain of joint distributions:**

1. **Prior:** $\mathbf{z} \sim p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$ — maximum entropy distribution over latent space
2. **Generative joint:** $p_\theta(\mathbf{x}, \mathbf{z})$ — the model's joint distribution
3. **Approximate posterior:** $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$
4. **Marginal likelihood:** $p_\theta(\mathbf{x}) = \int p_\theta(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})\,d\mathbf{z}$ — intractable; ELBO is a lower bound

**Reparameterisation implements the affine transformation:** The encoder outputs $(\boldsymbol{\mu}_\phi, \boldsymbol{\sigma}_\phi)$. Sampling $\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon}$ with $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$ uses the affine transformation property: $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_\phi, \text{diag}(\boldsymbol{\sigma}_\phi^2))$.

**Training objective decomposition:**
$$\mathcal{L} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{regularisation}}$$

The second term, computed in closed form (Appendix O.3), enforces that the approximate posterior $q_\phi(\mathbf{z} \mid \mathbf{x})$ stays close to the prior — preventing the encoder from ignoring the prior and using an arbitrary encoding.

### T.3 Case Study: Gaussian Process Regression as Conditional MVN

**Setting:** We observe noisy evaluations $\mathbf{y} = f(\mathbf{X}) + \boldsymbol{\varepsilon}$ at training inputs $\mathbf{X}$. We want to predict $f_* = f(\mathbf{X}_*)$ at test points.

**Joint prior:**
$$(f_*, \mathbf{y}) \sim \mathcal{N}\!\left(\mathbf{0}, \begin{pmatrix} K_{**} & K_{*n} \\ K_{n*} & K_{nn} + \sigma^2 I \end{pmatrix}\right)$$

where $K_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$ is the kernel matrix.

**Prediction = Conditional MVN:** Apply the Schur complement formula (Section 6.4):
$$f_* \mid \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_*, \Sigma_*)$$

with $\boldsymbol{\mu}_* = K_{*n}(K_{nn} + \sigma^2 I)^{-1}\mathbf{y}$ and $\Sigma_* = K_{**} - K_{*n}(K_{nn} + \sigma^2 I)^{-1}K_{n*}$.

The posterior mean $\boldsymbol{\mu}_*$ is a weighted combination of training outputs — the weights $K_{*n}(K_{nn} + \sigma^2 I)^{-1}$ are the GP analog of the Kalman gain. The posterior variance $\Sigma_*$ quantifies predictive uncertainty — it is reduced compared to the prior $K_{**}$ because we have data.

**Connection to Bayesian linear regression:** With a linear kernel $k(\mathbf{x},\mathbf{x}') = \mathbf{x}^\top \mathbf{x}'$, GP regression reduces to Bayesian linear regression with an isotropic prior on weights. The MVN conditional formula gives the posterior over weights and the predictive distribution simultaneously.

---

## Appendix U: Numerical Examples — Covariance Matrix Computations

### U.1 Computing the Schur Complement by Hand

**Problem:** 
$$\Sigma = \begin{pmatrix}4 & 2\\ 2 & 3\end{pmatrix}, \quad \Sigma_{11} = 4, \quad \Sigma_{12} = \Sigma_{21} = 2, \quad \Sigma_{22} = 3$$

**Schur complement:** $\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21} = 4 - 2 \cdot (1/3) \cdot 2 = 4 - 4/3 = 8/3$

**Interpretation:** The conditional variance of $X_1$ given $X_2$ is $8/3 \approx 2.67 < 4$ — conditioning reduces variance. The reduction $4/3$ is the variance explained by $X_2$.

**Correlation from covariance:** $\rho = 2/\sqrt{4 \cdot 3} = 2/\sqrt{12} \approx 0.577$.

**Conditional distribution** for $X_1 \mid X_2 = x_2$ (with $\mu_1 = \mu_2 = 0$):
$$X_1 \mid X_2 = x_2 \sim \mathcal{N}\!\left(\frac{2}{3}x_2, \frac{8}{3}\right)$$

### U.2 Eigendecomposition of a 2x2 Covariance Matrix

For $\Sigma = \begin{pmatrix}4 & 2\\ 2 & 3\end{pmatrix}$:

**Characteristic polynomial:** $\lambda^2 - 7\lambda + 8 = 0$

**Eigenvalues:** $\lambda_{1,2} = (7 \pm \sqrt{49-32})/2 = (7 \pm \sqrt{17})/2$

Numerically: $\lambda_1 \approx 5.56$, $\lambda_2 \approx 1.44$.

**Eigenvectors:** For $\lambda_1$: $(4-\lambda_1)v_1 + 2v_2 = 0 \Rightarrow v_1 \propto (2, \lambda_1-4) \propto (2, 1.56)$, normalised: $\approx (0.789, 0.614)$.

**Geometric interpretation:** The principal axis of the distribution is at angle $\arctan(0.614/0.789) \approx 37.9°$ from the $x_1$-axis. Variance along the long axis is $\lambda_1 \approx 5.56$; along the short axis $\lambda_2 \approx 1.44$.

### U.3 PSD Check via Cholesky

For $\Sigma = \begin{pmatrix}4 & 2\\ 2 & 3\end{pmatrix}$:

Cholesky: $\Sigma = LL^\top$ where $L = \begin{pmatrix}2 & 0\\ 1 & \sqrt{2}\end{pmatrix}$.

Verify: $LL^\top = \begin{pmatrix}2&0\\1&\sqrt{2}\end{pmatrix}\begin{pmatrix}2&1\\0&\sqrt{2}\end{pmatrix} = \begin{pmatrix}4&2\\2&3\end{pmatrix}$ ✓

Since Cholesky exists (all diagonal entries $2, \sqrt{2} > 0$), $\Sigma$ is SPD.

**Sampling:** Generate $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},I)$, then $\mathbf{x} = L\boldsymbol{\varepsilon}$ has covariance $L I L^\top = \Sigma$.

---

## Appendix V — Worked Problems: Marginals, Conditionals, and Independence Tests

This appendix provides fully worked solutions to eight representative problems at the level
of the exercises. Work through each problem before reading the solution.

---

### V.1 Marginalising a Triangular Joint PDF

**Problem.** Let $(X, Y)$ have joint PDF

$$f_{X,Y}(x, y) = 6x \cdot \mathbf{1}[0 \le y \le x \le 1].$$

Find (a) $f_X(x)$, (b) $f_Y(y)$, (c) $f_{X|Y}(x|y)$, (d) $\mathbb{E}[X|Y=y]$.

**Solution.**

**(a) Marginal of $X$.**  For $0 \le x \le 1$, integrate over $y \in [0, x]$:

$$f_X(x) = \int_0^x 6x \, dy = 6x \cdot x = 6x^2.$$

Check: $\int_0^1 6x^2 \, dx = 2$.  Wait — that gives 2, not 1.  Recheck the support:
the integrand is $6x$ but $x$ ranges over $[0,1]$.  Observe

$$\int_0^1 6x^2 \, dx = 6 \cdot \tfrac{1}{3} = 2 \ne 1.$$

The given density must be renormalised.  The normalisation constant $c$ satisfies

$$c \int_0^1 \int_0^x x \, dy \, dx = c \int_0^1 x^2 \, dx = \tfrac{c}{3} = 1 \implies c = 3.$$

So the correct joint is $f_{X,Y}(x,y) = 3x \cdot \mathbf{1}[0 \le y \le x \le 1]$ and $f_X(x) = 3x^2$.

**(b) Marginal of $Y$.**  For $0 \le y \le 1$, integrate over $x \in [y, 1]$:

$$f_Y(y) = \int_y^1 3x \, dx = 3 \cdot \tfrac{1-y^2}{2} = \tfrac{3(1-y^2)}{2}.$$

Check: $\int_0^1 \tfrac{3(1-y^2)}{2} \, dy = \tfrac{3}{2}[1 - \tfrac{1}{3}] = \tfrac{3}{2} \cdot \tfrac{2}{3} = 1$. ✓

**(c) Conditional $f_{X|Y}(x|y)$.**  By definition,

$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)} = \frac{3x}{\tfrac{3(1-y^2)}{2}} = \frac{2x}{1-y^2}, \quad x \in [y, 1].$$

**(d) Conditional mean $\mathbb{E}[X|Y=y]$.**

$$\mathbb{E}[X|Y=y] = \int_y^1 x \cdot \frac{2x}{1-y^2} \, dx
= \frac{2}{1-y^2} \int_y^1 x^2 \, dx
= \frac{2}{1-y^2} \cdot \frac{1-y^3}{3}
= \frac{2(1-y^3)}{3(1-y^2)}.$$

Factor: $1 - y^3 = (1-y)(1+y+y^2)$ and $1-y^2 = (1-y)(1+y)$, so

$$\mathbb{E}[X|Y=y] = \frac{2(1+y+y^2)}{3(1+y)}.$$

---

### V.2 Checking Independence via Factorisation

**Problem.** For each of the following joint PMFs, determine whether $X \perp Y$.

**(a)**

| $X \backslash Y$ | 0   | 1   |
| ---------------- | --- | --- |
| 0                | 0.3 | 0.2 |
| 1                | 0.1 | 0.4 |

**(b)**

| $X \backslash Y$ | 0    | 1    |
| ---------------- | ---- | ---- |
| 0                | 0.12 | 0.18 |
| 1                | 0.28 | 0.42 |

**Solution.**

**(a)** Marginals: $P(X=0) = 0.5$, $P(X=1) = 0.5$; $P(Y=0) = 0.4$, $P(Y=1) = 0.6$.
Check $(0,0)$: $P(X=0)P(Y=0) = 0.5 \times 0.4 = 0.20 \ne 0.30 = P(X=0,Y=0)$. **Not independent.**

**(b)** Marginals: $P(X=0) = 0.30$, $P(X=1) = 0.70$; $P(Y=0) = 0.40$, $P(Y=1) = 0.60$.
Check all four cells:
- $(0,0)$: $0.30 \times 0.40 = 0.12$ ✓
- $(0,1)$: $0.30 \times 0.60 = 0.18$ ✓
- $(1,0)$: $0.70 \times 0.40 = 0.28$ ✓
- $(1,1)$: $0.70 \times 0.60 = 0.42$ ✓

**Independent.** ✓

---

### V.3 Covariance Computation

**Problem.** Suppose $(X, Y)$ uniform on the unit disk $\{(x,y) : x^2 + y^2 \le 1\}$.
The joint PDF is $f(x,y) = \tfrac{1}{\pi}$.  Compute $\text{Cov}(X, Y)$.

**Solution.**

By symmetry of the unit disk under $(x,y) \mapsto (-x, y)$ and $(x,y) \mapsto (x, -y)$:
- $\mathbb{E}[X] = 0$ and $\mathbb{E}[Y] = 0$ (odd integrand over symmetric domain).
- $\mathbb{E}[XY] = \tfrac{1}{\pi} \iint_{x^2+y^2\le 1} xy \, dx \, dy = 0$ (same symmetry argument: $xy$ is
  antisymmetric under $x \mapsto -x$).

Therefore $\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y] = 0 - 0 \cdot 0 = 0$.

**However, $X$ and $Y$ are not independent:** if $X = 0.9$ then $Y$ must lie in
$[-\sqrt{1 - 0.81}, \sqrt{1 - 0.81}] = [-0.436, 0.436]$, a much tighter range than
without conditioning.  This is a canonical example of zero covariance $\not\Rightarrow$ independence.

---

### V.4 Bivariate Normal: Conditional Mean and Variance

**Problem.** $(X_1, X_2) \sim \mathcal{N}(\mathbf{0}, \Sigma)$ with $\Sigma = \begin{pmatrix} 4 & 3 \\ 3 & 9 \end{pmatrix}$.

Find the distribution of $X_1 | X_2 = x_2$.

**Solution.**

Using the Schur complement formula with $\mu_1 = \mu_2 = 0$:

$$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)
= 0 + 3 \cdot \frac{1}{9} \cdot x_2 = \frac{x_2}{3}.$$

$$\sigma^2_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
= 4 - 3 \cdot \frac{1}{9} \cdot 3 = 4 - 1 = 3.$$

Therefore $X_1 | X_2 = x_2 \sim \mathcal{N}\!\left(\tfrac{x_2}{3},\, 3\right)$.

The correlation is $\rho = 3 / \sqrt{4 \cdot 9} = 3/6 = 1/2$, and the conditional mean
$\mu_{1|2} = \rho \cdot \tfrac{\sigma_1}{\sigma_2} \cdot x_2 = \tfrac{1}{2} \cdot \tfrac{2}{3} \cdot x_2 = \tfrac{x_2}{3}$ confirms our result.

---

### V.5 Reparameterisation Trick Gradient

**Problem.** The ELBO of a VAE contains the term $\mathbb{E}_{z \sim q_\phi(z|x)}[f(z)]$ where
$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$.  Show that
$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[\nabla_\phi f(\mu_\phi + \sigma_\phi \odot \varepsilon)]$.

**Solution.**

The key insight is to push the gradient inside the expectation.  With the direct
parameterisation, $q_\phi$ depends on $\phi$, making $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)]$
require differentiating through the distribution — the REINFORCE estimator does this
but has high variance.

The reparameterisation trick decouples the stochasticity from the parameters.  Write

$$z = g(\phi, \varepsilon) = \mu_\phi + \sigma_\phi \odot \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I).$$

Then

$$\mathbb{E}_{z \sim q_\phi}[f(z)]
= \mathbb{E}_{\varepsilon \sim \mathcal{N}(0,I)}[f(g(\phi, \varepsilon))].$$

Now $\varepsilon$ is drawn from a **fixed** distribution independent of $\phi$.  Assuming
sufficient regularity to swap gradient and expectation:

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)]
= \nabla_\phi \mathbb{E}_{\varepsilon}[f(g(\phi,\varepsilon))]
= \mathbb{E}_{\varepsilon}[\nabla_\phi f(g(\phi,\varepsilon))]
= \mathbb{E}_{\varepsilon}[\nabla_\phi f(\mu_\phi + \sigma_\phi \odot \varepsilon)].$$

The gradient flows through the deterministic function $g$ via the chain rule:

$$\nabla_\phi f(g(\phi,\varepsilon))
= \nabla_z f(z) \cdot \frac{\partial g}{\partial \phi}
= \nabla_z f(z) \cdot \begin{pmatrix} I \\ \text{diag}(\varepsilon) \end{pmatrix}$$

where the second row gives $\partial g / \partial \sigma_\phi = \varepsilon$.  This low-variance
estimator is exactly what enables training VAEs with standard backpropagation.

---

### V.6 Change of Variables: Log-Normal

**Problem.** If $X \sim \mathcal{N}(\mu, \sigma^2)$, find the PDF of $Y = e^X$.

**Solution.**

The transformation $g(x) = e^x$ is strictly increasing, so its inverse is $g^{-1}(y) = \ln y$
with derivative $\frac{d}{dy}g^{-1}(y) = \frac{1}{y}$.  By the univariate change-of-variables formula:

$$f_Y(y) = f_X(\ln y) \cdot \left|\frac{d}{dy}\ln y\right|
= \frac{1}{\sqrt{2\pi}\sigma} \exp\!\left(-\frac{(\ln y - \mu)^2}{2\sigma^2}\right) \cdot \frac{1}{y},
\quad y > 0.$$

This is the **log-normal** distribution.  Its mean and variance are

$$\mathbb{E}[Y] = e^{\mu + \sigma^2/2}, \quad \text{Var}(Y) = (e^{\sigma^2}-1)e^{2\mu+\sigma^2}.$$

**AI relevance.** Log-normal distributions model quantities that are products of many
small independent factors — learning rate schedules, weight magnitudes after multiplicative
updates, and token probability products in sequence models.  When $\sigma^2 \gg 1$ the
distribution has a heavy right tail; this explains why gradient clipping is necessary:
the product of many near-1 weights can occasionally explode.

---

### V.7 Order Statistics of Uniform

**Problem.** Let $U_{(1)} \le U_{(2)} \le \cdots \le U_{(n)}$ be the order statistics of
$n$ iid $\text{Uniform}(0,1)$ random variables.  Find $f_{U_{(k)}}(u)$.

**Solution.**

For the $k$-th order statistic, we need exactly $k-1$ values below $u$, one value at $u$,
and $n-k$ values above $u$.  This is a multinomial counting argument:

$$f_{U_{(k)}}(u) = \frac{n!}{(k-1)!(n-k)!} \cdot u^{k-1} \cdot (1-u)^{n-k} \cdot 1,
\quad u \in [0,1].$$

Recognising the kernel: $U_{(k)} \sim \text{Beta}(k, n-k+1)$.  This is a remarkable
result connecting order statistics to the Beta distribution.

**Moments:**
$$\mathbb{E}[U_{(k)}] = \frac{k}{n+1}, \quad \text{Var}(U_{(k)}) = \frac{k(n-k+1)}{(n+1)^2(n+2)}.$$

**AI relevance.** Order statistics underlie Top-$k$ sampling in autoregressive generation.
When sampling the next token by selecting from the $k$ highest-probability tokens,
the threshold is the $k$-th order statistic of the probability vector (in decreasing order).
The Beta distribution for $U_{(k)}$ gives the distribution of this adaptive threshold.

---

### V.8 Conditional Independence: Chain vs Fork vs Collider

**Problem.** In each graphical model below, state whether $X \perp Y | Z$ and whether
$X \perp Y$ (marginal independence).

**(a)** Chain: $X \to Z \to Y$

**(b)** Fork: $X \leftarrow Z \rightarrow Y$

**(c)** Collider: $X \to Z \leftarrow Y$

**Solution.**

**(a) Chain.** The joint is $p(x,z,y) = p(x)p(z|x)p(y|z)$.
- Marginal: $p(x,y) = \sum_z p(x)p(z|x)p(y|z)$, generally not $p(x)p(y)$.  **Not marginally independent.**
- Conditional: $p(x,y|z) = p(x,z,y)/p(z) = p(x|z)p(y|z)$ after noting $p(x|z) = p(x)p(z|x)/p(z)$.  **$X \perp Y | Z$.** ✓

**(b) Fork.** The joint is $p(z)p(x|z)p(y|z)$.
- Marginal: $p(x,y) = \sum_z p(z)p(x|z)p(y|z)$, generally not $p(x)p(y)$ because both share cause $Z$.  **Not marginally independent.**
- Conditional: $p(x,y|z) = p(x|z)p(y|z)$.  **$X \perp Y | Z$.** ✓

**(c) Collider.** The joint is $p(x)p(y)p(z|x,y)$.
- Marginal: $p(x,y) = p(x)p(y)\sum_z p(z|x,y) = p(x)p(y)$.  **$X \perp Y$ (marginal).** ✓
- Conditional: knowing $Z$ creates dependence between $X$ and $Y$ (explaining away / Berkson's paradox).  **$X \not\perp Y | Z$ in general.**

**Summary table:**

| Structure | $X \perp Y$ (marginal)? | $X \perp Y \mid Z$? |
| --------- | ----------------------- | ------------------- |
| Chain     | No                      | Yes                 |
| Fork      | No                      | Yes                 |
| Collider  | Yes                     | No                  |

**The collider reversal** is the most counterintuitive result in probabilistic graphical
models and has real-world consequences: conditioning on a collider (e.g., selecting on a
trait influenced by two independent factors) induces spurious correlations between
otherwise independent causes.  In neural architecture search, selecting architectures
that achieve high validation accuracy (collider: architecture traits + dataset difficulty)
can create apparent anti-correlations between design choices that are truly independent.


---

## Appendix W — Notation Reference and Quick Formulas

### W.1 Notation Used in This Section

| Symbol | Meaning |
|--------|---------|
| $f_{X,Y}(x,y)$ | Joint PDF of $(X,Y)$ |
| $p_{X,Y}(x,y)$ | Joint PMF of $(X,Y)$ |
| $F_{X,Y}(x,y)$ | Joint CDF: $P(X \le x, Y \le y)$ |
| $f_{X}(x)$ | Marginal PDF of $X$ |
| $f_{Y\|X}(y\|x)$ | Conditional PDF of $Y$ given $X=x$ |
| $X \perp Y$ | $X$ and $Y$ are independent |
| $X \perp Y \| Z$ | $X$ and $Y$ are conditionally independent given $Z$ |
| $\boldsymbol{\mu}$ | Mean vector of multivariate distribution |
| $\Sigma$ | Covariance matrix |
| $\Sigma_{12}$ | Off-diagonal block $\text{Cov}(X_1, X_2)$ |
| $\Sigma_{1\|2}$ | Conditional covariance (Schur complement) |
| $\rho_{XY}$ | Pearson correlation coefficient |
| $\phi(\cdot)$ | Standard normal PDF |
| $C(u,v)$ | Copula function |
| $\odot$ | Element-wise (Hadamard) product |

### W.2 Master Formula Sheet

**Joint, marginal, conditional (continuous):**

$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)\,dy$$

$$f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

$$f_{X,Y}(x,y) = f_{Y|X}(y|x) \cdot f_X(x)$$

**Chain rule ($n$ variables):**

$$p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i \mid x_1, \ldots, x_{i-1})$$

**Independence:**

$$X \perp Y \iff f_{X,Y}(x,y) = f_X(x)f_Y(y) \text{ a.e.}$$

**Covariance and correlation:**

$$\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$$

$$\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sqrt{\text{Var}(X)\text{Var}(Y)}} \in [-1,1]$$

**Multivariate Gaussian:**

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$$
$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

**MVN conditional (Schur complement):**

$$\mathbf{X}_1|\mathbf{X}_2=\mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2},\, \Sigma_{1|2})$$

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$

**MVN affine transform:**

$$A\mathbf{X} + \mathbf{b} \sim \mathcal{N}(A\boldsymbol{\mu}+\mathbf{b},\; A\Sigma A^\top)$$

**Change of variables (bivariate):**

$$f_\mathbf{Y}(\mathbf{y}) = f_\mathbf{X}(g^{-1}(\mathbf{y})) \cdot |\det J_{g^{-1}}(\mathbf{y})|$$

**Convolution:**

$$f_{X+Y}(z) = \int_{-\infty}^{\infty} f_X(x)\, f_Y(z-x)\, dx$$

**Reparameterisation trick:**

$$\mathbf{z} = \boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

$$\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{\boldsymbol{\varepsilon}}[\nabla_\phi f(\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\varepsilon})]$$

**Sklar's Theorem:**

$$F_{X,Y}(x,y) = C(F_X(x), F_Y(y))$$

### W.3 Key Inequalities

| Inequality | Statement | Use Case |
|------------|-----------|----------|
| Cauchy-Schwarz | $|\text{Cov}(X,Y)|^2 \le \text{Var}(X)\text{Var}(Y)$ | Bounds correlation $|\rho| \le 1$ |
| Chebyshev | $P(|X-\mu| \ge k\sigma) \le 1/k^2$ | Tail bounds from mean+variance |
| Jensen | $f(\mathbb{E}[X]) \le \mathbb{E}[f(X)]$ for convex $f$ | ELBO derivation, log inequalities |
| Markov | $P(X \ge t) \le \mathbb{E}[X]/t$ for $X \ge 0$ | Simplest tail bound |
| Data Processing | $I(X;Y) \ge I(X;g(Y))$ for any function $g$ | Information is not increased by processing |

### W.4 Common Mistakes — Quick Reference

| # | Mistake | Fix |
|---|---------|-----|
| 1 | Confusing joint CDF and joint PDF | $F(x,y) = P(X \le x, Y \le y)$; PDF is the mixed partial $\partial^2 F / \partial x \partial y$ |
| 2 | $\text{Cov}=0 \Rightarrow$ independent | Only true for jointly Gaussian; counterexample: unit disk |
| 3 | Forgetting Jacobian in change of variables | Always compute $|\det J|$; add it to the PDF |
| 4 | Treating correlated Gaussians as independent | $X,Y$ Gaussian does not imply $(X,Y)$ jointly Gaussian |
| 5 | Conditioning on zero-probability events without limiting argument | Use $f_{Y|X}$ for continuous $X$ — not a ratio of point probabilities |
| 6 | Applying Schur formula without checking $\Sigma_{22} \succ 0$ | Verify positive definiteness before inverting |
| 7 | Confusing marginal and conditional independence | Collider: marginally independent, conditionally dependent |
| 8 | Forgetting the log-determinant term in MVN log-likelihood | $\log f = -\tfrac{d}{2}\log 2\pi - \tfrac{1}{2}\log|\Sigma| - \tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})$ |

