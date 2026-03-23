[← Back to Chapter 7: Statistics](../README.md) | [Next: Estimation Theory →](../02-Estimation-Theory/notes.md)

---

# Descriptive Statistics

> _"The first step in wisdom is knowing what you don't know — and in statistics, that begins by knowing what your data actually looks like."_
> — John Tukey

## Overview

Descriptive statistics is the art and science of summarising data before you model it. It answers a deceptively simple question: **what does this dataset look like?** Before fitting a model, optimising a loss, or drawing any inference, you must know whether your features are symmetric or skewed, whether outliers will dominate gradient updates, whether two variables move together, and whether the scale differences between features will derail gradient descent. Descriptive statistics provides the vocabulary and tools for all of this.

The discipline divides naturally into three layers. The first layer is **univariate**: for each variable in isolation, compute its centre (mean, median, mode), its spread (variance, IQR, MAD), and its shape (skewness, kurtosis). The second layer is **bivariate**: measure how pairs of variables relate through covariance, Pearson correlation, and rank correlation. The third layer is **multivariate**: characterise the joint behaviour of all variables together through the covariance matrix, Mahalanobis distance, and correlation heatmaps.

Every modern ML pipeline touches descriptive statistics constantly, often invisibly. Batch normalisation computes sample means and variances of activations. Adam's momentum terms are exponentially-weighted sample statistics. Dataset documentation standards require engineers to report label distributions, feature histograms, and correlation structures. Data drift is detected by monitoring shifts in descriptive statistics over time. This section builds the rigorous foundation for all of these applications.

## Prerequisites

- **Expectation and variance** ($\mathbb{E}[X]$, $\text{Var}(X)$, covariance of random variables) — [Ch6 §04 Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Common distributions** (Gaussian, Bernoulli, Poisson) — [Ch6 §02 Common Distributions](../../06-Probability-Theory/02-Common-Distributions/notes.md)
- **Basic probability** (sample spaces, random variables, CDF/PDF) — [Ch6 §01](../../06-Probability-Theory/01-Introduction-and-Random-Variables/notes.md)
- **Matrix operations** (matrix–vector products, transpose, positive definiteness) — [Ch2 §02](../../02-Linear-Algebra-Basics/02-Matrix-Operations/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive implementation of every statistic with visualisations and ML-case studies |
| [exercises.ipynb](exercises.ipynb) | 8 graded exercises from basic EDA through Batch Norm and Adam momentum analysis |

## Learning Objectives

After completing this section, you will:

- Compute and interpret all standard measures of central tendency, spread, and shape for a univariate sample
- Explain Bessel's correction and derive why $n-1$ gives an unbiased variance estimator
- Construct and interpret the empirical CDF, quantiles, five-number summary, and box plots
- Assess outliers using Z-score thresholds, Tukey fences, and MAD-based robust methods
- Define the breakdown point of a statistic and select between mean and median appropriately
- Compute Pearson, Spearman, and Kendall correlation and choose the right one for your data
- Build the empirical covariance matrix, check positive semi-definiteness, and compute Mahalanobis distance
- Explain how sample statistics relate to population parameters and what the iid assumption means
- Connect batch normalisation, layer normalisation, and RMS norm to their descriptive-statistics definitions
- Derive Adam's first and second moments as exponentially-weighted sample statistics
- Detect data drift by monitoring shifts in descriptive statistics over time
- Apply the full EDA workflow to a multivariate ML dataset

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 What Descriptive Statistics Does](#11-what-descriptive-statistics-does)
  - [1.2 Why It Matters for AI](#12-why-it-matters-for-ai)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 The EDA Workflow](#14-the-eda-workflow)
- [2. Formal Definitions and Notation](#2-formal-definitions-and-notation)
  - [2.1 Sample Space and the Data Matrix](#21-sample-space-and-the-data-matrix)
  - [2.2 Measures of Central Tendency](#22-measures-of-central-tendency)
  - [2.3 Measures of Spread](#23-measures-of-spread)
  - [2.4 Shape Statistics: Skewness and Kurtosis](#24-shape-statistics-skewness-and-kurtosis)
  - [2.5 Order Statistics and Quantiles](#25-order-statistics-and-quantiles)
- [3. Robust Statistics](#3-robust-statistics)
  - [3.1 Breakdown Point and Influence Function](#31-breakdown-point-and-influence-function)
  - [3.2 Robust Location Estimators](#32-robust-location-estimators)
  - [3.3 Robust Spread: MAD and IQR](#33-robust-spread-mad-and-iqr)
  - [3.4 Outlier Detection Methods](#34-outlier-detection-methods)
- [4. Bivariate Statistics](#4-bivariate-statistics)
  - [4.1 Sample Covariance and Pearson Correlation](#41-sample-covariance-and-pearson-correlation)
  - [4.2 Rank Correlation: Spearman and Kendall](#42-rank-correlation-spearman-and-kendall)
  - [4.3 Association Beyond Linear Correlation](#43-association-beyond-linear-correlation)
  - [4.4 Correlation vs Causation](#44-correlation-vs-causation)
- [5. Multivariate Descriptive Statistics](#5-multivariate-descriptive-statistics)
  - [5.1 Empirical Mean Vector and Covariance Matrix](#51-empirical-mean-vector-and-covariance-matrix)
  - [5.2 Correlation Matrix and Visualisation](#52-correlation-matrix-and-visualisation)
  - [5.3 Mahalanobis Distance](#53-mahalanobis-distance)
  - [5.4 Dimensionality and the Curse](#54-dimensionality-and-the-curse)
- [6. Data Standardisation and Preprocessing](#6-data-standardisation-and-preprocessing)
  - [6.1 Z-Score Standardisation](#61-z-score-standardisation)
  - [6.2 Min-Max and Robust Scaling](#62-min-max-and-robust-scaling)
  - [6.3 Normalisation in Neural Networks](#63-normalisation-in-neural-networks)
  - [6.4 Handling Missing Data](#64-handling-missing-data)
- [7. Visualisation for EDA](#7-visualisation-for-eda)
  - [7.1 Univariate Plots](#71-univariate-plots)
  - [7.2 Bivariate and Multivariate Plots](#72-bivariate-and-multivariate-plots)
  - [7.3 Assessing Normality](#73-assessing-normality)
- [8. Applications in Machine Learning](#8-applications-in-machine-learning)
  - [8.1 Batch Normalisation and Layer Normalisation](#81-batch-normalisation-and-layer-normalisation)
  - [8.2 Adam Optimiser as Online Sample Statistics](#82-adam-optimiser-as-online-sample-statistics)
  - [8.3 Dataset Characterisation and Model Cards](#83-dataset-characterisation-and-model-cards)
  - [8.4 Data Drift and Distribution Shift](#84-data-drift-and-distribution-shift)
- [9. Common Mistakes](#9-common-mistakes)
- [10. Exercises](#10-exercises)
- [11. Why This Matters for AI (2026 Perspective)](#11-why-this-matters-for-ai-2026-perspective)
- [12. Conceptual Bridge](#12-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Descriptive Statistics Does

Descriptive statistics does not make predictions. It does not test hypotheses. It does not fit models. Its sole purpose is to **look at the data** — to summarise its essential features in a form that a human can understand and that downstream methods can rely on.

The canonical demonstration of why this matters is **Anscombe's Quartet** (1973): four datasets with nearly identical means ($\bar{x} = 9.0$, $\bar{y} = 7.5$), variances ($s_x^2 = 11.0$, $s_y^2 = 4.12$), and Pearson correlations ($r = 0.816$) — yet one is a perfect linear relationship, one is a curved parabola, one is a near-perfect line with a single outlier, and one is a vertical cluster with a single high-leverage point. A statistician who skips visualisation and reports only the summary statistics would fit the same regression line to all four, missing everything important.

The lesson for ML is immediate: if you skip EDA on your training data, you may train a model on a dataset where a single feature with extreme variance (unnormalised pixel intensity vs normalised embedding values) dominates gradient updates; where two features are so highly correlated that they provide redundant information and slow convergence; where a label is heavily imbalanced, making accuracy a misleading metric; or where systematic missing-data patterns encode the label and will cause the model to cheat.

**Descriptive statistics is the gate through which all good modelling passes.** Every subsequent step — estimation, hypothesis testing, Bayesian inference — implicitly assumes you already know your data's basic structure.

### 1.2 Why It Matters for AI

The connection between descriptive statistics and modern AI is deeper than "preprocessing". Sample statistics are computed billions of times per training run in contemporary neural networks.

**Batch Normalisation** (Ioffe & Szegedy, 2015) computes the sample mean and variance of each feature channel across a mini-batch, then normalises activations to zero mean and unit variance. This is exactly the z-score standardisation formula applied at every layer, every forward pass. Without it, deep networks suffer from internal covariate shift — the distribution of activations drifts as parameters update, causing gradient vanishing or explosion.

**Layer Normalisation** (Ba et al., 2016), used in every transformer architecture (GPT, BERT, LLaMA, Mistral, Gemini), computes the mean and variance across features within a single token's representation. For a token embedding $\mathbf{h} \in \mathbb{R}^d$: $\text{LN}(\mathbf{h}) = \frac{\mathbf{h} - \hat{\mu}}{\hat{\sigma} + \epsilon} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$. The $\hat{\mu}$ and $\hat{\sigma}$ are sample statistics over the $d$ components.

**Adam optimiser** (Kingma & Ba, 2015) maintains an exponentially-weighted moving average of past gradients (first moment, analogous to a running mean) and past squared gradients (second moment, analogous to a running variance). The update rule corrects for initialisation bias exactly as one corrects for Bessel bias in sample statistics.

**Dataset cards** required by Hugging Face, Google, and the EU AI Act mandate reporting of label distributions (a sample frequency table), feature statistics (mean, std, min, max per column), and known biases (correlations between sensitive attributes and labels).

**For AI:** Every time you decide how to initialise weights, normalise activations, monitor training stability, document a dataset, or detect distribution shift in production, you are doing descriptive statistics.

### 1.3 Historical Timeline

```
DESCRIPTIVE STATISTICS — HISTORICAL TIMELINE
════════════════════════════════════════════════════════════════════════

  1662  John Graunt — Bills of Mortality: first tabulation of population
        statistics; birth/death rates; the discipline's birth

  1805  Legendre, Gauss — Least squares method; residual analysis;
        the mean as the least-squares centre

  1885  Francis Galton — Percentiles, regression to the mean, rank
        correlation; connected statistics to heredity and evolution

  1896  Karl Pearson — Product-moment correlation coefficient r;
        standard deviation (coined the term); chi-squared distribution

  1919  Ronald Fisher — ANOVA, experimental design; p-values; the
        distinction between sample and population statistics

  1925  Fisher — "Statistical Methods for Research Workers"; codified
        the sample mean, variance, and hypothesis testing framework

  1951  John Tukey — Jackknife resampling; robust estimation begins;
        "the future of data analysis" (1962) anticipates EDA

  1977  John Tukey — "Exploratory Data Analysis" textbook; box plots,
        stem-and-leaf plots; EDA as a discipline; look before you infer

  1973  Frank Anscombe — Anscombe's Quartet; the necessity of plotting

  1983  Peter Rousseeuw — Least Median of Squares; formal breakdown
        point theory; robust statistics becomes rigorous

  2015  Ioffe & Szegedy — Batch Normalization; sample statistics
        embedded inside neural network training loops

  2016  Ba et al. — Layer Normalization; per-token sample statistics
        for transformer architectures

  2020  Gebru et al. — "Datasheets for Datasets"; descriptive statistics
        as ethical AI documentation requirement

  2026  EU AI Act — Mandatory dataset statistics for high-risk AI systems

════════════════════════════════════════════════════════════════════════
```

### 1.4 The EDA Workflow

A disciplined exploratory analysis follows five stages:

```
THE EDA WORKFLOW
════════════════════════════════════════════════════════════════════════

  Stage 1: LOAD AND AUDIT
  ─────────────────────────
  • Check shapes, dtypes, memory usage
  • Count missing values per column
  • Identify data types: numeric / categorical / datetime / text
  • Check for duplicate rows
  • Verify expected value ranges (no age = -5, no probability > 1)

  Stage 2: UNIVARIATE ANALYSIS
  ──────────────────────────────
  For each feature:
  • Central tendency: mean, median, mode
  • Spread: variance, std, IQR, range
  • Shape: skewness, kurtosis
  • Extremes: min, max, outliers
  • Distribution: histogram, KDE, box plot, ECDF

  Stage 3: BIVARIATE ANALYSIS
  ──────────────────────────────
  For each pair (feature, target) and (feature, feature):
  • Pearson/Spearman correlation
  • Scatter plot with regression line
  • Conditional distributions P(Y | X = x)

  Stage 4: MULTIVARIATE ANALYSIS
  ─────────────────────────────────
  • Correlation matrix heatmap
  • Pair plot (scatter matrix)
  • PCA scree plot (variance explained)
  • Cluster structure via UMAP/t-SNE

  Stage 5: DOCUMENT AND ACT
  ──────────────────────────
  • Record all findings
  • Plan transformations (log, standardise, impute)
  • Flag problematic features for domain expert review
  • Update dataset card

════════════════════════════════════════════════════════════════════════
```

In ML practice, this workflow is applied to training data before modelling, to validation data at model evaluation time, and to production data continuously (as drift monitoring). The output is not a model — it is a map of the data landscape that guides every subsequent decision.

---

## 2. Formal Definitions and Notation

### 2.1 Sample Space and the Data Matrix

Let $\mathcal{X}$ be the population — the full set of all possible observations. A **sample** is a collection of $n$ realisations drawn from $\mathcal{X}$:

$$\{x_1, x_2, \ldots, x_n\} \quad x_i \overset{\text{iid}}{\sim} P$$

The **iid assumption** (independent, identically distributed) means each $x_i$ is drawn from the same distribution $P$ independently of all others. This is the foundation on which classical descriptive statistics rests — but it is violated routinely in practice:

- **Time series data**: $x_t$ depends on $x_{t-1}$ (autocorrelation)
- **Clustered data**: patients from the same hospital are correlated
- **Sequential text**: tokens in a sentence are deeply dependent
- **Mini-batches in SGD**: often drawn without replacement, not exactly iid

The **data matrix** for multivariate data is $\mathbf{X} \in \mathbb{R}^{n \times d}$, where $n$ is the number of observations (rows) and $d$ is the number of features (columns). Row $i$ is the feature vector $\mathbf{x}_i \in \mathbb{R}^d$; column $j$ is the vector of all observed values of feature $j$.

A critical distinction runs throughout this section:

| Symbol | Name | Meaning |
|---|---|---|
| $\mu$ | Population mean | $\mathbb{E}[X]$ — theoretical, unknown |
| $\bar{x}$ | Sample mean | $\frac{1}{n}\sum x_i$ — computed from data |
| $\sigma^2$ | Population variance | $\text{Var}(X)$ — theoretical, unknown |
| $s^2$ | Sample variance | $\frac{1}{n-1}\sum(x_i - \bar{x})^2$ — computed from data |
| $\rho$ | Population correlation | $\text{Corr}(X,Y)$ — theoretical |
| $r$ | Sample correlation | Pearson's $r$ — computed from data |

**For AI:** The distinction matters in Batch Norm. During training, BN uses the mini-batch sample mean $\bar{x}$ and sample variance $s^2$. During inference, it uses running estimates of the population mean and variance accumulated during training. The two are different statistics with different properties.

### 2.2 Measures of Central Tendency

**Definition (Sample Mean).** Given a sample $x_1, \ldots, x_n \in \mathbb{R}$, the sample mean is:

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

The mean is the unique minimiser of $\sum_{i=1}^n (x_i - c)^2$ over $c \in \mathbb{R}$. It is the least-squares centre of the data. It is also an **unbiased estimator** of $\mu$: $\mathbb{E}[\bar{x}] = \mu$ (proof: by linearity of expectation).

**Definition (Sample Median).** Sort the sample as $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$. The sample median is:

$$\tilde{x} = \begin{cases} x_{(m+1)} & \text{if } n = 2m+1 \text{ (odd)} \\ \frac{x_{(m)} + x_{(m+1)}}{2} & \text{if } n = 2m \text{ (even)} \end{cases}$$

The median is the minimiser of $\sum_{i=1}^n |x_i - c|$ over $c$. It is the least-absolute-deviations centre.

**Definition (Sample Mode).** The value that appears most frequently. Not unique if two values tie (bimodal). Not well-defined for continuous data without binning.

**Comparison:**

```
CENTRAL TENDENCY COMPARISON
════════════════════════════════════════════════════════════════════════

  Statistic  Loss minimised     Sensitivity    Use when
  ─────────  ─────────────────  ─────────────  ─────────────────────────
  Mean       Σ(xᵢ - c)²         High           Symmetric, no outliers
             (L2 loss)           to outliers    Gaussian-like data

  Median     Σ|xᵢ - c|          Low            Skewed distributions,
             (L1 loss)           to outliers    outliers present
                                                Income, house prices

  Mode       0/1 loss           N/A            Categorical data;
                                                multi-modal distributions

════════════════════════════════════════════════════════════════════════
```

**Non-examples:**
- A dataset of incomes $[25, 30, 32, 35, 1000]$ thousand: mean = $224.4$k is misleading; median = $32$k is representative. The mean is pulled catastrophically by the outlier.
- A bimodal dataset $[1, 1.1, 1.2, 8.9, 9, 9.1]$: the mean ($5.05$) and median ($5.05$) both fall in an empty region between the two clusters. The mode is more informative here.
- An exactly symmetric dataset: all three coincide. This is the exception, not the rule.

**For AI:** In label smoothing and temperature scaling for language models, the "centre" of the output distribution is the mode (the argmax token). In KL divergence minimisation, the mean of the approximate distribution is what's optimised. In median-of-means estimation (used in Byzantine-robust distributed learning), the median replaces the mean to guard against corrupted gradient submissions from compromised workers.

### 2.3 Measures of Spread

**Definition (Sample Variance).** The sample variance with Bessel's correction is:

$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Theorem (Bessel's Correction).** The estimator $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$ is unbiased for $\sigma^2$: $\mathbb{E}[s^2] = \sigma^2$.

*Proof.* Write $\sum_{i=1}^n (x_i - \bar{x})^2 = \sum_{i=1}^n x_i^2 - n\bar{x}^2$. Taking expectation:
$$\mathbb{E}\!\left[\sum_{i=1}^n (x_i - \bar{x})^2\right] = n(\sigma^2 + \mu^2) - n\!\left(\frac{\sigma^2}{n} + \mu^2\right) = (n-1)\sigma^2$$
Dividing by $n-1$ gives $\mathbb{E}[s^2] = \sigma^2$. $\square$

The divisor $n$ gives the **maximum likelihood estimator** of $\sigma^2$ under a Gaussian assumption, which is biased. NumPy uses $n$ by default (`ddof=0`); use `ddof=1` for the unbiased version.

**Definition (Standard Deviation).** $s = \sqrt{s^2}$. Note: $s$ is NOT an unbiased estimator of $\sigma$ (Jensen's inequality: $\mathbb{E}[\sqrt{s^2}] \leq \sqrt{\mathbb{E}[s^2]} = \sigma$), but the bias is negligible for large $n$.

**Definition (Range).** $R = x_{(n)} - x_{(1)}$. Maximal sensitivity to outliers; breakdown point = $1/n \approx 0$.

**Definition (Interquartile Range).** $\text{IQR} = Q_3 - Q_1 = Q(0.75) - Q(0.25)$. Contains the middle 50% of the data. Breakdown point = 25%.

**Definition (Coefficient of Variation).** $\text{CV} = s / \bar{x}$ (dimensionless). Useful for comparing spread across variables with different units. Not meaningful when $\bar{x} \approx 0$.

**For AI:** In gradient clipping, the gradient norm $\|\mathbf{g}\|$ is compared to a threshold — the threshold is often chosen based on the empirical standard deviation of gradient norms observed during early training. In the Adam optimiser, the second moment estimate $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$ estimates $\mathbb{E}[g^2]$, and $\sqrt{v_t/(1-\beta_2^t)}$ is the adaptive per-parameter learning rate — a measure of spread in the gradient history.

### 2.4 Shape Statistics: Skewness and Kurtosis

**Definition (Sample Skewness).** Fisher's $g_1$:

$$g_1 = \frac{\frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^3}{s^3} = \frac{m_3}{m_2^{3/2}}$$

where $m_k = \frac{1}{n}\sum (x_i - \bar{x})^k$ is the $k$-th central moment.

- $g_1 > 0$: right-skewed (long right tail; mean > median)
- $g_1 < 0$: left-skewed (long left tail; mean < median)
- $g_1 = 0$: symmetric

**Examples:** Income distributions are right-skewed ($g_1 > 0$). Exam scores near a ceiling are left-skewed ($g_1 < 0$). Gradient norms in deep networks are often right-skewed — a few extreme gradients can destabilise training.

**Definition (Sample Excess Kurtosis).** Fisher's $g_2$:

$$g_2 = \frac{m_4}{m_2^2} - 3$$

The subtraction of 3 normalises so that the Gaussian has $g_2 = 0$.

- $g_2 > 0$: **leptokurtic** (heavier tails than Gaussian; more extreme values)
- $g_2 < 0$: **platykurtic** (lighter tails; more concentrated)
- $g_2 = 0$: **mesokurtic** (Gaussian-like tails)

**For AI:** The distribution of gradients in neural networks is often **leptokurtic** — it has heavier tails than a Gaussian. This is why vanilla SGD with a Gaussian noise model is a poor fit; it's why gradient clipping is essential; and it's why the Student-$t$ distribution (which has heavier tails) is sometimes used to model stochastic gradient noise. Understanding kurtosis explains why "gradient explosion" is the right mental model for large-$g_2$ gradient distributions.

### 2.5 Order Statistics and Quantiles

**Definition (Order Statistics).** Given sample $x_1, \ldots, x_n$, the order statistics are $x_{(1)} \leq x_{(2)} \leq \cdots \leq x_{(n)}$.

**Definition (Quantile / Percentile).** The $p$-th quantile $Q(p)$, $p \in [0,1]$, is a value such that at least proportion $p$ of the data lies at or below it. Common choices: $Q_1 = Q(0.25)$, $Q_2 = Q(0.50) = \tilde{x}$, $Q_3 = Q(0.75)$.

**Definition (Empirical CDF).** The ECDF is:

$$\hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbf{1}[x_i \leq x]$$

This is a step function that jumps by $1/n$ at each observation. It is the non-parametric estimator of the true CDF $F(x) = P(X \leq x)$.

**Theorem (Glivenko-Cantelli).** For iid data, $\sup_{x \in \mathbb{R}} |\hat{F}_n(x) - F(x)| \overset{a.s.}{\to} 0$ as $n \to \infty$.

This guarantees the ECDF converges uniformly to the true CDF — the empirical distribution is a consistent estimator of the population distribution.

**Definition (Five-Number Summary).** $\{\min, Q_1, \tilde{x}, Q_3, \max\}$. The basis of the box plot.

**For AI:** Quantile functions are used in calibration evaluation (reliability diagrams), in quantile regression for uncertainty estimation, in fairness auditing (comparing median outcomes across demographic groups), and in the Wasserstein distance (used in GAN training and dataset comparison), which is computed from the quantile functions of two distributions.

---

## 3. Robust Statistics

### 3.1 Breakdown Point and Influence Function

Classical statistics (mean, variance, Pearson correlation) is vulnerable to outliers because it is designed to be **efficient** under Gaussian assumptions. Robust statistics replaces efficiency under Gaussianity with **resistance to contamination**.

**Definition (Breakdown Point).** The breakdown point $\varepsilon^*$ of an estimator $T$ is the smallest fraction of observations that can be replaced by arbitrarily bad values to make $T$ take an arbitrarily extreme value:

$$\varepsilon^* = \min\left\{\frac{m}{n} : \sup_{\text{corrupt }m\text{ pts}} \|T(\text{corrupted}) - T(\text{clean})\| = \infty\right\}$$

- **Sample mean**: $\varepsilon^* = 1/n \approx 0$. A single observation replaced by $+\infty$ drives the mean to infinity.
- **Sample median**: $\varepsilon^* = \lfloor(n-1)/2\rfloor / n \to 50\%$. You must corrupt more than half the data to move the median arbitrarily far.
- **Sample variance**: $\varepsilon^* = 1/n$ (same as mean — one outlier suffices).
- **IQR**: $\varepsilon^* = 25\%$.
- **MAD**: $\varepsilon^* = 50\%$.

**Definition (Influence Function).** The influence function $\text{IF}(x; T, F)$ measures the effect of a single infinitesimally contaminated observation at $x$ on the estimator $T$:

$$\text{IF}(x; T, F) = \lim_{\varepsilon \to 0} \frac{T((1-\varepsilon)F + \varepsilon\delta_x) - T(F)}{\varepsilon}$$

where $\delta_x$ is a point mass at $x$.

- **Mean**: $\text{IF}(x; \bar{x}, F) = x - \mu$. Unbounded — large $x$ causes large change.
- **Median**: $\text{IF}(x; \tilde{x}, F) = \text{sign}(x - \tilde{x}) / (2f(\tilde{x}))$. Bounded — extreme $x$ has finite effect.

**For AI:** Robust statistics is directly relevant to **Byzantine-robust federated learning** (e.g., Krum, median-of-means, geometric median aggregation), where some fraction of participating clients may send corrupted gradients. The median-of-means estimator achieves breakdown point $\varepsilon^* > 0$ against malicious workers, while naive FedAvg (a simple mean) has $\varepsilon^* = 1/n$.

### 3.2 Robust Location Estimators

**Definition (Trimmed Mean).** The $\alpha$-trimmed mean removes the bottom and top $\alpha$-fraction of observations before computing the mean:

$$\bar{x}_\alpha = \frac{1}{n - 2\lfloor\alpha n\rfloor} \sum_{i=\lfloor\alpha n\rfloor + 1}^{n - \lfloor\alpha n\rfloor} x_{(i)}$$

Breakdown point: $\alpha$. With $\alpha = 0.1$, this is the 10%-trimmed mean — discards the most extreme 10% on each side before averaging.

**Definition (Winsorised Mean).** Rather than discarding extreme values, Winsorisation **clips** them to the $\alpha$ and $1-\alpha$ quantiles:

$$x_i^W = \max(Q_\alpha, \min(Q_{1-\alpha}, x_i))$$

The Winsorised mean is the mean of $\{x_1^W, \ldots, x_n^W\}$. It retains all $n$ observations but limits the damage any single one can do.

**For AI:** **Gradient clipping** in neural network training is exactly Winsorisation of gradient norms. When $\|\mathbf{g}_t\| > \tau$, the gradient is rescaled to $\tau \cdot \mathbf{g}_t / \|\mathbf{g}_t\|$. This is the $\ell_2$-norm equivalent of Winsorising each component. The threshold $\tau$ is a hyperparameter analogous to the Winsorisation quantile. LLaMA, GPT-4, and most modern LLMs use gradient clipping with $\tau \in \{0.5, 1.0\}$.

### 3.3 Robust Spread: MAD and IQR

**Definition (Median Absolute Deviation).** The MAD is the median of absolute deviations from the median:

$$\text{MAD} = \text{median}(|x_i - \tilde{x}|)$$

For a Gaussian distribution, $\sigma \approx 1.4826 \cdot \text{MAD}$, so a **consistent** estimator of $\sigma$ is $\hat{\sigma}_{\text{MAD}} = 1.4826 \cdot \text{MAD}$.

The MAD has breakdown point 50% — the best possible for a location-scale estimator.

**Definition (IQR).** $\text{IQR} = Q_3 - Q_1$. For a Gaussian, $\sigma \approx \text{IQR} / 1.349$.

**Comparison table:**

```
SPREAD ESTIMATORS — PROPERTIES
════════════════════════════════════════════════════════════════════════

  Estimator  Breakdown  Efficiency  Computation   Best for
  ─────────  ─────────  ──────────  ────────────  ─────────────────────
  Std dev s  ~0%        100%        O(n)          Clean Gaussian data
  IQR        25%        37%         O(n log n)    Moderate contamination
  MAD        50%        37%         O(n log n)    Heavy contamination
  Trimmed s  α          depends     O(n log n)    Tunable robustness

════════════════════════════════════════════════════════════════════════
```

**For AI:** Robust spread estimation is used in **anomaly detection** for ML systems monitoring. When a model's loss distribution suddenly widens (increased IQR or MAD of batch losses), it signals a data quality issue or distribution shift — potentially faster and more reliably than watching the mean loss alone, since the mean is pulled by a few extreme batches.

### 3.4 Outlier Detection Methods

**Method 1: Z-score threshold.** Flag $x_i$ as an outlier if $|z_i| > k$ where $z_i = (x_i - \bar{x})/s$. Standard choice: $k = 3$ (flags ~0.3% of a Gaussian). Weakness: breakdown point $\approx 0$; the mean and std used to compute $z_i$ are themselves distorted by outliers.

**Method 2: Tukey fences.** Flag as an outlier if $x_i < Q_1 - 1.5 \cdot \text{IQR}$ or $x_i > Q_3 + 1.5 \cdot \text{IQR}$. This is the standard box plot definition. "Extreme" outliers: use factor $3.0$. Breakdown point: 25%.

**Method 3: Modified Z-score (Iglewicz-Hoaglin).** Uses the MAD instead of the standard deviation:

$$M_i = \frac{0.6745(x_i - \tilde{x})}{\text{MAD}}$$

Flag as outlier if $|M_i| > 3.5$. This is robust: breakdown point 50%.

**Summary:**

| Method | Breakdown | When to use |
|---|---|---|
| Z-score | ~0% | Clean data; fast first pass |
| Tukey fences | 25% | General purpose; default choice |
| Modified Z-score | 50% | Heavily contaminated data |

**For AI:** Outlier detection on training data is a data-cleaning step before model training. But outlier detection also occurs at **inference time**: inputs far outside the training distribution (OOD inputs) can be flagged using Mahalanobis distance (§5.3), energy scores, or model uncertainty. The descriptive statistics methods here are the simplest and most interpretable first pass.

---

## 4. Bivariate Statistics

### 4.1 Sample Covariance and Pearson Correlation

**Definition (Sample Covariance).** For paired samples $(x_1, y_1), \ldots, (x_n, y_n)$:

$$s_{xy} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})$$

Covariance measures the direction of linear association: $s_{xy} > 0$ means $x$ and $y$ tend to increase together; $s_{xy} < 0$ means they move in opposite directions; $s_{xy} = 0$ means no linear association (but possibly nonlinear).

**Definition (Pearson Correlation Coefficient).** The sample Pearson correlation is:

$$r = \frac{s_{xy}}{s_x s_y} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2} \cdot \sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

**Properties:**
- $r \in [-1, 1]$ (by Cauchy-Schwarz inequality)
- $r = 1$: perfect positive linear relationship; $r = -1$: perfect negative linear
- Invariant under affine transformations: $r(ax+b, cy+d) = \text{sign}(ac) \cdot r(x,y)$
- $r = 0$ does NOT imply independence (only linear independence)

**Theorem (Cauchy-Schwarz bound on $r$).** By the Cauchy-Schwarz inequality applied to the centred vectors $\mathbf{u} = (x_i - \bar{x})$ and $\mathbf{v} = (y_i - \bar{y})$:

$$\left|\sum_{i=1}^n u_i v_i\right| \leq \sqrt{\sum_{i=1}^n u_i^2} \cdot \sqrt{\sum_{i=1}^n v_i^2}$$

which gives $|s_{xy}| \leq s_x s_y$, hence $|r| \leq 1$. Equality holds iff $y_i - \bar{y} = c(x_i - \bar{x})$ for all $i$ (perfect linear relationship). $\square$

**Non-examples of $r = 0$ implying independence:**
- $x_i \sim \text{Uniform}(-1,1)$, $y_i = x_i^2$: $r = 0$ exactly, yet $y$ is a deterministic function of $x$.
- Anscombe's Dataset II: parabola, $r = 0.816$ but the fit is entirely non-linear.

**For AI:** In transformer attention, the attention weights form a row of a correlation-like matrix between query and key vectors. The dot-product $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ is an unnormalised, uncentred covariance between the query and key representations. The softmax normalisation maps these to probabilities. Understanding Pearson correlation provides intuition for what "attention scores" actually measure.

### 4.2 Rank Correlation: Spearman and Kendall

**Definition (Spearman's $\rho$).** Compute ranks $R(x_i)$ and $R(y_i)$ of each sample within their respective sequences, then compute Pearson correlation on the ranks:

$$\rho_S = r(\mathbf{R}(x), \mathbf{R}(y)) = 1 - \frac{6 \sum_{i=1}^n d_i^2}{n(n^2-1)}$$

where $d_i = R(x_i) - R(y_i)$ and the second formula holds when there are no ties.

**Definition (Kendall's $\tau$).** Count concordant pairs (both $x$ and $y$ increase together) and discordant pairs:

$$\tau = \frac{(\text{number of concordant pairs}) - (\text{number of discordant pairs})}{\binom{n}{2}}$$

**When to use each:**

| Method | Captures | Use when |
|---|---|---|
| Pearson $r$ | Linear association | Both vars approx. Gaussian, no outliers |
| Spearman $\rho_S$ | Monotone association | Ordinal data, outliers present, non-linear monotone |
| Kendall $\tau$ | Concordance probability | Small samples, ordinal data, robust inference |

**For AI:** When auditing ML models for fairness, Spearman correlation (rather than Pearson) between model outputs and sensitive attributes is more appropriate: the relationship between a model's confidence score and a demographic attribute is rarely linear. Rank correlation is also used in **information retrieval evaluation** — NDCG and Kendall's $\tau$ between predicted and true rankings.

### 4.3 Association Beyond Linear Correlation

**Mutual Information (preview).** For random variables $X$ and $Y$:

$$I(X; Y) = \int\!\!\int p(x,y) \log \frac{p(x,y)}{p(x)p(y)}\, dx\, dy$$

MI captures **all** statistical dependencies (linear and nonlinear). $I(X;Y) = 0$ iff $X \perp Y$. Unlike Pearson correlation, it detects the $y = x^2$ relationship.

> **Forward reference:** Mutual information is the canonical measure of statistical dependence with no linearity assumption. Full treatment: [Chapter 9 — Information Theory](../../09-Information-Theory/README.md).

**Distance Correlation.** Székely et al. (2007): $\text{dCor}(X,Y) = 0$ iff $X \perp Y$ (for all distributions with finite first moments). Computed via pairwise distance matrices. Captures non-linear dependence while remaining a proper correlation measure in $[0,1]$.

**Cramér's V (categorical association).** For two categorical variables with a contingency table, Cramér's V normalises the chi-squared statistic to $[0,1]$: $V = \sqrt{\chi^2 / (n \cdot \min(r-1, c-1))}$.

### 4.4 Correlation vs Causation

The maxim "correlation is not causation" is one of the most important lessons in statistics for ML practitioners — and one of the most frequently violated.

**Simpson's Paradox (formal example).** A dataset shows that treatment A appears better than treatment B when the full dataset is used ($r(\text{treatment}, \text{outcome}) > 0$), but treatment B is better within every subgroup (men and women separately). This reversal occurs when a **confounding variable** (sex) is correlated with both the treatment assignment and the outcome.

In ML: A model trained on internet text learns that "doctor" co-occurs more with "he" than "she" — not because being a doctor causes gender, but because historical data reflects a correlation induced by societal biases (the confounder). The model's correlation-based weights perpetuate the bias.

**Key insight:** Correlation between feature $X$ and label $Y$ is sufficient to use $X$ as a predictor **only if the test distribution matches the training distribution**. If the mechanism generating the correlation changes (distribution shift), the correlation can reverse or disappear — but a causal relationship would remain stable.

**For AI:** This is why OOD generalisation is fundamentally harder than in-distribution accuracy. Models trained to exploit spurious correlations (e.g., grass background → outdoor scene) fail when those correlations break (grass in an indoor setting). Causal ML is an active research area aimed at learning invariant, causal features rather than spurious correlations.

---

## 5. Multivariate Descriptive Statistics

### 5.1 Empirical Mean Vector and Covariance Matrix

**Definition (Sample Mean Vector).** For data matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$\hat{\boldsymbol{\mu}} = \frac{1}{n} \mathbf{X}^\top \mathbf{1}_n = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \in \mathbb{R}^d$$

**Definition (Centred Data Matrix).** $\mathbf{X}_c = \mathbf{X} - \mathbf{1}_n \hat{\boldsymbol{\mu}}^\top$ (subtract the mean from each row).

**Definition (Sample Covariance Matrix).** With Bessel's correction:

$$\hat{\Sigma} = \frac{1}{n-1} \mathbf{X}_c^\top \mathbf{X}_c = \frac{1}{n-1} \sum_{i=1}^n (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^\top \in \mathbb{R}^{d \times d}$$

**Properties of $\hat{\Sigma}$:**
- **Symmetric**: $\hat{\Sigma}_{jk} = \hat{\Sigma}_{kj} = s_{jk}$
- **Positive semi-definite (PSD)**: $\mathbf{v}^\top \hat{\Sigma} \mathbf{v} \geq 0$ for all $\mathbf{v}$. Proof: $\mathbf{v}^\top \hat{\Sigma} \mathbf{v} = \frac{1}{n-1}\|\mathbf{X}_c \mathbf{v}\|^2 \geq 0$.
- **Rank**: $\text{rank}(\hat{\Sigma}) \leq \min(d, n-1)$. If $n < d$, the matrix is **rank-deficient** — it cannot be inverted. This is the classical $p \gg n$ problem.
- **Diagonal entries**: $\hat{\Sigma}_{jj} = s_j^2$ (sample variance of feature $j$)

**For AI:** The empirical covariance matrix is the key object in PCA (eigendecomposition of $\hat\Sigma$ gives principal components), in Gaussian discriminant analysis, in the Mahalanobis distance (§5.3), and in LoRA-style low-rank adaptation (the weight update $\Delta W = BA$ is designed so that the effective covariance of updates is low-rank, matching the low intrinsic dimensionality of fine-tuning data).

### 5.2 Correlation Matrix and Visualisation

**Definition (Sample Correlation Matrix).** Standardise each column: $\tilde{x}_{ij} = (x_{ij} - \bar{x}_j)/s_j$. Then:

$$R = \frac{1}{n-1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} \in \mathbb{R}^{d \times d}$$

Properties: $R_{jj} = 1$ for all $j$; $R_{jk} = r_{jk} \in [-1,1]$; $R$ is PSD with diagonal ones.

The correlation matrix is related to the covariance matrix by $R_{jk} = \hat{\Sigma}_{jk} / (\hat{\sigma}_j \hat{\sigma}_k)$, i.e., $R = D^{-1/2} \hat{\Sigma} D^{-1/2}$ where $D = \text{diag}(\hat{\Sigma})$.

**Visualisation:** A heatmap of $R$ with a diverging colormap (e.g., `RdBu_r`) centred at 0 is the standard tool. Hierarchical clustering of rows/columns groups correlated features.

**For AI:** Analysing the correlation matrix of **attention heads** in a transformer layer reveals which heads focus on similar token relationships (high $r$) and which are functionally diverse (low $r$). Correlation matrices of **model weights** across layers are used in mechanistic interpretability to study how information is processed. High inter-layer weight correlation often signals redundancy — the basis for pruning.

### 5.3 Mahalanobis Distance

**Definition (Mahalanobis Distance).** The Mahalanobis distance from a point $\mathbf{x} \in \mathbb{R}^d$ to the sample centre $\hat{\boldsymbol{\mu}}$ is:

$$d_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \hat{\boldsymbol{\mu}})^\top \hat{\Sigma}^{-1} (\mathbf{x} - \hat{\boldsymbol{\mu}})}$$

**Interpretation:** The Mahalanobis distance is the Euclidean distance after **whitening** — transforming the data so that all features have unit variance and zero correlation. Equivalently, it accounts for the scale and correlation structure of the data: a point that is 3 standard deviations away along a low-variance direction is farther (in Mahalanobis terms) than 3 standard deviations along a high-variance direction.

- If $\hat{\Sigma} = I$, then $d_M(\mathbf{x}) = \|\mathbf{x} - \hat{\boldsymbol{\mu}}\|$ (ordinary Euclidean distance).
- For multivariate Gaussian data, $d_M(\mathbf{x})^2 \sim \chi^2(d)$ — so $d_M > \sqrt{\chi^2_{0.975, d}}$ flags multivariate outliers.

**For AI:** The Mahalanobis distance is used as an **out-of-distribution detection score** (Lee et al., 2018): compute $d_M(\mathbf{h})$ of a test example's penultimate-layer embedding relative to the training-set class means and covariances. OOD examples have large Mahalanobis distance. This outperforms simple Euclidean-distance methods and is used in production OOD monitoring for deployed LLMs.

### 5.4 Dimensionality and the Curse

As $d$ grows, the geometry of data changes in counterintuitive ways that make descriptive statistics increasingly unreliable.

**Concentration of pairwise distances.** In high dimensions, all pairwise Euclidean distances $\|x_i - x_j\|$ concentrate around the same value: $\text{Var}(\|x_i-x_j\|) / \mathbb{E}[\|x_i-x_j\|]^2 \to 0$ as $d \to \infty$. This means nearest-neighbour methods, clustering, and any distance-based statistic become meaningless — everything is equally far from everything else.

**Covariance matrix estimation degrades.** Estimating $\hat\Sigma \in \mathbb{R}^{d \times d}$ requires $O(d^2)$ parameters from $n$ observations. For $n < d^2$, the estimator is noise-dominated. This is the classical $p \gg n$ problem. Solution: regularised covariance (Ledoit-Wolf), diagonal approximation, or sparse graphical models.

**Implication for attention.** The softmax attention mechanism computes $d_k$-dimensional dot products. As $d_k$ grows, dot products concentrate (variance scales as $d_k$), causing the softmax to saturate (near-uniform or near-one-hot attention). The $1/\sqrt{d_k}$ scaling in scaled dot-product attention directly addresses this — it's a variance normalisation step grounded in descriptive statistics.

---

## 6. Data Standardisation and Preprocessing

### 6.1 Z-Score Standardisation

**Definition (Z-Score Standardisation / Standard Scaling).** Transform each observation $x_i$ to:

$$z_i = \frac{x_i - \bar{x}}{s}$$

The result has sample mean $\bar{z} = 0$ and sample standard deviation $s_z = 1$.

**Why it matters for ML:** Gradient descent convergence depends critically on the scale of features. If one feature ranges over $[0, 10^6]$ and another over $[0, 1]$, the gradient of the loss with respect to the first weight is $10^6 \times$ larger — the loss landscape is elongated, and gradient descent follows a zigzag path rather than heading directly toward the minimum. Standardising features makes the loss landscape more spherical, dramatically improving gradient descent convergence.

**Implementation note:** **Always fit the scaler on the training set only**, then apply the same mean and standard deviation to the validation and test sets. Fitting on the full dataset (including test) is **data leakage** — it allows test-set information to influence preprocessing.

**Properties:**
- The z-score is dimensionless (units cancel).
- Does not change the shape of the distribution (skewness and kurtosis are invariant).
- Sensitive to outliers (outliers inflate $s$, compressing the bulk of the data toward zero).

### 6.2 Min-Max and Robust Scaling

**Definition (Min-Max Scaling).** Maps to $[0, 1]$:

$$x_i^{\text{mm}} = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$

Useful when features must lie in a fixed range (e.g., inputs to sigmoid activations, image pixel values). Breakdown point: $1/n$ (one extreme outlier shifts the entire scale).

**Definition (Robust Scaling).** Uses median and IQR instead of mean and standard deviation:

$$x_i^{\text{rob}} = \frac{x_i - \tilde{x}}{\text{IQR}}$$

Result has sample median 0 and IQR 1. Breakdown point: 25%. Appropriate when the data is heavily contaminated.

**Decision guide:**

```
SCALING METHOD SELECTION
════════════════════════════════════════════════════════════════════════

  Data has outliers?  ──Yes──▶  Robust Scaling (median / IQR)

  No outliers, and
  need [0,1] range?  ──Yes──▶  Min-Max Scaling

  No outliers, and
  need zero mean,
  unit variance?     ──────▶  Z-Score Standardisation (default)

════════════════════════════════════════════════════════════════════════
```

### 6.3 Normalisation in Neural Networks

Modern neural networks embed descriptive statistics directly into their architecture through normalisation layers. Every such layer computes sample statistics — the difference lies in *which* samples are used.

**Batch Normalisation (Ioffe & Szegedy, 2015).** For a feature channel $j$ across a mini-batch of $n$ examples:

$$\hat{h}_{ij} = \frac{h_{ij} - \hat{\mu}_j^{(B)}}{\sqrt{(\hat{\sigma}_j^{(B)})^2 + \epsilon}}, \quad \text{BN}(h_{ij}) = \gamma_j \hat{h}_{ij} + \beta_j$$

where $\hat{\mu}_j^{(B)} = \frac{1}{n}\sum_{i=1}^n h_{ij}$ and $(\hat{\sigma}_j^{(B)})^2 = \frac{1}{n}\sum_{i=1}^n (h_{ij} - \hat{\mu}_j^{(B)})^2$. These are sample statistics **across the batch** for a fixed feature. Parameters $\gamma_j, \beta_j$ are learned.

**Layer Normalisation (Ba et al., 2016).** For a single token embedding $\mathbf{h}_i \in \mathbb{R}^d$:

$$\hat{h}_{ij} = \frac{h_{ij} - \hat{\mu}_i^{(L)}}{\sqrt{(\hat{\sigma}_i^{(L)})^2 + \epsilon}}, \quad \text{LN}(\mathbf{h}_i) = \boldsymbol{\gamma} \odot \hat{\mathbf{h}}_i + \boldsymbol{\beta}$$

where $\hat{\mu}_i^{(L)} = \frac{1}{d}\sum_{j=1}^d h_{ij}$ and $(\hat{\sigma}_i^{(L)})^2 = \frac{1}{d}\sum_{j=1}^d (h_{ij} - \hat{\mu}_i^{(L)})^2$. These are sample statistics **across features** for a fixed token. Used in all transformer models.

**RMS Norm (Zhang & Sennrich, 2019).** Used in LLaMA, Mistral, Gemma:

$$\text{RMSNorm}(\mathbf{h}_i) = \frac{\mathbf{h}_i}{\text{RMS}(\mathbf{h}_i)} \odot \boldsymbol{\gamma}, \quad \text{RMS}(\mathbf{h}_i) = \sqrt{\frac{1}{d}\sum_{j=1}^d h_{ij}^2}$$

This is standardisation without mean-centering: divide by the root mean square (which is the standard deviation when the mean is 0).

```
NORMALISATION LAYERS — WHAT STATISTICS THEY COMPUTE
════════════════════════════════════════════════════════════════════════

            Batch Norm    Layer Norm    RMS Norm   Group Norm
  ───────────────────────────────────────────────────────────
  Statistic  mean, var     mean, var     rms only   mean, var
  Over       batch × H×W   feature dim  feature dim feature groups
  Per        channel       token         token      group per token
  Train ≠    Yes (running  No            No         No
  Inference  stats used)
  Seq vary   Breaks for    Works fine    Works fine Works fine
  length     variable len
  ───────────────────────────────────────────────────────────
  Used in    CNNs, ResNet  All Transformers  LLaMA  CNNs+some
                           (GPT, BERT, etc)  Mistral transformers

════════════════════════════════════════════════════════════════════════
```

### 6.4 Handling Missing Data

Missing data is the rule rather than the exception in real ML datasets. The strategy for handling it depends on the **missing data mechanism**:

- **MCAR** (Missing Completely At Random): missingness is independent of all variables. Safe to delete rows or use simple imputation.
- **MAR** (Missing At Random): missingness depends on observed variables but not on the missing value itself. Model-based imputation works.
- **MNAR** (Missing Not At Random): missingness depends on the missing value (e.g., high-income people don't report income). Simple imputation is biased; need specialised methods.

**Simple imputation strategies:**

| Strategy | Use for | Advantage | Disadvantage |
|---|---|---|---|
| Mean imputation | Continuous, MCAR | Fast, simple | Reduces variance; biased for skewed data |
| Median imputation | Continuous, outliers | Robust | Reduces variance |
| Mode imputation | Categorical | Simple | Biases category frequencies |
| Forward fill | Time series | Preserves temporal structure | Wrong for gaps > 1 step |

**For AI:** Missing data in training datasets often encodes information that models will exploit as a spurious correlation. If high-income records are more likely to have missing income fields, a model that has access to missingness indicators may learn to predict high income from missing income — a label leakage that fails catastrophically when data collection improves.

---

## 7. Visualisation for EDA

### 7.1 Univariate Plots

**Histogram.** Bins the range $[x_{\min}, x_{\max}]$ into $k$ equal intervals and counts observations in each. Choice of $k$ is critical:

- **Sturges' rule**: $k = \lceil \log_2 n \rceil + 1$. Simple; undersmooths for large $n$.
- **Scott's rule**: $h = 3.49 \cdot s \cdot n^{-1/3}$. Optimal for Gaussian data.
- **Freedman-Diaconis rule**: $h = 2 \cdot \text{IQR} \cdot n^{-1/3}$. Robust to outliers; best general choice.

**Kernel Density Estimate (KDE).** A smooth non-parametric estimate of the PDF:

$$\hat{f}_h(x) = \frac{1}{nh} \sum_{i=1}^n K\!\left(\frac{x - x_i}{h}\right)$$

where $K$ is a kernel (typically Gaussian) and $h$ is the bandwidth. The optimal bandwidth under a Gaussian kernel for Gaussian data is Silverman's rule: $h = 1.06 \cdot s \cdot n^{-1/5}$.

**Box Plot (Tukey, 1977).** Shows five-number summary with: box from $Q_1$ to $Q_3$; median line; whiskers extending to the most extreme non-outlier points (within $1.5 \times \text{IQR}$ of the box); dots for outliers beyond the fences.

**Empirical CDF.** Plot $\hat{F}_n(x)$ as a step function. Advantages over histogram: no binning choice; shows quantiles directly; convergence to $F$ is guaranteed by Glivenko-Cantelli.

**Violin Plot.** Combines a box plot with a mirrored KDE — shows both summary statistics and distributional shape. Preferred over box plots when sample size is sufficient to estimate the density reliably ($n \gtrsim 100$).

### 7.2 Bivariate and Multivariate Plots

**Scatter Plot.** Plot $y_i$ vs $x_i$ for all $i$. Add a regression line and confidence band. Colour by a third variable. The go-to tool for bivariate exploratory analysis.

**Pair Plot (Scatter Matrix).** For a $d$-dimensional dataset, create a $d \times d$ grid where $(j,k)$ cell shows the scatter plot of feature $j$ vs feature $k$; diagonal cells show histograms or KDEs. Standard library: `seaborn.pairplot`.

**Correlation Heatmap.** Display $R \in [-1,1]^{d \times d}$ as a colour-coded matrix. Use a diverging colormap centred at 0 (e.g., `coolwarm` or `RdBu_r`). Cluster rows/columns by hierarchical clustering to reveal block structure.

**Parallel Coordinates.** Each axis corresponds to one feature; each observation is a polyline connecting its values across all axes. Reveals clusters and correlations in high-dimensional data without reducing dimensionality.

### 7.3 Assessing Normality

Many statistical methods (t-tests, confidence intervals, OLS regression) assume normally distributed data. Assessing normality is therefore a prerequisite for applying them.

**Q-Q Plot (Quantile-Quantile Plot).** Plot theoretical Gaussian quantiles (x-axis) against sample quantiles (y-axis). If the data is Gaussian, points lie on the line $y = x$. Deviations reveal:
- S-shaped curve → lighter tails than Gaussian (platykurtic)
- Bow-shaped curve → heavier tails (leptokurtic, common in financial and gradient data)
- Points above the line at the top → right skew

**Probability Plot.** Same idea as Q-Q but with standardised axes; equivalent to Q-Q for location-scale families.

**Shapiro-Wilk Test (preview).** A formal hypothesis test for normality. $H_0$: data is Gaussian; $W$ statistic close to 1 supports normality; small $p$-value rejects normality.

> **Forward reference:** Full hypothesis tests for distributional assumptions — Shapiro-Wilk, Kolmogorov-Smirnov, Anderson-Darling — are covered in [§03 Hypothesis Testing](../03-Hypothesis-Testing/notes.md).

---

## 8. Applications in Machine Learning

### 8.1 Batch Normalisation and Layer Normalisation

The two dominant normalisation strategies in deep learning are directly grounded in the sample statistics developed in §2.

**Batch Norm workflow:**

1. For each feature channel $j$ in a mini-batch of size $n$:
   - Compute $\hat{\mu}_j = \frac{1}{n}\sum_i h_{ij}$ (sample mean, $n$ divisor, NOT $n-1$)
   - Compute $\hat{\sigma}_j^2 = \frac{1}{n}\sum_i (h_{ij} - \hat{\mu}_j)^2$ (biased sample variance, MLE)
   - Normalise: $\hat{h}_{ij} = (h_{ij} - \hat{\mu}_j) / \sqrt{\hat{\sigma}_j^2 + \epsilon}$
2. Scale and shift: $\text{BN}(h_{ij}) = \gamma_j \hat{h}_{ij} + \beta_j$
3. At test time, use exponential moving averages $\bar{\mu}_j$ and $\bar{\sigma}_j^2$ accumulated during training.

**Why Batch Norm uses biased variance ($n$ not $n-1$):** Batch Norm is not trying to estimate the population variance; it is normalising the activations within the current mini-batch. Unbiasedness is not the goal — numerical stability and consistent normalisation are. The $\epsilon$ term (typically $10^{-5}$) prevents division by zero.

**Layer Norm workflow:**

1. For each token embedding $\mathbf{h}_i \in \mathbb{R}^d$ in the sequence:
   - Compute $\hat{\mu}_i = \frac{1}{d}\sum_j h_{ij}$ (mean over the $d$ feature dimensions)
   - Compute $\hat{\sigma}_i^2 = \frac{1}{d}\sum_j (h_{ij} - \hat{\mu}_i)^2$
   - Normalise: $\hat{h}_{ij} = (h_{ij} - \hat{\mu}_i) / \sqrt{\hat{\sigma}_i^2 + \epsilon}$
2. Apply learned $\gamma_j$ and $\beta_j$.

**Why transformers use Layer Norm, not Batch Norm:** Batch Norm requires a fixed batch size and breaks with variable-length sequences (because the normalisation statistics change with sequence length). Layer Norm normalises within each token independently, making it sequence-length-agnostic and suitable for autoregressive generation (where batch size = 1 at inference).

**Why RMSNorm (LLaMA, Mistral, Gemma) drops the mean:** Empirically, re-centering (subtracting the mean) provides marginal benefit compared to re-scaling (dividing by RMS). RMSNorm is cheaper to compute (no mean subtraction) and achieves comparable training stability.

### 8.2 Adam Optimiser as Online Sample Statistics

The Adam optimiser (Kingma & Ba, 2015) maintains running estimates of the first and second moments of the gradient distribution, updated at each step $t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment — running mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment — running mean of squared gradient)}$$

These are **exponentially-weighted moving averages** (EWMA) of $g_t$ and $g_t^2$. Initialising $m_0 = v_0 = 0$ introduces a **bias toward zero** at early steps (analogous to the bias in the $n$ vs $n-1$ estimator for finite samples). Adam corrects this with:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

These are the **bias-corrected** moment estimates. The analogy to Bessel's correction: both corrections account for the fact that a short history ($t$ steps or $n$ observations) underestimates the true statistic.

The Adam update is:

$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

The denominator $\sqrt{\hat{v}_t}$ is an estimate of the gradient's root-mean-square — an adaptive per-parameter learning rate that scales inversely with the gradient's historical variance. Parameters with high-variance gradients get smaller updates; parameters with low-variance gradients get larger updates.

**Connection to descriptive statistics:** $\hat{m}_t$ is a smoothed estimate of the gradient mean; $\hat{v}_t$ is a smoothed estimate of the uncentred second moment (related to variance + mean²). Adam is performing online, exponentially-weighted descriptive statistics on the gradient stream.

### 8.3 Dataset Characterisation and Model Cards

The **Datasheets for Datasets** framework (Gebru et al., 2020) and Hugging Face **Dataset Cards** require quantitative documentation of every training dataset. The required statistics are directly from descriptive statistics:

- **Sample size**: $n$ (rows), $d$ (features/columns)
- **Label distribution**: frequency table or histogram for each output class
- **Feature statistics**: mean, std, min, Q1, median, Q3, max for each numeric feature
- **Missing values**: count and fraction per column
- **Correlation structure**: correlation matrix or heatmap between features and between features and label
- **Demographic representation**: distribution of sensitive attributes (age, gender, race, geography)
- **Temporal coverage**: date range, recency bias

For embedding spaces (used in retrieval-augmented generation, semantic search, and vector databases): **intra-cluster variance** (how spread out a class's embeddings are), **inter-cluster distance** (how separated different classes are), and **cosine similarity distribution** (the histogram of pairwise cosine similarities across the corpus).

### 8.4 Data Drift and Distribution Shift

In production ML systems, the data generating distribution $P_{\text{prod}}(X, Y)$ may diverge from the training distribution $P_{\text{train}}(X, Y)$. This is called **distribution shift** or **data drift**. Monitoring descriptive statistics is the first line of defence.

**Types of drift:**
- **Covariate shift**: $P(X)$ changes, $P(Y|X)$ stays the same. Feature means shift.
- **Label shift**: $P(Y)$ changes, $P(X|Y)$ stays the same. Label proportions change.
- **Concept drift**: $P(Y|X)$ changes. Same inputs now map to different outputs.

**Detection via descriptive statistics:**

| Drift type | Monitoring statistic | Alert when |
|---|---|---|
| Covariate shift | Feature means, variances | Mean shift > $k\sigma$ |
| Skewness shift | Feature skewness | $|g_1^{\text{prod}} - g_1^{\text{train}}|$ large |
| Outlier rate | Fraction > Tukey fence | Rate doubles from baseline |
| Label imbalance | Class frequency table | Class proportions shift |
| PSI (Population Stability Index) | $\sum_k (p_k - q_k)\log(p_k/q_k)$ | PSI > 0.2 |

The **Population Stability Index (PSI)** is the symmetric KL divergence between the training and production distributions, computed on binned data — a descriptive statistic that quantifies how much the overall distribution has shifted.

> **Forward reference:** Formal hypothesis tests for detecting drift (KS test, two-sample t-test, chi-squared test) are covered in [§03 Hypothesis Testing](../03-Hypothesis-Testing/notes.md).

---

## 9. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using the mean for skewed or heavy-tailed data | A few extreme values (income, house prices, gradient norms) pull the mean far from the "typical" value; it no longer represents the centre | Report median alongside mean; use robust statistics for skewed features |
| 2 | Using $n$ instead of $n-1$ for sample variance | Dividing by $n$ gives the MLE of $\sigma^2$ under a Gaussian, which is biased: $\mathbb{E}[s_n^2] = \frac{n-1}{n}\sigma^2 < \sigma^2$ | Always use `ddof=1` in NumPy/SciPy (`np.var(x, ddof=1)`) unless computing BN-style normalisation |
| 3 | Fitting the scaler on train+test data | The test-set statistics (mean, std) leak into the preprocessing, artificially improving reported test performance | Fit the scaler only on the training set; `transform` the test set using training-set parameters |
| 4 | Interpreting $r = 0$ as independence | Zero Pearson correlation means no **linear** association; nonlinear dependence can be arbitrarily strong | Also compute distance correlation or mutual information; always scatter-plot the pair |
| 5 | Concluding causation from correlation | Spurious correlations (Simpson's paradox, confounders, selection bias) produce $r \neq 0$ with no causal link | Use controlled experiments (A/B tests) or causal inference methods to test causal claims |
| 6 | Removing outliers without investigation | An outlier may be the most interesting data point (a fraud transaction, a rare disease case, a critical edge case for a safety model) | Investigate each outlier's source before deciding to remove vs. keep vs. separate |
| 7 | Standardising after train/test split but using test statistics | Same as mistake 3 in reverse — computing the standard deviation on the test set introduces leakage | Always use training statistics for all scaling |
| 8 | Using Pearson correlation for ordinal data | Ordinal data (ratings 1–5, Likert scales) has rank order but not equal intervals; Pearson assumes interval scale | Use Spearman $\rho_S$ or Kendall $\tau$ for ordinal data |
| 9 | Ignoring the dimensionality of the covariance matrix | For $n < d$, $\hat\Sigma$ is rank-deficient and non-invertible; Mahalanobis distance and PCA are undefined or numerically unstable | Regularise: add $\lambda I$ (ridge), use diagonal approximation, or reduce dimensionality first |
| 10 | Reporting only mean accuracy as the evaluation metric | Mean accuracy aggregates over all classes; for imbalanced data (99% negative class), a trivial classifier gets 99% accuracy | Also report median, per-class accuracy, precision, recall, F1; use stratified evaluation |
| 11 | Choosing histogram bin width arbitrarily | Too few bins hides multimodality; too many bins creates noise that looks like structure | Use Freedman-Diaconis rule; compare histogram to KDE; be explicit about bin choice |
| 12 | Confusing sample skewness and distribution skewness | High sample skewness $g_1$ for small $n$ may be noise, not a true distributional property | Check confidence intervals for $g_1$; use bootstrap to assess sampling variability |

---

## 10. Exercises

**Exercise 1 ★ — Central Tendency and Spread**
Given the dataset $\{2, 4, 4, 4, 5, 5, 7, 9, 9, 100\}$:
(a) Compute mean, median, and mode.
(b) Compute sample variance (with Bessel correction), standard deviation, and IQR.
(c) Explain which measure of centre is most representative of the data and why.
(d) Remove the outlier 100 and recompute all statistics. By what factor does the mean change? The median?

**Exercise 2 ★ — Five-Number Summary and Outlier Detection**
Generate 200 observations from $\mathcal{N}(0,1)$ with 5 additional outliers at $\{-8, -7, 7, 8, 9\}$.
(a) Compute the five-number summary.
(b) Identify outliers using Tukey fences ($1.5 \times \text{IQR}$).
(c) Identify outliers using the modified Z-score with MAD.
(d) Compare which method flags more false positives on the clean Gaussian data (benchmark: $\hat F$ critical values for $n=200$).

**Exercise 3 ★ — Empirical CDF and Q-Q Plot**
Generate 100 samples from a $t_3$ (Student-$t$ with 3 degrees of freedom) distribution.
(a) Construct and plot the empirical CDF $\hat{F}_n(x)$.
(b) Construct a Q-Q plot against the Gaussian distribution.
(c) Describe what the Q-Q plot reveals about the tails of the $t_3$ distribution relative to the Gaussian.
(d) Compute the excess kurtosis $g_2$ and interpret the result.

**Exercise 4 ★★ — Robust Statistics Under Contamination**
Start with 95 samples from $\mathcal{N}(0,1)$. Add 5 outliers drawn from $\mathcal{N}(20, 1)$ (contamination fraction 5%).
(a) Compare mean vs median vs 10%-trimmed mean on the contaminated dataset. Which is closest to 0?
(b) Compare std vs MAD vs IQR. Which is closest to 1?
(c) Compute the influence of the single largest outlier on each statistic: replace it with $10^6$ and measure the change.
(d) Plot the estimates as a function of contamination fraction $\alpha \in \{0, 0.05, 0.1, 0.25, 0.4, 0.49\}$.

**Exercise 5 ★★ — Covariance Matrix and Mahalanobis Distance**
Generate 200 samples from $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ with $\boldsymbol{\mu} = (2, -1)^\top$ and $\Sigma = \begin{pmatrix}4 & 3 \\ 3 & 9\end{pmatrix}$.
(a) Compute the sample mean vector $\hat{\boldsymbol{\mu}}$ and covariance matrix $\hat{\Sigma}$. Verify they are close to the true values.
(b) Verify that $\hat{\Sigma}$ is PSD.
(c) Compute the Mahalanobis distance from each point to $\hat{\boldsymbol{\mu}}$. Plot a histogram of $d_M^2$.
(d) Compare the histogram to a $\chi^2(2)$ distribution. What percentage of points exceed the 97.5th percentile?

**Exercise 6 ★★ — Anscombe's Quartet**
Implement all four of Anscombe's datasets from scratch (exact values from the original paper).
(a) Compute and verify that all four datasets have the same mean $\bar{x} = 9$, $\bar{y} = 7.5$, variance $s_x^2 = 11$, $s_y^2 \approx 4.12$, and Pearson correlation $r \approx 0.816$.
(b) Fit a linear regression $\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$ to each dataset and verify all four give the same coefficients.
(c) Plot all four scatter plots with the fitted line.
(d) Compute skewness and kurtosis for each dataset's residuals. Do they differ? What does this reveal?

**Exercise 7 ★★★ — Batch Norm vs Layer Norm from Scratch**
(a) Implement `batch_norm(X, gamma, beta, eps=1e-5)` where `X` has shape `(batch, features)`, normalising across the batch dimension.
(b) Implement `layer_norm(X, gamma, beta, eps=1e-5)`, normalising across the feature dimension.
(c) Implement `rms_norm(X, gamma, eps=1e-5)`, normalising by RMS only (no mean subtraction).
(d) Apply all three to a random matrix of shape $(32, 512)$. Verify: for BN, each column (feature) has mean≈0, std≈1; for LN, each row (sample) has mean≈0, std≈1; for RMSNorm, each row has RMS≈1.
(e) Verify that when applied to variable-batch-size inputs (1, 4, 32), BN statistics change but LN statistics are invariant. Discuss implications for inference.

**Exercise 8 ★★★ — Adam Momentum as Running Sample Statistics**
(a) For a gradient sequence $g_1, \ldots, g_{1000}$ drawn from $\mathcal{N}(0.1, 1.0)$, compute Adam's first moment $m_t$ and second moment $v_t$ with $\beta_1 = 0.9$, $\beta_2 = 0.999$.
(b) Plot the bias-corrected estimates $\hat{m}_t$ and $\hat{v}_t$ alongside the true running mean and uncentred second moment.
(c) Show that without bias correction, $m_t$ underestimates $\mathbb{E}[g]$ by a factor of $(1-\beta_1^t)$ at step $t$.
(d) Vary $\beta_1 \in \{0.5, 0.9, 0.99\}$ and plot how quickly $\hat{m}_t$ converges to the true mean. Connect this to the "effective window size" $1/(1-\beta_1)$.

---

## 11. Why This Matters for AI (2026 Perspective)

| Concept | AI / LLM Application | Impact |
|---|---|---|
| Sample mean and variance | Batch Norm, Layer Norm, RMS Norm in every transformer | Core to training stability; without it, deep networks diverge |
| Robust statistics (MAD, trimmed mean) | Byzantine-robust federated learning; gradient aggregation under adversarial workers | Enables privacy-preserving training that resists poisoning attacks |
| Gradient clipping = Winsorisation | Training GPT-4, LLaMA, Mistral; prevents loss spikes | Without gradient clipping, large models regularly diverge during training |
| Adam = online sample statistics | Universal optimiser for neural networks (90%+ of published LLM training) | Bias-corrected moment estimates make adaptive learning rates work |
| Outlier detection | OOD detection, anomaly detection in production model monitoring | Flagging adversarial or corrupted inputs before they cause harmful outputs |
| Covariance matrix + Mahalanobis distance | OOD detection from penultimate-layer embeddings (Lee et al. 2018) | State-of-the-art OOD scoring without retraining the model |
| Correlation matrix | Attention head redundancy analysis; mechanistic interpretability | Identifies which attention heads can be pruned to reduce inference cost |
| $1/\sqrt{d_k}$ attention scaling | Prevents softmax saturation in high-dimensional attention (FlashAttention, MLA) | Directly derived from variance concentration in high dimensions |
| Empirical CDF + quantile functions | Calibration evaluation; Wasserstein distance for dataset comparison | Measuring whether a model's confidence matches its accuracy |
| Standardisation (Z-score) | Feature preprocessing for all tabular ML; embedding normalisation | Ensures equal-scale features; required for SGD convergence |
| Skewness and kurtosis | Gradient noise modelling; heavy-tailed gradient distributions | Justifies Student-$t$ noise models over Gaussian for stochastic gradient methods |
| Label distribution (frequency table) | Class-weighted loss, oversampling, dataset cards for responsible AI | Prevents models from learning to predict the majority class trivially |
| Data drift (PSI, KL) | Production monitoring for deployed LLMs and recommendation systems | Early warning system for model degradation; triggers retraining |
| Simpson's paradox + confounders | Fairness auditing; understanding why a model exhibits bias | Correlation-based bias is masked by aggregation; subgroup analysis reveals it |

---

## 12. Conceptual Bridge

Descriptive statistics is the foundation on which all statistical inference rests. To draw any conclusion from data — to estimate a parameter, test a hypothesis, train a Bayesian model, or fit a regression — you must first understand what the data actually looks like. This section has provided the complete vocabulary: the mean, median, and mode as measures of centre; variance, standard deviation, IQR, and MAD as measures of spread; skewness and kurtosis as measures of shape; quantiles and the ECDF as complete distributional summaries; covariance and correlation as measures of association; and the covariance matrix and Mahalanobis distance as multivariate generalisations.

The next section, [§02 Estimation Theory](../02-Estimation-Theory/notes.md), takes the sample statistics developed here and asks: **how good are they as estimators of the underlying population parameters?** The sample mean $\bar{x}$ is an estimator of $\mu$ — but what is its bias? What is its variance? Is there a better estimator? The Cramér-Rao lower bound sets a fundamental limit on how precisely any unbiased estimator can estimate a parameter, expressed in terms of Fisher information. Bessel's correction (§2.3) is a preview of estimator theory: the $n-1$ divisor corrects the bias of the sample variance estimator.

Looking back, this section builds directly on probability theory. The **population parameters** $\mu = \mathbb{E}[X]$, $\sigma^2 = \text{Var}(X)$, and $\rho = \text{Corr}(X,Y)$ were defined in [Chapter 6 §04 Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md). What we compute here are their **sample analogues** — functions of observed data designed to estimate those population quantities. The Glivenko-Cantelli theorem (§2.5) provides the theoretical guarantee connecting the ECDF to the population CDF: it is a specialisation of the law of large numbers proved in [Ch6 §06 Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md).

```
POSITION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  CHAPTER 6: PROBABILITY THEORY
  ┌────────────────────────────────────────────────────────────────┐
  │  §01 Intro + RVs  →  §02 Distributions  →  §03 Joint         │
  │  §04 Expectation  →  §05 Concentration  →  §06 Stochastic    │
  │  §07 Markov Chains                                             │
  │  (population parameters μ, σ², ρ defined here)                │
  └──────────────────────────────────┬─────────────────────────────┘
                                     │ sample analogues
                                     ▼
  CHAPTER 7: STATISTICS
  ┌────────────────────────────────────────────────────────────────┐
  │  §01 DESCRIPTIVE STATISTICS  ◄── YOU ARE HERE                  │
  │  "Summarise data before you model it"                          │
  │  x̄, s², r, ECDF, IQR, MAD, Σ̂, Mahalanobis, z-score          │
  └──────────────────────────────────┬─────────────────────────────┘
                                     │ "how good are these estimators?"
                                     ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  §02 Estimation Theory                                          │
  │  Bias, variance, MSE, Cramér-Rao, MLE, Fisher information      │
  └──────────────────────────────────┬─────────────────────────────┘
                                     │ "make decisions from estimates"
                                     ▼
  ┌────────────────────────────────────────────────────────────────┐
  │  §03 Hypothesis Testing  →  §04 Bayesian Inference             │
  │  §05 Time Series  →  §06 Regression Analysis                   │
  └────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

---

[← Back to Chapter 7: Statistics](../README.md) | [Next: Estimation Theory →](../02-Estimation-Theory/notes.md)

---

## Appendix A: Worked Examples

### A.1 Computing All Univariate Statistics by Hand

**Dataset:** $\{3, 7, 7, 13, 2, 8, 11, 1, 14, 7\}$ ($n = 10$)

**Step 1 — Sort:** $\{1, 2, 3, 7, 7, 7, 8, 11, 13, 14\}$

**Step 2 — Central tendency:**
- Mean: $\bar{x} = (1+2+3+7+7+7+8+11+13+14)/10 = 73/10 = 7.3$
- Median ($n=10$, even): $Q_2 = (x_{(5)} + x_{(6)})/2 = (7+7)/2 = 7.0$
- Mode: $7$ (appears 3 times)

**Step 3 — Spread:**
- Deviations: $(-6.3, -5.3, -4.3, -0.3, -0.3, -0.3, 0.7, 3.7, 5.7, 6.7)$
- Squared deviations sum: $39.69 + 28.09 + 18.49 + 0.09 + 0.09 + 0.09 + 0.49 + 13.69 + 32.49 + 44.89 = 178.10$
- Sample variance: $s^2 = 178.10/9 = 19.79$; $s = 4.45$
- Range: $14 - 1 = 13$
- $Q_1 = (2+3)/2 = 2.5$; $Q_3 = (11+13)/2 = 12.0$; IQR $= 9.5$

**Step 4 — Shape:**
- Skewness $g_1$: the distribution is nearly symmetric (mean ≈ median ≈ mode $= 7$); $g_1 \approx 0.08$ (very slight right skew due to 14)
- All values within Tukey fences: $[2.5 - 14.25, 12 + 14.25] = [-11.75, 26.25]$ — no outliers

**Step 5 — Five-number summary:** $\{1, 2.5, 7.0, 12.0, 14\}$

### A.2 Pearson vs Spearman on a Monotone Nonlinear Dataset

Consider $x_i = i$ for $i = 1, \ldots, 10$ and $y_i = x_i^3$:

- $x = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)$, $y = (1, 8, 27, 64, 125, 216, 343, 512, 729, 1000)$
- Pearson $r = 0.974$ (strong, but $y$ is highly curved so $r < 1$)
- Spearman $\rho_S = 1.000$ (perfect rank correlation — the mapping is monotone)

This illustrates: Spearman detects the perfect monotone relationship that Pearson misses.

Now consider $x_i = i$ and $y_i = \sin(x_i)$ for $i = 1, \ldots, 20$:
- Pearson $r \approx -0.17$: looks like no association
- Spearman $\rho_S \approx -0.19$: same conclusion

But the relationship is highly structured — it's non-monotone. Both Pearson and Spearman fail to detect it. This is why visualisation is always the first step.

### A.3 Covariance Matrix — A Complete Worked Example

**Dataset:** Standardised heights (cm), weights (kg), age (years) for 5 individuals:

$$\mathbf{X} = \begin{pmatrix} 170 & 65 & 25 \\ 165 & 58 & 30 \\ 180 & 80 & 28 \\ 175 & 72 & 35 \\ 160 & 55 & 22 \end{pmatrix}$$

- $\hat{\boldsymbol{\mu}} = (170, 66, 28)$

Centred matrix $\mathbf{X}_c$:
$$\mathbf{X}_c = \begin{pmatrix} 0 & -1 & -3 \\ -5 & -8 & 2 \\ 10 & 14 & 0 \\ 5 & 6 & 7 \\ -10 & -11 & -6 \end{pmatrix}$$

$$\hat{\Sigma} = \frac{1}{4} \mathbf{X}_c^\top \mathbf{X}_c = \begin{pmatrix} 62.5 & 89.5 & 6.25 \\ 89.5 & 134.5 & 11.5 \\ 6.25 & 11.5 & 24.5 \end{pmatrix}$$

Interpretation:
- Height and weight have high positive covariance ($89.5$): taller people tend to weigh more
- Height and age have low positive covariance ($6.25$): weak association in this sample
- Diagonal: variances are $62.5$, $134.5$, $24.5$ — weight is most variable

Correlation matrix: $R_{12} = 89.5/\sqrt{62.5 \cdot 134.5} = 0.976$; height and weight are very strongly correlated.

---

## Appendix B: Key Formulas Quick Reference

```
DESCRIPTIVE STATISTICS — FORMULA REFERENCE
════════════════════════════════════════════════════════════════════════

  CENTRAL TENDENCY
  ─────────────────
  Sample mean:       x̄ = (1/n) Σ xᵢ
  Sample median:     middle value of sorted data
  Trimmed mean:      mean of middle (1-2α)n observations

  SPREAD
  ────────
  Sample variance:   s² = (1/(n-1)) Σ(xᵢ - x̄)²
  Std deviation:     s = √s²
  IQR:               Q₃ - Q₁
  MAD:               median(|xᵢ - median(x)|)
  Robust σ̂:          1.4826 × MAD

  SHAPE
  ───────
  Skewness:          g₁ = m₃ / m₂^(3/2),   mₖ = (1/n)Σ(xᵢ-x̄)ᵏ
  Excess kurtosis:   g₂ = m₄ / m₂² - 3

  QUANTILES
  ──────────
  Empirical CDF:     F̂ₙ(x) = (1/n)Σ 1[xᵢ ≤ x]
  p-quantile:        Q(p) = inf{x : F̂ₙ(x) ≥ p}
  Five-number:       {min, Q₁, median, Q₃, max}

  BIVARIATE
  ──────────
  Covariance:        sxy = (1/(n-1)) Σ(xᵢ-x̄)(yᵢ-ȳ)
  Pearson r:         r = sxy / (sx sy)
  Spearman ρ:        Pearson r applied to ranks

  MULTIVARIATE
  ─────────────
  Mean vector:       μ̂ = (1/n) Xᵀ 1
  Covariance matrix: Σ̂ = (1/(n-1)) Xcᵀ Xc
  Correlation matrix: R = D^{-1/2} Σ̂ D^{-1/2}
  Mahalanobis dist:  d_M(x) = √[(x-μ̂)ᵀ Σ̂⁻¹(x-μ̂)]

  STANDARDISATION
  ─────────────────
  Z-score:           zᵢ = (xᵢ - x̄) / s
  Min-max:           xᵢ_mm = (xᵢ - min) / (max - min)
  Robust:            xᵢ_rob = (xᵢ - median) / IQR

  NORMALISATION LAYERS
  ─────────────────────
  Batch Norm:        ĥᵢⱼ = (hᵢⱼ - μ̂ⱼᴮ) / √(σ̂ⱼᴮ² + ε)   [over batch]
  Layer Norm:        ĥᵢⱼ = (hᵢⱼ - μ̂ᵢᴸ) / √(σ̂ᵢᴸ² + ε)   [over features]
  RMS Norm:          ĥᵢⱼ = hᵢⱼ / RMS(hᵢ)                  [no mean subtraction]

  ADAM MOMENTS
  ─────────────
  First moment:      mₜ = β₁mₜ₋₁ + (1-β₁)gₜ
  Second moment:     vₜ = β₂vₜ₋₁ + (1-β₂)gₜ²
  Bias correction:   m̂ₜ = mₜ/(1-β₁ᵗ),   v̂ₜ = vₜ/(1-β₂ᵗ)
  Update:            θₜ₊₁ = θₜ - α·m̂ₜ / (√v̂ₜ + ε)

════════════════════════════════════════════════════════════════════════
```

---

## Appendix C: Proofs and Derivations

### C.1 Proof: Sample Mean is Unbiased

**Claim:** $\mathbb{E}[\bar{x}] = \mu$ where $\bar{x} = \frac{1}{n}\sum_{i=1}^n X_i$ and $X_i \overset{\text{iid}}{\sim} P$ with mean $\mu$.

**Proof:**
$$\mathbb{E}[\bar{x}] = \mathbb{E}\!\left[\frac{1}{n}\sum_{i=1}^n X_i\right] = \frac{1}{n}\sum_{i=1}^n \mathbb{E}[X_i] = \frac{1}{n} \cdot n\mu = \mu \quad \square$$

This uses linearity of expectation and the iid assumption ($\mathbb{E}[X_i] = \mu$ for all $i$).

### C.2 Proof: Variance of the Sample Mean

**Claim:** $\text{Var}(\bar{x}) = \sigma^2/n$.

**Proof:** By independence ($\text{Cov}(X_i, X_j) = 0$ for $i \neq j$):
$$\text{Var}(\bar{x}) = \text{Var}\!\left(\frac{1}{n}\sum X_i\right) = \frac{1}{n^2}\sum_{i=1}^n \text{Var}(X_i) = \frac{n\sigma^2}{n^2} = \frac{\sigma^2}{n} \quad \square$$

This is the **standard error** result: $\text{SE}(\bar{x}) = \sigma/\sqrt{n}$. Doubling precision requires 4× the data.

### C.3 Proof: Pearson Correlation is Bounded in $[-1,1]$

**Claim:** $|r| \leq 1$.

**Proof:** Apply Cauchy-Schwarz to the centred vectors $\mathbf{u} = (x_1-\bar{x}, \ldots, x_n-\bar{x})$ and $\mathbf{v} = (y_1-\bar{y}, \ldots, y_n-\bar{y})$:
$$|\mathbf{u} \cdot \mathbf{v}| \leq \|\mathbf{u}\| \|\mathbf{v}\|$$
$$\left|\sum_i (x_i-\bar{x})(y_i-\bar{y})\right| \leq \sqrt{\sum_i (x_i-\bar{x})^2} \cdot \sqrt{\sum_i (y_i-\bar{y})^2}$$
Dividing both sides by $(n-1)$ and by $s_x s_y$ gives $|r| \leq 1$. Equality iff $\mathbf{u} = c\mathbf{v}$ for some $c \neq 0$, i.e., $y_i - \bar{y} = c(x_i - \bar{x})$ — a perfect linear relationship. $\square$

### C.4 Proof: Sample Covariance Matrix is PSD

**Claim:** $\hat{\Sigma} = \frac{1}{n-1}\mathbf{X}_c^\top \mathbf{X}_c$ is positive semi-definite.

**Proof:** For any $\mathbf{v} \in \mathbb{R}^d$:
$$\mathbf{v}^\top \hat{\Sigma} \mathbf{v} = \frac{1}{n-1} \mathbf{v}^\top \mathbf{X}_c^\top \mathbf{X}_c \mathbf{v} = \frac{1}{n-1} \|\mathbf{X}_c \mathbf{v}\|^2 \geq 0 \quad \square$$

The matrix is positive **definite** (not just semi-definite) iff $\mathbf{X}_c$ has full column rank, which requires $n > d$ and no perfect linear dependence among features.

---

## Appendix D: Python Implementation Reference

```python
import numpy as np
from scipy import stats

# ── Univariate statistics ──────────────────────────────────────
x = np.array([...])
n = len(x)

mean     = np.mean(x)
median   = np.median(x)
mode     = stats.mode(x).mode[0]
var      = np.var(x, ddof=1)       # unbiased (Bessel's correction)
std      = np.std(x, ddof=1)
q1, q3   = np.percentile(x, [25, 75])
iqr      = q3 - q1
mad      = np.median(np.abs(x - np.median(x)))
robust_s = 1.4826 * mad
skew     = stats.skew(x)
kurt     = stats.kurtosis(x)       # excess kurtosis (Fisher)

# Five-number summary
five_num = np.array([x.min(), q1, median, q3, x.max()])

# Empirical CDF
def ecdf(x_eval, data):
    return np.mean(data[:, None] <= x_eval, axis=0)

# ── Outlier detection ──────────────────────────────────────────
# Z-score
z = (x - mean) / std
outliers_z = np.abs(z) > 3

# Tukey fences
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
outliers_tukey = (x < lower) | (x > upper)

# Modified Z-score (MAD-based)
m_z = 0.6745 * (x - np.median(x)) / mad
outliers_mad = np.abs(m_z) > 3.5

# ── Bivariate statistics ───────────────────────────────────────
x, y = np.array([...]), np.array([...])
cov    = np.cov(x, y, ddof=1)[0, 1]
pearsr = np.corrcoef(x, y)[0, 1]   # Pearson
rho_s  = stats.spearmanr(x, y).statistic
tau    = stats.kendalltau(x, y).statistic

# ── Multivariate statistics ────────────────────────────────────
X = np.array([...])   # shape (n, d)
mu_hat  = X.mean(axis=0)
Xc      = X - mu_hat
Sigma   = (Xc.T @ Xc) / (len(X) - 1)    # covariance matrix
D       = np.sqrt(np.diag(Sigma))
R       = Sigma / np.outer(D, D)          # correlation matrix

# Mahalanobis distance
Sigma_inv = np.linalg.inv(Sigma)
def mahal(x, mu, Sigma_inv):
    diff = x - mu
    return np.sqrt(diff @ Sigma_inv @ diff)

# ── Standardisation ────────────────────────────────────────────
z_score   = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
minmax    = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
q1s       = np.percentile(X, 25, axis=0)
q3s       = np.percentile(X, 75, axis=0)
robust_sc = (X - np.median(X, axis=0)) / (q3s - q1s)
```

---

## Appendix E: Connections to Other Chapters

**To Chapter 6 — Probability Theory:**
Every sample statistic here is an estimator of a population parameter defined in Ch6:
- $\bar{x}$ estimates $\mathbb{E}[X]$ (defined in [Ch6 §04](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md))
- $s^2$ estimates $\text{Var}(X)$ (defined in [Ch6 §04](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md))
- $r$ estimates $\text{Corr}(X,Y)$ (defined in [Ch6 §04](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md))
- $\hat{F}_n(x)$ estimates $F(x) = P(X \leq x)$ (defined in [Ch6 §01](../../06-Probability-Theory/01-Introduction-and-Random-Variables/notes.md))
- The Glivenko-Cantelli guarantee uses the Law of Large Numbers from [Ch6 §06](../../06-Probability-Theory/06-Stochastic-Processes/notes.md)

**To §02 — Estimation Theory:**
Descriptive statistics IS estimation theory applied to the simplest possible estimands:
- Is $\bar{x}$ a good estimator of $\mu$? (Yes — unbiased, consistent, efficient for Gaussian data)
- What is $\text{Var}(\bar{x})$? ($\sigma^2/n$ — derived from the iid assumption)
- Can we do better than $\bar{x}$ for non-Gaussian distributions? (Median is more efficient for Laplace distributions)
All of these questions are answered formally in [§02 Estimation Theory](../02-Estimation-Theory/notes.md).

**To §03 — Hypothesis Testing:**
Many descriptive statistics have associated hypothesis tests:
- Is $\mu = \mu_0$? → one-sample t-test using $\bar{x}$
- Is the data Gaussian? → Shapiro-Wilk test using skewness and kurtosis
- Is there correlation? → test that $\rho = 0$ using Fisher's $z$-transformation of $r$
- Has the distribution shifted? → KS test comparing two ECDFs
Full treatment: [§03 Hypothesis Testing](../03-Hypothesis-Testing/notes.md).

**To §06 — Regression Analysis:**
The regression coefficient $\hat{\beta}_1 = r \cdot s_y / s_x$ is a function of descriptive statistics. The OLS estimator is derived from minimising the sample variance of residuals. The $R^2$ coefficient of determination equals $r^2$ in simple linear regression. Full treatment: [§06 Regression Analysis](../06-Regression-Analysis/notes.md).

**To Chapter 3 — Advanced Linear Algebra:**
The covariance matrix $\hat\Sigma$ is a real symmetric PSD matrix. Its eigendecomposition $\hat\Sigma = Q\Lambda Q^\top$ gives the **principal component directions** (columns of $Q$) and **variance explained** per component (diagonal of $\Lambda$). This is PCA — the most important linear dimensionality reduction method in ML. Full treatment: [Ch3 §03 Eigenvalues and Eigenvectors](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md).


---

## Appendix F: Advanced Topics

### F.1 Estimating the Population Mean: The Central Limit Theorem Connection

The sample mean $\bar{x}$ is an excellent estimator, but how confident can we be in it? The Central Limit Theorem tells us that for large $n$:

$$\frac{\bar{x} - \mu}{\sigma/\sqrt{n}} \overset{d}{\to} \mathcal{N}(0, 1)$$

This means the sampling distribution of $\bar{x}$ is approximately Gaussian with mean $\mu$ and standard deviation $\sigma/\sqrt{n}$ — regardless of the distribution of individual $X_i$. The spread of $\bar{x}$ decreases as $1/\sqrt{n}$: to halve the uncertainty, you need 4× the data.

> **Backward reference:** The CLT was proved formally in [Ch6 §06 Stochastic Processes](../../06-Probability-Theory/06-Stochastic-Processes/notes.md). Here we use it to interpret the reliability of the sample mean.

**Implication for ML:** The validation loss computed over a test set of size $n$ has uncertainty $\pm z_{\alpha/2} \cdot \hat{\sigma}_{\text{loss}} / \sqrt{n}$. This is a confidence interval for the true expected loss — a fundamental fact that is frequently ignored when comparing models. Two models that differ by less than $\pm 2\text{SE}$ are not statistically distinguishable. See [§02 Estimation Theory](../02-Estimation-Theory/notes.md) and [§03 Hypothesis Testing](../03-Hypothesis-Testing/notes.md) for formal treatment.

### F.2 L-Estimators: Unifying the Zoo of Robust Statistics

The mean, trimmed mean, Winsorised mean, and median are all members of a single family called **L-estimators**: linear combinations of order statistics.

$$T_L = \sum_{i=1}^n w_i x_{(i)}$$

where the weights $w_i$ depend on the estimator:
- **Sample mean**: $w_i = 1/n$ for all $i$
- **Median** ($n$ odd): $w_m = 1$, all others $0$
- **$\alpha$-trimmed mean**: $w_i = 1/(n-2k)$ for $i \in [k+1, n-k]$, zero otherwise
- **Winsorised mean**: $w_i = 1/n$ for $i \in [k+1, n-k]$; $w_k = (k+1)/n$; $w_{n-k} = (k+1)/n$

This unification shows that robustness and efficiency are controlled by the shape of the weight sequence $\{w_i\}$: flat weights = maximum efficiency under Gaussian; zero-outside-middle weights = robustness.

### F.3 The Efficiency-Robustness Tradeoff

For Gaussian data, the sample mean is the **minimum variance unbiased estimator (MVUE)** of $\mu$ — no other unbiased estimator can achieve lower variance. The relative efficiency of the median vs the mean for Gaussian data is:

$$\text{ARE}(\tilde{x}, \bar{x}) = \frac{\text{Var}(\bar{x})}{\text{Var}(\tilde{x})} = \frac{2}{\pi} \approx 0.637$$

The median needs $1/0.637 \approx 1.57\times$ more data to achieve the same precision as the mean — for Gaussian data. But for heavy-tailed distributions (like Student-$t_\nu$ for small $\nu$), the median becomes more efficient than the mean. At the Cauchy distribution ($\nu = 1$), the sample mean is **infinitely inefficient** — its variance is infinite.

This tradeoff is fundamental to the choice between Adam and SGD: Adam's adaptive second-moment estimate essentially adapts to the local variance of the gradient distribution, making it more efficient than SGD with a fixed learning rate when gradient variance varies across parameters.

### F.4 Bootstrap Confidence Intervals for Descriptive Statistics

**The bootstrap** (Efron, 1979) provides confidence intervals for any statistic without assuming a parametric form for the distribution. For a statistic $T(x_1, \ldots, x_n)$:

1. Draw $B$ bootstrap samples: for $b = 1, \ldots, B$, sample $n$ observations with replacement from the data.
2. Compute $T^{(b)}$ on each bootstrap sample.
3. The bootstrap distribution $\{T^{(1)}, \ldots, T^{(B)}\}$ estimates the sampling distribution of $T$.
4. The 95% bootstrap CI is the $[2.5, 97.5]$ percentile interval of $\{T^{(b)}\}$.

This works for the mean, median, correlation, MAD — any statistic you can compute. For the sample mean, the bootstrap CI coincides with the CLT-based CI for large $n$; for the median, it is more accurate.

**For AI:** Bootstrapping is used to compute confidence intervals on evaluation metrics (accuracy, F1, AUC) that do not have simple analytical distributions. When you report "model A has 94.2% accuracy vs model B's 93.8%", bootstrapping tells you whether this difference is statistically meaningful. Full treatment of hypothesis tests for model comparison: [§03 Hypothesis Testing](../03-Hypothesis-Testing/notes.md).

### F.5 Multivariate Outlier Detection: The Mahalanobis Envelope

For multivariate Gaussian data $\mathbf{x}_i \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$, the random variable $d_M(\mathbf{x}_i)^2 = (\mathbf{x}_i - \boldsymbol{\mu})^\top \Sigma^{-1} (\mathbf{x}_i - \boldsymbol{\mu}) \sim \chi^2(d)$.

This means that under normality, 95% of observations should have $d_M^2 \leq \chi^2_{0.95, d}$. An observation with $d_M^2 > \chi^2_{0.975, d}$ is a multivariate outlier at the 2.5% per-observation level.

**Example ($d = 2$):** $\chi^2_{0.975, 2} = 7.38$, so $d_M > 2.72$ flags multivariate outliers. An observation can be within the univariate ranges of both features but still be an outlier in the bivariate sense — e.g., if the features are highly positively correlated, an observation with a large positive $x_1$ but small $x_2$ is multivariate-unusual even if neither value is individually extreme.

**Robustified version (MCD estimator):** The Minimum Covariance Determinant estimator (Rousseeuw, 1984) computes the mean vector and covariance matrix using only the $h = \lfloor (n+d+1)/2 \rfloor$ observations whose covariance matrix has minimum determinant. This gives a robust, high-breakdown-point version of the Mahalanobis distance, widely used in OOD detection for deployed ML models.

---

## Appendix G: Glossary

| Term | Definition |
|---|---|
| **iid** | Independent and identically distributed: each observation drawn independently from the same distribution |
| **Sample statistic** | A function of observed data; an estimator of a population parameter |
| **Population parameter** | A fixed but unknown property of the data-generating distribution (e.g., $\mu$, $\sigma^2$) |
| **Breakdown point** | Smallest fraction of corrupted observations that can cause a statistic to take an arbitrarily extreme value |
| **Influence function** | The infinitesimal effect of adding a single observation at $x$ on a statistic |
| **ECDF** | Empirical Cumulative Distribution Function: $\hat{F}_n(x) = \frac{1}{n}\sum \mathbf{1}[x_i \leq x]$ |
| **L-estimator** | A linear combination of order statistics; unifies mean, median, trimmed mean |
| **Bessel's correction** | Using $n-1$ instead of $n$ in the sample variance formula to make it unbiased |
| **Mahalanobis distance** | Distance metric that accounts for the scale and correlation structure of multivariate data |
| **Anscombe's Quartet** | Four datasets with identical standard summary statistics but dramatically different distributions |
| **PSI** | Population Stability Index: symmetric KL divergence between training and production distributions |
| **ARE** | Asymptotic Relative Efficiency: ratio of asymptotic variances of two estimators |
| **KDE** | Kernel Density Estimate: smooth non-parametric estimate of a probability density |
| **Q-Q plot** | Quantile-Quantile plot: compares empirical quantiles to theoretical quantiles to assess distributional fit |
| **Covariate shift** | Distribution shift where $P(X)$ changes but $P(Y|X)$ is stable |
| **Concept drift** | Distribution shift where $P(Y|X)$ changes |
| **Batch Norm** | Normalisation layer that standardises activations across the batch dimension |
| **Layer Norm** | Normalisation layer that standardises activations across the feature dimension per token |
| **RMS Norm** | Variant of Layer Norm that divides by root mean square without subtracting the mean |


---

## Appendix H: Practice Problems with Solutions

### H.1 Basic Problems

**Problem 1.** The following are response times (seconds) for 8 API calls: $\{0.12, 0.15, 0.09, 0.87, 0.11, 0.14, 0.13, 0.10\}$. Compute: (a) mean, (b) median, (c) variance with Bessel's correction, (d) identify outliers using Tukey fences.

*Solution:*
(a) $\bar{x} = (0.12+0.15+0.09+0.87+0.11+0.14+0.13+0.10)/8 = 1.71/8 = 0.214$ seconds
(b) Sorted: $\{0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.87\}$; median $= (0.12+0.13)/2 = 0.125$ seconds
(c) Deviations from mean: $(-0.094, -0.064, -0.124, 0.656, -0.104, -0.074, -0.084, -0.114)$; $s^2 = \sum d_i^2/7 = 0.529/7 = 0.0756$
(d) $Q_1 = 0.105$, $Q_3 = 0.145$, IQR $= 0.04$; upper fence $= 0.145 + 0.06 = 0.205$; $0.87 > 0.205$ → outlier

Note: the mean ($0.214$s) is pulled above the upper fence — it is no longer representative of the typical response time. The median ($0.125$s) is the appropriate summary.

**Problem 2.** Show that for any dataset, $\sum_{i=1}^n (x_i - c)^2$ is minimised at $c = \bar{x}$.

*Solution:*
$$\frac{d}{dc} \sum (x_i - c)^2 = -2\sum(x_i - c) = -2\left(\sum x_i - nc\right) = 0 \implies c = \frac{\sum x_i}{n} = \bar{x}$$
Second derivative $= 2n > 0$ confirms a minimum. $\square$

**Problem 3.** The waiting times at a coffee shop follow an Exponential($\lambda=2$) distribution. What is the population skewness? How does this compare to sample estimates from $n = 20$ vs $n = 1000$?

*Solution:* For Exponential($\lambda$), the skewness is always $g_1 = 2$ (independent of $\lambda$). For $n=20$, the sample skewness estimate has high variance — approximately $\text{Var}(g_1) \approx 6/n = 0.3$ under normality, so a standard error of $\approx 0.55$. For $n=1000$, $\text{SE}(g_1) \approx \sqrt{6/1000} = 0.077$ — much more precise. This illustrates why shape statistics require larger samples than location/scale statistics.

### H.2 Intermediate Problems

**Problem 4.** A dataset has $\bar{x} = 50$, $s = 10$, $g_1 = 0$, $g_2 = 3$ (leptokurtic). Using Chebyshev's inequality, what fraction of observations must lie within $[\bar{x} - 2s, \bar{x} + 2s]$? How does this compare to the Gaussian prediction?

*Solution:* Chebyshev: $P(|X - \mu| \geq k\sigma) \leq 1/k^2$. For $k=2$: at least $1 - 1/4 = 75\%$ lies within $\pm 2$ standard deviations. For a Gaussian: $\approx 95.45\%$. The leptokurtic distribution ($g_2 = 3 > 0$) has heavier tails than Gaussian, so we expect a value between 75% and 95.45% to lie in this interval — Chebyshev's bound is not tight for leptokurtic distributions.

> **Backward reference:** Chebyshev's inequality was proved rigorously in [Ch6 §05 Concentration Inequalities](../../06-Probability-Theory/05-Concentration-Inequalities/notes.md).

**Problem 5.** Two features $X_1$ (salary, thousands) and $X_2$ (years of experience) have covariance $s_{12} = 15$, variances $s_1^2 = 900$, $s_2^2 = 4$. Compute Pearson correlation. Now rescale: $X_1' = X_1/1000$ (salary in millions). How does $r$ change?

*Solution:* $r = s_{12}/(s_1 s_2) = 15/(30 \cdot 2) = 0.25$. After rescaling: $s_1' = 0.9$, $s_{12}' = 0.015$. $r' = 0.015/(0.9 \cdot 2) = 0.0083$... wait, that's wrong. Let me redo: if $X_1' = X_1/1000$, then $s_1' = s_1/1000 = 0.03$, $s_{12}' = s_{12}/1000 = 0.015$. $r' = 0.015/(0.03 \cdot 2) = 0.25$. Pearson correlation is invariant to scaling. This confirms the affine invariance property.

### H.3 Advanced Problems

**Problem 6 (Batch Norm during inference).** During training, Batch Norm computes mini-batch statistics. During inference, it uses running averages. Why can't it use the current-batch statistics during inference for a batch of size 1?

*Solution:* For a batch of size $n=1$: $\hat{\mu}_j^{(B)} = h_{1j}$ and $(\hat{\sigma}_j^{(B)})^2 = 0$ (variance of a single point is zero). The normalised activation would be $0/\epsilon \approx 0$ for all inputs — the layer outputs a constant, destroying all information. This is why BN uses running averages accumulated during training for inference. Layer Norm has no this problem because it normalises across the feature dimension ($d$), which is fixed regardless of batch size.

**Problem 7 (Adam effective window size).** Show that the effective window size of Adam's first moment estimator is approximately $1/(1-\beta_1)$ time steps.

*Solution:* Unrolling the recursion $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$:
$$m_t = (1-\beta_1)\sum_{k=0}^{t-1} \beta_1^k g_{t-k}$$
The weight given to gradient $g_{t-k}$ is $(1-\beta_1)\beta_1^k$, which decays exponentially with delay $k$. The effective number of steps contributing significantly is approximately the "half-life" of the exponential: the average delay is $\mathbb{E}[k] = \beta_1/(1-\beta_1)$, and the effective window is approximately $1/(1-\beta_1)$. For $\beta_1 = 0.9$: window $\approx 10$ steps. For $\beta_1 = 0.99$: window $\approx 100$ steps.

---

## Appendix I: Software Reference

### NumPy / SciPy

| Task | Function | Notes |
|---|---|---|
| Sample mean | `np.mean(x)` | Defaults to full array |
| Sample median | `np.median(x)` | |
| Sample variance | `np.var(x, ddof=1)` | `ddof=1` for Bessel |
| Sample std | `np.std(x, ddof=1)` | |
| Quantiles | `np.percentile(x, [25,50,75])` | or `np.quantile` |
| Skewness | `scipy.stats.skew(x)` | Fisher's $g_1$ |
| Kurtosis | `scipy.stats.kurtosis(x)` | Excess (Fisher), $g_2$ |
| Pearson $r$ | `scipy.stats.pearsonr(x,y)` | Returns (r, p-value) |
| Spearman $\rho$ | `scipy.stats.spearmanr(x,y)` | |
| Kendall $\tau$ | `scipy.stats.kendalltau(x,y)` | |
| Covariance matrix | `np.cov(X.T, ddof=1)` | Rows = features |
| Correlation matrix | `np.corrcoef(X.T)` | |
| ECDF | `scipy.stats.ecdf(x)` | Or manual |

### scikit-learn

| Task | Class | Notes |
|---|---|---|
| Z-score scaling | `StandardScaler()` | `fit` on train, `transform` on test |
| Min-max scaling | `MinMaxScaler()` | |
| Robust scaling | `RobustScaler()` | Uses median and IQR |
| Covariance (robust) | `MinCovDet()` | MCD estimator |
| Mahalanobis OOD | `EllipticEnvelope()` | Fits Gaussian, flags outliers |

### Visualisation (matplotlib / seaborn)

| Plot | Code | Notes |
|---|---|---|
| Histogram + KDE | `sns.histplot(x, kde=True)` | |
| Box plot | `sns.boxplot(x=x)` | Tukey fences |
| Violin plot | `sns.violinplot(x=x)` | |
| ECDF | `sns.ecdfplot(x)` | |
| Q-Q plot | `scipy.stats.probplot(x, plot=plt)` | |
| Scatter + regression | `sns.regplot(x=x, y=y)` | |
| Pair plot | `sns.pairplot(df)` | |
| Correlation heatmap | `sns.heatmap(R, annot=True, cmap='coolwarm')` | |


---

## Appendix J: Common Distributions and Their Descriptive Statistics

Understanding the theoretical descriptive statistics of named distributions (derived from population parameters) helps calibrate expectations when you compute sample statistics from real data.

| Distribution | Mean $\mu$ | Variance $\sigma^2$ | Skewness $g_1$ | Excess Kurtosis $g_2$ |
|---|---|---|---|---|
| $\mathcal{N}(\mu, \sigma^2)$ | $\mu$ | $\sigma^2$ | $0$ | $0$ |
| $\text{Uniform}(a,b)$ | $(a+b)/2$ | $(b-a)^2/12$ | $0$ | $-6/5$ |
| $\text{Exponential}(\lambda)$ | $1/\lambda$ | $1/\lambda^2$ | $2$ | $6$ |
| $\text{Bernoulli}(p)$ | $p$ | $p(1-p)$ | $(1-2p)/\sqrt{p(1-p)}$ | $(1-6p(1-p))/(p(1-p))$ |
| $\text{Poisson}(\lambda)$ | $\lambda$ | $\lambda$ | $1/\sqrt{\lambda}$ | $1/\lambda$ |
| $t_\nu$ (Student-$t$) | $0$ ($\nu>1$) | $\nu/(\nu-2)$ ($\nu>2$) | $0$ | $6/(\nu-4)$ ($\nu>4$) |
| $\chi^2(k)$ | $k$ | $2k$ | $\sqrt{8/k}$ | $12/k$ |
| $\text{Laplace}(\mu,b)$ | $\mu$ | $2b^2$ | $0$ | $3$ |

**Key observations for ML:**

1. **Exponential distribution** has skewness $= 2$ and excess kurtosis $= 6$ — heavy right tail and heavy overall tails. This resembles the distribution of inter-event times in Poisson processes (token generation times, request arrival times).

2. **Student-$t_\nu$** has excess kurtosis $6/(\nu-4)$ for $\nu > 4$. For $\nu = 5$: $g_2 = 6$. As $\nu \to \infty$, $g_2 \to 0$ (approaches Gaussian). The leptokurtic behaviour of $t_\nu$ for small $\nu$ models heavy-tailed phenomena like stochastic gradient noise in LLM training.

3. **Laplace distribution** (double exponential) has excess kurtosis $= 3$. MAP estimation with a Gaussian likelihood and Laplace prior gives Lasso regression — the Laplace prior's heavier tails promote sparsity.

4. **$\chi^2(k)$** is always right-skewed with decreasing skewness as $k$ increases. The sum of squared standardised Gaussian values follows $\chi^2$; gradient squared norms tend toward this shape.

---

## Appendix K: Connections to Linear Algebra

The multivariate statistics developed in §5 are deeply connected to linear algebra concepts:

**Covariance matrix eigendecomposition (PCA):**
$$\hat{\Sigma} = Q \Lambda Q^\top$$
where $Q$ is orthogonal (columns are eigenvectors = principal component directions) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$ with $\lambda_1 \geq \ldots \geq \lambda_d \geq 0$ (eigenvalues = variance explained per component).

The fraction of total variance explained by the first $k$ components:
$$\text{VarExplained}(k) = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^d \lambda_j} = \frac{\sum_{j=1}^k \lambda_j}{\text{tr}(\hat{\Sigma})}$$

> **Forward reference:** PCA derivation, spectral theorem, and the relationship between SVD of the data matrix and eigendecomposition of the covariance matrix are covered in [Ch3 §01 Eigenvalues and Eigenvectors](../../03-Advanced-Linear-Algebra/01-Eigenvalues-and-Eigenvectors/notes.md) and [Ch3 §02 SVD](../../03-Advanced-Linear-Algebra/02-Singular-Value-Decomposition/notes.md).

**Whitening (Mahalanobis transform):**
The transformation $\mathbf{z} = \hat{\Sigma}^{-1/2}(\mathbf{x} - \hat{\boldsymbol{\mu}})$ maps the data to have $\hat{\Sigma}_z = I$ (identity covariance matrix). After whitening, Euclidean distance equals Mahalanobis distance: $\|\mathbf{z}_i - \mathbf{z}_j\|^2 = d_M(\mathbf{x}_i, \mathbf{x}_j)^2$.

Computing $\hat{\Sigma}^{-1/2}$: via eigendecomposition $\hat{\Sigma} = Q\Lambda Q^\top$, we get $\hat{\Sigma}^{-1/2} = Q \Lambda^{-1/2} Q^\top$ where $\Lambda^{-1/2} = \text{diag}(\lambda_j^{-1/2})$.

**Condition number and numerical issues:**
The condition number $\kappa(\hat{\Sigma}) = \lambda_{\max}/\lambda_{\min}$ measures how reliably $\hat{\Sigma}$ can be inverted. Large condition numbers (features with very different scales, highly correlated features) make the covariance matrix ill-conditioned — the Mahalanobis distance becomes numerically unreliable. This is why standardisation (§6.1) is important before computing the covariance matrix.

---

## Appendix L: Summary Diagram

```
DESCRIPTIVE STATISTICS — COMPLETE TAXONOMY
════════════════════════════════════════════════════════════════════════

  UNIVARIATE
  ──────────
  LOCATION        SPREAD          SHAPE           DISTRIBUTION
  ─────────       ──────          ─────           ────────────
  Mean x̄          Variance s²     Skewness g₁     Histogram
  Median x̃        Std dev s       Kurtosis g₂     KDE
  Mode            IQR             5-num summary   Box plot
  Trimmed mean    MAD             ECDF            Violin plot
  Winsorised mean Range           Q-Q plot

  BIVARIATE
  ─────────
  LINEAR              RANK              NONLINEAR
  ──────              ────              ─────────
  Covariance sxy      Spearman ρ        Distance corr
  Pearson r           Kendall τ         Mutual information
  Scatter plot        Rank scatter      Contingency table
  Anscombe's Q        (for ordinal)     Cramér's V

  MULTIVARIATE
  ────────────
  MATRIX              GEOMETRY          DIMENSION
  ──────              ────────          ─────────
  Mean vector μ̂       Mahalanobis dist  Curse of dim
  Cov matrix Σ̂        Whitening         PCA/scree plot
  Corr matrix R       Ellipsoidal sets  n>>d constraint
  Heatmap             Outlier envelope  Rank(Σ̂) = min(n-1,d)

  PREPROCESSING
  ─────────────
  Z-SCORE           MIN-MAX         ROBUST
  ──────            ───────         ──────
  (x - x̄) / s      (x-min)/(max-min) (x - x̃) / IQR
  mean=0, std=1     [0, 1]           median=0, IQR=1
  fits most cases   bounded range    resistant to outliers

  AI APPLICATIONS
  ───────────────
  NORMALISATION           OPTIMISATION        MONITORING
  ─────────────           ────────────        ──────────
  Batch Norm (per feat)   Adam m̂ₜ (mean)      Data drift (PSI)
  Layer Norm (per token)  Adam v̂ₜ (variance)  Outlier detection
  RMS Norm (no center)    Gradient clipping   OOD scoring
                          (Winsorisation)     Calibration

════════════════════════════════════════════════════════════════════════
```


---

## Appendix M: The Statistics of Neural Network Activations

This appendix connects descriptive statistics directly to modern neural network analysis — what happens to activation distributions at each layer, and how normalisation methods control them.

### M.1 Why Activations Drift (Internal Covariate Shift)

Consider a deep network with $L$ layers. Each layer computes $\mathbf{h}^{(l)} = f(W^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$. As parameters $W^{(l)}$ are updated during training, the distribution of $\mathbf{h}^{(l)}$ changes — even though the distribution of the input $\mathbf{h}^{(0)} = \mathbf{x}$ is fixed. This is **internal covariate shift**.

Descriptively: the sample mean $\hat{\mu}_j^{(l)}$ and variance $(\hat{\sigma}_j^{(l)})^2$ of channel $j$ at layer $l$ change with every gradient update. Deep networks without normalisation exhibit a cascade: a parameter update in layer 1 shifts the distribution fed to layer 2, which amplifies or distorts, leading to gradients that vanish (activations → 0) or explode (activations → ±∞).

Batch Norm breaks this cascade by **renormalising to zero mean and unit variance** after every parameter update, making the distribution of inputs to layer $l+1$ independent of the parameters of layer $l$.

### M.2 Analysing Pre-Norm vs Post-Norm Transformers

Modern transformers use either **Pre-LN** (Layer Norm before the attention/FFN sublayer) or **Post-LN** (Layer Norm after adding the residual). GPT-2 used Post-LN; GPT-3, LLaMA, Mistral, and modern models use Pre-LN.

**Why Pre-LN is more stable:** In Post-LN, the gradient of the loss with respect to early layer parameters passes through many unnormalized residual connections, amplifying gradient norms. In Pre-LN, the gradient is normalised at each layer before propagating back, keeping the effective sample variance of gradients more uniform across layers — a descriptive statistics insight with profound training stability consequences.

Concretely: in a 96-layer transformer (like GPT-3), with Post-LN, the gradient variance at layer 1 is approximately $96 \times$ the gradient variance at layer 95. With Pre-LN, the ratio is approximately $1$. This is directly measurable as a descriptive statistic on the gradient distributions across layers.

### M.3 The $1/\sqrt{d_k}$ Scaling in Attention

In scaled dot-product attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

Why divide by $\sqrt{d_k}$? Each element of $QK^\top$ is a dot product of two $d_k$-dimensional vectors. If $q_i, k_j \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$, then:
$$\text{Var}(q_i \cdot k_j) = d_k, \quad \text{Std}(q_i \cdot k_j) = \sqrt{d_k}$$

Without scaling, the dot products have standard deviation $\sqrt{d_k}$, which grows with model dimension. Feeding large-magnitude values into softmax causes it to saturate — the output approaches a one-hot vector, blocking gradient flow. Dividing by $\sqrt{d_k}$ restores unit variance:

$$\text{Var}\!\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) = 1$$

This is a **variance normalisation step** — standard scaling applied to the attention logits. It is derived directly from the sample variance formula and the property that the variance of a sum of $d_k$ independent unit-variance terms is $d_k$.

**Multi-head Latent Attention (MLA, DeepSeek-V2, 2024):** MLA compresses key-value representations into a low-dimensional latent space before attention. The projection reduces $d_k$ and hence the variance of dot products — a deliberate manipulation of the attention logit statistics to reduce KV cache memory while maintaining expressivity.

### M.4 Gradient Statistics as a Training Health Monitor

Monitoring the following descriptive statistics of gradient distributions during LLM training provides an early warning system for training instabilities:

| Statistic | Healthy range | Warning sign |
|---|---|---|
| Gradient $\ell_2$ norm per layer | Decreasing from input to output layers | Exploding: norm growing exponentially |
| Gradient norm variance across layers | Low, uniform | Spike: sudden increase in one layer |
| Gradient skewness $g_1$ | Near 0 | Large positive: rare large positive gradients |
| Gradient kurtosis $g_2$ | $\lesssim 5$ | Very large: heavy-tailed gradient distribution → clip threshold too high |
| Loss spike detection | Mean batch loss | Single batch loss > mean + 5 MAD → potentially corrupted batch |
| Parameter update scale | $\|W_{t+1} - W_t\| / \|W_t\|$ | Should be small ($\sim 10^{-3}$); large → instability |

This is why modern LLM training frameworks (e.g., Megatron-LM, Nanotron) log running statistics of gradient norms, weight update scales, and activation statistics at every layer — real-time EDA of the training process.

---

## Appendix N: Extended Exercise Solutions

### N.1 Exercise 7 (Batch Norm vs Layer Norm) — Key Implementation Notes

The critical implementation detail in Batch Norm is the distinction between training and inference modes. During training:

```python
# Training mode: use current mini-batch statistics
mu_B = X.mean(axis=0)           # shape (d,)
var_B = X.var(axis=0)           # biased variance (ddof=0)
X_norm = (X - mu_B) / np.sqrt(var_B + eps)
running_mean = momentum * running_mean + (1-momentum) * mu_B
running_var  = momentum * running_var  + (1-momentum) * var_B
```

During inference:
```python
# Inference mode: use accumulated running statistics
X_norm = (X - running_mean) / np.sqrt(running_var + eps)
```

For Layer Norm, the statistics are computed per-sample along the feature dimension:
```python
mu_L = X.mean(axis=1, keepdims=True)   # shape (n, 1)
var_L = X.var(axis=1, keepdims=True)   # shape (n, 1)
X_norm = (X - mu_L) / np.sqrt(var_L + eps)
```

The key test: for a batch of size 1 (`n=1`), Batch Norm produces $\hat{\mu} = x_{1j}$ and $\hat{\sigma}^2 = 0$, so normalisation maps every activation to $0/\sqrt{0+\varepsilon} \approx 0$. Layer Norm is unaffected because it normalises over the feature dimension.

### N.2 Exercise 8 (Adam Bias Correction) — Analytical Solution

At time step $t$ with $m_0 = 0$:
$$m_t = (1-\beta_1)\sum_{k=0}^{t-1} \beta_1^k g_{t-k}$$

Taking expectations (assuming iid gradients with $\mathbb{E}[g] = \mu_g$):
$$\mathbb{E}[m_t] = (1-\beta_1)\mu_g \sum_{k=0}^{t-1}\beta_1^k = (1-\beta_1)\mu_g \cdot \frac{1-\beta_1^t}{1-\beta_1} = \mu_g(1-\beta_1^t)$$

Without correction, $m_t$ underestimates $\mu_g$ by a factor $(1-\beta_1^t)$. At $t=1$: $m_1 = (1-\beta_1)g_1$, which for $\beta_1 = 0.9$ gives $m_1 = 0.1 g_1$ — only 10% of the first gradient. Bias correction: $\hat{m}_1 = m_1/(1-0.9) = g_1$. This recovers the full first gradient as the initial estimate. As $t \to \infty$, $\beta_1^t \to 0$ and the correction factor $1/(1-\beta_1^t) \to 1$ — the correction becomes negligible once the EWMA has warmed up.


---

## Appendix O: EDA on Real ML Datasets — Checklist and Patterns

### O.1 What to Always Report for a Tabular ML Dataset

Before any model development, the minimum descriptive statistics to report are:

**Dataset-level:**
- Shape: $n$ rows × $d$ columns
- Memory footprint
- Number and fraction of missing values (total and per column)
- Number of duplicate rows
- Data types (numeric, categorical, datetime, text, binary)

**Per numeric feature:**
- Mean, median, mode (if applicable)
- Std dev, IQR, min, max
- Skewness ($g_1$) and kurtosis ($g_2$)
- Number and fraction of outliers (Tukey fence)
- Distribution plot (histogram + KDE)

**Per categorical feature:**
- Number of unique values (cardinality)
- Top-$k$ most frequent values with counts and fractions
- Frequency of "other" category
- Missing value rate

**Bivariate with target variable:**
- Pearson/Spearman correlation for each numeric feature vs target
- Conditional distribution plots: $P(\text{feature} | \text{class})$ for classification
- Point-biserial correlation for numeric feature vs binary target

**Multivariate:**
- Correlation heatmap of all numeric features
- Variance inflation factor (VIF) for collinearity assessment
- PCA scree plot: how many components explain 90% of variance

### O.2 Common Patterns and What They Mean

| Pattern | Observation | Implication |
|---|---|---|
| Feature has $g_1 > 2$ | Right-skewed: long right tail | Consider log transform; mean >> median; outliers will inflate variance |
| Feature has $g_2 > 5$ | Leptokurtic: heavy tails | Outliers frequent; robust scaling preferred; may need Winsorisation |
| Two features with $|r| > 0.95$ | Near-perfect collinearity | Consider removing one; Ridge/Lasso needed; covariance matrix ill-conditioned |
| Feature correlation with target near 0 | Weak linear association | Feature may still be useful (nonlinear); check mutual information |
| Label class $k$ has $<5\%$ of samples | Severe imbalance | Accuracy misleading; use balanced accuracy, F1; consider oversampling |
| Feature has bimodal histogram | Two subpopulations | Consider splitting dataset or adding an interaction term |
| Many features have near-zero variance | Near-constant features | Remove or near-zero-variance filter (contributes nothing, hurts conditioning) |
| MAD $\approx 0$ but IQR > 0 | More than 50% identical values | Discrete-valued or heavily concentrated; robust stats unreliable |
| Correlation matrix has block structure | Correlated feature groups | Natural for grouped features (e.g., pixel neighbours, word $n$-grams) |

### O.3 Pre-Training Checklist for LLMs

For large-scale pretraining datasets (web text, code, books), descriptive statistics take a different form:

- **Token distribution**: frequency table of most/least common tokens; vocabulary coverage
- **Sequence length distribution**: mean, median, percentiles ($P_{95}$, $P_{99}$, $P_{99.9}$); right-skewed is typical
- **Document quality scores**: perplexity under a reference model; flagging low-quality text
- **Language distribution**: fraction of each language; imbalance affects multilingual capability
- **Temporal distribution**: date of creation; recency bias analysis
- **Duplication rate**: $n$-gram overlap statistics; near-duplicate detection
- **Toxicity and bias statistics**: frequency of flagged content; demographic representation in co-occurrence statistics

These are all descriptive statistics applied at scale — the same concepts (frequency tables, histograms, correlation structures, outlier detection) but applied to discrete text data rather than continuous measurements.

---

## Appendix P: Notation Summary

Following `docs/NOTATION_GUIDE.md`:

| Symbol | Meaning | Type |
|---|---|---|
| $x_i$ | $i$-th scalar observation | Scalar, lowercase |
| $\mathbf{x}_i$ | $i$-th feature vector | Bold lowercase vector |
| $\mathbf{X}$ | Data matrix ($n \times d$) | Uppercase matrix |
| $\mathbf{X}_c$ | Centred data matrix | Uppercase matrix |
| $\bar{x}$ | Sample mean (scalar) | Scalar |
| $\hat{\boldsymbol{\mu}}$ | Sample mean vector | Bold lowercase, hat |
| $s^2$ | Sample variance | Scalar |
| $\hat{\Sigma}$ | Sample covariance matrix | Uppercase, hat |
| $R$ | Sample correlation matrix | Uppercase |
| $r$ | Pearson correlation coefficient | Scalar lowercase |
| $\rho$ | Population correlation (or Spearman) | Greek |
| $\tau$ | Kendall's tau | Greek |
| $\tilde{x}$ | Sample median | Scalar, tilde |
| $Q(p)$ | $p$-th quantile function | |
| $\hat{F}_n(x)$ | Empirical CDF | Hat, subscript $n$ |
| $g_1, g_2$ | Sample skewness, excess kurtosis | Scalars |
| $d_M(\mathbf{x})$ | Mahalanobis distance | Scalar |
| $n$ | Sample size | Positive integer |
| $d$ | Number of features/dimensions | Positive integer |
| $\mu, \sigma^2$ | Population mean, variance | Greek (theoretical) |

---

*End of §07-Statistics/01-Descriptive-Statistics/notes.md*


---

## Appendix Q: Worked ML Case Study — EDA on a Classification Dataset

This appendix walks through a complete EDA on a synthetic classification dataset representative of tabular ML problems: predicting loan default from applicant features.

### Q.1 Dataset Overview

**Features:** age (years), income ($k), credit score (0–850), debt-to-income ratio, number of accounts, late payments (count).
**Target:** default (0/1), with 15% positive class.
**Size:** $n = 5000$ applicants, $d = 6$ features.

### Q.2 Univariate Analysis

**Age:** mean $= 42.3$, median $= 41.0$, $s = 11.8$, $g_1 = 0.2$ (nearly symmetric, slight right skew), $g_2 = -0.3$ (slightly platykurtic). Q-Q plot: slight deviation in upper tail only. No outliers by Tukey fences.

**Income:** mean $= 68.5$k, median $= 52.3$k, $s = 51.2$, $g_1 = 2.8$ (strongly right-skewed), $g_2 = 11.4$ (very leptokurtic). Tukey fence upper: $Q_3 + 1.5 \times \text{IQR} = 87 + 1.5 \times 45 = 154.5$k — 8.3% of observations flagged as outliers. **Recommendation:** log-transform income before modelling. After log transform: $g_1 = 0.1$, $g_2 = 0.4$ — approximately Gaussian.

**Credit score:** mean $= 672$, median $= 685$, $s = 89$, $g_1 = -0.8$ (left-skewed — ceiling effect near 850). A few scores above 840 represent 98th percentile; no action needed but note the ceiling.

**Late payments:** mean $= 1.2$, median $= 0$, mode $= 0$. Heavily right-skewed ($g_1 = 4.1$). 62% of applicants have zero late payments — consider treating as a mixed discrete-continuous variable or Poisson-distributed.

### Q.3 Bivariate Analysis with Target

| Feature | Pearson $r$ | Spearman $\rho_S$ | Note |
|---|---|---|---|
| Age | $-0.03$ | $-0.04$ | Negligible |
| log(Income) | $-0.31$ | $-0.34$ | Moderate negative |
| Credit score | $-0.42$ | $-0.45$ | Strongest predictor |
| Debt-to-income | $+0.38$ | $+0.40$ | Strong positive |
| Late payments | $+0.28$ | $+0.35$ | Moderate positive |

Pearson and Spearman are close for all features except late payments (the most skewed), where Spearman is meaningfully larger — confirming that the relationship is monotone but not linear.

### Q.4 Multivariate Analysis

The correlation matrix reveals:
- Income and credit score: $r = 0.41$ (expected: higher income → better credit history)
- Debt-to-income and late payments: $r = 0.58$ (financially stressed applicants)
- Age and all others: $r < 0.1$ (age is roughly independent of financial variables in this dataset)

The block structure (income-credit-debt cluster vs age) suggests two natural feature groups for regularisation. PCA on the standardised features: first two PCs explain 51% and 19% = 70% total variance. The first PC is approximately an equal-weight financial health score; the second PC is approximately age vs financial stress.

### Q.5 Preprocessing Decisions

Based on EDA:
1. Log-transform income (right-skewed, leptokurtic)
2. Z-score standardise all numeric features (using training-set statistics only)
3. No outlier removal — outliers in income are legitimate high-income applicants, not data errors
4. Handle 3.2% missing credit scores via median imputation (MAR assumption — missing-at-random; lower-income applicants slightly more likely to have missing credit scores)
5. Flag: debt-to-income and late payments are collinear ($r=0.58$) — regularisation will be important

### Q.6 Key EDA Findings Summary

```
EDA FINDINGS — LOAN DEFAULT PREDICTION
════════════════════════════════════════════════════════════════════════

  ✓ Class imbalance: 15% default → use balanced accuracy, F1, AUC
  ✓ Income: log-transform required (skew 2.8 → 0.1 after transform)
  ✓ Credit score: best single predictor (r = -0.42)
  ✓ Collinearity: debt-to-income ↔ late payments (r = 0.58)
  ✓ Missing data: 3.2% credit score (MAR) → median impute
  ✓ Age: near-zero correlation with target → likely weak feature
  ✓ Scale: all features need standardisation (income in $k,
    credit 0-850, DTI 0-1, late payments 0-20)

  RECOMMENDED PIPELINE:
  log(income) → StandardScaler → [classifier with L2 reg]
  Evaluate: AUC-ROC, PR-AUC, balanced accuracy
  Monitor: label distribution shift, credit score mean shift

════════════════════════════════════════════════════════════════════════
```

---

## Appendix R: Reading List

**Foundational:**
1. Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley. — The founding text; invented box plots, stem-and-leaf, resistant methods.
2. Huber, P.J. & Ronchetti, E.M. (2009). *Robust Statistics* (2nd ed.). Wiley. — Formal breakdown point theory; influence functions; M-estimators.
3. Anscombe, F.J. (1973). "Graphs in Statistical Analysis." *American Statistician*, 27(1), 17–21. — The Quartet paper; 4 pages, required reading.

**Applied Statistics:**
4. Freedman, D., Pisani, R., & Purves, R. (2007). *Statistics* (4th ed.). Norton. — Best non-technical introduction; strong on intuition.
5. Wackerly, D., Mendenhall, W., & Scheaffer, R. (2008). *Mathematical Statistics with Applications* (7th ed.). — Formal treatment; proofs; distributions.

**Machine Learning Connections:**
6. Ioffe, S. & Szegedy, C. (2015). "Batch Normalization." *ICML 2015*. — Original BN paper.
7. Ba, J.L., Kiros, J.R., & Hinton, G.E. (2016). "Layer Normalization." *arXiv:1607.06450*. — LN for transformers.
8. Kingma, D.P. & Ba, J.L. (2015). "Adam: A Method for Stochastic Optimization." *ICLR 2015*. — Adam; bias correction derivation.
9. Gebru, T. et al. (2020). "Datasheets for Datasets." *Communications of the ACM*, 64(12). — Descriptive statistics as documentation standard.
10. Lee, K. et al. (2018). "A Simple Unified Framework for Detecting Out-of-Distribution Samples." *NeurIPS 2018*. — Mahalanobis distance for OOD.


---

## Appendix S: Common Statistical Distributions — Shape at a Glance

Recognising the shape of a distribution from its descriptive statistics is a practical skill. This guide maps statistic ranges to likely distribution families.

### S.1 Shape Decision Tree

```
IDENTIFYING DISTRIBUTIONS FROM DESCRIPTIVE STATISTICS
════════════════════════════════════════════════════════════════════════

  Start: Look at histogram + g₁ + g₂

  g₁ ≈ 0, g₂ ≈ 0
  ├── Unimodal, bell-shaped  → Gaussian (test with Shapiro-Wilk)
  └── Bimodal                → Mixture of Gaussians

  g₁ > 1, support [0, ∞)
  ├── g₂ ≈ 6                 → Exponential (λ = 1/mean)
  ├── g₂ large, discrete     → Poisson (λ = mean = variance?)
  └── g₁ = 2, g₂ = 6        → exactly Exponential(λ)

  g₁ > 0, support bounded above
  ├── [0, 1], hump shape     → Beta(α>1, β>1)
  └── [0, 1], J/U shape      → Beta(α≤1 or β≤1)

  g₁ = 0, g₂ > 0 (heavy tails)
  ├── Symmetric, very heavy  → Student-t (ν ≈ 6/(g₂) + 4)
  └── Symmetric, long tails  → Laplace (g₂ = 3)

  g₁ < 0 (left-skewed)
  └── Bounded above, mode near max → ceiling effect; truncated distribution

════════════════════════════════════════════════════════════════════════
```

### S.2 What Gradient Distributions Look Like

Based on empirical measurements in large LLM training runs:

- **Individual parameter gradients**: approximately $\mathcal{N}(0, \sigma^2)$ with $g_1 \approx 0$ but $g_2 \in [3, 15]$ — leptokurtic, closer to Student-$t_5$ than Gaussian.
- **Gradient $\ell_2$ norms** (per-layer): approximately Exponential or $\chi$ distribution — always non-negative, right-skewed with $g_1 \approx 0.5$–$2$.
- **Loss values within a training batch**: usually right-skewed ($g_1 > 0$) with $g_2 \gg 3$ — a few hard examples cause very high loss.

Implication: gradient norm thresholds for clipping should be set based on the **high-quantile** (e.g., $P_{95}$ or $P_{99}$) of the gradient norm distribution, not the mean — because the mean is pulled by rare large-norm gradients.

---

## Appendix T: Summary of All Mathematical Results

| Result | Statement | Reference |
|---|---|---|
| Sample mean is unbiased | $\mathbb{E}[\bar{x}] = \mu$ | §2.2, App C.1 |
| Sample mean variance | $\text{Var}(\bar{x}) = \sigma^2/n$ | App C.2 |
| Bessel's correction | $\mathbb{E}[s^2] = \sigma^2$ with $n-1$ divisor | §2.3 |
| Cauchy-Schwarz bound on $r$ | $|r| \leq 1$ | §4.1, App C.3 |
| $\hat{\Sigma}$ is PSD | $\mathbf{v}^\top \hat{\Sigma} \mathbf{v} \geq 0$ | §5.1, App C.4 |
| Mahalanobis $\chi^2$ | $d_M^2 \sim \chi^2(d)$ under $\mathcal{N}(\mu, \Sigma)$ | §5.3 |
| Glivenko-Cantelli | ECDF converges to CDF uniformly a.s. | §2.5 |
| CLT for $\bar{x}$ | $\sqrt{n}(\bar{x}-\mu)/\sigma \to \mathcal{N}(0,1)$ | App F.1 |
| Breakdown point of median | $\varepsilon^* \to 50\%$ as $n \to \infty$ | §3.1 |
| Adam bias correction | $\hat{m}_t = m_t/(1-\beta_1^t)$ | §8.2, App N.2 |
| Attention variance | $\text{Var}(q \cdot k) = d_k$ for unit-Gaussian inputs | §5.4, App M.3 |
| ARE(median, mean) Gaussian | $2/\pi \approx 0.637$ | App F.3 |


---

## Appendix U: Practice Problems Bank

### U.1 Quick Computation Problems

1. Dataset: $\{1, 2, 2, 3, 3, 3, 4, 4, 5\}$. Compute all measures of central tendency and spread.
2. Prove that for any $c$, $\sum(x_i - c)^2 = ns^2 + n(\bar{x}-c)^2$. Interpret geometrically.
3. Show that adding a constant $c$ to all observations changes $\bar{x}$ by $c$ but leaves $s^2$ unchanged.
4. For samples $x_1, \ldots, x_n$ and $y_i = ax_i + b$, express $\bar{y}$, $s_y^2$, and $r_{xy}$ in terms of $\bar{x}$, $s_x^2$, and $a,b$.
5. Prove Spearman's $\rho = 1 - 6\sum d_i^2 / (n(n^2-1))$ when there are no ties.

### U.2 Conceptual Problems

6. The median of $\{2, 4, 6, 8, 10\}$ is 6. You add a 6th observation. For what values of the new observation does the median stay at 6? Change?
7. Two datasets have the same mean (50) and standard deviation (10). Dataset A has $g_2 = 0$; Dataset B has $g_2 = 5$. Without knowing the data, what can you say about which dataset has more extreme values? Which would be safer to model with a Gaussian?
8. A colleague reports that features $X_1$ and $X_2$ have Pearson correlation $r = 0.98$. They suggest dropping $X_2$ from the model. Is this always the right move? What information do you need?
9. You standardise your training data and achieve validation accuracy of 87%. A teammate says to also standardise the test data using test-set statistics for "fair comparison". What is wrong with this? What should you do instead?
10. Explain why the sample variance of activations in a neural network might increase as training progresses, and how Batch Norm addresses this.

### U.3 Implementation Challenges

11. Implement the ECDF from scratch without using any statistical libraries. Verify it converges to the true CDF for $n \in \{10, 100, 1000\}$ samples from $\mathcal{N}(0,1)$.
12. Implement a bootstrap confidence interval (95%) for the sample median using $B = 1000$ bootstrap resamples. Apply to the income dataset from the loan default case study.
13. Write a function `detect_drift(train, prod, alpha=0.05)` that returns a dictionary of descriptive statistics for both datasets and flags features where the PSI exceeds 0.2 or the mean has shifted by more than 2 standard errors.
14. Reproduce Anscombe's Quartet from scratch and verify all four statistical summaries match within floating-point precision.
15. Implement online (streaming) computation of mean, variance, and skewness using Welford's online algorithm — a numerically stable method that processes one observation at a time without storing the entire dataset.

**Welford's algorithm hint:**
$$M_k = M_{k-1} + (x_k - M_{k-1})/k$$
$$S_k = S_{k-1} + (x_k - M_{k-1})(x_k - M_k)$$
$$s^2_k = S_k/(k-1) \quad \text{for } k \geq 2$$

This is the numerically stable online update rule, used in BN's running mean/variance accumulation.

---

*End of Appendices*


---

## Appendix V: Welford's Algorithm — Numerically Stable Online Statistics

Welford's one-pass algorithm (1962) computes the sample mean and variance numerically stably in a single pass, processing one observation at a time. This is essential for streaming data and large datasets.

**Algorithm (mean and variance):**
```
Initialize: n = 0, M₁ = 0, S = 0
For each observation xₖ:
    n  ← n + 1
    Δ  ← xₖ - Mₙ₋₁
    Mₙ ← Mₙ₋₁ + Δ/n
    S  ← S + Δ·(xₖ - Mₙ)
Sample variance: s² = S/(n-1)
```

**Why numerically stable:** The naïve formula $s^2 = \frac{\sum x_i^2 - n\bar{x}^2}{n-1}$ suffers catastrophic cancellation when $\sum x_i^2 \approx n\bar{x}^2$ (near-constant data). Welford's maintains running deviations from the current mean, avoiding this subtraction.

**Connection to Batch Norm running statistics:**
```python
# Batch Norm running mean update (PyTorch default momentum=0.1):
running_mean = (1-momentum) * running_mean + momentum * batch_mean
```
This is an EWMA (exponentially weighted), not Welford's — but the principle is the same: update the running estimate incrementally without recomputing from scratch.

**Extension to skewness (Terriberry, 2007):**
```
Φ₃ ← Φ₃ + Δ²·(xₖ - Mₙ)·(1 - 2/n) + 3·S·(-Δ/n)
g₁ = √n · Φ₃ / S^(3/2)
```
This allows computing skewness in a single pass — useful for monitoring activation skewness during LLM inference.

