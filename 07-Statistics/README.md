[← Previous Chapter: Probability Theory](../06-Probability-Theory/README.md) | [Next Chapter: Optimization →](../08-Optimization/README.md)

---

# Chapter 7 — Statistics

> _"Statistics is the grammar of science. Without it, data is just noise; with it, data becomes evidence."_

## Overview

Statistics is the discipline of drawing principled conclusions from data. Where probability theory asks "given a model, what data should we expect?", statistics inverts the question: "given data, what can we infer about the underlying model?" This inversion — inference from observations — is the foundation of every machine learning training algorithm.

This chapter builds statistical reasoning from data summarisation (§01), through classical estimation and hypothesis testing (§02–§03), Bayesian inference (§04), time-series analysis (§05), and regression (§06). Every concept is grounded in its ML application: MLE underpins cross-entropy training, confidence intervals govern model evaluation, Bayesian inference enables uncertainty quantification in neural networks, and regression is the blueprint for supervised learning.

**The conceptual arc:** summarise data (§01) → estimate parameters from data (§02) → test hypotheses about data (§03) → update beliefs from data (§04) → model sequential data (§05) → model relationships between variables (§06).

---

## Subsection Map

| # | Subsection | What It Covers | Canonical Topics |
|---|---|---|---|
| 01 | [Descriptive Statistics](01-Descriptive-Statistics/notes.md) | Summarising and characterising datasets before modelling | Mean, median, mode, variance, standard deviation, skewness, kurtosis, quantiles, IQR, correlation, covariance matrices, data visualisation, outlier detection |
| 02 | [Estimation Theory](02-Estimation-Theory/notes.md) | Inferring population parameters from samples; MLE and method of moments | Point estimation, bias, variance, MSE, consistency, efficiency, Cramér-Rao bound, MLE derivation, method of moments, confidence intervals, Fisher information |
| 03 | [Hypothesis Testing](03-Hypothesis-Testing/notes.md) | Formal decision-making under uncertainty; p-values, power, and error rates | Null/alternative hypotheses, Type I/II errors, p-values, significance level, power, t-tests, z-tests, chi-squared tests, ANOVA, multiple testing correction, A/B testing |
| 04 | [Bayesian Inference](04-Bayesian-Inference/notes.md) | Treating parameters as random variables; posterior computation and uncertainty | Prior, likelihood, posterior, conjugate priors, MAP estimation, MCMC posterior sampling, variational inference, credible intervals, Bayesian model comparison |
| 05 | [Time Series](05-Time-Series/notes.md) | Modelling and forecasting sequential, temporally dependent data | Stationarity, autocorrelation, AR/MA/ARMA/ARIMA models, spectral analysis, seasonal decomposition, Kalman filter, forecasting |
| 06 | [Regression Analysis](06-Regression-Analysis/notes.md) | Modelling relationships between variables; the blueprint for supervised learning | Simple and multiple linear regression, OLS derivation, Gauss-Markov theorem, regularisation (Ridge/Lasso), logistic regression, GLMs, model diagnostics |

---

## Reading Order and Dependencies

```
01-Descriptive-Statistics         (foundation: summarise before modelling)
        ↓
02-Estimation-Theory              (core inference: MLE, confidence intervals, Fisher info)
        ↓
03-Hypothesis-Testing             (decisions: p-values, power, A/B testing)
        ↓
04-Bayesian-Inference             (probabilistic view: posterior, MAP, MCMC)
        ↓
05-Time-Series                    (sequential data: AR/MA, forecasting, Kalman)
        ↓
06-Regression-Analysis            (supervised learning blueprint: OLS, Ridge, Lasso)
        ↓
Chapter 8 — Optimization          (gradient methods, convexity, training algorithms)
```

---

## What Belongs Where — Canonical Homes

| Topic | Canonical Home | Preview Only In |
|---|---|---|
| Mean, median, mode, quantiles | §01 | §02 (sample mean as estimator) |
| Variance, standard deviation (sample) | §01 | §06 (residual variance) |
| Correlation and covariance (empirical) | §01 | §06 (design matrix structure) |
| Skewness, kurtosis, distribution shape | §01 | — |
| Outlier detection methods | §01 | — |
| Point estimators: bias, variance, MSE | §02 | — |
| Cramér-Rao lower bound, Fisher information | §02 | §04 (Laplace approximation) |
| MLE derivation | §02 | §04 (MAP as regularised MLE) |
| Method of moments | §02 | — |
| Confidence intervals (frequentist) | §02 | §03 (duality with tests) |
| Asymptotic normality of MLE | §02 | — |
| Null/alternative hypotheses, p-values | §03 | — |
| Type I/II errors, power, significance | §03 | — |
| t-test, z-test, chi-squared, ANOVA | §03 | — |
| Multiple testing correction | §03 | — |
| A/B testing framework | §03 | — |
| Bayes' theorem (prior × likelihood) | §04 | Ch6§03 (full derivation), §02 (preview) |
| Conjugate priors | §04 | — |
| MAP estimation | §04 | §02 (as regularised MLE) |
| Posterior predictive distribution | §04 | — |
| Credible intervals | §04 | — |
| Variational inference | §04 | — |
| Stationarity, ACF/PACF | §05 | — |
| AR/MA/ARIMA models | §05 | — |
| Kalman filter | §05 | — |
| Spectral density, Fourier in time series | §05 | — |
| OLS derivation ($\hat{\beta} = (X^TX)^{-1}X^Ty$) | §06 | — |
| Gauss-Markov theorem, BLUE | §06 | — |
| Ridge and Lasso regularisation | §06 | — |
| Logistic regression, GLMs | §06 | — |
| Residual analysis, model diagnostics | §06 | — |

---

## Overlap Danger Zones

### 1. Descriptive Statistics ↔ Probability Theory
- **§01** computes sample statistics from data (empirical mean, sample variance, sample correlation).
- **Ch6§04** defines population parameters (expected value, variance, covariance of a distribution).
- §01 must not re-derive the expectation operator; it applies sample analogues. §01 should backward-reference Ch6§04 for the population versions.

### 2. MLE ↔ MAP ↔ Bayesian Inference
- **§02** derives MLE from the likelihood principle and treats it as a point estimate.
- **§04** introduces MAP as MLE regularised by a prior, then extends to full posterior inference.
- §02 may note that adding a prior gives MAP, but the full treatment of priors, posteriors, and conjugacy belongs in §04.

### 3. Confidence Intervals ↔ Credible Intervals
- **§02** defines frequentist confidence intervals: in repeated experiments, 95% of such intervals contain the true parameter.
- **§04** defines Bayesian credible intervals: the posterior probability that the parameter lies in the interval is 95%.
- The philosophical distinction belongs in §04; §02 covers only the frequentist construction.

### 4. Hypothesis Testing ↔ Bayesian Model Comparison
- **§03** is the canonical home for p-values, power, and classical tests.
- **§04** covers Bayes factors and Bayesian model selection as the Bayesian analogue.
- §03 may note the Bayesian alternative briefly; §04 covers it in depth.

### 5. Regression ↔ Optimisation
- **§06** derives OLS and establishes regression as a statistical model (Gauss-Markov, residual assumptions).
- **Ch8** covers gradient-based optimisation, convexity, and SGD — the algorithmic machinery for fitting models.
- §06 may solve OLS by the normal equations (closed form) without invoking gradient descent; the algorithmic treatment belongs in Ch8.

### 6. Regression ↔ Descriptive Statistics
- **§01** computes the empirical correlation coefficient $r$ as a summary statistic.
- **§06** derives the regression coefficient $\hat{\beta}_1 = r \cdot s_y / s_x$ and explains the relationship.
- §01 must not derive OLS; §06 must backward-reference §01 for empirical correlation.

---

## Key Cross-Chapter Dependencies

**From Chapter 6 — Probability Theory:**
- §01 (Random Variables, CDF/PDF) → §02 likelihood functions and sampling distributions
- §02 (Common Distributions) → §02/§03 named test statistics (t, chi-squared, F distributions)
- §03 (Bayes' theorem, conditional distributions) → §04 Bayesian inference
- §04 (Expectation, variance) → §02 estimator properties (bias, variance, MSE)
- §05 (CLT, LLN) → §02 asymptotic normality of MLE; §03 large-sample tests
- §06 (Stochastic Processes) → §05 time-series modelling (stationary processes, ACF)
- §07 (Markov Chains, MH) → §04 MCMC posterior sampling

**From Chapter 3 — Advanced Linear Algebra:**
- SVD → §02 Fisher information geometry; §06 OLS via pseudoinverse
- Positive definite matrices → §06 covariance matrix of estimators; ridge regression

**Into Chapter 8 — Optimisation:**
- §02 (MLE) → cross-entropy and NLL as loss functions
- §06 (OLS) → normal equations as a linear system; extends to gradient descent
- §04 (ELBO, variational inference) → variational autoencoder training objective

**Into Chapter 9 — Information Theory:**
- §02 (Fisher information) → Cramér-Rao and the information-theoretic view of estimation
- §04 (KL divergence as posterior approximation criterion) → variational inference

---

## ML Concept Map

| ML Concept | Statistics Foundation | Section |
|---|---|---|
| Cross-entropy loss $-\sum y \log \hat{p}$ | Negative log-likelihood (MLE objective) | §02 |
| Weight decay / L2 regularisation | Ridge regression; MAP with Gaussian prior | §04, §06 |
| L1 / sparsity regularisation | Lasso regression; MAP with Laplace prior | §04, §06 |
| Confidence in model evaluation | Confidence intervals for test accuracy | §02, §03 |
| A/B testing for model comparison | Two-sample hypothesis test | §03 |
| Bayesian neural networks | Posterior over weights; variational inference | §04 |
| Dropout as Bayesian approximation | MC Dropout ↔ approximate posterior sampling | §04 |
| Batch normalisation | Sample mean/variance of activations | §01 |
| Layer normalisation | Sample statistics within layers | §01 |
| Early stopping | Train/validation loss as statistical estimators | §02 |
| Calibration (softmax temperature) | Posterior predictive calibration | §04 |
| Anomaly / OOD detection | Hypothesis testing; statistical distance | §03 |
| Data drift detection | Two-sample tests on feature distributions | §03 |
| Time-series forecasting | ARIMA, Kalman filter | §05 |
| Transformer positional encoding | Spectral methods, Fourier features | §05 |
| Linear probe evaluation | Logistic regression on embeddings | §06 |
| LoRA / low-rank adaptation | Regression on low-dimensional subspaces | §06 |
| Reward modelling (RLHF) | Logistic regression (Bradley-Terry model) | §06 |

---

## Prerequisites

Before starting this chapter, ensure you are comfortable with:

- **Probability distributions** — PDFs, CDFs, named distributions (Gaussian, Bernoulli, Poisson, Beta) — [Chapter 6 §01–§02](../06-Probability-Theory/01-Introduction-and-Random-Variables/notes.md)
- **Expectation and variance** — $\mathbb{E}[X]$, $\text{Var}(X)$, covariance — [Chapter 6 §04](../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Bayes' theorem** — $p(\theta|x) \propto p(x|\theta)p(\theta)$ — [Chapter 6 §03](../06-Probability-Theory/03-Joint-Distributions/notes.md)
- **Central limit theorem** — sample mean converges to Gaussian — [Chapter 6 §06](../06-Probability-Theory/06-Stochastic-Processes/notes.md)
- **Matrix algebra** — matrix inverse, SVD, positive definiteness — [Chapter 3](../03-Advanced-Linear-Algebra/README.md)
- **Calculus** — derivatives, optimisation (for MLE derivations) — [Chapter 4](../04-Calculus-Fundamentals/README.md)

---

[← Previous Chapter: Probability Theory](../06-Probability-Theory/README.md) | [Next Chapter: Optimization →](../08-Optimization/README.md)
