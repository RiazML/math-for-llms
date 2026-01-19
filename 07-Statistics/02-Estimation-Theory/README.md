# Estimation Theory

## Introduction

Estimation theory provides the mathematical framework for inferring population parameters from sample data. This is foundational for machine learning, where we estimate model parameters from training data.

## Prerequisites

- Probability distributions
- Expectation and variance
- Calculus (derivatives, optimization)

## Learning Objectives

1. Understand point vs interval estimation
2. Evaluate estimators (bias, variance, MSE)
3. Derive MLE and method of moments estimators
4. Apply estimation to ML contexts

---

## 1. Point Estimation

A **point estimator** $\hat{\theta}$ is a function of sample data that produces a single value to estimate an unknown parameter $\theta$.

$$\hat{\theta} = g(X_1, X_2, \ldots, X_n)$$

### Example: Estimating Mean

Given samples $X_1, \ldots, X_n$ from a distribution with mean $\mu$:

$$\hat{\mu} = \bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

---

## 2. Properties of Estimators

### 2.1 Bias

**Bias** measures systematic error:

$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

- **Unbiased**: $E[\hat{\theta}] = \theta$ (bias = 0)
- Sample mean $\bar{X}$ is unbiased for $\mu$

### 2.2 Variance

$$\text{Var}(\hat{\theta}) = E[(\hat{\theta} - E[\hat{\theta}])^2]$$

Lower variance = more precise estimates.

### 2.3 Mean Squared Error (MSE)

MSE combines bias and variance:

$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + [\text{Bias}(\hat{\theta})]^2$$

```
Bias-Variance Tradeoff:

        Low Variance           High Variance
       ┌─────────────┐        ┌─────────────┐
Low    │    ·  ·     │        │  ·      ·   │
Bias   │      ⊕      │        │    ⊕        │
       │   ·    ·    │        │      ·   ·  │
       └─────────────┘        └─────────────┘
          Ideal!              Imprecise but
                              centered
       ┌─────────────┐        ┌─────────────┐
High   │             │        │ ·           │
Bias   │    · · ·    │        │         ·   │
       │      ⊕      │        │    ⊕  ·     │
       └─────────────┘        └─────────────┘
       Precise but            Worst case
       off target

⊕ = True value, · = Estimates
```

### 2.4 Consistency

An estimator is **consistent** if:

$$\hat{\theta}_n \xrightarrow{p} \theta \text{ as } n \to \infty$$

(Converges in probability to true value)

### 2.5 Efficiency

Among unbiased estimators, the **efficient** one has minimum variance.

**Cramér-Rao Lower Bound:**

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

where $I(\theta)$ is Fisher Information.

---

## 3. Maximum Likelihood Estimation (MLE)

### 3.1 Likelihood Function

Given data $\mathbf{x} = (x_1, \ldots, x_n)$ and parameter $\theta$:

$$L(\theta | \mathbf{x}) = \prod_{i=1}^n f(x_i | \theta)$$

### 3.2 Log-Likelihood

$$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log f(x_i | \theta)$$

### 3.3 MLE

$$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta)$$

Find by solving: $\frac{\partial \ell}{\partial \theta} = 0$

### 3.4 Example: Normal Distribution

Data: $x_1, \ldots, x_n \sim N(\mu, \sigma^2)$

Log-likelihood:
$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2$$

**MLE for μ:**
$$\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0$$
$$\hat{\mu}_{MLE} = \bar{x}$$

**MLE for σ²:**
$$\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$

Note: MLE for variance uses $n$, not $n-1$ (slightly biased).

---

## 4. Properties of MLE

### 4.1 Asymptotic Properties

As $n \to \infty$:

1. **Consistency**: $\hat{\theta}_{MLE} \xrightarrow{p} \theta$

2. **Asymptotic Normality**:
   $$\sqrt{n}(\hat{\theta}_{MLE} - \theta) \xrightarrow{d} N(0, I(\theta)^{-1})$$

3. **Efficiency**: Achieves Cramér-Rao bound

### 4.2 Invariance

If $\hat{\theta}$ is MLE of $\theta$, then $g(\hat{\theta})$ is MLE of $g(\theta)$.

---

## 5. Fisher Information

### 5.1 Definition

$$I(\theta) = -E\left[\frac{\partial^2 \ell}{\partial \theta^2}\right] = E\left[\left(\frac{\partial \ell}{\partial \theta}\right)^2\right]$$

### 5.2 Interpretation

- Higher Fisher Information → More information about $\theta$ in data
- Inverse gives lower bound on estimator variance

### 5.3 Example: Bernoulli

$X \sim \text{Bernoulli}(p)$

$$\ell(p) = x\log p + (1-x)\log(1-p)$$

$$\frac{\partial \ell}{\partial p} = \frac{x}{p} - \frac{1-x}{1-p}$$

$$I(p) = E\left[\left(\frac{x}{p} - \frac{1-x}{1-p}\right)^2\right] = \frac{1}{p(1-p)}$$

---

## 6. Method of Moments (MoM)

### 6.1 Approach

1. Express population moments in terms of parameters
2. Equate sample moments to population moments
3. Solve for parameters

### 6.2 Population and Sample Moments

**k-th Population Moment:** $\mu_k = E[X^k]$

**k-th Sample Moment:** $m_k = \frac{1}{n}\sum_{i=1}^n x_i^k$

### 6.3 Example: Gamma Distribution

$X \sim \text{Gamma}(\alpha, \beta)$ with $E[X] = \alpha/\beta$, $\text{Var}(X) = \alpha/\beta^2$

From $E[X] = \alpha/\beta$ and $E[X^2] = \text{Var}(X) + E[X]^2$:

$$m_1 = \bar{x} = \frac{\alpha}{\beta}$$
$$m_2 = \overline{x^2} = \frac{\alpha}{\beta^2} + \frac{\alpha^2}{\beta^2}$$

Solving:
$$\hat{\beta}_{MoM} = \frac{\bar{x}}{\overline{x^2} - \bar{x}^2}$$
$$\hat{\alpha}_{MoM} = \bar{x} \cdot \hat{\beta}_{MoM}$$

---

## 7. MLE vs Method of Moments

| Aspect      | MLE                        | Method of Moments     |
| ----------- | -------------------------- | --------------------- |
| Efficiency  | Asymptotically optimal     | Can be less efficient |
| Computation | May need numerical methods | Often closed-form     |
| Complexity  | Can be complex             | Usually simpler       |
| Properties  | Well-studied theory        | Easier to derive      |

---

## 8. Bayesian Estimation

### 8.1 Prior and Posterior

**Prior:** $p(\theta)$ - belief about $\theta$ before seeing data

**Likelihood:** $p(\mathbf{x}|\theta)$

**Posterior:** $p(\theta|\mathbf{x}) \propto p(\mathbf{x}|\theta)p(\theta)$

### 8.2 MAP Estimation

**Maximum A Posteriori:**

$$\hat{\theta}_{MAP} = \arg\max_\theta p(\theta|\mathbf{x}) = \arg\max_\theta [p(\mathbf{x}|\theta)p(\theta)]$$

### 8.3 MLE vs MAP

$$\hat{\theta}_{MLE} = \arg\max_\theta \log p(\mathbf{x}|\theta)$$

$$\hat{\theta}_{MAP} = \arg\max_\theta [\log p(\mathbf{x}|\theta) + \log p(\theta)]$$

**Connection to Regularization:**

- Gaussian prior on $\theta$ → L2 regularization (Ridge)
- Laplace prior on $\theta$ → L1 regularization (Lasso)

---

## 9. ML Applications

### 9.1 Neural Network Training

Weight estimation via MLE (cross-entropy = negative log-likelihood):

$$\hat{\mathbf{w}}_{MLE} = \arg\min_\mathbf{w} \sum_{i=1}^n -\log p(y_i | \mathbf{x}_i, \mathbf{w})$$

### 9.2 Regularized Estimation (MAP)

$$\hat{\mathbf{w}}_{MAP} = \arg\min_\mathbf{w} \left[\sum_{i=1}^n -\log p(y_i | \mathbf{x}_i, \mathbf{w}) + \lambda||\mathbf{w}||^2\right]$$

### 9.3 Bias-Variance in ML

| Model                 | Bias   | Variance | MSE    |
| --------------------- | ------ | -------- | ------ |
| Simple (few params)   | High   | Low      | Medium |
| Complex (many params) | Low    | High     | Medium |
| Regularized           | Medium | Medium   | Low!   |

---

## 10. Summary

| Estimator | Formula                        | Properties                           |
| --------- | ------------------------------ | ------------------------------------ | ----------------------------------- |
| MLE       | $\arg\max_\theta \ell(\theta)$ | Consistent, asymptotically efficient |
| MoM       | Equate moments                 | Simple, may be inefficient           |
| MAP       | $\arg\max\_\theta [p(x         | \theta)p(\theta)]$                   | Incorporates prior (regularization) |

**Key Concepts:**

- Bias-Variance Tradeoff
- Fisher Information bounds variance
- MLE = cross-entropy minimization
- MAP = regularized MLE

---

## References

1. Casella & Berger - "Statistical Inference"
2. Bishop - "Pattern Recognition and Machine Learning"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
