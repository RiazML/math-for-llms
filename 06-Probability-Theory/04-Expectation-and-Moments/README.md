# Expectation, Variance, and Moments

## Introduction

Expectation and variance are fundamental measures that summarize the central tendency and spread of probability distributions. Higher moments provide additional information about shape (skewness, kurtosis). These concepts are essential for understanding loss functions, regularization, and uncertainty in ML.

## Prerequisites

- Basic probability theory
- Integration/summation
- Probability distributions

## Learning Objectives

1. Compute expectation and variance for various distributions
2. Apply properties of expectation and variance
3. Understand moment generating functions
4. Connect moments to ML applications

---

## 1. Expectation (Expected Value)

### Definition

**Discrete case:**
$$E[X] = \sum_x x \cdot P(X = x) = \sum_x x \cdot p(x)$$

**Continuous case:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

The expected value is also called the **mean** and denoted $\mu$ or $\mu_X$.

```
Interpretation:

       PMF             Weighted Average
     P(X=x)
0.4  │  ██                    ┌─────────┐
0.3  │  ██  ██                │ Balance │
0.2  │  ██  ██                │  Point  │
0.1  │  ██  ██  ██            └────┬────┘
     └────────────→                ▼
       1   2   3   x        E[X] ≈ 1.7
```

### Expectation of Function

$$E[g(X)] = \sum_x g(x) \cdot p(x) \quad \text{or} \quad E[g(X)] = \int g(x) \cdot f(x) \, dx$$

**Important:** $E[g(X)] \neq g(E[X])$ in general!

---

## 2. Properties of Expectation

### Linearity (Most Important!)

$$E[aX + b] = aE[X] + b$$

$$E[X + Y] = E[X] + E[Y]$$

**Always true**, even for dependent variables!

### Expectation of Product

- If X, Y independent: $E[XY] = E[X] \cdot E[Y]$
- In general: $E[XY] = E[X]E[Y] + \text{Cov}(X, Y)$

### Conditional Expectation

$$E[X|Y=y] = \sum_x x \cdot P(X=x|Y=y)$$

**Law of Total Expectation (Adam's Law):**
$$E[X] = E[E[X|Y]]$$

"Expected value of conditional expectations"

---

## 3. Variance

### Definition

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

The second form is often easier to compute.

**Standard Deviation:** $\sigma = \sqrt{\text{Var}(X)}$

```
Variance visualization:

Low variance:          High variance:
       │                     │
    ████████               ██    ██
       │                     │
   ────┼────              ──────┼──────
       μ                       μ
```

### Properties of Variance

| Property          | Formula                                                                |
| ----------------- | ---------------------------------------------------------------------- |
| Constant          | $\text{Var}(c) = 0$                                                    |
| Scaling           | $\text{Var}(aX) = a^2 \text{Var}(X)$                                   |
| Shift             | $\text{Var}(X + b) = \text{Var}(X)$                                    |
| Sum (independent) | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$                    |
| Sum (general)     | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ |

### Conditional Variance

$$\text{Var}(X|Y=y) = E[X^2|Y=y] - (E[X|Y=y])^2$$

**Law of Total Variance (Eve's Law):**
$$\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$$

"Total = within-group + between-group"

---

## 4. Covariance and Correlation

### Covariance

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

Properties:

- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX, bY) = ab \cdot \text{Cov}(X, Y)$
- If X ⊥ Y, then Cov(X, Y) = 0 (but converse is false!)

### Correlation

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

- Normalized measure: $-1 \leq \rho \leq 1$
- $\rho = 0$: no linear relationship
- $\rho = \pm 1$: perfect linear relationship

---

## 5. Higher Moments

### Raw Moments

$$\mu'_n = E[X^n]$$

- $\mu'_1 = E[X]$ (mean)
- $\mu'_2 = E[X^2]$ (used for variance)

### Central Moments

$$\mu_n = E[(X - \mu)^n]$$

- $\mu_1 = 0$ (always)
- $\mu_2 = \text{Var}(X)$
- $\mu_3$: related to skewness
- $\mu_4$: related to kurtosis

### Skewness

$$\text{Skewness} = \frac{E[(X-\mu)^3]}{\sigma^3}$$

```
Skewness:

Negative (left)    Zero (symmetric)   Positive (right)
      ▄▄▄█                              █▄▄▄
     ▄████                ██           ████▄
    ▄█████               ████          █████▄
   ▄██████              ██████        ██████▄
  ─────────            ─────────      ─────────
      │                   │               │
   mode>mean         mode=mean        mode<mean
```

### Kurtosis

$$\text{Kurtosis} = \frac{E[(X-\mu)^4]}{\sigma^4}$$

**Excess Kurtosis** = Kurtosis - 3 (Normal distribution has kurtosis = 3)

- Excess kurtosis > 0: heavy tails (leptokurtic)
- Excess kurtosis < 0: light tails (platykurtic)

---

## 6. Moment Generating Functions

### Definition

$$M_X(t) = E[e^{tX}]$$

### Key Properties

1. **Uniqueness:** MGF uniquely determines distribution
2. **Moments:** $E[X^n] = M_X^{(n)}(0) = \frac{d^n M_X}{dt^n}\bigg|_{t=0}$
3. **Independence:** $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$ if X ⊥ Y

### Common MGFs

| Distribution   | MGF                                             |
| -------------- | ----------------------------------------------- |
| Bernoulli(p)   | $1 - p + pe^t$                                  |
| Binomial(n,p)  | $(1 - p + pe^t)^n$                              |
| Poisson(λ)     | $\exp(\lambda(e^t - 1))$                        |
| Normal(μ,σ²)   | $\exp(\mu t + \frac{1}{2}\sigma^2 t^2)$         |
| Exponential(λ) | $\frac{\lambda}{\lambda - t}$ for $t < \lambda$ |

### Using MGF to Find Moments

Example for N(μ, σ²):
$$M(t) = e^{\mu t + \frac{1}{2}\sigma^2 t^2}$$

$$M'(t) = (\mu + \sigma^2 t)M(t)$$
$$M'(0) = \mu = E[X]$$

$$M''(t) = \sigma^2 M(t) + (\mu + \sigma^2 t)^2 M(t)$$
$$M''(0) = \sigma^2 + \mu^2 = E[X^2]$$

---

## 7. Inequalities

### Markov's Inequality

For non-negative X and a > 0:
$$P(X \geq a) \leq \frac{E[X]}{a}$$

### Chebyshev's Inequality

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

- At least 75% within 2σ of mean
- At least 89% within 3σ of mean

### Jensen's Inequality

For convex function g:
$$g(E[X]) \leq E[g(X)]$$

For concave function g:
$$g(E[X]) \geq E[g(X)]$$

Application: $E[\log X] \leq \log E[X]$ (log is concave)

---

## 8. ML Applications

### Loss Functions as Expectations

$$\text{Risk} = E[\ell(Y, \hat{Y})] = \int \ell(y, \hat{y}) \cdot p(x, y) \, dx \, dy$$

### Mean Squared Error

$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2$$

### Bias-Variance Tradeoff

$$E[(Y - \hat{f}(X))^2] = \underbrace{\text{Var}(\hat{f})}_{\text{variance}} + \underbrace{(\text{Bias}(\hat{f}))^2}_{\text{bias}^2} + \underbrace{\sigma^2}_{\text{irreducible}}$$

### Regularization

Ridge regression: minimize $E[\|Y - X\beta\|^2] + \lambda \|\beta\|^2$

The regularization term reduces variance at cost of bias.

### Entropy as Expected Log-Probability

$$H(X) = -E[\log p(X)] = -\sum_x p(x) \log p(x)$$

Cross-entropy loss is expected negative log-likelihood.

---

## 9. Computational Formulas

### Sample Mean

$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

- Unbiased: $E[\bar{X}] = \mu$
- Variance: $\text{Var}(\bar{X}) = \sigma^2/n$

### Sample Variance

$$S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

Division by n-1 makes it unbiased: $E[S^2] = \sigma^2$

### Bessel's Correction

Why n-1 instead of n?

The sample mean $\bar{X}$ is closer to the sample points than $\mu$ is.
Using n underestimates variance. The n-1 correction compensates.

---

## 10. Summary Table

| Quantity  | Formula             | Properties        |
| --------- | ------------------- | ----------------- |
| E[X]      | $\sum x \cdot p(x)$ | Linear            |
| Var(X)    | $E[X^2] - E[X]^2$   | Non-negative      |
| Cov(X,Y)  | $E[XY] - E[X]E[Y]$  | Symmetric         |
| Corr(X,Y) | Cov/(σ_X σ_Y)       | In [-1, 1]        |
| Skewness  | $E[(X-μ)^3]/σ^3$    | Symmetry measure  |
| Kurtosis  | $E[(X-μ)^4]/σ^4$    | Tail heaviness    |
| MGF       | $E[e^{tX}]$         | Generates moments |

---

## Exercises

1. Prove E[X+Y] = E[X] + E[Y] (no independence required)
2. Derive Var(aX + bY) for correlated X, Y
3. Find the MGF of Exponential(λ) and use it to get mean and variance
4. Prove Jensen's inequality for discrete case
5. Show that sample variance with n-1 is unbiased

---

## References

1. Casella & Berger - "Statistical Inference"
2. Rice - "Mathematical Statistics and Data Analysis"
3. Wasserman - "All of Statistics"
