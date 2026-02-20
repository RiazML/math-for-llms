# Expectation and Moments

> **Navigation**: [01-Introduction](../01-Introduction-and-Random-Variables/) | [02-Common-Distributions](../02-Common-Distributions/) | [03-Joint-Distributions](../03-Joint-Distributions/)

## Overview

**Expectation** and **moments** summarize distributions with single numbers. The mean tells us "where" a distribution is centered, variance tells us "how spread out" it is, and higher moments capture shape characteristics like skewness and heavy tails. These summary statistics are fundamental to loss functions, optimization, and uncertainty quantification in ML.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MOMENTS IN MACHINE LEARNING                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Moment 1: Mean (μ)          → Predictions, Loss functions              │
│  Moment 2: Variance (σ²)     → Uncertainty, Regularization              │
│  Moment 3: Skewness          → Asymmetric distributions                 │
│  Moment 4: Kurtosis          → Heavy tails, Outlier sensitivity         │
│                                                                          │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐     │
│  │   E[X]     │   │  E[X²]     │   │  E[X³]     │   │  E[X⁴]     │     │
│  │            │   │            │   │            │   │            │     │
│  │  Location  │   │  Spread    │   │  Symmetry  │   │   Tails    │     │
│  └────────────┘   └────────────┘   └────────────┘   └────────────┘     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- [01-Introduction-and-Random-Variables](../01-Introduction-and-Random-Variables/)
- [02-Common-Distributions](../02-Common-Distributions/)
- [03-Joint-Distributions](../03-Joint-Distributions/)
- Calculus (integration, series)

## Learning Objectives

1. Master expectation properties and computation
2. Understand variance and standard deviation
3. Interpret higher moments (skewness, kurtosis)
4. Apply moment generating functions

---

## 1. Expected Value (Mean)

### Definition

**Discrete:**
$$E[X] = \sum_x x \cdot P(X = x) = \sum_x x \cdot p(x)$$

**Continuous:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$$

> **💡 Intuition**: E[X] is the "center of mass" of the probability distribution. If you placed weights proportional to probabilities, E[X] is where the distribution would balance.

```
Expected Value as Center of Mass:

P(x)
  │
  │    █             
  │   ███            █
  │  █████          ███
  │ ███████        █████
  ├──┴────┴──────────┴────────
  │       ▲              ▲
  │    E[X]≈2         E[X]≈7
  │
  └────────────────────────── x
```

### Expectation of a Function

$$E[g(X)] = \sum_x g(x) \cdot p(x) \quad \text{or} \quad \int g(x) \cdot f(x) \, dx$$

> **⚠️ Warning**: $E[g(X)] \neq g(E[X])$ in general! (Jensen's inequality)

---

## 2. Properties of Expectation

### Linearity (Most Important!)

$$E[aX + b] = aE[X] + b$$
$$E[X + Y] = E[X] + E[Y] \quad \text{(ALWAYS, even if dependent!)}$$

```
Linearity of Expectation:

E[X₁ + X₂ + ... + Xₙ] = E[X₁] + E[X₂] + ... + E[Xₙ]

This works regardless of dependence!

Example: Expected sum of 10 dice = 10 × E[single die] = 10 × 3.5 = 35
```

### Summary of Properties

| Property | Formula | Note |
|----------|---------|------|
| Constant | $E[c] = c$ | |
| Scaling | $E[aX] = aE[X]$ | |
| Shift | $E[X + b] = E[X] + b$ | |
| Linearity | $E[aX + bY] = aE[X] + bE[Y]$ | **Always works!** |
| Products (if independent) | $E[XY] = E[X]E[Y]$ | **Only if $X \perp\!\!\!\perp Y$** |

---

## 3. Conditional Expectation

### Definition

$$E[Y | X = x] = \sum_y y \cdot P(Y = y | X = x)$$

$$E[Y | X = x] = \int y \cdot f_{Y|X}(y | x) \, dy$$

### Law of Total Expectation

$$E[Y] = E[E[Y | X]]$$

"Average of conditional averages = unconditional average"

```
Law of Total Expectation:

        ┌─────────────────┐
        │     E[Y]        │  ← Unconditional mean
        └────────┬────────┘
                 │
      ┌─────────────────────┐
      │  E[Y|X=x₁]·P(X=x₁)  │
      ├─────────────────────┤
      │  E[Y|X=x₂]·P(X=x₂)  │  ← Weighted sum of
      ├─────────────────────┤     conditional means
      │  E[Y|X=x₃]·P(X=x₃)  │
      └─────────────────────┘
```

### Law of Total Variance

$$\text{Var}(Y) = E[\text{Var}(Y|X)] + \text{Var}(E[Y|X])$$

```
Total Variance Decomposition:

Total Variance = Within-group Variance + Between-group Variance

    │
    │   ●●       ○○
    │  ●●●●     ○○○○    ← Two groups with means μ₁, μ₂
    │   ●●       ○○
    │
    ├──────┴───────┴────
           μ₁      μ₂

• Within-group: How spread out within each group
• Between-group: How different are the group means
```

---

## 4. Variance

### Definition

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

**Standard Deviation:** $\sigma_X = \sqrt{\text{Var}(X)}$

> **💡 Intuition**: Variance measures the average squared distance from the mean.

```
Variance Visualization:

Low Variance                    High Variance
    │                              │
    │       ████                   │
    │      ██████                  │ ██      ██
    │     ████████                 │████    ████
    │     ████████                 │████████████
    ├───────┴───────               ├───────────────
            μ                              μ

Points cluster tightly          Points spread widely
around the mean                 from the mean
```

### Properties of Variance

| Property | Formula | Note |
|----------|---------|------|
| Non-negative | $\text{Var}(X) \geq 0$ | |
| Constant | $\text{Var}(c) = 0$ | |
| Scaling | $\text{Var}(aX) = a^2 \text{Var}(X)$ | Squared! |
| Shift | $\text{Var}(X + c) = \text{Var}(X)$ | Shift doesn't affect spread |
| Independent sum | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ | Only if independent! |
| General sum | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ | |

> **⚠️ Common Mistake**: $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ only when $X \perp\!\!\!\perp Y$!

---

## 5. Higher Moments

### Raw Moments

$$\mu'_n = E[X^n]$$

### Central Moments

$$\mu_n = E[(X - \mu)^n]$$

| Moment | Definition | Interpretation |
|--------|-----------|----------------|
| $\mu_1$ | $E[(X - \mu)^1] = 0$ | Always zero |
| $\mu_2$ | $E[(X - \mu)^2] = \text{Var}(X)$ | Spread |
| $\mu_3$ | $E[(X - \mu)^3]$ | Asymmetry (unnormalized) |
| $\mu_4$ | $E[(X - \mu)^4]$ | Tail weight (unnormalized) |

---

## 6. Skewness

**Normalized third central moment:**

$$\gamma_1 = \frac{E[(X - \mu)^3]}{\sigma^3} = \frac{\mu_3}{\sigma^3}$$

| Value | Interpretation |
|-------|----------------|
| $\gamma_1 > 0$ | Right-skewed (long right tail) |
| $\gamma_1 = 0$ | Symmetric |
| $\gamma_1 < 0$ | Left-skewed (long left tail) |

```
Skewness Examples:

Left-skewed (γ₁ < 0)    Symmetric (γ₁ = 0)    Right-skewed (γ₁ > 0)
        ╱██                   ██                    ██╲
       ╱████                 ████                  ████╲
      ╱██████               ██████                ██████╲
    ╱████████             ████████              ████████╲
  ╱██████████           ██████████            ██████████╲
─────────────────     ─────────────────     ─────────────────
     median > mean       median = mean       median < mean

Examples:                Examples:               Examples:
• Exam scores          • Normal distribution   • Income distribution
  (hard test)          • Height                • Wealth
• Age at death                                 • File sizes
```

### ML Relevance

- Right-skewed features may benefit from log transformation
- Skewed target variables can affect loss functions
- Helps detect data quality issues

---

## 7. Kurtosis

**Normalized fourth central moment:**

$$\gamma_2 = \frac{E[(X - \mu)^4]}{\sigma^4} = \frac{\mu_4}{\sigma^4}$$

**Excess Kurtosis:** $\gamma_2 - 3$ (Normal distribution has kurtosis = 3)

| Value | Name | Interpretation |
|-------|------|----------------|
| $\gamma_2 < 3$ | Platykurtic | Light tails, fewer outliers |
| $\gamma_2 = 3$ | Mesokurtic | Normal-like tails |
| $\gamma_2 > 3$ | Leptokurtic | Heavy tails, more outliers |

```
Kurtosis Comparison:

Platykurtic (κ < 3)     Mesokurtic (κ = 3)      Leptokurtic (κ > 3)
   ┌─────────┐                ╱╲               
   │  ████   │               ╱  ╲                    █
   │████████ │              ╱    ╲                  ███
   │████████ │             ╱      ╲                █████
───┴────────┴───        ──╱────────╲──          ╱████████╲
                                               ╱          ╲
• Uniform dist         • Normal dist          • Student's t
• Bounded data         • Most common          • t-distribution
• Light tails          • Reference            • Financial returns
                                              • Heavy tails
```

### ML Relevance

- High kurtosis → more outliers → need robust methods
- Student's t distribution for robust regression
- Important for risk modeling in finance

---

## 8. Moment Generating Function (MGF)

### Definition

$$M_X(t) = E[e^{tX}]$$

**Discrete:**
$$M_X(t) = \sum_x e^{tx} p(x)$$

**Continuous:**
$$M_X(t) = \int_{-\infty}^{\infty} e^{tx} f(x) \, dx$$

### Computing Moments from MGF

$$E[X^n] = M_X^{(n)}(0) = \frac{d^n M_X(t)}{dt^n}\bigg|_{t=0}$$

```
MGF → Moments:

        ┌───────────────┐
        │   M_X(t)      │
        └───────┬───────┘
                │
    ┌───────────┼───────────┐
    │           │           │
    ▼           ▼           ▼
 M'(0)       M''(0)     M'''(0)
 = E[X]      = E[X²]    = E[X³]
    │           │           │
    │           ▼           │
    │    Var = E[X²]-E[X]²  │
    └───────────┴───────────┘
```

### Common MGFs

| Distribution | MGF $M_X(t)$ |
|-------------|--------------|
| Bernoulli($p$) | $1 - p + pe^t$ |
| Binomial($n,p$) | $(1 - p + pe^t)^n$ |
| Poisson($\lambda$) | $e^{\lambda(e^t - 1)}$ |
| Normal($\mu,\sigma^2$) | $e^{\mu t + \frac{1}{2}\sigma^2 t^2}$ |
| Exponential($\lambda$) | $\frac{\lambda}{\lambda - t}$, $t < \lambda$ |

### Key MGF Property

$$M_{X+Y}(t) = M_X(t) \cdot M_Y(t) \quad \text{if } X \perp\!\!\!\perp Y$$

> **🔑 Why MGFs matter**: MGF uniquely determines the distribution! If two RVs have the same MGF, they have the same distribution.

---

## 9. Probability Inequalities

### Markov's Inequality

For $X \geq 0$:
$$P(X \geq a) \leq \frac{E[X]}{a}$$

### Chebyshev's Inequality

$$P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}$$

```
Chebyshev's Inequality:

P(|X - μ| ≥ kσ) ≤ 1/k²

k=1:  At most 100% outside 1σ  (trivial)
k=2:  At most 25% outside 2σ
k=3:  At most 11% outside 3σ
k=4:  At most 6.25% outside 4σ

     ░░░░░░████████████████░░░░░░
         └──────────────────┘
          At least 1 - 1/k²
          within k std devs
```

### Jensen's Inequality

If $g$ is **convex**:
$$g(E[X]) \leq E[g(X)]$$

If $g$ is **concave**:
$$g(E[X]) \geq E[g(X)]$$

```
Jensen's Inequality (convex g):

g(x)
    │           ╱
    │         ╱   ● E[g(X)]
    │       ╱     │
    │     ╱───────┤
    │   ╱ ●       │ ● g(E[X])
    │ ╱    ╲     ╱
    │        ● ●
    └─────────────────
       X₁  E[X]  X₂

The average of g(X) is above g of the average
→ E[g(X)] ≥ g(E[X])
```

> **🔑 ML Applications of Jensen's:**
> - KL divergence is non-negative
> - Log-likelihood bounds (EM algorithm)
> - Variational inference (ELBO)

---

## 10. Bias-Variance Tradeoff

For an estimator $\hat{\theta}$ of parameter $\theta$:

$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2] = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$$

where:
$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

```
Bias-Variance Tradeoff:

High Bias, Low Var      Balanced             Low Bias, High Var
(Underfitting)                               (Overfitting)

   │   Target: ●         │   Target: ●        │   Target: ●
   │                     │                    │
   │    ○ ○              │      ○             │  ○
   │   ○ ○ ○             │    ○ ● ○           │        ○
   │    ○ ○              │      ○             │    ●
   │                     │                    │  ○       ○
   │  Clustered but      │   Close to         │  Scattered around
   │  away from target   │   target           │  target

MSE = Bias² + Variance
```

### Sample Mean as Estimator

For $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$:

$$E[\bar{X}] = \mu \quad \text{(unbiased)}$$
$$\text{Var}(\bar{X}) = \frac{\sigma^2}{n} \quad \text{(decreases with n)}$$

---

## 11. Sample Statistics

### Sample Mean

$$\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$$

### Sample Variance

**Biased (MLE):**
$$\hat{\sigma}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2$$

**Unbiased:**
$$S^2 = \frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2$$

> **💡 Why $n-1$?**: We use 1 degree of freedom to estimate $\bar{X}$, leaving only $n-1$ independent deviations.

### Central Limit Theorem

For i.i.d. $X_1, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$:

$$\bar{X}_n \xrightarrow{d} N\left(\mu, \frac{\sigma^2}{n}\right) \quad \text{as } n \to \infty$$

```
Central Limit Theorem:

Original Distribution          Sum of n samples          n → ∞
(any shape)                   (getting smoother)        (Normal!)

  █                              ╱╲                        ╱╲
 ███                            ╱  ╲                      ╱  ╲
█████                          ╱    ╲                    ╱    ╲
                              ╱      ╲                  ╱      ╲
n = 1                         n = 5                    n → ∞

The sample mean of any distribution approaches Normal!
```

---

## 12. ML Applications

### Expected Loss Minimization

Training = minimize expected loss:
$$\hat{\theta} = \arg\min_\theta E_{(x,y) \sim p_{\text{data}}}[\mathcal{L}(f_\theta(x), y)]$$

In practice, use sample mean:
$$\hat{\theta} \approx \arg\min_\theta \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(x_i), y_i)$$

### Variance in SGD

Mini-batch gradient variance:
$$\text{Var}(\nabla_\theta \mathcal{L}_{\text{batch}}) = \frac{\sigma^2}{B}$$

Larger batch size $B$ → lower variance → more stable training

### Uncertainty Quantification

Predictive uncertainty from posterior:
$$\text{Var}(y | x) = E[\text{Var}(y | x, \theta)] + \text{Var}(E[y | x, \theta])$$

- **Aleatoric uncertainty**: inherent noise (first term)
- **Epistemic uncertainty**: model uncertainty (second term)

---

## 13. Summary Tables

### Key Formulas

| Concept | Formula |
|---------|---------|
| E[X] (discrete) | $\sum_x x \cdot p(x)$ |
| E[X] (continuous) | $\int x \cdot f(x) dx$ |
| Linearity | $E[aX + bY] = aE[X] + bE[Y]$ |
| Var(X) | $E[X^2] - (E[X])^2$ |
| Var(aX + b) | $a^2 \text{Var}(X)$ |
| Var(X + Y) independent | $\text{Var}(X) + \text{Var}(Y)$ |
| Skewness | $E[(X-\mu)^3] / \sigma^3$ |
| Kurtosis | $E[(X-\mu)^4] / \sigma^4$ |
| MGF | $E[e^{tX}]$ |
| $n$-th moment | $M^{(n)}(0)$ |

### Quick Reference

```
┌──────────────────────────────────────────────────────────────┐
│                    MOMENT PROPERTIES                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  EXPECTATION (linear in everything!)                         │
│  ─────────────────────────────────                           │
│  E[aX + bY + c] = aE[X] + bE[Y] + c                          │
│  E[XY] = E[X]E[Y]  (only if X ⊥ Y)                           │
│  E[g(X)] ≠ g(E[X])  (Jensen: = only if g linear)             │
│                                                               │
│  VARIANCE (quadratic in scaling!)                            │
│  ──────────────────────────────                              │
│  Var(aX + b) = a²Var(X)                                      │
│  Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)                    │
│  Var(X + Y) = Var(X) + Var(Y)  (only if X ⊥ Y)               │
│                                                               │
│  SAMPLE STATISTICS                                           │
│  ─────────────────                                           │
│  E[X̄] = μ                                                    │
│  Var(X̄) = σ²/n                                               │
│  X̄ →ᵈ N(μ, σ²/n)  as n → ∞  (CLT)                           │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Exercises

1. Compute E[X²] for X ~ Uniform(0, 1) and verify Var(X) = 1/12
2. Prove that Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
3. Derive the MGF of the Normal distribution
4. Use Chebyshev to bound P(|X - μ| > 3σ) and compare to exact Normal probability
5. Show that sample variance with n-1 is unbiased

---

## References

1. Casella & Berger - "Statistical Inference"
2. Ross - "A First Course in Probability"
3. Wasserman - "All of Statistics"

---

> **Return to**: [01-Introduction](../01-Introduction-and-Random-Variables/) | [Section Overview](../)
