# Hypothesis Testing

## Introduction

Hypothesis testing provides a formal framework for making decisions based on data. In ML, we use these concepts for A/B testing, model comparison, feature selection, and validating experimental results.

## Prerequisites

- Probability distributions
- Estimation theory
- Random sampling

## Learning Objectives

1. Formulate null and alternative hypotheses
2. Understand Type I and Type II errors
3. Calculate and interpret p-values
4. Apply common statistical tests
5. Connect hypothesis testing to ML model validation

---

## 1. Hypothesis Testing Framework

### 1.1 Hypotheses

**Null Hypothesis ($H_0$)**: Default assumption (no effect/difference)

**Alternative Hypothesis ($H_1$ or $H_a$)**: What we want to show

| Type         | Null             | Alternative      |
| ------------ | ---------------- | ---------------- |
| Two-tailed   | $\mu = \mu_0$    | $\mu \neq \mu_0$ |
| Right-tailed | $\mu \leq \mu_0$ | $\mu > \mu_0$    |
| Left-tailed  | $\mu \geq \mu_0$ | $\mu < \mu_0$    |

### 1.2 Decision Process

```
                  ┌─────────────────────────────────────┐
                  │           Collect Data              │
                  └────────────────┬────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │    Calculate Test Statistic         │
                  └────────────────┬────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────┐
                  │      Compute p-value                │
                  └────────────────┬────────────────────┘
                                   ▼
              ┌────────────────────┴───────────────────┐
              │                                        │
              ▼                                        ▼
    ┌─────────────────┐                    ┌─────────────────┐
    │   p ≤ α         │                    │    p > α        │
    │   Reject H₀     │                    │ Fail to reject  │
    └─────────────────┘                    └─────────────────┘
```

---

## 2. Types of Errors

|                | $H_0$ True       | $H_0$ False           |
| -------------- | ---------------- | --------------------- |
| Reject $H_0$   | Type I Error (α) | Correct (Power = 1-β) |
| Fail to Reject | Correct          | Type II Error (β)     |

### 2.1 Type I Error (False Positive)

$$\alpha = P(\text{Reject } H_0 | H_0 \text{ true})$$

- Significance level (usually 0.05 or 0.01)
- Probability of false alarm

### 2.2 Type II Error (False Negative)

$$\beta = P(\text{Fail to reject } H_0 | H_0 \text{ false})$$

- Miss rate
- Often harder to control

### 2.3 Statistical Power

$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 | H_0 \text{ false})$$

Power increases with:

- Larger sample size
- Larger effect size
- Higher α
- Lower variance

```
Error Types Visualization:

           Distribution under H₀          Distribution under H₁
                   │                              │
              ╭────┴────╮                    ╭────┴────╮
             ╱          ╲                   ╱          ╲
            ╱            ╲                 ╱            ╲
           ╱              ╲               ╱              ╲
          ╱                ╲             ╱                ╲
         ╱                  ╲           ╱                  ╲
        ╱██████              ╲         ╱     ░░░░░░░░░░░░░░░╲
       ─────────────────|─────────────────|────────────────────
                   Rejection boundary

       ██ = α (Type I)     ░░░ = 1-β (Power)
       (shown if H₀ true)   (shown if H₁ true)
```

---

## 3. p-Values

### 3.1 Definition

The p-value is the probability of observing data as extreme or more extreme than what was observed, assuming $H_0$ is true.

$$p\text{-value} = P(T \geq t_{obs} | H_0)$$

### 3.2 Interpretation

| p-value         | Interpretation                |
| --------------- | ----------------------------- |
| p < 0.01        | Strong evidence against $H_0$ |
| 0.01 ≤ p < 0.05 | Moderate evidence             |
| 0.05 ≤ p < 0.10 | Weak evidence                 |
| p ≥ 0.10        | Little evidence               |

**Warning**: p-value is NOT:

- Probability that $H_0$ is true
- Probability that result is due to chance
- Measure of effect size

---

## 4. Common Statistical Tests

### 4.1 Z-Test (Known Variance)

**When**: Testing mean, σ² known, large n

**Test Statistic:**
$$Z = \frac{\bar{X} - \mu_0}{\sigma/\sqrt{n}} \sim N(0,1)$$

### 4.2 One-Sample t-Test

**When**: Testing mean, σ² unknown

**Test Statistic:**
$$t = \frac{\bar{X} - \mu_0}{S/\sqrt{n}} \sim t_{n-1}$$

### 4.3 Two-Sample t-Test

**When**: Comparing two means

**Independent samples (equal variance):**
$$t = \frac{\bar{X}_1 - \bar{X}_2}{S_p\sqrt{1/n_1 + 1/n_2}}$$

where $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$

**Welch's t-test (unequal variance):**
$$t = \frac{\bar{X}_1 - \bar{X}_2}{\sqrt{S_1^2/n_1 + S_2^2/n_2}}$$

### 4.4 Paired t-Test

**When**: Before/after measurements on same subjects

**Test Statistic:**
$$t = \frac{\bar{D}}{S_D/\sqrt{n}}$$

where $D_i = X_{1i} - X_{2i}$

### 4.5 Chi-Square Tests

**Goodness of Fit:**
$$\chi^2 = \sum_i \frac{(O_i - E_i)^2}{E_i}$$

**Independence:**
$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

### 4.6 F-Test (ANOVA)

**When**: Comparing multiple means

$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$

---

## 5. Confidence Intervals

### 5.1 Relationship to Hypothesis Testing

A $(1-\alpha)$ confidence interval contains all parameter values that would not be rejected at level $\alpha$.

### 5.2 CI for Mean

$$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}$$

### 5.3 Interpretation

"If we repeated the experiment many times, $(1-\alpha) \times 100\%$ of the constructed intervals would contain the true parameter."

---

## 6. Multiple Testing

### 6.1 Problem

If we do m tests at level α:
$$P(\text{at least one Type I error}) = 1 - (1-\alpha)^m$$

For m=20, α=0.05: $P = 0.64$

### 6.2 Bonferroni Correction

Use $\alpha_{adj} = \alpha/m$ for each test.

Conservative but simple.

### 6.3 False Discovery Rate (FDR)

**Benjamini-Hochberg procedure:**

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest k where $p_{(k)} \leq k \cdot \alpha / m$
3. Reject all hypotheses with $p \leq p_{(k)}$

Less conservative, controls expected proportion of false discoveries.

---

## 7. Effect Size

### 7.1 Why Effect Size Matters

- p-value depends on sample size
- Large n → small p even for tiny effects
- Effect size measures practical significance

### 7.2 Cohen's d

$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_{pooled}}$$

| d   | Interpretation |
| --- | -------------- |
| 0.2 | Small          |
| 0.5 | Medium         |
| 0.8 | Large          |

---

## 8. ML Applications

### 8.1 A/B Testing

Testing if new model B is better than baseline A:

- $H_0$: $\mu_B = \mu_A$
- $H_1$: $\mu_B > \mu_A$

### 8.2 Model Comparison

**Paired t-test** for comparing models on same test folds.

### 8.3 Feature Selection

- t-test for continuous features
- Chi-square for categorical features
- F-test (ANOVA) for multi-class

### 8.4 Statistical vs Practical Significance

| Scenario              | Statistical       | Practical             |
| --------------------- | ----------------- | --------------------- |
| Large n, tiny effect  | ✓ Significant     | ✗ Not meaningful      |
| Small n, large effect | ✗ Not significant | ✓ Might be meaningful |
| Large n, large effect | ✓ Significant     | ✓ Meaningful          |

---

## 9. Permutation Tests

### 9.1 Non-parametric Alternative

No distributional assumptions.

**Procedure:**

1. Calculate observed test statistic
2. Randomly permute labels many times
3. Calculate statistic for each permutation
4. p-value = proportion of permuted statistics ≥ observed

### 9.2 Advantages

- No normality assumption
- Exact p-values
- Works for any test statistic

---

## 10. Summary

| Test         | Use Case        | Assumptions          |
| ------------ | --------------- | -------------------- |
| z-test       | Mean, σ known   | Normal or large n    |
| t-test       | Mean, σ unknown | Normal or large n    |
| Two-sample t | Compare means   | Independence         |
| Paired t     | Before/after    | Paired data          |
| Chi-square   | Categorical     | Expected counts ≥ 5  |
| F-test       | Multiple means  | Normality, equal var |
| Permutation  | Any             | None                 |

**Key Concepts:**

- p-value: Evidence against $H_0$
- Type I/II errors: False positive/negative
- Power: Ability to detect true effects
- Effect size: Practical significance

---

## References

1. Casella & Berger - "Statistical Inference"
2. Wasserman - "All of Statistics"
3. Efron & Tibshirani - "An Introduction to the Bootstrap"
