# Hypothesis Testing

> **Navigation**: [← 02-Estimation-Theory](../02-Estimation-Theory/) | [Statistics](../) | [04-Bayesian-Inference →](../04-Bayesian-Inference/)

## Introduction

Hypothesis testing provides a formal framework for making decisions based on data. In ML, we use these concepts for A/B testing, model comparison, feature selection, and validating experimental results.

```
Hypothesis Testing in ML Workflow:
══════════════════════════════════════════════════════════════════

  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────┐
  │   Model A   │    │                  │    │                 │
  │  (Baseline) │───▶│   Statistical    │───▶│    Decision:    │
  └─────────────┘    │      Test        │    │  Deploy B or    │
  ┌─────────────┐    │                  │    │  Keep A?        │
  │   Model B   │───▶│   p-value,       │    │                 │
  │    (New)    │    │   confidence     │    │                 │
  └─────────────┘    └──────────────────┘    └─────────────────┘

                     Applications:
                     • A/B testing
                     • Model comparison
                     • Feature selection
                     • Detecting data drift
```

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

> 💡 **Intuition**: Think of it like a court trial. $H_0$ = "innocent" (default). We need strong evidence to reject $H_0$ and conclude $H_1$ = "guilty".

### 1.2 Decision Process

```
Hypothesis Testing Flowchart:
══════════════════════════════════════════════════════════════════

                    ┌─────────────────────────────────────┐
                    │     1. State H₀ and H₁              │
                    └────────────────┬────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────┐
                    │     2. Choose significance level α   │
                    │        (usually 0.05 or 0.01)       │
                    └────────────────┬────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────┐
                    │     3. Collect data & calculate     │
                    │        test statistic               │
                    └────────────────┬────────────────────┘
                                     ▼
                    ┌─────────────────────────────────────┐
                    │     4. Compute p-value              │
                    └────────────────┬────────────────────┘
                                     ▼
                ┌────────────────────┴───────────────────┐
                │                                        │
                ▼                                        ▼
      ┌─────────────────┐                    ┌─────────────────┐
      │   p ≤ α         │                    │    p > α        │
      │   Reject H₀     │                    │ Fail to reject  │
      │   "Significant" │                    │   H₀            │
      └─────────────────┘                    └─────────────────┘
```

---

## 2. Types of Errors

|                    | $H_0$ True           | $H_0$ False              |
| ------------------ | -------------------- | ------------------------ |
| **Reject $H_0$**   | Type I Error (α)     | ✓ Correct (Power = 1-β) |
| **Fail to Reject** | ✓ Correct            | Type II Error (β)        |

### 2.1 Type I Error (False Positive)

$$\alpha = P(\text{Reject } H_0 | H_0 \text{ true})$$

- Significance level (usually 0.05 or 0.01)
- Probability of false alarm
- **ML Example**: Concluding new model is better when it's not

### 2.2 Type II Error (False Negative)

$$\beta = P(\text{Fail to reject } H_0 | H_0 \text{ false})$$

- Miss rate
- Often harder to control
- **ML Example**: Keeping old model when new one is actually better

### 2.3 Statistical Power

$$\text{Power} = 1 - \beta = P(\text{Reject } H_0 | H_0 \text{ false})$$

Power increases with:

- Larger sample size
- Larger effect size
- Higher α (but more Type I errors)
- Lower variance

```
Error Types Visualization:
══════════════════════════════════════════════════════════════════

        Distribution if H₀ True         Distribution if H₁ True
                 
                 ╱╲                              ╱╲
                ╱  ╲                            ╱  ╲
               ╱    ╲                          ╱    ╲
              ╱      ╲                        ╱      ╲
             ╱        ╲                      ╱        ╲
            ╱          ╲                    ╱          ╲
           ╱     ▓▓▓▓▓▓▓╲                  ╱░░░░░░░░░░░░╲
    ──────────────|───────────────────────────|────────────────
               Critical                    Critical
               Value                       Value
                   
        ▓▓▓ = α (Type I Error)      ░░░ = Power (1-β)
        "False Positive"             "True Positive"
        
        The unshaded area under      The unshaded area under
        H₁ curve left of critical    H₀ curve right of critical
        value = β (Type II Error)    value = Correct rejection

┌───────────────────────────────────────────────────────────────┐
│  α ↓  →  Critical value moves right  →  β ↑ (less power)     │
│  n ↑  →  Distributions narrower      →  Both errors ↓        │
└───────────────────────────────────────────────────────────────┘
```

> ⚠️ **Common Mistake**: "Fail to reject $H_0$" does NOT mean "$H_0$ is true". It means we don't have enough evidence against $H_0$. Absence of evidence ≠ evidence of absence.

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

> ⚠️ **Critical Warnings about p-values**:
> 
> p-value is **NOT**:
> - Probability that $H_0$ is true
> - Probability that result is due to chance
> - Measure of effect size
> 
> p-value **IS**:
> - Probability of seeing data this extreme IF $H_0$ were true

```
p-value Intuition:
══════════════════════════════════════════════════════════════════

If H₀ is true, how surprising is our observed data?

      Distribution under H₀
              ╱╲
             ╱  ╲
            ╱    ╲
           ╱      ╲
          ╱        ╲
         ╱          ╲▓▓▓▓▓▓
        ╱________________▓▓▓
    ───────────────────|─────────▶
                       │
                   Our observed
                   test statistic
                   
        ▓▓▓ = p-value (area in tail beyond our observation)

Small p-value → Our data is surprising under H₀
              → Evidence against H₀
```

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

```
Test Selection Guide:
══════════════════════════════════════════════════════════════════

              ┌──────────────────────────────────────────┐
              │          What are you comparing?         │
              └────────────────────┬─────────────────────┘
                                   │
         ┌─────────────┬───────────┼───────────┬──────────────┐
         │             │           │           │              │
         ▼             ▼           ▼           ▼              ▼
    One Mean      Two Means    >2 Means   Proportions   Categorical
         │             │           │           │              │
         ▼             ▼           ▼           ▼              ▼
    ┌─────────┐  ┌──────────┐  ┌───────┐  ┌─────────┐  ┌──────────┐
    │σ known? │  │Paired or │  │ANOVA  │  │Z-test   │  │Chi-square│
    │Y: Z-test│  │Indep?    │  │(F-test)│ │for      │  │test      │
    │N: t-test│  │          │  │        │ │propor-  │  │          │
    └─────────┘  └──────────┘  └───────┘  │tions    │  └──────────┘
                      │                   └─────────┘
              ┌───────┴───────┐
              │               │
              ▼               ▼
         ┌─────────┐    ┌──────────┐
         │ Paired  │    │Two-sample│
         │ t-test  │    │ t-test   │
         └─────────┘    └──────────┘
```

---

## 5. Confidence Intervals

### 5.1 Relationship to Hypothesis Testing

A $(1-\alpha)$ confidence interval contains all parameter values that would not be rejected at level $\alpha$.

### 5.2 CI for Mean

$$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{S}{\sqrt{n}}$$

### 5.3 Interpretation

"If we repeated the experiment many times, $(1-\alpha) \times 100\%$ of the constructed intervals would contain the true parameter."

> ⚠️ **Common Misinterpretation**: A 95% CI does NOT mean "95% probability the true value is in this interval." The interval either contains the true value or it doesn't. The 95% refers to the long-run frequency.

```
Confidence Interval vs Hypothesis Test:
══════════════════════════════════════════════════════════════════

                    95% Confidence Interval
                    ├──────────────────────────────┤
                    ▼                              ▼
    ────────────────[████████████████████████████████]─────────────
                    │                              │
                    Lower                          Upper
                    Bound                          Bound

    If H₀: μ = μ₀:
    
    Case 1: μ₀ inside CI  →  Fail to reject H₀ at α = 0.05
    
            [███████████████μ₀████████████████]
                            ▲
                         Not rejected
    
    Case 2: μ₀ outside CI  →  Reject H₀ at α = 0.05
    
    μ₀      [██████████████████████████████████]
    ▲
    Rejected
```

---

## 6. Multiple Testing

### 6.1 The Problem

If we do m tests at level α:
$$P(\text{at least one Type I error}) = 1 - (1-\alpha)^m$$

For m=20, α=0.05: $P = 0.64$ — More likely than not to get a false positive!

```
Multiple Testing Problem:
══════════════════════════════════════════════════════════════════

Number of      P(at least one      Expected false
  tests         false positive)     positives
────────────────────────────────────────────────────
     1              0.05                0.05
     5              0.23                0.25
    10              0.40                0.50
    20              0.64                1.00
    50              0.92                2.50
   100              0.99                5.00

    ⚠️ Testing many features = Many false discoveries!
```

### 6.2 Bonferroni Correction

Use $\alpha_{adj} = \alpha/m$ for each test.

Conservative but simple. Guarantees family-wise error rate ≤ α.

### 6.3 False Discovery Rate (FDR)

**Benjamini-Hochberg procedure:**

1. Order p-values: $p_{(1)} \leq p_{(2)} \leq ... \leq p_{(m)}$
2. Find largest k where $p_{(k)} \leq k \cdot \alpha / m$
3. Reject all hypotheses with $p \leq p_{(k)}$

Less conservative, controls expected proportion of false discoveries.

> 💡 **When to Use Which**:
> - **Bonferroni**: When any false positive is costly (e.g., clinical trials)
> - **FDR**: When some false positives are acceptable (e.g., exploratory analysis, feature selection)

---

## 7. Effect Size

### 7.1 Why Effect Size Matters

- p-value depends on sample size
- Large n → small p even for tiny effects
- Effect size measures **practical significance**

### 7.2 Cohen's d

$$d = \frac{\bar{X}_1 - \bar{X}_2}{S_{pooled}}$$

| d   | Interpretation |
| --- | -------------- |
| 0.2 | Small          |
| 0.5 | Medium         |
| 0.8 | Large          |

```
Statistical vs Practical Significance:
══════════════════════════════════════════════════════════════════

                 Small n                      Large n
            ┌────────────────┐           ┌────────────────┐
Small       │ p > 0.05       │           │ p < 0.001      │
Effect      │ Not significant│           │ Significant    │
(d=0.1)     │ Not meaningful │           │ NOT meaningful │
            └────────────────┘           └────────────────┘
            
            ┌────────────────┐           ┌────────────────┐
Large       │ p = 0.08       │           │ p < 0.001      │
Effect      │ Not significant│           │ Significant    │
(d=0.8)     │ BUT meaningful!│           │ AND meaningful!│
            └────────────────┘           └────────────────┘
            
    ★ Always report BOTH p-value AND effect size!
```

---

## 8. ML Applications

### 8.1 A/B Testing

Testing if new model B is better than baseline A:

- $H_0$: $\mu_B = \mu_A$ (no difference)
- $H_1$: $\mu_B > \mu_A$ (B is better)

```
A/B Testing Pipeline:
══════════════════════════════════════════════════════════════════

┌─────────────┐                         ┌─────────────┐
│   Users     │                         │   Users     │
│  (Group A)  │─────┐           ┌───────│  (Group B)  │
└─────────────┘     │           │       └─────────────┘
                    ▼           ▼
              ┌──────────┐ ┌──────────┐
              │ Model A  │ │ Model B  │
              │(control) │ │(variant) │
              └────┬─────┘ └────┬─────┘
                   │            │
                   ▼            ▼
              ┌──────────────────────┐
              │  Collect Metrics     │
              │  (CTR, conversion,   │
              │   revenue, etc.)     │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Statistical Test    │
              │  (t-test, z-test)    │
              └──────────┬───────────┘
                         ▼
              ┌──────────────────────┐
              │  Decision: Deploy B? │
              └──────────────────────┘
```

### 8.2 Model Comparison

**Paired t-test** for comparing models on same test folds:

1. Run both models on k-fold cross-validation
2. Pair accuracy scores fold-by-fold
3. Test if mean difference is significant

### 8.3 Feature Selection

| Feature Type   | Recommended Test           |
| -------------- | -------------------------- |
| Continuous     | t-test, ANOVA, correlation |
| Categorical    | Chi-square test            |
| Mixed          | Mutual information         |

### 8.4 Statistical vs Practical Significance in ML

| Scenario              | Statistical       | Practical             | Action              |
| --------------------- | ----------------- | --------------------- | ------------------- |
| Large n, tiny effect  | ✓ Significant     | ✗ Not meaningful      | Don't deploy        |
| Small n, large effect | ✗ Not significant | ✓ Might be meaningful | Collect more data   |
| Large n, large effect | ✓ Significant     | ✓ Meaningful          | Deploy!             |

---

## 9. Permutation Tests

### 9.1 Non-parametric Alternative

No distributional assumptions required.

**Procedure:**

1. Calculate observed test statistic
2. Randomly permute labels many times (e.g., 10,000)
3. Calculate statistic for each permutation
4. p-value = proportion of permuted statistics ≥ observed

```
Permutation Test Procedure:
══════════════════════════════════════════════════════════════════

Original Data:
Group A: [a₁, a₂, a₃]    Group B: [b₁, b₂, b₃]

Observed difference: d_obs = mean(B) - mean(A) = 2.5

Permutation 1: Shuffle labels randomly
[a₁, b₂, a₃] vs [b₁, a₂, b₃] → d₁ = 0.3

Permutation 2: Shuffle again
[b₁, a₁, b₃] vs [a₂, b₂, a₃] → d₂ = -1.2

... repeat 10,000 times ...

┌─────────────────────────────────────────────────────────────┐
│     Distribution of permuted statistics                     │
│                                                             │
│              ╱╲                                             │
│             ╱  ╲                                            │
│            ╱    ╲                                 │         │
│           ╱      ╲                                │ d_obs   │
│          ╱        ╲                               │         │
│         ╱__________╲                              ▼         │
│    ────────────────────────────────────────|─────|─────     │
│                                           2.0   2.5         │
│                                                             │
│    p-value = (# permutations with d ≥ 2.5) / 10,000        │
│            = 23 / 10,000 = 0.0023                          │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Advantages

- No normality assumption  
- Exact p-values
- Works for any test statistic
- Great for small samples

---

## 10. Summary

### Test Selection Quick Reference

| Test         | Use Case              | Assumptions            |
| ------------ | --------------------- | ---------------------- |
| z-test       | Mean, σ known         | Normal or large n      |
| t-test       | Mean, σ unknown       | Normal or large n      |
| Two-sample t | Compare means         | Independence           |
| Paired t     | Before/after          | Paired data            |
| Chi-square   | Categorical           | Expected counts ≥ 5    |
| F-test       | Multiple means        | Normality, equal var   |
| Permutation  | Any                   | None                   |

### Key Concepts Cheat Sheet

```
Quick Reference:
══════════════════════════════════════════════════════════════════

Type I Error (α):   Reject H₀ when true    (False Positive)
Type II Error (β):  Accept H₀ when false   (False Negative)
Power (1-β):        Reject H₀ when false   (True Positive)

p-value:  P(data this extreme | H₀ true)
          Small p → Reject H₀

Effect Size (Cohen's d):  Practical magnitude of difference
          0.2 = small, 0.5 = medium, 0.8 = large

Multiple Testing:  More tests = More false positives
          Bonferroni: α/m (conservative)
          FDR: Controls proportion of false discoveries
```

---

## Exercises

1. **Error Types**: In spam detection, which is worse: Type I (marking good email as spam) or Type II (missing spam)? How would you adjust α?

2. **Power Calculation**: You expect effect size d = 0.5 and want 80% power at α = 0.05. How many samples per group do you need? (Hint: use power analysis tools)

3. **Multiple Testing**: You test 100 features for significance at α = 0.05. If all null hypotheses are true, how many false positives do you expect? What would Bonferroni correction make your per-test α?

4. **A/B Test**: Model A has accuracy 85% (n=1000), Model B has 86% (n=1000). Is this difference significant? Is it practically meaningful?

---

## References

1. Casella & Berger - "Statistical Inference"
2. Wasserman - "All of Statistics"
3. Efron & Tibshirani - "An Introduction to the Bootstrap"
4. Kohavi et al. - "Trustworthy Online Controlled Experiments" (A/B testing)
