# Descriptive Statistics

> **Navigation**: [← 04-Expectation-and-Moments](../../06-Probability-Theory/04-Expectation-and-Moments/) | [Statistics](../) | [02-Estimation-Theory →](../02-Estimation-Theory/)

## Introduction

Descriptive statistics summarize and describe the main features of a dataset. Before building ML models, understanding your data through descriptive statistics is crucial for feature engineering, outlier detection, and model selection.

```
The Data Science Pipeline - Where Descriptive Stats Fit:
═══════════════════════════════════════════════════════════

  ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐    ┌──────────────┐
  │  Raw Data   │ ─▶ │  DESCRIPTIVE     │ ─▶ │   Feature   │ ─▶ │    Model     │
  │             │    │  STATISTICS      │    │ Engineering │    │  Training    │
  └─────────────┘    └──────────────────┘    └─────────────┘    └──────────────┘
                            │
                            ▼
                     ┌──────────────────┐
                     │ • Central tendency│
                     │ • Spread/variance │
                     │ • Distribution    │
                     │ • Correlations    │
                     │ • Outlier detect  │
                     └──────────────────┘
```

## Prerequisites

- Basic probability
- Expectation and variance

## Learning Objectives

1. Compute measures of central tendency
2. Calculate measures of spread
3. Understand data distributions and shape
4. Apply descriptive statistics to ML preprocessing

---

## 1. Measures of Central Tendency

### 1.1 Mean (Arithmetic Average)

$$\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$$

Properties:

- Minimizes sum of squared deviations
- Sensitive to outliers
- Used in: MSE loss, batch normalization

### 1.2 Median

Middle value when data is sorted.

$$\text{Median} = \begin{cases} x_{(n+1)/2} & \text{if } n \text{ odd} \\ \frac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ even} \end{cases}$$

Properties:

- Robust to outliers
- Minimizes sum of absolute deviations
- Used in: MAE loss, robust statistics

### 1.3 Mode

Most frequently occurring value.

- Can have multiple modes (bimodal, multimodal)
- Useful for categorical data
- Used in: Classification (majority voting)

```
Comparison on Skewed Data:
═══════════════════════════════════════

Right-skewed distribution:

Frequency
    │
  8 │     █
  6 │    ██
  4 │   ███
  2 │  █████████
    └─────────────────────▶ Value
        ▲   ▲    ▲
        │   │    │
      Mode │   Mean
           │
         Median

Rule of thumb for right-skewed: Mode < Median < Mean
```

> 💡 **Key Insight**: The relationship between mean, median, and mode tells you about skewness. For symmetric distributions, all three are equal. For skewed data, prefer the median as it's more robust.

### 1.4 Weighted Mean

$$\bar{x}_w = \frac{\sum_{i=1}^n w_i x_i}{\sum_{i=1}^n w_i}$$

Used in: Sample weights, attention mechanisms

### 1.5 Geometric Mean

$$\bar{x}_g = \left(\prod_{i=1}^n x_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i=1}^n \ln x_i\right)$$

Used for: Growth rates, ratios, multiplicative processes

### 1.6 Harmonic Mean

$$\bar{x}_h = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}$$

Used for: F1-score (harmonic mean of precision and recall)

```
Relationship Between Means:
═══════════════════════════

For positive numbers: Harmonic ≤ Geometric ≤ Arithmetic

Example with [2, 8]:
  Harmonic Mean:   2/(1/2 + 1/8)  = 3.2
  Geometric Mean:  √(2 × 8)       = 4.0
  Arithmetic Mean: (2 + 8)/2      = 5.0

              ◀──────────────────────────▶
              3.2    4.0           5.0
               HM     GM            AM
```

---

## 2. Measures of Spread

### 2.1 Range

$$\text{Range} = x_{\max} - x_{\min}$$

Simple but sensitive to outliers.

### 2.2 Variance and Standard Deviation

**Sample Variance:**
$$s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$$

**Standard Deviation:**
$$s = \sqrt{s^2}$$

> 💡 **Why n-1?** Bessel's correction makes sample variance an unbiased estimator of population variance. We lose one degree of freedom because we estimate $\bar{x}$ from the data.

### 2.3 Interquartile Range (IQR)

$$\text{IQR} = Q_3 - Q_1$$

Where Q1 = 25th percentile, Q3 = 75th percentile.

Robust to outliers. Used in box plots.

### 2.4 Mean Absolute Deviation (MAD)

$$\text{MAD} = \frac{1}{n}\sum_{i=1}^n |x_i - \bar{x}|$$

More robust than standard deviation.

### 2.5 Coefficient of Variation

$$\text{CV} = \frac{s}{\bar{x}}$$

Normalized measure of dispersion (unitless). Useful for comparing variability between datasets with different scales.

```
Spread Measures - Robustness Comparison:
════════════════════════════════════════

Dataset: [1, 2, 3, 4, 5, 100]  ← Outlier!

Measure       │ Value  │ Robust?
──────────────┼────────┼─────────
Range         │   99   │  No ✗
Std Dev       │  39.6  │  No ✗
IQR           │   2.5  │  Yes ✓
MAD           │  15.6  │  Partial

Without outlier [1, 2, 3, 4, 5]:
Range = 4, Std = 1.58, IQR = 2.5, MAD = 1.2
```

---

## 3. Percentiles and Quantiles

### Percentile

The p-th percentile is the value below which p% of data falls.

- 25th percentile = Q1 (first quartile)
- 50th percentile = Q2 (median)
- 75th percentile = Q3 (third quartile)

### Five-Number Summary

$$(\min, Q_1, \text{median}, Q_3, \max)$$

```
Box Plot Anatomy:
═══════════════════════════════════════════════════

               Lower        Upper
               Fence        Fence
                 │            │
    ○     ├─────────┬────┼────┬─────────┤     ○
          │         │    │    │         │
         Min       Q1   Med  Q3        Max
          │         │◀──IQR──▶│         │
          │                             │
Potential │                             │ Potential
Outlier   │                             │ Outlier

Lower Fence = Q1 - 1.5 × IQR
Upper Fence = Q3 + 1.5 × IQR
```

### Outlier Detection (IQR Method)

- Lower fence: $Q_1 - 1.5 \times \text{IQR}$
- Upper fence: $Q_3 + 1.5 \times \text{IQR}$

Points outside fences are potential outliers.

> ⚠️ **Common Mistake**: The 1.5×IQR rule is a heuristic, not a rigorous statistical test. In high-dimensional data, apparent "outliers" may be valid data points. Always investigate before removing.

---

## 4. Shape of Distribution

### 4.1 Skewness

$$\text{Skewness} = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3$$

| Value | Interpretation               |
| ----- | ---------------------------- |
| = 0   | Symmetric                    |
| > 0   | Right-skewed (tail on right) |
| < 0   | Left-skewed (tail on left)   |

```
Skewness Visualization:
════════════════════════════════════════════════════════════

Left-Skewed             Symmetric              Right-Skewed
(Negative)              (Zero)                 (Positive)

       ╱╲                 ╱╲                      ╱╲
      ╱  ╲               ╱  ╲                    ╱  ╲
     ╱    ╲             ╱    ╲                  ╱    ╲
    ╱      ╲           ╱      ╲                ╱      ╲
▁▂▃▅        ▇▇▇      ▇▇        ▇▇            ▇▇▇        ▅▃▂▁

Mean < Median        Mean = Median          Mean > Median

Example: Exam          Example:              Example:
scores with ceiling    Height                Income
```

### 4.2 Kurtosis

$$\text{Kurtosis} = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4$$

**Excess Kurtosis** = Kurtosis - 3

| Value | Interpretation            |
| ----- | ------------------------- |
| = 0   | Normal-like tails         |
| > 0   | Heavy tails (leptokurtic) |
| < 0   | Light tails (platykurtic) |

```
Kurtosis Comparison:
════════════════════════════════════════════════════════════

Leptokurtic             Mesokurtic            Platykurtic
(Heavy tails)           (Normal)              (Light tails)
Excess K > 0            Excess K = 0          Excess K < 0

        █                    █                   ████
        █                   ███                 ██████
       ███                 █████               ████████
      █████               ███████             ██████████
   ███████████           █████████           ████████████

More extreme            Normal                Less extreme
values likely           tails                 values
```

> 💡 **ML Relevance**: High kurtosis (heavy tails) indicates more extreme values, which can affect loss functions. Consider robust losses (Huber, quantile) or data transformations.

---

## 5. Correlation and Association

### 5.1 Pearson Correlation

$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

Measures linear relationship: $-1 \leq r \leq 1$

```
Correlation Patterns:
════════════════════════════════════════════════════════════

r = +1          r = +0.7        r = 0           r = -0.7        r = -1
Perfect +       Strong +        None            Strong -         Perfect -

    ••             •              • • •           •                ••
   ••           •••              •••••            •••               ••
  ••          •••••            •••••••            •••••            ••
 ••           •••              •••••••              •••••          ••
••             •               • • •                   •          ••
```

### 5.2 Spearman Rank Correlation

$$\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $d_i$ = difference in ranks.

Measures monotonic relationship. Robust to outliers.

> ⚠️ **Common Pitfall**: Pearson correlation only captures LINEAR relationships. A classic example: X and X² have Pearson r ≈ 0 but are perfectly related. Always visualize your data!

### 5.3 Covariance Matrix

For multivariate data $\mathbf{X} \in \mathbb{R}^{n \times p}$:

$$\mathbf{S} = \frac{1}{n-1}(\mathbf{X} - \bar{\mathbf{X}})^T(\mathbf{X} - \bar{\mathbf{X}})$$

Used in: PCA, Mahalanobis distance

---

## 6. Data Normalization

### 6.1 Z-Score Standardization

$$z_i = \frac{x_i - \bar{x}}{s}$$

Result: mean = 0, std = 1

### 6.2 Min-Max Scaling

$$x'_i = \frac{x_i - x_{\min}}{x_{\max} - x_{\min}}$$

Result: values in [0, 1]

### 6.3 Robust Scaling

$$x'_i = \frac{x_i - \text{median}}{\text{IQR}}$$

Robust to outliers.

```
Normalization Methods Comparison:
════════════════════════════════════════════════════════════

Original Data: [1, 2, 3, 4, 100]

Method          │ Result            │ Best When
────────────────┼───────────────────┼──────────────────────
Z-Score         │ [-0.5, ..., 2.4]  │ Gaussian-ish data
Min-Max         │ [0, 0.01, ..., 1] │ Bounded features
Robust Scaling  │ [-0.67, ..., 32]  │ Outliers present

Z-Score vs Min-Max in presence of outliers:

Z-Score  ────○───○───○───○────────────────────────○────
         -0.5 -0.3  0  0.2                        2.4

Min-Max  ○○○○───────────────────────────────────────○
         0.01                                      1.0
         ▲
         │ Most data crushed near 0!
```

> 💡 **Practical Tip**: Use Z-score for most algorithms (neural nets, logistic regression). Use Min-Max for algorithms requiring bounded input (some neural net activations). Use Robust Scaling when outliers exist but shouldn't dominate.

---

## 7. ML Applications

| Statistic   | ML Application                             |
| ----------- | ------------------------------------------ |
| Mean        | Batch normalization, missing value imputation |
| Median      | Robust imputation, MAE loss metric         |
| Std         | Feature scaling, regularization decisions  |
| Correlation | Feature selection, multicollinearity check |
| Skewness    | Log/Box-Cox transform decisions            |
| Kurtosis    | Robust loss function selection             |
| Outliers    | Data cleaning, anomaly detection           |

```
Descriptive Stats → ML Decisions:
════════════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────────┐
  │ High Skewness?  ───▶  Apply log/sqrt transform          │
  │                                                         │
  │ High Correlation ───▶  Remove redundant features        │
  │ between features?      or use PCA                       │
  │                                                         │
  │ Many Outliers?  ───▶  Use robust scaler                 │
  │                       Use MAE instead of MSE            │
  │                                                         │
  │ Missing Values?  ───▶  Impute with mean (symmetric)     │
  │                        or median (skewed)               │
  │                                                         │
  │ Different Scales? ───▶  Standardize features            │
  │                         (especially for distance-based) │
  └─────────────────────────────────────────────────────────┘
```

---

## 8. Summary

### Quick Reference Table

| Measure  | Formula                            | Robust? | Use Case              |
| -------- | ---------------------------------- | ------- | --------------------- |
| Mean     | $\frac{1}{n}\sum x_i$              | No      | MSE, normalization    |
| Median   | Middle value                       | Yes     | MAE, skewed data      |
| Mode     | Most frequent                      | Yes     | Categorical data      |
| Variance | $\frac{1}{n-1}\sum(x_i-\bar{x})^2$ | No      | Spread measurement    |
| Std      | $\sqrt{\text{Var}}$                | No      | Feature scaling       |
| IQR      | $Q_3 - Q_1$                        | Yes     | Outlier detection     |
| Skewness | 3rd standardized moment            | No      | Distribution shape    |
| Kurtosis | 4th standardized moment            | No      | Tail behavior         |

### When to Use What

```
Decision Tree for Central Tendency:
════════════════════════════════════════

         ┌─── Categorical? ───▶ MODE
         │
Data ────┼─── Ordinal? ───▶ MEDIAN
         │
         └─── Continuous?
                  │
                  ├── Symmetric? ───▶ MEAN
                  │
                  └── Skewed or
                      Outliers? ───▶ MEDIAN
```

---

## Exercises

1. **Outlier Impact**: Given [10, 12, 11, 13, 12, 100], calculate mean, median, std, and IQR. Which measures are affected most by the outlier?

2. **Normalization Choice**: You have income data ranging from $20K to $10M with high right skew. Which normalization would you use and why?

3. **Correlation Types**: Two variables have Pearson r = 0.1 but Spearman ρ = 0.95. What does this tell you about their relationship?

4. **Feature Engineering**: Given a feature with skewness = 2.5, what transformation might you apply before feeding it to a linear model?

---

## References

1. Wasserman - "All of Statistics"
2. James et al. - "Introduction to Statistical Learning"
3. Freedman et al. - "Statistics" (4th edition)
