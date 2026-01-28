# Descriptive Statistics

## Introduction

Descriptive statistics summarize and describe the main features of a dataset. Before building ML models, understanding your data through descriptive statistics is crucial for feature engineering, outlier detection, and model selection.

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
Comparison on skewed data:

Right-skewed:
     █
    ██
   ███
  ████
 █████████
─────────────────→
 Mode Median Mean
```

### 1.4 Weighted Mean

$$\bar{x}_w = \frac{\sum_{i=1}^n w_i x_i}{\sum_{i=1}^n w_i}$$

Used in: Sample weights, attention mechanisms

### 1.5 Geometric Mean

$$\bar{x}_g = \left(\prod_{i=1}^n x_i\right)^{1/n} = \exp\left(\frac{1}{n}\sum_{i=1}^n \ln x_i\right)$$

Used for: Growth rates, ratios, multiplicative processes

### 1.6 Harmonic Mean

$$\bar{x}_h = \frac{n}{\sum_{i=1}^n \frac{1}{x_i}}$$

Used for: F1-score (harmonic mean of precision and recall)

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

Why n-1? Bessel's correction for unbiased estimation.

### 2.3 Interquartile Range (IQR)

$$\text{IQR} = Q_3 - Q_1$$

Where Q1 = 25th percentile, Q3 = 75th percentile.

Robust to outliers. Used in box plots.

### 2.4 Mean Absolute Deviation (MAD)

$$\text{MAD} = \frac{1}{n}\sum_{i=1}^n |x_i - \bar{x}|$$

More robust than standard deviation.

### 2.5 Coefficient of Variation

$$\text{CV} = \frac{s}{\bar{x}}$$

Normalized measure of dispersion (unitless).

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
Box Plot:

    ┌─────┬─────┐
────┤     │     ├────○
    └─────┴─────┘
   Q1    Q2    Q3   Outlier
    └──IQR──┘
```

### Outlier Detection (IQR Method)

- Lower fence: $Q_1 - 1.5 \times \text{IQR}$
- Upper fence: $Q_3 + 1.5 \times \text{IQR}$

Points outside fences are potential outliers.

---

## 4. Shape of Distribution

### 4.1 Skewness

$$\text{Skewness} = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^3$$

| Value | Interpretation               |
| ----- | ---------------------------- |
| = 0   | Symmetric                    |
| > 0   | Right-skewed (tail on right) |
| < 0   | Left-skewed (tail on left)   |

### 4.2 Kurtosis

$$\text{Kurtosis} = \frac{1}{n}\sum_{i=1}^n \left(\frac{x_i - \bar{x}}{s}\right)^4$$

**Excess Kurtosis** = Kurtosis - 3

| Value | Interpretation            |
| ----- | ------------------------- |
| = 0   | Normal-like tails         |
| > 0   | Heavy tails (leptokurtic) |
| < 0   | Light tails (platykurtic) |

```
Kurtosis comparison:

Leptokurtic (heavy tails):     Normal:     Platykurtic (light tails):
        █                        ██               ████
        █                       ████             ██████
       ███                     ██████           ████████
      █████                   ████████         ██████████
   ███████████               ██████████       ████████████
```

---

## 5. Correlation and Association

### 5.1 Pearson Correlation

$$r = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^n (x_i - \bar{x})^2}\sqrt{\sum_{i=1}^n (y_i - \bar{y})^2}}$$

Measures linear relationship: $-1 \leq r \leq 1$

### 5.2 Spearman Rank Correlation

$$\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $d_i$ = difference in ranks.

Measures monotonic relationship. Robust to outliers.

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

---

## 7. ML Applications

| Statistic   | ML Application                   |
| ----------- | -------------------------------- |
| Mean        | Batch normalization, imputation  |
| Median      | Robust imputation, MAE           |
| Std         | Feature scaling, regularization  |
| Correlation | Feature selection                |
| Skewness    | Log transform decision           |
| Outliers    | Data cleaning, anomaly detection |

---

## 8. Summary

| Measure  | Formula                            | Robust? |
| -------- | ---------------------------------- | ------- |
| Mean     | $\frac{1}{n}\sum x_i$              | No      |
| Median   | Middle value                       | Yes     |
| Variance | $\frac{1}{n-1}\sum(x_i-\bar{x})^2$ | No      |
| IQR      | $Q_3 - Q_1$                        | Yes     |
| Skewness | 3rd standardized moment            | No      |
| Kurtosis | 4th standardized moment            | No      |

---

## References

1. Wasserman - "All of Statistics"
2. James et al. - "Introduction to Statistical Learning"
