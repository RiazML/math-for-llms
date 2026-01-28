# Joint Distributions and Independence

## Introduction

Most real-world problems involve multiple random variables simultaneously. Joint distributions describe the probabilistic behavior of two or more random variables together. Understanding joint distributions is crucial for multivariate analysis and machine learning.

## Prerequisites

- Single-variable probability
- Basic calculus (integration)
- Matrix operations

## Learning Objectives

1. Work with joint, marginal, and conditional distributions
2. Understand independence and conditional independence
3. Compute covariance and correlation
4. Apply Bayes' theorem to multiple variables

---

## 1. Joint Probability Distributions

### 1.1 Discrete Joint Distribution

For discrete random variables X and Y:

**Joint PMF:** $P(X=x, Y=y) = p_{XY}(x, y)$

Properties:

- $p_{XY}(x, y) \geq 0$
- $\sum_x \sum_y p_{XY}(x, y) = 1$

```
Example joint PMF table:

           Y=0    Y=1    Y=2
        ┌──────┬──────┬──────┐
  X=0   │ 0.10 │ 0.15 │ 0.05 │  → 0.30
        ├──────┼──────┼──────┤
  X=1   │ 0.20 │ 0.30 │ 0.20 │  → 0.70
        └──────┴──────┴──────┘
           0.30   0.45   0.25    = 1.00
           ↓      ↓      ↓
        (Marginals of Y)
```

### 1.2 Continuous Joint Distribution

For continuous random variables X and Y:

**Joint PDF:** $f_{XY}(x, y)$

Properties:

- $f_{XY}(x, y) \geq 0$
- $\int \int f_{XY}(x, y) \, dx \, dy = 1$

**Probability over region:**

$$P((X, Y) \in A) = \iint_A f_{XY}(x, y) \, dx \, dy$$

---

## 2. Marginal Distributions

### Obtaining Marginals from Joint

**Discrete case:**
$$p_X(x) = \sum_y p_{XY}(x, y)$$
$$p_Y(y) = \sum_x p_{XY}(x, y)$$

**Continuous case:**
$$f_X(x) = \int_{-\infty}^{\infty} f_{XY}(x, y) \, dy$$
$$f_Y(y) = \int_{-\infty}^{\infty} f_{XY}(x, y) \, dx$$

```
Visual: Marginal from Joint

Joint PDF f(x,y)       Marginal f_X(x)
     ┌─────────┐       ┌─────────┐
     │ ╭───╮   │       │         │
     │╱     ╲  │  →    │   ╱╲    │
     │   •   │ │ sum   │  ╱  ╲   │
     │╲     ╱  │ over  │ ╱    ╲  │
     │ ╰───╯   │   y   └─────────┘
     └─────────┘
         x,y               x
```

---

## 3. Conditional Distributions

### Definition

$$P(Y = y | X = x) = \frac{P(X = x, Y = y)}{P(X = x)}$$

**Discrete:** $p_{Y|X}(y|x) = \frac{p_{XY}(x, y)}{p_X(x)}$

**Continuous:** $f_{Y|X}(y|x) = \frac{f_{XY}(x, y)}{f_X(x)}$

### Relationship: Joint = Marginal × Conditional

$$p_{XY}(x, y) = p_X(x) \cdot p_{Y|X}(y|x)$$

or equivalently:

$$p_{XY}(x, y) = p_Y(y) \cdot p_{X|Y}(x|y)$$

### Chain Rule (Product Rule)

For multiple variables:

$$P(X_1, X_2, \ldots, X_n) = P(X_1) \cdot P(X_2|X_1) \cdot P(X_3|X_1, X_2) \cdots P(X_n|X_1, \ldots, X_{n-1})$$

---

## 4. Independence

### Definition

X and Y are independent if and only if:

$$P(X = x, Y = y) = P(X = x) \cdot P(Y = y) \quad \forall x, y$$

or equivalently: $f_{XY}(x, y) = f_X(x) \cdot f_Y(y)$

### Tests for Independence

1. Joint = product of marginals
2. $P(Y|X) = P(Y)$ (knowing X doesn't help predict Y)
3. Covariance = 0 (necessary but not sufficient!)

```
Independence visualization:

Independent:            Not Independent:
  Y                       Y
  │  □ □ □ □               │    □
  │  □ □ □ □               │  □ □ □
  │  □ □ □ □               │□ □ □ □ □
  └──────────→ X           └──────────→ X

(uniform rows)         (pattern varies)
```

### Conditional Independence

X and Y are conditionally independent given Z:

$$X \perp Y \mid Z$$

if $P(X, Y | Z) = P(X|Z) \cdot P(Y|Z)$

**Important:** Independence and conditional independence are different!

- X ⊥ Y does NOT imply X ⊥ Y | Z
- X ⊥ Y | Z does NOT imply X ⊥ Y

---

## 5. Covariance and Correlation

### Covariance

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

Properties:

- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX + b, cY + d) = ac \cdot \text{Cov}(X, Y)$
- If X ⊥ Y, then Cov(X, Y) = 0

**Warning:** Cov(X, Y) = 0 does NOT imply independence!

### Correlation

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

Properties:

- $-1 \leq \rho \leq 1$
- $\rho = \pm 1$ iff Y = aX + b (perfect linear relationship)
- $\rho = 0$ means no linear relationship

```
Correlation examples:

ρ ≈ 1          ρ ≈ 0.5        ρ ≈ 0          ρ ≈ -1
  y              y              y              y
  │    •        │     •        │  • • •       │•
  │   •         │    •  •      │ • • • •      │ •
  │  •          │  •   •       │• • • • •     │  •
  │ •           │ •  •         │ • • • •      │   •
  │•            │•    •        │  • • •       │    •
  └─────→x      └─────→x       └─────→x       └─────→x
```

### Covariance Matrix

For random vector $\mathbf{X} = (X_1, \ldots, X_n)^T$:

$$\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

$$\boldsymbol{\Sigma}_{ij} = \text{Cov}(X_i, X_j)$$

Properties:

- Symmetric: $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$
- Positive semi-definite: $\mathbf{a}^T \boldsymbol{\Sigma} \mathbf{a} \geq 0$
- Diagonal elements are variances

---

## 6. Bayes' Theorem for Random Variables

### Discrete Case

$$P(Y = y | X = x) = \frac{P(X = x | Y = y) \cdot P(Y = y)}{\sum_{y'} P(X = x | Y = y') \cdot P(Y = y')}$$

### Continuous Case

$$f_{Y|X}(y|x) = \frac{f_{X|Y}(x|y) \cdot f_Y(y)}{\int f_{X|Y}(x|y') \cdot f_Y(y') \, dy'}$$

### Bayesian Interpretation

$$\underbrace{P(\theta | \text{data})}_{\text{posterior}} = \frac{\overbrace{P(\text{data} | \theta)}^{\text{likelihood}} \cdot \overbrace{P(\theta)}^{\text{prior}}}{\underbrace{P(\text{data})}_{\text{evidence}}}$$

---

## 7. Multivariate Normal Distribution

The most important multivariate distribution!

### Definition

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1}(\mathbf{x} - \boldsymbol{\mu})\right)$$

### Key Properties

**Marginals are Normal:**
If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, then each $X_i \sim \mathcal{N}(\mu_i, \Sigma_{ii})$

**Conditionals are Normal:**
Partition: $\mathbf{X} = \begin{pmatrix} \mathbf{X}_1 \\ \mathbf{X}_2 \end{pmatrix}$, $\boldsymbol{\mu} = \begin{pmatrix} \boldsymbol{\mu}_1 \\ \boldsymbol{\mu}_2 \end{pmatrix}$, $\boldsymbol{\Sigma} = \begin{pmatrix} \boldsymbol{\Sigma}_{11} & \boldsymbol{\Sigma}_{12} \\ \boldsymbol{\Sigma}_{21} & \boldsymbol{\Sigma}_{22} \end{pmatrix}$

$$\mathbf{X}_1 | \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2}, \boldsymbol{\Sigma}_{1|2})$$

where:

- $\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$
- $\boldsymbol{\Sigma}_{1|2} = \boldsymbol{\Sigma}_{11} - \boldsymbol{\Sigma}_{12}\boldsymbol{\Sigma}_{22}^{-1}\boldsymbol{\Sigma}_{21}$

**Linear Transformations:**
If $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ and $\mathbf{Y} = \mathbf{A}\mathbf{X} + \mathbf{b}$, then:
$$\mathbf{Y} \sim \mathcal{N}(\mathbf{A}\boldsymbol{\mu} + \mathbf{b}, \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T)$$

**Uncorrelated = Independent:**
For multivariate normal, zero covariance implies independence!

---

## 8. Functions of Random Variables

### Transformation Method

If $Y = g(X)$ and g is monotonic with inverse $g^{-1}$:

$$f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d}{dy}g^{-1}(y)\right|$$

### Sum of Random Variables

**Convolution:**
$$f_{X+Y}(z) = \int f_X(x) f_Y(z-x) \, dx$$

**Sum of Normals:**
If $X \sim N(\mu_X, \sigma_X^2)$ and $Y \sim N(\mu_Y, \sigma_Y^2)$ independent:
$$X + Y \sim N(\mu_X + \mu_Y, \sigma_X^2 + \sigma_Y^2)$$

---

## 9. ML Applications

### Naive Bayes Classifier

Assumes conditional independence:
$$P(Y | X_1, \ldots, X_n) \propto P(Y) \prod_{i=1}^n P(X_i | Y)$$

### Gaussian Mixture Models

$$P(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

### Hidden Markov Models

Chain of conditional independence:
$$P(X_{t+1} | X_1, \ldots, X_t) = P(X_{t+1} | X_t)$$

### Graphical Models

Joint distribution factorizes according to graph structure:
$$P(X_1, \ldots, X_n) = \prod_i P(X_i | \text{Parents}(X_i))$$

---

## 10. Summary

| Concept       | Definition                                  |
| ------------- | ------------------------------------------- | ----------------- |
| Joint PMF/PDF | $P(X=x, Y=y)$ or $f(x,y)$                   |
| Marginal      | Sum/integrate out other variable            |
| Conditional   | $P(Y                                        | X) = P(X,Y)/P(X)$ |
| Independence  | $P(X,Y) = P(X)P(Y)$                         |
| Covariance    | $E[XY] - E[X]E[Y]$                          |
| Correlation   | $\rho = \text{Cov}(X,Y)/(\sigma_X\sigma_Y)$ |

---

## Exercises

1. Given a joint PMF table, find marginals, conditionals, and check independence
2. Prove that uncorrelated normal RVs are independent
3. Derive the conditional distribution of bivariate normal
4. Show that Cov(X,Y)=0 doesn't imply independence with counterexample
5. Apply Bayes' theorem for a classification problem

---

## References

1. Casella & Berger - "Statistical Inference"
2. Bishop - "Pattern Recognition and Machine Learning"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
