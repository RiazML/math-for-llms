# Introduction to Probability and Random Variables

## Introduction

Probability theory is the mathematical foundation of machine learning. Every ML model involves uncertainty: noisy data, random initialization, stochastic optimization, and probabilistic predictions. Understanding probability enables rigorous reasoning about uncertain outcomes.

## Prerequisites

- Basic set theory
- Calculus (integration, differentiation)
- Series and summations

## Learning Objectives

1. Understand probability axioms and rules
2. Master discrete and continuous random variables
3. Compute expectations, variances, and moments
4. Apply probability to ML problems

---

## 1. Probability Basics

### Sample Space and Events

**Sample space** $\Omega$: Set of all possible outcomes

**Event** $A$: Subset of $\Omega$

```
Example: Rolling a die
  Ω = {1, 2, 3, 4, 5, 6}
  A = "even number" = {2, 4, 6}
  B = "greater than 4" = {5, 6}
```

### Probability Axioms (Kolmogorov)

For probability function $P$:

1. **Non-negativity:** $P(A) \geq 0$
2. **Normalization:** $P(\Omega) = 1$
3. **Additivity:** If $A \cap B = \emptyset$, then $P(A \cup B) = P(A) + P(B)$

### Basic Rules

| Rule       | Formula                                   |
| ---------- | ----------------------------------------- |
| Complement | $P(A^c) = 1 - P(A)$                       |
| Union      | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ |
| Impossible | $P(\emptyset) = 0$                        |

---

## 2. Conditional Probability

### Definition

$$P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

"Probability of A given B"

### Chain Rule (Product Rule)

$$P(A \cap B) = P(A | B) \cdot P(B) = P(B | A) \cdot P(A)$$

For multiple events:
$$P(A_1 \cap A_2 \cap \ldots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1, A_2) \cdots$$

### Law of Total Probability

If $B_1, B_2, \ldots, B_n$ partition $\Omega$:

$$P(A) = \sum_{i=1}^n P(A | B_i) P(B_i)$$

```
       ┌─────────────────────────────┐
       │           Ω                 │
       │   ┌─────┬─────┬─────┐      │
       │   │ B₁  │ B₂  │ B₃  │      │
       │   │ ┌───┼───┐ │     │      │
       │   │ │ A │   │ │     │      │
       │   │ └───┼───┘ │     │      │
       │   └─────┴─────┴─────┘      │
       └─────────────────────────────┘

P(A) = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + P(A|B₃)P(B₃)
```

### Bayes' Theorem

$$P(B | A) = \frac{P(A | B) \cdot P(B)}{P(A)}$$

In ML terminology:
$$\text{Posterior} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Evidence}}$$

---

## 3. Independence

### Definition

Events $A$ and $B$ are independent if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently: $P(A|B) = P(A)$

### Conditional Independence

$A$ and $B$ are conditionally independent given $C$ if:

$$P(A \cap B | C) = P(A|C) \cdot P(B|C)$$

Written as: $A \perp B | C$

---

## 4. Random Variables

### Definition

A **random variable** $X$ is a function that maps outcomes to real numbers:

$$X: \Omega \to \mathbb{R}$$

```
Example: Sum of two dice
  Ω = {(1,1), (1,2), ..., (6,6)}
  X((i,j)) = i + j
  X maps outcomes to {2, 3, ..., 12}
```

### Discrete vs Continuous

| Property     | Discrete                    | Continuous                       |
| ------------ | --------------------------- | -------------------------------- |
| Values       | Countable (finite/infinite) | Uncountable (intervals)          |
| Probability  | PMF: $P(X = x)$             | PDF: $f(x)$, $P(X = x) = 0$      |
| Sum/Integral | $\sum_x P(X=x) = 1$         | $\int f(x)dx = 1$                |
| Cumulative   | $F(x) = P(X \leq x)$        | $F(x) = \int_{-\infty}^x f(t)dt$ |

---

## 5. Discrete Random Variables

### Probability Mass Function (PMF)

$$p(x) = P(X = x)$$

Properties:

- $p(x) \geq 0$
- $\sum_x p(x) = 1$

### Common Discrete Distributions

#### Bernoulli($p$)

$$P(X = 1) = p, \quad P(X = 0) = 1-p$$

- Single binary trial
- Example: Coin flip, spam/not-spam

#### Binomial($n$, $p$)

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- Number of successes in $n$ independent trials
- $E[X] = np$, $\text{Var}(X) = np(1-p)$

#### Categorical($\mathbf{p}$)

$$P(X = k) = p_k, \quad \sum_{k=1}^K p_k = 1$$

- Single draw from $K$ categories
- Generalization of Bernoulli

#### Poisson($\lambda$)

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

- Count of events in fixed interval
- $E[X] = \text{Var}(X) = \lambda$

---

## 6. Continuous Random Variables

### Probability Density Function (PDF)

$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

Properties:

- $f(x) \geq 0$
- $\int_{-\infty}^{\infty} f(x) dx = 1$
- $f(x)$ can be $> 1$!

### Cumulative Distribution Function (CDF)

$$F(x) = P(X \leq x) = \int_{-\infty}^x f(t) dt$$

$$f(x) = \frac{dF(x)}{dx}$$

### Common Continuous Distributions

#### Uniform($a$, $b$)

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

- $E[X] = \frac{a+b}{2}$
- $\text{Var}(X) = \frac{(b-a)^2}{12}$

#### Gaussian/Normal($\mu$, $\sigma^2$)

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- $E[X] = \mu$, $\text{Var}(X) = \sigma^2$
- Central Limit Theorem: sum of many RVs → Normal

#### Exponential($\lambda$)

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

- Time between Poisson events
- $E[X] = 1/\lambda$, memoryless property

---

## 7. Expectation and Variance

### Expected Value (Mean)

**Discrete:**
$$E[X] = \sum_x x \cdot p(x)$$

**Continuous:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

### Properties of Expectation

| Property  | Formula                                          |
| --------- | ------------------------------------------------ |
| Linearity | $E[aX + bY] = aE[X] + bE[Y]$                     |
| Constant  | $E[c] = c$                                       |
| Function  | $E[g(X)] = \sum_x g(x)p(x)$ or $\int g(x)f(x)dx$ |

### Variance

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

### Properties of Variance

| Property        | Formula                                             |
| --------------- | --------------------------------------------------- |
| Non-negative    | $\text{Var}(X) \geq 0$                              |
| Constant        | $\text{Var}(c) = 0$                                 |
| Scaling         | $\text{Var}(aX) = a^2 \text{Var}(X)$                |
| Shift           | $\text{Var}(X + c) = \text{Var}(X)$                 |
| Independent sum | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ |

### Standard Deviation

$$\sigma_X = \sqrt{\text{Var}(X)}$$

---

## 8. Moments and Higher Statistics

### Raw Moments

$$\mu_n = E[X^n]$$

### Central Moments

$$\mu'_n = E[(X - \mu)^n]$$

- $\mu'_1 = 0$
- $\mu'_2 = \text{Var}(X)$

### Skewness (3rd standardized moment)

$$\text{Skewness} = E\left[\left(\frac{X - \mu}{\sigma}\right)^3\right]$$

- Measures asymmetry
- Positive: right tail longer
- Negative: left tail longer

### Kurtosis (4th standardized moment)

$$\text{Kurtosis} = E\left[\left(\frac{X - \mu}{\sigma}\right)^4\right]$$

- Measures tail heaviness
- Normal distribution: kurtosis = 3

---

## 9. ML Applications

### Binary Classification

$$P(\text{class} = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$$

### Likelihood

For data $D = \{x_1, \ldots, x_n\}$:

$$P(D | \theta) = \prod_{i=1}^n p(x_i | \theta)$$

### Maximum Likelihood Estimation

$$\hat{\theta}_{MLE} = \arg\max_\theta P(D | \theta)$$

### Loss as Negative Log-Likelihood

$$\text{Loss} = -\log P(D | \theta)$$

---

## 10. Summary Tables

### Distribution Quick Reference

| Distribution           | PMF/PDF                                                         | Mean                | Variance              |
| ---------------------- | --------------------------------------------------------------- | ------------------- | --------------------- |
| Bernoulli($p$)         | $p^x(1-p)^{1-x}$                                                | $p$                 | $p(1-p)$              |
| Binomial($n,p$)        | $\binom{n}{k}p^k(1-p)^{n-k}$                                    | $np$                | $np(1-p)$             |
| Poisson($\lambda$)     | $\frac{\lambda^k e^{-\lambda}}{k!}$                             | $\lambda$           | $\lambda$             |
| Uniform($a,b$)         | $\frac{1}{b-a}$                                                 | $\frac{a+b}{2}$     | $\frac{(b-a)^2}{12}$  |
| Normal($\mu,\sigma^2$) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$               | $\sigma^2$            |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$                                        | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

### Key Formulas

```
Probability Rules:
├── P(A ∪ B) = P(A) + P(B) - P(A ∩ B)
├── P(A | B) = P(A ∩ B) / P(B)
├── P(A ∩ B) = P(A|B)P(B) = P(B|A)P(A)
└── Bayes: P(B|A) = P(A|B)P(B) / P(A)

Expectation & Variance:
├── E[aX + bY] = aE[X] + bE[Y]
├── Var(X) = E[X²] - E[X]²
├── Var(aX) = a²Var(X)
└── If X ⊥ Y: Var(X+Y) = Var(X) + Var(Y)
```

---

## Exercises

1. Prove $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
2. Compute $E[X]$ and $\text{Var}(X)$ for $X \sim \text{Binomial}(n, p)$
3. Show that for standard normal, $E[X^2] = 1$
4. Derive the MLE for $\lambda$ in Poisson distribution
5. Use Bayes' theorem to update belief about spam given word occurrence

---

## References

1. Bertsekas & Tsitsiklis - "Introduction to Probability"
2. Ross - "A First Course in Probability"
3. Murphy - "Machine Learning: A Probabilistic Perspective"
