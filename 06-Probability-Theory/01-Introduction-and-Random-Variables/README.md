# Introduction to Probability and Random Variables

> **Navigation**: [02-Common-Distributions](../02-Common-Distributions/) | [03-Joint-Distributions](../03-Joint-Distributions/) | [04-Expectation-and-Moments](../04-Expectation-and-Moments/)

## Introduction

Probability theory is the mathematical foundation of machine learning. Every ML model involves uncertainty: noisy data, random initialization, stochastic optimization, and probabilistic predictions. Understanding probability enables rigorous reasoning about uncertain outcomes.

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PROBABILITY IN MACHINE LEARNING                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Data Uncertainty          Model Uncertainty        Predictions     │
│  ───────────────          ─────────────────        ───────────      │
│  • Noisy measurements     • Random initialization  • P(spam|email)  │
│  • Missing values         • Dropout regularization • Confidence     │
│  • Sampling bias          • Bayesian posteriors    • Calibration    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

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

| Term | Definition | Example (Die Roll) |
|------|-----------|-------------------|
| **Sample space** Ω | Set of all possible outcomes | {1, 2, 3, 4, 5, 6} |
| **Event** A | Subset of Ω | "even" = {2, 4, 6} |
| **Elementary outcome** ω | Single element of Ω | Rolling a 3 |

```
Sample Space Visualization:

       ┌─────────────────────────────────┐
       │            Ω (all outcomes)     │
       │   ┌───────────────────────┐     │
       │   │         A             │     │
       │   │    ┌─────────┐        │     │
       │   │    │  A ∩ B  │   B    │     │
       │   │    └─────────┘        │     │
       │   └───────────────────────┘     │
       │                                 │
       └─────────────────────────────────┘

Events can overlap (A ∩ B), be disjoint, or one contain another.
```

### Probability Axioms (Kolmogorov)

For probability function $P$:

1. **Non-negativity:** $P(A) \geq 0$ for all events A
2. **Normalization:** $P(\Omega) = 1$
3. **Additivity:** If $A \cap B = \emptyset$, then $P(A \cup B) = P(A) + P(B)$

> **💡 Intuition**: These axioms ensure probabilities behave like "portions" of certainty - they're non-negative, sum to 1 for all possibilities, and add up for mutually exclusive events.

### Basic Rules

| Rule | Formula | Interpretation |
|------|---------|---------------|
| Complement | $P(A^c) = 1 - P(A)$ | "Not A" is everything else |
| Union | $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Subtract overlap to avoid double-counting |
| Impossible | $P(\emptyset) = 0$ | Nothing never happens |
| Monotonicity | If $A \subseteq B$, then $P(A) \leq P(B)$ | More outcomes = higher probability |

---

## 2. Conditional Probability

### Definition

$$P(A | B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

> **💡 Intuition**: "Zoom in" on the world where B happened. What fraction of that world is A?

```
Conditional Probability Visualization:

Before conditioning:              After conditioning on B:
┌─────────────────────┐          ┌─────────────────────┐
│         Ω          │          │/////////B///////////│
│    ┌─────┐         │          │////┌─────┐//////////│
│    │  A  │   B     │    →     │////│A∩B │//////////│
│    │  ∩──┼───┐     │          │////│    │//////////│
│    └──┼──┘   │     │          │////└─────┘//////////│
│       └──────┘     │          │/////////////////////│
└─────────────────────┘          └─────────────────────┘

P(A|B) = shaded A∩B / all shaded B
```

### Chain Rule (Product Rule)

$$P(A \cap B) = P(A | B) \cdot P(B) = P(B | A) \cdot P(A)$$

For multiple events:
$$P(A_1 \cap A_2 \cap \ldots \cap A_n) = P(A_1) \cdot P(A_2|A_1) \cdot P(A_3|A_1, A_2) \cdots$$

### Law of Total Probability

If $B_1, B_2, \ldots, B_n$ partition $\Omega$ (mutually exclusive and exhaustive):

$$P(A) = \sum_{i=1}^n P(A | B_i) P(B_i)$$

```
Law of Total Probability:

           ┌─────────────────────────────────┐
           │               Ω                 │
           │   ┌─────┬─────┬─────┬─────┐     │
           │   │ B₁  │ B₂  │ B₃  │ B₄  │     │
           │   │╔═══╗│════╗│     │     │     │
           │   │║ A ║│══A═║│     │     │     │
           │   │╚═══╝│════╝│     │     │     │
           │   └─────┴─────┴─────┴─────┘     │
           └─────────────────────────────────┘

P(A) = P(A|B₁)P(B₁) + P(A|B₂)P(B₂) + P(A|B₃)P(B₃) + P(A|B₄)P(B₄)

"Split A into pieces based on which Bᵢ it falls in"
```

### Bayes' Theorem

$$P(B | A) = \frac{P(A | B) \cdot P(B)}{P(A)} = \frac{P(A | B) \cdot P(B)}{\sum_j P(A | B_j) P(B_j)}$$

**ML Terminology:**
$$\underbrace{P(\theta | \text{data})}_{\text{Posterior}} = \frac{\overbrace{P(\text{data} | \theta)}^{\text{Likelihood}} \times \overbrace{P(\theta)}^{\text{Prior}}}{\underbrace{P(\text{data})}_{\text{Evidence}}}$$

> **🔑 Key ML Application**: Bayes' theorem is the foundation of:
> - Naive Bayes classifiers
> - Bayesian neural networks
> - Probabilistic graphical models
> - MAP estimation

---

## 3. Independence

### Definition

Events $A$ and $B$ are **independent** if:

$$P(A \cap B) = P(A) \cdot P(B)$$

Equivalently: $P(A|B) = P(A)$ — knowing B doesn't change probability of A

```
Independence vs Dependence:

Independent:                    Dependent:
┌────────────────┐              ┌────────────────┐
│ B doesn't      │              │ B changes      │
│ affect A       │              │ probability    │
│                │              │ of A           │
│  P(A|B) = P(A) │              │  P(A|B) ≠ P(A) │
│                │              │                │
│  Examples:     │              │  Examples:     │
│  • Coin flips  │              │  • Cards w/o   │
│  • Dice rolls  │              │    replacement │
│  • Weather in  │              │  • Test score  │
│    diff cities │              │    & studying  │
└────────────────┘              └────────────────┘
```

### Conditional Independence

$A$ and $B$ are **conditionally independent** given $C$ if:

$$P(A \cap B | C) = P(A|C) \cdot P(B|C)$$

Written as: $A \perp\!\!\!\perp B \mid C$

> **⚠️ Warning**: Independence and conditional independence are different!
> - $A \perp\!\!\!\perp B$ does NOT imply $A \perp\!\!\!\perp B \mid C$
> - $A \perp\!\!\!\perp B \mid C$ does NOT imply $A \perp\!\!\!\perp B$

**ML Example**: In Naive Bayes, we assume features are conditionally independent given the class label, even if they're not marginally independent.

---

## 4. Random Variables

### Definition

A **random variable** $X$ is a function that maps outcomes to real numbers:

$$X: \Omega \to \mathbb{R}$$

```
Random Variable as a Function:

Sample Space Ω                Real Numbers ℝ
┌─────────────────┐           ┌─────────────┐
│  (1,1)  ●───────┼──────────→│  2          │
│  (1,2)  ●───────┼──────────→│  3          │
│  (2,1)  ●───────┼──────────→│  3          │
│  (3,6)  ●───────┼──────────→│  9          │
│  (6,6)  ●───────┼──────────→│  12         │
│   ...           │           │             │
└─────────────────┘           └─────────────┘
    Two dice                  X = sum of dice
```

### Discrete vs Continuous

| Property | Discrete | Continuous |
|----------|----------|------------|
| **Values** | Countable (finite/infinite) | Uncountable (intervals) |
| **Probability** | PMF: $P(X = x) > 0$ | PDF: $f(x)$, $P(X = x) = 0$ |
| **Normalization** | $\sum_x P(X=x) = 1$ | $\int f(x)dx = 1$ |
| **CDF** | $F(x) = P(X \leq x)$ | $F(x) = \int_{-\infty}^x f(t)dt$ |
| **Examples** | Counts, categories | Measurements, time |

---

## 5. Discrete Random Variables

### Probability Mass Function (PMF)

$$p(x) = P(X = x)$$

Properties:
- $p(x) \geq 0$ for all x
- $\sum_x p(x) = 1$

### Common Discrete Distributions

#### Bernoulli($p$) — Single Binary Trial

$$P(X = 1) = p, \quad P(X = 0) = 1-p$$

- **Use**: Coin flip, spam detection, click/no-click
- **Mean**: $p$, **Variance**: $p(1-p)$

#### Binomial($n$, $p$) — Number of Successes

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- **Use**: Number of heads in n flips, number of defects
- **Mean**: $np$, **Variance**: $np(1-p)$

```
Binomial(n=10, p=0.3):

P(X=k)
0.28 │      ██
0.24 │     ████
0.20 │    ██████
0.16 │   ████████
0.12 │  ██████████
0.08 │ ████████████
0.04 │██████████████
     └─────────────────
       0 1 2 3 4 5 6 7 8 9 10
                k
```

#### Categorical($\mathbf{p}$) — One of K Categories

$$P(X = k) = p_k, \quad \sum_{k=1}^K p_k = 1$$

- **Use**: Multi-class classification, word prediction
- Generalization of Bernoulli to K classes

#### Poisson($\lambda$) — Count of Rare Events

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

- **Use**: Website visits, network packets, mutation count
- **Mean** = **Variance** = $\lambda$

---

## 6. Continuous Random Variables

### Probability Density Function (PDF)

$$P(a \leq X \leq b) = \int_a^b f(x) dx$$

Properties:
- $f(x) \geq 0$
- $\int_{-\infty}^{\infty} f(x) dx = 1$
- $f(x)$ **can be $> 1$** (it's density, not probability!)

> **💡 Intuition**: For continuous RVs, $P(X = x) = 0$ for any exact value. Probability only makes sense for intervals.

### Cumulative Distribution Function (CDF)

$$F(x) = P(X \leq x) = \int_{-\infty}^x f(t) dt$$

$$f(x) = \frac{dF(x)}{dx}$$

```
PDF vs CDF:

PDF f(x)                        CDF F(x)
    │     ╱╲                        │         ┌────
    │    ╱  ╲                       │       ╱╱
    │   ╱    ╲                      │     ╱╱
    │  ╱      ╲                     │   ╱╱
    │ ╱        ╲                    │ ╱╱
    │╱          ╲                   │╱
    └────────────→               ──┴────────────→
          x               0←      x        →1

• PDF: Height = probability density
• CDF: Height = P(X ≤ x), always between 0 and 1
```

### Common Continuous Distributions

#### Uniform($a$, $b$)

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

- **Mean**: $\frac{a+b}{2}$, **Variance**: $\frac{(b-a)^2}{12}$

#### Normal/Gaussian($\mu$, $\sigma^2$)

$$f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- **Mean**: $\mu$, **Variance**: $\sigma^2$
- Central Limit Theorem → sum of many RVs approaches Normal

```
68-95-99.7 Rule for Normal Distribution:

                    ╱╲
                   ╱  ╲
                  ╱    ╲
                 ╱      ╲
                ╱        ╲
               ╱          ╲
              ╱  ┌──68%──┐  ╲
             ╱   │       │   ╲
            ╱  ┌─┴───────┴─┐  ╲
           ╱   │   95%     │   ╲
          ╱  ┌─┴───────────┴─┐  ╲
         ╱   │    99.7%      │   ╲
        ─────┴───────────────┴─────
           μ-3σ  μ-2σ  μ-σ   μ   μ+σ  μ+2σ  μ+3σ
```

#### Exponential($\lambda$)

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

- **Mean**: $1/\lambda$, **Variance**: $1/\lambda^2$
- Time between Poisson events
- **Memoryless**: $P(X > s+t | X > s) = P(X > t)$

---

## 7. Expectation and Variance

### Expected Value (Mean)

**Discrete:**
$$E[X] = \sum_x x \cdot p(x)$$

**Continuous:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x) dx$$

> **💡 Intuition**: E[X] is the "center of mass" of the distribution

### Properties of Expectation

| Property | Formula | Note |
|----------|---------|------|
| Linearity | $E[aX + bY] = aE[X] + bE[Y]$ | **Always works!** |
| Constant | $E[c] = c$ | |
| Function | $E[g(X)] = \sum_x g(x)p(x)$ | |

> **⚠️ Warning**: $E[g(X)] \neq g(E[X])$ in general! (Jensen's inequality)

### Variance

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**Standard Deviation:** $\sigma_X = \sqrt{\text{Var}(X)}$

### Properties of Variance

| Property | Formula |
|----------|---------|
| Non-negative | $\text{Var}(X) \geq 0$ |
| Constant | $\text{Var}(c) = 0$ |
| Scaling | $\text{Var}(aX) = a^2 \text{Var}(X)$ |
| Shift | $\text{Var}(X + c) = \text{Var}(X)$ |
| Independent sum | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ |

---

## 8. Moments and Higher Statistics

### Raw and Central Moments

$$\mu'_n = E[X^n] \quad \text{(raw)}$$
$$\mu_n = E[(X - \mu)^n] \quad \text{(central)}$$

### Skewness — Asymmetry Measure

$$\text{Skewness} = \frac{E[(X - \mu)^3]}{\sigma^3}$$

```
Skewness:

Left-skewed (< 0)      Symmetric (= 0)      Right-skewed (> 0)
       ╱█                    █                        █╲
      ╱██                   ███                      ██╲
     ╱███                  █████                    ███╲
    ╱████                 ███████                  ████╲
   ╱█████                █████████                █████╲
  ────────              ───────────              ────────
    mode > mean         mode = mean             mode < mean
```

### Kurtosis — Tail Heaviness

$$\text{Kurtosis} = \frac{E[(X - \mu)^4]}{\sigma^4}$$

- Normal distribution has kurtosis = 3
- **Excess Kurtosis** = Kurtosis - 3
- Positive: heavy tails (leptokurtic)
- Negative: light tails (platykurtic)

---

## 9. ML Applications

### Binary Classification

$$P(\text{class} = 1 | \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b)$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

### Likelihood and MLE

For data $D = \{x_1, \ldots, x_n\}$:

$$P(D | \theta) = \prod_{i=1}^n p(x_i | \theta)$$

$$\hat{\theta}_{MLE} = \arg\max_\theta P(D | \theta) = \arg\max_\theta \sum_{i=1}^n \log p(x_i | \theta)$$

### Loss as Negative Log-Likelihood

$$\text{Loss} = -\log P(D | \theta) = -\sum_{i=1}^n \log p(x_i | \theta)$$

> **🔑 Connection**: Cross-entropy loss = NLL for classification!

---

## 10. Summary Tables

### Distribution Quick Reference

| Distribution | PMF/PDF | Mean | Variance |
|--------------|---------|------|----------|
| Bernoulli($p$) | $p^x(1-p)^{1-x}$ | $p$ | $p(1-p)$ |
| Binomial($n,p$) | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| Poisson($\lambda$) | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |
| Uniform($a,b$) | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| Normal($\mu,\sigma^2$) | $\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| Exponential($\lambda$) | $\lambda e^{-\lambda x}$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

### Key Formulas Cheat Sheet

```
┌─────────────────────────────────────────────────────────────┐
│                    PROBABILITY RULES                        │
├─────────────────────────────────────────────────────────────┤
│  P(A ∪ B) = P(A) + P(B) - P(A ∩ B)                         │
│  P(A | B) = P(A ∩ B) / P(B)                                │
│  P(A ∩ B) = P(A|B)P(B) = P(B|A)P(A)                        │
│  Bayes: P(B|A) = P(A|B)P(B) / P(A)                         │
├─────────────────────────────────────────────────────────────┤
│              EXPECTATION & VARIANCE                         │
├─────────────────────────────────────────────────────────────┤
│  E[aX + bY] = aE[X] + bE[Y]  (always!)                     │
│  Var(X) = E[X²] - E[X]²                                    │
│  Var(aX) = a²Var(X)                                        │
│  If X ⊥ Y: Var(X+Y) = Var(X) + Var(Y)                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Exercises

1. Prove $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ using axioms
2. Compute $E[X]$ and $\text{Var}(X)$ for $X \sim \text{Binomial}(n, p)$
3. Show that for standard normal $Z$, $E[Z^2] = 1$
4. Derive the MLE for $\lambda$ in Poisson distribution
5. Use Bayes' theorem: given 1% spam rate and 90% word detection, find P(spam|word)

---

## References

1. Bertsekas & Tsitsiklis - "Introduction to Probability"
2. Ross - "A First Course in Probability"
3. Murphy - "Machine Learning: A Probabilistic Perspective"

---

> **Next**: [02-Common-Distributions](../02-Common-Distributions/) — Deep dive into specific distributions
