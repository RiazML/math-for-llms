# Entropy

> **Navigation**: [← 09-Hyperparameter-Optimization](../../08-Optimization/09-Hyperparameter-Optimization/) | [Information Theory](../) | [02-KL-Divergence →](../02-KL-Divergence/)

**Files in this section:**
- [theory.ipynb](theory.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Introduction

Entropy is the fundamental concept in information theory, quantifying the uncertainty or "information content" of a random variable. In machine learning, entropy appears in loss functions, decision trees, generative models, and information-theoretic analysis of learning algorithms.

## Prerequisites

- Probability theory basics
- Logarithms and their properties
- Expected values

## Learning Objectives

1. Understand Shannon entropy and its properties
2. Apply entropy to discrete and continuous distributions
3. Connect entropy to coding theory
4. Use entropy in ML applications (decision trees, maximum entropy models)

---

## 1. Shannon Entropy

### 1.1 Definition

For a discrete random variable $X$ with probability mass function $p(x)$:

$$H(X) = -\sum_{x \in \mathcal{X}} p(x) \log p(x) = \mathbb{E}[-\log p(X)]$$

Convention: $0 \log 0 = 0$ (by continuity)

### 1.2 Interpretation

- **Uncertainty:** Average surprise when observing $X$
- **Information:** Bits needed to describe $X$ on average
- **Randomness:** How unpredictable $X$ is

```
Low Entropy (predictable):     High Entropy (unpredictable):

p(X):                          p(X):
█                              ██ ██ ██ ██
█                              ██ ██ ██ ██
█ ░ ░ ░                        ██ ██ ██ ██
─────────                      ─────────────
A B C D                        A  B  C  D

H(X) ≈ 0 bits                  H(X) = 2 bits
```

### 1.3 Units

| Log base | Unit   | Common usage       |
| -------- | ------ | ------------------ |
| 2        | bits   | Information theory |
| $e$      | nats   | Machine learning   |
| 10       | digits | Less common        |

Conversion: $H_{\text{nats}} = H_{\text{bits}} \cdot \ln 2$

---

## 2. Properties of Entropy

### 2.1 Non-negativity

$$H(X) \geq 0$$

Equality when $X$ is deterministic (one outcome has probability 1).

### 2.2 Maximum Entropy

For a discrete variable with $n$ outcomes:

$$H(X) \leq \log n$$

Maximum achieved when $X$ is uniform: $p(x) = 1/n$ for all $x$.

**Proof:**
$$H(X) = -\sum_x p(x) \log p(x) \leq -\sum_x p(x) \log(1/n) = \log n$$

### 2.3 Concavity

Entropy is concave in the probability distribution:

$$H(\lambda p + (1-\lambda) q) \geq \lambda H(p) + (1-\lambda) H(q)$$

This means mixing distributions increases entropy.

### 2.4 Chain Rule

$$H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)$$

Joint entropy equals marginal plus conditional.

---

## 3. Binary Entropy

### 3.1 Definition

For a Bernoulli random variable with $P(X=1) = p$:

$$H_b(p) = -p \log p - (1-p) \log(1-p)$$

### 3.2 Properties

```
Binary Entropy Function:

H(p) ↑
  1 │      ●
    │    ╱   ╲
    │   ╱     ╲
    │  ╱       ╲
  0 ├─●─────────●─
    0   0.5     1    p

Maximum at p = 0.5
H(0.5) = 1 bit
```

| $p$  | $H_b(p)$ (bits) |
| ---- | --------------- |
| 0    | 0               |
| 0.1  | 0.469           |
| 0.25 | 0.811           |
| 0.5  | 1.0             |
| 0.75 | 0.811           |
| 1    | 0               |

---

## 4. Conditional Entropy

### 4.1 Definition

$$H(Y|X) = \sum_x p(x) H(Y|X=x) = -\sum_{x,y} p(x,y) \log p(y|x)$$

Average uncertainty in $Y$ given knowledge of $X$.

### 4.2 Properties

$$H(Y|X) \leq H(Y)$$

Conditioning reduces entropy (on average). Equality iff $X$ and $Y$ are independent.

### 4.3 Chain Rule Expansion

$$H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^{n} H(X_i | X_1, \ldots, X_{i-1})$$

---

## 5. Joint Entropy

### 5.1 Definition

$$H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

### 5.2 Bounds

$$\max(H(X), H(Y)) \leq H(X,Y) \leq H(X) + H(Y)$$

- Lower bound: joint has at least as much uncertainty as marginals
- Upper bound: equality when $X, Y$ independent

### 5.3 Venn Diagram View

```
         ┌─────────────────────────┐
         │       H(X,Y)            │
         │  ┌─────────┬─────────┐  │
         │  │         │         │  │
         │  │  H(X|Y) │ I(X;Y)  │  │
         │  │         │         │  │
         │  │    ─────┼─────    │  │
         │  │         │         │  │
         │  │ I(X;Y)  │  H(Y|X) │  │
         │  │         │         │  │
         │  └─────────┴─────────┘  │
         └─────────────────────────┘

H(X) = H(X|Y) + I(X;Y)
H(Y) = H(Y|X) + I(X;Y)
H(X,Y) = H(X) + H(Y) - I(X;Y)
```

---

## 6. Differential Entropy

### 6.1 Definition for Continuous Variables

$$h(X) = -\int_{-\infty}^{\infty} f(x) \log f(x) \, dx$$

where $f(x)$ is the probability density function.

### 6.2 Key Differences from Discrete Entropy

| Property             | Discrete $H(X)$    | Continuous $h(X)$ |
| -------------------- | ------------------ | ----------------- |
| Non-negative         | Always $\geq 0$    | Can be negative   |
| Coordinate invariant | Yes                | No                |
| Maximum              | $\log n$ (uniform) | No upper bound    |

### 6.3 Common Distributions

| Distribution                          | Differential Entropy               |
| ------------------------------------- | ---------------------------------- |
| Uniform$[a,b]$                        | $\log(b-a)$                        |
| Gaussian $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{2}\log(2\pi e \sigma^2)$ |
| Exponential$(\lambda)$                | $1 - \log \lambda$                 |
| Laplace$(0, b)$                       | $1 + \log(2b)$                     |

### 6.4 Gaussian Has Maximum Entropy

Among all distributions with fixed variance $\sigma^2$, the Gaussian has maximum differential entropy.

$$h(X) \leq \frac{1}{2}\log(2\pi e \sigma^2)$$

with equality iff $X \sim \mathcal{N}(\mu, \sigma^2)$.

---

## 7. Entropy in Machine Learning

### 7.1 Decision Trees

**Information Gain:** Reduction in entropy from splitting on feature $A$:

$$IG(S, A) = H(S) - \sum_{v \in \text{values}(A)} \frac{|S_v|}{|S|} H(S_v)$$

Choose splits that maximize information gain.

```
Before split:           After split on A:
   H(S)                    H(S|A)

███████                 ██████  (A=0)
░░░░░░░                 ░░

                        █  (A=1)
                        ░░░░░░░

High entropy    →     Lower weighted entropy
```

### 7.2 Maximum Entropy Models

Find distribution $p$ maximizing $H(p)$ subject to constraints:

$$\max_p H(p) \quad \text{s.t.} \quad \mathbb{E}_p[f_i(X)] = \alpha_i$$

Solution has exponential form:
$$p(x) = \frac{1}{Z} \exp\left(\sum_i \lambda_i f_i(x)\right)$$

This is the theoretical foundation for logistic regression and CRFs.

### 7.3 Entropy Regularization

In reinforcement learning, add entropy bonus to encourage exploration:

$$\mathcal{L} = \mathbb{E}[R] + \beta H(\pi)$$

where $\pi$ is the policy and $\beta$ controls exploration.

### 7.4 Variational Autoencoders

The ELBO includes an entropy-like term:

$$\mathcal{L} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

---

## 8. Entropy Estimation

### 8.1 Plug-in Estimator

Given samples, estimate probabilities then compute entropy:

$$\hat{H}(X) = -\sum_x \hat{p}(x) \log \hat{p}(x)$$

where $\hat{p}(x) = \frac{\text{count}(x)}{n}$

**Problem:** Biased (underestimates entropy for small samples)

### 8.2 Miller-Madow Correction

$$\hat{H}_{MM} = \hat{H} + \frac{m-1}{2n}$$

where $m$ is the number of bins with non-zero counts.

### 8.3 For Continuous Variables

- **Histogram-based:** Discretize and apply discrete entropy
- **KDE-based:** Estimate density, compute integral
- **k-NN based:** Use nearest neighbor distances

---

## 9. Rényi Entropy

### 9.1 Generalization

$$H_\alpha(X) = \frac{1}{1-\alpha} \log \sum_x p(x)^\alpha$$

for $\alpha > 0, \alpha \neq 1$.

### 9.2 Special Cases

| $\alpha$            | Entropy               | Name              |
| ------------------- | --------------------- | ----------------- | --- | ----------- |
| $\alpha \to 0$      | $\log                 | \text{support}    | $   | Max entropy |
| $\alpha \to 1$      | Shannon entropy       | Shannon           |
| $\alpha = 2$        | $-\log \sum_x p(x)^2$ | Collision entropy |
| $\alpha \to \infty$ | $-\log \max_x p(x)$   | Min entropy       |

### 9.3 Application

- $H_2$ used in randomness extraction
- $H_\infty$ important for cryptography (worst-case security)

---

## 10. Summary

| Concept         | Formula                                   | Key Property                      |
| --------------- | ----------------------------------------- | --------------------------------- | ---------- | ---- | ------------- |
| Shannon entropy | $H(X) = -\sum_x p(x) \log p(x)$           | Average surprise                  |
| Binary entropy  | $H_b(p) = -p\log p - (1-p)\log(1-p)$      | Max at $p=0.5$                    |
| Conditional     | $H(Y                                      | X) = -\sum\_{x,y} p(x,y) \log p(y | x)$        | $H(Y | X) \leq H(Y)$ |
| Joint           | $H(X,Y) = H(X) + H(Y                      | X)$                               | Chain rule |
| Differential    | $h(X) = -\int f(x) \log f(x) dx$          | Can be negative                   |
| Max entropy     | Uniform (discrete), Gaussian (continuous) | Least assumptions                 |

**Key insight:** Entropy measures uncertainty. In ML, we often minimize entropy of predictions (make them confident) while maximizing entropy of distributions (avoid assumptions).

---

## Exercises

1. **Binary Entropy**: Calculate $H_b(p)$ for $p \in \{0, 0.1, 0.5, 0.9, 1\}$. Plot the function and verify the maximum is at $p=0.5$.

2. **Distribution Comparison**: Compute the entropy of: (a) fair coin, (b) biased coin with $p=0.9$, (c) fair 6-sided die. Which has highest uncertainty?

3. **Chain Rule**: For joint distribution $P(X,Y)$ with $X,Y \in \{0,1\}$ and $P(0,0)=0.4, P(0,1)=0.1, P(1,0)=0.2, P(1,1)=0.3$, verify that $H(X,Y) = H(X) + H(Y|X)$.

4. **Maximum Entropy**: Show that among all distributions on $\{1,...,n\}$, the uniform distribution maximizes entropy. What is this maximum value?

5. **Differential Entropy**: Compute the differential entropy of a Gaussian $\mathcal{N}(\mu, \sigma^2)$. How does it change with $\sigma$?

---

## References

1. Cover & Thomas - "Elements of Information Theory"
2. MacKay - "Information Theory, Inference, and Learning Algorithms"
3. Bishop - "Pattern Recognition and Machine Learning" (Chapter 1.6)
