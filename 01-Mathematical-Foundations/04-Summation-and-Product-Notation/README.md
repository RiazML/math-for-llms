# Summation and Product Notation

## Introduction

Summation (Σ) and product (Π) notation are compact ways to express repeated addition and multiplication. These notations are ubiquitous in ML—from loss functions to probability calculations to matrix operations. Mastering this notation is essential for reading and understanding ML literature.

## Prerequisites

- Basic arithmetic operations
- Functions and mappings
- Basic algebra

## Learning Objectives

1. Read and write summation notation
2. Read and write product notation
3. Apply standard manipulation rules
4. Recognize common sums and products
5. Use double and nested summations

---

## 1. Summation Notation (Sigma Notation)

### Basic Definition

$$\sum_{i=m}^{n} a_i = a_m + a_{m+1} + a_{m+2} + \cdots + a_n$$

```
Anatomy of Summation:
                    n         ← Upper limit (end)
                   ___
                   \
                    \    aᵢ   ← General term
                    /
                   /___
                   i=m        ← Index variable = lower limit (start)

Example:
  4
 ___
 \
  \   i² = 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
  /
 /___
 i=1
```

### Components

| Component       | Description           | Example      |
| --------------- | --------------------- | ------------ |
| Index (i)       | Variable that changes | i, j, k, n   |
| Lower limit (m) | Starting value        | 1, 0, -2     |
| Upper limit (n) | Ending value          | 10, n, ∞     |
| Summand (aᵢ)    | Expression to sum     | i², 2i+1, xᵢ |

### Examples

$$\sum_{i=1}^{5} i = 1 + 2 + 3 + 4 + 5 = 15$$

$$\sum_{i=0}^{3} 2^i = 2^0 + 2^1 + 2^2 + 2^3 = 1 + 2 + 4 + 8 = 15$$

$$\sum_{k=2}^{4} (k^2 - 1) = (4-1) + (9-1) + (16-1) = 3 + 8 + 15 = 26$$

---

## 2. Summation Properties

### Linearity

$$\sum_{i=m}^{n} (a_i + b_i) = \sum_{i=m}^{n} a_i + \sum_{i=m}^{n} b_i$$

$$\sum_{i=m}^{n} c \cdot a_i = c \cdot \sum_{i=m}^{n} a_i \quad \text{(c is constant)}$$

### Constant Summation

$$\sum_{i=1}^{n} c = nc \quad \text{(sum of n copies of c)}$$

### Splitting

$$\sum_{i=m}^{n} a_i = \sum_{i=m}^{k} a_i + \sum_{i=k+1}^{n} a_i$$

### Index Shifting

$$\sum_{i=m}^{n} a_i = \sum_{j=m+k}^{n+k} a_{j-k}$$

Example:
$$\sum_{i=1}^{n} i = \sum_{j=0}^{n-1} (j+1)$$

### Reversing Order

$$\sum_{i=1}^{n} a_i = \sum_{i=1}^{n} a_{n+1-i}$$

---

## 3. Common Summation Formulas

### Arithmetic Series

$$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$$

$$\sum_{i=1}^{n} (2i-1) = n^2 \quad \text{(sum of first n odd numbers)}$$

### Power Sums

$$\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6}$$

$$\sum_{i=1}^{n} i^3 = \left(\frac{n(n+1)}{2}\right)^2$$

### Geometric Series

$$\sum_{i=0}^{n} r^i = \frac{1-r^{n+1}}{1-r} \quad (r \neq 1)$$

$$\sum_{i=0}^{\infty} r^i = \frac{1}{1-r} \quad (|r| < 1)$$

### Exponential Series

$$\sum_{i=0}^{\infty} \frac{x^i}{i!} = e^x$$

### Quick Reference Table

| Sum                       | Formula                  |
| ------------------------- | ------------------------ | --- | --- |
| $\sum_{i=1}^{n} 1$        | $n$                      |
| $\sum_{i=1}^{n} i$        | $\frac{n(n+1)}{2}$       |
| $\sum_{i=1}^{n} i^2$      | $\frac{n(n+1)(2n+1)}{6}$ |
| $\sum_{i=0}^{n} r^i$      | $\frac{1-r^{n+1}}{1-r}$  |
| $\sum_{i=0}^{\infty} r^i$ | $\frac{1}{1-r}$ for $    | r   | <1$ |

---

## 4. Product Notation (Pi Notation)

### Basic Definition

$$\prod_{i=m}^{n} a_i = a_m \cdot a_{m+1} \cdot a_{m+2} \cdots a_n$$

```
Anatomy of Product:
                    n         ← Upper limit
                   ___
                   | |
                   | |   aᵢ   ← General term
                   | |
                   i=m        ← Index variable = lower limit

Example:
  4
 ___
 | |
 | |   i = 1 × 2 × 3 × 4 = 24 = 4!
 | |
 i=1
```

### Examples

$$\prod_{i=1}^{4} i = 1 \cdot 2 \cdot 3 \cdot 4 = 24$$

$$\prod_{i=1}^{n} i = n! \quad \text{(factorial)}$$

$$\prod_{i=1}^{3} 2^i = 2^1 \cdot 2^2 \cdot 2^3 = 2 \cdot 4 \cdot 8 = 64$$

$$\prod_{i=1}^{n} x = x^n$$

---

## 5. Product Properties

### Multiplicativity

$$\prod_{i=m}^{n} (a_i \cdot b_i) = \left(\prod_{i=m}^{n} a_i\right) \cdot \left(\prod_{i=m}^{n} b_i\right)$$

### Power Rule

$$\prod_{i=m}^{n} a_i^c = \left(\prod_{i=m}^{n} a_i\right)^c$$

### Log Conversion

$$\log\left(\prod_{i=m}^{n} a_i\right) = \sum_{i=m}^{n} \log(a_i)$$

This is extremely important in ML for numerical stability!

### Splitting

$$\prod_{i=m}^{n} a_i = \left(\prod_{i=m}^{k} a_i\right) \cdot \left(\prod_{i=k+1}^{n} a_i\right)$$

---

## 6. Double Summations

### Notation

$$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$$

This means: for each i from 1 to m, sum over all j from 1 to n.

```
Double Sum Expansion:

  m     n                 n                   n                       n
 ___   ___              ___                 ___                     ___
 \     \                \                   \                       \
  \     \   aᵢⱼ    =     \   a₁ⱼ    +       \   a₂ⱼ    + ... +      \   aₘⱼ
  /     /                /                   /                       /
 /___  /___             /___                /___                    /___
 i=1   j=1              j=1                 j=1                     j=1

= (a₁₁ + a₁₂ + ... + a₁ₙ) + (a₂₁ + a₂₂ + ... + a₂ₙ) + ... + (aₘ₁ + aₘ₂ + ... + aₘₙ)
```

### Interchanging Order

If limits are independent:
$$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij} = \sum_{j=1}^{n} \sum_{i=1}^{m} a_{ij}$$

### Matrix Sum

For matrix A with elements aᵢⱼ:
$$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij} = \text{sum of all elements in A}$$

### Triangular Sums

When j depends on i:
$$\sum_{i=1}^{n} \sum_{j=i}^{n} a_{ij}$$

This sums over the upper triangular portion.

---

## 7. Applications in ML/AI

### 1. Mean (Average)

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

### 2. Variance

$$\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

### 3. Mean Squared Error

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 4. Cross-Entropy Loss

$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

### 5. Softmax Function

$$\text{softmax}(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^{K} \exp(z_j)}$$

### 6. Dot Product

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

### 7. Matrix Multiplication

$$(AB)_{ij} = \sum_{k=1}^{p} a_{ik} b_{kj}$$

### 8. Likelihood Function

$$L(\theta) = \prod_{i=1}^{n} P(x_i | \theta)$$

### 9. Log-Likelihood

$$\log L(\theta) = \sum_{i=1}^{n} \log P(x_i | \theta)$$

### 10. Neural Network Forward Pass

$$a_j^{(l)} = \sigma\left(\sum_{i} w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)}\right)$$

### 11. Gradient Descent Update

$$\theta_{t+1} = \theta_t - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} \nabla_\theta \mathcal{L}(x_i, y_i; \theta)$$

---

## 8. Manipulation Techniques

### Technique 1: Factor Out Constants

$$\sum_{i=1}^{n} 3x_i = 3 \sum_{i=1}^{n} x_i$$

### Technique 2: Split Sums

$$\sum_{i=1}^{n} (x_i + y_i) = \sum_{i=1}^{n} x_i + \sum_{i=1}^{n} y_i$$

### Technique 3: Expand Squares

$$\sum_{i=1}^{n} (x_i - \mu)^2 = \sum_{i=1}^{n} x_i^2 - 2\mu \sum_{i=1}^{n} x_i + n\mu^2$$

### Technique 4: Change of Index

Replace i with n-i to sum in reverse:
$$\sum_{i=0}^{n} f(i) = \sum_{i=0}^{n} f(n-i)$$

### Technique 5: Product to Sum via Logarithm

$$\prod_{i=1}^{n} a_i = \exp\left(\sum_{i=1}^{n} \ln(a_i)\right)$$

---

## 9. Common Patterns

### Pattern 1: Arithmetic Progression

$$\sum_{i=0}^{n-1} (a + id) = na + d\frac{n(n-1)}{2}$$

### Pattern 2: Telescoping Sum

$$\sum_{i=1}^{n} (a_i - a_{i-1}) = a_n - a_0$$

### Pattern 3: Binomial Expansion

$$\sum_{k=0}^{n} \binom{n}{k} x^k y^{n-k} = (x + y)^n$$

### Pattern 4: Probability Sum

$$\sum_{i} P(X = x_i) = 1$$

### Pattern 5: Expected Value

$$E[X] = \sum_{i} x_i P(X = x_i)$$

---

## 10. Summary

### Quick Notation Reference

```
Summation: Σ
─────────────
  n
 ___
 \
  \   f(i)  means f(m) + f(m+1) + ... + f(n)
  /
 /___
 i=m

Product: Π
──────────
  n
 ___
 | |
 | |   f(i)  means f(m) × f(m+1) × ... × f(n)
 | |
 i=m
```

### Key Properties

| Summation         | Product           |
| ----------------- | ----------------- |
| Σ(a+b) = Σa + Σb  | Π(a·b) = Πa · Πb  |
| Σ(ca) = c·Σa      | Π(aᶜ) = (Πa)ᶜ     |
| log(Πa) = Σlog(a) | Πa = exp(Σlog(a)) |

### ML Applications Summary

| Concept        | Formula                         |
| -------------- | ------------------------------- | -------- |
| Mean           | $\bar{x} = \frac{1}{n}\sum x_i$ |
| MSE            | $\frac{1}{n}\sum(y-\hat{y})^2$  |
| Cross-Entropy  | $-\sum y_i \log \hat{y}_i$      |
| Softmax        | $\frac{e^{z_i}}{\sum e^{z_j}}$  |
| Likelihood     | $\prod P(x_i                    | \theta)$ |
| Log-Likelihood | $\sum \log P(x_i                | \theta)$ |

---

## Exercises

1. Expand and compute: $\sum_{i=1}^{4} (2i - 1)$
2. Use the formula to compute: $\sum_{i=1}^{100} i$
3. Compute: $\prod_{i=1}^{5} i$
4. Simplify: $\sum_{i=1}^{n} (3x_i + 2)$
5. Write in summation notation: $1 + 4 + 9 + 16 + 25$
6. Expand the double sum: $\sum_{i=1}^{2} \sum_{j=1}^{3} ij$

---

## References

1. Graham, Knuth, Patashnik - "Concrete Mathematics"
2. Stewart - "Calculus: Early Transcendentals"
3. Bishop - "Pattern Recognition and Machine Learning"
