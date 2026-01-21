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

## Table of Contents

1. [Summation Notation](#1-summation-notation-sigma-notation)
2. [Summation Properties](#2-summation-properties)
3. [Common Summation Formulas](#3-common-summation-formulas)
4. [Product Notation](#4-product-notation-pi-notation)
5. [Product Properties](#5-product-properties)
6. [Double Summations](#6-double-summations)
7. [AI/ML Domain Connections](#7-aiml-domain-connections)
8. [Real-World Code Examples](#8-real-world-code-examples)
9. [Manipulation Techniques](#9-manipulation-techniques)
10. [Educational Extras](#10-educational-extras)
11. [Common Patterns](#11-common-patterns)
12. [References & Further Reading](#12-references--further-reading)
13. [Summary](#13-summary)

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

## 7. AI/ML Domain Connections

### 1. Einstein Summation Convention

Modern Deep Learning frameworks (PyTorch, TensorFlow, NumPy) use `einsum` to represent complex tensor operations compactly. The rule is: **Repeated indices are summed over.**

- **Matrix Multiplication**: $C_{ij} = \sum_k A_{ik} B_{kj}$
  - `einsum('ik,kj->ij', A, B)`
- **Dot Product**: $s = \sum_i a_i b_i$
  - `einsum('i,i->', a, b)`
- **Batch Matrix Multiplication**: $C_{bij} = \sum_k A_{bik} B_{bkj}$
  - `einsum('bik,bkj->bij', A, B)`

### 2. Gradient Accumulation

When batch sizes are too large for GPU memory, we split them into mini-batches and sum the gradients.

$$\nabla_\theta \mathcal{L}_{total} = \sum_{b=1}^{B} \nabla_\theta \mathcal{L}_{batch}(b)$$

This exploits the **Linearity Property**: The derivative of a sum is the sum of derivatives.

### 3. Attention Mechanisms

The core of Transformer models involves a weighted sum of values, where weights are computed via softmax of dot products.

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

In summation notation for a single query $q$ and key-value pairs $(k_i, v_i)$:

$$Attention(q, K, V) = \sum_{i=1}^{n} \frac{\exp(q \cdot k_i)}{\sum_{j=1}^{n} \exp(q \cdot k_j)} v_i$$

### 4. Expectation & Monte Carlo

In Reinforcement Learning and Variational Inference, we approximate expectations (integrals/sums) using sample averages (Monte Carlo).

$$\mathbb{E}_{x \sim p(x)}[f(x)] = \sum_{x} p(x)f(x) \approx \frac{1}{N} \sum_{i=1}^{N} f(x_i)$$

where $x_i$ are samples drawn from $p(x)$. This is the **Law of Large Numbers**.

### 5. Convolutions as Moving Sums

A 1D convolution is a sum of element-wise products between a kernel $w$ and input $x$ at different offsets.

$$(x * w)[t] = \sum_{k=-K}^{K} x[t-k] w[k]$$

In CNNs, this extends to 2D summations over height, width, and channels.

---

## 8. Real-World Code Examples

### 1. NumPy `einsum`

The most powerful tool for summation in Python.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix Multiplication: C_ij = Σ_k A_ik * B_kj
C = np.einsum('ik,kj->ij', A, B)

# Dot Product: s = Σ_i a_i * b_i
s = np.einsum('i,i->', A[0], B[0])
```

### 2. Vectorization vs Loops

Avoid explicit Python loops for summation!

```python
# SLOW (Python loop)
total = 0
for x in large_array:
    total += x

# FAST (Vectorized C backend)
total = np.sum(large_array)
```

### 3. Log-Sum-Exp Stability

Computing $\log(\sum e^{x_i})$ is prone to overflow if $x_i$ is large.

```python
def log_sum_exp(x):
    # Mathematical identity: log(Σ e^x) = a + log(Σ e^(x-a))
    # Best 'a' is max(x)
    a = np.max(x)
    return a + np.log(np.sum(np.exp(x - a)))
```

### 4. Scatter Add

In Graph Neural Networks (GNNs), we sum messages from neighbors. PyTorch `scatter_add` handles this.

```python
import torch
# src: values to sum, index: destination indices
src = torch.tensor([1, 2, 3, 4, 5, 6])
index = torch.tensor([0, 0, 0, 1, 1, 1])
out = torch.zeros(2, dtype=src.dtype)
out.scatter_add_(0, index, src)
# out[0] = 1+2+3 = 6, out[1] = 4+5+6 = 15
```

### 5. Cumulative Sum

Used for masking in Transformers (causal mask) and computing integral images.

```python
x = np.array([1, 2, 3, 4])
cumsum = np.cumsum(x)  # [1, 3, 6, 10]
# cumsum[i] = Σ_{j=0}^{i} x[j]
```

---

## 9. Manipulation Techniques

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

## 10. Educational Extras

### 1. Common Pitfalls

- **Broadcasting Errors**: Adding `(3,)` vector to `(3,3)` matrix works, but `(3,)` to `(4,3)` fails. Always check shapes!
- **Off-by-One**: $\sum_{i=1}^n$ has $n$ terms. $\sum_{i=0}^n$ has $n+1$ terms.
- **Overflow**: $\prod p_i \to 0$ for probabilities. Use $\sum \log p_i$.

### 2. Interview Questions

1.  **Q: Gradient of a Sum?**
    - A: $\nabla \sum f(x) = \sum \nabla f(x)$ by linearity.
2.  **Q: Explain Log-Sum-Exp?**
    - A: It's a smooth maximum function ($\approx \max(x)$) used for numerical stability in softmax.
3.  **Q: Write Matrix Mult purely in sums?**
    - A: $C_{ij} = \sum_{k} A_{ik} B_{kj}$.

### 3. Cheat Sheet: LaTeX for Sums

| Notation                              | LaTeX Code                            |
| :------------------------------------ | :------------------------------------ |
| $\sum_{i=1}^{n}$                      | `\sum_{i=1}^{n}`                      |
| $\prod_{i=1}^{n}$                     | `\prod_{i=1}^{n}`                     |
| $\underbrace{a + \cdots + a}_{n}$     | `\underbrace{a + \cdots + a}_{n}`     |
| $\sum_{\substack{0<i<n \\ i \neq j}}$ | `\sum_{\substack{0<i<n \\ i \neq j}}` |

---

## 11. Common Patterns

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

## 12. References & Further Reading

### Courses

- **fast.ai**: https://course.fast.ai/
- **Stanford CS229**: https://cs229.stanford.edu/
- **MIT 18.06**: https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/

### Books

- **Concrete Mathematics** (Graham, Knuth, Patashnik)
- **Deep Learning** (Goodfellow et al., free): https://www.deeplearningbook.org/
- **Pattern Recognition and Machine Learning** (Bishop)

### Key Papers / Notes

- **Attention Is All You Need** (Vaswani et al.)
- **The Matrix Calculus You Need For Deep Learning** (Parr, Howard)

---

## 13. Summary

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

Einstein:
─────────
ik,kj->ij  (Matrix Multiplication)
i,i->      (Dot Product)
```

### Key Properties

| Summation         | Product           | Einstein |
| ----------------- | ----------------- | -------- |
| Σ(a+b) = Σa + Σb  | Π(a·b) = Πa · Πb  | Linear   |
| Σ(ca) = c·Σa      | Π(aᶜ) = (Πa)ᶜ     | Compact  |
| log(Πa) = Σlog(a) | Πa = exp(Σlog(a)) | Implicit |

### ML Applications Summary

| Concept        | Formula                         |
| -------------- | ------------------------------- | -------- |
| Mean           | $\bar{x} = \frac{1}{n}\sum x_i$ |
| MSE            | $\frac{1}{n}\sum(y-\hat{y})^2$  |
| Cross-Entropy  | $-\sum y_i \log \hat{y}_i$      |
| Softmax        | $\frac{e^{z_i}}{\sum e^{z_j}}$  |
| Attention      | $\sum \text{soft}(q \cdot k) v$ |
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
