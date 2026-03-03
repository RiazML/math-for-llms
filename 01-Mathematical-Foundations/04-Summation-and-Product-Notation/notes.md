[← Functions and Mappings](../03-Functions-and-Mappings/notes.md) | [Home](../../README.md) | [Einstein Summation →](../05-Einstein-Summation-and-Index-Notation/notes.md)

---

# Summation and Product Notation

## Introduction

Summation (Σ) and product (Π) notation are the **mathematical shorthand for loops**. Just as programmers use `for` loops to repeat operations, mathematicians use Σ and Π to express repeated addition and multiplication compactly.

```
Python Loop                          Mathematical Notation
─────────────────────────────────    ─────────────────────────────
total = 0                              n
for i in range(1, n+1):               ___
    total += a[i]                     \
                                       \   aᵢ = a₁ + a₂ + ... + aₙ
                     ══════▶          /
                                      /___
                                      i=1
```

### Why This Matters for ML

**Every ML algorithm uses Σ and Π notation:**

| ML Concept    | Formula                                        | What It Means                       |
| ------------- | ---------------------------------------------- | ----------------------------------- |
| Mean          | $\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$      | Sum all values, divide by count     |
| MSE Loss      | $\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ | Sum of squared errors               |
| Cross-Entropy | $-\sum_{i=1}^{C} y_i \log(\hat{y}_i)$          | Sum over classes                    |
| Softmax       | $\frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$       | Normalize by sum of exponentials    |
| Likelihood    | $\prod_{i=1}^{n} P(x_i \mid \theta)$           | Product of individual probabilities |
| Dot Product   | $\sum_{i=1}^{n} a_i b_i$                       | Element-wise multiply then sum      |

---

## Prerequisites

- Basic arithmetic operations
- Functions and mappings
- Basic algebra

## Learning Objectives

By the end of this section, you will be able to:

1. ✅ Read and write Σ (summation) notation fluently
2. ✅ Read and write Π (product) notation fluently
3. ✅ Apply manipulation rules (splitting, factoring, index shifting)
4. ✅ Convert products to sums using logarithms (crucial for numerical stability)
5. ✅ Recognize common patterns in ML formulas

---

## Table of Contents

1. [Summation Notation (Σ)](#1-summation-notation-σ)
2. [Product Notation (Π)](#2-product-notation-π)
3. [Properties and Manipulation Rules](#3-properties-and-manipulation-rules)
4. [The Log-Sum Trick](#4-the-log-sum-trick-critical-for-ml)
5. [Double Summations](#5-double-summations)
6. [Common Formulas Cheat Sheet](#6-common-formulas-cheat-sheet)
7. [ML Applications Deep Dive](#7-ml-applications-deep-dive)
8. [Code Implementation](#8-code-implementation)
9. [Common Pitfalls](#9-common-pitfalls)
10. [Summary](#10-summary)

---

## 1. Summation Notation (Σ)

### Anatomy of a Sum

$$\sum_{i=1}^{n} a_i = a_1 + a_2 + a_3 + \cdots + a_n$$

```
                    n         ← Upper limit (where to stop)
                   ___
                   \
                    \    aᵢ   ← Summand (what to add each time)
                    /
                   /___
                   i=1        ← Index = Lower limit (where to start)
```

| Component          | Description              | Example Values  |
| ------------------ | ------------------------ | --------------- |
| **Index variable** | The counter that changes | i, j, k, n, t   |
| **Lower limit**    | Starting value of index  | 0, 1, -5        |
| **Upper limit**    | Ending value of index    | 10, n, ∞        |
| **Summand**        | Expression being summed  | i, i², xᵢ, f(i) |

### Step-by-Step Examples

**Example 1: Sum of first 5 integers**

$$\sum_{i=1}^{5} i = ?$$

Step through each value of i:

```
i = 1 → 1
i = 2 → 2
i = 3 → 3
i = 4 → 4
i = 5 → 5
        ───
        15
```

**Example 2: Sum of squares**

$$\sum_{i=1}^{4} i^2 = ?$$

```
i = 1 → 1² = 1
i = 2 → 2² = 4
i = 3 → 3² = 9
i = 4 → 4² = 16
           ────
           30
```

**Example 3: Powers of 2**

$$\sum_{i=0}^{3} 2^i = ?$$

```
i = 0 → 2⁰ = 1
i = 1 → 2¹ = 2
i = 2 → 2² = 4
i = 3 → 2³ = 8
          ────
          15
```

### Different Starting Points

The lower limit doesn't have to be 1:

$$\sum_{i=0}^{3} i = 0 + 1 + 2 + 3 = 6$$

$$\sum_{i=2}^{5} i = 2 + 3 + 4 + 5 = 14$$

$$\sum_{i=-2}^{2} i = -2 + (-1) + 0 + 1 + 2 = 0$$

### Counting Terms

**Important**: The number of terms in $\sum_{i=m}^{n}$ is $(n - m + 1)$

| Sum               | Number of Terms       |
| ----------------- | --------------------- |
| $\sum_{i=1}^{10}$ | 10 - 1 + 1 = 10 terms |
| $\sum_{i=0}^{5}$  | 5 - 0 + 1 = 6 terms   |
| $\sum_{i=3}^{7}$  | 7 - 3 + 1 = 5 terms   |

---

## 2. Product Notation (Π)

### Anatomy of a Product

$$\prod_{i=1}^{n} a_i = a_1 \times a_2 \times a_3 \times \cdots \times a_n$$

```
                    n         ← Upper limit
                   ___
                   | |
                   | |   aᵢ   ← Term being multiplied
                   | |
                   i=1        ← Index = Lower limit
```

### Step-by-Step Examples

**Example 1: Factorial**

$$\prod_{i=1}^{5} i = ?$$

```
i = 1 → 1
i = 2 → 2
i = 3 → 3
i = 4 → 4
i = 5 → 5
        ─────────────
1 × 2 × 3 × 4 × 5 = 120 = 5!
```

This is exactly the **factorial**: $n! = \prod_{i=1}^{n} i$

**Example 2: Powers**

$$\prod_{i=1}^{4} 2 = ?$$

```
2 × 2 × 2 × 2 = 16 = 2⁴
```

General pattern: $\prod_{i=1}^{n} c = c^n$ (constant repeated n times)

**Example 3: Joint Probability**

For independent events:

$$P(A \text{ and } B \text{ and } C) = P(A) \times P(B) \times P(C) = \prod_{i=1}^{3} P_i$$

If $P_1 = 0.9$, $P_2 = 0.8$, $P_3 = 0.7$:

$$\prod_{i=1}^{3} P_i = 0.9 \times 0.8 \times 0.7 = 0.504$$

---

## 3. Properties and Manipulation Rules

### Summation Properties

#### 1. Linearity (Most Important!)

**Constants come out:**
$$\sum_{i=1}^{n} c \cdot a_i = c \cdot \sum_{i=1}^{n} a_i$$

Example: $\sum_{i=1}^{3} 5i = 5(1 + 2 + 3) = 5 \times 6 = 30$

**Sums split:**
$$\sum_{i=1}^{n} (a_i + b_i) = \sum_{i=1}^{n} a_i + \sum_{i=1}^{n} b_i$$

Example: $\sum_{i=1}^{3} (i + i^2) = (1+2+3) + (1+4+9) = 6 + 14 = 20$

#### 2. Constant Summation

$$\sum_{i=1}^{n} c = n \cdot c$$

Example: $\sum_{i=1}^{100} 5 = 100 \times 5 = 500$

#### 3. Splitting Range

$$\sum_{i=1}^{n} a_i = \sum_{i=1}^{k} a_i + \sum_{i=k+1}^{n} a_i$$

Example: $\sum_{i=1}^{10} i = \sum_{i=1}^{5} i + \sum_{i=6}^{10} i = 15 + 40 = 55$

#### 4. Index Shifting

$$\sum_{i=1}^{n} a_i = \sum_{j=0}^{n-1} a_{j+1}$$

This is like changing loop variable: `for i in range(1, n+1)` vs `for j in range(0, n)`

### Product Properties

#### 1. Multiplicativity

$$\prod_{i=1}^{n} (a_i \cdot b_i) = \left(\prod_{i=1}^{n} a_i\right) \cdot \left(\prod_{i=1}^{n} b_i\right)$$

#### 2. Power Rule

$$\prod_{i=1}^{n} a_i^c = \left(\prod_{i=1}^{n} a_i\right)^c$$

#### 3. Product ↔ Sum Conversion (Critical!)

$$\log\left(\prod_{i=1}^{n} a_i\right) = \sum_{i=1}^{n} \log(a_i)$$

This is the **log-sum trick** - essential for numerical stability in ML!

---

## 4. The Log-Sum Trick (Critical for ML!)

### The Problem: Underflow

When multiplying many small probabilities, the result becomes too small for computers:

```python
probs = [0.01] * 100  # 100 probabilities of 0.01 each
product = 1.0
for p in probs:
    product *= p
print(product)  # 0.0 (underflow!)
```

The true answer is $0.01^{100} = 10^{-200}$, but computers can't represent this.

### The Solution: Log-Sum

Convert products to sums using logarithms:

$$\log\left(\prod_{i=1}^{n} p_i\right) = \sum_{i=1}^{n} \log(p_i)$$

```python
import numpy as np
probs = [0.01] * 100
log_sum = sum(np.log(p) for p in probs)
print(log_sum)  # -460.5... (computable!)
```

### Why This Works

| Operation | Before Log   | After Log         |
| --------- | ------------ | ----------------- |
| Multiply  | $a \times b$ | $\log a + \log b$ |
| Divide    | $a \div b$   | $\log a - \log b$ |
| Power     | $a^n$        | $n \cdot \log a$  |

Products become sums, which are numerically stable!

### ML Applications

| Concept           | Problematic Form                 | Stable Form                       |
| ----------------- | -------------------------------- | --------------------------------- |
| Likelihood        | $L = \prod_i P(x_i)$             | $\ell = \sum_i \log P(x_i)$       |
| Softmax           | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | Use log-sum-exp trick             |
| Probability chain | $P(x_1, x_2, ..., x_n)$          | $\log P(x_1) + \log P(x_2) + ...$ |

---

## 5. Double Summations

### What Are They?

Double sums are **nested loops** - for each value of the outer index, we do a complete inner sum.

$$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$$

```python
# Equivalent Python:
total = 0
for i in range(1, m+1):
    for j in range(1, n+1):
        total += a[i][j]
```

### Visual Expansion

For a 2×3 matrix:

$$\sum_{i=1}^{2} \sum_{j=1}^{3} a_{ij}$$

```
Matrix A:
┌─────────────────┐
│ a₁₁  a₁₂  a₁₃  │  ← Row 1 (i=1)
│ a₂₁  a₂₂  a₂₃  │  ← Row 2 (i=2)
└─────────────────┘

Expansion:
= (a₁₁ + a₁₂ + a₁₃) + (a₂₁ + a₂₂ + a₂₃)
   └──── i=1 ────┘   └──── i=2 ────┘
```

### Order Can Be Swapped

When limits are independent, you can swap the order:

$$\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij} = \sum_{j=1}^{n} \sum_{i=1}^{m} a_{ij}$$

This is like swapping nested loops - the total is the same.

### Row Sums vs Column Sums

**Row sums** (fix i, sum over j):
$$\text{row}_i = \sum_{j=1}^{n} a_{ij}$$

**Column sums** (fix j, sum over i):
$$\text{col}_j = \sum_{i=1}^{m} a_{ij}$$

**Total**:
$$\sum_i \text{row}_i = \sum_j \text{col}_j = \sum_i \sum_j a_{ij}$$

---

## 6. Common Formulas Cheat Sheet

### Arithmetic Sums

| Formula              | Result                            | Proof Hint    |
| -------------------- | --------------------------------- | ------------- |
| $\sum_{i=1}^{n} 1$   | $n$                               | n ones        |
| $\sum_{i=1}^{n} i$   | $\frac{n(n+1)}{2}$                | Gauss's trick |
| $\sum_{i=1}^{n} i^2$ | $\frac{n(n+1)(2n+1)}{6}$          | Induction     |
| $\sum_{i=1}^{n} i^3$ | $\left(\frac{n(n+1)}{2}\right)^2$ | = (Σi)²       |

**Gauss's Trick Visualized:**

```
S = 1 + 2 + 3 + ... + n
S = n + (n-1) + ... + 1
─────────────────────────
2S = (n+1) + (n+1) + ... + (n+1)  [n times]
2S = n(n+1)
S = n(n+1)/2
```

### Geometric Series

$$\sum_{i=0}^{n} r^i = \frac{1 - r^{n+1}}{1 - r} \quad (r \neq 1)$$

$$\sum_{i=0}^{\infty} r^i = \frac{1}{1 - r} \quad (|r| < 1)$$

**Example: Discount factor in RL**

With γ = 0.9:
$$\sum_{t=0}^{\infty} \gamma^t = \frac{1}{1-0.9} = 10$$

This means future rewards are worth at most 10× the immediate reward.

### Special Sums

| Sum                                  | Result | ML Use      |
| ------------------------------------ | ------ | ----------- |
| $\sum_{i=1}^{n} (2i-1)$              | $n^2$  | Odd numbers |
| $\sum_{i=0}^{\infty} \frac{x^i}{i!}$ | $e^x$  | Exponential |
| $\sum_{k=0}^{n} \binom{n}{k}$        | $2^n$  | Binomial    |

---

## 7. ML Applications Deep Dive

### 1. Loss Functions

**Mean Squared Error (MSE):**
$$\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Cross-Entropy Loss:**
$$\mathcal{L} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

**Batch Loss:**
$$\mathcal{L}_{batch} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}_i$$

### 2. Softmax Function

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

The denominator is a sum that normalizes all values to sum to 1.

### 3. Attention Mechanism

The core of Transformers:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

For a single query, this is:
$$\text{output} = \sum_{i=1}^{n} \alpha_i v_i$$

where $\alpha_i = \text{softmax}(q \cdot k_i)$ are attention weights.

### 4. Expected Value

$$\mathbb{E}[X] = \sum_{i} x_i \cdot P(X = x_i)$$

The expected value is a **weighted sum** of outcomes.

### 5. Dot Product (Foundation of Neural Networks)

$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i$$

Every neuron computes: $\text{output} = f\left(\sum_i w_i x_i + b\right)$

### 6. Einstein Summation Convention

Modern notation for tensor operations - **repeated indices are summed**:

| Operation       | Math                   | einsum        |
| --------------- | ---------------------- | ------------- |
| Dot product     | $\sum_i a_i b_i$       | `'i,i->'`     |
| Matrix multiply | $\sum_k A_{ik} B_{kj}$ | `'ik,kj->ij'` |
| Outer product   | $a_i b_j$              | `'i,j->ij'`   |
| Trace           | $\sum_i A_{ii}$        | `'ii->'`      |

---

## 8. Code Implementation

### Basic Sums in Python

```python
import numpy as np

# Σᵢ₌₁ⁿ i
n = 100
loop_sum = sum(range(1, n+1))          # Python loop
numpy_sum = np.sum(np.arange(1, n+1))  # NumPy (faster)
formula = n * (n + 1) // 2             # Closed form (fastest)

print(f"All equal: {loop_sum == numpy_sum == formula}")  # True
```

### Products and Log-Sum

```python
# Likelihood calculation (stable version)
probs = np.array([0.9, 0.8, 0.7, 0.6, 0.5])

# Unstable: direct product
likelihood = np.prod(probs)

# Stable: log-sum
log_likelihood = np.sum(np.log(probs))
likelihood_recovered = np.exp(log_likelihood)

print(f"Direct: {likelihood:.6f}")
print(f"Log-sum: {likelihood_recovered:.6f}")
```

### Loss Functions

```python
def mse_loss(y_true, y_pred):
    """MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²"""
    n = len(y_true)
    return np.sum((y_true - y_pred)**2) / n

def cross_entropy_loss(y_true, y_pred):
    """CE = -Σᵢ yᵢ log(ŷᵢ)"""
    return -np.sum(y_true * np.log(y_pred + 1e-10))

def softmax(z):
    """softmax(zᵢ) = exp(zᵢ) / Σⱼ exp(zⱼ)"""
    exp_z = np.exp(z - np.max(z))  # Subtract max for stability
    return exp_z / np.sum(exp_z)
```

### Einstein Summation

```python
import numpy as np

A = np.random.randn(3, 4)
B = np.random.randn(4, 5)

# Matrix multiplication: C_ij = Σ_k A_ik B_kj
C = np.einsum('ik,kj->ij', A, B)
C_check = A @ B  # Same result

# Batch matrix multiplication
batch_A = np.random.randn(10, 3, 4)  # 10 matrices
batch_B = np.random.randn(10, 4, 5)
batch_C = np.einsum('bik,bkj->bij', batch_A, batch_B)
```

---

## 9. Common Pitfalls

### 1. Off-by-One Errors

| Expression         | Number of Terms |
| ------------------ | --------------- |
| $\sum_{i=1}^{n}$   | **n** terms     |
| $\sum_{i=0}^{n}$   | **n+1** terms   |
| $\sum_{i=0}^{n-1}$ | **n** terms     |

```python
# Common mistake:
sum(range(1, n))    # Only n-1 terms! (1 to n-1)
sum(range(1, n+1))  # Correct: n terms (1 to n)
```

### 2. Product Underflow

```python
# DON'T: Direct product of many small numbers
prob = np.prod([0.1] * 50)  # Returns 0.0 (underflow)

# DO: Use log-sum
log_prob = np.sum(np.log([0.1] * 50))  # -115.13 (correct)
```

### 3. Empty Sums and Products

$$\sum_{i=1}^{0} a_i = 0 \quad \text{(empty sum)}$$
$$\prod_{i=1}^{0} a_i = 1 \quad \text{(empty product)}$$

```python
np.sum([])   # 0.0
np.prod([])  # 1.0
```

### 4. Broadcasting Confusion

```python
# Be careful with dimensions when summing
A = np.array([[1, 2, 3],
              [4, 5, 6]])

np.sum(A)         # 21 (all elements)
np.sum(A, axis=0) # [5, 7, 9] (column sums)
np.sum(A, axis=1) # [6, 15] (row sums)
```

---

## 10. Summary

### Quick Reference Card

```
SUMMATION (Σ)                    PRODUCT (Π)
═════════════                    ═══════════
  n                                n
 ___                              ___
 \                                | |
  \   aᵢ = a₁+a₂+...+aₙ           | |  aᵢ = a₁×a₂×...×aₙ
  /                               | |
 /___                             i=1
 i=1

Key Properties:                  Key Properties:
• Σ(a+b) = Σa + Σb               • Π(a·b) = Πa · Πb
• Σ(ca) = c·Σa                   • log(Πa) = Σlog(a)
• Σc = nc                        • Πc = cⁿ
```

### Essential Formulas

| What             | Formula                  |
| ---------------- | ------------------------ |
| Sum of 1 to n    | $\frac{n(n+1)}{2}$       |
| Geometric series | $\frac{1-r^{n+1}}{1-r}$  |
| Factorial        | $n! = \prod_{i=1}^{n} i$ |
| Dot product      | $\sum_i a_i b_i$         |
| Mean             | $\frac{1}{n}\sum_i x_i$  |

### The Golden Rule

> **When multiplying many small numbers, convert to log-sum:**
> $$\log\left(\prod_i a_i\right) = \sum_i \log(a_i)$$

---

## Companion Notebooks

| Notebook                           | Description                                                               |
| ---------------------------------- | ------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive examples: summations, products, log-sum trick, loss functions |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions                                          |

---

## What's Next?

After mastering summation and product notation, proceed to:
→ [Einstein Summation and Index Notation](../05-Einstein-Summation-and-Index-Notation/notes.md) - Compact tensor notation used in modern ML frameworks

---

## References

1. Graham, Knuth, Patashnik - "Concrete Mathematics" (the definitive reference)
2. Goodfellow, Bengio, Courville - "Deep Learning" (free online)
3. Bishop - "Pattern Recognition and Machine Learning"

---

---

[← Functions and Mappings](../03-Functions-and-Mappings/notes.md) | [Home](../../README.md) | [Einstein Summation →](../05-Einstein-Summation-and-Index-Notation/notes.md)
