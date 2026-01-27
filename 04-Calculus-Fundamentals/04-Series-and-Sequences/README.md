# Series and Sequences

## Introduction

Sequences and series are fundamental to understanding convergence, approximations, and many algorithms in machine learning. Taylor series allow us to approximate complex functions, power series form the basis for many analytical solutions, and understanding convergence is crucial for iterative optimization algorithms.

## Prerequisites

- Limits and continuity
- Derivatives
- Basic integration

## Learning Objectives

1. Understand sequences and their convergence
2. Work with infinite series and tests for convergence
3. Master Taylor and Maclaurin series
4. Apply series to ML function approximations

---

## 1. Sequences

### Definition

A **sequence** is an ordered list of numbers:
$$a_1, a_2, a_3, \ldots, a_n, \ldots$$

Written as $\{a_n\}$ or $(a_n)_{n=1}^{\infty}$

### Convergence

A sequence $\{a_n\}$ **converges** to $L$ if:
$$\lim_{n \to \infty} a_n = L$$

For any $\epsilon > 0$, there exists $N$ such that $|a_n - L| < \epsilon$ for all $n > N$.

### Examples

| Sequence            | Limit | Converges?      |
| ------------------- | ----- | --------------- |
| $a_n = 1/n$         | 0     | Yes             |
| $a_n = n$           | ∞     | No (diverges)   |
| $a_n = (-1)^n$      | -     | No (oscillates) |
| $a_n = (1 + 1/n)^n$ | $e$   | Yes             |
| $a_n = n!/n^n$      | 0     | Yes             |

### Important Limits

$$\lim_{n \to \infty} \frac{1}{n^p} = 0 \text{ for } p > 0$$

$$\lim_{n \to \infty} r^n = 0 \text{ for } |r| < 1$$

$$\lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n = e^x$$

$$\lim_{n \to \infty} \frac{n!}{n^n} = 0$$

---

## 2. Infinite Series

### Definition

An **infinite series** is the sum of an infinite sequence:
$$\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots$$

### Partial Sums

$$S_N = \sum_{n=1}^{N} a_n$$

The series converges if $\lim_{N \to \infty} S_N$ exists.

### Geometric Series

$$\sum_{n=0}^{\infty} r^n = \frac{1}{1-r} \text{ for } |r| < 1$$

$$\sum_{n=0}^{\infty} ar^n = \frac{a}{1-r} \text{ for } |r| < 1$$

### p-Series

$$\sum_{n=1}^{\infty} \frac{1}{n^p}$$

Converges if $p > 1$, diverges if $p \leq 1$.

### Harmonic Series

$$\sum_{n=1}^{\infty} \frac{1}{n} = 1 + \frac{1}{2} + \frac{1}{3} + \cdots \text{ (diverges)}$$

---

## 3. Convergence Tests

### Test Summary

```
Convergence Tests:
│
├── Divergence Test
│   └── If lim aₙ ≠ 0, series diverges
│
├── Geometric Series Test
│   └── |r| < 1: converges
│
├── p-Series Test
│   └── p > 1: converges
│
├── Comparison Tests
│   ├── Direct Comparison
│   └── Limit Comparison
│
├── Ratio Test
│   └── lim |aₙ₊₁/aₙ| < 1: converges
│
├── Root Test
│   └── lim |aₙ|^(1/n) < 1: converges
│
└── Alternating Series Test
    └── |aₙ₊₁| ≤ |aₙ| and lim aₙ = 0
```

### Ratio Test (Most Useful!)

For series $\sum a_n$, let $L = \lim_{n \to \infty} \left|\frac{a_{n+1}}{a_n}\right|$

- If $L < 1$: series converges absolutely
- If $L > 1$: series diverges
- If $L = 1$: test inconclusive

### Root Test

Let $L = \lim_{n \to \infty} |a_n|^{1/n}$

- If $L < 1$: converges
- If $L > 1$: diverges
- If $L = 1$: inconclusive

### Alternating Series Test

For $\sum_{n=1}^{\infty} (-1)^{n+1} b_n$ where $b_n > 0$:

If:

1. $b_{n+1} \leq b_n$ (decreasing)
2. $\lim_{n \to \infty} b_n = 0$

Then the series converges.

---

## 4. Power Series

### Definition

$$\sum_{n=0}^{\infty} c_n (x - a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots$$

centered at $x = a$.

### Radius of Convergence

$$R = \lim_{n \to \infty} \left|\frac{c_n}{c_{n+1}}\right|$$

or using root test:
$$R = \frac{1}{\limsup_{n \to \infty} |c_n|^{1/n}}$$

- Converges for $|x - a| < R$
- Diverges for $|x - a| > R$
- Check endpoints separately

### Important Power Series

$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

$$\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$$

$$\cos(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}$$

$$\ln(1+x) = \sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n}$$

$$\frac{1}{1-x} = \sum_{n=0}^{\infty} x^n$$

---

## 5. Taylor Series

### Definition

The Taylor series of $f(x)$ about $x = a$:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

$$= f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots$$

### Maclaurin Series

Taylor series centered at $a = 0$:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!}x^n$$

### Taylor's Theorem (Error Bound)

$$f(x) = T_n(x) + R_n(x)$$

where $T_n$ is the $n$-th degree Taylor polynomial and

$$R_n(x) = \frac{f^{(n+1)}(c)}{(n+1)!}(x-a)^{n+1}$$

for some $c$ between $a$ and $x$.

---

## 6. Common Taylor Series

### Table of Maclaurin Series

| Function        | Series                                                | Convergence     |
| --------------- | ----------------------------------------------------- | --------------- | --- | ---- |
| $e^x$           | $\sum_{n=0}^{\infty} \frac{x^n}{n!}$                  | All $x$         |
| $\sin(x)$       | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$ | All $x$         |
| $\cos(x)$       | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}$     | All $x$         |
| $\ln(1+x)$      | $\sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n}$        | $-1 < x \leq 1$ |
| $(1+x)^k$       | $\sum_{n=0}^{\infty} \binom{k}{n} x^n$                | $               | x   | < 1$ |
| $\frac{1}{1-x}$ | $\sum_{n=0}^{\infty} x^n$                             | $               | x   | < 1$ |

### Derivation Example: e^x

$$f(x) = e^x \implies f^{(n)}(x) = e^x \implies f^{(n)}(0) = 1$$

$$e^x = \sum_{n=0}^{\infty} \frac{1 \cdot x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

---

## 7. Applications in ML/AI

### 1. Taylor Approximations for Activation Functions

**Sigmoid approximation** around $x = 0$:
$$\sigma(x) \approx \frac{1}{2} + \frac{x}{4} - \frac{x^3}{48} + O(x^5)$$

**Softmax temperature** analysis:
$$\text{softmax}(x_i/T) \approx \frac{1}{K} + \frac{x_i - \bar{x}}{KT} + O(1/T^2)$$

### 2. Exponential Learning Rate Decay

$$\alpha_t = \alpha_0 \cdot e^{-\lambda t}$$

Using Taylor series for small $\lambda t$:
$$\alpha_t \approx \alpha_0 (1 - \lambda t + \frac{(\lambda t)^2}{2} - \cdots)$$

### 3. Newton's Method Derivation

Using Taylor expansion:
$$f(x) \approx f(x_n) + f'(x_n)(x - x_n)$$

Setting to zero:
$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

### 4. Second-Order Optimization

Loss function approximation:
$$L(\theta + \Delta\theta) \approx L(\theta) + \nabla L^T \Delta\theta + \frac{1}{2} \Delta\theta^T H \Delta\theta$$

This is a second-order Taylor expansion!

### 5. Log-Sum-Exp Approximation

$$\log\left(\sum_i e^{x_i}\right) \approx \max_i(x_i) + \log\left(\sum_i e^{x_i - \max(x)}\right)$$

### 6. Series in Neural Network Analysis

Weight initialization analysis uses:
$$\text{Var}(\sum_{i=1}^n w_i x_i) = n \cdot \text{Var}(w) \cdot \text{Var}(x)$$

---

## 8. Convergence of Optimization Algorithms

### Gradient Descent Convergence

For convex $L$-smooth function:
$$f(x_k) - f(x^*) \leq \frac{L\|x_0 - x^*\|^2}{2k}$$

This is $O(1/k)$ convergence.

### Convergence Rates

| Algorithm          | Rate         | Type      |
| ------------------ | ------------ | --------- |
| Gradient Descent   | $O(1/k)$     | Sublinear |
| GD + Strong Convex | $O(\rho^k)$  | Linear    |
| Newton's Method    | $O(c^{2^k})$ | Quadratic |
| Accelerated GD     | $O(1/k^2)$   | Sublinear |

```
Convergence Comparison:
│
│ Error
│ │
│ │╲  O(1/k) - GD
│ │ ╲╲
│ │  ╲ ╲ O(1/k²) - Nesterov
│ │   ╲  ╲
│ │    ╲   ╲╲╲ O(ρᵏ) - Linear
│ │     ╲     ╲╲╲
│ └──────────────────→ k
```

---

## 9. Summary

### Key Concepts

| Concept       | Description                           |
| ------------- | ------------------------------------- |
| Sequence      | Ordered list of numbers               |
| Series        | Sum of sequence terms                 |
| Convergence   | Approaches a finite limit             |
| Taylor Series | Polynomial approximation of functions |
| Radius        | Range where power series converges    |

### Important Series

```
Exponential:    e^x = Σ x^n/n!
Sine:           sin(x) = x - x³/3! + x⁵/5! - ...
Cosine:         cos(x) = 1 - x²/2! + x⁴/4! - ...
Geometric:      1/(1-x) = 1 + x + x² + x³ + ...
Logarithm:      ln(1+x) = x - x²/2 + x³/3 - ...
```

### ML Applications

```
Series in ML:
│
├── Function Approximation
│   ├── Activation function Taylor series
│   └── Loss function quadratic approximation
│
├── Optimization Analysis
│   ├── Convergence rate analysis
│   └── Newton's method derivation
│
├── Numerical Stability
│   ├── Computing exp, log safely
│   └── Softmax approximations
│
└── Learning Rate Schedules
    └── Exponential decay analysis
```

---

## Exercises

1. Determine if $\sum_{n=1}^{\infty} \frac{n}{2^n}$ converges using the ratio test
2. Find the first 4 terms of the Taylor series for $\ln(1+x)$ about $x=0$
3. Approximate $e^{0.1}$ using 4 terms of the Maclaurin series
4. Find the radius of convergence for $\sum_{n=0}^{\infty} \frac{x^n}{n!}$
5. Derive the Taylor series approximation for sigmoid around $x=0$

---

## References

1. Stewart - "Calculus: Early Transcendentals"
2. Rudin - "Principles of Mathematical Analysis"
3. Goodfellow et al. - "Deep Learning"
