# Integration

## Introduction

Integration is the reverse of differentiation. While derivatives measure rates of change, integrals measure accumulation. In ML, integration appears in probability distributions, expected values, information theory, and many optimization problems.

## Prerequisites

- Limits and continuity
- Derivatives and differentiation
- Basic algebra

## Learning Objectives

1. Understand definite and indefinite integrals
2. Apply the Fundamental Theorem of Calculus
3. Master integration techniques
4. Connect integration to probability and ML

---

## 1. The Definite Integral

### Area Under a Curve

The definite integral represents the **signed area** under a curve:

$$\int_a^b f(x) \, dx = \text{Area between } f(x) \text{ and x-axis from } a \text{ to } b$$

```
f(x)
  │
  │     ╱╲
  │    ╱  ╲
  │   ╱    ╲
  │  ╱██████╲
  │ ╱████████╲
  └─┼────────┼───→ x
    a        b

Area = ∫[a to b] f(x) dx
```

### Riemann Sums

Approximate the area with rectangles:

$$\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x$$

where $\Delta x = \frac{b-a}{n}$

---

## 2. The Indefinite Integral (Antiderivative)

### Definition

The indefinite integral is the **antiderivative**:

$$\int f(x) \, dx = F(x) + C$$

where $F'(x) = f(x)$ and $C$ is a constant.

### Basic Integrals

| Function            | Integral                  |
| ------------------- | ------------------------- | --- | ---- |
| $x^n$ $(n \neq -1)$ | $\frac{x^{n+1}}{n+1} + C$ |
| $x^{-1} = 1/x$      | $\ln                      | x   | + C$ |
| $e^x$               | $e^x + C$                 |
| $a^x$               | $\frac{a^x}{\ln a} + C$   |
| $\sin x$            | $-\cos x + C$             |
| $\cos x$            | $\sin x + C$              |
| $\sec^2 x$          | $\tan x + C$              |

---

## 3. Fundamental Theorem of Calculus

### Part 1

If $F(x) = \int_a^x f(t) \, dt$, then $F'(x) = f(x)$

(Integration and differentiation are inverse operations)

### Part 2

$$\int_a^b f(x) \, dx = F(b) - F(a)$$

where $F$ is any antiderivative of $f$.

**Notation**: $F(x) \Big|_a^b = F(b) - F(a)$

---

## 4. Integration Rules

### Linearity

$$\int [af(x) + bg(x)] \, dx = a\int f(x) \, dx + b\int g(x) \, dx$$

### Properties of Definite Integrals

| Property   | Formula                          |
| ---------- | -------------------------------- |
| Additivity | $\int_a^b + \int_b^c = \int_a^c$ |
| Reversal   | $\int_a^b = -\int_b^a$           |
| Zero width | $\int_a^a f = 0$                 |

---

## 5. Integration Techniques

### Substitution (u-substitution)

For $\int f(g(x))g'(x) \, dx$:

Let $u = g(x)$, then $du = g'(x) dx$

$$\int f(g(x))g'(x) \, dx = \int f(u) \, du$$

**Example**: $\int 2x \cdot e^{x^2} \, dx$

Let $u = x^2$, $du = 2x \, dx$
$$= \int e^u \, du = e^u + C = e^{x^2} + C$$

### Integration by Parts

$$\int u \, dv = uv - \int v \, du$$

Choose $u$ and $dv$ using **LIATE** priority:

- **L**ogarithmic
- **I**nverse trig
- **A**lgebraic (polynomials)
- **T**rigonometric
- **E**xponential

**Example**: $\int x e^x \, dx$

Let $u = x$, $dv = e^x dx$
Then $du = dx$, $v = e^x$

$$= xe^x - \int e^x \, dx = xe^x - e^x + C = e^x(x-1) + C$$

---

## 6. Important Integrals for ML

### Gaussian Integral

$$\int_{-\infty}^{\infty} e^{-x^2} \, dx = \sqrt{\pi}$$

### Standard Normal

$$\int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{-x^2/2} \, dx = 1$$

### Gamma Function

$$\Gamma(n) = \int_0^{\infty} x^{n-1} e^{-x} \, dx = (n-1)!$$

### Beta Function

$$B(a, b) = \int_0^1 x^{a-1}(1-x)^{b-1} \, dx = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$$

---

## 7. Applications in ML/AI

### 1. Probability Distributions

For continuous random variable with PDF $p(x)$:

$$P(a \leq X \leq b) = \int_a^b p(x) \, dx$$

Normalization:
$$\int_{-\infty}^{\infty} p(x) \, dx = 1$$

### 2. Expected Value

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$$

$$\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot p(x) \, dx$$

### 3. Variance

$$\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot p(x) \, dx$$

### 4. Cross-Entropy (Continuous)

$$H(p, q) = -\int p(x) \log q(x) \, dx$$

### 5. KL Divergence

$$D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx$$

### 6. Marginal Likelihood

$$p(x) = \int p(x|z) p(z) \, dz$$

(Integrating out latent variable $z$)

---

## 8. Numerical Integration

### Rectangle Rule

$$\int_a^b f(x) \, dx \approx \sum_{i=0}^{n-1} f(x_i) \Delta x$$

### Trapezoidal Rule

$$\int_a^b f(x) \, dx \approx \frac{\Delta x}{2} \sum_{i=0}^{n-1} [f(x_i) + f(x_{i+1})]$$

### Simpson's Rule

$$\int_a^b f(x) \, dx \approx \frac{\Delta x}{3} [f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \cdots + f(x_n)]$$

### Monte Carlo Integration

$$\int_a^b f(x) \, dx \approx (b-a) \cdot \frac{1}{n} \sum_{i=1}^n f(x_i)$$

where $x_i$ are random samples from $[a, b]$.

```
Monte Carlo Integration:
│
├── Sample random points
├── Evaluate function at those points
├── Average and scale by domain size
└── Converges as O(1/√n)
```

---

## 9. Improper Integrals

### Type 1: Infinite Limits

$$\int_a^{\infty} f(x) \, dx = \lim_{t \to \infty} \int_a^t f(x) \, dx$$

### Type 2: Discontinuous Integrand

$$\int_a^b f(x) \, dx = \lim_{c \to a^+} \int_c^b f(x) \, dx$$

(when $f$ has discontinuity at $a$)

### Convergence Tests

- **Comparison**: If $0 \leq f(x) \leq g(x)$ and $\int g$ converges, so does $\int f$
- **p-test**: $\int_1^{\infty} \frac{1}{x^p} dx$ converges iff $p > 1$

---

## 10. Summary

### Key Formulas

| Concept           | Formula                                |
| ----------------- | -------------------------------------- |
| Definite integral | $\int_a^b f(x) dx = F(b) - F(a)$       |
| Substitution      | $\int f(g(x))g'(x) dx = \int f(u) du$  |
| By parts          | $\int u \, dv = uv - \int v \, du$     |
| Expected value    | $\mathbb{E}[X] = \int x \cdot p(x) dx$ |

### Basic Antiderivatives

```
∫ x^n dx = x^(n+1)/(n+1) + C  (n ≠ -1)
∫ 1/x dx = ln|x| + C
∫ e^x dx = e^x + C
∫ sin(x) dx = -cos(x) + C
∫ cos(x) dx = sin(x) + C
```

### ML Connections

```
Integration in ML:
│
├── Probability
│   ├── P(a ≤ X ≤ b) = ∫[a,b] p(x) dx
│   └── Normalization: ∫ p(x) dx = 1
│
├── Expected Values
│   ├── E[X] = ∫ x·p(x) dx
│   └── E[g(X)] = ∫ g(x)·p(x) dx
│
├── Information Theory
│   ├── Entropy: H = -∫ p log p dx
│   └── KL divergence: ∫ p log(p/q) dx
│
└── Bayesian ML
    └── Marginal: p(x) = ∫ p(x|z)p(z) dz
```

---

## Exercises

1. Evaluate $\int (3x^2 + 2x - 1) \, dx$
2. Compute $\int_0^1 e^{-x} \, dx$
3. Use substitution to find $\int \frac{2x}{x^2 + 1} \, dx$
4. Find $\int x \cos(x) \, dx$ using integration by parts
5. For $X \sim \text{Uniform}(0, 1)$, compute $\mathbb{E}[X^2]$

---

## References

1. Stewart - "Calculus: Early Transcendentals"
2. Ross - "A First Course in Probability"
3. Bishop - "Pattern Recognition and Machine Learning"
