# Integration

## Introduction

Integration is the reverse of differentiation. While derivatives measure rates of change, integrals measure accumulation—the total effect of continuous change. In machine learning, integration is everywhere:

- **Probability**: Computing $P(a \leq X \leq b)$ requires integrating the PDF
- **Expected Values**: Loss functions are expectations requiring integration
- **Bayesian ML**: Marginalizing over latent variables involves integration
- **Information Theory**: Entropy, KL divergence, and cross-entropy are integrals
- **Generative Models**: VAEs and normalizing flows rely on integration

Unlike differentiation, many integrals have no closed-form solution. This is why **numerical integration** and **Monte Carlo methods** are essential tools in modern ML.

```
Integration in Machine Learning:

Forward Propagation              Training Objective           Inference
─────────────────────           ──────────────────           ─────────
                                                              
∫ p(x|z)p(z) dz                 E[loss] = ∫ L(x)p(x) dx     ∫ p(z|x)p(θ|z) dz
     ↓                               ↓                            ↓
  Decoder output                  Expected risk               Posterior
  (VAE marginal)                (risk minimization)          (Bayesian)
```

## Prerequisites

- Limits and continuity
- Derivatives and differentiation (for Fundamental Theorem)
- Basic algebra and trigonometry

## Learning Objectives

1. Understand definite and indefinite integrals geometrically and algebraically
2. Apply the Fundamental Theorem of Calculus
3. Master integration techniques (substitution, by parts)
4. Connect integration to probability and expected values
5. Use numerical integration when analytical solutions don't exist
6. Apply Monte Carlo integration to high-dimensional problems

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

The FTC connects differentiation and integration—two seemingly different operations.

### Part 1: Differentiation Undoes Integration

If $F(x) = \int_a^x f(t) \, dt$, then $F'(x) = f(x)$

**Interpretation**: If you integrate a function from $a$ to $x$, then differentiate with respect to $x$, you get back the original function.

**In ML Context**: This appears when computing gradients through integral-valued loss functions.

### Part 2: Computing Definite Integrals

$$\int_a^b f(x) \, dx = F(b) - F(a)$$

where $F$ is any antiderivative of $f$ (i.e., $F'(x) = f(x)$).

**Notation**: $F(x) \Big|_a^b = F(b) - F(a)$

**Example**: Compute $\int_0^2 3x^2 \, dx$
- Antiderivative: $F(x) = x^3$
- $F(2) - F(0) = 8 - 0 = 8$

```python
# Verification
import numpy as np
from scipy import integrate

f = lambda x: 3*x**2
result, _ = integrate.quad(f, 0, 2)
print(f"∫[0,2] 3x² dx = {result}")  # Output: 8.0
```

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

Normalization (valid PDF requirement):
$$\int_{-\infty}^{\infty} p(x) \, dx = 1$$

```python
# Probability that standard normal is between -1 and 1
from scipy import integrate, stats

p = lambda x: stats.norm.pdf(x, 0, 1)
prob, _ = integrate.quad(p, -1, 1)
print(f"P(-1 ≤ X ≤ 1) = {prob:.4f}")  # ≈ 0.6827
```

### 2. Expected Value and Loss Functions

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx$$

$$\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot p(x) \, dx$$

**In ML**: Training minimizes expected loss:
$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{data}}[\ell(f_\theta(x), y)]$$

Since we can't integrate over unknonw $p_{data}$, we approximate with sample mean:
$$\mathcal{L}(\theta) \approx \frac{1}{n}\sum_{i=1}^n \ell(f_\theta(x_i), y_i)$$

### 3. Variance

$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot p(x) \, dx$$

Or using the computational formula:
$$\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$$

### 4. Cross-Entropy Loss (Continuous)

$$H(p, q) = -\int p(x) \log q(x) \, dx$$

Cross-entropy measures how well distribution $q$ approximates $p$. In classification, $p$ is the true distribution and $q$ is our model's predictions.

### 5. KL Divergence

$$D_{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx = H(p, q) - H(p, p)$$

**In VAEs**: The ELBO loss contains KL divergence between approximate posterior and prior:
$$\mathcal{L}_{VAE} = \mathbb{E}_{z \sim q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

```python
# KL divergence between two Gaussians N(μ₁, σ₁²) and N(μ₂, σ₂²)
def kl_gaussians(mu1, sigma1, mu2, sigma2):
    """Closed-form KL divergence for Gaussians."""
    return (np.log(sigma2/sigma1) + 
            (sigma1**2 + (mu1-mu2)**2)/(2*sigma2**2) - 0.5)

# KL(N(0,1) || N(1,1)) 
print(f"KL divergence: {kl_gaussians(0, 1, 1, 1):.4f}")  # = 0.5
```

### 6. Marginal Likelihood (Bayesian ML)

$$p(x) = \int p(x|z) p(z) \, dz$$

Integrating out latent variable $z$. This integral is often **intractable**—motivating:
- Variational inference (approximate with ELBO)
- Monte Carlo sampling
- Importance sampling

### 7. Normalizing Flows

Change of variables formula:
$$p_X(x) = p_Z(f^{-1}(x)) \cdot \left|\det \frac{\partial f^{-1}}{\partial x}\right|$$

The Jacobian determinant accounts for how the transformation $f$ "stretches" probability density.

---

## 8. Numerical Integration

When analytical integration fails, numerical methods approximate the integral.

### Rectangle Rule (Midpoint)

$$\int_a^b f(x) \, dx \approx \sum_{i=0}^{n-1} f(x_i^*) \Delta x$$

where $x_i^*$ is the midpoint of each subinterval. Error: $O(\Delta x^2)$

### Trapezoidal Rule

$$\int_a^b f(x) \, dx \approx \frac{\Delta x}{2} \sum_{i=0}^{n-1} [f(x_i) + f(x_{i+1})]$$

Error: $O(\Delta x^2)$ — same order as midpoint, but different constant.

### Simpson's Rule

$$\int_a^b f(x) \, dx \approx \frac{\Delta x}{3} [f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \cdots + f(x_n)]$$

Uses parabolic interpolation. Error: $O(\Delta x^4)$ — much better for smooth functions.

```python
# Comparing numerical methods
import numpy as np
from scipy import integrate

f = lambda x: np.exp(-x**2)
exact, _ = integrate.quad(f, 0, 1)

for n in [10, 100, 1000]:
    x = np.linspace(0, 1, n+1)
    y = f(x)
    
    trap = integrate.trapezoid(y, x)
    simp = integrate.simpson(y, x=x)
    
    print(f"n={n:4d}: Trap error={abs(trap-exact):.2e}, "
          f"Simpson error={abs(simp-exact):.2e}")
```

### Monte Carlo Integration

$$\int_a^b f(x) \, dx \approx (b-a) \cdot \frac{1}{n} \sum_{i=1}^n f(x_i)$$

where $x_i \sim \text{Uniform}(a, b)$.

**Key Properties**:
- Error: $O(1/\sqrt{n})$ — independent of dimension!
- Essential for high-dimensional integrals (curse of dimensionality)
- Used in Bayesian inference, RL (policy evaluation), Physics simulations

```python
# Monte Carlo estimation
np.random.seed(42)
n = 10000
x = np.random.uniform(0, 1, n)
mc_estimate = np.mean(np.exp(-x**2))  # (b-a)=1
print(f"MC estimate: {mc_estimate:.6f}")
print(f"Exact value: {exact:.6f}")
print(f"Error: {abs(mc_estimate - exact):.6f}")
```

### Method Comparison

| Method | Error Order | Dimension Scaling | Best For |
|--------|-------------|-------------------|----------|
| Trapezoidal | $O(h^2)$ | $O(n^d)$ evaluations | Low-dim, smooth |
| Simpson's | $O(h^4)$ | $O(n^d)$ evaluations | Low-dim, very smooth |
| Monte Carlo | $O(1/\sqrt{n})$ | $O(n)$ evaluations | High-dim, complex domains |

**Curse of Dimensionality**: For $d$-dimensional integral with $n$ points per dimension:
- Grid methods: $n^d$ evaluations (exponential in $d$)
- Monte Carlo: $n$ evaluations (linear), same $O(1/\sqrt{n})$ error

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
