# Numerical Integration

> **Navigation**: [← 04-Interpolation-and-Approximation](../04-Interpolation-and-Approximation/) | [Numerical Methods](../) | [11-Graph-Theory →](../../11-Graph-Theory/)

**Files in this section:**
- [theory.ipynb](theory.ipynb) - 12 worked examples
- [exercises.ipynb](exercises.ipynb) - 10 practice problems with solutions

---

## Overview

Numerical integration computes definite integrals $\int_a^b f(x) dx$ when analytical solutions are unavailable or impractical. These methods are fundamental for computing expectations, normalizing constants, and areas in ML.

## Learning Objectives

- Understand quadrature rules and their error analysis
- Implement Newton-Cotes formulas
- Apply Gaussian quadrature for high accuracy
- Handle multi-dimensional integrals with Monte Carlo methods
- Recognize applications in probabilistic ML

## 1. Basic Quadrature Rules

### Newton-Cotes Formulas

Approximate integral using polynomial interpolation on equally spaced nodes.

**Midpoint Rule:**
$$\int_a^b f(x) dx \approx (b-a) f\left(\frac{a+b}{2}\right)$$

Error: $O((b-a)^3 f''(\xi))$

**Trapezoidal Rule:**
$$\int_a^b f(x) dx \approx \frac{b-a}{2}[f(a) + f(b)]$$

Error: $O((b-a)^3 f''(\xi))$

**Simpson's Rule:**
$$\int_a^b f(x) dx \approx \frac{b-a}{6}\left[f(a) + 4f\left(\frac{a+b}{2}\right) + f(b)\right]$$

Error: $O((b-a)^5 f^{(4)}(\xi))$ - exact for polynomials up to degree 3

```
Integration Rules Visualization:

   f(x)         Trapezoidal        Simpson's
    │               │                 │
    │   ┌───────┐   │   ┌─────┐      │   ┌──╮
    │  /│       │   │  /│     │\     │  /    \
    │ / │       │   │ / │     │ \    │ /      \
    │/  │       │   │/  │     │  \   │/        \
    ├───┴───────┤   ├───┴─────┴───┤  ├──────────┤
    a           b   a             b  a          b

    Straight top    Linear approx   Quadratic approx
```

### Composite Rules

Divide $[a,b]$ into $n$ subintervals for better accuracy.

**Composite Trapezoidal:**
$$\int_a^b f(x) dx \approx h\left[\frac{f(a) + f(b)}{2} + \sum_{i=1}^{n-1} f(a + ih)\right]$$

where $h = (b-a)/n$. Error: $O(h^2)$

**Composite Simpson's:**
$$\int_a^b f(x) dx \approx \frac{h}{3}\left[f(a) + 4\sum_{i=1}^{n/2} f(x_{2i-1}) + 2\sum_{i=1}^{n/2-1} f(x_{2i}) + f(b)\right]$$

Error: $O(h^4)$

## 2. Gaussian Quadrature

### Concept

Choose both nodes and weights optimally to maximize polynomial exactness.

**General Form:**
$$\int_a^b w(x) f(x) dx \approx \sum_{i=1}^n w_i f(x_i)$$

where $w(x)$ is a weight function, $x_i$ are nodes, $w_i$ are weights.

**Key Property:** n-point Gaussian quadrature is exact for polynomials of degree up to $2n-1$.

### Gauss-Legendre Quadrature

For $w(x) = 1$ on $[-1, 1]$:

| n   | Nodes        | Weights  |
| --- | ------------ | -------- |
| 1   | 0            | 2        |
| 2   | ±0.577...    | 1        |
| 3   | 0, ±0.775... | 8/9, 5/9 |

```
Gauss-Legendre Nodes (n=5):

   │                                    │
   │  ●      ●      ●      ●      ●    │
   │                                    │
   ├──┬──────┬──────┬──────┬──────┬────┤
  -1                 0                  1

   Nodes are roots of Legendre polynomials
   Clustered near boundaries
```

### Change of Variables

Transform $[a, b]$ to $[-1, 1]$:

$$\int_a^b f(x) dx = \frac{b-a}{2} \int_{-1}^{1} f\left(\frac{b-a}{2}t + \frac{a+b}{2}\right) dt$$

### Other Gaussian Quadratures

| Name            | Weight Function  | Interval            | Use Case              |
| --------------- | ---------------- | ------------------- | --------------------- |
| Gauss-Laguerre  | $e^{-x}$         | $[0, \infty)$       | Gamma function        |
| Gauss-Hermite   | $e^{-x^2}$       | $(-\infty, \infty)$ | Gaussian integrals    |
| Gauss-Chebyshev | $(1-x^2)^{-1/2}$ | $[-1, 1]$           | Oscillatory functions |

## 3. Adaptive Quadrature

### Concept

Recursively subdivide intervals where error is large.

```python
def adaptive_simpson(f, a, b, tol):
    # Compute Simpson's rule on [a,b] and [a,c], [c,b]
    c = (a + b) / 2
    S_ab = simpson(f, a, b)
    S_ac = simpson(f, a, c)
    S_cb = simpson(f, c, b)

    # Error estimate
    error = |S_ab - (S_ac + S_cb)| / 15

    if error < tol:
        return S_ac + S_cb
    else:
        # Subdivide
        return adaptive_simpson(f, a, c, tol/2) + \
               adaptive_simpson(f, c, b, tol/2)
```

### Error Estimation

Richardson extrapolation estimates error without knowing exact integral.

## 4. Monte Carlo Integration

### Basic Monte Carlo

For integral $I = \int_a^b f(x) dx$:

$$\hat{I} = (b-a) \frac{1}{N} \sum_{i=1}^N f(x_i)$$

where $x_i \sim \text{Uniform}(a, b)$

**Error:** Standard error $\propto 1/\sqrt{N}$ (dimension-independent!)

```
Monte Carlo Integration:

    f(x)
      │    ┌──╮
      │   /    \
      │  /      \────╮
      │ /             \
      │/               \
      ├────●──●─●──●────┤
      a    x₁ x₂ x₃ x₄  b

      Random samples estimate area under curve
```

### Importance Sampling

Sample from proposal distribution $q(x)$ instead of uniform:

$$I = \int f(x) dx = \int \frac{f(x)}{q(x)} q(x) dx \approx \frac{1}{N} \sum_{i=1}^N \frac{f(x_i)}{q(x_i)}$$

where $x_i \sim q(x)$

**Optimal choice:** $q(x) \propto |f(x)|$

### Variance Reduction Techniques

1. **Antithetic variates:** Use correlated samples
2. **Control variates:** Subtract known integral
3. **Stratified sampling:** Divide domain into strata

## 5. Multi-dimensional Integration

### Tensor Product Rules

For $d$-dimensional integral:

$$\int_{[0,1]^d} f(\mathbf{x}) d\mathbf{x} \approx \sum_{i_1=1}^{n} \cdots \sum_{i_d=1}^{n} w_{i_1} \cdots w_{i_d} f(x_{i_1}, \ldots, x_{i_d})$$

**Curse of dimensionality:** $n^d$ function evaluations

### Sparse Grids (Smolyak)

Reduce evaluations while maintaining accuracy for smooth functions.

```
Full Grid (2D)          Sparse Grid (2D)
●  ●  ●  ●  ●          ●     ●     ●
●  ●  ●  ●  ●                ●
●  ●  ●  ●  ●          ●  ●  ●  ●  ●
●  ●  ●  ●  ●                ●
●  ●  ●  ●  ●          ●     ●     ●

25 points               13 points
```

### Monte Carlo for High Dimensions

- Error $O(1/\sqrt{N})$ regardless of dimension
- Becomes competitive for $d > 4-5$

## 6. Applications in Machine Learning

### Computing Expectations

$$\mathbb{E}[g(X)] = \int g(x) p(x) dx$$

**Methods:**

- Gaussian quadrature for low-dimensional
- Monte Carlo for high-dimensional
- Importance sampling when $p(x)$ is complex

### Bayesian Inference

**Evidence computation:**
$$p(\mathcal{D}) = \int p(\mathcal{D}|\theta) p(\theta) d\theta$$

**Posterior mean:**
$$\mathbb{E}[\theta|\mathcal{D}] = \frac{\int \theta \, p(\mathcal{D}|\theta) p(\theta) d\theta}{p(\mathcal{D})}$$

### Normalizing Constants

**Partition function:**
$$Z = \int \exp(-E(x)) dx$$

Critical for:

- Boltzmann machines
- Probabilistic graphical models
- Variational inference

### Gauss-Hermite for Gaussian Integrals

For $X \sim \mathcal{N}(\mu, \sigma^2)$:

$$\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}} dx$$

Transform to standard form:
$$= \frac{1}{\sqrt{\pi}} \int_{-\infty}^{\infty} g(\sqrt{2}\sigma t + \mu) e^{-t^2} dt$$

Apply Gauss-Hermite quadrature.

## 7. Numerical Stability

### Cancelation Issues

Avoid subtracting nearly equal quantities:

```python
# Bad: I_1 - I_2 with similar values
# Better: Compute difference directly
```

### Overflow/Underflow

For log-scale computations (common in ML):
$$\log \int e^{f(x)} dx$$

Use log-sum-exp trick:
$$\log \int e^{f(x)} dx = M + \log \int e^{f(x) - M} dx$$

## Summary

| Method              | Accuracy      | Dimensions | Use Case           |
| ------------------- | ------------- | ---------- | ------------------ |
| Trapezoidal         | $O(h^2)$      | Low        | Simple functions   |
| Simpson             | $O(h^4)$      | Low        | Smooth functions   |
| Gauss-Legendre      | Optimal       | Low        | High accuracy      |
| Gauss-Hermite       | Optimal       | Low        | Gaussian integrals |
| Monte Carlo         | $O(N^{-1/2})$ | Any        | High dimensions    |
| Importance Sampling | Varies        | Any        | Known structure    |

## Key Takeaways

1. **Newton-Cotes:** Simple but limited accuracy
2. **Gaussian quadrature:** Optimal for low dimensions
3. **Monte Carlo:** Essential for high dimensions
4. **Importance sampling:** Reduces variance when well-designed
5. **Adaptive methods:** Handle difficult integrands automatically
6. **Curse of dimensionality:** Makes high-d integration challenging

## Practice Problems

1. Compare trapezoidal and Simpson's rule for $\int_0^{\pi} \sin(x) dx$
2. Implement Gauss-Legendre quadrature with $n=5$ nodes
3. Compute $\mathbb{E}[\exp(X)]$ where $X \sim \mathcal{N}(0,1)$ using Gauss-Hermite
4. Use Monte Carlo to estimate $\int_0^1 \int_0^1 \int_0^1 \int_0^1 \exp(-\|\mathbf{x}\|^2) d\mathbf{x}$
5. Implement importance sampling for a tail probability
