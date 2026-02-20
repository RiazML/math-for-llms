# Joint Distributions and Dependence

> **Navigation**: [01-Introduction](../01-Introduction-and-Random-Variables/) | [02-Common-Distributions](../02-Common-Distributions/) | [04-Expectation-and-Moments](../04-Expectation-and-Moments/)

## Overview

Real-world ML problems involve **multiple random variables**. Joint distributions capture how variables relate—enabling us to model dependencies, compute conditionals, and make predictions. Understanding joint distributions is essential for multivariate modeling, graphical models, and dimensionality reduction.

```
┌─────────────────────────────────────────────────────────────────────────┐
│              JOINT DISTRIBUTIONS IN MACHINE LEARNING                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Given: Joint P(X,Y)                                                     │
│  ─────                                                                   │
│       ┌──────────────┬───────────────┬───────────────┐                  │
│       │              │               │               │                  │
│       ▼              ▼               ▼               ▼                  │
│   Marginal       Conditional    Covariance      Independence           │
│   P(X), P(Y)     P(Y|X)        Cov(X,Y)        X ⊥ Y ?                  │
│       │              │               │               │                  │
│       ▼              ▼               ▼               ▼                  │
│   Feature        Prediction     Feature          Model                  │
│   Selection      Y from X       Engineering      Simplification         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- [01-Introduction-and-Random-Variables](../01-Introduction-and-Random-Variables/)
- [02-Common-Distributions](../02-Common-Distributions/)
- Linear algebra basics (matrices)

## Learning Objectives

1. Understand joint, marginal, and conditional distributions
2. Compute and interpret covariance and correlation
3. Master multivariate Gaussian properties
4. Apply independence concepts in ML

---

## 1. Joint Probability Mass Function (Discrete)

For discrete random variables $X$ and $Y$:

$$p_{X,Y}(x, y) = P(X = x, Y = y)$$

### Properties

1. **Non-negativity:** $p_{X,Y}(x, y) \geq 0$
2. **Normalization:** $\sum_x \sum_y p_{X,Y}(x, y) = 1$

### Example: Joint PMF Table

```
                        Y
                 0      1      2      P(X=x)
           ┌────────────────────────┬─────────
         0 │  0.10   0.15   0.05   │  0.30
    X    1 │  0.20   0.25   0.05   │  0.50
         2 │  0.10   0.05   0.05   │  0.20
           └────────────────────────┴─────────
      P(Y=y)   0.40   0.45   0.15     1.00

• Each cell: P(X=x, Y=y)
• Row sums: Marginal P(X=x)
• Column sums: Marginal P(Y=y)
• Grand total: 1.00
```

---

## 2. Joint Probability Density Function (Continuous)

For continuous random variables:

$$P((X, Y) \in A) = \iint_A f_{X,Y}(x, y) \, dx \, dy$$

### Properties

1. **Non-negativity:** $f_{X,Y}(x, y) \geq 0$
2. **Normalization:** $\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dx \, dy = 1$

```
Joint PDF Visualization (Bivariate Normal):

       y
       │               ░░░░
       │            ░░░████░░░
       │          ░░██████████░░
       │         ░████████████████░
       │        ░██████████████████░
       │         ░████████████████░
       │          ░░██████████░░
       │            ░░░████░░░
       │               ░░░░
       └────────────────────────── x

Contour lines show constant density levels
Volume under surface = 1
```

---

## 3. Marginal Distributions

**Marginal distribution** of $X$ "integrates out" $Y$:

$$\text{Discrete: } p_X(x) = \sum_y p_{X,Y}(x, y)$$
$$\text{Continuous: } f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x, y) \, dy$$

> **💡 Intuition**: To find P(X=x), sum over all possible values of Y

```
Marginalization:

Joint P(X,Y)                        Marginal P(X)
┌─────────────────┐                 ┌─────────────┐
│  ░░░░  ░░       │   Sum over Y    │    ████     │
│ ░████░░██░      │  ───────────→   │   ██████    │
│░██████████░     │                 │  ████████   │
│ ░████░░██░      │                 │   ██████    │
│  ░░░░  ░░       │                 │    ████     │
└─────────────────┘                 └─────────────┘
     (2D surface)                       (1D curve)

"Project" the joint distribution onto one axis
```

---

## 4. Conditional Distributions

### Definition

$$p_{Y|X}(y|x) = \frac{p_{X,Y}(x, y)}{p_X(x)}, \quad \text{if } p_X(x) > 0$$

$$f_{Y|X}(y|x) = \frac{f_{X,Y}(x, y)}{f_X(x)}, \quad \text{if } f_X(x) > 0$$

> **💡 Intuition**: "Slice" the joint distribution at X=x, then renormalize to get a valid distribution for Y.

```
Conditional Distribution:

Joint P(X,Y)                        Conditional P(Y|X=x₀)
┌─────────────────┐                 
│      │          │                      │
│   ░░░│░░        │   Slice at X=x₀      │   ░░
│  ░███│███░      │  ─────────────→      │  ████
│ ░████│████░     │                      │ ██████
│  ░███│███░      │                      │  ████
│   ░░░│░░        │                      │   ░░
│      │          │                      │
└──────┴──────────┘                      └─────────
       x₀                                    y
```

### Chain Rule

$$p_{X,Y}(x, y) = p_{Y|X}(y|x) \cdot p_X(x) = p_{X|Y}(x|y) \cdot p_Y(y)$$

For multiple variables:
$$p(x_1, x_2, \ldots, x_n) = p(x_1) \cdot p(x_2|x_1) \cdot p(x_3|x_1,x_2) \cdots$$

---

## 5. Independence

### Definition

$X$ and $Y$ are **independent** ($X \perp\!\!\!\perp Y$) if:

$$p_{X,Y}(x, y) = p_X(x) \cdot p_Y(y) \quad \text{for all } x, y$$

Equivalently:
- $p_{Y|X}(y|x) = p_Y(y)$ (conditioning doesn't change distribution)
- Knowing X gives no information about Y

```
Independence vs Dependence:

Independent Joint P(X,Y)          Dependent Joint P(X,Y)
┌─────────────────────┐          ┌─────────────────────┐
│                     │          │            ╱        │
│  ░░░░     ░░░░      │          │          ╱          │
│  ████     ████      │          │        ░░░          │
│  ░░░░     ░░░░      │          │      ░████░         │
│                     │          │    ░████████░       │
│  ░░░░     ░░░░      │          │      ░████░         │
│  ████     ████      │          │        ░░░          │
│  ░░░░     ░░░░      │          │          ╲          │
│                     │          │            ╲        │
└─────────────────────┘          └─────────────────────┘

Joint = Product of marginals      Joint ≠ Product of marginals
```

### Conditional Independence

$X \perp\!\!\!\perp Y \mid Z$ means:

$$p_{X,Y|Z}(x, y | z) = p_{X|Z}(x|z) \cdot p_{Y|Z}(y|z)$$

> **🔑 Key ML Concept**: Many graphical models rely on conditional independence to simplify computation!

```
Conditional Independence:

              Z (condition)
             ╱ ╲
            ╱   ╲
           ▼     ▼
          X       Y

Given Z, X and Y are independent.
But marginally (ignoring Z), X and Y may be dependent!
```

---

## 6. Covariance and Correlation

### Covariance

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

| Cov(X,Y) | Interpretation |
|----------|----------------|
| $> 0$ | X and Y tend to move together |
| $< 0$ | X and Y tend to move oppositely |
| $= 0$ | No linear relationship |

### Properties of Covariance

| Property | Formula |
|----------|---------|
| Symmetry | $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ |
| Self-covariance | $\text{Cov}(X, X) = \text{Var}(X)$ |
| Linearity | $\text{Cov}(aX + b, Y) = a \cdot \text{Cov}(X, Y)$ |
| Sum | $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$ |
| Constant | $\text{Cov}(c, X) = 0$ |

### Variance of Sum

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

$$\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i<j} \text{Cov}(X_i, X_j)$$

### Correlation

$$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}, \quad -1 \leq \rho \leq 1$$

```
Correlation Patterns:

ρ = 1 (perfect +)     ρ = 0.7 (strong +)    ρ = 0 (none)
    y                      y                     y
    │    ●●●               │      ●              │  ●   ●  ●
    │   ●●●●               │    ●●●●             │●  ●  ● ●
    │  ●●●●●               │   ●●●●●●            │ ● ●●●●
    │ ●●●●●                │  ●●●●●●●            │●●●● ●● ●
    │●●●●●                 │ ●●●●●●              │ ● ●  ●●●
    └────────── x          └────────── x         └────────── x

ρ = -0.7 (strong -)   ρ = -1 (perfect -)
    y                      y
    │●●                    │●●●●●
    │ ●●●●                 │ ●●●●
    │  ●●●●●               │  ●●●
    │   ●●●●●●             │   ●●
    │     ●●●              │    ●
    └────────── x          └────────── x
```

> **⚠️ Warning**: Correlation = 0 does NOT imply independence!
> (Counterexample: $X \sim N(0,1)$, $Y = X^2$ have zero correlation but are clearly dependent)

---

## 7. Covariance Matrix

For random vector $\mathbf{X} = [X_1, \ldots, X_n]^T$:

$$\Sigma = E[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T]$$

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

```
Covariance Matrix Structure:

         X₁       X₂       X₃       X₄
    ┌────────────────────────────────────┐
X₁  │ Var(X₁)  Cov₁₂    Cov₁₃    Cov₁₄  │
X₂  │ Cov₂₁   Var(X₂)   Cov₂₃    Cov₂₄  │
X₃  │ Cov₃₁   Cov₃₂    Var(X₃)   Cov₃₄  │
X₄  │ Cov₄₁   Cov₄₂    Cov₄₃    Var(X₄) │
    └────────────────────────────────────┘

• Diagonal: Variances
• Off-diagonal: Covariances
• Symmetric: Σᵢⱼ = Σⱼᵢ
• Positive semi-definite: xᵀΣx ≥ 0
```

### Properties

1. **Symmetric:** $\Sigma = \Sigma^T$
2. **Positive semi-definite:** $\mathbf{x}^T \Sigma \mathbf{x} \geq 0$
3. **Linear transformation:** If $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$, then $\Sigma_Y = A \Sigma_X A^T$

---

## 8. Multivariate Normal

The **multivariate normal** is the most important joint distribution in ML:

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)$$

### Key Properties

| Property | Result |
|----------|--------|
| Marginals | Gaussian |
| Conditionals | Gaussian |
| Linear combos | Gaussian |
| Uncorrelated ⟺ Independent | Only true for Gaussians! |

### Conditional Gaussian

For partitioned $\mathbf{X} = [X_1, X_2]^T$ with:
$$\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}, \quad \Sigma = \begin{bmatrix} \Sigma_{11} & \Sigma_{12} \\ \Sigma_{21} & \Sigma_{22} \end{bmatrix}$$

The conditional distribution $X_1 | X_2 = x_2$ is Gaussian with:
$$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(x_2 - \mu_2)$$
$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$

```
Conditioning in 2D Gaussian:

      y
      │         ╱╲
      │       ╱    ╲
      │     ╱░░░░░░░░╲  ← Contour of joint P(X,Y)
      │   ╱░░░░░░░░░░░░╲
  y₀ ─┼──▓▓▓▓▓▓▓▓▓▓▓▓▓▓──── ← Slice at Y = y₀
      │   ╲░░░░░░░░░░░░╱
      │     ╲░░░░░░░░╱
      │       ╲    ╱
      │         ╲╱
      └──────────────────── x
                 ↑
           Conditional P(X|Y=y₀)
           is also Gaussian!
```

> **🔑 ML Applications:**
> - Gaussian Processes: conditionals for prediction
> - Kalman Filter: state estimation
> - Linear regression: posterior distribution

---

## 9. Bayes' Theorem for Random Variables

For random variables:

$$f_{Y|X}(y|x) = \frac{f_{X|Y}(x|y) \cdot f_Y(y)}{f_X(x)}$$

In ML notation:

$$p(\theta | \mathbf{x}) = \frac{p(\mathbf{x} | \theta) \cdot p(\theta)}{p(\mathbf{x})} \propto p(\mathbf{x} | \theta) \cdot p(\theta)$$

```
Bayesian Inference:

Prior P(θ)              Likelihood P(x|θ)           Posterior P(θ|x)
    │                        │                          │
    │     ╱╲                 │          ╱╲              │      ╱╲
    │   ╱    ╲               │        ╱    ╲            │    ╱    ╲
    │ ╱        ╲             │  ────╱────────╲────     │  ╱        ╲
    │╱          ╲            │                          │╱            ╲
    └────────────            └─────────────────         └──────────────
                    ×                           =
     What we                  How well             Updated belief
     believe                  data fits            after seeing data
```

---

## 10. Transformations of Random Variables

### Single Variable

If $Y = g(X)$ and $g$ is monotonic:

$$f_Y(y) = f_X(g^{-1}(y)) \left| \frac{d g^{-1}(y)}{dy} \right|$$

### Multiple Variables (Jacobian)

If $\mathbf{Y} = g(\mathbf{X})$:

$$f_Y(\mathbf{y}) = f_X(g^{-1}(\mathbf{y})) \left| \det\left( \frac{\partial g^{-1}}{\partial \mathbf{y}} \right) \right|$$

> **🔑 ML Application**: Normalizing flows use this formula to transform simple distributions into complex ones!

---

## 11. ML Applications

### Feature Independence (Naive Bayes)

$$P(y | x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^n P(x_i | y)$$

Assumes: $x_i \perp\!\!\!\perp x_j \mid y$ for all $i \neq j$

### Gaussian Mixture Models

$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \cdot \mathcal{N}(\mathbf{x} | \boldsymbol{\mu}_k, \Sigma_k)$$

Joint over $(X, Z)$ where $Z$ is the latent cluster assignment.

### Principal Component Analysis

Find directions that maximize variance:
$$\text{maximize } \mathbf{w}^T \Sigma \mathbf{w} \quad \text{subject to } \|\mathbf{w}\| = 1$$

Solution: eigenvectors of covariance matrix $\Sigma$.

### Graphical Models

```
Bayesian Network:

     Rain
      │╲
      │ ╲
      ▼  ▼
  Sprinkler  Cloudy
      │╲      ╱
      │ ╲    ╱
      ▼  ▼  ▼
        Wet Grass

Joint = P(Rain) × P(Cloudy|Rain) × P(Sprinkler|Rain) × P(Wet|Sprinkler,Cloudy)
```

---

## 12. Summary Tables

### Key Formulas

| Concept | Formula |
|---------|---------|
| Marginal | $p_X(x) = \sum_y p_{X,Y}(x,y)$ |
| Conditional | $p_{Y\|X}(y\|x) = p_{X,Y}(x,y) / p_X(x)$ |
| Independence | $p_{X,Y}(x,y) = p_X(x) \cdot p_Y(y)$ |
| Covariance | $\text{Cov}(X,Y) = E[XY] - E[X]E[Y]$ |
| Correlation | $\rho = \text{Cov}(X,Y) / (\sigma_X \sigma_Y)$ |
| Var of sum | $\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$ |

### Independence Cheat Sheet

```
┌──────────────────────────────────────────────────────────────┐
│                    INDEPENDENCE FACTS                         │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  X ⊥ Y (independent)                                         │
│  ────────────────────                                        │
│  • P(X,Y) = P(X)P(Y)                                         │
│  • P(Y|X) = P(Y)                                             │
│  • Cov(X,Y) = 0                                              │
│  • Var(X+Y) = Var(X) + Var(Y)                                │
│                                                               │
│  BUT: Cov(X,Y) = 0 does NOT imply X ⊥ Y !                    │
│                                                               │
│  For Gaussians ONLY:                                         │
│  • Cov(X,Y) = 0 ⟺ X ⊥ Y                                      │
│                                                               │
│  X ⊥ Y | Z (conditionally independent)                      │
│  ─────────────────────────────────────                       │
│  • P(X,Y|Z) = P(X|Z)P(Y|Z)                                   │
│  • X ⊥ Y | Z does NOT imply X ⊥ Y                            │
│  • X ⊥ Y does NOT imply X ⊥ Y | Z                            │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Exercises

1. Compute marginal and conditional distributions from a given joint PMF table
2. Verify that $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ when $X \perp\!\!\!\perp Y$
3. Find the conditional distribution $X_1 | X_2 = 1$ for bivariate normal
4. Show that correlation = 0 doesn't imply independence (find counterexample)
5. Compute the Jacobian for polar coordinate transformation

---

## References

1. Casella & Berger - "Statistical Inference"
2. Bishop - "Pattern Recognition and Machine Learning"
3. Koller & Friedman - "Probabilistic Graphical Models"

---

> **Next**: [04-Expectation-and-Moments](../04-Expectation-and-Moments/) — Expected values and higher moments
