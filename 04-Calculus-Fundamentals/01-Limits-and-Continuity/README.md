# Limits and Continuity

## Introduction

Limits are the foundation of calculus. They formalize the idea of "approaching" a value and are essential for defining derivatives and integrals. Without limits, we cannot rigorously define:
- **Derivatives**: The instantaneous rate of change
- **Integrals**: The accumulation of infinitesimal quantities
- **Infinite series**: Sums of infinitely many terms

In machine learning, understanding limits is crucial for:
- **Convergence analysis**: Does gradient descent converge?
- **Asymptotic behavior**: How do algorithms scale?
- **Numerical stability**: Avoiding undefined computations
- **Activation functions**: Understanding saturation behavior

## Prerequisites

- Functions and graphs
- Basic algebra and inequalities
- Familiarity with polynomials, exponentials, and trigonometric functions

## Learning Objectives

1. Understand the intuitive and formal (ε-δ) definition of limits
2. Evaluate limits using algebraic techniques, L'Hôpital's Rule, and the Squeeze Theorem
3. Classify discontinuities (removable, jump, infinite)
4. Apply limit concepts to ML scenarios (softmax temperature, vanishing gradients, SGD convergence)

---

## 1. Intuitive Definition of Limits

### The Idea

$$\lim_{x \to a} f(x) = L$$

means: as $x$ gets arbitrarily close to $a$, $f(x)$ gets arbitrarily close to $L$.

**Important**: The limit describes what happens as we *approach* $a$, not what happens *at* $a$. The function doesn't even need to be defined at $a$ for the limit to exist!

### Visualization

```
f(x)
  │
L ─┼─ ─ ─ ─ ─ ─ ─●─ ─ ─ ─
  │           ╱     ╲
  │         ╱         ╲
  │       ╱
  │     ╱
  └──────────┼───────────→ x
             a

As x → a, f(x) → L
```

### One-Sided Limits

| Notation                | Meaning                                   |
| ----------------------- | ----------------------------------------- |
| $\lim_{x \to a^-} f(x)$ | Left-hand limit (approaching from below)  |
| $\lim_{x \to a^+} f(x)$ | Right-hand limit (approaching from above) |

The two-sided limit exists iff both one-sided limits exist and are equal.

### Examples of Different Cases

| Function | At $x = 0$ | Limit as $x \to 0$ |
|----------|------------|-------------------|
| $f(x) = x^2$ | $f(0) = 0$ | $\lim = 0$ (limit equals value) |
| $f(x) = \frac{x^2}{x}$ | undefined | $\lim = 0$ (limit exists despite hole) |
| $f(x) = \frac{\|x\|}{x}$ | undefined | DNE (left ≠ right) |
| $f(x) = \frac{1}{x}$ | undefined | DNE (approaches ±∞) |

---

## 2. Formal Definition (ε-δ)

$$\lim_{x \to a} f(x) = L$$

if for every $\varepsilon > 0$, there exists $\delta > 0$ such that:

$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$

### Interpretation

```
        f(x)
          │
    L+ε ──┼─────────────────────
          │       ╱ ─ ─ ╲
      L ──┼─ ─ ─●─ ─ ─ ─ ─ ─ ─
          │   ╱ ─ ─ ─ ─ ╲
    L-ε ──┼─────────────────────
          │
          └───┼───┼───┼───┼───→ x
            a-δ   a   a+δ

For any ε-band around L,
there's a δ-band around a
keeping f(x) in the ε-band.
```

### Reading the Definition

1. **"For every ε > 0"**: The challenger picks any precision goal
2. **"There exists δ > 0"**: We must find a response
3. **"0 < |x - a| < δ"**: When x is within δ of a (but not at a)
4. **"⟹ |f(x) - L| < ε"**: Then f(x) is within ε of L

### Why ε-δ Matters

The ε-δ definition:
- Makes limits rigorous and unambiguous
- Enables formal proofs of limit properties
- Generalizes to multivariable calculus and topology
- Provides the foundation for continuity and differentiability

### Example: Proving $\lim_{x \to 2} (3x - 1) = 5$

**Goal**: For any ε > 0, find δ such that |x - 2| < δ ⟹ |(3x-1) - 5| < ε

**Solution**:
- |(3x - 1) - 5| = |3x - 6| = 3|x - 2|
- Want: 3|x - 2| < ε
- So: |x - 2| < ε/3
- Choose δ = ε/3 ✓

---

## 3. Limit Laws

For $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$:

| Law        | Statement                       |
| ---------- | ------------------------------- |
| Sum        | $\lim(f + g) = L + M$           |
| Difference | $\lim(f - g) = L - M$           |
| Product    | $\lim(f \cdot g) = L \cdot M$   |
| Quotient   | $\lim(f/g) = L/M$ if $M \neq 0$ |
| Constant   | $\lim(c \cdot f) = c \cdot L$   |
| Power      | $\lim(f^n) = L^n$               |

---

## 4. Common Techniques

### Direct Substitution

If $f$ is continuous at $a$:
$$\lim_{x \to a} f(x) = f(a)$$

This is the simplest case - just plug in the value!

### Factoring (0/0 form)

When direct substitution gives 0/0, try factoring:

$$\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = \lim_{x \to 2} \frac{(x-2)(x+2)}{x-2} = \lim_{x \to 2} (x + 2) = 4$$

### Rationalization

For limits involving square roots:

$$\lim_{x \to 0} \frac{\sqrt{x+1} - 1}{x} = \lim_{x \to 0} \frac{(\sqrt{x+1}-1)(\sqrt{x+1}+1)}{x(\sqrt{x+1}+1)} = \lim_{x \to 0} \frac{x}{x(\sqrt{x+1}+1)} = \frac{1}{2}$$

### L'Hôpital's Rule

For indeterminate forms $\frac{0}{0}$ or $\frac{\infty}{\infty}$:

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

**When to use**: Only when direct methods fail and you have 0/0 or ∞/∞.

**Common mistake**: Applying L'Hôpital to non-indeterminate forms!

### Technique Selection Flowchart

```
Start: lim(x→a) f(x)
       │
       ▼
   Try substitution
       │
       ├── Works? → Done!
       │
       ▼
   Indeterminate?
       │
       ├── 0/0 → Factor, rationalize, or L'Hôpital
       │
       ├── ∞/∞ → Divide by highest power or L'Hôpital
       │
       ├── 0·∞ → Rewrite as 0/0 or ∞/∞
       │
       ├── ∞-∞ → Find common factor
       │
       └── 1^∞, 0^0, ∞^0 → Use logarithm
```

---

## 5. Important Limits

### Fundamental Limits

| Limit                                     | Value | Importance |
| ----------------------------------------- | ----- | ---------- |
| $\lim_{x \to 0} \frac{\sin x}{x}$         | $1$   | Derivative of sin |
| $\lim_{x \to 0} \frac{1 - \cos x}{x}$     | $0$   | Derivative of cos |
| $\lim_{x \to 0} \frac{1 - \cos x}{x^2}$   | $\frac{1}{2}$   | Taylor series |
| $\lim_{x \to 0} \frac{e^x - 1}{x}$        | $1$   | Derivative of exp |
| $\lim_{x \to 0} \frac{\ln(1+x)}{x}$       | $1$   | Derivative of ln |
| $\lim_{x \to \infty} (1 + \frac{1}{x})^x$ | $e$   | Definition of e |
| $\lim_{x \to 0} (1 + x)^{1/x}$            | $e$   | Compound interest |

### Generalized Exponential Limit

$$\lim_{n \to \infty} \left(1 + \frac{r}{n}\right)^n = e^r$$

This is the foundation of continuous compounding in finance and exponential growth models.

### Limits at Infinity (Polynomial Ratios)

$$\lim_{x \to \infty} \frac{P(x)}{Q(x)} = \begin{cases} 0 & \text{if } \deg P < \deg Q \\ \frac{a_n}{b_m} & \text{if } \deg P = \deg Q \\ \pm\infty & \text{if } \deg P > \deg Q \end{cases}$$

**Intuition**: The highest-degree terms dominate as x → ∞.

### Exponential vs Polynomial Growth

$$\lim_{x \to \infty} \frac{x^n}{e^x} = 0 \quad \text{for any } n$$

Exponential functions eventually outgrow any polynomial - important for algorithm complexity analysis.

---

## 6. Continuity

### Definition

$f$ is **continuous at $a$** if:

1. $f(a)$ is defined
2. $\lim_{x \to a} f(x)$ exists
3. $\lim_{x \to a} f(x) = f(a)$

**In words**: You can draw the graph without lifting your pen, and the function value equals the limit.

### Types of Discontinuities

```
Removable:           Jump:               Infinite:
    │                  │                    │  │
    ○                  ●                    │  │
    │                  │                    │  │
────●────          ────┤              ──────┼──┼──
    │              ●───┤                    │  │
    │                  │                   ╱    ╲
                                         ╱      ╲
(hole in graph)   (left ≠ right)    (vertical asymptote)
```

| Type | Limit Exists? | Can Be Fixed? | Example |
|------|--------------|---------------|---------|
| Removable | Yes | Yes, redefine f(a) | $\frac{x^2-1}{x-1}$ at $x=1$ |
| Jump | No (left ≠ right) | No | $\frac{\|x\|}{x}$ at $x=0$ |
| Infinite | No (→±∞) | No | $\frac{1}{x}$ at $x=0$ |

### Properties of Continuous Functions

| Property       | Statement                                 |
| -------------- | ----------------------------------------- |
| Sum/Difference | $f \pm g$ continuous                      |
| Product        | $f \cdot g$ continuous                    |
| Quotient       | $f/g$ continuous where $g \neq 0$         |
| Composition    | $f \circ g$ continuous if both continuous |
| Scalar multiple | $c \cdot f$ continuous for constant $c$  |

### Important Theorems

**Intermediate Value Theorem (IVT)**:
If $f$ is continuous on $[a,b]$ and $y$ is between $f(a)$ and $f(b)$, then there exists $c \in (a,b)$ such that $f(c) = y$.

**Extreme Value Theorem (EVT)**:
If $f$ is continuous on a closed interval $[a,b]$, then $f$ attains its maximum and minimum values.

### ML Relevance: Activation Functions

| Function | Continuous? | Differentiable? | Note |
|----------|-------------|-----------------|------|
| Sigmoid | Yes | Yes | Can saturate |
| Tanh | Yes | Yes | Can saturate |
| ReLU | Yes | No (at 0) | Non-differentiable kink |
| Leaky ReLU | Yes | No (at 0) | Small gradient for x < 0 |
| GELU | Yes | Yes | Smooth approximation of ReLU |

---

## 7. Squeeze Theorem

If $g(x) \leq f(x) \leq h(x)$ near $a$ and:
$$\lim_{x \to a} g(x) = \lim_{x \to a} h(x) = L$$

then:
$$\lim_{x \to a} f(x) = L$$

### Classic Example

$$\lim_{x \to 0} x^2 \sin\left(\frac{1}{x}\right) = 0$$

Since $-x^2 \leq x^2 \sin(1/x) \leq x^2$ and both bounds → 0.

---

## 8. Applications in ML/AI

### 1. Convergence of Gradient Descent

Learning rate decay:
$$\lim_{t \to \infty} \alpha_t = 0 \quad \text{but} \quad \sum_{t=1}^{\infty} \alpha_t = \infty$$

Example: $\alpha_t = \frac{1}{t}$

**Why both conditions?**
- $\sum \alpha_t = \infty$: Can reach any point in parameter space
- $\sum \alpha_t^2 < \infty$: Noise variance goes to zero

The harmonic series $\sum 1/t$ diverges, but $\sum 1/t^2 = \pi^2/6$ converges.

### 2. Softmax Limits

$$\lim_{T \to 0^+} \text{softmax}(z_i/T) = \begin{cases} 1 & \text{if } z_i = \max_j z_j \\ 0 & \text{otherwise} \end{cases}$$

(Temperature → 0 makes softmax "harder")

### 3. Sigmoid Saturation

$$\lim_{x \to \infty} \sigma(x) = 1, \quad \lim_{x \to -\infty} \sigma(x) = 0$$

**The Vanishing Gradient Problem**:
- Sigmoid derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- At saturation: $\sigma'(x) \approx 0$
- Deep networks: gradients multiply across layers
- Result: Early layers receive near-zero gradients

This is why ReLU and its variants became popular - they don't saturate for positive inputs.

### 4. Batch Normalization

As batch size $n \to \infty$:
$$\bar{x} \to \mu, \quad s^2 \to \sigma^2$$

The sample statistics converge to population parameters by the Law of Large Numbers.

### 5. Asymptotic Complexity

$$\lim_{n \to \infty} \frac{f(n)}{g(n)} = c \implies f(n) = \Theta(g(n))$$

Used to analyze algorithm complexity - what matters as the problem size grows?

### 6. Cross-Entropy Loss

$$\lim_{p \to 0^+} -\log(p) = +\infty$$

When the model predicts probability $p \approx 0$ for the true class:
- The cross-entropy loss → ∞
- This severely penalizes confident wrong predictions
- Makes cross-entropy effective for classification

### 7. Knowledge Distillation

In knowledge distillation, temperature $T$ controls softmax sharpness:
- High $T$: Soft labels reveal "dark knowledge"
- Low $T$: Sharp labels for final prediction

### 8. Weight Initialization

Xavier/Glorot initialization ensures:
$$\lim_{n \to \infty} \text{Var}(\text{activations}) = \text{constant}$$

Prevents vanishing/exploding signals in deep networks.

---

## 9. Numerical Considerations

### Finite Differences Approach Derivatives

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

But in practice, we use small finite $h$ (not 0).

**Forward difference**: $(f(x+h) - f(x))/h$ — Error: $O(h)$

**Central difference**: $(f(x+h) - f(x-h))/(2h)$ — Error: $O(h^2)$

### Numerical Stability

$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

But computing directly for small $x$ causes cancellation errors!

```python
# WRONG (cancellation error)
result = (np.exp(x) - 1) / x

# RIGHT (numerically stable)
result = np.expm1(x) / x
```

### Common Numerically Stable Functions

| Mathematical | Naive | Stable |
|-------------|-------|--------|
| $e^x - 1$ | `np.exp(x) - 1` | `np.expm1(x)` |
| $\log(1 + x)$ | `np.log(1 + x)` | `np.log1p(x)` |
| $\sqrt{x^2 + 1} - 1$ | `np.sqrt(x**2 + 1) - 1` | `x**2 / (np.sqrt(x**2 + 1) + 1)` |
| $\log(\sum e^{x_i})$ | `np.log(np.sum(np.exp(x)))` | `scipy.special.logsumexp(x)` |

### Softmax Numerical Stability

```python
# WRONG (overflow for large x)
def softmax_naive(x):
    return np.exp(x) / np.sum(np.exp(x))

# RIGHT (subtract max for stability)
def softmax_stable(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)
```

---

## 10. Summary

### Key Concepts

| Concept                   | Meaning                                     |
| ------------------------- | ------------------------------------------- |
| $\lim_{x \to a} f(x) = L$ | $f(x)$ approaches $L$ as $x$ approaches $a$ |
| Continuity                | No breaks, holes, or jumps                  |
| ε-δ definition            | Formal precision for limits                 |
| L'Hôpital                 | Tool for 0/0 or ∞/∞ forms                   |
| Squeeze Theorem           | Bound function between known limits         |

### Limit Evaluation Checklist

```
1. Try direct substitution
2. If 0/0: factor, rationalize, or L'Hôpital
3. If ∞/∞: divide by highest power or L'Hôpital
4. Check one-sided limits if needed
5. Use squeeze theorem for bounded oscillations
6. For 0·∞ or ∞-∞: algebraic manipulation first
```

### Indeterminate Forms Reference

| Form | Example | Strategy |
|------|---------|----------|
| $\frac{0}{0}$ | $\frac{x^2-1}{x-1}$ at $x=1$ | Factor, L'Hôpital |
| $\frac{\infty}{\infty}$ | $\frac{x^2}{e^x}$ as $x \to \infty$ | Divide, L'Hôpital |
| $0 \cdot \infty$ | $x \ln x$ as $x \to 0^+$ | Rewrite as fraction |
| $\infty - \infty$ | $\frac{1}{x} - \frac{1}{\sin x}$ | Common denominator |
| $1^\infty$ | $(1 + 1/n)^n$ | Take log, use L'Hôpital |
| $0^0$ | $x^x$ as $x \to 0^+$ | Take log |
| $\infty^0$ | $x^{1/x}$ as $x \to \infty$ | Take log |

### ML Connections

```
Limits in ML:
│
├── Convergence analysis
│   ├── Does training converge?
│   ├── Learning rate decay conditions
│   └── SGD convergence guarantees
│
├── Asymptotic behavior
│   ├── Algorithm complexity O(n)
│   ├── Model capacity scaling
│   └── Generalization bounds
│
├── Activation functions
│   ├── Saturation behavior (sigmoid, tanh)
│   ├── Vanishing gradients
│   └── Temperature in softmax
│
├── Loss functions
│   ├── Cross-entropy at extreme predictions
│   ├── Regularization as constraint
│   └── Huber loss smoothness
│
└── Numerical stability
    ├── Avoiding 0/0 and log(0)
    ├── Numerically stable implementations
    └── Gradient clipping rationale
```

### Quick Reference: Python Numerics

```python
import numpy as np
from scipy import special

# Stable functions for common limit-related computations
np.expm1(x)           # e^x - 1, accurate for small x
np.log1p(x)           # log(1 + x), accurate for small x
special.logsumexp(x)  # log(sum(exp(x))), stable softmax denominator
np.clip(p, eps, 1-eps) # Avoid log(0) in cross-entropy

# Numerical derivatives (use for gradient checking)
def numerical_grad(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)
```

---

## Exercises

1. Evaluate $\lim_{x \to 0} \frac{\sin(3x)}{x}$
2. Find $\lim_{x \to \infty} \frac{2x^2 + 3x}{x^2 - 1}$
3. Determine where $f(x) = \frac{x^2 - 1}{x - 1}$ is discontinuous
4. Use L'Hôpital to find $\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$
5. Show that ReLU is continuous everywhere
6. Find $k$ such that $f(x) = \{x^2 + k \text{ if } x < 2; 3x \text{ if } x \geq 2\}$ is continuous
7. Use Squeeze Theorem to evaluate $\lim_{x \to \infty} \frac{\sin x}{x}$
8. Analyze $\lim_{T \to 0^+} \text{softmax}(z/T)$ and interpret for ML

---

## References

1. Stewart - "Calculus: Early Transcendentals"
2. Rudin - "Principles of Mathematical Analysis"
3. Boyd & Vandenberghe - "Convex Optimization" (convergence analysis)
4. Goodfellow, Bengio, Courville - "Deep Learning" (numerical stability)
