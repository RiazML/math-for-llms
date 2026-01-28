# Limits and Continuity

## Introduction

Limits are the foundation of calculus. They formalize the idea of "approaching" a value and are essential for defining derivatives and integrals. In ML, understanding limits helps with convergence analysis, asymptotic behavior, and numerical stability.

## Prerequisites

- Functions and graphs
- Basic algebra
- Inequalities

## Learning Objectives

1. Understand the concept of a limit
2. Evaluate limits using various techniques
3. Recognize continuous and discontinuous functions
4. Apply limits to ML contexts

---

## 1. Intuitive Definition of Limits

### The Idea

$$\lim_{x \to a} f(x) = L$$

means: as $x$ gets arbitrarily close to $a$, $f(x)$ gets arbitrarily close to $L$.

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

### Factoring (0/0 form)

$$\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = \lim_{x \to 2} \frac{(x-2)(x+2)}{x-2} = \lim_{x \to 2} (x + 2) = 4$$

### Rationalization

$$\lim_{x \to 0} \frac{\sqrt{x+1} - 1}{x} = \lim_{x \to 0} \frac{(\sqrt{x+1}-1)(\sqrt{x+1}+1)}{x(\sqrt{x+1}+1)} = \lim_{x \to 0} \frac{x}{x(\sqrt{x+1}+1)} = \frac{1}{2}$$

### L'Hôpital's Rule

For indeterminate forms $\frac{0}{0}$ or $\frac{\infty}{\infty}$:

$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

---

## 5. Important Limits

### Fundamental Limits

| Limit                                     | Value |
| ----------------------------------------- | ----- |
| $\lim_{x \to 0} \frac{\sin x}{x}$         | $1$   |
| $\lim_{x \to 0} \frac{1 - \cos x}{x}$     | $0$   |
| $\lim_{x \to 0} \frac{e^x - 1}{x}$        | $1$   |
| $\lim_{x \to 0} \frac{\ln(1+x)}{x}$       | $1$   |
| $\lim_{x \to \infty} (1 + \frac{1}{x})^x$ | $e$   |
| $\lim_{x \to 0} (1 + x)^{1/x}$            | $e$   |

### Limits at Infinity

$$\lim_{x \to \infty} \frac{P(x)}{Q(x)} = \begin{cases} 0 & \text{if } \deg P < \deg Q \\ \frac{a_n}{b_m} & \text{if } \deg P = \deg Q \\ \pm\infty & \text{if } \deg P > \deg Q \end{cases}$$

---

## 6. Continuity

### Definition

$f$ is **continuous at $a$** if:

1. $f(a)$ is defined
2. $\lim_{x \to a} f(x)$ exists
3. $\lim_{x \to a} f(x) = f(a)$

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

### Properties of Continuous Functions

| Property       | Statement                                 |
| -------------- | ----------------------------------------- |
| Sum/Difference | $f \pm g$ continuous                      |
| Product        | $f \cdot g$ continuous                    |
| Quotient       | $f/g$ continuous where $g \neq 0$         |
| Composition    | $f \circ g$ continuous if both continuous |

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

### 2. Softmax Limits

$$\lim_{T \to 0^+} \text{softmax}(z_i/T) = \begin{cases} 1 & \text{if } z_i = \max_j z_j \\ 0 & \text{otherwise} \end{cases}$$

(Temperature → 0 makes softmax "harder")

### 3. Sigmoid Saturation

$$\lim_{x \to \infty} \sigma(x) = 1, \quad \lim_{x \to -\infty} \sigma(x) = 0$$

### 4. Batch Normalization

As batch size $n \to \infty$:
$$\bar{x} \to \mu, \quad s^2 \to \sigma^2$$

### 5. Asymptotic Complexity

$$\lim_{n \to \infty} \frac{f(n)}{g(n)} = c \implies f(n) = \Theta(g(n))$$

---

## 9. Numerical Considerations

### Finite Differences Approach Derivatives

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

But in practice, we use small finite $h$ (not 0).

### Numerical Stability

$$\lim_{x \to 0} \frac{e^x - 1}{x} = 1$$

But computing directly for small $x$ causes cancellation errors!
Use: `np.expm1(x) / x` instead.

---

## 10. Summary

### Key Concepts

| Concept                   | Meaning                                     |
| ------------------------- | ------------------------------------------- |
| $\lim_{x \to a} f(x) = L$ | $f(x)$ approaches $L$ as $x$ approaches $a$ |
| Continuity                | No breaks, holes, or jumps                  |
| ε-δ definition            | Formal precision for limits                 |
| L'Hôpital                 | Tool for 0/0 or ∞/∞ forms                   |

### Limit Evaluation Checklist

```
1. Try direct substitution
2. If 0/0: factor, rationalize, or L'Hôpital
3. If ∞/∞: divide by highest power or L'Hôpital
4. Check one-sided limits if needed
5. Use squeeze theorem for bounded oscillations
```

### ML Connections

```
Limits in ML:
│
├── Convergence analysis
│   └── Does training converge?
│
├── Asymptotic behavior
│   └── What happens as n → ∞?
│
├── Activation functions
│   └── Saturation behavior
│
└── Numerical stability
    └── Avoiding 0/0 computations
```

---

## Exercises

1. Evaluate $\lim_{x \to 0} \frac{\sin(3x)}{x}$
2. Find $\lim_{x \to \infty} \frac{2x^2 + 3x}{x^2 - 1}$
3. Determine where $f(x) = \frac{x^2 - 1}{x - 1}$ is discontinuous
4. Use L'Hôpital to find $\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$
5. Show that ReLU is continuous everywhere

---

## References

1. Stewart - "Calculus: Early Transcendentals"
2. Rudin - "Principles of Mathematical Analysis"
3. Boyd & Vandenberghe - "Convex Optimization" (convergence)
