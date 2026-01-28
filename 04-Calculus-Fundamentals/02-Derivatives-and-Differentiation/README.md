# Derivatives and Differentiation

## Introduction

Derivatives measure how functions change — the instantaneous rate of change. In ML, derivatives are the foundation of backpropagation and gradient-based optimization. Understanding derivatives deeply is essential for understanding how neural networks learn.

## Prerequisites

- Limits and continuity
- Function basics
- Algebraic manipulation

## Learning Objectives

1. Understand derivatives geometrically and analytically
2. Master differentiation rules
3. Compute derivatives of common functions
4. Apply derivatives to optimization
5. Connect to ML gradient computations

---

## 1. Definition of the Derivative

### Geometric Interpretation

The derivative $f'(a)$ is the **slope of the tangent line** to $f$ at $x = a$.

```
f(x)
  │         ╱ tangent line (slope = f'(a))
  │       ╱
  │     ●╱
  │    /╱
  │   /
  │  /
  │ /
  └──────────────→ x
       a
```

### Formal Definition

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}$$

Alternative notation:
$$\frac{df}{dx} = \frac{d}{dx}f(x) = f'(x) = Df(x)$$

### Computing from Definition

**Example**: $f(x) = x^2$

$$f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h} = \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h} = \lim_{h \to 0} (2x + h) = 2x$$

---

## 2. Basic Differentiation Rules

### Power Rule

$$\frac{d}{dx}x^n = nx^{n-1}$$

Works for any real $n$.

### Constant Multiple Rule

$$\frac{d}{dx}[cf(x)] = c \cdot f'(x)$$

### Sum/Difference Rule

$$\frac{d}{dx}[f(x) \pm g(x)] = f'(x) \pm g'(x)$$

### Product Rule

$$\frac{d}{dx}[f(x) \cdot g(x)] = f'(x)g(x) + f(x)g'(x)$$

### Quotient Rule

$$\frac{d}{dx}\left[\frac{f(x)}{g(x)}\right] = \frac{f'(x)g(x) - f(x)g'(x)}{[g(x)]^2}$$

### Chain Rule

$$\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$$

Or using Leibniz notation:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

---

## 3. Derivatives of Common Functions

| Function       | Derivative    |
| -------------- | ------------- |
| $c$ (constant) | $0$           |
| $x^n$          | $nx^{n-1}$    |
| $e^x$          | $e^x$         |
| $a^x$          | $a^x \ln a$   |
| $\ln x$        | $1/x$         |
| $\log_a x$     | $1/(x \ln a)$ |
| $\sin x$       | $\cos x$      |
| $\cos x$       | $-\sin x$     |
| $\tan x$       | $\sec^2 x$    |

---

## 4. ML Activation Functions

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

### ReLU

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined} & x = 0 \end{cases}$$

### Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$$

### Softplus

$$\text{softplus}(x) = \ln(1 + e^x)$$

$$\text{softplus}'(x) = \frac{e^x}{1 + e^x} = \sigma(x)$$

```
Activation Functions and Their Derivatives:

    Sigmoid              ReLU                Tanh
    ___________         _________           ________
   /                   /                  /
──/────────────  ─────/            ───────/
                                          \________

σ'(x) peaks at 0.25   ReLU'(x) = 0 or 1   tanh'(x) peaks at 1
```

---

## 5. Higher-Order Derivatives

### Second Derivative

$$f''(x) = \frac{d^2f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right)$$

**Interpretation**: Rate of change of the slope (curvature)

- $f''(x) > 0$: Concave up (curves like ∪)
- $f''(x) < 0$: Concave down (curves like ∩)

### nth Derivative

$$f^{(n)}(x) = \frac{d^nf}{dx^n}$$

---

## 6. Chain Rule Deep Dive

The chain rule is **the foundation of backpropagation**.

### Composite Functions

For $y = f(g(x))$:
$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

### Multiple Compositions

For $y = f(g(h(x)))$:
$$\frac{dy}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

### Example: Neural Network Layer

For $y = \sigma(wx + b)$:

$$\frac{\partial y}{\partial w} = \sigma'(wx + b) \cdot x$$
$$\frac{\partial y}{\partial b} = \sigma'(wx + b)$$
$$\frac{\partial y}{\partial x} = \sigma'(wx + b) \cdot w$$

---

## 7. Implicit Differentiation

When $y$ is defined implicitly by $F(x, y) = 0$:

$$\frac{dy}{dx} = -\frac{\partial F/\partial x}{\partial F/\partial y}$$

### Example

For $x^2 + y^2 = 1$ (circle):

Differentiate both sides:
$$2x + 2y\frac{dy}{dx} = 0$$
$$\frac{dy}{dx} = -\frac{x}{y}$$

---

## 8. Applications to Optimization

### Critical Points

$f'(c) = 0$ or $f'(c)$ undefined → $c$ is a critical point

### First Derivative Test

| Sign of $f'$             | Behavior          |
| ------------------------ | ----------------- |
| $f' > 0$                 | $f$ is increasing |
| $f' < 0$                 | $f$ is decreasing |
| $f'$ changes from + to - | Local maximum     |
| $f'$ changes from - to + | Local minimum     |

### Second Derivative Test

At critical point $c$ where $f'(c) = 0$:

| $f''(c)$     | Classification    |
| ------------ | ----------------- |
| $f''(c) > 0$ | Local minimum     |
| $f''(c) < 0$ | Local maximum     |
| $f''(c) = 0$ | Test inconclusive |

---

## 9. Gradient Descent Connection

### Single Variable

Update rule:
$$x_{t+1} = x_t - \alpha f'(x_t)$$

```
f(x)
  │\
  │ \
  │  \       ╱
  │   \     ╱
  │    \___╱
  │     ↓
  │   minimum
  └─────────────→ x

Move opposite to gradient direction
```

### Why It Works

- If $f'(x) > 0$: function increasing, move left (decrease $x$)
- If $f'(x) < 0$: function decreasing, move right (increase $x$)
- Step size proportional to steepness

---

## 10. Numerical Differentiation

### Forward Difference

$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$

Error: $O(h)$

### Central Difference

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

Error: $O(h^2)$ — more accurate!

### Second Derivative

$$f''(x) \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$$

---

## 11. Summary

### Key Rules

| Rule     | Formula                                |
| -------- | -------------------------------------- |
| Power    | $(x^n)' = nx^{n-1}$                    |
| Chain    | $(f \circ g)' = (f' \circ g) \cdot g'$ |
| Product  | $(fg)' = f'g + fg'$                    |
| Quotient | $(f/g)' = (f'g - fg')/g^2$             |

### ML Activations

| Function         | Derivative           |
| ---------------- | -------------------- |
| Sigmoid $\sigma$ | $\sigma(1-\sigma)$   |
| Tanh             | $1 - \tanh^2$        |
| ReLU             | $\mathbf{1}_{x > 0}$ |
| Softplus         | $\sigma$             |

### Optimization

```
Finding minimum of f(x):

1. Find f'(x)
2. Solve f'(x) = 0 for critical points
3. Use second derivative test:
   - f''(c) > 0 → minimum
   - f''(c) < 0 → maximum
4. Or use gradient descent:
   x_{t+1} = x_t - α·f'(x_t)
```

---

## Exercises

1. Compute $\frac{d}{dx}(3x^4 - 2x^2 + 5x - 1)$
2. Find the derivative of $f(x) = e^{x^2}$
3. Derive $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
4. Find critical points of $f(x) = x^3 - 3x + 1$
5. Use gradient descent to minimize $f(x) = (x - 2)^2$

---

## References

1. Stewart - "Calculus: Early Transcendentals"
2. Goodfellow et al. - "Deep Learning" (Backpropagation)
3. Boyd & Vandenberghe - "Convex Optimization"
