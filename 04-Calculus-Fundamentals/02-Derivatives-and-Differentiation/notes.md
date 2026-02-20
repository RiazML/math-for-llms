# Derivatives and Differentiation

## Introduction

Derivatives measure how functions change — the instantaneous rate of change. In machine learning, derivatives are the **foundation of backpropagation** and gradient-based optimization. Every parameter update in neural network training relies on computing derivatives of the loss function with respect to model parameters.

When you train a neural network:
- **Forward pass**: Compute predictions using current weights
- **Loss computation**: Measure how wrong the predictions are
- **Backward pass**: Compute derivatives of loss w.r.t. each parameter
- **Update**: Adjust parameters in the direction that reduces loss

This entire process hinges on understanding and computing derivatives correctly.

## Prerequisites

- Limits and continuity
- Function basics (domain, range, composition)
- Algebraic manipulation
- Basic exponential and logarithmic functions

## Learning Objectives

By the end of this section, you will be able to:

1. **Geometric Understanding**: Visualize derivatives as tangent line slopes
2. **Master Differentiation Rules**: Apply power, product, quotient, and chain rules fluently
3. **Activation Functions**: Compute and interpret derivatives of sigmoid, ReLU, tanh, and variants
4. **Higher-Order Derivatives**: Understand concavity and inflection points
5. **Optimization Connection**: Use derivatives for finding extrema and gradient descent
6. **Numerical Methods**: Implement and compare finite difference approximations

## Key Takeaways for ML

| Concept | ML Application |
|---------|----------------|
| Chain Rule | Backpropagation algorithm |
| Derivative of σ(x) | Vanishing gradient problem |
| ReLU derivative | Sparse gradients, dying ReLU |
| Critical points | Loss function optimization |
| Numerical differentiation | Gradient checking |

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

Understanding activation function derivatives is crucial for understanding gradient flow in neural networks.

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x)(1 - \sigma(x))$$

**Derivation**:
Let $u = 1 + e^{-x}$, so $\sigma = u^{-1}$
$$\sigma' = -u^{-2} \cdot (-e^{-x}) = \frac{e^{-x}}{(1+e^{-x})^2} = \sigma(1-\sigma)$$

**Properties**:
- Maximum gradient at x=0: $\sigma'(0) = 0.25$
- Saturates for |x| > 4: gradients nearly zero
- Output range: (0, 1)

### Tanh

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

$$\tanh'(x) = 1 - \tanh^2(x) = \text{sech}^2(x)$$

**Properties**:
- Maximum gradient at x=0: $\tanh'(0) = 1$
- Zero-centered output: (-1, 1)
- Still saturates, but better gradients than sigmoid

### ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined} & x = 0 \end{cases}$$

**Properties**:
- Gradient is 0 or 1 (no saturation for positive inputs)
- Computationally efficient (no exponentials)
- **Dying ReLU problem**: If neuron outputs are always negative, gradient is always 0

### Leaky ReLU

$$\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}$$

$$\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$$

Typically $\alpha = 0.01$. Fixes dying ReLU by allowing small gradient when x < 0.

### Softplus (Smooth ReLU)

$$\text{softplus}(x) = \ln(1 + e^x)$$

$$\text{softplus}'(x) = \frac{e^x}{1 + e^x} = \sigma(x)$$

Smooth approximation to ReLU. Differentiable everywhere.

### Comprehensive Comparison

| Activation | Derivative | Max Gradient | Vanishing? | Zero-Centered |
|------------|------------|--------------|------------|---------------|
| Sigmoid | $\sigma(1-\sigma)$ | 0.25 | Yes, severe | No |
| Tanh | $1-\tanh^2$ | 1.0 | Yes | Yes |
| ReLU | 0 or 1 | 1.0 | No (if x>0) | No |
| Leaky ReLU | $\alpha$ or 1 | 1.0 | No | No |
| Softplus | $\sigma(x)$ | 1.0 | No | No |
| GELU | complex | ~1.0 | No | Nearly |
| Swish | complex | ~1.0 | No | Nearly |

```
Gradient Saturation Comparison:

Sigmoid σ'(x):          Tanh'(x):              ReLU'(x):
   max=0.25               max=1.0                max=1.0
     ∧                      ∧                       ___
    / \                    / \                     |
___/   \___            ___/   \___           _____|

  Saturates              Saturates             No saturation
  at ±3                  at ±2                 for x > 0
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

The chain rule is **the foundation of backpropagation** — it's how gradients flow backward through neural networks.

### Composite Functions

For $y = f(g(x))$:
$$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

**Intuition**: If $g$ changes by a small amount, how much does $y$ change?
- First, $g$'s change affects the input to $f$
- Then, $f$ responds to that changed input
- The total effect is the product of these sensitivities

### Multiple Compositions

For $y = f(g(h(x)))$:
$$\frac{dy}{dx} = f'(g(h(x))) \cdot g'(h(x)) \cdot h'(x)$$

This is exactly what happens in a neural network with multiple layers!

### Example: Neural Network Layer

For a single neuron with activation: $y = \sigma(wx + b)$

```
Input    Weights    Linear      Activation   Output
  x   →   wx + b   →    z    →    σ(z)    →    y
```

Using chain rule:
$$\frac{\partial y}{\partial w} = \underbrace{\sigma'(z)}_{\text{activation gradient}} \cdot \underbrace{x}_{\text{input}}$$
$$\frac{\partial y}{\partial b} = \sigma'(z) \cdot 1$$
$$\frac{\partial y}{\partial x} = \sigma'(z) \cdot w$$

### Backpropagation Perspective

In a multi-layer network, the chain rule chains through all layers:

```
Layer 1         Layer 2         Layer 3         Loss
  z₁ → σ(z₁) →   z₂ → σ(z₂) →   z₃ → σ(z₃) →   L

∂L/∂z₁ = (∂L/∂z₃) · (∂z₃/∂z₂) · (∂z₂/∂z₁)
       ↑           ↑           ↑
    Gradients flow backward through the chain
```

**Vanishing Gradient Problem**: If each $\sigma'(z) < 1$, the product of many such terms approaches zero. This is why sigmoid/tanh networks have trouble learning deep representations.

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

Derivatives directly power the gradient descent algorithm, the workhorse of neural network training.

### Single Variable

Update rule:
$$x_{t+1} = x_t - \alpha f'(x_t)$$

Where $\alpha$ is the **learning rate** (step size).

```
f(x)
  │\
  │ \
  │  \       ╱
  │   \     ╱
  │    \___╱  ← minimum
  │     ↓
  │   move opposite to gradient
  └─────────────────────→ x

Move opposite to gradient direction:
• f'(x) > 0 → decrease x
• f'(x) < 0 → increase x
```

### Why It Works

The gradient points in the direction of **steepest ascent**. To minimize, we go in the opposite direction (steepest descent).

- If $f'(x) > 0$: function is increasing → move left (subtract)
- If $f'(x) < 0$: function is decreasing → move right (add negative)
- Step size proportional to gradient magnitude (steeper = bigger step)

### Learning Rate Effects

| Learning Rate | Behavior |
|---------------|----------|
| Too small | Slow convergence, may get stuck |
| Just right | Smooth convergence to minimum |
| Too large | Oscillation, may diverge |

### Connection to ML Training

In neural networks with parameters $\theta$:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(\theta_t)$$

Where:
- $\mathcal{L}$ is the loss function
- $\nabla_\theta \mathcal{L}$ is the gradient (vector of partial derivatives)
- The gradient is computed via backpropagation (chain rule applied recursively)

---

## 10. Numerical Differentiation

Numerical differentiation approximates derivatives using function evaluations. Essential for:
- **Gradient checking**: Verify analytical gradients during development
- **When analytical derivatives are unavailable**
- **Understanding approximation errors**

### Forward Difference

$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$

Error: $O(h)$ — first-order accurate

### Central Difference

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

Error: $O(h^2)$ — second-order accurate (much better!)

### Second Derivative

$$f''(x) \approx \frac{f(x + h) - 2f(x) + f(x - h)}{h^2}$$

### Gradient Checking in Practice

When implementing backpropagation, verify your analytical gradients:

```python
# Gradient checking pseudocode
h = 1e-7
for each parameter θ:
    θ_plus = θ + h
    θ_minus = θ - h
    numerical_grad = (L(θ_plus) - L(θ_minus)) / (2h)
    analytical_grad = backprop_gradient(θ)
    
    # Check relative error
    error = |numerical - analytical| / max(|numerical|, |analytical|)
    assert error < 1e-5, "Gradient mismatch!"
```

### Choosing Step Size h

| h Value | Issue |
|---------|-------|
| Too large (e.g., 0.1) | Truncation error (approximation too crude) |
| Too small (e.g., 1e-15) | Roundoff error (floating-point precision) |
| Just right (1e-5 to 1e-7) | Balance truncation and roundoff |

Optimal $h$ for central difference on 64-bit floats: approximately $h \approx \epsilon^{1/3} \approx 10^{-5}$ where $\epsilon \approx 10^{-16}$.

### Comparison of Methods

| Method | Formula | Error Order | Function Evals |
|--------|---------|-------------|----------------|
| Forward | $(f(x+h) - f(x))/h$ | $O(h)$ | 2 |
| Central | $(f(x+h) - f(x-h))/(2h)$ | $O(h^2)$ | 2 |
| Complex Step | $\text{Im}(f(x+ih))/h$ | $O(h^2)$ | 1 (complex) |

---

## 11. Summary

### Differentiation Rules Quick Reference

| Rule | Formula | Example |
|------|---------|---------|
| Power | $(x^n)' = nx^{n-1}$ | $(x^3)' = 3x^2$ |
| Constant | $(cf)' = cf'$ | $(5x^2)' = 10x$ |
| Sum | $(f+g)' = f' + g'$ | $(x^2+x)' = 2x+1$ |
| Product | $(fg)' = f'g + fg'$ | $(x \cdot e^x)' = e^x + xe^x$ |
| Quotient | $(f/g)' = (f'g - fg')/g^2$ | $(x/e^x)' = (e^x-xe^x)/e^{2x}$ |
| Chain | $(f \circ g)' = (f' \circ g) \cdot g'$ | $(e^{x^2})' = 2xe^{x^2}$ |

### Activation Function Derivatives

| Function | Derivative | Gradient Issue |
|----------|------------|----------------|
| Sigmoid $\sigma$ | $\sigma(1-\sigma)$ | Vanishes for |x| > 4 |
| Tanh | $1 - \tanh^2$ | Vanishes for |x| > 2 |
| ReLU | $\mathbf{1}_{x > 0}$ | Zero for x < 0 (dying) |
| Leaky ReLU | 1 or α | None |
| Softplus | $\sigma$ | None (smooth) |

### Optimization Decision Tree

```
Finding minimum of f(x):
─────────────────────────
        ┌──────────────────┐
        │ 1. Find f'(x)    │
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │ 2. Solve f'(x)=0 │
        │    for critical  │
        │    points        │
        └────────┬─────────┘
                 ↓
        ┌──────────────────┐
        │ 3. Classify with │
        │    f''(x)        │
        └────────┬─────────┘
                 ↓
  ┌──────────────┼──────────────┐
  ↓              ↓              ↓
f''>0         f''=0          f''<0
LOCAL MIN   INCONCLUSIVE    LOCAL MAX
```

### Python Quick Reference

```python
import numpy as np

# Numerical derivative (central difference)
def derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

# Sigmoid and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ReLU and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Gradient descent
def gradient_descent(f, df, x0, lr=0.01, n_iter=100):
    x = x0
    for _ in range(n_iter):
        x = x - lr * df(x)
    return x
```

---

## Exercises

See `exercises.ipynb` for comprehensive practice problems including:

1. Basic derivative computation
2. Chain rule applications
3. Product and quotient rules
4. Sigmoid derivative derivation
5. Implicit differentiation
6. Finding extrema
7. Gradient descent implementation
8. Concavity analysis
9. Leaky ReLU derivative
10. MSE loss gradient

---

## References

1. Stewart, J. - "Calculus: Early Transcendentals" (rigorous calculus foundation)
2. Goodfellow, I., Bengio, Y., Courville, A. - "Deep Learning" Chapter 6 (backpropagation)
3. Boyd, S., Vandenberghe, L. - "Convex Optimization" (optimization theory)
4. Ruder, S. - "An Overview of Gradient Descent Optimization Algorithms" (arXiv:1609.04747)
5. Nielsen, M. - "Neural Networks and Deep Learning" (online book, backprop derivation)
