# Multivariate Chain Rule and Backpropagation

> **Navigation**: [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/) | [02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/) | [04-Optimization-Theory](../04-Optimization-Theory/)

## Overview

The multivariate chain rule is the **mathematical foundation of backpropagation**, the algorithm that enables training of deep neural networks. Understanding how derivatives flow through composed functions is essential for both implementing and debugging neural networks.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    BACKPROP = CHAIN RULE                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Forward Pass:                                                          │
│                                                                          │
│     x ──→ [Layer 1] ──→ a₁ ──→ [Layer 2] ──→ a₂ ──→ [Loss] ──→ L       │
│             W₁               W₂                                         │
│                                                                          │
│  Backward Pass (Chain Rule):                                            │
│                                                                          │
│    ∂L    ∂L   ∂a₂    ∂L   ∂a₂   ∂a₁                                     │
│   ─── = ─── · ─── = ─── · ─── · ───                                     │
│   ∂W₁   ∂a₂  ∂W₁    ∂a₂  ∂a₁   ∂W₁                                     │
│                                                                          │
│     ∂L/∂W₁ ←── multiply ←── multiply ←── ∂L/∂L = 1                      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- [01-Partial-Derivatives-and-Gradients](../01-Partial-Derivatives-and-Gradients/)
- [02-Jacobians-and-Hessians](../02-Jacobians-and-Hessians/)
- Basic neural network concepts

## Learning Objectives

1. Master the multivariate chain rule
2. Understand backpropagation as chain rule application
3. Implement gradient computation for neural networks
4. Debug gradient computations

---

## 1. Single-Variable Chain Rule Review

For $y = f(g(x))$:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

where $u = g(x)$.

> **💡 Intuition**: Rate of change of $y$ with $x$ = (rate of $y$ with $u$) × (rate of $u$ with $x$)

---

## 2. Multivariate Chain Rule

### Case 1: Scalar Function of Vector

If $f: \mathbb{R}^n \to \mathbb{R}$ and $\mathbf{x}(t): \mathbb{R} \to \mathbb{R}^n$:

$$\frac{df}{dt} = \sum_{i=1}^n \frac{\partial f}{\partial x_i} \frac{dx_i}{dt} = \nabla f \cdot \frac{d\mathbf{x}}{dt}$$

### Case 2: Vector Function of Vector (Most Important!)

If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ and $\mathbf{g}: \mathbb{R}^p \to \mathbb{R}^n$:

$$\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}} \cdot \mathbf{J}_{\mathbf{g}}$$

> **🔑 THE KEY INSIGHT: Chain rule is matrix multiplication!**

```
Composition: x → g(x) → f(g(x))
             ℝᵖ → ℝⁿ  →   ℝᵐ

Jacobians:   J_g: (n×p)   J_f: (m×n)

J_{f∘g} = J_f × J_g : (m×p)

Dimensions work out like matrix multiplication!
```

### Case 3: Scalar Loss of Vector (The ML Case!)

For $L(\mathbf{y}(\mathbf{x}))$ where $\mathbf{y}: \mathbb{R}^n \to \mathbb{R}^m$:

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^m \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

In matrix/vector form:
$$\nabla_{\mathbf{x}} L = \mathbf{J}_{\mathbf{y}}^T \nabla_{\mathbf{y}} L$$

> **💡 This is the backprop equation!** Gradient with respect to earlier layer = Jacobian-transpose times gradient from later layer.

---

## 3. Computational Graph Perspective

### Nodes and Edges

Every computation can be represented as a directed acyclic graph (DAG):

```
Computational Graph Example:

     x₁ ─────┐
              ├──→ [×] ───→ z₁ ──→ [σ] ──→ a₁ ─┐
     x₂ ─────┘                                  │
                                                ├──→ [+] ──→ y ──→ [Loss] → L
     w  ────────→ [×] ───→ z₂ ──→ [σ] ──→ a₂ ─┘
              ╱
     b  ─────┘

Nodes: operations (×, +, σ, Loss)
Edges: data flow (tensors)
```

### Forward Pass

Compute values from inputs to outputs:
- Process nodes in topological order
- Store intermediate values (for backward pass)

### Backward Pass

Compute gradients from outputs to inputs:
- Process nodes in reverse topological order
- Apply chain rule at each node

```
Forward vs Backward:

FORWARD (left to right):
─────────────────────────────────────────→
x → f₁ → a₁ → f₂ → a₂ → f₃ → L

BACKWARD (right to left):
←─────────────────────────────────────────
∂L/∂x ← ∂L/∂a₁ ← ∂L/∂a₂ ← ∂L/∂L = 1
```

---

## 4. Backpropagation Derivation

### Simple Neural Network

```
Input  →  Linear   → Activation →  Linear   →  Loss
  x    →  z₁=Wx+b  →   a=σ(z₁)  →  y=Va+c   →  L(y,t)
```

### Forward Pass

1. $z_1 = Wx + b_1$
2. $a = \sigma(z_1)$
3. $y = Va + b_2$
4. $L = \text{loss}(y, \text{target})$

### Backward Pass (Chain Rule!)

Starting from $\frac{\partial L}{\partial L} = 1$:

```
Step 1: ∂L/∂y = loss_gradient(y, target)

Step 2: ∂L/∂a = Vᵀ × (∂L/∂y)           ← Linear layer backward

Step 3: ∂L/∂z₁ = (∂L/∂a) ⊙ σ'(z₁)      ← Activation backward

Step 4: ∂L/∂W = (∂L/∂z₁) × xᵀ          ← Parameter gradient
        ∂L/∂x = Wᵀ × (∂L/∂z₁)          ← Input gradient
```

---

## 5. Key Backprop Equations

### Linear Layer: $z = Wx + b$

| Gradient | Formula |
|----------|---------|
| $\frac{\partial L}{\partial x}$ | $W^T \frac{\partial L}{\partial z}$ |
| $\frac{\partial L}{\partial W}$ | $\frac{\partial L}{\partial z} \cdot x^T$ |
| $\frac{\partial L}{\partial b}$ | $\frac{\partial L}{\partial z}$ |

```
Linear Layer Gradients:

Forward:  z = Wx + b

               ┌───────────────────────────┐
               │    ∂L/∂z (from above)     │
               └───────────┬───────────────┘
                           │
         ┌─────────────────┼────────────────┐
         │                 │                │
         ▼                 ▼                ▼
    ∂L/∂x = Wᵀ(∂L/∂z)   ∂L/∂W = (∂L/∂z)xᵀ   ∂L/∂b = ∂L/∂z
```

### Activation: $a = \sigma(z)$

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \odot \sigma'(z)$$

($\odot$ is element-wise multiplication)

> **⚠️ Common Mistake**: This is element-wise multiplication, NOT matrix multiplication!

### Common Activations

| Activation | $\sigma(z)$ | $\sigma'(z)$ |
|------------|-------------|--------------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$ |
| Tanh | $\tanh(z)$ | $1 - \tanh^2(z)$ |
| ReLU | $\max(0, z)$ | $\mathbf{1}_{z > 0}$ |
| Leaky ReLU | $\max(\alpha z, z)$ | $\alpha$ if $z < 0$, else $1$ |
| Softplus | $\ln(1+e^z)$ | $\sigma(z)$ |

```
Activation Gradients:

ReLU:                    Sigmoid:
─────                    ────────

σ(z) = max(0, z)        σ(z) = 1/(1+e^{-z})

    │    ╱               │      ─────────
    │   ╱                │    ╱
    │  ╱                 │  ╱
────┼─●────              ├─/───────────
    │                    │╱
    0                    0

σ'(z) = {1 if z>0       σ'(z) = σ(z)(1-σ(z))
         {0 if z<0              max at z=0
```

### Loss Functions

| Loss | $L(y, \hat{y})$ | $\frac{\partial L}{\partial \hat{y}}$ |
|------|-----------------|--------------------------------------|
| MSE | $\frac{1}{n}\|\hat{y} - y\|^2$ | $\frac{2}{n}(\hat{y} - y)$ |
| Cross-entropy | $-y\log\hat{y}$ | $-\frac{y}{\hat{y}}$ |
| Softmax + CE | $-\log p_k$ | $p - y$ (one-hot) |

> **💡 Beautiful Result**: Softmax + Cross-entropy has the simple gradient $p - y$ (predictions minus targets)!

---

## 6. Backprop Through Common Operations

### Matrix Multiplication: $Y = XW$

```
X: (batch, in)   W: (in, out)   Y: (batch, out)

Forward:  Y = X @ W

Backward: ∂L/∂X = (∂L/∂Y) @ Wᵀ     ← shape: (batch, in)
          ∂L/∂W = Xᵀ @ (∂L/∂Y)     ← shape: (in, out)
```

### Element-wise Operations: $y = f(x)$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot f'(x)$$

### Sum/Reduce: $y = \sum_i x_i$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y}$$

The gradient "broadcasts" back!

### Broadcast: $y_i = x$ (scalar to vector)

$$\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial y_i}$$

The gradients "sum" back!

```
Broadcast and Sum:

Forward (broadcast):        Backward (sum):
     x                          ∂L/∂x
     │                            ▲
     ├───→ y₁                     │
     ├───→ y₂                ═════╪═════
     └───→ y₃                     │
                              ∂L/∂y₁ + ∂L/∂y₂ + ∂L/∂y₃

Forward (sum):              Backward (broadcast):
  x₁ ─┐                       ∂L/∂x₁ ←── ∂L/∂y
  x₂ ─┼→ y = Σxᵢ              ∂L/∂x₂ ←── ∂L/∂y
  x₃ ─┘                       ∂L/∂x₃ ←── ∂L/∂y
```

---

## 7. Implementing Backpropagation

### General Recipe

For each operation $y = f(x; \theta)$:

1. **Forward**: Compute output, store inputs for backward
2. **Backward**: Compute gradients given $\frac{\partial L}{\partial y}$

```python
class Layer:
    def forward(self, x):
        self.cache = x  # Store for backward
        return f(x, self.params)

    def backward(self, grad_output):
        x = self.cache
        grad_input = compute_grad_x(grad_output, x, self.params)
        grad_params = compute_grad_params(grad_output, x)
        return grad_input, grad_params
```

### Full Network Backprop

```python
# Forward pass - store intermediates
z1 = linear1.forward(x)
a1 = relu.forward(z1)
z2 = linear2.forward(a1)
loss = loss_fn.forward(z2, y)

# Backward pass - chain rule!
grad_z2 = loss_fn.backward()      # ∂L/∂z2
grad_a1 = linear2.backward(grad_z2)  # ∂L/∂a1
grad_z1 = relu.backward(grad_a1)     # ∂L/∂z1
grad_x = linear1.backward(grad_z1)   # ∂L/∂x
```

---

## 8. Gradient Checking

### Numerical Gradient

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}$$

### Relative Error

$$\text{rel\_error} = \frac{\|\nabla_{\text{analytic}} - \nabla_{\text{numerical}}\|}{\|\nabla_{\text{analytic}}\| + \|\nabla_{\text{numerical}}\|}$$

| Relative Error | Interpretation |
|---------------|----------------|
| $< 10^{-7}$ | Excellent |
| $< 10^{-5}$ | Good |
| $< 10^{-3}$ | Suspicious |
| $> 10^{-3}$ | Bug likely! |

```
Gradient Checking:

┌─────────────────────┐    ┌─────────────────────┐
│    ANALYTICAL       │    │    NUMERICAL        │
│                     │    │                     │
│  Backprop gradient  │    │   f(θ+ε) - f(θ-ε)   │
│  (fast, exact)      │    │   ───────────────   │
│                     │    │         2ε          │
└──────────┬──────────┘    └──────────┬──────────┘
           │                          │
           └──────────┬───────────────┘
                      │
               Compare: should be
               nearly identical!
```

---

## 9. Common Pitfalls

### 1. Forgetting Transposes

```
WRONG: ∂L/∂x = W @ ∂L/∂z
RIGHT: ∂L/∂x = Wᵀ @ ∂L/∂z
            ↑
        Transpose!
```

### 2. Wrong Multiplication Type

```
WRONG: ∂L/∂z = ∂L/∂a @ σ'(z)   ← matrix multiply
RIGHT: ∂L/∂z = ∂L/∂a ⊙ σ'(z)   ← element-wise!
```

### 3. Broadcasting Issues

When reducing dimensions, **sum** the gradients:

```python
# Forward: y = x + b  (b broadcast)
# Backward:
grad_x = grad_y          # Same shape
grad_b = grad_y.sum(0)   # Sum over batch!
```

### 4. In-place Operations

In-place modifications can corrupt stored values needed for backward!

```python
# WRONG: modifies x in place
x += 1  

# RIGHT: creates new tensor
x = x + 1
```

---

## 10. Automatic Differentiation

### Forward Mode (JVP)

Compute $\mathbf{J}\mathbf{v}$ alongside forward pass.

### Reverse Mode (VJP) = Backprop

Compute $\mathbf{J}^T\mathbf{v}$ in backward pass.

```
When to use which:

Forward Mode: Efficient when output dim >> input dim
              (e.g., computing full Jacobian)

Reverse Mode: Efficient when output dim << input dim
              (neural nets: 1 loss, millions of params!)

For loss (scalar output): Reverse mode needs just ONE
backward pass regardless of number of parameters!
```

---

## 11. Summary

### Chain Rule Forms

| Form | Formula | When to Use |
|------|---------|-------------|
| Scalar | $\frac{dL}{dx} = \frac{dL}{dy} \frac{dy}{dx}$ | Single variables |
| Vector | $\nabla_x L = J_y^T \nabla_y L$ | Layer gradients |
| Matrix | $J_{f \circ g} = J_f J_g$ | Full Jacobians |

### Backprop Rules Cheat Sheet

```
┌────────────────────────────────────────────────────────────────┐
│                    BACKPROP RULES                              │
├───────────────────────┬────────────────────────────────────────┤
│  Operation            │  Backward Rule                         │
├───────────────────────┼────────────────────────────────────────┤
│  z = Wx + b           │  ∂L/∂x = Wᵀ(∂L/∂z)                    │
│                       │  ∂L/∂W = (∂L/∂z)xᵀ                    │
│                       │  ∂L/∂b = ∂L/∂z                        │
├───────────────────────┼────────────────────────────────────────┤
│  a = σ(z)             │  ∂L/∂z = (∂L/∂a) ⊙ σ'(z)              │
├───────────────────────┼────────────────────────────────────────┤
│  y = x₁ + x₂          │  ∂L/∂x₁ = ∂L/∂y                       │
│                       │  ∂L/∂x₂ = ∂L/∂y                       │
├───────────────────────┼────────────────────────────────────────┤
│  y = x₁ ⊙ x₂          │  ∂L/∂x₁ = (∂L/∂y) ⊙ x₂                │
│                       │  ∂L/∂x₂ = (∂L/∂y) ⊙ x₁                │
├───────────────────────┼────────────────────────────────────────┤
│  y = Σᵢ xᵢ            │  ∂L/∂xᵢ = ∂L/∂y  (broadcast)          │
└───────────────────────┴────────────────────────────────────────┘
```

### Key Insights

```
Backpropagation Summary:
│
├── Forward Pass
│   └── Compute outputs, store intermediates
│
├── Backward Pass
│   └── Apply chain rule in reverse order
│
├── Gradient Flow
│   └── ∂L/∂earlier = Jᵀ @ ∂L/∂later
│
└── Efficiency
    └── Reverse mode: ONE backward pass for any # params
```

---

## Exercises

1. Derive backprop for a 2-layer MLP with ReLU activation
2. Implement gradient checking for a simple network
3. Compute $\frac{\partial L}{\partial W_1}$ for network with softmax output
4. Show that batch normalization gradient involves centered inputs
5. Derive backward pass for attention mechanism: $\text{Attention}(Q, K, V)$

---

## References

1. Goodfellow et al. - "Deep Learning" (Chapter 6)
2. Rumelhart, Hinton, Williams - "Learning representations by back-propagating errors"
3. Baydin et al. - "Automatic Differentiation in Machine Learning: a Survey"

---

> **Next**: [04-Optimization-Theory](../04-Optimization-Theory/) — Constrained optimization and KKT conditions
