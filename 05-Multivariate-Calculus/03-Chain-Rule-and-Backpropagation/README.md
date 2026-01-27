# Multivariate Chain Rule and Backpropagation

## Introduction

The multivariate chain rule is the mathematical foundation of backpropagation, the algorithm that enables training of deep neural networks. Understanding how derivatives flow through composed functions is essential for both implementing and debugging neural networks.

## Prerequisites

- Partial derivatives and gradients
- Jacobian matrices
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

---

## 2. Multivariate Chain Rule

### Case 1: Scalar Function of Vector

If $f: \mathbb{R}^n \to \mathbb{R}$ and $\mathbf{x}(t): \mathbb{R} \to \mathbb{R}^n$:

$$\frac{df}{dt} = \sum_{i=1}^n \frac{\partial f}{\partial x_i} \frac{dx_i}{dt} = \nabla f \cdot \frac{d\mathbf{x}}{dt}$$

### Case 2: Vector Function of Vector

If $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ and $\mathbf{g}: \mathbb{R}^p \to \mathbb{R}^n$:

$$\mathbf{J}_{\mathbf{f} \circ \mathbf{g}} = \mathbf{J}_{\mathbf{f}} \cdot \mathbf{J}_{\mathbf{g}}$$

**Chain rule is matrix multiplication!**

```
Composition: x → g(x) → f(g(x))
             ℝᵖ → ℝⁿ → ℝᵐ

Jacobians:   J_g: n×p    J_f: m×n
             J_{f∘g} = J_f × J_g : m×p
```

### Case 3: Scalar Loss of Vector

For $L(\mathbf{y}(\mathbf{x}))$ where $\mathbf{y}: \mathbb{R}^n \to \mathbb{R}^m$:

$$\frac{\partial L}{\partial x_i} = \sum_{j=1}^m \frac{\partial L}{\partial y_j} \frac{\partial y_j}{\partial x_i}$$

In matrix form:
$$\nabla_{\mathbf{x}} L = \mathbf{J}_{\mathbf{y}}^T \nabla_{\mathbf{y}} L$$

---

## 3. Computational Graph Perspective

### Nodes and Edges

```
Computational Graph:

     x₁ ─────┐
             ├──→ [*] ───→ z₁ ──→ [σ] ──→ a₁ ─┐
     x₂ ─────┘                                 ├──→ [L] → Loss
                                               │
     w ────────────────────────────────────────┘
```

### Forward Pass

Compute values from inputs to outputs.

### Backward Pass

Compute gradients from outputs to inputs using chain rule.

---

## 4. Backpropagation Derivation

### Simple Neural Network

```
Input → Linear → Activation → Linear → Loss
  x   →  z₁=Wx  →   a=σ(z₁)  →  z₂=Va  → L
```

### Forward Pass

1. $z_1 = Wx + b_1$
2. $a = \sigma(z_1)$
3. $z_2 = Va + b_2$
4. $L = \text{loss}(z_2, y)$

### Backward Pass

1. $\frac{\partial L}{\partial z_2}$ (from loss function)
2. $\frac{\partial L}{\partial a} = V^T \frac{\partial L}{\partial z_2}$
3. $\frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a} \odot \sigma'(z_1)$
4. $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial z_1} x^T$

---

## 5. Key Backprop Equations

### Linear Layer: $z = Wx + b$

| Gradient                        | Formula                             |
| ------------------------------- | ----------------------------------- |
| $\frac{\partial L}{\partial x}$ | $W^T \frac{\partial L}{\partial z}$ |
| $\frac{\partial L}{\partial W}$ | $\frac{\partial L}{\partial z} x^T$ |
| $\frac{\partial L}{\partial b}$ | $\frac{\partial L}{\partial z}$     |

### Activation: $a = \sigma(z)$

$$\frac{\partial L}{\partial z} = \frac{\partial L}{\partial a} \odot \sigma'(z)$$

($\odot$ is element-wise multiplication)

### Common Activations

| Activation | $\sigma(z)$          | $\sigma'(z)$         |
| ---------- | -------------------- | -------------------- |
| Sigmoid    | $\frac{1}{1+e^{-z}}$ | $\sigma(1-\sigma)$   |
| Tanh       | $\tanh(z)$           | $1 - \tanh^2(z)$     |
| ReLU       | $\max(0, z)$         | $\mathbf{1}_{z > 0}$ |
| Softplus   | $\ln(1+e^z)$         | $\sigma(z)$          |

### Loss Functions

| Loss          | $L(y, \hat{y})$                | $\frac{\partial L}{\partial \hat{y}}$ |
| ------------- | ------------------------------ | ------------------------------------- |
| MSE           | $\frac{1}{n}\|\hat{y} - y\|^2$ | $\frac{2}{n}(\hat{y} - y)$            |
| Cross-entropy | $-y\log\hat{y}$                | $-\frac{y}{\hat{y}}$                  |
| Softmax + CE  | $-\log p_k$                    | $p - y$                               |

---

## 6. Backprop Through Common Operations

### Matrix Multiplication: $Y = XW$

```
X: (batch, in)   W: (in, out)   Y: (batch, out)

∂L/∂X = (∂L/∂Y) W^T
∂L/∂W = X^T (∂L/∂Y)
```

### Element-wise Operations: $y = f(x)$

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \odot f'(x)$$

### Sum: $y = \sum_i x_i$

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial y}$$

### Broadcast: $y_i = x$ (scalar to vector)

$$\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial y_i}$$

---

## 7. Implementing Backpropagation

### General Recipe

For each operation $y = f(x; \theta)$:

1. **Forward**: Store inputs needed for backward
2. **Backward**: Compute gradients given $\frac{\partial L}{\partial y}$

```python
class Layer:
    def forward(self, x):
        self.cache = x  # Store for backward
        return f(x)

    def backward(self, grad_output):
        x = self.cache
        grad_input = compute_grad_x(grad_output, x)
        grad_params = compute_grad_params(grad_output, x)
        return grad_input, grad_params
```

### Full Network Backprop

```python
# Forward pass
z1 = linear1.forward(x)
a1 = relu.forward(z1)
z2 = linear2.forward(a1)
loss = loss_fn.forward(z2, y)

# Backward pass
grad_z2 = loss_fn.backward()
grad_a1 = linear2.backward(grad_z2)
grad_z1 = relu.backward(grad_a1)
grad_x = linear1.backward(grad_z1)
```

---

## 8. Gradient Checking

### Numerical Gradient

$$\frac{\partial L}{\partial \theta_i} \approx \frac{L(\theta_i + \epsilon) - L(\theta_i - \epsilon)}{2\epsilon}$$

### Relative Error

$$\text{rel\_error} = \frac{\|\nabla_{\text{analytic}} - \nabla_{\text{numerical}}\|}{\|\nabla_{\text{analytic}}\| + \|\nabla_{\text{numerical}}\|}$$

Should be $< 10^{-5}$ for correct implementation.

---

## 9. Common Pitfalls

### 1. Forgetting Transposes

```
WRONG: ∂L/∂x = W @ ∂L/∂z
RIGHT: ∂L/∂x = W^T @ ∂L/∂z
```

### 2. Missing Element-wise Multiplication

```
WRONG: ∂L/∂z = ∂L/∂a @ σ'(z)
RIGHT: ∂L/∂z = ∂L/∂a ⊙ σ'(z)
```

### 3. Broadcasting Issues

When reducing dimensions, sum the gradients.

### 4. In-place Operations

Modifying tensors in-place can corrupt gradient computation.

---

## 10. Automatic Differentiation

### Forward Mode (JVP)

Compute $\mathbf{J}\mathbf{v}$ alongside forward pass.

### Reverse Mode (VJP) = Backprop

Compute $\mathbf{J}^T\mathbf{v}$ in backward pass.

```
Forward Mode: Efficient when output dim > input dim
Reverse Mode: Efficient when output dim < input dim (neural nets!)
```

For loss function (scalar output), reverse mode needs **one** backward pass regardless of number of parameters.

---

## 11. Summary

### Chain Rule Forms

| Form   | Formula                                       |
| ------ | --------------------------------------------- |
| Scalar | $\frac{dL}{dx} = \frac{dL}{dy} \frac{dy}{dx}$ |
| Vector | $\nabla_x L = J_y^T \nabla_y L$               |
| Matrix | $J_{f \circ g} = J_f J_g$                     |

### Backprop Rules

```
Operation           │ Backward Rule
────────────────────┼──────────────────────────────
z = Wx + b         │ ∂L/∂x = Wᵀ(∂L/∂z)
                    │ ∂L/∂W = (∂L/∂z)xᵀ
────────────────────┼──────────────────────────────
a = σ(z)           │ ∂L/∂z = (∂L/∂a) ⊙ σ'(z)
────────────────────┼──────────────────────────────
y = x₁ + x₂        │ ∂L/∂x₁ = ∂L/∂y
                    │ ∂L/∂x₂ = ∂L/∂y
────────────────────┼──────────────────────────────
y = x₁ ⊙ x₂        │ ∂L/∂x₁ = (∂L/∂y) ⊙ x₂
                    │ ∂L/∂x₂ = (∂L/∂y) ⊙ x₁
────────────────────┼──────────────────────────────
y = Σᵢ xᵢ          │ ∂L/∂xᵢ = ∂L/∂y
```

### Key Insights

```
Backpropagation:
│
├── Forward Pass
│   └── Compute activations, store intermediates
│
├── Backward Pass
│   └── Apply chain rule in reverse order
│
├── Gradient Flow
│   └── ∂L/∂earlier = Jᵀ @ ∂L/∂later
│
└── Efficiency
    └── Reverse mode: O(1) backward passes for any # params
```

---

## Exercises

1. Derive backprop for a 2-layer MLP with ReLU activation
2. Implement gradient checking for a simple network
3. Compute $\frac{\partial L}{\partial W_1}$ for network with softmax output
4. Show that batch normalization gradient involves centered inputs
5. Derive backward pass for attention mechanism

---

## References

1. Goodfellow et al. - "Deep Learning" (Chapter 6)
2. Rumelhart, Hinton, Williams - "Learning representations by back-propagating errors"
3. Baydin et al. - "Automatic Differentiation in Machine Learning: a Survey"
