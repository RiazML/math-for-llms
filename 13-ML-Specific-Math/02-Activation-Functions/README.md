# Activation Functions for Neural Networks

## Overview

Activation functions introduce non-linearity into neural networks, enabling them to learn complex, non-linear mappings. Without activation functions, neural networks would only represent linear transformations regardless of depth.

## Mathematical Role

### Universal Approximation

A feed-forward network with one hidden layer and non-linear activation can approximate any continuous function on compact subsets of $\mathbb{R}^n$.

**Key requirement:** The activation must be non-polynomial (e.g., sigmoid, ReLU).

### Gradient Flow

For layer $l$ with activation $\sigma$:

$$a^{(l)} = \sigma(z^{(l)}) = \sigma(W^{(l)} a^{(l-1)} + b^{(l)})$$

Backpropagation gives:

$$\frac{\partial \mathcal{L}}{\partial z^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \odot \sigma'(z^{(l)})$$

The derivative $\sigma'$ critically affects gradient flow.

## Classic Activations

### Sigmoid

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Properties:**

- Range: $(0, 1)$ - interpretable as probability
- Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
- Maximum derivative: $\sigma'(0) = 0.25$
- Smooth, monotonic, bounded

**Issues:**

- **Vanishing gradients:** $\sigma'(x) \to 0$ for $|x| \gg 0$
- **Not zero-centered:** Outputs always positive
- **Saturating:** Gradients vanish for saturated neurons
- **Computationally expensive:** Exponential

### Hyperbolic Tangent (Tanh)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

**Properties:**

- Range: $(-1, 1)$ - zero-centered
- Derivative: $\tanh'(x) = 1 - \tanh^2(x)$
- Maximum derivative: $\tanh'(0) = 1$

**Advantages over sigmoid:**

- Zero-centered (better gradient dynamics)
- Larger gradients (max = 1 vs 0.25)

**Issues:**

- Still suffers from vanishing gradients
- Saturating for large inputs

## Modern Activations

### ReLU (Rectified Linear Unit)

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases}
x & x > 0 \\
0 & x \leq 0
\end{cases}
$$

**Properties:**

- Derivative: $\text{ReLU}'(x) = \mathbf{1}_{x > 0}$
- Unbounded for positive inputs
- Sparse activation (natural feature selection)
- Computationally efficient

**Advantages:**

- No vanishing gradient for $x > 0$
- Fast computation (comparison + multiplication)
- Promotes sparsity (≈50% neurons inactive)
- Converges faster than sigmoid/tanh

**Issues:**

- **Dying ReLU:** If $z < 0$ for all inputs, gradient is always 0
- Not zero-centered
- Unbounded (can cause instability)
- Non-differentiable at 0

### Leaky ReLU

$$
\text{LeakyReLU}(x) = \begin{cases}
x & x > 0 \\
\alpha x & x \leq 0
\end{cases}
$$

where $\alpha$ is a small positive constant (typically 0.01).

**Properties:**

- Derivative: $\alpha$ for $x < 0$, $1$ for $x > 0$
- No dead neurons
- Small gradient for negative inputs

**Variants:**

- **PReLU:** Learnable $\alpha$ per channel
- **RReLU:** Random $\alpha$ during training

### ELU (Exponential Linear Unit)

$$
\text{ELU}(x) = \begin{cases}
x & x > 0 \\
\alpha(e^x - 1) & x \leq 0
\end{cases}
$$

**Properties:**

- Derivative: $\alpha e^x$ for $x < 0$, $1$ for $x > 0$
- Smooth everywhere
- Negative values push mean activation towards zero
- Self-normalizing properties

**Advantages:**

- Zero-centered (approximately)
- No dead neurons
- Smooth (unlike ReLU variants)

### SELU (Scaled ELU)

$$
\text{SELU}(x) = \lambda \begin{cases}
x & x > 0 \\
\alpha(e^x - 1) & x \leq 0
\end{cases}
$$

where $\lambda \approx 1.0507$ and $\alpha \approx 1.6733$.

**Self-normalizing property:** With proper initialization, activations converge to mean 0 and variance 1 throughout the network.

### GELU (Gaussian Error Linear Unit)

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Approximation:**

$$\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{2/\pi}(x + 0.044715x^3)\right)\right]$$

**Properties:**

- Smooth, non-monotonic
- Combines properties of ReLU and dropout
- Default in transformers (BERT, GPT)

**Interpretation:** Expected value of stochastic regularization where input is randomly zeroed based on its magnitude.

### Swish / SiLU

$$\text{Swish}(x) = x \cdot \sigma(\beta x) = \frac{x}{1 + e^{-\beta x}}$$

**Properties:**

- Smooth, non-monotonic
- Unbounded above, bounded below
- Self-gated
- $\beta = 1$ commonly used (SiLU)

**Relation to ReLU:**

- $\beta \to \infty$: approaches ReLU
- $\beta = 0$: linear function $x/2$

### Mish

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

**Properties:**

- Smooth, non-monotonic
- Self-regularizing
- Similar to Swish but empirically better in some cases

### Softplus

$$\text{Softplus}(x) = \ln(1 + e^x)$$

**Properties:**

- Smooth approximation of ReLU
- Derivative is sigmoid: $\text{Softplus}'(x) = \sigma(x)$
- Always positive

## Output Layer Activations

### Softmax

For classification with $K$ classes:

$$\text{Softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$

**Properties:**

- Outputs sum to 1 (probability distribution)
- Derivative: $\frac{\partial \text{Softmax}_i}{\partial z_j} = \text{Softmax}_i(\delta_{ij} - \text{Softmax}_j)$
- With cross-entropy loss: simple gradient $\hat{p} - y$

**Numerical stability:**

$$\text{Softmax}(z)_i = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$$

**Temperature scaling:**

$$\text{Softmax}(z/\tau)_i = \frac{e^{z_i/\tau}}{\sum_j e^{z_j/\tau}}$$

- $\tau \to 0$: one-hot (argmax)
- $\tau \to \infty$: uniform

### Log-Softmax

$$\text{LogSoftmax}(z)_i = z_i - \log\sum_j e^{z_j}$$

**Advantages:**

- Numerically stable
- Efficient for NLL loss

## Specialized Activations

### Hard Variants

**Hard Sigmoid:**
$$\text{HardSigmoid}(x) = \max(0, \min(1, \frac{x + 1}{2}))$$

**Hard Swish:**
$$\text{HardSwish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6}$$

Used in mobile networks for efficiency.

### Maxout

$$\text{Maxout}(x) = \max_i (w_i^T x + b_i)$$

**Properties:**

- Piecewise linear
- Can approximate any convex function
- Generalizes ReLU (2 pieces) and leaky ReLU

### GLU (Gated Linear Unit)

$$\text{GLU}(x) = x_a \odot \sigma(x_b)$$

where $x$ is split into $x_a$ and $x_b$.

**Variants:**

- **ReGLU:** $x_a \odot \text{ReLU}(x_b)$
- **GEGLU:** $x_a \odot \text{GELU}(x_b)$
- **SwiGLU:** $x_a \odot \text{Swish}(x_b)$

Used in modern language models (PaLM, LLaMA).

## Analysis and Properties

### Gradient Flow Analysis

For deep networks, gradients accumulate multiplicatively:

$$\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \prod_{l=1}^{L} \text{diag}(\sigma'(z^{(l)})) W^{(l)T}$$

**Requirements for stable training:**

1. $|\sigma'(x)| \approx 1$ on average
2. Bounded or well-controlled gradients
3. Non-zero gradients (avoid dead neurons)

### Zero-Centered Activations

**Problem with non-zero-centered activations:**

If all activations are positive (sigmoid, ReLU), gradients for weights in a layer have the same sign:

$$\frac{\partial \mathcal{L}}{\partial w_j} = \delta \cdot a_{j-1}$$

where $a_{j-1} > 0$ always. This creates zig-zag gradient descent.

### Lipschitz Continuity

Activation $\sigma$ is $L$-Lipschitz if:

$$|\sigma(x) - \sigma(y)| \leq L|x - y|$$

- ReLU: 1-Lipschitz
- Sigmoid: 0.25-Lipschitz
- Tanh: 1-Lipschitz

Important for:

- Stability in adversarial training
- Controlling network Lipschitz constant
- Theoretical guarantees

### Initialization Considerations

**Xavier/Glorot (for sigmoid/tanh):**

$$W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]$$

**He (for ReLU):**

$$W \sim \mathcal{N}\left(0, \frac{2}{n_{in}}\right)$$

## Activation Selection Guidelines

| Scenario                     | Recommended Activation |
| ---------------------------- | ---------------------- |
| Hidden layers (default)      | ReLU, GELU, Swish      |
| Transformers                 | GELU, SwiGLU           |
| CNNs                         | ReLU, LeakyReLU        |
| RNNs/LSTMs                   | Tanh (gates: sigmoid)  |
| GANs                         | LeakyReLU              |
| Self-normalizing nets        | SELU                   |
| Mobile/efficient             | Hard variants          |
| Binary classification output | Sigmoid                |
| Multi-class output           | Softmax                |
| Regression output            | Linear (none)          |
| Bounded regression           | Sigmoid/Tanh           |

## Computational Considerations

### Efficiency

| Activation    | Relative Cost |
| ------------- | ------------- |
| ReLU          | 1×            |
| Leaky ReLU    | ~1×           |
| Hard Sigmoid  | ~1×           |
| Sigmoid       | ~4×           |
| Tanh          | ~4×           |
| GELU (approx) | ~3×           |
| Swish         | ~5×           |

### Memory

- In-place operations: ReLU, LeakyReLU can modify input
- Store mask: ReLU backward needs $\mathbf{1}_{x > 0}$
- Store activation: Sigmoid/Tanh backward needs $\sigma(x)$

### Fused Operations

Efficient implementations fuse activation with preceding operations:

- Linear + ReLU
- BatchNorm + ReLU
- Conv + BatchNorm + ReLU

## Recent Developments

### Adaptive Activations

- **PAU (Padé Activation Unit):** Learnable rational function
- **Learnable activations:** Optimize activation shape during training

### Attention as Activation

In transformers, attention mechanisms act as data-dependent activations:

$$\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

### Dynamic Activations

Activations that depend on the input distribution:

- Batch normalization + activation
- Layer normalization + GELU

## Summary

Key takeaways:

1. **ReLU family** dominates practical use due to simplicity and effectiveness
2. **GELU/Swish** for transformers and modern architectures
3. **Avoid sigmoid/tanh** in hidden layers (vanishing gradients)
4. **Match initialization** to activation function
5. **Consider computational cost** for deployment
6. **Zero-centered activations** improve optimization

The choice of activation function significantly impacts:

- Training dynamics and convergence speed
- Expressivity and function approximation
- Gradient flow and stability
- Computational efficiency
