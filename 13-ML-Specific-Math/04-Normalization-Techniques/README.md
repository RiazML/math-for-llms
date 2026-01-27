# Normalization Techniques in Machine Learning

## Overview

Normalization techniques are essential for stabilizing neural network training, reducing internal covariate shift, and enabling deeper architectures with higher learning rates.

## 1. Batch Normalization

### Definition

For a mini-batch $B = \{x_1, \ldots, x_m\}$:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

where:

- $\mu_B = \frac{1}{m}\sum_{i=1}^m x_i$ (batch mean)
- $\sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$ (batch variance)
- $\epsilon$ is a small constant for numerical stability

### Learnable Parameters

$$y_i = \gamma \hat{x}_i + \beta$$

where $\gamma$ (scale) and $\beta$ (shift) are learned parameters that allow the network to undo normalization if needed.

### Training vs. Inference

**Training**: Compute statistics from mini-batch

**Inference**: Use running (exponential moving average) statistics:
$$\mu_{running} \leftarrow (1 - \alpha)\mu_{running} + \alpha\mu_B$$
$$\sigma^2_{running} \leftarrow (1 - \alpha)\sigma^2_{running} + \alpha\sigma_B^2$$

### Gradient Flow

Let $L$ be the loss. The gradients are:

$$\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_B^2 + \epsilon}}\left(\frac{\partial L}{\partial y_i} - \frac{1}{m}\sum_j\frac{\partial L}{\partial y_j} - \frac{\hat{x}_i}{m}\sum_j\frac{\partial L}{\partial y_j}\hat{x}_j\right)$$

## 2. Layer Normalization

### Definition

Normalizes across features for each sample independently:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

where statistics are computed across the feature dimension:

- $\mu = \frac{1}{H}\sum_{i=1}^H x_i$
- $\sigma^2 = \frac{1}{H}\sum_{i=1}^H (x_i - \mu)^2$
- $H$ is the hidden dimension

### Advantages

- Works with any batch size (including batch size 1)
- Natural choice for sequence models (RNNs, Transformers)
- Consistent behavior between training and inference

### Comparison with Batch Norm

| Aspect                | Batch Norm | Layer Norm         |
| --------------------- | ---------- | ------------------ |
| Normalization axis    | Batch      | Features           |
| Batch size dependency | Yes        | No                 |
| Running statistics    | Required   | Not needed         |
| Typical use           | CNNs       | Transformers, RNNs |

## 3. Group Normalization

### Definition

Divides channels into groups and normalizes within each group:

For input with shape $(N, C, H, W)$, divide $C$ channels into $G$ groups:

$$\mu_g = \frac{1}{(C/G) \cdot H \cdot W}\sum_{c \in g}\sum_{h,w} x_{c,h,w}$$

$$\sigma_g^2 = \frac{1}{(C/G) \cdot H \cdot W}\sum_{c \in g}\sum_{h,w} (x_{c,h,w} - \mu_g)^2$$

### Special Cases

- **$G = 1$**: Layer Normalization (normalize all channels together)
- **$G = C$**: Instance Normalization (normalize each channel separately)

### Applications

- Object detection and segmentation (small batch sizes)
- Transfer learning scenarios

## 4. Instance Normalization

### Definition

Normalizes each feature map independently:

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}}$$

where:
$$\mu_{n,c} = \frac{1}{HW}\sum_{h,w} x_{n,c,h,w}$$
$$\sigma_{n,c}^2 = \frac{1}{HW}\sum_{h,w} (x_{n,c,h,w} - \mu_{n,c})^2$$

### Applications

- Style transfer (removes style information)
- Image generation tasks

## 5. RMS Normalization

### Definition

Root Mean Square Layer Normalization (RMSNorm):

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma_i$$

where:
$$\text{RMS}(x) = \sqrt{\frac{1}{H}\sum_{i=1}^H x_i^2}$$

### Advantages

- No mean computation (computational savings)
- Empirically similar performance to Layer Norm
- Used in LLaMA and other modern models

### Connection to Layer Norm

If we assume zero mean, Layer Norm and RMSNorm are equivalent:
$$\text{Var}(x) = E[x^2] - E[x]^2 \approx E[x^2] = \text{RMS}(x)^2$$

## 6. Weight Normalization

### Definition

Reparameterizes weight vector:

$$\mathbf{w} = \frac{g}{\|\mathbf{v}\|}\mathbf{v}$$

where:

- $g$ is a learned scalar (magnitude)
- $\mathbf{v}$ is a learned vector (direction)

### Gradient Computation

$$\nabla_g L = \nabla_\mathbf{w} L \cdot \frac{\mathbf{v}}{\|\mathbf{v}\|}$$

$$\nabla_\mathbf{v} L = \frac{g}{\|\mathbf{v}\|}\nabla_\mathbf{w} L - \frac{g \cdot \nabla_g L}{\|\mathbf{v}\|^2}\mathbf{v}$$

### Advantages

- Separates magnitude and direction learning
- No running statistics needed
- Deterministic (no batch statistics)

## 7. Spectral Normalization

### Definition

Normalizes weights by their spectral norm:

$$\bar{W} = \frac{W}{\sigma(W)}$$

where $\sigma(W)$ is the largest singular value of $W$.

### Power Iteration Approximation

For computational efficiency, estimate $\sigma(W)$ iteratively:

1. Initialize random unit vector $\mathbf{u}$
2. $\mathbf{v} \leftarrow \frac{W^T\mathbf{u}}{\|W^T\mathbf{u}\|}$
3. $\mathbf{u} \leftarrow \frac{W\mathbf{v}}{\|W\mathbf{v}\|}$
4. $\sigma(W) \approx \mathbf{u}^T W \mathbf{v}$

### Lipschitz Constraint

Spectral normalization ensures:
$$\|f(x_1) - f(x_2)\| \leq \|x_1 - x_2\|$$

This 1-Lipschitz constraint is crucial for GANs and stability.

## 8. Pre-Norm vs. Post-Norm

### Post-Normalization (Original Transformer)

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### Pre-Normalization (GPT-2, LLaMA)

$$\text{output} = x + \text{Sublayer}(\text{LayerNorm}(x))$$

### Comparison

| Aspect               | Post-Norm        | Pre-Norm      |
| -------------------- | ---------------- | ------------- |
| Gradient flow        | Potential issues | Better        |
| Training stability   | Lower            | Higher        |
| Learning rate warmup | Often needed     | Less critical |
| Final performance    | Slightly better  | Comparable    |

## 9. Normalization Statistics

### Dimension Summary

For input tensor of shape $(N, C, H, W)$:

| Method                  | Normalize Over | Stats Shape |
| ----------------------- | -------------- | ----------- |
| Batch Norm              | $N, H, W$      | $(C,)$      |
| Layer Norm              | $C, H, W$      | $(N,)$      |
| Instance Norm           | $H, W$         | $(N, C)$    |
| Group Norm ($G$ groups) | $C/G, H, W$    | $(N, G)$    |

### Visualization (for 2D input)

```
Batch Norm:    normalize across samples (column-wise)
Layer Norm:    normalize across features (row-wise)
Instance Norm: normalize each (sample, channel) independently
Group Norm:    normalize groups of channels per sample
```

## 10. Adaptive Normalization

### Adaptive Instance Normalization (AdaIN)

Used in style transfer:

$$\text{AdaIN}(x, y) = \sigma(y)\left(\frac{x - \mu(x)}{\sigma(x)}\right) + \mu(y)$$

where $x$ is content and $y$ is style.

### Conditional Batch Normalization

$$y = \gamma(c) \hat{x} + \beta(c)$$

where $\gamma(c)$ and $\beta(c)$ are conditioned on class $c$.

### SPADE (Spatially-Adaptive Normalization)

For semantic image synthesis:
$$\gamma_{c,y,x}(m) \cdot \frac{h_{n,c,y,x} - \mu_c}{\sigma_c} + \beta_{c,y,x}(m)$$

where $m$ is a semantic segmentation mask.

## 11. Numerical Stability

### Variance Computation

**Naive (unstable)**:
$$\sigma^2 = E[x^2] - E[x]^2$$

**Welford's algorithm (stable)**:

```
mean = 0
M2 = 0
for x in data:
    delta = x - mean
    mean += delta / n
    M2 += delta * (x - mean)
variance = M2 / n
```

### Floating Point Considerations

- Always use $\epsilon \approx 10^{-5}$ to $10^{-6}$
- Compute $\frac{1}{\sqrt{\sigma^2 + \epsilon}}$ once
- Use mixed precision carefully (accumulate in FP32)

## 12. Theoretical Analysis

### Internal Covariate Shift

Original motivation: reduce change in layer input distributions during training.

**Counter-argument**: Recent work suggests batch norm's success may be due to:

- Smoother loss landscape
- Better gradient flow
- Implicit regularization

### Loss Landscape Smoothing

Batch normalization makes the loss:
$$\|\nabla L\| \leq \gamma/\sqrt{\sigma^2 + \epsilon}$$

This bounds gradient magnitudes and smooths the optimization landscape.

### Regularization Effect

Batch norm introduces noise (via mini-batch statistics), acting as implicit regularization similar to dropout.

## 13. Practical Guidelines

### When to Use Which

| Scenario             | Recommended           |
| -------------------- | --------------------- |
| CNN with large batch | Batch Norm            |
| CNN with small batch | Group Norm            |
| Transformer/RNN      | Layer Norm            |
| LLM (efficiency)     | RMSNorm               |
| GAN discriminator    | Spectral Norm         |
| Style transfer       | Instance Norm / AdaIN |

### Placement

- After linear/conv layer, before activation (standard)
- Pre-norm for deep transformers
- After residual addition for ResNets

### Hyperparameters

- **$\epsilon$**: $10^{-5}$ (default), adjust if using FP16
- **momentum**: 0.1 for running stats (PyTorch default)
- **Groups**: 32 common for Group Norm

## Key Takeaways

1. **Batch Norm**: Original, powerful but batch-dependent
2. **Layer Norm**: Batch-independent, standard for Transformers
3. **Group Norm**: Bridges batch/layer norm, good for small batches
4. **RMSNorm**: Efficient alternative to Layer Norm
5. **Pre-norm**: Better training stability for deep models
6. **Normalization smooths loss landscape** and enables larger learning rates
