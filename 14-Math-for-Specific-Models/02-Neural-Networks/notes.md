# Neural Networks: Mathematical Foundations

[← Previous: Linear Models](../01-Linear-Models) | [Next: Probabilistic Models →](../03-Probabilistic-Models)

## Overview

Neural networks are parameterized function approximators built from layers of linear transformations and nonlinear activations. Understanding their mathematical foundations is crucial for designing, training, and analyzing deep learning models.

## Files in This Section

| File | Description |
|------|-------------|
| [theory.ipynb](theory.ipynb) | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## Why This Matters for Machine Learning

Neural networks are the workhorses of modern AI, yet their power stems from a surprisingly small set of mathematical principles applied at scale. The universal approximation theorem guarantees that a sufficiently wide single-hidden-layer network can represent any continuous function — but it says nothing about whether gradient-based training can *find* that representation. This gap between expressivity and trainability is the central tension in deep learning theory and motivates nearly every architectural innovation.

Backpropagation is simply the chain rule of calculus applied systematically through a computational graph. Every gradient that updates a weight in a 100-billion-parameter language model traces back to this same principle. Understanding how gradients flow — and why they vanish or explode — reveals why LSTMs, residual connections, and careful initialization are necessary, and why certain architectures train while others do not.

The loss landscape of a neural network is a high-dimensional, non-convex surface riddled with saddle points, plateaus, and sharp vs. flat minima. Modern optimizers like Adam navigate this landscape by adapting per-parameter learning rates, while regularization techniques like dropout and batch normalization reshape the landscape itself. Understanding these geometric properties explains why overparameterized networks generalize despite having more parameters than data points.

## Chapter Roadmap

- **Feed-Forward Networks**: Architecture, MLP definition, and the universal approximation theorem
- **Backpropagation**: The chain rule in computational graphs, forward and backward passes
- **Convolutional Networks**: Convolution as equivariant linear operation, pooling, and output dimensions
- **Recurrent Networks**: Vanilla RNN, vanishing/exploding gradients, LSTM and GRU gating
- **Attention and Transformers**: Scaled dot-product attention, multi-head attention, and positional encoding
- **Optimization**: SGD, momentum, Adam, and learning rate schedules
- **Regularization**: Dropout, weight decay, batch normalization, and layer normalization
- **Loss Functions**: Regression losses, classification losses, and label smoothing
- **Weight Initialization**: Xavier/Glorot, He, and orthogonal initialization
- **Architectural Components**: Residual connections, dense connections, bottleneck layers
- **Gradient Flow Analysis**: Jacobian spectrum, gradient clipping, and skip connections
- **Expressivity and Depth**: Depth vs width tradeoffs and the neural tangent kernel

## 1. Feed-Forward Networks

### Architecture

**Single layer**: $h = \sigma(Wx + b)$

**Multi-layer perceptron (MLP)**:
$$f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x + b_1) \cdots) + b_{L-1}) + b_L$$

### Universal Approximation

**Theorem**: A single hidden layer network with sufficient width can approximate any continuous function on compact sets.

$$f_\theta(x) = \sum_{i=1}^m v_i \sigma(w_i^T x + b_i)$$

can approximate any $f \in C([0,1]^d)$ as $m \to \infty$.

> 💡 **Insight:** The universal approximation theorem is both empowering and misleading. It tells you a wide-enough network *exists* that matches your target function, but it offers no guarantee about the required width, nor that gradient descent will discover the right weights. In practice, depth (composing many layers) is far more efficient than width for representing hierarchical features — which is why deep networks dominate despite the theorem being stated for shallow ones.

## 2. Backpropagation

### Chain Rule

For composed functions $f = f_L \circ f_{L-1} \circ \cdots \circ f_1$:

$$\frac{\partial \mathcal{L}}{\partial W_\ell} = \frac{\partial \mathcal{L}}{\partial z_L} \cdot \frac{\partial z_L}{\partial z_{L-1}} \cdots \frac{\partial z_{\ell+1}}{\partial z_\ell} \cdot \frac{\partial z_\ell}{\partial W_\ell}$$

### Forward Pass

$$z_\ell = W_\ell a_{\ell-1} + b_\ell, \quad a_\ell = \sigma(z_\ell)$$

### Backward Pass

$$\delta_\ell = \frac{\partial \mathcal{L}}{\partial z_\ell} = \left(W_{\ell+1}^T \delta_{\ell+1}\right) \odot \sigma'(z_\ell)$$

**Gradients**:
$$\frac{\partial \mathcal{L}}{\partial W_\ell} = \delta_\ell a_{\ell-1}^T, \quad \frac{\partial \mathcal{L}}{\partial b_\ell} = \delta_\ell$$

> 💡 **Insight:** Backpropagation is not a special algorithm — it is the chain rule executed in reverse topological order through a directed acyclic graph. Each node caches its local Jacobian during the forward pass, and the backward pass multiplies these Jacobians right-to-left. This means the cost of computing *all* gradients is proportional to a single forward pass, regardless of the number of parameters — a fact that makes training billion-parameter models feasible at all.

## 3. Convolutional Networks

### Convolution Operation

**1D**: $(f * g)[n] = \sum_k f[k] \cdot g[n-k]$

**2D**: $(K * X)[i,j] = \sum_{m,n} K[m,n] \cdot X[i+m, j+n]$

### Properties

- **Equivariance**: $T_a(K * X) = K * T_a(X)$ for translation $T_a$
- **Parameter sharing**: Same kernel applied across spatial locations
- **Locality**: Each output depends on local input region

### Output Dimensions

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - K}{S} \right\rfloor + 1$$

where $P$ = padding, $K$ = kernel size, $S$ = stride.

### Pooling

- **Max pooling**: $y_{ij} = \max_{m,n \in R_{ij}} x_{mn}$
- **Average pooling**: $y_{ij} = \frac{1}{|R_{ij}|}\sum_{m,n \in R_{ij}} x_{mn}$

## 4. Recurrent Networks

### Vanilla RNN

$$h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$$
$$y_t = W_{hy}h_t + b_y$$

### Vanishing/Exploding Gradients

$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t}^{T-1} W_{hh}^T \text{diag}(\sigma'(z_k))$$

- If $\|W_{hh}\| < 1$: gradients vanish
- If $\|W_{hh}\| > 1$: gradients explode

### Long Short-Term Memory (LSTM)

**Gates**:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \quad \text{(forget)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \quad \text{(input)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \quad \text{(output)}$$

**Cell state**:
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

### Gated Recurrent Unit (GRU)

$$z_t = \sigma(W_z [h_{t-1}, x_t]) \quad \text{(update)}$$
$$r_t = \sigma(W_r [h_{t-1}, x_t]) \quad \text{(reset)}$$
$$\tilde{h}_t = \tanh(W [r_t \odot h_{t-1}, x_t])$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

## 5. Attention and Transformers

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where $\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$

### Transformer Block

1. Multi-head self-attention
2. Add & normalize
3. Feed-forward network
4. Add & normalize

$$x' = \text{LayerNorm}(x + \text{MHA}(x))$$
$$x'' = \text{LayerNorm}(x' + \text{FFN}(x'))$$

### Positional Encoding

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

## 6. Optimization

### Gradient Descent Variants

**SGD**: $\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(\theta_t)$

**Momentum**:
$$v_t = \gamma v_{t-1} + \eta \nabla \mathcal{L}(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

**Adam**:
$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla \mathcal{L}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla \mathcal{L})^2$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t)$$
$$\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

### Learning Rate Schedules

- **Step decay**: $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$
- **Cosine annealing**: $\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t/T))$
- **Warmup**: Linear increase for first $k$ steps

## 7. Regularization

### Dropout

During training: $h' = h \odot m / (1-p)$ where $m_i \sim \text{Bernoulli}(1-p)$

During inference: $h' = h$

**Interpretation**: Approximately ensemble of $2^n$ sub-networks.

### Weight Decay (L2)

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2}\|W\|_F^2$$

$$\nabla_{W} \mathcal{L}_{reg} = \nabla_W \mathcal{L} + \lambda W$$

### Batch Normalization

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

Reduces internal covariate shift, allows higher learning rates.

### Layer Normalization

$$\hat{x}_i = \frac{x_i - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}}$$

Normalizes across features for each sample (used in transformers).

## 8. Loss Functions

### Regression

- **MSE**: $\mathcal{L} = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$
- **MAE**: $\mathcal{L} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$
- **Huber**: Smooth combination of MSE and MAE

### Classification

- **Cross-entropy**: $\mathcal{L} = -\sum_i y_i \log \hat{y}_i$
- **Focal loss**: $\mathcal{L} = -\alpha(1-\hat{y})^\gamma \log \hat{y}$
- **Label smoothing**: Replace one-hot with $(1-\epsilon)\cdot y + \epsilon/K$

## 9. Weight Initialization

### Xavier/Glorot

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right)$$

Maintains variance: $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$

### He Initialization

For ReLU: $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$

### Orthogonal Initialization

$W = QR$ decomposition, use $Q$.

Preserves gradient norms in deep networks.

## 10. Architectural Components

### Residual Connections

$$y = F(x) + x$$

**Benefit**: Gradients flow directly through identity path.

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y}\left(\frac{\partial F}{\partial x} + I\right)$$

### Dense Connections (DenseNet)

$$x_\ell = H_\ell([x_0, x_1, \ldots, x_{\ell-1}])$$

Feature reuse and stronger gradient flow.

### Bottleneck Layers

Reduce dimensionality:
$$\text{1×1 conv} \to \text{3×3 conv} \to \text{1×1 conv}$$

Reduces computational cost.

### Squeeze-and-Excitation

Channel attention:
$$s = \sigma(W_2 \text{ReLU}(W_1 \cdot \text{GAP}(x)))$$
$$y = s \odot x$$

## 11. Gradient Flow Analysis

### Jacobian Spectrum

For stable training, Jacobian singular values should be near 1:
$$\frac{\partial h_\ell}{\partial h_{\ell-1}} \approx 1$$

### Gradient Clipping

$$g \leftarrow \min\left(1, \frac{\tau}{\|g\|}\right) g$$

Prevents gradient explosion in RNNs.

### Skip Connection Analysis

With residual: $h_{L} = h_0 + \sum_{\ell=1}^{L} F_\ell(h_{\ell-1})$

Gradient: $\frac{\partial \mathcal{L}}{\partial h_0} = \frac{\partial \mathcal{L}}{\partial h_L}\left(1 + \sum_\ell \frac{\partial F_\ell}{\partial h_0}\right)$

> 💡 **Insight:** Residual connections change the gradient from a product of Jacobians to a *sum*. Without them, a 100-layer network multiplies 100 matrices together during backprop — if any has spectral radius less than 1, gradients vanish exponentially. The skip connection adds an identity term, so even if every $\partial F_\ell / \partial h_0$ is tiny, the gradient still includes a direct path of magnitude 1. This single architectural idea enabled the jump from ~20-layer to 1000+ layer networks.

## 12. Expressivity and Depth

### Depth vs Width

- **Width**: More basis functions at each layer
- **Depth**: Hierarchical feature composition

**Depth separation**: Some functions require exponentially more neurons in shallow networks.

### Neural Tangent Kernel

In infinite-width limit:
$$f(x) - f_0(x) \approx \nabla_\theta f(x)^T (\theta - \theta_0)$$

Training dynamics become linear (kernel regression with NTK).

## ML Connections

| Concept     | Application                  |
| ----------- | ---------------------------- |
| Backprop    | Training all neural networks |
| CNN         | Computer vision, images      |
| RNN/LSTM    | Sequences, time series       |
| Transformer | NLP, vision transformers     |
| ResNet      | Very deep networks           |
| Dropout     | Regularization               |
| Adam        | Default optimizer            |
| BatchNorm   | Training stability           |

## Key Equations Summary

| Component    | Equation                                                              |
| ------------ | --------------------------------------------------------------------- |
| Forward pass | $a_\ell = \sigma(W_\ell a_{\ell-1} + b_\ell)$                         |
| Backprop     | $\delta_\ell = (W_{\ell+1}^T \delta_{\ell+1}) \odot \sigma'(z_\ell)$  |
| Convolution  | $(K*X)[i,j] = \sum_{m,n} K[m,n] X[i+m,j+n]$                           |
| LSTM cell    | $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$                     |
| Attention    | $\text{softmax}(QK^T/\sqrt{d_k})V$                                    |
| Residual     | $y = F(x) + x$                                                        |
| BatchNorm    | $\hat{x} = (x - \mu)/\sqrt{\sigma^2 + \epsilon}$                      |
| Adam         | $\theta \leftarrow \theta - \eta \hat{m}/(\sqrt{\hat{v}} + \epsilon)$ |

## Key Takeaways

- A neural network is a composition of affine transformations and pointwise nonlinearities: depth creates hierarchical representations, width provides capacity within each level.
- The universal approximation theorem guarantees representational power for single-hidden-layer networks, but depth separation results show deep networks can be exponentially more efficient.
- Backpropagation computes all gradients in $O(\text{forward pass})$ time by applying the chain rule in reverse topological order through the computational graph.
- Vanishing and exploding gradients arise from repeated multiplication of Jacobians; LSTMs, GRUs, and residual connections address this by introducing additive gradient paths.
- The Transformer's scaled dot-product attention computes pairwise interactions in $O(n^2 d)$, enabling long-range dependencies without recurrence.
- Adam combines momentum (first moment) with RMSProp (second moment) and bias correction, making it robust across a wide range of loss landscapes.
- Batch normalization and layer normalization reshape the loss landscape, enabling higher learning rates and faster convergence.

## Exercises

1. **Backprop by Hand**: For a two-layer MLP $f(x) = W_2 \sigma(W_1 x + b_1) + b_2$ with ReLU activation, compute $\partial \mathcal{L}/\partial W_1$ for MSE loss. Write out the full chain rule expansion and identify which terms come from the forward pass.

2. **Vanishing Gradients**: For a vanilla RNN with $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t)$, show that $\|\partial h_T / \partial h_t\|$ decays exponentially with $T - t$ when the spectral radius of $W_{hh}$ is less than 1. Explain how LSTM's cell state alleviates this.

3. **Convolution Output Size**: Derive the general formula for the output spatial dimension $H_{out} = \lfloor(H_{in} + 2P - K)/S\rfloor + 1$. Verify it for $H_{in}=32$, $K=5$, $P=2$, $S=2$, and explain what happens when the result is not an integer.

4. **Attention Complexity**: Prove that the time complexity of scaled dot-product attention is $O(n^2 d)$ for sequence length $n$ and head dimension $d$. Describe one method (e.g., linear attention) that reduces this to $O(nd^2)$ and what approximation it makes.

5. **Loss Landscape and Minima**: Construct a simple 1D loss function with both a sharp minimum and a flat minimum. Argue, using the PAC-Bayes bound or a weight perturbation argument, why stochastic gradient descent tends to find flat minima and why these are expected to generalize better.

## References

1. Goodfellow, Bengio, Courville - "Deep Learning"
2. He et al. - "Deep Residual Learning for Image Recognition"
3. Vaswani et al. - "Attention Is All You Need"
4. Hochreiter & Schmidhuber - "Long Short-Term Memory"
