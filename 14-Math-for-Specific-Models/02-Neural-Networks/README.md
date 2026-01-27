# Neural Networks: Mathematical Foundations

## Overview

Neural networks are parameterized function approximators built from layers of linear transformations and nonlinear activations. Understanding their mathematical foundations is crucial for designing, training, and analyzing deep learning models.

## 1. Feed-Forward Networks

### Architecture

**Single layer**: $h = \sigma(Wx + b)$

**Multi-layer perceptron (MLP)**:
$$f(x) = W_L \sigma(W_{L-1} \sigma(\cdots \sigma(W_1 x + b_1) \cdots) + b_{L-1}) + b_L$$

### Universal Approximation

**Theorem**: A single hidden layer network with sufficient width can approximate any continuous function on compact sets.

$$f_\theta(x) = \sum_{i=1}^m v_i \sigma(w_i^T x + b_i)$$

can approximate any $f \in C([0,1]^d)$ as $m \to \infty$.

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

## References

1. Goodfellow, Bengio, Courville - "Deep Learning"
2. He et al. - "Deep Residual Learning for Image Recognition"
3. Vaswani et al. - "Attention Is All You Need"
4. Hochreiter & Schmidhuber - "Long Short-Term Memory"
