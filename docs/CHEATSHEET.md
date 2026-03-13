# Mathematics for AI/ML/LLM — Cheatsheet

> Every formula that actually matters. No filler.
> Organized by how you encounter them in practice.

---

## 1 · Linear Algebra

### Vectors

| Operation | Formula | Why It Matters |
|-----------|---------|----------------|
| Dot Product | $\mathbf{a} \cdot \mathbf{b} = \sum_{i} a_i b_i$ | Attention scores, similarity |
| Cosine Similarity | $\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ | Embeddings, RAG retrieval, semantic search |
| L2 Norm | $\|\mathbf{a}\| = \sqrt{\sum a_i^2}$ | Weight decay, distance metrics |
| L1 Norm | $\|\mathbf{a}\|_1 = \sum |a_i|$ | Sparsity, Lasso regularization |
| Projection | $\text{proj}_{\mathbf{b}}\mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{b}\|^2}\mathbf{b}$ | Orthogonal decomposition, GS process |

### Matrices

| Operation | Formula | NumPy / PyTorch |
|-----------|---------|-----------------|
| Matrix Multiply | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$ | `A @ B` |
| Transpose | $(A^T)_{ij} = A_{ji}$ | `A.T` |
| Inverse | $AA^{-1} = I$ | `np.linalg.inv(A)` |
| Trace | $\text{tr}(A) = \sum_i A_{ii}$ | `torch.trace(A)` |
| Hadamard (element-wise) | $(A \odot B)_{ij} = A_{ij} B_{ij}$ | `A * B` |

### Decompositions

**Eigendecomposition** — Used in: PCA, spectral clustering, graph analysis

$$A\mathbf{v} = \lambda\mathbf{v} \qquad A = PDP^{-1}$$

**SVD** — Used in: PCA, LoRA, matrix compression, recommender systems

$$A = U\Sigma V^T$$

- $U$ ($m \times m$): left singular vectors
- $\Sigma$ ($m \times n$): singular values on diagonal
- $V^T$ ($n \times n$): right singular vectors
- Low-rank approximation: $A_k = U_k \Sigma_k V_k^T$ (keep top-$k$ values)

**PCA via SVD**: Center data $X$, compute SVD, project onto top-$k$ columns of $V$.

### Matrix Properties That Matter

| Property | Meaning | Where You See It |
|----------|---------|-----------------|
| Positive Definite | $\mathbf{x}^T A \mathbf{x} > 0, \forall \mathbf{x} \neq 0$ | Covariance matrices, convex loss |
| Orthogonal | $A^T A = I$ | Rotation matrices, SVD components |
| Symmetric | $A = A^T$ | Covariance, Hessians, kernels |
| Rank | $\text{rank}(A) = $ # linearly independent rows/cols | LoRA exploits low rank |

---

## 2 · Calculus

### Derivatives You Need to Know

| Function | Derivative | Used In |
|----------|-----------|---------|
| $x^n$ | $nx^{n-1}$ | Polynomial features |
| $e^x$ | $e^x$ | Softmax, exponential LR decay |
| $\ln(x)$ | $1/x$ | Log-likelihood, cross-entropy |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid activation, logistic regression |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | RNN/LSTM gates |

### Rules That Drive Backpropagation

| Rule | Formula |
|------|---------|
| Chain Rule | $(f \circ g)' = f'(g(x)) \cdot g'(x)$ |
| Product Rule | $(fg)' = f'g + fg'$ |
| Sum Rule | $(f + g)' = f' + g'$ |

> **The chain rule IS backpropagation.** Every layer computes local gradient × upstream gradient.

### Multivariate Calculus

**Gradient** — direction of steepest ascent, used in every optimizer:

$$\nabla f = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right]^T$$

**Jacobian** ($\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$) — used in normalizing flows, diffusion models:

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

**Hessian** — second-order info, used in loss landscape analysis, Newton's method:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

- $H \succ 0$ (positive definite) → local minimum
- $H \prec 0$ (negative definite) → local maximum
- Mixed eigenvalues → saddle point (common in deep learning!)

---

## 3 · Probability & Statistics

### Core Rules

| Formula | Name |
|---------|------|
| $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ | Addition |
| $P(A \cap B) = P(A) \cdot P(B \mid A)$ | Multiplication |
| $P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}$ | **Bayes' Theorem** |

> **Bayes is everywhere:** Naive Bayes, MAP estimation, Bayesian neural nets, posterior inference.

### Key Distributions

| Distribution | PDF / PMF | Mean | Variance | Used In |
|-------------|-----------|------|----------|---------|
| Bernoulli | $P(X=1) = p$ | $p$ | $p(1-p)$ | Binary classification |
| Categorical | $P(X=k) = p_k$ | — | — | Multi-class output |
| Gaussian | $\frac{1}{\sigma\sqrt{2\pi}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | Weight init, noise, VAE |
| Multivariate Gaussian | $\frac{1}{(2\pi)^{n/2}\|\Sigma\|^{1/2}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})}$ | $\boldsymbol{\mu}$ | $\Sigma$ | Latent spaces, GMMs |

### Estimation

**MLE** — maximize likelihood of observed data:

$$\hat{\theta}_{MLE} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

**MAP** — MLE + prior belief (= regularization!):

$$\hat{\theta}_{MAP} = \arg\max_\theta \left[\sum_{i=1}^n \log p(x_i \mid \theta) + \log p(\theta)\right]$$

> Gaussian prior on $\theta$ → L2 regularization. Laplace prior → L1 regularization.

### Expectation & Variance

$$E[X] = \sum_x x \cdot P(x) \qquad \text{Var}(X) = E[X^2] - (E[X])^2$$

### Covariance

$$\text{Cov}(X,Y) = E[(X - \mu_X)(Y - \mu_Y)] \qquad \rho = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}$$

---

## 4 · Activation Functions

| Function | Formula | Derivative | Used In |
|----------|---------|-----------|---------|
| ReLU | $\max(0, x)$ | $\begin{cases}1 & x > 0 \\ 0 & x \leq 0\end{cases}$ | CNNs, default choice |
| Leaky ReLU | $\max(\alpha x, x)$ | $\begin{cases}1 & x > 0 \\ \alpha & x \leq 0\end{cases}$ | Avoids dead neurons |
| GELU | $x \cdot \Phi(x) \approx x \cdot \sigma(1.702x)$ | (complex) | **GPT, BERT, modern transformers** |
| SiLU / Swish | $x \cdot \sigma(x)$ | $\sigma(x) + x\sigma(x)(1-\sigma(x))$ | **LLaMA, modern LLMs** |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Gates (LSTM), binary output |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | RNN hidden states |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | $s_i(\delta_{ij} - s_j)$ | Classification output, attention |

---

## 5 · Loss Functions

### Classification

**Cross-Entropy** (multi-class) — THE standard classification loss:

$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$

**Binary Cross-Entropy:**

$$L = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

**Focal Loss** — handles class imbalance (used in object detection):

$$L = -\alpha_t (1 - \hat{y}_t)^\gamma \log(\hat{y}_t)$$

### Regression

**MSE:**

$$L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

**MAE / L1 Loss:**

$$L = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$

**Huber Loss** — smooth transition between MSE and MAE:

$$L_\delta = \begin{cases}\frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}\end{cases}$$

### Contrastive & Embedding Losses

**Contrastive Loss (SimCLR / CLIP):**

$$L = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

Where $\text{sim}$ = cosine similarity, $\tau$ = temperature

**Triplet Loss:**

$$L = \max(0, \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + \alpha)$$

- $a$ = anchor, $p$ = positive, $n$ = negative, $\alpha$ = margin

---

## 6 · Optimization

### Gradient Descent Variants

**Vanilla GD:**

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

**SGD with Momentum:**

$$v_t = \gamma v_{t-1} + \eta \nabla L(\theta_t) \qquad \theta_{t+1} = \theta_t - v_t$$

**Adam** (the default optimizer for most deep learning):

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(1st moment)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(2nd moment)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$

**AdamW** (Adam with decoupled weight decay — used in LLM training):

$$\theta_{t+1} = \theta_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t\right)$$

> Key difference from Adam: weight decay $\lambda\theta_t$ is applied directly, not through the gradient.

### Gradient Clipping

**By norm** (prevents exploding gradients — standard in LLM training):

$$\hat{g} = \begin{cases}g & \text{if } \|g\| \leq c \\ c \cdot \frac{g}{\|g\|} & \text{otherwise}\end{cases}$$

### Learning Rate Schedules

**Cosine Annealing** (used in most modern LLM training):

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{t\pi}{T}\right)$$

**Warmup + Cosine Decay** (the LLM standard):

$$\eta_t = \begin{cases}\eta_{max} \cdot \frac{t}{T_{warmup}} & t < T_{warmup} \\ \text{cosine decay} & t \geq T_{warmup}\end{cases}$$

### Regularization

| Method | Effect | Formula |
|--------|--------|---------|
| L2 / Weight Decay | Small weights | $L + \lambda\sum\theta_i^2$ |
| L1 | Sparse weights | $L + \lambda\sum\|\theta_i\|$ |
| Dropout | Random neuron masking | $h = \text{mask} \odot f(x) / (1-p)$ |

### Convexity

$$f \text{ convex} \iff f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$$

- Convex → single global minimum (guaranteed convergence)
- Deep learning losses are **non-convex** → saddle points, local minima

---

## 7 · Normalization

### Batch Normalization

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \qquad y_i = \gamma \hat{x}_i + \beta$$

Normalizes across the **batch dimension**. Used in CNNs.

### Layer Normalization

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}} \qquad y_i = \gamma \hat{x}_i + \beta$$

Normalizes across the **feature dimension**. Used in original Transformers, BERT.

### RMSNorm (Root Mean Square Normalization)

$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot \gamma \qquad \text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}$$

No mean subtraction, no bias $\beta$. Faster than LayerNorm. **Used in LLaMA, GPT-4, modern LLMs.**

---

## 8 · Transformer & Attention Math

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $Q = XW_Q$ (queries), $K = XW_K$ (keys), $V = XW_V$ (values)
- $d_k$ = key dimension. Scaling by $\sqrt{d_k}$ prevents softmax saturation.
- Complexity: $O(n^2 d)$ where $n$ = sequence length

### Why $\sqrt{d_k}$?

If $q, k$ have components with variance 1, then $q \cdot k$ has variance $d_k$.
Dividing by $\sqrt{d_k}$ restores unit variance → softmax gets reasonable gradients.

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

- $h$ heads, each with $d_k = d_{model}/h$
- Lets the model attend to different representation subspaces

### Causal (Autoregressive) Masking

$$\text{mask}_{ij} = \begin{cases}0 & i \geq j \\ -\infty & i < j\end{cases}$$

Applied before softmax: $\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{mask}\right)V$

Prevents token $i$ from attending to future tokens $j > i$. **Used in GPT, LLaMA, all decoder models.**

### Transformer Block

```
Input
  → RMSNorm / LayerNorm
  → Multi-Head Self-Attention + Residual
  → RMSNorm / LayerNorm
  → Feed-Forward Network + Residual
Output
```

**Feed-Forward Network (FFN):**

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

- $W_1$: $d_{model} \to d_{ff}$ (typically $d_{ff} = 4 \times d_{model}$)
- $W_2$: $d_{ff} \to d_{model}$

**SwiGLU FFN** (used in LLaMA, PaLM, modern LLMs):

$$\text{SwiGLU}(x) = (\text{SiLU}(W_1 x) \odot W_3 x) \cdot W_2$$

### Residual Connection

$$\text{output} = x + \text{Sublayer}(x)$$

Enables gradient flow through deep networks (50+ layers in LLMs).

---

## 9 · LLM-Specific Math

### Positional Encoding — Sinusoidal (Original Transformer)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right) \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### Rotary Position Embedding (RoPE) — Used in LLaMA, GPT-NeoX

Rotates query and key vectors based on position:

$$f(x_m, m) = R_m x_m \qquad R_m = \begin{pmatrix}\cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta\end{pmatrix}$$

Key property: $\langle f(q, m), f(k, n) \rangle$ depends only on $q$, $k$, and **relative position** $m - n$.

### Tokenization (BPE)

Byte-Pair Encoding greedily merges the most frequent adjacent token pair:

$$\text{pair}^* = \arg\max_{(a,b)} \text{count}(a, b)$$

Repeat until vocabulary size reached. Subword tokenization balances vocabulary size vs sequence length.

### Language Model Probability

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \ldots, x_{t-1})$$

Training objective (minimize negative log-likelihood):

$$L = -\frac{1}{T}\sum_{t=1}^T \log P(x_t \mid x_{<t})$$

### Temperature Scaling

$$P(x_i) = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

- $\tau \to 0$: greedy (picks highest logit)
- $\tau = 1$: standard sampling
- $\tau > 1$: more random / creative

### Top-k and Top-p (Nucleus) Sampling

- **Top-k**: Sample from the $k$ highest-probability tokens only
- **Top-p**: Sample from smallest set where $\sum P(x_i) \geq p$

### Scaling Laws (Chinchilla)

$$L(N, D) \approx A \cdot N^{-\alpha} + B \cdot D^{-\beta} + L_\infty$$

- $N$ = number of parameters, $D$ = number of training tokens
- Chinchilla-optimal: $D \approx 20 \times N$ (train on 20 tokens per parameter)

### LoRA (Low-Rank Adaptation)

$$W' = W_0 + \Delta W = W_0 + BA$$

- $W_0 \in \mathbb{R}^{d \times d}$ (frozen pretrained weights)
- $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$ where $r \ll d$
- Only $B$ and $A$ are trained. Parameters: $2dr$ instead of $d^2$
- Example: $d = 4096$, $r = 16$ → 99.2% fewer trainable parameters

### Quantization

**Uniform quantization** (maps float → int):

$$x_q = \text{round}\left(\frac{x}{s}\right) + z \qquad x_{dequant} = s(x_q - z)$$

- $s$ = scale factor, $z$ = zero point
- INT8: 4× memory reduction, INT4: 8× memory reduction

### KV Cache

During autoregressive generation, cache Key and Value matrices to avoid recomputation:

- Without cache: $O(n^2)$ per token
- With cache: $O(n)$ per new token
- Memory: $2 \times n_{layers} \times n_{heads} \times d_{head} \times \text{seq\_len} \times \text{precision}$

---

## 10 · Information Theory

| Concept | Formula | Used In |
|---------|---------|---------|
| Entropy | $H(X) = -\sum p(x) \log p(x)$ | Uncertainty measure, decision trees |
| Cross-Entropy | $H(p, q) = -\sum p(x) \log q(x)$ | **THE classification loss** |
| KL Divergence | $D_{KL}(p \| q) = \sum p(x) \log \frac{p(x)}{q(x)}$ | VAE loss, knowledge distillation |
| Perplexity | $\text{PPL} = \exp\left(-\frac{1}{T}\sum \log P(x_t \mid x_{<t})\right)$ | **LLM evaluation metric** |
| Mutual Info | $I(X;Y) = H(X) - H(X \mid Y)$ | Feature selection, info bottleneck |

> **Cross-entropy = Entropy + KL Divergence**: $H(p, q) = H(p) + D_{KL}(p \| q)$
>
> Minimizing cross-entropy loss = minimizing KL divergence from true distribution.

---

## 11 · Embeddings & Similarity

### Embedding Lookup

$$\text{embed}(x) = E[x, :] \qquad E \in \mathbb{R}^{|V| \times d}$$

One-hot × embedding matrix = row lookup. $|V|$ = vocab size, $d$ = embedding dim.

### Similarity Metrics

| Metric | Formula | Range | Used In |
|--------|---------|-------|---------|
| Cosine Similarity | $\frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ | $[-1, 1]$ | RAG, semantic search, CLIP |
| Dot Product | $\mathbf{a} \cdot \mathbf{b}$ | $(-\infty, \infty)$ | Attention scores |
| Euclidean (L2) | $\|\mathbf{a} - \mathbf{b}\|_2$ | $[0, \infty)$ | k-NN, clustering |

### Approximate Nearest Neighbor

For retrieval at scale (millions of vectors):

- **HNSW**: Hierarchical Navigable Small World graphs
- **IVF**: Inverted File Index — cluster then search within clusters
- **PQ**: Product Quantization — compress vectors, approximate distance

---

## 12 · Generative Models

### VAE (Variational Autoencoder)

**ELBO** (Evidence Lower Bound):

$$\log p(x) \geq E_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

- First term: reconstruction quality
- Second term: regularize latent space toward prior $p(z) = \mathcal{N}(0, I)$

**Reparameterization Trick** (enables backprop through sampling):

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### GAN (Generative Adversarial Network)

$$\min_G \max_D \; E_{x \sim p_{data}}[\log D(x)] + E_{z \sim p_z}[\log(1 - D(G(z)))]$$

- Optimal discriminator: $D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$
- At equilibrium: generator distribution = data distribution

### Diffusion Models (DDPM)

**Forward process** — add Gaussian noise over $T$ steps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} \, x_{t-1}, \beta_t I)$$

Direct jump to any timestep:

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) I) \qquad \bar{\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$$

**Reverse process** — neural network learns to denoise:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)$$

**Training loss** — predict the noise:

$$L = E_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

---

## 13 · Alignment & RLHF

### Reward Modeling

Train reward model $r_\phi$ on human preferences:

$$P(\text{response}_w \succ \text{response}_l) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

Bradley-Terry model: probability that response $w$ is preferred over $l$.

### PPO Objective (Proximal Policy Optimization)

$$L_{PPO} = E_t\left[\min\left(r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{old}}(a_t | s_t)}$$

With KL penalty to stay close to base model:

$$\text{objective} = E[r(x, y)] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

### DPO (Direct Preference Optimization)

Skips the reward model entirely:

$$L_{DPO} = -E\left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

- Simpler than PPO pipeline (no reward model, no RL loop)
- Equivalent to PPO under certain assumptions

### SFT (Supervised Fine-Tuning)

Standard next-token prediction on instruction-response pairs:

$$L_{SFT} = -\sum_{t=1}^T \log P_\theta(y_t \mid x, y_{<t})$$

---

## 14 · Backpropagation

For layer $l$ with $z^l = W^l a^{l-1} + b^l$ and $a^l = \sigma(z^l)$:

$$\delta^l = ((W^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l)$$
$$\frac{\partial L}{\partial W^l} = \delta^l (a^{l-1})^T \qquad \frac{\partial L}{\partial b^l} = \delta^l$$

### Vanishing / Exploding Gradients

After $n$ layers: $\frac{\partial L}{\partial W^1} \propto \prod_{l=1}^{n} W^l \cdot \sigma'(z^l)$

| Problem | Cause | Solution |
|---------|-------|----------|
| Vanishing | $\|W \cdot \sigma'\| < 1$ repeated | Residual connections, LSTM gates, ReLU |
| Exploding | $\|W \cdot \sigma'\| > 1$ repeated | Gradient clipping, proper initialization |

### Weight Initialization

| Method | Variance | Best For |
|--------|----------|----------|
| Xavier / Glorot | $\frac{2}{n_{in} + n_{out}}$ | Sigmoid, Tanh |
| He / Kaiming | $\frac{2}{n_{in}}$ | ReLU and variants |

---

## 15 · Model-Specific Math

### CNN — Convolution

$$(f * g)(t) = \sum_{\tau} f(\tau) \cdot g(t - \tau)$$

Output size: $\left\lfloor\frac{n + 2p - k}{s}\right\rfloor + 1$

- $n$ = input size, $k$ = kernel size, $p$ = padding, $s$ = stride

### RNN / LSTM

**RNN:**

$$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$$

**LSTM gates:**

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \qquad \text{(forget)}$$
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \qquad \text{(input)}$$
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \qquad \text{(candidate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \qquad \text{(cell state)}$$
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o) \qquad \text{(output)}$$
$$h_t = o_t \odot \tanh(c_t) \qquad \text{(hidden state)}$$

### GNN (Graph Convolutional Network)

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

Where $\hat{A} = A + I$ (adjacency + self-loops), $\hat{D}$ = degree matrix of $\hat{A}$.

---

## 16 · PyTorch Quick Reference

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Tensors ===
x = torch.randn(batch, seq_len, d_model)       # Random tensor
x.shape, x.dtype, x.device                      # Inspect

# === Linear Algebra ===
A @ B                                            # Matrix multiply
torch.linalg.svd(A)                              # SVD
torch.linalg.eig(A)                              # Eigendecomposition
torch.linalg.norm(x, dim=-1)                     # Norm

# === Autograd ===
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x
y.backward()
x.grad                                           # dy/dx = 2x + 3 = 7.0

# === Key Layers ===
nn.Linear(d_in, d_out)                           # Fully connected
nn.Embedding(vocab_size, d_model)                # Embedding lookup
nn.MultiheadAttention(d_model, num_heads)        # Multi-head attention
nn.LayerNorm(d_model)                            # Layer normalization
nn.Dropout(p=0.1)                                # Dropout

# === Loss Functions ===
nn.CrossEntropyLoss()                            # Classification
nn.MSELoss()                                     # Regression
nn.BCEWithLogitsLoss()                           # Binary classification
F.cosine_similarity(a, b)                        # Similarity

# === Optimizers ===
torch.optim.Adam(model.parameters(), lr=1e-4)
torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

# === LR Schedulers ===
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10000)
```

---

## 17 · Numbers to Know

| Metric | Value | Context |
|--------|-------|---------|
| GPT-3 parameters | 175B | $d_{model}=12288$, 96 layers, 96 heads |
| LLaMA-2 70B | 70B | $d_{model}=8192$, 80 layers, 64 heads |
| Typical batch size (LLM) | 1-4M tokens | Per gradient step |
| Typical learning rate (LLM) | $1\text{e-}4$ to $3\text{e-}4$ | With cosine decay |
| Chinchilla-optimal tokens | $20 \times N$ | For $N$ parameters |
| Float16 memory per param | 2 bytes | 70B model ≈ 140 GB |
| INT4 memory per param | 0.5 bytes | 70B model ≈ 35 GB |
| Attention FLOPs | $2n^2d$ | $n$ = seq len, $d$ = dim |
| FFN FLOPs | $16nd^2$ | Dominates for short sequences |

---

*This cheatsheet covers Ch.01–25 of the [Math for AI/ML/LLM](https://github.com/RiazML/math-for-llms) curriculum.*
*See also: [Notation Guide](NOTATION_GUIDE.md) · [ML Math Map](ML_MATH_MAP.md) · [Interview Prep](INTERVIEW_PREP.md)*
