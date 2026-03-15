# ML Math Map

> A precise mapping from mathematical concepts to their concrete roles in modern
> ML systems. Each entry cites the specific model, paper, or algorithm where the
> mathematics is load-bearing — not merely present.

---

## How to Read This Document

Each section identifies a mathematical domain, lists its core concepts, and maps
each concept to: (a) the ML context, (b) the specific operation or formula, and
(c) a concrete 2024–2026 example. This is a research-grade reference, not a survey.

---

## 1. Linear Algebra → ML

### 1.1 Matrix Multiplication

| Math concept | ML operation | Formula | Example |
| --- | --- | --- | --- |
| $\mathbf{y} = W\mathbf{x} + \mathbf{b}$ | Fully-connected layer | Forward pass | Every dense layer in every neural network |
| $\text{Attention} = \text{softmax}(QK^\top/\sqrt{d_k})V$ | Self-attention | $O(n^2 d)$ | Transformers: GPT-4, LLaMA-3, Gemini |
| $H^{(l+1)} = \sigma(\hat{A} H^{(l)} W^{(l)})$ | Graph convolution | Message passing | GCN (Kipf & Welling, 2017) |

### 1.2 Eigendecomposition

| Concept | ML role | Where |
| --- | --- | --- |
| $A\mathbf{v} = \lambda \mathbf{v}$ | Stability of RNN hidden state | $\rho(W_h) < 1$ prevents exploding gradients |
| Top eigenvector of $X^\top X$ | First principal component | PCA for dimensionality reduction |
| Eigenvalues of graph Laplacian $L$ | Spectral graph convolution | ChebNet, spectral GNNs |
| Hessian spectrum $\nabla^2 \mathcal{L}$ | Loss landscape sharpness | Sharpness-aware minimisation (Foret et al., 2021) |
| Neural Tangent Kernel eigenvalues | Training speed of wide networks | NTK theory (Jacot et al., 2018) |

### 1.3 SVD

| Concept | ML role | Where |
| --- | --- | --- |
| $A = U\Sigma V^\top$ | Low-rank weight decomposition | LoRA (Hu et al., 2022): $\Delta W = BA$ |
| Eckart-Young theorem | Optimal rank-$k$ approximation | Matrix factorisation for recommenders |
| Pseudoinverse $A^\dagger$ | Least-squares solution | Normal equations; ridge regression |
| Singular value spectrum | Weight matrix health | WeightWatcher (Martin & Mahoney, 2021) |
| Randomised SVD | Scalable approximation | Halko, Martinsson & Tropp (2011) |

### 1.4 Norms and Regularisation

| Norm | Regulariser | Effect | Used in |
| --- | --- | --- | --- |
| $\lVert \boldsymbol{\theta} \rVert_2^2$ | L2 / weight decay | Penalises large weights | AdamW, all modern LLMs |
| $\lVert \boldsymbol{\theta} \rVert_1$ | L1 / Lasso | Induces sparsity | Sparse fine-tuning |
| $\lVert W \rVert_2 = \sigma_{\max}(W)$ | Spectral normalisation | Lipschitz constraint | GANs (Miyato et al., 2018) |
| $\lVert A \rVert_*$ (nuclear) | Nuclear norm | Low-rank inductive bias | Matrix completion |

---

## 2. Calculus → ML

### 2.1 Chain Rule = Backpropagation

The chain rule is not merely used in backpropagation — it **is** backpropagation.

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[L]}} \cdot \prod_{k=l+1}^{L} \frac{\partial \mathbf{a}^{[k]}}{\partial \mathbf{a}^{[k-1]}} \cdot \frac{\partial \mathbf{a}^{[l]}}{\partial W^{[l]}}$$

Every automatic differentiation framework (PyTorch, JAX, TensorFlow) is a
symbolic implementation of this identity.

### 2.2 Key Derivatives in Practice

| Function | Derivative | Why it matters |
| --- | --- | --- |
| $\sigma(x) = 1/(1+e^{-x})$ | $\sigma(x)(1-\sigma(x))$ | LSTM gate updates; vanishes for large $\lvert x \rvert$ |
| $\tanh(x)$ | $1 - \tanh^2(x)$ | RNN hidden states; range $(-1,1)$ |
| $\text{ReLU}(x) = \max(0,x)$ | $\mathbb{1}[x > 0]$ | Avoids vanishing gradient for $x > 0$ |
| $\text{GELU}(x) = x\Phi(x)$ | $\Phi(x) + x\phi(x)$ | GPT-2, BERT, modern transformers |
| $\text{softmax}(\mathbf{z})_i$ | $s_i(\delta_{ij} - s_j)$ | Cross-entropy gradient; numerically use log-sum-exp |

### 2.3 Gradient Flow and Vanishing/Exploding

For a depth-$L$ network: $\frac{\partial \mathcal{L}}{\partial W^{[1]}} \propto \prod_{l=2}^{L} W^{[l]} \cdot \sigma'(\mathbf{z}^{[l]})$.

| Condition | Effect | Fix |
| --- | --- | --- |
| $\lVert W \sigma' \rVert < 1$ repeated | Vanishing gradient | Residual connections, LSTM gates, ReLU |
| $\lVert W \sigma' \rVert > 1$ repeated | Exploding gradient | Gradient clipping $\lVert g \rVert \le c$ |

### 2.4 Second-Order Methods

| Concept | Formula | ML application |
| --- | --- | --- |
| Hessian $H = \nabla^2 \mathcal{L}$ | $(H)_{ij} = \partial^2 \mathcal{L}/\partial\theta_i\partial\theta_j$ | Newton's method; Fisher information matrix |
| Gauss-Newton approximation | $H \approx J^\top J$ | K-FAC (Martens & Grosse, 2015) |
| Sharpness $\lambda_{\max}(H)$ | Largest Hessian eigenvalue | SAM (Foret et al., 2021); flat minima generalise better |

---

## 3. Probability → ML

### 3.1 Probabilistic View of Supervised Learning

| Framework | Objective | Equivalent to |
| --- | --- | --- |
| MLE | $\max \sum \log p(y^{(i)} \mid \mathbf{x}^{(i)}; \boldsymbol{\theta})$ | Minimise cross-entropy loss |
| MAP | $\max \sum \log p(y^{(i)} \mid \mathbf{x}^{(i)}; \boldsymbol{\theta}) + \log p(\boldsymbol{\theta})$ | L2 reg. (Gaussian prior), L1 reg. (Laplace prior) |
| ELBO | $\mathbb{E}_{q}[\log p(\mathbf{x} \mid \mathbf{z})] - D_\text{KL}(q(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))$ | VAE training objective (Kingma & Welling, 2014) |

### 3.2 Distributions in Active Use (2026)

| Distribution | Where | Formula |
| --- | --- | --- |
| $\mathcal{N}(\boldsymbol{\mu}, \Sigma)$ | Weight init (He/Xavier), VAE latent, diffusion | $p(\mathbf{x}) \propto \exp(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu}))$ |
| Categorical / softmax | LLM output token distribution | $p(x_i) = e^{z_i}/\sum_j e^{z_j}$ |
| Bernoulli | Dropout mask, binary classification | $P(X=1) = p$ |
| Dirichlet | Topic models, LDA, concentration prior | Conjugate to Categorical |

### 3.3 Bayes' Theorem in ML

$$p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})}{p(\mathcal{D})}$$

- **Likelihood** $p(\mathcal{D} \mid \boldsymbol{\theta})$: the loss function
- **Prior** $p(\boldsymbol{\theta})$: regularisation
- **Posterior** $p(\boldsymbol{\theta} \mid \mathcal{D})$: what we actually want
- **Evidence** $p(\mathcal{D})$: intractable; approximated by ELBO, Laplace, MCMC

---

## 4. Information Theory → ML

| Concept | Formula | ML role |
| --- | --- | --- |
| Cross-entropy | $H(p,q) = -\sum p \log q$ | **The** classification loss; training objective for all LLMs |
| KL divergence | $D_\text{KL}(p\|q) = \sum p \log(p/q)$ | VAE regulariser; knowledge distillation; RLHF KL penalty |
| Mutual information | $I(X;Y) = H(X) - H(X\mid Y)$ | InfoNCE loss; contrastive learning (SimCLR, CLIP) |
| Perplexity | $\exp(-\frac{1}{T}\sum_t \log p(x_t\mid x_{<t}))$ | LLM evaluation; lower = better language model |
| Bits-back coding | $-\mathbb{E}_q[\log p(\mathbf{x}\mid\mathbf{z})] + D_\text{KL}(q\|\,p)$ | VAE ELBO reinterpreted as compression |

---

## 5. Optimisation → ML

### 5.1 Gradient Descent Variants

| Algorithm | Update rule | Used in |
| --- | --- | --- |
| SGD + momentum | $\mathbf{v}_t = \beta\mathbf{v}_{t-1} + \nabla\mathcal{L}$; $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta\mathbf{v}_t$ | Vision models |
| Adam (Kingma & Ba, 2015) | Adaptive per-parameter $\eta$; bias-corrected moments | Default for LLMs |
| AdamW (Loshchilov & Hutter, 2019) | Adam + decoupled weight decay | GPT-3, LLaMA, all frontier LLMs |
| Muon (2024) | Orthogonalised Nesterov momentum | GPT-4o-scale training |
| SOAP (2024) | Shampoo + Adam preconditioner | State-of-art efficiency |

### 5.2 Learning Rate Schedules

| Schedule | Formula | Used in |
| --- | --- | --- |
| Cosine annealing | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max}-\eta_{\min})(1+\cos\frac{\pi t}{T})$ | GPT, LLaMA pretraining |
| Linear warmup | $\eta_t = \eta_{\max} \cdot t / T_\text{warm}$ | All large models (first 1–4K steps) |
| WSD (warmup-stable-decay) | Constant phase + sharp cosine decay | Mistral, Phi-3 |

---

## 6. Curriculum Map — Math to Model

This table maps each repository chapter to the specific models and papers
that use it as load-bearing mathematics.

| Chapter | Core math | Primary models / papers |
| --- | --- | --- |
| 02 Linear Algebra Basics | Matrix ops, rank, projections | Every neural network |
| 03 Advanced Linear Algebra | SVD, eigenvalues | LoRA, PCA, WeightWatcher |
| 04 Calculus Fundamentals | Derivatives, chain rule | Backpropagation (Rumelhart et al., 1986) |
| 05 Multivariate Calculus | Jacobian, Hessian | Adam, K-FAC, SAM |
| 06 Probability Theory | Distributions, Bayes | VAE, DDPM, Bayesian deep learning |
| 07 Statistics | MLE, MAP, hypothesis tests | Training objectives, model selection |
| 08 Optimisation | Convexity, GD, constraints | All training algorithms |
| 09 Information Theory | Entropy, KL, MI | Cross-entropy loss, RLHF, contrastive learning |
| 10 Numerical Methods | Condition number, stability | Mixed precision, numerical autograd |
| 11 Graph Theory | Laplacian, random walks | GCN, GAT, Node2Vec |
| 12 Functional Analysis | Hilbert spaces, RKHS | SVMs, kernel methods, NTK theory |
| 13 ML-Specific Math | Attention math, normalisation | Transformers (Vaswani et al., 2017) |
| 14 Math for Specific Models | RNN/LSTM, CNN, GAN | Sequence models, generative models |

---

*This map is updated with each new section added to the curriculum.
For the definitive reference on mathematics for ML, see:
Goodfellow, Bengio & Courville (2016); Bishop (2006); Shalev-Shwartz & Ben-David (2014).*
