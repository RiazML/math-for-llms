# Generative Models: Mathematical Foundations

## Overview

Generative models learn the underlying data distribution $p(x)$ or the conditional distribution $p(x|z)$ to generate new samples. This section covers the mathematical theory behind various generative model architectures used in modern AI systems.

## Learning Objectives

- Understand maximum likelihood estimation for generative models
- Master latent variable models and variational inference
- Learn the mathematics of normalizing flows
- Study GAN theory and training dynamics
- Explore diffusion models and score matching

---

## 1. Foundations of Generative Modeling

### Maximum Likelihood Framework

Given data $\mathcal{D} = \{x^{(1)}, \ldots, x^{(n)}\}$, maximize:

$$\theta^* = \arg\max_\theta \frac{1}{n} \sum_{i=1}^n \log p_\theta(x^{(i)})$$

### Density Estimation

**Explicit density models**: Directly define $p_\theta(x)$

- Autoregressive models
- Normalizing flows
- Energy-based models

**Implicit density models**: Learn to sample without explicit density

- Generative Adversarial Networks (GANs)

**Latent variable models**: Introduce latent $z$
$$p_\theta(x) = \int p_\theta(x|z)p(z)dz$$

---

## 2. Autoregressive Models

### Chain Rule Factorization

$$p(x) = \prod_{i=1}^d p(x_i | x_1, \ldots, x_{i-1})$$

### Masked Autoencoder (MADE)

Architecture ensuring autoregressive property:

- Mask connections to enforce $p(x_i | x_{<i})$
- Single forward pass for all conditionals

### Transformer Autoregressive Models

Self-attention with causal masking:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M$ is the causal mask.

---

## 3. Variational Autoencoders (VAE)

### Evidence Lower Bound (ELBO)

$$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))$$

The ELBO consists of:

- **Reconstruction term**: $\mathbb{E}_{q(z|x)}[\log p(x|z)]$
- **Regularization term**: $-D_{KL}(q(z|x) \| p(z))$

### Reparameterization Trick

For Gaussian encoder $q(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))$:
$$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

Enables gradient flow through sampling.

### VAE Variants

**β-VAE**: Disentanglement
$$\mathcal{L}_{\beta-VAE} = \mathbb{E}_{q}[\log p(x|z)] - \beta \cdot D_{KL}(q(z|x) \| p(z))$$

**VQ-VAE**: Discrete latent codes
$$z_q = \text{Quantize}(z_e) = e_k, \quad k = \arg\min_j \|z_e - e_j\|^2$$

---

## 4. Normalizing Flows

### Change of Variables

For invertible transformation $f: z \mapsto x$:
$$p_X(x) = p_Z(f^{-1}(x)) \left|\det \frac{\partial f^{-1}}{\partial x}\right|$$

Or in log form:
$$\log p_X(x) = \log p_Z(z) - \log \left|\det \frac{\partial f}{\partial z}\right|$$

### Flow Composition

Chain of transformations:
$$x = f_K \circ f_{K-1} \circ \cdots \circ f_1(z)$$

Log determinant sums:
$$\log \left|\det \frac{\partial x}{\partial z}\right| = \sum_{k=1}^K \log \left|\det \frac{\partial f_k}{\partial h_{k-1}}\right|$$

### Coupling Layers (RealNVP)

Split input: $z = [z_a, z_b]$

Forward:
$$y_a = z_a, \quad y_b = z_b \odot \exp(s(z_a)) + t(z_a)$$

Jacobian is triangular → determinant is product of diagonal.

### Continuous Normalizing Flows

ODE-based transformation:
$$\frac{dz}{dt} = f(z(t), t; \theta)$$

Instantaneous change of variables:
$$\frac{\partial \log p(z(t))}{\partial t} = -\text{Tr}\left(\frac{\partial f}{\partial z}\right)$$

---

## 5. Generative Adversarial Networks (GANs)

### Minimax Game

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

### Optimal Discriminator

For fixed $G$:
$$D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$$

### Jensen-Shannon Divergence

At optimum:
$$\min_G V(D^*, G) = -\log 4 + 2 \cdot D_{JS}(p_{data} \| p_g)$$

### Training Dynamics

**Mode collapse**: Generator produces limited variety
**Vanishing gradients**: When $D$ is too strong

### Wasserstein GAN

Using Wasserstein-1 distance:
$$W(p_r, p_g) = \inf_{\gamma \in \Pi(p_r, p_g)} \mathbb{E}_{(x,y)\sim\gamma}[\|x - y\|]$$

Kantorovich-Rubinstein duality:
$$W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]$$

Loss function:
$$\mathcal{L} = \mathbb{E}_{x \sim p_r}[D(x)] - \mathbb{E}_{z \sim p_z}[D(G(z))]$$

with Lipschitz constraint on $D$ (gradient penalty or spectral normalization).

---

## 6. Diffusion Models

### Forward Process

Gradually add noise:
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

Marginal:
$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

### Reverse Process

Learned denoising:
$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### Variational Bound

$$\mathcal{L} = \mathbb{E}_q\left[-\log p(x_T) - \sum_{t \geq 1} \log \frac{p_\theta(x_{t-1}|x_t)}{q(x_t|x_{t-1})}\right]$$

### Simplified Objective

Predicting noise $\epsilon$:
$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t)\|^2\right]$$

### Score Matching

Score function: $s_\theta(x) \approx \nabla_x \log p(x)$

Denoising score matching:
$$\mathcal{L}_{DSM} = \mathbb{E}_{x_0, \tilde{x}}\left[\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q(\tilde{x}|x_0)\|^2\right]$$

---

## 7. Energy-Based Models

### Boltzmann Distribution

$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z_\theta}$$

where $Z_\theta = \int e^{-E_\theta(x)}dx$ is the partition function.

### Contrastive Divergence

Approximate gradient:
$$\nabla_\theta \log p(x) \approx -\nabla_\theta E_\theta(x) + \nabla_\theta E_\theta(\tilde{x})$$

where $\tilde{x}$ from short MCMC chain.

### Score-Based Training

Train energy function via score matching:
$$\nabla_x \log p(x) = -\nabla_x E(x)$$

---

## 8. Flow Matching and Rectified Flows

### Conditional Flow Matching

Target vector field:
$$u_t(x | x_1) = \frac{x_1 - x}{1 - t}$$

for linear interpolation $x_t = (1-t)x_0 + tx_1$.

Training objective:
$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_0, x_1}\left[\|v_\theta(x_t, t) - u_t(x_t | x_1)\|^2\right]$$

### Optimal Transport Flow

OT path between $p_0$ and $p_1$:
$$x_t = (1-t)x_0 + t T(x_0)$$

where $T$ is the optimal transport map.

---

## 9. Autoencoder Variants for Generation

### Denoising Autoencoders

Learn to recover clean input:
$$\mathcal{L}_{DAE} = \mathbb{E}_{x, \tilde{x}}[\|f_\theta(\tilde{x}) - x\|^2]$$

Connection to score matching:
$$f_\theta(\tilde{x}) - \tilde{x} \propto \nabla_{\tilde{x}} \log q(\tilde{x})$$

### Masked Autoencoders

Random masking during training:
$$\mathcal{L}_{MAE} = \|x_{masked} - f_\theta(x_{visible})\|^2$$

---

## 10. Evaluation Metrics

### Inception Score (IS)

$$IS = \exp\left(\mathbb{E}_x[D_{KL}(p(y|x) \| p(y))]\right)$$

Higher = better quality and diversity.

### Fréchet Inception Distance (FID)

$$FID = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

Lower = closer to real distribution.

### Precision and Recall

- **Precision**: Fraction of generated samples in real manifold
- **Recall**: Fraction of real manifold covered by generated

---

## 11. Advanced Topics

### Hierarchical VAEs

Multi-scale latent structure:
$$p(z) = p(z_L)\prod_{l=1}^{L-1} p(z_l | z_{l+1})$$

### Consistency Models

Direct mapping from noise to data:
$$f_\theta(x_t, t) \approx x_0 \text{ for all } t$$

Self-consistency loss:
$$\mathcal{L} = \|f_\theta(x_{t_n}, t_n) - f_{\theta^-}(x_{t_{n+1}}, t_{n+1})\|^2$$

### Guidance in Diffusion

**Classifier guidance**:
$$\nabla_{x_t} \log p(x_t | y) = \nabla_{x_t} \log p(x_t) + \gamma \nabla_{x_t} \log p(y | x_t)$$

**Classifier-free guidance**:
$$\tilde{\epsilon}_\theta = (1 + w)\epsilon_\theta(x_t, y) - w\epsilon_\theta(x_t, \emptyset)$$

---

## Key Connections to ML

| Concept           | Application                               |
| ----------------- | ----------------------------------------- |
| VAE ELBO          | Image generation, representation learning |
| GAN minimax       | Realistic image synthesis                 |
| Normalizing flows | Exact likelihood computation              |
| Diffusion         | State-of-the-art generation               |
| Score matching    | Density estimation                        |
| Flow matching     | Efficient training                        |

---

## Summary

Generative models provide powerful tools for:

- Learning complex data distributions
- Generating realistic samples
- Learning meaningful representations
- Density estimation and likelihood computation

Modern approaches combine:

- Latent variable models (VAE)
- Adversarial training (GAN)
- Iterative refinement (Diffusion)
- Efficient deterministic mappings (Flow Matching)

---

## References

1. Kingma & Welling - "Auto-Encoding Variational Bayes"
2. Goodfellow et al. - "Generative Adversarial Networks"
3. Rezende & Mohamed - "Variational Inference with Normalizing Flows"
4. Ho et al. - "Denoising Diffusion Probabilistic Models"
5. Song et al. - "Score-Based Generative Modeling"
6. Lipman et al. - "Flow Matching for Generative Modeling"
