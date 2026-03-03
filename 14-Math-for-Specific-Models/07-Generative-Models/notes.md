# Generative Models: Mathematical Foundations

[← Previous: Reinforcement Learning](../06-Reinforcement-Learning) | [Next: Documentation →](../../docs)

## Overview

Generative models learn the underlying data distribution $p(x)$ or the conditional distribution $p(x|z)$ to generate new samples. This section covers the mathematical theory behind various generative model architectures used in modern AI systems.

## Learning Objectives

- Understand maximum likelihood estimation for generative models
- Master latent variable models and variational inference
- Learn the mathematics of normalizing flows
- Study GAN theory and training dynamics
- Explore diffusion models and score matching

## Files in This Section

| File                               | Description                              |
| ---------------------------------- | ---------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive examples with visualizations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions         |

## Why This Matters for Machine Learning

Generative models address the most ambitious goal in machine learning: learning the full data distribution $p(x)$ so that entirely new, realistic samples can be created from scratch. The mathematical frameworks behind modern generative models — VAEs, GANs, normalizing flows, and diffusion models — each offer a different tradeoff between exact likelihood computation, sample quality, training stability, and computational cost. Understanding these tradeoffs mathematically is essential for choosing the right tool and diagnosing failures.

The VAE’s ELBO connects generative modeling to variational inference: the encoder approximates the intractable posterior while the decoder learns a generative process, and the KL term prevents the latent space from collapsing. GANs take a radically different approach by framing generation as a minimax game where the optimum corresponds to the generator matching the data distribution exactly — a result tied to the Jensen-Shannon divergence. Wasserstein GANs replace this with the earth-mover distance, fixing vanishing gradient problems at the cost of enforcing a Lipschitz constraint.

Diffusion models and score matching represent the current state of the art. They decompose generation into a sequence of small denoising steps, each of which is easy to learn, and the simplified training objective reduces to predicting the noise added at each step. The connection to score matching $\nabla_x \log p(x)$ unifies denoising diffusion with Langevin dynamics and stochastic differential equations, providing a rigorous continuous-time framework. Normalizing flows complement these approaches by offering exact, tractable log-likelihoods through invertible transformations with efficient Jacobian determinants.

## Chapter Roadmap

- **Foundations of Generative Modeling**: Maximum likelihood, explicit vs implicit density models, and latent variable formulations
- **Autoregressive Models**: Chain rule factorization, MADE, and causal-masked transformers
- **Variational Autoencoders**: ELBO derivation, reparameterization trick, $\beta$-VAE, and VQ-VAE
- **Normalizing Flows**: Change of variables, coupling layers (RealNVP), and continuous normalizing flows (Neural ODE)
- **GANs**: Minimax formulation, optimal discriminator, Jensen-Shannon divergence, and Wasserstein distance
- **Diffusion Models**: Forward/reverse processes, the simplified denoising objective, and score matching
- **Energy-Based Models**: Boltzmann distributions, contrastive divergence, and score-based training
- **Flow Matching**: Conditional flow matching, optimal transport paths, and rectified flows
- **Autoencoder Variants**: Denoising autoencoders, masked autoencoders, and their connection to score matching
- **Evaluation Metrics**: Inception Score, FID, and precision/recall for generative models
- **Advanced Topics**: Hierarchical VAEs, consistency models, and classifier-free guidance

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

> 💡 **Insight:** The ELBO is the cornerstone of VAE theory. The reconstruction term $\mathbb{E}_q[\log p(x|z)]$ asks "given a latent code sampled from the encoder, can the decoder reconstruct the input?" while the KL term $D_{KL}(q(z|x)\|p(z))$ asks "how far is the encoder’s output from the prior?" If the KL term is zero, every input maps to the same prior — good for generation but useless for reconstruction. If the KL is unconstrained, the model memorizes inputs without learning a smooth latent space. The tension between these terms is what makes the VAE latent space both _structured_ (nearby codes produce similar outputs) and _complete_ (sampling from the prior produces valid data).

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

> 💡 **Insight:** The GAN minimax theorem reveals a deep connection to information theory. When the discriminator is optimal, the generator’s loss reduces to $2 \cdot D_{JS}(p_{\text{data}}\|p_g) - \log 4$, where $D_{JS}$ is the Jensen-Shannon divergence. At Nash equilibrium, $p_g = p_{\text{data}}$ and $D(x) = 1/2$ everywhere — the discriminator can’t tell real from fake. But this equilibrium is notoriously unstable: if $D$ becomes too strong, $\log(1 - D(G(z))) \to 0$ and gradients vanish for the generator. Wasserstein GANs fix this by replacing JSD with the earth-mover distance $W$, which provides gradients even when the two distributions have non-overlapping support.

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

> 💡 **Insight:** Diffusion models flip the generative modeling problem: instead of learning to generate data in one shot, they learn to _denoise_ — to remove small amounts of noise step by step. The forward process is fixed (just add Gaussian noise), so all learning happens in the reverse process. The simplified training objective $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ is just noise prediction: given a noisy image and the noise level $t$, predict the noise that was added. This is equivalent to score matching because $\nabla_{x_t}\log q(x_t|x_0) = -(x_t - \sqrt{\bar\alpha_t}x_0)/(1-\bar\alpha_t) = -\epsilon/\sqrt{1-\bar\alpha_t}$. The stunning result is that this simple denoising objective, repeated across all noise levels, suffices to learn the entire data distribution.

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

## Key Takeaways

- Generative models can be categorized by how they represent the data distribution: explicit density (autoregressive, flows), latent variable (VAE), implicit (GAN), or iterative refinement (diffusion).
- The VAE ELBO = reconstruction - KL divergence; the reparameterization trick enables gradient-based optimization through the stochastic sampling step by expressing $z = \mu + \sigma \odot \epsilon$.
- Normalizing flows provide exact log-likelihoods via the change of variables formula, but require invertible architectures with efficiently computable Jacobian determinants.
- GANs optimize a minimax game whose equilibrium corresponds to $p_g = p_{\text{data}}$; the Wasserstein formulation provides more stable gradients by using the earth-mover distance.
- Diffusion models learn a denoising process across many noise levels; the simplified objective is just noise prediction, which is equivalent to denoising score matching.
- Score matching learns $\nabla_x \log p(x)$ without requiring the normalizing constant, connecting energy-based models, diffusion, and Langevin dynamics into a unified framework.
- Flow matching offers a simulation-free alternative to diffusion with straight-line OT paths, achieving faster sampling while maintaining competitive generation quality.

## Exercises

1. **VAE ELBO Derivation**: Starting from $\log p(x)$, multiply and divide by $q(z|x)$ inside the integral, apply Jensen’s inequality, and arrive at the ELBO. Then decompose the ELBO into the reconstruction term and the KL term. Compute the KL analytically when both $q(z|x)$ and $p(z)$ are Gaussian.

2. **Normalizing Flow Jacobian**: For a RealNVP coupling layer where $y_a = z_a$ and $y_b = z_b \odot \exp(s(z_a)) + t(z_a)$, write out the full Jacobian $\partial y / \partial z$ in block form. Show that it is triangular and that its log-determinant reduces to $\sum_i s_i(z_a)$.

3. **GAN Optimal Discriminator**: Given the GAN value function $V(D, G)$, fix $G$ and maximize $V$ with respect to $D(x)$ pointwise. Show that $D^*(x) = p_{\text{data}}(x)/(p_{\text{data}}(x) + p_g(x))$. Substitute back and show the resulting generator objective equals $2D_{JS}(p_{\text{data}}\|p_g) - \log 4$.

4. **Diffusion Forward Process**: Prove that $q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$ by inductively composing the single-step transitions $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t I)$. Show that as $t \to \infty$ with appropriate $\beta_t$ schedule, $q(x_t|x_0) \to \mathcal{N}(0, I)$.

5. **Score Matching Equivalence**: Show that the denoising score matching objective $\mathbb{E}[\|s_\theta(\tilde{x}) - \nabla_{\tilde{x}}\log q(\tilde{x}|x_0)\|^2]$ is equivalent (up to a constant) to the explicit score matching objective $\mathbb{E}[\|s_\theta(x) - \nabla_x \log p(x)\|^2]$. Explain why the denoising formulation is tractable while the explicit one is not.

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
