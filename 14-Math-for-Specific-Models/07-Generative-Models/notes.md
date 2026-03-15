[← Back to Math for Specific Models](../README.md) | [Next: CNN and Convolution Math →](../08-CNN-and-Convolution-Math/notes.md)

---

# Generative Models: Mathematical Foundations

> _"A generative model is a compressed description of a universe — it must encode not just what exists, but the probability that each possibility exists, so that when you sample from it, you draw from the same distribution that produced reality."_

## Overview

Generative models address the most ambitious goal in machine learning: learn the full data distribution $p(\mathbf{x})$ well enough to draw new samples indistinguishable from real data. This is harder than classification or regression — instead of learning a decision boundary or a scalar function, you must implicitly or explicitly capture the full joint distribution over every pixel, token, or coordinate in high-dimensional space.

The mathematics of generative models is a meeting point of probability theory, variational inference, optimal transport, stochastic differential equations, and game theory. Each major family — autoregressive models, variational autoencoders, normalizing flows, GANs, diffusion models, and flow matching — approaches the problem with a distinct mathematical strategy, offering different tradeoffs between exact likelihood computation, sample quality, training stability, and inference speed.

This section develops the complete mathematical theory from first principles. We derive the ELBO from Jensen's inequality, the optimal GAN discriminator from calculus of variations, the diffusion forward marginal by composing Gaussian convolutions, and the flow matching objective from the continuity equation. Every derivation connects to the modern AI systems that depend on it: latent diffusion (Stable Diffusion, DALL-E 3), flow matching (Flux, MovieGen), and discrete diffusion for language generation.

## Prerequisites

- Probability theory: conditional distributions, Bayes' theorem, Gaussian distributions, KL divergence (Chapters 06–07)
- Variational inference: ELBO, mean-field approximation, reparameterisation (Section 14-03)
- Linear algebra: matrix decompositions, Jacobian determinants (Chapters 02–03)
- Calculus: partial derivatives, chain rule, ODEs (Chapters 04–05)
- Optimisation: gradient descent, convex duality, Lagrangians (Chapter 08)
- Information theory: entropy, mutual information, bits-per-dimension (Chapter 09)
- Neural networks: MLP, backpropagation (Section 14-02)
- Reinforcement Learning: RLHF context for reward-conditioned generation (Section 14-06)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: ELBO geometry, coupling layer Jacobian, GAN training dynamics, diffusion forward process, score matching, FID computation, flow matching trajectories |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems: ELBO derivation, coupling Jacobian, diffusion marginal, optimal discriminator, WGAN-GP, DDIM step, score matching equivalence, CFM objective, FID, classifier-free guidance |

## Learning Objectives

After completing this section, you will:

- Derive the ELBO from Jensen's inequality and decompose it into reconstruction and KL terms
- Implement the reparameterisation trick and explain why score-function estimators have higher variance
- Prove that the RealNVP coupling layer has a triangular Jacobian and compute its log-determinant
- Derive the optimal GAN discriminator and show the generator objective equals $2D_{\mathrm{JS}} - \log 4$
- State and prove the Kantorovich-Rubinstein duality for the Wasserstein distance
- Derive the diffusion forward marginal $q(\mathbf{x}_t\mid\mathbf{x}_0)$ by composing Gaussian transitions
- Explain why the simplified DDPM objective is equivalent to denoising score matching
- Implement DDIM deterministic sampling and explain the ODE interpretation
- Formulate the conditional flow matching objective from the continuity equation
- Compute FID using the matrix square root of the covariance product

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 What Is a Generative Model?](#11-what-is-a-generative-model)
  - [1.2 Why Generative Modeling Is Hard](#12-why-generative-modeling-is-hard)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Taxonomy of Approaches](#14-taxonomy-of-approaches)
- [2. Foundations: Maximum Likelihood and Density Estimation](#2-foundations-maximum-likelihood-and-density-estimation)
  - [2.1 The Maximum Likelihood Framework](#21-the-maximum-likelihood-framework)
  - [2.2 Explicit vs Implicit Density Models](#22-explicit-vs-implicit-density-models)
  - [2.3 Latent Variable Models](#23-latent-variable-models)
  - [2.4 The Autoregressive Factorisation](#24-the-autoregressive-factorisation)
- [3. Variational Autoencoders](#3-variational-autoencoders)
  - [3.1 ELBO Derivation](#31-elbo-derivation)
  - [3.2 The Reparameterisation Trick](#32-the-reparameterisation-trick)
  - [3.3 Gaussian VAE: Closed-Form KL and Posterior](#33-gaussian-vae-closed-form-kl-and-posterior)
  - [3.4 Posterior Collapse and β-VAE](#34-posterior-collapse-and-β-vae)
  - [3.5 VQ-VAE and Discrete Latents](#35-vq-vae-and-discrete-latents)
  - [3.6 Hierarchical VAEs](#36-hierarchical-vaes)
- [4. Normalizing Flows](#4-normalizing-flows)
  - [4.1 Change of Variables Formula](#41-change-of-variables-formula)
  - [4.2 Flow Composition and Log-Det Sum](#42-flow-composition-and-log-det-sum)
  - [4.3 Coupling Layers: NICE and RealNVP](#43-coupling-layers-nice-and-realnvp)
  - [4.4 Autoregressive Flows: MAF and IAF](#44-autoregressive-flows-maf-and-iaf)
  - [4.5 Continuous Normalizing Flows](#45-continuous-normalizing-flows)
- [5. Generative Adversarial Networks](#5-generative-adversarial-networks)
  - [5.1 The Minimax Game](#51-the-minimax-game)
  - [5.2 Optimal Discriminator and JSD](#52-optimal-discriminator-and-jsd)
  - [5.3 Training Dynamics](#53-training-dynamics)
  - [5.4 Wasserstein GAN and Optimal Transport](#54-wasserstein-gan-and-optimal-transport)
  - [5.5 Spectral Normalization and Gradient Penalty](#55-spectral-normalization-and-gradient-penalty)
  - [5.6 Conditional GANs](#56-conditional-gans)
- [6. Diffusion Models](#6-diffusion-models)
  - [6.1 Forward Process: Controlled Corruption](#61-forward-process-controlled-corruption)
  - [6.2 Reverse Process: Learned Denoising](#62-reverse-process-learned-denoising)
  - [6.3 Variational Lower Bound for Diffusion](#63-variational-lower-bound-for-diffusion)
  - [6.4 The Simplified Denoising Objective](#64-the-simplified-denoising-objective)
  - [6.5 Score Matching](#65-score-matching)
  - [6.6 DDIM: Deterministic Sampling](#66-ddim-deterministic-sampling)
  - [6.7 Guidance: Classifier and Classifier-Free](#67-guidance-classifier-and-classifier-free)
- [7. Energy-Based Models](#7-energy-based-models)
  - [7.1 The Boltzmann Distribution](#71-the-boltzmann-distribution)
  - [7.2 Score-Based Training Without MCMC](#72-score-based-training-without-mcmc)
  - [7.3 Langevin Dynamics Sampling](#73-langevin-dynamics-sampling)
- [8. Flow Matching](#8-flow-matching)
  - [8.1 Conditional Flow Matching](#81-conditional-flow-matching)
  - [8.2 Optimal Transport Paths](#82-optimal-transport-paths)
  - [8.3 Rectified Flows](#83-rectified-flows)
  - [8.4 Flow Matching vs Diffusion](#84-flow-matching-vs-diffusion)
- [9. Latent Diffusion and Modern Architectures](#9-latent-diffusion-and-modern-architectures)
  - [9.1 Latent Diffusion Models](#91-latent-diffusion-models)
  - [9.2 U-Net with Cross-Attention](#92-u-net-with-cross-attention)
  - [9.3 Discrete Diffusion for Language](#93-discrete-diffusion-for-language)
  - [9.4 Consistency Models](#94-consistency-models)
- [10. Evaluation Metrics](#10-evaluation-metrics)
  - [10.1 Inception Score](#101-inception-score)
  - [10.2 Fréchet Inception Distance](#102-fréchet-inception-distance)
  - [10.3 Precision and Recall](#103-precision-and-recall)
  - [10.4 Likelihood and Bits-Per-Dimension](#104-likelihood-and-bits-per-dimension)
- [11. Common Mistakes](#11-common-mistakes)
- [12. Exercises](#12-exercises)
- [13. Why This Matters for AI (2026 Perspective)](#13-why-this-matters-for-ai-2026-perspective)
- [14. Conceptual Bridge](#14-conceptual-bridge)

---
## 1. Intuition and Motivation

### 1.1 What Is a Generative Model?

A **generative model** is a probabilistic model of the data distribution $p(\mathbf{x})$. "Generative" means the model can, in principle, generate new samples $\mathbf{x} \sim p_\theta(\mathbf{x})$ that look like they came from the training set. There are three complementary ways to think about what a generative model actually does:

**Perspective 1: Density estimation.** The model explicitly represents a probability density $p_\theta(\mathbf{x})$ over data space. A good generative model assigns high probability to realistic data and near-zero probability to impossible data. This is the mathematically cleanest view: training maximises $\sum_i \log p_\theta(\mathbf{x}^{(i)})$, and generation samples from $p_\theta$.

**Perspective 2: Latent structure.** The model posits that each observation $\mathbf{x}$ is generated from a low-dimensional latent variable $\mathbf{z}$: first draw $\mathbf{z} \sim p(\mathbf{z})$, then draw $\mathbf{x} \sim p_\theta(\mathbf{x}\mid\mathbf{z})$. This factorisation imposes structure: the latent space should capture the semantically meaningful degrees of variation (style, content, pose) while the decoder fills in the details.

**Perspective 3: Distribution matching.** The model is a parameterised distribution $p_\theta$ that we want to match to the empirical data distribution $p_{\text{data}}$. Different generative model families correspond to different notions of "match": KL divergence (VAEs, autoregressive), Jensen-Shannon divergence (GANs), Wasserstein distance (WGANs), or implicit score matching (diffusion).

**For AI:** Every modern image, audio, and text synthesis system is a generative model. Stable Diffusion, GPT-4, DALL-E 3, and WaveNet all implement different mathematical strategies for learning $p_{\text{data}}(\mathbf{x})$ and sampling from it efficiently.

### 1.2 Why Generative Modeling Is Hard

Three fundamental difficulties make generative modeling harder than discriminative learning:

**1. The curse of dimensionality.** A $256 \times 256$ RGB image lives in $\mathbb{R}^{196608}$. The manifold of realistic images is a tiny, convoluted subspace. Any grid-based density estimator would need exponentially many cells. Generative models must learn to concentrate probability mass on this manifold without ever representing it explicitly.

**2. Intractable normalisation.** For an energy-based model $p_\theta(\mathbf{x}) = e^{-E_\theta(\mathbf{x})} / Z_\theta$, the partition function $Z_\theta = \int e^{-E_\theta(\mathbf{x})} d\mathbf{x}$ is a high-dimensional integral with no closed form. This makes maximum likelihood training $\nabla_\theta \log p_\theta(\mathbf{x}) = -\nabla_\theta E_\theta(\mathbf{x}) - \nabla_\theta \log Z_\theta$ intractable — computing $\nabla_\theta \log Z_\theta$ requires samples from the model itself.

**3. Coverage vs fidelity tradeoff.** A model trained to minimise $D_{\mathrm{KL}}(p_{\text{data}} \| p_\theta)$ (forward KL) is penalised for assigning zero probability to any region of the data distribution — it covers all modes but may produce blurry samples. A model trained to minimise $D_{\mathrm{KL}}(p_\theta \| p_{\text{data}})$ (reverse KL) is penalised for generating out-of-distribution samples — it produces sharp samples but may miss modes entirely. Different generative model families implicitly optimise different divergences.

**For AI:** The coverage-fidelity tradeoff directly explains why diffusion models produce diverse, high-quality images (approximately minimising forward KL via score matching) while GANs produce sharp but sometimes repetitive images (approximately minimising reverse KL).

### 1.3 Historical Timeline

```
GENERATIVE MODELS: HISTORICAL TIMELINE
════════════════════════════════════════════════════════════════════════

  1983  Boltzmann Machines (Hinton & Sejnowski) — first neural generative model
  1986  Restricted Boltzmann Machines (RBMs) — tractable shallow generative model
  1995  Helmholtz Machine (Dayan et al.) — wake-sleep algorithm, precursor to VAE
  2013  VAE (Kingma & Welling) — reparameterisation, amortised inference
  2014  GANs (Goodfellow et al.) — minimax game, implicit density
  2015  NICE (Dinh et al.) — coupling layers, exact likelihood flows
  2016  RealNVP (Dinh et al.) — affine coupling, multi-scale architecture
        WaveNet (van den Oord) — dilated causal convolutions, audio synthesis
  2017  WGAN (Arjovsky et al.) — Wasserstein distance, Lipschitz constraint
        MAF/IAF (Papamakarios, Kingma) — autoregressive flows
  2018  GLOW (Kingma & Dhariwal) — 1×1 invertible convolutions
        BigGAN (Brock et al.) — class-conditional image synthesis at scale
  2019  VQ-VAE-2 (Razavi et al.) — hierarchical discrete latents
        StyleGAN (Karras et al.) — style-based generator, unprecedented quality
  2020  DDPM (Ho et al.) — denoising diffusion, state-of-the-art FID
        NVAE (Vahdat & Kautz) — deep hierarchical VAE
  2021  Diffusion Beats GANs (Dhariwal & Nichol) — classifier guidance
        DALL-E (Ramesh et al.) — text-conditional image generation at scale
  2022  LDM/Stable Diffusion (Rombach et al.) — diffusion in latent space
        DALL-E 2 — CLIP + diffusion prior
        Flow Matching (Lipman et al.) — CFM, simulation-free training
  2023  Consistency Models (Song et al.) — single-step generation
        SDXL — improved latent diffusion architecture
        Imagen Video, Gen-2 — video generation at scale
  2024  Flux (Black Forest Labs) — flow matching + MMDiT architecture
        Stable Diffusion 3 — flow matching backbone
        Sora (OpenAI) — video diffusion transformers
  2025  Discrete diffusion for language — MDLM, masked diffusion LMs
        Flow matching LLMs — continuous-token generation

════════════════════════════════════════════════════════════════════════
```

### 1.4 Taxonomy of Approaches

```
TAXONOMY OF GENERATIVE MODELS
════════════════════════════════════════════════════════════════════════

  EXPLICIT DENSITY (tractable likelihood)
  ├── Autoregressive: p(x) = ∏ p(xᵢ|x<ᵢ)   [PixelCNN, GPT]
  └── Normalizing Flows: p(x) = pZ(f⁻¹(x))|det J|   [RealNVP, GLOW]

  EXPLICIT DENSITY (approximate likelihood)
  └── VAEs: log p(x) ≥ ELBO   [VAE, NVAE, VQ-VAE]

  IMPLICIT DENSITY (no likelihood, only sampling)
  └── GANs: min_G max_D V(G,D)   [GAN, WGAN, StyleGAN]

  ITERATIVE REFINEMENT (score / drift)
  ├── Diffusion: reverse Markov chain   [DDPM, DDIM, LDM]
  ├── Score-based: Langevin dynamics   [NCSN, SDE framework]
  └── Flow Matching: ODE with learned velocity   [CFM, Flux]

  ┌────────────────────────────────────────────────────────────────┐
  │ Model       │ Likelihood │ Sample quality │ Speed │ Stability │
  ├─────────────┼────────────┼────────────────┼───────┼───────────┤
  │ Autoregres. │ Exact      │ High           │ Slow  │ High      │
  │ Flow        │ Exact      │ Good           │ Fast  │ High      │
  │ VAE         │ Lower bnd  │ Blurry         │ Fast  │ High      │
  │ GAN         │ None       │ Highest*       │ Fast  │ Low       │
  │ Diffusion   │ Lower bnd  │ Highest        │ Slow  │ High      │
  │ Flow match. │ Approx.    │ Highest        │ Med   │ High      │
  └────────────────────────────────────────────────────────────────┘
  * GANs historically dominated FID; diffusion+flow matching now matches

════════════════════════════════════════════════════════════════════════
```

---

## 2. Foundations: Maximum Likelihood and Density Estimation

### 2.1 The Maximum Likelihood Framework

Given a dataset $\mathcal{D} = \{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$ drawn i.i.d. from the true data distribution $p_{\text{data}}(\mathbf{x})$, maximum likelihood estimation (MLE) finds parameters that maximise the probability of the observed data:

$$\boldsymbol{\theta}^* = \arg\max_{\boldsymbol{\theta}} \prod_{i=1}^n p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) = \arg\max_{\boldsymbol{\theta}} \frac{1}{n} \sum_{i=1}^n \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)})$$

The log transformation is monotone and converts the product to a sum, making optimisation tractable. The factor $1/n$ normalises by dataset size.

**MLE as KL minimisation.** MLE is equivalent to minimising the KL divergence from the model to the data distribution. By the law of large numbers, the empirical average converges to an expectation:

$$\frac{1}{n} \sum_{i=1}^n \log p_{\boldsymbol{\theta}}(\mathbf{x}^{(i)}) \xrightarrow{n \to \infty} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\boldsymbol{\theta}}(\mathbf{x})]$$

Now write out the KL divergence from data to model:

$$D_{\mathrm{KL}}(p_{\text{data}} \| p_{\boldsymbol{\theta}}) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\text{data}}(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\boldsymbol{\theta}}(\mathbf{x})]$$

The first term is the entropy of the data distribution — it does not depend on $\boldsymbol{\theta}$. Therefore:

$$\arg\min_{\boldsymbol{\theta}} D_{\mathrm{KL}}(p_{\text{data}} \| p_{\boldsymbol{\theta}}) = \arg\max_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log p_{\boldsymbol{\theta}}(\mathbf{x})]$$

MLE minimises $D_{\mathrm{KL}}(p_{\text{data}} \| p_{\boldsymbol{\theta}})$, the forward KL. This means MLE is penalised whenever $p_{\boldsymbol{\theta}}(\mathbf{x}) = 0$ but $p_{\text{data}}(\mathbf{x}) > 0$ — it must cover all modes of the data distribution, at the cost of sometimes generating blurry or averaged samples between modes.

**For AI:** GPT-family models optimise the cross-entropy loss $-\frac{1}{T}\sum_t \log p_{\boldsymbol{\theta}}(x_t \mid \mathbf{x}_{<t})$, which is exactly MLE for the autoregressive factorisation. The connection to KL minimisation explains why language models cover diverse topics rather than collapsing to the most common tokens.

**Non-examples (when MLE fails):**
- If $p_{\boldsymbol{\theta}}$ cannot represent the data distribution (model misspecification), MLE finds the closest model in KL sense — but "closest" may still be far.
- If the normalisation constant $Z_{\boldsymbol{\theta}}$ is intractable (energy-based models), the MLE gradient itself is intractable.
- For discrete data with continuous models, likelihood can be infinite (density model assigns a spike to each data point) — need regularisation or proper discrete models.

### 2.2 Explicit vs Implicit Density Models

**Explicit density models** define a closed-form (or at least computable) expression for $p_{\boldsymbol{\theta}}(\mathbf{x})$:

- **Fully tractable:** autoregressive models ($p(\mathbf{x}) = \prod_i p(x_i \mid \mathbf{x}_{<i})$), normalizing flows (change of variables formula). Enables exact likelihood evaluation — useful for compression (bits-per-dim) and model comparison.
- **Approximate (lower bound):** VAEs. The true likelihood $\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log \int p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$ is intractable, but the ELBO gives a lower bound trainable by gradient descent.

**Implicit density models** do not define $p_{\boldsymbol{\theta}}(\mathbf{x})$ explicitly — they only define a sampling procedure. GANs are the canonical example: the generator $G(\mathbf{z})$ maps noise to samples, but there is no closed-form expression for the induced density $p_g(\mathbf{x})$. This makes likelihood evaluation impossible but allows using powerful, flexible architectures as generators.

| Property | Explicit (tractable) | Explicit (approx) | Implicit |
|---|---|---|---|
| Likelihood evaluation | Yes | Lower bound | No |
| Sampling speed | Slow (AR) / Fast (flow) | Fast | Fast |
| Training objective | MLE | ELBO | Adversarial |
| Typical quality | High (AR), Good (flow) | Good | Highest |

### 2.3 Latent Variable Models

A **latent variable model** introduces an unobserved variable $\mathbf{z}$ that explains the observed data $\mathbf{x}$. The joint model is:

$$p_{\boldsymbol{\theta}}(\mathbf{x}, \mathbf{z}) = p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})$$

The marginal likelihood, obtained by integrating out $\mathbf{z}$, is:

$$p_{\boldsymbol{\theta}}(\mathbf{x}) = \int p_{\boldsymbol{\theta}}(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$$

This integral is generally intractable for continuous $\mathbf{z}$ in high dimensions, because there is no analytical form and Monte Carlo estimation has high variance (most $\mathbf{z}$ values give near-zero $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$).

**Intuition:** $\mathbf{z}$ encodes the "what" (identity, style, semantic content) and the decoder $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ fills in the "how" (pixel values, rendering details). The prior $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$ provides a simple distribution to sample from at generation time.

**For AI:** In Stable Diffusion, $\mathbf{z}$ is the latent encoding produced by a VAE encoder. The diffusion process operates entirely in latent space, making it $4\times$ cheaper than pixel-space diffusion. At generation time: sample $\mathbf{z} \sim p_\theta$ via denoising, then decode $\mathbf{x} = \mathcal{D}(\mathbf{z})$.

**Standard examples:**
- GMM: $\mathbf{z} \in \{1,\ldots,K\}$ discrete cluster label; $p(\mathbf{x}\mid\mathbf{z}=k) = \mathcal{N}(\boldsymbol{\mu}_k, \Sigma_k)$. Marginal is a mixture.
- VAE: $\mathbf{z} \in \mathbb{R}^d$ continuous code; $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ is a neural network decoder.
- LDA (text): $\mathbf{z}$ is topic proportion; $p(\text{word}\mid\mathbf{z})$ is topic-word distribution.

**Non-examples:**
- A purely discriminative model $p(y\mid\mathbf{x})$ with no generative story for $\mathbf{x}$.
- A deterministic autoencoder (no probabilistic decoder or prior).

### 2.4 The Autoregressive Factorisation

The **chain rule of probability** gives an exact factorisation of any joint distribution over $d$ variables:

$$p(\mathbf{x}) = p(x_1)\, p(x_2 \mid x_1)\, p(x_3 \mid x_1, x_2)\, \cdots\, p(x_d \mid x_1, \ldots, x_{d-1}) = \prod_{i=1}^d p(x_i \mid \mathbf{x}_{<i})$$

This is exact — no approximation. The challenge is parameterising each conditional $p(x_i \mid \mathbf{x}_{<i})$ with a neural network that scales to large $d$.

**MADE (Masked Autoencoder for Distribution Estimation, Germain et al. 2015):** Train a single MLP to predict all conditionals in one forward pass by masking connections. Assign each unit a random order $m(k) \in \{1,\ldots,d\}$ and mask weights $W_{ij}^{[l]} = 0$ if $m_l(j) < m_{l-1}(i)$ (upstream units can't depend on downstream inputs). Output unit $i$ then predicts $p(x_i \mid \mathbf{x}_{<i})$.

**Causal Transformer (GPT):** Self-attention with a **causal mask** $M_{ij} = -\infty$ for $j > i$ ensures each position $i$ only attends to positions $\leq i$:

$$\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V$$

Each row $i$ of the output aggregates only $\mathbf{v}_1, \ldots, \mathbf{v}_i$, enforcing the autoregressive property. This allows parallelised training (all conditionals computed in one forward pass) while maintaining exact likelihood.

**For AI:** All GPT-family models (GPT-4, Claude, LLaMA, Gemini) use the autoregressive factorisation over token sequences: $p(\mathbf{x}) = \prod_t p(x_t \mid x_1, \ldots, x_{t-1})$. Training minimises negative log-likelihood (cross-entropy). Generation samples token-by-token — the sequential nature means generation is $O(T)$ serial steps, unlike training which is $O(1)$ parallel steps.

---

## 3. Variational Autoencoders

### 3.1 ELBO Derivation

The core difficulty of latent variable models is that $\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log \int p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z}) p(\mathbf{z})\,d\mathbf{z}$ is intractable. The VAE (Kingma & Welling, 2013) introduces an **inference network** (encoder) $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ to approximate the true posterior $p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x})$.

**Derivation via Jensen's inequality.** For any distribution $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$:

$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log \int p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z}) p(\mathbf{z})\,d\mathbf{z}$$

Multiply and divide by $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$:

$$= \log \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\phi}}(\cdot\mid\mathbf{x})}\!\left[\frac{p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z}) p(\mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}\right]$$

Apply Jensen's inequality ($\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ since $\log$ is concave):

$$\geq \mathbb{E}_{\mathbf{z} \sim q_{\boldsymbol{\phi}}(\cdot\mid\mathbf{x})}\!\left[\log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z}) p(\mathbf{z})}{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}\right] =: \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x})$$

This is the **Evidence Lower BOund (ELBO)**. Expanding:

$$\mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}) = \underbrace{\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}[\log p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})]}_{\text{reconstruction term}} - \underbrace{D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z}))}_{\text{KL regularisation}}$$

**Gap between ELBO and log-likelihood.** The gap is exactly the KL between the approximate and true posteriors:

$$\log p_{\boldsymbol{\theta}}(\mathbf{x}) - \mathcal{L}(\boldsymbol{\theta}, \boldsymbol{\phi}; \mathbf{x}) = D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x})) \geq 0$$

So the ELBO is tight if and only if $q_{\boldsymbol{\phi}} = p_{\boldsymbol{\theta}}(\cdot\mid\mathbf{x})$ exactly.

**Alternative derivation via KL.** Starting from $D_{\mathrm{KL}}(q_{\boldsymbol{\phi}} \| p_{\boldsymbol{\theta}}) \geq 0$ and rearranging gives the same ELBO — illuminating that maximising the ELBO is equivalent to minimising the posterior approximation error.

**For AI:** In Stable Diffusion, the ELBO is used to train the VAE compressor/decompressor. The reconstruction term encourages pixel-accurate decoding; the KL term keeps the latent space close to $\mathcal{N}(\mathbf{0}, I)$ so diffusion can operate on it. The ELBO objective also appears in VQ-VAEs (DALL-E, Parti) with a discrete codebook.

### 3.2 The Reparameterisation Trick

The ELBO contains an expectation over $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$, which depends on the parameters $\boldsymbol{\phi}$. To optimise $\boldsymbol{\phi}$ with gradient descent, we need $\nabla_{\boldsymbol{\phi}} \mathcal{L}$.

**Problem:** Naive Monte Carlo gives the **score-function estimator** (REINFORCE):

$$\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}}[f(\mathbf{z})] = \mathbb{E}_{q_{\boldsymbol{\phi}}}[f(\mathbf{z}) \nabla_{\boldsymbol{\phi}} \log q_{\boldsymbol{\phi}}(\mathbf{z})]$$

This is unbiased but has **very high variance** because $f(\mathbf{z}) \nabla_{\boldsymbol{\phi}} \log q$ fluctuates wildly, especially when $f$ (the reconstruction loss) is large.

**Reparameterisation trick.** Express the random variable $\mathbf{z}$ as a deterministic function of $\boldsymbol{\phi}$ and a noise variable $\boldsymbol{\varepsilon}$ with a fixed distribution:

$$\mathbf{z} = g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon}, \mathbf{x}) = \boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}) + \boldsymbol{\sigma}_{\boldsymbol{\phi}}(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

Now $\boldsymbol{\phi}$ appears inside a deterministic function, so the gradient passes through:

$$\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}}[f(\mathbf{z})] = \nabla_{\boldsymbol{\phi}} \mathbb{E}_{\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},I)}[f(g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon}, \mathbf{x}))] = \mathbb{E}_{\boldsymbol{\varepsilon}}[\nabla_{\boldsymbol{\phi}} f(g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon}, \mathbf{x}))]$$

This **pathwise gradient** has much lower variance because it uses the chain rule through $g_{\boldsymbol{\phi}}$ rather than the noisy score function. In practice, a single sample $\boldsymbol{\varepsilon}^{(1)} \sim \mathcal{N}(\mathbf{0},I)$ suffices for training.

**Generalisation.** Reparameterisation works for any distribution with a location-scale family or an invertible CDF transform:
- Gaussian: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\varepsilon}$
- Laplace: $z = \mu - b\operatorname{sign}(u)\log(1 - 2\lvert u \rvert)$, $u \sim \mathcal{U}(-\frac{1}{2}, \frac{1}{2})$
- Exponential: $z = -\lambda^{-1}\log u$, $u \sim \mathcal{U}(0,1)$
- Discrete (approximately): Gumbel-softmax / concrete relaxation

**Non-example — Bernoulli:** $z \sim \operatorname{Bern}(p)$ has no reparameterisation. The Gumbel-softmax trick provides a biased-but-lower-variance continuous relaxation.

**For AI:** The reparameterisation trick is the key to training VAEs with backpropagation. It also appears in diffusion model training: the forward process $\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}$ is a reparameterisation that puts $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},I)$ as the free noise variable.

### 3.3 Gaussian VAE: Closed-Form KL and Posterior

**Standard VAE architecture.** The encoder outputs Gaussian parameters:
$$q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\phi}}(\mathbf{x}),\, \operatorname{diag}(\boldsymbol{\sigma}_{\boldsymbol{\phi}}^2(\mathbf{x})))$$

The prior is $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$. The KL term in the ELBO has a **closed-form solution**:

$$D_{\mathrm{KL}}\!\left(\mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2)) \| \mathcal{N}(\mathbf{0}, I)\right) = \frac{1}{2} \sum_{j=1}^d \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

**Derivation.** For two Gaussians $p = \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1)$ and $q = \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)$:

$$D_{\mathrm{KL}}(p \| q) = \frac{1}{2}\left[\operatorname{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1)^\top \Sigma_2^{-1} (\boldsymbol{\mu}_2 - \boldsymbol{\mu}_1) - d + \log\frac{\det \Sigma_2}{\det \Sigma_1}\right]$$

With $\boldsymbol{\mu}_1 = \boldsymbol{\mu}$, $\Sigma_1 = \operatorname{diag}(\boldsymbol{\sigma}^2)$, $\boldsymbol{\mu}_2 = \mathbf{0}$, $\Sigma_2 = I$:

$$= \frac{1}{2}\left[\operatorname{tr}(\operatorname{diag}(\boldsymbol{\sigma}^2)) + \boldsymbol{\mu}^\top \boldsymbol{\mu} - d - \log\det(\operatorname{diag}(\boldsymbol{\sigma}^2))\right] = \frac{1}{2}\sum_j\left(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2\right)$$

**The decoder.** If $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z}) = \mathcal{N}(f_{\boldsymbol{\theta}}(\mathbf{z}), \sigma_x^2 I)$, then the reconstruction term is:

$$\mathbb{E}_{q_{\boldsymbol{\phi}}}[\log p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})] = -\frac{1}{2\sigma_x^2}\mathbb{E}\!\left[\lVert\mathbf{x} - f_{\boldsymbol{\theta}}(\mathbf{z})\rVert_2^2\right] + \text{const}$$

So minimising negative ELBO corresponds to mean-squared reconstruction error (weighted by $1/\sigma_x^2$) plus the closed-form KL. In practice $\sigma_x^2 = 1$ is used as a hyperparameter, trading off reconstruction sharpness vs. latent regularity.

### 3.4 Posterior Collapse and β-VAE

**Posterior collapse** is a failure mode where the encoder learns $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}) \approx p(\mathbf{z}) = \mathcal{N}(\mathbf{0},I)$ for all $\mathbf{x}$: the KL term vanishes but the latent code carries no information about $\mathbf{x}$. The decoder ignores $\mathbf{z}$ and learns a fixed marginal distribution — the model becomes a pure autoregressive decoder.

**Why it happens.** If the decoder is powerful enough to model $p(\mathbf{x})$ without using $\mathbf{z}$ (e.g., a transformer decoder), it will do so — the KL penalty incentivises collapsing the posterior to the prior. This is an instance of the **information preference** of the ELBO: the model always prefers to eliminate the KL cost by not using the latent.

**KL annealing.** Train with a schedule that starts the KL weight at 0 and gradually increases it: $\mathcal{L}_{\text{anneal}} = \mathbb{E}_{q}[\log p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})] - \beta(t)\,D_{\mathrm{KL}}(q_{\boldsymbol{\phi}} \| p)$, with $\beta(t)$ increasing from 0 to 1. This forces the encoder to learn informative codes before the KL regularisation kicks in.

**Free bits (Kingma et al., 2016).** Fix a minimum information $\lambda$ per latent dimension: $\mathcal{L}_{\text{free}} = \mathbb{E}_{q}[\log p] - \sum_j \max(\lambda, D_{\mathrm{KL}}(q_j \| p_j))$. Dimensions already above $\lambda$ nats are not penalised further, preventing collapse while maintaining meaningful latent structure.

**β-VAE (Higgins et al., 2017).** Upweight the KL term to encourage disentanglement:

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})}[\log p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})] - \beta\, D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}) \| p(\mathbf{z})), \quad \beta > 1$$

With $\beta > 1$, the model is more strongly penalised for using correlated latent dimensions, encouraging each $z_j$ to capture a single independent factor of variation (e.g., colour, shape, orientation). At the cost of some reconstruction quality.

**For AI:** Posterior collapse is a major challenge for discrete VAEs (VQ-VAE) and hierarchical VAEs used in DALL-E and Parti. VQ-VAE sidesteps the problem by replacing the continuous Gaussian encoder with a discrete codebook lookup, making the KL term constant.

### 3.5 VQ-VAE and Discrete Latents

**Vector quantisation.** Instead of a continuous Gaussian encoder, VQ-VAE (van den Oord et al., 2017) uses a discrete codebook $\{e_k\}_{k=1}^K \subset \mathbb{R}^d$. The encoder produces a continuous representation $\mathbf{z}_e = E_{\boldsymbol{\phi}}(\mathbf{x})$, which is then quantised to the nearest codebook entry:

$$\mathbf{z}_q = e_k, \quad k = \arg\min_j \lVert \mathbf{z}_e - e_j \rVert_2^2$$

The decoder receives $\mathbf{z}_q$ and reconstructs $\hat{\mathbf{x}} = D_{\boldsymbol{\theta}}(\mathbf{z}_q)$.

**Straight-through estimator.** The quantisation step $\mathbf{z}_q = e_k$ is non-differentiable (argmin has zero gradient almost everywhere). The **straight-through estimator** copies gradients from $\mathbf{z}_q$ to $\mathbf{z}_e$: $\frac{\partial \mathcal{L}}{\partial \mathbf{z}_e} \approx \frac{\partial \mathcal{L}}{\partial \mathbf{z}_q}$. This is a biased estimator but works well in practice.

**Training objective.**

$$\mathcal{L}_{\text{VQ}} = \underbrace{\lVert \mathbf{x} - \hat{\mathbf{x}} \rVert_2^2}_{\text{reconstruction}} + \underbrace{\lVert \text{sg}[\mathbf{z}_e] - \mathbf{z}_q \rVert_2^2}_{\text{codebook}} + \underbrace{\beta\lVert \mathbf{z}_e - \text{sg}[\mathbf{z}_q] \rVert_2^2}_{\text{commitment}}$$

where $\text{sg}[\cdot]$ is the stop-gradient operator. The codebook loss moves embeddings toward encoder outputs; the commitment loss (weighted by $\beta$) prevents the encoder outputs from growing arbitrarily far from the codebook.

**For AI:** DALL-E 1 (Ramesh et al., 2021) used a VQ-VAE to compress images into $32 \times 32$ discrete tokens, then trained a GPT on the concatenation of text and image tokens. VQ-VAE-2 (Razavi et al., 2019) introduced a two-level hierarchy (local + global codebooks) for higher-fidelity generation.

### 3.6 Hierarchical VAEs

Standard VAEs use a single latent layer, limiting their expressiveness. **Hierarchical VAEs** introduce multiple latent layers $\mathbf{z}_1, \ldots, \mathbf{z}_L$ with a factored structure:

$$p(\mathbf{z}) = p(\mathbf{z}_L) \prod_{l=1}^{L-1} p(\mathbf{z}_l \mid \mathbf{z}_{l+1})$$

**Top-down inference (NVAE, Vahdat & Kautz, 2020).** The encoder runs top-down: a bottom-up pass extracts features from $\mathbf{x}$, then a top-down pass samples $\mathbf{z}_L$ first and conditions each subsequent $\mathbf{z}_l$ on both the bottom-up features and $\mathbf{z}_{l+1}$. This avoids the information bottleneck at each layer.

**VDVAE (Child, 2021).** Uses **residual normal distributions** for numerical stability: parameterise the encoder as a perturbation of the decoder's prior: $q(\mathbf{z}_l \mid \mathbf{z}_{l+1}, \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_{\text{dec}} + \Delta\boldsymbol{\mu}, \sigma_{\text{dec}}\odot\Delta\boldsymbol{\sigma})$. This makes the KL terms small and well-conditioned throughout training.

---

## 4. Normalizing Flows

### 4.1 Change of Variables Formula

A **normalizing flow** is an invertible, differentiable transformation $f: \mathcal{Z} \to \mathcal{X}$ that maps a simple base distribution $p_Z(\mathbf{z})$ (typically $\mathcal{N}(\mathbf{0}, I)$) to a complex data distribution $p_X(\mathbf{x})$. The name "normalizing" refers to transforming to the normal distribution in the reverse direction.

**Change of Variables Theorem.** If $\mathbf{x} = f(\mathbf{z})$ where $f$ is invertible and differentiable, then:

$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \cdot \left\lvert \det J_{f^{-1}}(\mathbf{x}) \right\rvert$$

In log form (using $\det J_{f^{-1}} = (\det J_f)^{-1}$):

$$\log p_X(\mathbf{x}) = \log p_Z(\mathbf{z}) - \log \left\lvert \det J_f(\mathbf{z}) \right\rvert, \quad \mathbf{z} = f^{-1}(\mathbf{x})$$

This is exact — no approximation, no lower bound. Flows give tractable, exact likelihoods.

**Intuition.** The Jacobian determinant $|\det J_f(\mathbf{z})|$ is a volume scaling factor: if $f$ expands a region by factor $c$, the probability density in that region decreases by $1/c$ to preserve the total probability mass. This is the continuous analogue of the discrete change-of-variables formula $P(Y = y) = P(X = f^{-1}(y)) / |f'(f^{-1}(y))|$.

**Two directions:**
- **Sampling (generation):** Sample $\mathbf{z} \sim p_Z$, compute $\mathbf{x} = f(\mathbf{z})$. Fast if $f$ is fast.
- **Density evaluation:** Given $\mathbf{x}$, compute $\mathbf{z} = f^{-1}(\mathbf{x})$ and evaluate $\log p_X(\mathbf{x})$. Fast if $f^{-1}$ is fast.

The fundamental design tension: $f$ must be (1) expressive enough to model complex distributions, (2) invertible, (3) have a tractably computable Jacobian determinant. Most flow architectures solve (3) by making $J_f$ triangular (determinant = product of diagonal).

**For AI:** Normalizing flows appear in latent diffusion: the VAE encoder/decoder is trained to be approximately invertible so the latent space has well-defined density. More directly, GLOW (Kingma & Dhariwal, 2018) achieved competitive image synthesis quality while providing exact likelihoods — enabling direct bits-per-dim comparisons.

### 4.2 Flow Composition and Log-Det Sum

Flows gain expressiveness by composing $K$ simpler transformations $f = f_K \circ f_{K-1} \circ \cdots \circ f_1$:

$$\mathbf{x} = f_K(\cdots f_2(f_1(\mathbf{z})))$$

By the chain rule of Jacobians, the total log-det decomposes as a sum:

$$\log \left\lvert \det J_f(\mathbf{z}) \right\rvert = \sum_{k=1}^K \log \left\lvert \det J_{f_k}(\mathbf{h}_{k-1}) \right\rvert, \quad \mathbf{h}_0 = \mathbf{z},\; \mathbf{h}_k = f_k(\mathbf{h}_{k-1})$$

This is crucial: even if each $f_k$ has a simple Jacobian, their composition can be highly expressive. Training maximises:

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \left[\log p_Z(\mathbf{z}^{(i)}) + \sum_{k=1}^K \log \left\lvert \det J_{f_k}(\mathbf{h}_{k-1}^{(i)}) \right\rvert \right]$$

**Memory vs computation.** During training, all intermediate activations $\mathbf{h}_1, \ldots, \mathbf{h}_{K-1}$ must be stored for backpropagation through the $K$ layers. For continuous normalizing flows (Neural ODEs), the adjoint method reduces memory to $O(1)$ at the cost of two ODE solves.

### 4.3 Coupling Layers: NICE and RealNVP

**NICE (Dinh et al., 2014)** introduced the **additive coupling layer**. Split $\mathbf{z} = [\mathbf{z}_a, \mathbf{z}_b]$ and define:

$$\mathbf{y}_a = \mathbf{z}_a, \qquad \mathbf{y}_b = \mathbf{z}_b + t(\mathbf{z}_a)$$

The Jacobian is:
$$J = \frac{\partial (\mathbf{y}_a, \mathbf{y}_b)}{\partial (\mathbf{z}_a, \mathbf{z}_b)} = \begin{pmatrix} I & 0 \\ \frac{\partial t}{\partial \mathbf{z}_a} & I \end{pmatrix}$$

This is lower triangular, so $\det J = 1$ — the log-det contribution is zero, no computation needed.

**RealNVP (Dinh et al., 2016)** extends to **affine coupling layers**:

$$\mathbf{y}_a = \mathbf{z}_a, \qquad \mathbf{y}_b = \mathbf{z}_b \odot \exp(s(\mathbf{z}_a)) + t(\mathbf{z}_a)$$

The Jacobian becomes:
$$J = \begin{pmatrix} I & 0 \\ \frac{\partial t}{\partial \mathbf{z}_a} + \frac{\partial s}{\partial \mathbf{z}_a}\cdot\text{diag}(\exp(s)\odot\mathbf{z}_b) & \operatorname{diag}(\exp(s(\mathbf{z}_a))) \end{pmatrix}$$

Still lower triangular, so:
$$\log \lvert \det J \rvert = \sum_j s_j(\mathbf{z}_a)$$

**Inverse:** Trivially invertible given $\mathbf{y}_a$:
$$\mathbf{z}_a = \mathbf{y}_a, \qquad \mathbf{z}_b = (\mathbf{y}_b - t(\mathbf{y}_a)) \odot \exp(-s(\mathbf{y}_a))$$

**Key insight:** $s$ and $t$ are arbitrary neural networks — they need not be invertible themselves, because the inverse is computed analytically. This allows using powerful, expressive networks as scale/shift functions while maintaining exact invertibility.

**Masking strategies.** Multiple coupling layers with alternating masks $(1,0,1,0,\ldots)$ and $(0,1,0,1,\ldots)$ ensure all dimensions are eventually transformed by a network.

### 4.4 Autoregressive Flows: MAF and IAF

**Masked Autoregressive Flow (MAF, Papamakarios et al., 2017).** Express the flow using the autoregressive structure:

$$y_i = x_i \cdot \exp(\alpha_i) + \mu_i, \quad [\mu_i, \alpha_i] = \text{MADE}(x_1, \ldots, x_{i-1})$$

The Jacobian is triangular (each $y_i$ depends only on $x_1,\ldots,x_i$), so the log-det is $\sum_i \alpha_i$. **Density evaluation** (given $\mathbf{x}$, compute $\log p$): parallel over all $i$ — fast. **Sampling** (given noise $\mathbf{u}$, compute $\mathbf{x}$): sequential over $i$ — $O(d)$ serial steps.

**Inverse Autoregressive Flow (IAF, Kingma et al., 2016).** Reverses the roles: density evaluation is serial, sampling is parallel. IAF is efficient for the VAE decoder (fast sampling) but slow for evaluation.

**Parallel Wavenet (van den Oord et al., 2018).** Trains a MAF teacher with slow sampling, then distils into an IAF student via probability density distillation. The student generates audio in parallel at deployment time — enabling real-time speech synthesis.

**For AI:** Autoregressive flows connect closely to LLMs: both use the chain rule factorisation with MADE/transformer architectures. The difference is that flows use continuous variables with invertible transformations, while LLMs use discrete tokens with softmax.

### 4.5 Continuous Normalizing Flows

**Neural ODE (Chen et al., 2018).** Replace a discrete chain of flow layers with a continuous-time ODE:

$$\frac{d\mathbf{z}(t)}{dt} = f_{\boldsymbol{\theta}}(\mathbf{z}(t), t), \qquad \mathbf{z}(0) = \mathbf{z}_0 \sim p_Z$$

The transformation from $t=0$ to $t=1$ gives $\mathbf{x} = \mathbf{z}(1)$.

**Instantaneous change of variables.** The log-density evolves as:

$$\frac{d\log p(\mathbf{z}(t))}{dt} = -\operatorname{tr}\!\left(\frac{\partial f_{\boldsymbol{\theta}}}{\partial \mathbf{z}}\right) = -\operatorname{tr}(J_{f_\theta})$$

This avoids computing the full Jacobian determinant — only the trace (sum of diagonal) is needed. However, the trace still costs $O(d)$ Hutchinson estimator calls in practice, making CNFs slower than coupling layers for large $d$.

**Adjoint method.** Training CNFs with backpropagation through the ODE solver would require storing all intermediate states. The adjoint method solves a second ODE backwards to compute gradients with $O(1)$ memory.

**For AI:** Continuous normalizing flows are the mathematical foundation of **flow matching** (Section 8). Instead of learning $f_\theta$ by maximising likelihood, flow matching directly regresses the velocity field onto a target — enabling much faster training.

---

## 5. Generative Adversarial Networks

### 5.1 The Minimax Game

GANs (Goodfellow et al., 2014) frame generation as a two-player game between a **generator** $G_{\boldsymbol{\theta}}: \mathcal{Z} \to \mathcal{X}$ and a **discriminator** $D_{\boldsymbol{\phi}}: \mathcal{X} \to [0,1]$:

$$\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}}\; V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]$$

**Discriminator objective.** For fixed $G$, the discriminator maximises the binary cross-entropy of classifying real ($y=1$) vs fake ($y=0$) samples. $D(\mathbf{x}) \in [0,1]$ outputs the probability that $\mathbf{x}$ is real.

**Generator objective.** For fixed $D$, the generator minimises $V$, i.e., tries to make $D(G(\mathbf{z})) \approx 1$ for generated samples.

**Nash equilibrium.** The game has a Nash equilibrium when neither player can improve unilaterally. As we will show in §5.2, the unique global Nash equilibrium satisfies $p_g = p_{\text{data}}$ and $D(\mathbf{x}) = 1/2$ everywhere.

**Non-saturating loss.** In practice, the generator objective $\min \log(1 - D(G(\mathbf{z})))$ saturates early in training when $D$ is strong ($D(G(\mathbf{z})) \approx 0$, so $\log(1 - D) \approx 0$ and gradients vanish). The **non-saturating alternative** $\max \log D(G(\mathbf{z}))$ has the same Nash equilibrium but provides larger gradients when the generator is losing — this is what is used in practice.

### 5.2 Optimal Discriminator and JSD

**Theorem (Goodfellow et al., 2014).** For fixed $G$, the optimal discriminator is:

$$D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$$

**Proof.** The value function $V(D, G)$ can be written as an integral over $\mathbf{x}$:

$$V(D, G) = \int_{\mathcal{X}} \left[p_{\text{data}}(\mathbf{x}) \log D(\mathbf{x}) + p_g(\mathbf{x}) \log(1 - D(\mathbf{x}))\right] d\mathbf{x}$$

For each $\mathbf{x}$, the integrand $a \log y + b \log(1-y)$ with $a = p_{\text{data}}(\mathbf{x})$, $b = p_g(\mathbf{x})$ is maximised at $y = a/(a+b)$, giving $D^* = p_{\text{data}}/(p_{\text{data}} + p_g)$. $\square$

**Generator objective at optimal discriminator.** Substituting $D^*$ into $V(D^*, G)$:

$$V(D^*, G) = \mathbb{E}_{p_{\text{data}}}\!\left[\log \frac{p_{\text{data}}}{p_{\text{data}} + p_g}\right] + \mathbb{E}_{p_g}\!\left[\log \frac{p_g}{p_{\text{data}} + p_g}\right]$$

Let $m = (p_{\text{data}} + p_g)/2$. Then:

$$V(D^*, G) = -\log 4 + D_{\mathrm{KL}}\!\left(p_{\text{data}} \,\Big\|\, \frac{p_{\text{data}} + p_g}{2}\right) + D_{\mathrm{KL}}\!\left(p_g \,\Big\|\, \frac{p_{\text{data}} + p_g}{2}\right) = -\log 4 + 2\, D_{\mathrm{JS}}(p_{\text{data}} \| p_g)$$

The global minimum is achieved at $p_g = p_{\text{data}}$, giving $V = -\log 4$ and $D = 1/2$ everywhere.

**For AI:** The connection to JSD explains the "realness" score that GAN discriminators implicitly compute. In RLHF, the reward model is conceptually similar to a discriminator — it distinguishes preferred from non-preferred responses, and its implicit objective is related to the JSD between the response distributions.

### 5.3 Training Dynamics

**Mode collapse.** The generator learns to produce a limited set of high-quality outputs that fool the discriminator, ignoring large portions of the data distribution. Geometrically, the generator collapses to a few modes of $p_{\text{data}}$. The discriminator then specialises in detecting these modes, forcing the generator to move — creating an unstable oscillation.

**Vanishing gradients.** When the discriminator is perfect ($D^*(G(\mathbf{z})) \approx 0$ for all generator outputs), the gradient of the generator loss $\nabla_{\boldsymbol{\theta}} \log(1 - D(G(\mathbf{z}))) \approx 0$ vanishes. This is a fundamental problem: the better the discriminator, the worse the generator's gradient signal.

**Non-overlapping supports.** If $p_{\text{data}}$ and $p_g$ have disjoint supports (common in high dimensions), there exists a perfect discriminator, $D_{\mathrm{JS}} = \log 2$ (maximum), and generator gradients vanish identically. This is the theoretical motivation for Wasserstein GANs.

**Training instabilities.** The minimax game is a saddle-point problem, not a standard minimisation. Gradient descent on both players simultaneously does not generally converge to the Nash equilibrium — it can cycle or diverge. Practical fixes include: two time-scale update rule (discriminator $k$ steps per generator step), spectral normalization, gradient penalty, and careful architecture choices.

### 5.4 Wasserstein GAN and Optimal Transport

**Wasserstein-1 distance** (earth-mover distance) between distributions $p$ and $q$:

$$W_1(p, q) = \inf_{\gamma \in \Pi(p, q)} \mathbb{E}_{(\mathbf{x}, \mathbf{y}) \sim \gamma}\!\left[\lVert \mathbf{x} - \mathbf{y} \rVert_2\right]$$

where $\Pi(p, q)$ is the set of all joint distributions (couplings) with marginals $p$ and $q$. Intuitively, $W_1$ is the minimum cost to transport mass from $p$ to $q$.

**Kantorovich-Rubinstein duality.** $W_1$ has a dual formulation:

$$W_1(p_{\text{data}}, p_g) = \sup_{\lVert f \rVert_L \leq 1} \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[f(\mathbf{x})] - \mathbb{E}_{\mathbf{x} \sim p_g}[f(\mathbf{x})]$$

where the supremum is over all 1-Lipschitz functions $f$ (i.e., $\lvert f(\mathbf{x}) - f(\mathbf{y}) \rvert \leq \lVert \mathbf{x} - \mathbf{y} \rVert_2$ for all $\mathbf{x},\mathbf{y}$).

**WGAN objective.** Replace the discriminator with a **critic** $f_{\boldsymbol{\phi}}$ constrained to be 1-Lipschitz:

$$\min_{\boldsymbol{\theta}} \max_{\boldsymbol{\phi}: \lVert f_{\boldsymbol{\phi}} \rVert_L \leq 1} \mathbb{E}_{p_{\text{data}}}[f_{\boldsymbol{\phi}}(\mathbf{x})] - \mathbb{E}_{p_z}[f_{\boldsymbol{\phi}}(G_{\boldsymbol{\theta}}(\mathbf{z}))]$$

**Advantages over standard GAN:**
1. $W_1$ is finite and continuous even when supports don't overlap — no vanishing gradients.
2. The critic provides a meaningful loss metric that correlates with sample quality.
3. The game is more stable: the critic can be trained to optimality without making generator training harder.

**For AI:** The Wasserstein distance is used as an evaluation metric in modern generative models (FID approximates it in feature space) and as an optimal transport objective in flow matching (Section 8.2).

### 5.5 Spectral Normalization and Gradient Penalty

**Weight clipping (original WGAN).** Arjovsky et al. (2017) enforced Lipschitz by clipping weights to $[-c, c]$. Simple but causes capacity underuse (weights concentrate at $\pm c$) and gradient explosion/vanishing for deep networks.

**Gradient Penalty (WGAN-GP, Gulrajani et al., 2017).** Instead of constraining weights, penalise the gradient norm directly:

$$\mathcal{L}_{\text{WGAN-GP}} = \mathbb{E}_{p_{\text{data}}}[f(\mathbf{x})] - \mathbb{E}_{p_g}[f(\tilde{\mathbf{x}})] + \lambda\, \mathbb{E}_{\hat{\mathbf{x}} \sim p_{\hat{\mathbf{x}}}}\!\left[(\lVert \nabla_{\hat{\mathbf{x}}} f(\hat{\mathbf{x}}) \rVert_2 - 1)^2\right]$$

where $\hat{\mathbf{x}} = \epsilon\mathbf{x} + (1-\epsilon)\tilde{\mathbf{x}}$, $\epsilon \sim \mathcal{U}(0,1)$ is sampled uniformly on the line between real and generated data. The penalty drives the gradient norm toward 1 everywhere, enforcing the Lipschitz constraint.

**Spectral normalization (Miyato et al., 2018).** Divide each weight matrix by its largest singular value (spectral norm): $\bar{W} = W / \sigma_1(W)$. This ensures the Lipschitz constant of each linear layer is $\leq 1$, and the Lipschitz constant of the whole network is $\leq 1$ (by composition). Spectral norm is estimated efficiently via power iteration. Used in BigGAN (Brock et al., 2019) and StyleGAN series.

### 5.6 Conditional GANs

**Conditional GAN (Mirza & Osindero, 2014).** Condition both generator and discriminator on class label $y$:
$$\min_G \max_D V(D,G) = \mathbb{E}_{\mathbf{x},y \sim p_{\text{data}}}[\log D(\mathbf{x},y)] + \mathbb{E}_{\mathbf{z},y}[\log(1-D(G(\mathbf{z},y),y))]$$

**Projection discriminator (Miyato & Koyama, 2018).** Instead of concatenating $y$ to the discriminator input, use the inner product of the class embedding with the discriminator's feature vector:

$$D(\mathbf{x}, y) = \phi(V^\top \mathbf{h}(\mathbf{x})) + \mathbf{v}_y^\top \mathbf{h}(\mathbf{x})$$

where $\mathbf{h}(\mathbf{x})$ is a feature map and $\mathbf{v}_y$ is a class embedding. This structure preserves the discriminator's role as a density ratio estimator while enabling class conditioning.

**For AI:** Conditional GANs are the forerunner of text-conditional image generation. The text encoder in DALL-E, Stable Diffusion, and Imagen plays the role of the condition $y$. The discriminator concept reappears in RLHF reward models.

---

## 6. Diffusion Models

### 6.1 Forward Process: Controlled Corruption

**Diffusion models** (Ho et al., DDPM 2020; Song & Ermon, NCSN 2020) define a **forward process** that gradually corrupts data $\mathbf{x}_0 \sim p_{\text{data}}$ into pure Gaussian noise over $T$ steps.

**Single-step transition:**
$$q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{1-\beta_t}\,\mathbf{x}_{t-1},\, \beta_t I\right)$$

where $\beta_1 < \beta_2 < \cdots < \beta_T$ is a **noise schedule** (typically small, e.g., $\beta_t \in [10^{-4}, 0.02]$).

**Forward marginal in closed form.** Define $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s=1}^t \alpha_s$. By composing $t$ Gaussian transitions (each is a linear map plus Gaussian noise, which compose as Gaussians):

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_t;\, \sqrt{\bar\alpha_t}\,\mathbf{x}_0,\, (1-\bar\alpha_t) I\right)$$

**Proof sketch.** By induction: $q(\mathbf{x}_1 \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\alpha_1}\,\mathbf{x}_0, \beta_1 I) = \mathcal{N}(\sqrt{\bar\alpha_1}\,\mathbf{x}_0, (1-\bar\alpha_1)I)$. Assume $q(\mathbf{x}_{t-1}\mid\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_{t-1}}\mathbf{x}_0, (1-\bar\alpha_{t-1})I)$. Then $\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\xi}$ with $\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\mathbf{x}_0 + \sqrt{1-\bar\alpha_{t-1}}\boldsymbol{\varepsilon}$, giving mean $\sqrt{\alpha_t\bar\alpha_{t-1}}\mathbf{x}_0 = \sqrt{\bar\alpha_t}\mathbf{x}_0$ and variance $\alpha_t(1-\bar\alpha_{t-1}) + \beta_t = (1-\bar\alpha_t)$. $\square$

**Reparameterisation for training.** Any noisy sample can be written as:
$$\mathbf{x}_t = \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

This allows sampling any $\mathbf{x}_t$ in one step without simulating the full chain.

**Noise schedules:**
- **Linear** (Ho et al., 2020): $\beta_t$ increases linearly from $\beta_1 = 10^{-4}$ to $\beta_T = 0.02$. Works but over-corrupts at the end.
- **Cosine** (Nichol & Dhariwal, 2021): $\bar\alpha_t = \cos^2(\pi t / (2T))$. Smoother corruption, better results for low-resolution images.
- **EDM** (Karras et al., 2022): parameterise in terms of noise levels $\sigma_t$ directly; $\bar\alpha_t = 1/(1+\sigma_t^2)$. Enables continuous-time analysis and optimal noise level sampling during training.

**For AI:** The forward process design directly impacts sample quality. Stable Diffusion uses the cosine schedule with $T=1000$ in pixel VAE latent space. The key insight: because $\bar\alpha_T \approx 0$, we have $q(\mathbf{x}_T \mid \mathbf{x}_0) \approx \mathcal{N}(\mathbf{0}, I)$ — the endpoint is pure noise, allowing generation by starting from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0},I)$.

### 6.2 Reverse Process: Learned Denoising

**Reverse process.** Generation runs the forward process backwards. The true reverse transition is:

$$q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\, \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0),\, \tilde\beta_t I\right)$$

where (by Bayes' theorem applied to the forward Gaussians):

$$\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}\mathbf{x}_t, \qquad \tilde\beta_t = \frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

This is tractable because conditioning on $\mathbf{x}_0$ makes everything Gaussian. The problem: at generation time, $\mathbf{x}_0$ is unknown.

**Learned reverse process.** Approximate $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t, \mathbf{x}_0)$ with a neural network:

$$p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t) = \mathcal{N}\!\left(\mathbf{x}_{t-1};\, \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t, t),\, \sigma_t^2 I\right)$$

where $\boldsymbol{\mu}_{\boldsymbol{\theta}}$ is a neural network (typically a U-Net) that predicts the denoised mean. Three equivalent parameterisations:

1. **Noise prediction** ($\boldsymbol{\varepsilon}$-prediction, Ho et al.): predict the noise $\boldsymbol{\varepsilon}$ added at step $t$, recover mean via $\boldsymbol{\mu}_{\boldsymbol{\theta}} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)\right)$
2. **Data prediction** ($\mathbf{x}_0$-prediction): directly predict $\hat{\mathbf{x}}_0$, then compute mean via $\tilde{\boldsymbol{\mu}}_t$
3. **Velocity prediction** ($\mathbf{v}$-prediction, Salimans & Ho): predict $\mathbf{v} = \sqrt{\bar\alpha_t}\boldsymbol{\varepsilon} - \sqrt{1-\bar\alpha_t}\mathbf{x}_0$

**For AI:** Stable Diffusion XL uses $\mathbf{v}$-prediction, which is more numerically stable at very low noise levels and better for high-frequency details. DDPM/DDIM use $\boldsymbol{\varepsilon}$-prediction. $\mathbf{x}_0$-prediction is used in some video diffusion models (Imagen Video).

### 6.3 Variational Lower Bound for Diffusion

The ELBO for the diffusion model decomposes into interpretable terms:

$$-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0) \leq \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_T \mid \mathbf{x}_0) \| p(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_{\mathrm{KL}}(q(\mathbf{x}_{t-1} \mid \mathbf{x}_t, \mathbf{x}_0) \| p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1} \mid \mathbf{x}_t))}_{L_{t-1}} - \underbrace{\log p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_1)}_{L_0}$$

- **$L_T$ (prior matching):** KL between the noised data and the pure noise prior. With large enough $T$, $q(\mathbf{x}_T \mid \mathbf{x}_0) \approx \mathcal{N}(\mathbf{0},I) = p(\mathbf{x}_T)$, so $L_T \approx 0$.
- **$L_{t-1}$ (denoising matching, $t=2,\ldots,T$):** KL between the true and learned reverse transitions. Each term compares $\boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ to the target $\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)$ — this is the training signal.
- **$L_0$ (reconstruction):** Log-likelihood of the original data given the slightly denoised $\mathbf{x}_1$.

For Gaussian transitions of equal variance, the KL $L_{t-1}$ simplifies to a squared error in the mean, leading directly to the simplified objective.

### 6.4 The Simplified Denoising Objective

**DDPM simplified loss (Ho et al., 2020).** Dropping weighting terms that empirically hurt performance:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\varepsilon}}\!\left[\lVert \boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\underbrace{\sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon}}_{\mathbf{x}_t},\, t) \rVert_2^2\right]$$

where $t \sim \mathcal{U}\{1,\ldots,T\}$ and $\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},I)$.

**Interpretation.** The model $\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}$ receives a noisy image $\mathbf{x}_t$ and the noise level $t$, and must predict the noise that was added. This is a classical **denoising** problem, but applied across all noise levels simultaneously.

**Why it works.** Predicting $\boldsymbol{\varepsilon}$ is equivalent to predicting the score (see §6.5). The simplification removes the time-dependent weighting $w(t)$ of the full ELBO, effectively giving higher weight to intermediate noise levels (where the model's predictions matter most for sample quality) and lower weight to very clean or very noisy steps.

**Parameterisations comparison:**
- $\boldsymbol{\varepsilon}$-prediction: $\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ — standard for DDPM/DDIM
- $\mathbf{x}_0$-prediction: $\hat{\mathbf{x}}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = (\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon}_\theta)/\sqrt{\bar\alpha_t}$ — equivalent, numerically different
- $\mathbf{v}$-prediction: $\mathbf{v}_{\boldsymbol{\theta}} = \sqrt{\bar\alpha_t}\boldsymbol{\varepsilon} - \sqrt{1-\bar\alpha_t}\mathbf{x}_0$ — used in Stable Diffusion XL

### 6.5 Score Matching

**Score function.** The **score** of a distribution is $\nabla_{\mathbf{x}} \log p(\mathbf{x})$ — the gradient of the log-density. It points toward regions of higher probability without requiring the normalising constant.

**Score-based generative models (Song & Ermon, 2019).** Train a neural network $s_{\boldsymbol{\theta}}(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log p(\mathbf{x})$ and generate samples using **Langevin dynamics**.

**Explicit score matching (Hyvärinen, 2005).** The score matching objective:
$$\mathcal{L}_{\text{ESM}} = \mathbb{E}_{p(\mathbf{x})}\!\left[\lVert s_{\boldsymbol{\theta}}(\mathbf{x}) - \nabla_\mathbf{x} \log p(\mathbf{x}) \rVert_2^2\right]$$

is intractable because $\nabla_\mathbf{x}\log p(\mathbf{x})$ is unknown. But using integration by parts, it reduces to:
$$= \mathbb{E}_{p(\mathbf{x})}\!\left[\operatorname{tr}(\nabla_\mathbf{x} s_{\boldsymbol{\theta}}(\mathbf{x})) + \frac{1}{2}\lVert s_{\boldsymbol{\theta}}(\mathbf{x}) \rVert_2^2\right] + \text{const}$$

Still expensive ($O(d)$ Jacobian-vector products for the trace term).

**Denoising Score Matching (DSM, Vincent, 2011).** Add noise $\tilde{\mathbf{x}} = \mathbf{x} + \boldsymbol{\xi}$, $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0},\sigma^2 I)$, and train to match the score of the noisy distribution:

$$\mathcal{L}_{\text{DSM}} = \mathbb{E}_{\mathbf{x},\tilde{\mathbf{x}}}\!\left[\left\lVert s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}) - \nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}} \mid \mathbf{x}) \right\rVert_2^2\right]$$

The conditional score is tractable: $\nabla_{\tilde{\mathbf{x}}} \log q_\sigma(\tilde{\mathbf{x}}\mid\mathbf{x}) = -(\tilde{\mathbf{x}} - \mathbf{x})/\sigma^2 = -\boldsymbol{\xi}/\sigma^2$.

**Connection to DDPM.** At noise level $\sigma_t = \sqrt{1-\bar\alpha_t}$, the conditional score is:
$$\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t \mid \mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t} = -\frac{\boldsymbol{\varepsilon}}{\sqrt{1-\bar\alpha_t}}$$

So the noise prediction network satisfies $\boldsymbol{\varepsilon}_{\boldsymbol{\theta}} \approx -\sqrt{1-\bar\alpha_t}\,s_{\boldsymbol{\theta}}$ — DDPM noise prediction is equivalent to scaled score matching.

**SDE framework (Song et al., 2021).** The forward process can be written as an SDE:
$$d\mathbf{x} = f(\mathbf{x}, t)\,dt + g(t)\,d\mathbf{W}$$

The reverse-time SDE (Anderson, 1982) is:
$$d\mathbf{x} = [f(\mathbf{x}, t) - g(t)^2 \nabla_\mathbf{x} \log p_t(\mathbf{x})]\,dt + g(t)\,d\bar{\mathbf{W}}$$

where $\bar{\mathbf{W}}$ is a backward Wiener process. Replacing the score with a learned $s_{\boldsymbol{\theta}}(\mathbf{x},t)$ and solving the reverse SDE gives a continuous-time generative model that unifies DDPM, NCSN, and flow models.

### 6.6 DDIM: Deterministic Sampling

**DDIM (Song et al., 2021)** derives a **non-Markovian** forward process that has the same marginals $q(\mathbf{x}_t \mid \mathbf{x}_0)$ as DDPM but allows deterministic sampling:

$$\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\underbrace{\left(\frac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t)}{\sqrt{\bar\alpha_t}}\right)}_{\hat{\mathbf{x}}_0 \text{ prediction}} + \sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\,\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t) + \sigma_t\boldsymbol{\varepsilon}$$

Setting $\sigma_t = 0$ gives a **deterministic** ODE that maps $\mathbf{x}_T \mapsto \mathbf{x}_0$:

$$\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\frac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}}{\sqrt{\bar\alpha_t}} + \sqrt{1-\bar\alpha_{t-1}}\,\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t)$$

**Advantages:**
1. **Fewer steps:** Because the sampling is now solving an ODE (not a stochastic process), we can use larger step sizes — 50 steps instead of 1000, with acceptable quality loss.
2. **Deterministic:** Same $\mathbf{x}_T$ always gives the same $\mathbf{x}_0$ — useful for interpolation and inversion.
3. **Invertible:** Can encode real images to noise by running the ODE forward, enabling image editing.

**For AI:** DDIM is the standard sampler in Stable Diffusion and most production diffusion systems. DDIM inversion enables ControlNet-style editing: encode the real image to noise, then decode with a modified condition.

### 6.7 Guidance: Classifier and Classifier-Free

**Classifier guidance (Dhariwal & Nichol, 2021).** To generate class-conditional samples, steer the reverse process with a pretrained classifier $p_{\boldsymbol{\phi}}(y \mid \mathbf{x}_t)$:

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t \mid y) = \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t) + \gamma \nabla_{\mathbf{x}_t}\log p_{\boldsymbol{\phi}}(y \mid \mathbf{x}_t)$$

The guidance scale $\gamma > 1$ amplifies the classifier signal, trading diversity for class fidelity. The modified noise prediction is:

$$\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{\theta}} = \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) - \sqrt{1-\bar\alpha_t}\,\gamma\,\nabla_{\mathbf{x}_t}\log p_{\boldsymbol{\phi}}(y \mid \mathbf{x}_t)$$

**Drawback:** Requires a separate classifier trained on noisy images at all noise levels $t$.

**Classifier-free guidance (Ho & Salimans, 2022).** Train a single network that can be conditioned ($y$ given) or unconditional ($y = \emptyset$, null condition). At inference, interpolate between conditional and unconditional predictions:

$$\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{\theta}}(\mathbf{x}_t, y) = (1 + w)\,\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, y) - w\,\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, \emptyset)$$

Equivalently (rearranging):
$$\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{\theta}} = \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, \emptyset) + (1 + w)[\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, y) - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, \emptyset)]$$

With $w = 0$ we get the conditional model; with $w > 0$ we amplify the conditional direction, trading diversity for prompt adherence.

**Connection to classifier guidance.** CFG implicitly uses an implicit classifier: $(1+w)\boldsymbol{\varepsilon}_\theta(y) - w\boldsymbol{\varepsilon}_\theta(\emptyset)$ corresponds to guidance by $\log[p_\theta(y|\mathbf{x}_t)/p_\theta(\emptyset|\mathbf{x}_t)]$.

**For AI:** CFG is used in essentially all modern text-to-image systems (Stable Diffusion, DALL-E 3, Midjourney, Flux). Guidance scale $w$ is the "prompt strength" parameter users control — higher $w$ gives sharper adherence to the text prompt but reduces diversity.

---

## 7. Energy-Based Models

### 7.1 The Boltzmann Distribution

**Energy-based models (EBMs)** define the probability density through an **energy function** $E_{\boldsymbol{\theta}}: \mathcal{X} \to \mathbb{R}$:

$$p_{\boldsymbol{\theta}}(\mathbf{x}) = \frac{e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}}{Z_{\boldsymbol{\theta}}}, \qquad Z_{\boldsymbol{\theta}} = \int e^{-E_{\boldsymbol{\theta}}(\mathbf{x})}\,d\mathbf{x}$$

Low energy $\Leftrightarrow$ high probability. $Z_{\boldsymbol{\theta}}$ is the **partition function** — a normalising constant that ensures $p_{\boldsymbol{\theta}}$ integrates to 1.

**The intractability problem.** The MLE gradient involves:

$$\nabla_{\boldsymbol{\theta}} \log p_{\boldsymbol{\theta}}(\mathbf{x}) = -\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x}) - \nabla_{\boldsymbol{\theta}} \log Z_{\boldsymbol{\theta}}$$

$$\nabla_{\boldsymbol{\theta}} \log Z_{\boldsymbol{\theta}} = -\mathbb{E}_{p_{\boldsymbol{\theta}}(\mathbf{x})}[\nabla_{\boldsymbol{\theta}} E_{\boldsymbol{\theta}}(\mathbf{x})]$$

Computing $\nabla_{\boldsymbol{\theta}} \log Z_{\boldsymbol{\theta}}$ requires samples from the model itself — the model's own distribution is needed to train the model.

**Contrastive Divergence (Hinton, 2002).** Approximate $\mathbb{E}_{p_{\boldsymbol{\theta}}}[\nabla_{\boldsymbol{\theta}} E]$ using a short MCMC chain starting from data:

1. Start from a data point $\mathbf{x}^+$
2. Run $k$ steps of Gibbs sampling or Langevin dynamics to get $\mathbf{x}^-$ ("fantasy particles")
3. Update: $\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} - \eta(-\nabla_{\boldsymbol{\theta}} E(\mathbf{x}^+) + \nabla_{\boldsymbol{\theta}} E(\mathbf{x}^-))$

CD-1 ($k=1$) works surprisingly well for RBMs but has known bias for deep models.

**For AI:** EBMs provide a flexible framework for any discriminative function. The reward model in RLHF can be interpreted as an EBM: $r_{\boldsymbol{\theta}}(\mathbf{x}, y)$ plays the role of $-E_{\boldsymbol{\theta}}$ where higher reward = lower energy = more probable response.

### 7.2 Score-Based Training Without MCMC

The key insight of **score-based models** (Song & Ermon, 2019) is that we can train the score function $s_{\boldsymbol{\theta}}(\mathbf{x}) \approx \nabla_\mathbf{x} \log p(\mathbf{x})$ without ever computing $Z_{\boldsymbol{\theta}}$.

The score is related to the energy: $\nabla_\mathbf{x} \log p_{\boldsymbol{\theta}}(\mathbf{x}) = -\nabla_\mathbf{x} E_{\boldsymbol{\theta}}(\mathbf{x})$. Note the gradient is with respect to $\mathbf{x}$, not $\boldsymbol{\theta}$ — $Z_{\boldsymbol{\theta}}$ drops out because it is constant in $\mathbf{x}$.

**Noise conditional score network (NCSN).** Perturb data at $L$ noise scales $\sigma_1 < \cdots < \sigma_L$ and train:

$$\mathcal{L}_{\text{NCSN}} = \sum_{l=1}^L \lambda(\sigma_l)\, \mathbb{E}_{\mathbf{x}, \tilde{\mathbf{x}}}\!\left[\left\lVert s_{\boldsymbol{\theta}}(\tilde{\mathbf{x}}, \sigma_l) + \frac{\tilde{\mathbf{x}} - \mathbf{x}}{\sigma_l^2} \right\rVert_2^2\right]$$

At large $\sigma_l$ the score guides the chain from noise to the data manifold; at small $\sigma_l$ it refines fine details.

### 7.3 Langevin Dynamics Sampling

Given the score $\nabla_\mathbf{x} \log p(\mathbf{x})$, **Langevin dynamics** generates samples by gradient ascent on the log-density with injected noise:

$$\mathbf{x}_{k+1} = \mathbf{x}_k + \frac{\eta}{2} \nabla_{\mathbf{x}} \log p(\mathbf{x}_k) + \sqrt{\eta}\, \boldsymbol{\xi}_k, \qquad \boldsymbol{\xi}_k \sim \mathcal{N}(\mathbf{0}, I)$$

Under mild conditions, as $\eta \to 0$ and $k \to \infty$, this Markov chain converges to $p(\mathbf{x})$. The noise term prevents getting stuck at local maxima of $\log p$.

**Connection to reverse diffusion.** The discrete-time diffusion reverse step $\mathbf{x}_{t-1} = \mathbf{x}_t - \frac{\beta_t}{2}s_{\boldsymbol{\theta}}(\mathbf{x}_t, t) + \sqrt{\beta_t}\boldsymbol{\xi}$ is Langevin dynamics applied to the score of $p_t(\mathbf{x}_t)$. Score-based diffusion and Langevin sampling are two views of the same underlying process.

**For AI:** Langevin dynamics underlies the sampling procedure of early score-based models (Song & Ermon, 2019). The connection was key to understanding that DDPM sampling IS score-based sampling — the noise prediction network is a score network in disguise.

---

## 8. Flow Matching

### 8.1 Conditional Flow Matching

**Flow matching (Lipman et al., 2022)** trains a continuous normalizing flow by directly regressing its velocity field — bypassing the expensive likelihood computation of maximum likelihood CNF training.

**Setup.** We want to learn a time-dependent velocity field $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t)$ that defines an ODE:

$$\frac{d\mathbf{x}(t)}{dt} = \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}(t), t), \qquad t \in [0,1]$$

whose flow $\phi_t: \mathbf{x}(0) \mapsto \mathbf{x}(t)$ transports the base distribution $p_0 = \mathcal{N}(\mathbf{0},I)$ to the data distribution $p_1 = p_{\text{data}}$.

**Flow matching objective (FM).** Regress $\mathbf{v}_{\boldsymbol{\theta}}$ onto the **marginal vector field** $u_t(\mathbf{x})$ that generates the probability path $p_t$:

$$\mathcal{L}_{\text{FM}} = \mathbb{E}_{t, \mathbf{x} \sim p_t(\mathbf{x})}\!\left[\lVert \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t) - u_t(\mathbf{x}) \rVert_2^2\right]$$

**Problem:** Computing $u_t(\mathbf{x})$ requires marginalising over all possible data points — still intractable.

**Conditional flow matching (CFM).** Condition on a specific data point $\mathbf{x}_1 \sim p_{\text{data}}$. Define a conditional probability path $p_t(\mathbf{x} \mid \mathbf{x}_1)$ and its associated conditional vector field $u_t(\mathbf{x} \mid \mathbf{x}_1)$. The **CFM objective**:

$$\mathcal{L}_{\text{CFM}} = \mathbb{E}_{t, \mathbf{x}_1 \sim p_{\text{data}}, \mathbf{x} \sim p_t(\mathbf{x}\mid\mathbf{x}_1)}\!\left[\lVert \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}, t) - u_t(\mathbf{x} \mid \mathbf{x}_1) \rVert_2^2\right]$$

**Key theorem.** $\mathcal{L}_{\text{FM}}$ and $\mathcal{L}_{\text{CFM}}$ have the same gradient with respect to $\boldsymbol{\theta}$, so training with $\mathcal{L}_{\text{CFM}}$ is equivalent to training with $\mathcal{L}_{\text{FM}}$.

**Simplest path: Linear interpolation.** Choose $p_t(\mathbf{x}\mid\mathbf{x}_1) = \mathcal{N}((1-t)\mathbf{x}_0 + t\mathbf{x}_1, \sigma^2 I)$ with $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0},I)$. The conditional vector field is:

$$u_t(\mathbf{x} \mid \mathbf{x}_1) = \frac{\mathbf{x}_1 - (1-t)\mathbf{x}_0}{1-t+\sigma^2/(1-t)} \approx \mathbf{x}_1 - \mathbf{x}_0 \quad \text{(for small } \sigma\text{)}$$

So the model learns to predict the direction from noise to data — a constant velocity field along straight lines!

**Training algorithm:**
1. Sample $\mathbf{x}_1 \sim p_{\text{data}}$, $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0},I)$, $t \sim \mathcal{U}(0,1)$
2. Compute $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$
3. Loss: $\lVert \mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t) - (\mathbf{x}_1 - \mathbf{x}_0) \rVert_2^2$

**For AI:** Flux (Black Forest Labs, 2024), Stable Diffusion 3 (Esser et al., 2024), and MovieGen (Meta, 2024) all use flow matching with the MMDiT (multimodal diffusion transformer) backbone. Flow matching trains faster than diffusion and produces trajectories with less curvature, enabling high-quality samples in fewer ODE steps.

### 8.2 Optimal Transport Paths

**The curvature problem.** With independent coupling ($\mathbf{x}_0$ and $\mathbf{x}_1$ drawn independently), the marginal velocity field $u_t(\mathbf{x})$ at intermediate times is curved — multiple straight paths cross at the same $(\mathbf{x},t)$ point, and the marginal field averages them. Curved fields require more ODE solver steps.

**OT-CFM (Tong et al., 2023).** Use the **minibatch optimal transport coupling** instead of independent coupling:

$$\gamma^* = \arg\min_{\gamma \in \Pi(p_0, p_{\text{data}})} \mathbb{E}_{(\mathbf{x}_0, \mathbf{x}_1)\sim\gamma}\!\left[\lVert\mathbf{x}_0 - \mathbf{x}_1\rVert_2^2\right]$$

Each noise sample $\mathbf{x}_0$ is matched to its nearest data point $\mathbf{x}_1$ (approximately, via Sinkhorn algorithm on minibatches). The paths $(1-t)\mathbf{x}_0 + t\mathbf{x}_1$ are then nearly parallel, making the marginal velocity nearly constant and reducing curvature.

**Advantages:**
- Straighter trajectories → fewer NFE (number of function evaluations) at inference
- Better sample quality at the same NFE budget
- Training stability (lower variance gradients)

### 8.3 Rectified Flows

**Rectified flows (Liu et al., 2022)** learn a linear ODE between $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0},I)$ and $\mathbf{x}_1 \sim p_{\text{data}}$ using independent coupling:

$$\min_{\boldsymbol{\theta}} \mathbb{E}_{\mathbf{x}_0, \mathbf{x}_1, t}\!\left[\lVert \mathbf{v}_{\boldsymbol{\theta}}((1-t)\mathbf{x}_0 + t\mathbf{x}_1, t) - (\mathbf{x}_1 - \mathbf{x}_0) \rVert_2^2\right]$$

**Reflow procedure.** After training $\mathbf{v}_{\boldsymbol{\theta}}$, generate $n$ pairs $(\mathbf{z}_0, \mathbf{z}_1)$ by running the ODE from $\mathbf{z}_0 \sim \mathcal{N}(\mathbf{0},I)$ to $\mathbf{z}_1 \approx p_{\text{data}}$. Retrain on these pairs as a new coupling. Repeated reflow straightens the trajectories — after $k$ reflowing iterations, the flow converges to a transport map with nearly straight paths.

**For AI:** Stable Diffusion 3 and Flux use rectified flows with one reflow iteration. The key observation: after reflow, single-step generation (Euler method with 1 step) produces competitive samples, enabling real-time generation.

### 8.4 Flow Matching vs Diffusion

```
FLOW MATCHING VS DIFFUSION: COMPARISON
════════════════════════════════════════════════════════════════════════

  Property          │ Diffusion (DDPM)        │ Flow Matching (CFM)
  ──────────────────┼─────────────────────────┼──────────────────────────
  Trajectory        │ Stochastic SDE          │ Deterministic ODE
  Training target   │ Noise εθ or score sθ    │ Velocity vθ = x1 - x0
  Forward process   │ Fixed Markov chain      │ Interpolation path
  Trajectory curve  │ High (crosses itself)   │ Low (nearly straight)
  NFE at inference  │ 50-1000 steps           │ 1-50 steps
  Likelihood        │ ELBO lower bound        │ Exact (via ODE adjoint)
  Sampling noise    │ Inherent (SDE)          │ Optional (add η√dt ξ)
  State of art      │ SD1.5, SD2.1, SDXL      │ SD3, Flux, MovieGen
  Key paper         │ Ho et al., 2020         │ Lipman et al., 2022
  ──────────────────┴─────────────────────────┴──────────────────────────
  VERDICT: Flow matching dominates for new models (2024+); diffusion
  models still widely deployed from 2020-2023 wave

════════════════════════════════════════════════════════════════════════
```

---

## 9. Latent Diffusion and Modern Architectures

### 9.1 Latent Diffusion Models

**Key idea (Rombach et al., 2022):** Run the diffusion process in the latent space of a pretrained VAE rather than in pixel space. This provides a $4\times$ to $8\times$ compression of spatial resolution, dramatically reducing compute.

**Architecture:**
1. **VAE encoder**: $\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h/f \times w/f \times c}$ with downsampling factor $f$ (typically $f=8$)
2. **Diffusion in latent space**: Forward/reverse process on $\mathbf{z}$, not $\mathbf{x}$
3. **VAE decoder**: $\hat{\mathbf{x}} = \mathcal{D}(\mathbf{z}_0)$ after denoising

**Why it works.** Natural images have strong spatial correlations — most information lies in a lower-dimensional perceptual space. The VAE learns this compression. The diffusion model only needs to capture the remaining, semantically rich variation.

**Perceptual compression loss.** The VAE in LDM is trained with a combination of reconstruction loss, KL penalty, and perceptual loss $\mathcal{L}_{\text{perceptual}} = \lVert \phi(\mathbf{x}) - \phi(\hat{\mathbf{x}}) \rVert_2^2$ where $\phi$ is a VGG feature extractor. This ensures the latent space captures perceptually meaningful features rather than per-pixel statistics.

**Cross-attention conditioning.** Text conditions are injected via cross-attention in the U-Net backbone. The text encoder (CLIP or T5) produces token embeddings $\mathbf{c}$; the U-Net features act as queries attending to text as keys/values:
$$\text{Attention}(Q,K,V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V, \quad Q = \phi_{\text{Q}}(\mathbf{z}_t),\; K = V = \phi_{\text{KV}}(\mathbf{c})$$

**For AI:** Stable Diffusion 1.x, 2.x use LDM with a U-Net backbone. SDXL upgrades to a larger U-Net with two text encoders (OpenCLIP + CLIP-ViT-L). Stable Diffusion 3 replaces U-Net with a **multimodal diffusion transformer (MMDiT)** and switches from diffusion to flow matching.

### 9.2 U-Net with Cross-Attention

The **U-Net** (Ronneberger et al., 2015) is the standard backbone for diffusion models. It processes spatial features at multiple resolutions with skip connections:

**Architecture:**
- **Encoder path**: Sequence of ResBlocks + downsampling (stride-2 conv). Resolution decreases $H \to H/2 \to H/4 \to \cdots$
- **Bottleneck**: Self-attention + ResBlocks at lowest resolution
- **Decoder path**: ResBlocks + upsampling + skip connections from encoder
- **Time conditioning**: Sinusoidal time embedding added to each ResBlock via AdaGN (adaptive group normalisation)
- **Cross-attention**: Text condition $\mathbf{c}$ injected at every resolution level

**Why U-Net for diffusion?** The skip connections allow the network to easily preserve fine spatial details (passed through skip) while the bottleneck focuses on global structure. The multi-scale processing matches the multi-scale nature of image generation: coarse structure (semantic layout) at low resolution, fine texture at high resolution.

**MMDiT (Esser et al., 2024 for SD3/Flux).** Replaces U-Net with a transformer that jointly processes image patches and text tokens in a shared sequence, with separate weight streams for each modality. The key innovation: image and text tokens attend to each other bidirectionally (no masking), enabling richer text-image alignment.

### 9.3 Discrete Diffusion for Language

Standard diffusion models operate on continuous variables, making them ill-suited for discrete tokens. **Discrete diffusion** adapts the framework to categorical distributions.

**Masked diffusion (MDLM, Sahoo et al., 2024).** Define a forward process that gradually masks tokens:
$$q(x_t \mid x_0) = \begin{cases} x_0 & \text{with prob } \bar\alpha_t \\ [\text{MASK}] & \text{with prob } 1-\bar\alpha_t \end{cases}$$

The reverse process predicts the original token given the masked sequence: $p_{\boldsymbol{\theta}}(x_0 \mid x_t)$. This is closely related to BERT-style masked language modelling.

**Training objective:** Cross-entropy over masked positions:
$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \mathbf{x}_t}\!\left[\sum_{i:\, x_t^i = \text{MASK}} -\log p_{\boldsymbol{\theta}}(x_0^i \mid \mathbf{x}_t)\right]$$

**Absorbing-state processes.** More generally, define a Markov chain on the vocabulary with an absorbing "mask" state. The forward rates control how quickly each token is masked; the reverse process denoises all positions in parallel.

**Advantage over autoregressive models:** Generation can run in parallel (all tokens denoised simultaneously) with iterative refinement, rather than left-to-right token-by-token. This allows arbitrary-length generation in a fixed number of steps (typically 10-50).

**For AI:** MDLM, PLAID, and similar discrete diffusion models show competitive perplexity to autoregressive models while supporting bidirectional conditioning and non-sequential generation. As of 2025, they are an active research area for LLM generation.

### 9.4 Consistency Models

**Consistency models (Song et al., 2023)** eliminate the need for multi-step ODE solving by training a network that maps any point on the ODE trajectory directly to the trajectory's endpoint ($\mathbf{x}_0$):

$$f_{\boldsymbol{\theta}}(\mathbf{x}_t, t) \approx \mathbf{x}_0 \text{ for all } t \in [0, T]$$

**Self-consistency property.** Points on the same ODE trajectory must map to the same $\mathbf{x}_0$:

$$f_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = f_{\boldsymbol{\theta}}(\mathbf{x}_{t'}, t') \quad \text{if } \mathbf{x}_t \text{ and } \mathbf{x}_{t'} \text{ lie on the same ODE trajectory}$$

**Consistency distillation.** Given a pretrained diffusion/flow model that defines the ODE trajectory, train:

$$\mathcal{L}_{\text{CD}} = \mathbb{E}\!\left[\lVert f_{\boldsymbol{\theta}}(\mathbf{x}_{t_n}, t_n) - f_{\boldsymbol{\theta}^-}(\mathbf{x}_{t_{n+1}}, t_{n+1}) \rVert_2^2\right]$$

where $\mathbf{x}_{t_{n+1}}$ is one ODE step from $\mathbf{x}_{t_n}$ using the pretrained model, and ${\boldsymbol{\theta}^-}$ is an exponential moving average (EMA) of $\boldsymbol{\theta}$ for stability.

**Consistency training.** Train without a pretrained model by using the ODE solver to provide $\mathbf{x}_{t_{n+1}}$ from $\mathbf{x}_{t_n}$ during training.

**For AI:** Consistency models enable 1-step or 2-step generation with quality competitive with 20-50 step diffusion. LCM (Latent Consistency Model) applies this to Stable Diffusion, enabling interactive-rate generation (6+ images/second on A100 GPUs). This is critical for real-time applications.

---

## 10. Evaluation Metrics

### 10.1 Inception Score

**IS (Salimans et al., 2016).** Uses a pretrained Inception-v3 classifier with conditional distribution $p(y \mid \mathbf{x})$ (class probabilities) and marginal $p(y) = \mathbb{E}_\mathbf{x}[p(y\mid\mathbf{x})]$:

$$\mathrm{IS} = \exp\!\left(\mathbb{E}_{\mathbf{x} \sim p_g}\!\left[D_{\mathrm{KL}}(p(y\mid\mathbf{x}) \| p(y))\right]\right)$$

**Interpretation:** High IS means each generated image is clearly classifiable (high confidence $p(y\mid\mathbf{x})$, measuring **fidelity**) AND the marginal distribution is diverse (high entropy $H(y)$, measuring **diversity**). The $\mathrm{IS}$ combines both:
$$\mathrm{IS} = \exp(H(y) - H(y\mid\mathbf{x})) = \exp(I(y;\mathbf{x}))$$

where $I(y;\mathbf{x})$ is the mutual information between class labels and generated images.

**Failures:** IS cannot detect memorisation (a model generating the training set perfectly gets a high IS). It only evaluates diversity in ImageNet label space — two models with identical class diversity but different within-class quality score the same. It is sensitive to the reference classifier's training data.

### 10.2 Fréchet Inception Distance

**FID (Heusel et al., 2017)** is the dominant metric for image generation quality. Extract Inception-v3 features $\phi(\mathbf{x}) \in \mathbb{R}^{2048}$ for real and generated images, fit Gaussians, and compute the Fréchet distance:

$$\mathrm{FID} = \lVert \boldsymbol{\mu}_r - \boldsymbol{\mu}_g \rVert_2^2 + \operatorname{tr}\!\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

where $(\boldsymbol{\mu}_r, \Sigma_r)$ and $(\boldsymbol{\mu}_g, \Sigma_g)$ are the mean and covariance of real and generated features. Lower FID is better (0 = perfect match).

**Matrix square root.** The term $(\Sigma_r \Sigma_g)^{1/2}$ is the **matrix geometric mean** (not element-wise square root). Computed via: $\Sigma_r^{1/2}[\Sigma_r^{1/2}\Sigma_g\Sigma_r^{1/2}]^{1/2}\Sigma_r^{-1/2}$ or eigendecomposition.

**FID as Wasserstein-2.** FID equals the Wasserstein-2 distance between the Gaussian approximations: $\mathrm{FID} = W_2^2(\mathcal{N}(\boldsymbol{\mu}_r, \Sigma_r), \mathcal{N}(\boldsymbol{\mu}_g, \Sigma_g))$.

**Practical notes:**
- Requires 50,000 real and generated samples for stable estimates
- Sensitive to image preprocessing (especially resize/crop choices)
- Sensitive to the Inception feature extractor version
- Does not account for text-image alignment (CLIP score is used separately)

### 10.3 Precision and Recall

**Precision and recall for generative models (Kynkäänniemi et al., 2019).** Decompose FID into fidelity and diversity components:

- **Precision:** Fraction of generated samples that fall within the support of the real distribution: $\Pr(\mathbf{x}_g \in \text{support}(p_r))$
- **Recall:** Fraction of the real distribution that is covered by the generated distribution: $\Pr(\mathbf{x}_r \in \text{support}(p_g))$

**Implementation.** Use $k$-nearest-neighbour manifold estimation: a sample is "in the manifold" of distribution $p$ if it is within distance $d$ of its $k$-th nearest neighbour in a sample from $p$.

$$\text{Precision} = \frac{1}{\lvert X_g \rvert}\sum_{\mathbf{x}_g \in X_g} \mathbf{1}[\mathbf{x}_g \in \text{manifold}(X_r)]$$

$$\text{Recall} = \frac{1}{\lvert X_r \rvert}\sum_{\mathbf{x}_r \in X_r} \mathbf{1}[\mathbf{x}_r \in \text{manifold}(X_g)]$$

**Tradeoff.** A model can have high precision (generates only realistic images) with low recall (misses many modes of the data), or vice versa. Guidance scale in diffusion models controls this tradeoff: high guidance → high precision, low recall.

### 10.4 Likelihood and Bits-Per-Dimension

**Bits per dimension (BPD).** For tractable density models (flows, autoregressive), the standard metric is:

$$\mathrm{BPD} = -\frac{\log_2 p_{\boldsymbol{\theta}}(\mathbf{x})}{d}$$

where $d$ is the number of dimensions (e.g., $d = 3 \times 256 \times 256 = 196608$ for a $256\times 256$ image). Lower BPD = better compression = better model.

**Connection to compression.** By Shannon's theorem, a model achieving BPD nats per dimension could compress the data to $\mathrm{BPD}$ bits per pixel — BPD directly measures compression efficiency.

**ELBO-BPD for diffusion.** Diffusion models compute a BPD estimate via the ELBO (the sum of KL terms), providing a lower bound on the true NLL. DDPM achieves 3.75 BPD on CIFAR-10, competitive with the best autoregressive models.

**Why GANs have no likelihood.** GANs are implicit models: $G(\mathbf{z})$ maps noise to data but there is no formula for $p_g(\mathbf{x})$. This makes likelihood-based comparison impossible. FID/IS/precision/recall are the primary evaluation metrics.

---

## 11. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Writing the ELBO as $\mathbb{E}_q[\log p] + D_{\mathrm{KL}}$ | The KL term in the ELBO is subtracted, not added. The ELBO = reconstruction $-$ KL. Adding KL gives a quantity above the log-likelihood. | $\mathcal{L} = \mathbb{E}_{q}[\log p(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q\|p)$ |
| 2 | KL direction: writing $D_{\mathrm{KL}}(p\|q)$ when deriving the ELBO | The ELBO derivation requires $D_{\mathrm{KL}}(q_\phi\|p_\theta(\mathbf{z}\mid\mathbf{x}))$; the KL term in the ELBO objective is $D_{\mathrm{KL}}(q_\phi\|p(\mathbf{z}))$. Reversing the direction changes the meaning and the optimisation. | Always write which direction: $D_{\mathrm{KL}}(\text{approximate}\|\text{prior})$ in the ELBO penalty |
| 3 | Using score-function gradient for the ELBO encoder | Score function (REINFORCE) is unbiased but has extremely high variance for continuous $\mathbf{z}$. Training will be very slow or fail entirely. | Use reparameterisation: $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},I)$ |
| 4 | Coupling layer inverse: computing $s(y_a)$ instead of $s(z_a)$ | Since $y_a = z_a$ (identity on the first half), $s(y_a) = s(z_a)$ — no bug here. But computing $s$ and $t$ on $y_b$ instead of $y_a$ is wrong: the inverse uses $x_1 = y_a$ to compute $s, t$, not $x_2 = y_b$. | Inverse: $z_b = (y_b - t(y_a)) \odot \exp(-s(y_a))$ |
| 5 | GAN non-saturation: using $\min_G \log(1-D(G(\mathbf{z})))$ | When $D$ is good, $D(G(\mathbf{z})) \approx 0$, so $\log(1-D)\approx 0$ and gradients vanish. The generator cannot learn. | Use $\max_G \log D(G(\mathbf{z}))$ (non-saturating) — same Nash equilibrium, better gradients early in training |
| 6 | WGAN gradient penalty: interpolating between two generated samples | The GP should be on points between real and generated data. Interpolating between two generated samples gives wrong penalty landscape. | $\hat\mathbf{x} = \epsilon\mathbf{x}_r + (1-\epsilon)G(\mathbf{z})$, $\epsilon\sim\mathcal{U}(0,1)$ |
| 7 | Diffusion forward: using $\bar\alpha_t = 1-t$ for linear noise schedule | The correct $\bar\alpha_t = \prod_{s=1}^t(1-\beta_s) \neq 1 - t\beta_{\max}$. The product compounds noise differently from a linear decrease. | Precompute $\bar\alpha_t = \prod_{s=1}^t(1-\beta_s)$ and cache; or use $\bar\alpha_t = \cos^2(\pi t/(2T))$ |
| 8 | Confusing $\mathbf{x}_0$-prediction and $\boldsymbol{\varepsilon}$-prediction | Both predict a reconstruction of $\mathbf{x}_0$, but with different loss weightings and numerical properties. $\boldsymbol{\varepsilon}$-prediction can have large magnitudes at high noise; $\mathbf{x}_0$-prediction saturates near 0 at low noise. | Be explicit: $\hat\mathbf{x}_0 = (\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon}_\theta)/\sqrt{\bar\alpha_t}$ gives the relationship |
| 9 | FID: computing with fewer than 10,000 samples | FID is highly sensitive to sample count — FID@1k can be 5-10 points lower than FID@50k. Results are not comparable across papers using different counts. | Always report FID@50k for standard benchmarks; report sample count explicitly |
| 10 | IS: comparing across different Inception versions | Inception Score depends on the specific Inception model's class probabilities. Newer models give different scores for the same images. | Use the same Inception checkpoint (inception-2015-12-05) across all comparisons |
| 11 | Flow composition order: applying $f_1$ last instead of first | The flow is $\mathbf{x} = f_K\circ\cdots\circ f_1(\mathbf{z})$. Applying in wrong order computes a different function. The Jacobian log-det sum depends on the correct ordering. | Maintain explicit list of layers; always verify: generate = apply in order $1\to K$, invert = apply in order $K\to 1$ |
| 12 | CFG: applying guidance scale $w$ to the wrong term | CFG formula: $\tilde\boldsymbol{\varepsilon} = (1+w)\boldsymbol{\varepsilon}(\mathbf{x}_t,y) - w\boldsymbol{\varepsilon}(\mathbf{x}_t,\emptyset)$. A common mistake is $w\boldsymbol{\varepsilon}(\mathbf{x}_t,y) + (1-w)\boldsymbol{\varepsilon}(\mathbf{x}_t,\emptyset)$, which is linear interpolation (not extrapolation) and doesn't amplify the conditional direction. | The correct formula extrapolates beyond the conditional prediction; verify $w=0$ gives conditional model, $w=-1$ gives unconditional |

---

## 12. Exercises

**Exercise 1** ★ **ELBO Derivation from Scratch**

Starting from $\log p_{\boldsymbol{\theta}}(\mathbf{x}) = \log \int p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})p(\mathbf{z})\,d\mathbf{z}$:

(a) Multiply and divide inside the log-integral by $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ and apply Jensen's inequality to derive the ELBO.

(b) Show that the gap between ELBO and log-likelihood is $D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}) \| p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x}))$.

(c) For $q = \mathcal{N}(\boldsymbol{\mu}, \operatorname{diag}(\boldsymbol{\sigma}^2))$ and $p = \mathcal{N}(\mathbf{0},I)$, derive the closed-form KL.

(d) Verify numerically that $D_{\mathrm{KL}} \geq 0$ for $\boldsymbol{\mu} = [2,1]$, $\boldsymbol{\sigma} = [0.5, 1.5]$.

**Exercise 2** ★ **Coupling Layer Jacobian**

For the RealNVP affine coupling $\mathbf{y}_b = \mathbf{z}_b \odot \exp(s(\mathbf{z}_a)) + t(\mathbf{z}_a)$, $\mathbf{y}_a = \mathbf{z}_a$:

(a) Write the full Jacobian $\partial(\mathbf{y}_a,\mathbf{y}_b)/\partial(\mathbf{z}_a,\mathbf{z}_b)$ in $2\times 2$ block form.

(b) Show it is lower triangular and compute $\log|\det J|$.

(c) Implement the forward and inverse passes and verify $f^{-1}(f(\mathbf{z})) = \mathbf{z}$.

(d) Verify the log-det matches $\sum_j s_j(\mathbf{z}_a)$ numerically.

**Exercise 3** ★ **Diffusion Forward Marginal**

(a) Given $q(\mathbf{x}_t \mid \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,\mathbf{x}_{t-1}, \beta_t I)$, prove by induction that $q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\,\mathbf{x}_0, (1-\bar\alpha_t)I)$.

(b) Show that $\mathbf{x}_t = \sqrt{\bar\alpha_t}\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon}$ is the reparameterisation form.

(c) For a linear noise schedule with $\beta_1 = 0.0001$, $\beta_T = 0.02$, $T = 1000$, plot $\bar\alpha_t$ vs $t$.

**Exercise 4** ★★ **Optimal GAN Discriminator and JSD**

(a) Show that $V(D,G) = \int [p_r(\mathbf{x})\log D(\mathbf{x}) + p_g(\mathbf{x})\log(1-D(\mathbf{x}))]\,d\mathbf{x}$ and find $D^*$ by setting the pointwise derivative to zero.

(b) Substitute $D^*$ into $V(D^*, G)$ and show $V(D^*, G) = -\log 4 + 2D_{\mathrm{JS}}(p_r \| p_g)$.

(c) Verify numerically: with $p_r = \mathcal{N}(0,1)$ and $p_g = \mathcal{N}(2,1)$, compute the JSD via Monte Carlo and check the formula gives the right $V(D^*, G)$.

**Exercise 5** ★★ **WGAN Gradient Penalty**

(a) Implement a simple 1D critic $f_\phi: \mathbb{R} \to \mathbb{R}$ (3-layer MLP).

(b) Compute the WGAN-GP loss with $p_r = \mathcal{N}(0,1)$, $p_g = \mathcal{N}(3,1)$, $\lambda = 10$.

(c) Verify that after 1000 training steps, the critic's gradient norm is close to 1 on the interpolated points.

(d) Show that the estimated Wasserstein distance grows as $|3 - 0| = 3$ (the true $W_1$ distance between these Gaussians).

**Exercise 6** ★★ **DDIM Deterministic Sampling Step**

(a) Starting from the DDIM update formula with $\sigma_t = 0$, express $\mathbf{x}_{t-1}$ in terms of $\mathbf{x}_t$, $\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$, $\bar\alpha_t$, and $\bar\alpha_{t-1}$.

(b) Show that this is an Euler step for an ODE in continuous time.

(c) Implement the DDIM sampler with 10 steps starting from $\mathbf{x}_T \sim \mathcal{N}(\mathbf{0},I)$ using a fixed (random) $\boldsymbol{\varepsilon}_\theta$.

(d) Verify that running the same sampler twice with the same initial noise gives identical $\mathbf{x}_0$.

**Exercise 7** ★★ **Score Matching Equivalence**

(a) Write out the explicit score matching objective $\mathbb{E}[\|s_\theta(\mathbf{x}) - \nabla_\mathbf{x}\log p(\mathbf{x})\|^2]$ and the denoising score matching objective.

(b) Show (by expanding the DSM objective) that the cross-term $-2\mathbb{E}[s_\theta(\tilde\mathbf{x}) \cdot \nabla\log q(\tilde\mathbf{x}|\mathbf{x})]$ equals $-2\mathbb{E}[s_\theta(\mathbf{x})\cdot\nabla\log p(\mathbf{x})]$ up to a constant, establishing equivalence.

(c) Verify numerically: train a score network on 1D Gaussian data using DSM; check that the learned score matches $-x/\sigma^2$ for $p = \mathcal{N}(0,\sigma^2)$.

**Exercise 8** ★★★ **CFM Objective**

(a) For the simple linear path $\mathbf{x}_t = (1-t)\mathbf{x}_0 + t\mathbf{x}_1$ with $\mathbf{x}_0 \sim \mathcal{N}(\mathbf{0},I)$, $\mathbf{x}_1 \sim p_{\text{data}}$, show the conditional vector field is $u_t(\mathbf{x}_t \mid \mathbf{x}_1) = \mathbf{x}_1 - \mathbf{x}_0$.

(b) Implement the CFM training loop for a 2D bimodal target distribution (mixture of two Gaussians).

(c) Train a small MLP as $\mathbf{v}_{\boldsymbol{\theta}}$, sample via Euler integration with 100 steps, and visualise the learned trajectories.

(d) Compare the curvature of trajectories with independent vs OT-matched coupling.

**Exercise 9** ★★★ **FID Computation**

(a) Given feature matrices $F_r \in \mathbb{R}^{n \times d}$ and $F_g \in \mathbb{R}^{n \times d}$, compute the FID using the formula with matrix square root.

(b) Implement FID using `scipy.linalg.sqrtm` and verify on synthetic data where $F_r \sim \mathcal{N}(\boldsymbol{\mu}_r, \Sigma_r)$ and $F_g \sim \mathcal{N}(\boldsymbol{\mu}_g, \Sigma_g)$ with known closed-form $W_2^2$.

(c) Show that FID increases as you shift $\boldsymbol{\mu}_g$ away from $\boldsymbol{\mu}_r$.

(d) Verify that FID = 0 when real and generated features come from the same distribution.

**Exercise 10** ★★★ **Classifier-Free Guidance Interpolation**

(a) Implement CFG: given a (randomly initialised) score network $\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, y)$ and $\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, \emptyset)$, compute $\tilde{\boldsymbol{\varepsilon}} = (1+w)\boldsymbol{\varepsilon}(\mathbf{x}_t,y) - w\boldsymbol{\varepsilon}(\mathbf{x}_t,\emptyset)$.

(b) Show that $w=0$ recovers $\boldsymbol{\varepsilon}(\mathbf{x}_t, y)$ and $w=-1$ recovers $\boldsymbol{\varepsilon}(\mathbf{x}_t,\emptyset)$.

(c) Verify that CFG corresponds to guiding by an implicit log-odds classifier.

(d) Simulate how guidance scale $w \in \{0, 1, 3, 7\}$ trades off between the conditional and unconditional directions in a toy 2D setting.

---

## 13. Why This Matters for AI (2026 Perspective)

| Concept | AI / LLM Impact |
|---|---|
| ELBO and VAE | VAE encoder/decoder is the compression backbone of Stable Diffusion, DALL-E 3, Parti. Latent space enables $4\times$–$8\times$ cheaper diffusion. |
| Reparameterisation trick | Enables gradient flow through stochastic nodes; used in diffusion forward process, normalising flow sampling, and differentiable simulation. |
| VQ-VAE | Discrete image tokenisation for DALL-E 1, Parti, LlamaGen. Enables treating images as sequences for GPT-style generation. |
| GAN discriminator | Direct ancestor of RLHF reward models. Discriminator training on preferred/non-preferred samples is the conceptual foundation of Bradley-Terry preference modelling. |
| WGAN / optimal transport | OT paths in flow matching (SD3, Flux) enable straight trajectories, fewer NFE, and faster generation. The Wasserstein metric underlies FID evaluation. |
| Score matching | Unified mathematical framework connecting DDPM, NCSN, SDE-based diffusion, and Langevin dynamics. Score functions = noise prediction networks. |
| DDPM simplified objective | Standard training objective for all diffusion-based image/video/audio generation (SD, DALL-E 2, Imagen, Sora, AudioLDM). |
| Classifier-free guidance | The "prompt strength" parameter in every text-to-image system. Controls text adherence vs diversity. Critical for RLHF-based alignment of diffusion models. |
| Flow matching (CFM) | State-of-the-art image generation (Flux, SD3, MovieGen). Trains faster than diffusion, enables 1-50 step generation. |
| DDIM / deterministic sampling | Production sampler in all deployed diffusion systems. Enables image editing via DDIM inversion. |
| Consistency models | Single-step generation for interactive applications (LCM). Deployed in real-time image editing tools. |
| Discrete diffusion | Emerging framework for language generation as parallel, non-autoregressive process. Competitive with GPT-style models on NLU/NLG benchmarks as of 2025. |
| FID / precision-recall | Universal evaluation benchmark for all generative models. Every image generation paper reports FID on COCO, LAION, or ImageNet. |
| Latent diffusion (LDM) | Stable Diffusion architecture — enabled open-source, affordable text-to-image generation. Foundation of entire open-source GenAI ecosystem. |

---

## 14. Conceptual Bridge

**Looking backward: What this section builds on.** Generative models sit at the convergence of several mathematical threads developed in earlier sections. The ELBO derivation requires Jensen's inequality and KL divergence (Section 14-03: Probabilistic Models). The VAE encoder is an amortised variational inference network — understanding variational inference from §14-03 is essential. GANs use the theory of convex duality (Kantorovich-Rubinstein duality for WGANs) developed in the optimisation chapters. Score matching and Langevin dynamics connect to MCMC theory (§14-03). Normalizing flows use Jacobian determinants from linear algebra (Chapters 02-03). Flow matching builds directly on continuous normalizing flows and the ODE theory sketched in calculus (Chapter 04). Reinforcement learning (§14-06) provides context for reward-conditioned generation and RLHF-fine-tuned diffusion models.

**Looking forward: What this section enables.** The architectures introduced here (VAE, U-Net, diffusion backbone) are the components of the next section on CNNs (§14-08), where the U-Net's convolutional architecture is analysed mathematically. The attention mechanism that enables text conditioning in diffusion (cross-attention) connects to the transformer section (§14-05). The flow matching framework connects to score SDEs and the broader theory of stochastic processes. Understanding diffusion model training objectives is essential for modern multimodal AI: text-to-image (Stable Diffusion, Flux), text-to-video (Sora, MovieGen), text-to-audio (AudioLDM, Stable Audio), and 3D generation (Zero-1-to-3, DreamFusion) all use the same mathematical skeleton.

**The central unifying idea.** Every generative model is a different strategy for the same problem: learn a mapping from a simple distribution (Gaussian noise) to the complex data distribution. VAEs do it with a probabilistic encoder-decoder. GANs do it with a minimax game. Flows do it with an invertible deterministic map. Diffusion models do it with a learned iterative denoiser. Flow matching does it with a learned velocity field for an ODE. The mathematical differences are:
- What notion of "distance" between distributions is optimised (KL, JSD, Wasserstein, score matching, OT)
- Whether the mapping is explicit or implicit
- Whether likelihood is tractable or bounded

Understanding all these simultaneously gives you the vocabulary to design new generative models and understand why they work.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  Chapter 06: Probability Theory
      ↓ KL divergence, Bayes theorem, Gaussians
  Chapter 09: Information Theory
      ↓ Entropy, mutual information, compression
  Section 14-03: Probabilistic Models
      ↓ ELBO, variational inference, MCMC, score matching
  Section 14-05: Transformer Architecture
      ↓ Attention, cross-attention for text conditioning
  Section 14-06: Reinforcement Learning
      ↓ RLHF, reward models, preference learning
      ↓
  ╔══════════════════════════════════╗
  ║  14-07: GENERATIVE MODELS        ║
  ║  VAEs, GANs, Flows, Diffusion,   ║
  ║  Score Matching, Flow Matching,  ║
  ║  LDM, Consistency Models         ║
  ╚══════════════════════════════════╝
      ↓
  Section 14-08: CNN and Convolution Math
      ↓ U-Net architecture, convolutional backbones
  Applications: Stable Diffusion, DALL-E 3, Flux,
                Sora, AudioLDM, LlamaGen

════════════════════════════════════════════════════════════════════════
```

---

## References

1. Kingma & Welling (2013). *Auto-Encoding Variational Bayes*. arXiv:1312.6114
2. Goodfellow et al. (2014). *Generative Adversarial Nets*. NeurIPS 2014
3. Dinh et al. (2016). *Density Estimation using Real-valued Non-Volume Preserving Transformations*. ICLR 2017
4. Arjovsky et al. (2017). *Wasserstein GAN*. ICML 2017
5. van den Oord et al. (2017). *Neural Discrete Representation Learning* (VQ-VAE). NeurIPS 2017
6. Ho et al. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS 2020
7. Song et al. (2021). *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR 2021
8. Song et al. (2021). *Denoising Diffusion Implicit Models*. ICLR 2021
9. Ho & Salimans (2022). *Classifier-Free Diffusion Guidance*. NeurIPS Workshop 2021
10. Rombach et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022
11. Lipman et al. (2022). *Flow Matching for Generative Modeling*. ICLR 2023
12. Song et al. (2023). *Consistency Models*. ICML 2023
13. Esser et al. (2024). *Scaling Rectified Flow Transformers for High-Resolution Image Synthesis* (SD3). ICML 2024
14. Sahoo et al. (2024). *Simple and Effective Masked Diffusion Language Models*. arXiv:2406.07524
15. Murphy (2023). *Probabilistic Machine Learning: Advanced Topics*. MIT Press

---

## Appendix A: Deeper Derivations

### A.1 Complete Diffusion ELBO Decomposition

We derive the full ELBO decomposition for the diffusion model. Define the variational family as the joint $q(\mathbf{x}_{1:T}\mid\mathbf{x}_0) = \prod_{t=1}^T q(\mathbf{x}_t\mid\mathbf{x}_{t-1})$.

The log-likelihood lower bound is:

$$\log p_{\boldsymbol{\theta}}(\mathbf{x}_0) \geq \mathbb{E}_q\!\left[\log \frac{p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T}\mid\mathbf{x}_0)}\right]$$

Expand using the Markov structure $p_{\boldsymbol{\theta}}(\mathbf{x}_{0:T}) = p(\mathbf{x}_T)\prod_{t=1}^T p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$:

$$= \mathbb{E}_q\!\left[\log p(\mathbf{x}_T) + \sum_{t=1}^T \log p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1}\mid\mathbf{x}_t) - \sum_{t=1}^T \log q(\mathbf{x}_t\mid\mathbf{x}_{t-1})\right]$$

After regrouping using the Markov factorisation of $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ (which is tractable as a Gaussian), we obtain:

$$\mathcal{L} = \underbrace{\mathbb{E}_q[-\log p(\mathbf{x}_T)/q(\mathbf{x}_T\mid\mathbf{x}_0)]}_{L_T\,\approx\,0} + \sum_{t=2}^T \underbrace{\mathbb{E}_q[D_{\mathrm{KL}}(q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)\|p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1}\mid\mathbf{x}_t))]}_{L_{t-1}} + \underbrace{\mathbb{E}_q[-\log p_{\boldsymbol{\theta}}(\mathbf{x}_0\mid\mathbf{x}_1)]}_{L_0}$$

**Each $L_{t-1}$ term.** Both $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$ and $p_{\boldsymbol{\theta}}(\mathbf{x}_{t-1}\mid\mathbf{x}_t)$ are Gaussian. The KL between two Gaussians of equal variance $\sigma_t^2$ is:

$$L_{t-1} = \mathbb{E}_q\!\left[\frac{1}{2\sigma_t^2}\lVert\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t,\mathbf{x}_0) - \boldsymbol{\mu}_{\boldsymbol{\theta}}(\mathbf{x}_t,t)\rVert_2^2\right] + \text{const}$$

Substituting $\mathbf{x}_0 = (\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon})/\sqrt{\bar\alpha_t}$ into $\tilde{\boldsymbol{\mu}}_t$ and rearranging:

$$L_{t-1} = \mathbb{E}\!\left[\frac{\beta_t^2}{2\sigma_t^2\alpha_t(1-\bar\alpha_t)}\lVert\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t)\rVert_2^2\right]$$

Dropping the time-dependent weight $\beta_t^2/(2\sigma_t^2\alpha_t(1-\bar\alpha_t))$ gives the simplified objective $\mathcal{L}_{\text{simple}}$.

### A.2 Kantorovich-Rubinstein Duality: Proof Sketch

**Theorem.** For probability measures $p$ and $q$ on a compact metric space $(\mathcal{X}, d)$:
$$W_1(p, q) = \sup_{f: \lVert f\rVert_L \leq 1} \int f\,dp - \int f\,dq$$

**Proof sketch.** This is a special case of linear programming duality. The primal is:
$$\min_{\gamma \geq 0}\int d(\mathbf{x},\mathbf{y})\,d\gamma(\mathbf{x},\mathbf{y}) \quad \text{s.t.} \quad \gamma_\mathbf{x} = p,\; \gamma_\mathbf{y} = q$$

The dual variables for the marginal constraints are functions $f, g: \mathcal{X} \to \mathbb{R}$. By strong duality (Kantorovich's duality theorem):
$$W_1(p,q) = \sup_{f+g \leq d(\cdot,\cdot)} \int f\,dp + \int g\,dq$$

Setting $g(\mathbf{y}) = \inf_\mathbf{x}[d(\mathbf{x},\mathbf{y}) - f(\mathbf{x})]$ (the $c$-transform) and using the 1-Lipschitz constraint $f(\mathbf{x}) - f(\mathbf{y}) \leq d(\mathbf{x},\mathbf{y})$ gives the result with $g = -f$. $\square$

**Implications for WGAN.** The Lipschitz constraint on the critic $f$ exactly enforces the dual constraint. The critic loss $\mathbb{E}_{p_r}[f] - \mathbb{E}_{p_g}[f]$ is the dual objective, so a trained-to-optimality critic computes $W_1$.

### A.3 Score SDE Framework: Unified View

**Forward SDE.** Any continuous-time forward noising process satisfies:
$$d\mathbf{x} = f(\mathbf{x},t)\,dt + g(t)\,d\mathbf{W}_t$$

**DDPM as SDE.** Setting $f(\mathbf{x},t) = -\frac{\beta(t)}{2}\mathbf{x}$ and $g(t) = \sqrt{\beta(t)}$ recovers the DDPM schedule in continuous time, with marginals $p_t(\mathbf{x}) = \mathcal{N}(\mathbf{x};\sqrt{\bar\alpha(t)}\mathbf{x}_0, (1-\bar\alpha(t))I)$.

**Reverse-time SDE (Anderson, 1982).** The reverse-time process satisfying the same marginals is:
$$d\mathbf{x} = \left[f(\mathbf{x},t) - g(t)^2\nabla_\mathbf{x}\log p_t(\mathbf{x})\right]dt + g(t)\,d\bar{\mathbf{W}}_t$$

Replacing the score with $s_{\boldsymbol{\theta}}(\mathbf{x},t)$ gives a generative SDE. Setting $g=0$ gives the **probability flow ODE**:
$$\frac{d\mathbf{x}}{dt} = f(\mathbf{x},t) - \frac{g(t)^2}{2}\nabla_\mathbf{x}\log p_t(\mathbf{x})$$

This ODE has the same marginals as the reverse SDE — it is the DDIM update in continuous time. The probability flow ODE is the continuous-time analogue of DDIM sampling.

---

## Appendix B: Connections Between Model Families

### B.1 VAE as a Special Case of Score Matching

Consider a VAE with Gaussian encoder $q_\phi(\mathbf{z}\mid\mathbf{x})$ and Gaussian decoder $p_\theta(\mathbf{x}\mid\mathbf{z})$. The reconstruction term $\mathbb{E}_q[\log p_\theta(\mathbf{x}\mid\mathbf{z})]$ can be interpreted as matching the score of $p_\theta(\mathbf{x}\mid\mathbf{z})$ at the encoded mean. This makes VAEs a special case of amortised variational score matching.

### B.2 GAN as Approximate KL Minimisation

The GAN generator minimises $D_{\mathrm{JS}}(p_r \| p_g) = \frac{1}{2}D_{\mathrm{KL}}(p_r \| m) + \frac{1}{2}D_{\mathrm{KL}}(p_g \| m)$ where $m = (p_r + p_g)/2$. This is neither forward nor reverse KL, but a symmetric combination. It explains GAN behavior: unlike forward KL (which is mode-covering) or reverse KL (which is mode-seeking), JSD-minimisation is intermediate — it can miss modes but is less prone to blurring than forward KL.

### B.3 Diffusion as Infinitely Deep VAE

A diffusion model with $T$ steps can be viewed as a $T$-layer hierarchical VAE where:
- Each "layer" has Gaussian transitions (linear encoder/decoder)
- The ELBO decomposes into $T$ KL terms
- The simplified objective drops the per-step weighting

This perspective (Ho et al., 2020) shows that DDPM training is exactly maximum likelihood for a specific hierarchical VAE with a fixed (non-trainable) encoder.

### B.4 Flow Matching as Score Matching + ODE Solver

The CFM velocity $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x}_t, t)$ is related to the score by:
$$\mathbf{v}_t(\mathbf{x}) = \mathbf{f}(\mathbf{x},t) - \frac{g(t)^2}{2}\nabla_\mathbf{x}\log p_t(\mathbf{x})$$

where $\mathbf{f}$ and $g$ are the SDE drift and diffusion coefficients. Flow matching directly regresses $\mathbf{v}_t$, while score matching regresses $\nabla\log p_t$ and then constructs $\mathbf{v}_t$ analytically. Both approaches work; flow matching avoids the need to specify the SDE parameters separately.

---

## Appendix C: Implementation Notes

### C.1 Numerical Stability for Diffusion

**Log-SNR parameterisation.** Instead of storing $\bar\alpha_t$ directly, use the log signal-to-noise ratio:
$$\lambda_t = \log\frac{\bar\alpha_t}{1-\bar\alpha_t}$$

This is more numerically stable near $t=0$ (clean data) and $t=T$ (pure noise), where $\bar\alpha_t$ approaches 1 and 0 respectively. The simplified objective in terms of $\lambda$ is:
$$\mathcal{L}_\lambda = \mathbb{E}_{\lambda\sim p(\lambda)}\mathbb{E}_{\mathbf{x}_0,\boldsymbol{\varepsilon}}\!\left[w(\lambda)\lVert\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_\lambda,\lambda)\rVert_2^2\right]$$

**Cholesky for VAE covariance.** When computing KL for a diagonal Gaussian:
```python
# Numerically stable: work in log-variance space
log_var = encoder(x)            # predict log(sigma^2)
kl = -0.5 * (1 + log_var - mu**2 - exp(log_var))
```
Never compute `sigma = sqrt(exp(log_var))` then `sigma**2` — this introduces unnecessary numerical error.

### C.2 FID Computation: Matrix Square Root

```python
import scipy.linalg as la

def compute_fid(mu_r, sigma_r, mu_g, sigma_g):
    # Matrix square root via eigendecomposition
    sqrt_sigma_r = la.sqrtm(sigma_r)
    M = sqrt_sigma_r @ sigma_g @ sqrt_sigma_r
    sqrt_M = la.sqrtm(M)
    # Handle numerical issues: take real part
    if np.iscomplexobj(sqrt_M):
        sqrt_M = sqrt_M.real
    trace_term = np.trace(sigma_r + sigma_g - 2 * sqrt_M)
    fid = np.sum((mu_r - mu_g)**2) + trace_term
    return float(fid)
```

**Warning:** `scipy.linalg.sqrtm` can produce complex outputs due to numerical errors. Always take the real part.

### C.3 Training Stability Tricks

**WGAN-GP:** Set $\lambda = 10$, train critic 5 steps per generator step, use Adam with $\alpha=10^{-4}$, $\beta_1=0.0$, $\beta_2=0.9$.

**VAE:** Use $\beta$-annealing with $\beta$ linearly increasing from 0 to 1 over the first 10k steps. Monitor the KL per latent dimension — if all dims collapse to 0, reduce learning rate or increase $\beta$-annealing duration.

**Diffusion:** Time step sampling matters. Instead of $t\sim\mathcal{U}\{1,\ldots,T\}$, use importance sampling proportional to $\lVert\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta\rVert^2$ variance (Min-SNR-$\gamma$ strategy, Hang et al. 2023) to reduce variance and accelerate training.

---

## Appendix D: Evaluation Beyond FID

### D.1 CLIP Score for Text-Conditional Models

For text-to-image models, FID measures visual quality but not text alignment. **CLIP Score** measures how well the generated image matches the prompt:

$$\text{CLIP-score}(\mathbf{x}_g, c) = \max(0, \cos(\phi_{\text{img}}(\mathbf{x}_g),\, \phi_{\text{txt}}(c)))$$

where $\phi_{\text{img}}$ and $\phi_{\text{txt}}$ are the CLIP image and text encoders. Higher = better text alignment. Modern models are evaluated on COCO-30k with CLIP-score + FID simultaneously.

### D.2 Human Evaluation

Automated metrics correlate imperfectly with human judgment. Standard human eval protocols:
- **Side-by-side preference:** Raters choose which of two images is better/more faithful to prompt
- **Likert scale:** Rate quality 1-5 on multiple axes (fidelity, diversity, text alignment, artefacts)
- **ELO rating:** Compute relative ranking from pairwise comparisons (used in DALL-E 3, Midjourney)

### D.3 Distributional Metrics for Language

For discrete diffusion / autoregressive models on text:
- **Perplexity:** $\operatorname{PPL} = \exp(-\frac{1}{T}\sum_t \log p_\theta(x_t\mid x_{<t}))$ — standard LM evaluation
- **MAUVE (Pillutla et al., 2021):** Measures divergence between human and model text distributions in a shared embedding space
- **Diversity:** Distinct-n (fraction of unique n-grams), self-BLEU (lower = more diverse)

---

## Appendix E: Notation Summary

| Symbol | Meaning |
|---|---|
| $p_{\text{data}}(\mathbf{x})$ | True data distribution |
| $p_{\boldsymbol{\theta}}(\mathbf{x})$ | Model distribution with parameters $\boldsymbol{\theta}$ |
| $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x})$ | Encoder / approximate posterior with parameters $\boldsymbol{\phi}$ |
| $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},I)$ | Prior distribution over latents |
| $\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\phi};\mathbf{x})$ | ELBO (evidence lower bound) |
| $G_{\boldsymbol{\theta}}, D_{\boldsymbol{\phi}}$ | GAN generator and discriminator |
| $f: \mathcal{Z}\to\mathcal{X}$ | Normalizing flow (forward direction) |
| $J_f(\mathbf{z})$ | Jacobian matrix of $f$ at $\mathbf{z}$ |
| $\bar\alpha_t = \prod_{s=1}^t(1-\beta_s)$ | Cumulative noise schedule |
| $\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t,t)$ | Noise prediction network (diffusion) |
| $s_{\boldsymbol{\theta}}(\mathbf{x},t)$ | Score network: $s_{\boldsymbol{\theta}} \approx \nabla_\mathbf{x}\log p_t(\mathbf{x})$ |
| $\mathbf{v}_{\boldsymbol{\theta}}(\mathbf{x},t)$ | Velocity field (flow matching) |
| $W_1(p,q)$ | Wasserstein-1 (earth-mover) distance |
| $D_{\mathrm{JS}}(p\|q)$ | Jensen-Shannon divergence |
| $\mathrm{FID}$ | Fréchet Inception Distance (lower = better) |
| $\mathrm{IS}$ | Inception Score (higher = better) |
| $w$ | CFG guidance scale |
| $\sigma_t$ | DDIM sampling noise level ($\sigma_t=0$ = deterministic) |

---

## Appendix F: Advanced Score SDE Theory

### F.1 Predictor-Corrector Sampling

**PC sampling (Song et al., 2021)** alternates between two sampling steps:

1. **Predictor:** Run one step of the reverse-time discretised SDE to move $\mathbf{x}_t \to \mathbf{x}_{t-1}$.
2. **Corrector:** Run several steps of Langevin MCMC at noise level $t-1$ to correct errors accumulated in the predictor step.

The corrector uses the score: $\mathbf{x} \leftarrow \mathbf{x} + \frac{\epsilon}{2}s_{\boldsymbol{\theta}}(\mathbf{x}, t-1) + \sqrt{\epsilon}\boldsymbol{\xi}$, $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0},I)$.

This achieves higher sample quality than pure predictor-only (DDPM/DDIM) sampling at a cost of more function evaluations.

### F.2 Continuous-Time Loss Weighting

The continuous-time score matching objective with weighting function $\lambda(t)$:

$$\mathcal{L}_{\text{weighted}} = \mathbb{E}_{t\sim\mathcal{U}(0,T)}\!\left[\lambda(t)\,\mathbb{E}_{\mathbf{x}_0,\mathbf{x}_t}\!\left[\lVert s_{\boldsymbol{\theta}}(\mathbf{x}_t,t) - \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t\mid\mathbf{x}_0)\rVert_2^2\right]\right]$$

**Choice of $\lambda(t)$:**
- $\lambda = g(t)^2$ (likelihood weighting): gives the exact log-likelihood gradient
- $\lambda = 1$ (simplified, Ho et al.): uniform weighting, good empirical performance
- $\lambda = \text{SNR}(t)^{-1}$ (Min-SNR, Hang et al.): prevents gradient dominance at high noise levels

The simplified objective $\mathcal{L}_{\text{simple}}$ with uniform weighting is justified by the empirical finding that over-weighting high-noise timesteps hurts visual quality even though it is theoretically better for NLL.

### F.3 Consistency Function Properties

A valid consistency function $f_{\boldsymbol{\theta}}: \mathbb{R}^d \times [0,T] \to \mathbb{R}^d$ must satisfy:

1. **Boundary condition:** $f_{\boldsymbol{\theta}}(\mathbf{x}_0, 0) = \mathbf{x}_0$ for all $\mathbf{x}_0$ (identity at $t=0$)
2. **Consistency:** $f_{\boldsymbol{\theta}}(\mathbf{x}_t, t) = f_{\boldsymbol{\theta}}(\mathbf{x}_{t'}, t')$ whenever $\mathbf{x}_t$ and $\mathbf{x}_{t'}$ lie on the same ODE trajectory

The boundary condition is enforced by parameterisation: $f_{\boldsymbol{\theta}}(\mathbf{x},t) = c_{\text{skip}}(t)\mathbf{x} + c_{\text{out}}(t)F_{\boldsymbol{\theta}}(\mathbf{x},t)$ with $c_{\text{skip}}(0) = 1$, $c_{\text{out}}(0) = 0$.

---

## Appendix G: Discrete Diffusion: Detailed Framework

### G.1 Categorical Diffusion

For discrete data $\mathbf{x}_0 \in \{1,\ldots,V\}^n$ (e.g., text tokens), define the forward process as a continuous-time Markov chain on the vocabulary. The transition rate matrix $Q_t$ defines how probability mass flows between states:

$$\frac{d}{dt}[\mathbf{x}_t = k] = \sum_{j} Q_t[k,j][\mathbf{x}_t = j]$$

**Absorbing state.** A common choice: each token transitions to a special [MASK] state at rate 1, and [MASK] is absorbing. Marginal at time $t$:
$$q(\mathbf{x}_t^i \mid \mathbf{x}_0^i) = (1-e^{-t})\delta_{\text{MASK}} + e^{-t}\delta_{\mathbf{x}_0^i}$$

**Reverse process.** The posterior $q(\mathbf{x}_{t-1}^i \mid \mathbf{x}_t^i, \mathbf{x}_0^i)$:
- If $\mathbf{x}_t^i = \mathbf{x}_0^i$: token survives with prob $\propto e^{-\beta_{t-1}}/e^{-\beta_t}$, gets masked otherwise
- If $\mathbf{x}_t^i = \text{MASK}$: remains masked with prob $1-e^{-\beta_{t-1}}(1-e^{\beta_t-\beta_{t-1}})/(1-e^{-\beta_t})$, or unmasked to $\mathbf{x}_0^i$

The denoising network $p_{\boldsymbol{\theta}}(\mathbf{x}_0 \mid \mathbf{x}_t)$ predicts the original tokens at all masked positions simultaneously.

### G.2 Connection to BERT

Masked language models (BERT, Devlin et al. 2018) can be viewed as training a single step ($t = 0.15T$) of a discrete diffusion model:
- BERT masks 15% of tokens randomly
- The model predicts the original tokens
- This is the discrete diffusion denoising objective at a fixed noise level

The key difference: BERT trains at a single fixed noise level, while MDLM trains across all noise levels $t \in [0,T]$, giving a generative model that can be used autoregressively-free.

---

## Appendix H: Modern Architectures (2024-2025)

### H.1 DiT: Diffusion Transformer

**DiT (Peebles & Xie, 2022)** replaces the U-Net backbone with a Vision Transformer (ViT):

1. **Patch embedding:** Divide latent $\mathbf{z} \in \mathbb{R}^{H\times W\times C}$ into $p\times p$ patches, linearly project to tokens
2. **Adaversarial Layer Norm (adaLN-zero):** Condition on time $t$ and class $y$ by modulating layer norm scale/shift parameters: $\gamma, \beta = \text{MLP}(\mathbf{c}_t + \mathbf{c}_y)$
3. **Transformer blocks:** Standard multi-head self-attention + MLP with adaLN conditioning
4. **Output:** Predict noise $\boldsymbol{\varepsilon}$ or $(\boldsymbol{\varepsilon}, \Sigma)$ at each patch position

**DiT-XL/2** achieves FID 2.27 on ImageNet-256 (class-conditional), surpassing U-Net diffusion models.

### H.2 MMDiT: Multimodal Diffusion Transformer

**MMDiT (Esser et al., 2024, used in Stable Diffusion 3)** extends DiT to joint image-text processing:

- Image patches and text tokens are processed in a **shared sequence** with separate weight streams
- Image self-attention attends to all tokens; text self-attention attends to all tokens
- **Cross-attention replaced by joint attention:** eliminates the asymmetry between image queries and text keys/values

The joint sequence formulation allows richer text-image interaction than cross-attention alone.

### H.3 Flux Architecture

**Flux (Black Forest Labs, 2024)** combines:
- **Flow matching** (rectified flows) instead of diffusion
- **MMDiT** transformer backbone
- **Rotary position embeddings (RoPE)** for spatial encoding in image tokens
- **Classifier-free guidance** with text+image joint conditioning

Flux achieves state-of-the-art quality on text-to-image benchmarks while requiring fewer sampling steps than diffusion-based predecessors.

---

## Appendix I: Common Mathematical Identities

### I.1 Gaussian Marginalisation

If $\mathbf{x} \mid \mathbf{z} \sim \mathcal{N}(A\mathbf{z} + \mathbf{b}, \Sigma_1)$ and $\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma_2)$, then:
$$\mathbf{x} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b},\; A\Sigma_2 A^\top + \Sigma_1)$$

Used in: composing diffusion forward transitions, VAE marginal likelihood.

### I.2 Gaussian Conditioning

If $(\mathbf{x}, \mathbf{z})$ are jointly Gaussian with mean $(\boldsymbol{\mu}_x, \boldsymbol{\mu}_z)$ and covariance $\begin{pmatrix}\Sigma_{xx} & \Sigma_{xz}\\ \Sigma_{zx} & \Sigma_{zz}\end{pmatrix}$, then:
$$\mathbf{x}\mid\mathbf{z} \sim \mathcal{N}(\boldsymbol{\mu}_x + \Sigma_{xz}\Sigma_{zz}^{-1}(\mathbf{z}-\boldsymbol{\mu}_z),\; \Sigma_{xx} - \Sigma_{xz}\Sigma_{zz}^{-1}\Sigma_{zx})$$

Used in: deriving the diffusion posterior $q(\mathbf{x}_{t-1}\mid\mathbf{x}_t,\mathbf{x}_0)$, GP regression.

### I.3 Matrix Square Root via Cholesky

For $\Sigma = LL^\top$ (Cholesky), the matrix square root satisfies $\Sigma^{1/2} = L$ only when $L$ is symmetric positive definite. In general, $\Sigma^{1/2}$ is defined via eigendecomposition: $\Sigma = Q\Lambda Q^\top \Rightarrow \Sigma^{1/2} = Q\Lambda^{1/2}Q^\top$.

### I.4 Log-Sum-Exp for Numerical Stability

$$\log\sum_{k=1}^K e^{a_k} = a_{\max} + \log\sum_{k=1}^K e^{a_k - a_{\max}}$$

Used in: computing GMM likelihoods, normalising attention weights, HMM forward algorithm.

### I.5 Score of a Gaussian

For $p(\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}, \Sigma)$:
$$\nabla_\mathbf{x}\log p(\mathbf{x}) = -\Sigma^{-1}(\mathbf{x} - \boldsymbol{\mu})$$

For the noisy data distribution $q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)I)$:
$$\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0}{1-\bar\alpha_t} = -\frac{\boldsymbol{\varepsilon}}{\sqrt{1-\bar\alpha_t}}$$

This directly gives the connection: score prediction $\equiv$ scaled noise prediction.

---

## Appendix J: EDM Framework (Karras et al., 2022)

The **EDM (Elucidating Diffusion Models)** framework provides a clean, principled reparameterisation of diffusion models that separates design choices.

### J.1 Noise Parameterisation

Instead of $\bar\alpha_t$, use noise level $\sigma$ directly. At noise level $\sigma$, the noisy data is:
$$\mathbf{x}_\sigma = \mathbf{x}_0 + \sigma\boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},I)$$

(Note: this is VP-SDE in the limit $\bar\alpha\to 1$, VE-SDE parameterisation.)

The score function is:
$$\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma) = -\frac{\boldsymbol{\varepsilon}}{\sigma} = \frac{\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_\sigma] - \mathbf{x}_\sigma}{\sigma^2}$$

### J.2 Denoiser Parameterisation

The **denoiser** $D_{\boldsymbol{\theta}}(\mathbf{x};\sigma)$ predicts $\mathbf{x}_0$ from $\mathbf{x}_\sigma$:

$$D_{\boldsymbol{\theta}}(\mathbf{x};\sigma) = c_{\text{skip}}(\sigma)\mathbf{x} + c_{\text{out}}(\sigma)F_{\boldsymbol{\theta}}(c_{\text{in}}(\sigma)\mathbf{x}; c_{\text{noise}}(\sigma))$$

where the preconditioning functions $c_{\text{skip}}, c_{\text{out}}, c_{\text{in}}, c_{\text{noise}}$ are chosen to:
- Normalise inputs and outputs to unit variance for stable training
- Interpolate between the identity (at $\sigma\to 0$) and pure noise prediction (at $\sigma\to\infty$)

**Optimal choices (Karras et al.):**
$$c_{\text{skip}}(\sigma) = \frac{\sigma_{\text{data}}^2}{\sigma^2 + \sigma_{\text{data}}^2}, \quad c_{\text{out}}(\sigma) = \frac{\sigma\cdot\sigma_{\text{data}}}{\sqrt{\sigma^2+\sigma_{\text{data}}^2}}$$

### J.3 Training Loss

The EDM training objective with weighting:
$$\mathcal{L}_{\text{EDM}} = \mathbb{E}_{\sigma\sim p_{\text{train}}, \mathbf{x}_0, \boldsymbol{\varepsilon}}\!\left[\lambda(\sigma)\lVert D_{\boldsymbol{\theta}}(\mathbf{x}_0+\sigma\boldsymbol{\varepsilon};\sigma) - \mathbf{x}_0\rVert_2^2\right]$$

with $\lambda(\sigma) = (\sigma^2 + \sigma_{\text{data}}^2)/(\sigma\cdot\sigma_{\text{data}})^2$ derived to give equal loss contribution at each noise level.

**Log-normal noise sampling:** $\ln\sigma \sim \mathcal{N}(P_{\text{mean}}, P_{\text{std}}^2)$ with $P_{\text{mean}} = -1.2$, $P_{\text{std}} = 1.2$. This concentrates training on the most informative noise levels (intermediate $\sigma$).

### J.4 Deterministic ODE Sampling

EDM uses a higher-order ODE solver (Heun's method, 2nd order):

**Step 1:** $\mathbf{x}_{i+1}' = \mathbf{x}_i + (\sigma_{i+1}-\sigma_i)\mathbf{d}_i$ where $\mathbf{d}_i = (\mathbf{x}_i - D_{\boldsymbol{\theta}}(\mathbf{x}_i;\sigma_i))/\sigma_i$

**Step 2 (correction):** $\mathbf{d}_i' = (\mathbf{x}_{i+1}' - D_{\boldsymbol{\theta}}(\mathbf{x}_{i+1}';\sigma_{i+1}))/\sigma_{i+1}$

**Final:** $\mathbf{x}_{i+1} = \mathbf{x}_i + (\sigma_{i+1}-\sigma_i)(\mathbf{d}_i + \mathbf{d}_i')/2$

With $N=35$ steps, EDM achieves competitive FID to DDPM with 1000 steps.

---

## Appendix K: Practical Guide to Building a VAE

### K.1 Architecture Choices

**Encoder.** Typically a CNN (for images) or transformer (for sequences):
```
x [B, C, H, W]
→ ResBlocks + downsampling → h [B, 512, H/8, W/8]
→ Flatten → Linear → (mu [B, d], log_var [B, d])
```

**Decoder.** Mirrors encoder with transposed convolutions or upsampling:
```
z [B, d]
→ Linear → Reshape [B, 512, H/8, W/8]
→ ResBlocks + upsampling → x_hat [B, C, H, W]
```

**Latent dimension.** Typical: $d = 4$ (Stable Diffusion VAE for $256\times256$), $d=8$ (SD XL), $d=16$ (SD3). Larger $d$ = more capacity, higher KL, slower diffusion.

### K.2 Loss Functions

**Reconstruction loss.** For images, MSE alone gives blurry results. Best practice (from Stable Diffusion):
$$\mathcal{L}_{\text{rec}} = \underbrace{\lVert\mathbf{x} - \hat\mathbf{x}\rVert_2^2}_{\text{pixel MSE}} + \lambda_{\text{perc}}\underbrace{\lVert\phi(\mathbf{x}) - \phi(\hat\mathbf{x})\rVert_2^2}_{\text{perceptual}} + \lambda_{\text{adv}}\underbrace{\mathcal{L}_{\text{GAN}}(\hat\mathbf{x})}_{\text{patch discriminator}}$$

The adversarial loss (from a patch-level discriminator) is critical for sharpness — without it, VAE outputs are characteristically blurry.

**KL weight.** $\beta \in [10^{-6}, 10^{-4}]$ (very small) for image VAEs in LDM: we want high reconstruction fidelity with minimal KL penalty, since the diffusion model will handle the generative modelling.

### K.3 Debugging Common Failures

| Symptom | Likely Cause | Fix |
|---|---|---|
| Blurry reconstructions | KL too high or no perceptual loss | Add perceptual loss + adversarial term |
| KL = 0 (all dims) | Posterior collapse | Use KL annealing or free bits |
| NaN in training | Numerical instability in KL | Use `log_var` clipping: `log_var.clamp(-30, 20)` |
| Artifacts at edges | Insufficient decoder capacity | Add skip connections or larger decoder |
| Poor generation quality | Latent space not smooth | Increase $\beta$ or add Gaussian MMD regulariser |

---

## Appendix L: Score Networks: Architecture Insights

### L.1 Why U-Net Works for Denoising

The U-Net's multi-resolution structure is ideal for the denoising task because:

1. **Different noise levels require different spatial scales.** At high noise ($\sigma$ large), global structure matters most — the bottleneck features handle this. At low noise, fine texture matters — the highest-resolution skip connections handle this.

2. **Skip connections prevent information loss.** In standard encoder-decoder networks, fine spatial details are lost at the bottleneck. Skip connections pass fine details directly from encoder to decoder, enabling sharp reconstructions.

3. **Equivariance.** Convolutional layers are translation-equivariant: shifting the input shifts the output by the same amount. This is appropriate for natural images where the same patterns (edges, textures) appear at different locations.

### L.2 Time Embedding

The noise level $t$ (or $\sigma_t$) must be communicated to the network. Standard approach:

**Sinusoidal embedding** (same as transformer positional encoding):
$$\mathbf{e}_t = [\sin(\omega_1 t), \cos(\omega_1 t), \sin(\omega_2 t), \cos(\omega_2 t), \ldots]$$

with $\omega_k = 10000^{-2k/d}$. This is then projected to a conditioning vector via two linear layers + SiLU:
$$\mathbf{c}_t = \text{Linear}_2(\operatorname{SiLU}(\text{Linear}_1(\mathbf{e}_t)))$$

**Adaptive Group Normalisation (adaGN).** Inject the conditioning via:
$$h_{\text{out}} = \mathbf{s}\cdot \text{GroupNorm}(h) + \mathbf{b}, \quad [\mathbf{s}, \mathbf{b}] = \text{Linear}(\mathbf{c}_t)$$

### L.3 Attention in U-Net

Self-attention at multiple resolutions captures long-range dependencies. Key observations:

- **Resolution matters.** Attention is $O(N^2)$ where $N$ = number of spatial positions. At full resolution ($64\times64$), self-attention has $N=4096$ — prohibitively expensive. Used only at lower resolutions ($8\times8$, $16\times16$, $32\times32$).

- **Cross-attention for text.** Text tokens act as keys/values; image features act as queries. Each spatial position in the image "asks" the text for relevant information.

- **FlashAttention (Dao et al., 2022).** Tiled matrix multiplication that avoids materialising the full attention matrix in memory: reduces memory from $O(N^2)$ to $O(N)$ and speeds up by $2\times$–$4\times$. Essential for training large diffusion models.

**For AI:** FlashAttention is used in essentially all large transformer and diffusion models (GPT-4, Stable Diffusion, Flux). It is one of the key infrastructure improvements that enabled scaling from 1B to 100B parameter models.

---

## Appendix M: The Rate-Distortion Theory of VAEs

### M.1 Information-Theoretic View

The VAE ELBO can be rewritten as a rate-distortion tradeoff:

$$\mathcal{L} = \underbrace{-\mathbb{E}_{q}[\log p_\theta(\mathbf{x}\mid\mathbf{z})]}_{\text{distortion }D} + \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\|p(\mathbf{z}))}_{\text{rate }R}$$

- **Rate $R$:** The number of bits used to encode $\mathbf{z}$ (relative to the prior). Equals the average code length under optimal coding.
- **Distortion $D$:** The reconstruction error. Measures how much information is lost.

The ELBO is $\mathcal{L} = D + R$ — maximising the ELBO minimises this sum.

### M.2 Rate-Distortion Tradeoff

By varying the KL weight $\beta$, we trace a **rate-distortion curve**:
$$\mathcal{L}_\beta = D + \beta R$$

- $\beta \to 0$: Model ignores rate, uses all available bits. Perfect reconstruction, no structure in latent space.
- $\beta \to \infty$: Model minimises rate, uses zero bits. Latent space = prior, no information encoded.
- Optimal $\beta$: Trade-off point where marginal reduction in distortion equals marginal increase in rate.

**Connection to $\beta$-VAE:** $\beta > 1$ prioritises low rate (compact, disentangled representations) over low distortion (high reconstruction quality). Disentanglement emerges when each latent dimension must maximally efficiently encode a single independent factor.

### M.3 Minimum Description Length

The VAE training objective is equivalent to the **Minimum Description Length (MDL)** principle: find $\boldsymbol{\theta}, \boldsymbol{\phi}$ that minimise the total code length:

$$\text{MDL} = \underbrace{\text{bits to encode } \mathbf{z}}_{\text{rate}} + \underbrace{\text{bits to encode } \mathbf{x} \text{ given } \mathbf{z}}_{\text{distortion}}$$

This provides an information-theoretic justification for the ELBO: the optimal model achieves the best compression of the data.

**For AI:** The rate-distortion view explains why VAE latent spaces at different $\beta$ values are useful for different downstream tasks. Low $\beta$ (high-rate) latent spaces preserve fine detail — ideal for Stable Diffusion's VAE where reconstruction fidelity is critical. High $\beta$ (low-rate, disentangled) spaces are ideal for downstream classification and attribute manipulation.

---

## Appendix N: Guidance and Conditioning: Deeper Analysis

### N.1 Guidance as Bayesian Posterior Sampling

Classifier-free guidance admits a clean Bayesian interpretation. Define:

$$\log\tilde{p}_{\boldsymbol{\theta}}(\mathbf{x}_t \mid y) = \log p_{\boldsymbol{\theta}}(\mathbf{x}_t) + (1+w)\log p_{\boldsymbol{\theta}}(y\mid\mathbf{x}_t) - \text{const}$$

The score of this tilted distribution is:
$$\nabla_{\mathbf{x}_t}\log\tilde{p}_{\boldsymbol{\theta}}(\mathbf{x}_t\mid y) = \nabla_{\mathbf{x}_t}\log p_{\boldsymbol{\theta}}(\mathbf{x}_t) + (1+w)\nabla_{\mathbf{x}_t}\log p_{\boldsymbol{\theta}}(y\mid\mathbf{x}_t)$$

The classifier $p_\theta(y\mid\mathbf{x}_t)$ is **implicit** — derived from the ratio of conditional to unconditional predictions via Bayes:
$$p_{\boldsymbol{\theta}}(y\mid\mathbf{x}_t) \propto p_{\boldsymbol{\theta}}(\mathbf{x}_t\mid y)/p_{\boldsymbol{\theta}}(\mathbf{x}_t)$$

The CFG modified score $\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,y) - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,\emptyset)$ estimates this log-ratio.

### N.2 Negative Guidance and Negative Prompts

The CFG formula extends to negative prompts. Let $y^+$ be the positive prompt and $y^-$ the negative:

$$\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{\theta}} = \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, y^-) + (1+w)[\boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, y^+) - \boldsymbol{\varepsilon}_{\boldsymbol{\theta}}(\mathbf{x}_t, y^-)]$$

Negative prompts steer generation away from specific attributes. This is equivalent to using $y^-$ as the "unconditional" baseline instead of the null prompt.

### N.3 Guidance Distillation

**Score distillation sampling (SDS, Poole et al., 2022).** Diffusion guidance can be used to optimise any differentiable function of generated images. SDS updates a NeRF or 3D model $\theta$ by:

$$\nabla_\theta\mathcal{L}_{\text{SDS}} = \mathbb{E}_{t,\boldsymbol{\varepsilon}}\!\left[w(t)(\tilde{\boldsymbol{\varepsilon}}_{\boldsymbol{\phi}}(\mathbf{x}_t, y) - \boldsymbol{\varepsilon})\frac{\partial\mathbf{x}}{\partial\theta}\right]$$

This is "guidance without an ODE solver" — the diffusion model's score estimates are used as a direct gradient signal to optimise 3D representations (DreamFusion, Magic3D, ProlificDreamer).

---

## Appendix O: The Geometry of Latent Spaces

### O.1 Interpolation in Latent Space

A key property of a good VAE latent space is **smoothness**: interpolating between two latent codes should produce smooth transitions in data space.

**Spherical interpolation (slerp)** is preferred over linear interpolation for latent codes:
$$\text{slerp}(\mathbf{z}_1, \mathbf{z}_2; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\mathbf{z}_1 + \frac{\sin(t\Omega)}{\sin\Omega}\mathbf{z}_2$$

where $\Omega = \arccos(\hat\mathbf{z}_1\cdot\hat\mathbf{z}_2)$. For high-dimensional Gaussians, most of the probability mass is on a sphere of radius $\sqrt{d}$ — slerp stays on this sphere while linear interpolation passes through low-probability regions near the origin.

### O.2 Manifold Hypothesis and Generative Models

The **manifold hypothesis** posits that natural data (images, text, audio) lies on a low-dimensional manifold embedded in high-dimensional space. Generative models learn this manifold:

- **VAE:** Explicit parametrisation of a manifold via the decoder $\mathbf{x} = f_\theta(\mathbf{z})$
- **GAN:** Implicit manifold via the generator image $G_\theta(\mathbf{z})$
- **Diffusion:** Reverse process traces a path from ambient noise down to the manifold
- **Flow matching:** ODE trajectory transport onto the manifold

**Topological constraint.** A continuous map from $\mathbb{R}^d$ (the latent space) cannot perfectly cover a disconnected data manifold (e.g., images of different classes). This is why the prior $p(\mathbf{z})$ must be continuous: the VAE decoder cannot create sharp mode boundaries.

### O.3 Latent Space Arithmetic

Well-trained generative models exhibit semantic arithmetic in latent space:

**VAE example:** $\mathbf{z}_{\text{smiling woman}} - \mathbf{z}_{\text{neutral woman}} + \mathbf{z}_{\text{neutral man}} \approx \mathbf{z}_{\text{smiling man}}$

This was first demonstrated with Word2Vec ($\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$) and shows that linearly structured latent spaces capture semantic relationships.

**Why it works.** If independent factors (gender, expression) are encoded in orthogonal subspaces (achieved by $\beta$-VAE disentanglement), then adding/subtracting latent vectors modifies individual factors independently.

**For AI:** DALL-E 2 uses latent arithmetic in CLIP embedding space: "a photo of a dog + watercolour painting style" produces a watercolour dog by adding the style vector. Prompt interpolation in CFG achieves similar effects at the conditioning level.

---

## Appendix P: Generative Models and LLMs: Direct Connections

### P.1 Diffusion Language Models

The success of diffusion for images motivates analogous models for language. Key challenges:

1. **Discrete tokens:** Standard diffusion requires continuous variables. Solutions: discrete diffusion (masking), continuous relaxation (soft tokens), embedding-space diffusion.

2. **Autoregressive baseline is strong:** GPT achieves near-human perplexity on many benchmarks. Diffusion must match this quality while offering parallelism benefits.

3. **Variable-length sequences:** Images have fixed $H\times W$ resolution; text has variable length. Length prediction must be part of the model.

**MDLM (2024) results:** On OpenWebText, MDLM achieves 7.41 BPD vs GPT-2's 7.14 BPD — within 4% of autoregressive baseline with parallel generation.

### P.2 Diffusion for Molecular Design

Protein structure (AlphaFold era) and molecular design use diffusion:

- **FrameDiff / Chroma / RFDiffusion:** Diffuse over protein backbone frames (SE(3) manifold)
- **DiffSBDD, DiffDock:** Diffuse over ligand poses and conformers
- **MolDiff:** Joint diffusion over atoms and bonds

The mathematical framework extends diffusion from Euclidean space to Lie groups (SE(3) = rotations + translations) using equivariant architectures.

### P.3 Reward-Guided Generation

RLHF for language models (§14-06) and diffusion guidance share the same structure:

| Language Model (RLHF) | Diffusion Model (CFG) |
|---|---|
| Base policy $\pi_{\text{ref}}$ | Unconditional model $p_\theta(\mathbf{x})$ |
| Reward model $r(y)$ | Classifier $\log p(y\mid\mathbf{x})$ |
| KL penalty $D_{\mathrm{KL}}(\pi\|\pi_{\text{ref}})$ | Implicit in CFG formula |
| Fine-tuned policy $\pi_\theta$ | Guided model $\tilde{p}_\theta(\mathbf{x}\mid y)$ |

In both cases, we want to sample from $\pi^* \propto \pi_{\text{ref}}\cdot e^{r/\beta}$ (the tilted distribution) without retraining the base model. CFG is an efficient approximation to this in diffusion; PPO is the standard approach for LLMs.

**DDPO (Black et al., 2023):** Directly applies policy gradient methods (PPO) to fine-tune diffusion models with respect to differentiable reward functions. This is the diffusion equivalent of RLHF.

---

## Appendix Q: Key Theorems and Proofs

### Q.1 The Reparameterisation Trick: Formal Statement

**Theorem.** Let $\mathbf{z} = g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon})$ where $\boldsymbol{\varepsilon} \sim p(\boldsymbol{\varepsilon})$ (independent of $\boldsymbol{\phi}$) and $g_{\boldsymbol{\phi}}$ is differentiable w.r.t. $\boldsymbol{\phi}$. Then:
$$\nabla_{\boldsymbol{\phi}} \mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z})}[f(\mathbf{z})] = \mathbb{E}_{p(\boldsymbol{\varepsilon})}\!\left[\nabla_{\boldsymbol{\phi}} f(g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon}))\right]$$

**Proof.** Under the change of variables $\mathbf{z} = g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon})$, the distribution of $\mathbf{z}$ is $q_{\boldsymbol{\phi}}$. Therefore:
$$\mathbb{E}_{q_{\boldsymbol{\phi}}(\mathbf{z})}[f(\mathbf{z})] = \mathbb{E}_{p(\boldsymbol{\varepsilon})}[f(g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon}))]$$

The right side is an expectation over a distribution that does not depend on $\boldsymbol{\phi}$. Under mild regularity conditions (Leibniz integral rule), we can interchange differentiation and expectation:
$$\nabla_{\boldsymbol{\phi}} \mathbb{E}_{p(\boldsymbol{\varepsilon})}[f(g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon}))] = \mathbb{E}_{p(\boldsymbol{\varepsilon})}[\nabla_{\boldsymbol{\phi}} f(g_{\boldsymbol{\phi}}(\boldsymbol{\varepsilon}))] \qquad \square$$

### Q.2 Change of Variables: Formal Statement

**Theorem.** Let $f: \mathbb{R}^d \to \mathbb{R}^d$ be an invertible differentiable map. If $Z \sim p_Z$ and $X = f(Z)$, then the density of $X$ is:
$$p_X(\mathbf{x}) = p_Z(f^{-1}(\mathbf{x})) \cdot |\det J_{f^{-1}}(\mathbf{x})|$$

**Proof.** For any Borel set $A$:
$$P(X \in A) = P(f(Z) \in A) = P(Z \in f^{-1}(A)) = \int_{f^{-1}(A)} p_Z(\mathbf{z})\,d\mathbf{z}$$

Under the substitution $\mathbf{z} = f^{-1}(\mathbf{x})$, $d\mathbf{z} = |\det J_{f^{-1}}(\mathbf{x})|\,d\mathbf{x}$:
$$= \int_A p_Z(f^{-1}(\mathbf{x})) |\det J_{f^{-1}}(\mathbf{x})|\,d\mathbf{x}$$

Since this holds for all $A$, the integrand is the density of $X$. $\square$

### Q.3 Optimality of the Gaussian ELBO

**Theorem.** For fixed $p_{\boldsymbol{\theta}}(\mathbf{x}\mid\mathbf{z})$ and $p(\mathbf{z}) = \mathcal{N}(\mathbf{0},I)$, the optimal encoder that maximises the ELBO over the class of Gaussian encoders $q_{\boldsymbol{\phi}}(\mathbf{z}\mid\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_{\boldsymbol{\phi}}, \Sigma_{\boldsymbol{\phi}})$ minimises $D_{\mathrm{KL}}(q_{\boldsymbol{\phi}}\|p_{\boldsymbol{\theta}}(\mathbf{z}\mid\mathbf{x}))$ over the class of Gaussians.

**Proof sketch.** The gap $\log p_\theta(\mathbf{x}) - \mathcal{L} = D_{\mathrm{KL}}(q\|p_\theta(\mathbf{z}\mid\mathbf{x}))$. Since $\log p_\theta(\mathbf{x})$ is fixed, maximising $\mathcal{L}$ is equivalent to minimising $D_{\mathrm{KL}}(q\|p_\theta(\mathbf{z}\mid\mathbf{x}))$, i.e., finding the Gaussian closest to the true posterior in reverse-KL. $\square$

**Implication.** The VAE encoder approximates the true posterior by the closest Gaussian under reverse-KL — a mean-field approximation. The quality of this approximation determines the tightness of the ELBO.

---

## Appendix R: Quick Reference

### R.1 Generative Model Objectives

| Model | Training Objective | Generation |
|---|---|---|
| Autoregressive | $\max\sum_t\log p_\theta(x_t\mid x_{<t})$ | Ancestral sampling |
| VAE | $\max\mathbb{E}_q[\log p_\theta(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q_\phi\|p)$ | Sample $\mathbf{z}\sim p$, decode |
| Flow | $\max\log p_Z(f^{-1}(\mathbf{x})) + \log|\det J_{f^{-1}}|$ | Sample $\mathbf{z}\sim p_Z$, apply $f$ |
| GAN | $\min_G\max_D\mathbb{E}[\log D(\mathbf{x})] + \mathbb{E}[\log(1-D(G(\mathbf{z})))]$ | Sample $\mathbf{z}\sim p_z$, apply $G$ |
| DDPM | $\min\mathbb{E}\lVert\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,t)\rVert^2$ | Iterative denoising |
| Flow Match. | $\min\mathbb{E}\lVert\mathbf{v}_\theta(\mathbf{x}_t,t) - (\mathbf{x}_1-\mathbf{x}_0)\rVert^2$ | ODE integration |

### R.2 Key Formulas

| Name | Formula |
|---|---|
| ELBO | $\mathbb{E}_q[\log p(\mathbf{x}\mid\mathbf{z})] - D_{\mathrm{KL}}(q_\phi(\mathbf{z}\mid\mathbf{x})\|p(\mathbf{z}))$ |
| Gaussian KL | $\frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$ |
| Reparam. | $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma}\odot\boldsymbol{\varepsilon}$, $\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},I)$ |
| Optimal disc. | $D^*(\mathbf{x}) = p_r(\mathbf{x})/(p_r(\mathbf{x})+p_g(\mathbf{x}))$ |
| Diffusion marginal | $q(\mathbf{x}_t\mid\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar\alpha_t}\mathbf{x}_0, (1-\bar\alpha_t)I)$ |
| Score-noise | $s_\theta \approx -\boldsymbol{\varepsilon}/\sqrt{1-\bar\alpha_t}$ |
| DDIM step | $\mathbf{x}_{t-1} = \sqrt{\bar\alpha_{t-1}}\hat\mathbf{x}_0 + \sqrt{1-\bar\alpha_{t-1}}\boldsymbol{\varepsilon}_\theta$ |
| CFG | $\tilde\boldsymbol{\varepsilon} = (1+w)\boldsymbol{\varepsilon}(\mathbf{x}_t,y) - w\boldsymbol{\varepsilon}(\mathbf{x}_t,\emptyset)$ |
| CFM loss | $\mathbb{E}\lVert\mathbf{v}_\theta(\mathbf{x}_t,t) - (\mathbf{x}_1-\mathbf{x}_0)\rVert^2$ |
| FID | $\lVert\boldsymbol{\mu}_r-\boldsymbol{\mu}_g\rVert_2^2 + \operatorname{tr}(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2})$ |
