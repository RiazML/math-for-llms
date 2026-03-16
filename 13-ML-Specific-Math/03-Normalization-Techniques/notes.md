[← Back to Activation Functions](../02-Activation-Functions/notes.md) | [Next: Sampling Methods →](../04-Sampling-Methods/notes.md)

---

# Normalization Techniques

> _"By normalizing the inputs to each layer, we can ensure they have the same distribution regardless of the parameters — dramatically accelerating training."_
> — Ioffe & Szegedy, 2015

## Overview

Normalization techniques are among the most impactful innovations in deep learning. Before Batch Normalization (2015), training networks with more than a dozen layers required careful initialization, tiny learning rates, and constant hyperparameter tuning. After BatchNorm, 100-layer networks became routine. The reason is mathematical: normalization controls the statistical properties of layer activations, smooths the loss landscape, and bounds gradient magnitudes — all of which are critical for reliable optimization.

This section develops the complete mathematics of every major normalization technique used in modern AI, from the BatchNorm backward pass (a non-trivial Jacobian derivation) through RMSNorm (the LLaMA standard) to AdaLN-Zero (used in diffusion transformers). A unifying theme runs throughout: each technique is a different answer to the question "over which axes of the activation tensor should we compute mean and variance?" Understanding this axis-choice perspective lets you immediately predict which normalization method suits a given architecture.

We also cover the theory: Santurkar et al.'s 2018 proof that BatchNorm makes the loss landscape β-smooth, the implicit regularization interpretation, spectral normalization's Lipschitz constraint, and the signal propagation perspective that motivates normalization-free networks. By the end you will be able to choose, implement from scratch, and debug any normalization method used in production 2026 LLMs, diffusion models, and GANs.

## Prerequisites

- **Activation functions** (`02-Activation-Functions/notes.md`) — pre-activations, gradient flow through nonlinearities, vanishing gradient analysis
- **Loss functions** (`01-Loss-Functions/notes.md`) — cross-entropy, the chain rule in backpropagation
- **Matrix calculus** — Jacobians, chain rule for vector-valued functions
- **Linear algebra basics** — SVD, singular values, spectral norm $\lVert A \rVert_2 = \sigma_{\max}(A)$
- **Statistics** — sample mean, variance, exponential moving averages

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: BN/LN/RMSNorm forward and backward passes, loss landscape smoothness, spectral norm power iteration, AdaIN style transfer, pre-norm vs post-norm gradient analysis |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems covering BN backward pass, normalisation axis comparison, RMSNorm vs LayerNorm, spectral norm implementation, AdaIN style transfer, pre-norm training dynamics, Welford algorithm, and LLM norm ablations |

## Learning Objectives

After completing this section, you will:

- Derive the complete BatchNorm backward pass Jacobian $\partial \mathcal{L}/\partial \mathbf{x}$ and explain why it couples gradients across samples in a mini-batch
- Explain the difference between training and inference behavior of BatchNorm and identify the running statistics mismatch bug
- State the normalisation axis for BatchNorm, LayerNorm, InstanceNorm, GroupNorm, and RMSNorm in unified tensor notation
- Prove that RMSNorm and LayerNorm are equivalent when the input has zero mean
- Implement spectral normalization via power iteration and verify the 1-Lipschitz property
- Explain the Santurkar et al. (2018) loss landscape smoothing theorem and its implications for learning rate choice
- Describe pre-norm vs post-norm placement and explain why pre-norm dominates in transformers deeper than 12 layers
- Identify which normalization is used in LLaMA, Mistral, GPT-2, BERT, ResNet-50, and DiT
- Implement AdaIN and explain how it separates content and style via first and second moments
- Implement Welford's online algorithm for numerically stable variance and explain when naive variance estimation fails

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 The Internal Covariate Shift Problem](#11-the-internal-covariate-shift-problem)
  - [1.2 Loss Landscape Perspective](#12-loss-landscape-perspective)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Taxonomy by Normalisation Axis](#14-taxonomy-by-normalisation-axis)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 The General Normalisation Map](#21-the-general-normalisation-map)
  - [2.2 Normalisation as a Projection](#22-normalisation-as-a-projection)
  - [2.3 Non-examples and Edge Cases](#23-non-examples-and-edge-cases)
- [3. Batch Normalisation](#3-batch-normalisation)
  - [3.1 Forward Pass](#31-forward-pass)
  - [3.2 Backward Pass](#32-backward-pass)
  - [3.3 Running Statistics](#33-running-statistics)
  - [3.4 Effect on Gradient Flow](#34-effect-on-gradient-flow)
- [4. Layer Normalisation](#4-layer-normalisation)
  - [4.1 Definition and Properties](#41-definition-and-properties)
  - [4.2 Jacobian and Backward Pass](#42-jacobian-and-backward-pass)
  - [4.3 Pre-Norm vs Post-Norm Architecture](#43-pre-norm-vs-post-norm-architecture)
- [5. Group and Instance Normalisation](#5-group-and-instance-normalisation)
  - [5.1 Group Norm](#51-group-norm)
  - [5.2 Instance Norm](#52-instance-norm)
  - [5.3 Dimensional Summary](#53-dimensional-summary)
- [6. RMSNorm](#6-rmsnorm)
  - [6.1 Definition](#61-definition)
  - [6.2 Why Mean-Centring May Be Redundant](#62-why-mean-centring-may-be-redundant)
  - [6.3 Backward Pass and Computational Savings](#63-backward-pass-and-computational-savings)
- [7. Weight Normalization and Spectral Normalization](#7-weight-normalization-and-spectral-normalization)
  - [7.1 Weight Normalization](#71-weight-normalization)
  - [7.2 Spectral Normalization](#72-spectral-normalization)
  - [7.3 Spectral Norm and Lipschitz Networks](#73-spectral-norm-and-lipschitz-networks)
- [8. Adaptive Normalization](#8-adaptive-normalization)
  - [8.1 Adaptive Instance Normalization (AdaIN)](#81-adaptive-instance-normalization-adain)
  - [8.2 Conditional Batch Normalization and FiLM](#82-conditional-batch-normalization-and-film)
  - [8.3 AdaLN-Zero in Diffusion Transformers](#83-adaln-zero-in-diffusion-transformers)
- [9. Theoretical Analysis](#9-theoretical-analysis)
  - [9.1 Loss Landscape Smoothing](#91-loss-landscape-smoothing)
  - [9.2 Implicit Regularisation](#92-implicit-regularisation)
  - [9.3 Normalisation-Free Networks](#93-normalisation-free-networks)
- [10. Numerical Stability](#10-numerical-stability)
  - [10.1 Welford's Online Algorithm](#101-welfords-online-algorithm)
  - [10.2 Mixed Precision Considerations](#102-mixed-precision-considerations)
- [11. Applications in Modern AI](#11-applications-in-modern-ai)
  - [11.1 Transformer Norms](#111-transformer-norms)
  - [11.2 ResNet and CNN Norms](#112-resnet-and-cnn-norms)
  - [11.3 GAN Norms](#113-gan-norms)
  - [11.4 Diffusion Models](#114-diffusion-models)
- [12. Common Mistakes](#12-common-mistakes)
- [13. Exercises](#13-exercises)
- [14. Why This Matters for AI (2026 Perspective)](#14-why-this-matters-for-ai-2026-perspective)
- [15. Conceptual Bridge](#15-conceptual-bridge)

---

## 1. Intuition and Motivation

### 1.1 The Internal Covariate Shift Problem

When training a deep network, the parameters of every layer change at every gradient step. This means the distribution of inputs to layer $l$ shifts as the parameters of layers $1, \ldots, l-1$ change. Ioffe and Szegedy (2015) named this phenomenon **internal covariate shift**: even if the distribution of the network's raw inputs $\mathbf{x}$ is fixed, the distribution of $\mathbf{z}^{[l]}$ drifts throughout training.

Internal covariate shift creates a compounding problem. Each layer must continuously adapt to a moving target distribution. If layer 3's inputs suddenly have mean 5 and variance 100 instead of the mean-0, unit-variance distribution it saw at initialization, the learning signal becomes noisy and the gradient steps taken for layer 3 may be based on stale statistics from layer 4. In practice, this forces practitioners to use very small learning rates (to prevent any single update from shifting the distribution too much) and very careful initialization.

Consider a linear layer followed by sigmoid: $y = \sigma(Wx + b)$. If the pre-activation $Wx + b$ has large variance, the sigmoid saturates and its gradient is nearly zero. This is exactly the vanishing gradient problem studied in §3 of the Activation Functions section — but internal covariate shift *dynamically creates* this saturation during training even if initialization was good.

**For AI:** In early transformers (before Pre-LN), a common failure mode was that after a few thousand training steps, the deeper layers' activations would drift to extreme values, causing gradient explosion. The introduction of LayerNorm to the transformer architecture directly addressed this — not by preventing all distribution shift, but by guaranteeing that the input to each sublayer has controlled statistics regardless of what the previous layers have learned.

### 1.2 Loss Landscape Perspective

While internal covariate shift is a useful intuition, a mathematically sharper explanation of why BatchNorm works comes from loss landscape theory. Santurkar et al. (2018) showed that BatchNorm does not primarily reduce internal covariate shift (the distributions still shift; they just do so more slowly). Instead, BatchNorm's key effect is to make the loss landscape **β-smooth**.

A loss $\mathcal{L}$ is β-smooth if its gradient is β-Lipschitz:
$$\lVert \nabla \mathcal{L}(\boldsymbol{\theta}) - \nabla \mathcal{L}(\boldsymbol{\theta}') \rVert \leq \beta \lVert \boldsymbol{\theta} - \boldsymbol{\theta}' \rVert$$

This means the gradient cannot change too fast. A smoother landscape (smaller β) allows larger gradient steps without overshooting minima, which is why BatchNorm empirically allows 10× larger learning rates.

Formally, Santurkar et al. proved that with BatchNorm, the gradient magnitude at each normalised layer is bounded:
$$\lVert \nabla_{\hat{\mathbf{x}}} \mathcal{L} \rVert \leq \frac{\lVert \gamma \rVert}{\sqrt{\sigma^2 + \epsilon}} \cdot C$$

where $C$ depends only on the loss curvature downstream. Since $\gamma / \sqrt{\sigma^2 + \epsilon}$ is controlled (γ is typically initialised to 1 and $\sigma^2$ is at least $\epsilon$), gradient magnitudes are bounded independently of the depth. Without normalisation, gradients can grow or shrink exponentially with depth.

**For AI:** The β-smoothness perspective explains why training instabilities in large transformers are often diagnosed by watching gradient norm histograms. When gradient norms explode at certain layers, it indicates those layers' loss landscape is not being properly smoothed. Adding normalization (or switching from Post-LN to Pre-LN) restores smoothness and stabilises training.

### 1.3 Historical Timeline

```
NORMALISATION TECHNIQUES — HISTORICAL TIMELINE
════════════════════════════════════════════════════════════════════════

  1998  LeCun et al.       Input whitening: normalise network inputs
                           to zero mean, unit variance before training
                           → first recognition of distribution mismatch

  2015  Ioffe & Szegedy    Batch Normalisation (BatchNorm)
                           → Trains 14× faster; ResNet enabled
                           → Internal covariate shift framing

  2016  Ba et al.          Layer Normalisation (LayerNorm)
                           → Works for RNNs and variable-length sequences
                           → No batch dependency

  2016  Ulyanov et al.     Instance Normalisation (InstanceNorm)
                           → Real-time style transfer
                           → Per-channel per-sample statistics

  2017  Salimans et al.    Weight Normalisation
                           → Reparameterises weights, not activations
                           → Useful for online learning, RL

  2018  Wu & He            Group Normalisation (GroupNorm)
                           → Works at batch size 1
                           → Standard for detection/segmentation

  2018  Miyato et al.      Spectral Normalisation
                           → SNGAN, 1-Lipschitz discriminators
                           → Stable GAN training

  2019  Huang & Belongie   AdaIN in style transfer (earlier 2017)
                           → Style via feature statistics matching

  2019  Zhang et al.       RMSNorm (Root Mean Square Norm)
                           → Drops mean centring
                           → Lower compute, similar accuracy

  2021  Brock et al.       NFNets (normalisation-free networks)
                           → Signal propagation theory replaces BN

  2022  Wang et al.        DeepNorm: Pre-LN + init scaling
                           → 1000-layer transformers stable

  2023  LLaMA, Mistral     RMSNorm becomes LLM standard
                           → All major open-source LLMs use RMSNorm

  2023  Peebles & Xie      DiT: AdaLN-Zero for diffusion transformers
                           → Class+timestep conditioned normalisation

════════════════════════════════════════════════════════════════════════
```

### 1.4 Taxonomy by Normalisation Axis

The cleanest way to understand all normalisation methods is through the lens of which axes of the input tensor they normalise over. Consider an input tensor with shape $(N, C, H, W)$ where $N$ is batch size, $C$ is channels/features, and $H, W$ are spatial dimensions (for sequence models, collapse $H, W$ to a single sequence dimension $T$ and set $C = d_{\text{model}}$).

```
NORMALISATION METHODS — AXIS OF STATISTICS COMPUTATION
════════════════════════════════════════════════════════════════════════

  Input tensor shape: (N, C, H, W)
  ─────────────────────────────────────────────────────

  Batch Norm       Axes: N, H, W   Stats shape: (C,)
  [BN]             ┌─────────────────────────────────┐
                   │ For each channel c:             │
                   │  μ_c = mean over N, H, W        │
                   │  σ²_c = var over N, H, W        │
                   └─────────────────────────────────┘

  Layer Norm       Axes: C, H, W   Stats shape: (N,)
  [LN]             ┌─────────────────────────────────┐
                   │ For each sample n:              │
                   │  μ_n = mean over C, H, W        │
                   │  σ²_n = var over C, H, W        │
                   └─────────────────────────────────┘

  Instance Norm    Axes: H, W      Stats shape: (N, C)
  [IN]             ┌─────────────────────────────────┐
                   │ For each (n, c) pair:           │
                   │  μ_{n,c} = mean over H, W       │
                   │  σ²_{n,c} = var over H, W       │
                   └─────────────────────────────────┘

  Group Norm       Axes: C/G, H, W   Stats shape: (N, G)
  [GN]             ┌─────────────────────────────────┐
                   │ Divide C channels into G groups │
                   │ For each (n, g) pair:           │
                   │  μ_{n,g} = mean over C/G, H, W  │
                   │  σ²_{n,g} = var over C/G, H, W  │
                   └─────────────────────────────────┘

  RMSNorm          Axes: C (no mean)   Stats shape: (N,)
  [RMS]            ┌─────────────────────────────────┐
                   │ For each sample n:              │
                   │  rms_n = sqrt(mean of x² over C)│
                   │  No mean subtraction            │
                   └─────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

**For AI:** This taxonomy is practically useful when designing architectures. The question "should I use BN or LN?" reduces to: "does my task have consistent batch statistics?" CNNs with large batches: use BN. Sequence models: use LN (batch statistics are not well-defined for variable-length sequences). Small-batch detection: use GN. Large language models where every FLOPs counts: use RMSNorm.


## 2. Formal Definitions

### 2.1 The General Normalisation Map

All normalisation methods are instances of a single template. Given an input $\mathbf{x} \in \mathbb{R}^d$ (a vector of activations at some "normalisation unit"), the general normalisation map is:

$$\operatorname{Norm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}} + \boldsymbol{\beta}$$

where:
- $\boldsymbol{\mu} \in \mathbb{R}^d$ is the mean vector (computed or zero, depending on the method)
- $\boldsymbol{\sigma} \in \mathbb{R}^d$ is the standard deviation vector (or RMS)
- $\boldsymbol{\gamma} \in \mathbb{R}^d$ is the learned **scale** (also called gain or weight)
- $\boldsymbol{\beta} \in \mathbb{R}^d$ is the learned **shift** (also called bias)
- $\odot$ denotes element-wise multiplication

The division $(\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$ is element-wise. In practice, $\boldsymbol{\sigma}$ is replaced by $\sqrt{\boldsymbol{\sigma}^2 + \epsilon}$ for numerical stability, where $\epsilon \approx 10^{-5}$ to $10^{-6}$.

The learnable parameters $\gamma$ and $\beta$ allow the network to undo the normalisation if it is beneficial. For example, after normalising to zero mean, the network can shift the mean to any value by learning $\beta$. This is crucial: without $\gamma$ and $\beta$, normalisation would permanently remove the representational capacity to have non-zero mean or non-unit variance in the activations.

**Shared vs. per-element parameters:** In BatchNorm (over the batch and spatial dimensions), $\gamma$ and $\beta$ have dimension $C$ (one per channel) because all spatial positions and batch elements share the same normalisation statistics. In LayerNorm (over features), $\gamma$ and $\beta$ have dimension $d_{\text{model}}$ (one per feature). In RMSNorm, $\beta = \mathbf{0}$ (no shift) and only $\gamma$ is learned.

**Initialization:** Standard initialization is $\gamma = \mathbf{1}$ (identity scale) and $\beta = \mathbf{0}$ (zero shift), so the normalisation starts as a pure standardisation. The AdaLN-Zero trick (§8.3) initializes $\gamma = \mathbf{0}$, making the entire normalised block output zero at initialization — a powerful training stability trick for very deep networks.

### 2.2 Normalisation as a Projection

There is an elegant geometric interpretation of normalisation: it is an orthogonal projection onto a constraint surface.

For a vector $\mathbf{x} \in \mathbb{R}^d$, standardisation maps it to:
$$\hat{\mathbf{x}} = \frac{\mathbf{x} - \mu(\mathbf{x})\mathbf{1}}{\sigma(\mathbf{x})}$$

where $\mu(\mathbf{x}) = \frac{1}{d}\sum_i x_i$ and $\sigma(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$.

The result $\hat{\mathbf{x}}$ lies on the intersection of:
- The hyperplane $\{\mathbf{v} : \sum_i v_i = 0\}$ (zero mean)
- The sphere $\{\mathbf{v} : \lVert \mathbf{v} \rVert_2 = \sqrt{d}\}$ (unit variance, since $\operatorname{Var}(\hat{\mathbf{x}}) = 1$ implies $\sum_i \hat{x}_i^2 = d$)

This $(d-2)$-dimensional manifold is sometimes called the **normalised sphere** in feature space. Each call to LayerNorm/BN projects the activation vector onto this manifold before applying the affine transform $\gamma \odot \hat{\mathbf{x}} + \beta$.

**Connection to whitening:** True whitening would decorrelate all features, mapping $\mathbf{x}$ to $\Sigma^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$ where $\Sigma$ is the full covariance matrix. Standard normalisation only equalises the *diagonal* of the covariance (variances), not the off-diagonal (correlations). Full whitening is rarely used in deep learning because it requires expensive covariance estimation and inversion, and the cross-feature correlations often carry useful information.

### 2.3 Non-examples and Edge Cases

**BatchNorm with batch size 1:** With $m = 1$, the batch variance $\sigma_B^2 = 0$ exactly, so $\hat{\mathbf{x}} = 0/\sqrt{\epsilon}$ — the entire normalised output is zero regardless of the input. BatchNorm with $m = 1$ is undefined/degenerate. This is why BatchNorm is never used for inference on a single sample without pre-computed running statistics, and why it fails for auto-regressive LM generation where each token is processed independently.

**BatchNorm for RNNs/LSTMs:** Consider applying BatchNorm to an LSTM hidden state $\mathbf{h}_t \in \mathbb{R}^{d}$ at time step $t$. For each time step, the statistics would be computed over the batch dimension only, giving a batch mean vector $\boldsymbol{\mu}_t \in \mathbb{R}^d$. But $\boldsymbol{\mu}_t$ depends on $t$ — the statistics at step 1 of a sequence are different from step 100. This means you need separate $\gamma_t, \beta_t$ for every time step, which is impractical for variable-length sequences. LayerNorm computes statistics over the feature dimension for each $(n, t)$ pair independently, which naturally handles variable lengths.

**GroupNorm with $G = 1$ vs. LayerNorm:** For a $(N, C)$ input (no spatial dimensions), GroupNorm with $G=1$ and LayerNorm are identical — both compute mean and variance over all $C$ features per sample. For a $(N, C, H, W)$ input, GroupNorm with $G=1$ normalises over all $C \times H \times W$ elements per sample, while LayerNorm in PyTorch normalises over the last $k$ dimensions specified. The difference matters for CNNs: LayerNorm normalises each pixel position independently across channels, GroupNorm normalises spatial positions together within groups.

**RMSNorm is not a special case of LayerNorm with $\beta = 0$:** RMSNorm normalises by the root mean square, not the standard deviation. The standard deviation is $\sqrt{\text{Var}(\mathbf{x})}$, while the RMS is $\sqrt{\mathbb{E}[\mathbf{x}^2]}$. These differ when $\mathbb{E}[\mathbf{x}] \neq 0$. RMSNorm without mean subtraction is strictly a different operation from LayerNorm with $\beta$ fixed to zero.


## 3. Batch Normalisation

### 3.1 Forward Pass

Given a mini-batch of $m$ pre-activations for a single feature/channel, $\mathcal{B} = \{x_1, \ldots, x_m\}$ where $x_i \in \mathbb{R}$:

**Step 1 — Batch statistics:**
$$\mu_\mathcal{B} = \frac{1}{m} \sum_{i=1}^{m} x_i, \qquad \sigma_\mathcal{B}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_\mathcal{B})^2$$

Note: most frameworks use the *biased* (population) variance $1/m$, not the unbiased $1/(m-1)$, because we are normalising this specific batch, not estimating a population parameter.

**Step 2 — Normalise:**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}}$$

**Step 3 — Affine transform:**
$$y_i = \gamma \hat{x}_i + \beta$$

For a full layer with $C$ channels and spatial dimensions $(H, W)$, the batch has shape $(m, C, H, W)$. The statistics $\mu_c, \sigma_c^2$ are computed over axes $(m, H, W)$ for each channel $c$ independently, giving vectors $\boldsymbol{\mu} \in \mathbb{R}^C$ and $\boldsymbol{\sigma}^2 \in \mathbb{R}^C$. The parameters $\gamma, \beta \in \mathbb{R}^C$ are then broadcast across all spatial positions and batch elements.

**For AI (ResNets):** In ResNet-50, each convolutional layer is followed by BatchNorm before the ReLU activation. The BatchNorm parameters add only $2C$ scalars per layer (γ and β), which is negligible compared to the convolutional weights. But the effect is enormous: ResNet-50 without BN fails to converge at standard learning rates, while with BN it trains reliably in hours.

### 3.2 Backward Pass

The BatchNorm backward pass is non-trivial because the normalisation operation couples all $m$ samples in the batch — the gradient for $x_i$ depends on all other $x_j$ through the shared mean and variance.

Let $\delta_i = \partial \mathcal{L} / \partial y_i$ be the upstream gradient. We derive $\partial \mathcal{L} / \partial x_i$ step by step.

**Gradients w.r.t. γ and β:**
$$\frac{\partial \mathcal{L}}{\partial \gamma} = \sum_{i=1}^{m} \delta_i \hat{x}_i, \qquad \frac{\partial \mathcal{L}}{\partial \beta} = \sum_{i=1}^{m} \delta_i$$

These are straightforward: γ scales $\hat{x}_i$, so its gradient is $\sum_i \delta_i \hat{x}_i$. β is a constant offset so its gradient is $\sum_i \delta_i$.

**Gradient w.r.t. $\hat{x}_i$:**
$$\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \delta_i \cdot \gamma$$

**Gradient w.r.t. $\sigma_\mathcal{B}^2$:**
$$\frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{-(x_i - \mu_\mathcal{B})}{2(\sigma_\mathcal{B}^2 + \epsilon)^{3/2}}$$

**Gradient w.r.t. $\mu_\mathcal{B}$:**
$$\frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} = \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} \cdot \frac{-2}{m}\sum_{i=1}^{m}(x_i - \mu_\mathcal{B})$$

**Final gradient w.r.t. $x_i$:**
$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} + \frac{\partial \mathcal{L}}{\partial \sigma_\mathcal{B}^2} \cdot \frac{2(x_i - \mu_\mathcal{B})}{m} + \frac{\partial \mathcal{L}}{\partial \mu_\mathcal{B}} \cdot \frac{1}{m}$$

Substituting and simplifying (substituting $\frac{\partial \mathcal{L}}{\partial \hat{x}_i} = \gamma \delta_i$ and collecting terms):

$$\boxed{\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} \left( \delta_i - \frac{1}{m}\sum_{j=1}^m \delta_j - \hat{x}_i \cdot \frac{1}{m}\sum_{j=1}^m \delta_j \hat{x}_j \right)}$$

This is the key formula. Three terms:
1. $\delta_i$: the upstream gradient passes through as-is
2. $-\frac{1}{m}\sum_j \delta_j$: subtract the batch mean of gradients (centring effect)
3. $-\hat{x}_i \frac{1}{m}\sum_j \delta_j \hat{x}_j$: subtract a term proportional to $\hat{x}_i$ (projection effect)

**Why BN couples gradients:** Terms 2 and 3 depend on all $m$ samples in the batch. This means the gradient for sample $i$ depends on the loss contributions from all other samples $j \neq i$ — a form of implicit data augmentation. Changing one sample's gradient changes all others.

### 3.3 Running Statistics

During training, BatchNorm maintains running (exponential moving average) estimates of the population mean and variance:

$$\mu_{\text{running}} \leftarrow (1 - \alpha) \mu_{\text{running}} + \alpha \mu_\mathcal{B}$$
$$\sigma^2_{\text{running}} \leftarrow (1 - \alpha) \sigma^2_{\text{running}} + \alpha \sigma^2_\mathcal{B}$$

where $\alpha$ is the **momentum** parameter (default $0.1$ in PyTorch, often confusingly the *inverse* of EMA decay: a larger $\alpha$ gives *more* weight to recent batches).

During inference, the running statistics are used instead of batch statistics:
$$\hat{x} = \frac{x - \mu_{\text{running}}}{\sqrt{\sigma^2_{\text{running}} + \epsilon}}, \quad y = \gamma \hat{x} + \beta$$

This is crucial: at inference time, BatchNorm must be in **eval mode**. Using training mode at inference on single samples gives nonsensical outputs because the "batch" of size 1 has $\sigma^2 = 0$.

**Common bug — train/eval mode mismatch:** The most common BatchNorm bug is forgetting to call `model.eval()` before inference. In PyTorch, this switches all BatchNorm layers from using batch statistics to using running statistics. Inference with training-mode BN can give radically different outputs, especially for small batches. A related bug: calling `model.eval()` during fine-tuning, which freezes the running statistics to values from pre-training, causing distribution mismatch on the new task.

### 3.4 Effect on Gradient Flow

BatchNorm has a remarkable implicit effect on the effective learning rate. Consider a weight matrix $W$ that feeds into a BatchNorm layer. The output of the network (and thus the loss) depends on $W$ only through the direction of $W$'s rows, not their magnitude, because BatchNorm normalises away the scale:

$$\operatorname{BN}(cWx) = \operatorname{BN}(Wx) \quad \text{for any scalar } c > 0$$

This scale invariance means that gradient steps in the magnitude direction of $W$ have no effect on the output. More precisely, if we decompose the weight into magnitude $\lVert \mathbf{w} \rVert$ and direction $\hat{\mathbf{w}} = \mathbf{w}/\lVert \mathbf{w} \rVert$, the effective learning rate for the direction is:

$$\eta_{\text{effective}} = \frac{\eta}{\lVert \mathbf{w} \rVert^2}$$

This is the **implicit learning rate normalisation** effect: as weights grow, the effective learning rate shrinks, providing automatic adaptation. Large weights are updated less aggressively in direction space, preventing runaway growth.

**For AI:** This is why networks with BatchNorm can be trained with much larger learning rates without divergence. The BN layer acts as a passive regulator: as weights grow larger, BN automatically reduces their effective learning rate in the optimization landscape. This is one reason why BatchNorm made the training of very deep ResNets feasible.


## 4. Layer Normalisation

### 4.1 Definition and Properties

Layer Normalisation (Ba et al., 2016) computes statistics over the feature dimension for each individual sample, independently of all other samples in the batch.

For input $\mathbf{x}^{(n)} \in \mathbb{R}^H$ (the $H$-dimensional feature vector for sample $n$):

$$\mu_n = \frac{1}{H} \sum_{i=1}^{H} x_i^{(n)}, \qquad \sigma_n^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i^{(n)} - \mu_n)^2$$

$$\hat{x}_i^{(n)} = \frac{x_i^{(n)} - \mu_n}{\sqrt{\sigma_n^2 + \epsilon}}, \qquad y_i^{(n)} = \gamma_i \hat{x}_i^{(n)} + \beta_i$$

Key properties of LayerNorm:

1. **No batch dependency:** $\mu_n$ and $\sigma_n^2$ depend only on sample $n$, not on any other sample. LayerNorm behaves identically with batch size 1 and batch size 1000.

2. **Consistent train/inference behaviour:** There are no running statistics to maintain. Inference is identical to training (both use the sample's own statistics), eliminating the train/eval mode distinction that plagues BatchNorm.

3. **Handles variable-length sequences:** For a sequence of tokens $\mathbf{x}_1, \ldots, \mathbf{x}_T$ where $\mathbf{x}_t \in \mathbb{R}^{d_{\text{model}}}$, LayerNorm is applied to each $\mathbf{x}_t$ independently, normalising over the $d_{\text{model}}$ features. The sequence length $T$ does not affect the normalisation.

4. **Equivariant to feature permutations (with tied γ, β):** If all $\gamma_i = \gamma$ and $\beta_i = \beta$ (scalar parameters), LayerNorm is invariant to permutations of the feature vector.

**For AI (Transformers):** LayerNorm is applied twice per transformer block: once before (or after) the attention sublayer, and once before (or after) the FFN sublayer. In the original "Attention is All You Need" transformer, LayerNorm appears *after* the residual addition (Post-LN). In GPT-2, LLaMA, and virtually all modern LLMs, LayerNorm appears *before* the sublayer (Pre-LN). See §4.3 for why.

### 4.2 Jacobian and Backward Pass

The LayerNorm backward pass is structurally similar to the BatchNorm backward pass, but gradients couple across features within a sample rather than across samples within a batch.

Let $\mathbf{x} \in \mathbb{R}^H$ (dropping the sample index for clarity), $\boldsymbol{\delta} = \partial \mathcal{L} / \partial \mathbf{y}$ be the upstream gradient.

**Gradients w.r.t. γ and β:**
$$\frac{\partial \mathcal{L}}{\partial \gamma_i} = \sum_n \delta_i^{(n)} \hat{x}_i^{(n)}, \qquad \frac{\partial \mathcal{L}}{\partial \beta_i} = \sum_n \delta_i^{(n)}$$

**Gradient w.r.t. $\mathbf{x}$** (for a single sample, dropping superscript $(n)$):

The full Jacobian $\partial \mathbf{y} / \partial \mathbf{x}$ is:
$$\frac{\partial y_i}{\partial x_j} = \frac{\gamma_i}{\sqrt{\sigma^2 + \epsilon}} \left( \delta_{ij} - \frac{1}{H} - \frac{\hat{x}_i \hat{x}_j}{H} \right)$$

where $\delta_{ij}$ is the Kronecker delta. This has the same three-term structure as the BN gradient, but now coupling is across features (the $1/H$ terms sum over features $j$ for a fixed sample $i$).

Collecting, the input gradient is:
$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{\sqrt{\sigma^2 + \epsilon}} \left( \delta_i - \frac{1}{H}\sum_{j=1}^H \delta_j - \hat{x}_i \cdot \frac{1}{H}\sum_{j=1}^H \delta_j \hat{x}_j \right)$$

where $\delta_i = \partial \mathcal{L}/\partial y_i \cdot \gamma_i / \sqrt{\sigma^2 + \epsilon}$... more precisely, letting $g_i = (\partial \mathcal{L}/\partial y_i) \gamma_i$:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sqrt{\sigma^2+\epsilon}} \left( g_i - \frac{1}{H}\sum_j g_j - \hat{x}_i \cdot \frac{1}{H}\sum_j g_j \hat{x}_j \right)$$

**Rank deficiency:** The Jacobian of LayerNorm (as a map $\mathbb{R}^H \to \mathbb{R}^H$) is rank $H - 2$, not $H$. Two directions in input space have zero Jacobian: the all-ones vector $\mathbf{1}/\sqrt{H}$ (translating all inputs by a constant changes the mean but not $\hat{\mathbf{x}}$) and the input direction $\hat{\mathbf{x}}$ itself (scaling all inputs by a constant changes the variance but not $\hat{\mathbf{x}}$). These are the two degrees of freedom removed by the normalisation.

### 4.3 Pre-Norm vs Post-Norm Architecture

The placement of LayerNorm relative to the residual connection has a profound effect on gradient flow and training stability.

**Post-Norm (Original Transformer, "Attention is All You Need"):**
$$\mathbf{x}_{l+1} = \operatorname{LayerNorm}\big(\mathbf{x}_l + \operatorname{Sublayer}_l(\mathbf{x}_l)\big)$$

**Pre-Norm (GPT-2, LLaMA, Mistral, all modern LLMs):**
$$\mathbf{x}_{l+1} = \mathbf{x}_l + \operatorname{Sublayer}_l\big(\operatorname{LayerNorm}(\mathbf{x}_l)\big)$$

The gradient flow analysis reveals the key difference. For Post-Norm, the gradient of the loss with respect to the residual stream $\mathbf{x}_l$ flows through:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{l+1}} \cdot \frac{\partial \operatorname{LayerNorm}}{\partial \mathbf{x}_l}$$

The LayerNorm Jacobian scales gradients by $1/\sqrt{\sigma^2 + \epsilon}$, which can be small if the activations have large variance. More critically, for Post-Norm at initialisation, the gradient through the residual connection passes through the LayerNorm, while for Pre-Norm the gradient flows directly through the residual path without any normalisation.

```
GRADIENT FLOW COMPARISON
════════════════════════════════════════════════════════════════════════

  Post-Norm:                      Pre-Norm:
  ─────────────────────           ─────────────────────
  x_l ──┬──→ Sublayer ──┐         x_l ──┬──────────────→ +──→ x_{l+1}
        │               ↓               │                ↑
        └───────────────+               └──→ LN → Sub ───┘
                        ↓
                        LN → x_{l+1}

  Gradient path:                  Gradient path:
  ∂L/∂x_l flows through LN       ∂L/∂x_l passes residual
  (possibly squashed/scaled)      directly without LN
  Gradient at l=1:                Gradient at l=1:
  ≈ (LN Jacobian)^L × ∂L/∂x_L   ≈ ∂L/∂x_L (direct path)

════════════════════════════════════════════════════════════════════════
```

**For AI:** At large depth ($L \geq 24$ layers), Post-Norm requires careful learning rate warmup to prevent gradient explosion/vanishing at the first few training steps. Pre-Norm does not require warmup and trains stably from the start. This is why all modern LLMs (GPT-2, GPT-3, LLaMA 1/2/3, Mistral, Gemma, Falcon, Phi) use Pre-Norm. The 2022 DeepNorm paper (Wang et al.) showed you can scale Post-Norm to 1000+ layers by scaling the residual connection as $\mathbf{x} + \alpha \cdot \operatorname{Sublayer}(\mathbf{x})$ with $\alpha < 1$ — but Pre-Norm remains the practical default.


## 5. Group and Instance Normalisation

### 5.1 Group Norm

Group Normalisation (Wu & He, 2018) divides the $C$ channels into $G$ equal groups of size $C/G$ and normalises within each group. For input $\mathbf{x}$ of shape $(N, C, H, W)$:

For sample $n$ and group $g$ (containing channels $c \in \{g \cdot C/G, \ldots, (g+1) \cdot C/G - 1\}$):

$$\mu_{n,g} = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in g} \sum_{h,w} x_{n,c,h,w}$$

$$\sigma_{n,g}^2 = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in g} \sum_{h,w} (x_{n,c,h,w} - \mu_{n,g})^2$$

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,g(c)}}{\sqrt{\sigma_{n,g(c)}^2 + \epsilon}}, \qquad y_{n,c,h,w} = \gamma_c \hat{x}_{n,c,h,w} + \beta_c$$

where $g(c)$ maps channel $c$ to its group index. The parameters $\gamma_c, \beta_c$ are per-channel (shape $C$), same as BatchNorm.

**Special cases:**
- $G = C$ (one channel per group): reduces to **Instance Normalisation** — statistics over $(H, W)$ for each $(n, c)$ pair
- $G = 1$ (all channels in one group): reduces to **Layer Normalisation** — statistics over $(C, H, W)$ for each sample $n$

**Why $G = 32$?** The group size $C/G$ should be large enough to give stable variance estimates. With $C = 256$ and $G = 32$, each group has 8 channels. Empirically, group sizes between 8 and 32 work well for feature maps in detection and segmentation networks.

**For AI (Object Detection):** Mask R-CNN (He et al., 2017) replaced BatchNorm with GroupNorm for two reasons: (1) detection models use small batches (2-4 images per GPU due to large memory requirements), making BN statistics unreliable; (2) RoI-pooled features are processed by the same head network, but different proposals have different sizes, making batch statistics across proposals ill-defined. GroupNorm solves both issues.

### 5.2 Instance Norm

Instance Normalisation (Ulyanov et al., 2016) normalises each feature map independently:

$$\mu_{n,c} = \frac{1}{HW} \sum_{h,w} x_{n,c,h,w}, \qquad \sigma_{n,c}^2 = \frac{1}{HW} \sum_{h,w} (x_{n,c,h,w} - \mu_{n,c})^2$$

$$\hat{x}_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}}$$

The key insight for style transfer: the mean and variance of a feature map encode the **style** (texture, colour palette) of the content. By normalising away these statistics (subtracting mean, dividing by std), InstanceNorm removes style information while preserving content (spatial structure). The style can then be injected back by adding the target style's statistics — this is the AdaIN operation (see §8.1).

**Gram matrix connection:** In texture synthesis, style is traditionally captured by the Gram matrix $G_c = F_c F_c^\top$ where $F_c \in \mathbb{R}^{C \times HW}$ is the reshaped feature map. The diagonal of $G_c / (HW)$ gives the per-channel variances $\sigma_c^2$. InstanceNorm zeroes these diagonal terms, which is a partial disentanglement of style from the representation.

**For AI (StyleGAN):** StyleGAN (Karras et al., 2019) uses **Adaptive InstanceNorm** (AdaIN) to inject style at each layer. The mapping network $f$ produces a style vector $\mathbf{w}$, which is projected to per-channel affine parameters $\gamma_c(\mathbf{w}), \beta_c(\mathbf{w})$. These are applied after normalising the feature maps with InstanceNorm. This lets StyleGAN control fine-grained style attributes at each resolution.

### 5.3 Dimensional Summary

```
NORMALISATION METHODS — UNIFIED TENSOR VIEW
════════════════════════════════════════════════════════════════════════

  For tensor shape (N, C, H, W):

  Method          Axes normalised      Stats shape   Params shape
  ─────────────────────────────────────────────────────────────────
  Batch Norm      N, H, W              (1, C, 1, 1)  γ,β ∈ R^C
  Layer Norm      C, H, W              (N, 1, 1, 1)  γ,β ∈ R^{C×H×W}
  Instance Norm   H, W                 (N, C, 1, 1)  γ,β ∈ R^C
  Group Norm (G)  C/G, H, W            (N, G, 1, 1)  γ,β ∈ R^C
  RMSNorm         C (no mean)          (N, 1, 1, 1)  γ ∈ R^{C×H×W}

  For sequence tensors (N, T, d_model):

  Method          Axes normalised      Notes
  ─────────────────────────────────────────────────────────────────
  Layer Norm      d_model              Standard for transformers
  RMSNorm         d_model (no mean)    LLaMA, Mistral, Gemma
  Batch Norm      N, T                 Requires fixed seq length
  Group Norm      d_model / G          Rare in transformers

  Batch dependency:
    Batch Norm: YES (requires multiple samples)
    All others: NO (computed per sample)

════════════════════════════════════════════════════════════════════════
```

For purely sequence-based models (LLMs, vision transformers), the relevant tensor is $(N, T, d)$ and the normalisation is applied to the $d$-dimensional feature vector at each sequence position independently. LayerNorm normalises over $d$ (features), while a hypothetical batch norm over a sequence would normalise over $N$ (batch) — problematic for variable-length sequences and auto-regressive inference.


## 6. RMSNorm

### 6.1 Definition

Root Mean Square Layer Normalization (RMSNorm; Zhang & Sennrich, 2019) simplifies LayerNorm by removing the mean-centring step. For input $\mathbf{x} \in \mathbb{R}^H$:

$$\operatorname{RMS}(\mathbf{x}) = \sqrt{\frac{1}{H} \sum_{i=1}^{H} x_i^2}$$

$$\operatorname{RMSNorm}(\mathbf{x})_i = \frac{x_i}{\operatorname{RMS}(\mathbf{x})} \cdot \gamma_i$$

There is no $\beta$ (shift) parameter and no mean subtraction. The normalisation is purely a scaling by the inverse RMS, followed by a learned per-feature scale $\gamma_i$.

Equivalently, RMSNorm normalises $\mathbf{x}$ to have unit $\ell^2$ norm (up to a $\sqrt{H}$ factor) and then applies a diagonal scale:

$$\operatorname{RMSNorm}(\mathbf{x}) = \frac{\sqrt{H}}{\lVert \mathbf{x} \rVert_2} \boldsymbol{\gamma} \odot \mathbf{x}$$

This is a projection onto the sphere $\lVert \mathbf{x} \rVert_2 = \sqrt{H}$ followed by element-wise scaling.

**FLOPs comparison:** For a vector of length $H$, the FLOPs count is:
- **LayerNorm:** $\approx 5H$ operations (mean: $H$ adds + 1 divide; variance: $H$ subtracts + $H$ squares + $H$ adds + 1 divide; normalise: $H$ subtracts + $H$ divides; affine: $2H$)
- **RMSNorm:** $\approx 3H$ operations (squares: $H$; mean: $H$ adds + 1 divide; normalise: $H$ divides; scale: $H$)

For $H = 4096$ (LLaMA-7B model dimension), applied at every token position $T$ in the sequence, the savings are significant. In LLaMA-7B with $T = 4096$ and 32 transformer layers (each with 2 RMSNorm applications), LayerNorm would require $5 \times 4096 \times 4096 \times 2 \times 32 \approx 5.4 \times 10^9$ operations just for normalisation, while RMSNorm reduces this by ~40%.

### 6.2 Why Mean-Centring May Be Redundant

The motivation for dropping mean-centring is that **re-centring invariance** is already provided by the following layer's bias term. Consider a transformer FFN:

$$\operatorname{FFN}(\mathbf{x}) = W_2 \sigma(W_1 \operatorname{LN}(\mathbf{x}) + \mathbf{b}_1) + \mathbf{b}_2$$

The bias $\mathbf{b}_1$ in the FFN can represent any mean shift in the normalised activations. If the LayerNorm mean-centres the input, the FFN bias immediately reintroduces a mean. The mean-centring step of LayerNorm and the bias of the next layer are thus *redundant*: removing either one leaves the network's representational capacity unchanged (the remaining parameter compensates).

More formally, for a linear layer $y = W\operatorname{LN}(\mathbf{x}) + \mathbf{b}$: if $\operatorname{LN}$ mean-centres, the mean of $\operatorname{LN}(\mathbf{x})$ is zero and $\mathbf{b}$ controls the mean of $y$. If $\operatorname{RMSNorm}$ is used instead (no mean-centring), the mean of $\operatorname{RMSNorm}(\mathbf{x})$ is non-zero, but $\mathbf{b}$ can still represent the desired mean offset for $y$ — it just needs to absorb both the desired output mean and the mean of the normalised input. As long as $\mathbf{b}$ is a free parameter (which it always is), no representational capacity is lost.

**Empirical evidence:** Zhang & Sennrich (2019) showed that RMSNorm achieves comparable or better machine translation BLEU scores than LayerNorm with 10-20% speedup. LLaMA (Touvron et al., 2023) adopted RMSNorm throughout, and subsequent models (Mistral, Gemma, Phi-2, Falcon) all followed.

**When mean-centring IS needed:** For tasks where the absolute mean of activations carries semantic meaning — e.g., some regression tasks or normalisation flows where distribution matching is critical — mean-centring may still be beneficial. For standard language modelling, it appears redundant.

### 6.3 Backward Pass and Computational Savings

The RMSNorm gradient is simpler than LayerNorm because there is no mean term.

Let $r = \operatorname{RMS}(\mathbf{x})$ and $\hat{x}_i = x_i / r$. The gradient of the loss with respect to $\mathbf{x}$:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{r} \left( g_i - \hat{x}_i \cdot \frac{1}{H} \sum_{j=1}^{H} g_j \hat{x}_j \right)$$

where $g_i = \partial \mathcal{L}/\partial y_i$. This has only two terms (vs. three for LayerNorm), corresponding to:
1. The upstream gradient scaled by $\gamma_i / r$
2. A correction term from the RMS dependence on all inputs (the $1/H \sum_j g_j \hat{x}_j$ term comes from $\partial r / \partial x_i = x_i / (Hr)$)

The missing term relative to LayerNorm is the mean-centring correction $-\frac{1}{H}\sum_j g_j / r$, which vanishes because RMSNorm has no mean subtraction.

**For AI (Flash Attention + RMSNorm):** In modern LLM training pipelines, RMSNorm is fused with the preceding linear projection using kernel fusion techniques. The reduced operation count of RMSNorm (no mean computation) translates to fewer memory reads in the fused kernel, improving both throughput and memory bandwidth utilisation.


## 7. Weight Normalization and Spectral Normalization

### 7.1 Weight Normalization

Weight Normalization (Salimans & Kingma, 2016) takes a different approach: instead of normalising activations (the output of a layer), it reparameterises the weight vectors themselves. For a weight vector $\mathbf{w} \in \mathbb{R}^k$ (one row of a weight matrix):

$$\mathbf{w} = \frac{g}{\lVert \mathbf{v} \rVert_2} \mathbf{v}$$

where $g \in \mathbb{R}$ is a learned scalar (magnitude) and $\mathbf{v} \in \mathbb{R}^k$ is a learned vector (direction). The weight norm is thus always $\lVert \mathbf{w} \rVert_2 = g$, and the direction of $\mathbf{w}$ is $\hat{\mathbf{v}} = \mathbf{v} / \lVert \mathbf{v} \rVert$.

**Gradient computation:** Differentiating the reparameterisation:
$$\nabla_g \mathcal{L} = \nabla_\mathbf{w} \mathcal{L} \cdot \frac{\mathbf{v}}{\lVert \mathbf{v} \rVert}$$

$$\nabla_\mathbf{v} \mathcal{L} = \frac{g}{\lVert \mathbf{v} \rVert} \nabla_\mathbf{w} \mathcal{L} - \frac{g(\nabla_\mathbf{w} \mathcal{L} \cdot \hat{\mathbf{v}})}{\lVert \mathbf{v} \rVert} \hat{\mathbf{v}}$$

The second term is a projection: $\nabla_\mathbf{v} \mathcal{L}$ has the component of $\nabla_\mathbf{w} \mathcal{L}$ along $\hat{\mathbf{v}}$ removed, meaning the gradient in $\mathbf{v}$ space only changes the direction, not the magnitude. The magnitude is controlled entirely by $g$.

**Advantages over BatchNorm:**
- No running statistics and no batch dependency
- Deterministic: the output of a layer is fully determined by the input (no stochastic batch statistics)
- Useful for online learning and reinforcement learning where batch statistics are not meaningful
- Equivalent to BatchNorm in the limit of infinite batch size and with suitable initialisation

**Disadvantage:** Unlike BatchNorm, weight normalisation does not automatically normalise the *activations* — only the weights. If the input $\mathbf{x}$ has large variance, the output $\mathbf{w}^\top \mathbf{x}$ can still have large variance. Salimans & Kingma recommend combining weight normalisation with **mean-only BatchNorm** (which only subtracts the batch mean, not dividing by std) to control activation scale.

### 7.2 Spectral Normalization

Spectral Normalization (Miyato et al., 2018) normalises each weight matrix by its **spectral norm** (largest singular value):

$$\bar{W} = \frac{W}{\sigma_1(W)}, \qquad \sigma_1(W) = \lVert W \rVert_2 = \max_{\lVert \mathbf{v} \rVert = 1} \lVert W\mathbf{v} \rVert_2$$

After spectral normalisation, $\lVert \bar{W} \rVert_2 = 1$, so the layer is 1-Lipschitz: $\lVert \bar{W}\mathbf{x} - \bar{W}\mathbf{x}' \rVert \leq \lVert \mathbf{x} - \mathbf{x}' \rVert$.

**Power iteration for $\sigma_1$:** Computing the full SVD at every training step is expensive. Instead, Miyato et al. use one step of power iteration per gradient update:

1. Maintain vectors $\tilde{\mathbf{u}} \in \mathbb{R}^m$ and $\tilde{\mathbf{v}} \in \mathbb{R}^n$ (initialised randomly, normalised)
2. At each step: $\hat{\mathbf{v}} = W^\top \tilde{\mathbf{u}} / \lVert W^\top \tilde{\mathbf{u}} \rVert_2$
3. $\hat{\mathbf{u}} = W \hat{\mathbf{v}} / \lVert W \hat{\mathbf{v}} \rVert_2$
4. $\tilde{\sigma} = \hat{\mathbf{u}}^\top W \hat{\mathbf{v}} \approx \sigma_1(W)$
5. Normalise: $\bar{W} = W / \tilde{\sigma}$

After a few training steps, $\hat{\mathbf{u}}$ and $\hat{\mathbf{v}}$ converge to the top left and right singular vectors, and the one-step update keeps them accurate as $W$ evolves.

**Gradient through spectral normalisation:** The gradient of the loss with respect to $W$ when using $\bar{W}$:

$$\nabla_W \mathcal{L} = \frac{1}{\tilde{\sigma}} \left( \nabla_{\bar{W}} \mathcal{L} - (\nabla_{\bar{W}} \mathcal{L} \cdot \hat{\mathbf{u}} \hat{\mathbf{v}}^\top) \hat{\mathbf{u}} \hat{\mathbf{v}}^\top \right)$$

The second term removes the component of the gradient along the top singular direction, preventing updates that would change the spectral norm. In practice, this term is often dropped (treated as a stop-gradient through $\tilde{\sigma}$) without loss of performance.

### 7.3 Spectral Norm and Lipschitz Networks

The Lipschitz constant of a composition of linear layers is bounded by the product of spectral norms. For a network $f = f_L \circ \cdots \circ f_1$ where each $f_l(\mathbf{x}) = \phi(W^{[l]}\mathbf{x} + \mathbf{b}^{[l]})$:

$$\operatorname{Lip}(f) \leq \prod_{l=1}^{L} \lVert W^{[l]} \rVert_2 \cdot \prod_{l=1}^{L} K_l$$

where $K_l$ is the Lipschitz constant of activation $\phi$ at layer $l$ (for ReLU, $K_l = 1$; for sigmoid, $K_l = 1/4$; see §6 of the Activation Functions section).

Spectral normalisation constrains each $\lVert W^{[l]} \rVert_2 \leq 1$, so with ReLU activations:
$$\operatorname{Lip}(f) \leq 1^L = 1$$

A 1-Lipschitz function satisfies $\lvert f(\mathbf{x}) - f(\mathbf{x}') \rvert \leq \lVert \mathbf{x} - \mathbf{x}' \rVert$. This constraint is crucial for:

1. **Wasserstein GAN (WGAN):** The Wasserstein distance requires the critic (discriminator) to be 1-Lipschitz. Spectral normalisation is one of two common ways to enforce this (the other is gradient penalty, WGAN-GP).

2. **Certified adversarial robustness:** If $f$ is 1-Lipschitz, then any perturbation $\lVert \boldsymbol{\delta} \rVert \leq \epsilon$ of the input changes the output by at most $\epsilon$. This gives a certified robustness guarantee.

3. **Stable GAN training:** Without Lipschitz constraint, the discriminator gradient grows unboundedly as the generator improves, causing training instabilities. Spectral normalisation keeps the discriminator gradient bounded throughout training.

**For AI (SNGAN, BigGAN):** SNGAN (Miyato et al., 2018) demonstrated that spectral normalisation alone (without gradient penalty) stabilises GAN training and produces state-of-the-art image generation on ImageNet. BigGAN (Brock et al., 2019) extended this to class-conditional generation at 512×512 resolution, using spectral normalisation in both generator and discriminator.


## 8. Adaptive Normalization

### 8.1 Adaptive Instance Normalization (AdaIN)

Adaptive Instance Normalization (AdaIN; Huang & Belongie, 2017) is the foundation of neural style transfer and many generative models. The key insight: the *mean* and *variance* of a feature map encode its **style** (texture, colour), while the spatial structure encodes **content**. By replacing the statistics of a content image with those of a style image, we transfer style while preserving content.

Given content feature map $\mathbf{x} \in \mathbb{R}^{C \times H \times W}$ and style feature map $\mathbf{y} \in \mathbb{R}^{C \times H \times W}$:

$$\operatorname{AdaIN}(\mathbf{x}, \mathbf{y})_c = \sigma_c(\mathbf{y}) \cdot \frac{\mathbf{x}_c - \mu_c(\mathbf{x})}{\sigma_c(\mathbf{x})} + \mu_c(\mathbf{y})$$

where $\mu_c(\cdot)$ and $\sigma_c(\cdot)$ denote the per-channel mean and standard deviation (computed over $H, W$). The operation:
1. Normalises the content feature map (removing its style statistics)
2. Denormalises with the style statistics (injecting the target style)

This is exactly InstanceNorm followed by affine transformation with parameters derived from the style image rather than learned.

**Speed advantage:** Unlike Gram matrix matching (Gatys et al., 2015), which requires an iterative optimisation loop at test time, AdaIN is a feed-forward operation. Style transfer is a single forward pass, enabling real-time (~30fps) video style transfer.

**For AI (StyleGAN):** StyleGAN's generator applies AdaIN at each resolution level. The mapping network $f: \mathcal{Z} \to \mathcal{W}$ maps a latent code $\mathbf{z}$ to a disentangled style space $\mathbf{w}$. Learned affine transformations $A$ produce per-layer styles $\mathbf{y} = A(\mathbf{w})$, which are then used as the affine parameters in AdaIN:
$$\operatorname{AdaIN}(\mathbf{x}_i, \mathbf{y}_i) = \mathbf{y}_{s,i} \frac{\mathbf{x}_i - \mu(\mathbf{x}_i)}{\sigma(\mathbf{x}_i)} + \mathbf{y}_{b,i}$$
where $\mathbf{y} = (\mathbf{y}_s, \mathbf{y}_b)$ are the scale and bias from the style affine transform.

### 8.2 Conditional Batch Normalization and FiLM

**Conditional BatchNorm** extends BatchNorm by making the affine parameters $\gamma$ and $\beta$ depend on an external conditioning signal $\mathbf{c}$ (class label, text embedding, etc.):

$$y_i = \gamma_c(\mathbf{c}) \hat{x}_i + \beta_c(\mathbf{c})$$

where $\gamma_c$ and $\beta_c$ are learned functions of $\mathbf{c}$, typically small MLP heads. For class-conditional image generation, $\mathbf{c}$ is a class embedding; for text-to-image generation, $\mathbf{c}$ is a text encoder output.

**FiLM (Feature-wise Linear Modulation; Perez et al., 2018)** generalises this idea: the conditioning information modulates feature representations at any layer via an affine transformation:

$$\operatorname{FiLM}(\mathbf{x}; \mathbf{c}) = \boldsymbol{\gamma}(\mathbf{c}) \odot \mathbf{x} + \boldsymbol{\beta}(\mathbf{c})$$

FiLM is used in visual question answering (VQA) to modulate visual features with language queries, in meta-learning to adapt features to new tasks, and in physics-informed networks to inject physical constraints.

**For AI:** The key insight of conditional normalisation is that the affine parameters after normalisation carry all the "style" or "condition" information. The normalisation removes the network's internal representation of the condition; the learned affine transform reinjects it from the external signal. This creates a clean separation: normalise first (remove internal biases), then modulate with condition.

### 8.3 AdaLN-Zero in Diffusion Transformers

Diffusion Transformers (DiT; Peebles & Xie, 2023) replaced the U-Net backbone of diffusion models (DDPM, Stable Diffusion) with a pure transformer architecture. The key normalisation design is **AdaLN-Zero**: an adaptive LayerNorm conditioned on the diffusion timestep $t$ and class label $c$, with zero-initialisation.

**Architecture:**
1. Encode conditioning: $\mathbf{c} = \operatorname{MLP}(\operatorname{embed}(t) + \operatorname{embed}(c)) \in \mathbb{R}^d$
2. Linear projection to affine parameters: $(\boldsymbol{\gamma}, \boldsymbol{\beta}, \boldsymbol{\alpha}, \boldsymbol{\gamma}', \boldsymbol{\beta}') = W_c \mathbf{c}$ (6 parameter sets per block)
3. Apply AdaLN: $\operatorname{AdaLN}(\mathbf{x}) = \boldsymbol{\gamma} \cdot \operatorname{LN}(\mathbf{x}) + \boldsymbol{\beta}$
4. Gate the sublayer output: $\mathbf{x} \leftarrow \mathbf{x} + \boldsymbol{\alpha} \odot \operatorname{Sublayer}(\operatorname{AdaLN}(\mathbf{x}))$

**The "Zero" trick:** $W_c$ is initialised to zero, so at the start of training, $\boldsymbol{\gamma} = \boldsymbol{\beta} = \boldsymbol{\alpha} = \mathbf{0}$. With $\boldsymbol{\gamma} = \mathbf{0}$, the AdaLN output is zero regardless of $\mathbf{x}$. With $\boldsymbol{\alpha} = \mathbf{0}$, the gated sublayer contributes nothing to the residual stream. The entire transformer block reduces to an identity function at initialisation: $\mathbf{x}_{l+1} = \mathbf{x}_l$.

This zero-initialisation means every transformer block starts as a residual connection, and the model gradually "activates" blocks as training progresses. It stabilises training of very deep transformers (DiT uses 28 or more blocks) and has been adopted in Stable Diffusion 3, FLUX.1, and other production diffusion models.

**For AI:** AdaLN-Zero is now the standard conditioning mechanism for diffusion transformers. It is strictly better than (1) cross-attention conditioning (higher compute), (2) token concatenation (increases sequence length), and (3) simple class embedding addition (no per-layer scale/shift control). The zero-init trick is a general technique applicable whenever you want to add a new learnable block to a pre-trained model without disrupting the initial output.


## 9. Theoretical Analysis

### 9.1 Loss Landscape Smoothing

The landmark theoretical result on BatchNorm is due to Santurkar et al. (2018), "How Does Batch Normalization Help Optimization?" The paper proved that BatchNorm improves the *β-smoothness* of the loss landscape, not just internal covariate shift.

**Definition (β-smoothness):** A function $f$ is β-smooth if its gradient is β-Lipschitz:
$$\lVert \nabla f(\mathbf{x}) - \nabla f(\mathbf{y}) \rVert \leq \beta \lVert \mathbf{x} - \mathbf{y} \rVert \quad \forall \mathbf{x}, \mathbf{y}$$

For gradient descent, β-smoothness gives a convergence guarantee: with step size $\eta \leq 1/\beta$, each step decreases the loss by at least $\frac{1}{2\beta} \lVert \nabla f \rVert^2$.

**Theorem (Santurkar et al., 2018):** For a network with BatchNorm, the loss satisfies:
$$\lVert \nabla_{\hat{\mathbf{x}}} \mathcal{L} \rVert^2 \leq \frac{\gamma^2}{\sigma^2 + \epsilon} \lVert \nabla_\mathbf{y} \mathcal{L} \rVert^2 \left(1 + \frac{\gamma^2 \lVert \nabla_\mathbf{y} \mathcal{L} \rVert^2}{m \sigma^2}\right)^{-1}$$

This bounds the gradient magnitude at the normalised layer. Crucially, the bound depends on $\gamma^2 / \sigma^2$, which is controlled (γ is initialised to 1, and $\sigma^2$ stays non-negligibly positive due to the diversity in mini-batches). For a non-normalised network, the gradient can grow unboundedly with depth.

**Practical implication:** A smaller β means a larger safe learning rate. BatchNorm effectively reduces β by a factor proportional to $\gamma/\sigma$, enabling the 10-30× learning rate increase that is observed empirically. The theoretical maximum step size scales as $\sigma/\gamma$, explaining why smaller γ values (sparser activations) allow larger learning rates.

### 9.2 Implicit Regularisation

BatchNorm has a well-documented regularisation effect that reduces the need for explicit regularisation (dropout, L2 weight decay). Several mechanisms contribute:

**1. Noise injection via batch statistics:** The mini-batch mean $\mu_\mathcal{B}$ and variance $\sigma_\mathcal{B}^2$ are stochastic estimates of the true population statistics $\mu$ and $\sigma^2$. Each mini-batch gives slightly different estimates, introducing noise into the normalised activations:

$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma_\mathcal{B}^2 + \epsilon}} = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} + \text{noise}(\mu_\mathcal{B} - \mu, \sigma_\mathcal{B}^2 - \sigma^2)$$

This noise acts like a per-sample perturbation, similar to dropout but applied to the normalised activations rather than the weights.

**2. Scale invariance → effective weight decay:** As shown in §3.4, BatchNorm makes the loss invariant to weight scaling. This means gradient descent effectively operates on the *direction* of weights rather than their magnitude. In the direction space, the norm is implicitly constrained to 1 (since magnitude doesn't matter), which is equivalent to a form of weight decay in directional parameterisation.

**3. Effective batch size dependence:** The regularisation strength from BN noise decreases as batch size increases (larger batches give more accurate estimates of $\mu$ and $\sigma^2$). This is why models trained with very large batches (128K+ tokens in LLM training) cannot rely on BN for regularisation and need other techniques.

### 9.3 Normalisation-Free Networks

**NFNets (Brock et al., 2021)** demonstrated that large-scale image classification can match BatchNorm performance without any normalisation, using two key techniques:

**1. Scaled Weight Standardisation (WSConv):** Normalises the weights (not activations) such that the initialisation satisfies a *signal propagation* property: for a random input, $\operatorname{Var}(\mathbf{z}^{[l]}) \approx \operatorname{Var}(\mathbf{z}^{[l-1]})$.

Specifically, weights are standardised to have zero mean and unit variance per filter, then scaled by a factor $\gamma / \sqrt{C}$ where $C$ is the input channels:
$$\hat{W}_{i,:} = \frac{W_{i,:} - \bar{W}_{i,:}}{\lVert W_{i,:} - \bar{W}_{i,:} \rVert_2} \cdot \frac{\gamma}{\sqrt{C}}$$

**2. Adaptive Gradient Clipping (AGC):** Clips gradients by the ratio of the weight norm to the gradient norm:
$$\mathbf{g} \leftarrow \begin{cases} \lambda \frac{\lVert \mathbf{w} \rVert}{\lVert \mathbf{g} \rVert} \mathbf{g} & \text{if } \frac{\lVert \mathbf{g} \rVert}{\lVert \mathbf{w} \rVert} > \lambda \\ \mathbf{g} & \text{otherwise} \end{cases}$$

with $\lambda = 0.01$ as default. This prevents any single gradient step from changing the relative size of weights significantly.

**Fixup Initialisation (Zhang et al., 2019):** An earlier approach to normalisation-free ResNets. Scales the residual branch by $L^{-1/(2m-2)}$ where $L$ is depth and $m$ is the number of multiplications in the residual branch. Ensures $\operatorname{Var}(\mathbf{z}^{[L]}) = O(1)$ at initialisation without any normalisation layers.

**For AI:** NFNets and Fixup show that normalisation is not fundamental — its effects (controlled variance, smooth loss landscape) can be achieved through careful initialisation and gradient management. However, BatchNorm and LayerNorm remain the practical choice because they require less hyperparameter tuning and are more robust to architecture changes.


## 10. Numerical Stability

### 10.1 Welford's Online Algorithm

The naive formula for variance, $\sigma^2 = \mathbb{E}[x^2] - \mathbb{E}[x]^2$, suffers from catastrophic cancellation in floating-point arithmetic. If $\mathbb{E}[x^2]$ and $\mathbb{E}[x]^2$ are both large and nearly equal, their difference can have very large relative error.

**Example of instability:** Let $x_1 = 10^8 + 1$, $x_2 = 10^8 - 1$. True variance = 1. Naive computation in FP32:
- $\mathbb{E}[x^2] = ((10^8+1)^2 + (10^8-1)^2)/2 = 10^{16} + 1$
- $\mathbb{E}[x]^2 = (10^8)^2 = 10^{16}$
- Difference: $10^{16} + 1 - 10^{16} = 1$ (correct in exact arithmetic)
- But in FP32 (24-bit mantissa, ~7 decimal digits): $10^{16}$ cannot be represented exactly; the $+1$ is lost, giving variance = 0.

**Welford's algorithm** computes mean and variance in a single pass with numerical stability:

```
Initialize: n = 0, mean = 0, M2 = 0

For each new value x:
  n += 1
  delta = x - mean
  mean += delta / n
  delta2 = x - mean
  M2 += delta * delta2

Variance = M2 / n  (population)
         = M2 / (n-1)  (sample, unbiased)
```

The key insight: instead of accumulating $\sum x_i^2$ and $\sum x_i$ separately (which requires large intermediate values), Welford tracks the difference of each new value from the *current* running mean. Since the values are compared to a nearby running estimate, cancellation is avoided.

**Stability analysis:** The algorithm maintains the exact relation $M2 = \sum_{i=1}^n (x_i - \bar{x}_n)^2$ with forward error bounded by $O(n \cdot \epsilon_{\text{machine}})$, compared to $O(n^2 \cdot \epsilon_{\text{machine}})$ for the naive two-pass algorithm and potentially much worse for the naive single-pass formula.

**In practice (PyTorch/CUDA):** Modern GPU kernels for BatchNorm and LayerNorm use variants of Welford's algorithm to compute mean and variance in fused single-pass kernels. The alternative two-pass approach (first pass: compute mean; second pass: compute variance) is also stable and is sometimes used because it maps better to parallel hardware.

### 10.2 Mixed Precision Considerations

LLM training uses mixed precision (FP16 or BF16 for activations and weights, FP32 for master weights and optimizer states). Normalisation layers have specific precision requirements:

**FP16 and BF16 properties:**
- FP16: 5 exponent bits, 10 mantissa bits. Range: $\approx 6 \times 10^{-5}$ to $65504$. Can underflow for small variances.
- BF16: 8 exponent bits, 7 mantissa bits. Range: same as FP32 ($\approx 10^{-38}$ to $3.4 \times 10^{38}$). No underflow risk, but less precision.

**Problem: FP16 variance underflow.** For a feature vector $\mathbf{x}$ with typical values in $[-1, 1]$, the variance $\sigma^2 \approx 0.1$ is well within FP16 range. But after many layers, if activations are small (say, post-LayerNorm with small $\gamma$), the variance can fall below FP16's representable minimum $\approx 6 \times 10^{-5}$, causing it to be rounded to 0. Then $1/\sqrt{\sigma^2 + \epsilon}$ overflows.

**Standard solution — compute statistics in FP32:**
```python
# PyTorch default for LayerNorm
with torch.cuda.amp.autocast():
    # activations in FP16/BF16
    x = layer(x)  # FP16 forward pass
    
# LayerNorm internally upcasts to FP32 for mean/variance computation
# (enabled by default, controlled by elementwise_affine parameter)
```

**ε tuning:** The default $\epsilon = 10^{-5}$ is chosen for FP32. For FP16 training without FP32 upcast, $\epsilon = 10^{-3}$ is often needed. For BF16 training, $\epsilon = 10^{-6}$ works since BF16 has no underflow risk.

**QK-norm:** In modern transformer variants (e.g., ViT-22B, SD3, FLUX), Query and Key projections in attention are normalised with RMSNorm or LayerNorm before the attention dot product. This prevents attention entropy collapse (where all attention concentrates on a few tokens) in very long sequences or with large learning rates, and specifically addresses FP16 overflow in $Q K^\top / \sqrt{d_k}$ when Q and K have large magnitude.


## 11. Applications in Modern AI

### 11.1 Transformer Norms

The normalisation choice has evolved systematically across transformer architectures:

```
NORMALISATION IN MAJOR TRANSFORMER ARCHITECTURES
════════════════════════════════════════════════════════════════════════

  Architecture    Year  Norm        Placement    Notes
  ─────────────────────────────────────────────────────────────────────
  Original TF     2017  LayerNorm   Post-LN      Vaswani et al.
  BERT            2019  LayerNorm   Post-LN      110M/340M params
  GPT-2           2019  LayerNorm   Pre-LN       First Pre-LN switch
  T5              2020  RMSNorm     Pre-LN       No bias in RMSNorm
  GPT-3           2020  LayerNorm   Pre-LN       175B params
  PaLM            2022  RMSNorm     Pre-LN       540B params
  LLaMA-1         2023  RMSNorm     Pre-LN       7B/13B/33B/65B
  LLaMA-2/3       2023  RMSNorm     Pre-LN       + QK-norm in v3
  Mistral         2023  RMSNorm     Pre-LN       7B/8x7B MoE
  Gemma           2024  RMSNorm     Pre-LN       2B/7B
  Phi-2/3         2024  LayerNorm   Pre-LN       2.7B/7B
  FLUX.1          2024  RMSNorm     Pre-LN       + AdaLN-Zero

════════════════════════════════════════════════════════════════════════
```

**QK-norm (LLaMA-3):** LLaMA-3 adds RMSNorm to Query and Key projections before the attention computation. This addresses attention score variance that grows with sequence length: without QK-norm, $\text{Var}(Q_i \cdot K_j) = d_k \cdot \text{Var}(q) \cdot \text{Var}(k)$, which grows proportionally to the number of attention heads and model width. QK-norm bounds this variance, enabling stable training at 128K context length.

**T5's RMSNorm without bias:** Google's T5 (Raffel et al., 2020) removed the bias $\beta$ from LayerNorm (equivalent to RMSNorm but still with mean-centring). Their ablation showed the bias contributes negligibly to quality. This was an early signal that the full affine transform in LayerNorm is over-parameterised.

### 11.2 ResNet and CNN Norms

**BatchNorm in ResNets:** ResNet-50 applies BatchNorm after every convolutional layer and before every ReLU. Each residual block has the structure:

$$\mathbf{h} = \operatorname{ReLU}(\operatorname{BN}(\operatorname{Conv}_2(\operatorname{ReLU}(\operatorname{BN}(\operatorname{Conv}_1(\mathbf{x}))))))$$

The 1×1 projection in the shortcut path also uses BatchNorm. Without BatchNorm, ResNet-50 achieves ~62% top-1 ImageNet accuracy; with BatchNorm, ~76%.

**GroupNorm in detection/segmentation:** Mask R-CNN uses GroupNorm (G=32) instead of BatchNorm for several reasons:
1. Detection uses batch size 2-4 (large images require more GPU memory), making BN statistics unreliable
2. Detection heads process RoI features of different sizes in the same batch, violating BN's assumption that all elements in the batch come from the same distribution
3. GroupNorm gives consistent behaviour across batch sizes and different feature sizes

EfficientDet, YOLOX, and most modern object detection systems follow the GN convention.

### 11.3 GAN Norms

GANs require normalization in both generator and discriminator, but for different reasons:

**Generator:** Uses InstanceNorm or AdaIN to control style at each resolution level. The key is that the generator needs to produce diverse outputs — using BatchNorm in the generator would couple all generated images through shared batch statistics, potentially causing mode collapse (all images in a batch look similar to minimise BN variance).

**Discriminator:** Uses SpectralNorm to enforce the Lipschitz constraint. A 1-Lipschitz discriminator has bounded gradients, preventing the discriminator from memorising the training data and providing stable training signal to the generator.

**WGAN-GP alternative:** Gradient Penalty (Gulrajani et al., 2017) adds a regularisation term to the loss:
$$\mathcal{L}_{\text{GP}} = \lambda \mathbb{E}_{\hat{\mathbf{x}}}[(\lVert \nabla_{\hat{\mathbf{x}}} D(\hat{\mathbf{x}}) \rVert_2 - 1)^2]$$

where $\hat{\mathbf{x}}$ is sampled uniformly between real and generated data. This directly enforces $\lVert \nabla D \rVert \approx 1$ on the data manifold, which is a stronger (and more local) Lipschitz condition than spectral normalisation.

### 11.4 Diffusion Models

Diffusion models have evolved through several normalisation choices:

**DDPM U-Net (Ho et al., 2020):** GroupNorm (G=32) in all convolutional blocks. Timestep conditioning via Conditional BatchNorm-style FiLM modulation: the timestep is encoded as a 2D vector $(a, b)$ and used to affinely transform features as $a \cdot \mathbf{x} + b$.

**Stable Diffusion (v1-v2):** GroupNorm with G=32 in the U-Net, LayerNorm in the text cross-attention transformer.

**DiT (Peebles & Xie, 2023):** Pure transformer with AdaLN-Zero (see §8.3). The class label and timestep jointly condition all normalisation layers. DiT-XL/2 outperforms all U-Net diffusion models on ImageNet class-conditional generation.

**Stable Diffusion 3 / FLUX.1 (2024):** Multi-modal Diffusion Transformer (MMDiT) architecture. Uses RMSNorm with QK-norm for attention stability. Text tokens and image tokens have separate normalisation parameters to account for their different statistical properties. AdaLN-Zero conditions all blocks on the timestep.

**For AI:** The shift from U-Net + GroupNorm to pure transformer + AdaLN-Zero in diffusion models mirrors the shift from CNN + BatchNorm to transformer + LayerNorm in discriminative models. The trend is clear: transformers with Pre-LN or AdaLN-Zero, normalising over the feature/model dimension, with RMSNorm for efficiency.


## 12. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using BatchNorm with batch size 1 | $\sigma^2 = 0$ exactly; normalised output is 0/ε, garbage | Use LayerNorm, GroupNorm, or InstanceNorm for small batches |
| 2 | Forgetting `model.eval()` before inference | BN uses batch statistics (stochastic) instead of running statistics (deterministic) | Always call `model.eval()` before evaluation and `model.train()` before training |
| 3 | Applying BatchNorm after the activation | Normalising post-ReLU throws away the negative half of the distribution info | Apply BN between linear/conv and activation: $\operatorname{ReLU}(\operatorname{BN}(\mathbf{z}))$ |
| 4 | Confusing RMSNorm with LayerNorm with $\beta=0$ | RMSNorm uses RMS = $\sqrt{E[x^2]}$; LayerNorm (β=0) uses std = $\sqrt{E[(x-\mu)^2]}$; differ when mean ≠ 0 | Check whether mean subtraction is performed |
| 5 | Using the same ε for FP16 and FP32 | FP16 can represent values as small as $6 \times 10^{-5}$; default ε=$10^{-5}$ may underflow | Use ε=$10^{-3}$ for FP16 or upcast to FP32 for statistics computation |
| 6 | Post-Norm placement in deep transformers (>12 layers) | Gradient flow passes through LN Jacobian $L$ times, causing gradient vanishing/explosion at initialisation | Switch to Pre-Norm or DeepNorm; add learning rate warmup for Post-Norm |
| 7 | Applying LayerNorm over the batch dimension in sequence models | LN over $N$ creates batch dependency; breaks variable-length inference | Apply LN over the feature dimension ($d_{\text{model}}$) for each position independently |
| 8 | Freezing BatchNorm running statistics during fine-tuning | If pre-training and fine-tuning have different input distributions, frozen BN stats cause mismatch | Set `requires_grad=True` for BN affine params and use `track_running_stats=True` during fine-tuning |
| 9 | Not including γ in AdaLN/AdaIN but using γ=1 constant | Loses the model's ability to gate off entire sublayers (the α gate in AdaLN-Zero enables this) | Include all affine parameters; initialise to zero for AdaLN-Zero stability |
| 10 | Computing variance with the naive formula $E[x^2] - E[x]^2$ in FP16 | Catastrophic cancellation when $\|x\|$ is large; variance can be negative | Use Welford's algorithm or two-pass computation; upcast to FP32 |


## 13. Exercises

**Exercise 1 ★ — BatchNorm Forward and Backward**
Implement BatchNorm from scratch for a mini-batch of shape $(4, 3)$. (a) Compute $\mu_B$ and $\sigma_B^2$ per feature. (b) Compute normalised output $\hat{X}$ and scaled output $Y = \gamma \hat{X} + \beta$ with $\gamma = [1, 2, 1]$, $\beta = [0, 0.5, -1]$. (c) Derive and implement the full gradient $\partial \mathcal{L}/\partial X$ using the formula from §3.2. (d) Verify your gradient against central differences.

**Exercise 2 ★ — Normalisation Axis Comparison**
Create a random tensor of shape $(8, 32, 16, 16)$. (a) Apply BatchNorm, LayerNorm, InstanceNorm, and GroupNorm (G=4) and print the shape of the statistics (mean, variance) for each. (b) Verify that each method produces the expected mean/variance of outputs. (c) Show that GroupNorm with G=1 and G=32 match LayerNorm and InstanceNorm respectively (on a 2D input).

**Exercise 3 ★ — RMSNorm vs LayerNorm**
(a) Implement both RMSNorm and LayerNorm from scratch. (b) For input drawn from $\mathcal{N}(5, 4)$ (non-zero mean), compute and compare their outputs. (c) For input drawn from $\mathcal{N}(0, 4)$ (zero mean), show they produce equal outputs. (d) Derive algebraically that when $\mathbb{E}[\mathbf{x}] = 0$, $\operatorname{RMS}(\mathbf{x}) = \sqrt{\operatorname{Var}(\mathbf{x})}$.

**Exercise 4 ★★ — BatchNorm Backward Derivation**
Derive the full BatchNorm backward pass formula from first principles. (a) Compute $\partial \mathcal{L}/\partial \gamma$ and $\partial \mathcal{L}/\partial \beta$. (b) Compute $\partial \mathcal{L}/\partial \hat{x}_i$. (c) Using the chain rule through variance and mean, derive $\partial \mathcal{L}/\partial x_i$. (d) Show the three-term structure and explain what each term does geometrically.

**Exercise 5 ★★ — Pre-Norm vs Post-Norm Gradient Flow**
Simulate a 12-layer transformer with both Pre-LN and Post-LN placements. (a) At initialisation, compute the gradient norm at each layer for a random input (no training). (b) Show that Post-LN has gradient norms that vary by orders of magnitude across layers, while Pre-LN has approximately constant gradient norms. (c) Measure the effective Lipschitz constant of each architecture as depth increases.

**Exercise 6 ★★ — Spectral Normalization**
(a) Implement spectral normalisation using 1 step of power iteration. (b) Apply it to a random $64 \times 64$ weight matrix and verify $\sigma_{\max}(\bar{W}) = 1$. (c) Compare the power-iteration estimate to the true spectral norm (via `np.linalg.svd`). (d) How many power-iteration steps are needed to reach 1% accuracy? (e) Explain why the Lipschitz constant of a 3-layer ReLU network with spectrally-normalised weights is at most 1.

**Exercise 7 ★★ — Welford's Algorithm**
(a) Implement Welford's online algorithm for computing mean and variance in a single pass. (b) Compare its numerical accuracy to the naive formula $E[x^2] - E[x]^2$ on a dataset where all values are close to $10^6$ (e.g., $\{10^6 + \epsilon_i\}$ with $\epsilon_i \sim \mathcal{N}(0, 1)$). (c) Demonstrate catastrophic cancellation in FP32 using the naive formula. (d) Show Welford's gives correct variance even in this pathological case.

**Exercise 8 ★★★ — AdaIN Style Transfer**
(a) Implement AdaIN: given content features $\mathbf{x}$ and style features $\mathbf{y}$ (both of shape $(C, H, W)$), compute $\operatorname{AdaIN}(\mathbf{x}, \mathbf{y})$. (b) Generate two random "content" and "style" feature maps (C=16, H=W=8). (c) Verify that the output has the same mean and variance as the style features (per channel). (d) Show that AdaIN is not the same as simply replacing content with style: the spatial structure of content is preserved.

**Exercise 9 ★★★ — Loss Landscape Smoothing**
(a) Train a simple 5-layer MLP on a synthetic classification task, both with and without BatchNorm. (b) At several points during training, compute the gradient norm $\lVert \nabla_\theta \mathcal{L} \rVert$ and the Hessian's largest eigenvalue (using power iteration). (c) Show empirically that BatchNorm results in smaller gradient norm variance across training steps. (d) Verify that larger learning rates cause divergence without BN but not with BN.

**Exercise 10 ★★★ — LLM Norm Ablation**
Build a miniature transformer language model (2 layers, $d_{\text{model}} = 128$, character-level). (a) Train with Post-LN LayerNorm, Pre-LN LayerNorm, and Pre-LN RMSNorm on a small text corpus. (b) Compare training curves (loss vs. steps). (c) Show that Pre-LN variants train without learning rate warmup while Post-LN requires warmup. (d) Measure the FLOPs per forward pass for each normalisation choice and verify the RMSNorm speedup.


## 14. Why This Matters for AI (2026 Perspective)

| Aspect | Impact |
|---|---|
| **LLM Training Speed** | RMSNorm reduces normalisation FLOPs by ~40% vs LayerNorm; at 100B+ parameter scale this saves significant compute over training runs |
| **Long-Context Transformers** | QK-norm (in LLaMA-3, SD3, FLUX) prevents attention entropy collapse at 128K+ context by bounding $Q K^\top$ magnitude |
| **Training Stability at Scale** | Pre-LN placement eliminates gradient explosion in 100+ layer transformers; enables training without warmup; critical for GPT-4/LLaMA-3 scale |
| **Diffusion Model Quality** | AdaLN-Zero's zero-init trick enables stable training of 28+ block DiTs; SD3 and FLUX use this for state-of-the-art generation |
| **GAN Stability** | SpectralNorm + WGAN-GP are the two standard approaches for stable adversarial training; all production GAN systems (BigGAN, StyleGAN) use them |
| **Small-Batch Fine-tuning** | GroupNorm enables fine-tuning detection/segmentation models on consumer GPUs with batch size 1-2; critical for accessible training |
| **Mechanistic Interpretability** | LayerNorm's scale invariance creates a "privileged basis" for the residual stream; understanding LN's projection is central to interpreting transformer circuits |
| **LoRA and PEFT** | Normalisation layers are typically *not* adapted in LoRA fine-tuning (parameters frozen); understanding which norm parameters to adapt is key for specialised tasks |
| **Mixed Precision** | Knowing when to upcast to FP32 for normalisation statistics is critical for preventing NaN/inf in large model training; BF16 vs FP16 choice interacts with ε |
| **Model Merging** | Batch statistics in BN create compatibility issues when merging models trained on different data; LN/RMSNorm models merge more cleanly |

## 15. Conceptual Bridge

### Looking Backward

This section builds on the foundations developed in **Activation Functions** (§02). The vanishing gradient analysis showed that sigmoid and tanh saturate when activations have large magnitude — normalization directly addresses this by controlling activation scale throughout training. The Glorot and He initialization rules derived in that section assume activations have unit variance at initialization; normalization ensures this assumption holds *throughout training*, not just at the start.

The **loss functions** section (§01) established that backpropagation propagates gradients through every layer of the network. The BatchNorm backward pass formula derived in §3.2 shows how normalization layers interact with gradient flow: they introduce a coupling between samples (through shared mean/variance) and add a "gradient centering" effect that removes the mean and variance components of the upstream gradient.

### Looking Forward

The normalisation techniques developed here are architectural components used throughout the curriculum's remaining sections:

**Sampling Methods (§04):** Diffusion models are the primary modern use case for AdaLN-Zero. The sampling dynamics of diffusion (score matching, denoising schedules) require stable activations throughout the reverse process — AdaLN-Zero's zero initialisation guarantees this.

**Neural Networks (Chapter 14, §02):** The complete transformer architecture builds directly on Pre-LN LayerNorm/RMSNorm placement. Understanding why normalization goes before (not after) the attention and FFN sublayers is prerequisite knowledge for the attention mechanism section.

**Optimization (Chapter 08):** The loss landscape smoothing theorem (§9.1) is a key application of smooth optimization theory. The maximum step size for gradient descent scales as $1/\beta$ where $\beta$ is the smoothness constant — BatchNorm directly improves this constant.

### Position in Curriculum

```
NORMALISATION IN THE CURRICULUM
════════════════════════════════════════════════════════════════════════

  ┌──────────────────────────────────────────────────────────────────┐
  │  CHAPTER 13: ML-SPECIFIC MATH                                    │
  │                                                                  │
  │  §01 Loss Functions ──────────────────────────────────────────┐  │
  │       CE, focal, KL, calibration                              │  │
  │                      │                                        │  │
  │  §02 Activation Fns ─┤──────────────────────────────────────┐ │  │
  │       sigmoid, GELU, SwiGLU, vanishing grad                  │ │  │
  │                      │                                       │ │  │
  │  §03 Normalisation ◄─┘ ← YOU ARE HERE                       │ │  │
  │       BN/LN/RMSNorm, Pre-Norm, AdaLN-Zero                    │ │  │
  │           │                                                  │ │  │
  │           ▼                                                  │ │  │
  │  §04 Sampling Methods                                        │ │  │
  │       Diffusion, MCMC, ancestral sampling                    │ │  │
  └──────────────────────────────────────────────────────────────┘ │  │
                                                                    │  │
  ┌──────────────────────────────────────────────────────────────┐  │  │
  │  CHAPTER 14: MATH FOR SPECIFIC MODELS                        │  │  │
  │                                                              │  │  │
  │  §02 Neural Networks ◄───────────────────────────────────────┘──┘  │
  │       Full transformer: Attn + FFN + Pre-LN + RMSNorm             │
  │           │                                                        │
  │           ▼                                                        │
  │  §03+ Generative Models                                            │
  │       VAE, Flow, GAN (SpectralNorm), Diffusion (AdaLN-Zero)        │
  └────────────────────────────────────────────────────────────────────┘

  Key dependency arrows:
  • Normalization → Neural Networks: Pre-LN placement, RMSNorm choice
  • Normalization → GANs: SpectralNorm for discriminator stability
  • Normalization → Diffusion: AdaLN-Zero conditioning
  • Activation Fns → Normalization: why we normalize (vanishing grads)
  • Loss Functions → Normalization: calibration, temperature scaling

════════════════════════════════════════════════════════════════════════
```

The normalisation choices made when designing an architecture have cascade effects on training stability, inference speed, and model quality. The mathematical tools developed in this section — Jacobians of normalisation maps, the β-smoothness theorem, power iteration for spectral norms, and Welford's algorithm — are applied routinely in modern deep learning practice. Every time you encounter a training instability in a deep network, the diagnosis and fix almost always involves understanding the normalisation layers.


---

## Appendix A: Complete BatchNorm Backward Pass Derivation

This appendix derives the full BN backward pass step by step, showing every chain rule application explicitly. This is a common interview question and a useful exercise in Jacobian computation.

### Setup

Given a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$ and scalar parameters $\gamma, \beta$:

| Forward step | Expression | Shape |
|---|---|---|
| Batch mean | $\mu = \frac{1}{m}\sum_{i=1}^m x_i$ | scalar |
| Batch variance | $\sigma^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu)^2$ | scalar |
| Standard deviation | $\sigma = \sqrt{\sigma^2 + \epsilon}$ | scalar |
| Normalised | $\hat{x}_i = (x_i - \mu)/\sigma$ | $m$ |
| Output | $y_i = \gamma \hat{x}_i + \beta$ | $m$ |

Let $\ell$ denote the loss. We are given $\frac{\partial \ell}{\partial y_i}$ for all $i$ and want $\frac{\partial \ell}{\partial x_i}$.

### Step 1: Gradient w.r.t. $\gamma$ and $\beta$

$$\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^m \frac{\partial \ell}{\partial y_i} \cdot \frac{\partial y_i}{\partial \gamma} = \sum_{i=1}^m \frac{\partial \ell}{\partial y_i} \hat{x}_i$$

$$\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^m \frac{\partial \ell}{\partial y_i} \cdot 1 = \sum_{i=1}^m \frac{\partial \ell}{\partial y_i}$$

### Step 2: Gradient w.r.t. $\hat{x}_i$

Since $y_i = \gamma \hat{x}_i + \beta$:
$$\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma$$

### Step 3: Gradient w.r.t. $\sigma$ (the standard deviation, not variance)

$\hat{x}_i = (x_i - \mu)/\sigma$, so:
$$\frac{\partial \ell}{\partial \sigma} = \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \sigma} = \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-(x_i - \mu)}{\sigma^2}$$

### Step 4: Gradient w.r.t. $\sigma^2$

Since $\sigma = (\sigma^2 + \epsilon)^{1/2}$:
$$\frac{\partial \ell}{\partial \sigma^2} = \frac{\partial \ell}{\partial \sigma} \cdot \frac{\partial \sigma}{\partial \sigma^2} = \frac{\partial \ell}{\partial \sigma} \cdot \frac{1}{2\sigma}$$

$$= \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-(x_i-\mu)}{\sigma^2} \cdot \frac{1}{2\sigma} = \frac{-1}{2\sigma^3} \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i} (x_i - \mu)$$

### Step 5: Gradient w.r.t. $\mu$

$\mu$ appears in three places: $\hat{x}_i = (x_i - \mu)/\sigma$ for all $i$, and in $\sigma^2 = \frac{1}{m}\sum_i (x_i-\mu)^2$.

From $\hat{x}_i$ terms:
$$\left.\frac{\partial \ell}{\partial \mu}\right|_{\hat{x}} = \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sigma}$$

From $\sigma^2$ term:
$$\left.\frac{\partial \ell}{\partial \mu}\right|_{\sigma^2} = \frac{\partial \ell}{\partial \sigma^2} \cdot \frac{-2}{m} \sum_{i=1}^m (x_i - \mu)$$

Note that $\sum_i (x_i - \mu) = 0$ by definition of $\mu$, so the second term vanishes! Thus:
$$\frac{\partial \ell}{\partial \mu} = \frac{-1}{\sigma} \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i}$$

### Step 6: Gradient w.r.t. $x_i$

$x_i$ appears in three places: $\hat{x}_i$ directly, $\mu = \frac{1}{m}\sum_j x_j$, and $\sigma^2 = \frac{1}{m}\sum_j(x_j-\mu)^2$.

$$\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sigma} + \frac{\partial \ell}{\partial \mu} \cdot \frac{1}{m} + \frac{\partial \ell}{\partial \sigma^2} \cdot \frac{2(x_i - \mu)}{m}$$

Substituting:

$$= \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sigma} + \frac{-1}{\sigma m}\sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j} + \frac{-1}{2\sigma^3}\sum_{j=1}^m \frac{\partial \ell}{\partial \hat{x}_j}(x_j-\mu) \cdot \frac{2(x_i-\mu)}{m}$$

Since $\hat{x}_i = (x_i - \mu)/\sigma$, the third term becomes:

$$= \frac{1}{\sigma}\left[\frac{\partial \ell}{\partial \hat{x}_i} - \frac{1}{m}\sum_j \frac{\partial \ell}{\partial \hat{x}_j} - \frac{\hat{x}_i}{m}\sum_j \frac{\partial \ell}{\partial \hat{x}_j}\hat{x}_j\right]$$

Substituting $\partial \ell / \partial \hat{x}_i = \gamma \cdot \partial \ell / \partial y_i$:

$$\boxed{\frac{\partial \ell}{\partial x_i} = \frac{\gamma}{\sigma}\left[\frac{\partial \ell}{\partial y_i} - \frac{1}{m}\sum_{j=1}^m \frac{\partial \ell}{\partial y_j} - \hat{x}_i \cdot \frac{1}{m}\sum_{j=1}^m \frac{\partial \ell}{\partial y_j}\hat{x}_j\right]}$$


---

## Appendix B: Normalisation Methods at a Glance

Complete reference for all methods covered in this section.

```
NORMALISATION REFERENCE TABLE
════════════════════════════════════════════════════════════════════════

  Method    Year  Paper                 Key Formula
  ────────────────────────────────────────────────────────────────────
  BatchNorm 2015  Ioffe & Szegedy       ŷ = γ(x-μ_B)/σ_B + β
                                         μ_B,σ_B over (N,H,W) per C
  LayerNorm 2016  Ba, Kiros, Hinton      ŷ = γ(x-μ_n)/σ_n + β
                                         μ_n,σ_n over (C,H,W) per N
  InstNorm  2016  Ulyanov et al.         ŷ = γ(x-μ_{n,c})/σ_{n,c} + β
                                         μ,σ over (H,W) per (N,C)
  GrpNorm   2018  Wu & He               ŷ = γ(x-μ_{n,g})/σ_{n,g} + β
                                         μ,σ over (C/G,H,W) per (N,G)
  WtNorm    2016  Salimans & Kingma      w = g·v/‖v‖
                                         Reparams weight vector
  SpectralN 2018  Miyato et al.          W̄ = W/σ₁(W)
                                         Power iteration for σ₁
  RMSNorm   2019  Zhang & Sennrich       ŷ = γ·x/RMS(x)
                                         RMS = √(mean(x²)), no mean
  AdaIN     2017  Huang & Belongie       ŷ = σ(y)·(x-μ(x))/σ(x) + μ(y)
                                         Content x, style y
  CondBN    2017  de Vries et al.        ŷ = γ(c)·x̂ + β(c)
  FiLM      2018  Perez et al.           ŷ = γ(c)⊙x + β(c)
  AdaLN-0   2023  Peebles & Xie          ŷ = γ·LN(x)+β, α-gated residual
                                         All init to 0

════════════════════════════════════════════════════════════════════════

  PARAMETER COUNTS for a layer with d_model = 4096:

  Method          Trainable params    Notes
  ────────────────────────────────────────────────────────────────────
  No norm         0                   NFNets
  LayerNorm       2 × 4096 = 8192     γ ∈ R^d, β ∈ R^d
  RMSNorm         1 × 4096 = 4096     γ ∈ R^d only
  BatchNorm       2 × C = 8192        C = num channels
  GroupNorm(G=32) 2 × C = 8192        Same as BN, diff statistics
  SpectralNorm    + 2 × min(m,n)      Singular vectors (cached)
  AdaLN-Zero      6 × 4096 = 24576    α,γ,β for attn + ffn

════════════════════════════════════════════════════════════════════════
```

---

## Appendix C: Practical Decision Guide

When choosing a normalisation method, answer these questions in order:

**Q1: Is your model a transformer/sequence model?**
- YES → Use **Pre-LN RMSNorm** (LLMs, ViTs) or **Pre-LN LayerNorm** (BERT-like, older models)
- NO → Continue to Q2

**Q2: Is your model a convolutional network?**
- YES → Continue to Q3
- NO (e.g., MLP, graph network) → Use **LayerNorm** (most flexible)

**Q3: Do you have large batches (≥ 8 per GPU)?**
- YES → Use **BatchNorm** (standard for CNNs, best training dynamics)
- NO → Use **GroupNorm** with G=32 (stable for small batches)

**Q4: Is your task style transfer / image generation?**
- YES → Use **InstanceNorm** (unconditional) or **AdaIN** (conditional)
- NO → Use the answer from Q3

**Q5: Is your model a GAN discriminator?**
- YES → Add **SpectralNorm** to all linear/conv layers (regardless of activation norm)
- NO → No spectral norm needed

**Q6: Are you training a diffusion model with class/text conditioning?**
- YES → Use **AdaLN-Zero** for transformer layers, **GroupNorm** for U-Net conv blocks
- NO → Standard choice from above

---

## Appendix D: Signal Propagation Perspective

A useful theoretical framework for understanding normalisation is **mean field theory** applied to deep networks. The key quantity is the *mean squared activation* at each layer: $q^l = \mathbb{E}[z_i^{[l]2}]$. For a network without normalisation, initialised with weights $W \sim \mathcal{N}(0, \sigma_w^2/n)$:

$$q^l = \sigma_w^2 \cdot \mathbb{E}[\sigma(z_i^{[l-1]})^2]$$

where $\sigma$ is the activation function. This recursion has a fixed point $q^*$ where $q^l = q^{l-1}$. At this fixed point, the network is at the **edge of chaos**: information propagates indefinitely without explosion or collapse.

For tanh, the fixed point is $q^* = 0$ (trivial) or depends on $\sigma_w^2$. For ReLU with $\sigma_w^2 = 2/n$ (He initialisation), $q^* = q^{[0]}$ (variance is preserved). The He initialisation condition is precisely the condition that keeps the signal propagation at the fixed point.

**How normalisation changes this:** LayerNorm/RMSNorm explicitly enforce $q^l = 1$ at every layer by normalising each activation vector to unit RMS. This is like forcing the signal propagation recursion to always be at the fixed point, regardless of the weights. The normalisation thus eliminates the need for careful initialisation schemes to achieve stable propagation — at the cost of adding the normalisation computation itself.

**For AI:** Understanding signal propagation theory explains why RMSNorm + LeCun initialisation (or standard He initialisation) works well for very deep transformers. The normalisation guarantees signal propagation stability; the initialisation only needs to ensure the affine transform does not immediately collapse or explode the first-layer output.

---

## Appendix E: Training Instability Diagnosis

When a model shows training instability (exploding/vanishing loss, NaN values), use this checklist:

| Symptom | Likely Cause | Fix |
|---|---|---|
| NaN in loss at step 1 | Variance underflow in FP16; ε too small | Set ε=1e-3, upcast stats to FP32 |
| Loss explodes after warmup ends | Post-LN with large LR; gradient amplification | Switch to Pre-LN; reduce LR |
| Loss stays flat for first 1000 steps | Post-LN requires warmup | Add cosine warmup over 2000 steps |
| Model evaluates differently in train/eval | BN in eval mode not updated | Set `track_running_stats=False` or freeze BN |
| GAN discriminator loss → 0 instantly | No Lipschitz constraint | Add SpectralNorm to all discriminator layers |
| Style transfer output is uniform colour | InstanceNorm applied to wrong axis | Verify IN normalises over H,W, not C |
| Large batch → worse generalisation | BN noise reduced (larger batch = better stats) | Add dropout; reduce batch size; use SAM optimiser |
| Activation norms grow across layers | No normalisation or wrong placement | Add Pre-LN before each sublayer |
| Different sequence lengths → inconsistent outputs | BN over sequence length dimension | Switch to LN over feature dimension |
| Fine-tuned model degrades on new domain | BN running stats frozen from pretraining | Unfreeze BN stats; run BN calibration pass |

---

## Appendix F: Mathematical Connections

Normalisation connects to several deep mathematical ideas:

**1. Normalisation as Riemannian geometry:** The normalised manifold $\{\hat{\mathbf{x}} : \bar{x} = 0, \operatorname{Var}(\hat{\mathbf{x}}) = 1\}$ is a Stiefel manifold in disguise. Gradient descent on the normalised sphere has different curvature than in Euclidean space, which is why normalisation changes the effective learning rate.

**2. Connection to whitening:** True whitening applies the transformation $\Sigma^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$ where $\Sigma$ is the full covariance. Diagonal normalisation (BN/LN) only normalises the diagonal, ignoring cross-feature correlations. Batch whitening (ZCA) is sometimes used but rarely in deep learning due to the $O(d^3)$ cost of the covariance inverse.

**3. Normalisation and information geometry:** The Fisher information matrix of a normalised layer has a specific structure that makes natural gradient (Adam with well-tuned ε) particularly effective. This is related to why Adam + LayerNorm is the standard combination for LLM training.

**4. Spectral norm and operator theory:** The spectral norm $\lVert W \rVert_2 = \sigma_{\max}(W)$ is the operator norm of $W$ as a linear map from $(\mathbb{R}^n, \lVert \cdot \rVert_2)$ to $(\mathbb{R}^m, \lVert \cdot \rVert_2)$. Spectral normalisation in neural networks is the discrete analogue of enforcing Lipschitz continuity in functional analysis — a requirement for well-posed inverse problems and stable mappings.


---

## Appendix G: Worked Examples

### G.1 BatchNorm Forward Pass — Numerical Example

Let mini-batch with $m = 4$ samples, scalar feature (one channel), $\epsilon = 10^{-5}$, $\gamma = 2$, $\beta = 0.5$:

$$\mathbf{x} = [1.0, 3.0, 5.0, 7.0]$$

**Step 1 — Mean:**
$$\mu_B = (1 + 3 + 5 + 7)/4 = 4.0$$

**Step 2 — Variance:**
$$\sigma_B^2 = \frac{1}{4}\left[(1-4)^2 + (3-4)^2 + (5-4)^2 + (7-4)^2\right] = \frac{9+1+1+9}{4} = 5.0$$

**Step 3 — Normalise:**
$$\hat{x}_i = \frac{x_i - 4.0}{\sqrt{5.0 + 10^{-5}}} \approx [-1.342, -0.447, 0.447, 1.342]$$

**Step 4 — Scale and shift:**
$$y_i = 2 \hat{x}_i + 0.5 \approx [-2.185, -0.394, 1.394, 3.185]$$

Verification: $\bar{y} = (-2.185 - 0.394 + 1.394 + 3.185)/4 = 0.5 = \beta$ ✓ (mean equals $\beta$ when $\gamma = 2$ and $\hat{x}$ is zero-mean... actually $\bar{y} = \gamma \cdot 0 + \beta = \beta$ ✓)

### G.2 LayerNorm Forward Pass — Numerical Example

Single sample with $H = 4$ features, $\epsilon = 10^{-5}$, $\boldsymbol{\gamma} = [1, 1, 1, 1]$, $\boldsymbol{\beta} = [0, 0, 0, 0]$:

$$\mathbf{x} = [2.0, 4.0, 6.0, 8.0]$$

**Mean:** $\mu = (2+4+6+8)/4 = 5.0$

**Variance:** $\sigma^2 = \frac{1}{4}[(2-5)^2+(4-5)^2+(6-5)^2+(8-5)^2] = \frac{9+1+1+9}{4} = 5.0$

**Normalised:** $\hat{\mathbf{x}} \approx [-1.342, -0.447, 0.447, 1.342]$

Note: same normalised values as the BN example! This is because both examples have the same data: the BN example has 4 samples of a scalar feature (processed together), the LN example has 1 sample with 4 features. The normalisation operation is identical — only the interpretation (across samples vs. across features) differs.

### G.3 RMSNorm Forward Pass — Numerical Example

Same input $\mathbf{x} = [2.0, 4.0, 6.0, 8.0]$:

**RMS:** $\operatorname{RMS}(\mathbf{x}) = \sqrt{(4+16+36+64)/4} = \sqrt{30} \approx 5.477$

**Normalised:** $\hat{\mathbf{x}} = \mathbf{x}/5.477 \approx [0.365, 0.730, 1.095, 1.461]$

Compare to LayerNorm output $[-1.342, -0.447, 0.447, 1.342]$: the RMSNorm output is positive throughout (since all inputs are positive), while LayerNorm output is zero-centred. The difference is precisely the mean-centring step.

### G.4 Spectral Norm Power Iteration — Numerical Example

Let $W = \begin{pmatrix} 3 & 1 \\ 1 & 2 \end{pmatrix}$. True SVD: $\sigma_1 = (5 + \sqrt{5})/2 \approx 3.618$.

**Power iteration from $\tilde{\mathbf{u}}_0 = [1, 0]^\top$:**

Step 1: $\hat{\mathbf{v}}_1 = W^\top \tilde{\mathbf{u}}_0 / \lVert W^\top \tilde{\mathbf{u}}_0 \rVert = [3, 1]^\top / \sqrt{10} \approx [0.949, 0.316]^\top$

$\hat{\mathbf{u}}_1 = W \hat{\mathbf{v}}_1 / \lVert W \hat{\mathbf{v}}_1 \rVert$: $W \hat{\mathbf{v}}_1 \approx [3.164, 1.581]^\top$, $\lVert \cdot \rVert \approx 3.543$, so $\hat{\mathbf{u}}_1 \approx [0.894, 0.447]^\top$

$\tilde{\sigma}_1 = \hat{\mathbf{u}}_1^\top W \hat{\mathbf{v}}_1 \approx 0.894 \times 3.164 + 0.447 \times 1.581 \approx 3.538$

Step 2: $\hat{\mathbf{v}}_2 = W^\top \hat{\mathbf{u}}_1 / \lVert \cdot \rVert$: $W^\top \hat{\mathbf{u}}_1 \approx [3.129, 1.788]^\top$, norm $\approx 3.592$, $\hat{\mathbf{v}}_2 \approx [0.871, 0.498]^\top$

$\hat{\mathbf{u}}_2 = W \hat{\mathbf{v}}_2 / \lVert \cdot \rVert$: $W \hat{\mathbf{v}}_2 \approx [3.111, 1.867]^\top$, norm $\approx 3.641$, $\hat{\mathbf{u}}_2 \approx [0.855, 0.513]^\top$

$\tilde{\sigma}_2 \approx 3.610$ (error from true 3.618: 0.2%)

After 3 steps the estimate converges to within 0.01% of the true value. This demonstrates why a single power-iteration step per gradient update is sufficient (the estimate improves incrementally as $W$ changes slowly during training).


---

## Appendix H: Self-Assessment Checklist

Use this checklist to verify your understanding of normalisation techniques.

### Core Mechanics
- [ ] Can you write the BatchNorm forward pass formula from memory, including ε?
- [ ] Can you explain the difference between training-mode and eval-mode BatchNorm?
- [ ] Can you identify which axes are normalised for BN, LN, IN, GN, and RMSNorm?
- [ ] Can you state the three-term structure of the BN backward pass?

### Theory
- [ ] Can you explain why BatchNorm allows larger learning rates (loss landscape smoothing)?
- [ ] Can you prove that RMSNorm = LayerNorm when the input has zero mean?
- [ ] Can you explain why Pre-LN has more stable gradients than Post-LN at initialisation?
- [ ] Can you describe the Welford algorithm and explain why the naive variance formula can fail?

### Applications
- [ ] Given a new architecture, can you choose the appropriate normalisation method?
- [ ] Do you know which normalisation method is used in LLaMA, ResNet-50, StyleGAN, and DiT?
- [ ] Can you implement spectral normalisation using power iteration?
- [ ] Can you implement AdaIN and explain how it separates content from style?

### Advanced
- [ ] Can you derive the full BatchNorm backward pass from scratch?
- [ ] Can you explain the implicit regularisation effect of BatchNorm and its dependence on batch size?
- [ ] Can you describe AdaLN-Zero's zero-initialisation trick and why it stabilises training?
- [ ] Can you explain how signal propagation theory motivates normalisation-free networks?

---

## Appendix I: Key Papers

| Paper | Year | Key Contribution |
|---|---|---|
| Ioffe & Szegedy | 2015 | Batch Normalisation; internal covariate shift; trains 14× faster |
| Ba, Kiros, Hinton | 2016 | Layer Normalisation; no batch dependency; works for RNNs |
| Ulyanov et al. | 2016 | Instance Normalisation; real-time style transfer |
| Salimans & Kingma | 2016 | Weight Normalisation; magnitude/direction reparameterisation |
| Wu & He | 2018 | Group Normalisation; works at batch size 1; detection standard |
| Miyato et al. | 2018 | Spectral Normalisation; SNGAN; 1-Lipschitz discriminators |
| Huang & Belongie | 2017 | AdaIN; arbitrary style transfer via feature statistics |
| Santurkar et al. | 2018 | How does BN help? β-smoothness; not ICS reduction |
| Zhang & Sennrich | 2019 | RMSNorm; drops mean-centring; 10-20% speedup |
| Brock et al. | 2021 | NFNets; normalisation-free via AGC + weight standardisation |
| Wang et al. | 2022 | DeepNorm; Pre-LN + α scaling; 1000-layer stable training |
| Touvron et al. | 2023 | LLaMA; RMSNorm becomes open LLM standard |
| Peebles & Xie | 2023 | DiT; AdaLN-Zero for diffusion transformers |


---

## Appendix J: Implementation Reference

### J.1 BatchNorm from Scratch (NumPy)

```python
import numpy as np

class BatchNorm:
    def __init__(self, C, epsilon=1e-5, momentum=0.1):
        self.gamma = np.ones(C)   # scale
        self.beta  = np.zeros(C)  # shift
        self.eps   = epsilon
        self.momentum = momentum
        # Running statistics (for inference)
        self.running_mean = np.zeros(C)
        self.running_var  = np.ones(C)
        self.training = True
        # Cache for backward pass
        self._cache = {}

    def forward(self, x):
        """x: shape (N, C) or (N, C, H, W)"""
        N = x.shape[0]
        # Reshape for general case: treat all non-channel dims together
        # Assume shape (N, C) for clarity
        if self.training:
            mu  = x.mean(axis=0)              # (C,)
            var = x.var(axis=0)               # (C,)
            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean \
                                + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  \
                                + self.momentum * var
        else:
            mu, var = self.running_mean, self.running_var
        
        x_hat = (x - mu) / np.sqrt(var + self.eps)  # (N, C)
        out   = self.gamma * x_hat + self.beta        # (N, C)
        
        # Cache for backward
        self._cache = {'x_hat': x_hat, 'mu': mu, 'var': var, 'x': x}
        return out

    def backward(self, dout):
        """dout: (N, C), returns dx: (N, C)"""
        x_hat = self._cache['x_hat']
        var   = self._cache['var']
        N     = dout.shape[0]
        std_inv = 1.0 / np.sqrt(var + self.eps)
        
        # Gradients w.r.t. gamma and beta
        dgamma = (dout * x_hat).sum(axis=0)   # (C,)
        dbeta  = dout.sum(axis=0)             # (C,)
        
        # Gradient w.r.t. x (three-term formula)
        dx_hat = dout * self.gamma
        dx = std_inv * (dx_hat
                        - dx_hat.mean(axis=0)
                        - x_hat * (dx_hat * x_hat).mean(axis=0))
        return dx, dgamma, dbeta
```

### J.2 LayerNorm from Scratch (NumPy)

```python
class LayerNorm:
    def __init__(self, d_model, epsilon=1e-5):
        self.gamma = np.ones(d_model)
        self.beta  = np.zeros(d_model)
        self.eps   = epsilon
        self._cache = {}

    def forward(self, x):
        """x: shape (N, d_model) or (N, T, d_model)"""
        mu  = x.mean(axis=-1, keepdims=True)   # (..., 1)
        var = x.var(axis=-1, keepdims=True)    # (..., 1)
        x_hat = (x - mu) / np.sqrt(var + self.eps)
        out   = self.gamma * x_hat + self.beta
        self._cache = {'x_hat': x_hat, 'var': var}
        return out

    def backward(self, dout):
        x_hat = self._cache['x_hat']
        var   = self._cache['var']
        d     = dout.shape[-1]
        std_inv = 1.0 / np.sqrt(var + self.eps)
        
        dgamma = (dout * x_hat).sum(axis=tuple(range(dout.ndim-1)))
        dbeta  = dout.sum(axis=tuple(range(dout.ndim-1)))
        
        g = dout * self.gamma
        dx = std_inv * (g
                        - g.mean(axis=-1, keepdims=True)
                        - x_hat * (g * x_hat).mean(axis=-1, keepdims=True))
        return dx, dgamma, dbeta
```

### J.3 RMSNorm from Scratch (NumPy)

```python
class RMSNorm:
    def __init__(self, d_model, epsilon=1e-6):
        self.gamma = np.ones(d_model)
        self.eps   = epsilon
        self._cache = {}

    def forward(self, x):
        """x: shape (..., d_model)"""
        rms  = np.sqrt((x**2).mean(axis=-1, keepdims=True) + self.eps)
        x_hat = x / rms
        out   = self.gamma * x_hat
        self._cache = {'x_hat': x_hat, 'rms': rms, 'x': x}
        return out

    def backward(self, dout):
        x_hat = self._cache['x_hat']
        rms   = self._cache['rms']
        
        dgamma = (dout * x_hat).sum(axis=tuple(range(dout.ndim-1)))
        
        g  = dout * self.gamma            # upstream scaled by gamma
        # Two-term formula (no mean-centring correction)
        dx = (g - x_hat * (g * x_hat).mean(axis=-1, keepdims=True)) / rms
        return dx, dgamma
```

### J.4 Spectral Normalization (NumPy)

```python
class SpectralNorm:
    """Wraps a linear layer with spectral normalisation."""
    def __init__(self, W, n_power_iter=1):
        m, n = W.shape
        self.W = W
        self.n_iter = n_power_iter
        # Initialise singular vectors
        self.u = np.random.randn(m) 
        self.u /= np.linalg.norm(self.u)
        self.v = np.random.randn(n)
        self.v /= np.linalg.norm(self.v)

    def _power_iter(self):
        u, v = self.u.copy(), self.v.copy()
        for _ in range(self.n_iter):
            v = self.W.T @ u
            v /= np.linalg.norm(v) + 1e-12
            u = self.W @ v
            u /= np.linalg.norm(u) + 1e-12
        sigma = u @ self.W @ v
        self.u, self.v = u, v  # update cached vectors
        return sigma

    def normalized_weight(self):
        sigma = self._power_iter()
        return self.W / (sigma + 1e-12), sigma
```


---

## Appendix K: Normalisation in the Transformer Block

This appendix traces the complete data flow through a modern Pre-LN transformer block, showing exactly where each normalisation is applied and what it does.

```
PRE-LN TRANSFORMER BLOCK DATA FLOW
════════════════════════════════════════════════════════════════════════

  Input: x ∈ R^{T × d}  (sequence of T tokens, d-dimensional)

  ─── ATTENTION SUBLAYER ──────────────────────────────────────────
  
  1. RMSNorm (or LayerNorm):
     x_norm = RMSNorm(x)         # normalise over d for each token
     x_norm ∈ R^{T × d},  RMS(x_norm[t,:]) = 1 for each t
  
  2. Q, K, V projections:
     Q = x_norm @ W_Q            # R^{T × d_k}
     K = x_norm @ W_K            # R^{T × d_k}
     V = x_norm @ W_V            # R^{T × d_v}
  
  2a. QK-Norm (LLaMA-3, SD3): 
     Q = RMSNorm(Q)              # bounds Q magnitude per head
     K = RMSNorm(K)              # prevents attn score explosion
  
  3. Scaled dot-product attention:
     A = softmax(Q K^T / √d_k) V  # R^{T × d_v}
  
  4. Output projection + residual:
     x = x + A @ W_O             # residual connection bypasses norm
  
  ─── FFN SUBLAYER ────────────────────────────────────────────────
  
  5. RMSNorm:
     x_norm = RMSNorm(x)
  
  6. SwiGLU FFN:
     x_ffn = (W_gate @ x_norm ⊙ SiLU(W_up @ x_norm)) @ W_down
     # gate branch (SiLU activation) × up branch, projected down
  
  7. Residual:
     x = x + x_ffn
  
  Output: x ∈ R^{T × d}  (same shape as input)
  
  ─── KEY OBSERVATIONS ────────────────────────────────────────────
  
  • RMSNorm appears BEFORE sublayers (Pre-LN)
  • Residual connections bypass normalization
  • Gradient highway: ∂L/∂x passes directly through residuals
    without any normalisation operation
  • QK-Norm is inside the attention computation, not on residual
  • Final output has no normalisation → post-stack LN or RMSNorm
    applied once before the unembedding head

════════════════════════════════════════════════════════════════════════
```

### Why the Residual Highway Matters

In Pre-LN, the gradient flows through:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[0]}} = \prod_{l=1}^{L} \left(I + \frac{\partial \operatorname{Sublayer}_l}{\partial \mathbf{x}^{[l-1]}}\right) \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[L]}}$$

At initialization, $\partial \operatorname{Sublayer}_l / \partial \mathbf{x} \approx 0$ (small random weights), so:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[0]}} \approx I^L \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[L]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}^{[L]}}$$

Every layer has gradient norm $\approx \lVert \partial \mathcal{L}/\partial \mathbf{x}^{[L]} \rVert$ — constant across depth. This is what enables training 100-layer transformers without warmup.

In Post-LN, the gradient from $\mathbf{x}^{[L]}$ to $\mathbf{x}^{[0]}$ passes through $L$ LayerNorm Jacobians, each of which scales the gradient by $\approx \gamma/\sigma$. If $\sigma \approx 1$ and $\gamma = 1$, this is fine — but during the first few hundred steps, before $\sigma$ stabilises, the Jacobian can have unexpected scaling, causing gradient explosion.

---

## Appendix L: Conditioning Mechanisms Comparison

A systematic comparison of how different models inject conditioning information:

| Method | How | Where Used | Pros | Cons |
|---|---|---|---|---|
| Token concatenation | Prepend condition tokens to sequence | BERT (CLS token), GPT-3 (few-shot) | Simple; uses same attention | Increases sequence length |
| Cross-attention | Separate KV from condition | T5, DALL-E 2, SD text cross-attn | Full interaction | High compute |
| Conditional BN / FiLM | γ(c), β(c) from condition | Class-cond BN, VQA, ControlNet | Efficient; per-feature | Less expressive than cross-attn |
| AdaIN | Replace mean+std with condition's | StyleGAN, fast style transfer | Clean style/content split | Only controls statistics |
| AdaLN-Zero | Zero-init FiLM on each layer | DiT, SD3, FLUX | Stable init; efficient | More params than simple bias |
| LoRA | Low-rank updates to W_Q,W_K,W_V | LLM fine-tuning | Few params; composable | No conditioning at norm level |

**For AI (ControlNet):** ControlNet (Zhang et al., 2023) adds spatial conditioning (edge maps, depth, pose) to Stable Diffusion by copying the U-Net encoder and connecting it to the denoising U-Net via zero-convolutions (1×1 conv initialised to zero). The zero-init trick is the same idea as AdaLN-Zero: start as an identity, learn the conditioning path. This is now a standard pattern for adding new conditioning modalities to pre-trained models without disrupting the base model.


---

## Appendix M: Advanced Topics

### M.1 QK-Norm Analysis

Query-Key normalisation is a 2023-2024 innovation that addresses attention score instability in long-context models. Without QK-norm, the dot product $q_i \cdot k_j$ has variance:

$$\operatorname{Var}(q_i \cdot k_j) = \sum_{d=1}^{d_k} \operatorname{Var}(q_{id}) \operatorname{Var}(k_{jd}) = d_k \cdot \operatorname{Var}(q) \cdot \operatorname{Var}(k)$$

The $\sqrt{d_k}$ scaling in $\operatorname{softmax}(QK^\top/\sqrt{d_k})$ controls this variance at initialisation. But as training progresses and the model learns sharper attention patterns, $\operatorname{Var}(q)$ and $\operatorname{Var}(k)$ can grow, making the pre-softmax logits very large (sharp attention).

Very sharp attention has several problems:
1. **Entropy collapse:** Attention becomes a hard argmax, losing the ability to aggregate information from multiple positions
2. **Gradient issues:** The softmax Jacobian $\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\top$ approaches a rank-1 matrix as $\mathbf{p}$ approaches a one-hot vector, reducing the gradient magnitude
3. **FP16 overflow:** Large attention logits before softmax can overflow FP16

QK-norm normalises Q and K to unit RMS before the dot product:
$$\hat{Q}_i = \frac{Q_i}{\operatorname{RMS}(Q_i)}, \quad \hat{K}_j = \frac{K_j}{\operatorname{RMS}(K_j)}$$

After this normalisation, $\operatorname{Var}(\hat{q} \cdot \hat{k}) \leq 1$ regardless of training stage, providing a stable upper bound on attention logit magnitude throughout training.

**Trade-off:** QK-norm prevents the model from learning *un-normalised* query-key comparisons. This may slightly reduce the model's ability to implement hard attention (which can be useful for copying tokens exactly). In practice, the stability benefit outweighs this, especially for training at scale or long contexts.

### M.2 Pre-LN vs Post-LN: Formal Gradient Analysis at Initialisation

At initialisation (weights $\sim \mathcal{N}(0, 1/d)$, biases $= 0$):

**Post-LN gradient at layer $l$ (from depth $L$):**

The backward pass through a Post-LN block involves:
$$\frac{\partial \mathbf{x}^{[l]}}{\partial \mathbf{x}^{[l-1]}} = J_{\text{LN}}(\mathbf{x}^{[l-1]} + \text{Sub}(\mathbf{x}^{[l-1]})) \cdot (I + J_{\text{Sub}}(\mathbf{x}^{[l-1]}))$$

At init, $J_{\text{Sub}} \approx O(\sqrt{1/d})$ (small), so:
$$\frac{\partial \mathbf{x}^{[l]}}{\partial \mathbf{x}^{[l-1]}} \approx J_{\text{LN}}(\mathbf{x}^{[l-1]})$$

The LN Jacobian $J_{\text{LN}}$ has spectral norm $\leq 1$ but can scale gradients in certain directions. Composing $L$ such Jacobians gives:
$$\frac{\partial \mathbf{x}^{[L]}}{\partial \mathbf{x}^{[0]}} \approx \prod_{l=1}^{L} J_{\text{LN}}^{[l]}$$

This product can collapse to near-zero in the directions orthogonal to the top eigenvectors of each Jacobian. After a warmup period, $J_{\text{Sub}}$ grows large enough to dominate, and the gradient becomes stable — but the first few thousand steps are unstable.

**Pre-LN gradient at layer $l$:**
$$\frac{\partial \mathbf{x}^{[l]}}{\partial \mathbf{x}^{[l-1]}} = I + J_{\text{Sub}}(J_{\text{LN}}(\mathbf{x}^{[l-1]}))$$

At init, $J_{\text{Sub}} \approx O(\sqrt{1/d})$, so:
$$\frac{\partial \mathbf{x}^{[l]}}{\partial \mathbf{x}^{[l-1]}} \approx I$$

The identity is the dominant term. Composing $L$ of these gives $I^L = I$. The gradient at depth 0 is approximately the same as at depth $L$ — no vanishing or explosion.

### M.3 The DeepNorm Approach

DeepNorm (Wang et al., 2022) achieves stable training of 1000-layer Post-LN transformers by modifying the residual scaling:

$$\mathbf{x}^{[l]} = \operatorname{LayerNorm}(\alpha \mathbf{x}^{[l-1]} + \operatorname{Sublayer}(\mathbf{x}^{[l-1]}))$$

where $\alpha > 1$ amplifies the residual connection relative to the sublayer output. The key insight: by making the residual $\alpha \mathbf{x}$ larger, the LN Jacobian at initialisation is dominated by the residual direction, which is well-conditioned. After LN, the residual direction has variance $\approx 1$, and the sublayer contributes a small perturbation.

The optimal $\alpha$ depends on model depth $L$ and architecture (encoder vs. decoder):
- Encoder: $\alpha = (2L)^{1/4}$
- Decoder (self-attention): $\alpha = (6L)^{1/4}$

Combined with a specific weight initialisation (scaling weights by $\beta = (8L)^{-1/4}$), DeepNorm enables training of 1000-layer models with Post-LN that outperform equivalent Pre-LN models.

**For AI:** While Pre-LN is the practical default, DeepNorm shows that Post-LN can be scaled with the right initialisation. The Post-LN advantage (empirically slightly better final performance on benchmarks) is worth pursuing at extreme scale with the DeepNorm recipe.

### M.4 Normalisation in Multimodal Models

Modern multimodal models (image-text, video-text, audio-text) face a specific challenge: different modalities have different statistical properties. A natural language token embedding has different mean and variance than a patch embedding from a ViT, which differs from an audio spectrogram embedding.

**Separate normalisation per modality:** MM-DiT (the architecture in SD3 and FLUX) uses separate normalisation parameters for text tokens and image tokens:

$$\operatorname{MNorm}(\mathbf{x}_{\text{img}}, \mathbf{x}_{\text{txt}}) = [\operatorname{RMSNorm}_{\text{img}}(\mathbf{x}_{\text{img}}), \operatorname{RMSNorm}_{\text{txt}}(\mathbf{x}_{\text{txt}})]$$

where $\operatorname{RMSNorm}_{\text{img}}$ and $\operatorname{RMSNorm}_{\text{txt}}$ have separate learned $\gamma$ vectors. The two streams interact through attention (sharing Q, K, V attention) but have separate normalisation, allowing each modality's statistics to be adapted independently.

**Joint normalisation for aligned modalities:** CLIP-style models (Radford et al., 2021) use shared LayerNorm across the image and text encoders when embeddings are projected to the same space. The assumption is that after alignment training, the two modalities live in the same statistical space.


---

## Appendix N: Extended Mathematical Analysis

### N.1 Batch Normalization as a Re-parameterisation

Consider a network where layer $l$ consists of a linear transform followed by BatchNorm:

$$\mathbf{y}^{[l]} = \operatorname{BN}(W^{[l]} \mathbf{a}^{[l-1]})$$

The BN normalises over the batch dimension. Let the batch of pre-activations be $Z^{[l]} = W^{[l]} A^{[l-1]}$ where $A^{[l-1]}$ has columns $\mathbf{a}^{(1)}, \ldots, \mathbf{a}^{(m)}$.

**Scale invariance:** For any scalar $c > 0$:
$$\operatorname{BN}(cW\mathbf{a}) = \operatorname{BN}(W\mathbf{a})$$

because BN subtracts the mean and divides by the standard deviation — both scale by $c$, so it cancels. This means the loss is constant along the ray $\{cW : c > 0\}$ in parameter space. Gradient descent on $W$ can make no progress in the radial direction $W/\lVert W \rVert$, only in the angular (direction) component.

**Implicit constraint:** The effective parameter space for a BN-normalised linear layer is the $(d^2 - 1)$-dimensional manifold of directions (unit vectors of rows), plus the scalar $\gamma$ per neuron. This reduces the effective degrees of freedom compared to a plain linear layer ($d^2$ parameters), which can be interpreted as a form of structural regularisation.

### N.2 LayerNorm and the Attention Mechanism

In the transformer attention mechanism:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

LayerNorm is applied to the input $\mathbf{x}$ before computing $Q = \mathbf{x} W_Q$ and $K = \mathbf{x} W_K$. Since LN normalises $\mathbf{x}$ to unit variance, the initial variance of $Q_{ij} = \langle \hat{\mathbf{x}}_i, \mathbf{w}_Q^j \rangle$ is:

$$\operatorname{Var}(Q_{ij}) = \sum_{k=1}^{d} \operatorname{Var}(\hat{x}_{ik}) \cdot (w_Q^{jk})^2 \approx \frac{1}{d} \sum_k (w_Q^{jk})^2 = \frac{\lVert \mathbf{w}_Q^j \rVert^2}{d}$$

With initialisation $W_Q \sim \mathcal{N}(0, 1/d)$, the weight norm squared is $\approx d_k$ (sum of $d_k$ squared Gaussians each with variance $1/d$), giving $\operatorname{Var}(Q_{ij}) \approx d_k/d$. For typical transformer dimensions ($d_k = d/h$ where $h$ is number of heads), this is $1/h$, which is small and consistent.

The $\sqrt{d_k}$ scaling in attention then gives:
$$\operatorname{Var}\left(\frac{Q_i \cdot K_j}{\sqrt{d_k}}\right) = \frac{d_k \cdot \operatorname{Var}(Q) \cdot \operatorname{Var}(K)}{d_k} = \operatorname{Var}(Q) \cdot \operatorname{Var}(K) \approx 1$$

This is why the combination of LN + $\sqrt{d_k}$ scaling keeps attention logits unit-variance at initialisation.

### N.3 Group Norm — Relationship Between G and Stability

The choice of group size $C/G$ involves a bias-variance trade-off:

- **Large $G$ (small groups, e.g., G=C = InstanceNorm):** Each group has few channels, giving a noisy variance estimate (high variance). The normalisation is more aggressive — each channel is treated independently. Pros: removes all inter-channel statistics, useful for style transfer. Cons: unstable for small feature maps (when $H \times W$ is small).

- **Small $G$ (large groups, e.g., G=1 = LayerNorm):** Each group has all channels, giving a stable variance estimate (low variance). Pros: stable statistics. Cons: all channels share the same normalisation, reducing the ability to normalise channels with very different scales independently.

The empirically optimal $G = 32$ for typical detection/segmentation networks (C=256) gives groups of size 8. This provides enough channels per group for a stable variance estimate while allowing some per-group specialisation.

**For AI — when to choose G:** A simple rule: choose $G$ such that $C/G \geq 8$. This ensures each group's variance estimate has at least 8 degrees of freedom, giving a coefficient of variation (standard error / mean) of $\approx 1/\sqrt{2 \times 8} \approx 18\%$, which is acceptable for normalisation purposes.

### N.4 Numerical Stability of Welford's Algorithm

Let $\bar{x}_n = \frac{1}{n}\sum_{i=1}^n x_i$ and $M_{2,n} = \sum_{i=1}^n (x_i - \bar{x}_n)^2$.

Welford's update for adding a new observation $x_{n+1}$:
$$\bar{x}_{n+1} = \bar{x}_n + \frac{x_{n+1} - \bar{x}_n}{n+1}$$
$$M_{2,n+1} = M_{2,n} + (x_{n+1} - \bar{x}_n)(x_{n+1} - \bar{x}_{n+1})$$

**Forward error bound (Higham 2002):** The computed $M_{2,n}$ satisfies:
$$|fl(M_{2,n}) - M_{2,n}| \leq c_n \cdot n \cdot u \cdot M_{2,n}$$

where $u$ is the unit roundoff ($\approx 10^{-7}$ for FP32) and $c_n$ is a small constant depending on $n$. This is a *relative* error bound: the error grows as $O(n \cdot u \cdot \sigma^2)$.

For the naive formula $M_{2,n} = \sum x_i^2 - n\bar{x}^2$:
$$|fl(\sum x_i^2 - n\bar{x}^2) - M_{2,n}| \lesssim n \cdot u \cdot |\sum x_i^2|$$

Since $|\sum x_i^2| \gg M_{2,n}$ when the data has large mean relative to variance, the naive formula has much larger relative error. In the worst case, the computed variance can be negative (when the rounding errors in $\sum x_i^2$ are larger than the true variance).

**Example quantifying the difference:** Data: $x_i = 10^4 + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, 1)$, $n = 100$.

- True variance: $\approx 1$
- $\sum x_i^2 \approx 100 \times 10^8 = 10^{10}$, $n\bar{x}^2 \approx 10^{10}$
- FP32 relative error in $\sum x_i^2$: $\approx 10^{10} \times 10^{-7} = 10^3$
- Naive estimate error: $O(10^3)$ — orders of magnitude larger than the true variance of 1!
- Welford estimate error: $O(n \cdot u \cdot \sigma^2) = O(100 \times 10^{-7} \times 1) = O(10^{-5})$ — accurate

This is why production implementations of BatchNorm and LayerNorm always use Welford's algorithm or a numerically equivalent two-pass approach.

### N.5 Spectral Norm and Layer Width

An interesting property of spectral normalisation: the spectral norm $\sigma_1(W)$ grows with layer width. For a random matrix $W \in \mathbb{R}^{m \times n}$ with $W_{ij} \sim \mathcal{N}(0, 1/n)$:

By the Marchenko-Pastur law (for $m, n \to \infty$ with $m/n \to r$):
$$\sigma_1(W) \to (1 + \sqrt{r})$$

For $m = n$ (square), $\sigma_1(W) \to 2$. After spectral normalisation, $\sigma_1(\bar{W}) = 1$, scaling all singular values by $1/2$ on average.

**For AI:** This means spectral normalisation removes the scaling that would otherwise grow with network width. Wider networks (with more channels) have larger spectral norms, and spectral normalisation corrects for this. This is related to the $\mu$P (maximal update parameterisation) theory, where the appropriate weight initialisation scale should depend on the number of neurons in the layer.


---

## Appendix O: Glossary

| Term | Definition |
|---|---|
| **Internal covariate shift** | Change in the distribution of layer inputs during training as the parameters of preceding layers are updated |
| **β-smooth loss** | A loss function whose gradient is β-Lipschitz; enables gradient descent with step size up to 1/β |
| **Batch statistics** | Mean and variance computed over all samples in a mini-batch for a fixed channel/feature |
| **Running statistics** | Exponential moving average of batch statistics, used for inference in BatchNorm |
| **Normalised sphere** | The $(d-2)$-dimensional manifold of zero-mean, unit-variance vectors in $\mathbb{R}^d$ |
| **Pre-LN / Post-LN** | Whether LayerNorm is applied before or after the sublayer in a residual block |
| **Scale invariance (BN)** | Property that $\operatorname{BN}(cW) = \operatorname{BN}(W)$ for any scalar $c > 0$; makes loss flat in weight magnitude |
| **Spectral norm** | Largest singular value of a matrix; equals the $\ell^2 \to \ell^2$ operator norm |
| **1-Lipschitz** | A function $f$ satisfying $\lVert f(\mathbf{x}) - f(\mathbf{y}) \rVert \leq \lVert \mathbf{x} - \mathbf{y} \rVert$ for all $\mathbf{x}, \mathbf{y}$ |
| **Power iteration** | Algorithm to approximate the dominant eigenvector/singular value by repeated matrix-vector products |
| **AdaIN** | Adaptive Instance Normalisation; transfers style by replacing content feature statistics with style feature statistics |
| **FiLM** | Feature-wise Linear Modulation; modulates features via $\gamma(\mathbf{c}) \odot \mathbf{x} + \beta(\mathbf{c})$ |
| **AdaLN-Zero** | Adaptive LayerNorm with all conditioning weights initialised to zero; DiT standard conditioning mechanism |
| **QK-norm** | RMSNorm applied to Query and Key projections; prevents attention score explosion in long-context models |
| **DeepNorm** | Post-LN variant with $\alpha \mathbf{x}$ residual scaling; enables 1000-layer stable training |
| **NFNet** | Normalisation-Free Network; achieves BN-equivalent performance via Scaled Weight Standardisation + AGC |
| **Welford's algorithm** | Online algorithm for numerically stable one-pass mean and variance computation |
| **Mean field theory** | Statistical physics approach to analysing signal propagation in random neural networks |
| **Catastrophic cancellation** | Severe loss of significant digits when subtracting two nearly equal floating-point numbers |
| **ε (epsilon)** | Small constant added to variance before taking square root for numerical stability; typical value $10^{-5}$ to $10^{-6}$ |

---

## Appendix P: Quick Implementation Checklist

When implementing normalisation in a new architecture, verify:

**BatchNorm:**
- [ ] Computing statistics over the correct axes (N, H, W per channel)
- [ ] Maintaining separate running stats with momentum 0.1
- [ ] Calling `model.train()` / `model.eval()` correctly
- [ ] ε ≥ 1e-4 if using FP16 without FP32 upcast

**LayerNorm:**
- [ ] Normalising over the feature dimension (last axis), not batch
- [ ] Parameters $\gamma, \beta$ have shape matching the normalised dimension
- [ ] No running statistics needed (same forward pass for train/eval)

**RMSNorm:**
- [ ] Using RMS (not std): $\sqrt{\frac{1}{H}\sum x_i^2}$ without mean subtraction
- [ ] Only $\gamma$ parameter, no $\beta$
- [ ] ε $\approx$ 1e-6 (BF16 safe) or 1e-5 (FP32 default)

**Spectral Normalization:**
- [ ] Initialising $\tilde{\mathbf{u}}$ and $\tilde{\mathbf{v}}$ as unit random vectors
- [ ] Updating singular vectors after each forward pass
- [ ] Using 1-3 power iterations per step (1 is usually sufficient)
- [ ] Dividing by $\sigma + \epsilon$ (not $\sigma$ alone) to avoid div by zero at init

**AdaLN-Zero:**
- [ ] Initialising the conditioning linear layer weights to zero
- [ ] Projecting conditioning signal to 6 affine parameter vectors per block
- [ ] Applying $\alpha$ gate to the sublayer output before residual addition
- [ ] The conditioning embedding includes timestep + class/text signals


---

## Appendix Q: Advanced Derivations

### Q.1 Weight Normalization Gradient Geometry

Weight normalisation reparameterises $\mathbf{w} = g \hat{\mathbf{v}}$ where $\hat{\mathbf{v}} = \mathbf{v}/\lVert \mathbf{v} \rVert$. The gradient of the loss $\ell$ with respect to $g$ and $\mathbf{v}$ can be decomposed geometrically.

The gradient w.r.t. $\mathbf{v}$ lies in the tangent space of the unit sphere at $\hat{\mathbf{v}}$:

$$\nabla_\mathbf{v} \ell = \frac{g}{\lVert \mathbf{v} \rVert} M_{\hat{\mathbf{v}}} \nabla_\mathbf{w} \ell$$

where $M_{\hat{\mathbf{v}}} = I - \hat{\mathbf{v}}\hat{\mathbf{v}}^\top$ is the projection onto the subspace orthogonal to $\hat{\mathbf{v}}$ (the tangent space of the sphere).

**Interpretation:** The gradient in direction space ($\mathbf{v}$) is always orthogonal to the current direction $\hat{\mathbf{v}}$. This means gradient steps change the direction of $\mathbf{w}$ without changing its magnitude (since $g$ handles the magnitude independently). This decoupling makes optimisation geometry clearer: $g$ can be updated aggressively (it's a scalar) while $\mathbf{v}$ is updated on the sphere manifold.

**Effective learning rate for $\mathbf{v}$:** The gradient magnitude satisfies:
$$\lVert \nabla_\mathbf{v} \ell \rVert \leq \frac{g}{\lVert \mathbf{v} \rVert} \lVert \nabla_\mathbf{w} \ell \rVert = \frac{\lVert \nabla_\mathbf{w} \ell \rVert}{\lVert \hat{\mathbf{v}} \rVert} = \lVert \nabla_\mathbf{w} \ell \rVert$$

So the gradient magnitude is preserved (it's just projected). But the effective step in weight space is $\eta \lVert \nabla_\mathbf{v} \ell \rVert / \lVert \mathbf{v} \rVert$, which scales inversely with $\lVert \mathbf{v} \rVert$. As the direction vector grows, the effective learning rate for direction shrinks — the same implicit learning rate normalisation as BatchNorm.

### Q.2 The Lipschitz Constant of a Composed BN Network

Consider a $L$-layer network where each layer has form $f^{[l]}(\mathbf{x}) = \gamma^{[l]} \hat{\mathbf{z}}^{[l]} + \boldsymbol{\beta}^{[l]}$ where $\hat{\mathbf{z}}^{[l]}$ is the BN-normalised pre-activation. The Lipschitz constant of the $l$-th layer (treating BN statistics as fixed):

$$\operatorname{Lip}(f^{[l]}) = \frac{\lVert \gamma^{[l]} \rVert_\infty}{\sigma^{[l]}_{\min}}$$

where $\sigma^{[l]}_{\min} = \min_c \sqrt{\sigma^{[l]2}_c + \epsilon}$ is the smallest normalised std across channels. With $\gamma^{[l]} = \mathbf{1}$ (standard init) and $\sigma^{[l]}_{\min} \approx \sqrt{\sigma^2 + \epsilon}$:

$$\operatorname{Lip}(f^{[l]}) \approx \frac{1}{\sqrt{\sigma^2 + \epsilon}}$$

The Lipschitz constant of each layer is bounded, and the product over layers gives:

$$\operatorname{Lip}(f^{[L]} \circ \cdots \circ f^{[1]}) \leq \prod_{l=1}^{L} \operatorname{Lip}(f^{[l]}) \leq \prod_{l=1}^{L} \frac{\lVert \gamma^{[l]} \rVert_\infty}{\sqrt{\sigma^{[l]2}_{\min} + \epsilon}}$$

This bound is the formal version of the gradient magnitude bound in the Santurkar et al. theorem.

### Q.3 Fixed Point Analysis of Normalisation

A normalisation scheme is **self-consistent** if applying it iteratively converges to a fixed point. For BatchNorm:

Define the mapping $T: (\mu, \sigma^2) \mapsto (\mu', \sigma'^2)$ where $\mu'$ and $\sigma'^2$ are the mean and variance of the BN output $y = \gamma \hat{x} + \beta$ when the input has mean $\mu$ and variance $\sigma^2$.

Since $\hat{x}$ always has mean 0 and variance 1:
$$\mu' = \mathbb{E}[\gamma \hat{x} + \beta] = \beta$$
$$\sigma'^2 = \operatorname{Var}(\gamma \hat{x} + \beta) = \gamma^2$$

So the fixed point is $(\mu^*, \sigma^{*2}) = (\beta, \gamma^2)$, independent of the input distribution! This means after one application of BatchNorm, the output always has mean $\beta$ and variance $\gamma^2$. BatchNorm forces the output distribution's first two moments to be the learned values $(\beta, \gamma^2)$.

For LayerNorm and RMSNorm, the same analysis applies per-sample rather than per-batch, with the same conclusion: normalisation forces a fixed output moment structure.

**Implication for initialisation:** Because normalisation forces fixed moments, the initialisation of $\gamma$ and $\beta$ matters more than the initialisation of the weights feeding into the normalised layer. The weights determine the *direction* of the output, but the *scale* is set by $\gamma$.

---

## Appendix R: Common Interview Questions

**Q: What happens to BatchNorm when batch size = 1?**
A: The batch variance is 0 (only one sample), so the normalised output is 0 regardless of the input. BatchNorm requires batch size ≥ 2; practically needs ≥ 8 for stable statistics. Use GroupNorm or LayerNorm for batch size 1.

**Q: Why does Pre-LN not need learning rate warmup but Post-LN does?**
A: At initialisation, Pre-LN residual blocks approximate the identity (sublayer output is small), so gradients flow directly through residuals without passing through any LN Jacobian. Post-LN gradients pass through LN Jacobians, which can scale gradients unpredictably in the first few thousand steps until the sublayer outputs grow to be comparable to the residual.

**Q: What is the difference between RMSNorm and LayerNorm with β=0?**
A: RMSNorm normalises by the root mean square $\sqrt{E[x^2]}$. LayerNorm (even with β=0) normalises by the standard deviation $\sqrt{E[(x-\mu)^2]}$. When $E[x] \neq 0$, RMS > std, and the outputs differ. They are equal only when $E[x] = 0$.

**Q: How many power iterations does spectral normalisation use, and why?**
A: One power iteration per gradient step is standard. Since the weight matrix changes slowly (small gradient steps), the singular vectors from the previous step are already close to the true ones. One step refines the estimate, which is sufficient for an accurate spectral norm. More iterations improve accuracy but increase compute.

**Q: Why is AdaLN-Zero's initialisation to zero important?**
A: With all conditioning weights initialised to zero, every transformer block starts as an identity function (output = input). The entire model begins as a trivial residual network. As training progresses, blocks gradually activate. This prevents the conditioning signal from disrupting the model at the start of training, enabling stable training of very deep conditioned transformers.

**Q: In what situation would you use Welford's algorithm over two-pass variance?**
A: Online settings where you cannot store all data (streaming, very large datasets). Welford's computes exact mean and variance in one pass. Two-pass (first pass for mean, second pass for variance) is also numerically stable and sometimes preferred for parallel computation. Both are better than the naive single-pass formula.

**Q: Why does BatchNorm add regularisation?**
A: BN statistics are computed from mini-batches, which are noisy estimates of population statistics. This noise adds stochastic perturbations to each sample's normalised activation, similar to dropout. Additionally, BN's scale invariance effectively constrains weights to the unit sphere in direction space, acting like implicit weight decay.


---

## Appendix S: Summary Reference Card

```
NORMALISATION SUMMARY REFERENCE CARD
════════════════════════════════════════════════════════════════════════

  BATCH NORM              LAYER NORM              RMSNORM
  ─────────────────────   ─────────────────────   ────────────────────
  Norm axes: (N,H,W)      Norm axes: (C,H,W)      Norm axes: (C,H,W)
  Stats: per-channel      Stats: per-sample        No mean subtraction
  Params: γ,β ∈ R^C      Params: γ,β ∈ R^d       Params: γ ∈ R^d
  Batch dep: YES          Batch dep: NO            Batch dep: NO
  Needs eval mode: YES    Needs eval mode: NO      Needs eval mode: NO
  Use in: CNNs            Use in: Transformers     Use in: LLMs
  ε default: 1e-5         ε default: 1e-5          ε default: 1e-6

  GROUP NORM              INSTANCE NORM            SPECTRAL NORM
  ─────────────────────   ─────────────────────   ────────────────────
  Norm axes: (C/G,H,W)    Norm axes: (H,W)         Normalises weights
  Per (N,G)               Per (N,C)                W̄ = W/σ₁(W)
  Params: γ,β ∈ R^C      Params: γ,β ∈ R^C       Lipschitz = 1
  Batch dep: NO           Batch dep: NO            Use: GAN discriminator
  Use in: Detection       Use in: Style transfer   Power iter: 1 step

  ADAIN                   CONDITIONAL BN           ADALN-ZERO
  ─────────────────────   ─────────────────────   ────────────────────
  σ(y)·(x-μ(x))/σ(x)    γ(c)·x̂+β(c)            γ,β,α from MLP(c)
  +μ(y)                   c = class/text           Zero-init all params
  Separates content/style  FiLM variant             Residual gating
  Use: Style transfer     Use: Class-cond gen      Use: DiT, FLUX

  ────────────────────────────────────────────────────────────────────

  PLACEMENT GUIDE:
  Pre-LN:   x → LN → Sublayer → + → x     (GPT-2, LLaMA, modern LLMs)
  Post-LN:  x → Sublayer → + → LN → x     (original transformer, BERT)
  AdaLN-0:  x → AdaLN → Sublayer → ⊙α → + → x  (DiT, SD3, FLUX)

  KEY EQUATIONS:
  BN backward:   dx_i = (γ/σ)[dŷ_i - mean_j(dŷ_j) - x̂_i·mean_j(dŷ_j·x̂_j)]
  LN backward:   same formula with axes over features not batch
  RMS backward:  dx_i = (γ_i/r)[g_i - x̂_i·mean_j(g_j·x̂_j)]
  Spect norm:    σ₁ ← û^T W v̂  (one power iter step per gradient step)

════════════════════════════════════════════════════════════════════════
```

This reference card summarises the key distinguishing properties of each normalisation method. Keep it handy when designing new architectures or debugging training instabilities.

---

*End of Normalization Techniques notes. Continue to [Sampling Methods →](../04-Sampling-Methods/notes.md)*


---

## Appendix T: Learning Rate Interaction with Normalisation

One of the most practically important effects of normalisation is how it changes the sensitivity of the training process to the learning rate.

### T.1 BatchNorm Allows Larger Learning Rates

The classical advice for training deep networks without normalisation is to use small learning rates ($\eta \approx 10^{-4}$). With BatchNorm, learning rates of $10^{-2}$ to $10^{-1}$ are common. Why?

The stability condition for gradient descent is $\eta \leq 1/\beta$ where $\beta$ is the smoothness constant. Without BN, $\beta$ grows exponentially with depth (each layer multiplies the smoothness by its Lipschitz constant). With BN:

$$\beta_{\text{BN}} \leq \frac{\gamma^2}{\sigma^2 + \epsilon} \cdot C_{\text{downstream}}$$

This is bounded independently of depth — the BN layer acts as a "smoothness bottleneck" that prevents $\beta$ from growing with depth. The implication: you can use the same learning rate for a 10-layer and 100-layer BN network, but for a plain network, the 100-layer network requires a learning rate $\approx (1/L_{\text{factor}})^{90}$ smaller.

### T.2 Pre-LN and Learning Rate Schedule

Pre-LN transformers do not require learning rate warmup because the gradient magnitudes are approximately constant at initialisation (as derived in §4.3 and Appendix Q.2). This enables a simpler learning rate schedule:
- **With warmup:** Linear warmup from $0$ to $\eta_{\max}$ over $W$ steps, then cosine decay
- **Without warmup (Pre-LN only):** Start directly at $\eta_{\max}$, cosine decay

For LLaMA and similar models, warmup is still used as a conservative practice, but it is not *required* for Pre-LN. Ablations show that removing warmup from Pre-LN models causes at most a small loss spike in the first few steps, which quickly recovers.

### T.3 RMSNorm and Learning Rate

RMSNorm and LayerNorm have nearly identical learning rate sensitivity. The slight computational savings of RMSNorm do not change the learning rate requirements. Both allow the same large learning rates enabled by Pre-LN placement.

However, the choice of ε interacts with learning rate stability in mixed precision:
- **BF16 with ε=1e-5:** Safe, BF16 has large dynamic range
- **FP16 with ε=1e-5:** Can underflow; consider ε=1e-3 or FP32 upcasting
- **FP16 with large learning rate (>0.01):** Weight updates can overflow; gradient clipping at norm 1.0 is standard for LLM training in FP16

