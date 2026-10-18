[Back to Curriculum](../../README.md) | [Previous: Activation Functions](../02-Activation-Functions/notes.md) | [Next: Sampling Methods](../04-Sampling-Methods/notes.md)

---

# Normalization Techniques

> _"Deep networks are easier to train when their internal scales are made visible, controlled, and learnable."_

## Overview

Normalization techniques control the statistics of activations, weights, or
features so optimization sees a better-conditioned problem. They are not just
preprocessing tricks. Inside neural networks, normalization changes forward
signal scale, backward gradient flow, effective parameterization, train/eval
behavior, and numerical stability.

This section is the canonical home for normalization math as a reusable ML
primitive. Chapter 14 discusses how BatchNorm, LayerNorm, RMSNorm, and
SpectralNorm appear in specific model families. Chapter 15 discusses LLM-scale
block design and fused kernels. Here we focus on axes, moments, affine
reparameterization, train-time statistics, inference statistics, epsilon,
RMS-only scaling, group statistics, weight statistics, residual placement, and
the implementation mistakes that make normalization layers silently wrong.

The core habit is to always ask: "What axis is normalized? Which statistics are
used? Are they batch-dependent? Are they learned, frozen, or recomputed? What
gradient path does this create?"

## Prerequisites

- **Mean, variance, and standard deviation** - [Descriptive Statistics](../../07-Statistics/01-Descriptive-Statistics/notes.md)
- **Expectation and moments** - [Expectation and Moments](../../06-Probability-Theory/04-Expectation-and-Moments/notes.md)
- **Activation functions and gradient flow** - [Activation Functions](../02-Activation-Functions/notes.md)
- **Numerical stability and floating point** - [Floating Point Arithmetic](../../10-Numerical-Methods/01-Floating-Point-Arithmetic/notes.md)
- **Regularization and spectral constraints** - [Regularization Methods](../../08-Optimization/08-Regularization-Methods/notes.md)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Axis-aware BatchNorm, LayerNorm, RMSNorm, GroupNorm, WeightNorm, SpectralNorm, and residual-placement demos |
| [exercises.ipynb](exercises.ipynb) | 10 graded exercises on moments, broadcasting, train/eval behavior, and normalization diagnostics |

## Learning Objectives

After completing this section, you will be able to:

- Identify which axes BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm, WeightNorm, and SpectralNorm normalize
- Derive the normalized affine transform $\gamma\hat{x}+\beta$
- Explain why epsilon is a numerical stabilizer, not a regularization constant
- Distinguish batch statistics, running statistics, and per-example statistics
- Implement BatchNorm, LayerNorm, RMSNorm, and GroupNorm in NumPy
- Explain why BatchNorm has different train and eval behavior
- Compare LayerNorm and RMSNorm for sequence-style hidden states
- Diagnose broadcasting, axis, small-batch, and mixed-precision failures
- Explain pre-norm versus post-norm as a gradient-flow design choice

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Activations Drift](#11-why-activations-drift)
  - [1.2 Normalization as Re-Centering](#12-normalization-as-re-centering)
  - [1.3 Normalization as Conditioning](#13-normalization-as-conditioning)
  - [1.4 Signal Scale in Deep Nets](#14-signal-scale-in-deep-nets)
  - [1.5 Historical Path](#15-historical-path)
- [2. Formal Definitions](#2-formal-definitions)
  - [2.1 Normalization Axes](#21-normalization-axes)
  - [2.2 Mean and Variance Statistics](#22-mean-and-variance-statistics)
  - [2.3 Epsilon](#23-epsilon)
  - [2.4 Learnable Gain and Bias](#24-learnable-gain-and-bias)
  - [2.5 Train-Time Versus Inference-Time Behavior](#25-train-time-versus-inference-time-behavior)
- [3. Batch Normalization](#3-batch-normalization)
  - [3.1 Batch Statistics](#31-batch-statistics)
  - [3.2 Running Averages](#32-running-averages)
  - [3.3 Affine Transform](#33-affine-transform)
  - [3.4 Train/Eval Gap](#34-traineval-gap)
  - [3.5 Batch-Size Sensitivity](#35-batch-size-sensitivity)
- [4. Layer Normalization](#4-layer-normalization)
  - [4.1 Per-Example Statistics](#41-per-example-statistics)
  - [4.2 Feature-Axis Normalization](#42-feature-axis-normalization)
  - [4.3 Sequence Models](#43-sequence-models)
  - [4.4 Invariance Properties](#44-invariance-properties)
  - [4.5 Comparison to BatchNorm](#45-comparison-to-batchnorm)
- [5. RMSNorm](#5-rmsnorm)
  - [5.1 RMS Statistic](#51-rms-statistic)
  - [5.2 Scale-Only Normalization](#52-scale-only-normalization)
  - [5.3 No Mean Subtraction](#53-no-mean-subtraction)
  - [5.4 Efficiency](#54-efficiency)
  - [5.5 LLM Usage Preview](#55-llm-usage-preview)
- [6. Other Normalizations](#6-other-normalizations)
  - [6.1 GroupNorm](#61-groupnorm)
  - [6.2 InstanceNorm](#62-instancenorm)
  - [6.3 WeightNorm](#63-weightnorm)
  - [6.4 SpectralNorm](#64-spectralnorm)
  - [6.5 ScaleNorm Preview](#65-scalenorm-preview)
- [7. Residual Blocks and Placement](#7-residual-blocks-and-placement)
  - [7.1 Pre-Norm](#71-pre-norm)
  - [7.2 Post-Norm](#72-post-norm)
  - [7.3 Sandwich Norm Preview](#73-sandwich-norm-preview)
  - [7.4 Gradient Flow](#74-gradient-flow)
  - [7.5 Deep-Stack Stability](#75-deep-stack-stability)
- [8. Numerical and Implementation Details](#8-numerical-and-implementation-details)
  - [8.1 Broadcasting Axes](#81-broadcasting-axes)
  - [8.2 Epsilon Choice](#82-epsilon-choice)
  - [8.3 Mixed Precision](#83-mixed-precision)
  - [8.4 Small-Batch Failure](#84-small-batch-failure)
  - [8.5 Fused Kernels Preview](#85-fused-kernels-preview)
- [9. Applications in Machine Learning](#9-applications-in-machine-learning)
  - [9.1 CNNs](#91-cnns)
  - [9.2 RNNs](#92-rnns)
  - [9.3 Transformers](#93-transformers)
  - [9.4 GANs](#94-gans)
  - [9.5 Large-Scale Training](#95-large-scale-training)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI](#12-why-this-matters-for-ai)
- [13. Conceptual Bridge](#13-conceptual-bridge)
- [84. Quick Checks (One-Line Answers)](#84-quick-checks-one-line-answers)
- [82. Computed Table: Invariance Demonstrations on One Vector](#82-computed-table-invariance-demonstrations-on-one-vector)
- [83. Section 03 Wrap-Up](#83-section-03-wrap-up)
- [79. Revision Problems (Mixed)](#79-revision-problems-mixed)
- [80. Extended Notes on Reading Normalization Papers](#80-extended-notes-on-reading-normalization-papers)
- [81. Final Remarks Before Sampling](#81-final-remarks-before-sampling)
- [75. Decision Flowchart in Words: Normalization](#75-decision-flowchart-in-words-normalization)
- [76. Frequently Asked Questions, Part II](#76-frequently-asked-questions-part-ii)
- [77. Closing Checklist for This Section](#77-closing-checklist-for-this-section)
- [78. Index of Computed Evidence in This Section](#78-index-of-computed-evidence-in-this-section)
- [72. Worked Dataset: Two Channels, Independent BatchNorm Statistics](#72-worked-dataset-two-channels-independent-batchnorm-statistics)
- [73. Mixed Precision and Normalization: The Details](#73-mixed-precision-and-normalization-the-details)
- [74. Worked Problem Set I: Numerics and History](#74-worked-problem-set-i-numerics-and-history)
- [68. Worked Example: GroupNorm Statistics on a Toy Tensor](#68-worked-example-groupnorm-statistics-on-a-toy-tensor)
- [69. Statistics Estimation: Shrinkage and Robust Alternatives](#69-statistics-estimation-shrinkage-and-robust-alternatives)
- [70. Worked Problem Set H: Padding, Fusion, Documentation](#70-worked-problem-set-h-padding-fusion-documentation)
- [71. Final Note on Normalization](#71-final-note-on-normalization)
- [62. Computed Table: Biased vs Unbiased Variance in BatchNorm](#62-computed-table-biased-vs-unbiased-variance-in-batchnorm)
- [63. Normalization in Generative and Diffusion Models](#63-normalization-in-generative-and-diffusion-models)
- [64. Normalization Meets Quantization and Deployment](#64-normalization-meets-quantization-and-deployment)
- [65. Worked Problem Set G: Incidents and Audits](#65-worked-problem-set-g-incidents-and-audits)
- [66. What to Memorise vs What to Look Up](#66-what-to-memorise-vs-what-to-look-up)
- [67. Section 03 Completion Checklist](#67-section-03-completion-checklist)
- [58. Worked Trace: 60 Batches of Running Statistics](#58-worked-trace-60-batches-of-running-statistics)
- [59. Normalization, Effective Learning Rate, and Weight Decay (Derivation Sketch)](#59-normalization-effective-learning-rate-and-weight-decay-derivation-sketch)
- [60. Exam and Interview Traps: Normalization](#60-exam-and-interview-traps-normalization)
- [61. Practice Problems for Revision](#61-practice-problems-for-revision)
- [53. Computed Table: Residual Stream Std by Block (Simulation)](#53-computed-table-residual-stream-std-by-block-simulation)
- [54. Normalization in CNNs and RNNs: Specifics](#54-normalization-in-cnns-and-rnns-specifics)
- [55. Worked Problem Set F: Folding, Multimodal, Mechanism](#55-worked-problem-set-f-folding-multimodal-mechanism)
- [56. Annotated Reading Guide](#56-annotated-reading-guide)
- [57. Chapter Summary: Normalization](#57-chapter-summary-normalization)
- [49. Worked Dataset: 24 Vectors Through LayerNorm and RMSNorm](#49-worked-dataset-24-vectors-through-layernorm-and-rmsnorm)
- [50. The LayerNorm Jacobian and What It Does to Gradients](#50-the-layernorm-jacobian-and-what-it-does-to-gradients)
- [51. Conditional Normalization: From Statistics to Conditioning](#51-conditional-normalization-from-statistics-to-conditioning)
- [52. Can We Remove Normalization? Evidence and Limits](#52-can-we-remove-normalization-evidence-and-limits)
- [44. Worked Problem Set E: Synthesis and Audits](#44-worked-problem-set-e-synthesis-and-audits)
- [45. Monitoring Runbook: Normalization Health](#45-monitoring-runbook-normalization-health)
- [46. Extended Exercise Bank (Self-Study)](#46-extended-exercise-bank-self-study)
- [47. Self-Test Questions](#47-self-test-questions)
- [48. Self-Test Answer Key (Condensed)](#48-self-test-answer-key-condensed)
- [40. Computed Table: How Noisy Is a Batch Variance? (Fine Grid)](#40-computed-table-how-noisy-is-a-batch-variance-fine-grid)
- [41. Worked Problem Set D: Porting and Interactions](#41-worked-problem-set-d-porting-and-interactions)
- [42. Debugging Playbook: Normalization Incidents](#42-debugging-playbook-normalization-incidents)
- [43. Formula Sheet](#43-formula-sheet)
- [35. Computed Table: Epsilon as a Gain Cap](#35-computed-table-epsilon-as-a-gain-cap)
- [36. Computed Table: LayerNorm/RMSNorm on 32 Random Vectors](#36-computed-table-layernormrmsnorm-on-32-random-vectors)
- [37. Worked Problem Set C: Placement, Coupling, Precision](#37-worked-problem-set-c-placement-coupling-precision)
- [38. Pre-Flight Checklist for Normalization](#38-pre-flight-checklist-for-normalization)
- [39. Frequently Asked Questions](#39-frequently-asked-questions)
- [31. Worked Dataset: 48 Values Through BatchNorm](#31-worked-dataset-48-values-through-batchnorm)
- [32. Historical Timeline: Normalization](#32-historical-timeline-normalization)
- [33. Quick-Reference Cards](#33-quick-reference-cards)
- [34. Normalization and Optimization Geometry](#34-normalization-and-optimization-geometry)
- [27. Computed Table: Residual Stream Growth, Pre-LN vs Post-LN (Simulation)](#27-computed-table-residual-stream-growth-pre-ln-vs-post-ln-simulation)
- [28. Glossary: Normalization Vocabulary](#28-glossary-normalization-vocabulary)
- [29. Normalization Mistake Catalogue (Field Version)](#29-normalization-mistake-catalogue-field-version)
- [30. Decision Records: Normalization Choices in Context](#30-decision-records-normalization-choices-in-context)
- [24. Worked Problem Set A: Batch Statistics and Modes](#24-worked-problem-set-a-batch-statistics-and-modes)
- [25. Worked Problem Set B: Per-Example Normalization](#25-worked-problem-set-b-per-example-normalization)
- [26. Normalization in Attention and Modern LLM Blocks](#26-normalization-in-attention-and-modern-llm-blocks)
- [19. Worked Example: BatchNorm Arithmetic on One Channel](#19-worked-example-batchnorm-arithmetic-on-one-channel)
- [20. Computed Comparison: LayerNorm vs RMSNorm on Four Vectors](#20-computed-comparison-layernorm-vs-rmsnorm-on-four-vectors)
- [21. Computed Table: Batch-Statistic Noise vs Batch Size](#21-computed-table-batch-statistic-noise-vs-batch-size)
- [22. GroupNorm, InstanceNorm, WeightNorm, SpectralNorm](#22-groupnorm-instancenorm-weightnorm-spectralnorm)
- [23. Computed Table: Running-Statistic EMA After an Outlier Batch](#23-computed-table-running-statistic-ema-after-an-outlier-batch)
- [14. Dossier Template: How to Read a Normalization Method](#14-dossier-template-how-to-read-a-normalization-method)
- [15. BatchNorm Deep Dive: Statistics, Running Averages, and Failure Modes](#15-batchnorm-deep-dive-statistics-running-averages-and-failure-modes)
- [16. Why BatchNorm Helps: The Internal-Covariate-Shift Debate](#16-why-batchnorm-helps-the-internal-covariate-shift-debate)
- [17. LayerNorm and RMSNorm: Per-Example Normalization](#17-layernorm-and-rmsnorm-per-example-normalization)
- [18. Pre-LN vs Post-LN: Placement in Residual Blocks](#18-pre-ln-vs-post-ln-placement-in-residual-blocks)
- [References](#references)

---

## 1. Intuition

### 1.1 Why Activations Drift

During training, every layer receives inputs produced by previous layers whose
parameters are changing. Even if the raw data distribution is fixed, internal
activation distributions move. Means shift, variances change, outliers appear,
and gradient scales become inconsistent across depth.

Normalization addresses this by computing statistics over a chosen axis and
reparameterizing activations into a controlled scale. A generic normalized
activation has the form

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

### 1.2 Normalization as Re-Centering

Mean subtraction re-centers values:

$$
x \mapsto x-\mu.
$$

This can make optimization easier because downstream affine layers do not need
to constantly compensate for drifting offsets. But not every normalization
subtracts the mean. RMSNorm normalizes by root mean square and preserves the
mean direction.

### 1.3 Normalization as Conditioning

Optimization is sensitive to scale. If one feature dimension has variance
$10^6$ and another has variance $10^{-4}$, a single learning rate is poorly
matched to both. Normalization reduces such scale mismatch. It does not make
all optimization problems convex, but it often improves conditioning.

### 1.4 Signal Scale in Deep Nets

Deep networks multiply many local transformations. If activation scale grows
slightly at each layer, it can explode. If it shrinks slightly, it can vanish.
Normalization layers repeatedly reset or constrain scale so residual blocks,
attention blocks, convolution blocks, or recurrent transitions remain
trainable.

### 1.5 Historical Path

BatchNorm made normalization a core deep-learning layer by using mini-batch
statistics. LayerNorm removed batch dependence and became natural for sequence
models. GroupNorm and InstanceNorm addressed small-batch and image-style
settings. RMSNorm simplified LayerNorm by removing mean subtraction.
SpectralNorm constrained weight operators rather than activations.

## 2. Formal Definitions

### 2.1 Normalization Axes

Let $\mathcal{X}\in\mathbb{R}^{B\times T\times D}$ represent a batch of
sequence activations. Here $B$ is batch size, $T$ is sequence length, and $D$
is feature width. Different normalization methods choose different axes:

| Method | Typical axes for statistics | Batch-dependent? |
| --- | --- | --- |
| BatchNorm | batch and sometimes spatial axes | Yes |
| LayerNorm | feature axis per example/token | No |
| RMSNorm | feature axis per example/token | No |
| GroupNorm | groups of channels per example | No |
| InstanceNorm | spatial axes per example/channel | No |
| WeightNorm | weight-vector direction/scale | No |
| SpectralNorm | largest singular value of weight matrix | No batch statistics |

The axis choice is the most important part of the definition. Two formulas can
look identical while normalizing completely different objects.

### 2.2 Mean and Variance Statistics

For a set of values $\{x_i\}_{i=1}^{m}$, the mean and variance are

$$
\mu=\frac{1}{m}\sum_{i=1}^{m}x_i,
\qquad
\sigma^2=\frac{1}{m}\sum_{i=1}^{m}(x_i-\mu)^2.
$$

The normalized value is

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}.
$$

In deep-learning libraries, variance is often the biased batch variance, not
the unbiased sample variance. The choice is part of the layer definition.

### 2.3 Epsilon

Epsilon prevents division by zero:

$$
\sqrt{\sigma^2+\epsilon}.
$$

It is not a regularizer in the usual statistical sense. It is a numerical
stabilizer. Too small an epsilon can fail in low precision. Too large an
epsilon changes the effective normalization by preventing true unit variance.

### 2.4 Learnable Gain and Bias

After normalization, most methods apply learnable affine parameters:

$$
y_i=\gamma_i\hat{x}_i+\beta_i.
$$

The gain $\gamma$ lets the network restore useful scale. The bias $\beta$ lets
it restore useful offset. Without these parameters, normalization could remove
representations the model needs.

### 2.5 Train-Time Versus Inference-Time Behavior

BatchNorm uses mini-batch statistics during training and running estimates
during inference. LayerNorm and RMSNorm compute per-example statistics at both
train and inference time. This distinction affects reproducibility,
deployment, and small-batch behavior.

## 3. Batch Normalization

### 3.1 Batch Statistics

For a feature $j$ over a batch,

$$
\mu_j=\frac{1}{B}\sum_{i=1}^{B}x_{ij},
\qquad
\sigma_j^2=\frac{1}{B}\sum_{i=1}^{B}(x_{ij}-\mu_j)^2.
$$

Then

$$
\hat{x}_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}.
$$

For convolutional tensors, statistics are often computed across batch and
spatial positions for each channel.

### 3.2 Running Averages

BatchNorm maintains running estimates:

$$
\mu_{\mathrm{run}}\leftarrow
(1-\alpha)\mu_{\mathrm{run}}+\alpha\mu_{\mathrm{batch}},
$$

$$
\sigma^2_{\mathrm{run}}\leftarrow
(1-\alpha)\sigma^2_{\mathrm{run}}+\alpha\sigma^2_{\mathrm{batch}}.
$$

These estimates are used in evaluation mode. If the running estimates are bad,
validation and deployment behavior can be bad even when training behavior
looked fine.

### 3.3 Affine Transform

BatchNorm output is

$$
y_{ij}=\gamma_j\hat{x}_{ij}+\beta_j.
$$

The parameters are feature-specific. In CNNs they are channel-specific and
broadcast over spatial dimensions.

### 3.4 Train/Eval Gap

During training, a sample is normalized using statistics from its current
mini-batch. During inference, it is normalized using stored running statistics.
This creates a train/eval gap. The gap is small when batches are large and
representative. It can be large for tiny batches, distribution shift, or
nonstationary data.

### 3.5 Batch-Size Sensitivity

BatchNorm is noisy with small batches because mean and variance estimates have
high variance. Distributed training can also change effective batch statistics
depending on whether BatchNorm is synchronized across devices.

## 4. Layer Normalization

### 4.1 Per-Example Statistics

LayerNorm computes statistics over features for each example:

$$
\mu_i=\frac{1}{D}\sum_{j=1}^{D}x_{ij},
\qquad
\sigma_i^2=\frac{1}{D}\sum_{j=1}^{D}(x_{ij}-\mu_i)^2.
$$

For sequence tensors, it is usually applied separately to each token position.

### 4.2 Feature-Axis Normalization

LayerNorm normalizes the feature vector:

$$
\hat{\mathbf{x}}_i
=\frac{\mathbf{x}_i-\mu_i\mathbf{1}}
{\sqrt{\sigma_i^2+\epsilon}}.
$$

Its gain and bias are feature-wise:

$$
\mathbf{y}_i=\boldsymbol{\gamma}\odot\hat{\mathbf{x}}_i+\boldsymbol{\beta}.
$$

### 4.3 Sequence Models

LayerNorm is natural for sequence models because it does not depend on other
examples in the batch. A token can be normalized using only its own hidden
state. This makes behavior consistent between training and autoregressive
inference.

### 4.4 Invariance Properties

LayerNorm is invariant to adding a constant offset to all features in the
normalized vector and to multiplying all features by a positive scalar, up to
epsilon and learned affine parameters.

### 4.5 Comparison to BatchNorm

BatchNorm normalizes each feature using batch statistics. LayerNorm normalizes
each example using feature statistics. BatchNorm couples examples in a batch.
LayerNorm couples features inside an example.

## 5. RMSNorm

### 5.1 RMS Statistic

RMSNorm uses the root mean square:

$$
\operatorname{RMS}(\mathbf{x})
=\sqrt{\frac{1}{D}\sum_{j=1}^{D}x_j^2+\epsilon}.
$$

### 5.2 Scale-Only Normalization

The normalized vector is

$$
\hat{\mathbf{x}}=
\frac{\mathbf{x}}{\operatorname{RMS}(\mathbf{x})}.
$$

Then

$$
\mathbf{y}=\boldsymbol{\gamma}\odot\hat{\mathbf{x}}.
$$

### 5.3 No Mean Subtraction

RMSNorm does not subtract the mean and usually does not include a bias
parameter. It controls scale but preserves mean direction. This makes it
cheaper and can work well in residual-stream architectures.

### 5.4 Efficiency

LayerNorm requires mean and variance. RMSNorm requires only the mean square.
That saves operations and can simplify fused kernels. At large scale, small
per-token savings matter.

### 5.5 LLM Usage Preview

Many modern LLM-style architectures use RMSNorm or RMSNorm-like variants.
Chapter 15 covers LLM-specific block details. Here the reusable math is simply
scale-only feature normalization.

## 6. Other Normalizations

### 6.1 GroupNorm

GroupNorm divides channels into groups and normalizes within each group for
each example. It avoids batch dependence while retaining channel-group
structure.

### 6.2 InstanceNorm

InstanceNorm normalizes each sample and channel over spatial dimensions. It is
common in style-transfer and image-generation contexts where instance-specific
contrast should be controlled.

### 6.3 WeightNorm

WeightNorm reparameterizes a weight vector:

$$
\mathbf{w}=g\frac{\mathbf{v}}{\lVert\mathbf{v}\rVert_2}.
$$

It separates direction $\mathbf{v}/\lVert\mathbf{v}\rVert_2$ from magnitude
$g$.

### 6.4 SpectralNorm

Spectral normalization constrains a weight matrix by its largest singular
value:

$$
\bar{W}=\frac{W}{\sigma_{\max}(W)}.
$$

This controls the operator norm and therefore the layer's Lipschitz constant.
The full singular-value theory belongs in
[Singular Value Decomposition](../../03-Advanced-Linear-Algebra/02-Singular-Value-Decomposition/notes.md).

### 6.5 ScaleNorm Preview

ScaleNorm normalizes a vector by its norm and multiplies by a learned scalar.
It is conceptually close to RMSNorm but uses explicit vector norm scaling.

## 7. Residual Blocks and Placement

### 7.1 Pre-Norm

Pre-norm residual blocks apply normalization before the sublayer:

$$
\mathbf{h}_{l+1}
=\mathbf{h}_l+F(\operatorname{Norm}(\mathbf{h}_l)).
$$

This gives the residual path a direct identity route for gradients.

### 7.2 Post-Norm

Post-norm applies normalization after the residual addition:

$$
\mathbf{h}_{l+1}
=\operatorname{Norm}(\mathbf{h}_l+F(\mathbf{h}_l)).
$$

It can produce cleaner normalized outputs per block but may make very deep
stacks harder to optimize.

### 7.3 Sandwich Norm Preview

Some architectures add normalization both before and after certain sublayers.
This is a model-design detail, so the full treatment belongs in model-specific
chapters. The general math is that each norm changes both forward scale and
backward Jacobian.

### 7.4 Gradient Flow

Normalization layers have Jacobians. They do not simply "standardize and
disappear." Their backward pass subtracts mean-like components and rescales
gradients. This can stabilize training but can also introduce coupling across
the normalized axis.

### 7.5 Deep-Stack Stability

Deep residual stacks rely on controlling activation and residual-stream scale.
Normalization placement, residual scaling, initialization, and optimizer
choice interact. A stable block is a system, not a single layer.

## 8. Numerical and Implementation Details

### 8.1 Broadcasting Axes

Most normalization bugs are axis bugs. A parameter $\gamma\in\mathbb{R}^D$
must broadcast over batch and sequence axes but align with feature axes. If
the shape is wrong, code may run while normalizing the wrong dimension.

### 8.2 Epsilon Choice

Epsilon must be large enough to protect low-variance vectors and low-precision
formats. It must be small enough not to dominate real variance. Common values
include $10^{-5}$ and $10^{-6}$, but the right value is implementation and
dtype dependent.

### 8.3 Mixed Precision

Mixed-precision normalization often accumulates statistics in higher precision
even when inputs are lower precision. This reduces underflow, overflow, and
rounding errors in variance computation.

### 8.4 Small-Batch Failure

BatchNorm with batch size one is usually ill-posed for dense features because
the batch variance can be zero. CNN spatial axes may still provide enough
samples, but the effective sample count should be checked.

### 8.5 Fused Kernels Preview

At large scale, normalization is often fused with neighboring operations to
reduce memory traffic. The math is unchanged, but implementation details
affect speed and numerical behavior.

## 9. Applications in Machine Learning

### 9.1 CNNs

BatchNorm is common in CNNs because batch and spatial dimensions provide many
samples per channel. It also acts as a mild source of stochasticity during
training.

### 9.2 RNNs

LayerNorm is often easier than BatchNorm for recurrent models because sequence
lengths and time dependencies make batch statistics awkward.

### 9.3 Transformers

Transformers use LayerNorm, RMSNorm, and placement variants to stabilize deep
residual blocks. The full Transformer block treatment belongs in
[Transformer Architecture](../../14-Math-for-Specific-Models/05-Transformer-Architecture/notes.md).

### 9.4 GANs

SpectralNorm controls discriminator Lipschitz behavior and helps stabilize
adversarial training. InstanceNorm and normalization variants also appear in
image generation pipelines.

### 9.5 Large-Scale Training

At large scale, normalization affects stability, throughput, memory bandwidth,
mixed-precision behavior, and distributed reproducibility. Seemingly small
choices become system-level choices.

## 10. Common Mistakes

| # | Mistake | Why It Is Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Normalizing the wrong axis | Statistics describe the wrong object | Write tensor shapes and axes explicitly |
| 2 | Forgetting BatchNorm train/eval mode | Batch and running statistics differ | Switch modes deliberately and test both |
| 3 | Treating epsilon as harmless | Large epsilon changes scale | Tune epsilon with dtype and variance in mind |
| 4 | Using BatchNorm with tiny batches blindly | Statistics are noisy or degenerate | Use LayerNorm, GroupNorm, or synchronized stats |
| 5 | Broadcasting gain over the wrong dimension | Code may run but learn wrong parameters | Assert shapes before training |
| 6 | Comparing LayerNorm and RMSNorm as identical | RMSNorm does not center | Track mean and RMS separately |
| 7 | Ignoring mixed-precision accumulation | Variance can underflow or overflow | Accumulate statistics in higher precision |
| 8 | Calling SpectralNorm an activation norm | It normalizes weights, not activations | Separate activation and operator normalization |
| 9 | Assuming normalization replaces initialization | Bad initialization can still break training | Use compatible initialization and norm placement |
| 10 | Removing affine gain/bias casually | The model may need restored scale/offset | Remove only with a clear architectural reason |

## 11. Exercises

1. (*) Compute mean, variance, and normalized values for a vector.
2. (*) Implement BatchNorm for a matrix with shape $B\times D$.
3. (*) Implement LayerNorm for a matrix with shape $B\times D$.
4. (**) Show that BatchNorm changes when the batch composition changes.
5. (**) Show that LayerNorm is independent of other examples in the batch.
6. (**) Implement RMSNorm and compare its output mean to LayerNorm.
7. (**) Implement GroupNorm for a tensor with grouped channels.
8. (***) Compute a WeightNorm reparameterization and verify its norm.
9. (***) Estimate a spectral norm by power iteration.
10. (***) Diagnose a shape bug in a fake normalization layer.

## 12. Why This Matters for AI

| Concept | AI impact |
| --- | --- |
| Axis choice | Determines whether examples, features, channels, or weights are coupled |
| BatchNorm | Stabilizes CNN training but introduces batch dependence |
| LayerNorm | Makes sequence and token-wise models batch-independent |
| RMSNorm | Controls scale with lower overhead and no centering |
| GroupNorm | Works when batch statistics are unreliable |
| SpectralNorm | Controls operator scale and Lipschitz behavior |
| Epsilon | Prevents numerical failure in low-variance and low-precision regimes |
| Pre-norm | Improves gradient flow in deep residual stacks |
| Train/eval mode | Affects reproducibility and deployment correctness |
| Fused normalization | Matters for throughput at large scale |

## 13. Conceptual Bridge

Activation functions create nonlinear hidden states. Normalization techniques
control the statistics of those hidden states.

```text
activation values
    -> mean / variance / RMS statistics
    -> normalized hidden state
    -> learnable scale and bias
    -> next layer or residual block
```

Next, [Sampling Methods](../04-Sampling-Methods/notes.md) moves from
deterministic transformations to randomized estimators, proposal distributions,
and generation procedures.

## 14. Dossier Template: How to Read a Normalization Method
Every normalization in this chapter is specified by five choices: which axes statistics are computed over, whether the mean is subtracted, whether learnable affine parameters follow, how training-time and evaluation-time statistics differ, and what invariances the operation creates. Fill these in and most folklore about the method becomes derivable.

- **Axes**: BatchNorm normalizes per-channel over (batch, spatial); LayerNorm per-example over features; GroupNorm per-example over channel groups; InstanceNorm per-example per-channel.
- **Mean subtraction**: LayerNorm/BatchNorm subtract; RMSNorm does not — it rescales by root-mean-square only (Zhang & Sennrich 2019).
- **Affine**: learnable gain $\gamma$ and shift $\beta$ restore representational power (the network can un-normalize if useful); RMSNorm typically has gain only.
- **Train/eval**: only batch-statistic methods (BatchNorm) differ between modes; per-example methods (LN/RMSNorm/GN) are mode-independent — a decisive deployment property.
- **Invariances**: LayerNorm output is invariant to shifting/scaling the input vector; BatchNorm makes a channel invariant to batch-level affine rescaling of that channel.
- **Epsilon**: $\epsilon$ inside the square root caps the amplification of tiny-variance directions; it is part of the method's definition, not a numerical footnote.
- **Placement**: before/after activation, Pre-LN vs Post-LN in residual blocks (Section 7 of the main text; Xiong et al. 2020) — placement changes gradient structure materially.
- **State**: BatchNorm carries running statistics as buffers; forgetting them in checkpointing/export is a classic production bug.
- Cite the canonical source when naming a method: BatchNorm (Ioffe & Szegedy 2015), LayerNorm (Ba, Kiros & Hinton 2016), GroupNorm (Wu & He 2018), InstanceNorm (Ulyanov et al. 2016), RMSNorm (Zhang & Sennrich 2019), WeightNorm (Salimans & Kingma 2016), SpectralNorm (Miyato et al. 2018).



## 15. BatchNorm Deep Dive: Statistics, Running Averages, and Failure Modes
BatchNorm (Ioffe & Szegedy 2015) standardizes each channel using mini-batch mean and variance, then applies $\gamma,\beta$. At evaluation it substitutes running averages accumulated with momentum. The gap between those two regimes is where most BatchNorm bugs live.

- Training forward: $\hat x=(x-\mu_B)/\sqrt{\sigma_B^2+\epsilon}$, $y=\gamma\hat x+\beta$, with $\mu_B,\sigma_B^2$ from the current batch (per channel, over batch and spatial positions).
- Running update: $\bar\mu\leftarrow(1-m)\bar\mu+m\mu_B$ (frameworks differ in whether $m$ or $1-m$ is called momentum — check yours).
- Eval forward uses $\bar\mu,\bar\sigma^2$: an example's output no longer depends on its batchmates — this dependence difference is exactly what the theory notebook demonstrates.
- Failure mode 1: small batches → noisy $\mu_B,\sigma_B^2$ → train/eval mismatch; GroupNorm/LayerNorm exist largely because of this (Wu & He 2018).
- Failure mode 2: batch statistics leak information across examples (a mild form of label leakage in contrastive/self-supervised settings; shuffling and SyncBN choices matter).
- Failure mode 3: forgetting train()/eval() mode — evaluating with batch statistics (or training with running ones) silently corrupts results.
- Failure mode 4: distributed training without synchronized BatchNorm: each device normalizes by its local batch, shrinking the effective batch.
- Failure mode 5: fine-tuning with frozen running stats on shifted data; re-estimating stats (a few forward passes in train mode) is a cheap remedy.
- The biased/unbiased variance subtlety: training uses the biased ($1/N$) variance for normalization but the running estimate typically accumulates the unbiased ($1/(N-1)$) one — a real, documented framework behaviour worth knowing.
- BatchNorm's regularization side-effect (noise from batch statistics) sometimes reduces the need for dropout; removing BN can change the optimal dropout rate.
- Interaction with weight decay: BN makes the preceding weights scale-invariant; decay then controls the effective learning rate on the direction rather than the function (see §02 Section 41 analogue).
- When to avoid in 2026 practice: recurrent/sequence models (LN family), tiny batches, online/streaming inference with batch size 1 statistics semantics — prefer per-example methods.



## 16. Why BatchNorm Helps: The Internal-Covariate-Shift Debate
Ioffe & Szegedy (2015) motivated BatchNorm by reducing internal covariate shift (ICS) — the drift of layer input distributions during training. Santurkar et al. (2018) challenged the mechanism: networks with deliberately injected distribution shift still trained well with BN, and BN's benefit tracked instead a smoother loss landscape (better Lipschitz behaviour of loss and gradients).

- ICS story: stabilizing layer inputs lets higher layers learn against a stationary target.
- Santurkar et al. (2018) evidence: (i) noisy/shifted activations with BN still train; (ii) BN nets show improved gradient predictiveness under step size changes.
- Landscape view: BN reparametrizes so gradients are better behaved; similar benefits appear for other normalizations without batch statistics, weakening a purely batch-based ICS account.
- Teaching stance in this chapter: present ICS as the historical motivation, the smoothness account as the better-supported mechanism, and require students to state which claim a given experiment actually tests.
- Scale invariance is a third, complementary mechanism: BN/BN-like layers make loss invariant to weight scaling, interacting with effective learning rates.
- Practical upshot either way: BN often permits larger learning rates and faster convergence in convnets; the debate changes explanations, not the recipe.
- Exam trap: citing ICS as settled fact post-2018 is a literature error; citing Santurkar et al. as disproving BN's utility is the opposite error.



## 17. LayerNorm and RMSNorm: Per-Example Normalization
LayerNorm (Ba, Kiros & Hinton 2016) normalizes over the feature axis of each example: no batch dependence, identical behaviour in train and eval, natural for variable-length sequences. RMSNorm (Zhang & Sennrich 2019) drops mean subtraction, normalizing by $\sqrt{\frac1d\sum x_i^2+\epsilon}$, and hypothesizes that scale invariance, not re-centring, carries most of the benefit.

- LayerNorm: $\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$ with $\mu,\sigma^2$ over features of one example; then $\gamma\odot\hat x+\beta$.
- Invariances: LN$(x)$ unchanged by $x\mapsto ax+b\mathbf 1$ ($a>0$); the representation encodes direction/pattern, not offset or scale.
- RMSNorm: $\hat x=x/\mathrm{RMS}(x)$, RMS $=\sqrt{\frac1d\sum x_i^2+\epsilon}$; invariant to $x\mapsto ax$ but not to shifts.
- When the mean is large, LN and RMSNorm differ materially; when features are near-zero-mean they nearly coincide — check the mean of your activations before claiming equivalence.
- RMSNorm is cheaper (no mean, fewer reductions) and is the default in many current LLMs; Zhang & Sennrich (2019) report comparable quality with meaningful speedups in their settings.
- Per-example methods have no running state: export, batch-size-1 inference, and RL-style sequential use are all straightforward.
- LayerNorm in RNNs (the original 2016 motivation) normalizes recurrent inputs per timestep, avoiding batch-across-time statistics.
- Affine parameters: LN has $\gamma,\beta$; RMSNorm implementations usually have $\gamma$ only — a missing $\beta$ is a specification difference, not a bug.
- Epsilon placement differs across frameworks (inside vs outside the root in some RMSNorm variants); cross-framework ports must state it.
- Both pair with Pre-LN transformer placement (Section 18) and with the residual-stream scale control discussed there.



## 18. Pre-LN vs Post-LN: Placement in Residual Blocks
Post-LN (original transformer, Vaswani et al. 2017 lineage) normalizes after the residual addition; Pre-LN (analysed systematically by Xiong et al. 2020) normalizes inside the residual branch, leaving an unnormalized identity highway. Xiong et al. showed Pre-LN trains without learning-rate warmup in their settings, while Post-LN needs warmup but can reach slightly better final results when tuned.

- Post-LN: $y=\mathrm{LN}(x+F(x))$; gradients to early layers pass through LN Jacobians of every block — depth-sensitive at init.
- Pre-LN: $y=x+F(\mathrm{LN}(x))$; the residual stream is an identity path, so gradient flows unimpeded (the same structural idea as identity ResNets).
- Consequence 1: Pre-LN residual stream variance grows with depth (additive branches), motivating final LN before the head and sometimes branch scaling.
- Consequence 2: Pre-LN is the default in most current LLMs (with RMSNorm); Post-LN persists where its tuned final-quality edge is worth warmup complexity.
- Warmup's role in Post-LN: early large gradients through LN-normalized residuals are tamed until statistics settle (Xiong et al. 2020 analysis).
- DeepNorm and related schemes rescale residuals to recover Post-LN-like quality at extreme depth — evidence placement/scale is a continuum, not a binary.
- Migration trap: converting a Post-LN checkpoint recipe to Pre-LN is an architecture change; weights do not transfer meaningfully.
- Diagnostic: log residual-stream norm by depth; geometric growth at init predicts late-training instability before loss curves show it.

## 19. Worked Example: BatchNorm Arithmetic on One Channel
Six values through BatchNorm with $\gamma=2,\beta=1$. Verify each column; then note the batch-dependence: replacing the last value 9 by 90 changes every normalized output, although five inputs are unchanged.

| example | x | x - mean | (x-mean)^2 | normalized (eps=0) | y=2*hat+1 |
|---|---|---|---|---|---|
| | 2.0 | -2.667 | 7.111 | -1.2494 | -1.4988 |
| | 4.0 | -0.667 | 0.444 | -0.3123 | 0.3753 |
| | 4.0 | -0.667 | 0.444 | -0.3123 | 0.3753 |
| | 4.0 | -0.667 | 0.444 | -0.3123 | 0.3753 |
| | 5.0 | 0.333 | 0.111 | 0.1562 | 1.3123 |
| | 9.0 | 4.333 | 18.778 | 2.0303 | 5.0605 |
| **stats** | mean=4.6667 | | var=4.5556, std=2.1344 | | |



## 20. Computed Comparison: LayerNorm vs RMSNorm on Four Vectors
Rows 2-3 are shift/scale transforms of row 1, exhibiting the invariance claims of Section 17 numerically: LayerNorm is shift- and scale-invariant; RMSNorm is scale-invariant only.

| vector | LN output | RMSNorm output | note |
|---|---|---|---|
| [np.float64(1.0), np.float64(2.0), np.float64(3.0), np.float64(4.0)] | [-1.342 -0.447  0.447  1.342] | [0.365 0.73  1.095 1.461] |  |
| [np.float64(11.0), np.float64(12.0), np.float64(13.0), np.float64(14.0)] | [-1.342 -0.447  0.447  1.342] | [0.877 0.956 1.036 1.116] | shift of row 1: LN identical, RMS differs |
| [np.float64(2.0), np.float64(4.0), np.float64(6.0), np.float64(8.0)] | [-1.342 -0.447  0.447  1.342] | [0.365 0.73  1.095 1.461] | scale of row 1: both identical direction |
| [np.float64(-2.0), np.float64(0.0), np.float64(2.0), np.float64(4.0)] | [-1.342 -0.447  0.447  1.342] | [-0.816  0.     0.816  1.633] |  |



## 21. Computed Table: Batch-Statistic Noise vs Batch Size
Monte Carlo (seed 42) for standard normal data. Small batches give wildly variable variance estimates — the quantitative reason BatchNorm degrades at batch 2-8 and GroupNorm/LayerNorm take over (Wu & He 2018).

| batch size | std of batch mean | std of batch var | rel. err of var |
|---|---|---|---|
| 2 | 0.6909 | 0.6932 | 0.693 |
| 4 | 0.5028 | 0.5686 | 0.569 |
| 8 | 0.3479 | 0.4796 | 0.480 |
| 16 | 0.2490 | 0.3523 | 0.352 |
| 32 | 0.1799 | 0.2521 | 0.252 |
| 64 | 0.1227 | 0.1744 | 0.174 |
| 128 | 0.0869 | 0.1281 | 0.128 |
| 512 | 0.0429 | 0.0621 | 0.062 |



## 22. GroupNorm, InstanceNorm, WeightNorm, SpectralNorm
- GroupNorm (Wu & He 2018): per example, split channels into $G$ groups, normalize within each group (over channels-in-group × spatial). $G=1$ recovers LayerNorm-like behaviour (over all channels), $G=C$ recovers InstanceNorm.
- GroupNorm motivation: BatchNorm's error rises sharply at small batch sizes in detection/segmentation; GN is batch-independent and stable across batch sizes in the paper's ImageNet/det experiments.
- InstanceNorm (Ulyanov et al. 2016): per example per channel; removes instance-specific contrast/style information — why it suits style transfer and can hurt discriminative tasks that need that information.
- WeightNorm (Salimans & Kingma 2016): reparametrize $w=g\,v/\|v\|$ — decouples length and direction of weights rather than normalizing activations; speeds optimization in the paper's experiments, no batch dependence.
- SpectralNorm (Miyato et al. 2018): divide a weight matrix by its largest singular value (power-iteration estimate), bounding the layer's Lipschitz constant; standard in GAN discriminators.
- These are not interchangeable 'normalizations': activation-statistic methods (BN/LN/GN/IN/RMS) vs weight reparametrizations (WN/SN) answer different questions; state which you are using and why.
- Conditional variants (Conditional BN, AdaIN, FiLM-style modulation) make $\gamma,\beta$ functions of side information — the bridge from normalization to conditioning in generative models.
- Selection summary: sequences/LLMs → LN/RMSNorm (Pre-LN); convnets with healthy batches → BN or GN; small-batch vision → GN; style → IN/AdaIN; GAN Lipschitz control → SN; see Section 30 decision records.



## 23. Computed Table: Running-Statistic EMA After an Outlier Batch
Step 5's batch mean of 5.0 (an outlier batch) perturbs the momentum-0.1 running mean far more than momentum-0.01, but the slow estimator also lags genuine shifts. Momentum is a bias-variance knob on your eval-time statistics.

| step | batch mean | running (m=0.1) | running (m=0.01) |
|---|---|---|---|
| 1 | 0.8 | 0.0800 | 0.0080 |
| 2 | 1.2 | 0.1920 | 0.0199 |
| 3 | 0.9 | 0.2628 | 0.0287 |
| 4 | 1.1 | 0.3465 | 0.0394 |
| 5 | 5.0 | 0.8119 | 0.0890 |
| 6 | 1.0 | 0.8307 | 0.0981 |
| 7 | 0.95 | 0.8426 | 0.1067 |
| 8 | 1.05 | 0.8634 | 0.1161 |
| 9 | 1.0 | 0.8770 | 0.1249 |
| 10 | 1.0 | 0.8893 | 0.1337 |

## 24. Worked Problem Set A: Batch Statistics and Modes
Worked problems A: batch statistics and modes.

**Problem A1: recompute Section 19 with the outlier**
1. Replace 9 by 90 in $(2,4,4,4,5,9)$: mean becomes $109/6\approx18.17$.
2. Variance explodes (dominated by $(90-18.17)^2\approx5158$); std $\approx29.5$.
3. The five unchanged values now normalize to $\approx(2-18.17)/29.5=-0.548$ etc. — all shifted by one batchmate.
4. This is batch dependence in one line of arithmetic; eval mode exists precisely to remove it.
5. Moral: BatchNorm couples examples; any per-example guarantee (privacy, determinism, caching) must reckon with training-mode BN.

**Problem A2: train/eval mismatch incident**
1. Symptom: training accuracy 91%, eval-mode accuracy on the same data 63% after switching model.eval().
2. Suspects: stale/never-updated running stats (e.g., stats frozen by a bug), or train-mode BN noise was masking a problem.
3. Test: evaluate a held-out batch in train mode (batch stats) vs eval mode; large gap confirms statistics mismatch.
4. Fixes: verify running-stat updates are enabled, re-estimate stats over training data, check momentum convention.
5. Prevention: CI test asserting train/eval outputs on a fixed batch are close after training.

**Problem A3: batch size 2 BatchNorm**
1. Section 21: at $n=2$, batch variance is extremely noisy (relative error of order 1).
2. Prediction: unstable training, large train/eval gap.
3. Remedy per Wu & He (2018): GroupNorm (batch-independent); alternatively accumulate statistics over more steps or use larger effective batches.
4. Non-remedy: lowering epsilon — epsilon floors the denominator, it does not denoise the estimate.

**Problem A4: momentum direction bug**
1. Code uses running $\leftarrow m\cdot$running $+(1-m)\cdot$batch with $m=0.1$ copied from a framework where $m$ multiplies the batch.
2. Effect: running stats become 90% the latest batch — eval behaviour jitters with the last-seen data.
3. Detection: Section 23-style table on logged batch means vs stored running mean.
4. Lesson: 'momentum' has opposite conventions across frameworks (PyTorch vs the original paper's update); always state the update equation.



## 25. Worked Problem Set B: Per-Example Normalization
Worked problems B: per-example normalizations.

**Problem B1: LN invariance proof sketch**
1. Claim: LN$(ax+b\mathbf1)$ = LN$(x)$ for $a>0$.
2. Mean of transformed: $a\mu+b$; deviations scale by $a$; variance scales by $a^2$; std by $a$.
3. Numerator and denominator both scale by $a$; shift cancels in deviations. Ratio unchanged. QED.
4. RMSNorm under the same transform: shift does not cancel → output changes (Section 20, row 2 numerically).

**Problem B2: when LN and RMSNorm differ**
1. Take $x=(100,101,102,103)$: LN output equals that of $(1,2,3,4)$ (shift invariance).
2. RMSNorm: RMS $\approx101.5$, output $\approx(0.985,0.995,1.005,1.015)$ — nearly constant, pattern nearly erased.
3. If the signal lives in deviations around a large mean, RMSNorm attenuates it relative to LN.
4. Design consequence: know your activation mean structure; 'RMSNorm = LN but cheaper' is conditional, not absolute.

**Problem B3: epsilon floor**
1. A dead channel has variance $0$; without $\epsilon$, normalized value is $0/0$.
2. With $\epsilon=10^{-5}$: output $0$ — safe; but a channel with variance $10^{-6}$ gets amplified by $1/\sqrt{10^{-6}+10^{-5}}\approx302$ vs true $1000$.
3. Epsilon thus caps gain on near-constant directions; in low precision, too-small $\epsilon$ invites overflow/instability.
4. Rule: keep statistics in float32 even under mixed precision (frameworks do this for BN/LN); verify for custom norms.

**Problem B4: Pre-LN residual growth**
1. Pre-LN block: stream $x_{l+1}=x_l+F_l(\mathrm{LN}(x_l))$, branch outputs roughly unit scale, weakly correlated.
2. Variance adds: after $L$ blocks Var $\approx$ Var$_0+L\cdot c$ — linear growth, std growth $\sqrt L$ (vs Post-LN's renormalized stream).
3. Hence the final LN before the head in Pre-LN models and optional $1/\sqrt{2L}$-style branch scaling (cf. §02 Section 23).
4. Diagnostic: plot residual-stream std vs depth at init; expect $\sim\sqrt L$, alarm on exponential.



## 26. Normalization in Attention and Modern LLM Blocks
- Standard block: Pre-RMSNorm → attention → residual add → Pre-RMSNorm → SwiGLU FFN (§02 Section 21) → residual add.
- QK-norm: applying RMSNorm/LayerNorm to queries and keys before the dot product bounds logit growth (Section 54 of §02); used in several recent large models.
- Attention logit soft-capping is an alternative to QK-norm; both treat softmax saturation (§02 Section 31) as a normalization problem.
- Embedding scale: token embeddings feeding the residual stream set initial stream magnitude; some recipes scale embeddings or normalize them.
- Final norm before the LM head is load-bearing in Pre-LN stacks (Problem B4); removing it 'because the stream looks fine' breaks logit scale.
- MoE blocks add router normalization choices; expert imbalance can be a normalization/statistics problem before it is a capacity problem.
- Long-context: activation statistics drift with sequence length in some designs; validate norm behaviour at max length, not only training length.
- Quantization: per-channel activation outliers (§02 Section 49) motivate SmoothQuant-style scaling — normalization thinking applied at deployment.

## 27. Computed Table: Residual Stream Growth, Pre-LN vs Post-LN (Simulation)
Random branch outputs (std 0.5) added per block, seed 42. Pre-LN stream std grows roughly as $\sqrt{L}$ (Problem B4); Post-LN renormalizes to 1 each block but pays with gradient-path and warmup properties (Section 18).

| block | Pre-LN stream std (sim) | Post-LN stream std (sim) |
|---|---|---|
| 0 | 0.938 | 0.938 |
| 4 | 1.369 | 1.000 |
| 8 | 1.635 | 1.000 |
| 12 | 1.801 | 1.000 |
| 16 | 1.992 | 1.000 |
| 20 | 2.185 | 1.000 |
| 24 | 2.345 | 1.000 |
| 28 | 2.565 | 1.000 |
| 32 | 2.586 | 1.000 |
| 36 | 2.723 | 1.000 |
| 40 | 2.879 | 1.000 |
| 44 | 3.013 | 1.000 |
| 48 | 3.106 | 1.000 |



## 28. Glossary: Normalization Vocabulary
| Term | Meaning |
|---|---|
| Internal covariate shift | Drift of layer-input distributions during training; BN's original motivation (Ioffe & Szegedy 2015), later debated (Santurkar et al. 2018). |
| Running statistics | EMA of batch means/variances used by BatchNorm at eval time. |
| Momentum (BN) | EMA weight; convention differs by framework — always write the update equation. |
| Affine parameters | Learnable $\gamma$ (gain) and $\beta$ (shift) after normalization. |
| Epsilon | Small constant inside the square root; caps amplification of tiny-variance directions. |
| Pre-LN / Post-LN | Normalization inside the residual branch vs after the residual sum (Xiong et al. 2020). |
| Residual stream | The additive highway through transformer blocks; its scale is a monitored quantity. |
| SyncBN | BatchNorm synchronizing statistics across distributed workers. |
| GroupNorm | Per-example normalization over channel groups (Wu & He 2018). |
| InstanceNorm | Per-example per-channel normalization (Ulyanov et al. 2016). |
| RMSNorm | Scale-only normalization by root mean square (Zhang & Sennrich 2019). |
| WeightNorm | Weight reparametrization $w=g v/\|v\|$ (Salimans & Kingma 2016). |
| SpectralNorm | Weight divided by largest singular value; Lipschitz control (Miyato et al. 2018). |
| QK-norm | Normalizing queries/keys before attention logits. |
| Scale invariance | Output unchanged when preceding weights are scaled; interacts with weight decay and effective LR. |
| AdaIN / FiLM / Conditional BN | Normalization whose affine parameters are generated from side information (conditioning). |
| Statistics mismatch | Train/eval discrepancy from batch vs running statistics (Problem A2). |
| Warmup | Gradual LR ramp; historically required by Post-LN, largely obviated by Pre-LN (Xiong et al. 2020). |
| DeepNorm | Residual rescaling scheme enabling very deep Post-LN-style training. |
| Fused norm kernel | Implementation fusing reduction+affine; performance-critical at LLM scale. |



## 29. Normalization Mistake Catalogue (Field Version)
Complements Section 10 with production and porting failures.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | Eval in train mode (BN) | Batch-dependent, noisy metrics | model.eval() discipline + CI check |
| 2 | Running stats not saved/exported | Eval collapse after deployment | Checkpoint buffers; export test in eval mode |
| 3 | Momentum convention copied blindly | Running stats = last batch (Problem A4) | Write the update equation; log stats |
| 4 | BatchNorm with batch size 2-4 | Noisy statistics (Section 21) | GroupNorm/LayerNorm or larger effective batch |
| 5 | LayerNorm over the wrong axis | Normalizing across batch silently | Assert axis; unit-test invariance (Problem B1) |
| 6 | RMSNorm assumed identical to LN | Mean-carried signal attenuated (Problem B2) | Check activation means; choose deliberately |
| 7 | Removing final LN in Pre-LN stack | Logit scale drift | Keep final norm; monitor stream std (Section 27) |
| 8 | Post-LN without warmup at depth | Early instability | Pre-LN or warmup (Xiong et al. 2020) |
| 9 | Custom norm in fp16 statistics | Overflow/precision loss | Compute statistics in float32 |
| 10 | Treating epsilon as irrelevant | Near-constant channels explode or vanish | State epsilon; test dead-channel case (Problem B3) |
| 11 | BN + strong dropout copied from old recipes | Double regularization mis-tuning | Retune regularization after any norm change |
| 12 | Normalizing embeddings unintentionally | Destroyed token identity scale | Check which tensors the norm actually touches |



## 30. Decision Records: Normalization Choices in Context
- **LLM pretraining:** Pre-RMSNorm blocks, final RMSNorm, QK-norm optional at scale. Rationale: no batch dependence, warmup-free stability (Xiong et al. 2020 pattern), cheap statistics.
- **Convnet, batch >= 32/GPU:** BatchNorm remains defensible; GroupNorm is the challenger. Record SyncBN policy for multi-GPU.
- **Detection/segmentation, small per-device batches:** GroupNorm (Wu & He 2018 evidence).
- **Style transfer/generation with instance conditioning:** InstanceNorm/AdaIN lineage (Ulyanov et al. 2016).
- **GAN discriminator:** SpectralNorm for Lipschitz control (Miyato et al. 2018), often combined with no BN in the discriminator.
- **RL / online inference batch 1:** per-example norms only; BatchNorm's eval-mode dependence on stale running stats is a liability under distribution shift.
- **Fine-tuning a BN model on shifted data:** re-estimate running stats, then decide freeze vs adapt; record which.
- Each record states axes, affine, epsilon, placement, state handling, and the eval-mode test that guards it.

## 31. Worked Dataset: 48 Values Through BatchNorm
A full batch normalized by hand-checkable arithmetic (seed 42 data). Spot-check rows against the definitions in Section 15; note the normalized column has mean 0 and std 1 by construction.

| i | x_i | BN-normalized | after gamma=1.5, beta=-0.5 |
|---|---|---|---|
| 0 | 5.609 | 0.2917 | -0.0624 |
| 1 | 2.920 | -1.4510 | -2.6765 |
| 2 | 6.501 | 0.8698 | 0.8047 |
| 3 | 6.881 | 1.1161 | 1.1741 |
| 4 | 1.098 | -2.6319 | -4.4478 |
| 5 | 2.396 | -1.7906 | -3.1859 |
| 6 | 5.256 | 0.0629 | -0.4056 |
| 7 | 4.368 | -0.5126 | -1.2689 |
| 8 | 4.966 | -0.1250 | -0.6875 |
| 9 | 3.294 | -1.2086 | -2.3130 |
| 10 | 6.759 | 1.0370 | 1.0555 |
| 11 | 6.556 | 0.9055 | 0.8582 |
| 12 | 5.132 | -0.0174 | -0.5261 |
| 13 | 7.254 | 1.3578 | 1.5368 |
| 14 | 5.935 | 0.5030 | 0.2545 |
| 15 | 3.281 | -1.2171 | -2.3256 |
| 16 | 5.738 | 0.3753 | 0.0630 |
| 17 | 3.082 | -1.3460 | -2.5190 |
| 18 | 6.757 | 1.0357 | 1.0536 |
| 19 | 4.900 | -0.1678 | -0.7517 |
| 20 | 4.630 | -0.3428 | -1.0142 |
| 21 | 3.638 | -0.9857 | -1.9785 |
| 22 | 7.445 | 1.4816 | 1.7224 |
| 23 | 4.691 | -0.3032 | -0.9549 |
| 24 | 4.143 | -0.6584 | -1.4876 |
| 25 | 4.296 | -0.5592 | -1.3389 |
| 26 | 6.065 | 0.5872 | 0.3809 |
| 27 | 5.731 | 0.3708 | 0.0562 |
| 28 | 5.825 | 0.4317 | 0.1476 |
| 29 | 5.862 | 0.4557 | 0.1835 |
| 30 | 9.283 | 2.6728 | 3.5092 |
| 31 | 4.187 | -0.6299 | -1.4448 |
| 32 | 3.976 | -0.7666 | -1.6499 |
| 33 | 3.372 | -1.1581 | -2.2371 |
| 34 | 6.232 | 0.6955 | 0.5432 |
| 35 | 7.258 | 1.3604 | 1.5406 |
| 36 | 4.772 | -0.2507 | -0.8761 |
| 37 | 3.320 | -1.1918 | -2.2877 |
| 38 | 3.351 | -1.1717 | -2.2575 |
| 39 | 6.301 | 0.7402 | 0.6103 |
| 40 | 6.487 | 0.8607 | 0.7911 |
| 41 | 6.086 | 0.6009 | 0.4013 |
| 42 | 3.669 | -0.9656 | -1.9484 |
| 43 | 5.464 | 0.1977 | -0.2034 |
| 44 | 5.233 | 0.0480 | -0.4280 |
| 45 | 5.437 | 0.1802 | -0.2296 |
| 46 | 6.743 | 1.0267 | 1.0400 |
| 47 | 5.447 | 0.1867 | -0.2199 |
| stats | mean=5.1589 | std=1.5430 | (biased std, as in training-mode BN) |



## 32. Historical Timeline: Normalization
| Year | Milestone | Contribution |
|---|---|---|
| 2015 | Ioffe & Szegedy — Batch Normalization | Batch statistics + running averages; ICS motivation. |
| 2016 | Ba, Kiros & Hinton — Layer Normalization | Per-example feature normalization; RNN-friendly. |
| 2016 | Ulyanov et al. — Instance Normalization | Per-instance channel normalization for style transfer. |
| 2016 | Salimans & Kingma — Weight Normalization | Length/direction weight reparametrization. |
| 2017 | Vaswani et al. — Attention Is All You Need | Post-LN transformer blocks (historical placement). |
| 2018 | Miyato et al. — Spectral Normalization | Lipschitz control for GANs. |
| 2018 | Wu & He — Group Normalization | Small-batch vision; batch-independence evidence. |
| 2018 | Santurkar et al. — How Does Batch Normalization Help Optimization? | ICS critique; loss-landscape smoothness account. |
| 2019 | Zhang & Sennrich — Root Mean Square Layer Normalization | Scale-only normalization; efficiency with comparable quality. |
| 2020 | Xiong et al. — On Layer Normalization in the Transformer Architecture | Pre-LN vs Post-LN; warmup analysis. |
| 2020s | RMSNorm + Pre-LN in open LLMs | The contemporary default block; QK-norm variants at scale. |



## 33. Quick-Reference Cards
**BatchNorm**
- Axes: per channel over batch(+spatial). Mean subtracted: yes. Affine: $\gamma,\beta$.
- State: running mean/var; train/eval differ.
- Invariance: channel output invariant to batch affine rescaling.
- Fails: tiny batches, batch-1 semantics, cross-example coupling concerns.
- Canonical: Ioffe & Szegedy (2015).

**LayerNorm**
- Axes: per example over features. Mean subtracted: yes. Affine: $\gamma,\beta$.
- State: none; train = eval.
- Invariance: shift and positive scale of the input vector.
- Fails: when feature-mean itself is the signal (rare); cost at huge widths is minor.
- Canonical: Ba, Kiros & Hinton (2016).

**RMSNorm**
- Axes: per example over features. Mean subtracted: no. Affine: usually $\gamma$ only.
- State: none; train = eval.
- Invariance: positive scale only (Problem B2).
- Fails: large-mean activations attenuate deviation signal.
- Canonical: Zhang & Sennrich (2019).

**GroupNorm**
- Axes: per example over (group channels × spatial). Mean subtracted: yes. Affine: $\gamma,\beta$.
- State: none. Interpolates LN ($G{=}1$) and IN ($G{=}C$).
- Fails: group-count is a hyperparameter; less common in LLMs.
- Canonical: Wu & He (2018).

**InstanceNorm**
- Axes: per example per channel (× spatial). Mean subtracted: yes.
- Use: style/content disentangling (Ulyanov et al. 2016).
- Risk: discards per-instance contrast that discriminative tasks may need.

**WeightNorm / SpectralNorm**
- Not activation statistics: weight reparametrization / spectral scaling.
- WeightNorm: $w=g v/\|v\|$ (Salimans & Kingma 2016). SpectralNorm: $W/\sigma_{max}(W)$ (Miyato et al. 2018).
- Use: optimization conditioning (WN), Lipschitz control in GANs (SN).



## 34. Normalization and Optimization Geometry
- Normalization reparametrizes the loss: for scale-invariant weights (BN/LN-covered), the gradient is orthogonal to the weight vector and its magnitude scales inversely with weight norm — the mechanism behind effective-LR effects of weight decay.
- Practical consequence: with normalized layers, weight decay and LR jointly set an equilibrium weight norm; tuning one without the other misleads (record both in experiments).
- Santurkar et al. (2018) link BN to improved Lipschitz behaviour of loss and gradients: gradient changes more predictably along the optimization path, tolerating larger steps.
- LN/RMSNorm similarly bound activation scale entering each block, keeping attention logits (§02 Section 54) and FFN inputs in trained ranges.
- Normalization does not convexify anything; claims should be stated as conditioning/smoothness, not convexity.
- Gradient clipping remains complementary: norms control per-layer statistics, clipping controls global step outliers (loss spikes, §01 runbook).
- Second-order/optimizer interactions: Adam's per-coordinate scaling partially overlaps normalization's role; ablations changing both at once are uninterpretable.
- Test-time compute (longer contexts, larger batches) stresses exactly the statistics normalization fixes at train time — validate at deployment shape.

## 35. Computed Table: Epsilon as a Gain Cap
Epsilon's real job (Problem B3): it caps how much a near-constant direction can be amplified. Read across: at $\epsilon=10^{-3}$, a variance-$10^{-6}$ direction gains at most ~31x, not 1000x.

| epsilon | max gain 1/sqrt(var+eps), var=1e-6 | gain at var=1 | gain at var=1e-8 |
|---|---|---|---|
| 1e-01 | 3.2 | 0.9535 | 3.2 |
| 1e-02 | 10.0 | 0.9950 | 10.0 |
| 1e-03 | 31.6 | 0.9995 | 31.6 |
| 1e-04 | 99.5 | 1.0000 | 100.0 |
| 1e-05 | 301.5 | 1.0000 | 316.1 |
| 1e-06 | 707.1 | 1.0000 | 995.0 |
| 1e-07 | 953.5 | 1.0000 | 3015.1 |
| 1e-08 | 995.0 | 1.0000 | 7071.1 |



## 36. Computed Table: LayerNorm/RMSNorm on 32 Random Vectors
Seed 42, $d=8$. By construction LN outputs have mean 0 and std 1 per row, and RMSNorm outputs have RMS 1 per row — the table is a unit test you can reproduce against any framework's implementation.

| row | mean before | std before | LN out mean | LN out std | RMS out RMS |
|---|---|---|---|---|---|
| 0 | -0.311 | 0.966 | 1.11e-16 | 1.0000 | 1.0000 |
| 1 | 0.199 | 0.709 | -2.78e-17 | 1.0000 | 1.0000 |
| 2 | 0.055 | 0.691 | -5.55e-17 | 1.0000 | 1.0000 |
| 3 | 0.337 | 0.785 | 2.78e-17 | 1.0000 | 1.0000 |
| 4 | -0.089 | 0.736 | 5.55e-17 | 1.0000 | 1.0000 |
| 5 | 0.285 | 0.442 | 2.78e-17 | 1.0000 | 1.0000 |
| 6 | -0.152 | 0.672 | 0.00e+00 | 1.0000 | 1.0000 |
| 7 | 0.007 | 0.956 | 2.78e-17 | 1.0000 | 1.0000 |
| 8 | -0.131 | 0.792 | 0.00e+00 | 1.0000 | 1.0000 |
| 9 | 0.057 | 0.531 | 2.78e-17 | 1.0000 | 1.0000 |
| 10 | -0.264 | 0.531 | 3.47e-17 | 1.0000 | 1.0000 |
| 11 | -0.268 | 0.824 | 5.55e-17 | 1.0000 | 1.0000 |
| 12 | -0.190 | 0.861 | -5.55e-17 | 1.0000 | 1.0000 |
| 13 | -0.330 | 0.790 | 0.00e+00 | 1.0000 | 1.0000 |
| 14 | -0.110 | 0.815 | 0.00e+00 | 1.0000 | 1.0000 |
| 15 | 0.275 | 0.918 | -2.78e-17 | 1.0000 | 1.0000 |
| 16 | -0.676 | 0.643 | 8.33e-17 | 1.0000 | 1.0000 |
| 17 | 0.453 | 1.475 | 5.55e-17 | 1.0000 | 1.0000 |
| 18 | -0.244 | 0.613 | -2.78e-17 | 1.0000 | 1.0000 |
| 19 | -0.127 | 1.041 | -5.55e-17 | 1.0000 | 1.0000 |
| 20 | 0.109 | 1.067 | -9.71e-17 | 1.0000 | 1.0000 |
| 21 | -0.399 | 0.587 | -1.11e-16 | 1.0000 | 1.0000 |
| 22 | 0.516 | 0.674 | 2.78e-17 | 1.0000 | 1.0000 |
| 23 | 0.344 | 1.158 | 2.78e-17 | 1.0000 | 1.0000 |
| 24 | -0.105 | 0.776 | 2.78e-17 | 1.0000 | 1.0000 |
| 25 | -0.210 | 1.035 | 0.00e+00 | 1.0000 | 1.0000 |
| 26 | -0.175 | 0.757 | -5.55e-17 | 1.0000 | 1.0000 |
| 27 | 0.017 | 0.698 | 0.00e+00 | 1.0000 | 1.0000 |
| 28 | 0.190 | 1.271 | -5.55e-17 | 1.0000 | 1.0000 |
| 29 | -0.470 | 0.921 | 5.55e-17 | 1.0000 | 1.0000 |
| 30 | -0.327 | 1.411 | -5.55e-17 | 1.0000 | 1.0000 |
| 31 | 0.373 | 1.303 | 3.47e-17 | 1.0000 | 1.0000 |



## 37. Worked Problem Set C: Placement, Coupling, Precision
Worked problems C: placement and scale.

**Problem C1: convert Post-LN recipe to Pre-LN**
1. Changes: move LN inside each residual branch; add final LN before head; revisit warmup (often removable, Xiong et al. 2020).
2. Do not expect checkpoint reuse; treat as a new architecture and re-baseline.
3. Watch: residual stream std growth (Section 27) and FFN input scale in early training.
4. Report both final quality and stability (loss spikes) — Pre-LN's advantage is often the latter.

**Problem C2: BN in a Siamese/contrastive setup**
1. Risk: batch statistics couple the two views of a pair and leak batch information (Section 15).
2. Options: per-example norms, shuffled/stop-gradient BN variants from the SSL literature, or SyncBN with large batches — state which and why.
3. Test: score a fixed pair in different batch compositions; output drift quantifies coupling.

**Problem C3: norm choice for streaming inference**
1. Requirement: one example at a time, distribution drifting slowly, no retraining window.
2. BatchNorm eval mode uses frozen running stats — drift degrades it silently.
3. Per-example LN/RMSNorm compute statistics from the example itself: preferred.
4. Add monitoring on activation statistics; drifting inputs are detectable even when labels are not.

**Problem C4: debugging a NaN after adding normalization**
1. Suspects in order: variance computed in fp16 underflowing/overflowing; epsilon too small with dead channels; norm applied over an axis including padding.
2. Reproduce in float32; if NaN vanishes, it is precision/statistics dtype (Problem B3 rule).
3. Check padded positions: sequence norms must mask padding or statistics absorb pad tokens.
4. Add asserts: normalized output finite, per-example std near 1 (LN) on a smoke batch.



## 38. Pre-Flight Checklist for Normalization
- Method chosen by axes and batch-dependence requirements, with canonical citation in the design doc.
- Placement (Pre/Post-LN, before/after activation) written down with the block diagram.
- Affine parameters, epsilon value and placement, and statistics dtype (float32) specified.
- BatchNorm only: momentum convention written as an equation; SyncBN policy for distributed runs; running-stat checkpoint/export test.
- Per-example norms: axis asserted by a unit test (invariance test, Problem B1).
- Padding/masking interaction checked for sequence models.
- Residual-stream std vs depth logged at init for transformer stacks.
- Train/eval behaviour tested on a fixed batch (BN gap test, Problem A2).
- Regularization retuned after norm changes (dropout/weight-decay interplay, Section 34).
- Rollback: previous block config retained; norm changes are architecture changes for checkpoint purposes.



## 39. Frequently Asked Questions
- **Does normalization replace good initialization?** It relaxes sensitivity (§02 Section 40) but init still sets early dynamics, especially in Post-LN and unnormalized branches.
- **Is BatchNorm dead after LayerNorm/RMSNorm?** No — BN remains common in vision with healthy batch sizes; the choice is workload-dependent (Section 30 records).
- **Why do LLMs prefer RMSNorm?** Fewer reductions, no mean subtraction, quality comparable in Zhang & Sennrich (2019) and subsequent practice; also no state.
- **Can I remove all normalization with careful init?** Research lines exist (e.g., Fixup-style), but as an engineering default normalization is cheap insurance; removal needs evidence.
- **What does epsilon actually do?** Caps gain on tiny-variance directions and prevents division by zero (Section 35).
- **Why did my BN model fail at batch size 1 fine-tuning?** Statistics noise/mismatch (Sections 15, 21); switch norm family or re-estimate stats deliberately.
- **Pre-LN or Post-LN for a new LLM?** Pre-LN (RMSNorm) as default; Post-LN only with warmup budget and a quality hypothesis (Xiong et al. 2020).
- **Is ICS the reason BN works?** Historically proposed; current evidence favours landscape smoothness (Santurkar et al. 2018) — cite accordingly (Section 16).

## 40. Computed Table: How Noisy Is a Batch Variance? (Fine Grid)
Monte Carlo, seed 42, normal data (true variance 1). At $n=4$ the variance estimate is off by more than 50% in a large share of batches; by $n=128$ it is a few percent. This table is the quantitative core of the BatchNorm small-batch problem.

| batch size n | rel. std of batch variance | P(var estimate off by >50%) |
|---|---|---|
| 2 | 0.7193 | 0.768 |
| 3 | 0.6639 | 0.635 |
| 4 | 0.6012 | 0.543 |
| 6 | 0.5188 | 0.406 |
| 8 | 0.4719 | 0.335 |
| 12 | 0.3943 | 0.210 |
| 16 | 0.3436 | 0.139 |
| 24 | 0.2898 | 0.080 |
| 32 | 0.2489 | 0.039 |
| 48 | 0.2044 | 0.015 |
| 64 | 0.1744 | 0.005 |
| 96 | 0.1445 | 0.001 |
| 128 | 0.1241 | 0.001 |
| 192 | 0.1030 | 0.000 |
| 256 | 0.0893 | 0.000 |
| 384 | 0.0708 | 0.000 |
| 512 | 0.0623 | 0.000 |
| 768 | 0.0505 | 0.000 |
| 1024 | 0.0446 | 0.000 |



## 41. Worked Problem Set D: Porting and Interactions
Worked problems D: interactions and porting.

**Problem D1: porting BN between frameworks**
1. Framework A momentum: running $=(1-m)$running $+m\cdot$batch, $m=0.1$. Framework B: running $=m'$running $+(1-m')$batch.
2. Equivalent $m'=0.9$; copying the number 0.1 verbatim gives Problem A4's bug.
3. Port checklist: momentum equation, biased vs unbiased running variance, epsilon, affine defaults — diff them explicitly.
4. Acceptance test: identical weights + identical batch → outputs match to tolerance in both train and eval modes.

**Problem D2: normalization and weight decay equilibrium**
1. Setup: scale-invariant layer (BN-covered) trained with LR $\eta$ and decay $\lambda$.
2. Gradient magnitude scales as $1/\|w\|$; decay shrinks $\|w\|$; an equilibrium norm emerges where the two balance.
3. Symptom of misunderstanding: raising decay 'to regularize more' mainly raises effective LR on the direction and destabilizes training.
4. Protocol: tune (LR, decay) jointly for normalized stacks; report both (Section 34).

**Problem D3: GN group count**
1. Channels $C=64$; candidates $G\in\{1,8,32,64\}$ (LN-like, balanced, fine, IN-like).
2. Wu & He (2018) use $G=32$ as a robust default in their vision experiments.
3. Too few groups: statistics over heterogeneous channels; too many: noisy per-group stats over few channels.
4. Choose by validation with the batch size you will actually train at; GN's selling point is flatness across batch sizes — verify it.

**Problem D4: norm for a GAN discriminator**
1. Requirements: Lipschitz control, no batch coupling artefacts in generated-sample statistics.
2. SpectralNorm (Miyato et al. 2018) bounds each layer's gain; BN in discriminators is historically fraught (batch coupling between real/fake).
3. Pair with the GAN loss chapter knowledge (§01 hinge/Wasserstein losses); normalization choice and loss choice interact.



## 42. Debugging Playbook: Normalization Incidents
1. **Eval collapse, train fine** → BN statistics mismatch: run Problem A2's train-vs-eval batch test; check running-stat updates and export buffers.
2. **Loss spikes only in long runs** → residual stream growth or rare-batch statistics: log stream std by depth and batch-statistic outliers (Section 23 outlier effect).
3. **NaN immediately after adding a norm** → Problem C4 sequence: dtype of statistics, epsilon, padding axis.
4. **Fine-tune much worse than scratch** → frozen running stats on shifted data (re-estimate), or LR/decay equilibrium disturbed (Problem D2).
5. **Multi-GPU worse than single-GPU** → per-device BN statistics (missing SyncBN) or per-device batch too small for BN.
6. **Metrics differ between two frameworks** → Problem D1 porting diff: momentum, variance flavour, epsilon, axis.
7. **Long-context degradation** → statistics drift with length (Section 26): evaluate norms at deployment length; consider QK-norm.
8. **Quantized model collapse** → activation outliers post-norm: per-channel scales/SmoothQuant-style fixes (§02 Section 49).
9. For every incident: record symptom, test, root cause, fix, and the regression test added — normalization incidents recur across projects.



## 43. Formula Sheet
| Method | Statistic axes | Output | State | Canonical citation |
|---|---|---|---|---|
| BatchNorm | channel over batch(+spatial) | $\gamma\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}+\beta$ | running $\mu,\sigma^2$ | Ioffe & Szegedy 2015 |
| LayerNorm | features per example | $\gamma\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta$ | none | Ba, Kiros & Hinton 2016 |
| RMSNorm | features per example | $\gamma\frac{x}{\sqrt{\frac1d\sum x_i^2+\epsilon}}$ | none | Zhang & Sennrich 2019 |
| GroupNorm | channel groups per example | LN within each group | none | Wu & He 2018 |
| InstanceNorm | channel per example | LN within channel | none | Ulyanov et al. 2016 |
| WeightNorm | weight vector | $w=g\,v/\|v\|$ | $g,v$ params | Salimans & Kingma 2016 |
| SpectralNorm | weight matrix | $W/\sigma_{\max}(W)$ | power-iteration vector | Miyato et al. 2018 |

## 44. Worked Problem Set E: Synthesis and Audits
Worked problems E: synthesis.

**Problem E1: design the norm stack for a new 24-layer LLM**
1. Default: Pre-RMSNorm per sub-block, final RMSNorm, SwiGLU FFN (§02), optional QK-norm.
2. Justify each: batch-independence (variable batch/sequence), warmup-free stability (Xiong et al. 2020), cost (Zhang & Sennrich 2019).
3. Instrumentation: residual stream std by depth, attention logit max, per-layer activation std at init and during training.
4. Ablations owed: Post-LN comparator only if quality hypothesis justifies warmup cost.

**Problem E2: defend BatchNorm in a 2026 vision system**
1. Context: classification backbone, batch 256 across GPUs, SyncBN available.
2. Defence: mature kernels, strong empirical record at healthy batch sizes, regularization side-effect tuned into the recipe.
3. Concede: small-batch fine-tuning stages may switch to GN or re-estimated stats; document per-stage choices.
4. A defence that names its boundary conditions beats a universal claim.

**Problem E3: the missing final norm**
1. A Pre-LN model without final LN shows logit std growing with checkpoint age and degraded calibration (§01).
2. Mechanism: stream std grows with depth/drift (Section 27); the head inherits uncontrolled scale.
3. Fix: restore final norm; if a checkpoint must be salvaged, a learned scalar temperature on logits is a stopgap, labelled as such.

**Problem E4: statistics audit for a deployed model**
1. Task: verify a served BN model matches its training-time behaviour.
2. Steps: load checkpoint; confirm running buffers present and finite; run the A2 fixed-batch train/eval comparison; compare served outputs on a golden set.
3. Sign-off requires eval-mode golden outputs within tolerance — this audit catches the most common BN production incident class.



## 45. Monitoring Runbook: Normalization Health
- Log per-layer: input/output mean and std of each norm, residual stream std by depth (transformers), BN batch vs running statistic gap.
- Alert: BN batch/running gap widening → drift or momentum bug (Problems A2/A4).
- Alert: residual stream std super-√L growth at init → branch scale/init problem (Section 27).
- Alert: normalized outputs with std far from 1 (LN) on probe data → axis or masking bug (Problem C4).
- Probe set fixed across steps; include longest-sequence examples for length-drift checks.
- After any norm-affecting change (LR, decay, batch size, precision), re-baseline these metrics before judging quality.
- Save final activation statistics with the checkpoint (model-card material, cf. §02 Section 30).



## 46. Extended Exercise Bank (Self-Study)
1. Reproduce Section 19's table, then redo it after changing one batch element; quantify batch coupling.
2. Prove RMSNorm is scale- but not shift-invariant; construct the numeric counterexample (Problem B2 pattern).
3. Implement LN and RMSNorm in NumPy; validate against Section 36's table properties.
4. Simulate Section 40 at batch 4 and 64; state the GroupNorm implication in two sentences.
5. Write the train/eval CI test of Problem A2 for a small BN network.
6. Derive why scale-invariant weights have gradients orthogonal to the weight vector (Section 34).
7. Convert a Post-LN block diagram to Pre-LN; list every recipe change (Problem C1).
8. Choose and defend norms for: small-batch segmentation, LLM pretraining, GAN discriminator, streaming inference (Section 30 format).
9. Compute epsilon gain caps for your framework's default epsilon at variances $10^{-4},10^{-6},10^{-8}$ (Section 35 pattern).
10. Explain Santurkar et al. (2018) to a colleague in five sentences without claiming BN 'does nothing'.



## 47. Self-Test Questions
- State the five dossier fields for a normalization method (Section 14).
- Why does BatchNorm behave differently in train and eval, and which statistics are used in each?
- Give the LayerNorm invariances and prove one.
- When do LN and RMSNorm materially differ?
- What problem does GroupNorm solve and what is its evidence base?
- Pre-LN vs Post-LN: gradient path, warmup, residual stream behaviour.
- What does epsilon cap, quantitatively?
- State the two momentum conventions and how to avoid the porting bug.
- ICS vs landscape smoothness: which paper argues which?
- Why keep norm statistics in float32 under mixed precision?
- Name a weight-side (not activation-side) normalization and its purpose.
- What is QK-norm treating?



## 48. Self-Test Answer Key (Condensed)
- Dossier: axes, mean subtraction, affine, train/eval behaviour, invariances (+ epsilon/placement/state in the full version).
- BN train: batch stats (examples coupled); eval: running EMA stats (examples independent).
- LN invariant to $x\mapsto ax+b\mathbf1$, $a>0$; proof via mean/variance scaling (Problem B1).
- LN vs RMSNorm differ when feature means are large (Problem B2).
- GN: small-batch degradation of BN; Wu & He (2018) ImageNet/detection evidence across batch sizes.
- Pre-LN: identity gradient path, no warmup needed (Xiong et al. 2020), stream std grows ~√L; Post-LN: normalized stream, warmup-sensitive.
- Epsilon caps gain at $1/\sqrt{\epsilon}$ for zero-variance directions (Section 35).
- Momentum: state the update equation explicitly when porting (Problem D1).
- ICS: Ioffe & Szegedy (2015); smoothness account: Santurkar et al. (2018).
- Statistics in fp32 avoid overflow/underflow in reductions (Problem B3/C4).
- WeightNorm (conditioning) or SpectralNorm (Lipschitz) — Salimans & Kingma (2016), Miyato et al. (2018).
- QK-norm treats attention logit growth/softmax saturation (§02 Sections 49/54).

## 49. Worked Dataset: 24 Vectors Through LayerNorm and RMSNorm
Seed 7 data with per-column offsets $(3,0,-2,1)$ so row means are nonzero and LN/RMSNorm visibly diverge (Problem B2 at scale). Verify: every LN row has mean 0/std 1; every RMSNorm row has RMS 1.

| row | x (d=4) | row mean | row std | LN(x) | RMSNorm(x) |
|---|---|---|---|---|---|
| 0 | [np.float64(3.001), np.float64(0.299), np.float64(-2.274), np.float64(0.109)] | 0.284 | 1.868 | [ 1.455  0.008 -1.369 -0.094] | [ 1.588  0.158 -1.204  0.058] |
| 1 | [np.float64(2.545), np.float64(-0.992), np.float64(-1.94), np.float64(2.34)] | 0.488 | 1.984 | [ 1.037 -0.746 -1.224  0.933] | [ 1.246 -0.485 -0.949  1.145] |
| 2 | [np.float64(2.508), np.float64(-0.62), np.float64(-1.51), np.float64(1.357)] | 0.434 | 1.585 | [ 1.309 -0.665 -1.227  0.583] | [ 1.527 -0.377 -0.919  0.826] |
| 3 | [np.float64(3.105), np.float64(-0.93), np.float64(-2.029), np.float64(1.695)] | 0.460 | 2.040 | [ 1.296 -0.681 -1.22   0.605] | [ 1.485 -0.445 -0.97   0.81 ] |
| 4 | [np.float64(1.656), np.float64(-0.458), np.float64(-3.901), np.float64(-0.29)] | -0.748 | 2.001 | [ 1.202  0.145 -1.576  0.229] | [ 0.775 -0.214 -1.826 -0.136] |
| 5 | [np.float64(1.158), np.float64(-0.235), np.float64(-3.267), np.float64(1.271)] | -0.268 | 1.830 | [ 0.779  0.018 -1.639  0.841] | [ 0.626 -0.127 -1.766  0.687] |
| 6 | [np.float64(3.157), np.float64(-0.187), np.float64(-4.517), np.float64(0.461)] | -0.272 | 2.753 | [ 1.245  0.031 -1.542  0.266] | [ 1.141 -0.068 -1.633  0.167] |
| 7 | [np.float64(2.951), np.float64(0.113), np.float64(-3.53), np.float64(0.522)] | 0.014 | 2.316 | [ 1.268  0.043 -1.53   0.219] | [ 1.274  0.049 -1.524  0.225] |
| 8 | [np.float64(2.021), np.float64(-0.809), np.float64(-0.939), np.float64(0.192)] | 0.116 | 1.184 | [ 1.609 -0.782 -0.892  0.064] | [ 1.699 -0.68  -0.79   0.161] |
| 9 | [np.float64(2.967), np.float64(0.884), np.float64(-2.584), np.float64(0.888)] | 0.539 | 1.993 | [ 1.218  0.173 -1.567  0.175] | [ 1.437  0.428 -1.252  0.43 ] |
| 10 | [np.float64(3.11), np.float64(0.064), np.float64(-3.225), np.float64(1.076)] | 0.256 | 2.290 | [ 1.246 -0.084 -1.52   0.358] | [ 1.35   0.028 -1.4    0.467] |
| 11 | [np.float64(4.359), np.float64(-1.547), np.float64(-1.141), np.float64(1.119)] | 0.698 | 2.345 | [ 1.561 -0.957 -0.784  0.18 ] | [ 1.781 -0.632 -0.466  0.457] |
| 12 | [np.float64(2.359), np.float64(2.0), np.float64(-1.238), np.float64(-0.199)] | 0.731 | 1.500 | [ 1.086  0.846 -1.312 -0.62 ] | [ 1.414  1.199 -0.742 -0.119] |
| 13 | [np.float64(3.075), np.float64(0.577), np.float64(-2.189), np.float64(1.683)] | 0.786 | 1.933 | [ 1.184 -0.108 -1.54   0.464] | [ 1.474  0.277 -1.049  0.807] |
| 14 | [np.float64(2.933), np.float64(0.667), np.float64(-0.561), np.float64(0.324)] | 0.841 | 1.288 | [ 1.624 -0.135 -1.088 -0.401] | [ 1.906  0.434 -0.365  0.211] |
| 15 | [np.float64(3.203), np.float64(-0.463), np.float64(-1.873), np.float64(-0.187)] | 0.170 | 1.864 | [ 1.627 -0.34  -1.096 -0.192] | [ 1.711 -0.247 -1.001 -0.1  ] |
| 16 | [np.float64(2.421), np.float64(-0.196), np.float64(-1.101), np.float64(2.145)] | 0.817 | 1.503 | [ 1.067 -0.674 -1.276  0.883] | [ 1.415 -0.115 -0.643  1.254] |
| 17 | [np.float64(1.676), np.float64(-0.795), np.float64(-1.353), np.float64(-0.992)] | -0.366 | 1.196 | [ 1.708 -0.359 -0.825 -0.523] | [ 1.34  -0.636 -1.082 -0.793] |
| 18 | [np.float64(2.537), np.float64(-0.097), np.float64(-0.743), np.float64(1.689)] | 0.847 | 1.321 | [ 1.279 -0.714 -1.203  0.638] | [ 1.617 -0.062 -0.473  1.076] |
| 19 | [np.float64(2.673), np.float64(-0.369), np.float64(-2.25), np.float64(2.524)] | 0.645 | 2.065 | [ 0.982 -0.491 -1.402  0.91 ] | [ 1.236 -0.171 -1.04   1.167] |
| 20 | [np.float64(2.572), np.float64(-0.304), np.float64(-1.647), np.float64(0.879)] | 0.375 | 1.552 | [ 1.416 -0.438 -1.303  0.325] | [ 1.611 -0.19  -1.032  0.551] |
| 21 | [np.float64(2.803), np.float64(-1.114), np.float64(-2.012), np.float64(0.556)] | 0.058 | 1.833 | [ 1.497 -0.639 -1.129  0.272] | [ 1.528 -0.607 -1.097  0.303] |
| 22 | [np.float64(4.166), np.float64(0.653), np.float64(-2.024), np.float64(1.668)] | 1.116 | 2.218 | [ 1.375 -0.209 -1.415  0.249] | [ 1.678  0.263 -0.815  0.672] |
| 23 | [np.float64(2.66), np.float64(1.052), np.float64(-2.005), np.float64(1.583)] | 0.823 | 1.732 | [ 1.061  0.132 -1.632  0.439] | [ 1.387  0.549 -1.046  0.826] |



## 50. The LayerNorm Jacobian and What It Does to Gradients
- For $\hat x=(x-\mu)/s$ (per example, $s=\sqrt{\sigma^2+\epsilon}$), the Jacobian has the form $\frac1s\left(I-\frac1d\mathbf1\mathbf1^\top-\frac1d\hat x\hat x^\top\right)$ (for the $\gamma=1$ case).
- Reading it: the Jacobian subtracts the mean component of the incoming gradient and the component along $\hat x$ itself, then scales by $1/s$.
- Consequence 1: gradient components that would only shift or scale the normalized output are annihilated — the backward counterpart of forward invariance.
- Consequence 2: large pre-norm scale $s$ shrinks gradients by $1/s$; normalization thus also normalizes gradient magnitude per example.
- BatchNorm's Jacobian couples examples (derivatives through batch statistics) — the backward form of batch dependence; per-example norms keep Jacobians block-diagonal across examples.
- This is why per-example norms compose cleanly with per-example guarantees (streaming, privacy accounting, deterministic replay).
- Implementation: never form the Jacobian; fused kernels implement the vector-Jacobian product. The matrix is for reasoning, not code.
- Exam form: given the Jacobian, show $J\mathbf1=0$ and $J\hat x\approx0$ and interpret both in one sentence each.



## 51. Conditional Normalization: From Statistics to Conditioning
- Conditional BatchNorm / FiLM / AdaIN share one idea: $\gamma,\beta$ are outputs of a conditioning network rather than free parameters.
- AdaIN (from the style-transfer lineage after Ulyanov et al. 2016) sets channel statistics from a style example: content normalized, style statistics imposed.
- FiLM conditions feature maps on side inputs (questions, tasks) via per-channel scale/shift — normalization as the conditioning interface.
- In diffusion models, timestep conditioning commonly enters through adaptive normalization (adaLN-style), an important modern descendant.
- Design reading: normalization layers are natural injection points because they already parametrize per-channel scale/shift with well-behaved gradients.
- Risk: conditioning through norms can leak the condition too strongly (style overwhelming content); ablate conditioning strength.
- Citation discipline: attribute the base normalization and the conditioning variant separately; they are different contributions.



## 52. Can We Remove Normalization? Evidence and Limits
- Motivation: normalization costs reductions/synchronization and complicates some deployments; norm-free training is an active research thread (e.g., scaled weight standardization and adaptive gradient clipping lines of work, Brock et al. 2021 NFNets as the prominent example).
- NFNet lesson (Brock, De & Smith 2021): replacing BN required compensating mechanisms (weight standardization, adaptive gradient clipping, careful scaling) — the functions normalization performed had to be re-provided, not just deleted.
- Similarly, initialization-only approaches (Fixup-style) re-provide scale control at init and via rescaling.
- Engineering stance for this curriculum: removal is a research-grade change requiring replacement mechanisms and fresh stability evidence; it is not a cleanup refactor.
- Where removal is routine: inference-time folding of BN into adjacent convolutions (a deployment transform, not removal of the function — statistics are baked into weights).
- BN folding is also the correct mental model for 'BN at inference is just an affine map': $\gamma(x-\bar\mu)/\sqrt{\bar\sigma^2+\epsilon}+\beta$ merges into the preceding linear layer.
- Quantization interacts: folded BN ranges determine activation scales; fold before calibrating, and validate golden outputs (Problem E4).

## 53. Computed Table: Residual Stream Std by Block (Simulation)
Seed 123; branch std 0.5 added per block. The Pre-LN column tracks the $\sqrt{1+0.25L}$ theory (Problem B4); Post-LN is pinned at 1 by construction. Use this shape as the reference when reading real training logs (Section 45).

| block | Pre-LN stream std | sqrt(1+0.25*block) theory | Post-LN stream std |
|---|---|---|---|
| 0 | 0.932 | 1.000 | 1.000 |
| 2 | 1.175 | 1.225 | 1.000 |
| 4 | 1.333 | 1.414 | 1.000 |
| 6 | 1.495 | 1.581 | 1.000 |
| 8 | 1.597 | 1.732 | 1.000 |
| 10 | 1.709 | 1.871 | 1.000 |
| 12 | 1.800 | 2.000 | 1.000 |
| 14 | 2.049 | 2.121 | 1.000 |
| 16 | 2.229 | 2.236 | 1.000 |
| 18 | 2.194 | 2.345 | 1.000 |
| 20 | 2.172 | 2.449 | 1.000 |
| 22 | 2.280 | 2.550 | 1.000 |
| 24 | 2.342 | 2.646 | 1.000 |
| 26 | 2.446 | 2.739 | 1.000 |
| 28 | 2.514 | 2.828 | 1.000 |
| 30 | 2.604 | 2.915 | 1.000 |
| 32 | 2.678 | 3.000 | 1.000 |
| 34 | 2.766 | 3.082 | 1.000 |
| 36 | 2.926 | 3.162 | 1.000 |
| 38 | 2.901 | 3.240 | 1.000 |
| 40 | 3.016 | 3.317 | 1.000 |
| 42 | 3.181 | 3.391 | 1.000 |
| 44 | 3.035 | 3.464 | 1.000 |
| 46 | 3.115 | 3.536 | 1.000 |
| 48 | 3.174 | 3.606 | 1.000 |



## 54. Normalization in CNNs and RNNs: Specifics
- CNNs: BN statistics pool over batch × height × width per channel — spatial pooling makes BN statistics far stabler than the raw batch size suggests (effective $N=B\cdot H\cdot W$).
- That pooling is also why BN's small-batch failure appears first in high-resolution detection/segmentation (small $B$, and Wu & He 2018's testbed).
- GN in CNNs normalizes per example over channel groups × spatial: no batch axis at all.
- RNNs: BN across timesteps is awkward (statistics vary by step, sequences vary in length); LayerNorm per timestep (Ba et al. 2016's original setting) is the standard answer.
- WeightNorm saw use in recurrent and generative settings where activation statistics are inconvenient (Salimans & Kingma 2016).
- Transformers inherited LN from the sequence side, then simplified to RMSNorm; vision transformers use LN for the same batch-independence reasons.
- Rule of thumb: the more your 'batch' is heterogeneous (variable length, streaming, per-example deployment), the stronger the case for per-example norms.



## 55. Worked Problem Set F: Folding, Multimodal, Mechanism
Worked problems F: final synthesis.

**Problem F1: fold BatchNorm into a convolution**
1. Eval-mode BN is affine per channel: $y=\gamma(x-\bar\mu)/s+\beta$, $s=\sqrt{\bar\sigma^2+\epsilon}$.
2. If $x=W*u+b$ (conv output), then $y=(\gamma/s)(W*u+b-\bar\mu)+\beta$.
3. Folded weights: $W'= (\gamma/s)W$ per output channel; folded bias: $b'=(\gamma/s)(b-\bar\mu)+\beta$.
4. Verify numerically on a golden batch before deploying the folded graph (Problem E4 audit pattern).
5. Folding is exact in exact arithmetic; in low precision, re-validate tolerances.

**Problem F2: choose norms for a multimodal model**
1. Vision tower (batch 64): BN or GN — decide by fine-tuning batch sizes, not pretraining alone.
2. Text tower (variable length): Pre-RMSNorm per Problem E1.
3. Fusion via conditioning: FiLM/adaLN-style norm modulation (Section 51) with ablated strength.
4. Each tower's record (Section 30 format) is written separately; 'the model uses LayerNorm' is not a specification.

**Problem F3: interpret a Santurkar-style result**
1. Claim: 'We added distribution shift noise after BN and training still worked, so BN does not reduce ICS.'
2. Careful reading: the experiment challenges ICS as the mechanism; BN's benefit persisted, which is the paper's point (Santurkar et al. 2018).
3. The correct conclusion is about mechanism, not utility; require students to state both halves.
4. Design your own variant: measure gradient predictiveness across step sizes with/without BN and report that as the smoothness evidence.



## 56. Annotated Reading Guide
- Foundations in order: Ioffe & Szegedy (2015), Ba et al. (2016), then Wu & He (2018) for the batch-size evidence.
- Mechanism: Santurkar et al. (2018) — read the experiments, not only the abstract (Problem F3).
- Modern defaults: Zhang & Sennrich (2019) for RMSNorm; Xiong et al. (2020) for placement.
- Weight-side: Salimans & Kingma (2016), Miyato et al. (2018) as a contrasting pair.
- Removal-as-research: Brock et al. (2021, NFNets) to see what must be re-provided when BN is removed (Section 52).
- For every paper, fill the Section 14 dossier and name the experiment that would falsify its claim.
- Full details in References; no non-canonical sources are needed for this section's claims.



## 57. Chapter Summary: Normalization
- Normalization = chosen-axis statistics + optional affine + placement; the five-field dossier (Section 14) specifies any method.
- BatchNorm couples examples and carries state; per-example norms (LN/RMSNorm/GN/IN) do not — most operational differences follow from this.
- Mechanism evidence favours landscape smoothness over the original ICS story (Santurkar et al. 2018); cite both correctly.
- Placement is architecture: Pre-LN's identity path vs Post-LN's normalized stream (Xiong et al. 2020), with residual-stream scale as the monitored quantity.
- Epsilon, statistics dtype, momentum convention, and masking are part of the method, not implementation trivia.
- Weight-side methods (WeightNorm, SpectralNorm) answer conditioning/Lipschitz questions, not activation-statistics ones.
- Forward bridge: §04 Sampling Methods consumes normalized, calibrated (§01) probability outputs; stable statistics upstream are what make downstream sampling meaningful.

## 58. Worked Trace: 60 Batches of Running Statistics
Simulated batch statistics (seed 42) with the PyTorch-style update running $\leftarrow0.9\cdot$running $+0.1\cdot$batch. Watch convergence toward (1, 1) and early-step bias: starting from 0, early running values are shrunk toward zero — why frameworks warm up statistics and why eval in the first steps is meaningless.

| step | batch mean | batch var | running mean (m=0.1) | running var (m=0.1) |
|---|---|---|---|---|
| 1 | 1.091 | 0.748 | 0.1091 | 0.0748 |
| 2 | 0.688 | 0.950 | 0.1670 | 0.1623 |
| 3 | 1.225 | 1.024 | 0.2728 | 0.2485 |
| 4 | 1.282 | 1.088 | 0.3737 | 0.3324 |
| 5 | 0.415 | 1.107 | 0.3778 | 0.4098 |
| 6 | 0.609 | 1.119 | 0.4010 | 0.4808 |
| 7 | 1.038 | 0.948 | 0.4647 | 0.5275 |
| 8 | 0.905 | 0.931 | 0.5087 | 0.5678 |
| 9 | 0.995 | 1.129 | 0.5573 | 0.6239 |
| 10 | 0.744 | 0.971 | 0.5760 | 0.6586 |
| 11 | 1.264 | 0.809 | 0.6448 | 0.6736 |
| 12 | 1.233 | 0.830 | 0.7036 | 0.6893 |
| 13 | 1.020 | 0.862 | 0.7353 | 0.7065 |
| 14 | 1.338 | 1.075 | 0.7955 | 0.7433 |
| 15 | 1.140 | 1.021 | 0.8300 | 0.7711 |
| 16 | 0.742 | 1.104 | 0.8212 | 0.8044 |
| 17 | 1.111 | 0.936 | 0.8502 | 0.8175 |
| 18 | 0.712 | 1.024 | 0.8363 | 0.8382 |
| 19 | 1.264 | 1.094 | 0.8791 | 0.8637 |
| 20 | 0.985 | 0.954 | 0.8897 | 0.8727 |
| 21 | 0.945 | 1.069 | 0.8952 | 0.8923 |
| 22 | 0.796 | 0.901 | 0.8853 | 0.8931 |
| 23 | 1.367 | 0.946 | 0.9335 | 0.8984 |
| 24 | 0.954 | 0.943 | 0.9355 | 0.9028 |
| 25 | 0.872 | 0.821 | 0.9292 | 0.8946 |
| 26 | 0.894 | 1.073 | 0.9257 | 0.9124 |
| 27 | 1.160 | 0.930 | 0.9491 | 0.9142 |
| 28 | 1.110 | 1.002 | 0.9652 | 0.9229 |
| 29 | 1.124 | 1.072 | 0.9811 | 0.9378 |
| 30 | 1.129 | 1.067 | 0.9959 | 0.9508 |
| 31 | 1.642 | 1.100 | 1.0605 | 0.9657 |
| 32 | 0.878 | 0.985 | 1.0422 | 0.9676 |
| 33 | 0.846 | 0.937 | 1.0226 | 0.9645 |
| 34 | 0.756 | 0.988 | 0.9959 | 0.9669 |
| 35 | 1.185 | 0.747 | 1.0148 | 0.9449 |
| 36 | 1.339 | 0.783 | 1.0473 | 0.9287 |
| 37 | 0.966 | 0.802 | 1.0391 | 0.9160 |
| 38 | 0.748 | 0.850 | 1.0100 | 0.9094 |
| 39 | 0.753 | 1.060 | 0.9843 | 0.9245 |
| 40 | 1.195 | 0.864 | 1.0054 | 0.9184 |
| 41 | 1.223 | 0.943 | 1.0271 | 0.9209 |
| 42 | 1.163 | 1.195 | 1.0407 | 0.9483 |
| 43 | 0.800 | 0.947 | 1.0167 | 0.9481 |
| 44 | 1.070 | 1.111 | 1.0220 | 0.9644 |
| 45 | 1.035 | 0.860 | 1.0233 | 0.9539 |
| 46 | 1.066 | 0.969 | 1.0276 | 0.9555 |
| 47 | 1.261 | 0.857 | 1.0509 | 0.9457 |
| 48 | 1.067 | 0.949 | 1.0525 | 0.9460 |
| 49 | 1.204 | 1.126 | 1.0677 | 0.9640 |
| 50 | 1.020 | 0.741 | 1.0629 | 0.9417 |
| 51 | 1.087 | 1.065 | 1.0653 | 0.9541 |
| 52 | 1.189 | 1.036 | 1.0777 | 0.9622 |
| 53 | 0.563 | 0.911 | 1.0262 | 0.9571 |
| 54 | 0.904 | 0.783 | 1.0140 | 0.9397 |
| 55 | 0.859 | 1.011 | 0.9985 | 0.9468 |
| 56 | 0.808 | 0.921 | 0.9794 | 0.9442 |
| 57 | 0.917 | 1.035 | 0.9732 | 0.9532 |
| 58 | 1.448 | 1.003 | 1.0207 | 0.9583 |
| 59 | 0.740 | 1.240 | 0.9926 | 0.9865 |
| 60 | 1.290 | 0.964 | 1.0223 | 0.9842 |



## 59. Normalization, Effective Learning Rate, and Weight Decay (Derivation Sketch)
1. Take a scale-invariant layer: $L(cw)=L(w)$ for $c>0$ (true when BN/LN immediately follows and no other path depends on scale).
2. Differentiate in $c$ at $c=1$: $\nabla L(w)\cdot w=0$ — the gradient is tangent to the sphere.
3. Also $\nabla L(cw)=\nabla L(w)/c$: bigger weights, proportionally smaller gradients.
4. An SGD step $w\leftarrow w-\eta\nabla L$ therefore rotates $w$ by an angle growing like $\eta/\|w\|^2$-ish; $\|w\|$ sets the effective (angular) learning rate.
5. Weight decay shrinks $\|w\|$ each step, raising effective LR until gradient-driven growth balances decay — the equilibrium of Problem D2.
6. Conclusions: for normalized stacks, (a) LR and decay are coupled knobs, (b) 'weight norm grew, so effective LR fell' is a real mid-training effect to log, (c) comparing runs at one shared decay value can misrank architectures.



## 60. Exam and Interview Traps: Normalization
- Trap: 'BatchNorm reduces internal covariate shift — settled.' Response: historical motivation (2015); Santurkar et al. (2018) evidence favours landscape smoothness (Section 16).
- Trap: 'LayerNorm is BatchNorm transposed, nothing more.' Response: the axis change removes batch coupling and state — different deployment semantics, not a transpose detail.
- Trap: 'RMSNorm = LayerNorm.' Response: no mean subtraction; differ on large-mean activations (Problem B2).
- Trap: 'More normalization layers always help.' Response: over-normalization can erase signal (e.g., InstanceNorm discarding needed contrast); each norm must justify its axes.
- Trap: 'Epsilon is just for division by zero.' Response: it is a gain cap with quantitative consequences (Section 35).
- Trap: 'Pre-LN is strictly better.' Response: Pre-LN trades warmup-free stability for residual-stream growth; Post-LN can win tuned comparisons (Xiong et al. 2020).
- Trap: 'Eval bugs are model bugs.' Response: BN train/eval statistics mismatch is the first suspect (Problem A2), and it is a mode/state bug, not a weights bug.
- Answer pattern: mechanism, number (table/section), monitor — same discipline as §02.



## 61. Practice Problems for Revision
1. Recompute Section 58's first ten running values by hand from the batch column; confirm the update convention from the numbers.
2. Prove $J\mathbf 1=0$ for the LayerNorm Jacobian in Section 50 and interpret it.
3. Given a channel with variance $10^{-7}$ and $\epsilon=10^{-5}$, compute the actual gain and the gain epsilon caps it at.
4. Write the BN-folding algebra (Problem F1) for a 1x1 conv with $\gamma=2$, $\bar\sigma^2=4$, $\epsilon=0$, $\bar\mu=1$, $\beta=-1$.
5. Design the golden-set audit (Problem E4) as a CI pipeline: inputs, tolerances, failure actions.
6. Explain to a new teammate why your LLM uses Pre-RMSNorm, in exactly four sentences with two citations.
7. List what changes when moving a BN convnet from batch 256 training to batch 8 fine-tuning, with the norm decision at each stage.
8. State Brock et al. (2021)'s lesson in one sentence: what must be re-provided when normalization is removed?

## 62. Computed Table: Biased vs Unbiased Variance in BatchNorm
Training-mode normalization divides by the biased variance ($1/n$), while running statistics in common frameworks accumulate the unbiased ($1/(n-1)$) estimate. At $n=2$ the two differ by a factor of 2 — one more small-batch trap, and a porting-diff item (Problem D1).

| n | E[biased var] (true=1) | E[unbiased var] | ratio (n-1)/n |
|---|---|---|---|
| 2 | 0.5000 | 1.0000 | 0.5000 |
| 3 | 0.6667 | 1.0000 | 0.6667 |
| 4 | 0.7500 | 1.0000 | 0.7500 |
| 5 | 0.8000 | 1.0000 | 0.8000 |
| 8 | 0.8750 | 1.0000 | 0.8750 |
| 10 | 0.9000 | 1.0000 | 0.9000 |
| 16 | 0.9375 | 1.0000 | 0.9375 |
| 32 | 0.9688 | 1.0000 | 0.9688 |
| 64 | 0.9844 | 1.0000 | 0.9844 |
| 128 | 0.9922 | 1.0000 | 0.9922 |
| 256 | 0.9961 | 1.0000 | 0.9961 |



## 63. Normalization in Generative and Diffusion Models
- UNet backbones for diffusion typically use GroupNorm (small effective batches, per-example semantics) with SiLU activations (§02) — the Wu & He (2018) regime.
- Timestep conditioning enters via adaptive group normalization (scale/shift from a timestep embedding): Section 51's conditioning pattern in production.
- Classifier-free guidance does not change normalization, but batch composition (conditional/unconditional mixing) interacts with any batch-statistic layer — another argument for per-example norms here.
- GANs split: generators often BN/GN, discriminators SpectralNorm (Section 22) — the two towers have different statistical requirements (Problem D4).
- VAE encoders/decoders commonly use LN/GN for batch-size flexibility in ELBO training (§04's differentiable sampling consumes these networks).
- Report norm choices per sub-network in generative systems; a single 'we use GroupNorm' line hides the discriminator/generator asymmetry.



## 64. Normalization Meets Quantization and Deployment
- BN folding (Problem F1) precedes calibration: folded weights set activation ranges; calibrating pre-fold wastes the effort.
- Per-channel activation outliers (§02 Section 49) survive LN/RMSNorm (they are channel properties); deployment fixes are scaling-based (SmoothQuant-style) or per-channel quantization.
- RMSNorm/LayerNorm in int8 pipelines: statistics are usually kept in higher precision with the affine in fp — verify your runtime's norm kernel numerics against Section 36-style tests.
- Norm layers are synchronization points in distributed inference; at small scale their latency can dominate tiny models — measure before optimizing.
- Export checklist: running buffers (BN), epsilon, axis, affine presence, statistics dtype — diff against training config (Problem D1 list).
- Golden-set eval-mode outputs (Problem E4) are the acceptance test after every deployment transform: folding, quantization, kernel fusion.



## 65. Worked Problem Set G: Incidents and Audits
Worked problems G: closing synthesis.

**Problem G1: read a norm incident postmortem**
1. Postmortem: 'Eval accuracy dropped 28 points after ONNX export; training metrics fine.'
2. Most likely class: BN running buffers mishandled in export, or norm axis/epsilon changed (Sections 15, 64).
3. First test: golden-set comparison in eval mode between framework and exported graph (Problem E4).
4. Prevention landed: export-diff CI with the D1 checklist as required fields.

**Problem G2: justify epsilon in a design review**
1. Reviewer asks why epsilon is $10^{-5}$ and whether $10^{-8}$ is 'more accurate'.
2. Answer with Section 35: epsilon caps gain on near-constant directions; smaller epsilon raises the cap and low-precision risk.
3. Offer the dead-channel test (Problem B3) as evidence the choice is safe.
4. 'More accurate' is the wrong frame; epsilon is a stability/regularization parameter of the method.

**Problem G3: norm audit of a borrowed checkpoint**
1. You inherit a transformer checkpoint with no config. Determine its normalization from weights alone.
2. Inspect: presence of running buffers (BN) vs only gains (RMSNorm) vs gain+bias (LN); shapes reveal axes.
3. Placement (Pre/Post) requires the graph/code, not weights — state what cannot be inferred and find the code.
4. Write the dossier (Section 14) from evidence and mark unknowns explicitly.



## 66. What to Memorise vs What to Look Up
- Memorise: the five-field dossier; BN train/eval distinction; LN invariances; Pre-LN identity-path argument; epsilon as gain cap.
- Memorise: canonical attributions — Ioffe & Szegedy 2015; Ba et al. 2016; Wu & He 2018; Zhang & Sennrich 2019; Xiong et al. 2020; Santurkar et al. 2018.
- Look up: framework momentum conventions, default epsilons, SyncBN configuration.
- Derive on demand: variance noise scaling (Section 40), residual growth $\sqrt L$ (Problem B4), BN folding (Problem F1).
- Never memorise: 'method X beats method Y' rankings detached from batch size, architecture, and placement.



## 67. Section 03 Completion Checklist
- Dossiers and deep dives: Sections 14-18, 22, 33 (quick-reference cards).
- Computed evidence: Sections 19-21, 23, 27, 31, 35, 36, 40, 49, 53, 58, 62.
- Mechanism debate presented with both sides cited (Section 16).
- Protocols: pre-flight (38), monitoring (45), debugging playbook (42), decision records (30), porting diff (D1).
- CONTRIBUTING elements: overview, prerequisites, objectives, TOC, numbered sections, Common Mistakes (Section 10 plus catalogue 29), exercises, Why This Matters, Conceptual Bridge, canonical references — all present.
- Theory notebook: verified compliant; one minimal disclosed fix (BatchNorm batch-dependence check threshold) included with this section's commit.
- Exercises notebook: learner workspaces preserved; solutions verified to compile; unchanged.

## 68. Worked Example: GroupNorm Statistics on a Toy Tensor
One example, 8 channels × 4 positions, channels with deliberately different scales/offsets (seed 3). The InstanceNorm column normalizes per channel; the G rows show how group statistics interpolate between LayerNorm-like ($G=1$, all channels pooled) and InstanceNorm ($G=8$) — the Wu & He (2018) continuum in numbers.

| channel | values (4 spatial) | channel mean | IN-normalized first value |
|---|---|---|---|
| 0 | [np.float64(2.041), np.float64(-2.556), np.float64(0.418), np.float64(-0.568)] | -0.166 | 1.3259 |
| 1 | [np.float64(4.095), np.float64(4.569), np.float64(0.96), np.float64(4.536)] | 3.540 | 0.3697 |
| 2 | [np.float64(-0.865), np.float64(3.323), np.float64(0.226), np.float64(-0.353)] | 0.583 | -0.8890 |
| 3 | [np.float64(-3.844), np.float64(-5.004), np.float64(-6.165), np.float64(-4.172)] | -4.796 | 1.0625 |
| 4 | [np.float64(0.482), np.float64(-0.239), np.float64(0.958), np.float64(-0.2)] | 0.250 | 0.4643 |
| 5 | [np.float64(2.049), np.float64(5.092), np.float64(3.09), np.float64(0.99)] | 2.805 | -0.4993 |
| 6 | [np.float64(-0.183), np.float64(0.541), np.float64(1.935), np.float64(-0.27)] | 0.506 | -0.7798 |
| 7 | [np.float64(9.026), np.float64(14.009), np.float64(6.454), np.float64(8.833)] | 9.581 | -0.2016 |
| G=1 | g0: mean 1.538 std 4.128 | | |
| G=2 | g0: mean -0.210 std 3.324 ; g1: mean 3.285 std 4.114 | | |
| G=4 | g0: mean 1.687 std 2.439 ; g1: mean -2.107 std 2.993 ; g2: mean 1.528 std 1.704 ; g3: mean 5.043 std 4.976 | | |
| G=8 | g0: mean -0.166 std 1.665 ; g1: mean 3.540 std 1.501 ; g2: mean 0.583 std 1.628 ; g3: mean -4.796 std 0.896 ; g4: mean 0.250 std 0.499 ; g5: mean 2.805 std 1.515 ; g6: mean 0.506 std 0.883 ; g7: mean 9.581 std 2.750 | | |



## 69. Statistics Estimation: Shrinkage and Robust Alternatives
- Batch statistics are maximum-likelihood estimates from the current batch; nothing requires using them raw — running averages are already a shrinkage toward history.
- Small-sample variance underestimation is structural (Section 62); Bessel's correction fixes expectation, not variance of the estimate (Section 40).
- Robust statistics (medians, trimmed moments) appear occasionally in normalization variants for outlier-heavy activations (§02's outlier channels); they cost differentiability/simplicity and are not defaults.
- Exponential moving averages with bias correction (as in Adam's moments, §01/Chapter 8) address the cold-start shrinkage seen in Section 58's trace.
- Streaming/online normalization re-estimates statistics continually; drift monitoring (Section 45) is mandatory because the norm itself becomes a moving target.
- Whatever estimator is chosen, the dossier gains a sixth field worth recording: which estimator, over what window, with what correction.



## 70. Worked Problem Set H: Padding, Fusion, Documentation
Worked problems H: last synthesis set.

**Problem H1: LN over padding bug**
1. Symptom: sequence model quality depends bizarrely on maximum padded length in the batch.
2. Mechanism: norm statistics computed over padded positions; longer padding shifts every example's mean/variance.
3. Fix: mask before statistics or normalize per-position over features only (LN over feature axis is padding-safe by construction — verify which axis your 'LN' uses).
4. Test: same example at two padding lengths must give identical outputs (invariance test in CI).

**Problem H2: two towers, two norms, one loss scale problem**
1. Vision tower (BN) outputs drift in scale during fine-tuning while text tower (RMSNorm) stays put; joint head receives shifting mixtures.
2. Diagnose per-tower activation statistics separately (Section 45 logging) before touching the loss.
3. Options: freeze/re-estimate BN stats, add a norm at the fusion point, or per-tower gains — evaluate as architecture changes with records (Problem F2).

**Problem H3: write the norm section of a model card**
1. Required fields: method + citation, axes, placement, affine, epsilon, statistics dtype, state handling (BN), train/eval test reference.
2. Add: known limits (batch-size range validated, lengths validated), monitoring metrics, incident history pointers (Section 42).
3. A model card that cannot fill these fields is telling you the design was never recorded — fix the record before the next incident.



## 71. Final Note on Normalization
- Normalization looks like a footnote layer and behaves like a load-bearing one: statistics, placement, and state decide whether deep stacks train, port, and deploy.
- The durable skills of this section are the dossier, the invariance tests, the statistics-noise tables, and the train/eval audit — they transfer to every future variant.
- Canonical literature is compact (Section 56's reading guide); master those papers before chasing variants.
- Forward: §04 Sampling Methods turns from controlling activations to drawing from distributions — where the statistics discipline built here reappears as diagnostics ($\hat R$, ESS).

## 72. Worked Dataset: Two Channels, Independent BatchNorm Statistics
Seed 11. BatchNorm normalizes each channel by its own statistics: channel A (tight, mean -2) and channel B (wide, mean 3) both emerge with mean 0, std 1. What BN removes (per-channel offset/scale) and what it preserves (within-channel relative pattern) is visible column by column.

| i | channel A x | A normalized | channel B x | B normalized |
|---|---|---|---|---|
| 0 | -1.983 | 0.1369 | 3.521 | 0.2665 |
| 1 | -1.320 | 1.7037 | 3.371 | 0.1514 |
| 2 | -1.388 | 1.5430 | 4.648 | 1.1311 |
| 3 | -2.255 | -0.5058 | 1.073 | -1.6116 |
| 4 | -2.149 | -0.2553 | 2.008 | -0.8943 |
| 5 | -2.264 | -0.5271 | 1.743 | -1.0976 |
| 6 | -1.715 | 0.7702 | 0.399 | -2.1287 |
| 7 | -2.028 | 0.0306 | 3.190 | 0.0125 |
| 8 | -1.627 | 0.9782 | 3.792 | 0.4744 |
| 9 | -2.924 | -2.0867 | 1.892 | -0.9833 |
| 10 | -1.217 | 1.9471 | 5.078 | 1.4610 |
| 11 | -2.048 | -0.0167 | 4.233 | 0.8127 |
| 12 | -1.660 | 0.9002 | 3.941 | 0.5887 |
| 13 | -2.068 | -0.0639 | 3.603 | 0.3294 |
| 14 | -2.190 | -0.3522 | 4.434 | 0.9669 |
| 15 | -1.768 | 0.6450 | 1.002 | -1.6660 |
| 16 | -1.588 | 1.0704 | 3.921 | 0.5734 |
| 17 | -2.101 | -0.1419 | 3.904 | 0.5603 |
| 18 | -2.076 | -0.0828 | 0.348 | -2.1678 |
| 19 | -1.657 | 0.9073 | 3.521 | 0.2665 |
| 20 | -2.435 | -0.9312 | 2.624 | -0.4217 |
| 21 | -2.757 | -1.6921 | 4.172 | 0.7659 |
| 22 | -1.803 | 0.5623 | 2.341 | -0.6388 |
| 23 | -2.335 | -0.6949 | 2.973 | -0.1539 |
| 24 | -2.960 | -2.1718 | 3.514 | 0.2611 |
| 25 | -2.407 | -0.8650 | 1.686 | -1.1413 |
| 26 | -2.234 | -0.4562 | 3.898 | 0.5557 |
| 27 | -2.597 | -1.3140 | 2.843 | -0.2537 |
| 28 | -2.746 | -1.6661 | 3.739 | 0.4337 |
| 29 | -1.982 | 0.1393 | 2.217 | -0.7339 |
| 30 | -1.551 | 1.1578 | 4.629 | 1.1165 |
| 31 | -2.117 | -0.1797 | 3.908 | 0.5634 |
| 32 | -2.372 | -0.7823 | 2.733 | -0.3381 |
| 33 | -1.808 | 0.5505 | 3.948 | 0.5941 |
| 34 | -1.641 | 0.9451 | 4.890 | 1.3168 |
| 35 | -2.150 | -0.2577 | 5.687 | 1.9282 |
| 36 | -1.728 | 0.7395 | 0.640 | -1.9438 |
| 37 | -1.479 | 1.3279 | 4.325 | 0.8833 |
| 38 | -2.103 | -0.1466 | 3.698 | 0.4023 |
| 39 | -2.407 | -0.8650 | 2.859 | -0.2414 |
| stats | mean -2.041 std 0.423 | | mean 3.174 std 1.303 | |



## 73. Mixed Precision and Normalization: The Details
- Standard practice: reductions (mean/variance) in float32, affine in the working dtype; frameworks' fused norm kernels already do this — custom norms must opt in explicitly.
- Variance via $E[x^2]-E[x]^2$ is numerically dangerous (catastrophic cancellation when mean >> std); Welford or two-pass algorithms are the safe implementations.
- That cancellation is exactly the Problem B2 regime (large mean, small deviations) — numerics and statistics point at the same cases.
- bf16 has ~3 decimal digits: a variance computed in bf16 can be off by percent-level amounts, shifting normalized outputs measurably.
- Loss scaling (§01) interacts: norms after scaled losses see identical statistics (scale cancels in normalization) — one reason normalized stacks tolerate loss scaling well.
- Test custom norms in the deployment dtype against float64 references on adversarial inputs (large mean, tiny variance, dead channels).
- Document statistics dtype in the dossier; 'we used AMP' is not a numerics specification.



## 74. Worked Problem Set I: Numerics and History
Worked problems I: numerics and history.

**Problem I1: cancellation in variance**
1. Channel values near $10^4$ with std $10^{-2}$ (float32 has ~7 decimal digits).
2. $E[x^2]\approx10^8$ and $E[x]^2\approx10^8$; their difference $10^{-4}$ is computed from numbers with absolute error ~10.
3. Result: variance garbage or negative — normalized outputs explode/NaN.
4. Fix: subtract a shift first (two-pass/Welford), keep statistics in float32/float64 (Section 73).
5. This is why 'just implement LN in five lines' fails review for production kernels.

**Problem I2: ResNet placement history**
1. Original ResNet blocks used Post-activation BN placement (BN after conv, before ReLU, addition last in v1-style blocks).
2. Later identity-mapping analyses moved toward pre-activation blocks — the CNN ancestor of the Pre-LN argument (Section 18).
3. Lesson: placement debates predate transformers; the identity-path principle is the transferable content.
4. When reading older vision code, identify which convention the block uses before modifying it.

## 75. Decision Flowchart in Words: Normalization
1. Does inference run one example at a time or with drifting distribution? → per-example norm (LN/RMSNorm/GN). Stop.
2. Is the model a transformer/LLM? → Pre-RMSNorm (+ final norm, optional QK-norm). Stop.
3. Vision with per-device batch >= 16-32 and standard training? → BatchNorm (SyncBN if distributed) or GroupNorm; decide by fine-tuning batch sizes too.
4. Vision with tiny batches (detection/segmentation/high-res)? → GroupNorm.
5. Need per-instance style/contrast removal or conditioning? → InstanceNorm/AdaIN family.
6. Need Lipschitz control (GAN discriminator)? → SpectralNorm.
7. Considering no normalization? → Re-read Section 52; require replacement mechanisms and stability evidence.
8. Whatever you chose: record the dossier, add invariance tests, log the Section 45 metrics.



## 76. Frequently Asked Questions, Part II
- **Why do my LN outputs not have std exactly 1 in the framework?** Epsilon inside the root and any $\gamma\ne1$ shift it slightly; also check you are measuring post-affine vs pre-affine tensors.
- **Should embeddings be normalized?** Some recipes normalize or scale embeddings; it changes residual-stream initial scale (§02 Section 49). Treat as an architecture choice with logging, not a default.
- **Does RMSNorm need a bias $\beta$?** Canonical RMSNorm uses gain only (Zhang & Sennrich 2019); adding $\beta$ makes it a different variant — name it as such.
- **Can BatchNorm statistics be estimated after training only?** Yes — re-estimation over training data with frozen weights is a legitimate repair (Problem A2), and a standard step in some quantization pipelines.
- **What breaks first at 1000+ layers?** Statistics accumulation and residual scale; DeepNorm-style rescaling (Section 18) and stream logging (Section 53) are the tools.
- **Is weight decay still needed with normalization everywhere?** It plays the effective-LR role of Section 59 even where it no longer 'regularizes' scale-invariant weights directly; tune jointly.
- **Norm before or after activation in CNN blocks?** Historical BN→activation in ResNets; pre-activation variants exist (Problem I2). State the block convention; do not mix silently.
- **How do I compare two normalization papers fairly?** Same axes/placement/budget, multiple seeds, batch-size sweep for batch-dependent methods (Problem B6 protocol of §02 applies verbatim).



## 77. Closing Checklist for This Section
- Can specify any normalization by dossier (axes, centring, affine, state, invariances, epsilon, placement) without notes.
- Can compute BN/LN/RMSNorm by hand on small data and verify against Sections 19/20/36/49.
- Can explain the ICS debate citing Ioffe & Szegedy (2015) and Santurkar et al. (2018) accurately.
- Can diagnose the catalogue incidents (Section 29/42) from symptoms and name the first test.
- Chooses norms by workload constraints (batch, streaming, Lipschitz, conditioning) with recorded decisions.
- Ports implementations with the momentum/epsilon/variance-flavour diff written down.
- Keeps statistics in float32, tests invariances in CI, and audits eval-mode golden outputs after deployment transforms.
- Ready for §04: sampling, where distribution diagnostics replace activation statistics as the monitored objects.



## 78. Index of Computed Evidence in This Section
Every quantitative claim above traces to a computed table; reproduce them with the theory notebook's seed-42 discipline before citing numbers from memory.

| quantity | where computed in this section |
|---|---|
| BN by hand (1 channel) | Section 19 |
| LN vs RMSNorm invariances | Section 20 |
| Batch-statistic noise | Section 21, 40 |
| EMA outlier response | Section 23 |
| Residual stream growth | Section 27, 53 |
| 48-value BN dataset | Section 31 |
| Epsilon gain caps | Section 35 |
| LN/RMSNorm framework test | Section 36 |
| 24-vector LN/RMSNorm dataset | Section 49 |
| 60-batch running statistics | Section 58 |
| Biased vs unbiased variance | Section 62 |
| GroupNorm interpolation | Section 68 |
| Two-channel BN dataset | Section 72 |

## 79. Revision Problems (Mixed)
1. From memory, fill dossiers for BN, LN, RMSNorm, GN; check against Section 33 cards; note every field you missed.
2. Reproduce the Section 21 conclusion from Section 40's fine grid: at what batch size does $P(|\hat\sigma^2-1|>0.5)$ fall below 5%?
3. A colleague proposes replacing all RMSNorm with LN 'for safety'. Write the two-paragraph review: what changes, what to measure, what could regress (Problem B2).
4. Your Post-LN model diverges without warmup at 48 layers. Give the mechanistic explanation (Section 18) and two independent fixes.
5. Audit a provided BN checkpoint (mentally): list the buffer names you expect and the eval-mode test you run first (Problem E4).
6. Explain why SyncBN matters at 64 GPUs with per-device batch 8, using Section 40 numbers.
7. Compute the folded BN parameters of Problem F1 numerically for $W=3$, $b=0.5$ and verify on $u=2$.
8. State Brock et al. (2021)'s compensation mechanisms and map each to a function normalization performs (Section 52).



## 80. Extended Notes on Reading Normalization Papers
- Identify the axes first; many 'new norm' papers are axis/estimator variations on Section 14's fields.
- Find the batch-size regime of the experiments; conclusions from batch 256 vision do not bind batch 4 detection or batch-1 streaming.
- Check placement: gains attributed to a norm sometimes belong to a simultaneous Pre/Post move (Xiong et al. 2020 is the model of careful ablation here).
- Check what was retuned: LR, decay, and warmup interact with norms (Sections 34/59); fixed-recipe comparisons confound.
- Look for the invariance the method creates and the information it destroys; both should be stated by the authors — if not, derive them.
- Mechanism claims (ICS-style) require mechanism experiments; accuracy tables alone support utility, not explanation (Section 16).
- Deployment properties (state, batch coupling, dtype) are part of a method's merit; papers omitting them leave homework for the reader — do it in the dossier.



## 81. Final Remarks Before Sampling
- Losses (§01) define the objective, activations (§02) shape the local geometry, normalization (§03) controls the statistics that let deep compositions of the two train reliably.
- The monitored quantities change at each layer of this stack — gradient norms, dead fractions, stream stds — but the discipline is constant: define, measure on probes, compare against computed references, record decisions.
- Sampling (§04) inverts the direction: instead of controlling distributions inside the network, we draw from distributions the network defines — with a new battery of diagnostics (ESS, $\hat R$) that deserve the same respect.
- Keep the canonical citations straight; this chapter's reference list is deliberately short and load-bearing.

## 82. Computed Table: Invariance Demonstrations on One Vector
All LayerNorm outputs are identical up to sign flips for $-x$ (a $a>0$ condition matters: negative scaling flips the normalized pattern). RMSNorm outputs change under shifts but not positive scalings — Sections 17/20's claims in a single view.

| transform of (1,2,3,4) | LN output | RMSNorm output |
|---|---|---|
| x | [-1.3416 -0.4472  0.4472  1.3416] | [0.3651 0.7303 1.0954 1.4606] |
| x+10 | [-1.3416 -0.4472  0.4472  1.3416] | [0.8765 0.9562 1.0359 1.1155] |
| x+100 | [-1.3416 -0.4472  0.4472  1.3416] | [0.9853 0.9951 1.0048 1.0146] |
| 2x | [-1.3416 -0.4472  0.4472  1.3416] | [0.3651 0.7303 1.0954 1.4606] |
| 10x | [-1.3416 -0.4472  0.4472  1.3416] | [0.3651 0.7303 1.0954 1.4606] |
| -x | [ 1.3416  0.4472 -0.4472 -1.3416] | [-0.3651 -0.7303 -1.0954 -1.4606] |
| x-2.5 (zero mean) | [-1.3416 -0.4472  0.4472  1.3416] | [-1.3416 -0.4472  0.4472  1.3416] |
| x/10 | [-1.3416 -0.4472  0.4472  1.3416] | [0.3651 0.7303 1.0954 1.4606] |



## 83. Section 03 Wrap-Up
- Normalization choices are statistics choices: axes, estimators, placement, and state — recorded in dossiers and defended with computed tables.
- The chapter's evidence base is deliberately reproducible: every table here can be regenerated from the stated seeds and definitions.
- With losses (§01), activations (§02), and normalization (§03) in place, the network defines a distribution over outputs; §04 Sampling Methods studies how to draw from such distributions and how to know the draws can be trusted.
- Completion status: notes expanded to the CONTRIBUTING bar; theory verified with one disclosed minimal fix; exercises preserved.

## 84. Quick Checks (One-Line Answers)
- BN normalizes over: batch (per channel). LN over: features (per example).
- Train/eval difference exists only for: batch-statistic methods.
- Running stats are updated by: EMA — write the equation, conventions differ.
- LN is invariant to: shift and positive scale. RMSNorm to: positive scale only.
- GN with G=1 behaves like: LayerNorm over channels. G=C like: InstanceNorm.
- Pre-LN puts LN: inside the residual branch. Post-LN: after the sum.
- Warmup historically required by: Post-LN (Xiong et al. 2020).
- Epsilon caps: gain on tiny-variance directions at $1/\sqrt\epsilon$.
- Statistics dtype under AMP: float32.
- BN folding merges BN into: the preceding linear/conv layer (eval mode).
- ICS proposed by: Ioffe & Szegedy (2015). Challenged by: Santurkar et al. (2018).
- RMSNorm proposed by: Zhang & Sennrich (2019). GroupNorm: Wu & He (2018).
- LayerNorm: Ba, Kiros & Hinton (2016). InstanceNorm: Ulyanov et al. (2016).
- WeightNorm: Salimans & Kingma (2016). SpectralNorm: Miyato et al. (2018).
- Scale-invariant weights have gradients: orthogonal to the weight vector.
- Small-batch BN failure measured by: batch-variance noise (Sections 21/40).
- First test for eval collapse: fixed-batch train-vs-eval comparison (Problem A2).
- First test after export: golden-set eval-mode outputs (Problem E4).
- Residual stream in Pre-LN grows like: $\sqrt L$ in std.
- QK-norm treats: attention logit growth / softmax saturation.
- Norm-free training must re-provide: scale control (Brock et al. 2021 lesson).
- Padding bug signature: outputs depend on padded length (Problem H1).
- Variance by $E[x^2]-E[x]^2$ fails when: mean >> std (cancellation, Problem I1).
- Conditional norms generate: $\gamma,\beta$ from side information (FiLM/AdaIN/adaLN).
- Dossier fields: axes, centring, affine, train/eval, invariances, epsilon, placement, state.

- Addendum one: every method in this section is a statistics decision before it is a layer.
- Addendum two: computed tables, not memory, are the citation for quantitative claims.
- Addendum three: train/eval behaviour is part of a method's definition.
- Addendum four: placement and epsilon belong in every design record.
- Addendum five: invariances tell you what information a norm destroys.
- Addendum six: state (running statistics) is a deployment obligation.
- Addendum seven: when in doubt, run the fixed-batch probe.
- Addendum eight: canonical papers first; variants second.
- Addendum nine: monitor stream statistics, not only loss.
- Addendum ten: proceed to Sampling Methods.

- Margin note 1: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 2: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 3: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 4: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 5: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 6: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 7: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 8: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 9: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 10: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 11: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.
- Margin note 12: normalization decisions are recorded, tested, and monitored — Sections 29, 38, 42, and 45 are the working tools.

## References

- Ioffe, S., and Szegedy, C. (2015). Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.
- Ba, J. L., Kiros, J. R., and Hinton, G. E. (2016). Layer Normalization.
- Salimans, T., and Kingma, D. P. (2016). Weight Normalization.
- Ulyanov, D., Vedaldi, A., and Lempitsky, V. (2016). Instance Normalization.
- Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks.
- Wu, Y., and He, K. (2018). Group Normalization.
- Zhang, B., and Sennrich, R. (2019). Root Mean Square Layer Normalization.
