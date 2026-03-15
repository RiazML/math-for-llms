[← Back to Math for Specific Models](../README.md) | [Next: Probabilistic Models →](../03-Probabilistic-Models/notes.md)

---

# Neural Networks: Mathematical Foundations

> _"The difference between a neural network and a polynomial is that the neural network can be trained in $O(n)$ time, and the polynomial cannot. The difference between a polynomial and a neural network is that the polynomial can be understood, and the neural network cannot — yet."_
> — paraphrasing Barron (1993) and the mechanistic interpretability community

## Overview

Neural networks are parameterised compositions of affine transformations and pointwise nonlinearities. Every feedforward layer computes $\mathbf{h}^{[l]} = \sigma(W^{[l]}\mathbf{h}^{[l-1]} + \mathbf{b}^{[l]})$ — a formula simple enough to fit in a tweet, yet whose iterated application underlies GPT-4, AlphaFold, and Stable Diffusion. The mathematical challenge is not writing down the formula but understanding why gradient descent finds good solutions in these vast non-convex landscapes, why depth is exponentially more efficient than width for certain function classes, and why 100-billion-parameter models trained on finite data generalise at all.

This section develops the full mathematical theory from scratch. We begin with universal approximation (what can a network represent?) and depth separation (why is depth better than width?), then derive backpropagation as reverse-mode automatic differentiation on a directed acyclic graph. We analyse weight initialisation through variance propagation, study normalisation layers as gradient preconditioners, and dissect residual connections as the architectural intervention that changed products of Jacobians into sums. The final sections connect to modern AI practice: the neural tangent kernel regime, LoRA fine-tuning, scaling laws, and the mechanistic interpretability programme that seeks to reverse-engineer learned representations.

The treatment assumes you have completed the Linear Models section (§14-01) — neural network layers are linear maps with shared architectural constraints, and many regularisation and optimisation ideas carry over directly.

## Prerequisites

- **Linear models** (§14-01): linear maps, ridge regression, logistic regression, NTK introduction
- **Linear algebra** (Chapters 02–03): matrix multiplication, eigendecomposition, SVD, norms
- **Calculus and matrix calculus** (Chapter 04): partial derivatives, Jacobians, chain rule
- **Probability** (Chapter 06): Gaussian, KL divergence, softmax, cross-entropy
- **Optimisation** (Chapter 08): gradient descent, convexity, saddle points, convergence rates

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive derivations: activation function zoo, backprop from scratch, He init variance experiment, Adam dynamics, BatchNorm train/eval gap, residual gradient flow, NTK, double descent |
| [exercises.ipynb](exercises.ipynb) | 10 graded problems: UAT width bound, backprop by hand, Xavier derivation, Adam update, dropout ensemble, BatchNorm affine, residual Jacobian, NTK kernel matrix, LoRA gradient analysis, scaling laws |

## Learning Objectives

After completing this section, you will:

- Prove the Cybenko (1989) universal approximation theorem and state its limitations
- Explain depth separation: why some functions require exponentially wider shallow networks
- Derive backpropagation as reverse-mode AD and prove its $O(\text{forward pass})$ gradient cost
- Derive Xavier and He initialisations from forward/backward variance propagation conditions
- Distinguish MSE/MAE/cross-entropy losses by their probabilistic generative models
- Derive the Adam update and prove why bias correction is necessary in early iterations
- Explain dropout as approximate ensemble averaging and MC dropout for uncertainty estimation
- Prove that residual connections convert products of Jacobians to sums, resolving vanishing gradients
- Derive the neural tangent kernel and explain the lazy training vs feature learning dichotomy
- Connect MLP sublayers, LoRA, mechanistic interpretability, and scaling laws to the core theory

---

## Table of Contents

- [1. Intuition and Motivation](#1-intuition-and-motivation)
  - [1.1 What a Neural Network Actually Is](#11-what-a-neural-network-actually-is)
  - [1.2 Why Depth? The Hierarchy Argument](#12-why-depth-the-hierarchy-argument)
  - [1.3 Neural Networks Inside Modern LLMs](#13-neural-networks-inside-modern-llms)
  - [1.4 Historical Timeline 1943–2024](#14-historical-timeline-19432024)
- [2. Network Architecture as Composed Maps](#2-network-architecture-as-composed-maps)
  - [2.1 MLP Formal Definition](#21-mlp-formal-definition)
  - [2.2 Computation Graph as DAG](#22-computation-graph-as-dag)
  - [2.3 Activation Functions: Zoo and Properties](#23-activation-functions-zoo-and-properties)
  - [2.4 Function Classes Representable by MLPs](#24-function-classes-representable-by-mlps)
- [3. Universal Approximation Theory](#3-universal-approximation-theory)
  - [3.1 Cybenko 1989: Width-∞ Single Hidden Layer](#31-cybenko-1989-width--single-hidden-layer)
  - [3.2 Barron 1993: Function Class and Approximation Rate](#32-barron-1993-function-class-and-approximation-rate)
  - [3.3 Depth Separation: Telgarsky 2016](#33-depth-separation-telgarsky-2016)
  - [3.4 Practical Implications](#34-practical-implications)
- [4. Backpropagation and the Chain Rule](#4-backpropagation-and-the-chain-rule)
  - [4.1 Reverse-Mode Autodiff on a DAG](#41-reverse-mode-autodiff-on-a-dag)
  - [4.2 Layer-Wise Delta Equations](#42-layer-wise-delta-equations)
  - [4.3 Jacobian Matrix View](#43-jacobian-matrix-view)
  - [4.4 Vectorised Backprop and Batching](#44-vectorised-backprop-and-batching)
  - [4.5 Numerical Gradient Checking](#45-numerical-gradient-checking)
- [5. Weight Initialisation](#5-weight-initialisation)
  - [5.1 Signal Propagation at Initialisation](#51-signal-propagation-at-initialisation)
  - [5.2 Xavier/Glorot Initialisation](#52-xavierglot-initialisation)
  - [5.3 He (Kaiming) Initialisation](#53-he-kaiming-initialisation)
  - [5.4 Orthogonal Initialisation](#54-orthogonal-initialisation)
  - [5.5 Mean-Field Theory of Initialisation](#55-mean-field-theory-of-initialisation)
- [6. Loss Functions](#6-loss-functions)
  - [6.1 MSE and MAE — Geometry and Robustness](#61-mse-and-mae--geometry-and-robustness)
  - [6.2 Cross-Entropy and KL Divergence](#62-cross-entropy-and-kl-divergence)
  - [6.3 Contrastive and Metric Losses](#63-contrastive-and-metric-losses)
  - [6.4 Autoregressive Language Modelling Loss](#64-autoregressive-language-modelling-loss)
- [7. Gradient Descent and Optimisers](#7-gradient-descent-and-optimisers)
  - [7.1 SGD and Mini-Batch Gradient Descent](#71-sgd-and-mini-batch-gradient-descent)
  - [7.2 Momentum and Nesterov](#72-momentum-and-nesterov)
  - [7.3 Adaptive Methods: AdaGrad, RMSProp](#73-adaptive-methods-adagrad-rmsprop)
  - [7.4 Adam: Derivation and Bias Correction](#74-adam-derivation-and-bias-correction)
  - [7.5 AdamW and Decoupled Weight Decay](#75-adamw-and-decoupled-weight-decay)
  - [7.6 Learning Rate Schedules](#76-learning-rate-schedules)
  - [7.7 Loss Landscape Geometry](#77-loss-landscape-geometry)
- [8. Regularisation](#8-regularisation)
  - [8.1 Weight Decay and Its Spectrum](#81-weight-decay-and-its-spectrum)
  - [8.2 Dropout: Bernoulli Masking and Inference](#82-dropout-bernoulli-masking-and-inference)
  - [8.3 Data Augmentation as Implicit Regularisation](#83-data-augmentation-as-implicit-regularisation)
  - [8.4 Implicit Regularisation of SGD](#84-implicit-regularisation-of-sgd)
- [9. Normalisation Layers](#9-normalisation-layers)
  - [9.1 Batch Normalisation: Derivation and Train/Test Gap](#91-batch-normalisation-derivation-and-traintest-gap)
  - [9.2 Layer Normalisation](#92-layer-normalisation)
  - [9.3 RMSNorm](#93-rmsnorm)
  - [9.4 Normalisation as Signal Propagation Control](#94-normalisation-as-signal-propagation-control)
- [10. Residual Connections and Modern Architectures](#10-residual-connections-and-modern-architectures)
  - [10.1 Residual Networks: Skip Connection Identity](#101-residual-networks-skip-connection-identity)
  - [10.2 Gradient Analysis: Sum vs Product of Jacobians](#102-gradient-analysis-sum-vs-product-of-jacobians)
  - [10.3 Pre-Norm vs Post-Norm](#103-pre-norm-vs-post-norm)
  - [10.4 Depth Efficiency: Why Residuals Enable 1000+ Layers](#104-depth-efficiency-why-residuals-enable-1000-layers)
- [11. Gradient Flow and Training Dynamics](#11-gradient-flow-and-training-dynamics)
  - [11.1 Vanishing and Exploding Gradients](#111-vanishing-and-exploding-gradients)
  - [11.2 Gradient Clipping](#112-gradient-clipping)
  - [11.3 Jacobian Spectral Analysis Across Layers](#113-jacobian-spectral-analysis-across-layers)
  - [11.4 Grokking and Delayed Generalisation](#114-grokking-and-delayed-generalisation)
- [12. Expressivity and Depth](#12-expressivity-and-depth)
  - [12.1 Depth Separation Results](#121-depth-separation-results)
  - [12.2 Linear Regions and ReLU Networks](#122-linear-regions-and-relu-networks)
  - [12.3 Circuit Complexity and Boolean Functions](#123-circuit-complexity-and-boolean-functions)
  - [12.4 Over-Parameterisation and Implicit Bias](#124-over-parameterisation-and-implicit-bias)
- [13. Neural Tangent Kernel](#13-neural-tangent-kernel)
  - [13.1 Infinite-Width Limit and Kernel Ridge Regression](#131-infinite-width-limit-and-kernel-ridge-regression)
  - [13.2 Lazy Training vs Feature Learning](#132-lazy-training-vs-feature-learning)
  - [13.3 NTK Spectrum and Convergence Rate](#133-ntk-spectrum-and-convergence-rate)
  - [13.4 Beyond NTK: μP and Feature Learning](#134-beyond-ntk-p-and-feature-learning)
- [14. Deep Learning Connections to Modern AI](#14-deep-learning-connections-to-modern-ai)
  - [14.1 MLP Sublayers in Transformers](#141-mlp-sublayers-in-transformers)
  - [14.2 LoRA Revisited: Full Neural-Network Perspective](#142-lora-revisited-full-neural-network-perspective)
  - [14.3 Mechanistic Interpretability](#143-mechanistic-interpretability)
  - [14.4 Scaling Laws and Chinchilla](#144-scaling-laws-and-chinchilla)
- [15. Common Mistakes](#15-common-mistakes)
- [16. Exercises](#16-exercises)
- [17. Why This Matters for AI (2026 Perspective)](#17-why-this-matters-for-ai-2026-perspective)
- [Conceptual Bridge](#conceptual-bridge)
- [References](#references)

---

## 1. Intuition and Motivation

### 1.1 What a Neural Network Actually Is

Strip away the hype and a neural network is a **parameterised function** $f: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^{d_{\text{out}}}$ built by composing two kinds of operations: affine maps $\mathbf{z} = W\mathbf{x} + \mathbf{b}$ and pointwise nonlinearities $\mathbf{a} = \sigma(\mathbf{z})$. A depth-$L$ feedforward network computes:

$$f(\mathbf{x}; \boldsymbol{\theta}) = W^{[L]}\sigma\!\left(W^{[L-1]}\sigma\!\left(\cdots\sigma\!\left(W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}\right)\cdots\right) + \mathbf{b}^{[L-1]}\right) + \mathbf{b}^{[L]}$$

where $\boldsymbol{\theta} = \{W^{[l]}, \mathbf{b}^{[l]}\}_{l=1}^L$ are the learnable parameters. Without the nonlinearity $\sigma$, any composition of affine maps collapses to a single affine map — the depth would be meaningless. The nonlinearity is what makes depth productive.

**Why not polynomials or Fourier bases?** Any universal approximator must handle the curse of dimensionality — representing a function on $[0,1]^d$ to precision $\epsilon$ requires $O(\epsilon^{-d})$ basis functions for most classical bases. Neural networks escape this (partially) because they learn the representation: instead of pre-specifying a basis, they adapt the basis to the data. This is the key distinction. A polynomial approximator uses fixed monomials; a neural network learns which "features" to extract and then uses those features linearly.

**The two-part structure.** Every network separates into (1) a **feature extractor** $\phi: \mathbb{R}^{d_{\text{in}}} \to \mathbb{R}^k$ (all layers except the last) and (2) a **linear readout** $W^{[L]}\phi(\mathbf{x})$. This separation is not merely conceptual — it is operationally important. When you fine-tune a pretrained model by freezing the backbone and retraining the head, you are explicitly exploiting this two-part structure. When you train a linear probe on frozen representations, you are measuring the quality of $\phi$ independently of the readout. The linear models of §14-01 are precisely the readout layer; neural networks add learned features.

**For AI:** Every transformer block contains two affine maps separated by a nonlinearity: the attention sublayer (linear mixing of values) and the FFN sublayer ($\text{FFN}(\mathbf{x}) = W_2 \, \text{GELU}(W_1 \mathbf{x})$). GPT-4 stacks 96 such blocks. The mathematical foundations developed here apply identically to each block.

### 1.2 Why Depth? The Hierarchy Argument

Consider recognising a face in an image. Pixels encode individual intensity values — meaningless in isolation. Edges arise from spatial differences of pixel pairs. Curves arise from aligned sequences of edges. Eyes, noses, and mouths arise from specific configurations of curves. A face arises from the spatial arrangement of these parts. This is a **compositional hierarchy** that requires $O(\log n)$ layers to represent but $O(n)$ neurons per layer — a depth-logarithmic representation that would require exponentially wider shallow networks.

This intuition is formalised by **depth separation results** (§3.3, §12.1). The canonical example (Telgarsky, 2016): the function $f_k(x)$ defined by iterating the hat function $k$ times — $f_1(x) = \max(2x, 2-2x) \cdot \mathbf{1}_{[0,1]}(x)$, $f_k = f_1 \circ f_{k-1}$ — can be computed by a depth-$k$ network with $O(1)$ neurons per layer. Any depth-2 network computing $f_k$ to $\epsilon$-precision (for $\epsilon < 1/3$) requires $\Omega(2^k)$ neurons. Depth buys an **exponential compression** for functions with recursive structure.

**For practical deep learning:** The 7-billion parameter LLaMA-3 model uses 32 transformer layers rather than a single wide layer for precisely this reason — not because of any hard theorem but because hierarchical composition empirically learns better representations per unit of compute. Depth allows different layers to specialise: early layers in language models represent syntactic features (part-of-speech, phrase boundaries), middle layers represent semantic features (entity type, argument structure), and late layers represent task-specific features (answer type, factual associations).

### 1.3 Neural Networks Inside Modern LLMs

Every modern large language model contains neural networks in multiple roles:

**Embedding tables** are lookup matrices $E \in \mathbb{R}^{V \times d}$ where $V$ is vocabulary size. The embedding $E_{v,:}$ is simply the $v$-th row — a linear map from a one-hot vector. These are the first and last linear layers of any LLM.

**MLP sublayers** (FFN blocks) in transformers follow $\text{FFN}(\mathbf{x}) = W_2 \, \text{GELU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$ with $W_1 \in \mathbb{R}^{4d \times d}$, $W_2 \in \mathbb{R}^{d \times 4d}$. The factor-of-4 expansion is a design convention from the original Transformer — making the hidden dimension $4d$ keeps $W_1^\top W_2$ approximately rank-$d$, analogous to a low-rank bottleneck. Geva et al. (2021) showed these FFN layers act as **key-value memories**: the rows of $W_1$ act as keys, the rows of $W_2^\top$ act as values, and GELU selects which memories activate for a given input.

**Probing classifiers** are linear models trained atop frozen network activations $\phi(\mathbf{x})$ to test whether specific linguistic or factual properties are linearly decodable. They are literally logistic regression (§14-01, §8) applied to neural features — bridging the two sections of this chapter.

**LoRA adapters** fine-tune a frozen weight matrix $W_0 \in \mathbb{R}^{m \times n}$ by learning $\Delta W = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$. This is directly the low-rank OLS analysis from §14-01, applied to each weight matrix update independently.

### 1.4 Historical Timeline 1943–2024

```
NEURAL NETWORK HISTORY
════════════════════════════════════════════════════════════════════════

  1943  McCulloch & Pitts — threshold logic neuron, Boolean computation
  1957  Rosenblatt — Perceptron algorithm, convergence theorem
  1969  Minsky & Papert — XOR impossibility for single-layer networks
  1986  Rumelhart, Hinton & Williams — backpropagation rediscovery
  1989  Cybenko — universal approximation theorem (width)
  1991  Hochreiter — vanishing gradient problem identified
  1993  Barron — approximation rates for neural networks
  1997  Hochreiter & Schmidhuber — LSTM gates solve vanishing gradients
  1998  LeCun et al. — LeNet, convolutional networks for vision
  2006  Hinton & Salakhutdinov — deep belief nets, layer-wise pretraining
  2009  Glorot & Bengio — Xavier initialisation, sigmoid saturation analysis
  2012  Krizhevsky et al. — AlexNet, GPU training, ImageNet breakthrough
  2014  Goodfellow et al. — dropout, Srivastava et al. formalise ensemble view
  2015  He et al. — ResNets, He initialisation, 152-layer networks
  2015  Ioffe & Szegedy — batch normalisation
  2016  Telgarsky — depth separation theorems
  2017  Vaswani et al. — Transformer, attention replaces recurrence
  2018  Jacot et al. — neural tangent kernel theory
  2019  Liu et al. — warm-up + cosine LR schedule standard practice
  2020  Bartlett et al. — benign overfitting, double descent theory
  2020  Brown et al. — GPT-3, 175B parameters, few-shot learning
  2022  Hu et al. — LoRA low-rank fine-tuning
  2022  Power et al. — grokking: delayed generalisation phenomenon
  2023  LeCun et al. — μP maximal update parametrisation
  2024  Elhage et al. — superposition and polysemanticity, circuits

════════════════════════════════════════════════════════════════════════
```


---

## 2. Network Architecture as Composed Maps

### 2.1 MLP Formal Definition

**Definition (Multi-Layer Perceptron).** A depth-$L$ MLP with input dimension $d_0$, hidden dimensions $d_1, \ldots, d_{L-1}$, and output dimension $d_L$ is the function $f: \mathbb{R}^{d_0} \to \mathbb{R}^{d_L}$ defined by:

$$\mathbf{h}^{[0]} = \mathbf{x}, \qquad \mathbf{z}^{[l]} = W^{[l]}\mathbf{h}^{[l-1]} + \mathbf{b}^{[l]}, \qquad \mathbf{h}^{[l]} = \sigma(\mathbf{z}^{[l]}), \quad l = 1, \ldots, L-1$$
$$f(\mathbf{x}) = \mathbf{h}^{[L]} = W^{[L]}\mathbf{h}^{[L-1]} + \mathbf{b}^{[L]}$$

where $W^{[l]} \in \mathbb{R}^{d_l \times d_{l-1}}$ and $\mathbf{b}^{[l]} \in \mathbb{R}^{d_l}$ are learnable parameters. The final layer applies no activation (for regression) or softmax (for classification).

**Parameter count.** The total number of parameters is:

$$|\boldsymbol{\theta}| = \sum_{l=1}^{L} d_l(d_{l-1} + 1) = \sum_{l=1}^{L} (d_l d_{l-1} + d_l)$$

For a 4-layer MLP with widths $[768, 3072, 3072, 768]$ (a single transformer FFN block with $d=768$): $(768 \cdot 3072 + 3072) + (3072 \cdot 768 + 768) \approx 4.7\text{M}$ parameters. A 96-layer GPT-4 architecture at $d=12288$ has FFN blocks alone contributing $\sim 150\text{B}$ parameters.

**Notation conventions (from `NOTATION_GUIDE.md`):**
- $\mathbf{x} \in \mathbb{R}^{d_0}$: input vector (bold lowercase)
- $W^{[l]}$: weight matrix at layer $l$ (uppercase, square bracket for layer index)
- $\mathbf{h}^{[l]}$: post-activation hidden state at layer $l$ (bold lowercase, square bracket)
- $\mathbf{z}^{[l]}$: pre-activation (logit) at layer $l$

### 2.2 Computation Graph as DAG

Any computation can be represented as a **directed acyclic graph (DAG)** where:
- **Nodes** are variables (scalars, vectors, matrices)
- **Edges** represent functional dependencies: an edge from $u$ to $v$ means $v = g(u, \ldots)$

For an MLP, the forward pass traces a path $\mathbf{x} \to \mathbf{z}^{[1]} \to \mathbf{h}^{[1]} \to \cdots \to \mathbf{z}^{[L]} \to \mathcal{L}$. The **topological order** of the DAG determines the execution order: every node is computed only after all its inputs are available.

```
COMPUTATION GRAPH (2-layer MLP)
════════════════════════════════════════════════════════════════════════

  FORWARD PASS (left to right):

  x ──[W¹,b¹]──► z¹ ──[σ]──► h¹ ──[W²,b²]──► z² ──[σ]──► ŷ ──[L]──► ℒ

  BACKWARD PASS (right to left, chain rule):

  ℒ ◄──[∂ℒ/∂ŷ]── ŷ ◄──[∂ŷ/∂z²]── z² ◄──[∂z²/∂h¹]── h¹ ◄──[∂h¹/∂z¹]── z¹

  Each node stores its LOCAL JACOBIAN during forward pass.
  Backward pass multiplies these Jacobians right-to-left.

════════════════════════════════════════════════════════════════════════
```

**Key insight:** The DAG representation separates the *structure* of the computation from the *values* flowing through it. PyTorch/TensorFlow build this graph dynamically during the forward pass, which is why arbitrary Python control flow (if statements, loops) is supported — the graph is just whatever operations were actually executed.

**For AI:** Transformer architectures have more complex DAGs than simple chains — residual connections create parallel paths, and multi-head attention merges multiple branches. The backward pass through these complex graphs follows the same principle: sum gradients arriving at a node from multiple outgoing edges.

### 2.3 Activation Functions: Zoo and Properties

The activation function $\sigma: \mathbb{R} \to \mathbb{R}$ is applied pointwise: $\sigma(\mathbf{z})_i = \sigma(z_i)$. The choice of activation profoundly affects (1) gradient flow, (2) expressivity, (3) biological plausibility (sometimes), and (4) training stability.

| Activation | Formula | Derivative | Range | Issues |
|---|---|---|---|---|
| Sigmoid | $1/(1+e^{-z})$ | $\sigma(z)(1-\sigma(z))$ | $(0,1)$ | Saturates at $\pm\infty$, vanishing gradients |
| Tanh | $(e^z - e^{-z})/(e^z+e^{-z})$ | $1 - \tanh^2(z)$ | $(-1,1)$ | Saturates at $\pm\infty$, zero-centred |
| ReLU | $\max(0,z)$ | $\mathbf{1}[z>0]$ | $[0,\infty)$ | Dying ReLU (negative half always zero) |
| Leaky ReLU | $\max(\alpha z, z)$ | $\alpha$ if $z<0$ else $1$ | $\mathbb{R}$ | Fixes dying ReLU, adds hyperparameter |
| ELU | $e^z - 1$ if $z<0$ else $z$ | smooth | $(-1,\infty)$ | Smooth negative, mean-zero activations |
| GELU | $z\Phi(z)$ | $\Phi(z) + z\phi(z)$ | $\mathbb{R}$ | Default in BERT, GPT; smooth gating |
| SiLU/Swish | $z \cdot \sigma(z)$ | $\sigma(z)(1 + z(1-\sigma(z)))$ | $\mathbb{R}$ | Non-monotone, used in LLaMA |
| Mish | $z\tanh(\ln(1+e^z))$ | complex | $\mathbb{R}$ | Smooth, non-monotone, EfficientNet |

**ReLU geometry.** $\text{ReLU}(z) = \max(0,z)$ partitions the input space into two regions: $\{z>0\}$ (identity, gradient 1) and $\{z\leq 0\}$ (zero output, gradient 0). A network with ReLU is **piecewise linear** — the function is linear on each polyhedral region of the partition, and the number of regions grows exponentially with depth (§12.2). This piecewise-linear structure makes ReLU networks analytically tractable and computationally efficient (no exponential evaluation).

**Dying ReLU.** If a neuron's pre-activation is always negative (due to a large negative bias or destructive weight updates), $\text{ReLU}(z) = 0$ always and $\partial\text{ReLU}/\partial z = 0$ always. The neuron never receives a gradient and can never recover. Leaky ReLU ($\alpha = 0.01$ or $0.1$) and ELU solve this by allowing a small negative output.

**GELU.** The Gaussian Error Linear Unit $\text{GELU}(z) = z \cdot \Phi(z)$ where $\Phi$ is the standard normal CDF can be interpreted as a **stochastic gating**: in expectation, each input is scaled by the probability that a standard normal exceeds $z$, effectively a soft version of ReLU that allows smooth gradients everywhere. BERT, GPT-2/3/4, and LLaMA all use GELU or SiLU.

**Approximation $\text{GELU}(z) \approx 0.5z(1 + \tanh[\sqrt{2/\pi}(z + 0.044715z^3)])$** is used in practice to avoid the expensive CDF evaluation.

### 2.4 Function Classes Representable by MLPs

With no nonlinearity: MLPs represent **affine functions** $f(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$ — composition of affine maps collapses.

With ReLU: MLPs represent **continuous piecewise-linear (CPWL) functions**. This class includes all affine functions, all indicator functions of polyhedral sets, and all splines with polyhedral knots. The number of pieces grows as $O((n/d)^{(L-1)d}n^d)$ for width-$n$, depth-$L$ networks in $\mathbb{R}^d$ (Montufar et al., 2014).

With sigmoid or tanh: MLPs represent **smooth nonlinear functions**. The universal approximation theorem (§3) states that these can approximate any continuous function to arbitrary precision.

**Non-representable functions:** Discontinuous functions, functions with strict discontinuities, or highly oscillatory functions require impractically wide/deep networks. No finite network exactly represents $\sin(nx)$ for large $n$ — it merely approximates it with some error.

**For AI:** The piecewise-linear structure of ReLU networks means transformers with ReLU FFNs partition the token embedding space into polyhedral regions, with the same linear map applied to all tokens in a region. GELU/SiLU networks are smooth versions of this: they softly interpolate between identity and zero rather than hard switching.


---

## 3. Universal Approximation Theory

### 3.1 Cybenko 1989: Width-∞ Single Hidden Layer

**Theorem (Cybenko, 1989).** Let $\sigma: \mathbb{R} \to \mathbb{R}$ be any continuous **discriminatory** function (defined below). Then for any $f \in C([0,1]^d)$ (continuous functions on the unit hypercube) and any $\epsilon > 0$, there exist $m \in \mathbb{N}$, weights $\{v_i, w_i, b_i\}$ such that:

$$\left|f(\mathbf{x}) - \sum_{i=1}^m v_i \, \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)\right| < \epsilon \quad \text{for all } \mathbf{x} \in [0,1]^d$$

**Definition (Discriminatory).** $\sigma$ is discriminatory if for any finite signed measure $\mu$ on $[0,1]^d$:
$$\int_{[0,1]^d} \sigma(\mathbf{w}^\top \mathbf{x} + b)\, d\mu(\mathbf{x}) = 0 \text{ for all } \mathbf{w}, b \implies \mu = 0$$

Sigmoidal functions (including logistic sigmoid and tanh) are discriminatory. ReLU is also discriminatory.

**Proof sketch (via Hahn-Banach).** Consider the subspace $S \subseteq C([0,1]^d)$ spanned by $\{\sigma(\mathbf{w}^\top \mathbf{x} + b)\}$. Suppose $S$ is not dense — by Hahn-Banach, there exists a continuous linear functional (equivalently, a finite signed measure $\mu \neq 0$) that annihilates $S$. But the discriminatory property means no such $\mu$ exists. Contradiction. Therefore $S$ is dense in $C([0,1]^d)$. $\square$

**What the theorem guarantees:** For any target function and tolerance, a shallow network exists that approximates it. The network may need $m \to \infty$ neurons.

**What the theorem does NOT guarantee:**
1. The **width needed** for a given $\epsilon$ and $f$ (may be exponential in $d$)
2. Whether **gradient descent** finds the approximating weights
3. Whether the approximation **generalises** from $[0,1]^d$ to nearby points
4. **Depth** efficiency — a shallow network may need exponentially more neurons than a deep one

**Hornik (1991) extension:** The theorem holds for any non-polynomial continuous activation function — including ReLU, GELU, and SiLU.

### 3.2 Barron 1993: Function Class and Approximation Rate

Cybenko's theorem tells us a shallow network *exists* but not *how wide* it needs to be. Barron (1993) gave an explicit rate by defining a function class with controlled spectral content.

**Definition (Barron class).** A function $f: \mathbb{R}^d \to \mathbb{R}$ belongs to the **Barron class** $\mathcal{B}_C$ if it has a Fourier representation $f(\mathbf{x}) = \int e^{i\boldsymbol{\omega}^\top\mathbf{x}}\hat{f}(\boldsymbol{\omega})\,d\boldsymbol{\omega}$ with finite spectral norm:

$$C_f := \int_{\mathbb{R}^d} \|\boldsymbol{\omega}\| |\hat{f}(\boldsymbol{\omega})|\, d\boldsymbol{\omega} < \infty$$

**Theorem (Barron, 1993).** For any $f \in \mathcal{B}_C$ and any distribution $\mu$ on inputs, there exists an $m$-neuron single-hidden-layer network $f_m$ such that:

$$\int \left(f(\mathbf{x}) - f_m(\mathbf{x})\right)^2 d\mu(\mathbf{x}) \leq \frac{C_f^2}{m}$$

This is an **$O(1/m)$ approximation rate independent of dimension $d$** — neural networks escape the curse of dimensionality for Barron-class functions. Classical nonparametric methods (kernel smoothing, polynomials) achieve only $O(m^{-2/d})$ — exponentially worse in high dimensions.

**Intuition:** Barron class functions have band-limited spectral content — they do not oscillate wildly at high frequencies. For such functions, a single neuron "captures" a chunk of spectral content proportional to its weight, so $m$ neurons collectively achieve $O(1/m)$ error.

**Relevance to deep learning:** Real-world functions encountered in language modelling (predicting the next word) appear to have controlled spectral content — empirically, neural networks achieve much better than the classical non-parametric rate. Barron's result is the theoretical backing.

### 3.3 Depth Separation: Telgarsky 2016

**Theorem (Telgarsky, 2016).** For any $k, p \geq 1$, define $f_k: [0,1] \to [0,1]$ as the $k$-fold composition of the hat function $t(x) = 1 - |2x - 1|$: $f_k = t^{(k)}$. Then:

1. $f_k$ can be computed by a ReLU network of depth $O(k)$ and width $O(1)$ (specifically, width 2).
2. Any ReLU network of depth $\leq k/2$ that approximates $f_k$ with $L^1([0,1])$ error $< 1/3$ must have width $\geq 2^{k/2}$ neurons.

This gives a **depth-exponential-width tradeoff**: going from depth $k$ to depth $k/2$ forces width to grow from $O(1)$ to $\Omega(2^{k/2})$ — exponential cost for cutting depth in half.

**Why the hat function?** $t^{(k)}$ is a "triangle wave" with $2^k$ oscillations on $[0,1]$. A depth-$k$ recursive composition makes $2^k$ oscillations with only $O(k)$ parameters by reusing the same pattern. Any single-layer network must independently represent each oscillation — requiring width $\Omega(2^k)$.

**Extensions:**
- **Montufar et al. (2014):** Lower bound on the number of linear regions of a ReLU network: $\Omega\left(\lfloor n/d \rfloor^{(L-1)d} n^d\right)$ for width $n$, depth $L$, input dim $d$.
- **Eldan & Shamir (2016):** A specific radially symmetric function in $\mathbb{R}^d$ requires width $\Omega(d)$ for depth-2 networks but only $O(1)$ for depth-3 networks.
- **Cohen et al. (2016):** Sum-product networks and tensor decompositions — connections between depth and tensor rank.

### 3.4 Practical Implications

**Depth is efficient for hierarchical functions.** If the target function $f$ decomposes as $f = g_L \circ g_{L-1} \circ \cdots \circ g_1$ where each $g_l$ is simple (low-complexity), then a depth-$L$ network represents $f$ with $O(Lk)$ parameters where $k$ is the complexity of each $g_l$. A depth-2 network requires $O(k^L)$ parameters — exponential blow-up.

**Width is efficient for non-hierarchical functions.** For functions that do not have recursive structure (e.g., a radial function $f(\mathbf{x}) = g(\|\mathbf{x}\|)$ for smooth $g$), depth beyond 2 provides minimal benefit and only width helps.

**The theorem gap.** Universal approximation theorems say: *there exists* a network that approximates $f$. They do not say: *gradient descent will find it*. This gap — between approximation theory and optimisation theory — is where most of the interesting open questions in deep learning live. The NTK (§13) partially bridges this gap by showing that in the infinite-width limit, gradient descent converges to a kernel regression solution.

**Table: Approximation theorems summary**

| Theorem | What it proves | Width needed | Depth |
|---|---|---|---|
| Cybenko (1989) | Dense in $C([0,1]^d)$ | Unbounded | 1 hidden layer |
| Hornik (1991) | Same for non-polynomial $\sigma$ | Unbounded | 1 hidden layer |
| Barron (1993) | $O(1/m)$ for $\mathcal{B}_C$ class | $m$ for $O(1/m)$ error | 1 hidden layer |
| Telgarsky (2016) | Depth exp saves width | $\Omega(2^{k/2})$ for depth-$k/2$ | Exponential savings |
| Montufar (2014) | Linear regions count | Polynomial | Exponential growth |


---

## 4. Backpropagation and the Chain Rule

### 4.1 Reverse-Mode Autodiff on a DAG

Backpropagation is **reverse-mode automatic differentiation** applied to the DAG of a neural network. The key theorem that makes it efficient:

**Theorem (gradient cost = forward pass cost).** Let $f: \mathbb{R}^n \to \mathbb{R}$ be computed by a DAG with $T$ operations. Then all $n$ partial derivatives $\partial f / \partial x_i$ can be computed in $O(T)$ operations — the same cost as one forward pass, regardless of $n$.

This is remarkable: computing $\nabla_{\boldsymbol{\theta}} \mathcal{L}$ for a network with 175 billion parameters costs the same as one forward pass. The entire feasibility of training modern LLMs rests on this theorem.

**Proof of the theorem.** Assign each intermediate variable $v_i$ an **adjoint** $\bar{v}_i = \partial f / \partial v_i$. Compute adjoints in reverse topological order. For each operation $v_j = g(v_{i_1}, \ldots, v_{i_k})$:

$$\bar{v}_{i_m} \mathrel{+}= \bar{v}_j \cdot \frac{\partial g}{\partial v_{i_m}}\bigg|_{v_{i_1},\ldots,v_{i_k}}$$

Each operation contributes a constant number of multiplications to each input's adjoint, so total cost is $O(T)$. $\square$

**Forward mode vs reverse mode.** Forward-mode AD computes directional derivatives $\nabla_{\boldsymbol{\theta}} f \cdot \mathbf{v}$ for one direction $\mathbf{v}$ per forward pass — efficient for $f: \mathbb{R}^n \to \mathbb{R}^m$ when $n \ll m$. Reverse-mode AD computes $\nabla_{\boldsymbol{\theta}} f$ for a scalar $f$ in one backward pass — efficient when $n \gg m = 1$, which is exactly the neural network training setting ($n = |\boldsymbol{\theta}| \gg 1$).

### 4.2 Layer-Wise Delta Equations

For an MLP with loss $\mathcal{L}$, define the **error signal** (delta) at layer $l$:

$$\boldsymbol{\delta}^{[l]} := \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} \in \mathbb{R}^{d_l}$$

**Output layer** ($l = L$, no activation, MSE loss $\mathcal{L} = \frac{1}{2}\|\mathbf{h}^{[L]} - \mathbf{y}\|^2$):

$$\boldsymbol{\delta}^{[L]} = \mathbf{h}^{[L]} - \mathbf{y}$$

For cross-entropy with softmax output: $\boldsymbol{\delta}^{[L]} = \hat{\mathbf{p}} - \mathbf{y}$ (the clean gradient that makes softmax + cross-entropy the canonical output combination).

**Hidden layer backprop** (for $l = L-1, \ldots, 1$):

$$\boldsymbol{\delta}^{[l]} = \left(W^{[l+1]\top} \boldsymbol{\delta}^{[l+1]}\right) \odot \sigma'(\mathbf{z}^{[l]})$$

This is the chain rule: gradients flow backwards through the transpose weight matrix, then are modulated by the activation derivative (element-wise product $\odot$).

**Weight and bias gradients:**

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{h}^{[l-1]})^\top \in \mathbb{R}^{d_l \times d_{l-1}}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \boldsymbol{\delta}^{[l]} \in \mathbb{R}^{d_l}$$

**Memory budget.** Backpropagation requires storing all intermediate activations $\mathbf{h}^{[l]}$ from the forward pass to compute $\boldsymbol{\delta}^{[l]}$. This is why training a 70B-parameter model requires far more GPU memory than inference: you must store activations for every layer simultaneously. Gradient checkpointing trades compute for memory by recomputing activations during the backward pass rather than storing them.

### 4.3 Jacobian Matrix View

For a layer $l$, the full Jacobian $J^{[l]} = \partial \mathbf{h}^{[l]} / \partial \mathbf{h}^{[l-1]} \in \mathbb{R}^{d_l \times d_{l-1}}$ is:

$$J^{[l]} = \text{diag}(\sigma'(\mathbf{z}^{[l]})) \cdot W^{[l]}$$

The product of all layer Jacobians gives the total Jacobian from input to output:

$$\frac{\partial \mathbf{h}^{[L]}}{\partial \mathbf{h}^{[0]}} = J^{[L]} J^{[L-1]} \cdots J^{[1]}$$

This **product of Jacobians** is the source of vanishing/exploding gradients (§11.1): if each $\|J^{[l]}\| < 1$, the product shrinks exponentially; if each $\|J^{[l]}\| > 1$, it grows exponentially.

**For sigmoid:** $\sigma'(z) = \sigma(z)(1-\sigma(z)) \leq 0.25$, so $\|\text{diag}(\sigma'(\mathbf{z}^{[l]}))\|_\infty \leq 0.25$. Deep sigmoid networks will always have vanishing gradients unless weights are carefully large.

**For ReLU:** $\sigma'(z) \in \{0, 1\}$, so $J^{[l]}$ is a sub-matrix of $W^{[l]}$ (rows zeroed where neuron is inactive). The spectral radius depends entirely on the weight matrix spectrum.

### 4.4 Vectorised Backprop and Batching

For a mini-batch of $B$ samples, forward pass produces $Z^{[l]} \in \mathbb{R}^{d_l \times B}$ and $H^{[l]} \in \mathbb{R}^{d_l \times B}$:

$$Z^{[l]} = W^{[l]} H^{[l-1]} + \mathbf{b}^{[l]}\mathbf{1}^\top$$

The batch gradient is the average over samples:

$$\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{1}{B}\Delta^{[l]} (H^{[l-1]})^\top, \qquad \Delta^{[l]} \in \mathbb{R}^{d_l \times B}$$

where $\Delta^{[l]}_{:,i} = \boldsymbol{\delta}^{[l]}$ for sample $i$. This is a **matrix-matrix product** — the dominant cost in both forward and backward passes, and why GPU parallelism is so effective.

**For AI:** The entire training of a language model is this matrix-matrix product loop, repeated for every batch and every layer. FlashAttention's efficiency gains come from fusing operations to avoid repeated reads from slow GPU memory — but the underlying computation is identical.

### 4.5 Numerical Gradient Checking

**Finite-difference gradient:** For scalar parameter $\theta_i$:

$$\left[\nabla_{\boldsymbol{\theta}}\mathcal{L}\right]_i \approx \frac{\mathcal{L}(\boldsymbol{\theta} + \epsilon \mathbf{e}_i) - \mathcal{L}(\boldsymbol{\theta} - \epsilon \mathbf{e}_i)}{2\epsilon}$$

The **centred difference** has error $O(\epsilon^2)$ vs $O(\epsilon)$ for the one-sided difference. With $\epsilon = 10^{-5}$, this achieves ~10 decimal digits of accuracy.

**Relative error threshold:**

$$\text{rel error} = \frac{\|\nabla_{\boldsymbol{\theta}}^{\text{analytic}}\mathcal{L} - \nabla_{\boldsymbol{\theta}}^{\text{numeric}}\mathcal{L}\|}{\|\nabla_{\boldsymbol{\theta}}^{\text{analytic}}\mathcal{L}\| + \|\nabla_{\boldsymbol{\theta}}^{\text{numeric}}\mathcal{L}\|}$$

- $< 10^{-7}$: Almost certainly correct
- $10^{-7}$ to $10^{-5}$: Suspicious — check activation derivative formulas
- $> 10^{-3}$: Bug in backward pass

**When gradient checking fails:**
- Kink in activation function at $z = 0$ for ReLU/Leaky-ReLU — the numerical gradient samples both sides of the kink
- Batch normalisation with running statistics (must be in training mode)
- Dropout (must use the same mask for both forward passes)


---

## 5. Weight Initialisation

### 5.1 Signal Propagation at Initialisation

Poor initialisation causes either **gradient vanishing** (signals decay to zero before reaching early layers) or **gradient explosion** (signals blow up, causing NaN losses). Proper initialisation ensures that activations and gradients have $O(1)$ variance at the start of training.

**Forward variance propagation.** At initialisation, assume weights $W^{[l]}_{ij} \sim \mathcal{N}(0, \sigma_W^2)$ i.i.d. and inputs $x_j$ with $\text{Var}(x_j) = \sigma_x^2$. The pre-activation variance is:

$$\text{Var}(z_i^{[l]}) = d_{l-1} \cdot \sigma_W^2 \cdot \text{Var}(h_j^{[l-1]})$$

For variance to be preserved across layers: $d_{l-1} \cdot \sigma_W^2 \cdot \mathbb{E}[(\sigma')^2] = 1$.

**Backward variance propagation.** Gradient $\partial\mathcal{L}/\partial h_j^{[l]}$ has variance proportional to $d_l \cdot \sigma_W^2 \cdot \text{Var}(\delta^{[l+1]})$. For variance preservation: $d_l \cdot \sigma_W^2 \cdot \mathbb{E}[(\sigma')^2] = 1$.

In general $d_{l-1} \neq d_l$, so these two conditions conflict — **Xavier** resolves this by averaging them.

### 5.2 Xavier/Glorot Initialisation

**Derivation (Glorot & Bengio, 2010).** Compromise between forward and backward variance conditions:

$$\sigma_W^2 = \frac{2}{d_{l-1} + d_l}$$

In practice this is implemented as uniform or Gaussian:
$$W^{[l]}_{ij} \sim \mathcal{U}\!\left[-\sqrt{\frac{6}{d_{l-1}+d_l}},\, \sqrt{\frac{6}{d_{l-1}+d_l}}\right] \quad \text{or} \quad W^{[l]}_{ij} \sim \mathcal{N}\!\left(0,\, \frac{2}{d_{l-1}+d_l}\right)$$

**Assumption:** The activation function has $\mathbb{E}[(\sigma')^2] \approx 1$, which holds for tanh near zero (since $\tanh'(0) = 1$) and for sigmoid near zero.

**Fails for ReLU:** ReLU kills half the pre-activations (those $< 0$), so the effective variance is halved: $\mathbb{E}[(\text{ReLU}')^2] = 0.5$. Xavier underestimates the needed variance for ReLU networks.

### 5.3 He (Kaiming) Initialisation

**Derivation (He et al., 2015).** For ReLU, the forward variance condition becomes:

$$d_{l-1} \cdot \sigma_W^2 \cdot \mathbb{E}[(\text{ReLU}')^2] = d_{l-1} \cdot \sigma_W^2 \cdot \frac{1}{2} = 1 \implies \sigma_W^2 = \frac{2}{d_{l-1}}$$

$$W^{[l]}_{ij} \sim \mathcal{N}\!\left(0,\, \frac{2}{d_{l-1}}\right)$$

The factor of 2 compensates for ReLU zeroing negative activations. This is the standard initialisation for any ReLU network.

**For Leaky ReLU** with negative slope $a$: $\mathbb{E}[(\text{LeakyReLU}')^2] = (1 + a^2)/2$, giving $\sigma_W^2 = 2/(d_{l-1}(1+a^2))$.

**Empirical validation:** With He init, the standard deviation of activations across layers is approximately 1.0 at initialisation. Without it, in a 100-layer ReLU network, activations shrink by $\sim (0.5)^{50} \approx 10^{-15}$ — numerically zero.

### 5.4 Orthogonal Initialisation

**Definition.** For a square weight matrix $W \in \mathbb{R}^{n \times n}$: sample a random matrix $A \sim \mathcal{N}(0,1)^{n \times n}$, compute its QR decomposition $A = QR$, and set $W = Q$. For rectangular matrices $W \in \mathbb{R}^{m \times n}$ with $m \neq n$: use the first $m$ columns of $Q$ or the SVD.

**Property:** An orthogonal matrix preserves the $\ell^2$ norm: $\|W\mathbf{x}\| = \|\mathbf{x}\|$. All singular values are exactly 1. This means gradients are neither amplified nor attenuated when passing through an orthogonally-initialised layer.

**Deep linear networks:** For a network $f(\mathbf{x}) = W^{[L]} W^{[L-1]} \cdots W^{[1]} \mathbf{x}$ (all linear), orthogonal initialisation ensures $\partial\mathbf{h}^{[L]}/\partial\mathbf{h}^{[0]}$ has singular values equal to 1 at init. Saxe et al. (2013) proved that orthogonally-initialised deep linear networks undergo **linear mode connectivity** — gradient flow from one point to the optimum is straight.

### 5.5 Mean-Field Theory of Initialisation

**Mean-field theory** (Poole et al., 2016; Schoenholz et al., 2017) provides a framework for understanding initialisation through the lens of statistical physics. Consider the correlation between two inputs $\mathbf{x}$ and $\mathbf{x}'$ as they propagate through a network:

$$q^{[l]} := \frac{1}{d_l}\|\mathbf{h}^{[l]}\|^2, \qquad c^{[l]} := \frac{(\mathbf{h}^{[l]})^\top (\mathbf{h}^{\prime[l]})}{d_l \sqrt{q^{[l]} q^{\prime[l]}}}$$

In the infinite-width limit, both quantities obey deterministic recursions determined by $(\sigma_W^2, \sigma_b^2)$.

**Two phases:**
- **Ordered phase** ($c^{[l]} \to 1$): Different inputs become indistinguishable — network is unable to distinguish $\mathbf{x}$ from $\mathbf{x}'$. Over-smoothing.
- **Chaotic phase** ($c^{[l]} \to 0$): Different inputs diverge exponentially — tiny differences in input lead to wildly different outputs. Unstable.

**Edge of chaos:** The boundary between phases is the initialisation where $c^{[l]}$ neither vanishes nor saturates — information about input differences propagates through arbitrary depth. Xavier and He initialisations are designed to operate at or near the edge of chaos for their respective activation functions.

**Correlation length $\xi$.** Near the edge of chaos, the correlation length diverges: $\xi \sim 1/(1 - \lambda_1)$ where $\lambda_1$ is the largest eigenvalue of the Jacobian covariance. Large $\xi$ means gradients can carry information across many layers — a necessary condition for training deep networks.


---

## 6. Loss Functions

### 6.1 MSE and MAE — Geometry and Robustness

**Mean Squared Error (MSE).** For regression with targets $\mathbf{y} \in \mathbb{R}^n$:

$$\mathcal{L}_{\text{MSE}}(\boldsymbol{\theta}) = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|^2$$

**Probabilistic interpretation:** MSE is the negative log-likelihood of a Gaussian noise model $y_i = f(\mathbf{x}^{(i)}) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$:

$$-\log p(\mathbf{y}|\mathbf{x}, \boldsymbol{\theta}) = \frac{n}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\|\mathbf{y} - \hat{\mathbf{y}}\|^2 \propto \mathcal{L}_{\text{MSE}}$$

**Geometry:** MSE penalises large residuals quadratically — an outlier with residual $10$ contributes $100$ to the loss, whereas the majority of points with residual $1$ each contribute $1$. This makes MSE **sensitive to outliers**.

**Mean Absolute Error (MAE).** $\mathcal{L}_{\text{MAE}} = \frac{1}{n}\sum_i |y_i - \hat{y}_i|$. This is the negative log-likelihood of a Laplace noise model: $\epsilon_i \sim \text{Laplace}(0, b)$.

**Huber loss** interpolates: for residual $r = y - \hat{y}$:
$$\mathcal{L}_\delta(r) = \begin{cases} \frac{1}{2}r^2 & |r| \leq \delta \\ \delta|r| - \frac{1}{2}\delta^2 & |r| > \delta \end{cases}$$

Huber is differentiable everywhere (unlike MAE at $r=0$), quadratic near zero (like MSE), and linear for large residuals (robust like MAE). Widely used in reinforcement learning (DQN, TD-learning).

### 6.2 Cross-Entropy and KL Divergence

**Binary cross-entropy (BCE).** For binary targets $y \in \{0,1\}$ and predicted probability $\hat{p} = \sigma(f(\mathbf{x}))$:

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{n}\sum_i \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

This is the MLE objective for the Bernoulli model. Gradient with respect to the logit $z = f(\mathbf{x})$: $\partial\mathcal{L}/\partial z = \hat{p} - y$ — the clean residual form that makes sigmoid + BCE the standard binary classification combination.

**Categorical cross-entropy.** For $K$-class targets $\mathbf{y} \in \{0,1\}^K$ (one-hot) and softmax output $\hat{\mathbf{p}} = \text{softmax}(\mathbf{z})$:

$$\mathcal{L}_{\text{CE}} = -\sum_k y_k \log \hat{p}_k = -\log \hat{p}_{y^*} \quad \text{(where } y^* \text{ is the true class)}$$

**KL divergence connection.** $\mathcal{L}_{\text{CE}}(y, \hat{p}) = \text{KL}(y \| \hat{p}) + H(y)$. Since $H(y)$ is constant (true labels are fixed), minimising cross-entropy = minimising KL divergence between the empirical label distribution and the predicted distribution.

**Label smoothing.** Replace one-hot target $\mathbf{y}$ with:

$$\tilde{y}_k = (1-\varepsilon) y_k + \frac{\varepsilon}{K}$$

This prevents the model from becoming overconfident (logit $\to +\infty$ for the true class). Vaswani et al. used $\varepsilon = 0.1$ in the original Transformer, and it has been standard in LLMs since. Label smoothing implicitly adds an entropy regularisation term $\varepsilon \cdot H(\hat{p})$ to the loss.

**Focal loss** (Lin et al., 2017): $\mathcal{L}_{\text{focal}} = -\alpha(1-\hat{p})^\gamma \log \hat{p}$. Down-weights easy examples (high $\hat{p}$), focuses training on hard negatives. Used in object detection (RetinaNet) and recently in language modelling when class imbalance is extreme.

### 6.3 Contrastive and Metric Losses

**Triplet loss.** Given anchor $\mathbf{a}$, positive $\mathbf{p}$ (same class), negative $\mathbf{n}$ (different class):

$$\mathcal{L}_{\text{triplet}} = \max\!\left(0,\; \|\phi(\mathbf{a}) - \phi(\mathbf{p})\|^2 - \|\phi(\mathbf{a}) - \phi(\mathbf{n})\|^2 + m\right)$$

where $m > 0$ is the margin. This trains the embedding $\phi$ so positives are closer than negatives by margin $m$.

**NT-Xent (SimCLR).** Given $N$ samples with two augmented views each ($2N$ total), the InfoNCE-style loss for sample $i$ and its positive pair $j$:

$$\mathcal{L}_{\text{NT-Xent}} = -\log \frac{\exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(\mathbf{z}_i, \mathbf{z}_k)/\tau)}$$

Temperature $\tau$ controls sharpness. InfoNCE is a lower bound on mutual information $I(\mathbf{z}_i; \mathbf{z}_j)$ — contrastive learning can be understood as mutual information maximisation.

### 6.4 Autoregressive Language Modelling Loss

**Next-token prediction.** For a sequence $\mathbf{x} = (x_1, \ldots, x_T)$, the causal language modelling (CLM) objective is:

$$\mathcal{L}_{\text{CLM}}(\boldsymbol{\theta}) = -\frac{1}{T}\sum_{t=1}^T \log p_{\boldsymbol{\theta}}(x_t | x_1, \ldots, x_{t-1})$$

By the chain rule of probability: $\log p(\mathbf{x}) = \sum_t \log p(x_t | x_{<t})$, so minimising $\mathcal{L}_{\text{CLM}}$ maximises the joint log-probability of the sequence — a maximum likelihood estimate of a causal language model.

**Bits per byte (BPB).** In practice, loss is reported in **bits per byte** rather than nats per token: $\text{BPB} = \mathcal{L}_{\text{CLM}} / \log 2 \cdot d_{\text{bytes/token}}$ where $d_{\text{bytes/token}} \approx 4$ for BPE tokenisers on English text. GPT-4 achieves approximately 0.7–0.8 BPB on English text.

**Perplexity:** $\text{PPL} = \exp(\mathcal{L}_{\text{CLM}})$. For a uniform distribution over vocabulary of size $V$: $\text{PPL} = V$. A perfect model: $\text{PPL} = 1$. Current LLMs: $\text{PPL} \approx 3$–10 on standard benchmarks.


---

## 7. Gradient Descent and Optimisers

### 7.1 SGD and Mini-Batch Gradient Descent

**Full-batch gradient descent:**

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\boldsymbol{\theta}_t)$$

**Stochastic gradient descent (SGD):** Replace full gradient with gradient on one sample or mini-batch of size $B$:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \hat{\nabla} \mathcal{L}(\boldsymbol{\theta}_t), \qquad \hat{\nabla} \mathcal{L} := \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla_{\boldsymbol{\theta}} \ell(\mathbf{x}^{(i)}, y^{(i)}; \boldsymbol{\theta}_t)$$

**Unbiasedness:** $\mathbb{E}[\hat{\nabla}\mathcal{L}] = \nabla\mathcal{L}$ (assuming i.i.d. sampling). The variance is $\text{Var}(\hat{\nabla}\mathcal{L}) \sim \sigma^2/B$ — halved by doubling batch size.

**Convergence for convex $\mathcal{L}$** (with $L$-smooth loss and step size $\eta \leq 1/L$):
$$\mathcal{L}(\boldsymbol{\theta}_T) - \mathcal{L}^* \leq \frac{\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|^2}{2\eta T} + \frac{\eta \sigma^2}{2}$$

The variance term $\eta\sigma^2/2$ does not vanish — SGD has a **noise floor** that can only be reduced by shrinking $\eta$ (slower convergence) or increasing $B$ (more memory/compute).

**Linear scaling rule (Goyal et al., 2017):** When multiplying batch size by $k$, multiply learning rate by $k$ and warmup for 5 epochs. This approximately preserves the noise-to-signal ratio and enables distributed training.

### 7.2 Momentum and Nesterov

**Heavy-ball momentum** (Polyak, 1964):

$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t - \eta \nabla\mathcal{L}(\boldsymbol{\theta}_t), \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{v}_{t+1}$$

where $\gamma \in (0,1)$ is the momentum coefficient (typically 0.9). The velocity $\mathbf{v}$ accumulates gradient history: $\mathbf{v}_t = -\eta \sum_{k=0}^{t} \gamma^{t-k} \nabla\mathcal{L}(\boldsymbol{\theta}_k)$.

**Why it helps:** In a ravine (loss surface with very different curvatures along different directions), vanilla gradient descent oscillates across the narrow direction while progressing slowly along the flat direction. Momentum accumulates velocity along the consistent flat direction, damping oscillations in the narrow direction.

**Nesterov accelerated gradient (NAG)**:

$$\boldsymbol{\theta}_t^{\text{look-ahead}} = \boldsymbol{\theta}_t + \gamma \mathbf{v}_t$$
$$\mathbf{v}_{t+1} = \gamma \mathbf{v}_t - \eta \nabla\mathcal{L}(\boldsymbol{\theta}_t^{\text{look-ahead}}), \qquad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \mathbf{v}_{t+1}$$

Nesterov computes the gradient at the **look-ahead** point $\boldsymbol{\theta}_t + \gamma\mathbf{v}_t$ rather than the current point. This gives $O(1/T^2)$ convergence for convex $\mathcal{L}$ vs $O(1/T)$ for vanilla gradient descent — **Nesterov acceleration**.

### 7.3 Adaptive Methods: AdaGrad, RMSProp

**AdaGrad** (Duchi et al., 2011):

$$G_t = G_{t-1} + (\nabla\mathcal{L})^2 \quad \text{(element-wise)}, \qquad \theta_i \leftarrow \theta_i - \frac{\eta}{\sqrt{G_t + \epsilon}} (\nabla\mathcal{L})_i$$

Per-parameter learning rates: parameters with large accumulated gradients get small steps; parameters with small accumulated gradients get large steps. Ideal for sparse gradients (NLP embedding tables — most tokens are rare, should get large steps when they appear).

**Problem with AdaGrad:** $G_t$ is monotonically increasing — learning rate $\to 0$ as $t \to \infty$. In non-convex settings (neural networks), this premature learning rate decay is undesirable.

**RMSProp** (Hinton, unpublished 2012): Use exponential moving average instead of sum:

$$v_t = \rho v_{t-1} + (1-\rho)(\nabla\mathcal{L})^2, \qquad \theta_i \leftarrow \theta_i - \frac{\eta}{\sqrt{v_t + \epsilon}}(\nabla\mathcal{L})_i$$

The decay $\rho$ (typically 0.99) "forgets" old gradient magnitudes, adapting to local curvature.

### 7.4 Adam: Derivation and Bias Correction

**Adam** (Kingma & Ba, 2015) combines momentum with RMSProp:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla\mathcal{L}(\boldsymbol{\theta}_t) \qquad \text{(1st moment: mean)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla\mathcal{L}(\boldsymbol{\theta}_t))^2 \qquad \text{(2nd moment: uncentred variance)}$$

**Why bias correction is necessary.** At $t=1$, if $m_0 = v_0 = \mathbf{0}$ (zero-init):
$$m_1 = (1-\beta_1)\nabla\mathcal{L}_1$$
This is $(1-\beta_1) \approx 0.1$ times the true gradient — badly underestimated. The correction:
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
As $t \to \infty$: $1-\beta_1^t \to 1$ and correction vanishes. Only important in early iterations.

**Update rule:**

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

Default hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\eta = 10^{-3}$.

**Effective step size.** Ignoring bias correction: $\Delta\theta_i \approx \eta \cdot m_i / \sqrt{v_i}$. For a parameter with consistent gradient sign, $m_i \approx \sqrt{v_i}$, so the effective step is approximately $\eta$. Adam's maximum step size is bounded by $\eta$ regardless of gradient magnitude — unlike vanilla SGD where a large gradient spike can cause parameter to jump far.

### 7.5 AdamW and Decoupled Weight Decay

**The problem with Adam + L2 regularisation.** L2 regularisation adds $\lambda/2 \cdot \|\boldsymbol{\theta}\|^2$ to the loss, giving gradient $\nabla\mathcal{L} + \lambda\boldsymbol{\theta}$. In Adam, this gradient is divided by $\sqrt{\hat{v}_t}$ — so the effective weight decay is $\lambda/\sqrt{\hat{v}_t}$, which is **smaller for parameters with large gradients**. This is the wrong behaviour: we want the same decay for all parameters.

**AdamW** (Loshchilov & Hutter, 2019) decouples weight decay from the gradient update:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \boldsymbol{\theta}_t\right)$$

The weight decay $\lambda\boldsymbol{\theta}_t$ is applied directly in parameter space, not through the adaptive gradient scaling. This is the standard optimiser for LLM training (GPT, LLaMA, Mistral all use AdamW).

**Connection to proximal gradient:** AdamW's weight decay step is a proximal operator for the $\ell^2$ regulariser: $\text{prox}_{\eta\lambda\|\cdot\|^2/2}(\boldsymbol{\theta}) = \boldsymbol{\theta}/(1 + \eta\lambda) \approx \boldsymbol{\theta} - \eta\lambda\boldsymbol{\theta}$.

### 7.6 Learning Rate Schedules

**Linear warmup.** Starting from learning rate 0, linearly increase to peak $\eta_{\max}$ over $T_{\text{warm}}$ steps:
$$\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warm}}}, \quad t \leq T_{\text{warm}}$$

Warmup stabilises early training when Adam's $\hat{v}_t$ is small (bias correction denominator $1-\beta_2^t$ is small), making effective step sizes large and unstable. LLMs typically warm up for 1–4% of total training steps.

**Cosine annealing:**
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi(t - T_{\text{warm}})}{T - T_{\text{warm}}}\right)\right)$$

Smoothly decays from $\eta_{\max}$ to $\eta_{\min}$ following a cosine curve. The slow initial decay allows continued learning at a high rate, while the slow final decay enables fine-scale exploration near convergence.

**Chinchilla scaling:** Hoffmann et al. (2022) showed that for a given compute budget, the optimal strategy is to train for longer with a smaller model (not train a huge model for fewer steps). The learning rate schedule should span the entire training duration — cosine decay to near-zero at the end. Models trained with truncated schedules are undertrained.

### 7.7 Loss Landscape Geometry

**Saddle points dominate.** For a random function in $\mathbb{R}^n$, the fraction of critical points that are local minima decays as $2^{-n}$ — the vast majority are saddle points. In neural network losses, this means:
- Most non-convergence is due to saddle points, not local minima
- Gradient descent with noise (SGD) escapes saddle points via the random perturbation
- Pure gradient flow converges to saddle points — SGD's noise is beneficial

**Flat vs sharp minima.** A minimum $\boldsymbol{\theta}^*$ is **sharp** if the Hessian $\nabla^2\mathcal{L}(\boldsymbol{\theta}^*)$ has large eigenvalues (loss rises steeply around it). It is **flat** if eigenvalues are small (loss is approximately flat around it).

**Generalisation and flatness (Hochreiter & Schmidhuber, 1997; Keskar et al., 2017):** Flat minima generalise better because a flat minimum found on training data likely sits in the same flat region for test data. Sharp minima may have small training loss but high test loss (the minimum shifted slightly for test data falls outside the sharp basin).

**PAC-Bayes perspective.** For a Gaussian perturbation $\mathbf{p} \sim \mathcal{N}(\boldsymbol{\theta}, \sigma^2 I)$, the expected test loss is bounded by:
$$\mathcal{L}_{\text{test}}(\boldsymbol{\theta}) \leq \mathcal{L}_{\text{train}}(\boldsymbol{\theta}) + \sqrt{\frac{\sum_i \lambda_i(\nabla^2\mathcal{L}) \sigma^2 + \text{KL}}{n}}$$
The Hessian trace appears — flat minima (small $\lambda_i$) have better PAC-Bayes generalisation bounds.

**Sharpness-Aware Minimisation (SAM)** explicitly minimises the maximum loss in a ball around $\boldsymbol{\theta}$:
$$\min_{\boldsymbol{\theta}} \max_{\|\boldsymbol{\epsilon}\|\leq\rho} \mathcal{L}(\boldsymbol{\theta} + \boldsymbol{\epsilon})$$
SAM requires two forward-backward passes per step but consistently improves generalisation on vision and language tasks.


---

## 8. Regularisation

### 8.1 Weight Decay and Its Spectrum

**L2 regularisation** adds a penalty $\mathcal{L}_{\text{reg}} = \mathcal{L} + \frac{\lambda}{2}\|\boldsymbol{\theta}\|^2$, giving gradient $\nabla\mathcal{L} + \lambda\boldsymbol{\theta}$.

**Spectral view.** For a linear model $f(\mathbf{x}) = W\mathbf{x}$, weight decay corresponds to ridge regression: the optimum shrinks singular values $\sigma_j \to \sigma_j^2/(\sigma_j^2 + \lambda)$ (see §14-01). For neural networks, the same shrinkage applies to the **effective weight** in each direction — directions with small signal-to-noise ratio are suppressed more aggressively.

**Connection to Bayesian inference.** Weight decay is the MAP estimator under a Gaussian prior $\boldsymbol{\theta} \sim \mathcal{N}(\mathbf{0}, \frac{1}{\lambda}I)$. This prior encodes the belief that weights should be small, with no preference for direction.

**Effective degrees of freedom** for a regularised network are reduced — similar to how ridge regression has effective df $= \sum_j \sigma_j^2/(\sigma_j^2+\lambda)$.

### 8.2 Dropout: Bernoulli Masking and Inference

**Training.** At each forward pass, independently zero each neuron with probability $p$ (the "dropout rate"), then **rescale** active neurons by $1/(1-p)$ (inverted dropout):

$$h'_i = \frac{m_i}{1-p} h_i, \qquad m_i \sim \text{Bernoulli}(1-p) \text{ i.i.d.}$$

The rescaling ensures $\mathbb{E}[h'_i] = h_i$ — the expected activation equals the full-network activation. Without rescaling, test-time inference (which uses all neurons) would produce activations $(1-p)$ times larger than training.

**Inference.** Use all neurons without masking. The rescaling during training already accounts for this.

**Ensemble interpretation (Srivastava et al., 2014).** Dropout trains an exponential ensemble of $2^n$ sub-networks (each obtained by dropping a subset of neurons) with shared weights. At test time, the full network with rescaled weights approximates the geometric mean of predictions from this ensemble.

**MC Dropout (Gal & Ghahramani, 2016).** Keep dropout active at test time and run $T$ forward passes; the variance of outputs is an estimate of **epistemic uncertainty**:
$$\text{Var}_{\text{epistemic}}(\hat{y}) \approx \frac{1}{T}\sum_{t=1}^T \hat{y}_t^2 - \left(\frac{1}{T}\sum_{t=1}^T \hat{y}_t\right)^2$$
This provides cheap uncertainty quantification without Bayesian computation.

**Optimal dropout rate** empirically: $p = 0.1$–$0.2$ for first layers, $p = 0.5$ for penultimate dense layers. Transformer models use $p = 0.1$ on residual streams, attention weights, and FFN outputs.

### 8.3 Data Augmentation as Implicit Regularisation

**Manifold hypothesis.** Natural data (images, text, audio) lies near a low-dimensional manifold $\mathcal{M}$ embedded in high-dimensional space. Data augmentation applies transformations that remain on or near $\mathcal{M}$:
- **Images:** random crops, flips, colour jitter, rotations, Gaussian blur (all preserve semantic content)
- **Text:** back-translation, synonym replacement, random deletion
- **Audio:** pitch shift, time stretch, additive noise

**Mathematical effect.** Augmentation increases the effective dataset size from $n$ to $n \cdot |T|$ where $|T|$ is the number of augmentation transforms applied per sample. It implicitly enforces **equivariance/invariance** of the learned representation to the chosen transforms.

**Mixup (Zhang et al., 2018).** Interpolate between pairs of samples:
$$\tilde{\mathbf{x}} = \lambda \mathbf{x}^{(i)} + (1-\lambda)\mathbf{x}^{(j)}, \quad \tilde{\mathbf{y}} = \lambda \mathbf{y}^{(i)} + (1-\lambda)\mathbf{y}^{(j)}, \quad \lambda \sim \text{Beta}(\alpha, \alpha)$$

Mixup trains the network to be linear between data points — encouraging flat decision boundaries and reducing confidence on out-of-distribution inputs. Used in training LLaMA and other LLMs in the form of **curriculum learning** and **data mixing** across domains.

### 8.4 Implicit Regularisation of SGD

Even without explicit regularisation, SGD exhibits **implicit regularisation** — it has an inherent preference for certain solutions over others, even among all global minima.

**Minimum norm solution.** For overparameterised linear models ($p > n$), SGD (and gradient descent) converges to the **minimum $\ell^2$-norm** solution among all global minima. This is identical to the ridge regression solution at $\lambda \to 0^+$, but achieved by optimisation dynamics rather than explicit penalty.

**Gunasekar et al. (2018):** For matrix factorisation (and related problems), gradient descent with small step size and small initialisation converges to the minimum nuclear-norm solution — a form of implicit low-rank regularisation. This partially explains why neural networks generalise: they are biased toward solutions with simple structure.

**SGD vs full-batch GD.** Mini-batch SGD finds flatter minima than full-batch GD (Keskar et al., 2017), because the gradient noise prevents convergence to sharp minima. This is an argument for small batch sizes during training — large batch training converges to sharper minima and generalises worse unless special measures (SAM, lr scaling, longer training) are taken.

---

## 9. Normalisation Layers

### 9.1 Batch Normalisation: Derivation and Train/Test Gap

**Motivation.** During training, the distribution of inputs to each layer changes as parameters update — Ioffe & Szegedy (2015) called this **internal covariate shift**. BatchNorm normalises layer inputs, stabilising training and allowing higher learning rates.

**Forward pass (training).** For a batch of pre-activations $\{z_i\}_{i=1}^B$:

$$\mu_B = \frac{1}{B}\sum_{i=1}^B z_i, \qquad \sigma_B^2 = \frac{1}{B}\sum_{i=1}^B (z_i - \mu_B)^2$$
$$\hat{z}_i = \frac{z_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma \hat{z}_i + \beta$$

where $\gamma, \beta$ are learnable affine parameters (per-feature scale and shift).

**Train/test gap.** During training, BatchNorm uses the batch statistics $\mu_B, \sigma_B^2$. During inference, these must be replaced by population statistics estimated from training data via exponential moving averages:
$$\mu_{\text{pop}} \leftarrow (1-\alpha)\mu_{\text{pop}} + \alpha \mu_B$$
Forgetting to switch to eval mode (in PyTorch: `model.eval()`) causes a systematic distribution mismatch between training and inference — one of the most common bugs in deep learning.

**Why BatchNorm accelerates training:** BatchNorm makes the loss landscape smoother and more Lipschitz-continuous (Santurkar et al., 2018). This allows larger learning rates without divergence. The effect is equivalent to an adaptive preconditioner — similar to second-order methods but achieved by normalisation.

### 9.2 Layer Normalisation

**Definition.** LayerNorm normalises across the feature dimension (within each sample), rather than across the batch:

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad \mu = \frac{1}{d}\sum_i x_i, \qquad \sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$$

**Why transformers use LayerNorm, not BatchNorm:**
1. **Variable sequence length.** Language model inputs have variable length — batch statistics are ill-defined.
2. **Autoregressive inference.** At inference time, each token is generated one at a time (batch size = 1). BatchNorm with batch size 1 is just the identity (mean and variance of a single element are trivially 0 and 0).
3. **Batch-size independence.** LayerNorm statistics depend only on the current sample, making it suitable for both training and inference without a train/test gap.

### 9.3 RMSNorm

**Definition (Zhang & Sennrich, 2019).** RMSNorm removes the mean-centering step, normalising only by the root-mean-square:

$$\hat{x}_i = \frac{x_i}{\text{RMS}(\mathbf{x})} \cdot \gamma_i, \qquad \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}$$

No bias parameter $\beta$ (since there is no mean subtraction to re-centre).

**Why it works.** The mean-centering in LayerNorm subtracts $\mu$, which is empirically near zero in many neural network settings (especially with zero-initialised biases). RMSNorm eliminates a redundant computation. Llama, Llama-2, Mistral, and Falcon all use RMSNorm for this reason — it is $\sim$10–20% faster than LayerNorm.

### 9.4 Normalisation as Signal Propagation Control

**BatchNorm as gradient preconditioner.** The gradient through BatchNorm has a projection effect:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \frac{\gamma}{\sqrt{\sigma_B^2+\epsilon}}\left(I - \frac{1}{B}\mathbf{1}\mathbf{1}^\top - \frac{\hat{\mathbf{z}}\hat{\mathbf{z}}^\top}{B}\right)\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{z}}}$$

This gradient projection removes the component of the upstream gradient that would shift the batch mean or scale — preventing large parameter updates from causing covariate shift in the next layer. The effect is a **second-order-like curvature correction** applied to the gradient.

**Sharpness reduction.** Santurkar et al. (2018) showed empirically that BatchNorm reduces the maximum eigenvalue of the loss Hessian by $\sim$50–100$\times$, directly explaining why it allows larger learning rates and more stable training.


---

## 10. Residual Connections and Modern Architectures

### 10.1 Residual Networks: Skip Connection Identity

**Definition (He et al., 2015).** A residual block computes:

$$\mathbf{h}^{[l+1]} = \mathbf{h}^{[l]} + F(\mathbf{h}^{[l]}; \boldsymbol{\theta}^{[l]})$$

where $F$ is the "residual function" (typically: Conv-BN-ReLU-Conv-BN or Linear-LayerNorm-Nonlinear-Linear). The identity shortcut $+\mathbf{h}^{[l]}$ bypasses the block.

**Key intuition.** Without residuals, the network must learn the full mapping $\mathbf{h}^{[l+1]} = G(\mathbf{h}^{[l]})$ from scratch. With residuals, it only needs to learn the **residual** $F = G - I$. If the optimal transformation is near-identity (as it is in many layers of trained networks), learning $F \approx 0$ is much easier than learning $G \approx I$.

**Empirical evidence:** In trained ResNets, the Frobenius norm of $F(\mathbf{h})$ is typically $10\times$ smaller than the norm of $\mathbf{h}$ — residual blocks are indeed learning near-zero residuals, confirming the hypothesis.

### 10.2 Gradient Analysis: Sum vs Product of Jacobians

**Without residuals.** For a depth-$L$ network, the gradient from layer $L$ to layer $0$ is:

$$\frac{\partial\mathcal{L}}{\partial \mathbf{h}^{[0]}} = \frac{\partial\mathcal{L}}{\partial \mathbf{h}^{[L]}} \cdot \prod_{l=1}^{L} J^{[l]}$$

where $J^{[l]} = \partial \mathbf{h}^{[l]}/\partial \mathbf{h}^{[l-1]}$. If each $J^{[l]}$ has spectral radius $< 1 - \epsilon$, this product vanishes as $\leq (1-\epsilon)^L \to 0$.

**With residuals.** For a residual network:

$$\frac{\partial \mathbf{h}^{[L]}}{\partial \mathbf{h}^{[0]}} = \prod_{l=1}^{L}(I + \partial F^{[l]}/\partial \mathbf{h}^{[l-1]})$$

Expanding the product:

$$= I + \sum_{l=1}^L \frac{\partial F^{[l]}}{\partial \mathbf{h}^{[l-1]}} + \sum_{l < l'} \frac{\partial F^{[l']}}{\partial \mathbf{h}^{[l'-1]}}\frac{\partial F^{[l]}}{\partial \mathbf{h}^{[l-1]}} + \cdots$$

The **identity term $I$** ensures that even if all $\partial F^{[l]}/\partial\mathbf{h}$ are tiny, the gradient still has a component of magnitude 1 flowing from the output directly to the input. No vanishing possible through the identity path.

**Variance of gradient.** For a random residual network at initialisation with $\mathbb{E}[\partial F^{[l]}/\partial\mathbf{h}] = 0$:

$$\text{Var}\left(\frac{\partial\mathcal{L}}{\partial\mathbf{h}^{[0]}}\right) \approx 1 + L \cdot \text{Var}\left(\frac{\partial F}{\partial \mathbf{h}}\right)$$

Gradient variance grows **linearly** with depth — not exponentially. Proper initialisation keeps this linear growth manageable.

### 10.3 Pre-Norm vs Post-Norm

**Original Transformer (post-norm):**
$$\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{MHA}(\mathbf{x})), \qquad \mathbf{x}'' = \text{LayerNorm}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$

**GPT-2/3, LLaMA (pre-norm):**
$$\mathbf{x}' = \mathbf{x} + \text{MHA}(\text{LayerNorm}(\mathbf{x})), \qquad \mathbf{x}'' = \mathbf{x}' + \text{FFN}(\text{LayerNorm}(\mathbf{x}'))$$

**Why pre-norm is preferred for deep networks.** In post-norm, the residual stream passes through LayerNorm after each block — which normalises the scale of the residual. This means the residual contribution $F(\mathbf{x})$ must compete with the normalised $\mathbf{x}$, and gradient flow through the shortcut is disrupted. In pre-norm, LayerNorm acts on a copy of $\mathbf{x}$ before the block — the residual stream itself remains unnormalised, allowing gradient magnitudes to be consistent across depths.

**Empirical stability:** Pre-norm models are more numerically stable at large depth and can be trained without warmup in some configurations. Post-norm requires careful learning rate warmup to avoid divergence at early steps.

### 10.4 Depth Efficiency: Why Residuals Enable 1000+ Layers

**Veit et al. (2016): Ensemble interpretation.** Unrolling the product $(I + F^{[1]})(I + F^{[2]})\cdots(I + F^{[L]})$ yields $2^L$ paths from input to output, each passing through a subset of residual blocks. A ResNet-110 has $2^{55} \approx 10^{16}$ paths. The network behaves as an ensemble of exponentially many shallow networks.

**Effective path length distribution.** The ensemble is dominated by **short paths** — paths that skip many blocks. The expected path length is $L/2$, but most gradient signal comes from paths shorter than $L/4$. This means the effective network training depth is much smaller than the nominal depth, resolving the vanishing gradient problem empirically.

**Highway networks (Srivastava et al., 2015)** were the predecessor:
$$\mathbf{y} = T(\mathbf{x}) \odot H(\mathbf{x}) + (1 - T(\mathbf{x})) \odot \mathbf{x}$$
where $T$ is a learned gate. ResNets simplify this to $T = \mathbf{1}$ (always carry 100% of identity).

---

## 11. Gradient Flow and Training Dynamics

### 11.1 Vanishing and Exploding Gradients

**Formal analysis.** For a vanilla RNN $\mathbf{h}_t = \tanh(W\mathbf{h}_{t-1} + U\mathbf{x}_t)$, the gradient from step $T$ to step $t$ is:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t}^{T-1} \frac{\partial \mathbf{h}_{k+1}}{\partial \mathbf{h}_k} = \prod_{k=t}^{T-1} \text{diag}(\tanh'(\mathbf{z}_k)) W^\top$$

Since $\|\text{diag}(\tanh'(\mathbf{z}_k))\|_2 \leq 1$ and $\|\text{diag}(\tanh'(\mathbf{z}_k))W^\top\|_2 \leq \|W\|_2$:
- If $\|W\|_2 < 1$: gradient norm $\leq \|W\|_2^{T-t} \to 0$ exponentially
- If $\|W\|_2 > 1$: gradient norm $\geq (1/\sqrt{d}\|W\|_2)^{T-t} \to \infty$ exponentially (for random $W$)

Only $\|W\|_2 = 1$ (spectral radius = 1) avoids both pathologies — but this is a knife-edge condition that breaks under gradient updates.

**MLP case:** Same analysis applies with $W^{[l]}$ replacing $W$ and activation derivatives replacing $\tanh'$. For deep MLPs with sigmoid/tanh, vanishing gradients limit trainable depth to $\sim 10$ layers without ResNets.

### 11.2 Gradient Clipping

**Global norm clipping.** Compute the global gradient norm $\|g\| = \sqrt{\sum_i g_i^2}$; if it exceeds threshold $\tau$, rescale:

$$\mathbf{g} \leftarrow \frac{\tau}{\max(\|\mathbf{g}\|, \tau)} \mathbf{g}$$

This preserves the **direction** of the gradient while capping its magnitude. Used in all RNN training (LSTM, GRU) and in transformer LLM training. Typical threshold: $\tau = 1.0$ for LLMs, $\tau = 5.0$ for RNNs.

**Why global norm, not per-parameter clipping?** Per-parameter clipping changes the gradient direction, potentially causing optimisation to move in an unexpected direction. Global norm clipping maintains the gradient direction — it is equivalent to reducing the learning rate when gradients are large.

**When clipping is needed vs unnecessary:** Necessary for RNNs (spectral radius $> 1$ leads to exploding gradients). Less critical for ResNets and transformers (residual connections + normalisation prevent gradient explosion in most cases), but still standard practice for numerical stability.

### 11.3 Jacobian Spectral Analysis Across Layers

**Ordered and chaotic phases (revisited with Jacobians).** The Jacobian $J^{[l]} = \text{diag}(\sigma'(\mathbf{z}^{[l]}))W^{[l]}$ has singular values that determine gradient flow. In the mean-field limit (infinite width, random weights):

$$\chi_1 := \sigma_W^2 \mathbb{E}_{z \sim \mathcal{N}(0, q^{[l]})}[\sigma'(z)^2]$$

- $\chi_1 < 1$: ordered phase — gradients vanish
- $\chi_1 > 1$: chaotic phase — gradients explode
- $\chi_1 = 1$: edge of chaos — gradients are stable

**Gradient explode probability.** A surprising result: in very deep randomly-initialised networks, even if $\chi_1 = 1$ on average, fluctuations cause the product of Jacobians to have enormous variance. The mean of $\prod_l J^{[l]}$ may be finite while the typical value is near zero — a **self-averaging failure** for gradient flow. This is why initialisation and normalisation are jointly necessary.

### 11.4 Grokking and Delayed Generalisation

**The grokking phenomenon (Power et al., 2022).** For certain algorithmic tasks (modular addition, permutation group composition), neural networks first **memorise** the training set (train accuracy $\to 100\%$, test accuracy $\approx$ random), then after many more steps **generalise** (test accuracy $\to 100\%$) — even with early stopping based on training loss.

**Mathematical explanation.** Two competing regularisation effects:
1. **Memorisation** (fast): the network finds a large-norm solution that perfectly fits training data
2. **Generalisation** (slow): weight decay gradually shrinks the solution to a lower-norm region with a qualitatively different algorithm

Grokking occurs when there exists a small-norm solution that generalises, but the optimisation path first passes through a large-norm memorising solution. Weight decay provides the eventual drive toward the generalising solution — but it operates slowly against a steep gradient pointing toward memorisation.

**Connection to double descent.** Grokking is a temporal analogue of double descent (§12.4): as training continues, the model transitions from an overfitting regime to a generalising regime. The transition requires crossing a "grokking threshold" in weight norm.


---

## 12. Expressivity and Depth

### 12.1 Depth Separation Results

**Telgarsky (2016) — revisited.** The exact theorem states: for any $k, q \geq 1$, there exists a function $g_k: [0,1] \to [0,1]$ computed by a ReLU network of depth $O(k)$ and width $O(1)$ such that any depth-$q$ network approximating $g_k$ with error $< 1/3$ in $L^1$ norm must have at least $2^{\Omega(\min(k, q/k))}$ neurons.

**Intuition:** The function $g_k$ is a "sawtooth" with $2^k$ teeth, representable recursively. Each level of the recursion corresponds to one layer.

**Eldan & Shamir (2016).** There exists a radially symmetric function $f: \mathbb{R}^d \to \mathbb{R}$ such that:
- Any depth-3 network can approximate $f$ with width $O(d)$ neurons
- Any depth-2 network approximating $f$ requires width $\Omega(e^d)$ neurons

This exponential separation shows that going from 2 to 3 layers can save exponentially many neurons for certain functions.

**Montufar et al. (2014).** The number of **linear regions** (polyhedral pieces) of a depth-$L$, width-$n$ ReLU network in $\mathbb{R}^d$ satisfies:

$$\text{# regions} \geq \left(\lfloor n/d \rfloor^{d(L-1)}\right) \cdot \binom{n}{d}$$

A depth-1 network with $m$ neurons in $\mathbb{R}^d$ has at most $\binom{m}{d}$ regions. A depth-$L$ network with the same number of total neurons has exponentially more regions.

### 12.2 Linear Regions and ReLU Networks

**ReLU as a piecewise-linear map.** Each neuron $\max(0, \mathbf{w}^\top \mathbf{x} + b)$ partitions input space with a hyperplane $\{\mathbf{x}: \mathbf{w}^\top \mathbf{x} + b = 0\}$. On each side, the neuron is either inactive (0) or active (identity). A network with $m$ neurons has at most $2^m$ activation patterns, each giving a different linear map.

**Counting linear regions.** For a single layer with $m$ neurons in $\mathbb{R}^d$, the number of distinct activation patterns is at most $\sum_{k=0}^d \binom{m}{k} = O(m^d)$ (bounded by the number of regions created by $m$ hyperplanes in $\mathbb{R}^d$). This is exponentially fewer than $2^m$ for small $d$.

**Depth multiplies regions.** At the second layer, each input region from layer 1 is further subdivided by the layer-2 neurons. The subdivision is multiplicative — hence the exponential growth in linear regions with depth.

### 12.3 Circuit Complexity and Boolean Functions

**$\text{TC}^0$ circuits.** Neural networks (with rational weights and polynomial-time computable activations) can approximate any function in the circuit complexity class $\text{TC}^0$ — functions computable by constant-depth, polynomial-size threshold circuits. This includes integer addition, multiplication, and sorting.

**What transformers compute.** Merrill & Sabharwal (2023) showed that transformers with hard attention and $O(\log n)$ precision compute exactly $\text{TC}^0$ — meaning transformers cannot (provably) compute functions like counting parity bits in a sequence, unless $\text{TC}^0 = \text{NC}^1$ (widely believed false). This theoretically limits what transformers can do with bounded precision and depth.

### 12.4 Over-Parameterisation and Implicit Bias

**The double descent phenomenon (Belkin et al., 2019).** Classical bias-variance says test error is U-shaped in model complexity. For modern neural networks, test error has a **double descent curve**:

1. **Under-parameterised regime** ($p < n$): classical U-shaped curve — test error rises as model overfits
2. **Interpolation threshold** ($p \approx n$): sharp spike in test error — the model barely fits training data
3. **Over-parameterised regime** ($p > n$): test error decreases again as $p \to \infty$

**Minimum-norm interpolators.** In the over-parameterised regime, the model interpolates training data (zero training error) while converging to the minimum-norm solution. For linear models, this is the Moore-Penrose pseudoinverse. For neural networks, the implicit bias of gradient descent determines which minimum-norm interpolator is found.

**Benign overfitting (Bartlett et al., 2020).** For linear regression with $p \gg n$, interpolation is "benign" — the test error converges to the irreducible noise level as $p \to \infty$ — if the covariance of input features has sufficiently fast eigenvalue decay. The excess risk is bounded by $\sigma^2 \cdot n / (\sum_{j>n} \lambda_j)$ where $\lambda_j$ are the data covariance eigenvalues.

**Implication for LLMs.** GPT-4 has $\sim 10^{12}$ parameters trained on $\sim 10^{12}$ tokens — roughly at the interpolation threshold. The benign overfitting theory partially explains why it generalises despite memorising vast amounts of training data.

---

## 13. Neural Tangent Kernel

### 13.1 Infinite-Width Limit and Kernel Ridge Regression

**Theorem (Jacot, Gabriel & Hongler, 2018).** Consider a neural network $f(\mathbf{x}; \boldsymbol{\theta})$ with width $n \to \infty$. Define the **Neural Tangent Kernel**:

$$\Theta(\mathbf{x}, \mathbf{x}') = \nabla_{\boldsymbol{\theta}} f(\mathbf{x}; \boldsymbol{\theta})^\top \nabla_{\boldsymbol{\theta}} f(\mathbf{x}'; \boldsymbol{\theta})$$

As $n \to \infty$, $\Theta$ converges to a deterministic limit $\Theta^*$ at initialisation, and **remains constant throughout gradient descent training** (lazy training regime).

**Consequence.** Under gradient flow (continuous-time gradient descent) on MSE loss:

$$\hat{f}(X_*) = K(X_*, X)(K(X,X) + \epsilon I)^{-1}\mathbf{y}$$

where $K_{ij} = \Theta^*(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$. Training an infinite-width network with gradient descent is **exactly kernel ridge regression** with kernel $\Theta^*$.

**NTK for a 2-layer linear network.** For $f(\mathbf{x}) = W_2 W_1 \mathbf{x}$ with $W_1 \in \mathbb{R}^{n \times d}$, $W_2 \in \mathbb{R}^{1 \times n}$:

$$\Theta(\mathbf{x}, \mathbf{x}') = \|W_2\|^2 (\mathbf{x}^\top \mathbf{x}') + \|W_1 \mathbf{x}\|^2 \cdot \|W_1 \mathbf{x}'\|^2 / (?)$$

(Full computation in theory.ipynb §13.1)

### 13.2 Lazy Training vs Feature Learning

**Lazy training (NTK regime).** When network weights change little during training — $\boldsymbol{\theta}_T \approx \boldsymbol{\theta}_0$ — the network is in the NTK regime. This happens for:
- Very large initialisation scale $\sigma_W^2 \to \infty$ (gradients are small relative to weights)
- Very small learning rate $\eta \to 0$
- Infinite width $n \to \infty$ (NTK theorem)

**Feature learning (rich regime).** When weights change substantially, the learned features $\phi_T(\mathbf{x})$ are qualitatively different from the initialisation features $\phi_0(\mathbf{x})$. This is the regime in which neural networks outperform kernel methods — they learn representations tailored to the task.

**The NTK kernel is task-agnostic** (fixed by architecture and initialisation). Feature learning networks adapt their representation to the specific task. Empirically, feature learning networks outperform NTK/kernel regression by a large margin on real tasks — which means practical networks are in the feature learning regime, not the NTK regime.

### 13.3 NTK Spectrum and Convergence Rate

**Convergence of gradient descent.** Under gradient flow on MSE loss with NTK $\Theta$:

$$\mathcal{L}(t) = \mathcal{L}(0) \cdot \exp(-2\lambda_{\min}(\Theta) t)$$

where $\lambda_{\min}(\Theta)$ is the smallest eigenvalue of the NTK matrix on training data. If $\Theta$ is PSD (which Jacot et al. proved for infinite-width networks), $\lambda_{\min} > 0$ and gradient flow converges exponentially.

**Positive-definiteness of NTK.** For commonly used activations (tanh, ReLU, GELU) and non-degenerate data, $\Theta^*$ is strictly positive definite. This guarantees that:
1. Gradient flow achieves zero training loss
2. The convergence rate depends on the condition number $\kappa(\Theta) = \lambda_{\max}/\lambda_{\min}$

Large $\kappa$ means slow convergence — analogous to poorly-conditioned linear systems.

### 13.4 Beyond NTK: μP and Feature Learning

**Maximal Update Parametrisation (μP).** The standard (NTK) parametrisation scales weights as $W \sim \mathcal{N}(0, 1/n)$, placing the network in the NTK regime at infinite width. The **μP** (pronounced "mu-P") parametrisation (Yang & Hu, 2021) scales weights differently:

| Layer type | NTK param | μP param |
|---|---|---|
| Input weights $W^{[1]}$ | $\sigma_W^2 = 1/n$ | $\sigma_W^2 = 1/d_{\text{in}}$ |
| Hidden weights $W^{[l]}$ | $\sigma_W^2 = 1/n$ | $\sigma_W^2 = 1/n$, lr $\propto 1/n$ |
| Output weights $W^{[L]}$ | $\sigma_W^2 = 1/n$ | $\sigma_W^2 = 1/n^2$ |

Under μP, the width $n \to \infty$ limit is the **feature learning regime** — updates to the hidden layer are $O(1)$ rather than $O(1/n)$, so features change meaningfully.

**Hyperparameter transfer.** A key practical implication: optimal hyperparameters (learning rate, weight decay) transfer from small-width to large-width models under μP. This means you can tune hyperparameters on a cheap small model and transfer them to a large model — a significant practical advantage.


---

## 14. Deep Learning Connections to Modern AI

### 14.1 MLP Sublayers in Transformers

**Architecture.** The FFN (feed-forward network) sublayer in each transformer block computes:

$$\text{FFN}(\mathbf{x}) = W_2 \, \text{GELU}(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

with $W_1 \in \mathbb{R}^{4d \times d}$, $W_2 \in \mathbb{R}^{d \times 4d}$ for model dimension $d$. The FFN contains approximately **2/3 of all parameters** in a transformer — not the attention layers, as commonly assumed.

**Key-value memory interpretation (Geva et al., 2021).** Expand the FFN:

$$\text{FFN}(\mathbf{x}) = \sum_{i=1}^{4d} [W_1\mathbf{x}]_i^+ \cdot W_2^{:,i}$$

(using GELU $\approx$ ReLU for illustration). The $i$-th column of $W_1^\top$ (row of $W_1$) acts as a **key** — it determines how much the $i$-th memory is activated for input $\mathbf{x}$. The $i$-th column of $W_2$ acts as the corresponding **value** — what is added to the residual stream. The FFN is a **content-addressable memory** that reads and writes information to the residual stream based on the current token representation.

**Superposition in FFN.** Elhage et al. (2022) showed that FFN neurons are **polysemantic** — a single neuron activates for multiple unrelated concepts (e.g., "Golden Gate Bridge", "mathematics", and "the word 'it'"). The FFN represents more concepts than it has neurons by encoding them in superposition (non-orthogonal directions), relying on the fact that most concepts are rarely co-active.

**SwiGLU variant (LLaMA, Mistral, Falcon).** Replace GELU with a gated activation:

$$\text{SwiGLU}(\mathbf{x}) = W_3(\text{SiLU}(W_1\mathbf{x}) \odot W_2\mathbf{x})$$

Three weight matrices instead of two, but the overall FFN is larger than 4× expansion — 8/3× is typical. The gating mechanism provides sharper selection of which memory entries activate.

### 14.2 LoRA Revisited: Full Neural-Network Perspective

**Full fine-tuning** of a pretrained model updates every weight: $W \leftarrow W_0 + \Delta W$. For GPT-4 with 1.8T parameters, $\Delta W$ would require as much memory as $W_0$ — impractical for most users.

**LoRA** (Hu et al., 2022) constrains $\Delta W = BA$ with $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$, $r \ll \min(m,n)$. Only $r(m+n)$ new parameters are trained; $W_0$ is frozen. Parameter savings for a $4096 \times 4096$ weight matrix at $r=16$: $4096^2 = 16.8\text{M}$ vs $16 \cdot 8192 = 131\text{K}$ — a 128× reduction.

**Why low rank works.** The empirical observation underlying LoRA: the update $\Delta W^*$ from full fine-tuning has **low numerical rank** — its singular values decay rapidly. Formally, for many NLP tasks, the intrinsic dimensionality of the fine-tuning update is $r^* \ll d$. LoRA exploits this by explicitly constraining to rank $r \geq r^*$.

**Initialisation:** $A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$ at init — ensures $\Delta W = BA = 0$ at the start of fine-tuning (fine-tuned model equals pretrained model initially).

**DoRA (Liu et al., 2024).** Decompose weight update as direction + magnitude:
$$W = \frac{W_0 + \Delta W}{\|W_0 + \Delta W\|_c} \cdot m$$
where $\|\cdot\|_c$ is the column-wise norm and $m$ is a learnable magnitude vector. DoRA matches full fine-tuning quality at lower rank than LoRA.

### 14.3 Mechanistic Interpretability

**Goal.** Mechanistic interpretability (MI) seeks to reverse-engineer what computations a trained neural network performs — not just its input-output behaviour, but the algorithm it implements.

**Superposition hypothesis (Elhage et al., 2022).** A network with $d$ neurons can represent more than $d$ features by encoding them as nearly-orthogonal directions in $\mathbb{R}^d$. If features are sparse (rarely co-active), the interference between non-orthogonal features is negligible most of the time. Formally: if $\mathbf{f} \in \mathbb{R}^k$ with $k > d$ is the feature vector and $W \in \mathbb{R}^{d \times k}$ is the encoding matrix, the reconstruction error $\|W^\top W \mathbf{f} - \mathbf{f}\|^2$ can be small when $\mathbf{f}$ is sparse.

**Circuits.** A "circuit" is a subgraph of the neural network that computes a specific function. Elhage et al. identified circuits for:
- **Induction heads** (attend to previous occurrence of current token)
- **Name mover heads** (move subject name to output position)
- **Factual association** (retrieve entity attributes from MLP layers)

**Linear representation hypothesis.** Many high-level concepts are linearly encoded in transformer residual streams — directions in residual space correspond to interpretable concepts. This justifies probing classifiers (logistic regression on activations) as a tool for interpretability.

### 14.4 Scaling Laws and Chinchilla

**Kaplan et al. (2020) scaling laws.** For autoregressive language models, test loss follows a power law in:
- Model parameters $N$: $\mathcal{L}(N) \approx (N_c/N)^\alpha$
- Dataset tokens $D$: $\mathcal{L}(D) \approx (D_c/D)^\beta$
- Compute $C$: $\mathcal{L}(C) \approx (C_c/C)^\gamma$

with $\alpha \approx 0.076$, $\beta \approx 0.095$, $\gamma \approx 0.050$. The compute-loss frontier combines these.

**Chinchilla (Hoffmann et al., 2022).** Given a fixed compute budget $C \approx 6ND$ FLOPs (approximately), the optimal allocation balances model size $N$ and training tokens $D$. The Chinchilla formula:

$$N_{\text{opt}} \propto \sqrt{C}, \qquad D_{\text{opt}} \propto \sqrt{C}$$

**Chinchilla scaling:** Optimal models should train for $\approx 20$ tokens per parameter. GPT-3 (175B parameters) was trained on 300B tokens — significantly undertrained. LLaMA-1 (65B) trained on 1.4T tokens — 5× more than Chinchilla-optimal, but cheaper inference.

**Implications for architecture.** The Chinchilla result implies that, for a fixed compute budget, it is more efficient to train a smaller model for longer than a larger model for fewer steps. This has shifted the community toward smaller but more efficiently trained models (Mistral 7B, LLaMA-3.1 8B).


---

## 15. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Using sigmoid activation in deep hidden layers | Max derivative is 0.25 — gradients shrink by $4\times$ per sigmoid layer, causing vanishing gradients for depth $> 5$ | Use ReLU, GELU, or SiLU in hidden layers; sigmoid only at output for binary classification |
| 2 | Forgetting `model.eval()` at test time | BatchNorm uses batch statistics in train mode (wrong at inference) and Dropout drops neurons (also wrong) | Always call `model.eval()` before inference and `model.train()` before training |
| 3 | Initialising all weights to zero | All neurons compute the same function and receive the same gradient — symmetry breaking never occurs, network stays at zero | Always use random initialisation (Xavier/He); biases can be zero |
| 4 | Same learning rate for all parameters | Parameters at different depths have vastly different gradient magnitudes; a LR good for the output layer explodes the input layer | Use Adam (adaptive per-parameter LR) or layer-wise LR scaling; reduce LR for embeddings and input projections |
| 5 | Not normalising inputs | Large input magnitudes cause large pre-activations, saturating sigmoid/tanh or creating scale-dependent ReLU patterns | Normalise inputs to zero-mean, unit-variance; use BatchNorm/LayerNorm after first layer |
| 6 | Computing cross-entropy as `log(softmax(z))` explicitly | Numerical overflow: $\exp(z_i)$ overflows for $z_i > 709$ (float32 max) before log is applied | Use `log_softmax(z)` (numerically stable: subtracts max before exp) or the fused `cross_entropy_loss` function |
| 7 | Comparing logits directly as probabilities | Network logits are not probabilities — they can be any real number, including negative | Apply softmax/sigmoid to convert to probabilities; or use argmax for classification (logits preserve rank) |
| 8 | Not clipping gradients in RNNs/LSTMs | Spectral radius $> 1$ causes exponential gradient growth, leading to NaN parameters in a few steps | Apply global norm gradient clipping ($\tau = 1.0$–5.0) whenever training RNNs |
| 9 | Applying weight decay to bias and normalisation parameters | Biases and LayerNorm/BatchNorm parameters should not be regularised — they are already controlled by the normalisation scale | Exclude biases and norm parameters from the weight decay parameter group in the optimiser |
| 10 | Evaluating loss without reducing over batch | Broadcasting bugs cause shape mismatches in loss computation; some losses return per-sample vectors | Always check loss shape; ensure loss is reduced to a scalar before backward pass; print `loss.shape` during debugging |
| 11 | Setting learning rate without warmup for transformers | Adam's adaptive estimates are unreliable for first $\sim 1000$ steps (bias correction catches up); high LR early causes divergence | Add 1–5% linear warmup; use cosine decay schedule for remainder |
| 12 | Training with large batches without adjusting LR | Larger batches produce less noisy gradients, requiring larger effective learning rate to maintain similar dynamics | Apply linear scaling rule: $\eta_{\text{new}} = \eta_{\text{base}} \cdot B_{\text{new}} / B_{\text{base}}$ with warmup |

---

## 16. Exercises

**Exercise 1 ★ — Universal Approximation Width Bound**

A single-hidden-layer network $f(\mathbf{x}) = \sum_{i=1}^m v_i \sigma(\mathbf{w}_i^\top \mathbf{x} + b_i)$ is said to achieve $\epsilon$-approximation of $f^* \in \mathcal{B}_C$ if $\mathbb{E}[(f(\mathbf{x})-f^*(\mathbf{x}))^2] \leq \epsilon$.

(a) Using Barron's theorem, derive the minimum width $m$ needed for $\epsilon$-approximation of a function with Barron constant $C_f = 10$.

(b) Repeat for $\epsilon = 10^{-3}$ and compare to the classical polynomial approximation rate $O(\epsilon^{-d/2})$ in dimension $d = 100$. How many polynomial terms are needed vs. neurons?

(c) State two conditions on the target function that Barron's theorem requires. Construct a counterexample of a function that violates one condition and cannot be efficiently approximated by shallow networks.

**Exercise 2 ★ — Backpropagation by Hand**

For a two-layer MLP $f(\mathbf{x}) = W^{[2]}\text{ReLU}(W^{[1]}\mathbf{x} + \mathbf{b}^{[1]}) + \mathbf{b}^{[2]}$ with:
$$W^{[1]} = \begin{pmatrix}1 & -1\\1 & 1\end{pmatrix}, \quad W^{[2]} = \begin{pmatrix}1 & 1\end{pmatrix}, \quad \mathbf{x} = \begin{pmatrix}1\\0\end{pmatrix}, \quad y = 1$$

(a) Perform the forward pass: compute $\mathbf{z}^{[1]}$, $\mathbf{h}^{[1]}$, $\hat{y}$.

(b) Compute the MSE loss $\mathcal{L} = \frac{1}{2}(\hat{y} - y)^2$.

(c) Perform the backward pass: compute $\delta^{[2]}$, $\partial\mathcal{L}/\partial W^{[2]}$, $\boldsymbol{\delta}^{[1]}$, $\partial\mathcal{L}/\partial W^{[1]}$.

(d) Verify your gradients with finite differences: compute $(\mathcal{L}(W^{[1]}_{11}+\epsilon) - \mathcal{L}(W^{[1]}_{11}-\epsilon))/(2\epsilon)$ for $\epsilon = 10^{-5}$ and check it matches your analytic gradient.

**Exercise 3 ★ — Xavier and He Initialisation**

(a) Derive Xavier initialisation $\sigma_W^2 = 2/(d_{\text{in}} + d_{\text{out}})$ from the forward and backward variance preservation conditions for tanh activations.

(b) Modify the derivation for ReLU to obtain He initialisation $\sigma_W^2 = 2/d_{\text{in}}$.

(c) Simulate a 20-layer network at init: use Xavier for tanh, He for ReLU. Plot the mean and standard deviation of activation magnitudes across layers for both. Verify that He init maintains $O(1)$ variance for ReLU, while Xavier causes decay.

**Exercise 4 ★ — Adam Update and Bias Correction**

(a) Implement one Adam update step from scratch. Given gradient $g = 0.1$ at step $t=1$ with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, $\eta=10^{-3}$: compute $m_1$, $v_1$, $\hat{m}_1$, $\hat{v}_1$, and $\Delta\theta_1$.

(b) Show why bias correction is necessary: compute $m_1/\sqrt{v_1}$ without correction and compare to $\hat{m}_1/\sqrt{\hat{v}_1}$.

(c) Run Adam for 1000 steps on the quadratic loss $\mathcal{L}(\theta) = \theta^2$ starting from $\theta_0 = 1$. Plot convergence of $\theta_t$ and $\mathcal{L}_t$. At what step does Adam effectively converge?

**Exercise 5 ★★ — Dropout as Ensemble Averaging**

(a) For a single-hidden-layer network with 4 hidden units and dropout rate $p = 0.5$, enumerate all $2^4 = 16$ possible sub-networks. What fraction of networks keep at least 2 neurons active?

(b) Prove that the expected output of a dropout network (averaging over all masks $\mathbf{m}$) equals the output of the full network with weights rescaled by $(1-p)$: $\mathbb{E}_{\mathbf{m}}[f_{\mathbf{m}}(\mathbf{x})] = f_{(1-p)W}(\mathbf{x})$ for a one-hidden-layer linear network.

(c) Explain why (b) is an approximation for nonlinear networks. Implement MC Dropout: run 50 forward passes with dropout active on a simple regression task and estimate the predictive mean and variance. Compare to the analytic Bayesian posterior from §14-01 Ex. 4.

**Exercise 6 ★★ — BatchNorm Backpropagation**

(a) Derive $\partial\mathcal{L}/\partial \mathbf{z}$ (gradient through BatchNorm) starting from $\partial\mathcal{L}/\partial \hat{\mathbf{z}}$.

(b) Show that the BatchNorm gradient has a projection component: it removes the mean and the component parallel to $\hat{\mathbf{z}}$ from the upstream gradient.

(c) Implement BatchNorm forward and backward pass from scratch (without autograd). Verify your gradients numerically. Confirm that gradient checkpointing (recomputing $\hat{\mathbf{z}}$ in the backward pass instead of storing it) gives the same result.

**Exercise 7 ★★ — Residual Connections and Gradient Flow**

For a depth-$L$ residual network where each block satisfies $\|F^{[l]}(\mathbf{h})\| \leq \alpha\|\mathbf{h}\|$ with $\alpha = 0.1$:

(a) Derive a lower bound on $\|\partial\mathcal{L}/\partial\mathbf{h}^{[0]}\|$ in terms of $\|\partial\mathcal{L}/\partial\mathbf{h}^{[L]}\|$, $L$, and $\alpha$.

(b) Compare this to the same bound for a plain (non-residual) network with identical blocks. At what depth $L$ does the residual bound become $10\times$ better than the non-residual bound?

(c) Simulate gradient flow through a 100-layer network: measure the gradient norm at each layer for both residual and non-residual architectures with ReLU activation. Verify the theoretical bounds hold empirically.

**Exercise 8 ★★★ — Neural Tangent Kernel Matrix**

(a) Implement `ntk_matrix(X, net)` that computes the $n \times n$ NTK matrix for a mini-dataset $X \in \mathbb{R}^{n \times d}$ and a 2-layer MLP net.

(b) Verify that the NTK matrix is positive semi-definite by checking all eigenvalues $\geq 0$.

(c) Compute the kernel regression prediction $\hat{f}(X_*) = K(X_*, X)(K(X,X) + \epsilon I)^{-1}\mathbf{y}$ for a test set $X_*$. Train the actual network with gradient descent for 10,000 steps. Plot both predictions and verify they converge as width $n$ increases.

**Exercise 9 ★★★ — LoRA Gradient Analysis**

(a) For a linear model $\hat{\mathbf{y}} = (W_0 + BA)\mathbf{x}$, derive the gradients $\partial\mathcal{L}/\partial B$ and $\partial\mathcal{L}/\partial A$ for MSE loss.

(b) Show that with $B_0 = 0$ at initialisation, the initial gradient $\partial\mathcal{L}/\partial B$ equals the full gradient of a model trained with $\Delta W = 0$. Explain why this is the correct behaviour for a fine-tuning method.

(c) Compare LoRA at rank $r = 1, 4, 16, 64$ to full fine-tuning on a synthetic dataset: generate $X \in \mathbb{R}^{100 \times 512}$, $W_{\text{true}} = W_0 + U_r\Sigma_r V_r^\top$ (true update has rank $r_{\text{true}} = 8$). Measure test MSE as a function of LoRA rank and find the minimum rank achieving near-optimal performance.

**Exercise 10 ★★★ — Scaling Laws Fit**

(a) Simulate power-law scaling: generate synthetic loss vs. compute data with $\mathcal{L}(C) = A \cdot C^{-\gamma} + \epsilon$ for $A = 10$, $\gamma = 0.05$, $\epsilon = 0.5$ (irreducible noise), and fit the parameters using least squares on log-log data.

(b) Estimate the compute budget needed to reach $\mathcal{L} = 0.55$ (just above irreducible). How sensitive is this estimate to the value of $\gamma$?

(c) Simulate the Chinchilla result: for fixed $C$ FLOPS, parameterise models as $C = 6ND$ (approximately). Optimise the allocation $(N, D)$ numerically to minimise the predicted loss $\mathcal{L}(N, D) = \frac{A_N}{N^\alpha} + \frac{A_D}{D^\beta} + \epsilon$ and verify the optimal $N \propto D$ relationship.


---

## 17. Why This Matters for AI (2026 Perspective)

| Concept | AI/LLM Impact | Where You See It |
|---|---|---|
| **Universal approximation** | Theoretical justification that NNs can represent any target function — but says nothing about learning efficiency | Foundational argument for neural architectures; explains why we use NNs over fixed-basis methods |
| **Depth separation** | Explains why 96-layer GPT-4 >> 2-layer shallow network of same parameter count | Architecture design: 32–96 transformer layers standard; 1-layer baselines consistently weaker |
| **Backpropagation** | All gradient-based training of LLMs reduces to this — 175B gradient computations per forward pass, same cost as forward | PyTorch autograd, FlashAttention (fused backprop), gradient checkpointing |
| **He initialisation** | Without it, 96-layer pre-activation ResNets/transformers have vanishing gradients at init | PyTorch `nn.Linear` default, `init.kaiming_normal_`, σ$W_Q$/$W_K$ scaling |
| **Adam / AdamW** | De-facto standard optimiser for all LLM training (GPT, LLaMA, Gemini, Claude) | AdamW with β₁=0.9, β₂=0.999; cosine decay + warmup is universal recipe |
| **Dropout** | Regularisation in pre-2022 transformers; MC Dropout for uncertainty; dropout in attention weights | BERT, GPT-2 use p=0.1; modern LLMs reduce/remove dropout at scale |
| **Residual connections** | Enabled training of 100→1000+ layer networks; identity gradient path prevents vanishing | Every transformer block: x' = x + Attn(LN(x)), x'' = x' + FFN(LN(x')) |
| **LayerNorm / RMSNorm** | Stabilises transformer training; enables pre-norm architecture for deep models | LN: BERT, GPT-2/3; RMSNorm: LLaMA, Mistral, Falcon — all modern LLMs |
| **Grokking** | Phase transition from memorisation to generalisation; explains delayed improvement on benchmarks during training | Algorithmic reasoning (math, code) benchmarks; continual pre-training phenomena |
| **NTK / lazy training** | Infinite-width networks = kernel methods; explains linear probing success on frozen features | Linear probing, NLP probing classifiers, transfer learning theory |
| **Feature learning / μP** | Optimal hyperparameter transfer from small to large models; enables efficient large-scale hyperparameter search | GPT-4 hyperparameter search via μP; implemented in Tensor Programs libraries |
| **FFN as key-value memory** | Factual associations stored in FFN weights; model editing targets specific MLP layers | ROME, MEMIT model editing; factual knowledge attribution; KV cache analysis |
| **LoRA / DoRA** | Efficient fine-tuning of LLMs with 100–1000× parameter reduction; standard PEFT technique | Hugging Face PEFT library; RLHF uses LoRA for policy fine-tuning; inference adapters |
| **Mechanistic interpretability** | Understanding which circuits implement specific behaviours; induction heads explain in-context learning | Anthropic Circuits work; EleutherAI probing; Neel Nanda's TransformerLens library |
| **Scaling laws / Chinchilla** | Optimal compute allocation (model size vs. training tokens); predicts final loss from intermediate checkpoints | Chinchilla: 20 tokens/param optimal; Llama-3 70B trained 15T tokens (much more than Chinchilla optimal for inference efficiency) |

---

## Conceptual Bridge

**Looking back.** This section builds directly on the linear models of §14-01. Every neural network layer is a linear map $W\mathbf{x} + \mathbf{b}$ — the nonlinearity is the only structural addition. Ridge regression is the simplest neural network (a linear network with L2 regularisation). Logistic regression is a one-layer network with sigmoid output and cross-entropy loss. The OLS hat matrix, the Bayesian posterior, and the NTK kernel regression are all connected to the same underlying geometry of projections onto learned feature spaces.

**The unifying theme** is that neural networks learn to map inputs into a **representation space** where a simple (linear) readout can solve the task. All the mathematical machinery — backpropagation, initialisation, normalisation, regularisation, optimisation — serves one purpose: ensuring that gradient descent finds a good representation efficiently. Understanding each component through the lens of its role in representation learning makes the theory coherent rather than a collection of disconnected tricks.

**Looking forward.** The next section (§14-03, Probabilistic Models) develops the probabilistic framework that underlies:
- The loss functions derived in §6 (cross-entropy = Gaussian/Bernoulli MLE; contrastive losses = mutual information)
- The Bayesian perspective on regularisation (weight decay = Gaussian prior; dropout = variational inference)
- Generative models that produce rather than classify: VAEs, diffusion models, and autoregressive language models

The Transformer architecture (§14-05) applies neural networks to sequence modelling, replacing recurrence with attention — but every individual transformer operation (linear projections, LayerNorm, MLP) is exactly what we derived here. RNNs and LSTMs (§14-04) fill in the historical bridge between the sequential inductive bias and the attention mechanism.

```
CURRICULUM POSITION — NEURAL NETWORKS
════════════════════════════════════════════════════════════════════════

  PREREQUISITES                    THIS SECTION              LEADS TO

  §14-01 Linear Models ───────────► §14-02 Neural Networks ──► §14-03 Probabilistic
    OLS, Ridge, Lasso                 Backprop, UAT              Models (VAE, EM)
    NTK introduction                  Initialisation
    Logistic regression               Optimisers (Adam)        ──► §14-04 RNN/LSTM
                                      Normalisation              Sequential models
  Chapters 02-05                      Residuals, Dropout         BPTT
    Linear algebra                    NTK regime
    Calculus, Jacobians               LoRA revisited           ──► §14-05 Transformer
    Probability                       Scaling laws               Attention = §6.4
    Optimisation                      Mechanistic interp         + residuals
                                                                 + LN + MLP

════════════════════════════════════════════════════════════════════════

  CENTRAL INSIGHT: A neural network is a learned feature extractor φ
  followed by a linear readout W^[L]. All mathematics in this section
  serves to make gradient descent find a good φ efficiently.

════════════════════════════════════════════════════════════════════════
```

---

## References

1. Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
2. Barron, A. R. (1993). Universal approximation bounds for superpositions of a sigmoidal function. *IEEE Transactions on Information Theory*, 39(3), 930–945.
3. Telgarsky, M. (2016). Benefits of depth in neural networks. *COLT 2016*.
4. Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *AISTATS 2010*.
5. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers. *ICCV 2015*.
6. Ioffe, S., & Szegedy, C. (2015). Batch normalization. *ICML 2015*.
7. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. *ICLR 2015*.
8. Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR 2019*.
9. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
10. Jacot, A., Gabriel, F., & Hongler, C. (2018). Neural tangent kernel. *NeurIPS 2018*.
11. Belkin, M., Hsu, D., Ma, S., & Mandal, S. (2019). Reconciling modern machine learning practice and the classical bias-variance trade-off. *PNAS 2019*.
12. Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.
13. Geva, M., et al. (2021). Transformer feed-forward layers are key-value memories. *EMNLP 2021*.
14. Power, A., et al. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *ICLR 2022*.
15. Hoffmann, J., et al. (2022). Training compute-optimal large language models (Chinchilla). *NeurIPS 2022*.
16. Yang, G., & Hu, E. J. (2021). Feature learning in infinite-width neural networks. *ICML 2021*.
17. Elhage, N., et al. (2022). Toy models of superposition. *Transformer Circuits Thread*.
18. Santurkar, S., et al. (2018). How does batch normalization help optimization? *NeurIPS 2018*.
19. Poole, B., et al. (2016). Exponential expressivity in deep neural networks through transient chaos. *NeurIPS 2016*.
20. Montufar, G., et al. (2014). On the number of linear regions of deep neural networks. *NeurIPS 2014*.

---

## Appendix A: Key Derivations in Full

### A.1 Softmax Cross-Entropy Gradient

The softmax function maps logit vector $\mathbf{z} \in \mathbb{R}^K$ to probability vector $\hat{\mathbf{p}}$:

$$\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

The Jacobian of softmax is:
$$\frac{\partial \hat{p}_k}{\partial z_j} = \hat{p}_k(\delta_{kj} - \hat{p}_j)$$

In matrix form: $\frac{\partial \hat{\mathbf{p}}}{\partial \mathbf{z}} = \text{diag}(\hat{\mathbf{p}}) - \hat{\mathbf{p}}\hat{\mathbf{p}}^\top$.

For categorical cross-entropy loss $\mathcal{L} = -\sum_k y_k \log \hat{p}_k$ with one-hot $\mathbf{y}$:

$$\frac{\partial \mathcal{L}}{\partial \hat{p}_k} = -\frac{y_k}{\hat{p}_k}$$

Applying chain rule through softmax:
$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_k \frac{\partial \mathcal{L}}{\partial \hat{p}_k} \cdot \frac{\partial \hat{p}_k}{\partial z_j} = \sum_k \left(-\frac{y_k}{\hat{p}_k}\right)\hat{p}_k(\delta_{kj} - \hat{p}_j) = \hat{p}_j - y_j$$

**The clean gradient** $\partial\mathcal{L}/\partial\mathbf{z} = \hat{\mathbf{p}} - \mathbf{y}$ is the reason softmax + cross-entropy is the canonical output combination for classification. The gradient has magnitude proportional to prediction error and vanishes exactly when the model predicts perfectly.

**Numerical stability.** Computing $\text{softmax}(\mathbf{z})$ directly overflows for $z_k > 709$ (float32). The numerically stable version uses the log-sum-exp trick:
$$\log \hat{p}_k = z_k - \log\sum_j e^{z_j} = z_k - z_{\max} - \log\sum_j e^{z_j - z_{\max}}$$
where $z_{\max} = \max_j z_j$. This shifts the exponents to be $\leq 0$, preventing overflow while preserving the result.

### A.2 LSTM Gates and the Constant Error Carousel

**The vanishing gradient problem for RNNs** is most clearly illustrated by the gradient magnification required for long-range dependencies. For a vanilla RNN, the gradient from time step $T$ to time step $t$ involves a product of $T-t$ matrices:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t}^{T-1} W_{hh}^\top \text{diag}(\tanh'(\mathbf{z}_k))$$

Hochreiter (1991) identified this and proposed the **Constant Error Carousel (CEC)**: a memory cell $c_t$ whose update equation has a fixed-point gradient of 1:

$$c_t = c_{t-1} \cdot f_t + \tilde{c}_t \cdot i_t$$

The gradient through the cell state: $\partial c_t / \partial c_{t-1} = f_t$. When the forget gate $f_t \approx 1$ (remember everything), the gradient flows back unchanged — the CEC.

**Full LSTM equations with gradient analysis:**

$$f_t = \sigma(W_f[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_f) \qquad \text{(forget gate: what to erase)}$$
$$i_t = \sigma(W_i[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_i) \qquad \text{(input gate: what to write)}$$
$$\tilde{c}_t = \tanh(W_c[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_c) \qquad \text{(candidate cell state)}$$
$$o_t = \sigma(W_o[\mathbf{h}_{t-1}, \mathbf{x}_t] + \mathbf{b}_o) \qquad \text{(output gate: what to read)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$\mathbf{h}_t = o_t \odot \tanh(c_t)$$

The cell state $c_t$ is the critical innovation: it has an **additive** (not multiplicative) update path, allowing gradients to flow backwards through time across hundreds of steps when $f_t \approx 1$.

**Connection to residual networks:** The LSTM cell state $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$ is a gated residual connection across time. ResNets (§10.1) can be seen as taking this idea to the spatial/depth domain: $\mathbf{h}^{[l+1]} = \mathbf{h}^{[l]} + F(\mathbf{h}^{[l]})$ is an LSTM cell with $f = i = \mathbf{1}$ (always forget 0, always write 100%).

### A.3 Attention Mechanism as Weighted Averaging

Although attention is covered in depth in §14-05, we derive it here from the neural network perspective to connect it to the MLP framework.

**Motivation.** In a standard MLP processing sequences, each output position computes a fixed-weight combination of input positions determined by learned weights. There is no mechanism to dynamically weight different positions based on content. Attention provides this: the **weights** are themselves a function of the inputs.

**Scaled dot-product attention.** Given queries $Q \in \mathbb{R}^{n \times d_k}$, keys $K \in \mathbb{R}^{m \times d_k}$, values $V \in \mathbb{R}^{m \times d_v}$:

$$\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

The term $QK^\top/\sqrt{d_k} \in \mathbb{R}^{n \times m}$ is the matrix of attention scores — each query's compatibility with each key. Softmax converts scores to weights (non-negative, sum to 1 per row). The output is a weighted sum of values.

**The $1/\sqrt{d_k}$ scaling.** If $Q$ and $K$ have i.i.d. entries from $\mathcal{N}(0,1)$, then $(QK^\top)_{ij} = \mathbf{q}_i^\top \mathbf{k}_j \sim \mathcal{N}(0, d_k)$ (sum of $d_k$ products of standard normals). Variance grows linearly with $d_k$. Without scaling, for large $d_k$ (e.g., $d_k = 64$ in GPT-2), the logits are so large that softmax saturates — the attention distribution becomes nearly one-hot, providing no useful aggregation. Scaling by $1/\sqrt{d_k}$ restores unit variance.

### A.4 The Adam Learning Rate Schedule in LLM Training

**Standard recipe for LLM training (2024 practice):**

1. **Warmup:** Linear increase from 0 to $\eta_{\max}$ over $T_{\text{warm}}$ steps (typically 1–4% of total)
2. **Cosine decay:** $\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi(t - T_{\text{warm}})/(T - T_{\text{warm}})))$
3. **End:** $\eta_{\min} = 0.1 \cdot \eta_{\max}$ (do not decay to zero)

**Peak learning rates by model size** (approximate, based on published papers):

| Model | $N$ (params) | $\eta_{\max}$ | Batch size | $T_{\text{warm}}$ |
|---|---|---|---|---|
| LLaMA-7B | 7B | $3 \times 10^{-4}$ | 4M tokens | 2000 steps |
| LLaMA-65B | 65B | $1.5 \times 10^{-4}$ | 4M tokens | 2000 steps |
| GPT-3 175B | 175B | $6 \times 10^{-5}$ | 3.2M tokens | 375M tokens |
| Chinchilla 70B | 70B | $1 \times 10^{-4}$ | 1.5M tokens | - |

Larger models use smaller learning rates — consistent with the Adam effective step size scaling as $\eta/\sqrt{\hat{v}}$. For very large models, the gradient variance $\hat{v}$ is larger, so a smaller $\eta$ is needed for the same effective step size.


---

## Appendix B: Activation Functions — Detailed Analysis

### B.1 ReLU and Its Variants

**ReLU:** $\text{ReLU}(z) = \max(0, z)$

- **Forward:** $O(1)$ per element (max operation)
- **Backward:** $\partial/\partial z = \mathbf{1}[z > 0]$ — gradient is either 0 or 1, no saturation for positive inputs
- **Mean activation:** For $z \sim \mathcal{N}(0, \sigma^2)$: $\mathbb{E}[\text{ReLU}(z)] = \sigma/\sqrt{2\pi}$ (half-normal mean)
- **Variance of activation:** $\text{Var}(\text{ReLU}(z)) = \sigma^2/2$ — precisely the He init $\sigma^2 = 2/d_{\text{in}}$ factor

**The dying ReLU problem.** A neuron "dies" when its pre-activation is always negative. This happens when:
1. A large negative bias develops during training
2. A weight update pushes the neuron into the negative region for all training examples

Once dead, the neuron never receives a nonzero gradient (since $\partial\text{ReLU}/\partial z = 0$ for $z < 0$) and cannot recover.

**Leaky ReLU:** $\text{LReLU}(z) = \max(\alpha z, z)$ for $\alpha \in (0,1)$, typically $\alpha = 0.01$. Prevents dying neurons by ensuring gradient is always $\geq \alpha > 0$. Widely used in GANs (DCGAN uses LReLU in discriminator).

**Parametric ReLU (PReLU):** Learn the slope $\alpha$ per neuron or per channel. He et al. (2015) used PReLU in the original He initialisation paper. Slight overfitting risk.

### B.2 GELU Detailed

**GELU** (Hendrycks & Gimpel, 2016): $\text{GELU}(z) = z \cdot \Phi(z)$ where $\Phi$ is the standard normal CDF.

**Stochastic interpretation:** $\text{GELU}(z) = \mathbb{E}_{m \sim \text{Bernoulli}(\Phi(z))}[m \cdot z]$. The neuron stochastically gates its input proportional to the probability that a standard normal exceeds $z$. This is a **differentiable** and **smooth** version of ReLU where the gating probability increases smoothly from 0 to 1.

**Comparison to ReLU:**
- Both compute a form of $z \cdot [\text{gate}]$, but ReLU's gate is hard ($\mathbf{1}[z > 0]$) while GELU's is soft ($\Phi(z)$)
- GELU is smooth everywhere (differentiable at $z = 0$), while ReLU has a kink
- GELU is slightly negative for $z \in (-0.17, 0)$ (non-monotone in this small region) — this is by design, providing a slight regularisation effect

**Fast approximation:** $\text{GELU}(z) \approx 0.5z(1 + \tanh(\sqrt{2/\pi}(z + 0.044715z^3)))$ — used in GPT-2, BERT for efficiency.

### B.3 SiLU/Swish

**SiLU (Sigmoid Linear Unit) / Swish:** $\text{SiLU}(z) = z \cdot \sigma(z) = z/(1+e^{-z})$

- Proposed simultaneously by Ramachandran et al. (2017) as "Swish" via architecture search and by Elfwang et al. as SiLU
- **Non-monotone:** $\text{SiLU}'(z) = \sigma(z) + z\sigma(z)(1-\sigma(z))$. The derivative can be $< 0$ for $z < 0$ (around $z \approx -1.7$)
- Used in LLaMA (combined with gating as SwiGLU), EfficientNet, and various vision transformers

**SwiGLU:** $\text{SwiGLU}(\mathbf{x}) = W_3(\text{SiLU}(W_1\mathbf{x}) \odot W_2\mathbf{x})$. Noam Shazeer (2020) showed this consistently outperforms ReLU/GELU in transformers. The gating mechanism allows the network to learn to completely block some channels ($W_2\mathbf{x} \approx 0$) while amplifying others.

---

## Appendix C: Optimiser Comparison Table

| Optimiser | Update | LR sensitivity | Memory | Best for |
|---|---|---|---|---|
| SGD | $-\eta g$ | High | $1\times\theta$ | Convex, simple losses |
| SGD+Mom | $-v$, $v = \gamma v + \eta g$ | Medium | $2\times\theta$ | CV with LR schedule |
| AdaGrad | $-\eta g/\sqrt{G}$ | Low | $2\times\theta$ | Sparse gradients (NLP embeds) |
| RMSProp | $-\eta g/\sqrt{v}$, $v = \rho v + (1-\rho)g^2$ | Low-medium | $2\times\theta$ | RNN training |
| Adam | $-\eta \hat{m}/(\sqrt{\hat{v}}+\epsilon)$ | Low | $3\times\theta$ | General deep learning |
| AdamW | Adam + direct weight decay | Low | $3\times\theta$ | Transformer LLM training |
| LAMB | Layer-wise adaptive moments | Very low | $3\times\theta$ | Very large batch training |
| Lion | Sign(momentum) update | Low | $2\times\theta$ | Efficient training, same quality |
| Sophia | Second-order Hessian diagonal | Very low | $3\times\theta$ | Language models, fewer steps |

**Lion** (Chen et al., 2023): Uses only the sign of the first moment: $\theta \leftarrow \theta - \eta \cdot \text{sign}(\beta_1 m + (1-\beta_1)g)$; then $m \leftarrow \beta_1 m + (1-\beta_1)g$. More memory-efficient than Adam (no $v_t$) with comparable quality. Google Brain used it for Gemini pre-training.

**Sophia** (Liu et al., 2023): Uses a diagonal Hessian estimate for per-parameter curvature, allowing larger steps in flat directions. Achieves same validation loss as Adam in $\sim$2× fewer steps.

---

## Appendix D: Practical Debugging Checklist

```
NEURAL NETWORK DEBUGGING PROTOCOL
════════════════════════════════════════════════════════════════════════

  SANITY CHECK (before training)
  ─────────────────────────────────────────────────────────────────────
  □ Forward pass: output shape matches expected
  □ Loss at init: should be -log(1/K) for K-class CE (= log K)
                  should be ~var(y) for MSE regression
  □ Gradient check: relative error < 1e-5 (use double precision)
  □ Overfit one batch: 0 training loss achievable? (tests expressivity)

  TRAINING INSTABILITY
  ─────────────────────────────────────────────────────────────────────
  □ NaN loss? → gradient explosion → add clipping, reduce LR, check init
  □ Loss not decreasing? → LR too small, or bug in backward pass
  □ Loss decreasing too slowly? → LR too large (oscillating) or momentum
  □ Validation much worse than train? → overfitting, add regularisation

  GRADIENT FLOW
  ─────────────────────────────────────────────────────────────────────
  □ Print gradient norms per layer: should be O(1), roughly equal
  □ Any layer with near-zero grads? → vanishing, use residuals/BN
  □ Any layer with large grads? → exploding, add clipping
  □ Dead ReLU check: count neurons always producing 0

  NORMALISATION
  ─────────────────────────────────────────────────────────────────────
  □ BatchNorm: call model.eval() at test time (CRITICAL)
  □ Verify batch statistics are being tracked (model.training flag)
  □ LayerNorm: double-check axis (features, not batch)

════════════════════════════════════════════════════════════════════════
```


---

## Appendix E: Mathematical Properties Glossary

### E.1 Convexity and Neural Networks

**The non-convexity of neural network losses.** For linear models (§14-01), the cross-entropy loss is strictly convex — there is a unique global minimum and gradient descent converges to it. Neural networks are not convex: $\mathcal{L}(\boldsymbol{\theta})$ is non-convex in $\boldsymbol{\theta}$ whenever depth $L \geq 2$ (because composing two or more linear layers gives permutation symmetry — swapping two neurons in a hidden layer with their weights gives an identical function but a different $\boldsymbol{\theta}$).

**Permutation symmetry.** For a two-hidden-layer network, permuting the neurons in layer 1 (simultaneously permuting rows of $W^{[1]}$ and rows of $W^{[2]}$) gives the same function. This means every non-degenerate critical point has at least $(d_1!)$ equivalent points — a form of non-uniqueness that does not affect training but makes the loss landscape highly symmetric.

**Implications for optimisation:**
1. Local minima that are not global minima **do exist** (e.g., Du & Lee, 2018 constructed examples)
2. However, for overparameterised networks on smooth losses, all local minima that are not global minima are **measure-zero saddle points** in practice
3. SGD's noise prevents convergence to saddle points (random perturbation provides escape direction)

### E.2 Lipschitz Continuity and Robustness

**Definition.** A function $f: \mathbb{R}^d \to \mathbb{R}^k$ is $L$-Lipschitz if for all $\mathbf{x}, \mathbf{x}'$:
$$\|f(\mathbf{x}) - f(\mathbf{x}')\| \leq L\|\mathbf{x} - \mathbf{x}'\|$$

**Lipschitz constant of an MLP.** By the chain rule and submultiplicativity of norms:

$$L_f \leq \prod_{l=1}^L \|W^{[l]}\|_2 \cdot L_\sigma^L$$

where $L_\sigma = 1$ for ReLU, GELU, tanh (all 1-Lipschitz). The product of spectral norms bounds the global Lipschitz constant.

**Spectral normalisation (Miyato et al., 2018).** Divides each weight matrix by its spectral norm $\|W\|_2$: $W \leftarrow W/\|W\|_2$. This constrains $L_f \leq 1$ — a 1-Lipschitz network. Used in Wasserstein GANs (WGAN-GP) and robustness-certified classifiers.

**For LLMs:** The output probability of a language model changes continuously with the input embedding. The Lipschitz constant bounds how much the probability changes for small embedding perturbations — important for understanding adversarial examples and the stability of in-context learning.

### E.3 Equivariance and Invariance

**Definition.** A function $f: \mathbb{R}^d \to \mathbb{R}^k$ is **equivariant** to transformation group $G$ if $f(g\mathbf{x}) = g'f(\mathbf{x})$ for all $g \in G$ (transformation in input maps to corresponding transformation in output). It is **invariant** if $f(g\mathbf{x}) = f(\mathbf{x})$.

**Standard MLPs** are **not equivariant** to permutation of input features — if you permute the pixels of an image, the MLP output changes arbitrarily. This is why MLPs are sample-inefficient for vision and sequences: they must learn each permutation separately.

**Convolutional networks** are **translation-equivariant**: shifting the input shifts the feature map by the same amount (before pooling). This allows convolutions to learn features that work anywhere in the image with a single kernel.

**Transformer self-attention** is **permutation-equivariant** (without positional encoding): permuting the sequence of tokens permutes the output sequence in the same way. Positional encoding breaks this equivariance to introduce order sensitivity.

### E.4 The No-Free-Lunch Theorem and Inductive Bias

**No-Free-Lunch (NFL) theorem (Wolpert, 1996).** Averaged over all possible data distributions, every supervised learning algorithm has the same expected out-of-sample error. There is no universally best learning algorithm.

**Implication for neural networks.** Neural networks are not universally better than other methods — they encode specific **inductive biases**:
- **MLPs:** Smooth function composition; no preference for spatial locality
- **CNNs:** Local feature detection; translation equivariance; weight sharing
- **RNNs:** Sequential processing; long-range memory (LSTM)
- **Transformers:** Pairwise token interactions; learned positional structure

NFL says: neural networks only outperform other methods when the data distribution matches the inductive bias. Transformers dominate language modelling because language has pairwise token interactions. CNNs dominate image recognition because images have local spatial structure.

**Practical takeaway.** When designing an architecture, ask: "What inductive bias am I encoding, and does it match the structure of my data?" This is the engineering design question that underlies all modern architecture innovations.

---

## Appendix F: Numerical Recipes for Common Operations

### F.1 Log-Sum-Exp (Numerically Stable)

```
log(exp(a) + exp(b)) = a + log(1 + exp(b - a))    [if a > b]
                     = log(exp(a) + exp(b))
```

Generalised to vector $\mathbf{z}$:
$$\text{LSE}(\mathbf{z}) = \log\sum_k e^{z_k} = z_{\max} + \log\sum_k e^{z_k - z_{\max}}$$

Used in: softmax (stable version), log-sum-exp pooling, log-partition function of CRFs.

### F.2 Softmax with Temperature

$$\text{softmax}(\mathbf{z}/T)_k = \frac{e^{z_k/T}}{\sum_j e^{z_j/T}}$$

- $T \to 0$: argmax (one-hot), perfectly sharp
- $T = 1$: standard softmax
- $T \to \infty$: uniform distribution, maximum entropy

Temperature sampling is used in LLM text generation: $T = 0.7$–$1.0$ gives reasonable diversity; $T = 0$ gives greedy decoding.

### F.3 Gradient Norm Monitoring

During training, log gradient norms per layer every 100 steps:
```
global_norm = sqrt(sum(||grad||^2 for all params))
per_layer_norm[l] = ||grad_W^[l]||_F
```

**Healthy patterns:**
- Global norm: stable $\in [0.1, 10]$ for typical models; spikes (up to $100\times$ normal) are ok if gradient clipping catches them
- Per-layer norms: roughly equal across layers (may decrease for early layers in plain networks; should be roughly equal for ResNets)
- Ratio $\|\Delta\boldsymbol{\theta}\|/\|\boldsymbol{\theta}\|$ (update ratio): should be $\sim 10^{-3}$ (LR/grad magnitude); too large = LR too high, too small = LR too low


---

## Appendix G: Detailed Proofs

### G.1 Proof: Adam Bias Correction Is Necessary

**Claim.** Without bias correction, Adam systematically underestimates gradient magnitudes in early steps.

**Proof.** Expanding the first-moment recursion:
$$m_t = (1-\beta_1)\sum_{k=1}^t \beta_1^{t-k} g_k$$

Under the assumption that gradients $g_1, \ldots, g_t$ are i.i.d. with $\mathbb{E}[g] = \bar{g}$:
$$\mathbb{E}[m_t] = (1-\beta_1)\sum_{k=1}^t \beta_1^{t-k} \bar{g} = (1-\beta_1^t)\bar{g}$$

So $m_t$ estimates $(1-\beta_1^t)\bar{g}$, not $\bar{g}$. The factor $(1-\beta_1^t)$ is:
- At $t=1$: $(1-0.9) = 0.1$ — gradient underestimated by $10\times$
- At $t=10$: $(1-0.9^{10}) \approx 0.65$ — underestimated by $1.5\times$
- At $t=100$: $(1-0.9^{100}) \approx 1$ — fully converged

The corrected estimate $\hat{m}_t = m_t / (1-\beta_1^t)$ removes this bias. Similarly for $v_t$. $\square$

**Why the early underestimation matters.** In step 1, without correction: effective step = $\eta \cdot 0.1g / \sqrt{0.001 g^2} = \eta \cdot 0.1/\sqrt{0.001} \approx 3\eta$. With correction: effective step $\approx \eta$. The uncorrected early steps are $3\times$ too large for Adam — leading to instability at high learning rates.

### G.2 Proof: Residual Networks Have Non-Vanishing Gradients

**Claim.** For a residual network with blocks satisfying $\|F^{[l]}(\mathbf{h})\| \leq \alpha\|\mathbf{h}\|$ ($\alpha < 1$), the gradient satisfies:
$$\left\|\frac{\partial\mathcal{L}}{\partial\mathbf{h}^{[0]}}\right\| \geq (1-\alpha)^L \left\|\frac{\partial\mathcal{L}}{\partial\mathbf{h}^{[L]}}\right\|$$

**Proof.** The Jacobian of one residual block: $\partial\mathbf{h}^{[l+1]}/\partial\mathbf{h}^{[l]} = I + \partial F^{[l]}/\partial\mathbf{h}^{[l]}$.

By triangle inequality: $\|I + \partial F^{[l]}/\partial\mathbf{h}^{[l]}\| \geq \|I\| - \|\partial F^{[l]}/\partial\mathbf{h}^{[l]}\| \geq 1 - \alpha$.

For the chain of $L$ blocks:
$$\left\|\frac{\partial\mathbf{h}^{[L]}}{\partial\mathbf{h}^{[0]}}\right\| \geq \prod_{l=1}^L (1-\alpha) = (1-\alpha)^L$$

Since $\|\partial\mathcal{L}/\partial\mathbf{h}^{[0]}\| \geq (1-\alpha)^L \|\partial\mathcal{L}/\partial\mathbf{h}^{[L]}\|$, the gradient at layer 0 is bounded below by an exponentially decaying but non-vanishing factor. For $\alpha = 0.1$ and $L = 100$: $(0.9)^{100} \approx 2.66 \times 10^{-5}$ — tiny but nonzero. $\square$

**Compare to non-residual.** A non-residual network with Lipschitz constant $\alpha$ per block: $\|\partial\mathbf{h}^{[L]}/\partial\mathbf{h}^{[0]}\| \leq \alpha^L$. For $\alpha = 0.9$: same $(0.9)^{100} \approx 2.66 \times 10^{-5}$ as an upper bound. The crucial difference: for residual networks, the factor is a lower bound on the gradient; for non-residual, it is an upper bound. In practice, non-residual blocks have $\alpha \ll 1$ (many singular values of $W$ are small), making the upper bound exponentially tight.

### G.3 Proof: NTK Is Positive Semi-Definite

**Claim.** The NTK matrix $\Theta_{ij} = \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^{(i)})^\top \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^{(j)})$ is positive semi-definite.

**Proof.** For any vector $\mathbf{c} \in \mathbb{R}^n$:

$$\mathbf{c}^\top \Theta \mathbf{c} = \sum_{i,j} c_i c_j \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^{(i)})^\top \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^{(j)}) = \left\|\sum_i c_i \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^{(i)})\right\|^2 \geq 0$$

since it is the squared norm of a vector. $\square$

**Strict positive definiteness** requires that $\sum_i c_i \nabla_{\boldsymbol{\theta}} f(\mathbf{x}^{(i)}) \neq \mathbf{0}$ for all nonzero $\mathbf{c}$. This holds generically for over-parameterised networks — the gradients of the network function at distinct input points span the full gradient space.

### G.4 Proof: BatchNorm Gradient Has Centring Property

**Claim.** The gradient $\partial\mathcal{L}/\partial\mathbf{z}$ through BatchNorm has zero mean and is orthogonal to $\hat{\mathbf{z}}$.

**Proof.** From the BatchNorm backward pass:
$$\frac{\partial\mathcal{L}}{\partial z_i} = \frac{\gamma}{\sqrt{\sigma_B^2+\epsilon}}\left[\frac{\partial\mathcal{L}}{\partial \hat{z}_i} - \frac{1}{B}\sum_j\frac{\partial\mathcal{L}}{\partial \hat{z}_j} - \hat{z}_i \cdot \frac{1}{B}\sum_j \frac{\partial\mathcal{L}}{\partial\hat{z}_j}\hat{z}_j\right]$$

**Zero mean:** $\sum_i \partial\mathcal{L}/\partial z_i = 0$ because the $1/B\sum_j(\cdot)$ terms cancel the mean.

**Orthogonal to $\hat{\mathbf{z}}$:** $\sum_i \hat{z}_i \cdot \partial\mathcal{L}/\partial z_i = 0$ because the third term removes this component. $\square$

These two properties mean BatchNorm prevents gradient updates that would shift the batch mean (mean-change is discarded) or re-scale the batch variance (variance-preserving direction is discarded). Only orthogonal updates — those that change the shape of the activation distribution — pass through. This is why BatchNorm makes training more stable: uninformative components of the gradient (scale and shift) are automatically filtered out.

---

## Appendix H: Historical Notes on Activation Functions

The history of activation functions mirrors the history of our understanding of the vanishing gradient problem:

**1943 (McCulloch-Pitts):** Step function $\mathbf{1}[z > 0]$ — Boolean logic, no gradient at all.

**1957–1980 (Rosenblatt, Minsky-Papert era):** Sigmoid $\sigma(z)$ adopted because it is differentiable and matches the firing rate model of neurons. The saturation problem was not yet recognised as critical.

**1986 (Rumelhart, Hinton & Williams):** Backpropagation rediscovered with sigmoid. Networks worked for shallow architectures (2–3 layers) but gradient vanishing limited depth.

**1991 (Hochreiter):** First rigorous analysis of vanishing gradients for sigmoid in deep networks. Identified maximum derivative 0.25 as the source of exponential decay.

**2010 (Nair & Hinton):** ReLU proposed for restricted Boltzmann machines. Showed that $\max(0,z)$ avoids saturation for positive inputs.

**2011 (Glorot, Bordes & Bengio):** First systematic study showing ReLU outperforms sigmoid/tanh for deep networks. The "ReLU revolution" begins.

**2015 (Maas et al.; He et al.):** Leaky ReLU and PReLU to fix dying neurons. Empirically slight improvement over ReLU.

**2016 (Hendrycks & Gimpel):** GELU proposed. Largely ignored until BERT (2018) used it, after which it became the default for language models.

**2017 (Ramachandran et al.):** Neural architecture search discovers Swish/SiLU as best activation on image tasks. Adopted by EfficientNet (2019) and LLaMA (2023, as SwiGLU variant).

**2020+ (LLaMA, Mistral, Falcon):** SwiGLU (Shazeer, 2020) becomes the default for open-source LLMs. ReLU/GELU used in proprietary models. Mish, ELU, and other variants continue to be explored.

The progression: step (not differentiable) → sigmoid (differentiable but saturating) → ReLU (non-saturating, sparse) → GELU/SiLU (smooth, non-monotone gating) reflects accumulating understanding of what properties matter for deep learning.


---

## Appendix I: Connections Between Sections

### I.1 The Unified View: From Linear to Nonlinear

The following table shows how every concept in §14-01 (Linear Models) generalises to §14-02 (Neural Networks):

| Linear Models (§14-01) | Neural Networks (§14-02) | Mathematical Relationship |
|---|---|---|
| OLS $\hat{\boldsymbol{\beta}} = (X^\top X)^{-1}X^\top\mathbf{y}$ | Full fine-tuning with MSE loss | NN generalises OLS by learning features $\phi(\mathbf{x})$ |
| Ridge regression $\hat{\boldsymbol{\beta}}_\lambda$ | Weight decay (AdamW) | Same $\ell^2$ penalty; connects AdamW to Bayesian Gaussian prior |
| Lasso soft-thresholding | Dropout, sparse attention | Both induce sparsity; dropout ≈ Bernoulli prior on weights |
| Logistic regression gradient $X^\top(\hat{p}-y)/n$ | Backpropagation output delta $\boldsymbol{\delta}^{[L]} = \hat{p}-y$ | Identical formula — neural networks generalise LR |
| OLS hat matrix $H = X(X^\top X)^{-1}X^\top$ | Attention matrix $A = \text{softmax}(QK^\top/\sqrt{d_k})$ | Both are projection/averaging matrices; A is learned and input-dependent |
| Ridge bias-variance tradeoff | Training/test error with regularisation | Same mathematical framework; ridge is linear case |
| Bayesian posterior = Ridge MAP | NTK kernel regression | Both are kernel methods; NTK kernel is architecture-dependent |
| Double descent at interpolation threshold | Grokking phase transition | Both: non-monotone test error vs. complexity/time |
| LoRA $\Delta W = BA$, low rank | LoRA in neural networks | Identical construction; generalises to any layer |
| NTK introduction | Full NTK theory (§13) | §14-01 NTK is for linear networks; §14-02 extends to nonlinear |
| LDA/QDA | Pre-training + linear probes | LDA = linear readout on Gaussian features; probe = linear readout on learned features |
| SVM dual $\sum \alpha_i y_i \mathbf{x}_i$ | Last-layer features as support vectors | Linear separability in feature space; SVM = optimal linear classifier |

### I.2 How This Section Connects Forward

**§14-03 Probabilistic Models** takes the loss functions of §6 and gives them a formal probabilistic foundation. The cross-entropy loss is the MLE for the Bernoulli/categorical distribution (derived here informally, formalised there). Variational autoencoders use the same backpropagation algorithm but with a KL divergence term in the loss (the ELBO).

**§14-04 RNN and LSTM Math** applies backpropagation to sequences: the same $\boldsymbol{\delta}^{[l]}$ equations but with a time dimension (BPTT). The vanishing gradient analysis (§11.1) is central to motivating LSTMs. The LSTM derivation in Appendix A.2 serves as direct preparation.

**§14-05 Transformer Architecture** is the synthesis: self-attention replaces recurrence for sequence processing, but the feedforward layers, normalisation, residual connections, and training dynamics (Adam, warmup, cosine decay) are all from this section. Section 2.4 on the computation graph, §9.2 on LayerNorm, §10 on residuals, and §7.4 on Adam are all directly prerequisite.

**§14-06 Reinforcement Learning** uses neural networks as value function approximators and policy networks. The policy gradient theorem requires computing $\nabla_{\boldsymbol{\theta}} \log \pi_{\boldsymbol{\theta}}(\mathbf{a}|\mathbf{s})$ — directly using backpropagation (§4). Trust-region methods (TRPO, PPO) constrain the parameter update to prevent instability — analogous to gradient clipping (§11.2).

---

## Appendix J: Proof of Universal Approximation via Stone-Weierstrass

An alternative, arguably more elegant proof of the universal approximation theorem uses the Stone-Weierstrass theorem from functional analysis.

**Stone-Weierstrass Theorem.** Let $K$ be a compact Hausdorff space and $\mathcal{A} \subseteq C(K)$ be a subalgebra (closed under addition, scalar multiplication, and pointwise multiplication) such that:
1. $\mathcal{A}$ separates points: for $x \neq y \in K$, there exists $f \in \mathcal{A}$ with $f(x) \neq f(y)$
2. $\mathcal{A}$ contains the constant functions

Then $\mathcal{A}$ is dense in $C(K)$ with the sup-norm.

**Application to neural networks.** Define $\mathcal{A}_\sigma$ as the closure of the span of $\{\mathbf{x} \mapsto \sigma(\mathbf{w}^\top\mathbf{x} + b) : \mathbf{w} \in \mathbb{R}^d, b \in \mathbb{R}\}$. If $\sigma$ is a polynomial (i.e., not piecewise constant), then products of functions in $\mathcal{A}_\sigma$ are polynomials in $\sigma$ evaluations — these can approximate any continuous function.

For non-polynomial $\sigma$ (including sigmoid, tanh, ReLU), the argument requires more care: the key property is that the linear span of neurons already separates points (for discriminatory $\sigma$), so no closure under products is needed.

**Lore and limitations.** The Stone-Weierstrass route provides clean proof structure but is less constructive than Cybenko's original proof. Both proofs are non-constructive: they prove existence of an approximating network but give no algorithm to find it. The constructive approximation theory (Barron, 1993, §3.2) provides explicit width bounds at the cost of restricting the function class.


---

## Appendix K: Layer-by-Layer Analysis of a Trained Language Model

A detailed anatomy of what each layer type computes in a trained GPT-style transformer, viewed through the lens of §14-02's neural network mathematics.

### K.1 Embedding Layer (Input)

**Mathematical operation:** Lookup table $E \in \mathbb{R}^{V \times d}$. Input token $t \in \{1,\ldots,V\}$ maps to $E_{t,:} \in \mathbb{R}^d$.

**Neural network interpretation:** This is a one-hidden-layer MLP with no activation: $E \cdot \mathbf{e}_t$ where $\mathbf{e}_t$ is the one-hot token vector. The embedding layer is a linear projection of discrete token identity into a continuous vector space.

**What is learned:** Syntactic and morphological similarity (nearby embeddings = similar words). After training, $\|E_{t,:} - E_{t',:}\|_2$ correlates with semantic distance. Principal components of $E$ capture interpretable directions (e.g., PC1: word frequency; PC2/3: part-of-speech).

### K.2 Attention Sublayer

**Mathematical operation:** $\mathbf{x}' = \mathbf{x} + \text{MHA}(\text{LN}(\mathbf{x}))$ (pre-norm style).

**Neural network interpretation:** The QKV projections ($W_Q$, $W_K$, $W_V$) are linear maps. The attention weights $A = \text{softmax}(QK^\top/\sqrt{d_k})$ are a data-dependent mixing matrix. The output $AV$ is a weighted average of value vectors — a linear combination with learned, input-dependent weights. The residual add ensures gradient flow (§10).

**What is learned:** Early layers: syntactic relationships (subject-verb agreement, coreference). Middle layers: semantic relationships (entity types, logical arguments). Late layers: task-specific patterns (question-answering, completion).

### K.3 FFN Sublayer

**Mathematical operation:** $\mathbf{x}'' = \mathbf{x}' + \text{FFN}(\text{LN}(\mathbf{x}'))$ where $\text{FFN}(\mathbf{x}) = W_2\text{GELU}(W_1\mathbf{x})$.

**Neural network interpretation:** A two-layer MLP with GELU activation (§2.3, §B.2). The first linear map $W_1$ projects into a high-dimensional space ($4d$); GELU gates each dimension; $W_2$ projects back to $d$.

**What is learned:** "Factual knowledge" — Geva et al. (2021) showed that individual FFN neurons fire for specific factual patterns ("Paris is the capital of", "The author of Hamlet is"). The FFN layers contain the majority of the model's knowledge about world facts.

### K.4 Output Layer (Unembedding)

**Mathematical operation:** $\text{logits} = E_{\text{unembed}}\mathbf{h}^{[L]}$ where $E_{\text{unembed}} \in \mathbb{R}^{V \times d}$. Often $E_{\text{unembed}} = E^\top$ (weight tying with the input embedding).

**Neural network interpretation:** This is a linear readout layer (§1.1 feature extractor + linear readout). The final hidden state $\mathbf{h}^{[L]}$ is the learned feature $\phi(\mathbf{x})$; the unembedding matrix is the linear classifier $W^{[L]}$.

**What is learned (logit lens technique):** Applying the unembedding to intermediate hidden states shows that even at layer 1–2, the most probable next token is already somewhat correct. By layer 10–20 (in a 32-layer model), the top-1 prediction is usually stable. This demonstrates that the transformer builds up the prediction incrementally across layers — consistent with the hierarchical representation view (§1.2).

---

## Appendix L: Sample Implementation Patterns

### L.1 Numerically Stable Attention

```python
# Naive (AVOID): overflows for large d_k
scores = Q @ K.T / np.sqrt(d_k)
weights = np.exp(scores) / np.exp(scores).sum(-1, keepdims=True)

# Stable (USE): subtract max before exp
scores = Q @ K.T / np.sqrt(d_k)
scores -= scores.max(-1, keepdims=True)  # log-sum-exp trick
exp_scores = np.exp(scores)
weights = exp_scores / exp_scores.sum(-1, keepdims=True)
output = weights @ V
```

### L.2 Gradient Clipping

```python
def clip_grad_norm(params, max_norm):
    total_norm = np.sqrt(sum(np.sum(p.grad**2) for p in params))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad *= clip_coef
    return total_norm
```

### L.3 Adam Update (from scratch)

```python
def adam_step(theta, grad, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * grad          # 1st moment
    v = b2 * v + (1 - b2) * grad**2       # 2nd moment
    m_hat = m / (1 - b1**t)               # bias correction
    v_hat = v / (1 - b2**t)               # bias correction
    theta -= lr * m_hat / (np.sqrt(v_hat) + eps)
    return theta, m, v
```

### L.4 He Initialisation

```python
def he_init(fan_in, fan_out, activation='relu'):
    if activation == 'relu':
        std = np.sqrt(2.0 / fan_in)
    elif activation in ('gelu', 'silu', 'swish'):
        std = np.sqrt(2.0 / fan_in)  # same as relu in practice
    elif activation == 'tanh':
        std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
    else:
        std = np.sqrt(1.0 / fan_in)  # LeCun
    return np.random.randn(fan_out, fan_in) * std
```


---

## Appendix M: Convergence Theory Summary

### M.1 SGD Convergence Regimes

The convergence of SGD depends on the structure of the loss:

**Strongly convex losses** ($\mathcal{L}(\boldsymbol{\theta}) \geq \mathcal{L}^* + \nabla\mathcal{L}^{*\top}(\boldsymbol{\theta}-\boldsymbol{\theta}^*) + \frac{\mu}{2}\|\boldsymbol{\theta}-\boldsymbol{\theta}^*\|^2$):

$$\mathbb{E}[\mathcal{L}(\boldsymbol{\theta}_T)] - \mathcal{L}^* \leq \frac{L\sigma^2}{2\mu T} \quad \text{with } \eta = \frac{1}{\mu T}$$

This is $O(1/T)$ for strongly convex losses — fast convergence. Logistic regression and ridge regression satisfy this; neural networks generally do not.

**Convex but not strongly convex** (e.g., hinge loss, unregularised logistic):
$$\mathbb{E}[\mathcal{L}(\boldsymbol{\theta}_T)] - \mathcal{L}^* \leq \frac{\|\boldsymbol{\theta}_0 - \boldsymbol{\theta}^*\|\sigma}{\sqrt{T}} \quad \text{with } \eta = \Theta(1/\sqrt{T})$$

Rate $O(1/\sqrt{T})$ — slower, and the optimal LR schedule requires knowing $T$ in advance.

**Non-convex losses** (neural networks): The standard guarantee is convergence to a **stationary point** (gradient $\approx 0$):
$$\min_{t \leq T} \mathbb{E}[\|\nabla\mathcal{L}(\boldsymbol{\theta}_t)\|^2] \leq \frac{2(\mathcal{L}(\boldsymbol{\theta}_0) - \mathcal{L}^*) + \eta L \sigma^2}{\eta T}$$

With $\eta \sim 1/\sqrt{T}$, this gives convergence rate $O(1/\sqrt{T})$ to a stationary point. Not a minimum — but empirically, stationary points found by SGD in neural networks are usually good enough.

### M.2 Adam's Convergence Properties

Adam's original paper (Kingma & Ba, 2015) claimed $O(1/\sqrt{T})$ convergence for convex losses. This was later shown to be incorrect (Reddi et al., 2018): Adam can diverge on simple convex problems due to the exponential moving average of $v_t$ not tracking the true second moment.

**AMSGrad** (Reddi et al., 2018) fixes this: $\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$ (monotone second moment) guarantees convergence. Used less in practice than Adam because the convergence issues rarely manifest with standard hyperparameters.

**In practice:** Adam converges empirically for neural networks despite the lack of theoretical guarantees. The combination of adaptive learning rates + momentum makes it robust across a wide range of loss landscapes.

### M.3 Linear Convergence in the NTK Regime

In the NTK regime (§13.1), gradient flow on MSE loss with constant NTK matrix $\Theta$ converges linearly:

$$\mathcal{L}(t) = \frac{1}{2}\|\mathbf{y} - f(X; \boldsymbol{\theta}(t))\|^2 \leq \mathcal{L}(0) \cdot e^{-2\lambda_{\min}(\Theta) t}$$

**Rate depends on $\lambda_{\min}(\Theta)$:** For overparameterised networks ($p \gg n$) with sufficient width, $\lambda_{\min}(\Theta) = \Omega(\text{poly}(1/n))$ — the smallest eigenvalue scales inversely with network depth and polynomial in $1/n$. The convergence can be slow for very deep networks.

**Du et al. (2019):** For a 2-layer network with $m$ neurons on $n$ samples, if $m = \Omega(n^6/\delta^3)$ (sufficiently wide), then gradient descent converges to zero training loss at rate $O(1/m)$ per iteration. This is the first polynomial-time convergence guarantee for training neural networks.

---

## Appendix N: Modern Architecture Variants

### N.1 Gated Linear Units (GLU)

**GLU** (Dauphin et al., 2017): $\text{GLU}(\mathbf{x}) = (W_1\mathbf{x} + \mathbf{b}_1) \odot \sigma(W_2\mathbf{x} + \mathbf{b}_2)$

A learned gating mechanism: the first branch produces values, the second produces gates in $[0,1]$. This allows the network to **silence** (gate to 0) or **pass** (gate to 1) individual dimensions.

**SwiGLU** (Shazeer, 2020): Replace $\sigma$ with SiLU: $\text{SwiGLU}(\mathbf{x}) = (\text{SiLU}(W_1\mathbf{x})) \odot (W_2\mathbf{x})$

Used in PaLM, LLaMA-2/3, Mistral, Gemma. Parameter counts differ from vanilla FFN:
- Vanilla FFN: $2 \times 4d \times d = 8d^2$
- SwiGLU FFN: $3 \times (8d/3) \times d = 8d^2$ (same total flops with $8d/3$ expansion)

### N.2 Mamba and Selective State Spaces

**S4/Mamba** (Gu et al., 2021; Gu & Dao, 2023) replaces transformer attention with a **selective state space model (SSM)**:

$$\mathbf{h}_t = A\mathbf{h}_{t-1} + B\mathbf{x}_t, \qquad y_t = C\mathbf{h}_t$$

where $A, B, C$ are **input-dependent** (selective): the model learns to selectively remember or forget based on the current input. This achieves $O(n)$ sequence processing (vs. $O(n^2)$ for attention) while maintaining long-range dependencies.

**Connection to LSTMs (§A.2).** Mamba's selective SSM is a continuous-time generalisation of LSTMs: the forget gate $f_t$ becomes $A(\mathbf{x}_t)$, the input gate is $B(\mathbf{x}_t)$, and the structure is a structured state matrix rather than a vector. Mamba can be interpreted as an LSTM with a linear state update and learned selective gating.

### N.3 Mixture of Experts (MoE)

**MoE FFN:** Replace the standard FFN with $K$ expert FFNs and a routing network:

$$\text{MoE}(\mathbf{x}) = \sum_{k=1}^K g_k(\mathbf{x}) \cdot E_k(\mathbf{x}), \qquad g(\mathbf{x}) = \text{TopK}(\text{softmax}(W_g \mathbf{x}))$$

Only the top-$r$ experts (typically $r=2$) process each token, making the effective compute per token $r/K$ times a dense FFN. GPT-4, Mixtral-8x7B, and many frontier models use sparse MoE.

**Mathematical property:** MoE is a piecewise function — each "piece" corresponds to a particular combination of active experts. With $K=8$ experts and $r=2$ active, there are $\binom{8}{2} = 28$ possible expert combinations, each defining a different linear map for that input region. This is similar to the piecewise-linear structure of ReLU networks but at a much coarser granularity.


---

## Appendix O: Quick Reference Card

```
NEURAL NETWORKS: KEY EQUATIONS
════════════════════════════════════════════════════════════════════════

  FORWARD PASS
  ─────────────────────────────────────────────────────────────────────
  Pre-activation:    z^[l] = W^[l] h^[l-1] + b^[l]
  Activation:        h^[l] = σ(z^[l])
  Network output:    ŷ = h^[L] = W^[L] h^[L-1] + b^[L]

  BACKPROPAGATION
  ─────────────────────────────────────────────────────────────────────
  Output delta:      δ^[L] = ŷ - y              (CE+softmax)
  Hidden delta:      δ^[l] = (W^[l+1]T δ^[l+1]) ⊙ σ'(z^[l])
  Weight gradient:   ∂L/∂W^[l] = δ^[l] (h^[l-1])T
  Bias gradient:     ∂L/∂b^[l] = δ^[l]

  INITIALISATION
  ─────────────────────────────────────────────────────────────────────
  Xavier (tanh):     σ²W = 2/(d_in + d_out)
  He (ReLU):         σ²W = 2/d_in
  Orthogonal:        W = Q from QR decomposition of random matrix

  OPTIMISERS
  ─────────────────────────────────────────────────────────────────────
  SGD:               θ ← θ - η∇L
  Momentum:          v ← γv - η∇L;  θ ← θ + v
  Adam:              m ← β₁m + (1-β₁)g; v ← β₂v + (1-β₂)g²
                     m̂ = m/(1-β₁ᵗ); v̂ = v/(1-β₂ᵗ)
                     θ ← θ - η m̂/(√v̂ + ε)
  AdamW:             θ ← θ - η(m̂/(√v̂+ε) + λθ)   (decoupled decay)

  NORMALISATION
  ─────────────────────────────────────────────────────────────────────
  BatchNorm:         x̂ = (x-μB)/√(σ²B+ε);  y = γx̂ + β
  LayerNorm:         x̂ = (x-μ)/√(σ²+ε);  y = γx̂ + β  (per sample)
  RMSNorm:           x̂ = x/RMS(x);  y = γx̂              (no mean)

  RESIDUAL CONNECTIONS
  ─────────────────────────────────────────────────────────────────────
  ResNet block:      h^[l+1] = h^[l] + F(h^[l]; θ^[l])
  Pre-norm transf:   x' = x + MHA(LN(x));  x'' = x' + FFN(LN(x'))
  Gradient:          ∂L/∂h^[0] has identity path — no vanishing

  REGULARISATION
  ─────────────────────────────────────────────────────────────────────
  L2/weight decay:   L_reg = L + λ/2||θ||²
  Dropout:           h' = m⊙h/(1-p), m_i ~ Bernoulli(1-p)
  Label smoothing:   ỹ_k = (1-ε)y_k + ε/K

  NEURAL TANGENT KERNEL
  ─────────────────────────────────────────────────────────────────────
  NTK:               Θ(x,x') = ∇θf(x)T ∇θf(x')
  NTK solution:      f(X*) = K(X*,X)(K(X,X)+εI)⁻¹y
  Convergence:       L(t) ≤ L(0)·exp(-2λ_min(Θ)·t)

════════════════════════════════════════════════════════════════════════
```

