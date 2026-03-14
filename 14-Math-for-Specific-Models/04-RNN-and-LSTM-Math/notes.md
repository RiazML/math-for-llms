[← Back to Math for Specific Models](../README.md) | [Next: Transformer Architecture →](../05-Transformer-Architecture/notes.md)

---

# RNNs and LSTMs: Mathematical Foundations

> _"The LSTM is one of the most important contributions in the history of neural networks — not because it's complicated, but because it's exactly complicated enough."_
> — Andrej Karpathy

## Overview

Recurrent Neural Networks (RNNs) and their gated variants — Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) — are the foundational architectures for processing sequential data. Unlike feedforward networks, which treat every input independently, recurrent architectures maintain a **hidden state** that evolves over time, allowing them to model temporal dependencies across arbitrarily long sequences.

This section develops the mathematics of recurrent computation from first principles. We begin with the simple recurrence equation and immediately confront its fundamental challenge: **the vanishing gradient problem**, which arises from multiplying Jacobians across hundreds of time steps. We then study how the LSTM resolves this through gated additive updates — the **constant error carousel** — and how the GRU achieves similar results with fewer parameters. The analysis extends to the dynamical systems view of RNNs, where spectral radius controls stability and the edge-of-chaos regime maximises information propagation.

In 2026, attention mechanisms and transformers dominate NLP, but recurrent architectures are experiencing a renaissance. **State space models** (S4, S5, Mamba) reinterpret recurrence as structured linear dynamical systems and achieve transformer-level performance on long-context tasks with linear (rather than quadratic) complexity. The mathematical ideas in this section — gating, cell state highways, selective updates — reappear directly in Mamba's selective scan and in linear attention variants. Understanding RNN mathematics is not merely historical: it is prerequisite knowledge for anyone working on efficient sequence models.

## Prerequisites

- Neural networks: forward pass, backpropagation, chain rule (Section 14-02)
- Matrix calculus: Jacobians, gradient chain rule, spectral radius (Section 03-01)
- Activation functions: sigmoid, tanh, softmax (Section 14-02)
- Eigenvalues and stability: spectral radius $\rho(A) = \max_i |\lambda_i(A)|$ (Section 03-01)
- Probability and cross-entropy loss (Section 13-01)

## Companion Notebooks

| Notebook | Description |
|---|---|
| [theory.ipynb](theory.ipynb) | Interactive demos: RNN forward pass, BPTT gradient flow, LSTM gate visualisation, spectral radius stability, attention alignment |
| [exercises.ipynb](exercises.ipynb) | 8 graded problems: vanilla RNN, BPTT, LSTM gates, GRU, spectral analysis, gradient clipping, Bahdanau attention, character-level LM |

## Learning Objectives

After completing this section, you will:

- Write the recurrence equation for a vanilla RNN and explain the role of each parameter matrix
- Derive BPTT by unrolling the computation graph and applying the chain rule
- Prove that the vanishing gradient problem is caused by repeated multiplication of Jacobians with spectral radius < 1
- State and implement all four LSTM gate equations and explain how the cell state highway prevents vanishing gradients
- Implement a GRU and compare its parameter count and performance with an LSTM
- Analyse RNN stability using the spectral radius of $W_h$ and connect it to the edge-of-chaos regime
- Apply gradient clipping and explain why it is necessary for stable RNN training
- Implement Bahdanau attention and explain how it solves the fixed-bottleneck problem
- Connect LSTM gating to modern state space models (Mamba)
- Compute perplexity for a character-level language model

---

## Table of Contents

- [1. Intuition](#1-intuition)
  - [1.1 Why Sequences Need Special Architecture](#11-why-sequences-need-special-architecture)
  - [1.2 The Memory Bottleneck](#12-the-memory-bottleneck)
  - [1.3 Historical Timeline](#13-historical-timeline)
  - [1.4 Computational Graph: Unrolled RNN](#14-computational-graph-unrolled-rnn)
- [2. Vanilla RNNs: Formal Definitions](#2-vanilla-rnns-formal-definitions)
  - [2.1 The Recurrence Equation](#21-the-recurrence-equation)
  - [2.2 Output Layer and Prediction](#22-output-layer-and-prediction)
  - [2.3 Parameter Count and Weight Sharing](#23-parameter-count-and-weight-sharing)
  - [2.4 Activation Functions in Context](#24-activation-functions-in-context)
- [3. Backpropagation Through Time (BPTT)](#3-backpropagation-through-time-bptt)
  - [3.1 The Loss Function](#31-the-loss-function)
  - [3.2 Gradient Flow Through Time](#32-gradient-flow-through-time)
  - [3.3 Vanishing Gradients: The Mathematics](#33-vanishing-gradients-the-mathematics)
  - [3.4 Exploding Gradients and Gradient Clipping](#34-exploding-gradients-and-gradient-clipping)
  - [3.5 Truncated BPTT](#35-truncated-bptt)
- [4. Long Short-Term Memory (LSTM)](#4-long-short-term-memory-lstm)
  - [4.1 The Core Idea: Gated Memory](#41-the-core-idea-gated-memory)
  - [4.2 The Four Gate Equations](#42-the-four-gate-equations)
  - [4.3 Cell State Highway: Why Gradients Flow](#43-cell-state-highway-why-gradients-flow)
  - [4.4 Hidden State and Output](#44-hidden-state-and-output)
  - [4.5 LSTM Parameter Count and Complexity](#45-lstm-parameter-count-and-complexity)
  - [4.6 BPTT for LSTM: Gradient Highway](#46-bptt-for-lstm-gradient-highway)
- [5. Gated Recurrent Units (GRU)](#5-gated-recurrent-units-gru)
  - [5.1 GRU Architecture](#51-gru-architecture)
  - [5.2 GRU vs LSTM: Mathematical Comparison](#52-gru-vs-lstm-mathematical-comparison)
  - [5.3 Minimal Gated Unit and Ablations](#53-minimal-gated-unit-and-ablations)
- [6. Stability Analysis and Dynamical Systems View](#6-stability-analysis-and-dynamical-systems-view)
  - [6.1 RNN as a Discrete Dynamical System](#61-rnn-as-a-discrete-dynamical-system)
  - [6.2 Spectral Radius and Stability](#62-spectral-radius-and-stability)
  - [6.3 Edge of Chaos](#63-edge-of-chaos)
  - [6.4 Orthogonal and Unitary RNNs](#64-orthogonal-and-unitary-rnns)
  - [6.5 Lyapunov Exponents and Information Propagation](#65-lyapunov-exponents-and-information-propagation)
- [7. Initialisation and Training Strategies](#7-initialisation-and-training-strategies)
  - [7.1 Weight Initialisation for RNNs](#71-weight-initialisation-for-rnns)
  - [7.2 Gradient Clipping: Theory and Practice](#72-gradient-clipping-theory-and-practice)
  - [7.3 Regularisation](#73-regularisation)
  - [7.4 Layer Normalisation in RNNs](#74-layer-normalisation-in-rnns)
- [8. Variants and Extensions](#8-variants-and-extensions)
  - [8.1 Bidirectional RNNs](#81-bidirectional-rnns)
  - [8.2 Deep (Stacked) RNNs](#82-deep-stacked-rnns)
  - [8.3 Clockwork RNNs and Multi-Timescale Models](#83-clockwork-rnns-and-multi-timescale-models)
  - [8.4 Attention-Augmented RNNs](#84-attention-augmented-rnns)
- [9. Applications in Modern AI](#9-applications-in-modern-ai)
  - [9.1 Language Modelling: Character and Word Level](#91-language-modelling-character-and-word-level)
  - [9.2 Sequence-to-Sequence and Encoder-Decoder](#92-sequence-to-sequence-and-encoder-decoder)
  - [9.3 Time-Series and Scientific ML](#93-time-series-and-scientific-ml)
  - [9.4 State Space Models and Mamba: The 2024 Connection](#94-state-space-models-and-mamba-the-2024-connection)
  - [9.5 Neural ODE Connection](#95-neural-ode-connection)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Exercises](#11-exercises)
- [12. Why This Matters for AI (2026 Perspective)](#12-why-this-matters-for-ai-2026-perspective)
- [13. Conceptual Bridge](#13-conceptual-bridge)

---

## 1. Intuition

### 1.1 Why Sequences Need Special Architecture

A feedforward neural network applies the same function $\mathbf{y} = f(\mathbf{x}; \theta)$ independently to every input. This is ideal for tasks where inputs are exchangeable — classifying an image, predicting a house price — but it fails completely for tasks where **order and history matter**. Consider predicting the next word in a sentence: the word "bank" has opposite meanings in "river bank" and "bank account", and the correct interpretation requires reading multiple preceding words. A feedforward network given only "bank" as input has no access to this context.

The core insight of recurrent networks is **parameter sharing across time**: use the same weight matrices at every step, but carry forward a **hidden state** $\mathbf{h}_t \in \mathbb{R}^d$ that accumulates information from all previous inputs. The hidden state is updated at each step:

$$\mathbf{h}_t = \sigma(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$$

This single equation defines the entire vanilla RNN. The weight matrices $W_h$ and $W_x$ are shared across all time steps — the network applies the same transformation at $t=1, t=2, \ldots, t=T$. This sharing achieves two goals: (1) it allows the network to generalise across sequence positions without needing to learn separate weights for each position, and (2) it keeps parameter count constant regardless of sequence length.

**For AI:** The idea of maintaining state across steps is fundamental to all sequence models. In LLMs, the key-value cache in transformers is a form of explicit state storage. In Mamba (2024), the selective state space mechanism is a direct successor to LSTM gating — it selectively writes to and reads from a hidden state vector based on the current input. Understanding the vanilla RNN is the entry point to all of these ideas.

### 1.2 The Memory Bottleneck

The hidden state $\mathbf{h}_t \in \mathbb{R}^d$ must summarise the entire history $(\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_t)$ in just $d$ real numbers. This creates an **information bottleneck**: as $T \to \infty$, the hidden state must discard information about early inputs to make room for recent ones. The amount of information retained decays geometrically with the spectral radius of $W_h$ — a deep result that we will make precise in Section 6.

This bottleneck is fundamental and cannot be avoided in fixed-dimensional recurrence. The only solutions are: (1) increase $d$ (expensive), (2) use gating to selectively preserve important information (LSTM/GRU), or (3) abandon recurrence entirely and use attention, which scales memory with sequence length (Transformer). In practice, option (2) dominates for streaming inference on long sequences, while option (3) dominates when the full sequence is available at training time.

**For AI:** The encoder-decoder bottleneck in seq2seq models (Section 9.2) is exactly this problem — compressing an entire source sentence into one vector. Bahdanau attention (Section 8.4) was introduced specifically to bypass this bottleneck by allowing the decoder to directly access all encoder hidden states.

### 1.3 Historical Timeline

```
RECURRENT NETWORKS: TIMELINE
════════════════════════════════════════════════════════════════════════

  1986  Rumelhart, Hinton, Williams — Backpropagation Through Time (BPTT)
  1990  Elman — Simple Recurrent Network (SRN / "Elman network")
  1991  Hochreiter — Diploma thesis: vanishing gradient problem identified
  1993  Robinson, Fallside — RTRL (Real-Time Recurrent Learning)
  1997  Hochreiter & Schmidhuber — LSTM (Long Short-Term Memory)
  1998  Williams & Zipser — Full BPTT formalised
  2000  Gers, Schmidhuber, Cummins — Forget gate added to LSTM
  2001  Graves — CTC (Connectionist Temporal Classification) + LSTM for ASR
  2013  Graves — Deep LSTM wins ICASSP speech competition
  2014  Cho et al. — GRU (Gated Recurrent Unit); Seq2Seq encoder-decoder
  2015  Bahdanau et al. — Neural attention for translation
  2015  Zaremba et al. — Dropout regularization for LSTMs
  2016  Karpathy — "Unreasonable Effectiveness of RNNs" (character-level LM)
  2017  Vaswani et al. — "Attention is All You Need" (Transformer)
  2018  ELMo — Bidirectional LSTM pretrained language model
  2019  BERT, GPT-2 — Transformers dominate; RNNs decline for NLP
  2022  Gu et al. — S4 (Structured State Space Sequence Model)
  2023  Smith et al. — S5; Fu et al. — H3
  2024  Gu & Dao — Mamba: selective state spaces with hardware-efficient scan
  2025  Mamba-2, RWKV-6 — Linear-complexity sequence models at scale

════════════════════════════════════════════════════════════════════════
```

The key inflection points: (1) 1991–1997: the vanishing gradient problem motivates the LSTM; (2) 2014–2015: sequence models mature with GRU and attention; (3) 2017: Transformer replaces RNNs for most NLP; (4) 2022–2024: SSMs revisit recurrence with hardware-efficient parallelism.

### 1.4 Computational Graph: Unrolled RNN

An RNN is most clearly understood by **unrolling** the recurrence into a directed acyclic graph (DAG). For a sequence of length $T$, we create $T$ copies of the same computation, connected by the hidden state:

```
UNROLLED RNN COMPUTATION GRAPH
════════════════════════════════════════════════════════════════════════

   x₁        x₂        x₃             xT
   │         │         │              │
   ▼         ▼         ▼              ▼
h₀─►[RNN]──►[RNN]──►[RNN]──► ··· ──►[RNN]
      │         │         │              │
      ▼         ▼         ▼              ▼
      y₁        y₂        y₃             yT

  Each [RNN] block computes:
    h_t = σ(W_h · h_{t-1} + W_x · x_t + b)
    y_t = W_o · h_t + b_o

  SAME W_h, W_x, b used at EVERY step (weight sharing)

════════════════════════════════════════════════════════════════════════
```

The unrolled graph makes the backward pass visible: gradients flow backward through time from the loss at step $T$ back to step $1$. The path from $\frac{\partial \mathcal{L}}{\partial W_h}$ passes through every intermediate hidden state, multiplying Jacobians $\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t}$ at each step. This repeated multiplication is the source of the vanishing gradient problem, analysed rigorously in Section 3.

---

## 2. Vanilla RNNs: Formal Definitions

### 2.1 The Recurrence Equation

**Definition (Vanilla RNN).** Given an input sequence $(\mathbf{x}_1, \ldots, \mathbf{x}_T)$ with $\mathbf{x}_t \in \mathbb{R}^p$, a vanilla RNN with hidden dimension $d$ computes hidden states $\mathbf{h}_t \in \mathbb{R}^d$ via:

$$\mathbf{h}_t = \sigma(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b}_h), \quad t = 1, 2, \ldots, T$$

where:
- $W_h \in \mathbb{R}^{d \times d}$ — **hidden-to-hidden weight matrix** (recurrent weights)
- $W_x \in \mathbb{R}^{d \times p}$ — **input-to-hidden weight matrix**
- $\mathbf{b}_h \in \mathbb{R}^d$ — **hidden bias**
- $\sigma: \mathbb{R} \to \mathbb{R}$ — activation function (tanh or ReLU), applied element-wise
- $\mathbf{h}_0 \in \mathbb{R}^d$ — **initial hidden state**, typically $\mathbf{h}_0 = \mathbf{0}$

The combined pre-activation can be written compactly. Concatenate $[\mathbf{h}_{t-1}; \mathbf{x}_t] \in \mathbb{R}^{d+p}$ and let $W = [W_h \mid W_x] \in \mathbb{R}^{d \times (d+p)}$:

$$\mathbf{h}_t = \sigma\!\left(W \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b}_h\right)$$

This concatenated form reveals that the RNN is simply an MLP applied to the concatenation of the previous hidden state and the current input — the recurrence comes entirely from feeding $\mathbf{h}_{t-1}$ back in.

**Non-examples that fail to be valid RNNs:**
- A network that uses a different $W_h^{(t)}$ at each step is not an RNN (it is an unrolled MLP); it cannot handle variable-length sequences and does not share parameters.
- Setting $W_h = 0$ reduces the RNN to an MLP with no memory; each output depends only on the current input.
- Using $\sigma = $ identity produces a linear dynamical system; useful for analysis but unstable in practice without spectral control.

### 2.2 Output Layer and Prediction

Given hidden states $\mathbf{h}_t$, the output is:

$$\hat{\mathbf{y}}_t = \text{softmax}(W_o \mathbf{h}_t + \mathbf{b}_o)$$

where $W_o \in \mathbb{R}^{k \times d}$ and $k$ is the vocabulary or class size. Three standard sequence architectures exist:

| Architecture | Input | Output | Use Case |
|---|---|---|---|
| **Many-to-one** | $T$ inputs | 1 output (at $t=T$) | Sentiment classification |
| **One-to-many** | 1 input | $T$ outputs | Image captioning |
| **Many-to-many (sync)** | $T$ inputs | $T$ outputs | Language modelling, POS tagging |
| **Many-to-many (async)** | $T_x$ inputs | $T_y$ outputs | Machine translation (encoder-decoder) |

For language modelling (predicting the next token), the loss at each step is cross-entropy:

$$\ell_t = -\log \hat{y}_{t, c_t}$$

where $c_t$ is the correct token at step $t$. The total loss is $\mathcal{L} = \sum_{t=1}^T \ell_t$.

### 2.3 Parameter Count and Weight Sharing

The total number of trainable parameters in a vanilla RNN is:

$$|\theta| = d \cdot d + d \cdot p + d + k \cdot d + k = d^2 + dp + d + kd + k$$

Breaking this down:
- $W_h$: $d^2$ parameters — dominant term for large hidden dimension
- $W_x$: $dp$ parameters
- $\mathbf{b}_h$: $d$ parameters
- $W_o$: $kd$ parameters (often tied with the embedding matrix: $W_o = W_{\text{emb}}^\top$)
- $\mathbf{b}_o$: $k$ parameters

Crucially, this count is **independent of sequence length $T$**. The same $d^2 + dp$ recurrent parameters are reused at every step. This contrasts sharply with a Transformer, where the attention computation scales with $T^2$ in memory (though not parameters). For $T$ very large and $d$ moderate, RNNs are highly parameter-efficient.

**For AI:** In 2024–2025, this parameter efficiency is exploited by Mamba and RWKV for long-context inference. A Mamba model at inference time runs in $O(d^2)$ per token regardless of context length, while a Transformer requires $O(T \cdot d)$ memory for the KV cache.

### 2.4 Activation Functions in Context

The choice of $\sigma$ in recurrence is critical and non-obvious:

**Tanh** ($\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$):
- Range $(-1, 1)$: bounded outputs prevent hidden state explosion
- Derivative: $\sigma'(z) = 1 - \tanh^2(z) \in (0, 1]$, maximised at $z=0$
- Symmetric around zero: helps with gradient flow compared to sigmoid
- Standard choice for vanilla RNNs

**Sigmoid** ($\sigma(z) = \frac{1}{1+e^{-z}}$):
- Range $(0, 1)$: natural for gate activations in LSTM/GRU
- Not used as main recurrent activation: squashes gradients too aggressively

**ReLU** ($\sigma(z) = \max(0, z)$):
- Unbounded: hidden state can grow without bound if $\rho(W_h) > 1$
- Derivative is 0 or 1: avoids vanishing gradients but enables exploding
- Requires careful initialisation (identity init for $W_h$) and gradient clipping
- Used in IRNN (Le et al., 2015): $W_h = I$, ReLU activation

**For AI:** LSTM and GRU gates use sigmoid activations (range $(0,1)$ for gating); the candidate state uses tanh (range $(-1,1)$ for bounded updates). This combination is not arbitrary — it is the mathematical structure that enables the cell state highway (Section 4.3).


---

## 3. Backpropagation Through Time (BPTT)

### 3.1 The Loss Function

For a sequence-labelling task (e.g., language modelling), the loss is the sum of per-step losses:

$$\mathcal{L} = \sum_{t=1}^{T} \ell_t(\hat{\mathbf{y}}_t, \mathbf{y}_t)$$

where $\ell_t$ is typically cross-entropy. For regression tasks, $\ell_t = \|\hat{\mathbf{y}}_t - \mathbf{y}_t\|^2$. The gradient of the total loss with respect to any parameter $\theta$ decomposes as:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \sum_{t=1}^{T} \frac{\partial \ell_t}{\partial \theta}$$

Each term $\frac{\partial \ell_t}{\partial \theta}$ contributes a gradient, but all share the same $W_h$, $W_x$ — the total gradient is the **sum** of contributions from every time step.

### 3.2 Gradient Flow Through Time

To compute $\frac{\partial \ell_T}{\partial W_h}$ (gradient of the final loss w.r.t. recurrent weights), we apply the chain rule through the hidden states:

$$\frac{\partial \ell_T}{\partial W_h} = \frac{\partial \ell_T}{\partial \hat{\mathbf{y}}_T} \cdot \frac{\partial \hat{\mathbf{y}}_T}{\partial \mathbf{h}_T} \cdot \frac{\partial \mathbf{h}_T}{\partial W_h}$$

The last factor, $\frac{\partial \mathbf{h}_T}{\partial W_h}$, must account for the fact that $\mathbf{h}_T$ depends on $W_h$ both directly and through all previous hidden states. By the total derivative:

$$\frac{\partial \mathbf{h}_T}{\partial W_h} = \sum_{k=1}^{T} \left(\prod_{j=k}^{T-1} \frac{\partial \mathbf{h}_{j+1}}{\partial \mathbf{h}_j}\right) \frac{\partial \mathbf{h}_k}{\partial W_h}\bigg|_{\text{direct}}$$

where the **temporal Jacobian** at each step is:

$$\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \text{diag}\!\left(\sigma'(\mathbf{z}_{t+1})\right) W_h \in \mathbb{R}^{d \times d}$$

and $\mathbf{z}_{t+1} = W_h \mathbf{h}_t + W_x \mathbf{x}_{t+1} + \mathbf{b}_h$ is the pre-activation. The gradient of the loss at step $T$ with respect to hidden state at step $k$ is:

$$\frac{\partial \ell_T}{\partial \mathbf{h}_k} = \frac{\partial \ell_T}{\partial \mathbf{h}_T} \prod_{j=k}^{T-1} \frac{\partial \mathbf{h}_{j+1}}{\partial \mathbf{h}_j} = \frac{\partial \ell_T}{\partial \mathbf{h}_T} \prod_{j=k}^{T-1} D_j W_h$$

where $D_j = \text{diag}(\sigma'(\mathbf{z}_{j+1}))$. This is a product of $(T-k)$ matrices, and its norm controls the gradient magnitude.

### 3.3 Vanishing Gradients: The Mathematics

**Theorem (Vanishing Gradient).** Let $J_t = D_t W_h$ be the temporal Jacobian at step $t$. The gradient $\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k}$ involves the product $\prod_{j=k}^{T-1} J_j$. By submultiplicativity of the spectral norm:

$$\left\|\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k}\right\|_2 \leq \prod_{j=k}^{T-1} \|J_j\|_2 \leq \left(\|D\|_2 \cdot \|W_h\|_2\right)^{T-k}$$

For tanh activation, $\|D_j\|_2 = \max_i |\sigma'(z_i)| \leq 1$ (with equality only at $z_i = 0$). Therefore:

$$\left\|\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k}\right\|_2 \leq \|W_h\|_2^{T-k} = \sigma_{\max}(W_h)^{T-k}$$

- If $\sigma_{\max}(W_h) < 1$: gradients vanish exponentially at rate $(T-k)$. Gradients from early steps are negligible, and the network cannot learn long-range dependencies.
- If $\sigma_{\max}(W_h) > 1$: the upper bound exceeds 1, but gradients may still explode if eigenvalues are large.

```
VANISHING vs EXPLODING GRADIENT REGIMES
════════════════════════════════════════════════════════════════════════

  Gradient norm: ‖∂h_T/∂h_k‖ ~ ρ(W_h)^(T-k)

  ρ(W_h) < 1                 ρ(W_h) ≈ 1              ρ(W_h) > 1
  ──────────────────         ────────────────         ──────────────
  Vanishing gradient         Stable propagation       Exploding grad
  Cannot learn long          Ideal regime             Numerical NaN
  dependencies               (hard to achieve)        gradient clip ✓

  Gradient after T steps:
  ρ = 0.9, T=50:  0.9^50 ≈ 0.005  ← effectively zero
  ρ = 1.0, T=50:  1.0^50 = 1.000  ← perfect
  ρ = 1.1, T=50:  1.1^50 ≈ 117    ← exploding

════════════════════════════════════════════════════════════════════════
```

**Historical note:** Hochreiter identified this problem in his 1991 diploma thesis (German). The result was largely overlooked until his 1997 LSTM paper provided the solution. The mathematical analysis above is from Bengio et al. (1994) "Learning Long-Term Dependencies with Gradient Descent is Difficult".

**For AI:** This analysis directly motivates residual connections in ResNets and transformers — adding an identity shortcut ensures that gradients can flow directly through depth without passing through activation Jacobians. The LSTM's cell state is exactly such a shortcut through time.

### 3.4 Exploding Gradients and Gradient Clipping

When $\|W_h\|_2 > 1$, gradients can grow exponentially, causing parameter updates to overshoot and numerical instability (NaN). The standard remedy is **gradient norm clipping**:

**Algorithm (Gradient Norm Clipping).**
1. Compute the gradient $\mathbf{g} = \nabla_\theta \mathcal{L}$ (full parameter vector gradient)
2. Compute the global gradient norm: $\|\mathbf{g}\| = \sqrt{\sum_i g_i^2}$
3. If $\|\mathbf{g}\| > \tau$ (threshold), rescale: $\mathbf{g} \leftarrow \frac{\tau}{\|\mathbf{g}\|} \mathbf{g}$
4. Apply update: $\theta \leftarrow \theta - \eta \mathbf{g}$

This preserves the **direction** of the gradient while capping its magnitude at $\tau$. Typical values: $\tau \in [1, 5]$ for LSTM training. Clipping is distinct from weight decay (which regularises the parameters, not the gradient).

**Why clipping works but does not solve the root problem:** Clipping prevents numerical explosion but does not fix the underlying issue — gradients from distant time steps are either zero (vanishing) or have been artificially capped (clipping masks the explosion). The only true solution is architectural: LSTM or GRU.

**For AI:** Gradient clipping is standard in modern LLM training. GPT-3 uses global gradient norm clipping with $\tau = 1.0$. It is one of the few training tricks that is universally applied regardless of architecture.

### 3.5 Truncated BPTT

For very long sequences ($T \gg 1$), full BPTT is computationally prohibitive: it requires storing all $T$ hidden states and backpropagating through all $T$ steps. **Truncated BPTT** approximates the gradient by limiting the lookback window:

**Algorithm (Truncated BPTT with window $k$).**
1. Process sequence in chunks of length $k$
2. For each chunk $[t, t+k]$: run forward pass, compute gradients only through the last $k$ steps
3. Carry forward the hidden state between chunks (no gradient through the boundary)

The gradient approximation error is:

$$\left\|\frac{\partial \mathcal{L}}{\partial W_h}\right\|_{\text{true}} - \left\|\frac{\partial \mathcal{L}}{\partial W_h}\right\|_{\text{truncated}} \leq C \cdot \rho(W_h)^k$$

For $\rho(W_h) < 1$, the approximation error decays exponentially with $k$. In practice, $k \in [20, 100]$ is sufficient.

**Trade-off:**
- Small $k$: fast training, but the network cannot learn dependencies longer than $k$ steps
- Large $k$: better gradient estimates, higher memory cost ($O(k \cdot d)$ for hidden states)
- Truncated BPTT is the standard training procedure for language models on long documents


---

## 4. Long Short-Term Memory (LSTM)

### 4.1 The Core Idea: Gated Memory

The LSTM (Hochreiter & Schmidhuber, 1997) solves the vanishing gradient problem by introducing two separate state vectors:

- **Cell state** $\mathbf{c}_t \in \mathbb{R}^d$: the long-term memory; updated via **addition**, not multiplication
- **Hidden state** $\mathbf{h}_t \in \mathbb{R}^d$: the short-term working representation; exposed as output

The key architectural insight is that the cell state is updated **additively**:

$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

where $\odot$ denotes the Hadamard (element-wise) product. This additive update creates a **gradient highway**: the gradient of $\mathbf{c}_t$ with respect to $\mathbf{c}_{t-1}$ is simply $\text{diag}(\mathbf{f}_t)$, a diagonal matrix with values in $(0,1)$. Crucially, the error can flow through the cell state without passing through the activation Jacobian $D_t W_h$ that caused vanishing in vanilla RNNs. Hochreiter called this the **Constant Error Carousel (CEC)**.

**Analogy to ResNets:** The cell state highway is the temporal analogue of the residual connection in ResNets. Just as $\mathbf{x}_{l+1} = F(\mathbf{x}_l) + \mathbf{x}_l$ allows gradients to bypass nonlinear layers in depth, the LSTM allows gradients to bypass nonlinear steps in time.

### 4.2 The Four Gate Equations

The LSTM at each step $t$ computes four intermediate vectors from the concatenated input $[\mathbf{h}_{t-1}; \mathbf{x}_t] \in \mathbb{R}^{d+p}$:

**Forget gate** — decides what to erase from cell state:
$$\mathbf{f}_t = \sigma(W_f [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f) \in (0,1)^d$$

**Input gate** — decides what new information to write:
$$\mathbf{i}_t = \sigma(W_i [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_i) \in (0,1)^d$$

**Cell update (candidate cell state)** — the new content to potentially add:
$$\tilde{\mathbf{c}}_t = \tanh(W_c [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_c) \in (-1,1)^d$$

**Output gate** — decides what to expose from cell state:
$$\mathbf{o}_t = \sigma(W_o [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_o) \in (0,1)^d$$

**Cell state update:**
$$\mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t$$

**Hidden state output:**
$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

```
LSTM CELL DIAGRAM
════════════════════════════════════════════════════════════════════════

  c_{t-1} ──────────────────────────────────────────────► c_t
               │                        │
               ▼                        ▼
          [forget: f_t ⊙]         [input: i_t ⊙ c̃_t]
               │                        │
               └──────────┬─────────────┘
                          │  (addition)
                          ▼
  h_{t-1} ──►[gates: f,i,c̃,o]         ┌─► tanh ──► o_t ⊙ ──► h_t
  x_t ──────►(4 linear + act)          │
                          │            │
                          └────────────┘

════════════════════════════════════════════════════════════════════════
```

In practice, all four gates are computed in a single batched matrix multiply. Define the stacked weight matrix $W \in \mathbb{R}^{4d \times (d+p)}$ and stacked bias $\mathbf{b} \in \mathbb{R}^{4d}$:

$$\begin{bmatrix} \mathbf{f}_t \\ \mathbf{i}_t \\ \tilde{\mathbf{c}}_t \\ \mathbf{o}_t \end{bmatrix} = \begin{bmatrix} \sigma \\ \sigma \\ \tanh \\ \sigma \end{bmatrix} \left( W \begin{bmatrix} \mathbf{h}_{t-1} \\ \mathbf{x}_t \end{bmatrix} + \mathbf{b} \right)$$

### 4.3 Cell State Highway: Why Gradients Flow

The gradient of the loss $\mathcal{L}$ with respect to the cell state at step $k$ can reach step $t$ through the cell state pathway:

$$\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_k} = \prod_{j=k}^{T-1} \frac{\partial \mathbf{c}_{j+1}}{\partial \mathbf{c}_j} = \prod_{j=k}^{T-1} \text{diag}(\mathbf{f}_{j+1})$$

This is a product of **diagonal matrices**, which is:
1. **Diagonal** (no cross-term interference between dimensions)
2. **Elementwise controlled** — each dimension $i$ has its own independent gradient decay $\prod_{j=k}^{T-1} f_{j+1,i}$
3. **Learnable** — the network can set $f_{j,i} \approx 1$ to preserve a dimension and $f_{j,i} \approx 0$ to reset it

When $\mathbf{f}_t \approx \mathbf{1}$ (forget gate fully open), we get $\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_k} \approx I$, a near-identity gradient — no vanishing. This is the CEC in action. Contrast with the vanilla RNN where the Jacobian product $\prod D_j W_h$ involves the full matrix $W_h$ at every step.

**Remark:** The forget gate was not in the original 1997 LSTM — it was added by Gers et al. (2000) after empirical evidence showed that it dramatically improved performance on tasks requiring the network to "reset" its memory at specific events (e.g., sentence boundaries in language modelling).

### 4.4 Hidden State and Output

The hidden state is:

$$\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$$

The output gate $\mathbf{o}_t$ acts as a **filter** on the cell state: it selects which dimensions of the stored memory to expose at the current step. The $\tanh(\mathbf{c}_t)$ squashes the cell state back to $(-1,1)$, ensuring that $\mathbf{h}_t$ is bounded even if $\mathbf{c}_t$ has grown large.

**Intuition for the two states:**
- **Cell state $\mathbf{c}_t$**: like long-term memory in a human — holds context accumulated over many steps; not directly observable
- **Hidden state $\mathbf{h}_t$**: like working memory — what is currently "in mind" and relevant to the immediate output; passed to the next step and to the output layer

### 4.5 LSTM Parameter Count and Complexity

The LSTM has four weight matrices, each of size $d \times (d+p)$, plus four bias vectors:

$$|\theta_{\text{LSTM}}| = 4 \cdot d(d + p) + 4d = 4d(d + p + 1)$$

For $d = 512$, $p = 512$ (typical for medium-sized LM): $|\theta| = 4 \times 512 \times 1025 \approx 2.1\text{M}$ parameters per layer.

**Forward pass FLOPs per step:** The dominant cost is the $4d \times (d+p)$ matrix multiply: $\approx 8d(d+p)$ multiply-adds.

**Memory per step:** Store $\mathbf{h}_t, \mathbf{c}_t \in \mathbb{R}^d$, all gates $\in \mathbb{R}^{4d}$ for BPTT: $O(T \cdot d)$ for a sequence of length $T$.

### 4.6 BPTT for LSTM: Gradient Highway

The gradient of the loss with respect to $\mathbf{c}_k$ receives contributions from two paths: (1) the direct path through the cell state highway, and (2) the indirect path through hidden states $\mathbf{h}_{k+1}, \ldots, \mathbf{h}_T$. The key result is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{c}_k} = \frac{\partial \mathcal{L}}{\partial \mathbf{c}_T} \prod_{j=k}^{T-1} \text{diag}(\mathbf{f}_{j+1}) + \sum_{t=k+1}^{T} (\text{path through } \mathbf{h}_t)$$

The first term — the cell state gradient — can be large even for small $k$ (long-range) as long as forget gates are open. The second term involves the output gate and is analogous to the vanilla RNN gradient, but since it is additive, it supplements rather than replaces the cell state path.

**Empirical observation:** In practice, LSTM gradients through the cell state are 10-100× larger than through the hidden state path for sequences longer than 20-30 steps. This is directly measurable via gradient norm tracking per pathway.


---

## 5. Gated Recurrent Units (GRU)

### 5.1 GRU Architecture

The GRU (Cho et al., 2014) simplifies the LSTM by merging the cell state and hidden state into a single hidden state $\mathbf{h}_t$, using only two gates:

**Update gate** — interpolates between previous and new state:
$$\mathbf{z}_t = \sigma(W_z [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_z) \in (0,1)^d$$

**Reset gate** — controls how much of the previous state is exposed to the candidate:
$$\mathbf{r}_t = \sigma(W_r [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_r) \in (0,1)^d$$

**Candidate hidden state:**
$$\tilde{\mathbf{h}}_t = \tanh(W_h [\mathbf{r}_t \odot \mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_h)$$

**Hidden state update** (linear interpolation):
$$\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t$$

The update gate $\mathbf{z}_t$ plays the combined role of LSTM's forget and input gates: when $\mathbf{z}_t \approx 0$, the hidden state is copied unchanged ($\mathbf{h}_t \approx \mathbf{h}_{t-1}$); when $\mathbf{z}_t \approx 1$, the hidden state is replaced by the candidate.

```
GRU vs LSTM: ARCHITECTURE COMPARISON
════════════════════════════════════════════════════════════════════════

  LSTM                                  GRU
  ────────────────────────              ────────────────────────
  Two states: h_t, c_t                  One state: h_t
  4 gates: f, i, c̃, o                  2 gates: z, r
  4d(d+p+1) parameters                  3d(d+p+1) parameters
  Separate memory/working               Combined
  Cell state highway (additive)         Update gate (interpolation)
  Output gate filters memory            No output gate

════════════════════════════════════════════════════════════════════════
```

### 5.2 GRU vs LSTM: Mathematical Comparison

**Parameter count:**
$$|\theta_{\text{GRU}}| = 3d(d+p+1), \qquad |\theta_{\text{LSTM}}| = 4d(d+p+1)$$

GRU uses 25% fewer parameters. For $d=512$, $p=512$: GRU ≈ 1.57M vs LSTM ≈ 2.1M per layer.

**Gradient flow:** In GRU, the gradient through the update gate path is:

$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_k} = \prod_{j=k}^{T-1} \text{diag}(1 - \mathbf{z}_{j+1}) + \text{cross terms}$$

When $\mathbf{z}_t \approx 0$, this is approximately identity — analogous to the LSTM's open forget gate. The gradient highway in GRU relies on the update gate rather than a separate cell state.

**Empirical comparison:**
- On short sequences ($T < 50$): GRU typically matches LSTM performance
- On long sequences ($T > 200$): LSTM's explicit cell state separation slightly outperforms GRU
- On small datasets: GRU's lower parameter count reduces overfitting
- In practice (2019–2024 literature): the difference is usually negligible; use GRU for faster experiments

### 5.3 Minimal Gated Unit and Ablations

Researchers have systematically ablated LSTM gates to understand which components are essential:

| Variant | Description | Params | Notes |
|---|---|---|---|
| LSTM (full) | 4 gates + cell state | $4d(d+p+1)$ | Baseline |
| LSTM - forget gate | $f_t = 1$ fixed | $3d(d+p+1)$ | Much worse on long sequences |
| LSTM - input gate | $i_t = 1 - f_t$ | $3d(d+p+1)$ | Minimal degradation |
| LSTM - output gate | $o_t = 1$ fixed | $3d(d+p+1)$ | Modest degradation |
| GRU | 2 gates, 1 state | $3d(d+p+1)$ | Competitive |
| MGU (Minimal Gated) | 1 gate | $2d(d+p+1)$ | Often competitive |

**Key finding** (Jozefowicz et al., 2015; Greff et al., 2017): The **forget gate** is the single most important gate. Removing it causes catastrophic performance degradation on long-range tasks. The output gate matters less. This explains why the GRU (which uses the update gate as a combined forget+input gate) works well: it preserves the essential forget-gate functionality.

---

## 6. Stability Analysis and Dynamical Systems View

### 6.1 RNN as a Discrete Dynamical System

A vanilla RNN (with no input, or with $\mathbf{x}_t$ viewed as a driving signal) defines a **discrete-time dynamical system**:

$$\mathbf{h}_t = F(\mathbf{h}_{t-1}) = \sigma(W_h \mathbf{h}_{t-1} + \mathbf{b})$$

The long-run behaviour of this system determines what the RNN "remembers":
- **Fixed points**: $\mathbf{h}^* = F(\mathbf{h}^*)$ — stable equilibria that the hidden state converges to
- **Limit cycles**: periodic orbits — the hidden state oscillates with period $p$
- **Strange attractors / chaos**: sensitive dependence on $\mathbf{h}_0$; exponential divergence of trajectories

For sequence modelling, we want the RNN to reside in a regime that is sensitive enough to distinguish inputs ($\rho \geq 1$) but stable enough to be trainable ($\rho \leq 1$) — the **edge of chaos**.

### 6.2 Spectral Radius and Stability

**Definition.** The spectral radius of $W_h$ is $\rho(W_h) = \max_i |\lambda_i(W_h)|$, the magnitude of the largest eigenvalue.

**Theorem (Asymptotic stability).** For the linear RNN $\mathbf{h}_t = W_h \mathbf{h}_{t-1}$ (no input, no activation), $\mathbf{h}_t \to \mathbf{0}$ as $t \to \infty$ if and only if $\rho(W_h) < 1$.

**Proof sketch:** Write $\mathbf{h}_t = W_h^t \mathbf{h}_0$. By the Jordan decomposition, $\|W_h^t\| \to 0$ iff all eigenvalues have magnitude $< 1$, i.e., $\rho(W_h) < 1$. $\square$

For the nonlinear RNN with tanh: the fixed point $\mathbf{h}^* = \mathbf{0}$ is locally asymptotically stable if $\rho(W_h) < 1/\max \sigma'(0) = 1$ (since $\max \sigma'(z) = 1$ for tanh at $z=0$).

**Implication for memory:** The spectral radius controls the timescale of memory. An RNN with $\rho(W_h) = 0.9$ has an effective memory horizon of $\approx 1/(1-0.9) = 10$ steps — information from 10+ steps ago contributes negligibly to the current hidden state.

### 6.3 Edge of Chaos

The "edge of chaos" at $\rho(W_h) \approx 1$ is the computational sweet spot for RNNs:

- $\rho < 1$: **ordered regime** — all perturbations decay; limited memory; easy to train
- $\rho = 1$: **critical regime** — perturbations neither grow nor decay; maximal memory
- $\rho > 1$: **chaotic regime** — perturbations grow; rich dynamics but gradient explosion; hard to train

This observation motivated **echo state networks** (Jaeger, 2001) and **reservoir computing**: fix $W_h$ with $\rho \approx 1$, and train only $W_o$. The reservoir provides rich dynamics without the training difficulty of BPTT through a chaotic system.

**For AI:** Understanding the edge of chaos motivates **orthogonal initialisation** of $W_h$ — initialising with $W_h \in O(d)$ (orthogonal matrix) ensures $\rho(W_h) = 1$ exactly. This is the initialisation scheme for uRNN and expRNN.

### 6.4 Orthogonal and Unitary RNNs

**Motivation:** If $\rho(W_h) = 1$ exactly (and $W_h$ is normal), gradients do not vanish or explode through the recurrent path. The spectral norm $\|W_h\|_2 = 1$ implies $\|W_h^T\|_2 = 1$ for all $T$ — perfect gradient magnitude preservation.

**Orthogonal RNN (oRNN):** Constrain $W_h \in O(d) = \{W : W^\top W = I\}$ during training. Since $O(d)$ is a Riemannian manifold, gradient steps on it use the **Cayley transform**:

$$W_h^{(t+1)} = \left(I - \frac{\eta}{2} A\right)\left(I + \frac{\eta}{2} A\right)^{-1} W_h^{(t)}$$

where $A = G W_h^\top - W_h G^\top$ is the skew-symmetric part of the gradient $G$.

**expRNN** (Lezcano-Casado & Martínez-Rubio, 2019): Parametrise $W_h = \exp(K)$ where $K$ is a skew-symmetric matrix ($K^\top = -K$). Since $\exp(K)$ is always orthogonal, unconstrained optimisation of $K$ yields valid orthogonal $W_h$.

**Unitary RNN (uRNN):** Generalise to $W_h \in U(d)$ (unitary matrices over $\mathbb{C}$). Allows full $O(d^2)$ parametrisation with efficient structured implementations.

### 6.5 Lyapunov Exponents and Information Propagation

The **maximal Lyapunov exponent** $\lambda_{\max}$ measures the average exponential rate of divergence of nearby trajectories:

$$\lambda_{\max} = \lim_{T \to \infty} \frac{1}{T} \log \frac{\|\delta \mathbf{h}_T\|}{\|\delta \mathbf{h}_0\|}$$

- $\lambda_{\max} < 0$: ordered regime (perturbations shrink)
- $\lambda_{\max} = 0$: critical regime (perturbations stay constant)
- $\lambda_{\max} > 0$: chaotic regime (perturbations grow)

For trained RNNs, the Lyapunov spectrum directly measures the **information propagation capacity** — how reliably a small change in an early input affects later hidden states. Empirically, well-trained LSTMs on long-range tasks have $\lambda_{\max} \approx 0$, consistent with the edge-of-chaos hypothesis.

**For AI:** Lyapunov analysis was applied to transformers (Dong et al., 2021) to explain rank collapse in attention — when attention maps become rank-1, the token representations collapse to the same vector, and $\lambda_{\max} \to -\infty$. This motivates layer normalisation and residual connections as stability mechanisms.


---

## 7. Initialisation and Training Strategies

### 7.1 Weight Initialisation for RNNs

Initialisation profoundly affects whether an RNN trains at all. Poor initialisation places the network in the chaotic or dead regime from step one.

**$W_x$ (input-to-hidden):** Use Xavier/Glorot initialisation:
$$W_x \sim \mathcal{U}\!\left(-\sqrt{\frac{6}{d+p}},\; \sqrt{\frac{6}{d+p}}\right)$$
This targets unit variance for the pre-activation assuming inputs have unit variance.

**$W_h$ (hidden-to-hidden):** The critical matrix. Standard options:
- **Gaussian, rescaled:** $W_h \sim \mathcal{N}(0, 1/d)$, giving $\rho(W_h) \approx 1$ in expectation by the circular law
- **Orthogonal initialisation:** Draw $W_h$ from the Haar measure on $O(d)$ via QR decomposition of a random Gaussian matrix; guarantees $\rho(W_h) = 1$ exactly
- **Identity (IRNN):** $W_h = I$; used with ReLU activation (Le et al., 2015)

**Forget gate bias ($\mathbf{b}_f$):** Initialise to $+1$ (or $+2$ for long dependencies). This ensures $\mathbf{f}_t \approx \sigma(1) \approx 0.73$ at the start of training, so the cell state is mostly preserved. Initialising to 0 means the forget gate starts at 0.5 — the LSTM is already discarding half of its memory before it has learned anything. This is one of the most impactful single-value hyperparameters in LSTM training.

**$\mathbf{b}_i$, $\mathbf{b}_c$, $\mathbf{b}_o$:** Initialise to 0. This sets input and output gates to $\approx 0.5$ at start.

### 7.2 Gradient Clipping: Theory and Practice

Gradient clipping (introduced in Section 3.4) is the universal remedy for exploding gradients. The algorithm:

```
GRADIENT CLIPPING ALGORITHM
════════════════════════════════════════════════════════════════════════

  Input: gradient g, threshold τ
  1. Compute L2 norm: ‖g‖ = sqrt(Σ gᵢ²)
  2. If ‖g‖ > τ:
       g ← (τ / ‖g‖) · g    # rescale direction preserved
  3. Update: θ ← θ - η · g

  Key properties:
  - Direction of g is unchanged
  - ‖g‖ after clipping ≤ τ
  - If ‖g‖ ≤ τ: no change (clipping is inactive)

════════════════════════════════════════════════════════════════════════
```

**Choosing $\tau$:** Inspect the gradient norm histogram during early training. Set $\tau$ at approximately the 95th percentile of the norm distribution when training is stable. Typical values:
- Language modelling LSTMs: $\tau = 5$
- Character-level models: $\tau = 1$–$3$
- Modern LLM training: $\tau = 1.0$ (GPT-3, GPT-4 training configs)

**Gradient norm tracking as a diagnostic:** If the gradient norm consistently exceeds $\tau$ by large factors, this indicates deeper training instability (bad learning rate, unstable architecture). If it consistently stays well below $\tau$, the threshold is unnecessarily conservative.

### 7.3 Regularisation

**Standard dropout** (Srivastava et al., 2014) applied naively to RNNs creates problems: if applied to hidden-to-hidden connections, a different mask at each step disrupts the temporal structure of the hidden state and damages long-term memory.

**Zaremba et al. (2015)** showed that dropout should be applied only to **non-recurrent connections** (input-to-hidden and hidden-to-output), not to $W_h$:

$$\mathbf{h}_t = \sigma(W_h \mathbf{h}_{t-1} + W_x \text{drop}(\mathbf{x}_t) + \mathbf{b})$$

**Variational dropout (Gal & Ghahramani, 2016):** Use the same dropout mask for all time steps within a sequence. This masks entire input/output channels consistently and preserves temporal structure. Mathematically, sample mask $\mathbf{m} \sim \text{Bernoulli}(1-p)^d$ once per sequence, then apply the same $\mathbf{m}$ at every step.

**Zoneout (Krueger et al., 2016):** Instead of zeroing hidden state units, randomly **preserve** previous hidden state values:
$$\mathbf{h}_t = \mathbf{m} \odot \mathbf{h}_{t-1} + (1 - \mathbf{m}) \odot F(\mathbf{h}_{t-1}, \mathbf{x}_t)$$
where $\mathbf{m} \sim \text{Bernoulli}(p)$. Zoneout can be interpreted as stochastic depth through time.

**Weight decay ($L_2$ regularisation):** Apply to $W_x$ and $W_o$ but use caution with $W_h$ — large weight decay on the recurrent matrix can push $\rho(W_h)$ below 1 and damage long-range memory.

### 7.4 Layer Normalisation in RNNs

**Layer Normalisation** (Ba et al., 2016), applied to the pre-activation vector at each step, stabilises the hidden state distribution:

$$\text{LN}(\mathbf{z}) = \frac{\mathbf{z} - \mu_z}{\sigma_z + \epsilon} \odot \boldsymbol{\gamma} + \boldsymbol{\beta}$$

where $\mu_z = \frac{1}{d}\sum_i z_i$ and $\sigma_z^2 = \frac{1}{d}\sum_i (z_i - \mu_z)^2$ are computed over the $d$ hidden units at the current step.

**Layer-Norm LSTM:** Apply LN before each gate activation:
$$\mathbf{f}_t = \sigma(\text{LN}(W_f [\mathbf{h}_{t-1}; \mathbf{x}_t] + \mathbf{b}_f))$$
(and similarly for $\mathbf{i}_t$, $\tilde{\mathbf{c}}_t$, $\mathbf{o}_t$).

Benefits: (1) stabilises the gradient scale across long sequences; (2) reduces sensitivity to initialisation; (3) allows larger learning rates; (4) acts as an implicit regulariser.

**For AI:** Layer norm is now standard in all transformers (applied before attention and FFN sublayers). Understanding its role in RNNs provides intuition for why it is effective in transformers as well.

---

## 8. Variants and Extensions

### 8.1 Bidirectional RNNs

A **bidirectional RNN** processes the sequence in both forward and backward directions:

$$\overrightarrow{\mathbf{h}}_t = \sigma(W_h^{(\to)} \overrightarrow{\mathbf{h}}_{t-1} + W_x^{(\to)} \mathbf{x}_t + \mathbf{b}^{(\to)})$$
$$\overleftarrow{\mathbf{h}}_t = \sigma(W_h^{(\leftarrow)} \overleftarrow{\mathbf{h}}_{t+1} + W_x^{(\leftarrow)} \mathbf{x}_t + \mathbf{b}^{(\leftarrow)})$$

The final representation concatenates both directions: $\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2d}$.

This doubles the parameter count but allows $\mathbf{h}_t$ to incorporate both past and future context — essential for tasks like named entity recognition and machine translation where the full sentence is available.

**Constraint:** Bidirectional RNNs are inherently **non-causal** — the backward pass requires the full sequence to be available. They cannot be used for autoregressive generation.

**For AI:** ELMo (2018) used bidirectional LSTMs as a pretrained language model. BERT (2019) is conceptually a deeply bidirectional transformer — it uses masked language modelling to "see" both directions simultaneously. The bidirectional LSTM is the direct predecessor of BERT's architecture.

### 8.2 Deep (Stacked) RNNs

Stack $L$ recurrent layers, feeding the output of layer $l$ as input to layer $l+1$:

$$\mathbf{h}_t^{(l)} = \sigma\!\left(W_h^{(l)} \mathbf{h}_{t-1}^{(l)} + W_x^{(l)} \mathbf{h}_t^{(l-1)} + \mathbf{b}^{(l)}\right)$$

where $\mathbf{h}_t^{(0)} = \mathbf{x}_t$. Deep RNNs learn hierarchical temporal representations: lower layers capture local patterns (phonemes, character n-grams), while higher layers capture long-range semantic structure.

**Residual connections between layers** (Wu et al., 2016):
$$\mathbf{h}_t^{(l)} = \sigma\!\left(\cdots\right) + \mathbf{h}_t^{(l-1)}$$
These prevent gradient vanishing through depth and allow training of 6-8 layer LSTMs.

**Practical recommendation:** For most sequence tasks, 2-3 LSTM layers with residual connections outperform both 1 layer and 6+ layers without residuals. Graves et al. (2013) showed 5-layer deep LSTMs win on speech recognition.

### 8.3 Clockwork RNNs and Multi-Timescale Models

A **Clockwork RNN** (Koutník et al., 2014) partitions the hidden state into groups operating at different timescales. Group $k$ updates only every $T_k$ steps:

$$\mathbf{h}_t^{(k)} = \begin{cases} \sigma(W^{(k)} \mathbf{h}_{t-1}^{(k)} + \cdots) & \text{if } t \bmod T_k = 0 \\ \mathbf{h}_{t-1}^{(k)} & \text{otherwise} \end{cases}$$

With $T_1 = 1 < T_2 < T_3 < \cdots$, the slow groups retain information over longer horizons while fast groups respond to rapid changes. This induces a temporal hierarchy without explicit gating.

**For AI:** The multi-scale temporal hierarchy of Clockwork RNNs connects to **multi-scale attention** in modern transformers (Longformer, BigBird) and the **hierarchical processing** in speech and video models where different components process signals at 10ms, 100ms, and 1s timescales.

### 8.4 Attention-Augmented RNNs

Bahdanau et al. (2015) added an attention mechanism to the encoder-decoder RNN that directly addresses the fixed-bottleneck problem:

**Attention score** (alignment model):
$$e_{t,s} = \mathbf{v}^\top \tanh(W_a \mathbf{h}_s^{(\text{enc})} + U_a \mathbf{h}_{t-1}^{(\text{dec})})$$

**Attention weights:**
$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_{s'} \exp(e_{t,s'})}$$

**Context vector** (weighted sum of encoder states):
$$\mathbf{c}_t = \sum_s \alpha_{t,s} \mathbf{h}_s^{(\text{enc})}$$

**Decoder step:**
$$\mathbf{h}_t^{(\text{dec})} = \text{LSTM}(\mathbf{h}_{t-1}^{(\text{dec})}, [\mathbf{y}_{t-1}; \mathbf{c}_t])$$

The attention weights $\alpha_{t,s}$ form an **alignment matrix**: $\alpha_{t,s}$ is large when the $t$-th output aligns with the $s$-th input. For translation, this produces a near-diagonal matrix for language pairs with similar word order.

**For AI:** This mechanism is the direct precursor to **multi-head self-attention** in transformers. The transformer replaces the recurrent structure entirely, using attention to mix all positions directly — but the mathematical form of the attention score ($\mathbf{q}^\top \mathbf{k}$ replaces $\mathbf{v}^\top \tanh(W\mathbf{h})$) is a direct generalisation. Understanding Bahdanau attention makes the transformer intuition much clearer.


---

## 9. Applications in Modern AI

### 9.1 Language Modelling: Character and Word Level

A **language model** assigns a probability to a sequence of tokens. For a sequence $(x_1, \ldots, x_T)$, the language model factorises:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, \ldots, x_{t-1})$$

An RNN language model estimates each conditional $P(x_t \mid x_1, \ldots, x_{t-1}) \approx \hat{y}_{t, x_t}$ by running the hidden state forward and applying a softmax output layer.

**Teacher forcing:** During training, the ground-truth token $x_{t-1}$ is fed as input at step $t$ (not the predicted $\hat{x}_{t-1}$). This stabilises training but creates a **train-test discrepancy** — at inference, only predicted tokens are available. **Scheduled sampling** (Bengio et al., 2015) gradually transitions from teacher-forced to model-generated inputs during training.

**Perplexity:** The standard metric for language models:
$$\text{PP} = \exp\!\left(-\frac{1}{T} \sum_{t=1}^T \log P(x_t \mid x_{1:t-1})\right) = \exp(\mathcal{L} / T)$$

Lower perplexity is better. A perplexity of $k$ means the model is on average as uncertain as a uniform distribution over $k$ equally likely outcomes. Historical benchmarks: vanilla RNN ≈ 130 PPL (Penn Treebank); LSTM ≈ 60 PPL; current transformers ≈ 20-30 PPL.

**For AI:** Character-level LSTMs (Karpathy, 2015) demonstrated that an LSTM trained on text learns to generate syntactically valid code, XML, and prose. This was the first clear public demonstration that neural networks could learn hierarchical sequential structure.

### 9.2 Sequence-to-Sequence and Encoder-Decoder

The **encoder-decoder** framework (Sutskever et al., 2014; Cho et al., 2014) handles variable-length input-to-output mappings (translation, summarisation, speech recognition):

**Encoder:** Process input sequence $(\mathbf{x}_1, \ldots, \mathbf{x}_{T_x})$, produce context vector $\mathbf{c} = \mathbf{h}_{T_x}^{(\text{enc})}$.

**Decoder:** Generate output sequence $(\mathbf{y}_1, \ldots, \mathbf{y}_{T_y})$ conditioned on $\mathbf{c}$:
$$\mathbf{h}_t^{(\text{dec})} = \text{LSTM}(\mathbf{h}_{t-1}^{(\text{dec})}, [\mathbf{y}_{t-1}; \mathbf{c}])$$

**The bottleneck problem:** The entire input sequence is compressed into a single fixed-size vector $\mathbf{c} \in \mathbb{R}^d$. For long input sequences ($T_x > 20$), information is inevitably lost. This motivated Bahdanau attention (Section 8.4), and subsequently the Transformer, which eliminates the bottleneck entirely by giving the decoder direct access to all encoder hidden states.

### 9.3 Time-Series and Scientific ML

RNNs and LSTMs remain competitive on time-series tasks where:
1. **Streaming inference** is required (no future context)
2. **Irregular sampling** is present (Neural ODE-based variants)
3. **Long-range dependencies** are less critical than local patterns

Applications:
- **Financial time-series:** Stock price prediction, volatility modelling; LSTMs capture momentum effects across days-to-weeks timescales
- **Medical signals:** ECG anomaly detection, ICU patient monitoring; LSTMs trained on 24-hour windows
- **Weather forecasting:** GraphCast (Google, 2023) uses message-passing, but LSTM baselines remain strong for single-station forecasting

**Comparison (2024 landscape):**

| Method | Long-range | Streaming | Parameters | When to use |
|---|---|---|---|---|
| LSTM | Good | Yes | Moderate | Streaming, resource-constrained |
| Temporal CNN | Good | Yes | Low | 1D conv for fixed-length patterns |
| Transformer | Excellent | No | High | Offline, full-sequence tasks |
| Mamba/SSM | Excellent | Yes | Low–Moderate | Long sequences, streaming |

### 9.4 State Space Models and Mamba: The 2024 Connection

**State space models (SSMs)** are linear dynamical systems:
$$\mathbf{h}_t = A\mathbf{h}_{t-1} + B\mathbf{x}_t, \qquad \mathbf{y}_t = C\mathbf{h}_t$$

with $A \in \mathbb{R}^{d \times d}$, $B \in \mathbb{R}^{d \times p}$, $C \in \mathbb{R}^{q \times d}$. This is a linear RNN — powerful enough to approximate long-range dependencies when $A$ is designed carefully.

**S4 (Gu et al., 2022):** Structures $A$ as a **HiPPO matrix** — a specific design that provably approximates the history with Legendre polynomial projections. This allows efficient training via convolution (parallel over time) while retaining the recurrent form for streaming inference.

**Mamba (Gu & Dao, 2024):** Adds **selective scanning** — makes $B$, $C$, and $\Delta$ (discretisation step) input-dependent:

$$B_t = B(\mathbf{x}_t), \quad C_t = C(\mathbf{x}_t), \quad \Delta_t = \Delta(\mathbf{x}_t)$$

This is directly analogous to LSTM gating: $\Delta_t$ controls the effective memory timescale at each step (like the forget gate $\mathbf{f}_t$). When $\Delta_t$ is large, the state integrates over a short window; when small, over a long window. Mamba achieves transformer-level performance on language benchmarks with $O(T)$ rather than $O(T^2)$ compute.

```
LSTM ↔ MAMBA CORRESPONDENCE
════════════════════════════════════════════════════════════════════════

  LSTM                          Mamba (selective SSM)
  ─────────────────────         ──────────────────────────
  Cell state c_t                State h_t (continuous)
  Forget gate f_t               Step size Δ_t (input-dependent)
  Input gate i_t                Input projection B_t
  Output gate o_t               Output projection C_t
  Additive update               Discretised linear recurrence
  Hidden dim d                  State dim N (per channel)

════════════════════════════════════════════════════════════════════════
```

### 9.5 Neural ODE Connection

A vanilla RNN hidden state update can be written as an **Euler discretisation** of a continuous ODE:

$$\frac{d\mathbf{h}}{dt} = f(\mathbf{h}(t), \mathbf{x}(t); \theta) \approx \mathbf{h}_{t+1} - \mathbf{h}_t = \sigma(W_h \mathbf{h}_t + W_x \mathbf{x}_t + \mathbf{b}) - \mathbf{h}_t$$

A **Neural ODE** (Chen et al., 2018) makes this continuous-time limit exact:
$$\frac{d\mathbf{h}(t)}{dt} = f_\theta(\mathbf{h}(t), \mathbf{x}(t), t)$$

and uses an ODE solver (e.g., Runge-Kutta) to integrate. Gradients are computed via the **adjoint method** (a continuous analogue of BPTT):

$$\frac{d\mathbf{a}}{dt} = -\mathbf{a}(t)^\top \frac{\partial f}{\partial \mathbf{h}}, \quad \frac{d\mathcal{L}}{d\theta} = -\int_{t_0}^{t_1} \mathbf{a}(t)^\top \frac{\partial f}{\partial \theta} \, dt$$

where $\mathbf{a}(t) = \frac{\partial \mathcal{L}}{\partial \mathbf{h}(t)}$ is the adjoint (co-state). This avoids storing all intermediate hidden states (unlike BPTT), using $O(1)$ memory per time point.

**For AI:** Neural ODEs handle **irregular time series** naturally — the ODE solver can evaluate $\mathbf{h}(t)$ at any time $t$, including non-integer and unequal intervals. This is useful for medical time series (blood draws at irregular intervals) and event-driven systems.

---

## 10. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
|---|---|---|---|
| 1 | Applying dropout to hidden-to-hidden connections at each step | Destroys temporal memory; different masks at each step disrupt learned structure | Apply dropout only to input/output connections, or use variational (same mask per sequence) |
| 2 | Initialising forget gate bias to 0 | LSTM starts discarding half its memory before learning anything | Initialise $\mathbf{b}_f = 1$ (or $2$ for long sequences) |
| 3 | Confusing cell state $\mathbf{c}_t$ with hidden state $\mathbf{h}_t$ | They serve different roles: $\mathbf{c}_t$ is long-term memory, $\mathbf{h}_t$ is working output | $\mathbf{c}_t$ accumulates via addition; $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{c}_t)$ is filtered |
| 4 | Using spectral radius $\rho(W_h) < 1$ as the only stability criterion | True for linear systems; the nonlinear tanh introduces additional stability that $\rho$ does not capture | Spectral radius governs linear regime; also check empirical gradient norms |
| 5 | Teacher forcing at inference time | Teacher forcing uses ground truth tokens; at inference, only model predictions are available; distribution mismatch | Use scheduled sampling or fully autoregressive inference from step 1 |
| 6 | Truncated BPTT with window shorter than the dependency | Gradient cannot reach earlier timesteps; long-range patterns are not learned | Set truncation window ≥ the expected dependency length |
| 7 | Using ReLU without gradient clipping in a vanilla RNN | Unbounded activations and spectral radius > 1 cause immediate gradient explosion | Use gradient clipping ($\tau = 1$–$5$) or switch to tanh |
| 8 | Treating bidirectional RNNs as causal | The backward pass sees future inputs; cannot be used for real-time or autoregressive generation | Use bidirectional only for offline/full-sequence tasks (classification, NER, translation encoder) |
| 9 | Ignoring the distinction between perplexity and cross-entropy loss | Perplexity = exp(loss) — a 10% reduction in loss is a large reduction in perplexity | Track both; compare models using PPL for interpretability, loss for training stability |
| 10 | Stacking many LSTM layers without residual connections | Deep LSTMs (6+) suffer from gradient vanishing through depth | Add residual (skip) connections between LSTM layers for depth > 3 |
| 11 | Using the same dropout rate for all sequence lengths | Short sequences tolerate high dropout; long sequences need lower rates to preserve gradient flow | Tune dropout rate; prefer zoneout for long sequences |

---

## 11. Exercises

**Exercise 1 (★):** Implement a vanilla RNN forward pass from scratch using only NumPy. Given matrices $W_h$, $W_x$, bias $\mathbf{b}$, and an input sequence $(\mathbf{x}_1, \ldots, \mathbf{x}_T)$:
- (a) Implement the recurrence $\mathbf{h}_t = \tanh(W_h \mathbf{h}_{t-1} + W_x \mathbf{x}_t + \mathbf{b})$
- (b) Verify that your implementation matches `numpy.dot` computations step-by-step
- (c) Show that setting $W_h = 0$ reduces to independent per-step MLPs

**Exercise 2 (★):** Derive BPTT gradients and verify numerically. Given a 2-step RNN with scalar hidden state:
- (a) Compute $\frac{\partial \mathcal{L}}{\partial W_h}$ analytically via chain rule
- (b) Verify against numerical gradient: $\frac{\partial \mathcal{L}}{\partial W_h} \approx \frac{\mathcal{L}(W_h + \epsilon) - \mathcal{L}(W_h - \epsilon)}{2\epsilon}$
- (c) Show that the gradient decomposes as a sum over time steps

**Exercise 3 (★★):** Implement all four LSTM gate equations and verify the cell-state gradient highway:
- (a) Implement LSTM forward pass: compute $\mathbf{f}_t, \mathbf{i}_t, \tilde{\mathbf{c}}_t, \mathbf{o}_t, \mathbf{c}_t, \mathbf{h}_t$
- (b) Compare to `torch.nn.LSTMCell` for numerical agreement
- (c) Set $\mathbf{f}_t = \mathbf{1}$ and show that $\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_0} \approx I$ (identity gradient)
- (d) Set $\mathbf{f}_t = \mathbf{0}$ and show that $\frac{\partial \mathbf{c}_T}{\partial \mathbf{c}_0} \approx 0$ (complete forgetting)

**Exercise 4 (★★):** Implement GRU and compare parameter count with LSTM:
- (a) Implement GRU: compute $\mathbf{z}_t, \mathbf{r}_t, \tilde{\mathbf{h}}_t, \mathbf{h}_t$
- (b) Count parameters for both GRU and LSTM with $d=128$, $p=64$
- (c) Show that GRU with $\mathbf{z}_t = \mathbf{0}$ copies the previous state (identity)
- (d) Show that GRU with $\mathbf{z}_t = \mathbf{1}$ replaces state with candidate

**Exercise 5 (★★):** Analyse spectral radius vs gradient norm across sequence lengths:
- (a) Create RNNs with $\rho(W_h) \in \{0.5, 0.9, 1.0, 1.1\}$ using scaled orthogonal matrices
- (b) For each, run a forward pass over $T = 100$ steps and compute $\|\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_0}\|_F$
- (c) Plot gradient norm vs $T$ for each $\rho$ value; verify exponential decay/growth
- (d) Compute the effective memory horizon: largest $k$ with $\|\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_{T-k}}\|_F > 0.01$

**Exercise 6 (★★):** Implement gradient clipping and demonstrate its effect on training stability:
- (a) Train a vanilla RNN (without clipping) on a synthetic copy task; observe gradient explosion
- (b) Add gradient norm clipping with $\tau = 5$; compare loss curves
- (c) Plot gradient norm histogram over training steps for both cases
- (d) Show that clipping preserves gradient direction: $\cos(\mathbf{g}_{\text{clipped}}, \mathbf{g}_{\text{original}}) = 1$

**Exercise 7 (★★★):** Build Bahdanau attention over an LSTM encoder-decoder:
- (a) Implement encoder: bidirectional LSTM over source sequence
- (b) Implement attention: compute $e_{t,s} = \mathbf{v}^\top \tanh(W_a \mathbf{h}_s + U_a \mathbf{h}_{t-1})$, then softmax
- (c) Implement decoder: LSTM step conditioned on context vector $\mathbf{c}_t = \sum_s \alpha_{t,s} \mathbf{h}_s$
- (d) Visualise the attention weight matrix $\alpha$ as a heatmap; verify it produces a near-diagonal pattern for a simple copy task

**Exercise 8 (★★★):** Implement a character-level RNN language model and compute perplexity:
- (a) Tokenise a text string at the character level; create train/validation split
- (b) Implement RNN LM with cross-entropy loss and teacher forcing
- (c) Implement the same architecture with LSTM instead of vanilla RNN
- (d) Compare validation perplexity of RNN vs LSTM after 50 epochs; report the improvement
- (e) Sample from the trained model: start from a seed string and generate 200 characters

---

## 12. Why This Matters for AI (2026 Perspective)

| Concept | AI/ML Impact |
|---|---|
| Vanishing gradient (BPTT) | Motivated residual connections (ResNet), layer norm, and LSTM — foundational for all deep learning |
| LSTM gating mechanism | Direct ancestor of Mamba's selective state spaces; gating appears in GRU, highway networks, transformer FFN gating (SwiGLU) |
| Cell state highway | Conceptual precursor to residual connections; same mathematical structure (additive updates bypass multiplicative decay) |
| Spectral radius stability | Motivates orthogonal initialisation, spectral normalisation (GANs), and careful weight initialisation in all deep networks |
| Bidirectional LSTM | Architectural ancestor of BERT and all encoder-only transformers; bidirectionality as a design principle |
| Encoder-decoder with LSTM | Foundation of seq2seq learning; Bahdanau attention directly inspired the transformer's attention mechanism |
| Gradient clipping | Universal training trick used in GPT-3, GPT-4, LLaMA, Gemini — every large-scale LLM training run |
| Character-level LM | Demonstrated that neural networks learn structured patterns in sequences; opened the path to GPT |
| Teacher forcing | Still used in LLM training; scheduled sampling and RLHF address the train-test discrepancy |
| Neural ODE | Enables continuous-time sequence models; used in latent ODE for irregular medical time-series |
| Mamba/SSM connection | LSTMs and SSMs are unified by their gating and state-space structure; the field is revisiting recurrence at scale |

---

## 13. Conceptual Bridge

### Where We Came From

This section builds directly on two foundational prerequisites. From **probabilistic models** (Section 14-03), we inherit the framework of Hidden Markov Models — a probabilistic recurrent architecture where the hidden state is discrete and the transitions are stochastic. RNNs can be seen as the deterministic, continuous-state generalisation of HMMs: instead of a finite state machine with learned transition probabilities, an RNN has a continuous hidden state with a learned nonlinear transition function. BPTT is the deterministic analogue of the forward-backward algorithm, and the vanishing gradient problem is the analogue of the mixing time issue in long Markov chains.

From **neural networks** (Section 14-02), we inherit the core backpropagation algorithm. BPTT is simply backpropagation applied to a computation graph that has been unrolled in time. The new insight is that **depth in time** has the same pathological gradient properties as **depth in space** — and the LSTM's cell state highway is the temporal equivalent of ResNet's spatial skip connections.

### Where We Are Going

The **Transformer architecture** (Section 14-05) is the natural successor to attention-augmented RNNs. Having understood Bahdanau attention as a mechanism for allowing the decoder to query all encoder hidden states, the Transformer's key innovation — **self-attention, applied at every layer to every position** — becomes a natural extrapolation. The query-key-value formalism ($\mathbf{q}^\top \mathbf{k} / \sqrt{d}$ replacing $\mathbf{v}^\top \tanh(W\mathbf{h})$) is a cleaner, more scalable implementation of the same idea.

The transformer eliminates the recurrent bottleneck entirely: instead of compressing history into a fixed-size state, it simply keeps all previous token representations in memory (the KV cache) and attends to all of them at each step. The trade-off is the $O(T^2)$ attention cost — which is why, in 2024, the field is revisiting recurrence through Mamba and other linear-attention variants that combine the constant-time-per-step property of RNNs with the long-range memory of attention.

```
POSITION IN CURRICULUM
════════════════════════════════════════════════════════════════════════

  14-01 Linear Models          14-02 Neural Networks
        │                            │
        └────────────────────────────┘
                        │
                        ▼
              14-03 Probabilistic Models
              (HMMs → continuous-state RNNs)
                        │
                        ▼
         ┌──── 14-04 RNNs & LSTMs ────┐  ◄── YOU ARE HERE
         │                             │
         │  Recurrence, BPTT,          │
         │  Gating, Stability,         │
         │  Attention (Bahdanau)       │
         └──────────────┬──────────────┘
                        │
                        ▼
              14-05 Transformer Architecture
              (Self-attention replaces recurrence;
               KV cache replaces hidden state;
               positional encoding replaces temporal order)
                        │
                        ▼
              14-06 Reinforcement Learning
              (Policy gradient, value functions,
               sequence of state-action-reward)

════════════════════════════════════════════════════════════════════════
```

The mathematics of this section — gating, state updates, gradient flow through time — will reappear throughout the remainder of the curriculum. Every time you see a residual connection, a gating mechanism (SwiGLU, GLU, Mamba's selective scan), or a KV cache, you are seeing a descendant of the ideas developed here.

