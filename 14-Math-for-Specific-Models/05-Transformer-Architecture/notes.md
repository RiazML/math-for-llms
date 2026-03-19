[← Back to Math for Specific Models](../README.md) | [Next: Reinforcement Learning →](../06-Reinforcement-Learning/notes.md)

---

# Transformer Architecture: Mathematical Foundations

> _"Attention is all you need — but understanding why requires linear algebra, probability, and careful analysis of computational complexity."_
> — paraphrasing Vaswani et al. (2017)

## Overview

The Transformer is the architecture behind every frontier language model — GPT-4, LLaMA-3, Gemini, Claude — and increasingly dominates vision, speech, and multimodal systems. Unlike recurrent networks that process sequences step-by-step through a hidden state bottleneck, the Transformer processes all positions simultaneously through **self-attention**: a mechanism that lets every token in a sequence directly query every other token for relevant information.

This section develops the full mathematical theory of the Transformer from first principles. We derive the attention function as a kernel smoother, prove why the $1/\sqrt{d_k}$ scaling is necessary from variance analysis, study multi-head attention as subspace decomposition, and analyse positional encodings through the lens of group theory (rotations in complex subspaces for RoPE). We then examine the feed-forward network as an associative key–value memory, study normalisation layers as signal propagation controllers, and derive the $O(n^2 d)$ complexity bound that motivates FlashAttention and linear attention variants.

The treatment connects every mathematical concept to its concrete role in modern LLMs: why GPT uses decoder-only architecture, how LoRA exploits low-rank structure in attention weight updates, why RMSNorm replaced LayerNorm, how mechanistic interpretability reveals induction heads and superposition, and how Mamba's selective scan relates attention back to gated recurrence. By the end, you will not merely know the Transformer equations — you will understand **why each design choice is mathematically necessary**.

## Prerequisites

- Neural networks: forward pass, backpropagation, chain rule (Section 14-02)
- RNNs and LSTMs: sequential bottleneck, BPTT, vanishing gradients (Section 14-04)
- Linear algebra: matrix multiplication, eigenvalues, SVD, low-rank approximation (Chapters 02–03)
- Calculus: partial derivatives, Jacobians, gradient flow (Chapters 04–05)
- Probability: softmax, cross-entropy, Boltzmann distribution (Sections 06, 13-01)
- Norms: Frobenius, spectral, nuclear norms (Section 03-06)

## Companion Notebooks

| Notebook | Description |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Interactive demos: attention as kernel smoother, variance scaling proof, RoPE rotation visualisation, FlashAttention tiling, attention head patterns, scaling laws |
| [exercises.ipynb](exercises.ipynb) | 8 graded problems: dot-product attention, multi-head from scratch, RoPE implementation, causal masking, parameter counting, FlashAttention, LoRA, induction heads |

## Learning Objectives

After completing this section, you will:

- Derive scaled dot-product attention from first principles and prove the $1/\sqrt{d_k}$ scaling from variance analysis
- Implement multi-head attention as parallel low-rank projections and explain why single-head has rank limitations
- Prove that self-attention is permutation-equivariant and explain why positional encodings are mathematically necessary
- Implement sinusoidal encodings and RoPE, and explain RoPE's rotation-group structure
- Derive the SwiGLU feed-forward variant and explain the FFN-as-memory interpretation
- Compare Pre-Norm vs Post-Norm signal propagation and explain why RMSNorm replaced LayerNorm
- Calculate FLOPs and memory for attention ($O(n^2 d)$) and FFN ($O(nd^2)$), and explain FlashAttention's tiling strategy
- Apply LoRA to attention weights and compute the parameter savings from low-rank adaptation

---

## Table of Contents

- [Transformer Architecture: Mathematical Foundations](#transformer-architecture-mathematical-foundations)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition and Motivation](#1-intuition-and-motivation)
    - [1.1 The Sequential Bottleneck](#11-the-sequential-bottleneck)
    - [1.2 Attention as Soft Database Lookup](#12-attention-as-soft-database-lookup)
    - [1.3 From Bahdanau to Vaswani](#13-from-bahdanau-to-vaswani)
    - [1.4 Historical Timeline](#14-historical-timeline)
  - [2. The Attention Mechanism — First Principles](#2-the-attention-mechanism--first-principles)
    - [2.1 Attention as a Weighted Average](#21-attention-as-a-weighted-average)
    - [2.2 Dot-Product Similarity](#22-dot-product-similarity)
    - [2.3 The Variance Problem and Scaling](#23-the-variance-problem-and-scaling)
    - [2.4 Softmax as Boltzmann Distribution](#24-softmax-as-boltzmann-distribution)
    - [2.5 Attention as Kernel Smoothing](#25-attention-as-kernel-smoothing)
    - [2.6 The Attention Matrix](#26-the-attention-matrix)
  - [3. Multi-Head Attention](#3-multi-head-attention)
    - [3.1 Single-Head Formulation](#31-single-head-formulation)
    - [3.2 Why Multiple Heads](#32-why-multiple-heads)
    - [3.3 Output Projection](#33-output-projection)
    - [3.4 Causal Masking](#34-causal-masking)
    - [3.5 Cross-Attention](#35-cross-attention)
    - [3.6 GQA and MQA](#36-gqa-and-mqa)
  - [4. Positional Encoding Theory](#4-positional-encoding-theory)
    - [4.1 Permutation Equivariance Proof](#41-permutation-equivariance-proof)
    - [4.2 Sinusoidal Encoding](#42-sinusoidal-encoding)
    - [4.3 Rotary Position Embedding (RoPE)](#43-rotary-position-embedding-rope)
    - [4.4 ALiBi](#44-alibi)
    - [4.5 Length Generalisation](#45-length-generalisation)
  - [5. Feed-Forward Networks as Key–Value Memories](#5-feed-forward-networks-as-keyvalue-memories)
    - [5.1 Position-Wise FFN](#51-position-wise-ffn)
    - [5.2 Expansion Ratio and Capacity](#52-expansion-ratio-and-capacity)
    - [5.3 SwiGLU](#53-swiglu)
    - [5.4 FFN as Associative Memory](#54-ffn-as-associative-memory)
  - [6. Normalisation and Residual Connections](#6-normalisation-and-residual-connections)
    - [6.1 Residual Stream](#61-residual-stream)
    - [6.2 Layer Norm](#62-layer-norm)
    - [6.3 RMSNorm](#63-rmsnorm)
    - [6.4 Pre-Norm vs Post-Norm](#64-pre-norm-vs-post-norm)
    - [6.5 DeepNorm](#65-deepnorm)
  - [7. Complete Transformer Block and Variants](#7-complete-transformer-block-and-variants)
    - [7.1 The Full Block](#71-the-full-block)
    - [7.2 Decoder-Only (GPT, LLaMA)](#72-decoder-only-gpt-llama)
    - [7.3 Encoder-Only (BERT)](#73-encoder-only-bert)
    - [7.4 Encoder–Decoder (T5, Whisper)](#74-encoderdecoder-t5-whisper)
    - [7.5 Parameter Counting](#75-parameter-counting)
  - [8. Computational Complexity and Efficient Attention](#8-computational-complexity-and-efficient-attention)
    - [8.1 FLOPs Analysis](#81-flops-analysis)
    - [8.2 Memory Bottleneck and KV Cache](#82-memory-bottleneck-and-kv-cache)
    - [8.3 FlashAttention](#83-flashattention)
    - [8.4 Multi-Latent Attention (MLA)](#84-multi-latent-attention-mla)
    - [8.5 Linear Attention](#85-linear-attention)
  - [9. Training Dynamics and Optimisation](#9-training-dynamics-and-optimisation)
    - [9.1 Weight Initialisation](#91-weight-initialisation)
    - [9.2 AdamW + Warmup + Cosine Decay](#92-adamw--warmup--cosine-decay)
    - [9.3 Gradient Clipping](#93-gradient-clipping)
    - [9.4 Mixed Precision](#94-mixed-precision)
    - [9.5 Scaling Laws](#95-scaling-laws)
  - [10. Interpretability](#10-interpretability)
    - [10.1 Attention Patterns](#101-attention-patterns)
    - [10.2 Residual Stream Hypothesis](#102-residual-stream-hypothesis)
    - [10.3 Superposition and Polysemanticity](#103-superposition-and-polysemanticity)
    - [10.4 Probing](#104-probing)
  - [11. Modern Extensions (2024–2026)](#11-modern-extensions-20242026)
    - [11.1 LoRA](#111-lora)
    - [11.2 KV Cache Optimisation](#112-kv-cache-optimisation)
    - [11.3 Mixture of Experts](#113-mixture-of-experts)
    - [11.4 From Transformers to SSMs](#114-from-transformers-to-ssms)
  - [12. Common Mistakes](#12-common-mistakes)
  - [13. Exercises](#13-exercises)
  - [14. Why This Matters for AI (2026)](#14-why-this-matters-for-ai-2026)
  - [15. Conceptual Bridge](#15-conceptual-bridge)
    - [Looking Backward](#looking-backward)
    - [Looking Forward](#looking-forward)
    - [The Transformer as a Mathematical Object](#the-transformer-as-a-mathematical-object)
    - [Key References](#key-references)

---

## 1. Intuition and Motivation

### 1.1 The Sequential Bottleneck

Recall from Section 14-04 that an RNN compresses an entire input sequence into a single hidden state vector $\mathbf{h}_T \in \mathbb{R}^d$. For a sequence of length $T$, the final hidden state must encode all information from positions $1, 2, \ldots, T$ into a fixed-dimensional vector. This creates an **information bottleneck**: as $T$ grows, the model must pack more information into the same $d$ dimensions, and earlier positions are systematically forgotten due to the vanishing gradient problem.

Concretely, the gradient of the loss with respect to an early hidden state decays as:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=t+1}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}}$$

When the spectral radius of the Jacobian $\partial \mathbf{h}_k / \partial \mathbf{h}_{k-1}$ is less than 1, this product shrinks exponentially in $T - t$. The LSTM mitigates this through gated additive updates, but the fundamental problem remains: information must flow through a sequential chain.

**For AI:** This bottleneck is why early machine translation systems (before 2017) struggled with long sentences. The encoder had to compress "The cat sat on the mat that was next to the door of the house at the end of the street" into a single vector before the decoder could begin translating.

**Information-theoretic perspective.** The hidden state $\mathbf{h}_T \in \mathbb{R}^d$ can store at most $d \cdot b$ bits of information (where $b$ is the bits per float, typically 16 or 32). For a sequence of $T$ tokens from a vocabulary of size $V$, the input contains up to $T \log_2 V$ bits. For $V = 32000$ and $T = 1000$: $15{,}000$ bits of input compressed into $d \cdot 32$ bits of state. With $d = 512$: $16{,}384$ bits — barely enough, and only if the hidden state is used with perfect efficiency (which it is not).

This quantitative argument shows the bottleneck is real, not merely theoretical. The attention mechanism resolves it by allowing the output at each step to access the full $T \times d$ matrix of hidden states — a $T$-fold increase in available information.

### 1.2 Attention as Soft Database Lookup

The key insight of the attention mechanism is to bypass the sequential chain entirely. Instead of forcing information through a bottleneck, attention allows the model to **directly access** any position in the input sequence.

The analogy is a database query. Given:
- A **query** $\mathbf{q}$: "what information do I need right now?"
- A set of **keys** $\{\mathbf{k}_1, \ldots, \mathbf{k}_n\}$: "what does each position contain?"
- A set of **values** $\{\mathbf{v}_1, \ldots, \mathbf{v}_n\}$: "the actual content at each position"

Attention computes a weighted average of the values, where the weights depend on how well each key matches the query:

$$\operatorname{Attn}(\mathbf{q}, K, V) = \sum_{j=1}^{n} \frac{\exp(\text{score}(\mathbf{q}, \mathbf{k}_j))}{\sum_{l=1}^{n} \exp(\text{score}(\mathbf{q}, \mathbf{k}_l))} \cdot \mathbf{v}_j$$

This is a **soft** lookup because it returns a weighted combination of all values rather than selecting a single entry. The softmax ensures the weights form a probability distribution: $\alpha_j \ge 0$ and $\sum_j \alpha_j = 1$.

**Hard vs soft attention.** Hard attention selects a single key–value pair: $\operatorname{HardAttn}(\mathbf{q}) = \mathbf{v}_{j^*}$ where $j^* = \arg\max_j \text{score}(\mathbf{q}, \mathbf{k}_j)$. This is non-differentiable (requires REINFORCE or straight-through estimators for training). Soft attention is differentiable through the softmax and thus trainable by standard backpropagation. The Transformer's success depends on this differentiability — it enables efficient end-to-end training with gradient descent.

**For AI:** In a language model generating the word "it" in "The cat sat on the mat. It was comfortable.", attention allows "it" to directly query all previous tokens and assign high weight to "cat" — regardless of how far back "cat" appeared.

**The parallel processing advantage.** Attention computes all pairwise interactions in a single matrix multiplication $QK^\top$. For a sequence of length $n$, this requires $O(1)$ sequential steps (parallelised across the $n^2$ pairs), compared to $O(n)$ sequential steps for an RNN. On modern GPUs with thousands of cores, this parallelism translates directly to wall-clock speedup:

| Model type | Sequential steps | Parallelism | Training speed |
| --- | --- | --- | --- |
| RNN | $O(n)$ | None across time | Slow for long sequences |
| LSTM | $O(n)$ | None across time | Slow, but better gradients |
| Transformer | $O(1)$ | $O(n^2)$ pairs in parallel | Fast — limited by memory |

This is the practical reason Transformers won: not just better quality, but orders-of-magnitude faster training through parallelism.

**Hardware co-evolution.** The Transformer's reliance on matrix multiplication ($QK^\top$, $AV$, linear projections) maps directly onto GPU tensor cores, which are optimised for exactly this operation. The NVIDIA A100 achieves 312 TFLOPS for fp16 matrix multiply but only ~19 TFLOPS for general-purpose compute — a 16× gap. An architecture built on matrix multiplication (Transformer) exploits the hardware 16× more efficiently than one built on sequential operations (RNN). This hardware–algorithm co-design is not coincidental: the Transformer was designed at Google, which builds both the models and the TPU hardware.

### 1.3 From Bahdanau to Vaswani

**Bahdanau attention (2014)** introduced additive attention for machine translation, where the decoder at each step attends to all encoder hidden states. The score function was a learned MLP:

$$\text{score}(\mathbf{s}_t, \mathbf{h}_j) = \mathbf{v}^\top \tanh(W_1 \mathbf{s}_t + W_2 \mathbf{h}_j)$$

**Luong attention (2015)** simplified this to dot-product attention: $\text{score}(\mathbf{s}_t, \mathbf{h}_j) = \mathbf{s}_t^\top \mathbf{h}_j$, which is faster and equally effective.

**Vaswani et al. (2017)** made the conceptual leap: if attention can replace the decoder's dependence on sequential hidden states, why not replace the encoder's recurrence too? The Transformer uses **self-attention** — where queries, keys, and values all come from the same sequence — as the only mechanism for mixing information across positions. No recurrence, no convolution.

**The three types of attention:**

1. **Encoder self-attention:** every position attends to every position in the input. Used in BERT-style models.
2. **Decoder self-attention (causal):** each position attends only to previous positions. Used in GPT-style language models.
3. **Cross-attention:** decoder positions attend to encoder positions. Used in encoder–decoder models (T5, Whisper).

The Transformer paper introduced all three simultaneously. The subsequent history of LLMs is largely the story of decoder self-attention becoming dominant — because autoregressive language modelling turned out to be sufficient for most tasks.

### 1.4 Historical Timeline

```
TRANSFORMER TIMELINE
════════════════════════════════════════════════════════════════════════

  2014  Bahdanau et al.      Additive attention for seq2seq MT
  2015  Luong et al.         Dot-product attention variants
  2017  Vaswani et al.       "Attention Is All You Need" — the Transformer
  2018  GPT-1 (OpenAI)       Decoder-only Transformer for language modelling
  2018  BERT (Google)        Encoder-only, masked language model pretraining
  2019  GPT-2                Scaling up decoder-only (1.5B params)
  2020  GPT-3                175B params; in-context learning emerges
  2020  T5 (Google)          Encoder–decoder for text-to-text
  2021  RoPE (Su et al.)     Rotary positional embeddings
  2022  Chinchilla           Scaling laws: optimal data/param ratio
  2022  FlashAttention       IO-aware exact attention (Dao et al.)
  2023  LLaMA (Meta)         Efficient open-source: RoPE + SwiGLU + RMSNorm
  2023  Mistral / Mixtral    Sliding window + Mixture of Experts
  2024  DeepSeek-V2          Multi-Latent Attention (MLA)
  2024  Mamba (Gu & Dao)     Selective state space: attention → gated RNN
  2025  GPT-4o, Claude-3.5   Frontier multimodal transformers
  2026  Hybrid architectures Transformer + SSM + MoE combinations

════════════════════════════════════════════════════════════════════════
```

---

## 2. The Attention Mechanism — First Principles

### 2.1 Attention as a Weighted Average

At its core, attention is a parametric weighted average. Given a query vector $\mathbf{q} \in \mathbb{R}^{d_k}$ and a set of $n$ key–value pairs $\{(\mathbf{k}_j, \mathbf{v}_j)\}_{j=1}^n$ with $\mathbf{k}_j \in \mathbb{R}^{d_k}$ and $\mathbf{v}_j \in \mathbb{R}^{d_v}$, the attention output is:

$$\operatorname{Attn}(\mathbf{q}, K, V) = \sum_{j=1}^{n} \alpha_j \mathbf{v}_j, \qquad \alpha_j = \frac{\exp(e_j)}{\sum_{l=1}^{n} \exp(e_l)}, \qquad e_j = \text{score}(\mathbf{q}, \mathbf{k}_j)$$

where $\alpha_j \ge 0$ and $\sum_j \alpha_j = 1$. The attention weights $\boldsymbol{\alpha} \in \Delta^{n-1}$ lie on the probability simplex.

In matrix form, for $n$ queries simultaneously: let $Q \in \mathbb{R}^{n \times d_k}$, $K \in \mathbb{R}^{n \times d_k}$, $V \in \mathbb{R}^{n \times d_v}$. Then:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where softmax is applied row-wise. The matrix $A = \operatorname{softmax}(QK^\top / \sqrt{d_k}) \in \mathbb{R}^{n \times n}$ is the **attention matrix**: each row is a probability distribution over the $n$ positions.

### 2.2 Dot-Product Similarity

The score function in the Transformer is the **dot product**: $\text{score}(\mathbf{q}, \mathbf{k}) = \mathbf{q}^\top \mathbf{k}$. Why dot product?

The dot product measures the cosine of the angle between two vectors, scaled by their magnitudes:

$$\mathbf{q}^\top \mathbf{k} = \lVert \mathbf{q} \rVert \lVert \mathbf{k} \rVert \cos\theta$$

When queries and keys are approximately unit-norm (which normalisation layers encourage), the dot product is proportional to cosine similarity. Vectors pointing in similar directions (similar semantic content) get high scores; orthogonal vectors (unrelated content) get near-zero scores.

**Computational advantage:** The dot product for all $n^2$ query–key pairs can be computed as a single matrix multiplication $QK^\top$, which runs at near-peak throughput on GPUs. Additive attention (Bahdanau) requires $O(n^2 d)$ operations that cannot be parallelised as efficiently.

**Non-example: Euclidean distance as score.** One might consider $\text{score}(\mathbf{q}, \mathbf{k}) = -\lVert \mathbf{q} - \mathbf{k} \rVert^2$. Expanding: $-\lVert \mathbf{q} - \mathbf{k} \rVert^2 = 2\mathbf{q}^\top \mathbf{k} - \lVert \mathbf{q} \rVert^2 - \lVert \mathbf{k} \rVert^2$. If queries and keys are normalised ($\lVert \mathbf{q} \rVert = \lVert \mathbf{k} \rVert = c$), this reduces to $2\mathbf{q}^\top \mathbf{k} - 2c^2$, which is equivalent to dot-product attention up to a constant. The dot product is simpler and sufficient.

**Attention score comparison:**

| Score function | Formula | Complexity | Used in |
| --- | --- | --- | --- |
| Dot product | $\mathbf{q}^\top \mathbf{k}$ | $O(d)$ | Transformer (Vaswani) |
| Scaled dot product | $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ | $O(d)$ | All modern Transformers |
| Additive (Bahdanau) | $\mathbf{v}^\top \tanh(W_1 \mathbf{q} + W_2 \mathbf{k})$ | $O(d^2)$ | Seq2seq attention |
| Cosine | $\mathbf{q}^\top \mathbf{k} / (\lVert \mathbf{q} \rVert \lVert \mathbf{k} \rVert)$ | $O(d)$ | Some retrieval models |
| Bilinear | $\mathbf{q}^\top W \mathbf{k}$ | $O(d^2)$ | Luong attention |

### 2.3 The Variance Problem and Scaling

**Theorem (Variance of dot products in high dimensions).** Let $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with independent components $q_i, k_i$ each having mean 0 and variance 1. Then:

$$\mathbb{E}[\mathbf{q}^\top \mathbf{k}] = 0, \qquad \operatorname{Var}(\mathbf{q}^\top \mathbf{k}) = d_k$$

*Proof.* We have $\mathbf{q}^\top \mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$. Since the components are independent with $\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0$:

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

$$\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$

By independence across $i$: $\operatorname{Var}(\mathbf{q}^\top \mathbf{k}) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k$. $\square$

**The problem:** For $d_k = 64$ (a typical head dimension), the dot products have standard deviation $\sqrt{64} = 8$. This means some logits will be very large ($\pm 16$ or more), pushing the softmax into saturation where $\alpha_j \approx 0$ or $\alpha_j \approx 1$. In this regime, gradients through softmax vanish.

**The fix:** Divide by $\sqrt{d_k}$:

$$\operatorname{Var}\!\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

This restores unit variance regardless of the head dimension, keeping softmax in its sensitive (gradient-rich) regime.

### 2.4 Softmax as Boltzmann Distribution

The softmax function is the Boltzmann (Gibbs) distribution from statistical mechanics:

$$\alpha_j = \frac{\exp(e_j / \tau)}{\sum_{l=1}^{n} \exp(e_l / \tau)}$$

where $\tau$ is the **temperature**. In the Transformer, $\tau = \sqrt{d_k}$ (absorbed into the score scaling).

**Properties of the attention distribution:**

- **$\tau \to 0$ (low temperature):** The distribution collapses to a one-hot vector selecting $\arg\max_j e_j$. This is "hard" attention — equivalent to a hash table lookup.
- **$\tau \to \infty$ (high temperature):** The distribution approaches uniform $\alpha_j = 1/n$ for all $j$. Every position gets equal weight — attention becomes a simple average.
- **$\tau = \sqrt{d_k}$ (standard):** A balanced regime where attention can be sharp or diffuse depending on the learned queries and keys.

**Entropy of attention:** The entropy $H(\boldsymbol{\alpha}) = -\sum_j \alpha_j \log \alpha_j$ measures how "spread out" the attention is. Low entropy = focused on few positions (typical for syntactic heads). High entropy = distributed across many positions (typical for positional heads).

**For AI:** During LLM inference, the temperature parameter in sampling ($P(x_i) \propto \exp(z_i / \tau)$) is mathematically identical to attention temperature. Low temperature → deterministic, high temperature → creative.

**Numerical stability of softmax.** Computing $\exp(e_j)$ directly overflows for large logits. The standard trick uses the identity:

$$\operatorname{softmax}(e_j) = \frac{\exp(e_j - \max_l e_l)}{\sum_l \exp(e_l - \max_l e_l)}$$

Subtracting $\max_l e_l$ ensures all exponents are $\le 0$, preventing overflow. The result is mathematically identical. This is the **log-sum-exp trick**, and every deep learning framework implements it automatically.

**Softmax Jacobian.** Let $\mathbf{s} = \operatorname{softmax}(\mathbf{e})$. The Jacobian is:

$$\frac{\partial s_i}{\partial e_j} = s_i(\delta_{ij} - s_j) = \begin{cases} s_i(1 - s_i) & i = j \\ -s_i s_j & i \ne j \end{cases}$$

In matrix form: $\frac{\partial \mathbf{s}}{\partial \mathbf{e}} = \operatorname{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$. This Jacobian is rank $n-1$ (the constraint $\sum s_i = 1$ removes one degree of freedom). When one $s_i \approx 1$ (saturation), the diagonal entries are $\approx 0$ — gradients vanish. This is why the $1/\sqrt{d_k}$ scaling is essential: it prevents saturation.

**Example: attention computation step-by-step.** Consider $n = 3$, $d_k = 2$:

$$Q = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix}, \quad K = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 0.5 & 0.5 \end{pmatrix}, \quad V = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}$$

Step 1: $S = QK^\top / \sqrt{2} = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 0 & 0.5 \\ 0 & 1 & 0.5 \\ 1 & 1 & 1 \end{pmatrix}$

Step 2: Softmax row-wise gives attention weights.

Step 3: $O = AV$ produces a weighted average of values per row. Row 1 attends primarily to position 1 (where its query direction matches key 1), Row 2 to position 2, and Row 3 attends broadly (its query matches all keys).

### 2.5 Attention as Kernel Smoothing

There is a deep connection between attention and classical nonparametric statistics. The **Nadaraya–Watson kernel regression estimator** is:

$$\hat{f}(\mathbf{x}) = \frac{\sum_{j=1}^n K(\mathbf{x}, \mathbf{x}_j) y_j}{\sum_{l=1}^n K(\mathbf{x}, \mathbf{x}_l)}$$

where $K$ is a kernel function. Setting $K(\mathbf{q}, \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d_k})$ (the softmax kernel), $\mathbf{x}_j = \mathbf{k}_j$, and $y_j = \mathbf{v}_j$, we recover exactly scaled dot-product attention.

This means attention is a **learned kernel smoother**: queries define the points at which we evaluate, keys define the data points, values define the function values, and the softmax kernel controls the bandwidth. The key difference from classical kernel regression is that the kernel is parameterised (through the learned projection matrices $W_Q, W_K$) rather than fixed.

**For AI:** This perspective explains why attention heads can learn different "kernels" — some heads learn narrow kernels (attending to the immediately previous token), while others learn broad kernels (attending to semantically similar tokens regardless of distance).

**Other kernel choices.** The softmax kernel is not the only option:

| Kernel | Formula | Properties |
| --- | --- | --- |
| Softmax (standard) | $k(\mathbf{q}, \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})$ | Sharp; non-decomposable; $O(n^2)$ |
| RBF (Gaussian) | $k(\mathbf{q}, \mathbf{k}) = \exp(-\lVert \mathbf{q} - \mathbf{k} \rVert^2 / 2)$ | Translation-invariant; equivalent to softmax up to normalisation |
| Polynomial | $k(\mathbf{q}, \mathbf{k}) = (\mathbf{q}^\top \mathbf{k})^p$ | Decomposable for fixed $p$; linear attention variant |
| Random features (Performers) | $k(\mathbf{q}, \mathbf{k}) \approx \phi(\mathbf{q})^\top \phi(\mathbf{k})$ | Approximates softmax; $O(nd)$; quality gap |

The softmax kernel's non-decomposability is precisely what creates the $O(n^2)$ cost — and also what makes attention so effective at sharp, selective retrieval.

### 2.6 The Attention Matrix

The attention matrix $A = \operatorname{softmax}(QK^\top / \sqrt{d_k}) \in \mathbb{R}^{n \times n}$ has several important structural properties:

**Row-stochastic:** Each row sums to 1 (it is a probability distribution). $A$ is not column-stochastic in general.

**Non-negative:** All entries $A_{ij} \ge 0$ by definition of softmax.

**Rank:** For generic $Q, K$, the pre-softmax matrix $S = QK^\top / \sqrt{d_k}$ has rank $\min(n, d_k)$. Since $d_k = d_{\text{model}}/h$ is typically 64–128 while sequence lengths can be 2K–128K, the score matrix is **low-rank** ($d_k \ll n$). The softmax applies an element-wise nonlinearity that can increase the rank, but empirically the attention matrix remains approximately low-rank — a fact exploited by efficient attention methods.

**Sparsity:** In practice, trained attention matrices are often approximately sparse: most of the probability mass concentrates on a small number of positions. This observation motivates sparse attention patterns (Longformer, BigBird) that only compute a subset of the $n^2$ entries.

**Spectral properties.** The singular values of the attention matrix reveal its effective rank. For a random softmax attention matrix (random $Q, K$), the spectrum decays rapidly: the top singular value captures a large fraction of the Frobenius norm. This low effective rank means attention can be well-approximated by low-rank matrices — the theoretical basis for Linformer and other efficient methods.

**Doubly stochastic attention.** Sinkhorn attention (Tay et al., 2020) adds column normalisation so that $A$ is doubly stochastic ($A\mathbf{1} = \mathbf{1}$ and $A^\top \mathbf{1} = \mathbf{1}$). This ensures every key position receives roughly equal total attention, preventing "attention sink" phenomena where certain positions (e.g., the first token) absorb disproportionate attention weight.

**Attention sink phenomenon.** Xiao et al. (2024) observed that in autoregressive LLMs, the first few tokens receive extremely high attention weights regardless of content. These positions act as "attention sinks" — the model uses them as a default when no other position is clearly relevant. This has practical implications for KV cache management: the initial tokens must always be kept in cache even during streaming.

```
ATTENTION MATRIX STRUCTURE
════════════════════════════════════════════════════════════════════════

  Full attention (n × n):    Causal mask:          Sparse (local + global):
  ┌─────────────┐            ┌─────────────┐       ┌─────────────┐
  │ ■ ■ ■ ■ ■ ■│            │ ■ · · · · ·│       │ ■ ■ · · · ■│
  │ ■ ■ ■ ■ ■ ■│            │ ■ ■ · · · ·│       │ ■ ■ ■ · · ■│
  │ ■ ■ ■ ■ ■ ■│            │ ■ ■ ■ · · ·│       │ · ■ ■ ■ · ■│
  │ ■ ■ ■ ■ ■ ■│            │ ■ ■ ■ ■ · ·│       │ · · ■ ■ ■ ■│
  │ ■ ■ ■ ■ ■ ■│            │ ■ ■ ■ ■ ■ ·│       │ · · · ■ ■ ■│
  │ ■ ■ ■ ■ ■ ■│            │ ■ ■ ■ ■ ■ ■│       │ ■ ■ ■ ■ ■ ■│
  └─────────────┘            └─────────────┘       └─────────────┘
  O(n²) compute              Lower triangular      O(n·w) compute

════════════════════════════════════════════════════════════════════════
```

---

## 3. Multi-Head Attention

### 3.1 Single-Head Formulation

In practice, queries, keys, and values are not the raw input vectors. They are **learned linear projections** of the input. Given input $X \in \mathbb{R}^{n \times d_{\text{model}}}$:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$. The projection maps each token from the $d_{\text{model}}$-dimensional representation space into a $d_k$-dimensional query/key space and a $d_v$-dimensional value space.

The complete single-head attention is:

$$\operatorname{head}(X) = \operatorname{softmax}\!\left(\frac{(XW_Q)(XW_K)^\top}{\sqrt{d_k}}\right)(XW_V) \in \mathbb{R}^{n \times d_v}$$

**Parameter count:** $W_Q$ has $d_{\text{model}} \cdot d_k$ parameters, similarly for $W_K$ and $W_V$. Total: $d_{\text{model}}(2d_k + d_v)$ parameters.

### 3.2 Why Multiple Heads

**Rank limitation of single-head attention.** The attention matrix $A = \operatorname{softmax}(QK^\top/\sqrt{d_k})$ is derived from a score matrix $S = XW_Q W_K^\top X^\top / \sqrt{d_k}$. The matrix $W_Q W_K^\top \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ has rank at most $d_k$. This means a single attention head can only express score functions that live in a $d_k$-dimensional subspace of the $d_{\text{model}}$-dimensional representation.

Multi-head attention addresses this by running $h$ independent attention operations in parallel, each with its own projection matrices:

$$\operatorname{head}_i = \operatorname{softmax}\!\left(\frac{(XW_i^Q)(XW_i^K)^\top}{\sqrt{d_k}}\right)(XW_i^V)$$

Each head can attend to different aspects: Head 1 might focus on syntactic relations ("subject → verb"), Head 2 on coreference ("pronoun → antecedent"), Head 3 on positional patterns ("attend to previous token").

**Subspace interpretation (Bhojanapalli et al., 2020):** With $h$ heads and $d_k = d_{\text{model}} / h$, the $h$ heads collectively span a rank-$d_{\text{model}}$ subspace. Multi-head attention decomposes the full attention computation into $h$ independent subspace operations, which is both more expressive and more parameter-efficient than a single head with $d_k = d_{\text{model}}$.

### 3.3 Output Projection

The $h$ head outputs are concatenated and projected:

$$\operatorname{MultiHead}(X) = \operatorname{Concat}(\operatorname{head}_1, \ldots, \operatorname{head}_h) W_O$$

where $W_O \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ is the output projection. Since $hd_v = d_{\text{model}}$ (by the standard choice $d_v = d_{\text{model}} / h$), the output has the same dimension as the input — essential for the residual connection.

**Total parameter count for multi-head attention:**

$$\text{params} = h(d_{\text{model}} \cdot d_k + d_{\text{model}} \cdot d_k + d_{\text{model}} \cdot d_v) + hd_v \cdot d_{\text{model}} = 4 d_{\text{model}}^2$$

(using $d_k = d_v = d_{\text{model}} / h$). This is independent of the number of heads — a remarkable fact. Whether you use 8 heads or 64 heads with the same $d_{\text{model}}$, the parameter count is the same.

### 3.4 Causal Masking

For autoregressive language models (GPT, LLaMA), token $i$ must not attend to future tokens $j > i$. This is enforced by adding $-\infty$ to the score matrix before softmax:

$$\text{mask}_{ij} = \begin{cases} 0 & \text{if } i \ge j \\ -\infty & \text{if } i < j \end{cases}$$

$$A = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \text{mask}\right)$$

Since $\exp(-\infty) = 0$, the masked positions contribute zero weight. The resulting attention matrix is **lower-triangular**: position $i$ can only attend to positions $1, \ldots, i$.

**For AI:** Causal masking is what makes autoregressive generation possible. During training, all positions are computed in parallel (teacher forcing), but the mask ensures each position only sees its past — exactly matching the generation-time constraint.

**Prefix attention (prefix LMs).** A variant used in T5 and some instruction-tuned models: the first $p$ positions (the "prefix" or prompt) use bidirectional attention, while positions $p+1, \ldots, n$ (the "generation" portion) use causal attention. This allows the model to fully process the prompt bidirectionally while generating autoregressively.

**Sliding window attention (Mistral).** Instead of attending to all previous positions, each token attends only to the most recent $w$ positions: $\text{mask}_{ij} = 0$ iff $i - w < j \le i$. With $L$ layers, information can propagate $L \cdot w$ positions through the residual stream, giving an effective context of $L \cdot w$ even though each layer's attention window is only $w$. For Mistral-7B with $w = 4096$ and $L = 32$: effective context is $32 \times 4096 = 131{,}072$ positions.

**Example: multi-head with $d_{\text{model}} = 8$, $h = 2$.** Each head gets $d_k = 4$. Head 1 uses $W_1^Q \in \mathbb{R}^{8 \times 4}$ to project queries into a 4D subspace focused on (say) syntactic relations. Head 2 uses $W_2^Q \in \mathbb{R}^{8 \times 4}$ for a different 4D subspace focused on semantic similarity. The two heads capture different types of dependencies, and the output projection $W_O$ combines them back into the full 8D space.

**Attention score decomposition.** The full score matrix can be written as a sum over heads:

$$S_{\text{full}} = \sum_{i=1}^{h} X W_i^Q (W_i^K)^\top X^\top / \sqrt{d_k}$$

Each head contributes a rank-$d_k$ matrix $W_i^Q (W_i^K)^\top$. The full attention is a sum of $h$ rank-$d_k$ components, giving total rank up to $h \cdot d_k = d_{\text{model}}$.

### 3.5 Cross-Attention

In encoder–decoder models (T5, Whisper), the decoder attends to the encoder's output. Here, queries come from the decoder and keys/values come from the encoder:

$$Q = X_{\text{dec}} W_Q, \quad K = X_{\text{enc}} W_K, \quad V = X_{\text{enc}} W_V$$

The attention matrix is $n_{\text{dec}} \times n_{\text{enc}}$ (not square). Cross-attention is the mechanism that allows the decoder to "read" the encoder's representation of the input.

### 3.6 GQA and MQA

**Multi-Query Attention (MQA, Shazeer 2019):** All heads share a single set of keys and values, but each head has its own queries. This reduces the KV cache by a factor of $h$ during inference.

$$K = XW_K, \quad V = XW_V \qquad \text{(shared across all heads)}$$

**Grouped-Query Attention (GQA, Ainslie et al. 2023):** A middle ground — $h$ query heads are divided into $g$ groups, and each group shares one set of K/V. LLaMA-2 70B uses GQA with 8 KV heads and 64 query heads ($g = 8$).

```
ATTENTION HEAD SHARING
════════════════════════════════════════════════════════════════════════

  MHA (standard):      GQA (grouped):        MQA (single KV):
  Q₁ K₁ V₁             Q₁ ─┐                 Q₁ ─┐
  Q₂ K₂ V₂             Q₂ ─┤ K₁ V₁           Q₂ ─┤
  Q₃ K₃ V₃             Q₃ ─┤                 Q₃ ─┤
  Q₄ K₄ V₄             Q₄ ─┘                 Q₄ ─┤ K₁ V₁
  Q₅ K₅ V₅             Q₅ ─┐                 Q₅ ─┤
  Q₆ K₆ V₆             Q₆ ─┤ K₂ V₂           Q₆ ─┤
  Q₇ K₇ V₇             Q₇ ─┤                 Q₇ ─┤
  Q₈ K₈ V₈             Q₈ ─┘                 Q₈ ─┘

  8 KV heads            2 KV groups            1 KV head
  KV cache: 8×          KV cache: 2×           KV cache: 1×

════════════════════════════════════════════════════════════════════════
```

---

## 4. Positional Encoding Theory

### 4.1 Permutation Equivariance Proof

**Theorem.** Self-attention (without positional encodings) is permutation-equivariant: if we permute the input sequence, the output is permuted in the same way.

*Proof.* Let $\Pi \in \{0,1\}^{n \times n}$ be a permutation matrix. The permuted input is $\tilde{X} = \Pi X$. Then:

$$\tilde{Q}\tilde{K}^\top = (\Pi X W_Q)(\Pi X W_K)^\top = \Pi (X W_Q)(X W_K)^\top \Pi^\top = \Pi S \Pi^\top$$

Softmax is applied row-wise, and $\Pi S \Pi^\top$ simply reorders both rows and columns. The output becomes:

$$\operatorname{softmax}(\Pi S \Pi^\top)(\Pi V) = \Pi \operatorname{softmax}(S) V = \Pi \cdot \operatorname{Attn}(X)$$

The output is permuted by the same $\Pi$. $\square$

**Consequence:** Without positional information, a Transformer treats "The cat sat on the mat" identically to "mat the on sat cat The". Position must be explicitly injected.

### 4.2 Sinusoidal Encoding

Vaswani et al. (2017) proposed adding a deterministic signal based on sine and cosine functions at different frequencies:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the position and $i$ is the dimension index. Each dimension has a different wavelength, forming a geometric progression from $2\pi$ to $10000 \cdot 2\pi$.

**Key property:** For any fixed offset $k$, there exists a linear transformation $M_k$ such that $PE_{pos+k} = M_k \cdot PE_{pos}$. This is because:

$$\begin{pmatrix} \sin(\omega(pos + k)) \\ \cos(\omega(pos + k)) \end{pmatrix} = \begin{pmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{pmatrix} \begin{pmatrix} \sin(\omega \cdot pos) \\ \cos(\omega \cdot pos) \end{pmatrix}$$

This rotation matrix structure means that relative positions can be represented as linear transformations of absolute positions — the model can learn to attend to "3 positions back" using a fixed linear map.

### 4.3 Rotary Position Embedding (RoPE)

RoPE (Su et al., 2021) injects position directly into the attention score computation rather than adding it to the input. The key idea: apply a position-dependent **rotation** to query and key vectors before computing the dot product.

For a pair of dimensions $(2i, 2i+1)$, define:

$$R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}, \qquad \theta_i = 10000^{-2i/d}$$

RoPE applies $R_m$ to the query at position $m$ and $R_n$ to the key at position $n$:

$$\mathbf{q}_m' = R_m \mathbf{q}_m, \qquad \mathbf{k}_n' = R_n \mathbf{k}_n$$

The attention score becomes:

$$(\mathbf{q}_m')^\top \mathbf{k}_n' = \mathbf{q}_m^\top R_m^\top R_n \mathbf{k}_n = \mathbf{q}_m^\top R_{n-m} \mathbf{k}_n$$

Since rotation matrices satisfy $R_m^\top R_n = R_{n-m}$, the score depends only on the **relative position** $n - m$, not on the absolute positions. This is the mathematical elegance of RoPE: relative position bias emerges automatically from rotation group structure.

**For AI:** RoPE is used in LLaMA, GPT-NeoX, PaLM, and most modern open-source LLMs. Its superiority over sinusoidal encodings comes from encoding position in the attention score directly, rather than contaminating the residual stream with additive position signals.

**RoPE in complex notation.** An elegant way to express RoPE uses complex numbers. Pair the dimensions and treat $(x_{2i}, x_{2i+1})$ as a complex number $z_i = x_{2i} + i \cdot x_{2i+1}$. The rotation becomes:

$$\tilde{z}_i^{(m)} = z_i \cdot e^{im\theta_i}$$

Multiplication by $e^{im\theta_i}$ rotates the complex number by angle $m\theta_i$. The dot product of two rotated vectors:

$$\operatorname{Re}\left(\sum_i \tilde{z}_i^{(m)} \overline{\tilde{z}_i^{(n)}}\right) = \operatorname{Re}\left(\sum_i z_i \bar{z}_i' e^{i(m-n)\theta_i}\right)$$

depends only on $m - n$. This complex formulation makes the group-theoretic structure transparent: RoPE applies an element of the torus group $\mathbb{T}^{d/2} = (S^1)^{d/2}$ to each query/key, and the relative position is the group element $e^{i(m-n)\theta}$.

**Frequency base and NTK-aware scaling.** The frequencies $\theta_i = 10000^{-2i/d}$ span 5 orders of magnitude. Low frequencies ($i$ near 0) encode coarse position (nearby vs distant tokens), while high frequencies ($i$ near $d/2$) encode fine position (exact offset). For context extension beyond training length, **NTK-aware RoPE** (bloc97, 2023) modifies the frequency base:

$$\theta_i' = \left(\frac{L_{\text{target}}}{L_{\text{train}}} \cdot 10000\right)^{-2i/d}$$

This stretches the low frequencies (which need to cover more positions) while leaving high frequencies unchanged (relative to the training distribution). YaRN further combines this with attention temperature scaling and fine-tuning on a small amount of long-context data.

### 4.4 ALiBi

Attention with Linear Biases (Press et al., 2022) takes the simplest possible approach: add a fixed, non-learned bias to the attention scores that decays linearly with distance:

$$\text{score}(i, j) = \mathbf{q}_i^\top \mathbf{k}_j - m \cdot |i - j|$$

where $m$ is a head-specific slope. Different heads use different slopes (geometric sequence: $m \in \{2^{-8/h}, 2^{-16/h}, \ldots, 2^{-8}\}$ for $h$ heads), so some heads attend locally and others attend more broadly. ALiBi requires no learned parameters for position.

**Mathematical properties of ALiBi:**

- The bias is equivalent to an exponential decay kernel: $\exp(-m|i-j|) \propto$ the attention weight contribution from position $j$ at query $i$ (before combining with the content-based score $\mathbf{q}^\top \mathbf{k}$).
- The linear decay means ALiBi attention weights decrease geometrically with distance, implementing a learnable locality bias per head.
- Since the bias is additive and independent of content, it can be precomputed and cached — zero additional inference cost.
- For any unseen distance $|i - j| > L_{\text{train}}$, the bias extrapolates linearly, providing a principled (if aggressive) decay.

### 4.5 Length Generalisation

A critical question: can a model trained on sequences of length $L_{\text{train}}$ generalise to length $L_{\text{test}} > L_{\text{train}}$?

- **Sinusoidal:** Poor extrapolation. The model has never seen position indices beyond $L_{\text{train}}$.
- **RoPE:** Moderate extrapolation with position interpolation (Chen et al., 2023). NTK-aware scaling extends context further by modifying the frequency base.
- **ALiBi:** Good extrapolation by design — the linear bias naturally extends to unseen distances.

**For AI:** Length generalisation is critical for LLMs that must handle variable-length documents. The YaRN (Yet another RoPE extensioN) method scales the RoPE frequency base to extend LLaMA from 4K to 128K context with minimal fine-tuning.

**Positional encoding desiderata.** An ideal positional encoding should satisfy:

1. **Unique representation:** Different positions get different encodings.
2. **Bounded magnitude:** Encodings should not grow with sequence length (otherwise they dominate the residual stream).
3. **Relative position sensitivity:** The model should be able to determine the distance $|i - j|$ between two positions.
4. **Length generalisation:** The encoding should work for sequences longer than those seen during training.
5. **Efficiency:** Minimal additional parameters and compute.

| Method | Unique | Bounded | Relative | Generalisation | Params |
| --- | --- | --- | --- | --- | --- |
| Sinusoidal | Yes | Yes | Yes (linear) | Poor | 0 |
| Learned absolute | Yes | Trained | No | None | $n \cdot d$ |
| RoPE | Yes | Yes | Yes (rotation) | Moderate → good with scaling | 0 |
| ALiBi | N/A (bias) | Yes | Yes (linear decay) | Good | 0 |

No method perfectly satisfies all desiderata, which is why positional encoding remains an active research area.

**Context length evolution (enabled by positional encoding advances):**

| Year | Model | Context length | Positional encoding |
| --- | --- | --- | --- |
| 2017 | Transformer | 512 | Sinusoidal |
| 2018 | GPT-1 | 512 | Learned absolute |
| 2019 | GPT-2 | 1024 | Learned absolute |
| 2020 | GPT-3 | 2048 | Learned absolute |
| 2023 | LLaMA-1 | 2048 | RoPE |
| 2023 | LLaMA-2 | 4096 | RoPE |
| 2023 | Mistral | 8192 (32K effective) | RoPE + sliding window |
| 2024 | LLaMA-3 | 8192 → 128K | RoPE + NTK-aware scaling |
| 2024 | Claude-3 | 200K | Proprietary |
| 2025 | Gemini-1.5 | 1M+ | Proprietary |

The 2000× increase in context length (512 → 1M) over 8 years was enabled by three advances: (1) RoPE replacing learned absolute positions, (2) FlashAttention making long sequences computationally feasible, (3) NTK-aware and YaRN scaling extending RoPE beyond training length.

---

## 5. Feed-Forward Networks as Key–Value Memories

### 5.1 Position-Wise FFN

After the attention sublayer, each token's representation passes independently through a two-layer fully-connected network. For input $\mathbf{x} \in \mathbb{R}^{d_{\text{model}}}$:

$$\operatorname{FFN}(\mathbf{x}) = W_2 \,\sigma(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

where $W_1 \in \mathbb{R}^{d_{ff} \times d_{\text{model}}}$, $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{ff}}$, and $\sigma$ is a nonlinear activation. The term "position-wise" means the same parameters are applied identically at every position — there is no cross-position interaction in the FFN. This is a critical design choice: attention handles inter-position communication, and the FFN handles per-position computation.

**Parameter count:** $2 d_{\text{model}} \cdot d_{ff} + d_{ff} + d_{\text{model}}$. For the standard expansion ratio $d_{ff} = 4 d_{\text{model}}$, this gives $8 d_{\text{model}}^2$ parameters — **twice** the parameter count of the attention sublayer. The FFN is the majority of parameters in a Transformer.

### 5.2 Expansion Ratio and Capacity

Why expand to $d_{ff} = 4 d_{\text{model}}$ and then compress back? The intermediate dimension $d_{ff}$ controls the **capacity** of the per-position computation. In the expanded space, the network can represent more complex functions before projecting back to $d_{\text{model}}$.

Consider the first layer $\mathbf{h} = \sigma(W_1 \mathbf{x})$ with ReLU activation. Each row $\mathbf{w}_i$ of $W_1$ defines a hyperplane in $\mathbb{R}^{d_{\text{model}}}$. The ReLU activation partitions the input space into $2^{d_{ff}}$ linear regions (in the generic case). With $d_{ff} = 4 d_{\text{model}}$, the network has exponentially more linear regions than with $d_{ff} = d_{\text{model}}$.

**For AI:** The expansion ratio is a key hyperparameter. GPT-3 uses $4\times$, while LLaMA uses $\frac{8}{3}\times$ with SwiGLU (which has a third weight matrix, so the total parameter count matches the $4\times$ ReLU variant).

### 5.3 SwiGLU

The original Transformer used ReLU activation. Modern architectures (LLaMA, PaLM, Mistral) use **SwiGLU** (Shazeer, 2020), a gated variant:

$$\operatorname{SwiGLU}(\mathbf{x}) = (\operatorname{Swish}(W_1 \mathbf{x}) \odot (W_3 \mathbf{x})) W_2$$

where $\operatorname{Swish}(z) = z \cdot \sigma(z)$ (with $\sigma$ the sigmoid function) and $\odot$ is element-wise multiplication. The **gate** $W_3 \mathbf{x}$ modulates the activation — this is the "GLU" (Gated Linear Unit) component.

The SwiGLU has three weight matrices $(W_1, W_2, W_3)$ instead of two. To match the parameter count of the standard $4\times$ FFN, the intermediate dimension is reduced to $d_{ff} = \frac{8}{3} d_{\text{model}} \approx 2.67 d_{\text{model}}$. This gives:

$$\text{params} = 3 \cdot d_{\text{model}} \cdot d_{ff} = 3 \cdot d_{\text{model}} \cdot \frac{8}{3} d_{\text{model}} = 8 d_{\text{model}}^2$$

identical to the standard FFN.

**Why SwiGLU works better:** The gating mechanism allows the network to selectively suppress or amplify activations. The Swish function is smooth (unlike ReLU) and has non-zero gradients for negative inputs, which improves optimisation. Empirically, SwiGLU gives a consistent 1–3% improvement on language modelling benchmarks across model scales (Shazeer, 2020).

### 5.4 FFN as Associative Memory

Geva et al. (2021, "Transformer Feed-Forward Layers Are Key-Value Memories") showed that the FFN can be interpreted as a memory lookup. Decompose the operation:

$$\operatorname{FFN}(\mathbf{x}) = \sum_{i=1}^{d_{ff}} \sigma(\mathbf{k}_i^\top \mathbf{x}) \cdot \mathbf{v}_i$$

where $\mathbf{k}_i$ is the $i$-th row of $W_1$ (a "key" pattern) and $\mathbf{v}_i$ is the $i$-th column of $W_2$ (a "value" vector). The activation $\sigma(\mathbf{k}_i^\top \mathbf{x})$ measures how well the input matches key $i$, and the output is a weighted sum of the corresponding values.

This is structurally identical to attention — but with fixed, **learned** keys stored in $W_1$ rather than input-dependent keys. The FFN is a lookup table of $d_{ff}$ (key, value) pairs, learned during training:

- **Keys** (rows of $W_1$): patterns the network has learned to detect (e.g., "input represents a capital city", "input is a number")
- **Activations** $\sigma(\mathbf{k}_i^\top \mathbf{x})$: how strongly each pattern matches the current input
- **Values** (columns of $W_2$): the information to write into the residual stream when a pattern is detected

**For AI:** This interpretation explains why scaling $d_{ff}$ improves factual knowledge: more memory slots = more facts. It also explains why LoRA applied to $W_1, W_2$ can efficiently update stored knowledge — you are modifying the key–value lookup table.

```text
FFN AS KEY-VALUE MEMORY
════════════════════════════════════════════════════════════════════════

  Input x ∈ R^d_model

  ┌─────────────────────────────────────────────────────────────────┐
  │  W₁ (keys): d_ff rows, each a pattern detector                │
  │                                                                │
  │  k₁ᵀx ──▶ σ(k₁ᵀx) ──┐                                       │
  │  k₂ᵀx ──▶ σ(k₂ᵀx) ──┤                                       │
  │  k₃ᵀx ──▶ σ(k₃ᵀx) ──┼──▶ Σ σ(kᵢᵀx) · vᵢ ──▶ output       │
  │    ⋮           ⋮      │                                       │
  │  kₘᵀx ──▶ σ(kₘᵀx) ──┘                                       │
  │                                                                │
  │  W₂ (values): d_ff columns, each a memory value               │
  └─────────────────────────────────────────────────────────────────┘

  Compare with attention:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Attention:  Σ softmax(qᵀkⱼ) · vⱼ    (keys from input)       │
  │  FFN:        Σ σ(kᵢᵀx) · vᵢ           (keys from parameters)  │
  └─────────────────────────────────────────────────────────────────┘

════════════════════════════════════════════════════════════════════════
```

**Activation function comparison for FFN:**

| Activation | Formula | Derivative | Properties |
| --- | --- | --- | --- |
| ReLU | $\max(0, z)$ | $\mathbb{1}[z > 0]$ | Sparse; dead neurons; fast |
| GELU | $z \Phi(z)$ | $\Phi(z) + z\phi(z)$ | Smooth; used in BERT, GPT-2 |
| Swish | $z \cdot \sigma(\beta z)$ | $\sigma(\beta z)(1 + \beta z(1 - \sigma(\beta z)))$ | Smooth; $\beta = 1$ for SiLU |
| SwiGLU | $\operatorname{Swish}(W_1 \mathbf{x}) \odot W_3 \mathbf{x}$ | Product rule + Swish derivative | Gated; best empirical performance |

**Non-examples (what the FFN is NOT):**

1. **Not a classifier:** The FFN does not produce a probability distribution. It maps $\mathbb{R}^{d_{\text{model}}} \to \mathbb{R}^{d_{\text{model}}}$ — same input and output dimensions.
2. **Not cross-position:** Unlike convolutions, the FFN applies independently at each position. Position $i$'s FFN output depends only on position $i$'s input.
3. **Not a single linear layer:** The nonlinearity $\sigma$ is essential. Without it, $W_2 W_1 \mathbf{x}$ is a rank-$d_{ff}$ linear map — the expansion to $d_{ff}$ would be pointless since $W_2 W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$.

---

## 6. Normalisation and Residual Connections

### 6.1 Residual Stream

The Transformer uses **residual connections** (He et al., 2016) around both the attention and FFN sublayers:

$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \operatorname{Sublayer}(\operatorname{Norm}(\mathbf{x}^{(l)}))$$

The residual connection creates a "highway" through the network: each layer adds a small update $\Delta\mathbf{x}^{(l)}$ to the running representation, rather than completely transforming it. After $L$ layers:

$$\mathbf{x}^{(L)} = \mathbf{x}^{(0)} + \sum_{l=0}^{L-1} \Delta\mathbf{x}^{(l)}$$

This is an additive decomposition: the final representation is the input embedding plus the sum of all layer contributions. The **residual stream hypothesis** (Elhage et al., 2021) says that the residual stream is the Transformer's primary "communication bus" — layers read from it and write to it.

**Gradient flow:** The residual connection ensures:

$$\frac{\partial \mathbf{x}^{(L)}}{\partial \mathbf{x}^{(0)}} = I + \sum_{\text{paths}} \prod_{l \in \text{path}} \frac{\partial \operatorname{Sublayer}^{(l)}}{\partial \mathbf{x}^{(l)}}$$

The identity matrix $I$ guarantees that gradients can flow unchanged from the loss to early layers, avoiding the vanishing gradient problem that plagues deep networks without residuals.

**Quantitative analysis.** For a network without residual connections, the gradient norm decays as $\lVert \partial \mathcal{L} / \partial \mathbf{x}^{(0)} \rVert \sim \prod_{l=1}^L \lVert J^{(l)} \rVert$ where $J^{(l)}$ is the Jacobian of layer $l$. If $\lVert J^{(l)} \rVert < 1$ for most layers, this product vanishes exponentially.

With residual connections, the gradient decomposes into $2^L$ paths (each layer's residual can be included or excluded). The shortest path (through all skip connections) has gradient exactly $I$ — no decay. The longer paths may decay, but the shortest path guarantees non-vanishing gradients. This is why Transformers can be trained with 100+ layers while vanilla networks fail beyond ~20 layers.

**Residual stream as a finite-dimensional communication bus.** The residual stream has exactly $d_{\text{model}}$ dimensions. With $L$ layers, each containing $h$ attention heads and 1 FFN, there are $L(h + 1)$ writers competing for $d_{\text{model}}$ dimensions. For LLaMA-7B: $32 \times (32 + 1) = 1056$ writers sharing $d_{\text{model}} = 4096$ dimensions. This is possible only because each writer uses a low-rank projection ($d_v = d/h = 128$ dimensions per head) and because the superposition phenomenon allows packing more features than dimensions.

**Writing and reading from the residual stream.** Each attention head writes to the residual stream through its output projection: $\Delta \mathbf{x} = W_O \mathbf{a}$ where $\mathbf{a}$ is the attention output. The rank of $W_O$ is $d_v = d_{\text{model}} / h$, so each head can only write in a $d_v$-dimensional subspace. With $h$ heads, the $h$ subspaces collectively span $d_{\text{model}}$ dimensions (assuming the heads learn orthogonal subspaces, which is approximately true in practice).

Similarly, each head reads from the residual stream through its query and key projections. The QK circuit $W_Q W_K^\top$ determines which features in the residual stream are used for computing attention patterns, while the OV circuit $W_O W_V$ determines what information is written based on what is read.

### 6.2 Layer Norm

**Layer Normalisation** (Ba et al., 2016) normalises across the feature dimension for each token independently:

$$\operatorname{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ and $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$ are the mean and variance computed across the $d$ features, and $\gamma, \beta \in \mathbb{R}^d$ are learned affine parameters.

**Why not BatchNorm?** Batch Normalisation computes statistics across the batch dimension. For sequence models: (a) batch sizes are often small due to memory constraints, making batch statistics noisy; (b) sequences have variable length, making it unclear how to normalise across the batch at each position; (c) at inference time, the model processes one sequence at a time.

**Jacobian of LayerNorm:** Let $\mathbf{y} = \operatorname{LayerNorm}(\mathbf{x})$. The Jacobian $\partial \mathbf{y} / \partial \mathbf{x}$ is:

$$\frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \left(I - \frac{1}{d}\mathbf{1}\mathbf{1}^\top - \frac{1}{d}\hat{\mathbf{x}}\hat{\mathbf{x}}^\top\right)$$

where $\hat{\mathbf{x}} = (\mathbf{x} - \mu) / \sqrt{\sigma^2 + \epsilon}$ is the normalised vector. This Jacobian projects out the mean direction and the current normalised direction — it constrains gradients to a $(d-2)$-dimensional subspace.

### 6.3 RMSNorm

**Root Mean Square Normalisation** (Zhang & Sennrich, 2019) drops the mean-centering step:

$$\operatorname{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}$$

This has only $d$ learned parameters ($\gamma$) instead of $2d$ ($\gamma, \beta$), and requires one fewer reduction operation (no mean computation).

**Why RMSNorm replaced LayerNorm:** The mean-centering in LayerNorm is redundant when combined with the bias terms in surrounding linear layers. Removing it: (a) reduces computation by ~7–15% for the normalisation step, (b) is more numerically stable (one fewer subtraction), and (c) empirically performs equally well or better.

**For AI:** LLaMA, Mistral, GPT-NeoX, PaLM-2, and most 2023+ open-source LLMs use RMSNorm exclusively. The simplicity of RMSNorm also makes it easier to implement in custom CUDA kernels and to fuse with adjacent operations.

### 6.4 Pre-Norm vs Post-Norm

The original Transformer (Vaswani et al., 2017) placed normalisation **after** the sublayer:

$$\mathbf{x}^{(l+1)} = \operatorname{Norm}(\mathbf{x}^{(l)} + \operatorname{Sublayer}(\mathbf{x}^{(l)})) \qquad \text{(Post-Norm)}$$

Modern Transformers place normalisation **before** the sublayer:

$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \operatorname{Sublayer}(\operatorname{Norm}(\mathbf{x}^{(l)})) \qquad \text{(Pre-Norm)}$$

**Signal propagation analysis.** In Post-Norm, the normalisation is applied to the sum $\mathbf{x}^{(l)} + \operatorname{Sublayer}(\mathbf{x}^{(l)})$, which means it also normalises the residual. This can destabilise training for deep networks because the residual stream magnitude is repeatedly rescaled.

In Pre-Norm, the residual stream flows unmodified through the skip connection. The normalisation only affects the sublayer input, leaving the residual stream magnitude to grow naturally. This makes Pre-Norm significantly easier to train at depth — Xiong et al. (2020) proved that Pre-Norm eliminates the need for learning rate warmup.

**The trade-off:** Post-Norm often achieves slightly better final performance when it can be trained stably (the normalisation after the residual acts as a regulariser). Pre-Norm is more robust and is the universal default for LLMs trained at scale.

### 6.5 DeepNorm

For very deep Transformers (100+ layers), even Pre-Norm can exhibit training instability. **DeepNorm** (Wang et al., 2022) modifies the residual connection with a depth-dependent scaling:

$$\mathbf{x}^{(l+1)} = \operatorname{Norm}(\alpha \cdot \mathbf{x}^{(l)} + \operatorname{Sublayer}(\mathbf{x}^{(l)}))$$

where $\alpha = (2L)^{1/4}$ for a network with $L$ layers, and the sublayer weights are initialised with scale $\beta = (8L)^{-1/4}$. This ensures that the residual stream magnitude stays bounded as $L \to \infty$.

**For AI:** DeepNorm enabled training of 1000-layer Transformers (Wang et al., 2022), though in practice frontier LLMs use 80–120 layers with Pre-Norm + RMSNorm.

**Normalisation comparison:**

| Method | Formula | Params | Compute | Used in |
| --- | --- | --- | --- | --- |
| LayerNorm | $\gamma \odot (\mathbf{x} - \mu) / \sqrt{\sigma^2 + \epsilon} + \beta$ | $2d$ | 2 reductions | Original Transformer, BERT |
| RMSNorm | $\gamma \odot \mathbf{x} / \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$ | $d$ | 1 reduction | LLaMA, Mistral, PaLM-2 |
| DeepNorm | $\operatorname{Norm}(\alpha \mathbf{x} + \text{Sublayer}(\mathbf{x}))$ | $d$ or $2d$ | +1 scalar mult | Very deep models (100+ layers) |

**Non-examples (what is NOT normalisation):**

1. **Batch Normalisation** normalises across the batch dimension, not the feature dimension. It requires batch statistics and behaves differently at train vs. inference time. Not used in Transformers.
2. **Weight Normalisation** reparameterises the weight matrix $W = g \cdot \hat{W} / \lVert \hat{W} \rVert$. This normalises the weights, not the activations.
3. **Instance Normalisation** normalises each channel of each sample independently. Used in style transfer CNNs, not in Transformers.

```text
SIGNAL PROPAGATION: PRE-NORM vs POST-NORM
════════════════════════════════════════════════════════════════════════

  Post-Norm (Vaswani 2017):         Pre-Norm (GPT-2+):

  x ──▶ Sublayer ──▶ (+) ──▶ Norm   x ──▶ Norm ──▶ Sublayer ──▶ (+)
        │              ▲                   │                     ▲
        └──────────────┘                   └─────────────────────┘
                                           (residual is UN-normalised)
  Problem: Norm rescales the          Benefit: Residual stream grows
  residual stream at every layer.     naturally. Gradients flow through
  Signal magnitude is bounded.        identity path without rescaling.
  Requires warmup for stability.      Stable without warmup.

  Post-Norm gradient path:           Pre-Norm gradient path:
  ∂L/∂x⁰ passes through L norms     ∂L/∂x⁰ = I + sublayer terms
  → magnitude controlled but         → gradient magnitude preserved
    information can be lost           → easier optimisation at depth

════════════════════════════════════════════════════════════════════════
```

---

## 7. Complete Transformer Block and Variants

### 7.1 The Full Block

A single Pre-Norm Transformer block combines attention, FFN, normalisation, and residuals:

$$\mathbf{z}^{(l)} = \mathbf{x}^{(l)} + \operatorname{MultiHead}(\operatorname{RMSNorm}(\mathbf{x}^{(l)}))$$
$$\mathbf{x}^{(l+1)} = \mathbf{z}^{(l)} + \operatorname{FFN}(\operatorname{RMSNorm}(\mathbf{z}^{(l)}))$$

This is the building block of every modern LLM. A complete model stacks $L$ such blocks, preceded by token + position embedding and followed by a final normalisation and language model head:

$$\text{logits} = \operatorname{LMHead}(\operatorname{RMSNorm}(\mathbf{x}^{(L)}))$$

where $\operatorname{LMHead}$ is a linear projection to vocabulary size $V$: $\text{logits} \in \mathbb{R}^{n \times V}$.

```
TRANSFORMER BLOCK (PRE-NORM, LLaMA STYLE)
════════════════════════════════════════════════════════════════════════

  Input x^(l) ─────────────────────────────────┐
       │                                        │ (residual)
       ▼                                        │
  ┌──────────┐                                  │
  │ RMSNorm  │                                  │
  └────┬─────┘                                  │
       ▼                                        │
  ┌──────────────────┐                          │
  │  Multi-Head      │  Q,K,V from norm'd x     │
  │  Attention       │  + causal mask + RoPE     │
  └────┬─────────────┘                          │
       │                                        │
       ▼                                        │
  z^(l) = x^(l) + Attn output  ◄───────────────┘
       │
       ├───────────────────────────────────┐
       │                                   │ (residual)
       ▼                                   │
  ┌──────────┐                             │
  │ RMSNorm  │                             │
  └────┬─────┘                             │
       ▼                                   │
  ┌──────────────────┐                     │
  │  SwiGLU FFN      │  W₁, W₂, W₃        │
  └────┬─────────────┘                     │
       │                                   │
       ▼                                   │
  x^(l+1) = z^(l) + FFN output  ◄─────────┘

════════════════════════════════════════════════════════════════════════
```

### 7.2 Decoder-Only (GPT, LLaMA)

The decoder-only architecture uses a single stack of blocks with **causal masking**. Each token can only attend to previous tokens and itself. This is the dominant architecture for language modelling (GPT-4, LLaMA-3, Claude, Mistral, Gemini).

**Why decoder-only won:** Encoder–decoder models require two stacks of layers and cross-attention, nearly doubling the parameters for the same compute. Decoder-only models with causal masking can be trained with simple next-token prediction on raw text — no special masking schemes or multiple objectives needed. The simplicity scales better.

**Mathematical reason for decoder-only dominance.** The training objective for a decoder-only model is:

$$\mathcal{L}(\theta) = -\frac{1}{T} \sum_{t=1}^{T} \log P_\theta(x_t \mid x_1, \ldots, x_{t-1})$$

This is simply the negative log-likelihood of the data under an autoregressive factorisation. By the chain rule of probability: $P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{<t})$. Minimising the loss is equivalent to maximum likelihood estimation of the joint distribution — the most natural and theoretically grounded training objective. No task-specific design decisions are needed.

**The LLaMA recipe (2023):** The specific combination that became the open-source standard:
- Pre-Norm with RMSNorm
- SwiGLU activation in FFN
- RoPE positional embeddings
- GQA (Grouped-Query Attention)
- No bias terms in linear layers

### 7.3 Encoder-Only (BERT)

BERT (Devlin et al., 2019) uses a stack of blocks with **bidirectional** attention (no causal mask). Every token attends to every other token. Trained with masked language modelling (predict 15% of randomly masked tokens) and next-sentence prediction.

**Mathematical difference:** Without causal masking, the attention matrix $A$ is a full $n \times n$ matrix. This gives each token strictly more information than the causal case, but makes autoregressive generation impossible — BERT cannot generate text left to right.

**For AI:** BERT-style models are used for encoding (sentence embeddings, classification, retrieval) but not generation. The encoder in modern retrieval-augmented generation (RAG) systems is often a BERT variant.

**Masked language modelling objective.** BERT's training corrupts 15% of tokens and predicts the original:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P_\theta(x_i \mid x_{\setminus \mathcal{M}})$$

where $\mathcal{M}$ is the set of masked positions and $x_{\setminus \mathcal{M}}$ is the sequence with masked positions replaced by `[MASK]`. Unlike autoregressive training, this objective does not factorise the joint distribution — it models conditional distributions $P(x_i \mid \text{context})$ independently. This is why BERT excels at understanding tasks (where you need the representation of a full sentence) but cannot generate text coherently.

### 7.4 Encoder–Decoder (T5, Whisper)

The encoder–decoder architecture has two stacks:
1. **Encoder:** bidirectional self-attention (like BERT)
2. **Decoder:** causal self-attention + cross-attention to encoder

At each decoder layer, the cross-attention sublayer computes queries from the decoder state and keys/values from the encoder output. T5 (Raffel et al., 2020) and Whisper (Radford et al., 2023) use this architecture.

**When encoder–decoder makes sense:** Tasks with distinct input and output (translation, speech-to-text, image captioning) benefit from the encoder processing the full input bidirectionally before the decoder generates the output autoregressively.

```text
TRANSFORMER ARCHITECTURAL VARIANTS
════════════════════════════════════════════════════════════════════════

  Encoder-Only (BERT):      Decoder-Only (GPT):      Encoder-Decoder (T5):
  ┌──────────────────┐      ┌──────────────────┐     ┌─────────┐ ┌─────────┐
  │ Bidirectional     │      │ Causal            │     │ Bidir.  │ │ Causal  │
  │ Self-Attention    │      │ Self-Attention    │     │ Self-   │ │ Self-   │
  │ (full n×n)        │      │ (lower-triangular)│     │ Attn.   │ │ Attn.   │
  │                  │      │                  │     │         │ │ Cross-  │
  │ FFN              │      │ FFN              │     │ FFN     │ │ Attn.   │
  │ × L layers       │      │ × L layers       │     │ × L     │ │ FFN     │
  └──────────────────┘      └──────────────────┘     └─────────┘ │ × L     │
                                                                  └─────────┘
  Use: encoding,            Use: generation,         Use: translation,
  classification,           dialogue, completion     summarisation,
  retrieval                                          speech-to-text

════════════════════════════════════════════════════════════════════════
```

### 7.5 Parameter Counting

For a decoder-only Transformer with $L$ layers, $d_{\text{model}} = d$, $h$ heads, $d_{ff} = 4d$ (ReLU) or $\frac{8}{3}d$ (SwiGLU), and vocabulary $V$:

| Component | Parameters | Notes |
| --- | --- | --- |
| Token embedding | $V \cdot d$ | Often tied with LM head |
| Per-layer attention ($W_Q, W_K, W_V, W_O$) | $4d^2$ | Same for any $h$ |
| Per-layer FFN (ReLU, $4\times$) | $8d^2 + 5d$ | Or $8d^2$ ignoring bias |
| Per-layer FFN (SwiGLU, $\frac{8}{3}\times$) | $8d^2$ | Three matrices |
| Per-layer norms (2 RMSNorm) | $2d$ | Just $\gamma$ vectors |
| Final norm | $d$ | Before LM head |
| LM head | $V \cdot d$ | Often tied with embedding |

**Total (approximate):** $\text{params} \approx 12 L d^2 + V d$ (ignoring small terms).

For LLaMA-7B: $L = 32$, $d = 4096$, $V = 32000$: $12 \cdot 32 \cdot 4096^2 + 32000 \cdot 4096 \approx 6.4\text{B} + 0.13\text{B} \approx 6.7\text{B}$. The published count is 6.7B — the formula works.

**For AI:** Parameter counting is essential for estimating training cost ($\text{FLOPs} \approx 6 \cdot N \cdot D$ where $N$ = params and $D$ = tokens), memory requirements ($\approx 2N$ bytes in fp16), and choosing LoRA rank ($r \ll d$ means $r(d + d) \ll d^2$ additional params per adapted matrix).

**Concrete parameter counts for major models:**

| Model | $L$ | $d$ | $h$ | $d_{ff}$ | Vocab $V$ | Total params |
| --- | --- | --- | --- | --- | --- | --- |
| GPT-2 Small | 12 | 768 | 12 | 3072 | 50257 | 124M |
| BERT-Base | 12 | 768 | 12 | 3072 | 30522 | 110M |
| GPT-3 | 96 | 12288 | 96 | 49152 | 50257 | 175B |
| LLaMA-7B | 32 | 4096 | 32 | 11008 | 32000 | 6.7B |
| LLaMA-70B | 80 | 8192 | 64 | 28672 | 32000 | 70B |
| Mistral-7B | 32 | 4096 | 32 | 14336 | 32000 | 7.2B |

**Memory budget during training (fp32 master + bf16 forward + AdamW):**

| Component | Bytes per param | 7B model | 70B model |
| --- | --- | --- | --- |
| bf16 weights (forward) | 2 | 14 GB | 140 GB |
| fp32 master weights | 4 | 28 GB | 280 GB |
| fp32 first moment ($m$) | 4 | 28 GB | 280 GB |
| fp32 second moment ($v$) | 4 | 28 GB | 280 GB |
| Activations (estimate) | ~10–20 | ~70–140 GB | ~700 GB+ |
| **Total** | **~24+** | **~170 GB+** | **~1.7 TB+** |

This is why training a 7B model requires at least 2× A100-80GB (for model parallelism), and 70B requires a full node of 8× A100s with tensor/pipeline parallelism.

---

## 8. Computational Complexity and Efficient Attention

### 8.1 FLOPs Analysis

The dominant operations in a Transformer forward pass are matrix multiplications. For sequence length $n$, model dimension $d$, and $L$ layers:

**Attention FLOPs per layer:**
1. Compute $Q, K, V$: three matrix multiplications of $(n \times d) \cdot (d \times d)$ = $3 \times 2nd^2$ FLOPs
2. Compute $QK^\top$: $(n \times d) \cdot (d \times n)$ = $2n^2 d$ FLOPs
3. Compute $AV$: $(n \times n) \cdot (n \times d)$ = $2n^2 d$ FLOPs
4. Output projection: $(n \times d) \cdot (d \times d)$ = $2nd^2$ FLOPs

Total attention: $8nd^2 + 4n^2 d$ FLOPs.

**FFN FLOPs per layer:** Two (or three for SwiGLU) matrix multiplications of $(n \times d) \cdot (d \times 4d)$ and $(n \times 4d) \cdot (4d \times d)$: $16nd^2$ FLOPs.

**Total per layer:** $24nd^2 + 4n^2d$ FLOPs. Across $L$ layers: $L(24nd^2 + 4n^2d)$.

The $4n^2 d$ term is the **quadratic attention cost**. For short sequences ($n \ll 6d$), the $24nd^2$ FFN term dominates. For long sequences ($n \gg 6d$), the $4n^2 d$ attention term dominates. With $d = 4096$ and $n = 32768$, both terms are comparable — this is the crossover regime for modern LLMs.

### 8.2 Memory Bottleneck and KV Cache

During autoregressive generation, the model generates one token at a time. At step $t$, the new token's query must attend to all previous keys and values. Naive recomputation would require $O(t^2)$ work per step. The **KV cache** stores the key and value projections from all previous steps:

$$\text{KV cache size} = 2 \cdot L \cdot n \cdot d \cdot \text{bytes per element}$$

For LLaMA-70B ($L = 80$, $d = 8192$) with fp16 and $n = 4096$: $2 \times 80 \times 4096 \times 8192 \times 2 = 10.7$ GB per sequence. This is often the binding constraint on batch size during inference.

**For AI:** KV cache memory is why GQA/MQA are critical for deployment. With $g = 8$ KV groups instead of 64 KV heads, the cache shrinks by $8\times$.

### 8.3 FlashAttention

**FlashAttention** (Dao et al., 2022) is an IO-aware algorithm that computes **exact** attention without materialising the full $n \times n$ attention matrix. The key insight: the bottleneck is not compute (FLOPs) but memory bandwidth — reading and writing the attention matrix to GPU HBM.

**Tiling strategy:** FlashAttention partitions $Q$, $K$, $V$ into blocks that fit in SRAM (on-chip memory, ~20 MB on an A100). For each block of queries:
1. Load a block of $K$ and $V$ into SRAM
2. Compute local attention scores and values in SRAM
3. Accumulate using the online softmax trick (Milakov & Gimelshein, 2018)
4. Write only the final output block to HBM

The **online softmax trick** maintains running statistics $m_j = \max$ and $\ell_j = \sum \exp$ that allow computing softmax incrementally without storing all logits simultaneously:

$$m_{j+1} = \max(m_j, \tilde{m}_{j+1})$$
$$\ell_{j+1} = e^{m_j - m_{j+1}} \ell_j + e^{\tilde{m}_{j+1} - m_{j+1}} \tilde{\ell}_{j+1}$$
$$\mathbf{o}_{j+1} = \frac{e^{m_j - m_{j+1}} \ell_j \mathbf{o}_j + e^{\tilde{m}_{j+1} - m_{j+1}} \tilde{\ell}_{j+1} \tilde{\mathbf{o}}_{j+1}}{\ell_{j+1}}$$

**Complexity:** FlashAttention computes attention in $O(n^2 d)$ FLOPs (same as standard) but $O(n)$ memory instead of $O(n^2)$. The wall-clock speedup comes from reduced HBM reads/writes: $O(n^2 d^2 / M)$ where $M$ is SRAM size, compared to $O(n^2 d + n^2)$ for standard attention.

**For AI:** FlashAttention-2 and FlashAttention-3 are used in essentially all modern LLM training and inference. The 2–4x speedup and memory savings are what made 128K+ context windows practical.

**FlashAttention backward pass.** The backward pass must recompute the attention matrix (since it was not saved to HBM). This requires re-reading $Q, K, V$ from HBM and re-computing attention in blocks. The key insight: the attention matrix is a function of $Q, K$ only, so it can be reconstructed from the saved running statistics $(m, \ell)$ and the inputs. The backward FlashAttention algorithm:

1. Load blocks of $Q, K, V, O, dO$ (output gradient) from HBM
2. Recompute local attention scores $S_{ij} = Q_i K_j^\top / \sqrt{d_k}$
3. Recompute local attention weights from saved $(m, \ell)$
4. Compute local gradients $dV_j, dK_j, dQ_i$
5. Accumulate into output gradient buffers in HBM

The recomputation adds ~25% to the total forward+backward FLOPs, but the memory savings (from $O(n^2)$ to $O(n)$) more than compensate.

### 8.4 Multi-Latent Attention (MLA)

DeepSeek-V2 (2024) introduced **Multi-Latent Attention**, which compresses the KV cache through learned low-rank projections:

$$\mathbf{c}^{KV} = W_{DKV} \mathbf{x} \in \mathbb{R}^{d_c}$$
$$K = W_{UK} \mathbf{c}^{KV}, \quad V = W_{UV} \mathbf{c}^{KV}$$

where $d_c \ll d_{\text{model}}$. Instead of caching full $K$ and $V$ matrices (dimension $d_{\text{model}}$ per head), MLA caches the compressed latent $\mathbf{c}^{KV}$ (dimension $d_c$). The up-projection matrices $W_{UK}, W_{UV}$ are absorbed into the query and output projections during inference.

**KV cache reduction:** From $2 \cdot n_{\text{heads}} \cdot d_{\text{head}}$ to $d_c$ per token per layer. For DeepSeek-V2 with $d_c = 512$ vs $128 \times 128 = 16384$ for standard MHA, this is a $32\times$ reduction.

**The absorption trick.** During inference, MLA avoids the up-projection compute by absorbing $W_{UK}$ into the query projection:

$$\mathbf{q}^\top W_{UK} \mathbf{c}^{KV} = (W_{UK}^\top \mathbf{q})^\top \mathbf{c}^{KV} = \tilde{\mathbf{q}}^\top \mathbf{c}^{KV}$$

The "absorbed query" $\tilde{\mathbf{q}} = W_{UK}^\top \mathbf{q}$ lives in the compressed $d_c$-dimensional space. This means the attention computation uses the $d_c$-dimensional compressed keys directly, avoiding the cost of expanding back to $d_{\text{model}}$. The same trick applies to $W_{UV}$ and the output projection. The result: MLA achieves the quality of MHA with the KV cache footprint of MQA.

### 8.5 Linear Attention

The softmax attention kernel $k(\mathbf{q}, \mathbf{k}) = \exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d_k})$ creates the $O(n^2)$ bottleneck because it cannot be factorised. **Linear attention** (Katharopoulos et al., 2020) replaces the kernel with a decomposable feature map $\phi$:

$$\operatorname{Attn}(\mathbf{q}_i) = \frac{\sum_j \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j) \mathbf{v}_j^\top}{\sum_j \phi(\mathbf{q}_i)^\top \phi(\mathbf{k}_j)} = \frac{\phi(\mathbf{q}_i)^\top \sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top}{\phi(\mathbf{q}_i)^\top \sum_j \phi(\mathbf{k}_j)}$$

The key: the sums $S = \sum_j \phi(\mathbf{k}_j) \mathbf{v}_j^\top$ and $\mathbf{z} = \sum_j \phi(\mathbf{k}_j)$ can be computed once and reused for all queries. This reduces attention from $O(n^2 d)$ to $O(n d^2)$ — linear in sequence length.

**The quality gap:** Linear attention consistently underperforms softmax attention for language modelling. The softmax kernel's ability to produce sharp, sparse attention patterns is critical for tasks requiring precise information retrieval (e.g., copying, factual recall). This gap motivated hybrid approaches like Mamba (Section 11.4).

```text
COMPLEXITY COMPARISON
════════════════════════════════════════════════════════════════════════

  Method              Time            Memory          Exact?
  ──────              ────            ──────          ──────
  Standard Attention  O(n²d)          O(n²)           Yes
  FlashAttention      O(n²d)          O(n)            Yes (IO-aware)
  FlashAttention-2    O(n²d)          O(n)            Yes (better tiling)
  Linear Attention    O(nd²)          O(d²)           No (approx kernel)
  Sparse (local+gbl)  O(n·w·d)        O(n·w)          No (subset of pairs)
  MLA                 O(n²d)          O(n·d_c)        Yes (compressed KV)

  Crossover point: n = 6d
  ─────────────────────────────────────────────────────
  n < 6d (e.g., n=2K, d=4K):  FFN dominates (24nd²)
  n > 6d (e.g., n=32K, d=4K): Attention dominates (4n²d)
  n ≈ 6d (e.g., n=24K, d=4K): Both terms comparable

════════════════════════════════════════════════════════════════════════
```

**Memory hierarchy and the IO bottleneck.** The A100 GPU has two levels of memory:

- **HBM (High Bandwidth Memory):** 80 GB, bandwidth 2 TB/s
- **SRAM (on-chip):** 20 MB per streaming multiprocessor, bandwidth ~19 TB/s

Standard attention writes the full $n \times n$ attention matrix to HBM (at $n = 4096$, this is $4096^2 \times 2 = 32$ MB in fp16). FlashAttention keeps this matrix in SRAM and never writes it to HBM. The speedup comes not from fewer FLOPs but from fewer bytes transferred between SRAM and HBM — the algorithm is **compute-bound** rather than **memory-bound**.

**Example: LLaMA-2 7B inference costs.** For $n = 4096$, $d = 4096$, $L = 32$:

- Attention FLOPs per layer: $8 \times 4096 \times 4096^2 + 4 \times 4096^2 \times 4096 = 549\text{B} + 275\text{B} = 824\text{B}$ FLOPs
- FFN FLOPs per layer: $16 \times 4096 \times 4096^2 = 1.1\text{T}$ FLOPs
- Total: $32 \times (824\text{B} + 1.1\text{T}) \approx 61\text{T}$ FLOPs per forward pass

---

## 9. Training Dynamics and Optimisation

### 9.1 Weight Initialisation

Proper initialisation is critical for Transformers. The residual connections create a signal propagation problem: with $L$ layers, the output magnitude grows as $\sqrt{L}$ if each sublayer contributes unit-variance updates.

**GPT-2 initialisation (Radford et al., 2019):** Scale the output projection of each sublayer by $1/\sqrt{2L}$:

$$W_O^{(l)} \sim \mathcal{N}\left(0, \frac{0.02}{\sqrt{2L}}\right)$$

This ensures the residual stream variance stays approximately constant across depth. The factor $2L$ accounts for two sublayers (attention + FFN) per block.

**Xavier/Glorot initialisation** for other weights: $W \sim \mathcal{N}(0, 1/d_{\text{in}})$ or $\mathcal{U}(-\sqrt{6/(d_{\text{in}} + d_{\text{out}})}, +\sqrt{6/(d_{\text{in}} + d_{\text{out}})})$. This preserves variance through the forward pass for linear layers.

**For AI:** Initialisation interacts with the learning rate schedule. The muP (maximal update parameterisation, Yang et al., 2022) framework shows how to set initialisation scales and learning rates jointly so that hyperparameters transfer across model widths — enabling hyperparameter search on small models and scaling up.

**Signal propagation at initialisation.** Consider a Pre-Norm block: $\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + W_O \operatorname{Attn}(\operatorname{Norm}(\mathbf{x}^{(l)}))$. At initialisation, $\operatorname{Norm}(\mathbf{x}^{(l)})$ has unit variance, and the attention output has variance $\sim d_v / d_{\text{model}}$. With the output projection, the sublayer contribution has variance $\sim d_v \cdot \sigma_{W_O}^2$. After $L$ layers:

$$\operatorname{Var}(\mathbf{x}^{(L)}) \approx \operatorname{Var}(\mathbf{x}^{(0)}) + 2L \cdot d_v \cdot \sigma_{W_O}^2$$

To keep this $\approx 2 \operatorname{Var}(\mathbf{x}^{(0)})$, we need $\sigma_{W_O}^2 \sim 1 / (2L \cdot d_v)$, matching the GPT-2 prescription.

### 9.2 AdamW + Warmup + Cosine Decay

The standard optimiser for Transformer training is **AdamW** (Loshchilov & Hutter, 2019):

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad \text{(first moment)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad \text{(second moment)}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t), \quad \hat{v}_t = v_t / (1 - \beta_2^t) \qquad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \eta_t (\hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) + \lambda \theta_t) \qquad \text{(update + decoupled weight decay)}$$

Standard hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.95$ (not 0.999), $\epsilon = 10^{-8}$, $\lambda = 0.1$.

**Learning rate schedule:** The standard recipe combines linear warmup with cosine decay:

$$\eta_t = \begin{cases} \eta_{\max} \cdot t / T_{\text{warmup}} & t \le T_{\text{warmup}} \\ \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi \cdot (t - T_{\text{warmup}}) / (T_{\max} - T_{\text{warmup}}))) & t > T_{\text{warmup}} \end{cases}$$

**Why warmup is necessary:** At initialisation, the Adam second moment estimates $v_t$ are near zero, so the effective step size $\hat{m}_t / \sqrt{\hat{v}_t}$ is very large. The warmup period gives the optimiser time to accumulate accurate moment estimates before taking full-size steps.

**Why $\beta_2 = 0.95$ instead of $0.999$.** The original Adam paper used $\beta_2 = 0.999$, but Transformer training benefits from faster adaptation of the second moment. With $\beta_2 = 0.999$, the effective window for the second moment is $\sim 1/(1 - \beta_2) = 1000$ steps. For LLM training with rapidly changing gradient statistics (especially early in training), this window is too long — the optimiser does not adapt quickly enough to changes in curvature. Setting $\beta_2 = 0.95$ reduces the window to ~20 steps.

**The cosine decay schedule** was proposed by Loshchilov & Hutter (2017) and has become universal for LLM training. The cosine shape provides a smooth transition from the peak learning rate to near-zero, and the final learning rate $\eta_{\min}$ is typically set to $0.1 \times \eta_{\max}$ or $0$. Variants include:

- **WSD (Warmup-Stable-Decay):** Used in Mistral and Phi-3. The learning rate is constant during the "stable" phase (most of training), with cosine decay only in the final 10–20% of steps. This is simpler to implement and allows extending training by adding more steps.
- **Trapezoidal:** Linear warmup, constant, linear decay — even simpler than cosine.

### 9.3 Gradient Clipping

The standard practice for Transformer training is **global gradient norm clipping**: if $\lVert \nabla_\theta \mathcal{L} \rVert_2 > c$, rescale:

$$\nabla_\theta \mathcal{L} \leftarrow \frac{c}{\lVert \nabla_\theta \mathcal{L} \rVert_2} \nabla_\theta \mathcal{L}$$

Typical threshold: $c = 1.0$. This prevents catastrophic parameter updates from rare high-loss examples (e.g., data corruption, distribution shift).

**Mathematical effect:** Clipping changes the optimisation landscape from unbounded gradients to a trust region of radius $c$ around the current parameters. The direction is preserved, only the magnitude is capped.

**When clipping activates.** In practice, gradient spikes occur when the model encounters unusual batches (rare tokens, distribution shift, data corruption). For LLaMA-style training, gradient clipping activates on roughly 1–5% of steps, with the clip ratio $\lVert g \rVert / c$ occasionally reaching 10–100×. Without clipping, a single spike can destabilise training irreversibly (the "loss spike" phenomenon).

### 9.4 Mixed Precision

Modern Transformers train in **mixed precision** (Micikevicius et al., 2018): the forward/backward passes use fp16 or bf16, while the master weights and optimiser states remain in fp32.

**bf16 (bfloat16):** Same exponent range as fp32 (8 bits) but only 7 mantissa bits. This means bf16 can represent the same range of magnitudes as fp32, avoiding the overflow/underflow issues of fp16. The reduced precision (3 decimal digits vs 7) is acceptable for gradient computation.

**Memory savings:** With bf16 training, the forward/backward memory is halved. Combined with AdamW (which stores fp32 master weights + two fp32 moment estimates), the total memory per parameter is: 2 bytes (bf16 weight) + 4 bytes (fp32 master) + 4 bytes ($m$) + 4 bytes ($v$) = 14 bytes. For a 7B model: ~98 GB for training state.

**fp8 training (2024+).** NVIDIA Hopper GPUs (H100) support fp8 (E4M3 and E5M2 formats) with 2× the throughput of bf16. The challenge is maintaining training stability with only 3 mantissa bits. Per-tensor scaling factors and loss scaling mitigate the reduced precision. DeepSeek-V3 demonstrated fp8 training at frontier scale, reducing training cost by ~40%.

**Number format comparison:**

| Format | Bits | Exponent | Mantissa | Range | Precision | Use |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 | 32 | 8 | 23 | $\pm 3.4 \times 10^{38}$ | ~7 digits | Master weights, optimiser |
| bf16 | 16 | 8 | 7 | $\pm 3.4 \times 10^{38}$ | ~3 digits | Forward/backward pass |
| fp16 | 16 | 5 | 10 | $\pm 65504$ | ~4 digits | Older training; overflow risk |
| fp8 (E4M3) | 8 | 4 | 3 | $\pm 448$ | ~2 digits | H100 matmuls |

### 9.5 Scaling Laws

**Chinchilla scaling laws** (Hoffmann et al., 2022) describe how loss decreases with compute budget:

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + L_\infty$$

where $N$ = parameters, $D$ = training tokens, $\alpha \approx 0.34$, $\beta \approx 0.28$, and $L_\infty$ is the irreducible loss (entropy of natural language). For a fixed compute budget $C \approx 6ND$, the optimal allocation is roughly $N \propto C^{0.5}$ and $D \propto C^{0.5}$ — train a model on approximately 20 tokens per parameter.

**For AI:** Chinchilla showed that most LLMs before 2022 were significantly undertrained. GPT-3 (175B params, 300B tokens = 1.7 tokens/param) was trained on 12x too few tokens by Chinchilla's ratio. LLaMA-1 (7B params, 1T tokens = 143 tokens/param) went in the opposite direction, overtrained by Chinchilla standards but producing a better model for inference cost.

**Scaling law predictions — concrete examples:**

| Model | Params $N$ | Tokens $D$ | $D/N$ ratio | Compute $\approx 6ND$ | Notes |
| --- | --- | --- | --- | --- | --- |
| GPT-3 | 175B | 300B | 1.7 | $3.1 \times 10^{23}$ | Undertrained |
| Chinchilla | 70B | 1.4T | 20 | $5.9 \times 10^{23}$ | Optimal by Chinchilla |
| LLaMA-1 7B | 7B | 1T | 143 | $4.2 \times 10^{22}$ | Inference-optimal |
| LLaMA-2 70B | 70B | 2T | 29 | $8.4 \times 10^{23}$ | Slightly overtrained |
| LLaMA-3 8B | 8B | 15T | 1875 | $7.2 \times 10^{23}$ | Massively overtrained |

The trend since 2023 has been to **overtrain** relative to Chinchilla — training on far more tokens than the compute-optimal ratio. The reason: at inference time, a smaller model is cheaper to serve, and the extra training compute is a one-time cost. LLaMA-3 trains an 8B model on 15T tokens, achieving performance competitive with larger models at a fraction of the inference cost.

**The three regimes of scaling:**

1. **Compute-optimal (Chinchilla):** Minimise loss for a fixed training budget. $N \propto C^{0.5}$, $D \propto C^{0.5}$.
2. **Inference-optimal (LLaMA-3):** Minimise model size for a target loss. Overtrain a smaller model on more data.
3. **Data-constrained:** When high-quality data is exhausted before the compute budget. Motivates synthetic data, data mixing, and curriculum strategies.

---

## 10. Interpretability

### 10.1 Attention Patterns

Trained attention heads learn stereotyped patterns that can be categorised:

**Positional heads:** Attend to fixed relative positions. "Previous-token" heads have attention matrices that are approximately shift matrices $A_{ij} \approx \delta_{i,j-1}$. These implement basic bigram statistics.

**Syntactic heads:** Attend along dependency parse arcs. Clark et al. (2019) showed that specific BERT heads correspond to subject–verb, determiner–noun, and possessive relations.

**Induction heads (Olsson et al., 2022):** A two-head circuit that implements in-context learning. Head 1 is a "previous-token" head that copies position $i-1$'s information into position $i$. Head 2 uses this to match the current context against earlier occurrences: if "... A B ... A" has appeared, the induction head predicts "B" will follow the second "A". Formally:

$$\text{Induction head: } \text{attn}(i) \propto \exp\left(\mathbf{q}_i^\top \mathbf{k}_j \mid x_{j-1} = x_{i-1}\right)$$

The induction head is hypothesised to be the fundamental circuit underlying in-context learning in Transformers.

### 10.2 Residual Stream Hypothesis

Elhage et al. (2021) proposed that the residual stream is a **shared communication channel**. Each attention head and FFN reads from and writes to this channel via low-rank projections. This leads to a decomposition of the final output:

$$\mathbf{x}^{(L)} = \mathbf{x}^{(0)} + \sum_{l=1}^{L} \left(\sum_{h=1}^{H} \operatorname{head}_h^{(l)}(\mathbf{x}) + \operatorname{FFN}^{(l)}(\mathbf{x})\right)$$

Each term in this sum is a "path" through the network. The **path decomposition** enables mechanistic interpretability: we can identify which specific heads and FFN layers contribute to a particular output.

**Logit attribution:** To understand why the model predicts token $t$ at position $i$, project each term's contribution onto the unembedding vector $\mathbf{u}_t$:

$$\text{logit contribution of head } h = \operatorname{head}_h^{(l)}(\mathbf{x})_i^\top \mathbf{u}_t$$

### 10.3 Superposition and Polysemanticity

**Superposition hypothesis (Elhage et al., 2022):** Networks represent more features than they have dimensions by using nearly-orthogonal directions. If there are $m \gg d$ features, each used sparsely, the network can store them as approximately orthogonal vectors with small interference.

The mathematical framework: represent $m$ features using $d$ directions $\mathbf{f}_1, \ldots, \mathbf{f}_m \in \mathbb{R}^d$ where $m > d$. The interference between features $i$ and $j$ is $|\mathbf{f}_i^\top \mathbf{f}_j|$. By the Johnson–Lindenstrauss lemma, $m = \exp(O(d))$ approximately orthogonal directions exist in $\mathbb{R}^d$, with pairwise inner products $O(1/\sqrt{d})$.

**Polysemanticity:** A consequence of superposition — individual neurons respond to multiple unrelated features. Neuron 347 might activate for both "academic citations" and "prices in dollars" because these features are represented by non-orthogonal directions that overlap on this neuron.

**Quantifying superposition.** For $m$ features in $d$ dimensions with sparsity $s$ (fraction of inputs where each feature is active), the optimal strategy is:

- If $s > d/m$: **no superposition** — store only $d$ features in $d$ dimensions
- If $s < d/m$: **superposition** — pack $m > d$ features with pairwise interference $\propto 1/\sqrt{d}$
- As $s \to 0$: nearly all $m$ features can be stored with negligible interference

The Johnson–Lindenstrauss lemma makes this precise: $m = \exp(O(d \epsilon^2))$ nearly-orthogonal directions exist with pairwise inner products $< \epsilon$.

**For AI:** Superposition is a fundamental challenge for mechanistic interpretability. Sparse autoencoders (Cunningham et al., 2023; Bricken et al., 2023) decompose neuron activations into monosemantic features, providing a path toward understanding what LLMs compute.

### 10.4 Probing

**Linear probes** test whether specific information is linearly decodable from the residual stream. Train a linear classifier $P(\text{label} | \mathbf{x}^{(l)}) = \text{softmax}(W_p \mathbf{x}^{(l)})$ to predict a linguistic property (POS tag, syntactic role, entity type) from the hidden state at layer $l$.

If a linear probe achieves high accuracy, the information is represented in a linearly accessible format. The probe accuracy across layers reveals **when** information is encoded: syntactic features typically peak in middle layers, while semantic features peak in later layers.

**Limitation:** A linear probe does not prove the model **uses** the information — only that it is present. The information might be a byproduct of computation rather than a causally relevant representation. **Causal interventions** (activation patching, causal tracing) test whether perturbing a representation changes the model's output, establishing a causal role.

```text
MECHANISTIC INTERPRETABILITY TOOLKIT
════════════════════════════════════════════════════════════════════════

  Technique             What it reveals              Limitation
  ─────────             ─────────────────            ──────────
  Attention patterns    Which positions attend to    Doesn't show what
                        which (per head)             information flows

  Logit attribution     Per-head/layer contribution  Ignores nonlinear
                        to output token probability  interactions

  Activation patching   Causal role of specific      Combinatorial
                        activations                  explosion of paths

  Linear probing        Whether info is linearly     Correlation, not
                        decodable at each layer      necessarily causation

  Sparse autoencoders   Monosemantic features from   Reconstruction loss;
                        polysemantic neurons         feature splitting

  Circuit analysis      Complete computational       Only feasible for
                        subgraph for a task          simple behaviours

════════════════════════════════════════════════════════════════════════
```

**Example: Induction head circuit.** The complete induction circuit uses two layers:

- **Layer $l$, Head A** (previous-token head): implements $A_{ij} \propto \mathbb{1}[j = i - 1]$. This copies the identity of token $i-1$ into position $i$'s residual stream via the value projection $W_V^A$.
- **Layer $l+1$, Head B** (induction head): uses $W_Q^B$ and $W_K^B$ composed with the OV circuit of Head A. The query at position $i$ encodes "what token preceded me?", and the key at position $j$ encodes "what token am I?". The match occurs when $x_j = x_{i-1}$ appeared earlier as $x_{j'}$, and position $j'+1$ provides the prediction.

This two-head circuit explains the "phase transition" in in-context learning observed around 200M parameters — it requires sufficient capacity to form the composition $W_Q^B W_K^B (W_O^A W_V^A)$.

**Formal definition of a circuit.** A circuit in a Transformer is a computational subgraph $C = (V, E)$ where:

- **Vertices** $V$ are attention heads and FFN layers
- **Edges** $E$ are information flows through the residual stream (from one component's output to another's input)
- The circuit computes a specific function $f_C: \text{input tokens} \to \text{output logits}$

A circuit is **complete** for a task if ablating (zeroing out) all components outside $C$ preserves the task performance. Circuits can be discovered through **activation patching**: systematically replacing activations with clean/corrupt values to identify which components are causally relevant.

**Known circuits in GPT-2 scale models:**

| Circuit | Function | Components | Reference |
| --- | --- | --- | --- |
| Induction heads | In-context pattern completion | 2 attention heads across 2 layers | Olsson et al. (2022) |
| IOI (Indirect Object ID) | "Mary gave the book to John. John gave..." → "Mary" | 26 heads across 10 layers | Wang et al. (2022) |
| Greater-than | "The war lasted from 1732 to 17..." → digit > 32 | 5 attention heads + MLPs | Hanna et al. (2023) |

---

## 11. Modern Extensions (2024–2026)

### 11.1 LoRA

**Low-Rank Adaptation** (Hu et al., 2022) fine-tunes a pretrained Transformer by adding low-rank updates to weight matrices. For a pretrained weight $W_0 \in \mathbb{R}^{d \times d}$:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, and $r \ll d$ (typically $r = 4, 8, 16$). The pretrained weight $W_0$ is frozen; only $A$ and $B$ are trained.

**Parameter savings:** Standard fine-tuning updates $d^2$ parameters per matrix. LoRA updates $2rd$ parameters — a reduction factor of $d / (2r)$. For $d = 4096$ and $r = 16$: $128\times$ fewer parameters.

**Mathematical justification:** Aghajanyan et al. (2021) showed that fine-tuning updates to pretrained weights have low intrinsic rank — the weight changes $\Delta W$ during fine-tuning lie in a low-dimensional subspace. LoRA makes this explicit by parameterising $\Delta W$ as a rank-$r$ matrix.

**Which matrices to adapt:** Typically $W_Q$ and $W_V$ in attention (the original LoRA paper). DoRA (Liu et al., 2024) decomposes $W$ into magnitude and direction components, adapting only the direction. QLoRA (Dettmers et al., 2023) combines LoRA with 4-bit quantisation of $W_0$.

**LoRA initialisation:** $A$ is initialised from $\mathcal{N}(0, \sigma^2)$ and $B$ is initialised to zero, so that $\Delta W = BA = 0$ at the start of training. This means the fine-tuned model begins at exactly the pretrained checkpoint. The scaling factor $\alpha / r$ (where $\alpha$ is a hyperparameter, typically $\alpha = r$ or $\alpha = 2r$) controls the effective learning rate of the low-rank update.

**SVD connection:** If we performed full fine-tuning and then computed the SVD of $\Delta W = U \Sigma V^\top$, LoRA is equivalent to keeping only the top-$r$ singular components: $\Delta W \approx U_r \Sigma_r V_r^\top$. LoRA does this implicitly by constraining the rank from the start, which acts as a regulariser and prevents overfitting to the fine-tuning dataset.

**Example computation:** For LLaMA-7B ($d = 4096$), adapting $W_Q$ and $W_V$ across all 32 layers with $r = 16$:

- Full fine-tuning: $32 \times 2 \times 4096^2 = 1.07\text{B}$ trainable params
- LoRA: $32 \times 2 \times 2 \times 16 \times 4096 = 8.4\text{M}$ trainable params
- Reduction: $128\times$ fewer parameters, fitting in ~33 MB of fp16 storage

### 11.2 KV Cache Optimisation

Beyond GQA/MQA and MLA, several techniques compress the KV cache:

**Paged Attention (vLLM, Kwon et al., 2023):** Manages KV cache memory like virtual memory pages, eliminating fragmentation and enabling efficient batching of variable-length sequences. The idea is borrowed from operating system virtual memory: KV cache blocks are allocated in non-contiguous physical memory but addressed contiguously through a page table. This eliminates the 60–80% memory waste from pre-allocated contiguous buffers.

**Sliding Window KV Cache (Mistral):** Only cache the last $w$ tokens per layer. Combined with sliding window attention, this gives constant memory per layer regardless of sequence length.

**Token merging/pruning:** Dynamically reduce the number of cached tokens by merging similar key–value pairs (averaging their vectors) or dropping low-importance positions (those with consistently low attention weights). H2O (Heavy-Hitter Oracle, Zhang et al., 2023) keeps only the "heavy hitter" tokens that receive high cumulative attention weight.

### 11.3 Mixture of Experts

**Mixture of Experts (MoE)** replaces the dense FFN with multiple "expert" FFNs, routing each token to only $k$ of $E$ experts:

$$\operatorname{MoE}(\mathbf{x}) = \sum_{i=1}^{E} g_i(\mathbf{x}) \cdot \operatorname{FFN}_i(\mathbf{x})$$

where $g(\mathbf{x}) = \operatorname{TopK}(\operatorname{softmax}(W_g \mathbf{x}))$ is a sparse gating function. Typically $E = 8$ and $k = 2$ (each token uses 2 of 8 experts).

**Scaling advantage:** MoE increases model capacity (total parameters) without proportionally increasing compute (FLOPs). Mixtral 8x7B has 47B total parameters but only uses ~13B per token — achieving performance comparable to a dense 40B+ model at the cost of a 13B model.

**Load balancing:** Without regularisation, the router collapses to sending all tokens to one expert. An auxiliary loss encourages uniform expert utilisation:

$$\mathcal{L}_{\text{balance}} = E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average gating probability for expert $i$.

**MoE parameter counting:** Mixtral 8×7B has:

- 8 expert FFN copies per layer (each ~7B FFN params)
- 1 shared attention module per layer
- Total: ~47B params, but each token uses only 2 experts → ~13B active params
- Memory: must store all 47B params, but compute is 13B equivalent

**Expert specialisation.** Analysis of trained MoE models shows that experts often specialise by domain or linguistic function: one expert for code, one for mathematics, one for dialogue, etc. This specialisation emerges naturally from the routing optimisation — the gating network learns to direct different types of inputs to the most capable expert.

### 11.4 From Transformers to SSMs

**Mamba** (Gu & Dao, 2023) showed that selective state space models (SSMs) can match Transformer performance on language modelling. The core idea: replace attention's $O(n^2)$ all-pairs interaction with a recurrent state that is selectively updated:

$$\mathbf{h}_t = \bar{A} \mathbf{h}_{t-1} + \bar{B} x_t, \qquad y_t = C \mathbf{h}_t$$

where $\bar{A}, \bar{B}$ are **input-dependent** (this is the "selective" part — the model learns which inputs to remember and which to forget).

**Connection to attention:** The SSM output $y_t = \sum_{s=1}^{t} C \bar{A}^{t-s} \bar{B} x_s$ is structurally similar to attention: a weighted sum over all previous positions, but with weights determined by a recurrence ($\bar{A}^{t-s}$) rather than a quadratic score matrix.

**The Transformer–SSM duality.** Both architectures implement a function $y_i = \sum_j w_{ij} v_j$:

| Property | Transformer attention | SSM |
| --- | --- | --- |
| Weights $w_{ij}$ | Content-dependent: $\operatorname{softmax}(\mathbf{q}_i^\top \mathbf{k}_j)$ | Position-dependent: $C \bar{A}^{i-j} \bar{B}$ |
| Complexity | $O(n^2 d)$ — quadratic | $O(n d s)$ — linear ($s$ = state dim) |
| Sharp retrieval | Excellent (softmax can concentrate) | Poor (exponential decay) |
| Long-range | Good but expensive | Excellent and cheap |
| Parallelisable | Yes (matrix multiply) | Yes (parallel scan) |

The weakness of SSMs (poor sharp retrieval) is exactly the strength of attention, and vice versa. This complementarity is why hybrid architectures work: use SSM layers for cheap long-range context and attention layers for precise information retrieval.

**Hybrid architectures (2025–2026):** Models like Jamba (AI21) and Griffin (Google) alternate Transformer layers with SSM layers, combining attention's precise information retrieval with SSMs' efficient long-range propagation.

```text
EVOLUTION OF EFFICIENT ATTENTION (2017-2026)
════════════════════════════════════════════════════════════════════════

  2017  Standard Attention        O(n²d) time, O(n²) memory
         │
  2019  Sparse Transformer        O(n√n · d) — fixed sparse patterns
         │
  2020  Linformer                  O(nd) — project K,V to low dim
         │      Performer          O(nd²) — random feature approx
         │
  2022  FlashAttention            O(n²d) time, O(n) memory — IO-aware
         │
  2023  FlashAttention-2          2× faster tiling, better GPU util.
         │      Sliding Window     O(nwd) — Mistral, local context
         │
  2024  MLA (DeepSeek-V2)         O(n²d) time, O(n·d_c) KV cache
         │      Mamba              O(nd) — selective SSM, no attention
         │
  2025  FlashAttention-3          Hopper GPU optimised
         │      Hybrid Attn+SSM   Best of both: precise + efficient
         │
  2026  Ring Attention            Distribute across devices for >1M ctx

════════════════════════════════════════════════════════════════════════
```

**Positional encoding comparison summary:**

| Method | Type | Learned? | Relative position? | Length generalisation | Used in |
| --- | --- | --- | --- | --- | --- |
| Sinusoidal | Additive to input | No | Via rotation matrix | Poor | Original Transformer |
| Learned absolute | Additive to input | Yes ($n \times d$ params) | No | None (fixed $n$) | GPT-2, BERT |
| RoPE | Multiplicative on Q,K | No (freq. only) | Yes (rotation group) | Moderate + extensible | LLaMA, PaLM, Mistral |
| ALiBi | Additive bias on scores | No ($h$ slopes only) | Yes (linear decay) | Good | BLOOM, MPT |
| NoPE | None | — | Via causal structure | Depends on task | Some small models |

---

## 12. Common Mistakes

| # | Mistake | Why It's Wrong | Fix |
| --- | --- | --- | --- |
| 1 | Forgetting the $1/\sqrt{d_k}$ scaling | Softmax saturates, gradients vanish | Always divide by $\sqrt{d_k}$ before softmax |
| 2 | Applying softmax column-wise instead of row-wise | Each query must produce a distribution over keys | Row-wise: each row of $QK^\top$ is a query's scores |
| 3 | Adding positional encoding to keys/values (not input) | RoPE modifies Q/K; sinusoidal adds to input before projection | Follow the specific scheme exactly |
| 4 | Using BatchNorm instead of LayerNorm | Batch statistics are noisy for variable-length sequences | Use LayerNorm or RMSNorm |
| 5 | Placing norm after residual (Post-Norm) without warmup | Post-Norm is unstable at depth without careful scheduling | Use Pre-Norm (default for LLMs) |
| 6 | Ignoring the causal mask during training | Model sees future tokens, learns to cheat | Add $-\infty$ mask to upper triangle of score matrix |
| 7 | Counting parameters wrong for MHA | Thinking more heads = more params | Total is $4d^2$ regardless of head count |
| 8 | Confusing attention FLOPs ($O(n^2 d)$) with FFN FLOPs ($O(nd^2)$) | For short sequences, FFN dominates; for long, attention dominates | Account for both terms when estimating compute |
| 9 | Materialising the $n \times n$ attention matrix for long sequences | $O(n^2)$ memory; OOM for large $n$ | Use FlashAttention |
| 10 | Applying LoRA to all weight matrices indiscriminately | Some matrices benefit more than others | Start with $W_Q, W_V$; ablate from there |
| 11 | Thinking attention is $O(n^2)$ in all regimes | For short sequences, FFN ($O(nd^2)$) dominates | Account for crossover at $n \approx 6d$ |
| 12 | Using learned positional embeddings for long contexts | Cannot extrapolate beyond training length | Use RoPE or ALiBi for length generalisation |

---

## 13. Exercises

**Exercise 1** (★). **Scaled Dot-Product Attention.**
Implement attention from scratch. Given $Q, K, V \in \mathbb{R}^{n \times d}$:
(a) Compute the score matrix $S = QK^\top / \sqrt{d}$.
(b) Apply row-wise softmax to get attention weights $A$.
(c) Compute the output $O = AV$.
(d) Verify that each row of $A$ sums to 1 and all entries are non-negative.
(e) Test with $n = 4$, $d = 8$ using random inputs.

**Exercise 2** (★). **Parameter Counting.**
For a single Transformer block with $d_{\text{model}} = 512$, $h = 8$, $d_{ff} = 2048$:
(a) Count the parameters in multi-head attention ($W_Q, W_K, W_V, W_O$).
(b) Count the parameters in a ReLU FFN ($W_1, W_2, \mathbf{b}_1, \mathbf{b}_2$).
(c) Count the parameters in a SwiGLU FFN ($W_1, W_2, W_3$) with $d_{ff} = \frac{8}{3} \times 512$.
(d) Count the parameters in 2 RMSNorm layers.
(e) Compute the total and verify the formula $12d^2$ (approximate).

**Exercise 3** (★). **Causal Masking.**
(a) Create a causal mask matrix for $n = 6$.
(b) Implement masked attention by adding $-10^9$ (representing $-\infty$) to masked positions.
(c) Show that position $i$'s output depends only on positions $1, \ldots, i$ by comparing with full attention.
(d) Verify that the attention matrix is lower-triangular.

**Exercise 4** (★★). **Rotary Position Embeddings.**
(a) Implement the 2D rotation matrix $R_m^{(i)}$ for position $m$ and frequency $\theta_i$.
(b) Apply RoPE to query and key vectors for positions $m = 3$ and $n = 7$.
(c) Compute $(\mathbf{q}_m')^\top \mathbf{k}_n'$ and $\mathbf{q}_m^\top R_{n-m} \mathbf{k}_n$. Verify they are equal.
(d) Show that the score depends only on the relative position $n - m = 4$ by testing with other $(m, n)$ pairs that have the same difference.

**Exercise 5** (★★). **Variance Scaling Verification.**
(a) For $d_k \in \{4, 16, 64, 128, 256, 512\}$, generate 10,000 random query–key pairs with i.i.d. $\mathcal{N}(0, 1)$ components.
(b) Compute the empirical variance of $\mathbf{q}^\top \mathbf{k}$ for each $d_k$.
(c) Plot empirical variance vs $d_k$ and overlay the theoretical line $\operatorname{Var} = d_k$.
(d) Repeat with scaled scores $\mathbf{q}^\top \mathbf{k} / \sqrt{d_k}$ and verify $\operatorname{Var} \approx 1$.
(e) Show the effect on softmax entropy: compute mean entropy of $\operatorname{softmax}(\mathbf{q}^\top K)$ vs $\operatorname{softmax}(\mathbf{q}^\top K / \sqrt{d_k})$.

**Exercise 6** (★★). **Multi-Head Attention from Scratch.**
(a) Implement single-head attention as a function.
(b) Implement multi-head attention: split $d_{\text{model}} = 64$ into $h = 8$ heads with $d_k = 8$.
(c) Create random projection matrices $W_Q^i, W_K^i, W_V^i$ for each head and $W_O$.
(d) Run all heads in parallel, concatenate, and project.
(e) Verify the output shape is $(n, d_{\text{model}})$ and matches a reference implementation.

**Exercise 7** (★★★). **FlashAttention Online Softmax.**
(a) Implement standard attention as a reference.
(b) Implement the online softmax trick: process $K, V$ in blocks of size $B = 2$, maintaining running max $m$ and sum-exp $\ell$.
(c) Show that the block-wise computation produces the exact same output as standard attention (up to floating-point precision).
(d) Count the peak memory usage of both methods and verify FlashAttention uses $O(n)$ vs $O(n^2)$.

**Exercise 8** (★★★). **LoRA Adaptation.**
(a) Create a random pretrained weight matrix $W_0 \in \mathbb{R}^{64 \times 64}$.
(b) Add a LoRA adapter with rank $r = 4$: $\Delta W = BA$ with $B \in \mathbb{R}^{64 \times 4}$, $A \in \mathbb{R}^{4 \times 64}$.
(c) Compute the modified attention output with $W = W_0 + \Delta W$ and verify it differs from the original.
(d) Compute the parameter reduction ratio: $2rd / d^2$.
(e) Compute the SVD of $\Delta W$ and verify it has rank exactly $r = 4$.

---

## 14. Why This Matters for AI (2026)

| Concept | AI Impact |
| --- | --- |
| Scaled dot-product attention | Core compute primitive of every LLM — GPT-4, Claude, Gemini, LLaMA-3 |
| Multi-head attention | Enables parallel processing of syntactic, semantic, and positional relations |
| RoPE | Standard positional encoding for open-source LLMs; enables context extension to 128K+ |
| SwiGLU FFN | 1–3% perplexity improvement over ReLU; standard in all 2023+ LLMs |
| RMSNorm | Replaced LayerNorm for efficiency; universal in LLaMA-family models |
| FlashAttention | Made long-context training practical; 2–4x speedup is load-bearing for 128K context |
| GQA/MQA | Reduces KV cache by 8–64x; critical for high-throughput inference serving |
| LoRA | Enables fine-tuning 70B+ models on a single GPU; standard for adaptation |
| MoE | Mixtral, DeepSeek-V2: more capacity at lower compute cost |
| Scaling laws | Guide optimal allocation of training compute budget |
| Induction heads | Mechanistic explanation of in-context learning |
| Superposition | Fundamental challenge for AI safety and interpretability |

---

## 15. Conceptual Bridge

### Looking Backward

The Transformer synthesises nearly every mathematical topic in this curriculum. Matrix multiplication (Chapter 02) gives us the $QK^\top V$ computation. Eigenvalues and SVD (Chapter 03) explain the low-rank structure of attention and enable LoRA. The chain rule (Chapter 04) powers backpropagation through the residual stream. Probability theory (Chapter 06) gives us softmax and cross-entropy. Optimisation (Chapter 08) provides AdamW and learning rate schedules. Information theory (Chapter 09) defines perplexity and the KL divergence in RLHF. The Transformer is where all of this mathematics becomes load-bearing.

### Looking Forward

Understanding the Transformer's mathematics opens three directions: (1) **efficiency** — FlashAttention, linear attention, and SSMs are all attempts to preserve the Transformer's expressiveness while reducing its $O(n^2)$ complexity; (2) **interpretability** — the residual stream decomposition and attention pattern analysis provide tools for understanding what these models compute; (3) **adaptation** — LoRA, QLoRA, and prompt tuning exploit the mathematical structure (low-rank updates, softmax temperature) to efficiently customise models for specific tasks.

The mathematics also reveals fundamental limitations. The softmax bottleneck limits the rank of next-token predictions. The $O(n^2)$ attention cost creates a hard trade-off between context length and compute. The superposition phenomenon means that representations are fundamentally difficult to interpret. These mathematical constraints shape the frontier of AI research in 2026.

### The Transformer as a Mathematical Object

The Transformer can be viewed through multiple mathematical lenses:

- **As a kernel machine:** Attention implements a parameterised Nadaraya–Watson estimator with a softmax kernel. Multi-head attention is a mixture of kernels. The FFN is a fixed-key kernel machine.
- **As a dynamical system:** The layer-to-layer update $\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + f^{(l)}(\mathbf{x}^{(l)})$ is an Euler discretisation of the ODE $d\mathbf{x}/dt = f(\mathbf{x}, t)$. This connection to neural ODEs (Chen et al., 2018) explains why deeper Transformers can implement more complex transformations.
- **As a computational graph:** Each attention head and FFN computes a specific function, and these compose through the residual stream. Mechanistic interpretability aims to reverse-engineer this computational graph.
- **As an information-processing channel:** The residual stream has finite capacity ($d_{\text{model}}$ dimensions). Each layer must decide what information to write, update, and delete within this fixed bandwidth.

Each perspective yields different insights and different research directions. The mathematical richness of the Transformer is why it remains the dominant architecture despite intense competition from alternatives.

### Key References

The following papers form the essential reading list for the mathematics of Transformers:

1. **Vaswani et al. (2017)** — "Attention Is All You Need". The original Transformer paper.
2. **Radford et al. (2019)** — GPT-2. Decoder-only scaling + training recipe.
3. **Devlin et al. (2019)** — BERT. Bidirectional encoder pretraining.
4. **Su et al. (2021)** — RoPE. Rotary Position Embeddings.
5. **Dao et al. (2022)** — FlashAttention. IO-aware exact attention.
6. **Hu et al. (2022)** — LoRA. Low-rank adaptation for efficient fine-tuning.
7. **Hoffmann et al. (2022)** — Chinchilla. Scaling laws for optimal compute allocation.
8. **Touvron et al. (2023)** — LLaMA. Open-source recipe: RoPE + SwiGLU + RMSNorm + GQA.
9. **Olsson et al. (2022)** — Induction heads. In-context learning mechanism.
10. **Elhage et al. (2022)** — Superposition. Features packed into lower-dimensional spaces.

```
CONCEPTUAL POSITION
════════════════════════════════════════════════════════════════════════

  PREREQUISITES                    THIS SECTION                 ENABLES
  ─────────────                    ────────────                 ───────

  Matrix Multiply (02) ──────┐
  Eigenvalues/SVD (03) ──────┤
  Chain Rule (04) ───────────┤     ┌──────────────────────┐
  Jacobians (05) ────────────┼────▶│  TRANSFORMER          │───▶ LLM Training
  Softmax/Prob (06) ─────────┤     │  ARCHITECTURE         │───▶ FlashAttention
  Cross-Entropy (09) ────────┤     │                      │───▶ LoRA / PEFT
  Optimisation (08) ─────────┤     │  Attention + FFN +    │───▶ Interpretability
  RNNs/LSTMs (14-04) ───────┘     │  Norms + Positional   │───▶ Scaling Laws
                                   └──────────────────────┘───▶ SSMs / Mamba

════════════════════════════════════════════════════════════════════════
```
