[← Embedding Space Math](../02-Embedding-Space-Math/notes.md) | [Home](../../README.md) | [Positional Encodings →](../04-Positional-Encodings/notes.md)

---

# Attention Mechanism Math

> _"Attention is a differentiable soft-lookup that lets every token dynamically decide which other tokens matter — it is the single primitive on which the Transformer revolution is built."_

## Overview

The attention mechanism is the computational core of all Transformer-based language models. It takes positionally-encoded token embeddings X ∈ ℝⁿˣᵈ and produces context-aware representations X' ∈ ℝⁿˣᵈ by letting every token attend to every other token. This section derives scaled dot-product attention from first principles, proves the √dₖ scaling factor, covers masking (causal, padding, sliding window), multi-head attention with full parameter counting, the KV cache and its memory mathematics, all modern attention variants (MHA, MQA, GQA, MLA, linear attention, FlashAttention), positional encodings as they interact with attention (RoPE, ALiBi), complexity analysis, attention interpretability (entropy, induction heads, circuits), and the 2024–2026 research frontier (hybrid SSM-attention architectures, FP8 attention, multi-million token contexts).

## Prerequisites

- Linear algebra: matrix multiplication, transpose, dot product, norms
- Calculus: softmax derivative, chain rule, gradient flow
- Probability: softmax as probability distribution, entropy
- Completed: [01-Tokenization-Math](../01-Tokenization-Math/notes.md) and [02-Embedding-Space-Math](../02-Embedding-Space-Math/notes.md)

## Companion Notebooks

| Notebook                           | Description                                                                                          |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Scaled dot-product attention, multi-head attention, masking, KV cache, RoPE in attention, complexity |
| [exercises.ipynb](exercises.ipynb) | Manual attention, scaling proof, causal mask, parameter counting, KV cache sizing, entropy, RoPE     |

## Learning Objectives

After completing this section, you will:

- Derive scaled dot-product attention step by step and prove why √dₖ scaling is necessary
- Implement single-head and multi-head attention from scratch using only NumPy
- Construct causal, padding, and sliding-window masks and explain their effect on attention weights
- Compute the exact parameter count for MHA, MQA, and GQA and compare their KV cache requirements
- Calculate KV cache memory for real models (LLaMA-3, GPT-4 class) at various sequence lengths
- Implement RoPE and ALiBi and verify their relative-position properties inside the attention computation
- Analyse attention weight distributions: entropy, attention sinks, induction head patterns
- Compare attention complexity across standard, FlashAttention, linear, and sliding-window variants
- Explain the full Transformer block (attention + FFN + residual + LayerNorm) and count total parameters
- Discuss 2024–2026 frontiers: MLA (DeepSeek), hybrid SSM-attention, FP8 attention, multi-million token contexts

## Table of Contents

- [Attention Mechanism Math](#attention-mechanism-math)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is Attention?](#11-what-is-attention)
    - [1.2 Why Attention Works](#12-why-attention-works)
    - [1.3 The Retrieval Analogy](#13-the-retrieval-analogy)
    - [1.4 Historical Timeline](#14-historical-timeline)
    - [1.5 Pipeline Position](#15-pipeline-position)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 Input Sequence](#21-input-sequence)
    - [2.2 Projection Matrices](#22-projection-matrices)
    - [2.3 Q, K, V Matrices](#23-q-k-v-matrices)
    - [2.4 Attention Types](#24-attention-types)
  - [3. Scaled Dot-Product Attention — Full Derivation](#3-scaled-dot-product-attention--full-derivation)
    - [3.1 Step-by-Step Formula](#31-step-by-step-formula)
    - [3.2 Why √dₖ Scaling — Full Proof](#32-why-dₖ-scaling--full-proof)
    - [3.3 Softmax Properties](#33-softmax-properties)
    - [3.4 Attention as Kernel Regression](#34-attention-as-kernel-regression)
  - [4. Masking — Complete Treatment](#4-masking--complete-treatment)
    - [4.1 Causal Mask](#41-causal-mask)
    - [4.2 Padding Mask](#42-padding-mask)
    - [4.3 Prefix / Document Mask](#43-prefix--document-mask)
    - [4.4 Sliding Window Mask](#44-sliding-window-mask)
    - [4.5 Combining Multiple Masks](#45-combining-multiple-masks)
  - [5. Multi-Head Attention (MHA)](#5-multi-head-attention-mha)
    - [5.1 Motivation](#51-motivation)
    - [5.2 Per-Head Computation](#52-per-head-computation)
    - [5.3 Concatenation and Output Projection](#53-concatenation-and-output-projection)
    - [5.4 Dimension Constraints and Parameter Count](#54-dimension-constraints-and-parameter-count)
    - [5.5 Head Redundancy and Pruning](#55-head-redundancy-and-pruning)
  - [6. KV Cache — Inference Mathematics](#6-kv-cache--inference-mathematics)
    - [6.1 What Is the KV Cache?](#61-what-is-the-kv-cache)
    - [6.2 KV Cache Memory Formula](#62-kv-cache-memory-formula)
    - [6.3 KV Cache at Scale](#63-kv-cache-at-scale)
    - [6.4 KV Cache Compression](#64-kv-cache-compression)
  - [7. Attention Variants — Complete Taxonomy](#7-attention-variants--complete-taxonomy)
    - [7.1 Standard MHA](#71-standard-mha)
    - [7.2 Multi-Query Attention (MQA)](#72-multi-query-attention-mqa)
    - [7.3 Grouped Query Attention (GQA)](#73-grouped-query-attention-gqa)
    - [7.4 Multi-head Latent Attention (MLA)](#74-multi-head-latent-attention-mla)
    - [7.5 Sliding Window Attention](#75-sliding-window-attention)
    - [7.6 Sparse Attention Patterns](#76-sparse-attention-patterns)
    - [7.7 Linear Attention](#77-linear-attention)
    - [7.8 FlashAttention Family](#78-flashattention-family)
    - [7.9 Hybrid Architectures (2024–2026)](#79-hybrid-architectures-20242026)
  - [8. Positional Encodings in Attention](#8-positional-encodings-in-attention)
    - [8.1 Why Attention Is Position-Agnostic](#81-why-attention-is-position-agnostic)
    - [8.2 Sinusoidal PE](#82-sinusoidal-pe)
    - [8.3 Learned Absolute PE](#83-learned-absolute-pe)
    - [8.4 Relative Position Encodings](#84-relative-position-encodings)
    - [8.5 RoPE](#85-rope)
    - [8.6 ALiBi](#86-alibi)
    - [8.7 Context Length Timeline](#87-context-length-timeline)
  - [9. Complexity Analysis](#9-complexity-analysis)
    - [9.1 Standard MHA Complexity](#91-standard-mha-complexity)
    - [9.2 Space Bottleneck](#92-space-bottleneck)
    - [9.3 FlashAttention Complexity](#93-flashattention-complexity)
    - [9.4 Efficient Attention Comparison](#94-efficient-attention-comparison)
  - [10. Attention Score Analysis and Interpretability](#10-attention-score-analysis-and-interpretability)
    - [10.1 Attention Entropy](#101-attention-entropy)
    - [10.2 Attention Sink Phenomenon](#102-attention-sink-phenomenon)
    - [10.3 Induction Heads](#103-induction-heads)
    - [10.4 Attention vs Gradient Attribution](#104-attention-vs-gradient-attribution)
    - [10.5 Attention Rollout](#105-attention-rollout)
    - [10.6 Circuit Analysis](#106-circuit-analysis)
  - [11. Attention in the Full Transformer Block](#11-attention-in-the-full-transformer-block)
    - [11.1 Residual Stream Perspective](#111-residual-stream-perspective)
    - [11.2 Layer Normalisation Variants](#112-layer-normalisation-variants)
    - [11.3 Parameter Count Per Block](#113-parameter-count-per-block)
    - [11.4 Total Model Parameter Estimate](#114-total-model-parameter-estimate)
  - [12. 2024–2026 Research Frontiers](#12-20242026-research-frontiers)
  - [13. Common Mistakes](#13-common-mistakes)
  - [14. Exercises](#14-exercises)
  - [15. Why This Matters for AI](#15-why-this-matters-for-ai)

---

## 1. Intuition

### 1.1 What Is Attention?

Every token in a sequence simultaneously broadcasts three things:

- A **query** (Q): "what information do I need?"
- A **key** (K): "what information do I contain?"
- A **value** (V): "what do I contribute if selected?"

Attention computes a soft, differentiable, weighted retrieval over all tokens simultaneously. The result: each token's representation is updated by blending information from all other tokens, with the blend weights determined by query-key compatibility.

Unlike hard database lookups that return exactly one item, attention returns a **weighted combination** of all values. This makes it end-to-end differentiable — gradients flow through the attention weights to learn which tokens should attend to which.

### 1.2 Why Attention Works

| Property                 | RNN/LSTM                       | Attention/Transformer         |
| ------------------------ | ------------------------------ | ----------------------------- |
| Token-to-token path      | O(n) through sequential states | O(1) direct connection        |
| Parallelism              | Sequential (can't parallelise) | Fully parallel                |
| Gradient path length     | O(n) → vanishing gradient      | O(1) → gradient highway       |
| Information routing      | Fixed by architecture          | Dynamic, content-dependent    |
| Long-range dependency    | Degrades with distance         | Equal access to all positions |
| Computational complexity | O(n·d²) sequential             | O(n²·d) parallel              |

The key insight: attention provides **direct paths** between every pair of tokens, eliminating the sequential bottleneck that made RNNs struggle with long-range dependencies. The cost is quadratic complexity in sequence length, which motivates the efficient attention variants in §7.

### 1.3 The Retrieval Analogy

Think of attention as a **soft database query**:

| Database Analogy  | Attention Component | Dimensionality |
| ----------------- | ------------------- | -------------- |
| Search query      | Q (query)           | ℝⁿˣᵈₖ          |
| Index labels      | K (key)             | ℝⁿˣᵈₖ          |
| Stored content    | V (value)           | ℝⁿˣᵈᵥ          |
| Similarity score  | QKᵀ / √dₖ           | ℝⁿˣⁿ           |
| Retrieval weights | softmax(scores)     | ℝⁿˣⁿ           |
| Retrieved result  | A·V                 | ℝⁿˣᵈᵥ          |

**Hard attention** (early NLP, 2015): select one item → non-differentiable → requires REINFORCE.  
**Soft attention** (standard today): weighted blend of all items → differentiable → standard backpropagation.

### 1.4 Historical Timeline

| Year | Paper / System                 | Contribution                                                     |
| ---- | ------------------------------ | ---------------------------------------------------------------- |
| 2014 | Bahdanau et al.                | Additive attention for seq2seq MT; first neural attention        |
| 2015 | Luong et al.                   | Multiplicative (dot-product) attention; simpler and faster       |
| 2017 | Vaswani et al.                 | "Attention Is All You Need" — self-attention as sole primitive   |
| 2018 | Devlin et al. (BERT)           | Bidirectional self-attention for representation learning         |
| 2020 | Brown et al. (GPT-3)           | Scaled autoregressive attention to 175B parameters               |
| 2022 | Dao et al. (FlashAttention)    | IO-aware exact attention; O(n) memory; changed practice          |
| 2023 | Dao (FlashAttention-2)         | Better parallelism, 2× faster than v1                            |
| 2023 | Gu & Dao (Mamba)               | Selective state-space model; O(n) compute; attention alternative |
| 2024 | Shah et al. (FlashAttention-3) | Hopper GPU optimised, FP8 support, ~3× FA2                       |
| 2024 | DeepSeek-V2 (MLA)              | Multi-head latent attention; ~10× KV cache reduction (d/d_c)     |
| 2025 | Hybrid SSM-attention           | Mamba-2, Jamba, Hymba at production scale                        |
| 2026 | Multi-million token contexts   | 1M–10M contexts becoming practical; native long-context training |

### 1.5 Pipeline Position

```
Token IDs → [Embedding] → X ∈ ℝⁿˣᵈ → [ Attention ] → Contextualised X' → [FFN] → … → logits
                                        ^^^^^^^^^^^^^
                                        THIS section
```

The attention layer sits between the embedding layer (§02) and the feed-forward network. It takes position-encoded embeddings and outputs context-aware representations where each token "knows about" the entire sequence.

---

## 2. Formal Definitions

### 2.1 Input Sequence

The input to attention is a matrix of token embeddings:

$$X \in \mathbb{R}^{n \times d}$$

where:

- **n** = sequence length (number of tokens)
- **d** = model dimension (embedding dimension, also called `d_model`)
- **xᵢ** ∈ ℝᵈ is row i — the embedding vector for token i

In practice, inputs are batched: X ∈ ℝᴮˣⁿˣᵈ where B is the batch size. We omit the batch dimension throughout for clarity.

### 2.2 Projection Matrices

Four learned weight matrices define the attention operation:

| Matrix | Dimensions   | Role                           |
| ------ | ------------ | ------------------------------ |
| Wᵠ     | ℝᵈˣᵈₖ        | Projects embeddings to queries |
| Wᴷ     | ℝᵈˣᵈₖ        | Projects embeddings to keys    |
| Wᵛ     | ℝᵈˣᵈᵥ        | Projects embeddings to values  |
| Wᴼ     | ℝ^(H·dᵥ) × d | Projects concatenated heads    |

All four matrices are learned by gradient descent during training. In standard Transformers, dₖ = dᵥ = d/H where H is the number of attention heads.

### 2.3 Q, K, V Matrices

The three central matrices are computed by linear projection of the input:

$$Q = XW^Q \in \mathbb{R}^{n \times d_k} \qquad K = XW^K \in \mathbb{R}^{n \times d_k} \qquad V = XW^V \in \mathbb{R}^{n \times d_v}$$

Each row qᵢ, kᵢ, vᵢ corresponds to one token position. The query says "what I'm looking for," the key says "what I offer," and the value says "what I contribute." The dot product qᵢ · kⱼ measures compatibility between position i's query and position j's key.

### 2.4 Attention Types

| Type                        | Q source | K, V source | Mask     | Used in                     |
| --------------------------- | -------- | ----------- | -------- | --------------------------- |
| **Self-attention**          | X        | X           | None     | Encoder, bidirectional      |
| **Causal self-attention**   | X        | X           | Causal   | Autoregressive decoders     |
| **Cross-attention**         | X        | Y (≠ X)     | Optional | Encoder-decoder, multimodal |
| **Bidirectional self-attn** | X        | X           | None     | BERT-style encoders         |

- **Self-attention**: Q, K, V all derived from the same sequence X.
- **Cross-attention**: Q from one sequence (decoder), K and V from another (encoder output). Used in encoder-decoder models (T5, BART) and multimodal models (GPT-4V, Gemini).
- **Causal self-attention**: self-attention with an upper-triangular mask that prevents future tokens from being attended to — required for autoregressive generation (GPT, LLaMA, Mistral).

---

## 3. Scaled Dot-Product Attention — Full Derivation

### 3.1 Step-by-Step Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

This decomposes into five concrete steps:

**Step 1 — Raw scores (dot products):**

$$S = QK^\top \in \mathbb{R}^{n \times n}, \quad S_{ij} = q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$$

Each entry S\_{ij} measures how much query i is compatible with key j. This is an n × n matrix: every token scores against every other token.

**Step 2 — Scaling:**

$$\tilde{S} = \frac{S}{\sqrt{d_k}}$$

Divides all scores by √dₖ to prevent softmax saturation (see §3.2 for full proof).

**Step 3 — Masking (optional):**

$$\hat{S}_{ij} = \tilde{S}_{ij} + M_{ij}, \quad M_{ij} \in \{0, -\infty\}$$

Adds a mask matrix M. Positions with M\_{ij} = −∞ will have zero attention weight after softmax. See §4 for mask types.

**Step 4 — Softmax (row-wise normalisation):**

$$A_{ij} = \frac{\exp(\hat{S}_{ij})}{\sum_{j'=1}^{n} \exp(\hat{S}_{ij'})}, \quad \sum_j A_{ij} = 1 \;\;\forall\, i$$

Each row of A is a probability distribution over positions. A\_{ij} is the attention weight: how much token i attends to token j.

**Step 5 — Weighted aggregation of values:**

$$O = AV \in \mathbb{R}^{n \times d_v}, \quad o_i = \sum_{j=1}^{n} A_{ij} v_j$$

The output for each token is a weighted average of all value vectors, where the weights are the attention probabilities.

### 3.2 Why √dₖ Scaling — Full Proof

**Claim**: Without scaling, the variance of dot-product scores grows linearly with dₖ, causing softmax to saturate.

**Proof**:

Assume query and key elements are i.i.d. random variables with zero mean and unit variance: q*{il}, k*{jl} ~ N(0, 1).

For a single product term: E[q_{il} · k_{jl}] = E[q_{il}] · E[k_{jl}] = 0 (since they're independent and zero-mean).

The variance of a single product: Var(q*{il} · k*{jl}) = E[q²_{il}] · E[k²_{jl}] = 1 · 1 = 1.

The dot product is a sum of dₖ independent terms:

$$q_i \cdot k_j = \sum_{l=1}^{d_k} q_{il} k_{jl}$$

By linearity of variance for independent terms:

$$\text{Var}(q_i \cdot k_j) = d_k, \quad \text{std}(q_i \cdot k_j) = \sqrt{d_k}$$

**Consequence without scaling**: For dₖ = 64, the standard deviation is 8. Softmax receives inputs with std ≈ 8, meaning typical differences between scores are ~16. With such large differences, softmax produces near-one-hot distributions (e.g., [0.999, 0.001, 0.000, ...]). Near-one-hot softmax → vanishing gradients for all non-attended tokens.

**Fix**: Divide by √dₖ to restore unit variance:

$$\text{Var}\!\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) = \frac{\text{Var}(q_i \cdot k_j)}{d_k} = \frac{d_k}{d_k} = 1$$

Now softmax inputs have std ≈ 1, producing well-spread attention weights with healthy gradients.

| dₖ   | std(q·k) unscaled | std(q·k/√dₖ) scaled | Softmax behaviour |
| ---- | ----------------- | ------------------- | ----------------- |
| 4    | 2.0               | 1.0                 | Smooth            |
| 16   | 4.0               | 1.0                 | Smooth            |
| 64   | 8.0               | 1.0                 | Smooth            |
| 256  | 16.0              | 1.0                 | Smooth            |
| 1024 | 32.0              | 1.0                 | Smooth            |
| 4096 | 64.0              | 1.0                 | Smooth            |

### 3.3 Softmax Properties

- **Output is a probability simplex**: Aᵢ ∈ Δⁿ (non-negative, sums to 1 per row)
- **Numerically stable computation**: softmax(z) = softmax(z − max(z)); subtracting max prevents overflow in exp
- **Temperature scaling**: softmax(S/τ) where τ is temperature
  - τ < 1 → sharpens attention (more peaked)
  - τ > 1 → smooths attention (more uniform)
  - τ → 0 → argmax (one-hot, non-differentiable) — this is hard attention
  - τ → ∞ → uniform distribution (1/n for all positions)
- **Jacobian**: ∂softmax(z)ᵢ/∂zⱼ = softmax(z)ᵢ(δᵢⱼ − softmax(z)ⱼ) — needed for backpropagation

### 3.4 Attention as Kernel Regression

Attention can be rewritten as a Nadaraya-Watson kernel regression:

$$o_i = \frac{\sum_j k(q_i, k_j)\, v_j}{\sum_j k(q_i, k_j)}$$

where the kernel is:

$$k(q, k) = \exp\!\left(\frac{q \cdot k}{\sqrt{d_k}}\right)$$

This perspective reveals attention as a **nonparametric regression** over the value vectors, with the kernel determining the similarity measure. It also opens the door to **linear attention**: approximate the kernel with a feature map φ such that k(q, k) ≈ φ(q) · φ(k), then exploit associativity:

$$O \approx \frac{\phi(Q)(\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}} \quad \text{(compute } \phi(K)^\top V \in \mathbb{R}^{d \times d} \text{ first → O(nd²) not O(n²d))}$$

---

## 4. Masking — Complete Treatment

### 4.1 Causal Mask (Autoregressive)

Prevents position i from attending to any future position j > i:

$$M_{ij}^{\text{causal}} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

This is an upper-triangular matrix of −∞ (with diagonal and below set to 0). After softmax, A\_{ij} = 0 exactly for all j > i.

**Required for**: all autoregressive generation — GPT, LLaMA, Mistral, Falcon, Gemma, Qwen, Phi.

```
Example (n=4):

     j=0   j=1   j=2   j=3
i=0 [ 0    -∞    -∞    -∞  ]
i=1 [ 0     0    -∞    -∞  ]
i=2 [ 0     0     0    -∞  ]
i=3 [ 0     0     0     0  ]
```

### 4.2 Padding Mask

When batching sequences of different lengths, shorter sequences are padded to match the longest. Pad tokens are meaningless and should be ignored:

$$M_{ij}^{\text{pad}} = \begin{cases} 0 & \text{if token } j \text{ is real} \\ -\infty & \text{if token } j \text{ is a pad token} \end{cases}$$

Applied column-wise: pad tokens cannot be attended to by any position.

### 4.3 Prefix / Document Mask

In multi-document contexts (e.g., continued pretraining on packed sequences), a document mask prevents cross-document attention. The resulting attention matrix has a **block-diagonal** structure — each document only attends within itself.

### 4.4 Sliding Window Mask (Longformer / Mistral)

Token i only attends to positions within window [i − w, i + w]:

$$M_{ij}^{\text{window}} = \begin{cases} 0 & |i - j| \le w \\ -\infty & |i - j| > w \end{cases}$$

This reduces O(n²) to O(n · w), trading global context for efficiency. Used by Longformer, Mistral (w = 4096), and many long-context models.

### 4.5 Combining Multiple Masks

All masks are added before softmax:

$$\hat{S} = \frac{S}{\sqrt{d_k}} + M_{\text{causal}} + M_{\text{pad}} + M_{\text{window}}$$

Since −∞ + anything = −∞ and exp(−∞) = 0, masks compose naturally. Any position masked by any mask will have zero attention weight.

---

## 5. Multi-Head Attention (MHA)

### 5.1 Motivation

A single attention head computes one weighted combination of values. But different linguistic relationships require different attention patterns simultaneously:

| Head | Pattern example                 | What it learns                  |
| ---- | ------------------------------- | ------------------------------- |
| A    | Verb ↔ subject distance         | Syntactic dependency            |
| B    | "she" → "Alice" (earlier)       | Coreference resolution          |
| C    | Nearby tokens                   | Local context / n-gram patterns |
| D    | BOS / special tokens            | Global information aggregation  |
| E    | Rare/out-of-distribution tokens | Novelty detection               |

Multiple heads run in parallel on different projection subspaces. Each head independently learns which relationships to focus on.

### 5.2 Per-Head Computation

For head h ∈ {1, …, H}:

$$Q_h = XW_h^Q \in \mathbb{R}^{n \times d_k}, \quad K_h = XW_h^K \in \mathbb{R}^{n \times d_k}, \quad V_h = XW_h^V \in \mathbb{R}^{n \times d_v}$$

$$\text{head}_h = \text{Attention}(Q_h, K_h, V_h) = \text{softmax}\!\left(\frac{Q_h K_h^\top}{\sqrt{d_k}}\right) V_h \in \mathbb{R}^{n \times d_v}$$

Each head has its own projection matrices W_h^Q, W_h^K, W_h^V, so it projects the input into a different subspace.

### 5.3 Concatenation and Output Projection

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot W^O$$

- Concatenated output: ℝⁿˣ(H·dᵥ)
- Output projection: W^O ∈ ℝ^(H·dᵥ) × d
- Final output: ℝⁿˣᵈ — same shape as input

The output projection W^O mixes information across heads, allowing the model to combine patterns learned by different heads.

### 5.4 Dimension Constraints and Parameter Count

Standard constraint: dₖ = dᵥ = d/H (heads partition the model dimension).

| Component            | Parameters      | For d=4096, H=32 |
| -------------------- | --------------- | ---------------- |
| All Q projections    | d × dₖ × H = d² | 16,777,216       |
| All K projections    | d × dₖ × H = d² | 16,777,216       |
| All V projections    | d × dᵥ × H = d² | 16,777,216       |
| Output projection Wᴼ | H·dᵥ × d = d²   | 16,777,216       |
| **Total MHA params** | **4d²**         | **67,108,864**   |

**Key insight**: total MHA parameters = 4d², independent of the number of heads H. Changing H changes the per-head dimension but not the total parameter count.

Example: d = 4096, L = 32 layers → per-layer MHA ≈ 67M params → all-layer MHA ≈ 2.1B params.

### 5.5 Head Redundancy and Pruning

Michel et al. (2019) showed that most attention heads can be pruned with minimal performance loss:

- In BERT-base (12 heads per layer), typically only 4–6 heads matter per layer
- Some heads are nearly identical in their attention patterns
- Head pruning is used for inference efficiency

This redundancy motivates more efficient variants:

- **MQA** (§7.2): all heads share K, V → KV cache reduced by factor H
- **GQA** (§7.3): groups of heads share K, V → balanced quality-efficiency tradeoff

---

## 6. KV Cache — Inference Mathematics

### 6.1 What Is the KV Cache?

During autoregressive generation, each new token i must attend to all previous tokens 1…i:

- **Without cache**: recompute K and V for all past tokens at every step → O(n²) total compute per sequence
- **With cache**: store K and V matrices for all past tokens; only compute new token's Q, K, V → O(n) per token

At each step:

1. Compute Q, K, V for the new token only (1 row each)
2. Append new K, V rows to the cached K, V matrices
3. Compute attention: new Q against all cached K → scores → softmax → weighted sum of all cached V

### 6.2 KV Cache Memory Formula

$$\text{KV Cache (bytes)} = 2 \times n \times d \times L \times \text{bytes\_per\_element}$$

| Factor | Meaning                                           |
| ------ | ------------------------------------------------- |
| 2      | One K matrix + one V matrix                       |
| n      | Current sequence length (grows during generation) |
| d      | Model dimension (or d_head × H_kv for GQA)        |
| L      | Number of layers                                  |
| bytes  | 2 (fp16/bf16), 1 (int8), 0.5 (int4)               |

### 6.3 KV Cache at Scale

| Model              | d     | L   | n       | Precision | KV Cache Size |
| ------------------ | ----- | --- | ------- | --------- | ------------- |
| LLaMA-3 8B         | 4096  | 32  | 8,192   | fp16      | **4.0 GB**    |
| LLaMA-3 70B        | 8192  | 80  | 8,192   | fp16      | **17.2 GB**   |
| LLaMA-3 70B        | 8192  | 80  | 128,000 | fp16      | **268 GB**    |
| LLaMA-3.1 405B     | 16384 | 126 | 128,000 | fp16      | **1,056 GB**  |
| GPT-4 class (est.) | 12288 | 96  | 128,000 | fp16      | **580 GB**    |

The KV cache is often the **primary memory bottleneck** at inference — frequently exceeding the model weights themselves for long sequences.

> **Note on GQA**: LLaMA-3 models use Grouped-Query Attention (GQA) with fewer KV heads than query heads (e.g., 8 KV vs 64 query heads for 70B). With GQA, replace $d$ with $d_{\text{kv}} = n_{\text{kv\_heads}} \times d_{\text{head}}$ in the formula above. For LLaMA-3 70B with GQA, actual KV cache at $n{=}8192$ is ~2.5 GB (8× reduction). The table above shows MHA-equivalent values using $d = d_{\text{model}}$.

### 6.4 KV Cache Compression (2024–2026)

| Method                      | Year | Approach                                              | Reduction |
| --------------------------- | ---- | ----------------------------------------------------- | --------- |
| Quantised KV cache          | 2023 | Store K, V in int8 or int4                            | 2–4×      |
| StreamingLLM                | 2023 | Keep attention sink tokens + recent window            | Fixed mem |
| H2O (Heavy Hitter Oracle)   | 2023 | Evict entries with low cumulative attention           | 2–5×      |
| ScissorHands                | 2023 | Exploit attention persistence (keep important tokens) | 2–4×      |
| SnapKV                      | 2024 | Observation-based KV pruning; keep representative K,V | 3–6×      |
| KVSharer                    | 2024 | Share K, V across nearby layers                       | 2×        |
| Cross-Layer Attention (CLA) | 2024 | Adjacent layers share same KV cache entirely          | 2×        |
| MLA (DeepSeek)              | 2024 | Low-rank latent compression of K, V                   | ~10×      |

---

## 7. Attention Variants — Complete Taxonomy

### 7.1 Standard MHA

- H independent Q, K, V projections
- Full O(n²) attention matrix per head
- Most expressive but most memory-intensive
- Used by: original Transformer, BERT, GPT-2

### 7.2 Multi-Query Attention (MQA)

All H heads share **one** K and **one** V projection; each head has its own Q:

$$Q_h = XW_h^Q, \quad K = XW^K, \quad V = XW^V \quad \text{(shared across heads)}$$

- KV cache reduced by factor H (e.g., 32× for H=32)
- Parameter savings: 2d² → 2d·dₖ (small savings since dₖ = d/H)
- Small quality loss; significant inference speedup
- Noam Shazeer (2019); adopted by PaLM, Falcon, early Gemini

### 7.3 Grouped Query Attention (GQA)

Heads are divided into G groups; heads within a group share K and V:

| Setting | Configuration            | KV Cache Size | Quality |
| ------- | ------------------------ | ------------- | ------- |
| G = 1   | MQA (all share)          | Smallest      | Lowest  |
| G = 2   | 2 KV sets                | ↓             | ↑       |
| G = 4   | 4 KV sets                | ↓             | ↑       |
| G = 8   | 8 KV sets (2026 default) | ↓             | ↑       |
| G = H   | MHA (none shared)        | Largest       | Highest |

Ainslie et al. (2023). Adopted by LLaMA-2/3, Mistral, Gemma, Qwen2, Phi-3. **GQA with G=8 is the 2026 production default** for most models.

KV cache reduction factor: H/G relative to MHA.

### 7.4 Multi-head Latent Attention (MLA) — DeepSeek (2024)

Instead of caching full K, V matrices, compress them into a low-rank latent:

$$c = XW^{DKV} \in \mathbb{R}^{n \times d_c}, \quad d_c \ll d$$

At attention time, upproject back:

$$K = cW^{KU}, \quad V = cW^{VU}$$

Only c is cached instead of full K, V — massive memory savings. DeepSeek-V2/V3 achieve **~10× KV cache reduction** vs MHA ($d/d_c = 5120/512$; accounting for RoPE dimensions the full reduction is even larger). This was the key innovation enabling DeepSeek's cost-efficient inference at scale.

### 7.5 Sliding Window Attention (SWA)

Token i attends to positions [i − w, i + w] only:

- Complexity: O(n · w · d) vs O(n² · d)
- Global tokens (BOS, special) may attend to all positions
- Used by Longformer (2020), Mistral (w = 4096, 2023)
- Standard for long-context models; often combined with full attention at certain layers

### 7.6 Sparse Attention Patterns

| Pattern                 | Method                    | Complexity |
| ----------------------- | ------------------------- | ---------- |
| Local + strided         | Sparse Transformer (2019) | O(n√n)     |
| Local + global + random | BigBird (2020)            | O(n)       |
| Block-sparse            | LongT5 (2021)             | O(n)       |

BigBird proved that local + global + random attention is **Turing-complete** — theoretically as powerful as full attention.

### 7.7 Linear Attention

Replace softmax with a feature map φ:

$$k(q, k) \approx \phi(q) \cdot \phi(k)^\top$$

Exploit associativity: compute φ(K)ᵀV ∈ ℝᵈˣᵈ first → O(nd²) instead of O(n²d).

| Model      | Year | Approach                            | Per-token inference |
| ---------- | ---- | ----------------------------------- | ------------------- |
| Performer  | 2020 | FAVOR+ random feature approximation | O(d²)               |
| RetNet     | 2023 | Gated linear recurrence             | O(d²)               |
| RWKV-4/5/6 | 2023 | Linear attention as RNN             | O(d)                |

### 7.8 FlashAttention Family

| Version          | Year | Key Innovation                          | Speedup vs prior |
| ---------------- | ---- | --------------------------------------- | ---------------- |
| FlashAttention   | 2022 | Tiled SRAM computation; O(n) memory     | Baseline         |
| FlashAttention-2 | 2023 | Better thread parallelism               | 2× FA1           |
| FlashAttention-3 | 2024 | Hopper (H100) optimisation, FP8         | ~3× FA2          |
| Flash-Decoding   | 2023 | Parallelise across seq dim at inference | ~8× decoding     |

**FlashAttention does not change the mathematical result** — it computes exact attention. The innovation is purely in how the computation is scheduled on GPU hardware (tiling into SRAM, never materialising the full n × n attention matrix to HBM).

Standard in all serious training stacks: HuggingFace, Megatron-LM, NanoGPT, etc.

### 7.9 Hybrid Architectures (2024–2026)

| Model        | Year | Architecture                               | Context  |
| ------------ | ---- | ------------------------------------------ | -------- |
| Mamba        | 2023 | Pure selective SSM                         | O(n)     |
| Mamba-2      | 2024 | Structured SSM ≡ linear attention          | O(n)     |
| Jamba        | 2024 | Alternating Mamba + Transformer layers     | 256K     |
| Zamba        | 2024 | Hybrid with shared attention layers        | Variable |
| Falcon-Mamba | 2024 | First pure-Mamba at 7B competitive w/ Tfmr | O(n)     |
| Hymba        | 2024 | Parallel SSM + attention heads per layer   | O(n)     |

**2025–2026 trend**: Pure-attention dominance increasingly challenged by hybrid SSM-attention architectures that offer O(n) inference with competitive quality.

---

## 8. Positional Encodings in Attention

### 8.1 Why Attention Is Position-Agnostic

Self-attention is **permutation equivariant**: if you permute the input tokens, the output tokens are permuted in the same way (with correspondingly permuted attention weights). In particular, QKᵀ produces the same set of pairwise scores regardless of token order — attention has no built-in notion of position.

Position must be explicitly injected. Methods differ in **where** and **how** position information enters the computation.

### 8.2 Sinusoidal PE (Vaswani 2017)

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad \text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

- Added to embeddings before attention: x̃ᵢ = xᵢ + PE(i)
- Fixed (not learned); deterministic
- Each dimension oscillates at a different frequency: low dimensions = high frequency (distinguish nearby positions), high dimensions = low frequency (distinguish distant positions)
- **Key property**: PE(pos + k) can be expressed as a linear function of PE(pos) for any fixed offset k — this encodes relative position information
- Theoretically extrapolates to unseen lengths; empirically degrades past training length

### 8.3 Learned Absolute PE

$$P \in \mathbb{R}^{L_{\max} \times d} \quad \text{(learned like token embeddings)}$$

- x̃ᵢ = xᵢ + pᵢ; position embedding added before attention
- Cannot generalise beyond L_max (training sequence length)
- Used by BERT, GPT-2, early GPT-3

### 8.4 Relative Position Encodings (Shaw 2018, T5)

Modify the attention score to include relative position:

$$S_{ij} = q_i \cdot k_j + q_i \cdot a_{i-j}$$

where a\_{i-j} ∈ ℝᵈₖ is a learned embedding for relative offset (i − j).

**T5 relative position bias**: add a scalar bias b\_{i-j} to the attention score. Only the relative offset matters; better length generalisation than absolute PE.

### 8.5 RoPE — Rotary Positional Encoding (Su et al. 2021)

RoPE applies a position-dependent rotation matrix Rₘ to Q and K vectors:

$$\text{score}(i, j) = (R_i q_i)^\top (R_j k_j) = q_i^\top R_i^\top R_j k_j = q_i^\top R_{j-i} k_j$$

The rotation matrix Rₘ is block-diagonal with 2×2 rotation blocks:

$$R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}, \quad \theta_i = 10000^{-2i/d}$$

**Properties**:

- Relative position (j − i) emerges naturally from the dot product — no additional parameters
- No change to model dimension; applied to Q and K only
- Standard for all modern models: LLaMA 1/2/3, Mistral, Gemma, Qwen, Phi, Falcon

**RoPE scaling for long context**:

| Method               | Year | Approach                                   | Context extension |
| -------------------- | ---- | ------------------------------------------ | ----------------- |
| Linear scaling (PI)  | 2023 | Divide position by factor s                | Moderate          |
| NTK-aware scaling    | 2023 | Scale base frequency; preserve high-freq   | Good              |
| YaRN                 | 2023 | Non-uniform scaling + magnitude correction | Strong            |
| LongRoPE (Microsoft) | 2024 | Non-uniform per-dimension scaling          | 2M tokens         |

### 8.6 ALiBi — Attention with Linear Biases (Press et al. 2021)

No positional vectors in embeddings at all. Instead, add a distance penalty directly to attention logits:

$$S_{ij} = q_i \cdot k_j - m_h \cdot |i - j|$$

where mₕ is a head-specific slope (geometric sequence: m₁ = 2^{−8/H}, m₂ = 2^{−16/H}, …).

- Strong extrapolation: train on 1K tokens, infer on 4K+
- Used by BLOOM, MPT

### 8.7 Context Length Timeline (2020–2026)

| Year | Model          | Max Context              |
| ---- | -------------- | ------------------------ |
| 2020 | GPT-3          | 2,048                    |
| 2022 | GPT-3.5        | 4,096                    |
| 2023 | Claude 2       | 100,000                  |
| 2023 | GPT-4-Turbo    | 128,000                  |
| 2024 | Gemini 1.5 Pro | 1,000,000                |
| 2024 | LLaMA-3.1      | 128,000                  |
| 2025 | Gemini 2.0     | 2,000,000                |
| 2026 | Leading models | 1M–10M (active research) |

---

## 9. Complexity Analysis

### 9.1 Standard MHA Complexity

| Operation          | Time Complexity  | Space Complexity |
| ------------------ | ---------------- | ---------------- |
| Q, K, V projection | O(nd²)           | O(nd)            |
| Score matrix QKᵀ   | O(n²dₖ)          | O(n²)            |
| Softmax            | O(n²)            | O(n²)            |
| Output AV          | O(n²dᵥ)          | O(ndᵥ)           |
| Output projection  | O(nd²)           | O(nd)            |
| **Total**          | **O(n²d + nd²)** | **O(n² + nd)**   |

- For n ≪ d: dominated by O(nd²) — the projection cost
- For n ≫ d: dominated by O(n²d) — the attention matrix
- Crossover at n ≈ d; typical models: d = 4096, so crossover at ~4096 tokens

### 9.2 Space Bottleneck

The attention matrix A ∈ ℝⁿˣⁿ per head per layer:

- H heads × L layers × n² floats each
- GPT-4 class (H=96, L=96, n=8192, fp16): 96 × 96 × 8192² × 2 bytes ≈ **100 TB**
- Impossible to materialise; this is exactly what FlashAttention solves

### 9.3 FlashAttention Complexity

| Metric        | Standard | FlashAttention |
| ------------- | -------- | -------------- |
| Time          | O(n²d)   | O(n²d)         |
| Memory        | O(n²)    | **O(n)**       |
| IO complexity | O(n²)    | O(n²d / M)     |

Where M = SRAM size. FlashAttention achieves the same result (exact attention) with dramatically less memory by tiling the computation into SRAM-sized blocks and never writing the full n × n matrix to GPU HBM.

In practice 2–4× faster despite no asymptotic time improvement, because GPU compute is bottlenecked by memory bandwidth, not FLOPs.

### 9.4 Efficient Attention Comparison

| Method           | Time   | Memory   | Quality            | KV Cache  |
| ---------------- | ------ | -------- | ------------------ | --------- |
| Standard MHA     | O(n²d) | O(n²)    | Exact              | O(nLd)    |
| FlashAttention   | O(n²d) | O(n)     | Exact              | O(nLd)    |
| Sliding Window   | O(nwd) | O(nw)    | Local only         | O(wLd)    |
| Linear Attention | O(nd²) | O(d²)    | Approximate        | O(Ld²)    |
| MQA              | O(n²d) | O(n²/H)  | Exact              | O(nLd/H)  |
| GQA (G groups)   | O(n²d) | O(n²G/H) | Exact              | O(nLdG/H) |
| MLA              | O(n²d) | O(n²)    | Exact (approx K,V) | O(nLdₖ)   |

---

## 10. Attention Score Analysis and Interpretability

### 10.1 Attention Entropy

The entropy of the attention distribution for query position i:

$$H(A_i) = -\sum_j A_{ij} \log A_{ij} \in [0, \log n]$$

| Entropy value | Interpretation                              |
| ------------- | ------------------------------------------- |
| 0             | All attention on one token (one-hot)        |
| log n         | Uniform attention over all positions        |
| Low           | Focused heads (specific dependency learned) |
| High          | Diffuse heads (information broadcasting)    |

**Observations**: Early layers tend to have higher entropy (diffuse attention); later layers tend to have lower entropy (focused attention). Entropy varies significantly by head type and token type.

### 10.2 Attention Sink Phenomenon

**Observation**: The BOS token (position 0) receives disproportionately high attention mass across many heads and layers, regardless of content.

**Reason**: Softmax must assign probability mass somewhere. BOS is always present and semantically "safe" — it doesn't carry misleading content. This is a mathematical artefact of softmax normalisation, not a semantic signal.

**Exploitation**: StreamingLLM (2023) always keeps BOS in the KV cache, enabling infinite-length generation with fixed memory. Any always-visible token can serve as an attention sink — it doesn't have to be BOS.

### 10.3 Induction Heads

A key mechanistic interpretability discovery (Olsson et al. 2022):

**Pattern**: If the sequence contains [A][B] ... [A], induction heads help predict [B] after the second occurrence of [A].

**Mechanism** (two-head composition):

1. **Head A** (previous-token head): at position of second [A], copies information about what followed [A] elsewhere
2. **Head B** (induction head): uses that information to promote [B] in the prediction

Induction heads are believed to be a primary mechanism for **in-context learning** (ICL). They form at a specific training step ("phase change") and are reproducible across model sizes.

### 10.4 Attention vs Gradient Attribution

⚠️ **Attention weights ≠ causal importance** (Jain & Wallace 2019, Wiegreffe & Pinter 2019):

- High attention can be misleading: a token can be attended to heavily but contribute little to the output
- Attention weights show which tokens were looked at, not which were important for the final answer
- For faithful interpretability, use gradient-based methods: Integrated Gradients, SHAP, attention rollout
- Raw attention is useful for visualisation but should not be treated as ground-truth importance

### 10.5 Attention Rollout

Approximate information propagation through L layers by aggregating attention matrices:

$$A_{\text{rollout}} = \prod_{l=1}^{L} \hat{A}_l, \quad \hat{A}_l = 0.5 A_l + 0.5 I$$

The identity term accounts for residual connections (each layer adds to the residual stream rather than replacing it). This gives a rough estimate of which input tokens influence each output token.

### 10.6 Circuit Analysis (Mechanistic Interpretability)

Attention heads can be identified as functional circuits:

- **IOI circuit** (Indirect Object Identification, Wang et al. 2022): specific heads identified for detecting "Alice gave the book to Bob" patterns
- **Compositional circuits**: head outputs feed into later head keys/queries, forming multi-step reasoning chains
- **Tools**: activation patching, path patching, causal tracing
- Active research area (Anthropic, EleutherAI, DeepMind, MATS) — 2024–2026

---

## 11. Attention in the Full Transformer Block

### 11.1 Residual Stream Perspective

The residual stream x^l ∈ ℝᵈ passes through the network; each sublayer adds a delta:

$$x^{l+1} = x^l + \text{MHA}(\text{LN}(x^l))$$
$$x^{l+2} = x^{l+1} + \text{FFN}(\text{LN}(x^{l+1}))$$

The residual stream is the central communication channel (Elhage et al. 2021). Attention writes to the stream; the stream accumulates contributions from all layers. The final representation is the initial embedding plus the sum of all attention and FFN outputs.

### 11.2 Layer Normalisation Variants

| Variant       | Formula                              | When used               |
| ------------- | ------------------------------------ | ----------------------- |
| **Post-norm** | LN(x + sublayer(x))                  | Vaswani 2017 (original) |
| **Pre-norm**  | x + sublayer(LN(x))                  | Modern default          |
| **RMSNorm**   | x / RMS(x) · γ, RMS(x) = √(1/d Σxᵢ²) | 2025–2026 standard      |

- **Post-norm**: unstable for deep networks (L > 12); requires learning-rate warmup
- **Pre-norm**: stable training; enables much deeper networks; used by all modern LLMs
- **RMSNorm**: no mean subtraction → faster; used by LLaMA, Gemma, Mistral, Qwen

### 11.3 Parameter Count Per Block

| Component             | Parameters (MHA) | Parameters (GQA, G groups) |
| --------------------- | ---------------- | -------------------------- |
| Q projection          | d²               | d²                         |
| K projection          | d²               | d² × G/H = d·dₖ·G          |
| V projection          | d²               | d² × G/H = d·dₖ·G          |
| Output projection Wᴼ  | d²               | d²                         |
| FFN W₁ (up-project)   | d × 4d = 4d²     | 4d²                        |
| FFN W₂ (down-project) | 4d × d = 4d²     | 4d²                        |
| LayerNorm/RMSNorm     | ~2d (negligible) | ~2d                        |
| **MHA subtotal**      | **4d²**          | **(2 + 2G/H)d²**           |
| **Full block total**  | **~12d²**        | **~(10 + 2G/H)d²**         |

### 11.4 Total Model Parameter Estimate

$$\text{Total params} \approx 12Ld^2 + 2Nd$$

where L = number of layers, d = model dimension, N = vocabulary size. With **weight tying** (shared embedding and LM head), the $2Nd$ term reduces to $Nd$.

This formula is accurate to ~10% for most Transformer LLMs:

| Model       | L   | d    | N       | Formula estimate   | Actual |
| ----------- | --- | ---- | ------- | ------------------ | ------ |
| GPT-2 Small | 12  | 768  | 50,257  | 124M (weight-tied) | 124M   |
| LLaMA-3 8B  | 32  | 4096 | 128,000 | 6.7B               | 8.0B   |
| LLaMA-3 70B | 80  | 8192 | 128,000 | 64.7B              | 70.6B  |

Discrepancy comes from: bias terms, gate projections in SwiGLU FFN, GQA adjustments, and LM head.

---

## 12. 2024–2026 Research Frontiers

### 12.1 Long Context Attention

- RoPE-based scaling enabling 128K–2M token contexts
- Ring attention (2023): distribute attention across multiple GPUs along sequence dimension
- Striped attention: load-balanced ring attention for causal models
- 2025: 1M-token context models becoming practical (Gemini 2.0, Claude 3.5 family)

### 12.2 Mixture of Experts (MoE) + Attention

- MoE replaces dense FFN; attention typically remains dense
- Some work on sparse attention + sparse FFN (SparseMoE)
- DeepSeek-V3 (2024): MoE with MLA attention; 671B params, 37B active per token

### 12.3 Attention-Free and Hybrid Models

- State-space models (SSM): Mamba, S4, H3 — O(n) inference
- Linear recurrences: RWKV, RetNet — O(1) per-token inference, O(n) training
- Hybrid: interleave attention layers with SSM/linear layers
- 2025–2026: emerging consensus that full Transformers may not be needed for all contexts

### 12.4 Subquadratic Attention

- HyperAttention (2023): random sampling for approximate O(n√n) attention
- Monarch Mixer: butterfly matrices instead of dense Q, K, V projections
- GQA + sliding window + sink tokens: pragmatic O(n) for most tokens, O(n²) for sink

### 12.5 FP8 and Low-Precision Attention

- H100/B200 GPUs support FP8 matmul natively
- FlashAttention-3 supports FP8 Q, K, V with FP32 accumulation
- Quantised attention: K, V cached in int4/int8 for inference
- 2025–2026: FP8 training becoming standard for frontier models

---

## 13. Common Mistakes

| Mistake                                          | Why It's Wrong                                                  | Fix                                                              |
| ------------------------------------------------ | --------------------------------------------------------------- | ---------------------------------------------------------------- |
| "Attention weights = token importance"           | Attention is not causal; high weight ≠ high influence           | Use gradient attribution (IG, SHAP) for importance               |
| "More heads always better"                       | Heads are redundant; most prunable with minimal loss            | Tune H to d; GQA reduces KV heads without hurting quality        |
| "Attention is O(n)"                              | It's O(n²d); the quadratic bottleneck is real                   | Use FlashAttention + GQA + sliding window for long contexts      |
| "All attention is self-attention"                | Cross-attention (encoder-decoder, multimodal) is distinct       | Distinguish architecture type before analysis                    |
| "Causal mask is optional for decoders"           | Future tokens leaking = data contamination, trivial loss        | Always verify causal mask in autoregressive code                 |
| "Pre-norm and post-norm are equivalent"          | Post-norm unstable for L > 12; pre-norm enables deeper networks | Default to pre-norm + RMSNorm for any serious model              |
| "KV cache is free"                               | KV cache dominates memory at long context and large batch       | Plan KV budget; use GQA, quantised cache, or eviction            |
| "RoPE works at any length"                       | RoPE degrades past training length without scaling              | Apply YaRN / NTK-aware scaling for context extension             |
| "Hybrid models are worse than pure Transformers" | Mamba-2, Jamba match or beat Transformers at efficiency         | Evaluate hybrids; pure-attention dominance increasingly outdated |

---

## 14. Exercises

See [exercises.ipynb](exercises.ipynb) for full implementations with scaffolds and solutions.

### Exercise 1: Manual Scaled Dot-Product Attention

For n=3, dₖ=2, hand-compute Q, K, V → scores → scale → softmax → output. Verify that each output row is a weighted average of value vectors with weights summing to 1.

### Exercise 2: √dₖ Scaling Proof (Numerical)

If q, k ~ N(0,1) in ℝᵈₖ, show that Var(q·k) = dₖ. Verify numerically for dₖ ∈ {1, 4, 64, 512}. Show that scaling by √dₖ restores unit variance.

### Exercise 3: Causal Mask Implementation

Construct a 4×4 causal mask M. Apply to a random score matrix. Verify A\_{ij} = 0 for all j > i after softmax. Implement sliding-window mask and show combined mask output.

### Exercise 4: Parameter Counting

For LLaMA-3 8B (d=4096, H=32, GQA G=8, L=32, N=128K): compute MHA params per layer, FFN params per layer, total model params. Compare to the reported 8B.

### Exercise 5: KV Cache Memory Calculator

Compute KV cache size for LLaMA-3 70B at n=8192, n=128K in fp16 and int8. Compare with model weight size. Show when KV cache exceeds model weights.

### Exercise 6: Attention Entropy Analysis

Compute H(A) for: (a) uniform attention over 4 positions, (b) peaked attention [0.97, 0.01, 0.01, 0.01], (c) one-hot [1, 0, 0, 0]. Interpret what each pattern means for information flow.

### Exercise 7: RoPE in Attention

For d=4, position m=2, θ=(1.0, 0.01): compute the block-diagonal rotation matrix Rₘ. Apply to q=(1,0,1,0). Verify that dot(R_m q, R_n k) depends only on (m−n).

### Exercise 8: FlashAttention Tiling

Sketch why tiling avoids materialising the full A ∈ ℝⁿˣⁿ matrix. Implement a block-tiled attention that processes the attention matrix in tiles of size B, storing only one tile at a time.

---

## 15. Why This Matters for AI (2026 Perspective)

| Aspect               | Impact                                                                                            |
| -------------------- | ------------------------------------------------------------------------------------------------- |
| **Long context**     | 1M+ token windows now practical; retrieval-augmentation partially replaced by in-context learning |
| **Inference cost**   | KV cache is the memory bottleneck; GQA/MLA/quantisation are production necessities                |
| **Interpretability** | Induction heads, IOI circuits — mechanistic understanding of LLM behaviour emerging               |
| **Safety**           | Attention circuits identified as loci for deceptive or unwanted behaviour                         |
| **Fine-tuning**      | LoRA adapts attention projections (Wᵠ, Wᴷ, Wᵛ, Wᴼ) with rank-r updates                            |
| **Multimodal**       | Cross-attention connects vision tokens to language tokens in GPT-4V, Gemini, LLaVA                |
| **Efficiency**       | FlashAttention-3 + GQA + FP8 = 3–5× training cost reduction vs 2022 baseline                      |
| **Hybrid future**    | Pure-attention dominance challenged; SSM-attention hybrids competitive at scale                   |

---

## Further Reading

### Papers

- Vaswani et al. "Attention Is All You Need" (2017)
- Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
- Dao "FlashAttention-2: Faster Attention with Better Parallelism" (2023)
- Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
- Press et al. "Train Short, Test Long: Attention with Linear Biases" (2021)
- Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- DeepSeek-AI "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (2024)
- Olsson et al. "In-context Learning and Induction Heads" (2022)
- Michel et al. "Are Sixteen Heads Really Better than One?" (2019)
- Shazeer "Fast Transformer Decoding: One Write-Head is All You Need" (2019)

### Implementations

- [NanoGPT](https://github.com/karpathy/nanoGPT) — minimal GPT implementation
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) — official FlashAttention implementation
- [LLaMA](https://github.com/meta-llama/llama) — Meta's LLaMA with GQA + RoPE

---

## Conceptual Bridge

Attention transforms positionally-encoded token embeddings into **context-aware representations**. Each token now "knows about" the full sequence via weighted aggregation. The attention output X' has the same shape as the input X, but encodes rich inter-token relationships.

**Next**: [Positional Encodings](../04-Positional-Encodings/notes.md) — how Transformers encode token order. Without positional information, attention is permutation-invariant and has no sense of sequence structure. We cover sinusoidal, learned, RoPE, and ALiBi approaches.

```
X ∈ ℝⁿˣᵈ → [MHA] → X' ∈ ℝⁿˣᵈ → [FFN] → X'' ∈ ℝⁿˣᵈ → … → logits
              ^^^^^
         THIS section
```

---

[← Embedding Space Math](../02-Embedding-Space-Math/notes.md) | [Home](../../README.md) | [Positional Encodings →](../04-Positional-Encodings/notes.md)
