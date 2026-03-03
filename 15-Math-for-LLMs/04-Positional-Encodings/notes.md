[← Attention Mechanism Math](../03-Attention-Mechanism-Math/notes.md) | [Home](../../README.md) | [Language Model Probability →](../05-Language-Model-Probability/notes.md)

---

# Positional Encodings

> _"Without positional encoding, a Transformer is a bag-of-words model with a matrix-multiply budget — it knows what words are present but has no idea where they are."_

## Overview

Attention is permutation-invariant: if you shuffle the input tokens, the attention scores are identical (up to reordering). This means a Transformer with no positional signal cannot distinguish "the cat sat on the mat" from "mat the on sat cat the." Positional encoding breaks this symmetry by injecting information about each token's position into the representation, either by adding a vector to the embedding, rotating the query and key vectors, or biasing the attention logits. This section derives sinusoidal PE from first principles, proves the linear-transformation property between positions, covers learned absolute PE, relative PE (Shaw, Transformer-XL, T5 bias), ALiBi, RoPE with full rotation-matrix derivation, all RoPE scaling methods for long context (position interpolation, NTK-aware, YaRN, LongRoPE), modern alternatives (xPos, FIRE, CoPE, NoPE), PE in different architecture families, formal length-generalisation analysis (why models fail beyond training length, the lost-in-the-middle phenomenon), implementation details with numerical stability, and the 2024–2026 research frontier where 1M–10M token contexts make PE design the critical bottleneck.

## Prerequisites

- Linear algebra: rotation matrices, block-diagonal matrices, dot products, orthogonality
- Trigonometry: sin/cos identities, angle addition formulas
- Calculus: frequency analysis, wavelength interpretation
- Completed: [01-Tokenization-Math](../01-Tokenization-Math/notes.md), [02-Embedding-Space-Math](../02-Embedding-Space-Math/notes.md), and [03-Attention-Mechanism-Math](../03-Attention-Mechanism-Math/notes.md)

## Companion Notebooks

| Notebook                           | Description                                                                                     |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Sinusoidal PE, RoPE, ALiBi, frequency analysis, extrapolation, YaRN scaling, PE comparison      |
| [exercises.ipynb](exercises.ipynb) | Manual PE computation, dot-product decay, RoPE 2D, ALiBi slopes, frequency analysis, YaRN zones |

## Learning Objectives

After completing this section, you will:

- Prove that attention is permutation-invariant and explain why positional encoding is necessary
- Derive sinusoidal positional encoding and prove the linear-transformation property PE(pos+k) = M_k · PE(pos)
- Compute and visualise the frequency structure of sinusoidal PE across all dimension pairs
- Implement learned absolute PE and analyse the undertrained-tail problem
- Explain relative PE variants (Shaw, Transformer-XL, T5 bias) and their parameter/complexity tradeoffs
- Derive ALiBi from first principles, compute head-specific slopes, and prove its extrapolation mechanism
- Derive RoPE: build 2D rotation matrices, extend to d-dimensional block-diagonal form, verify the relative-position property
- Implement efficient RoPE without constructing the full rotation matrix
- Analyse RoPE frequency structure and explain why high-frequency dimensions fail at long contexts
- Apply RoPE scaling methods (position interpolation, NTK-aware, YaRN, LongRoPE) and compare their tradeoffs
- Discuss modern PE alternatives: xPos, FIRE, CoPE, NoPE
- Explain how PE design affects length generalisation, the lost-in-the-middle phenomenon, and effective context length

## Table of Contents

- [Positional Encodings](#positional-encodings)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 The Core Problem](#11-the-core-problem)
    - [1.2 What Positional Encoding Must Do](#12-what-positional-encoding-must-do)
    - [1.3 The Design Space](#13-the-design-space)
    - [1.4 Historical Timeline](#14-historical-timeline)
    - [1.5 Pipeline Position](#15-pipeline-position)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 The Permutation Invariance Problem — Formal](#21-the-permutation-invariance-problem--formal)
    - [2.2 Absolute vs Relative Position](#22-absolute-vs-relative-position)
    - [2.3 Additive vs Rotational Injection](#23-additive-vs-rotational-injection)
    - [2.4 Extrapolation vs Interpolation](#24-extrapolation-vs-interpolation)
  - [3. Sinusoidal Positional Encoding (Original Transformer)](#3-sinusoidal-positional-encoding-original-transformer)
    - [3.1 Formula](#31-formula)
    - [3.2 Geometric Interpretation](#32-geometric-interpretation)
    - [3.3 Frequency Structure](#33-frequency-structure)
    - [3.4 Linear Relationship Between Positions](#34-linear-relationship-between-positions)
    - [3.5 Properties](#35-properties)
    - [3.6 Dot Product Between Positions](#36-dot-product-between-positions)
  - [4. Learned Absolute Positional Embeddings](#4-learned-absolute-positional-embeddings)
    - [4.1 Definition](#41-definition)
    - [4.2 Initialisation](#42-initialisation)
    - [4.3 Properties](#43-properties)
    - [4.4 Undertrained Tail Problem](#44-undertrained-tail-problem)
  - [5. Relative Positional Encodings](#5-relative-positional-encodings)
    - [5.1 Motivation](#51-motivation)
    - [5.2 Shaw et al. (2018) — Relative PE in Keys](#52-shaw-et-al-2018--relative-pe-in-keys)
    - [5.3 Transformer-XL (Dai et al. 2019)](#53-transformer-xl-dai-et-al-2019)
    - [5.4 T5 Relative Bias (Raffel et al. 2019)](#54-t5-relative-bias-raffel-et-al-2019)
    - [5.5 Bucket Formula (T5)](#55-bucket-formula-t5)
  - [6. ALiBi — Attention with Linear Biases](#6-alibi--attention-with-linear-biases)
    - [6.1 Core Idea (Press et al. 2021)](#61-core-idea-press-et-al-2021)
    - [6.2 Slope Schedule](#62-slope-schedule)
    - [6.3 Extrapolation Mechanism](#63-extrapolation-mechanism)
    - [6.4 Properties](#64-properties)
  - [7. RoPE — Rotary Positional Encoding](#7-rope--rotary-positional-encoding)
    - [7.1 Core Idea (Su et al. 2021)](#71-core-idea-su-et-al-2021)
    - [7.2 2D Case — Building Intuition](#72-2d-case--building-intuition)
    - [7.3 General d-Dimensional RoPE](#73-general-d-dimensional-rope)
    - [7.4 Efficient Implementation](#74-efficient-implementation)
    - [7.5 Key Properties](#75-key-properties)
    - [7.6 Frequency Analysis of RoPE](#76-frequency-analysis-of-rope)
    - [7.7 RoPE Used By](#77-rope-used-by)
  - [8. RoPE Scaling for Long Context](#8-rope-scaling-for-long-context)
    - [8.1 The Problem](#81-the-problem)
    - [8.2 Linear Scaling (Position Interpolation, Chen et al. 2023)](#82-linear-scaling-position-interpolation-chen-et-al-2023)
    - [8.3 NTK-Aware Scaling (Community 2023, bloc97)](#83-ntk-aware-scaling-community-2023-bloc97)
    - [8.4 YaRN — Yet Another RoPE extensioN (Peng et al. 2023)](#84-yarn--yet-another-rope-extension-peng-et-al-2023)
    - [8.5 LongRoPE (Ding et al. 2024, Microsoft)](#85-longrope-ding-et-al-2024-microsoft)
    - [8.6 RoPE Base Scaling Comparison](#86-rope-base-scaling-comparison)
  - [9. Other Modern Positional Schemes](#9-other-modern-positional-schemes)
    - [9.1 xPos (Sun et al. 2022)](#91-xpos-sun-et-al-2022)
    - [9.2 FIRE — Functional Interpolation for Relative Encodings (Li et al. 2023)](#92-fire--functional-interpolation-for-relative-encodings-li-et-al-2023)
    - [9.3 CoPE — Contextualised Positional Encoding (Olsson et al. 2024)](#93-cope--contextualised-positional-encoding-olsson-et-al-2024)
    - [9.4 NoPE — No Positional Encoding (Kazemnejad et al. 2023)](#94-nope--no-positional-encoding-kazemnejad-et-al-2023)
    - [9.5 Continuous / Learned Frequency PE](#95-continuous--learned-frequency-pe)
    - [9.6 3D / Multi-Dimensional PE](#96-3d--multi-dimensional-pe)
  - [10. Positional Encoding in Different Architectures](#10-positional-encoding-in-different-architectures)
    - [10.1 Encoder-Only (BERT family)](#101-encoder-only-bert-family)
    - [10.2 Encoder-Decoder (T5 family)](#102-encoder-decoder-t5-family)
    - [10.3 Decoder-Only (GPT family)](#103-decoder-only-gpt-family)
    - [10.4 Vision Transformers (ViT)](#104-vision-transformers-vit)
    - [10.5 Multimodal Models](#105-multimodal-models)
  - [11. Mathematical Analysis — Length Generalisation](#11-mathematical-analysis--length-generalisation)
    - [11.1 Why Models Fail Beyond Training Length](#111-why-models-fail-beyond-training-length)
    - [11.2 Formal Extrapolation Bound (ALiBi)](#112-formal-extrapolation-bound-alibi)
    - [11.3 The Lost in the Middle Problem](#113-the-lost-in-the-middle-problem)
    - [11.4 Effective Context Length vs Nominal Context Length](#114-effective-context-length-vs-nominal-context-length)
  - [12. Implementation Details](#12-implementation-details)
    - [12.1 Sinusoidal PE — Code Pattern](#121-sinusoidal-pe--code-pattern)
    - [12.2 RoPE — Efficient Implementation Pattern](#122-rope--efficient-implementation-pattern)
    - [12.3 ALiBi — Implementation Pattern](#123-alibi--implementation-pattern)
    - [12.4 Numerical Stability](#124-numerical-stability)
  - [13. Common Mistakes](#13-common-mistakes)
  - [14. Exercises](#14-exercises)
  - [15. Why This Matters for AI (2026 Perspective)](#15-why-this-matters-for-ai-2026-perspective)
  - [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 The Core Problem

Attention is **permutation-invariant**: shuffle input tokens → identical attention scores.

**Proof.** Let X ∈ ℝⁿˣᵈ be the input and let P be a permutation matrix. The permuted input X' = PX produces:

```
Q' = PXW_Q = PQ
K' = PXW_K = PK
S' = Q'K'ᵀ = PQ(PK)ᵀ = PQKᵀPᵀ = PSPᵀ
```

The scores S' = PSPᵀ are simply the original scores with rows and columns reordered. Every token attends to every other token with the same weights — only the indexing changes. **The model cannot distinguish word order.**

This means:

```
"The cat sat on the mat"    →  identical attention weights as
"mat the on sat cat the"    →  (just reordered)
```

Position is not optional — it is fundamental to language meaning.

### 1.2 What Positional Encoding Must Do

Any PE scheme must satisfy these requirements:

| Requirement                    | Why                                                              |
| ------------------------------ | ---------------------------------------------------------------- |
| **Break permutation symmetry** | Different positions must produce distinguishable representations |
| **Preserve order**             | Nearby positions should be more similar than distant positions   |
| **Generalise**                 | Ideally work beyond training sequence length                     |
| **Not interfere**              | Positional signal should not overwrite semantic content          |
| **Be efficient**               | Minimal parameter overhead; ideally no extra parameters          |

### 1.3 The Design Space

Four orthogonal axes define the PE design choices:

| Axis                | Options                                                    |
| ------------------- | ---------------------------------------------------------- |
| **Where to inject** | Input embeddings, attention logits, or both                |
| **What to encode**  | Absolute position, relative offset, or continuous time     |
| **How to learn**    | Fixed/deterministic vs learned parameters                  |
| **Shape**           | Additive to embedding, multiplicative, rotational, or bias |

### 1.4 Historical Timeline

| Year    | Method                 | Key Idea                                        | Used By                     |
| ------- | ---------------------- | ----------------------------------------------- | --------------------------- |
| 2017    | Sinusoidal PE          | Fixed sin/cos added to embeddings               | Original Transformer        |
| 2018    | Learned PE             | Trainable embedding per position                | BERT, GPT-2                 |
| 2018    | Shaw Relative PE       | Learned relative key embeddings                 | Google research             |
| 2019    | Transformer-XL         | Segment-level recurrence + relative PE          | Transformer-XL, XLNet       |
| 2019    | T5 Relative Bias       | Scalar bias per bucketed offset                 | T5, Flan-T5, UL2            |
| 2021    | ALiBi                  | Linear distance penalty in logits               | BLOOM (176B), MPT           |
| 2021    | RoPE                   | Rotary encoding of Q and K                      | LLaMA, Mistral, Gemma, Qwen |
| 2022    | xPos                   | RoPE + exponential decay                        | Long-context research       |
| 2023    | Position Interpolation | Linear rescaling of RoPE positions              | LLaMA-2 long-context        |
| 2023    | NTK-Aware Scaling      | Rescale RoPE base instead of positions          | Community long-context      |
| 2023    | YaRN                   | Non-uniform per-dimension RoPE scaling          | Mistral, LLaMA-3 long       |
| 2023    | NoPE                   | No positional encoding; causal mask only        | Research                    |
| 2023    | FIRE                   | Continuous learned relative function            | Research                    |
| 2024    | LongRoPE               | Evolutionary per-dim scaling; 2M tokens         | Phi-3 (Microsoft)           |
| 2024    | CoPE                   | Context-dependent positional encoding           | Meta research               |
| 2025–26 | —                      | Context 1M–10M tokens; PE = critical bottleneck | Frontier models             |

### 1.5 Pipeline Position

```
Token IDs → [Embedding E] → x = Eᵢ → [+ PE] → x̃ → [Attention] → …
                                         ^^^^
                                    THIS section
```

For additive PE (sinusoidal, learned): position vector added to embedding **before** attention.
For rotational PE (RoPE): rotation applied to Q and K **inside** attention.
For bias PE (ALiBi, T5): scalar bias added to attention scores **inside** attention.

---

## 2. Formal Definitions

### 2.1 The Permutation Invariance Problem — Formal

Self-attention output for token i:

$$o_i = \sum_j A_{ij} v_j, \quad A_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d_k})}{\sum_{j'} \exp(q_i \cdot k_{j'} / \sqrt{d_k})}$$

If we project identical tokens x at different positions, Q, K, V are identical. The attention weights A\_{ij} become uniform: every token attends equally to every other. The output is the same vector for every position — a complete loss of sequential information.

**Need**: a function f(xᵢ, i) such that f(xᵢ, i) ≠ f(xᵢ, j) for i ≠ j. This is what positional encoding provides.

### 2.2 Absolute vs Relative Position

| Type           | Encodes                           | Formula            | Generalisation        |
| -------------- | --------------------------------- | ------------------ | --------------------- |
| **Absolute**   | Position index i ∈ {0, 1, …, n−1} | x̃ᵢ = xᵢ + pᵢ       | Poor past L           |
| **Relative**   | Offset i − j between any pair     | S*{ij} += b*{i−j}  | Good (offset bounded) |
| **Continuous** | Real number t ∈ [0, 1]            | x̃(t) = x(t) + p(t) | Domain-dependent      |

### 2.3 Additive vs Rotational Injection

**Additive**: Position vector added to token embedding before projection.

$$\tilde{x}_i = x_i + p_i, \quad p_i \in \mathbb{R}^d$$

**Rotational**: Rotation applied to Q and K after projection.

$$\tilde{q}_i = R_i q_i, \quad \tilde{k}_j = R_j k_j$$

The attention score becomes:

$$\tilde{q}_i \cdot \tilde{k}_j = (R_i q_i)^\top (R_j k_j) = q_i^\top R_i^\top R_j k_j = q_i^\top R_{j-i} k_j$$

Since R is a rotation matrix, R^⊤ = R^{−1}, and the product R_i^⊤ R_j depends only on j − i.

**Bias**: Scalar added directly to attention logits.

$$S_{ij} = q_i \cdot k_j / \sqrt{d_k} + b_{i-j}$$

### 2.4 Extrapolation vs Interpolation

- **Interpolation**: inference on sequences shorter than training length n_train. Trivially well-behaved.
- **Extrapolation**: inference on sequences longer than n_train. The hard problem.

Most PE methods fail at extrapolation:

- **Learned PE**: no embedding stored at position > L → crash or garbage
- **Sinusoidal PE**: model sees new PE vectors outside training distribution → degraded performance
- **RoPE (unscaled)**: angles mθᵢ exceed trained range → especially high-frequency dimensions alias
- **ALiBi**: smoothly extends linear penalty → strong extrapolation ✅
- **RoPE + YaRN**: controlled interpolation within trained frequency range → strong extrapolation ✅

---

## 3. Sinusoidal Positional Encoding (Original Transformer)

### 3.1 Formula

The original Transformer (Vaswani et al. 2017) defines positional encoding PE ∈ ℝⁿˣᵈ:

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Where:

- pos ∈ {0, 1, …, n−1}: token position in the sequence
- i ∈ {0, 1, …, d/2−1}: dimension pair index
- d: embedding dimension
- 10000: base constant (chosen empirically; larger base = lower frequencies)

The complete PE vector for position pos is:

```
PE(pos) = [sin(pos/10000^0), cos(pos/10000^0),
           sin(pos/10000^{2/d}), cos(pos/10000^{2/d}),
           …,
           sin(pos/10000^{(d-2)/d}), cos(pos/10000^{(d-2)/d})]
```

### 3.2 Geometric Interpretation

Each pair of dimensions (2i, 2i+1) forms a **2D circle** in that subspace:

- Position pos traces a point on the circle at angle α = pos / 10000^{2i/d}
- The coordinates are (sin α, cos α) — a point on the unit circle
- **Low i** (low dimensions): high frequency; position changes rapidly per step
- **High i** (high dimensions): low frequency; position changes slowly per step

The full PE vector is a point on a **d/2-dimensional torus** — the Cartesian product of d/2 circles:

$$\text{PE}(pos) \in S^1 \times S^1 \times \cdots \times S^1 = T^{d/2}$$

**Analogy**: binary counting. The least-significant bit flips every step, the next bit every 2 steps, the next every 4, etc. Sinusoidal PE is the continuous version of this — different dimensions oscillate at exponentially-spaced frequencies.

### 3.3 Frequency Structure

Dimension pair i oscillates with angular frequency:

$$\omega_i = \frac{1}{10000^{2i/d}}$$

And wavelength (positions per full cycle):

$$\lambda_i = \frac{2\pi}{\omega_i} = 2\pi \cdot 10000^{2i/d}$$

| Dimension pair i           | Angular frequency ωᵢ | Wavelength λᵢ | Interpretation                  |
| -------------------------- | -------------------- | ------------- | ------------------------------- |
| i = 0 (lowest dims)        | 1.0                  | 2π ≈ 6.28     | Oscillates every ~6 positions   |
| i = d/4                    | 0.01                 | 628           | Oscillates every ~628 positions |
| i = d/2 − 1 (highest dims) | 0.0001               | 62,832        | Oscillates every ~63K positions |

For d = 512: wavelengths span from 6.28 to 62,832 — a **10,000× range**.

This multi-scale structure ensures that:

- Short-range position differences are captured by high-frequency dimensions
- Long-range position differences are captured by low-frequency dimensions
- Every position in [0, 62832] gets a unique PE vector

### 3.4 Linear Relationship Between Positions

**Key property**: there exists a fixed linear transformation M_k such that:

$$\text{PE}(pos + k) = M_k \cdot \text{PE}(pos)$$

**Proof.** For each dimension pair i, apply the angle addition identity:

$$\sin(\alpha + \beta) = \sin\alpha \cos\beta + \cos\alpha \sin\beta$$
$$\cos(\alpha + \beta) = \cos\alpha \cos\beta - \sin\alpha \sin\beta$$

Let α = pos · ωᵢ and β = k · ωᵢ. Then:

$$\begin{pmatrix} \text{PE}(pos+k, 2i) \\ \text{PE}(pos+k, 2i+1) \end{pmatrix} = \begin{pmatrix} \cos(k\omega_i) & \sin(k\omega_i) \\ -\sin(k\omega_i) & \cos(k\omega_i) \end{pmatrix} \begin{pmatrix} \text{PE}(pos, 2i) \\ \text{PE}(pos, 2i+1) \end{pmatrix}$$

The full M_k is a block-diagonal matrix of d/2 rotation matrices — one 2×2 block per dimension pair. Since M_k depends only on offset k (not on pos), the model can theoretically learn to extract relative position from absolute encodings via a linear operation.

### 3.5 Properties

| Property                   | Status                                                          |
| -------------------------- | --------------------------------------------------------------- |
| No learned parameters      | ✅ Fully deterministic                                          |
| Reproducible               | ✅ Same PE every run                                            |
| Unique per position        | ✅ (for positions within wavelength range)                      |
| Nearby positions similar   | ✅ Continuous; smooth change                                    |
| Generate for any position  | ✅ Compute sin/cos for arbitrary pos at inference               |
| Linear offset relationship | ✅ PE(pos+k) = M_k · PE(pos)                                    |
| Extrapolation              | ❌ Empirically poor; model sees new PE vectors OOD              |
| Signal separation          | ❌ Adding PE to embedding mixes positional and semantic signals |

### 3.6 Dot Product Between Positions

The dot product between two PE vectors:

$$\text{PE}(pos) \cdot \text{PE}(pos') = \sum_{i=0}^{d/2-1} \left[\sin(pos \cdot \omega_i)\sin(pos' \cdot \omega_i) + \cos(pos \cdot \omega_i)\cos(pos' \cdot \omega_i)\right]$$

By the product-to-sum identity sin(a)sin(b) + cos(a)cos(b) = cos(a − b):

$$\text{PE}(pos) \cdot \text{PE}(pos') = \sum_{i=0}^{d/2-1} \cos\!\left((pos - pos') \cdot \omega_i\right)$$

**Key insight**: the dot product depends **only on the relative offset** pos − pos'. This means:

- Larger offset → more oscillating cosine terms → lower expected dot product
- Zero offset → all cosine terms equal 1 → dot product = d/2 (maximum)
- Sinusoidal PE implicitly encodes relative distance through the dot product

---

## 4. Learned Absolute Positional Embeddings

### 4.1 Definition

Position embedding matrix P ∈ ℝ^{L×d} where L is the maximum sequence length:

- Each row pᵢ ∈ ℝᵈ is a learned vector for position i
- Input to the model: x̃ᵢ = xᵢ + pᵢ
- P is trained jointly with all other model parameters via gradient descent
- Implemented as a standard embedding lookup: `position_emb = P[position_ids]`

### 4.2 Initialisation

Two common strategies:

| Strategy        | Method                                                 | When Used            |
| --------------- | ------------------------------------------------------ | -------------------- |
| Random          | pᵢ ∼ N(0, σ²); σ matched to embedding scale            | BERT, GPT-2          |
| From sinusoidal | Warm-start with sinusoidal values; let gradient refine | Some research models |

### 4.3 Properties

| Property                             | Status                                                |
| ------------------------------------ | ----------------------------------------------------- |
| Task-optimal representations         | ✅ Gradients learn best PE for the task               |
| Outperforms sinusoidal on benchmarks | ✅ Often yes (within training length)                 |
| Simple implementation                | ✅ Same as token embedding lookup                     |
| Hard length ceiling at L             | ❌ Cannot generalise to positions > L                 |
| Undertrained tail                    | ❌ Positions near L are undertrained                  |
| Extra parameters                     | ❌ L × d parameters (e.g. L=2048, d=768: 1.6M params) |

Used by: **BERT** (L=512), **GPT-2** (L=1024), **early GPT-3** (L=2048).

### 4.4 Undertrained Tail Problem

Training sequences are not uniformly distributed over length:

- Most training examples are shorter than L
- Positions near L appear in very few training examples
- Gradients for these positions are sparse → poor learned representations
- Result: **performance cliff** at lengths approaching L

```
Performance
    ↑
    |  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░
    |  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒░░░░  ← Degradation
    |             Well-trained      ↑ Undertrained
    +──────────────────────────────────────→ Position
    0               L/2            L
```

---

## 5. Relative Positional Encodings

### 5.1 Motivation

Absolute position i is less linguistically meaningful than relative offset i − j:

- "The word two positions to my left" matters more than "the word at position 47"
- Relative PE naturally handles variable-length inputs
- Generalises better: offset 3 at position 100 is the same relationship as offset 3 at position 1000
- Matches the inductive bias of natural language: meaning comes from local context and relative structure

### 5.2 Shaw et al. (2018) — Relative PE in Keys

Modify the attention score to include a relative position term:

$$S_{ij} = \frac{q_i \cdot k_j + q_i \cdot a_{i-j}^K}{\sqrt{d_k}}$$

Where a^K\_{i−j} ∈ ℝ^{d_k} is a **learned relative key embedding** for offset i − j.

- Clip offsets: max relative distance = ±k; positions beyond ±k share the same embedding
- Also modify value aggregation: oᵢ = ∑ⱼ Aᵢⱼ(vⱼ + a^V\_{i−j})
- Adds 2 × (2k+1) × d_k parameters for relative embeddings
- Content and position interact: the query determines how much to weight positional info

### 5.3 Transformer-XL (Dai et al. 2019)

Designed for **segment-level recurrence**: cache hidden states from the previous segment and attend across segment boundaries. Absolute PE would be inconsistent across segments (position 0 in segment 2 is not the same as position 0 in segment 1). Solution: decompose attention into four terms:

$$S_{ij} = \underbrace{q_i^\top W_Q^\top W_K k_j}_{\text{content-content}} + \underbrace{q_i^\top W_Q^\top W_{K,R} r_{i-j}}_{\text{content-position}} + \underbrace{u^\top W_K k_j}_{\text{global content}} + \underbrace{v^\top W_{K,R} r_{i-j}}_{\text{global position}}$$

Where:

- r\_{i−j}: sinusoidal encoding of relative distance (fixed, not learned)
- u, v ∈ ℝ^{d_k}: learned global query vectors (replacing absolute position queries)
- W\_{K,R}: separate key projection for relative positions

### 5.4 T5 Relative Bias (Raffel et al. 2019)

The simplest relative scheme — add a scalar bias to each attention logit:

$$S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} + b_{i-j}$$

Where b\_{i−j} ∈ ℝ is a **single scalar** per relative offset. In the per-head variant, each head has its own bias table.

- Very few parameters: one scalar per bucket per head
- **Buckets** reduce the number of parameters for large offsets (Section 5.5)
- Strong generalisation; simple and effective
- Used by: **T5**, **UL2**, **Flan-T5**, **mT5**

### 5.5 Bucket Formula (T5)

For offset δ = |i − j|:

| Range             | Bucketing          | Resolution             |
| ----------------- | ------------------ | ---------------------- |
| δ < 16            | bucket = δ (exact) | 1 position per bucket  |
| 16 ≤ δ < max_dist | Logarithmic        | Decreasing resolution  |
| δ ≥ max_dist      | Same final bucket  | No further distinction |

The formula for δ ≥ 16:

$$\text{bucket}(\delta) = 16 + \left\lfloor \frac{\log(\delta / 16)}{\log(\text{max\_dist} / 16)} \times 16 \right\rfloor$$

This gives exact resolution for nearby tokens (where precise position matters most) and coarser resolution for distant tokens (where approximate distance suffices). With max_dist = 128 and 32 total buckets: 16 exact + 16 logarithmic = 32 buckets covering all offsets.

---

## 6. ALiBi — Attention with Linear Biases

### 6.1 Core Idea (Press et al. 2021)

Do **not** add any positional vector to token embeddings. Instead, add a **linear distance penalty** directly to attention scores:

$$S_{ij} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m_h \cdot (i - j)$$

Where:

- m_h > 0: head-specific slope; larger m_h = stronger recency bias
- (i − j): distance between positions (for causal attention, i ≥ j so this is non-negative)
- The penalty is always **negative**: farther tokens receive larger penalties

The bias matrix for a sequence of length n (causal):

```
         k₀    k₁    k₂    k₃
q₀  [   0
q₁  [ -m     0
q₂  [ -2m   -m     0
q₃  [ -3m   -2m   -m     0    ]
```

### 6.2 Slope Schedule

Slopes form a **geometric sequence** across H heads:

$$m_h = \frac{1}{2^{h \cdot 8/H}}, \quad h = 1, \ldots, H$$

| H (heads) | Slopes                                        |
| --------- | --------------------------------------------- |
| 4         | 1/4, 1/16, 1/64, 1/256                        |
| 8         | 1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/256 |
| 16        | 2^{-0.5}, 2^{-1}, 2^{-1.5}, …, 2^{-8}         |

- **Large slope** (e.g. 1/2): sharp distance decay → head attends **locally**
- **Small slope** (e.g. 1/256): gentle decay → head attends **globally**
- This forces different heads to specialise in different attention ranges

### 6.3 Extrapolation Mechanism

Why ALiBi extrapolates beyond training length:

1. **During training** (length n_train): model sees distances 0, 1, …, n_train − 1 and learns to produce content scores q·k that are calibrated relative to penalties up to m_h · n_train
2. **At inference** (length n_test > n_train): distances n_train, …, n_test−1 produce larger penalties
3. These larger penalties are just **bigger negative numbers** — not fundamentally different from what the model has seen
4. There are no OOD embedding vectors, no new angles, no new basis functions — just smoothly-extended scalars
5. **Result**: train on 1024 tokens → near-perfect perplexity at 2048 tokens

### 6.4 Properties

| Property             | Status                                                       |
| -------------------- | ------------------------------------------------------------ |
| Extra parameters     | ✅ Zero — slopes are deterministic                           |
| Length extrapolation | ✅ Strong; smoothly extends                                  |
| Implementation       | ✅ Simple — single line added to attention scores            |
| Head specialisation  | ✅ Different slopes → local vs global attention              |
| Fixed linear decay   | ❌ May not be optimal for all tasks                          |
| Expressiveness       | ❌ Less expressive than learned relative PE                  |
| Compatibility        | ❌ Cannot be combined with RoPE (different injection points) |

Used by: **BLOOM** (176B), **MPT**, **OpenLLaMA** variants.

---

## 7. RoPE — Rotary Positional Encoding

### 7.1 Core Idea (Su et al. 2021)

Encode position by **rotating** the query and key vectors. The rotation angle is proportional to the position index: a token at position m is rotated m times more than a token at position 0. The key insight: the dot product of two rotated vectors depends only on their **relative offset**.

No positional vectors are added to embeddings. Position is encoded entirely in the **geometry** of Q and K.

### 7.2 2D Case — Building Intuition

For d = 2, rotate vector v by angle mθ at position m:

$$R_m v = \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} v_1 \\ v_2 \end{pmatrix}$$

Now consider the dot product of rotated query q at position m and key k at position n:

$$(R_m q)^\top (R_n k) = q^\top R_m^\top R_n k = q^\top R_{n-m} k$$

**Why?** Because R*m is an orthogonal matrix (R^⊤ = R^{−1}), and R_m^{−1} R_n = R*{n−m} (rotations compose by adding angles).

Result: the dot product depends **only** on n − m (relative offset), not on absolute positions. ✅

### 7.3 General d-Dimensional RoPE

Apply 2D rotation independently to each consecutive pair of dimensions:

$$R_m = \text{blockdiag}\!\left(R_m^{(0)}, R_m^{(1)}, \ldots, R_m^{(d/2-1)}\right)$$

Each 2×2 block:

$$R_m^{(i)} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix}$$

With **frequencies**:

$$\theta_i = \frac{1}{\text{base}^{2i/d}}, \quad i = 0, 1, \ldots, d/2 - 1$$

Default base = 10000 (same as sinusoidal PE).

The full formula for the rotated query at position m:

$$\tilde{q}_m = R_m q_m$$

$$(R_m q)_{2i} = q_{2i}\cos(m\theta_i) - q_{2i+1}\sin(m\theta_i)$$

$$(R_m q)_{2i+1} = q_{2i}\sin(m\theta_i) + q_{2i+1}\cos(m\theta_i)$$

The relative-offset property holds for each dimension pair independently:

$$(R_m^{(i)} u)^\top (R_n^{(i)} v) = u^\top R_{n-m}^{(i)} v$$

Summing over all pairs: q̃_m · k̃_n = f(q, k, n − m) only.

### 7.4 Efficient Implementation

**Never construct the full R_m matrix** — that would be O(d²). Instead:

1. Split q into pairs: q = (q₀, q₁, q₂, q₃, …, q*{d−2}, q*{d−1})
2. Precompute cos(mθᵢ) and sin(mθᵢ) for all positions m and dimension pairs i
3. Apply element-wise with a "rotate-half" trick:

$$\tilde{q} = q \odot \cos(m\theta) + \text{rotate\_half}(q) \odot \sin(m\theta)$$

Where rotate_half swaps and negates pairs: [q₀, q₁, q₂, q₃] → [−q₁, q₀, −q₃, q₂]

This is **O(nd)** computation — no matrix multiply needed. The cos/sin values are precomputed once and cached.

### 7.5 Key Properties

| Property                         | Status                                       |
| -------------------------------- | -------------------------------------------- |
| Relative position encoding       | ✅ q̃ᵢ·k̃ⱼ = f(content, i−j)                   |
| No extra parameters              | ✅ θᵢ are deterministic from base            |
| Compatible with linear attention | ✅ Rotation commutes with many kernels       |
| Rich positional signal           | ✅ Each dimension pair has its own frequency |
| FlashAttention compatible        | ✅ Rotation applied before tiling            |
| Norm-preserving                  | ✅ Rotation matrices are orthogonal          |
| Extrapolation (unscaled)         | ❌ Degrades beyond training length           |
| Implementation complexity        | ❌ Slightly more complex than additive PE    |

### 7.6 Frequency Analysis of RoPE

Dimension pair i has frequency θᵢ = base^{−2i/d} and wavelength:

$$\lambda_i = \frac{2\pi}{\theta_i} = 2\pi \cdot \text{base}^{2i/d}$$

For base = 10000, d = 128:

| Pair i | Frequency θᵢ | Wavelength λᵢ | Context coverage          |
| ------ | ------------ | ------------- | ------------------------- |
| 0      | 1.0          | 6.28          | 6 positions per cycle     |
| 16     | 0.056        | 112           | 112 positions per cycle   |
| 32     | 0.0032       | 1,990         | ~2K positions per cycle   |
| 48     | 0.00018      | 35,300        | ~35K positions per cycle  |
| 63     | 0.00001      | 628,318       | ~628K positions per cycle |

**Long-context problem**: high-i dimensions (low frequency) barely rotate within training length.

- For n_train = 4096: pair 63 completes only 4096/628318 ≈ 0.0065 of a full rotation
- These dimensions carry almost **no positional information** for typical contexts
- At very long contexts, only these slow dimensions can distinguish distant positions — but they're essentially untrained

### 7.7 RoPE Used By

As of 2025–2026, RoPE is the dominant PE standard:

**LLaMA** 1/2/3, **Mistral**, **Gemma**/Gemma-2, **Qwen**/Qwen2, **Phi-2/3**, **Falcon**, **Yi**, **Baichuan**, **DeepSeek**, **Command R**, **OLMo**, **Grok** — virtually every major open model.

---

## 8. RoPE Scaling for Long Context

### 8.1 The Problem

RoPE trained with base = 10000 encodes positions well up to approximately n_train. Beyond n_train:

- **Angles mθᵢ exceed the trained range** for high-frequency dimensions (low i)
- These dimensions have completed many full rotations within n_train; new positions look like **already-seen positions** (aliasing)
- Low-frequency dimensions (high i) barely budge at n_train; they can distinguish longer positions but were **undertrained**
- Result: **catastrophic perplexity increase** at positions > n_train

### 8.2 Linear Scaling (Position Interpolation, Chen et al. 2023)

Scale all positions by factor s = n_train / n_test:

$$\text{pos} \rightarrow \text{pos} / s$$

This **interpolates** between trained positions rather than extrapolating:

- Position 0 maps to 0 (unchanged)
- Position n_test maps to n_train (the furthest trained position)
- All intermediate positions map within [0, n_train]

**Tradeoff**: compresses positional resolution. Nearby tokens (offset 1) now map to offset 1/s — the model must distinguish finer differences than it was trained on.

Requires short fine-tuning (~1000 steps) to adapt.

### 8.3 NTK-Aware Scaling (Community 2023, bloc97)

Instead of scaling positions, scale the **base**:

$$\theta_i' = \left(\text{base} \cdot s^{d/(d-2)}\right)^{-2i/d}$$

Effect:

- High-frequency dimensions (low i): frequencies decrease → extends their effective range
- Low-frequency dimensions (high i): frequencies barely change → preserves local positional resolution
- Redistributes scaling across frequencies; high-frequency dimensions less compressed than linear scaling

**No fine-tuning required** (zero-shot extension); slight quality loss.

### 8.4 YaRN — Yet Another RoPE extensioN (Peng et al. 2023)

YaRN identifies **three zones** of dimension pairs based on the ratio of wavelength to training length:

| Zone                 | Dimensions | Wavelength vs n_train              | Action                |
| -------------------- | ---------- | ---------------------------------- | --------------------- |
| **High frequency**   | Low i      | λᵢ ≪ n_train (many full rotations) | Interpolate (like PI) |
| **Medium frequency** | Mid i      | λᵢ ≈ n_train                       | NTK-scale             |
| **Low frequency**    | High i     | λᵢ ≫ n_train (barely rotated)      | No scaling needed     |

YaRN applies **different scaling per dimension**, with a smooth ramp function γ(i) ∈ [0, 1] blending between interpolation and no-scaling:

$$\theta_i' = \theta_i \cdot \frac{1}{(1 - \gamma_i) / s + \gamma_i}$$

Where γ(i) = 0 → full interpolation (divide by s) and γ(i) = 1 → no scaling.

Additionally, YaRN adds a **magnitude correction factor**:

$$\sqrt{\frac{1}{d} \sum_i \gamma_i^2 + (1 - \gamma_i)^2 / s^2}^{-1}$$

This compensates for the reduced attention score magnitude caused by frequency compression.

Requires only ~400 fine-tuning steps. Used by: **LLaMA-3** long-context, **Mistral** long-context, many community fine-tunes.

### 8.5 LongRoPE (Ding et al. 2024, Microsoft)

Extends context to **2M tokens** in Phi-3:

1. **Non-uniform per-dimension rescaling** factors found by evolutionary search (not a closed-form formula)
2. **Two-stage**: find optimal rescale factors at 256K → interpolate to 2M
3. Maintains **short-context performance** (≤4K) by preserving high-frequency dimensions
4. **Critical finding**: optimal scaling factors are **non-monotone** across dimensions — some mid-range dimensions need more scaling than nearby ones

### 8.6 RoPE Base Scaling Comparison

| Method      | Approach                  | Fine-tune?       | Short ctx quality | Long ctx quality | Max demonstrated |
| ----------- | ------------------------- | ---------------- | ----------------- | ---------------- | ---------------- |
| No scaling  | —                         | —                | ✅ Best           | ❌ Fails         | n_train          |
| Linear (PI) | Scale positions by s      | Yes (~1K steps)  | ⚠️ Slight loss    | ✅ Good          | 32K              |
| NTK-Aware   | Scale base by s^{d/(d−2)} | No (zero-shot)   | ✅ Good           | ✅ Good          | 16K              |
| YaRN        | Non-uniform + correction  | Yes (~400 steps) | ✅ Best           | ✅ Best          | 128K             |
| LongRoPE    | Per-dim evolutionary      | Yes              | ✅ Best           | ✅ Best          | 2M               |

---

## 9. Other Modern Positional Schemes

### 9.1 xPos (Sun et al. 2022)

Extends RoPE with an **exponential decay** factor on attention scores:

- Multiplies Q, K by position-dependent scaling factor: eˢᵐ for queries, e⁻ˢⁿ for keys
- Net effect: attention score includes exp(−s|m−n|) decay for distant tokens
- Improves length extrapolation by smoothly suppressing very long-range attention
- Used by some research models and long-context fine-tunes

### 9.2 FIRE — Functional Interpolation for Relative Encodings (Li et al. 2023)

- Learn a **continuous function** f(i − j) → scalar bias using a small MLP
- The MLP takes the normalised relative position as input; outputs a per-head scalar bias
- Generalises to **any offset** at inference time (the MLP can interpolate/extrapolate)
- Very few parameters (small MLP); strong extrapolation
- Shows that even a tiny network can capture complex distance patterns

### 9.3 CoPE — Contextualised Positional Encoding (Olsson et al. 2024)

Position is computed **from context**, not just the integer index:

- Each token computes a **gate value** from its content: gᵢ = σ(w · xᵢ)
- Position is the cumulative sum of gates: p̃ᵢ = ∑\_{j≤i} gⱼ
- Allows the model to assign "logical" position rather than "physical" position
- Example: in code, closing brackets at position 50 might have logical position close to the opening bracket at position 10
- Better for structured inputs: code, tables, lists, nested structures

### 9.4 NoPE — No Positional Encoding (Kazemnejad et al. 2023)

Some architectures work **without explicit PE**:

- Positional information comes from the **causal mask alone**
- Causal mask breaks symmetry: token i can only attend to j ≤ i
- This means token 0 attends to 1 token, token 1 to 2 tokens, … implicitly encoding position
- Surprisingly competitive on short sequences; fails on long sequences
- Demonstrates that causal masking provides a weak but real implicit positional signal

### 9.5 Continuous / Learned Frequency PE

- Replace fixed base 10000 with a **learned base** parameter
- Or learn the θᵢ values directly rather than computing from a formula
- Adds d/2 learnable parameters (negligible)
- Used in some **ViT** (Vision Transformer) variants for image patches

### 9.6 3D / Multi-Dimensional PE

| Modality | Dimensions                          | PE Approach                       |
| -------- | ----------------------------------- | --------------------------------- |
| Text     | 1D (sequence position)              | RoPE, ALiBi, sinusoidal           |
| Images   | 2D (row, column)                    | Factored 1D+1D or full 2D learned |
| Video    | 3D (time, row, column)              | Factored 1D+1D+1D or learned      |
| Code     | Hierarchical (file, function, line) | Research area; hierarchical PE    |
| Audio    | 1D (time samples)                   | Same as text; RoPE typical        |

For images: **DINOv2** uses learned absolute PE with bilinear interpolation at different resolutions.
For video: **Sora** and video generation models use 3D sinusoidal or factored learned PE.

---

## 10. Positional Encoding in Different Architectures

### 10.1 Encoder-Only (BERT family)

- **Bidirectional** attention: all tokens see all others
- **No causal mask**: PE must distinguish positions without directional cues
- Absolute learned PE most common: **BERT** (L=512), **RoBERTa** (L=514), **ALBERT** (L=512)
- Maximum length is a hard constraint — the model crashes on inputs longer than L

### 10.2 Encoder-Decoder (T5 family)

- **Encoder**: bidirectional T5 relative bias
- **Decoder**: causal T5 relative bias
- **Cross-attention**: typically no PE needed (queries from decoder, keys from encoder — positions are in different spaces)
- T5 relative bias is the reference implementation for scalar-bias PE

### 10.3 Decoder-Only (GPT family)

- **Causal mask**: tokens only see past and present
- **GPT-2**: learned absolute PE (L=1024)
- **GPT-3**: learned absolute PE (L=2048)
- **LLaMA/Mistral/Gemma/Qwen**: **RoPE** — the current standard for decoder-only models
- Code generation: RoPE with large base (e.g. base=500,000 for **CodeLLaMA**) to handle long code files

### 10.4 Vision Transformers (ViT)

- Image split into **patches** (16×16 or 14×14 pixels)
- Patches linearised into a sequence; each patch is a "token"
- **2D spatial position**: row and column in the patch grid
- Options: learned 2D PE, factored 1D+1D PE (row PE + column PE), or flat 1D learned PE
- At inference for different image resolutions: **bilinear interpolation** of learned PE

### 10.5 Multimodal Models

- **Text tokens**: RoPE (standard)
- **Image tokens**: may use separate 2D PE
- **Cross-modal alignment**: position in text vs position in image are fundamentally different spaces
- **LLaVA, Flamingo**: image features injected as prefix tokens; text token positions shifted accordingly
- **Gemini 1.5**: unified positional encoding across modalities — current research frontier

---

## 11. Mathematical Analysis — Length Generalisation

### 11.1 Why Models Fail Beyond Training Length

The failure mode depends on the PE type:

| PE Type          | Failure Mode at n > n_train                                         |
| ---------------- | ------------------------------------------------------------------- |
| Learned absolute | No embedding at position > L → index error or zeros                 |
| Sinusoidal       | Model receives valid PE vectors but has never trained on them → OOD |
| RoPE (unscaled)  | Angles mθᵢ exceed trained range; high-freq dims alias               |
| ALiBi            | Smoothly extends → no OOD (strong extrapolation)                    |
| T5 bias          | Offsets > max_dist all share bucket → limited discrimination        |

The fundamental issue: any PE that produces **representations the model has never seen during training** will cause degradation. ALiBi avoids this because its output is just a scalar (larger negative number), not a high-dimensional vector.

### 11.2 Formal Extrapolation Bound (ALiBi)

For ALiBi, the attention score at distance δ:

$$S_{ij} = \underbrace{q_i \cdot k_j / \sqrt{d_k}}_{\text{content score}} - \underbrace{m_h \delta}_{\text{distance penalty}}$$

As δ → ∞: S\_{ij} → −∞, so the softmax weight → 0. The token becomes effectively **invisible**.

Within training distribution: δ ≤ n_train. The model trains on the full range of content scores paired with penalties up to m_h · n_train.

Beyond training: δ > n_train. The penalty m_h · δ is just a larger negative number — the same kind of number the model has seen, just bigger. There are **no OOD vectors, no new dimensions, no aliasing**.

### 11.3 The Lost in the Middle Problem

Liu et al. (2023) showed that LLMs perform **worse on information in the middle** of long contexts:

```
Retrieval Accuracy
    ↑
    |  ▓▓▓▓                                    ▓▓▓▓
    |  ▓▓▓▓▓▓                                ▓▓▓▓▓▓
    |  ▓▓▓▓▓▓▓▓▒▒▒▒▒▒░░░░░░░░░░░░▒▒▒▒▒▒▓▓▓▓▓▓▓▓
    |                  ↑ weak middle
    +──────────────────────────────────────────→ Position in context
    Start                                      End
```

This **U-shaped performance curve** results from:

- **Recency bias**: recent tokens have shorter distances → stronger attention weights (amplified by ALiBi's linear decay)
- **Primacy bias**: BOS / early tokens attract disproportionate attention (attention sinks)
- **Positional encoding contribution**: PE design directly affects this pattern — ALiBi worsens recency; RoPE distributes more evenly

### 11.4 Effective Context Length vs Nominal Context Length

| Metric        | Definition                                        | Typical Ratio               |
| ------------- | ------------------------------------------------- | --------------------------- |
| **Nominal**   | Maximum n the model accepts without error         | — (stated context window)   |
| **Effective** | Length at which performance meaningfully degrades | ~0.5× nominal for retrieval |

Standard benchmarks:

- **Needle-in-a-haystack** (Kamradt 2023): hide a fact in a long document; test retrieval at every position
- **RULER** (2024): multi-hop retrieval test across context lengths
- **∞Bench** (2024): tasks requiring understanding of the full context, not just retrieval

---

## 12. Implementation Details

### 12.1 Sinusoidal PE — Code Pattern

```python
def sinusoidal_pe(n, d):
    """Generate sinusoidal positional encoding matrix (n, d)."""
    pos = np.arange(n)[:, None]                  # (n, 1)
    i   = np.arange(0, d, 2)[None, :]            # (1, d/2)
    angles = pos / 10000 ** (i / d)              # (n, d/2)
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(angles)                 # even dimensions
    pe[:, 1::2] = np.cos(angles)                 # odd dimensions
    return pe                                     # (n, d)
```

### 12.2 RoPE — Efficient Implementation Pattern

```python
def apply_rope(q, positions, base=10000):
    """Apply RoPE to query (or key) tensor. q: (n, d), positions: (n,)."""
    d = q.shape[-1]
    i = np.arange(0, d, 2) / d                  # (d/2,)
    freqs = 1.0 / (base ** i)                   # (d/2,)
    angles = positions[:, None] * freqs[None, :] # (n, d/2)
    cos_a = np.cos(angles)                       # (n, d/2)
    sin_a = np.sin(angles)                       # (n, d/2)
    # Rotate pairs: (q0,q1) → (q0*cos - q1*sin, q0*sin + q1*cos)
    q_even = q[:, 0::2]                          # (n, d/2)
    q_odd  = q[:, 1::2]                          # (n, d/2)
    q_rot = np.zeros_like(q)
    q_rot[:, 0::2] = q_even * cos_a - q_odd * sin_a
    q_rot[:, 1::2] = q_even * sin_a + q_odd * cos_a
    return q_rot
```

### 12.3 ALiBi — Implementation Pattern

```python
def alibi_slopes(num_heads):
    """Compute ALiBi slopes for each head."""
    return 1.0 / (2 ** (np.arange(1, num_heads + 1) * 8 / num_heads))

def alibi_bias(n, slopes):
    """Build ALiBi bias matrix (H, n, n) for causal attention."""
    dist = np.arange(n)[None, :] - np.arange(n)[:, None]  # (n, n)
    dist = np.minimum(dist, 0)  # causal: only past (non-positive)
    return dist[None, :, :] * slopes[:, None, None]        # (H, n, n)
```

### 12.4 Numerical Stability

| PE Type    | Stability Issue                                    | Mitigation                                                  |
| ---------- | -------------------------------------------------- | ----------------------------------------------------------- |
| Sinusoidal | None — sin/cos bounded in [−1, 1]                  | —                                                           |
| RoPE       | None — rotation preserves vector norm (orthogonal) | —                                                           |
| ALiBi      | Large negative biases can cause softmax underflow  | Use log-space softmax; compute max before exp               |
| Learned PE | Large PE values can dominate embedding             | Initialise with small σ; layer normalisation after addition |

---

## 13. Common Mistakes

| #   | Mistake                                            | Why It's Wrong                                                           | Fix                                                 |
| --- | -------------------------------------------------- | ------------------------------------------------------------------------ | --------------------------------------------------- |
| 1   | "Sinusoidal PE extrapolates perfectly"             | Empirically fails past training length; model sees new PE vectors OOD    | Use RoPE + YaRN for reliable long context           |
| 2   | "Learned PE is always better than fixed"           | Learned PE cannot generalise past L; sinusoidal/RoPE can extend          | Use RoPE for any model needing length flexibility   |
| 3   | "RoPE needs no modification for long context"      | Base=10000 fails past ~4K–8K tokens without scaling                      | Apply NTK-aware or YaRN scaling for longer contexts |
| 4   | "PE only affects encoder; decoder doesn't need it" | Decoder needs causal PE too; position of generated tokens matters        | Apply PE consistently to all attention layers       |
| 5   | "Absolute position is most important"              | Relative offset is linguistically more meaningful                        | Prefer relative or rotary PE for most tasks         |
| 6   | "All heads should use the same positional signal"  | ALiBi shows different heads benefit from different slopes                | Use head-specific PE to enable specialisation       |
| 7   | "PE is fixed forever after pretraining"            | RoPE base can be rescaled at fine-tune time for context extension        | YaRN / NTK rescaling requires only ~400 steps       |
| 8   | "2D PE for images = apply 1D PE twice"             | 2D structure has row-column interactions; factored PE misses cross-terms | Use full 2D PE or patch-level learned PE            |

---

## 14. Exercises

1. **Sinusoidal by hand** — for d=4, pos=0,1,2: compute all PE vectors; verify the linear relationship PE(pos+1) = M₁ · PE(pos)
2. **Dot product decay** — compute PE(0)·PE(k) for k=0,1,5,50 with d=512; plot the decay curve and explain the oscillatory structure
3. **RoPE 2D** — for d=2, θ=π/4, positions 0,1,2,3: compute rotated query vectors; verify q̃ᵢ·k̃ⱼ depends only on i−j
4. **ALiBi slopes** — for H=8: compute all slopes; plot the bias matrix for n=16; identify which heads attend locally vs globally
5. **RoPE frequency analysis** — for d=128, base=10000: compute all wavelengths λᵢ; identify which dimensions encode local vs global position; find how many dimensions are effectively "dead" for context length 4096
6. **Extrapolation comparison** — implement sinusoidal PE, RoPE, and ALiBi; compare dot-product patterns at 1×, 2×, 4× training length
7. **YaRN zones** — for d=64, base=10000, n_train=2048: classify all dimension pairs into high/mid/low frequency zones; compute the recommended scaling factors for 4× context extension
8. **Context length benchmark** — implement needle-in-a-haystack retrieval test; compare RoPE vs ALiBi at 4K, 8K, 16K context

---

## 15. Why This Matters for AI (2026 Perspective)

| Aspect                | Impact                                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------- |
| **Long context**      | PE design is the primary technical bottleneck for 1M–10M token contexts                                   |
| **Code models**       | Code has deep hierarchical structure; PE must encode nested scopes and indentation levels                 |
| **Retrieval quality** | Lost-in-the-middle effect reduces effective context; PE design is partially responsible                   |
| **Multimodal**        | Unified PE across text/image/video/audio is an open research problem                                      |
| **Fine-tuning**       | Context extension via YaRN requires only ~400 fine-tuning steps — very cheap per-token cost reduction     |
| **Edge deployment**   | NoPE models have zero PE parameters; beneficial for tiny models on mobile/embedded devices                |
| **Interpretability**  | PE affects which tokens each head attends to; circuit analysis requires understanding PE structure        |
| **Fairness**          | Languages with longer average token sequences (non-English) are penalised more by limited context windows |

---

## Conceptual Bridge

Positional encoding solves the fundamental **permutation-invariance problem** of attention. Without it, a Transformer is a bag-of-words model with an expensive matrix-multiply budget — it knows what words are present but has no idea where they are. We covered the full spectrum from the original sinusoidal encoding through today's dominant RoPE and its scaling variants for million-token contexts.

**Next**: [Feed-Forward Network Math](../05-Language-Model-Probability/notes.md) — the second sublayer in each transformer block, where approximately two-thirds of all parameters live and where factual knowledge is stored.

---

[← Attention Mechanism Math](../03-Attention-Mechanism-Math/notes.md) | [Home](../../README.md) | [Language Model Probability →](../05-Language-Model-Probability/notes.md)
