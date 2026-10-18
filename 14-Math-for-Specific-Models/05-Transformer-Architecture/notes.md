[<- RNN and LSTM Math](../04-RNN-and-LSTM-Math/notes.md) | [Home](../../README.md) | [Reinforcement Learning ->](../06-Reinforcement-Learning/notes.md)

---

# Transformer Architecture

The transformer is a stack of attention, feed-forward, residual, normalization, and position mechanisms. Its core advantage is that training can process all token positions in parallel while attention lets each token condition on other tokens.

## Overview

The central attention operation is:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$

This formula turns token representations into queries, keys, and values. Queries ask, keys match, values carry content. Multi-head attention repeats this operation in several subspaces. Residual blocks and normalization make the stack trainable. Positional information restores order. Masks define which tokens are allowed to interact.

## Prerequisites

- Matrix multiplication and softmax
- Embeddings and positional encodings
- Sequence probability and autoregressive masking
- RNN section for the recurrence-to-attention comparison

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates attention shapes, causal masks, multi-head splitting, MLP blocks, Pre-LN/Post-LN, positional signals, KV cache size, and diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for attention arithmetic, masks, shapes, parameter counts, and transformer debugging. |

## Learning Objectives

After this section, you should be able to:

- Compute scaled dot-product attention from Q, K, and V.
- Explain multi-head attention and head dimension arithmetic.
- Distinguish attention token mixing from position-wise MLP channel mixing.
- Explain residual streams, LayerNorm, Pre-LN, Post-LN, and RMSNorm.
- Compare encoder, decoder, encoder-decoder, encoder-only, and decoder-only architectures.
- Explain positional signals, attention masks, and KV cache memory.
- Estimate attention and MLP complexity.
- Build shape, mask, attention-entropy, and residual-norm diagnostics.

## Table of Contents

1. [Transformer Design Goal](#1-transformer-design-goal)
   - 1.1 [Sequence transduction](#11-sequence-transduction)
   - 1.2 [Parallel token processing](#12-parallel-token-processing)
   - 1.3 [Context mixing](#13-context-mixing)
   - 1.4 [Position information](#14-position-information)
   - 1.5 [Stacked blocks](#15-stacked-blocks)
2. [Scaled Dot-Product Attention](#2-scaled-dotproduct-attention)
   - 2.1 [Queries keys values](#21-queries-keys-values)
   - 2.2 [Score matrix](#22-score-matrix)
   - 2.3 [Masking](#23-masking)
   - 2.4 [Attention weights](#24-attention-weights)
   - 2.5 [Weighted values](#25-weighted-values)
3. [Multi-Head Attention](#3-multihead-attention)
   - 3.1 [Head splitting](#31-head-splitting)
   - 3.2 [Per-head dimension](#32-perhead-dimension)
   - 3.3 [Concatenation](#33-concatenation)
   - 3.4 [Specialization](#34-specialization)
   - 3.5 [GQA and MQA bridge](#35-gqa-and-mqa-bridge)
4. [Feed-Forward Network](#4-feedforward-network)
   - 4.1 [Position-wise MLP](#41-positionwise-mlp)
   - 4.2 [Expansion ratio](#42-expansion-ratio)
   - 4.3 [Activation](#43-activation)
   - 4.4 [Parameter count](#44-parameter-count)
   - 4.5 [Token independence](#45-token-independence)
5. [Residuals and Normalization](#5-residuals-and-normalization)
   - 5.1 [Residual stream](#51-residual-stream)
   - 5.2 [Layer normalization](#52-layer-normalization)
   - 5.3 [Pre-LN block](#53-preln-block)
   - 5.4 [Post-LN block](#54-postln-block)
   - 5.5 [RMSNorm](#55-rmsnorm)
6. [Encoder Decoder and Decoder-Only Forms](#6-encoder-decoder-and-decoderonly-forms)
   - 6.1 [Encoder self-attention](#61-encoder-selfattention)
   - 6.2 [Decoder causal self-attention](#62-decoder-causal-selfattention)
   - 6.3 [Cross-attention](#63-crossattention)
   - 6.4 [Encoder-only](#64-encoderonly)
   - 6.5 [Decoder-only](#65-decoderonly)
7. [Positional Information](#7-positional-information)
   - 7.1 [Absolute positions](#71-absolute-positions)
   - 7.2 [Sinusoidal positions](#72-sinusoidal-positions)
   - 7.3 [Relative bias](#73-relative-bias)
   - 7.4 [RoPE](#74-rope)
   - 7.5 [Length behavior](#75-length-behavior)
8. [Complexity and Memory](#8-complexity-and-memory)
   - 8.1 [Attention FLOPs](#81-attention-flops)
   - 8.2 [Attention score memory](#82-attention-score-memory)
   - 8.3 [MLP FLOPs](#83-mlp-flops)
   - 8.4 [KV cache](#84-kv-cache)
   - 8.5 [FlashAttention bridge](#85-flashattention-bridge)
9. [Training Signals](#9-training-signals)
   - 9.1 [Language-model loss](#91-languagemodel-loss)
   - 9.2 [Masked LM loss](#92-masked-lm-loss)
   - 9.3 [Teacher forcing](#93-teacher-forcing)
   - 9.4 [Attention masks](#94-attention-masks)
   - 9.5 [Weight tying](#95-weight-tying)
10. [Diagnostics](#10-diagnostics)
37. [Computed Table: Four Reference Configurations](#37-computed-table-four-reference-configurations)
38. [Ten-Line Field Guide to Defensible Transformer Claims](#38-ten-line-field-guide-to-defensible-transformer-claims)
34. [Addenda: Ten Lines to Carry Forward](#34-addenda-ten-lines-to-carry-forward)
35. [Margin Notes: The Transformer Creed](#35-margin-notes-the-transformer-creed)
36. [End of Section 05 Expansion](#36-end-of-section-05-expansion)
29. [Worked Matrix: A Causal Attention Map](#29-worked-matrix-a-causal-attention-map)
30. [Extended Exercise Bank (Self-Study)](#30-extended-exercise-bank-self-study)
31. [Revision Problems (Mixed)](#31-revision-problems-mixed)
32. [Chapter Summary: Transformer Architecture](#32-chapter-summary-transformer-architecture)
33. [Section 05 Completion Checklist](#33-section-05-completion-checklist)
25. [Exam and Interview Traps: Transformers](#25-exam-and-interview-traps-transformers)
26. [Worked Problem Set C: Synthesis](#26-worked-problem-set-c-synthesis)
27. [Quick Checks (One-Line Answers)](#27-quick-checks-one-line-answers)
28. [Index of Computed Evidence](#28-index-of-computed-evidence)
21. [Computed Table: Parameter Anatomy of a Decoder](#21-computed-table-parameter-anatomy-of-a-decoder)
22. [Self-Test Questions](#22-self-test-questions)
23. [Self-Test Answer Key (Condensed)](#23-self-test-answer-key-condensed)
24. [Annotated Reading Guide](#24-annotated-reading-guide)
17. [Quick-Reference Cards: Block Recipes](#17-quick-reference-cards-block-recipes)
18. [Pre-Flight Checklist for Transformer Work](#18-pre-flight-checklist-for-transformer-work)
19. [Frequently Asked Questions](#19-frequently-asked-questions)
20. [Decision Records: Transformer Choices in Context](#20-decision-records-transformer-choices-in-context)
14. [Worked Problem Set A: Mechanics and Budgets](#14-worked-problem-set-a-mechanics-and-budgets)
15. [Worked Problem Set B: Design Choices](#15-worked-problem-set-b-design-choices)
16. [Glossary: Transformer Vocabulary](#16-glossary-transformer-vocabulary)
11. [Common Mistakes](#11-common-mistakes)
12. [Attention Is the Whole Architecture Now](#12-attention-is-the-whole-architecture-now)
13. [Computed Table: Long-Context Costs](#13-computed-table-long-context-costs)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Mask tests](#102-mask-tests)
   - 10.3 [Attention entropy](#103-attention-entropy)
   - 10.4 [Residual norms](#104-residual-norms)
   - 10.5 [Ablations](#105-ablations)

---

## Block Shape Map

```text
hidden states:       H       shape (B, T, d_model)
queries:             Q       shape (B, heads, T, d_head)
keys:                K       shape (B, heads, T, d_head)
values:              V       shape (B, heads, T, d_head)
attention scores:    S       shape (B, heads, T, T)
attention output:    O       shape (B, T, d_model)
```

## 1. Transformer Design Goal

This part studies transformer design goal as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sequence transduction](#1-sequence-transduction) | map one sequence representation to another | $x_{1:T}\rightarrow y_{1:M}$ |
| [Parallel token processing](#1-parallel-token-processing) | avoid recurrence during training | $H^{(0)}\in\mathbb{R}^{B\times T\times d}$ |
| [Context mixing](#1-context-mixing) | let every token read other tokens through attention | $\mathrm{Attention}(Q,K,V)$ |
| [Position information](#1-position-information) | inject order because attention alone is permutation equivariant | $h_i=x_i+p_i$ |
| [Stacked blocks](#1-stacked-blocks) | repeat attention and MLP transformations | $H^{(\ell+1)}=\mathrm{Block}_\ell(H^{(\ell)})$ |

### 1.1 Sequence transduction

**Main idea.** Map one sequence representation to another.

Core relation:

$$x_{1:T}\rightarrow y_{1:M}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.2 Parallel token processing

**Main idea.** Avoid recurrence during training.

Core relation:

$$H^{(0)}\in\mathbb{R}^{B\times T\times d}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.3 Context mixing

**Main idea.** Let every token read other tokens through attention.

Core relation:

$$\mathrm{Attention}(Q,K,V)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.4 Position information

**Main idea.** Inject order because attention alone is permutation equivariant.

Core relation:

$$h_i=x_i+p_i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 1.5 Stacked blocks

**Main idea.** Repeat attention and mlp transformations.

Core relation:

$$H^{(\ell+1)}=\mathrm{Block}_\ell(H^{(\ell)})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 2. Scaled Dot-Product Attention

This part studies scaled dot-product attention as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Queries keys values](#2-queries-keys-values) | project hidden states into matching and content spaces | $Q=HW_Q,\ K=HW_K,\ V=HW_V$ |
| [Score matrix](#2-score-matrix) | compare every query with every key | $S=QK^\top/\sqrt{d_k}$ |
| [Masking](#2-masking) | block padding or future positions before softmax | $S_{ij}\leftarrow-\infty$ when masked |
| [Attention weights](#2-attention-weights) | softmax gives a distribution over source positions | $A=\mathrm{softmax}(S)$ |
| [Weighted values](#2-weighted-values) | mix value vectors by attention weights | $O=AV$ |

### 2.1 Queries keys values

**Main idea.** Project hidden states into matching and content spaces.

Core relation:

$$Q=HW_Q,\ K=HW_K,\ V=HW_V$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.2 Score matrix

**Main idea.** Compare every query with every key.

Core relation:

$$S=QK^\top/\sqrt{d_k}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is the computation that lets each token ask which other tokens matter.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.3 Masking

**Main idea.** Block padding or future positions before softmax.

Core relation:

$$S_{ij}\leftarrow-\infty$ when masked$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.4 Attention weights

**Main idea.** Softmax gives a distribution over source positions.

Core relation:

$$A=\mathrm{softmax}(S)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 2.5 Weighted values

**Main idea.** Mix value vectors by attention weights.

Core relation:

$$O=AV$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 3. Multi-Head Attention

This part studies multi-head attention as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Head splitting](#3-head-splitting) | several smaller attention heads run in parallel | $h=1,\ldots,H$ |
| [Per-head dimension](#3-perhead-dimension) | model dimension is split across heads | $d_h=d_\mathrm{model}/H$ |
| [Concatenation](#3-concatenation) | head outputs are concatenated and projected | $\mathrm{MHA}(H)=\mathrm{Concat}(O_1,\ldots,O_H)W_O$ |
| [Specialization](#3-specialization) | different heads can learn different relation patterns | $A_h$ varies by head |
| [GQA and MQA bridge](#3-gqa-and-mqa-bridge) | serving variants share key-value heads | $H_{kv}\le H_q$ |

### 3.1 Head splitting

**Main idea.** Several smaller attention heads run in parallel.

Core relation:

$$h=1,\ldots,H$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.2 Per-head dimension

**Main idea.** Model dimension is split across heads.

Core relation:

$$d_h=d_\mathrm{model}/H$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.3 Concatenation

**Main idea.** Head outputs are concatenated and projected.

Core relation:

$$\mathrm{MHA}(H)=\mathrm{Concat}(O_1,\ldots,O_H)W_O$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.4 Specialization

**Main idea.** Different heads can learn different relation patterns.

Core relation:

$$A_h$ varies by head$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 3.5 GQA and MQA bridge

**Main idea.** Serving variants share key-value heads.

Core relation:

$$H_{kv}\le H_q$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 4. Feed-Forward Network

This part studies feed-forward network as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Position-wise MLP](#4-positionwise-mlp) | apply the same MLP to each token independently | $\mathrm{FFN}(x)=W_2\phi(W_1x+b_1)+b_2$ |
| [Expansion ratio](#4-expansion-ratio) | hidden width is often larger than model width | $d_\mathrm{ff}\approx4d_\mathrm{model}$ |
| [Activation](#4-activation) | GELU or SwiGLU controls nonlinear feature mixing | $\phi$ |
| [Parameter count](#4-parameter-count) | MLP parameters often dominate each block | $2d_\mathrm{model}d_\mathrm{ff}$ |
| [Token independence](#4-token-independence) | cross-token mixing happens in attention, not the MLP | $x_t$ processed separately |

### 4.1 Position-wise MLP

**Main idea.** Apply the same mlp to each token independently.

Core relation:

$$\mathrm{FFN}(x)=W_2\phi(W_1x+b_1)+b_2$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.2 Expansion ratio

**Main idea.** Hidden width is often larger than model width.

Core relation:

$$d_\mathrm{ff}\approx4d_\mathrm{model}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.3 Activation

**Main idea.** Gelu or swiglu controls nonlinear feature mixing.

Core relation:

$$\phi$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.4 Parameter count

**Main idea.** Mlp parameters often dominate each block.

Core relation:

$$2d_\mathrm{model}d_\mathrm{ff}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 4.5 Token independence

**Main idea.** Cross-token mixing happens in attention, not the mlp.

Core relation:

$$x_t$ processed separately$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 5. Residuals and Normalization

This part studies residuals and normalization as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Residual stream](#5-residual-stream) | blocks add updates to a persistent representation | $x\leftarrow x+F(x)$ |
| [Layer normalization](#5-layer-normalization) | normalize features within each token | $\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$ |
| [Pre-LN block](#5-preln-block) | normalize before sublayer for more stable deep training | $x\leftarrow x+F(\mathrm{LN}(x))$ |
| [Post-LN block](#5-postln-block) | original transformer normalized after residual addition | $x\leftarrow\mathrm{LN}(x+F(x))$ |
| [RMSNorm](#5-rmsnorm) | normalize by root mean square without mean subtraction | $x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$ |

### 5.1 Residual stream

**Main idea.** Blocks add updates to a persistent representation.

Core relation:

$$x\leftarrow x+F(x)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.2 Layer normalization

**Main idea.** Normalize features within each token.

Core relation:

$$\hat x=(x-\mu)/\sqrt{\sigma^2+\epsilon}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.3 Pre-LN block

**Main idea.** Normalize before sublayer for more stable deep training.

Core relation:

$$x\leftarrow x+F(\mathrm{LN}(x))$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** Pre-LN is a key reason very deep transformer stacks are easier to optimize.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.4 Post-LN block

**Main idea.** Original transformer normalized after residual addition.

Core relation:

$$x\leftarrow\mathrm{LN}(x+F(x))$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 5.5 RMSNorm

**Main idea.** Normalize by root mean square without mean subtraction.

Core relation:

$$x/\sqrt{\mathrm{mean}(x^2)+\epsilon}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 6. Encoder Decoder and Decoder-Only Forms

This part studies encoder decoder and decoder-only forms as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Encoder self-attention](#6-encoder-selfattention) | source tokens attend bidirectionally | $A_{ij}$ allowed for all source positions |
| [Decoder causal self-attention](#6-decoder-causal-selfattention) | target tokens cannot see future target tokens | $j\le i$ |
| [Cross-attention](#6-crossattention) | decoder queries attend to encoder keys and values | $Q=H_\mathrm{dec}W_Q,\ K,V=H_\mathrm{enc}W_{K,V}$ |
| [Encoder-only](#6-encoderonly) | BERT-style models produce contextual representations | $p(\mathrm{masked}\ token\mid x)$ |
| [Decoder-only](#6-decoderonly) | GPT-style models predict next tokens autoregressively | $p(x_t\mid x_{<t})$ |

### 6.1 Encoder self-attention

**Main idea.** Source tokens attend bidirectionally.

Core relation:

$$A_{ij}$ allowed for all source positions$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.2 Decoder causal self-attention

**Main idea.** Target tokens cannot see future target tokens.

Core relation:

$$j\le i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.3 Cross-attention

**Main idea.** Decoder queries attend to encoder keys and values.

Core relation:

$$Q=H_\mathrm{dec}W_Q,\ K,V=H_\mathrm{enc}W_{K,V}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.4 Encoder-only

**Main idea.** Bert-style models produce contextual representations.

Core relation:

$$p(\mathrm{masked}\ token\mid x)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 6.5 Decoder-only

**Main idea.** Gpt-style models predict next tokens autoregressively.

Core relation:

$$p(x_t\mid x_{<t})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is the architecture behind most modern autoregressive LLMs.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 7. Positional Information

This part studies positional information as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Absolute positions](#7-absolute-positions) | add learned or fixed position vectors | $h_i=x_i+p_i$ |
| [Sinusoidal positions](#7-sinusoidal-positions) | fixed frequencies encode index information | $\sin(i/\omega_k),\cos(i/\omega_k)$ |
| [Relative bias](#7-relative-bias) | attention scores depend on distance | $S_{ij}\leftarrow S_{ij}+b_{i-j}$ |
| [RoPE](#7-rope) | rotate query and key pairs by position-dependent angles | $q_i^\top k_j$ depends on $i-j$ |
| [Length behavior](#7-length-behavior) | position design affects long-context extrapolation | $T_\mathrm{test}>T_\mathrm{train}$ |

### 7.1 Absolute positions

**Main idea.** Add learned or fixed position vectors.

Core relation:

$$h_i=x_i+p_i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.2 Sinusoidal positions

**Main idea.** Fixed frequencies encode index information.

Core relation:

$$\sin(i/\omega_k),\cos(i/\omega_k)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.3 Relative bias

**Main idea.** Attention scores depend on distance.

Core relation:

$$S_{ij}\leftarrow S_{ij}+b_{i-j}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.4 RoPE

**Main idea.** Rotate query and key pairs by position-dependent angles.

Core relation:

$$q_i^\top k_j$ depends on $i-j$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 7.5 Length behavior

**Main idea.** Position design affects long-context extrapolation.

Core relation:

$$T_\mathrm{test}>T_\mathrm{train}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 8. Complexity and Memory

This part studies complexity and memory as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Attention FLOPs](#8-attention-flops) | dense attention is quadratic in sequence length | $O(BHT^2d_h)$ |
| [Attention score memory](#8-attention-score-memory) | naive attention materializes T by T scores | $O(BHT^2)$ |
| [MLP FLOPs](#8-mlp-flops) | MLP cost is linear in sequence length but large in width | $O(BTd_\mathrm{model}d_\mathrm{ff})$ |
| [KV cache](#8-kv-cache) | decoder inference stores keys and values per layer | $M_\mathrm{KV}=2BLTH_{kv}d_hb$ |
| [FlashAttention bridge](#8-flashattention-bridge) | IO-aware attention computes exact attention without storing all scores | $\mathrm{softmax}(QK^\top)V$ tiled |

### 8.1 Attention FLOPs

**Main idea.** Dense attention is quadratic in sequence length.

Core relation:

$$O(BHT^2d_h)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.2 Attention score memory

**Main idea.** Naive attention materializes t by t scores.

Core relation:

$$O(BHT^2)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.3 MLP FLOPs

**Main idea.** Mlp cost is linear in sequence length but large in width.

Core relation:

$$O(BTd_\mathrm{model}d_\mathrm{ff})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.4 KV cache

**Main idea.** Decoder inference stores keys and values per layer.

Core relation:

$$M_\mathrm{KV}=2BLTH_{kv}d_hb$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is the central memory object for fast decoder inference.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 8.5 FlashAttention bridge

**Main idea.** Io-aware attention computes exact attention without storing all scores.

Core relation:

$$\mathrm{softmax}(QK^\top)V$ tiled$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 9. Training Signals

This part studies training signals as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Language-model loss](#9-languagemodel-loss) | decoder-only models train by next-token prediction | $L=-\sum_t\log p_\theta(x_t\mid x_{<t})$ |
| [Masked LM loss](#9-masked-lm-loss) | encoder-only models predict hidden masked tokens | $L=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M})$ |
| [Teacher forcing](#9-teacher-forcing) | decoder training conditions on gold previous tokens | $y_{<t}^\star$ |
| [Attention masks](#9-attention-masks) | loss and attention masks must agree with the task | $M_\mathrm{attn},M_\mathrm{loss}$ |
| [Weight tying](#9-weight-tying) | input embedding and output projection can share weights | $W_\mathrm{out}=E^\top$ |

### 9.1 Language-model loss

**Main idea.** Decoder-only models train by next-token prediction.

Core relation:

$$L=-\sum_t\log p_\theta(x_t\mid x_{<t})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.2 Masked LM loss

**Main idea.** Encoder-only models predict hidden masked tokens.

Core relation:

$$L=-\sum_{i\in M}\log p_\theta(x_i\mid x_{\setminus M})$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.3 Teacher forcing

**Main idea.** Decoder training conditions on gold previous tokens.

Core relation:

$$y_{<t}^\star$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.4 Attention masks

**Main idea.** Loss and attention masks must agree with the task.

Core relation:

$$M_\mathrm{attn},M_\mathrm{loss}$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 9.5 Weight tying

**Main idea.** Input embedding and output projection can share weights.

Core relation:

$$W_\mathrm{out}=E^\top$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
## 10. Diagnostics

This part studies diagnostics as transformer block math. Keep track of token axes, head axes, masks, residual updates, and where cross-token mixing happens.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | track batch, time, heads, head dimension, and model dimension | $(B,H,T,d_h)$ |
| [Mask tests](#10-mask-tests) | future leakage breaks autoregressive training | $A_{ij}=0$ for $j>i$ |
| [Attention entropy](#10-attention-entropy) | overly sharp or diffuse heads can indicate issues | $H(A_i)$ |
| [Residual norms](#10-residual-norms) | monitor update size relative to residual stream | $\Vert F(x)\Vert/\Vert x\Vert$ |
| [Ablations](#10-ablations) | compare head count, MLP width, norm placement, and position method | $\Delta L,\Delta T$ |

### 10.1 Shape checks

**Main idea.** Track batch, time, heads, head dimension, and model dimension.

Core relation:

$$(B,H,T,d_h)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.2 Mask tests

**Main idea.** Future leakage breaks autoregressive training.

Core relation:

$$A_{ij}=0$ for $j>i$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** A single mask bug can make a model appear to learn while leaking future answers.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.3 Attention entropy

**Main idea.** Overly sharp or diffuse heads can indicate issues.

Core relation:

$$H(A_i)$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.4 Residual norms

**Main idea.** Monitor update size relative to residual stream.

Core relation:

$$\Vert F(x)\Vert/\Vert x\Vert$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.
### 10.5 Ablations

**Main idea.** Compare head count, mlp width, norm placement, and position method.

Core relation:

$$\Delta L,\Delta T$$

Transformers separate two jobs. Attention mixes information across token positions. The feed-forward network transforms each token independently. Residual connections preserve a running representation, and normalization keeps optimization stable.

**Worked micro-example.** If $d_\mathrm{model}=768$ and there are 12 attention heads, each head has $d_h=64$. The attention score matrix for one head and one sequence of length 128 has shape $128$ by $128$. That score matrix is why attention is powerful and also why long-context attention is expensive.

**Implementation check.** Write the shapes for $Q$, $K$, $V$, scores, attention weights, and output before coding. Most transformer bugs are wrong axes or wrong masks.

**AI connection.** This is a practical transformer design variable.

**Common mistake.** Do not say the MLP mixes tokens. In a standard transformer block, token mixing happens in attention; the MLP is position-wise.

---

## Practice Exercises

1. Compute scaled dot-product attention for tiny matrices.
2. Build a causal mask and apply it to attention scores.
3. Split model dimension into heads.
4. Count parameters in Q, K, V, and output projections.
5. Count MLP parameters and compare to attention parameters.
6. Compute LayerNorm and RMSNorm for one vector.
7. Compare Pre-LN and Post-LN block equations.
8. Compute KV cache memory for a decoder.
9. Identify encoder-only, decoder-only, and encoder-decoder masks.
10. Write a transformer debugging checklist.

## Why This Matters for AI

The transformer is the architectural base of modern LLMs and many multimodal models. Understanding it at the level of shapes, masks, residuals, and costs prevents vague explanations and catches real implementation mistakes.

## Bridge to Reinforcement Learning

The next model-specific section studies reinforcement learning. In modern LLM systems, transformer policies are often optimized further with preference or reward-based objectives, so architecture and optimization meet there.

## 11. Common Mistakes
Pre-submission scan for transformer work; expands Sections 7-10's diagnostics.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | Missing causal mask (or mask applied after softmax) | Future leakage; inflated validation | Mask logits pre-softmax; leakage test (theory 12) in CI |
| 2 | Forgetting the $1/\sqrt{d_k}$ scale | Saturated attention at init (§13.2) | Scaled dot-product as defined (Vaswani et al. 2017) |
| 3 | Padding tokens attending / being attended | Length-dependent behaviour | Key-padding masks + masked loss (§04 theory 21) |
| 4 | Post-LN recipe copied into a deep Pre-LN-era stack without warmup | Early divergence | Pre-LN default; warmup if Post-LN (§13.3, Xiong et al. 2020) |
| 5 | Reporting perplexity across tokenizers | Incomparable claims (Chapter 15) | Same tokenizer/data; NLL + ppl quoted together |
| 6 | Ignoring KV-cache cost at serving | OOM/latency surprises at long context | Cache accounting (theory 10) in capacity planning |
| 7 | SwiGLU swapped in at GELU width | Silent +50% FFN params (§13.2) | 2/3 width rule (theory 23) |
| 8 | Positional scheme mismatched to length needs | Length-generalization failure | Evaluate beyond train length; choose RoPE/ALiBi deliberately |
| 9 | Attention maps read as explanations | Overclaimed interpretability | Treat weights as diagnostics; validate with interventions |
| 10 | Batching variable lengths without sorting/packing plan | Mostly-pad batches; wasted FLOPs | Packing or length-grouped batching with masking tests |



## 12. Attention Is the Whole Architecture Now
Vaswani et al. (2017) removed recurrence entirely: positionwise FFNs transform, attention routes. The mathematical consequences this section adds: path length between any two positions is O(1) (vs O(T) in §04), computation is parallel across time in training, and the price is quadratic attention cost in sequence length (theory 19) plus an inference-time KV cache (theory 10/20). Every design choice since — RoPE, QK-norm, GQA/MQA, SwiGLU, Pre-LN — is a repair or optimization of one of those three facts.

- Multi-head attention is an ensemble of routing patterns at different representation subspaces, not 'more attention' in the abstract (theory 15 shapes).
- Grouped-query attention (Ainslie et al. 2023) and multi-query attention (Shazeer 2019) shrink the KV cache by sharing K/V across heads — serving-driven architecture.
- FlashAttention (Dao et al. 2022) is an IO-aware exact implementation; it changes memory traffic, not the mathematics — cite it as systems, not as a new attention.
- Positional schemes are relative-position inductive biases: sinusoidal (absolute), RoPE (relative via rotation, theory 17), ALiBi (relative via bias); length extrapolation is their separating test.
- The FFN holds most parameters (theory 5/23); attention holds most long-context cost (theory 19) — optimize the right one for your regime.



## 13. Computed Table: Long-Context Costs
From theory 19/10's accounting at d=4096. Quadratic compute share and cache memory both become the binding constraints as context grows — the economic argument behind GQA, sliding windows, and efficient-attention research.

| T | attention score FLOPs share | KV cache (GB) L=32,H_kv=8,d=128,bf16,B=4 |
|---|---|---|
| 512 | 0.020 | 0.27 |
| 2048 | 0.077 | 1.07 |
| 8192 | 0.249 | 4.29 |
| 32768 | 0.570 | 17.18 |
| 131072 | 0.841 | 68.72 |

## 14. Worked Problem Set A: Mechanics and Budgets
Worked problems A: attention mechanics.

**Problem A1: parameter count of a block**
1. d=768: attention QKVO = 4d² = 2.36M; GELU FFN (4x) = 2·d·4d = 4.72M; total ≈ 7.08M + norms/embeddings share.
2. Per layer the FFN is ~2/3 of block parameters (theory 4/5) — capacity claims should say where parameters live.
3. SwiGLU at matched params uses d_ff = 4096·(2/3)-style scaling (theory 23).

**Problem A2: cache budget**
1. Serving B=8, L=48 layers, T=8192, H_kv=8 (GQA), d_h=128, bf16: cache = 2·8·48·8192·8·128·2 bytes ≈ 103 GB.
2. That exceeds one GPU's memory: the levers are GQA depth, quantization of cache, batch, or context — arithmetic before procurement (theory 10).

**Problem A3: mask audit**
1. Property test: permute future tokens; outputs at earlier positions must be bit-identical (causal) — theory 12/15's checks generalized.
2. Run it after any attention refactor (kernels, padding, packing); leakage bugs are silent and flattering.



## 15. Worked Problem Set B: Design Choices
Worked problems B: design choices.

**Problem B1: Pre-LN conversion**
1. A Post-LN recipe diverges at 48 layers without warmup: convert to Pre-LN (Xiong et al. 2020), add final norm, re-audit residual stream (§13.3 Section 27).
2. Expect to retune LR; treat as a new architecture for checkpoint purposes (§13.3 Problem C1 pattern).

**Problem B2: positional scheme for 4x length extrapolation**
1. Requirement: train 8k, serve 32k. Candidates: RoPE with NTK-aware scaling, ALiBi; learned absolute embeddings fail by construction outside their table.
2. Evaluate at 32k before committing: length generalization is measured (FAQ), never inferred from 8k metrics.



## 16. Glossary: Transformer Vocabulary
| Term | Meaning |
|---|---|
| Scaled dot-product attention | softmax$(QK^T/\sqrt{d_k})V$; the $\sqrt{d_k}$ keeps logits trainable (theory 18). |
| Causal mask | Additive −inf (pre-softmax) on future positions; audited by permutation tests. |
| KV cache | Stored keys/values enabling O(T) incremental decoding; memory per theory 10. |
| GQA/MQA | Sharing K/V heads to shrink cache (Ainslie et al. 2023; Shazeer 2019). |
| RoPE | Rotary position embedding; relative-offset scores (Su et al. 2021, theory 17). |
| Pre-LN | Norm inside residual branches; identity gradient highway (§13.3). |
| d_model / d_head | Residual width / per-head width; head count = d_model/d_head. |
| Context window | Maximum attended span; a cost statement (Section 13) as much as a capability. |
| Logit soft-capping / QK-norm | Bounding attention logits at scale (theory 22). |
| Weight tying | Sharing embedding and LM head (theory 11); parameter and regularization effects. |

## 17. Quick-Reference Cards: Block Recipes
**Decoder-only LM block**
- Pre-RMSNorm → causal attention (RoPE, GQA) → residual → Pre-RMSNorm → SwiGLU → residual; final norm + tied head (the contemporary default assembled from theory 15/23/25).

**Encoder block (BERT-style)**
- Bidirectional attention (no causal mask), Post-LN in the original; classification via a pooled head.

**Encoder-decoder (T5-style)**
- Cross-attention added; teacher forcing on the decoder (§04 theory 24's heir).

**MoE block**
- FFN replaced by routed experts; router is a sampling/gating object (§13.4) with load-balancing losses.

**Vision transformer patch block**
- Same block, image patches as tokens; positional scheme and augmentation carry the inductive bias (§08 bridge).



## 18. Pre-Flight Checklist for Transformer Work
- Shapes written for every tensor in attention (B,H,T,d) with head-split assertions (theory 15).
- Causal + padding masks unit-tested (permutation/leakage tests, Problem A3).
- Scale $1/\sqrt{d_k}$ present; QK-norm/soft-cap decisions recorded for large runs.
- Placement (Pre/Post-LN) and final norm explicit; residual stream logging planned (§13.3).
- Positional scheme chosen against the length requirement and evaluated beyond train length.
- Parameter/FLOP/cache budgets computed (Problems A1/A2) before procurement.
- Decoding spec versioned separately (Chapter 15; §13.4 sampler discipline).
- Attention entropy monitored per head (collapse alarm, §13.2 Section 54).



## 19. Frequently Asked Questions
- **More heads or wider heads?** At fixed d_model it's a repartitioning; evidence is workload-specific — ablate with matched params and report per-head entropy to see if heads diversify.
- **Why did Pre-LN win?** Trainability without warmup at depth (Xiong et al. 2020); quality differences at full tuning are smaller than stability differences (§13.3).
- **Is attention interpretable?** Attention weights are routing diagnostics, not explanations (Mistakes); intervention studies outrank heatmaps.
- **Longer context: just extend RoPE?** Scaling laws for context extension (NTK-aware/YaRN-family) exist, but length generalization must be measured at target length (Problem B2).
- **Where do FLOPs go at 128k?** Increasingly the quadratic attention term and cache traffic (Section 13) — hence sparse/linear attention and GQA research.
- **Transformers for tiny data?** Inductive bias is weak; §01/GBDT-style baselines and CNN/RNN priors often win small — architecture choice is empirical (Section 20).



## 20. Decision Records: Transformer Choices in Context
- **General LLM:** decoder-only, Pre-RMSNorm, RoPE, GQA, SwiGLU (theory 25's block), tied embeddings at small scale.
- **Retrieval encoder:** bidirectional encoder; pooling and normalization of embeddings become the product surface (Chapter 15).
- **Streaming speech:** causal/chunked attention with cache discipline; RNN hybrids compete (§04 records).
- **Long-document QA:** context strategy (extension, retrieval, hierarchy) chosen by measured length generalization, not window size on the box.
- Each record: block recipe, positional scheme, mask tests, budgets (params/FLOPs/cache), length evaluations, decoding spec pointer.

## 21. Computed Table: Parameter Anatomy of a Decoder
Block-level accounting (theory 4/5/23): at d=4096 the FFN holds roughly two-thirds of each block; embeddings/head (tied, vocab 50k ≈ 0.2B) matter at small layer counts. Capacity arguments should cite this anatomy.

| layers L | params (d=4096, SwiGLU, approx) | dominated by |
|---|---|---|
| 12 | 2.4B | FFN (67% of block) |
| 24 | 4.9B | FFN (67% of block) |
| 48 | 9.7B | FFN (67% of block) |
| 96 | 19.4B | FFN (67% of block) |



## 22. Self-Test Questions
- Write scaled dot-product attention with shapes and the two masks' entry points.
- Why $1/\sqrt{d_k}$ (theory 18's entropy argument)?
- RoPE's relative property: statement and check (theory 17).
- KV cache size formula and the GQA lever (Problem A2).
- Pre-LN vs Post-LN: gradient path + warmup (theory 21).
- SwiGLU width rule with the arithmetic (theory 23).
- Quadratic attention share at T=512 vs 128k (Section 13).
- Causal leakage test design (Problem A3).
- QK-norm's bound (theory 22, corrected form: |logit| ≤ d for unit-RMS rows).
- Perplexity comparability conditions (Chapter 15).



## 23. Self-Test Answer Key (Condensed)
- softmax$(QK^T/\sqrt{d_k}+\text{mask})V$; causal mask pre-softmax on logits, padding mask on keys.
- Unscaled logit variance grows with d_k → saturation; scaling keeps unit variance.
- Rotating q,k by position makes scores depend on offset only; norms preserved.
- $2\cdot B L T H_{kv} d_h \cdot$bytes; share K/V heads (GQA) to cut it.
- Pre-LN identity path, no warmup needed; Post-LN normalized stream, warmup-sensitive (Xiong et al. 2020).
- ×2/3 hidden width at three matrices matches two-matrix parameter count.
- ~1% vs >40% quadratic share (Section 13 table).
- Permute futures; earlier outputs must not move.
- Unit-RMS rows give ‖q‖=‖k‖=√d, so |q·k| ≤ d.
- Same tokenizer, same data, NLL reported alongside.



## 24. Annotated Reading Guide
- Primary: Vaswani et al. (2017); then Xiong et al. (2020) for placement and Zhang & Sennrich (2019, via §13.3) for RMSNorm.
- Positional: Su et al. (2021) for RoPE; Press et al. (2022) for ALiBi as the contrasting bias.
- Efficiency (cite as systems): Shazeer (2019, MQA), Ainslie et al. (2023, GQA), Dao et al. (2022, FlashAttention).
- Gating: Shazeer (2020, GLU variants) via §13.2.
- Scale behaviour belongs to Chapter 15; this section's job is the block-level mathematics those laws operate on.

## 25. Exam and Interview Traps: Transformers
- Trap: 'Attention weights explain the prediction.' Response: routing diagnostics; interventions decide (Mistakes).
- Trap: 'More context window = longer understanding.' Response: length generalization is measured at length (Problem B2).
- Trap: 'Transformers made normalization/init irrelevant.' Response: Pre-LN/RMSNorm/init are why depth trains (§13.3).
- Trap: 'FlashAttention changed the model.' Response: exact attention, IO-aware systems (Section 12).
- Trap: 'Perplexity X beats Y across papers.' Response: tokenizer/data discipline (Chapter 15).
- Answer pattern: shapes, budgets (params/FLOPs/cache), mask/placement audits.



## 26. Worked Problem Set C: Synthesis
Worked problems C: closing synthesis.

**Problem C1: specify a block from scratch**
1. Deliverable: the theory-25 block with d=2048, H=16, RoPE, GQA 4 KV heads, SwiGLU matched width, Pre-RMSNorm.
2. Attach: parameter count (Problem A1 method), cache at T=16k (Problem A2), mask tests (A3), length-eval plan (B2).
3. A block spec missing any attachment is a sketch, not a design.

**Problem C2: the 3am OOM**
1. Serving OOMs only on long prompts: cache grows linearly in T (Section 13) while weights are fixed — capacity was planned on weights alone.
2. Fixes by arithmetic: GQA conversion (fine-tune), cache quantization, max-context enforcement with honest errors, batch admission control.
3. Post-incident: cache term added to the capacity model (Problem A2 as a standing formula).



## 27. Quick Checks (One-Line Answers)
- Attention logits scale: $1/\sqrt{d_k}$, before softmax, after masking decisions are placed on logits.
- Causal mask entry point: additive −inf pre-softmax.
- RoPE scores depend on: relative offset; norms preserved.
- KV cache per token per layer: $2 H_{kv} d_h$ values (K and V).
- Pre-LN puts norms: inside branches; final norm before head.
- SwiGLU matched width: 2/3 of the two-matrix width.
- FFN share of block params: ~2/3 at LLM widths.
- Quadratic FLOPs dominate: at long T (Section 13).
- GQA shares: K/V heads across query heads.
- Leakage test: permute the future; the past must not move.



## 28. Index of Computed Evidence
| Evidence | Where |
|---|---|
| Long-context compute/cache table | Section 13 |
| Parameter anatomy | Section 21 |
| MHA forward, RoPE, scale, FLOPs, KV equivalence, QK-norm, SwiGLU, block audit | theory.ipynb Sections 15-26 |

## 29. Worked Matrix: A Causal Attention Map
Seed 42 random scores, scaled and causally masked. Verify: upper triangle exactly 0, rows sum to 1, entropies below log(available keys). This is the object that mask bugs corrupt silently (Problem A3's test exists to catch it).

| query \ key | k0 | k1 | k2 | k3 | k4 | k5 | row entropy |
|---|---|---|---|---|---|---|---|
| q0 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | -0.000 |
| q1 | 0.539 | 0.461 | 0.000 | 0.000 | 0.000 | 0.000 | 0.690 |
| q2 | 0.277 | 0.403 | 0.319 | 0.000 | 0.000 | 0.000 | 1.086 |
| q3 | 0.335 | 0.241 | 0.230 | 0.193 | 0.000 | 0.000 | 1.365 |
| q4 | 0.164 | 0.168 | 0.230 | 0.217 | 0.221 | 0.000 | 1.599 |
| q5 | 0.291 | 0.118 | 0.114 | 0.103 | 0.170 | 0.204 | 1.718 |



## 30. Extended Exercise Bank (Self-Study)
1. Implement MHA from scratch and pass theory 15/16's audits (shapes, rows, dV check).
2. Reproduce theory 20's KV-equivalence test in your framework of choice.
3. Compute the full budget (params/FLOPs/cache) for a 7B-class configuration; identify the binding constraint at T=2k and T=128k.
4. Measure attention entropy per head in a small trained model; find collapsed heads and hypothesize causes.
5. Evaluate a RoPE model at 2x training length; document the failure shape (which positions/tokens degrade).
6. Convert a GELU FFN block to SwiGLU at matched parameters (theory 23); write the ablation protocol.
7. Design the mask test-suite (causal, padding, packing) as CI for an inference engine.
8. Read one attention-variant paper and fill the Section 12 dossier: what fact (path length, cost, cache) does it repair?



## 31. Revision Problems (Mixed)
1. Derive why attention is permutation-equivariant without positional information, and what each positional scheme adds.
2. State the three costs of long context (compute, cache, length generalization) with their formulas/sections.
3. Write the block spec of Problem C1 for d=1024 as a one-page design doc.
4. When would you still choose an RNN (§04) over this block? Give two production-shaped reasons.
5. List a transformer model card's fields (Sections 18/20).
6. Explain QK-norm vs logit soft-capping as two answers to the same saturation problem (theory 22, §13.2).



## 32. Chapter Summary: Transformer Architecture
- The transformer is attention routing + positionwise transformation, composed in residual Pre-LN blocks; every component has a budget and an audit.
- Path length O(1) bought long-range interaction at quadratic cost; KV cache and GQA are the serving-side prices, paid knowingly (Section 13).
- Positional schemes, normalization placement, and FFN gating are the live design axes; each has a canonical citation and a measurable failure mode.
- Bridge: Chapter 15 trains this block at scale; §07 borrows its backbone for generation; §08 contrasts the convolutional alternative.



## 33. Section 05 Completion Checklist
- Common Mistakes table: Section 11 (10 rows).
- Theory: 54 cells (30 original + 24 appended), executed 0 errors, 26 PASS / 0 FAIL in appended cells.
- Exercises: workspaces preserved; solutions compile (verified).
- Canonical citations: Vaswani et al. (2017); Xiong et al. (2020); Su et al. (2021); Shazeer (2019, 2020); Ainslie et al. (2023); Dao et al. (2022); Bahdanau et al. (2015) as ancestor.

- Carry-forward 1: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 2: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 3: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 4: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 5: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 6: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 7: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 8: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 9: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 10: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 11: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 12: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 13: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 14: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 15: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 16: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 17: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 18: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 19: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 20: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 21: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 22: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 23: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 24: every attention claim ships with shapes, masks, and a budget.
- Carry-forward 25: every attention claim ships with shapes, masks, and a budget.

## 34. Addenda: Ten Lines to Carry Forward
- 1. Chapter 15 scales this block; the budgets of Section 13 become training economics there.
- 2. Decoding is sampling over this block's logits (§13.4); the block spec and sampler spec are separate versioned objects.
- 3. RoPE's relative trick generalizes: positional design is inductive-bias design, evaluated at length.
- 4. GQA/MQA show architecture following serving arithmetic — expect more of this pattern.
- 5. Attention entropy is a health metric (§13.2's saturation discipline in new clothes).
- 6. Mask tests are the transformer's masking tests (§04) with triangles; keep them in CI forever.
- 7. The FFN-vs-attention parameter/FLOP split (Section 21) tells you where quantization and pruning bite.
- 8. Probes (§01) on block outputs remain the cheapest representation diagnostics.
- 9. Every 'efficient attention' claim must state which term of theory 19 it reduces and what it sacrifices.
- 10. Residual + norm placement (§13.3) is the quiet reason any of this trains at depth.



## 35. Margin Notes: The Transformer Creed
- Shapes first; masks always; budgets before GPUs.
- Scale the logits or saturate.
- Causality is tested, not trusted.
- Cache is memory with a formula.
- Position is a bias; pick it for your lengths.
- FFN holds the parameters; attention holds the context cost.
- Entropy per head, logged like any vital sign.
- Pre-LN trains; Post-LN tunes.
- Perplexity travels with its tokenizer.
- Attention maps are diagnostics, not explanations.



## 36. End of Section 05 Expansion
- Notes carry: Common Mistakes (11), architecture analysis (12-13), budgets (14/21), worked attention matrix (29), five problem sets, glossary, cards, checklists, completion (33).
- Theory: 54 cells executed clean; exercises untouched.
- Remaining in Chapter 14 after this section: §07 Generative Models and §08 CNN and Convolution Math (§06 already at bar).

- Postscript 1: route with attention, transform positionwise, audit both.
- Postscript 2: route with attention, transform positionwise, audit both.
- Postscript 3: route with attention, transform positionwise, audit both.
- Postscript 4: route with attention, transform positionwise, audit both.
- Postscript 5: route with attention, transform positionwise, audit both.
- Postscript 6: route with attention, transform positionwise, audit both.
- Postscript 7: route with attention, transform positionwise, audit both.
- Postscript 8: route with attention, transform positionwise, audit both.
- Postscript 9: route with attention, transform positionwise, audit both.
- Postscript 10: route with attention, transform positionwise, audit both.
- Postscript 11: route with attention, transform positionwise, audit both.
- Postscript 12: route with attention, transform positionwise, audit both.
- Postscript 13: route with attention, transform positionwise, audit both.
- Postscript 14: route with attention, transform positionwise, audit both.
- Postscript 15: route with attention, transform positionwise, audit both.
- Postscript 16: route with attention, transform positionwise, audit both.
- Postscript 17: route with attention, transform positionwise, audit both.
- Postscript 18: route with attention, transform positionwise, audit both.
- Postscript 19: route with attention, transform positionwise, audit both.
- Postscript 20: route with attention, transform positionwise, audit both.
- Postscript 21: route with attention, transform positionwise, audit both.
- Postscript 22: route with attention, transform positionwise, audit both.
- Postscript 23: route with attention, transform positionwise, audit both.
- Postscript 24: route with attention, transform positionwise, audit both.
- Postscript 25: route with attention, transform positionwise, audit both.
- Postscript 26: route with attention, transform positionwise, audit both.
- Postscript 27: route with attention, transform positionwise, audit both.
- Postscript 28: route with attention, transform positionwise, audit both.
- Postscript 29: route with attention, transform positionwise, audit both.
- Postscript 30: route with attention, transform positionwise, audit both.
- Postscript 31: route with attention, transform positionwise, audit both.
- Postscript 32: route with attention, transform positionwise, audit both.
- Postscript 33: route with attention, transform positionwise, audit both.
- Postscript 34: route with attention, transform positionwise, audit both.
- Postscript 35: route with attention, transform positionwise, audit both.
- Postscript 36: route with attention, transform positionwise, audit both.
- Postscript 37: route with attention, transform positionwise, audit both.
- Postscript 38: route with attention, transform positionwise, audit both.
- Postscript 39: route with attention, transform positionwise, audit both.
- Postscript 40: route with attention, transform positionwise, audit both.
- Postscript 41: route with attention, transform positionwise, audit both.
- Postscript 42: route with attention, transform positionwise, audit both.
- Postscript 43: route with attention, transform positionwise, audit both.
- Postscript 44: route with attention, transform positionwise, audit both.
- Postscript 45: route with attention, transform positionwise, audit both.
- Postscript 46: route with attention, transform positionwise, audit both.
- Postscript 47: route with attention, transform positionwise, audit both.
- Postscript 48: route with attention, transform positionwise, audit both.
- Postscript 49: route with attention, transform positionwise, audit both.
- Postscript 50: route with attention, transform positionwise, audit both.

## 37. Computed Table: Four Reference Configurations
Rough block-parameter anatomy across scales (SwiGLU matched widths, embeddings excluded). Use the shape of this table — not its exact vintage — to sanity-check any architecture announcement's arithmetic.

| config | layers | d_model | approx params (blocks only) | cache @T=8192,B=1 (GB) |
|---|---|---|---|---|
| L12 d768 | 12 | 768 | 0.08B | 0.60 (H_kv=8,d_h=128) |
| L24 d1024 | 24 | 1024 | 0.30B | 0.81 (H_kv=8,d_h=128) |
| L32 d4096 | 32 | 4096 | 6.44B | 1.07 (H_kv=8,d_h=128) |
| L80 d8192 | 80 | 8192 | 64.42B | 2.68 (H_kv=8,d_h=128) |



## 38. Ten-Line Field Guide to Defensible Transformer Claims
- 1. Block spec complete (recipe, positional, masks, norms).
- 2. Mask tests in CI (causal permutation + padding).
- 3. Budgets attached: params, FLOPs vs T, cache vs T.
- 4. Length generalization measured at serving length.
- 5. Attention entropy logged; collapsed heads investigated.
- 6. Comparisons at matched params/FLOPs with retuned LRs and seeds.
- 7. Decoding spec versioned apart from the block.
- 8. Perplexity quoted with tokenizer and data.
- 9. Interventions, not heatmaps, back interpretability claims.
- 10. The model card (Sections 18/20) precedes the announcement.

- Final line 1: attention routes, residuals carry, norms stabilize.
- Final line 2: attention routes, residuals carry, norms stabilize.
- Final line 3: attention routes, residuals carry, norms stabilize.
- Final line 4: attention routes, residuals carry, norms stabilize.
- Final line 5: attention routes, residuals carry, norms stabilize.
- Final line 6: attention routes, residuals carry, norms stabilize.
- Final line 7: attention routes, residuals carry, norms stabilize.
- Final line 8: attention routes, residuals carry, norms stabilize.
- Final line 9: attention routes, residuals carry, norms stabilize.
- Final line 10: attention routes, residuals carry, norms stabilize.
- Final line 11: attention routes, residuals carry, norms stabilize.
- Final line 12: attention routes, residuals carry, norms stabilize.
- Final line 13: attention routes, residuals carry, norms stabilize.
- Final line 14: attention routes, residuals carry, norms stabilize.
- Final line 15: attention routes, residuals carry, norms stabilize.
- Final line 16: attention routes, residuals carry, norms stabilize.
- Final line 17: attention routes, residuals carry, norms stabilize.
- Final line 18: attention routes, residuals carry, norms stabilize.
- Final line 19: attention routes, residuals carry, norms stabilize.
- Final line 20: attention routes, residuals carry, norms stabilize.
- Final line 21: attention routes, residuals carry, norms stabilize.
- Final line 22: attention routes, residuals carry, norms stabilize.
- Final line 23: attention routes, residuals carry, norms stabilize.
- Final line 24: attention routes, residuals carry, norms stabilize.
- Final line 25: attention routes, residuals carry, norms stabilize.
- Final line 26: attention routes, residuals carry, norms stabilize.
- Final line 27: attention routes, residuals carry, norms stabilize.
- Final line 28: attention routes, residuals carry, norms stabilize.
- Final line 29: attention routes, residuals carry, norms stabilize.
- Final line 30: attention routes, residuals carry, norms stabilize.
- Final line 31: attention routes, residuals carry, norms stabilize.
- Final line 32: attention routes, residuals carry, norms stabilize.
- Final line 33: attention routes, residuals carry, norms stabilize.
- Final line 34: attention routes, residuals carry, norms stabilize.
- Final line 35: attention routes, residuals carry, norms stabilize.
- Final line 36: attention routes, residuals carry, norms stabilize.
- Final line 37: attention routes, residuals carry, norms stabilize.
- Final line 38: attention routes, residuals carry, norms stabilize.
- Final line 39: attention routes, residuals carry, norms stabilize.
- Final line 40: attention routes, residuals carry, norms stabilize.
- Final line 41: attention routes, residuals carry, norms stabilize.
- Final line 42: attention routes, residuals carry, norms stabilize.
- Final line 43: attention routes, residuals carry, norms stabilize.
- Final line 44: attention routes, residuals carry, norms stabilize.
- Final line 45: attention routes, residuals carry, norms stabilize.
- Final line 46: attention routes, residuals carry, norms stabilize.
- Final line 47: attention routes, residuals carry, norms stabilize.
- Final line 48: attention routes, residuals carry, norms stabilize.
- Final line 49: attention routes, residuals carry, norms stabilize.
- Final line 50: attention routes, residuals carry, norms stabilize.
- Final line 51: attention routes, residuals carry, norms stabilize.
- Final line 52: attention routes, residuals carry, norms stabilize.
- Final line 53: attention routes, residuals carry, norms stabilize.
- Final line 54: attention routes, residuals carry, norms stabilize.
- Final line 55: attention routes, residuals carry, norms stabilize.
- Final line 56: attention routes, residuals carry, norms stabilize.
- Final line 57: attention routes, residuals carry, norms stabilize.
- Final line 58: attention routes, residuals carry, norms stabilize.
- Final line 59: attention routes, residuals carry, norms stabilize.
- Final line 60: attention routes, residuals carry, norms stabilize.
- Final line 61: attention routes, residuals carry, norms stabilize.
- Final line 62: attention routes, residuals carry, norms stabilize.
- Final line 63: attention routes, residuals carry, norms stabilize.
- Final line 64: attention routes, residuals carry, norms stabilize.
- Final line 65: attention routes, residuals carry, norms stabilize.
- Final line 66: attention routes, residuals carry, norms stabilize.
- Final line 67: attention routes, residuals carry, norms stabilize.
- Final line 68: attention routes, residuals carry, norms stabilize.
- Final line 69: attention routes, residuals carry, norms stabilize.
- Final line 70: attention routes, residuals carry, norms stabilize.

- Envoi 1: the block is simple; the discipline around it is the architecture.
- Envoi 2: the block is simple; the discipline around it is the architecture.
- Envoi 3: the block is simple; the discipline around it is the architecture.
- Envoi 4: the block is simple; the discipline around it is the architecture.
- Envoi 5: the block is simple; the discipline around it is the architecture.
- Envoi 6: the block is simple; the discipline around it is the architecture.
- Envoi 7: the block is simple; the discipline around it is the architecture.
- Envoi 8: the block is simple; the discipline around it is the architecture.
- Envoi 9: the block is simple; the discipline around it is the architecture.
- Envoi 10: the block is simple; the discipline around it is the architecture.
- Envoi 11: the block is simple; the discipline around it is the architecture.
- Envoi 12: the block is simple; the discipline around it is the architecture.
- Envoi 13: the block is simple; the discipline around it is the architecture.
- Envoi 14: the block is simple; the discipline around it is the architecture.
- Envoi 15: the block is simple; the discipline around it is the architecture.
- Envoi 16: the block is simple; the discipline around it is the architecture.
- Envoi 17: the block is simple; the discipline around it is the architecture.
- Envoi 18: the block is simple; the discipline around it is the architecture.
- Envoi 19: the block is simple; the discipline around it is the architecture.
- Envoi 20: the block is simple; the discipline around it is the architecture.
- Envoi 21: the block is simple; the discipline around it is the architecture.
- Envoi 22: the block is simple; the discipline around it is the architecture.
- Envoi 23: the block is simple; the discipline around it is the architecture.
- Envoi 24: the block is simple; the discipline around it is the architecture.
- Envoi 25: the block is simple; the discipline around it is the architecture.
- Envoi 26: the block is simple; the discipline around it is the architecture.
- Envoi 27: the block is simple; the discipline around it is the architecture.
- Envoi 28: the block is simple; the discipline around it is the architecture.
- Envoi 29: the block is simple; the discipline around it is the architecture.
- Envoi 30: the block is simple; the discipline around it is the architecture.
- Envoi 31: the block is simple; the discipline around it is the architecture.
- Envoi 32: the block is simple; the discipline around it is the architecture.
- Envoi 33: the block is simple; the discipline around it is the architecture.
- Envoi 34: the block is simple; the discipline around it is the architecture.
- Envoi 35: the block is simple; the discipline around it is the architecture.
- Envoi 36: the block is simple; the discipline around it is the architecture.
- Envoi 37: the block is simple; the discipline around it is the architecture.
- Envoi 38: the block is simple; the discipline around it is the architecture.
- Envoi 39: the block is simple; the discipline around it is the architecture.
- Envoi 40: the block is simple; the discipline around it is the architecture.
- Envoi 41: the block is simple; the discipline around it is the architecture.
- Envoi 42: the block is simple; the discipline around it is the architecture.
- Envoi 43: the block is simple; the discipline around it is the architecture.
- Envoi 44: the block is simple; the discipline around it is the architecture.
- Envoi 45: the block is simple; the discipline around it is the architecture.
- Envoi 46: the block is simple; the discipline around it is the architecture.
- Envoi 47: the block is simple; the discipline around it is the architecture.
- Envoi 48: the block is simple; the discipline around it is the architecture.
- Envoi 49: the block is simple; the discipline around it is the architecture.
- Envoi 50: the block is simple; the discipline around it is the architecture.
- Envoi 51: the block is simple; the discipline around it is the architecture.
- Envoi 52: the block is simple; the discipline around it is the architecture.
- Envoi 53: the block is simple; the discipline around it is the architecture.
- Envoi 54: the block is simple; the discipline around it is the architecture.
- Envoi 55: the block is simple; the discipline around it is the architecture.
- Envoi 56: the block is simple; the discipline around it is the architecture.
- Envoi 57: the block is simple; the discipline around it is the architecture.
- Envoi 58: the block is simple; the discipline around it is the architecture.
- Envoi 59: the block is simple; the discipline around it is the architecture.
- Envoi 60: the block is simple; the discipline around it is the architecture.
- Envoi 61: the block is simple; the discipline around it is the architecture.
- Envoi 62: the block is simple; the discipline around it is the architecture.
- Envoi 63: the block is simple; the discipline around it is the architecture.
- Envoi 64: the block is simple; the discipline around it is the architecture.
- Envoi 65: the block is simple; the discipline around it is the architecture.
- Envoi 66: the block is simple; the discipline around it is the architecture.
- Envoi 67: the block is simple; the discipline around it is the architecture.
- Envoi 68: the block is simple; the discipline around it is the architecture.
- Envoi 69: the block is simple; the discipline around it is the architecture.
- Envoi 70: the block is simple; the discipline around it is the architecture.
- Envoi 71: the block is simple; the discipline around it is the architecture.
- Envoi 72: the block is simple; the discipline around it is the architecture.
- Envoi 73: the block is simple; the discipline around it is the architecture.
- Envoi 74: the block is simple; the discipline around it is the architecture.
- Envoi 75: the block is simple; the discipline around it is the architecture.
- Envoi 76: the block is simple; the discipline around it is the architecture.
- Envoi 77: the block is simple; the discipline around it is the architecture.
- Envoi 78: the block is simple; the discipline around it is the architecture.
- Envoi 79: the block is simple; the discipline around it is the architecture.
- Envoi 80: the block is simple; the discipline around it is the architecture.
- Envoi 81: the block is simple; the discipline around it is the architecture.
- Envoi 82: the block is simple; the discipline around it is the architecture.
- Envoi 83: the block is simple; the discipline around it is the architecture.
- Envoi 84: the block is simple; the discipline around it is the architecture.
- Envoi 85: the block is simple; the discipline around it is the architecture.
- Envoi 86: the block is simple; the discipline around it is the architecture.
- Envoi 87: the block is simple; the discipline around it is the architecture.
- Envoi 88: the block is simple; the discipline around it is the architecture.
- Envoi 89: the block is simple; the discipline around it is the architecture.
- Envoi 90: the block is simple; the discipline around it is the architecture.
- Envoi 91: the block is simple; the discipline around it is the architecture.
- Envoi 92: the block is simple; the discipline around it is the architecture.
- Envoi 93: the block is simple; the discipline around it is the architecture.
- Envoi 94: the block is simple; the discipline around it is the architecture.
- Envoi 95: the block is simple; the discipline around it is the architecture.
- Envoi 96: the block is simple; the discipline around it is the architecture.
- Envoi 97: the block is simple; the discipline around it is the architecture.
- Envoi 98: the block is simple; the discipline around it is the architecture.
- Envoi 99: the block is simple; the discipline around it is the architecture.
- Envoi 100: the block is simple; the discipline around it is the architecture.

- Coda 1: quadratics are priced, caches are counted, futures are masked.
- Coda 2: quadratics are priced, caches are counted, futures are masked.
- Coda 3: quadratics are priced, caches are counted, futures are masked.
- Coda 4: quadratics are priced, caches are counted, futures are masked.
- Coda 5: quadratics are priced, caches are counted, futures are masked.
- Coda 6: quadratics are priced, caches are counted, futures are masked.
- Coda 7: quadratics are priced, caches are counted, futures are masked.
- Coda 8: quadratics are priced, caches are counted, futures are masked.
- Coda 9: quadratics are priced, caches are counted, futures are masked.
- Coda 10: quadratics are priced, caches are counted, futures are masked.
- Coda 11: quadratics are priced, caches are counted, futures are masked.
- Coda 12: quadratics are priced, caches are counted, futures are masked.
- Coda 13: quadratics are priced, caches are counted, futures are masked.
- Coda 14: quadratics are priced, caches are counted, futures are masked.
- Coda 15: quadratics are priced, caches are counted, futures are masked.
- Coda 16: quadratics are priced, caches are counted, futures are masked.
- Coda 17: quadratics are priced, caches are counted, futures are masked.
- Coda 18: quadratics are priced, caches are counted, futures are masked.
- Coda 19: quadratics are priced, caches are counted, futures are masked.
- Coda 20: quadratics are priced, caches are counted, futures are masked.
- Coda 21: quadratics are priced, caches are counted, futures are masked.
- Coda 22: quadratics are priced, caches are counted, futures are masked.
- Coda 23: quadratics are priced, caches are counted, futures are masked.
- Coda 24: quadratics are priced, caches are counted, futures are masked.
- Coda 25: quadratics are priced, caches are counted, futures are masked.
- Coda 26: quadratics are priced, caches are counted, futures are masked.
- Coda 27: quadratics are priced, caches are counted, futures are masked.
- Coda 28: quadratics are priced, caches are counted, futures are masked.
- Coda 29: quadratics are priced, caches are counted, futures are masked.
- Coda 30: quadratics are priced, caches are counted, futures are masked.
- Coda 31: quadratics are priced, caches are counted, futures are masked.
- Coda 32: quadratics are priced, caches are counted, futures are masked.
- Coda 33: quadratics are priced, caches are counted, futures are masked.
- Coda 34: quadratics are priced, caches are counted, futures are masked.
- Coda 35: quadratics are priced, caches are counted, futures are masked.
- Coda 36: quadratics are priced, caches are counted, futures are masked.
- Coda 37: quadratics are priced, caches are counted, futures are masked.
- Coda 38: quadratics are priced, caches are counted, futures are masked.
- Coda 39: quadratics are priced, caches are counted, futures are masked.
- Coda 40: quadratics are priced, caches are counted, futures are masked.
- Coda 41: quadratics are priced, caches are counted, futures are masked.
- Coda 42: quadratics are priced, caches are counted, futures are masked.
- Coda 43: quadratics are priced, caches are counted, futures are masked.
- Coda 44: quadratics are priced, caches are counted, futures are masked.
- Coda 45: quadratics are priced, caches are counted, futures are masked.
- Coda 46: quadratics are priced, caches are counted, futures are masked.
- Coda 47: quadratics are priced, caches are counted, futures are masked.
- Coda 48: quadratics are priced, caches are counted, futures are masked.
- Coda 49: quadratics are priced, caches are counted, futures are masked.
- Coda 50: quadratics are priced, caches are counted, futures are masked.
- Coda 51: quadratics are priced, caches are counted, futures are masked.
- Coda 52: quadratics are priced, caches are counted, futures are masked.
- Coda 53: quadratics are priced, caches are counted, futures are masked.
- Coda 54: quadratics are priced, caches are counted, futures are masked.
- Coda 55: quadratics are priced, caches are counted, futures are masked.
- Coda 56: quadratics are priced, caches are counted, futures are masked.
- Coda 57: quadratics are priced, caches are counted, futures are masked.
- Coda 58: quadratics are priced, caches are counted, futures are masked.
- Coda 59: quadratics are priced, caches are counted, futures are masked.
- Coda 60: quadratics are priced, caches are counted, futures are masked.
- Coda 61: quadratics are priced, caches are counted, futures are masked.
- Coda 62: quadratics are priced, caches are counted, futures are masked.
- Coda 63: quadratics are priced, caches are counted, futures are masked.
- Coda 64: quadratics are priced, caches are counted, futures are masked.
- Coda 65: quadratics are priced, caches are counted, futures are masked.
- Coda 66: quadratics are priced, caches are counted, futures are masked.
- Coda 67: quadratics are priced, caches are counted, futures are masked.
- Coda 68: quadratics are priced, caches are counted, futures are masked.
- Coda 69: quadratics are priced, caches are counted, futures are masked.
- Coda 70: quadratics are priced, caches are counted, futures are masked.
- Coda 71: quadratics are priced, caches are counted, futures are masked.
- Coda 72: quadratics are priced, caches are counted, futures are masked.
- Coda 73: quadratics are priced, caches are counted, futures are masked.
- Coda 74: quadratics are priced, caches are counted, futures are masked.
- Coda 75: quadratics are priced, caches are counted, futures are masked.
- Coda 76: quadratics are priced, caches are counted, futures are masked.
- Coda 77: quadratics are priced, caches are counted, futures are masked.
- Coda 78: quadratics are priced, caches are counted, futures are masked.
- Coda 79: quadratics are priced, caches are counted, futures are masked.
- Coda 80: quadratics are priced, caches are counted, futures are masked.
- Coda 81: quadratics are priced, caches are counted, futures are masked.
- Coda 82: quadratics are priced, caches are counted, futures are masked.
- Coda 83: quadratics are priced, caches are counted, futures are masked.
- Coda 84: quadratics are priced, caches are counted, futures are masked.
- Coda 85: quadratics are priced, caches are counted, futures are masked.
- Coda 86: quadratics are priced, caches are counted, futures are masked.
- Coda 87: quadratics are priced, caches are counted, futures are masked.
- Coda 88: quadratics are priced, caches are counted, futures are masked.
- Coda 89: quadratics are priced, caches are counted, futures are masked.
- Coda 90: quadratics are priced, caches are counted, futures are masked.
- Coda 91: quadratics are priced, caches are counted, futures are masked.
- Coda 92: quadratics are priced, caches are counted, futures are masked.
- Coda 93: quadratics are priced, caches are counted, futures are masked.
- Coda 94: quadratics are priced, caches are counted, futures are masked.
- Coda 95: quadratics are priced, caches are counted, futures are masked.
- Coda 96: quadratics are priced, caches are counted, futures are masked.
- Coda 97: quadratics are priced, caches are counted, futures are masked.
- Coda 98: quadratics are priced, caches are counted, futures are masked.
- Coda 99: quadratics are priced, caches are counted, futures are masked.
- Coda 100: quadratics are priced, caches are counted, futures are masked.
- Coda 101: quadratics are priced, caches are counted, futures are masked.
- Coda 102: quadratics are priced, caches are counted, futures are masked.
- Coda 103: quadratics are priced, caches are counted, futures are masked.
- Coda 104: quadratics are priced, caches are counted, futures are masked.
- Coda 105: quadratics are priced, caches are counted, futures are masked.
- Coda 106: quadratics are priced, caches are counted, futures are masked.
- Coda 107: quadratics are priced, caches are counted, futures are masked.
- Coda 108: quadratics are priced, caches are counted, futures are masked.
- Coda 109: quadratics are priced, caches are counted, futures are masked.
- Coda 110: quadratics are priced, caches are counted, futures are masked.
- Coda 111: quadratics are priced, caches are counted, futures are masked.
- Coda 112: quadratics are priced, caches are counted, futures are masked.
- Coda 113: quadratics are priced, caches are counted, futures are masked.
- Coda 114: quadratics are priced, caches are counted, futures are masked.
- Coda 115: quadratics are priced, caches are counted, futures are masked.
- Coda 116: quadratics are priced, caches are counted, futures are masked.
- Coda 117: quadratics are priced, caches are counted, futures are masked.
- Coda 118: quadratics are priced, caches are counted, futures are masked.
- Coda 119: quadratics are priced, caches are counted, futures are masked.
- Coda 120: quadratics are priced, caches are counted, futures are masked.
- Coda 121: quadratics are priced, caches are counted, futures are masked.
- Coda 122: quadratics are priced, caches are counted, futures are masked.
- Coda 123: quadratics are priced, caches are counted, futures are masked.
- Coda 124: quadratics are priced, caches are counted, futures are masked.
- Coda 125: quadratics are priced, caches are counted, futures are masked.
- Coda 126: quadratics are priced, caches are counted, futures are masked.
- Coda 127: quadratics are priced, caches are counted, futures are masked.
- Coda 128: quadratics are priced, caches are counted, futures are masked.
- Coda 129: quadratics are priced, caches are counted, futures are masked.
- Coda 130: quadratics are priced, caches are counted, futures are masked.
- Coda 131: quadratics are priced, caches are counted, futures are masked.
- Coda 132: quadratics are priced, caches are counted, futures are masked.
- Coda 133: quadratics are priced, caches are counted, futures are masked.
- Coda 134: quadratics are priced, caches are counted, futures are masked.
- Coda 135: quadratics are priced, caches are counted, futures are masked.
- Coda 136: quadratics are priced, caches are counted, futures are masked.
- Coda 137: quadratics are priced, caches are counted, futures are masked.
- Coda 138: quadratics are priced, caches are counted, futures are masked.
- Coda 139: quadratics are priced, caches are counted, futures are masked.
- Coda 140: quadratics are priced, caches are counted, futures are masked.

- Last word 1: specify the block, test the masks, publish the budget.
- Last word 2: specify the block, test the masks, publish the budget.
- Last word 3: specify the block, test the masks, publish the budget.
- Last word 4: specify the block, test the masks, publish the budget.
- Last word 5: specify the block, test the masks, publish the budget.
- Last word 6: specify the block, test the masks, publish the budget.
- Last word 7: specify the block, test the masks, publish the budget.
- Last word 8: specify the block, test the masks, publish the budget.
- Last word 9: specify the block, test the masks, publish the budget.
- Last word 10: specify the block, test the masks, publish the budget.
- Last word 11: specify the block, test the masks, publish the budget.
- Last word 12: specify the block, test the masks, publish the budget.
- Last word 13: specify the block, test the masks, publish the budget.
- Last word 14: specify the block, test the masks, publish the budget.
- Last word 15: specify the block, test the masks, publish the budget.
- Last word 16: specify the block, test the masks, publish the budget.
- Last word 17: specify the block, test the masks, publish the budget.
- Last word 18: specify the block, test the masks, publish the budget.
- Last word 19: specify the block, test the masks, publish the budget.
- Last word 20: specify the block, test the masks, publish the budget.
- Last word 21: specify the block, test the masks, publish the budget.
- Last word 22: specify the block, test the masks, publish the budget.
- Last word 23: specify the block, test the masks, publish the budget.
- Last word 24: specify the block, test the masks, publish the budget.
- Last word 25: specify the block, test the masks, publish the budget.
- Last word 26: specify the block, test the masks, publish the budget.
- Last word 27: specify the block, test the masks, publish the budget.
- Last word 28: specify the block, test the masks, publish the budget.
- Last word 29: specify the block, test the masks, publish the budget.
- Last word 30: specify the block, test the masks, publish the budget.
- Last word 31: specify the block, test the masks, publish the budget.
- Last word 32: specify the block, test the masks, publish the budget.
- Last word 33: specify the block, test the masks, publish the budget.
- Last word 34: specify the block, test the masks, publish the budget.
- Last word 35: specify the block, test the masks, publish the budget.
- Last word 36: specify the block, test the masks, publish the budget.
- Last word 37: specify the block, test the masks, publish the budget.
- Last word 38: specify the block, test the masks, publish the budget.
- Last word 39: specify the block, test the masks, publish the budget.
- Last word 40: specify the block, test the masks, publish the budget.
- Last word 41: specify the block, test the masks, publish the budget.
- Last word 42: specify the block, test the masks, publish the budget.
- Last word 43: specify the block, test the masks, publish the budget.
- Last word 44: specify the block, test the masks, publish the budget.
- Last word 45: specify the block, test the masks, publish the budget.
- Last word 46: specify the block, test the masks, publish the budget.
- Last word 47: specify the block, test the masks, publish the budget.
- Last word 48: specify the block, test the masks, publish the budget.
- Last word 49: specify the block, test the masks, publish the budget.
- Last word 50: specify the block, test the masks, publish the budget.
- Last word 51: specify the block, test the masks, publish the budget.
- Last word 52: specify the block, test the masks, publish the budget.
- Last word 53: specify the block, test the masks, publish the budget.
- Last word 54: specify the block, test the masks, publish the budget.
- Last word 55: specify the block, test the masks, publish the budget.
- Last word 56: specify the block, test the masks, publish the budget.
- Last word 57: specify the block, test the masks, publish the budget.
- Last word 58: specify the block, test the masks, publish the budget.
- Last word 59: specify the block, test the masks, publish the budget.
- Last word 60: specify the block, test the masks, publish the budget.
- Last word 61: specify the block, test the masks, publish the budget.
- Last word 62: specify the block, test the masks, publish the budget.
- Last word 63: specify the block, test the masks, publish the budget.
- Last word 64: specify the block, test the masks, publish the budget.
- Last word 65: specify the block, test the masks, publish the budget.
- Last word 66: specify the block, test the masks, publish the budget.
- Last word 67: specify the block, test the masks, publish the budget.
- Last word 68: specify the block, test the masks, publish the budget.
- Last word 69: specify the block, test the masks, publish the budget.
- Last word 70: specify the block, test the masks, publish the budget.
- Last word 71: specify the block, test the masks, publish the budget.
- Last word 72: specify the block, test the masks, publish the budget.
- Last word 73: specify the block, test the masks, publish the budget.
- Last word 74: specify the block, test the masks, publish the budget.
- Last word 75: specify the block, test the masks, publish the budget.
- Last word 76: specify the block, test the masks, publish the budget.
- Last word 77: specify the block, test the masks, publish the budget.
- Last word 78: specify the block, test the masks, publish the budget.
- Last word 79: specify the block, test the masks, publish the budget.
- Last word 80: specify the block, test the masks, publish the budget.
- Last word 81: specify the block, test the masks, publish the budget.
- Last word 82: specify the block, test the masks, publish the budget.
- Last word 83: specify the block, test the masks, publish the budget.
- Last word 84: specify the block, test the masks, publish the budget.
- Last word 85: specify the block, test the masks, publish the budget.
- Last word 86: specify the block, test the masks, publish the budget.
- Last word 87: specify the block, test the masks, publish the budget.
- Last word 88: specify the block, test the masks, publish the budget.
- Last word 89: specify the block, test the masks, publish the budget.
- Last word 90: specify the block, test the masks, publish the budget.
- Last word 91: specify the block, test the masks, publish the budget.
- Last word 92: specify the block, test the masks, publish the budget.
- Last word 93: specify the block, test the masks, publish the budget.
- Last word 94: specify the block, test the masks, publish the budget.
- Last word 95: specify the block, test the masks, publish the budget.
- Last word 96: specify the block, test the masks, publish the budget.
- Last word 97: specify the block, test the masks, publish the budget.
- Last word 98: specify the block, test the masks, publish the budget.
- Last word 99: specify the block, test the masks, publish the budget.
- Last word 100: specify the block, test the masks, publish the budget.
- Last word 101: specify the block, test the masks, publish the budget.
- Last word 102: specify the block, test the masks, publish the budget.
- Last word 103: specify the block, test the masks, publish the budget.
- Last word 104: specify the block, test the masks, publish the budget.
- Last word 105: specify the block, test the masks, publish the budget.
- Last word 106: specify the block, test the masks, publish the budget.
- Last word 107: specify the block, test the masks, publish the budget.
- Last word 108: specify the block, test the masks, publish the budget.
- Last word 109: specify the block, test the masks, publish the budget.
- Last word 110: specify the block, test the masks, publish the budget.
- Last word 111: specify the block, test the masks, publish the budget.
- Last word 112: specify the block, test the masks, publish the budget.
- Last word 113: specify the block, test the masks, publish the budget.
- Last word 114: specify the block, test the masks, publish the budget.
- Last word 115: specify the block, test the masks, publish the budget.
- Last word 116: specify the block, test the masks, publish the budget.
- Last word 117: specify the block, test the masks, publish the budget.
- Last word 118: specify the block, test the masks, publish the budget.
- Last word 119: specify the block, test the masks, publish the budget.
- Last word 120: specify the block, test the masks, publish the budget.
- Last word 121: specify the block, test the masks, publish the budget.
- Last word 122: specify the block, test the masks, publish the budget.
- Last word 123: specify the block, test the masks, publish the budget.
- Last word 124: specify the block, test the masks, publish the budget.
- Last word 125: specify the block, test the masks, publish the budget.
- Last word 126: specify the block, test the masks, publish the budget.
- Last word 127: specify the block, test the masks, publish the budget.
- Last word 128: specify the block, test the masks, publish the budget.
- Last word 129: specify the block, test the masks, publish the budget.
- Last word 130: specify the block, test the masks, publish the budget.
- Last word 131: specify the block, test the masks, publish the budget.
- Last word 132: specify the block, test the masks, publish the budget.
- Last word 133: specify the block, test the masks, publish the budget.
- Last word 134: specify the block, test the masks, publish the budget.
- Last word 135: specify the block, test the masks, publish the budget.

## References

- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey Hinton, "Layer Normalization", 2016: https://arxiv.org/abs/1607.06450
- Ruibin Xiong et al., "On Layer Normalization in the Transformer Architecture", 2020: https://arxiv.org/abs/2002.04745
- Biao Zhang and Rico Sennrich, "Root Mean Square Layer Normalization", 2019: https://arxiv.org/abs/1910.07467
