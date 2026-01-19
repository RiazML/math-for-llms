# Attention Mechanisms in Deep Learning

## Overview

Attention mechanisms allow neural networks to dynamically focus on relevant parts of the input when producing each part of the output. Originally developed for sequence-to-sequence models, attention has become the foundation of transformer architectures.

## Core Intuition

Traditional sequence models process input sequentially, compressing all information into a fixed-size hidden state. Attention provides direct connections to all input positions, weighted by relevance.

**Key idea:** Instead of encoding input as a single vector, allow the model to "look back" at all input positions and compute a weighted combination.

## Mathematical Formulation

### General Attention

Given:

- **Query** $q \in \mathbb{R}^{d_q}$: what we're looking for
- **Keys** $K \in \mathbb{R}^{n \times d_k}$: what the inputs contain
- **Values** $V \in \mathbb{R}^{n \times d_v}$: information to retrieve

Attention computes:

$$\text{Attention}(q, K, V) = \sum_{i=1}^{n} \alpha_i v_i$$

where attention weights $\alpha_i$ satisfy $\sum_i \alpha_i = 1$, $\alpha_i \geq 0$.

### Scoring Functions

The score function determines compatibility between query and keys:

**Dot-Product (Scaled):**
$$\text{score}(q, k_i) = \frac{q^T k_i}{\sqrt{d_k}}$$

**Additive (Bahdanau):**
$$\text{score}(q, k_i) = v^T \tanh(W_q q + W_k k_i)$$

**General:**
$$\text{score}(q, k_i) = q^T W k_i$$

**Concatenation:**
$$\text{score}(q, k_i) = v^T \tanh(W[q; k_i])$$

### Attention Weights

Weights are computed via softmax:

$$\alpha_i = \frac{\exp(\text{score}(q, k_i))}{\sum_{j=1}^{n} \exp(\text{score}(q, k_j))}$$

## Scaled Dot-Product Attention

The most common form, used in Transformers:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Why scale by $\sqrt{d_k}$?**

For large $d_k$, dot products can have large magnitude, pushing softmax into regions with tiny gradients. Scaling ensures variance of scores ≈ 1.

If $q, k$ have i.i.d. components with variance 1:
$$\text{Var}(q^T k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

After scaling: $\text{Var}\left(\frac{q^T k}{\sqrt{d_k}}\right) = 1$

## Multi-Head Attention

Instead of single attention, compute multiple attention "heads" with different learned projections:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

where each head is:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameters:**

- $W_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $W_i^V \in \mathbb{R}^{d_{model} \times d_v}$
- $W^O \in \mathbb{R}^{hd_v \times d_{model}}$

Typically: $d_k = d_v = d_{model} / h$

**Benefits:**

- Different heads learn different relationships
- Attends to information from different representation subspaces
- Comparable computation to single-head attention

## Self-Attention

When queries, keys, and values come from the same sequence:

$$Q = XW^Q, \quad K = XW^K, \quad V = XW^V$$

Self-attention captures relationships within a single sequence.

**Complexity:** $O(n^2 d)$ for sequence length $n$ and dimension $d$.

## Cross-Attention

When queries come from one sequence and keys/values from another:

$$Q = X_1 W^Q, \quad K = X_2 W^K, \quad V = X_2 W^V$$

Used in encoder-decoder models where decoder attends to encoder outputs.

## Masking

### Padding Mask

Prevents attention to padding tokens:

$$\text{Attention}(Q, K, V, M) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M_{ij} = -\infty$ if position $j$ is padding, else 0.

### Causal Mask (Look-ahead)

Prevents attending to future positions (for autoregressive models):

$$
M_{ij} = \begin{cases}
0 & i \geq j \\
-\infty & i < j
\end{cases}
$$

This creates lower-triangular attention patterns.

## Positional Encoding

Self-attention is permutation-equivariant. To inject position information:

### Sinusoidal (Original Transformer)

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

**Properties:**

- Deterministic, no learned parameters
- Can extrapolate to longer sequences
- $PE_{pos+k}$ is a linear function of $PE_{pos}$

### Learned Positional Embeddings

$$X'_{pos} = X_{pos} + E_{pos}$$

where $E \in \mathbb{R}^{n_{max} \times d}$ is learned.

### Rotary Position Embedding (RoPE)

Applies rotation to query/key vectors based on position:

$$f_q(x_m, m) = e^{im\theta} \cdot x_m$$

Encodes relative position in the dot product of rotated vectors.

### Alibi (Attention with Linear Biases)

Adds position-dependent bias to attention scores:

$$\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} - m \cdot |i-j|\right)$$

where $m$ is a head-specific slope.

## Efficient Attention Variants

### Linear Attention

Replace softmax with feature maps to get linear complexity:

$$\text{Attention}(Q, K, V) = \phi(Q) \cdot (\phi(K)^T V)$$

where $\phi$ is a feature map. Reduces complexity from $O(n^2)$ to $O(n)$.

### Sparse Attention

Only compute attention for subset of positions:

- **Local attention:** attend to nearby positions only
- **Strided attention:** attend to every $k$-th position
- **Block-sparse:** predefined sparsity patterns

### Flash Attention

Memory-efficient attention using tiling and recomputation:

- Process attention in blocks that fit in fast memory (SRAM)
- Avoid materializing full $n \times n$ attention matrix
- Reduces memory from $O(n^2)$ to $O(n)$

### Sliding Window Attention

Each position attends to a fixed window:

$$\alpha_{ij} = 0 \text{ if } |i - j| > w$$

Reduces complexity to $O(nw)$.

## Attention Patterns Analysis

### Attention Entropy

Measures how diffuse/focused attention is:

$$H(\alpha) = -\sum_i \alpha_i \log \alpha_i$$

- Low entropy: focused attention (few positions)
- High entropy: diffuse attention (many positions)

### Attention Distance

Average distance of attended positions:

$$D = \sum_i \alpha_i |i - pos|$$

### Effective Receptive Field

Which positions significantly contribute to output:

$$\text{ERF} = \{i : \alpha_i > \tau\}$$

## Gradient Flow in Attention

For input $X$ and output $Y = \text{Attention}(X, X, X)$:

$$\frac{\partial Y}{\partial X} = \frac{\partial Y}{\partial A}\frac{\partial A}{\partial X} + A$$

where $A$ is the attention matrix.

**Key property:** Direct gradient paths through attention weights allow information to flow across long distances without vanishing.

## Attention and Memory

Attention can be viewed as content-addressable memory:

- **Keys:** memory addresses
- **Values:** memory contents
- **Query:** address to look up
- **Attention weights:** soft retrieval based on address similarity

## Applications

### Machine Translation

- Encoder-decoder attention aligns source and target
- Self-attention captures dependencies within each language

### Language Modeling

- Causal self-attention for autoregressive generation
- Captures long-range dependencies

### Vision Transformers

- Patch embeddings as tokens
- Self-attention over spatial positions

### Speech Recognition

- Time-frequency attention
- Streaming/online attention for real-time

### Graph Neural Networks

- Graph attention: attention weights based on edge structure
- GAT: learned attention between neighbors

## Implementation Considerations

### Numerical Stability

```
# Stable softmax
scores = scores - scores.max(dim=-1, keepdim=True)
weights = softmax(scores)
```

### Memory Efficiency

- Attention recomputation in backward pass
- Chunked attention for long sequences
- Gradient checkpointing

### Parallelization

- Self-attention is highly parallelizable
- All positions can be computed simultaneously
- Contrast with RNNs (sequential)

## Recent Developments

### Mixture of Experts (MoE)

Sparse activation combined with attention for scaling.

### Grouped Query Attention (GQA)

Share key-value heads across multiple query heads:

- Reduces memory for KV cache
- Used in LLaMA 2, Mistral

### Multi-Query Attention (MQA)

Single key-value head for all query heads:

- Maximum KV cache efficiency
- Slight quality trade-off

### State Space Models

Alternatives to attention (Mamba, S4):

- Linear complexity in sequence length
- Competitive with transformers on some tasks

## Mathematical Properties

### Permutation Equivariance

Self-attention without positional encoding is permutation equivariant:

$$\text{Attention}(\Pi X) = \Pi \cdot \text{Attention}(X)$$

for any permutation matrix $\Pi$.

### Lipschitz Continuity

Attention with bounded inputs is Lipschitz continuous, important for:

- Adversarial robustness
- Optimization stability

### Expressive Power

Self-attention layers can:

- Approximate any continuous sequence-to-sequence function
- Implement certain algorithms (sorting, copying)
- Simulate Turing machines (with sufficient depth)

## Summary

Key concepts:

1. **Queries, Keys, Values:** the fundamental abstraction
2. **Scaling:** prevents softmax saturation
3. **Multi-head:** parallel attention in subspaces
4. **Masking:** control information flow (padding, causal)
5. **Position:** inject sequence order information
6. **Efficiency:** linear attention, sparsity, Flash Attention

Attention enables:

- Parallel computation over sequences
- Direct long-range dependencies
- Interpretable alignment/importance weights
- Flexible architectural compositions
