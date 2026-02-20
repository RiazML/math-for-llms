# Attention Mechanisms in Deep Learning

[← Previous: Activation Functions](../02-Activation-Functions) | [Next: Normalization Techniques →](../04-Normalization-Techniques)

---

## Overview

Attention mechanisms allow neural networks to dynamically focus on relevant parts of the input when producing each part of the output. Originally developed for sequence-to-sequence models, attention has become the foundation of transformer architectures.

### Files in This Section

| File | Description |
|------|-------------|
| [README.md](README.md) | Comprehensive theory and mathematical foundations |
| [theory.ipynb](theory.ipynb) | Worked examples with Python implementations |
| [exercises.ipynb](exercises.ipynb) | Practice problems with solutions |

## Why This Matters for Machine Learning

Attention mechanisms represent arguably the most important architectural innovation in modern deep learning. The transformer architecture—built entirely on attention—underpins virtually every state-of-the-art model across NLP (GPT, BERT, LLaMA), vision (ViT, DINO), and multimodal AI (CLIP, DALL-E). Understanding the mathematics of attention is no longer optional for anyone working in ML; it is foundational knowledge.

The mathematical elegance of scaled dot-product attention lies in its simplicity: it computes a weighted sum of values based on query-key compatibility, with the $1/\sqrt{d_k}$ scaling factor ensuring stable softmax gradients. Multi-head attention extends this by projecting queries, keys, and values into multiple subspaces, allowing the model to simultaneously attend to different types of relationships—syntactic structure in one head, semantic similarity in another.

Equally important is the mathematics of positional encoding. Self-attention is inherently permutation-equivariant: it treats its input as a set, not a sequence. Sinusoidal encodings, learned embeddings, RoPE, and ALiBi each inject position information differently, with profound implications for length generalization and computational efficiency. Understanding these tradeoffs—along with masking strategies, efficient attention variants like Flash Attention, and modern extensions like grouped-query attention—is essential for both building and optimizing transformer-based systems.

## Chapter Roadmap

- Core intuition: from fixed-size bottlenecks to dynamic attention
- General attention formulation: queries, keys, values, and scoring functions
- Scaled dot-product attention and the variance-stabilizing $1/\sqrt{d_k}$ factor
- Multi-head attention: parallel subspace projections and their benefits
- Self-attention and cross-attention: intra-sequence and inter-sequence interactions
- Masking strategies: padding masks and causal (autoregressive) masks
- Positional encoding: sinusoidal, learned, RoPE, and ALiBi approaches
- Efficient attention: linear attention, sparse attention, Flash Attention, and sliding window
- Attention pattern analysis: entropy, effective receptive field, and interpretability
- Modern variants: grouped-query attention (GQA), multi-query attention (MQA), and state-space alternatives

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

## Key Takeaways

- **Attention replaces recurrence with parallelism**: by computing all pairwise interactions simultaneously, attention removes the sequential bottleneck of RNNs and enables GPU-efficient training on long sequences
- **Scaling by $1/\sqrt{d_k}$ is mathematically necessary**: without it, dot products grow with dimension, pushing softmax into saturated regions where gradients vanish
- **Multi-head attention learns diverse relationships**: each head operates in its own low-dimensional subspace, collectively capturing both local syntactic and long-range semantic patterns
- **Positional encoding is essential for sequence tasks**: since self-attention is permutation-equivariant, position information must be explicitly injected via sinusoidal functions, learned embeddings, or rotary encodings (RoPE)
- **Masking controls information flow**: causal masks enforce autoregressive constraints in language models, while padding masks prevent attention to invalid positions
- **$O(n^2)$ complexity is the fundamental bottleneck**: Flash Attention, linear attention, and sparse patterns address this through algorithmic and hardware-aware optimizations
- **Attention weights provide interpretability**: unlike opaque hidden states, attention distributions can be visualized and analyzed, though they are not always faithful explanations of model behavior

## Exercises

1. **Scaled Dot-Product Attention by Hand**: For query $q = [1, 0, 1]$, keys $K = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1 \\ 1 & 1 & 0 \end{bmatrix}$, and values $V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}$, compute the scaled dot-product attention output step by step. What are the attention weights $\alpha_i$? How would the weights change if you doubled $d_k$ without scaling?

2. **Multi-Head vs. Single-Head**: Implement both single-head attention (with $d_{model} = 64$) and multi-head attention (with $h = 8$, $d_k = d_v = 8$). Apply both to a random sequence of length 20 and compare the parameter counts. Show that multi-head attention has the same total parameter count as a single attention head with the same $d_{model}$.

3. **Positional Encoding Analysis**: Implement sinusoidal positional encodings for $d = 128$ and sequence length $n = 100$. Plot the encodings as a heatmap. Compute the dot product $PE_{pos} \cdot PE_{pos+k}$ as a function of relative offset $k$ and show that it depends only on $k$, not on the absolute position. Compare with random learned embeddings—does this property hold?

4. **Causal Masking and Autoregression**: Implement causal self-attention for a sequence of length 10. Verify that the output at position $i$ depends only on positions $\leq i$ by computing the Jacobian $\partial y_i / \partial x_j$ and confirming it is zero for $j > i$. What happens to the attention entropy $H(\alpha)$ at position 1 vs. position 10?

5. **Flash Attention vs. Standard**: Implement standard attention and tiled (block-wise) attention for sequence lengths $n \in \{128, 512, 2048, 8192\}$. Measure peak memory usage and wall-clock time. At what sequence length does the standard implementation run out of memory on your GPU (or system RAM)? How does the tiled version compare?
