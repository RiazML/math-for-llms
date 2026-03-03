[← Tokenization Math](../01-Tokenization-Math/notes.md) | [Home](../../README.md) | [Attention Math →](../03-Attention-Math/notes.md)

---

# Embedding Space Math

> _"An embedding table is where discrete symbols become continuous geometry. Every relationship the model will ever learn — synonymy, analogy, syntax — is encoded as structure in ℝᵈ."_

## Overview

After tokenization maps text to integer IDs, the embedding layer maps those IDs into a continuous vector space ℝᵈ where neural networks can operate. This section covers the mathematics of that mapping: the embedding matrix, similarity metrics, geometric properties (analogy, isotropy, curse of dimensionality), positional encodings (sinusoidal, learned, RoPE, ALiBi), how embeddings are trained, the distinction between static and contextual representations, and how embedding geometry feeds into attention. Every concept is grounded in how it affects real LLM architecture and performance.

## Prerequisites

- Linear algebra: vectors, matrices, dot product, norms
- Calculus: gradients, chain rule
- Basic probability (for cross-entropy loss)
- Completed: [01-Tokenization-Math](../01-Tokenization-Math/notes.md)

## Companion Notebooks

| Notebook                           | Description                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| [theory.ipynb](theory.ipynb)       | Embedding lookup, similarity metrics, analogy arithmetic, positional encodings, PCA/t-SNE        |
| [exercises.ipynb](exercises.ipynb) | Hands-on: cosine similarity, analogy verification, PE visualisation, parameter counting, probing |

## Learning Objectives

After completing this section, you will:

- Explain the embedding matrix E ∈ ℝᴺˣᵈ and compute parameter counts for real models
- Implement dot product, cosine similarity, and Euclidean distance and know when to use each
- Demonstrate vector analogy arithmetic (king − man + woman ≈ queen) and its geometric interpretation
- Distinguish isotropic vs anisotropic embedding distributions and their impact on similarity
- Derive sinusoidal positional encodings and explain why they allow length generalisation
- Compare RoPE, ALiBi, and learned positional embeddings mathematically
- Trace gradient flow from cross-entropy loss through the LM head into the embedding matrix
- Explain contextual vs static embeddings and layer-wise representation structure
- Perform dimensionality reduction (PCA, t-SNE) on embedding spaces and interpret clusters

## Table of Contents

- [Embedding Space Math](#embedding-space-math)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [What Is an Embedding?](#what-is-an-embedding)
    - [Why Continuous Space?](#why-continuous-space)
    - [Pipeline Position](#pipeline-position)
    - [Historical Context](#historical-context)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 Embedding Matrix](#21-embedding-matrix)
    - [2.2 Vector Space ℝᵈ](#22-vector-space-ℝᵈ)
    - [2.3 Embedding Function](#23-embedding-function)
  - [3. Mathematical Structure](#3-mathematical-structure)
    - [3.1 Dot Product and Similarity](#31-dot-product-and-similarity)
    - [3.2 Cosine Similarity](#32-cosine-similarity)
    - [3.3 Euclidean Distance](#33-euclidean-distance)
    - [3.4 Norms](#34-norms)
  - [4. Geometric Properties](#4-geometric-properties)
    - [4.1 Linear Relationships (Analogy Structure)](#41-linear-relationships-analogy-structure)
    - [4.2 Subspaces and Directions](#42-subspaces-and-directions)
    - [4.3 Isotropy vs Anisotropy](#43-isotropy-vs-anisotropy)
    - [4.4 The Curse of Dimensionality](#44-the-curse-of-dimensionality)
  - [5. Positional Encodings](#5-positional-encodings)
    - [5.1 Why Position Matters](#51-why-position-matters)
    - [5.2 Sinusoidal Positional Encoding](#52-sinusoidal-positional-encoding)
    - [5.3 Learned Positional Embeddings](#53-learned-positional-embeddings)
    - [5.4 Rotary Positional Encoding (RoPE)](#54-rotary-positional-encoding-rope)
    - [5.5 ALiBi (Attention with Linear Biases)](#55-alibi-attention-with-linear-biases)
  - [6. Embedding Training](#6-embedding-training)
    - [6.1 Random Initialisation](#61-random-initialisation)
    - [6.2 Cross-Entropy Loss and Gradient Flow](#62-cross-entropy-loss-and-gradient-flow)
    - [6.3 Tied Embeddings](#63-tied-embeddings)
    - [6.4 Pretrained vs Fine-tuned Embeddings](#64-pretrained-vs-fine-tuned-embeddings)
  - [7. Contextual vs Static Embeddings](#7-contextual-vs-static-embeddings)
    - [7.1 Static Embeddings (word2vec, GloVe)](#71-static-embeddings-word2vec-glove)
    - [7.2 Contextual Embeddings (BERT, GPT)](#72-contextual-embeddings-bert-gpt)
    - [7.3 Layer-wise Representation](#73-layer-wise-representation)
  - [8. Dimensionality and Model Scale](#8-dimensionality-and-model-scale)
    - [8.1 Embedding Dimension vs Model Size](#81-embedding-dimension-vs-model-size)
    - [8.2 Scaling Laws for Embeddings](#82-scaling-laws-for-embeddings)
    - [8.3 Dimensionality Reduction](#83-dimensionality-reduction)
  - [9. Embedding Space in Attention](#9-embedding-space-in-attention)
    - [9.1 Query, Key, Value Projections](#91-query-key-value-projections)
    - [9.2 Attention as Soft Lookup](#92-attention-as-soft-lookup)
    - [9.3 Residual Stream](#93-residual-stream)
  - [10. Common Mistakes](#10-common-mistakes)
  - [11. Exercises](#11-exercises)
    - [Exercise 1: Cosine Similarity Calculator](#exercise-1-cosine-similarity-calculator)
    - [Exercise 2: Analogy Arithmetic](#exercise-2-analogy-arithmetic)
    - [Exercise 3: Sinusoidal PE Visualisation](#exercise-3-sinusoidal-pe-visualisation)
    - [Exercise 4: Embedding Isotropy Analysis](#exercise-4-embedding-isotropy-analysis)
    - [Exercise 5: RoPE Implementation](#exercise-5-rope-implementation)
    - [Exercise 6: Dimensionality Reduction](#exercise-6-dimensionality-reduction)
    - [Exercise 7: Parameter Counting](#exercise-7-parameter-counting)
    - [Exercise 8: Layer Probing](#exercise-8-layer-probing)
    - [Exercise 9: Weight Tying Analysis](#exercise-9-weight-tying-analysis)
    - [Exercise 10: Curse of Dimensionality Demo](#exercise-10-curse-of-dimensionality-demo)
  - [12. Why This Matters for AI](#12-why-this-matters-for-ai)
  - [13. Further Reading](#13-further-reading)
    - [Papers](#papers)
    - [Implementations](#implementations)
    - [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### What Is an Embedding?

An embedding is a learned mapping from a finite set of discrete symbols (token IDs) to points in a continuous vector space. If the vocabulary has N tokens and the embedding dimension is d, then each token is represented by a vector in ℝᵈ:

$$\text{embed}: \{0, 1, \ldots, N-1\} \to \mathbb{R}^d$$

Concretely, the word "cat" might have token ID 2368 and be represented by the vector $[0.12, -0.45, 0.73, \ldots]$ with d = 768 components. This vector is **learned** during training — not hand-designed.

### Why Continuous Space?

Discrete IDs carry no relational structure: ID 2368 ("cat") tells you nothing about ID 2369 ("dog"). Continuous vectors enable three critical capabilities:

1. **Similarity**: $\cos(\text{embed}(\text{"cat"}), \text{embed}(\text{"dog"})) \approx 0.8$ — nearby in ℝᵈ means semantically related
2. **Interpolation**: Moving along a direction in ℝᵈ smoothly transitions between concepts
3. **Gradient-based learning**: Backpropagation requires differentiable operations — you can't take the gradient of an integer lookup, but you can take the gradient of a matrix-vector product

### Pipeline Position

```
Text → [Tokenizer] → Token IDs → [Embedding] → Vectors in ℝᵈ → [Attention] → …
                                   ^^^^^^^^^^^^
                                   THIS section
```

The embedding layer is a **table lookup**: given token ID $i$, return row $i$ of the embedding matrix $E$. Despite its simplicity, this matrix contains $N \times d$ learnable parameters and encodes the model's entire understanding of individual token meanings.

### Historical Context

| Era        | Method             | Key Idea                                                | Embedding Type |
| ---------- | ------------------ | ------------------------------------------------------- | -------------- |
| 2003       | Neural LM (Bengio) | First learned word vectors as byproduct of LM training  | Static         |
| 2013       | word2vec           | Skip-gram / CBOW: efficient training on massive corpora | Static         |
| 2014       | GloVe              | Factorising co-occurrence matrix: log-bilinear model    | Static         |
| 2018       | ELMo               | Bidirectional LSTM: context-dependent embeddings        | Contextual     |
| 2018       | BERT               | Transformer encoder: deep bidirectional context         | Contextual     |
| 2018–today | GPT family         | Autoregressive transformer: causal contextual vectors   | Contextual     |

The fundamental shift: from **one vector per word** (word2vec) to **one vector per token-in-context** (transformers). A transformer's "embedding" at layer $l$ depends on the entire surrounding sequence.

---

## 2. Formal Definitions

### 2.1 Embedding Matrix

The core data structure is the **embedding matrix**:

$$E \in \mathbb{R}^{N \times d}$$

where:

- $N$ = vocabulary size (number of distinct tokens)
- $d$ = embedding dimension (width of each vector)

Each row $E_i \in \mathbb{R}^d$ is the embedding vector for token $i$. The lookup operation is:

$$\text{embed}(i) = E_i = E^\top \mathbf{e}_i$$

where $\mathbf{e}_i$ is the one-hot vector with 1 at position $i$. In practice, this is implemented as an array index, not a matrix multiply — but mathematically they're equivalent.

**Parameter count**: The embedding matrix stores $N \times d$ floating-point values. For real models:

| Model       | N (vocab) | d     | Embedding Params | % of Total |
| ----------- | --------- | ----- | ---------------- | ---------- |
| GPT-2 Small | 50,257    | 768   | 38.6M            | 31%        |
| GPT-2 XL    | 50,257    | 1,600 | 80.4M            | 5.3%       |
| BERT-base   | 30,522    | 768   | 23.4M            | 21%        |
| LLaMA-7B    | 32,000    | 4,096 | 131M             | 1.9%       |
| LLaMA-70B   | 32,000    | 8,192 | 262M             | 0.4%       |

As models scale, the embedding fraction shrinks — the transformer layers dominate.

### 2.2 Vector Space ℝᵈ

Each embedding vector lives in **d-dimensional real coordinate space** $\mathbb{R}^d$. This space has:

- **Origin**: the zero vector $\mathbf{0} = (0, 0, \ldots, 0)$
- **Basis**: the standard basis $\{\mathbf{e}_1, \ldots, \mathbf{e}_d\}$ where $\mathbf{e}_j$ has a 1 in position $j$
- **Inner product**: $\langle \mathbf{u}, \mathbf{v} \rangle = \sum_{j=1}^d u_j v_j$ (the dot product)
- **Norm**: $\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$

Typical embedding dimensions by model scale:

| Model Scale | d          | Why                                             |
| ----------- | ---------- | ----------------------------------------------- |
| Small       | 256–512    | Lightweight; mobile/edge deployment             |
| Medium      | 768–1024   | BERT-base, GPT-2 small; good quality/cost ratio |
| Large       | 2048–4096  | GPT-3, LLaMA-7B; production LLMs                |
| Very Large  | 8192–12288 | LLaMA-70B, GPT-4; diminishing returns beyond    |

Each token occupies **exactly one point** in this space (at the input layer). The geometry of these points — their distances, angles, and clustering — encodes the model's learned knowledge about language.

### 2.3 Embedding Function

Formally, the embedding function is a composition:

$$\text{embed} = E \circ \text{id}: \mathcal{V} \to \mathbb{R}^d$$

where $\text{id}: \mathcal{V} \to \{0, \ldots, N-1\}$ maps tokens to integer IDs (from the tokenizer) and $E$ performs the row lookup.

**Initialisation**: Before training, embedding vectors are randomly initialised — typically:

$$E_{ij} \sim \mathcal{N}(0, 1/\sqrt{d}) \quad \text{or} \quad E_{ij} \sim \text{Uniform}(-c, c), \quad c \approx 0.02$$

The $1/\sqrt{d}$ scaling ensures that the dot product $\mathbf{u} \cdot \mathbf{v} = \sum u_j v_j$ (a sum of $d$ terms) has variance $\approx 1$ regardless of $d$, preventing exploding or vanishing signals at initialisation.

**Training**: Embeddings are updated via backpropagation like all other parameters:

$$E_i \leftarrow E_i - \eta \frac{\partial \mathcal{L}}{\partial E_i}$$

Tokens that appear frequently in the training data receive more gradient updates and develop richer, more informative representations. Rare tokens remain close to their random initialisation.

---

## 3. Mathematical Structure

### 3.1 Dot Product and Similarity

The **dot product** (inner product) is the fundamental operation on embedding vectors:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{j=1}^d u_j v_j = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where $\theta$ is the angle between the vectors. It measures two things simultaneously:

1. **Alignment** (angle): how much the vectors point in the same direction
2. **Magnitude**: longer vectors produce larger dot products

In transformers, the dot product is used to compute **attention scores**:

$$\text{score}(i, j) = \mathbf{q}_i \cdot \mathbf{k}_j = \sum_{l=1}^{d_k} q_{il} \cdot k_{jl}$$

This means the geometry of the embedding space directly determines which tokens attend to which other tokens.

**Properties**:

- Symmetric: $\mathbf{u} \cdot \mathbf{v} = \mathbf{v} \cdot \mathbf{u}$
- Bilinear: $(a\mathbf{u} + b\mathbf{w}) \cdot \mathbf{v} = a(\mathbf{u} \cdot \mathbf{v}) + b(\mathbf{w} \cdot \mathbf{v})$
- Positive definite: $\mathbf{v} \cdot \mathbf{v} \geq 0$, equality iff $\mathbf{v} = \mathbf{0}$

### 3.2 Cosine Similarity

**Cosine similarity** normalises the dot product by vector magnitudes:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \cdot \|\mathbf{v}\|} = \frac{\sum_{j=1}^d u_j v_j}{\sqrt{\sum_j u_j^2} \cdot \sqrt{\sum_j v_j^2}} \in [-1, 1]$$

**Interpretation**:

| cos value | Meaning             | Example              |
| --------- | ------------------- | -------------------- |
| 1.0       | Identical direction | "cat" vs "cat"       |
| 0.7–0.9   | Highly similar      | "cat" vs "kitten"    |
| 0.3–0.5   | Somewhat related    | "cat" vs "animal"    |
| ≈ 0       | Unrelated           | "cat" vs "democracy" |
| −0.5 to 0 | Weakly opposite     | "hot" vs "mild"      |
| −1.0      | Opposite direction  | Rare in practice     |

Cosine similarity is the **standard metric** for semantic similarity in NLP because:

- It's invariant to vector magnitude (a 2× longer vector has the same cosine)
- It ranges from −1 to +1, giving a natural similarity score
- It's fast to compute (dot product + two norms)

**When cosine fails**: For tasks where magnitude carries information (e.g., word frequency encoded in norm), raw dot product or Euclidean distance may be more appropriate.

### 3.3 Euclidean Distance

The **Euclidean distance** (L2 distance) measures the straight-line distance between two points:

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_{j=1}^d (u_j - v_j)^2}$$

**Relationship to cosine similarity** (for unit-normalised vectors $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$):

$$d(\mathbf{u}, \mathbf{v})^2 = \|\mathbf{u}\|^2 + \|\mathbf{v}\|^2 - 2\mathbf{u} \cdot \mathbf{v} = 2(1 - \cos(\mathbf{u}, \mathbf{v}))$$

So for normalised embeddings, Euclidean distance and cosine similarity are **monotonically related** — minimising one maximises the other. Many vector databases pre-normalise embeddings for this reason.

**When to use Euclidean distance**:

- Clustering algorithms (k-means operates on Euclidean distance)
- When magnitude differences matter
- In conjunction with distance-based similarity kernels (e.g., RBF: $K(\mathbf{u}, \mathbf{v}) = \exp(-\gamma d(\mathbf{u}, \mathbf{v})^2)$)

### 3.4 Norms

A **norm** measures the "size" of a vector. The three most common norms in ML:

$$\|\mathbf{v}\|_1 = \sum_{j=1}^d |v_j| \quad \text{(L1 / Manhattan norm)}$$

$$\|\mathbf{v}\|_2 = \sqrt{\sum_{j=1}^d v_j^2} \quad \text{(L2 / Euclidean norm)}$$

$$\|\mathbf{v}\|_\infty = \max_{j} |v_j| \quad \text{(L∞ / max norm)}$$

**General p-norm**:

$$\|\mathbf{v}\|_p = \left(\sum_{j=1}^d |v_j|^p\right)^{1/p}$$

**Ordering**: $\|\mathbf{v}\|_\infty \leq \|\mathbf{v}\|_2 \leq \|\mathbf{v}\|_1 \leq \sqrt{d} \|\mathbf{v}\|_2 \leq d \|\mathbf{v}\|_\infty$

In embedding spaces:

- **L2 normalisation** ($\hat{\mathbf{v}} = \mathbf{v} / \|\mathbf{v}\|_2$) maps all embeddings onto the unit hypersphere $S^{d-1}$, making cosine similarity equivalent to dot product
- **L1 norm** appears in sparse regularisation (pushing embedding components to zero)
- **L∞ norm** bounds the worst-case component magnitude; relevant for quantisation

---

## 4. Geometric Properties

### 4.1 Linear Relationships (Analogy Structure)

The most celebrated property of embedding spaces is **vector arithmetic for analogies**:

$$\text{embed}(\text{king}) - \text{embed}(\text{man}) + \text{embed}(\text{woman}) \approx \text{embed}(\text{queen})$$

Geometrically, this means the **offset vector** $\mathbf{d}_{\text{gender}} = \text{embed}(\text{man}) - \text{embed}(\text{woman})$ is approximately constant across word pairs:

$$\text{embed}(\text{king}) - \text{embed}(\text{queen}) \approx \text{embed}(\text{man}) - \text{embed}(\text{woman}) \approx \mathbf{d}_{\text{gender}}$$

This forms a **parallelogram** in ℝᵈ:

```
  king ─────────────────── queen
    │                        │
    │   d_gender             │   d_gender
    │                        │
  man ──────────────────── woman
```

**Why this works**: During training, the model learns to capture semantic relationships as geometric relationships. If "king" co-occurs with "royal" contexts and "man" co-occurs with "male" contexts, and "queen" co-occurs with both, then the difference vectors align.

**Common analogy types**:

| Relationship    | Example                           |
| --------------- | --------------------------------- |
| Gender          | king − man + woman ≈ queen        |
| Tense           | walking − walk + swim ≈ swimming  |
| Country-capital | France − Paris + Berlin ≈ Germany |
| Comparative     | bigger − big + small ≈ smaller    |
| Plural          | cats − cat + dog ≈ dogs           |

**Caveats**: Analogy arithmetic works best with word2vec/GloVe (optimised for linear structure). In contextual models like GPT, relationships are more complex and layer-dependent.

### 4.2 Subspaces and Directions

The embedding space contains **meaningful directions** — unit vectors along which semantics vary smoothly:

- **Gender direction**: The difference $\text{embed}(\text{he}) - \text{embed}(\text{she})$, averaged over many gendered pairs, defines a direction in ℝᵈ
- **Sentiment direction**: positive − negative word pairs define a sentiment axis
- **Tense direction**: past − present verb pairs

These directions span **semantic subspaces**. PCA (Principal Component Analysis) on the embedding matrix reveals the dominant axes of variation:

$$E = U \Sigma V^\top$$

The top principal components (columns of $V$ with largest singular values) capture the most important directions in the embedding space. Empirically:

- **First few PCs**: often capture frequency effects (common vs rare tokens)
- **Next PCs**: semantic categories (content words vs function words)
- **Later PCs**: fine-grained distinctions (occupation, topic, register)

**Key insight**: Individual dimensions of $\mathbf{e} = (e_1, e_2, \ldots, e_d)$ are **not interpretable**. Any rotation $R \in O(d)$ of the embedding space gives an equally valid representation with the same dot products: $\langle R\mathbf{u}, R\mathbf{v} \rangle = \langle \mathbf{u}, \mathbf{v} \rangle$. Only **directions** (which are rotation-invariant relationships) are meaningful.

### 4.3 Isotropy vs Anisotropy

**Isotropy** means embeddings are uniformly distributed across all directions in ℝᵈ. **Anisotropy** means they're clustered in a narrow cone or low-dimensional subspace.

**Measuring isotropy**: Compute the average cosine similarity across random pairs of embeddings:

$$\bar{c} = \frac{1}{\binom{N}{2}} \sum_{i < j} \cos(\mathbf{e}_i, \mathbf{e}_j)$$

| $\bar{c}$ | Distribution       | Consequence                                     |
| --------- | ------------------ | ----------------------------------------------- |
| ≈ 0       | Isotropic (ideal)  | Full use of ℝᵈ; cosine similarity is meaningful |
| 0.2–0.5   | Mildly anisotropic | Common; partially degenerate                    |
| > 0.5     | Highly anisotropic | Cosine similarity becomes unreliable            |

**Why anisotropy occurs**:

1. **Frequency bias**: Common tokens (function words like "the", "is") dominate training, pulling the mean embedding away from origin
2. **Low-rank structure**: The embedding matrix tends to be approximately low-rank (most variance in first few PCs)
3. **Training dynamics**: SGD with momentum can cause embeddings to drift in a common direction

**Fixing anisotropy**:

- **Mean centring**: $\hat{\mathbf{e}}_i = \mathbf{e}_i - \bar{\mathbf{e}}$ where $\bar{\mathbf{e}} = \frac{1}{N}\sum_i \mathbf{e}_i$
- **Whitening**: $\hat{\mathbf{e}}_i = \Sigma^{-1/2}(\mathbf{e}_i - \bar{\mathbf{e}})$ where $\Sigma$ is the covariance matrix
- **All-but-the-top**: Remove the top $k$ principal components that capture the anisotropic drift

These corrections dramatically improve the quality of cosine similarity for retrieval and sentence comparison tasks.

### 4.4 The Curse of Dimensionality

In high-dimensional spaces ($d \gg 100$), intuition from 2D/3D geometry breaks down:

**Concentration of distances**: For random vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^d$ with i.i.d. components:

$$\frac{\|\mathbf{u} - \mathbf{v}\|}{\sqrt{d}} \xrightarrow{d \to \infty} \text{constant}$$

All pairwise distances converge to the same value. The ratio:

$$\frac{\max d(\mathbf{u}, \mathbf{v}) - \min d(\mathbf{u}, \mathbf{v})}{\min d(\mathbf{u}, \mathbf{v})} \to 0 \quad \text{as } d \to \infty$$

This means nearest-neighbour search becomes unreliable — the "nearest" and "farthest" points are almost equidistant.

**Concentration of dot products**: For random Gaussian vectors:

$$\mathbf{u} \cdot \mathbf{v} = \sum_{j=1}^d u_j v_j$$

By CLT, this is approximately $\mathcal{N}(0, d)$, so $\cos(\mathbf{u}, \mathbf{v}) \approx \mathcal{N}(0, 1/d)$ — cosine similarities concentrate around zero with vanishing variance.

**Practical consequences**:

- Embedding dimensions beyond $d \approx 4096$ show diminishing returns
- Approximate nearest-neighbour (ANN) algorithms (FAISS, Annoy) are essential for retrieval
- Dimensionality reduction (PCA to $d' \ll d$) loses surprisingly little information

---

## 5. Positional Encodings

### 5.1 Why Position Matters

Self-attention is **permutation-equivariant**: for any permutation $\pi$,

$$\text{Attention}(\pi(X)) = \pi(\text{Attention}(X))$$

This means "the cat sat" and "sat the cat" produce the same attention patterns (just permuted). Without positional information, the model cannot distinguish word order.

Position must be **injected** into the embedding vectors. The combined input to the transformer is:

$$\mathbf{x}_i = \text{embed}(t_i) + \text{pos}(i)$$

where $\text{pos}(i) \in \mathbb{R}^d$ encodes the position index $i$.

### 5.2 Sinusoidal Positional Encoding

The original Transformer (Vaswani et al., 2017) uses fixed sinusoidal functions:

$$\text{PE}(\text{pos}, 2k) = \sin\left(\frac{\text{pos}}{10000^{2k/d}}\right)$$

$$\text{PE}(\text{pos}, 2k+1) = \cos\left(\frac{\text{pos}}{10000^{2k/d}}\right)$$

for $k = 0, 1, \ldots, d/2 - 1$.

**Why sinusoids?**

1. **Unique encoding**: Each position gets a unique vector — no two positions have the same PE
2. **Smooth variation**: Nearby positions have similar PE vectors: $\text{PE}(\text{pos}) \cdot \text{PE}(\text{pos}+1) \approx 1$
3. **Relative position via linear transformation**: For any fixed offset $k$, there exists a matrix $M_k$ such that $\text{PE}(\text{pos} + k) = M_k \cdot \text{PE}(\text{pos})$. This is because:

$$\sin(\alpha + \beta) = \sin\alpha \cos\beta + \cos\alpha \sin\beta$$
$$\cos(\alpha + \beta) = \cos\alpha \cos\beta - \sin\alpha \sin\beta$$

So the sin/cos pair at each frequency can be rotated by a 2×2 matrix to shift position.

4. **Wavelength spectrum**: Dimension pair $k$ has wavelength $\lambda_k = 2\pi \cdot 10000^{2k/d}$, ranging from $2\pi$ (dimension 0) to $2\pi \cdot 10000$ (dimension $d-1$). Lower dimensions encode **fine-grained** position; higher dimensions encode **coarse** position — like a binary representation where each bit has a different period.

5. **Length generalisation**: Since the PE is a deterministic function (not learned), it can be evaluated at any position, even positions not seen during training.

**Inner product between positions**:

$$\text{PE}(\text{pos}_1) \cdot \text{PE}(\text{pos}_2) = \sum_{k} \cos\left(\frac{\text{pos}_1 - \text{pos}_2}{10000^{2k/d}}\right)$$

This depends only on the **relative** distance $|\text{pos}_1 - \text{pos}_2|$, not the absolute positions.

### 5.3 Learned Positional Embeddings

An alternative is a learned positional embedding matrix:

$$P \in \mathbb{R}^{L \times d}$$

where $L$ is the maximum sequence length. The positional encoding is simply:

$$\text{pos}(i) = P_i \quad (i\text{-th row of } P)$$

**Used by**: BERT ($L = 512$), GPT-2 ($L = 1024$), GPT-3 ($L = 2048$).

**Advantages**:

- Can learn arbitrary position-dependent patterns
- No assumption about sinusoidal structure

**Disadvantages**:

- Cannot generalise beyond training length $L$ (position $L+1$ has no embedding)
- Additional $L \times d$ parameters (e.g., GPT-2: $1024 \times 768 = 786K$ params)
- Positions near the end of training sequences may be undertrained if inputs are shorter than $L$

### 5.4 Rotary Positional Encoding (RoPE)

RoPE (Su et al., 2021) encodes position via **rotation** of query and key vectors in the attention computation, rather than adding a positional vector to the input embedding.

For a 2D subspace (dimensions $2k$ and $2k+1$), position $m$ applies the rotation:

$$R_m^{(k)} = \begin{pmatrix} \cos(m\theta_k) & -\sin(m\theta_k) \\ \sin(m\theta_k) & \cos(m\theta_k) \end{pmatrix}$$

where $\theta_k = 10000^{-2k/d}$. The full rotation $R_m \in \mathbb{R}^{d \times d}$ is block-diagonal with these 2×2 rotation matrices.

**Key property**: The dot product between rotated query and key depends only on **relative** position:

$$(R_m \mathbf{q}) \cdot (R_n \mathbf{k}) = \mathbf{q}^\top R_m^\top R_n \mathbf{k} = \mathbf{q}^\top R_{n-m} \mathbf{k}$$

This is because rotation matrices satisfy $R_m^\top R_n = R_{n-m}$ (group property).

**Advantages over sinusoidal PE**:

- Relative position naturally encoded in attention scores (no need for the model to learn this)
- Decays with distance: $|(R_m \mathbf{q}) \cdot (R_n \mathbf{k})|$ tends to decrease as $|m - n|$ grows
- Extrapolates well beyond training length with NTK-aware scaling or YaRN

**Used by**: LLaMA, LLaMA-2, Mistral, GPT-NeoX, PaLM, CodeLlama.

### 5.5 ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) takes a radically different approach: no positional vector is added to the embeddings at all. Instead, a **linear bias** is added directly to the attention logits:

$$\text{score}(i, j) = \mathbf{q}_i \cdot \mathbf{k}_j - m \cdot |i - j|$$

where $m$ is a head-specific slope. Different attention heads use different slopes (geometrically spaced from $2^{-1}$ down to $2^{-8}$ for 8 heads), giving each head a different "attention span":

| Head | Slope $m$  | Effective span |
| ---- | ---------- | -------------- |
| 1    | 0.5        | Very local     |
| 2    | 0.25       | Local          |
| 3    | 0.125      | Medium         |
| ...  | ...        | ...            |
| 8    | 0.00390625 | Nearly global  |

**Advantages**:

- Zero additional parameters
- Strong length extrapolation (train on 1K, inference on 128K)
- Simple to implement

**Used by**: BLOOM, MPT.

**Comparison of positional encoding methods**:

| Method     | Type              | Relative pos?         | Length extrapolation? | Extra params | Used in              |
| ---------- | ----------------- | --------------------- | --------------------- | ------------ | -------------------- |
| Sinusoidal | Additive, fixed   | Via linear transform  | Yes (theoretically)   | 0            | Original Transformer |
| Learned PE | Additive, learned | No (absolute)         | No                    | L × d        | GPT-2, BERT          |
| RoPE       | Multiplicative    | Yes (by construction) | Yes (with scaling)    | 0            | LLaMA, Mistral       |
| ALiBi      | Attention bias    | Yes                   | Yes (strong)          | 0            | BLOOM, MPT           |

---

## 6. Embedding Training

### 6.1 Random Initialisation

Before any training, embedding vectors are randomly initialised. Common strategies:

**Xavier/Glorot**: $E_{ij} \sim \mathcal{N}(0, \sigma^2)$ where $\sigma = \sqrt{2 / (N + d)}$

**Standard**: $E_{ij} \sim \mathcal{N}(0, 1/\sqrt{d})$

**Small uniform**: $E_{ij} \sim \text{Uniform}(-c, c)$ with $c \approx 0.02$

The $1/\sqrt{d}$ scaling is critical. If embeddings are initialised with unit variance, then the dot product $\mathbf{u} \cdot \mathbf{v} = \sum_{j=1}^d u_j v_j$ has variance $d \cdot \text{Var}(u_j) \cdot \text{Var}(v_j) = d$. With $1/\sqrt{d}$ scaling, the variance of each component is $1/d$, so:

$$\text{Var}(\mathbf{u} \cdot \mathbf{v}) = d \cdot (1/d) \cdot (1/d) = 1/d$$

This keeps attention logits $\mathbf{q} \cdot \mathbf{k}$ from blowing up at initialisation (the same motivation behind the $1/\sqrt{d_k}$ scaling in attention).

### 6.2 Cross-Entropy Loss and Gradient Flow

The standard training objective for language models is **cross-entropy loss**:

$$\mathcal{L} = -\log P(t_{\text{target}} | \text{context}) = -\log \frac{\exp(\mathbf{h} \cdot \mathbf{w}_{t_{\text{target}}})}{\sum_{j=0}^{N-1} \exp(\mathbf{h} \cdot \mathbf{w}_j)}$$

where $\mathbf{h}$ is the final hidden state and $\mathbf{w}_j = W_j$ is the $j$-th row of the output projection (LM head) $W \in \mathbb{R}^{N \times d}$.

**Gradient flow to the embedding**:

1. Loss $\mathcal{L}$ → gradient on logits $\mathbf{z} = W\mathbf{h}$
2. Gradient on $\mathbf{h}$ → backprop through all transformer layers
3. At the input: $\mathbf{h}_0 = \text{embed}(t) + \text{pos}(i)$, so $\frac{\partial \mathcal{L}}{\partial E_t} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_0}$

The embedding for token $t$ only receives a gradient **when $t$ appears in the input**. This creates a frequency bias:

| Token frequency | Gradient updates | Result                                 |
| --------------- | ---------------- | -------------------------------------- |
| Very common     | Millions         | Well-trained, stable embedding         |
| Common          | Thousands        | Adequate representation                |
| Rare            | Tens             | Poorly trained, stays near random init |
| Unseen          | Zero             | Completely random                      |

This is why rare token embeddings are unreliable, and why subword tokenisation (BPE) helps — it ensures even rare words are composed of well-trained subword tokens.

### 6.3 Tied Embeddings

In many models, the input embedding matrix $E$ and the output LM head $W$ share the same weights:

$$W = E \quad \text{(weight tying)}$$

The output logit for token $j$ is:

$$z_j = \mathbf{h} \cdot E_j = \mathbf{h} \cdot \text{embed}(j)$$

**Interpretation**: The score for predicting token $j$ is the **dot product** between the final hidden state and the embedding of $j$. This means the model predicts the token whose embedding is most similar to the hidden state — prediction is literally a nearest-neighbour lookup in embedding space.

**Parameter savings**: Without tying: $2 \times N \times d$ params. With tying: $N \times d$ params. For GPT-2 Small: saving 38.6M params (31% of model).

**Trade-off**: The embedding must serve dual roles:

1. As **input representation**: encoding the meaning of the token in context
2. As **output target**: being the vector that the hidden state should point toward for correct prediction

These roles are not necessarily aligned, which is why some large models (GPT-3) use separate matrices.

### 6.4 Pretrained vs Fine-tuned Embeddings

**Freezing**: Keep $E$ fixed during fine-tuning; only update task-specific layers.

- Avoids catastrophic forgetting of general language knowledge
- Faster training (fewer parameters to update)
- Used when fine-tuning data is small

**Full fine-tuning**: Update $E$ along with all other parameters.

- Best performance on large task-specific datasets
- Risk: rare tokens may drift from meaningful representations

**Parameter-efficient fine-tuning (PEFT)**:

- **LoRA**: Add low-rank update $\Delta E = BA$ where $B \in \mathbb{R}^{N \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$. Only $r(N + d)$ trainable params instead of $Nd$
- **Adapter layers**: Insert small bottleneck layers; leave embeddings frozen
- **Prompt tuning**: Learn a small set of "soft prompt" embeddings prepended to the input; original embeddings frozen

---

## 7. Contextual vs Static Embeddings

### 7.1 Static Embeddings (word2vec, GloVe)

In static embedding models, each token has **one fixed vector** regardless of context:

$$\text{embed}(\text{"bank"}) = \mathbf{e}_{\text{bank}} \quad \text{(same in every sentence)}$$

**word2vec (Skip-gram)**: Trained to predict context words from a centre word. The objective maximises:

$$\sum_{(w, c) \in D} \log \sigma(\mathbf{e}_w \cdot \mathbf{e}_c) + k \cdot \mathbb{E}_{c' \sim P_n} [\log \sigma(-\mathbf{e}_w \cdot \mathbf{e}_{c'})]$$

where $\sigma$ is the sigmoid function, $D$ is the set of observed (word, context) pairs, and $c'$ is a negative sample.

**GloVe**: Factorises the log co-occurrence matrix:

$$\mathbf{e}_i \cdot \mathbf{e}_j + b_i + b_j \approx \log X_{ij}$$

where $X_{ij}$ is the number of times word $j$ appears in the context of word $i$.

**Limitations**: "bank" (river) and "bank" (finance) have the same vector. The model cannot represent polysemy.

### 7.2 Contextual Embeddings (BERT, GPT)

In transformer models, the representation of token $t_i$ at layer $l$ depends on **all tokens in the sequence**:

$$\mathbf{h}_i^{(l)} = f_l(\mathbf{h}_1^{(l-1)}, \mathbf{h}_2^{(l-1)}, \ldots, \mathbf{h}_n^{(l-1)})$$

The input to layer 0 is the static embedding + position: $\mathbf{h}_i^{(0)} = E_{t_i} + P_i$. Each subsequent layer transforms these representations through self-attention and feedforward networks.

**"bank" in context**:

- "I deposited money at the **bank**" → $\mathbf{h}_{\text{bank}}^{(12)}$ points toward financial vectors
- "The river **bank** was muddy" → $\mathbf{h}_{\text{bank}}^{(12)}$ points toward geographical vectors

The same token ID gets **different vectors** depending on surrounding context — this is the fundamental advance of contextual embeddings.

**Computational cost**: Static embeddings are a table lookup ($O(1)$ per token). Contextual embeddings require a full transformer forward pass ($O(n^2 d)$ for a sequence of length $n$).

### 7.3 Layer-wise Representation

Different transformer layers encode different types of information. This has been extensively studied via **probing classifiers** — training a simple classifier on frozen intermediate representations to test what information is present:

| Layer range         | Information encoded            | Evidence                                    |
| ------------------- | ------------------------------ | ------------------------------------------- |
| Layer 0 (input)     | Token identity, position       | Essentially the static embedding + PE       |
| Layers 1–3 (lower)  | Syntax: POS tags, constituency | High probing accuracy for syntax tasks      |
| Layers 4–8 (middle) | Semantics: word sense, NER     | Peak performance on semantic tasks          |
| Layers 9–12 (upper) | Task-specific: next-token pred | Best for the specific pretraining objective |

**Practical guideline**: When extracting embeddings for downstream use:

- **Sentence similarity**: Use average of middle layers (layers 4–8)
- **Syntax tasks**: Use lower layers
- **Classification**: Use the [CLS] token or last-layer average

---

## 8. Dimensionality and Model Scale

### 8.1 Embedding Dimension vs Model Size

The embedding dimension $d$ grows with model scale, but **sublinearly** relative to total parameters:

| Model       | Total Params | d     | Layers | Heads | Emb Params | Emb % |
| ----------- | ------------ | ----- | ------ | ----- | ---------- | ----- |
| GPT-2 Small | 124M         | 768   | 12     | 12    | 38.6M      | 31.1% |
| GPT-2 Med   | 355M         | 1,024 | 24     | 16    | 51.5M      | 14.5% |
| GPT-2 Large | 774M         | 1,280 | 36     | 20    | 64.3M      | 8.3%  |
| GPT-2 XL    | 1,558M       | 1,600 | 48     | 25    | 80.4M      | 5.2%  |
| LLaMA-7B    | 6,738M       | 4,096 | 32     | 32    | 131M       | 1.9%  |
| LLaMA-13B   | 13,015M      | 5,120 | 40     | 40    | 164M       | 1.3%  |
| LLaMA-70B   | 64,868M      | 8,192 | 80     | 64    | 262M       | 0.4%  |

**Pattern**: Embedding percentage drops from ~31% (small models) to <1% (very large models). The transformer layers grow as $\sim d^2 \times L$ (where $L$ is layer count), while the embedding grows as $\sim N \times d$.

### 8.2 Scaling Laws for Embeddings

Empirical scaling laws (Kaplan et al., 2020) suggest:

$$d \propto P^{0.5} \quad \text{(roughly)}$$

where $P$ is the total parameter count. More precisely, given a parameter budget, optimal allocation puts most capacity into depth (layers) and width ($d$), with the embedding matrix being a fixed overhead.

**Diminishing returns**: Increasing $d$ from 4096 to 8192 doubles the embedding parameters and all attention/FFN matrices, but empirical quality improvements are marginal per parameter. The benefit of larger $d$ is primarily more attention heads (each with dimension $d_k = d / h$), not richer per-token representations.

### 8.3 Dimensionality Reduction

High-dimensional embeddings ($d = 768, 4096, \ldots$) cannot be directly visualised. Three standard reduction techniques:

**PCA (Principal Component Analysis)**:

$$X_{\text{reduced}} = (X - \bar{X}) V_k$$

where $V_k$ contains the top-$k$ eigenvectors of the covariance matrix. PCA is **linear** and preserves global structure (distances between distant clusters). Typically reveals 80%+ of variance in the top 50–100 components.

**t-SNE** (van der Maaten & Hinton, 2008):

Minimises KL divergence between high-dimensional and low-dimensional neighbourhood probability distributions. Produces excellent 2D plots showing local cluster structure, but:

- Non-deterministic (depends on random initialisation)
- Distances between clusters are **not meaningful**
- Cannot embed new points without rerunning

**UMAP** (McInnes et al., 2018):

Similar to t-SNE but preserves more global structure and runs faster. Based on manifold theory and topological data analysis. Preferred for large-scale embedding visualisation.

**What reductions reveal**:

- Parts-of-speech cluster together (nouns, verbs, adjectives)
- Semantic fields form neighbourhoods (animals, countries, emotions)
- Multilingual models show aligned clusters across languages

---

## 9. Embedding Space in Attention

### 9.1 Query, Key, Value Projections

From the embedding $\mathbf{e} \in \mathbb{R}^d$ (after adding positional encoding), the model projects into three spaces:

$$\mathbf{q} = W^Q \mathbf{e}, \quad \mathbf{k} = W^K \mathbf{e}, \quad \mathbf{v} = W^V \mathbf{e}$$

where $W^Q, W^K \in \mathbb{R}^{d_k \times d}$ and $W^V \in \mathbb{R}^{d_v \times d}$. Typically $d_k = d_v = d / h$ where $h$ is the number of attention heads.

The attention score between positions $i$ and $j$ is:

$$\text{score}(i, j) = \frac{\mathbf{q}_i \cdot \mathbf{k}_j}{\sqrt{d_k}}$$

The **$\sqrt{d_k}$ scaling** prevents dot products from growing too large. Without it:

- $\text{Var}(\mathbf{q} \cdot \mathbf{k}) \approx d_k$ at initialisation
- Large dot products → softmax saturates → gradients vanish
- With scaling: $\text{Var}(\mathbf{q} \cdot \mathbf{k} / \sqrt{d_k}) \approx 1$ — safe for softmax

### 9.2 Attention as Soft Lookup

Attention computes a **weighted average** of value vectors:

$$\mathbf{o}_i = \sum_{j=1}^n \alpha_{ij} \mathbf{v}_j, \quad \alpha_{ij} = \frac{\exp(\text{score}(i, j))}{\sum_{l=1}^n \exp(\text{score}(i, l))}$$

This is interpretable as a **soft dictionary lookup**:

- **Query** $\mathbf{q}_i$: "what am I looking for?"
- **Keys** $\mathbf{k}_j$: "what does position $j$ have?"
- **Values** $\mathbf{v}_j$: "what information does position $j$ contribute?"

The attention weight $\alpha_{ij}$ measures how well the query at position $i$ matches the key at position $j$. The output is a mixture of all values, weighted by compatibility.

**Geometric interpretation**: The query defines a direction in $\mathbb{R}^{d_k}$. Keys that point in a similar direction get high attention weights. The output is a weighted centroid of value vectors in $\mathbb{R}^{d_v}$.

### 9.3 Residual Stream

The transformer uses **residual connections** at every layer:

$$\mathbf{x}^{(l+1)} = \mathbf{x}^{(l)} + \text{Attention}^{(l)}(\mathbf{x}^{(l)}) + \text{FFN}^{(l)}(\mathbf{x}^{(l)})$$

This means the original embedding vector is **never overwritten** — each layer adds a correction to it. The "residual stream" view (Elhage et al., 2021) shows that:

1. The input embedding $\mathbf{x}^{(0)} = E_{t_i} + P_i$ persists throughout the network
2. Each attention head reads from and writes to this shared stream
3. The final prediction uses the full accumulated representation: $\mathbf{x}^{(L)}$

**Consequence**: The embedding geometry at layer 0 directly affects what every subsequent layer can compute. A poorly initialised or poorly trained embedding creates a bottleneck for the entire model.

---

## 10. Common Mistakes

| Mistake                                                    | Why It's Wrong                                                                     | Fix                                                                      |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| "Embedding dimensions are interpretable"                   | Individual dimensions are arbitrary; any rotation gives equivalent representations | Analyse **directions** (e.g., PCA components), not individual dimensions |
| "Cosine similarity is always best"                         | Ignores magnitude information; can mislead for length-varying texts                | Choose metric based on task; consider L2 distance for clustering         |
| "Larger d is always better"                                | Diminishing returns; more params, slower inference, more memory                    | Tune d to model scale; d ∝ √(total params)                               |
| "Static embeddings are obsolete"                           | Fast, cheap, and sufficient for many structured/retrieval tasks                    | Use contextual only when you need context-dependent representations      |
| "Embeddings are the same at all layers"                    | Layer 1 and layer 12 encode very different information                             | Specify which layer when extracting embeddings                           |
| "Position 0 and position 100 are equally well represented" | Later positions may be undertrained if inputs are typically short                  | Check positional coverage in training data                               |
| "Tied weights always help"                                 | Dual role (input repr + output target) can be conflicting                          | Untie for very large models where parameter budget allows                |

---

## 11. Exercises

See [exercises.ipynb](exercises.ipynb) for full implementations with scaffolds and solutions.

### Exercise 1: Cosine Similarity Calculator

Compute $\cos(\text{embed}(\text{"cat"}), \text{embed}(\text{"dog"}))$ vs $\cos(\text{embed}(\text{"cat"}), \text{embed}(\text{"democracy"}))$ using random embeddings. Verify that related words should have higher similarity.

### Exercise 2: Analogy Arithmetic

Implement $\text{embed}(\text{king}) - \text{embed}(\text{man}) + \text{embed}(\text{woman})$ and find the nearest neighbour. Test 5 analogy types.

### Exercise 3: Sinusoidal PE Visualisation

Plot PE vectors for positions 0–99 across all dimensions. Observe the frequency structure and verify orthogonality properties.

### Exercise 4: Embedding Isotropy Analysis

Compute average pairwise cosine similarity for a random embedding matrix. Simulate anisotropy and show how mean-centering and whitening correct it.

### Exercise 5: RoPE Implementation

Implement rotary positional encoding for a pair of (query, key) vectors. Verify that the dot product depends only on relative position $m - n$.

### Exercise 6: Dimensionality Reduction

Apply PCA and t-SNE to 500 token embeddings. Colour by semantic category and identify clusters.

### Exercise 7: Parameter Counting

For GPT-2 Small through LLaMA-70B, compute embedding params as % of total. Plot the trend.

### Exercise 8: Layer Probing

Extract embeddings at layers 1, 6, 12 of a 12-layer model. Compare cosine similarity structure across layers.

### Exercise 9: Weight Tying Analysis

Train a small model with and without tied embeddings. Compare parameter count, training loss, and embedding–LM head similarity.

### Exercise 10: Curse of Dimensionality Demo

Generate random Gaussian vectors in $\mathbb{R}^d$ for $d \in \{2, 10, 100, 1000\}$. Plot nearest/farthest distance ratio as $d$ increases.

---

## 12. Why This Matters for AI

| Aspect               | Impact                                                                                           |
| -------------------- | ------------------------------------------------------------------------------------------------ |
| **Retrieval / RAG**  | Cosine similarity over embeddings is how documents are retrieved in RAG pipelines                |
| **Fine-tuning**      | Embedding geometry determines what the model can specialise on during fine-tuning                |
| **Interpretability** | Probing embedding space reveals what the model has learned (and what biases it encodes)          |
| **Multilingual**     | Cross-lingual embeddings share space across languages, enabling zero-shot cross-lingual transfer |
| **Bias detection**   | Social biases are encoded as geometric directions: gender, race, religion axes in ℝᵈ             |
| **Efficiency**       | Embedding table is often the memory bottleneck on edge devices; quantisation targets it first    |
| **Semantic search**  | All modern search engines use embedding similarity; geometry directly affects search quality     |

---

## 13. Further Reading

### Papers

1. Mikolov et al. (2013) — "Efficient Estimation of Word Representations in Vector Space" (word2vec)
2. Pennington et al. (2014) — "GloVe: Global Vectors for Word Representation"
3. Vaswani et al. (2017) — "Attention Is All You Need" (sinusoidal PE)
4. Su et al. (2021) — "RoFormer: Enhanced Transformer with Rotary Position Embedding" (RoPE)
5. Press et al. (2022) — "Train Short, Test Long: Attention with Linear Biases" (ALiBi)
6. Ethayarajh (2019) — "How Contextual are Contextualised Word Representations?" (isotropy analysis)
7. Kaplan et al. (2020) — "Scaling Laws for Neural Language Models"
8. Elhage et al. (2021) — "A Mathematical Framework for Transformer Circuits" (residual stream)

### Implementations

- [word2vec (gensim)](https://radimrehurek.com/gensim/) — Static embeddings in Python
- [SentenceTransformers](https://www.sbert.net/) — Contextual sentence embeddings
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [UMAP](https://umap-learn.readthedocs.io/) — Dimensionality reduction

### Conceptual Bridge

The embedding space is where tokens become meaning. The next section, [Attention Math](../03-Attention-Math/notes.md), covers how the model uses embedding geometry to route information across the sequence via the self-attention mechanism.

```
Token IDs → [Embedding] → Vectors in ℝᵈ → [Attention] → Contextualised vectors → …
              ^^^^^^^^^^^^                   ^^^^^^^^^^^
              THIS section                   NEXT section
```

---

[← Tokenization Math](../01-Tokenization-Math/notes.md) | [Home](../../README.md) | [Attention Math →](../03-Attention-Math/notes.md)
