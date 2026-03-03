# RAG Math and Retrieval

[← Quantization and Distillation](../11-Quantization-and-Distillation/notes.md) | [Home](../../README.md) | [Serving and Systems Tradeoffs →](../13-Serving-and-Systems-Tradeoffs/notes.md)

---

## 1. Intuition

### 1.1 What Is Retrieval-Augmented Generation?

RAG is a hybrid architecture that combines a retrieval system with a generative
language model. Instead of relying purely on knowledge encoded in model weights,
RAG retrieves relevant information from an external corpus at inference time.

Two-component system:

- **Retriever:** finds relevant documents from a large corpus
- **Reader/Generator:** uses retrieved documents to produce an answer

The model's **parametric memory** (weights) is augmented with **non-parametric memory**
(external document store). Analogy: the difference between a professor answering
from memory vs a researcher who first searches the library then formulates an answer.

### 1.2 Why RAG Exists — The Problems It Solves

| Problem                | Without RAG                              | With RAG                                |
| ---------------------- | ---------------------------------------- | --------------------------------------- |
| **Knowledge cutoff**   | Weights frozen at training time; stale   | Document store updated in real time     |
| **Hallucination**      | Confidently generates false information  | Retrieved documents ground the response |
| **Knowledge capacity** | 7B model can't memorise all corpora      | On-demand access to any corpus          |
| **Attributability**    | No source for claims                     | Retrieved documents provide citations   |
| **Cost of updates**    | Retraining 70B costs millions            | Updating a document store costs pennies |
| **Privacy**            | Sensitive data baked into shared weights | Data stays in retrieval store           |

### 1.3 The Core Mathematical Problem

Given query $q$ and corpus $\mathcal{D} = \{d_1, d_2, \ldots, d_n\}$:

1. **Find** subset $\mathcal{D}_q \subset \mathcal{D}$ most relevant to $q$
2. **Generate** response $y$ conditioned on $q$ and $\mathcal{D}_q$

- Relevance is a mathematical similarity problem in high-dimensional vector space
- Generation is a conditional probability problem: $P(y \mid q, \mathcal{D}_q)$
- The bridge: embedding functions that map both queries and documents into comparable
  vector representations

### 1.4 RAG vs Fine-tuning vs Long Context

| Approach         | Mechanism                    | Cost      | Update Speed           | Best For                              |
| ---------------- | ---------------------------- | --------- | ---------------------- | ------------------------------------- |
| **Fine-tuning**  | Bake knowledge into weights  | High      | Slow (retrain)         | Style, behaviour, static knowledge    |
| **RAG**          | Retrieve at inference        | Low       | Instant (update store) | Dynamic factual knowledge, citations  |
| **Long context** | Everything in context window | Per-query | N/A                    | Small corpus, full-document coherence |

2026 reality: hybrid approaches dominate. RAG + fine-tuning + long context all used
together in production systems.

### 1.5 Historical Timeline

| Year    | Work                     | Key Contribution                                                      |
| ------- | ------------------------ | --------------------------------------------------------------------- |
| 1975    | Salton et al.            | TF-IDF; foundation of information retrieval                           |
| 1976    | Robertson & Sparck Jones | Probabilistic relevance framework; BM25 foundations                   |
| 1990    | Deerwester et al.        | Latent Semantic Analysis (LSA); dense retrieval precursor             |
| 1994    | Robertson et al.         | BM25 algorithm; gold standard sparse retrieval for decades            |
| 2020    | Karpukhin et al.         | Dense Passage Retrieval (DPR); bi-encoder learned retrieval           |
| 2020    | Lewis et al.             | RAG paper (Facebook AI); first end-to-end trainable RAG system        |
| 2020    | Guu et al.               | REALM (Google); retrieval-augmented pretraining                       |
| 2021    | Izacard & Grave          | FiD (Fusion in Decoder); read multiple documents simultaneously       |
| 2021    | Borgeaud et al.          | RETRO (DeepMind); chunked cross-attention retrieval; 7B + 2T token DB |
| 2023    | Shi et al.               | REPLUG; retrieval as plug-in; no model fine-tuning needed             |
| 2023    | Asai et al.              | Self-RAG; model decides when to retrieve; self-reflection             |
| 2024    | Edge et al.              | GraphRAG (Microsoft); knowledge graphs for RAG                        |
| 2024–26 | Industry-wide            | Agentic RAG, multi-hop reasoning, multimodal retrieval, real-time web |

### 1.6 Pipeline Position

```
User Query
    ↓
[Query Encoder] → query vector q ∈ ℝᵈ
    ↓
[Vector Store / Index] → top-k document vectors
    ↓
[Document Decoder] → retrieved text d₁, …, dₖ
    ↓
[Generator LLM] conditioned on (query + documents)
    ↓
Generated Response with Citations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        THIS section
```

---

## 2. Formal Definitions

### 2.1 Corpus and Documents

- **Corpus:** $\mathcal{D} = \{d_1, d_2, \ldots, d_{|\mathcal{D}|}\}$; set of all retrievable documents
- **Document** $d_i$: a string of text; may be a sentence, paragraph, page, or arbitrary chunk
- **Chunk:** a fixed-length or semantically-bounded segment of a larger document
- **Passage:** a retrievable unit; typically 100–512 tokens
- $|\mathcal{D}|$: corpus size; ranges from thousands (enterprise) to billions (web-scale)

### 2.2 Retrieval Function

Maps query $q$ and corpus $\mathcal{D}$ to a ranked list of passages:

$$R: (q, \mathcal{D}) \to [(d_1, s_1), (d_2, s_2), \ldots, (d_k, s_k)]$$

- $s_i \in \mathbb{R}$: relevance score for document $d_i$
- Ranked by score: $s_1 \geq s_2 \geq \ldots \geq s_k$
- Top-$k$ retrieval: return $k$ documents with highest scores; $k = 3\text{–}20$ typical

### 2.3 Embedding Function

Encoder $E$ maps text to a fixed-dimensional dense vector:

$$E: \Sigma^* \to \mathbb{R}^d$$

- $d$ = embedding dimension; typically 384, 768, 1024, 1536, 3072
- Query embedding: $\mathbf{q} = E_q(\text{query}) \in \mathbb{R}^d$
- Document embedding: $\mathbf{p} = E_d(\text{document}) \in \mathbb{R}^d$
- **Bi-encoder:** separate encoders for query and document (efficient; asymmetric)
- **Cross-encoder:** single encoder processes concatenated (query, document) pair (accurate; slow)

### 2.4 Similarity Functions

| Function               | Formula                                                 | Range               | Notes                                     |
| ---------------------- | ------------------------------------------------------- | ------------------- | ----------------------------------------- |
| **Dot product**        | $\text{sim}(q,p) = q \cdot p = \sum_i q_i p_i$          | $(-\infty, \infty)$ | Fast; assumes normalised vectors          |
| **Cosine similarity**  | $\text{sim}(q,p) = \frac{q \cdot p}{\|q\| \cdot \|p\|}$ | $[-1, 1]$           | Direction only; most common               |
| **Euclidean distance** | $d(q,p) = \|q - p\|_2$                                  | $[0, \infty)$       | Magnitude-sensitive; $\text{sim} = -d$    |
| **Negative L2**        | $\text{sim} = -\|q - p\|^2$                             | $(-\infty, 0]$      | Equivalent to dot product when normalised |

For L2-normalised vectors: cosine similarity = dot product (monotonically related).
Standard practice: L2-normalise all embeddings; use dot product = cosine similarity.

### 2.5 RAG Generative Model

Generator produces response $y$ given query $q$ and retrieved documents $\mathcal{D}_q = \{d_1, \ldots, d_k\}$:

$$P(y \mid q, \mathcal{D}_q) = \prod_{t=1}^{|y|} P(y_t \mid y_{<t}, q, d_1, \ldots, d_k)$$

Model attends over both query and retrieved content during generation.

**Marginalised RAG** (Lewis et al. 2020):

$$P(y \mid q) = \sum_{d \in \mathcal{D}} P(d \mid q) \cdot P(y \mid q, d)$$

---

## 3. Sparse Retrieval

### 3.1 TF-IDF — Term Frequency–Inverse Document Frequency

Classic lexical retrieval; no neural networks; fast; interpretable.

**Term Frequency** (TF): how often term $t$ appears in document $d$:

$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

**Inverse Document Frequency** (IDF): how rare term $t$ is across corpus:

$$\text{IDF}(t, \mathcal{D}) = \log \frac{|\mathcal{D}|}{|\{d \in \mathcal{D} : t \in d\}| + 1}$$

**TF-IDF weight:** $w(t, d) = \text{TF}(t, d) \times \text{IDF}(t, \mathcal{D})$

Document vector: $\mathbf{w}(d) \in \mathbb{R}^{|V|}$ where $|V|$ = vocabulary size (sparse; mostly zeros).

### 3.2 BM25 — Best Match 25 (Robertson et al. 1994)

Improved probabilistic term weighting; gold standard for sparse retrieval:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

- $k_1 \in [1.2, 2.0]$: term frequency saturation; controls diminishing returns of repeated terms
- $b \in [0, 1]$: length normalisation; $b=1$ full normalisation; $b=0$ none; $b=0.75$ typical
- $\text{avgdl}$: average document length across corpus
- IDF component: $\log\!\left(\frac{N - \text{df} + 0.5}{\text{df} + 0.5}\right)$ where $N$ = corpus size, $\text{df}$ = document frequency

### 3.3 BM25 Properties

- **Saturating TF:** repeated terms have diminishing contribution (unlike raw TF-IDF)
- **Length normalisation:** longer documents don't unfairly dominate
- **IDF weighting:** rare terms more informative; matches human intuition
- **No learning:** entirely hand-crafted; fast to compute; no GPU needed
- **Limitations:** exact lexical match only; synonyms not handled; no semantic understanding

### 3.4 Inverted Index

Data structure: maps each term $t$ → list of (document_id, term_frequency) pairs.

- Construction: $O(|\mathcal{D}| \times \text{avg\_doc\_length})$; done once offline
- Query: look up each query term; merge posting lists; score with BM25
- Space: $O(\text{total\_terms} \times \text{avg\_term\_doc\_frequency})$
- Compression: posting lists compressed with delta encoding; typical 4–10 bytes per entry
- Production: Elasticsearch, Lucene, Solr

### 3.5 SPLADE — Sparse Learned Vocabulary Expansion (Formal et al. 2021)

Learned sparse retrieval: neural network produces sparse vectors in vocabulary space.

$$w(t, d) = \text{ReLU}(\text{BERT}(d))_t \cdot \text{IDF}(t)$$

FLOPS regularisation encourages sparsity:

$$\mathcal{L}_{\text{FLOPS}} = \sum_t \left(\frac{1}{|B|}\sum_{d \in B} w(t,d)\right)^2$$

Combines learned semantic matching with efficient inverted index retrieval. Bridges
dense and sparse: semantic expansion of query/document terms without dense vector search.

---

## 4. Dense Retrieval

### 4.1 Bi-Encoder Architecture

Two separate encoders; one for queries, one for documents:

- Query encoder: $E_q(q) \to \hat{\mathbf{q}} \in \mathbb{R}^d$
- Document encoder: $E_d(d) \to \hat{\mathbf{p}} \in \mathbb{R}^d$
- Similarity: $\text{sim}(q, d) = \hat{\mathbf{q}} \cdot \hat{\mathbf{p}}$ (after L2 normalisation)

Key property: document embeddings precomputed offline; only query embedding at inference.
Scalability: encode all $|\mathcal{D}|$ documents once; store in vector index; $O(d)$ query
encoding + $O(\log|\mathcal{D}|)$ search.

### 4.2 DPR — Dense Passage Retrieval (Karpukhin et al. 2020)

First large-scale successful dense retrieval for open-domain QA.

- Architecture: two independent BERT encoders $(E_q, E_d)$
- Training: contrastive learning with in-batch negatives + hard negatives
- Loss (in-batch softmax):

$$\mathcal{L}_{\text{DPR}} = -\log \frac{\exp(\mathbf{q} \cdot \mathbf{p}^+)}{\exp(\mathbf{q} \cdot \mathbf{p}^+) + \sum_{j=1}^{B-1} \exp(\mathbf{q} \cdot \mathbf{p}_j^-)}$$

- $\mathbf{p}^+$: positive passage (contains answer);
  $\mathbf{p}_j^-$: negative passages from other batch examples
- Hard negatives: passages retrieved by BM25 that don't contain answer but look relevant
- Result: significantly outperforms BM25 on NQ and TriviaQA

### 4.3 Contrastive Learning for Retrieval

Goal: pull query embeddings close to relevant document embeddings; push apart irrelevant.

InfoNCE loss (generalisation of DPR loss):

$$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\text{sim}(q_i, p_i^+) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(q_i, p_j) / \tau)}$$

- $\tau$: temperature; lower $\tau$ = sharper distinction between positive and negative
- **In-batch negatives:** $B-1$ negatives per query from same batch; efficient; scales with batch size
- **Cross-batch negatives:** share negatives across GPUs; effectively larger batch
- **MoCo-style queue:** maintain running queue of recent negatives; decouple batch size from negative count

### 4.4 Hard Negative Mining

| Strategy             | Description                                             | Quality                                            |
| -------------------- | ------------------------------------------------------- | -------------------------------------------------- |
| Random negatives     | Random documents from corpus                            | Low; easy to distinguish                           |
| BM25 hard negatives  | Top BM25 results that don't contain answer              | Good; lexically similar but semantically different |
| Model hard negatives | Current retriever's near-misses                         | Best; iteratively refined                          |
| ANCE (Xiong 2021)    | Refresh hard negatives every N steps using latest model | State-of-the-art                                   |
| Denoised negatives   | Verify with cross-encoder; remove false negatives       | Highest quality                                    |

### 4.5 Asymmetric vs Symmetric Retrieval

- **Symmetric:** query and document from same distribution; same encoder; QA pairs, duplicate detection
- **Asymmetric:** query is short question; document is long passage; different encoders; standard for QA
- Symmetric models: sentence-transformers/all-mpnet-base-v2; SBERT
- Asymmetric models: DPR; E5; BGE; multi-qa models

### 4.6 Late Interaction — ColBERT (Khattab & Zaharia 2020)

Bridge between bi-encoder (fast) and cross-encoder (accurate).

Both query and document encoded as **sequences of vectors** (not single vector):

- Query: $Q = E_q(\text{query}) \in \mathbb{R}^{|q| \times d}$; one vector per query token
- Document: $P = E_d(\text{doc}) \in \mathbb{R}^{|d| \times d}$; one vector per document token

MaxSim scoring:

$$\text{score}(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{q}_i \cdot \mathbf{d}_j$$

Each query token independently finds its best matching document token; sum over query tokens.
More expressive than bi-encoder; cheaper than cross-encoder (precompute doc vectors).

ColBERT v2 (2022): compressed token representations; residual compression; state-of-the-art
efficiency-quality tradeoff.

---

## 5. Approximate Nearest Neighbour Search

### 5.1 The Exact Search Problem

Brute-force nearest neighbour: compute similarity to all $|\mathcal{D}|$ documents; $O(|\mathcal{D}| \times d)$.

| Corpus Size | Storage (768d BF16) | Query Time  | Feasibility         |
| ----------- | ------------------- | ----------- | ------------------- |
| 1M          | 1.5 GB              | ~0.05ms GPU | Fast; exact is fine |
| 10M         | 15.4 GB             | ~0.5ms GPU  | Manageable          |
| 100M        | 154 GB              | ~5ms GPU    | Slow for real-time  |
| 1B          | 1.5 TB              | ~50ms GPU   | Too slow; need ANN  |

### 5.2 FAISS — Facebook AI Similarity Search (Johnson et al. 2019)

Production-grade library for efficient similarity search; GPU and CPU.
Multiple index types; each trades accuracy, memory, speed differently.

### 5.3 FAISS Flat Index (Exact Search)

Stores all vectors; brute-force L2 or dot product.

- Time: $O(|\mathcal{D}| \times d)$ per query; exact; no accuracy loss
- Use when: $|\mathcal{D}| < 1\text{M}$ and GPU available; baseline for accuracy comparison

### 5.4 Inverted File Index (IVF)

Partition embedding space into $n_{\text{list}}$ Voronoi cells using k-means clustering.

- Assignment: each document assigned to nearest centroid
- Query: find $n_{\text{probe}}$ nearest centroids; search only those cells
- Time: $O(n_{\text{list}} \times d)$ for centroid search + $O(|\mathcal{D}|/n_{\text{list}} \times n_{\text{probe}} \times d)$ for cell search
- $n_{\text{list}} = \sqrt{|\mathcal{D}|}$ typical; $n_{\text{probe}} = 1\text{–}100$
- With $n_{\text{list}}=1000$, $n_{\text{probe}}=10$: search ~1% of database; ~100× speedup

### 5.5 Product Quantization (PQ)

Compress each $d$-dimensional vector to $m \times \log_2 K$ bits:

- Partition $d$ dimensions into $m$ subvectors of $d/m$ dimensions each
- Quantize each subvector to one of $K$ centroids ($K=256 \to 8$ bits per subspace)
- Storage: $m$ bytes per vector (vs $d \times 4$ bytes FP32)
- For $d=768$, $m=4$: each vector stored as 4 bytes vs FP32 original $768 \times 4 = 3{,}072$ bytes → $768\times$ compression
- Distance computation: lookup table; $O(m)$ instead of $O(d)$
- **IVFPQ:** combine IVF (fast cell selection) + PQ (compressed within cell); standard for large-scale retrieval

### 5.6 HNSW — Hierarchical Navigable Small World (Malkov & Yashunin 2020)

Graph-based ANN index; best recall-speed tradeoff for most applications.

- **Hierarchical layers:** top layers sparse (long-range connections); bottom layer dense (all nodes)
- **Construction:** each new vector connects to $M$ nearest neighbours at each layer; probabilistic layer assignment
- **Query:** enter at top layer; greedily descend; explore $\text{ef\_search}$ candidates at bottom layer
- **Complexity:**
  - Build: $O(|\mathcal{D}| \times M \times \log|\mathcal{D}|)$
  - Query: $O(\log|\mathcal{D}|)$ expected
  - Memory: $O(|\mathcal{D}| \times M \times d)$

Parameters: $M = 16\text{–}64$; $\text{ef\_construction} = 200\text{–}400$; $\text{ef\_search} = 50\text{–}200$.

### 5.7 ScaNN — Scalable Nearest Neighbours (Google 2020)

- **Anisotropic quantization:** quantize dimensions proportionally to their contribution to inner product
- **Asymmetric hashing:** query and database vectors quantized differently
- **Parallel queries:** batched GPU execution
- Best-in-class throughput for high-recall search; Google's production retrieval backend

### 5.8 Recall and Precision Metrics for ANN

$$\text{Recall@k} = \frac{|ANN_k \cap \text{True}_k|}{k}$$

- **QPS (Queries Per Second):** throughput metric
- **Pareto frontier:** recall vs QPS; choose index type and parameters based on requirements
- ann-benchmarks.com: standard benchmark; compare FAISS, HNSW, ScaNN

### 5.9 Vector Database Systems

| System              | Type                  | Key Feature                                     |
| ------------------- | --------------------- | ----------------------------------------------- |
| **Pinecone**        | Managed cloud         | Serverless; automatic scaling                   |
| **Weaviate**        | Open-source           | Hybrid search (dense + sparse); graph traversal |
| **Qdrant**          | Open-source (Rust)    | Payload filtering; efficient disk-based index   |
| **Milvus / Zilliz** | Open-source + managed | Multiple index types; GPU acceleration          |
| **Chroma**          | Lightweight           | Python-native; in-process for development       |
| **pgvector**        | PostgreSQL extension  | SQL + vector search; familiar interface         |
| **Redis Vector**    | In-memory             | Ultra-low latency; HNSW or flat index           |

---

## 6. Embedding Models for Retrieval

### 6.1 BERT-Based Encoders

- BERT (Devlin et al. 2018): original transformer encoder; 110M–340M params
- **Sentence-BERT** (Reimers & Gurevych 2019): BERT fine-tuned with siamese network
  - Twin network: process sentence A and B separately; compare embeddings
  - Cosine similarity loss or contrastive loss on sentence pairs
- **Mean pooling:** average all token embeddings; better than CLS for sentence representation
- **CLS pooling:** use [CLS] token embedding; common for classification tasks

### 6.2 E5 — Text Embeddings by Weakly Supervised Contrastive Pre-training (Wang et al. 2022)

- Trained on massive weakly-supervised text pairs from web
- Instruction prefix: "query: " for queries; "passage: " for documents
- Contrastive pretraining then fine-tuning on BEIR
- E5-large-v2 (335M), E5-mistral-7B (7B): scale dramatically improves quality
- State-of-the-art dense retrieval as of 2023

### 6.3 BGE — BAAI General Embeddings (2023)

- Flagship open-source embedding family
- **BGE-M3** (2024): multi-lingual, multi-functionality, multi-granularity
  - Dense retrieval: single embedding per text
  - Sparse retrieval: SPLADE-style term weights
  - Multi-vector (ColBERT-style): per-token embeddings
  - All three from single model; unified retrieval
- 100+ languages; up to 8192 token input

### 6.4 OpenAI Embedding Models

| Model                  | Dimension | Notes                                       |
| ---------------------- | --------- | ------------------------------------------- |
| text-embedding-ada-002 | 1536      | Strong general-purpose; widely deployed     |
| text-embedding-3-small | 1536      | Cheaper; competitive with ada-002           |
| text-embedding-3-large | 3072      | Strongest OpenAI; flexible output dimension |

**Matryoshka embeddings (MRL):** embeddings where truncated prefix is still useful.
3072d → 256d truncation retains most information.

### 6.5 Matryoshka Representation Learning (MRL — Kusupati et al. 2022)

Train embedding such that first $m$ dimensions are useful at any $m \leq d$:

$$\mathcal{L}_{\text{MRL}} = \sum_{m \in M} \frac{1}{|M|} \mathcal{L}(f(x)[1:m])$$

- $M = \{8, 16, 32, 64, 128, 256, 512, 1024, 2048\}$ for $d=2048$
- Single model; use 128d for fast initial retrieval; 2048d for reranking
- **Cascaded retrieval:** coarse search with small $d$; refine with full $d$

### 6.6 Embedding Model Fine-tuning

- Domain adaptation: general embeddings may miss domain-specific terminology
- Fine-tuning data: (query, positive_passage, negative_passage) triples
- Typical recipe: start from strong base (E5 or BGE); fine-tune with domain pairs; 1–3 epochs
- Data generation: use LLM to generate synthetic query-passage pairs for domain corpus
  - Prompt: "Write a question that this passage answers: [passage text]"
  - Filter: remove low-quality pairs with cross-encoder score
- Result: 5–15% improvement on domain-specific retrieval

### 6.7 Embedding Dimension and Quality Tradeoff

| Model                  | Dimension | Storage per 1M docs | MTEB Score |
| ---------------------- | --------- | ------------------- | ---------- |
| all-MiniLM-L6-v2       | 384       | 768 MB BF16         | 56.3       |
| all-mpnet-base-v2      | 768       | 1.5 GB BF16         | 57.8       |
| E5-large-v2            | 1024      | 2.0 GB BF16         | 62.2       |
| text-embedding-3-large | 3072      | 6.0 GB BF16         | 64.6       |
| E5-mistral-7B          | 4096      | 8.0 GB BF16         | 66.6       |

---

## 7. Chunking Strategies

### 7.1 Why Chunking Matters

- Full documents too large for embedding: semantic dilution
- Too small: insufficient context; retrieved chunks miss surrounding information
- Wrong chunking can split key information across chunks; retrieval misses both halves
- No universally optimal chunk size; depends on content type and retrieval task

### 7.2 Fixed-Size Chunking

- Split document every $N$ tokens with optional overlap
- Chunk size $N$: typically 128–512 tokens
- Overlap $O$: typically 10–25% of chunk size (e.g. 32–64 tokens for $N=256$)
- Pros: simple; fast; predictable
- Cons: may split mid-sentence; ignores document structure

### 7.3 Sentence-Level Chunking

- Split at sentence boundaries detected by NLP parser or regex
- Variable chunk size; short sentences may not provide enough context
- **Sentence windowing:** group $k$ consecutive sentences; slide window by 1 sentence
- Typical: groups of 3–5 sentences; overlap of 1–2 sentences

### 7.4 Semantic Chunking

- Split when semantic similarity between consecutive sentences drops below threshold
- Embed each sentence; compute cosine similarity with next sentence
- Split when similarity $< \tau$ (threshold)
- Produces semantically coherent chunks; variable size; captures topic shifts
- Higher quality than fixed-size but slower (requires embedding every sentence)

### 7.5 Document Structure-Aware Chunking

- Respect document structure: headers, paragraphs, sections, code blocks
- Markdown splitting: split at ## headings; preserves section coherence
- Code splitting: split at function/class boundaries; AST-aware
- PDF chunking: detect page breaks, columns, tables; layout-aware parsing

### 7.6 Hierarchical / Parent-Child Chunking

- Index small chunks for precise retrieval; retrieve larger parent chunks for context
- Child chunks: 128 tokens (for precise embedding matching)
- Parent chunks: 512 tokens (for full context in generation)
- Process: retrieve child chunk; return parent chunk to LLM
- Balances retrieval precision and context richness

### 7.7 Chunking Evaluation

Metrics:

- **Context precision:** fraction of retrieved chunks that are relevant
- **Context recall:** fraction of relevant information covered by retrieved chunks
- **Chunk utilisation:** how much of each retrieved chunk is actually used in response
- **RAGAS framework** (2023): automated RAG evaluation; measures context precision/recall + answer faithfulness

---

## 8. Reranking

### 8.1 The Two-Stage Retrieval Pipeline

```
Stage 1 (Recall):    query → bi-encoder → top-k candidates (k=50–200)
                              ↓
Stage 2 (Precision): candidates → cross-encoder → top-r results (r=3–10)
```

Motivation: bi-encoder trades accuracy for speed; cross-encoder more accurate but slower.
Combined: fast recall + precise ranking; best of both worlds.

### 8.2 Cross-Encoder Architecture

- Concatenate: [CLS] query [SEP] document [SEP]
- BERT-style encoding; classification head produces relevance score
- Single scalar output per (query, document) pair
- Cannot precompute document representations; must process every $(q, d)$ pair at query time
- Time: $O(k \times \text{encode\_time})$ per query; $k=100$, encode=10ms → 1s latency

### 8.3 Cross-Encoder Training

| Approach      | Description                                        | Loss                       |
| ------------- | -------------------------------------------------- | -------------------------- |
| **Pointwise** | Predict relevance score                            | MSE loss                   |
| **Pairwise**  | Predict which of two docs is more relevant         | Hinge / binary CE          |
| **Listwise**  | Predict optimal ranking of $k$ docs simultaneously | More complex; best quality |

Hard negative mining: same strategies as bi-encoder; crucial for cross-encoder quality.

### 8.4 Reranking Models

| Model                   | Notes                                                        |
| ----------------------- | ------------------------------------------------------------ |
| MS-MARCO trained models | Standard benchmark; many open models                         |
| Cohere Rerank API       | Commercial; strong; easy integration                         |
| bge-reranker-v2 (2024)  | BAAI; open; multilingual; strong                             |
| Jina Reranker v2 (2024) | 137M; multilingual; 8192 token input                         |
| RankLLaMA (2023)        | LLaMA-based reranker; state-of-the-art                       |
| GPT-4 as reranker       | Prompt LLM to score relevance; expensive but highest quality |

### 8.5 LLM-Based Reranking

- **Pointwise:** "Is this document relevant to the query? Score 1–10"
- **Pairwise:** "Which document is more relevant? A or B?" — most reliable; $O(k^2)$
- **Listwise:** "Rank these documents most to least relevant" — efficient
- **Setwise:** compare small sets; $O(k \log k)$ with sorting
- **RankGPT** (Sun et al. 2023): sliding window permutation; LLM ranks window of 20 docs

### 8.6 Reciprocal Rank Fusion (RRF)

Combine rankings from multiple retrieval systems without score normalisation:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

- $k = 60$: smoothing constant; prevents excessive weight for top-ranked docs
- $R$: set of retrieval systems (e.g. BM25 + dense retrieval)
- Robust: no need to normalise scores across systems; works with any ranking
- Simple but effective: often outperforms score fusion

---

## 9. Hybrid Retrieval

### 9.1 Why Hybrid Retrieval

- Dense retrieval: strong semantic matching; misses exact keyword matches
- Sparse retrieval: exact keyword matching; misses paraphrases and synonyms
- Hybrid combines both; handles both semantic and lexical relevance
- Example: query "BERT acronym meaning" → dense finds BERT papers; sparse finds "Bidirectional Encoder"
- Standard practice (2024–2026): always use hybrid retrieval; pure dense rarely optimal

### 9.2 Score Fusion Methods

| Method                      | Description                                      | Pros                       |
| --------------------------- | ------------------------------------------------ | -------------------------- |
| **Normalised score fusion** | Normalise each system to [0,1]; weighted average | Intuitive                  |
| **RRF**                     | Rank-based; no normalisation                     | Robust; standard           |
| **CombMNZ**                 | Sum scores × number of systems that retrieved it | Rewards consensus          |
| **Learned fusion**          | Train lightweight model to combine scores        | Best quality; needs labels |

### 9.3 Linear Score Combination

$$\text{score}_{\text{hybrid}}(d, q) = \alpha \cdot \text{score}_{\text{dense}}(d, q) + (1-\alpha) \cdot \text{score}_{\text{sparse}}(d, q)$$

- $\alpha \in [0, 1]$: weight; $\alpha=0.5$ equal blend; tune on validation set
- Requires score normalisation: min-max or softmax normalisation
- Sensitivity: different query types have different score distributions

### 9.4 Query Routing

- Classify query type; route to optimal retrieval system
- Factoid queries → sparse + dense hybrid
- Semantic similarity queries → dense only
- Keyword / Boolean queries → sparse only
- **Adaptive retrieval:** try multiple systems; select based on confidence scores

### 9.5 Multi-Vector Hybrid (SPLADE + Dense)

$$\text{score} = w_1 \cdot \text{sim}_{\text{dense}}(q, d) + w_2 \cdot \text{sim}_{\text{sparse}}(q, d) + w_3 \cdot \text{sim}_{\text{colbert}}(q, d)$$

BGE-M3: single model producing all three; state-of-the-art on BEIR 2024.

---

## 10. Advanced RAG Architectures

### 10.1 Naive RAG

Simple pipeline: query → BM25/dense retrieval → top-$k$ passages → [query + passages] → LLM → answer.

Problems: retrieval may return irrelevant content; LLM may ignore retrieved content;
no iterative refinement.

### 10.2 Modular RAG (2023–2026)

Decompose RAG into interchangeable modules:

- **Modules:** query transformer, retriever, reranker, memory, generator, evaluator
- Flexible orchestration: chain modules in different orders for different tasks
- Examples:
  - Query → expand → retrieve → rerank → generate
  - Query → retrieve → validate → re-retrieve if poor → generate
  - Query → retrieve → summarise passages → retrieve again → generate

### 10.3 Query Transformation

| Technique               | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| **Query expansion**     | Add related terms via LLM or pseudo-relevance feedback        |
| **HyDE** (Gao 2022)     | Generate hypothetical answer; embed it for search             |
| **Step-back prompting** | Rewrite specific query as more general question               |
| **Multi-query**         | Generate $k$ diverse reformulations; retrieve for each; merge |

**HyDE — Hypothetical Document Embeddings:**

1. Generate hypothetical answer to query using LLM (no retrieval yet)
2. Embed hypothetical answer; use its embedding to search document store
3. Works because query and answer have different distributions; HyDE bridges the gap

### 10.4 Iterative / Recursive RAG

Single retrieval insufficient for complex multi-step questions.

- **Iterative RAG:** retrieve → read → identify gaps → retrieve again → read → answer
- **Recursive retrieval:** decompose question into sub-questions; retrieve for each; compose
- **FLARE** (Jiang et al. 2023): active retrieval; generate until low-confidence token; retrieve; continue
  - Retrieve exactly when needed; not just at query time

### 10.5 Self-RAG (Asai et al. 2023)

LLM learns to decide: when to retrieve, what to retrieve, how to use retrieved content.

Special tokens trained into LLM:

- `[Retrieve]`: should I retrieve now?
- `[IsRel]`: is this document relevant?
- `[IsSup]`: does retrieved content support my claim?
- `[IsUse]`: is my response useful?

Inference: LLM decides retrieval dynamically; evaluates its own outputs.

### 10.6 CRAG — Corrective RAG (Yan et al. 2024)

Add a retrieval evaluator: judge quality of retrieved documents.

- If quality **high**: use documents directly
- If quality **low**: trigger web search for better information
- If quality **ambiguous**: decompose into fine-grained knowledge strips; filter irrelevant

### 10.7 GraphRAG (Edge et al. 2024 — Microsoft)

Build knowledge graph from corpus; use graph structure for retrieval:

- **Entities:** extract named entities and attributes from all documents
- **Relations:** extract relationships between entities
- **Graph:** nodes = entities; edges = relationships; weighted by document evidence
- **Community detection:** cluster related entities; summarise each community
- **Local search:** retrieve specific entities; traverse subgraph
- **Global search:** synthesise across communities for broad questions

Strength: multi-hop reasoning; global synthesis; relationship-aware.
Weakness: expensive to build; requires entity/relation extraction at scale.

### 10.8 Agentic RAG (2024–2026)

RAG as part of a larger agentic loop; retrieval as a tool.

- Agent decides: when to retrieve, from which source, what follow-up retrieval to do
- Multi-source: internal DB, web search, APIs, code execution simultaneously
- **ReAct pattern:** Reason → Act (retrieve) → Observe → Reason → Act → …
- Challenges: retrieval latency in agentic loops; cost; error propagation

---

## 11. Context Integration and Generation

### 11.1 Context Window Organisation

How to arrange retrieved documents in context matters significantly.

- **"Lost in the middle"** effect (Liu et al. 2023): information in middle of long
  context retrieved less reliably
- Best positions: beginning and end of context window
- Optimal ordering: most relevant first; second most relevant last; others in middle

### 11.2 Fusion-in-Decoder (FiD — Izacard & Grave 2021)

Process each retrieved document independently in encoder; fuse in decoder:

- Encoder: encode (query + document_i) separately for each of $k$ documents
- Decoder: cross-attend over all $k$ encoded representations simultaneously
- Avoids context length limitation: each document processed independently
- Scales to $k=100$ without quadratic attention cost

### 11.3 Attention over Retrieved Documents

- Standard RAG: concatenate all documents; model attends over full context
- Cross-attention RAG: separate attention blocks for query and retrieved context
- **Token attribution:** which retrieved tokens does the model attend to?
- **Grounded attribution:** cite which document supports each claim in output

### 11.4 Retrieval-Augmented Prompting Patterns

| Pattern        | Description                                                | Use Case                                    |
| -------------- | ---------------------------------------------------------- | ------------------------------------------- |
| **Stuff**      | Concatenate all chunks into context                        | Small number of chunks; fits in window      |
| **Map-reduce** | Summarise each chunk (map); combine (reduce)               | Large corpus; parallel                      |
| **Map-rerank** | Answer from each chunk; rank answers                       | When best answer may come from single chunk |
| **Refine**     | Generate initial answer; refine with each subsequent chunk | Sequential; preserves context               |

### 11.5 Citation Generation

- **Post-hoc attribution:** generate response; then attribute each sentence to source
- **Inline citation:** generate [1], [2] markers inline during generation
- **Attribution training:** fine-tune model to generate citations alongside responses
- **ALCE benchmark** (Gao et al. 2023): measures citation accuracy in RAG responses

### 11.6 Hallucination in RAG

Types of RAG hallucinations:

- **Retrieved but ignored:** relevant document retrieved but model ignores it
- **Contradictory:** model contradicts retrieved content
- **Conflated:** mixes information from different retrieved documents incorrectly
- **Confabulated:** generates claim not supported by any retrieved document

Measurement:

- **Faithfulness score:** fraction of claims supported by retrieved documents
- **NLI-based faithfulness:** use NLI model to check entailment
- **RAGAS faithfulness metric:** automated LLM-as-judge evaluation

---

## 12. RAG Evaluation

### 12.1 End-to-End Metrics

| Metric               | Description                               | Usage          |
| -------------------- | ----------------------------------------- | -------------- |
| **Exact Match (EM)** | Binary; answer exactly matches gold       | Strict QA      |
| **F1 score**         | Token-level overlap                       | Less strict QA |
| **BLEU / ROUGE**     | N-gram overlap                            | Longer answers |
| **LLM-as-judge**     | GPT-4 evaluates correctness, faithfulness | Scalable       |

### 12.2 Component Metrics

**MRR — Mean Reciprocal Rank:**

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$

**NDCG — Normalised Discounted Cumulative Gain:**

$$\text{DCG@k} = \sum_{i=1}^{k} \frac{2^{\text{rel}_i} - 1}{\log_2(i+1)}, \quad \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

**MAP — Mean Average Precision:** area under precision-recall curve; averaged over queries.

### 12.3 RAGAS Framework (Es et al. 2023)

End-to-end RAG evaluation using LLM-as-judge. Four metrics:

- **Faithfulness:** fraction of claims supported by retrieved context
- **Answer relevance:** how relevant is answer to the question?
- **Context precision:** fraction of retrieved context relevant to answering
- **Context recall:** fraction of gold answer information covered by context

All metrics automated; no human annotation needed.

### 12.4 Retrieval Benchmarks

| Benchmark         | Task             | Corpus Size     | Metric        |
| ----------------- | ---------------- | --------------- | ------------- |
| MS-MARCO          | Passage ranking  | 8.8M passages   | MRR@10        |
| Natural Questions | Open-domain QA   | Wikipedia       | Recall@20     |
| TriviaQA          | Open-domain QA   | Wikipedia + Web | Exact Match   |
| BEIR              | 18 diverse tasks | Various         | NDCG@10       |
| LoTTE             | Long-tail QA     | StackExchange   | Success@5     |
| MTEB              | 56 tasks         | Various         | Average score |

### 12.5 RAG-Specific Benchmarks

- **RGB** (Chen et al. 2023): evaluates LLM ability to use retrieved documents; noise robustness
- **ARES** (Saad-Falcon et al. 2023): automated RAG evaluation with synthetic data
- **RECALL** (2024): counterfactual evaluation; test if model uses retrieved content over parametric knowledge
- **CRUD-RAG** (2024): create/read/update/delete operations; knowledge management

---

## 13. Indexing at Scale

### 13.1 Index Construction Pipeline

```
Raw Documents
    ↓ [Parsing] — PDF, HTML, DOCX, code extraction
    ↓ [Chunking] — fixed/semantic/structural splitting
    ↓ [Embedding] — batch encode all chunks
    ↓ [Vector Index] — FAISS/HNSW/ScaNN index build
    ↓ [Metadata Store] — document IDs, sources, timestamps
    ↓ [Keyword Index] — BM25 inverted index construction
    ↓ Deployed Retrieval System
```

### 13.2 Embedding Throughput

- GPU throughput: BERT-large, batch=512, A100: ~50K sentences/second
- For 10M documents × 256 tokens average: 2.56B tokens; ~14 hours on single A100
- Multi-GPU: linear scaling; 8×A100 → ~1.7 hours for 10M documents
- Smart batching by length reduces padding waste

### 13.3 Incremental Indexing

- HNSW: supports incremental insertion; $O(M \times \log|\mathcal{D}|)$ per new vector
- IVF: requires index rebuild when cluster assignments change; expensive
- Practical solution: small "hot" HNSW for recent documents; large "cold" IVF for historical
- Merge periodically: absorb hot index into cold; rebuild cold IVF

### 13.4 Index Compression for Scale

| Configuration | Storage for 1B vectors (768d) | Notes                        |
| ------------- | ----------------------------- | ---------------------------- |
| FP32 flat     | 3.072 TB                      | Impractical single machine   |
| BF16 flat     | 1.536 TB                      | Still very large             |
| IVFOPQ (m=32) | ~32 GB                        | Standard for billion-scale   |
| ScaNN (32×)   | ~96 GB                        | 90%+ recall retention        |
| DiskANN       | ~3 GB RAM + SSD               | Billion-scale single machine |

### 13.5 Metadata Filtering

- **Pre-filtering:** apply filter before ANN search; reduces search space
- **Post-filtering:** ANN search then filter results; may miss relevant filtered docs
- **Hybrid filtering:** retrieve filtered + unfiltered; merge results
- **Payload filtering** (Qdrant): filter on document metadata alongside vectors; production standard

---

## 14. Production RAG Systems

### 14.1 Latency Budget

| Component       | Typical Latency | Budget                             |
| --------------- | --------------- | ---------------------------------- |
| Query embedding | 10–50ms         | Small encoder on GPU               |
| ANN search      | 10–100ms        | Depends on index type and size     |
| Reranking       | 100–500ms       | Cross-encoder on 50–100 candidates |
| LLM generation  | 500ms–5s        | Depends on model and output length |
| **Total**       | **500ms–2s**    | Interactive RAG target             |

Optimisation: parallel retrieval; cache frequent queries; streaming generation.

### 14.2 Caching Strategies

- **Query result cache:** cache (query_embedding → top-$k$ documents); TTL based on data freshness
- **Semantic cache:** cache queries similar to previous queries (cosine sim > threshold)
  - GPTCache (2023): semantic cache for LLM queries; 2–10× cost reduction
- **KV cache reuse:** if system prompt + docs same across requests, reuse KV cache
- **Prefix caching:** cache embeddings of common prefixes (system prompt, doc headers)

### 14.3 Observability and Monitoring

Track per-request:

- Retrieval latency; ANN search time; reranking time
- Retrieval quality (user feedback or auto-eval)
- Context utilisation: did LLM use retrieved content?
- Answer quality: accuracy, faithfulness, completeness
- **Retrieval drift:** monitor embedding distribution shift; detect when re-indexing needed
- **Coverage:** fraction of queries where retrieval found relevant content

### 14.4 Security and Privacy

- **Access control:** users should only retrieve documents they have permission to access
- **Row-level security:** filter retrieved documents by user permissions
- **Data isolation:** multi-tenant RAG; separate namespaces per customer
- **PII in retrieval:** retrieved chunks may contain sensitive information; redact before generating
- **Prompt injection:** malicious documents in corpus can inject instructions

### 14.5 RAG Frameworks

| Framework                       | Focus                                          | Language     |
| ------------------------------- | ---------------------------------------------- | ------------ |
| **LangChain**                   | Most widely used; loaders, splitters, chains   | Python       |
| **LlamaIndex**                  | Data indexing; strong RAG abstractions         | Python       |
| **Haystack** (deepset)          | Production ML pipelines; modular               | Python       |
| **DSPy** (Stanford 2023)        | Programmatic LM pipelines; prompt optimisation | Python       |
| **Semantic Kernel** (Microsoft) | Enterprise RAG; Azure integration              | .NET, Python |

---

## 15. Mathematical Analysis of RAG

### 15.1 Retrieval as Probabilistic Inference

Generative model:

$$P(y, d \mid q) = P(y \mid q, d) \cdot P(d \mid q)$$

Marginalisation:

$$P(y \mid q) = \sum_{d \in \mathcal{D}} P(y \mid q, d) \cdot P(d \mid q)$$

Approximation — sum over top-$k$ only:

$$P(y \mid q) \approx \sum_{d \in \text{TopK}(q)} P(y \mid q, d) \cdot P(d \mid q)$$

### 15.2 Retrieval Score to Probability

Convert raw similarity scores to probabilities via softmax:

$$P(d_i \mid q) = \frac{\exp(s_i / \tau)}{\sum_{j=1}^{k} \exp(s_j / \tau)}$$

- $\tau$: temperature; lower → more weight on top document; higher → more uniform

RAG with marginalisation (Lewis 2020):

$$P(y_t \mid y_{<t}, q) = \sum_{i=1}^{k} P(d_i \mid q) \cdot P(y_t \mid y_{<t}, q, d_i)$$

Requires $k$ forward passes through generator; often approximated with top-1.

### 15.3 Optimal Retrieval — Information-Theoretic View

Information gain from retrieval:

$$\text{IG}(d; y \mid q) = H(y \mid q) - H(y \mid q, d)$$

- Optimal document: maximises information gain about answer $y$
- Relevance score ≈ proxy for information gain; not exact but correlated
- Random retrieval: zero information gain on average
- Perfect retrieval: $\text{IG} = H(y|q)$ = answer uncertainty fully resolved

### 15.4 Embedding Space Geometry for Retrieval

- Good embedding space: semantically similar texts → geometrically close
- **Isotropy condition:** embeddings should be spread uniformly; not clustered in narrow cone
- **Anisotropy problem:** transformer embeddings often anisotropic; degrade cosine similarity
- BERT embeddings: highly anisotropic; popular tokens dominate space; similarity inflated
- Fix: whitening transformation; BERT-Whitening (Su et al. 2021)

### 15.5 Retrieval Augmented Generation Bias

| Bias           | Description                                       | Mitigation               |
| -------------- | ------------------------------------------------- | ------------------------ |
| **Recency**    | Newer documents more likely retrieved             | Temporal downweighting   |
| **Popularity** | Frequently referenced docs have better embeddings | Popularity normalisation |
| **Length**     | Longer docs have more matching tokens             | Length normalisation     |
| **Position**   | Start of documents better embedded                | Position-aware chunking  |

---

## 16. Common Mistakes

| Mistake                                | Why It's Wrong                                                 | Fix                                             |
| -------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------- |
| "Dense retrieval always beats BM25"    | BM25 outperforms on specific keyword-match queries             | Use hybrid retrieval; always include BM25       |
| "Larger chunks always better"          | Large chunks dilute embedding signal                           | Test chunk sizes; 256–512 tokens typical        |
| "More retrieved docs always helps"     | Too many docs confuse LLM; lost in middle                      | Test $k$; typically $k=3\text{–}10$ optimal     |
| "Cosine similarity always appropriate" | Dot product may differ for asymmetric retrieval                | Match encoder's training similarity function    |
| "RAG eliminates hallucination"         | LLM can still ignore or misinterpret retrieved content         | Measure faithfulness; add attribution           |
| "Any chunking works"                   | Poor chunking splits key info                                  | Test multiple strategies; use semantic chunking |
| "Embedding once is sufficient"         | Domain shift degrades retrieval                                | Fine-tune embeddings; re-embed periodically     |
| "Retrieval latency is negligible"      | ANN + reranking can add 500ms+                                 | Profile all components; set budgets; cache      |
| "Single-stage retrieval good enough"   | Bi-encoder errors not recoverable without reranking            | Two-stage: bi-encoder + cross-encoder           |
| "FAISS flat index scales to any size"  | $O(\lvert \mathcal{D} \rvert \times d)$ impractical beyond 10M | Use HNSW or IVFPQ for large corpora             |

---

## 17. Exercises

1. **BM25 by hand** — corpus of 3 documents; query "neural retrieval"; compute TF, IDF,
   BM25 score for each document with $k_1=1.5$, $b=0.75$; rank results.

2. **Contrastive loss** — batch of 4 (query, positive) pairs with similarity matrix;
   compute InfoNCE loss at $\tau=0.1$ and $\tau=1.0$; interpret temperature effect.

3. **HNSW parameter tuning** — for 1M vectors $d=768$: estimate memory for $M=16$ vs
   $M=32$; estimate recall@10 tradeoff; recommend for latency < 5ms.

4. **HyDE pipeline** — query: "What causes transformer training instability?"; write
   prompt for hypothetical document; describe embedding-based retrieval; explain benefit.

5. **RRF fusion** — BM25 ranks doc A at 3, doc B at 7; dense ranks A at 8, B at 2;
   compute RRF scores with $k=60$; determine final ranking.

6. **Chunking analysis** — 2000-token document; compare fixed-size (256 tokens, 64
   overlap) vs semantic (average 180 tokens, 0 overlap); compute chunks; discuss tradeoffs.

7. **NDCG calculation** — relevance scores [3, 0, 2, 1, 0] for positions 1–5; compute
   DCG@5; ideal ranking [3, 2, 1, 0, 0]; compute IDCG@5; compute NDCG@5.

8. **RAG marginalisation** — top-3 retrieved scores [2.1, 1.8, 0.9]; convert to
   probabilities with $\tau=1.0$; LLM log-probs for answer $y$: [−1.2, −2.5, −4.1];
   compute marginalised $\log P(y|q)$.

---

## 18. Why This Matters for AI (2026 Perspective)

| Aspect                      | Impact                                                                          |
| --------------------------- | ------------------------------------------------------------------------------- |
| **Knowledge freshness**     | RAG gives any LLM access to real-time information; solves knowledge cutoff      |
| **Hallucination reduction** | Grounding in docs measurably reduces factual errors; enterprise-critical        |
| **Enterprise AI**           | Most enterprise AI deployments are RAG-based; internal docs, policies, KBs      |
| **Cost**                    | Updating a doc store costs pennies; retraining a frontier model costs millions  |
| **Interpretability**        | Retrieved documents provide citations; builds trust                             |
| **Legal / compliance**      | Regulated industries require attribution; RAG enables auditable AI              |
| **Personalisation**         | Per-user doc stores enable personalised AI without shared weights               |
| **Multimodal**              | Retrieval extends to images, audio, video; active research frontier             |
| **Agents**                  | Retrieval is the primary tool for giving agents access to knowledge             |
| **Scale**                   | Billion-document retrieval enables LLMs to access all human knowledge on demand |

---

## Conceptual Bridge

RAG extends the LLM's effective knowledge beyond what fits in weights or context window.
Retrieval is a learned similarity problem in embedding space; generation is conditional
probability over retrieved context. Together they enable accurate, attributable,
updatable AI that bridges parametric and non-parametric memory.

Next: **Serving and Systems Tradeoffs** — how models are deployed at scale with
latency, throughput, and cost constraints.

```
Query → [Embedding] → ℝᵈ → [ANN Search] → Top-k Docs → [Reranker] → [LLM] → Answer + Citations
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              THIS section
    → [Serving Systems]       ← NEXT section
```

---

[← Quantization and Distillation](../11-Quantization-and-Distillation/notes.md) | [Home](../../README.md) | [Serving and Systems Tradeoffs →](../13-Serving-and-Systems-Tradeoffs/notes.md)
