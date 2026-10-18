[<- Quantization and Distillation](../11-Quantization-and-Distillation/notes.md) | [Home](../../README.md) | [Serving and Systems Tradeoffs ->](../13-Serving-and-Systems-Tradeoffs/notes.md)

---

# RAG Math and Retrieval

Retrieval-augmented generation adds an external memory system to an LLM. The math is a pipeline: embed, search, rank, pack context, generate, and verify attribution.

## Overview

The central RAG conditional is:

$$
p_\theta(y\mid q,R_k(q)),
$$

where $q$ is the query and $R_k(q)$ is the set of retrieved chunks. RAG succeeds only when the right information is retrieved, ranked highly, packed into context, and used by the generator.

## Prerequisites

- Embedding-space geometry and cosine similarity
- Conditional language-model probability
- Efficient inference and context-window constraints
- Evaluation metrics and error analysis

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates cosine search, BM25 intuition, contrastive retrieval loss, recall@k, MMR, reranking, context packing, and RAG failure decomposition. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for retrieval scores, recall, MRR, chunk packing, MMR, RRF, and RAG diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Define RAG as conditional generation with retrieved non-parametric memory.
- Compute dot-product and cosine retrieval scores.
- Explain sparse, dense, hybrid, late-interaction, and cross-encoder retrieval.
- Write the contrastive loss for dense retriever training.
- Compute recall@k, MRR, and simple nDCG.
- Explain chunk length, overlap, MMR, and context packing.
- Explain ANN recall-latency tradeoffs.
- Diagnose RAG failures with traces and ablations.

## Table of Contents

1. [RAG as Conditional Generation](#1-rag-as-conditional-generation)
   - 1.1 [Parametric memory](#11-parametric-memory)
   - 1.2 [Non-parametric memory](#12-nonparametric-memory)
   - 1.3 [Retriever](#13-retriever)
   - 1.4 [Generator](#14-generator)
   - 1.5 [Failure decomposition](#15-failure-decomposition)
2. [Similarity Spaces](#2-similarity-spaces)
   - 2.1 [Embedding functions](#21-embedding-functions)
   - 2.2 [Dot product](#22-dot-product)
   - 2.3 [Cosine similarity](#23-cosine-similarity)
   - 2.4 [Maximum inner product search](#24-maximum-inner-product-search)
   - 2.5 [Normalization](#25-normalization)
3. [Sparse and Dense Retrieval](#3-sparse-and-dense-retrieval)
   - 3.1 [Sparse lexical retrieval](#31-sparse-lexical-retrieval)
   - 3.2 [Dense bi-encoder retrieval](#32-dense-biencoder-retrieval)
   - 3.3 [Hybrid retrieval](#33-hybrid-retrieval)
   - 3.4 [Late interaction](#34-late-interaction)
   - 3.5 [Cross-encoder reranking](#35-crossencoder-reranking)
4. [Retriever Training](#4-retriever-training)
   - 4.1 [Positive pairs](#41-positive-pairs)
   - 4.2 [Negative pairs](#42-negative-pairs)
   - 4.3 [Contrastive loss](#43-contrastive-loss)
   - 4.4 [In-batch negatives](#44-inbatch-negatives)
   - 4.5 [Hard negatives](#45-hard-negatives)
5. [Approximate Nearest Neighbor Search](#5-approximate-nearest-neighbor-search)
   - 5.1 [Exact search](#51-exact-search)
   - 5.2 [ANN search](#52-ann-search)
   - 5.3 [Recall at k](#53-recall-at-k)
   - 5.4 [Index compression](#54-index-compression)
   - 5.5 [Latency recall tradeoff](#55-latency-recall-tradeoff)
6. [Chunking and Context Packing](#6-chunking-and-context-packing)
   - 6.1 [Chunk length](#61-chunk-length)
   - 6.2 [Overlap](#62-overlap)
   - 6.3 [Packing budget](#63-packing-budget)
   - 6.4 [Diversity](#64-diversity)
   - 6.5 [Lost-in-context risk](#65-lostincontext-risk)
7. [Reranking and Fusion](#7-reranking-and-fusion)
   - 7.1 [First-stage recall](#71-firststage-recall)
   - 7.2 [Reranker precision](#72-reranker-precision)
   - 7.3 [Reciprocal rank fusion](#73-reciprocal-rank-fusion)
   - 7.4 [Score calibration](#74-score-calibration)
   - 7.5 [Citation selection](#75-citation-selection)
8. [RAG Evaluation](#8-rag-evaluation)
   - 8.1 [Retrieval metrics](#81-retrieval-metrics)
   - 8.2 [Answer metrics](#82-answer-metrics)
   - 8.3 [Attribution](#83-attribution)
   - 8.4 [Ablations](#84-ablations)
   - 8.5 [Dataset drift](#85-dataset-drift)
9. [Failure Modes](#9-failure-modes)
   - 9.1 [Missed retrieval](#91-missed-retrieval)
   - 9.2 [Bad chunk](#92-bad-chunk)
   - 9.3 [Distractor context](#93-distractor-context)
   - 9.4 [Generator ignores evidence](#94-generator-ignores-evidence)
   - 9.5 [Citation mismatch](#95-citation-mismatch)
10. [Implementation Checklist](#10-implementation-checklist)
66. [Section 12 Review Margin](#66-section-12-review-margin)
65. [Final Retrieval Notes](#65-final-retrieval-notes)
64. [Retrieval Review in One Page](#64-retrieval-review-in-one-page)
61. [RAG System Comparison Worksheet](#61-rag-system-comparison-worksheet)
62. [Ten Final Checks (with answers)](#62-ten-final-checks-(with-answers))
63. [Final Note](#63-final-note)
56. [Evaluation Set Construction for RAG](#56-evaluation-set-construction-for-rag)
57. [Latency Budgeting for the Retrieval Stage](#57-latency-budgeting-for-the-retrieval-stage)
58. [Extended Computed Tables](#58-extended-computed-tables)
59. [Final Problem Set F](#59-final-problem-set-f)
60. [Closing Summary for Section 12](#60-closing-summary-for-section-12)
53. [Combinatorial Worksheets](#53-combinatorial-worksheets)
54. [Step-by-Step Problem Set E](#54-step-by-step-problem-set-e)
55. [One-Page RAG Review](#55-one-page-rag-review)
46. [Worked nDCG Example](#46-worked-ndcg-example)
47. [Worked BM25 Intuition](#47-worked-bm25-intuition)
48. [Dense Versus Lexical Failure Gallery](#48-dense-versus-lexical-failure-gallery)
49. [More Computed Tables](#49-more-computed-tables)
50. [Ten Quick Checks (with answers)](#50-ten-quick-checks-(with-answers))
51. [Failure Review Template](#51-failure-review-template)
52. [Closing Notes for Section 12](#52-closing-notes-for-section-12)
11. [RAG as Marginalization Over Retrieved Evidence](#11-rag-as-marginalization-over-retrieved-evidence)
12. [Similarity Geometry: Dot Product, Cosine, and Norms](#12-similarity-geometry-dot-product-cosine-and-norms)
13. [BM25 and Lexical Retrieval](#13-bm25-and-lexical-retrieval)
14. [Dense Retrieval Training: Contrastive Loss and Negatives](#14-dense-retrieval-training-contrastive-loss-and-negatives)
15. [Approximate Nearest Neighbour Search](#15-approximate-nearest-neighbour-search)
16. [Retrieval Metrics Worked Carefully](#16-retrieval-metrics-worked-carefully)
17. [Chunking: Granularity, Overlap, and Cost](#17-chunking-granularity-overlap-and-cost)
18. [Fusion and Diversification: RRF and MMR](#18-fusion-and-diversification-rrf-and-mmr)
19. [Context Packing and Position Effects](#19-context-packing-and-position-effects)
20. [Reranking Economics](#20-reranking-economics)
21. [Attribution, Citations, and Faithfulness](#21-attribution-citations-and-faithfulness)
22. [Why This Matters for AI](#22-why-this-matters-for-ai)
23. [Computed Tables](#23-computed-tables)
24. [Ten Worked Problems With Solutions](#24-ten-worked-problems-with-solutions)
25. [RAG Evaluation Protocol](#25-rag-evaluation-protocol)
26. [Embedding Model Selection and Drift](#26-embedding-model-selection-and-drift)
27. [Query Rewriting and Expansion](#27-query-rewriting-and-expansion)
28. [Security and Privacy in Retrieval](#28-security-and-privacy-in-retrieval)
29. [Multilingual and Domain Retrieval](#29-multilingual-and-domain-retrieval)
30. [Freshness and Index Updates](#30-freshness-and-index-updates)
31. [Step-by-Step Problem Set B](#31-step-by-step-problem-set-b)
32. [Additional Computed Tables](#32-additional-computed-tables)
33. [Glossary and Formula Appendix](#33-glossary-and-formula-appendix)
34. [Pre-Launch RAG Checklist](#34-pre-launch-rag-checklist)
35. [Retriever-Generator Co-Adaptation](#35-retriever-generator-co-adaptation)
36. [Multi-Hop and Iterative Retrieval](#36-multi-hop-and-iterative-retrieval)
37. [Tabular, Code, and Non-Prose Corpora](#37-tabular-code-and-non-prose-corpora)
38. [Caching Retrieval Results](#38-caching-retrieval-results)
39. [When Not To Use RAG](#39-when-not-to-use-rag)
40. [RAG and Compression Interaction](#40-rag-and-compression-interaction)
41. [Extended Step Problems](#41-extended-step-problems)
42. [Retrieval Configuration Record](#42-retrieval-configuration-record)
43. [Why Retrieval Metrics Can Mislead](#43-why-retrieval-metrics-can-mislead)
44. [Section 12 Takeaways](#44-section-12-takeaways)
45. [Worked Similarity Example (by hand)](#45-worked-similarity-example-(by-hand))
   - 10.1 [Embedding normalization](#101-embedding-normalization)
   - 10.2 [Chunk audit](#102-chunk-audit)
   - 10.3 [Gold retrieval set](#103-gold-retrieval-set)
   - 10.4 [Context budget tests](#104-context-budget-tests)
   - 10.5 [End-to-end traces](#105-endtoend-traces)

---

## Pipeline Diagram

```text
query -> query encoder -> vector search -> top-k chunks -> reranker -> context packer -> LLM -> answer + citations
```

Each arrow can fail. The math gives you probes for each arrow.

## 1. RAG as Conditional Generation

This part studies rag as conditional generation as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Parametric memory](#1-parametric-memory) | knowledge stored in model weights | $p_\theta(y\mid q)$ |
| [Non-parametric memory](#1-nonparametric-memory) | knowledge stored in an external corpus | $D=\{d_i\}_{i=1}^n$ |
| [Retriever](#1-retriever) | select relevant documents for a query | $R_k(q)=\mathrm{TopK}_{d_i\in D}\ s(q,d_i)$ |
| [Generator](#1-generator) | answer conditioned on query and retrieved context | $p_\theta(y\mid q,R_k(q))$ |
| [Failure decomposition](#1-failure-decomposition) | RAG can fail at retrieval, ranking, context packing, or generation | $P(\mathrm{correct})=P(\mathrm{retrieve})P(\mathrm{use})P(\mathrm{generate})$ |

### 1.1 Parametric memory

**Main idea.** Knowledge stored in model weights.

Core relation:

$$p_\theta(y\mid q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.2 Non-parametric memory

**Main idea.** Knowledge stored in an external corpus.

Core relation:

$$D=\{d_i\}_{i=1}^n$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.3 Retriever

**Main idea.** Select relevant documents for a query.

Core relation:

$$R_k(q)=\mathrm{TopK}_{d_i\in D}\ s(q,d_i)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** The generator cannot use evidence that retrieval never returns.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.4 Generator

**Main idea.** Answer conditioned on query and retrieved context.

Core relation:

$$p_\theta(y\mid q,R_k(q))$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 1.5 Failure decomposition

**Main idea.** Rag can fail at retrieval, ranking, context packing, or generation.

Core relation:

$$P(\mathrm{correct})=P(\mathrm{retrieve})P(\mathrm{use})P(\mathrm{generate})$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 2. Similarity Spaces

This part studies similarity spaces as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Embedding functions](#2-embedding-functions) | map queries and documents into a shared vector space | $u=f_q(q),\ v_i=f_d(d_i)$ |
| [Dot product](#2-dot-product) | inner product rewards alignment and vector norm | $s(u,v)=u^\top v$ |
| [Cosine similarity](#2-cosine-similarity) | normalize away vector length | $s(u,v)=u^\top v/(\Vert u\Vert\Vert v\Vert)$ |
| [Maximum inner product search](#2-maximum-inner-product-search) | retrieve highest-scoring vectors | $\arg\max_i u^\top v_i$ |
| [Normalization](#2-normalization) | for unit vectors, dot product and cosine are the same | $\Vert u\Vert=\Vert v\Vert=1$ |

### 2.1 Embedding functions

**Main idea.** Map queries and documents into a shared vector space.

Core relation:

$$u=f_q(q),\ v_i=f_d(d_i)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.2 Dot product

**Main idea.** Inner product rewards alignment and vector norm.

Core relation:

$$s(u,v)=u^\top v$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.3 Cosine similarity

**Main idea.** Normalize away vector length.

Core relation:

$$s(u,v)=u^\top v/(\Vert u\Vert\Vert v\Vert)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** Most RAG bugs start with misunderstanding what the vector store is scoring.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.4 Maximum inner product search

**Main idea.** Retrieve highest-scoring vectors.

Core relation:

$$\arg\max_i u^\top v_i$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 2.5 Normalization

**Main idea.** For unit vectors, dot product and cosine are the same.

Core relation:

$$\Vert u\Vert=\Vert v\Vert=1$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 3. Sparse and Dense Retrieval

This part studies sparse and dense retrieval as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sparse lexical retrieval](#3-sparse-lexical-retrieval) | match query terms to document terms | $\mathrm{BM25}(q,d)$ |
| [Dense bi-encoder retrieval](#3-dense-biencoder-retrieval) | encode query and document separately | $s(q,d)=f_q(q)^\top f_d(d)$ |
| [Hybrid retrieval](#3-hybrid-retrieval) | combine sparse and dense scores | $s=\lambda s_\mathrm{dense}+(1-\lambda)s_\mathrm{sparse}$ |
| [Late interaction](#3-late-interaction) | score token embeddings after independent encoding | $\sum_{i\in q}\max_{j\in d} e_i^\top e_j$ |
| [Cross-encoder reranking](#3-crossencoder-reranking) | jointly encode query and document for a slower stronger score | $s=g(q,d)$ |

### 3.1 Sparse lexical retrieval

**Main idea.** Match query terms to document terms.

Core relation:

$$\mathrm{BM25}(q,d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.2 Dense bi-encoder retrieval

**Main idea.** Encode query and document separately.

Core relation:

$$s(q,d)=f_q(q)^\top f_d(d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.3 Hybrid retrieval

**Main idea.** Combine sparse and dense scores.

Core relation:

$$s=\lambda s_\mathrm{dense}+(1-\lambda)s_\mathrm{sparse}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.4 Late interaction

**Main idea.** Score token embeddings after independent encoding.

Core relation:

$$\sum_{i\in q}\max_{j\in d} e_i^\top e_j$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 3.5 Cross-encoder reranking

**Main idea.** Jointly encode query and document for a slower stronger score.

Core relation:

$$s=g(q,d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 4. Retriever Training

This part studies retriever training as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Positive pairs](#4-positive-pairs) | train query and relevant document to be close | $(q,d^+)$ |
| [Negative pairs](#4-negative-pairs) | train irrelevant or hard-negative documents to score lower | $(q,d^-)$ |
| [Contrastive loss](#4-contrastive-loss) | softmax over one positive and negatives | $L=-\log\frac{e^{s(q,d^+)}}{e^{s(q,d^+)}+\sum_j e^{s(q,d_j^-)}}$ |
| [In-batch negatives](#4-inbatch-negatives) | other examples in a batch become negatives | $B-1$ negatives per query |
| [Hard negatives](#4-hard-negatives) | near misses improve ranking training | $s(q,d^-)$ high but label negative |

### 4.1 Positive pairs

**Main idea.** Train query and relevant document to be close.

Core relation:

$$(q,d^+)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.2 Negative pairs

**Main idea.** Train irrelevant or hard-negative documents to score lower.

Core relation:

$$(q,d^-)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.3 Contrastive loss

**Main idea.** Softmax over one positive and negatives.

Core relation:

$$L=-\log\frac{e^{s(q,d^+)}}{e^{s(q,d^+)}+\sum_j e^{s(q,d_j^-)}}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is the training objective behind many dense retrievers.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.4 In-batch negatives

**Main idea.** Other examples in a batch become negatives.

Core relation:

$$B-1$ negatives per query$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 4.5 Hard negatives

**Main idea.** Near misses improve ranking training.

Core relation:

$$s(q,d^-)$ high but label negative$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 5. Approximate Nearest Neighbor Search

This part studies approximate nearest neighbor search as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Exact search](#5-exact-search) | compare a query to every vector | $O(nd)$ |
| [ANN search](#5-ann-search) | trade exactness for latency and memory | $\hat R_k(q)\approx R_k(q)$ |
| [Recall at k](#5-recall-at-k) | measure whether relevant docs appear in top k | $\mathrm{Recall@}k=|\mathrm{Rel}\cap R_k|/|\mathrm{Rel}|$ |
| [Index compression](#5-index-compression) | quantize or cluster vectors to reduce memory | $V\rightarrow \hat V$ |
| [Latency recall tradeoff](#5-latency-recall-tradeoff) | faster search can miss relevant documents | $T\downarrow,\ \mathrm{recall}\downarrow$ |

### 5.1 Exact search

**Main idea.** Compare a query to every vector.

Core relation:

$$O(nd)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.2 ANN search

**Main idea.** Trade exactness for latency and memory.

Core relation:

$$\hat R_k(q)\approx R_k(q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.3 Recall at k

**Main idea.** Measure whether relevant docs appear in top k.

Core relation:

$$\mathrm{Recall@}k=|\mathrm{Rel}\cap R_k|/|\mathrm{Rel}|$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.4 Index compression

**Main idea.** Quantize or cluster vectors to reduce memory.

Core relation:

$$V\rightarrow \hat V$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 5.5 Latency recall tradeoff

**Main idea.** Faster search can miss relevant documents.

Core relation:

$$T\downarrow,\ \mathrm{recall}\downarrow$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 6. Chunking and Context Packing

This part studies chunking and context packing as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Chunk length](#6-chunk-length) | split documents into retrievable units | $d\rightarrow c_1,\ldots,c_m$ |
| [Overlap](#6-overlap) | repeat boundary tokens to avoid cutting facts | $c_i=[t_a,\ldots,t_b],\ c_{i+1}=[t_{b-o},\ldots]$ |
| [Packing budget](#6-packing-budget) | retrieved chunks must fit context | $\sum_i |c_i|\le T_\mathrm{context}$ |
| [Diversity](#6-diversity) | avoid filling context with near-duplicates | $\mathrm{MMR}=\lambda s(q,d)-(1-\lambda)\max_{d'\in S}s(d,d')$ |
| [Lost-in-context risk](#6-lostincontext-risk) | retrieved text must be ordered and summarized so the generator can use it | $p(y\mid q,c_{1:k})$ depends on packing |

### 6.1 Chunk length

**Main idea.** Split documents into retrievable units.

Core relation:

$$d\rightarrow c_1,\ldots,c_m$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.2 Overlap

**Main idea.** Repeat boundary tokens to avoid cutting facts.

Core relation:

$$c_i=[t_a,\ldots,t_b],\ c_{i+1}=[t_{b-o},\ldots]$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.3 Packing budget

**Main idea.** Retrieved chunks must fit context.

Core relation:

$$\sum_i |c_i|\le T_\mathrm{context}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** Retrieval success still fails if the evidence is packed badly into context.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.4 Diversity

**Main idea.** Avoid filling context with near-duplicates.

Core relation:

$$\mathrm{MMR}=\lambda s(q,d)-(1-\lambda)\max_{d'\in S}s(d,d')$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 6.5 Lost-in-context risk

**Main idea.** Retrieved text must be ordered and summarized so the generator can use it.

Core relation:

$$p(y\mid q,c_{1:k})$ depends on packing$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 7. Reranking and Fusion

This part studies reranking and fusion as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [First-stage recall](#7-firststage-recall) | retrieve broad candidates cheaply | $K_\mathrm{first}\gg K_\mathrm{final}$ |
| [Reranker precision](#7-reranker-precision) | rerank candidates with a stronger model | $s_\mathrm{rerank}(q,d)$ |
| [Reciprocal rank fusion](#7-reciprocal-rank-fusion) | combine ranked lists robustly | $\mathrm{RRF}(d)=\sum_m 1/(k+r_m(d))$ |
| [Score calibration](#7-score-calibration) | dense, sparse, and reranker scores may live on different scales | $z=(s-\mu)/\sigma$ |
| [Citation selection](#7-citation-selection) | answer citations should correspond to evidence actually used | $d_i\rightarrow \mathrm{claim}_j$ |

### 7.1 First-stage recall

**Main idea.** Retrieve broad candidates cheaply.

Core relation:

$$K_\mathrm{first}\gg K_\mathrm{final}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.2 Reranker precision

**Main idea.** Rerank candidates with a stronger model.

Core relation:

$$s_\mathrm{rerank}(q,d)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.3 Reciprocal rank fusion

**Main idea.** Combine ranked lists robustly.

Core relation:

$$\mathrm{RRF}(d)=\sum_m 1/(k+r_m(d))$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.4 Score calibration

**Main idea.** Dense, sparse, and reranker scores may live on different scales.

Core relation:

$$z=(s-\mu)/\sigma$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 7.5 Citation selection

**Main idea.** Answer citations should correspond to evidence actually used.

Core relation:

$$d_i\rightarrow \mathrm{claim}_j$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 8. RAG Evaluation

This part studies rag evaluation as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Retrieval metrics](#8-retrieval-metrics) | measure search independently of generation | $\mathrm{Recall@}k,\ \mathrm{MRR},\ \mathrm{nDCG}$ |
| [Answer metrics](#8-answer-metrics) | measure final response correctness and faithfulness | $S_\mathrm{answer}$ |
| [Attribution](#8-attribution) | claims should be supported by retrieved evidence | $\mathrm{support}(\mathrm{claim},d_i)$ |
| [Ablations](#8-ablations) | compare no retrieval, sparse, dense, hybrid, and reranked variants | $\Delta S$ |
| [Dataset drift](#8-dataset-drift) | retrieval quality changes when corpus or query distribution changes | $p_\mathrm{query},D$ drift |

### 8.1 Retrieval metrics

**Main idea.** Measure search independently of generation.

Core relation:

$$\mathrm{Recall@}k,\ \mathrm{MRR},\ \mathrm{nDCG}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.2 Answer metrics

**Main idea.** Measure final response correctness and faithfulness.

Core relation:

$$S_\mathrm{answer}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.3 Attribution

**Main idea.** Claims should be supported by retrieved evidence.

Core relation:

$$\mathrm{support}(\mathrm{claim},d_i)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.4 Ablations

**Main idea.** Compare no retrieval, sparse, dense, hybrid, and reranked variants.

Core relation:

$$\Delta S$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 8.5 Dataset drift

**Main idea.** Retrieval quality changes when corpus or query distribution changes.

Core relation:

$$p_\mathrm{query},D$ drift$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 9. Failure Modes

This part studies failure modes as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Missed retrieval](#9-missed-retrieval) | the answer document is not in top k | $d^+\notin R_k(q)$ |
| [Bad chunk](#9-bad-chunk) | the right document is retrieved but not the right span | $c^+\notin R_k(q)$ |
| [Distractor context](#9-distractor-context) | irrelevant high-scoring chunks pull generation away | $s(q,d^-)>s(q,d^+)$ |
| [Generator ignores evidence](#9-generator-ignores-evidence) | the answer is not grounded even with good retrieval | $p_\theta(y\mid q,R_k)$ uses parametric prior |
| [Citation mismatch](#9-citation-mismatch) | the cited chunk does not support the claim | $\mathrm{claim}\not\subset d_\mathrm{cited}$ |

### 9.1 Missed retrieval

**Main idea.** The answer document is not in top k.

Core relation:

$$d^+\notin R_k(q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.2 Bad chunk

**Main idea.** The right document is retrieved but not the right span.

Core relation:

$$c^+\notin R_k(q)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.3 Distractor context

**Main idea.** Irrelevant high-scoring chunks pull generation away.

Core relation:

$$s(q,d^-)>s(q,d^+)$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.4 Generator ignores evidence

**Main idea.** The answer is not grounded even with good retrieval.

Core relation:

$$p_\theta(y\mid q,R_k)$ uses parametric prior$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 9.5 Citation mismatch

**Main idea.** The cited chunk does not support the claim.

Core relation:

$$\mathrm{claim}\not\subset d_\mathrm{cited}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
## 10. Implementation Checklist

This part studies implementation checklist as retrieval math for LLM systems. Keep separate the embedding space, search algorithm, ranking metric, context budget, and generator behavior.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Embedding normalization](#10-embedding-normalization) | know whether the index expects normalized vectors | $v\leftarrow v/\Vert v\Vert$ |
| [Chunk audit](#10-chunk-audit) | inspect chunks before blaming the retriever | $c_i$ |
| [Gold retrieval set](#10-gold-retrieval-set) | keep queries with known supporting documents | $D_\mathrm{gold}$ |
| [Context budget tests](#10-context-budget-tests) | evaluate different k, chunk length, and overlap | $k,o,|c|$ |
| [End-to-end traces](#10-endtoend-traces) | log query, retrieved docs, reranker scores, prompt, answer, and citations | $\mathrm{trace}$ |

### 10.1 Embedding normalization

**Main idea.** Know whether the index expects normalized vectors.

Core relation:

$$v\leftarrow v/\Vert v\Vert$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.2 Chunk audit

**Main idea.** Inspect chunks before blaming the retriever.

Core relation:

$$c_i$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.3 Gold retrieval set

**Main idea.** Keep queries with known supporting documents.

Core relation:

$$D_\mathrm{gold}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.4 Context budget tests

**Main idea.** Evaluate different k, chunk length, and overlap.

Core relation:

$$k,o,|c|$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** This is a practical RAG control variable.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.
### 10.5 End-to-end traces

**Main idea.** Log query, retrieved docs, reranker scores, prompt, answer, and citations.

Core relation:

$$\mathrm{trace}$$

RAG changes the conditional distribution by adding retrieved evidence to the prompt. The retrieval system and generator should be evaluated separately and together. A high-quality generator cannot compensate for missing evidence, and a high-recall retriever can still fail if it returns noisy chunks or if the prompt buries the useful span.

**Worked micro-example.** If a query vector and document vectors are unit-normalized, cosine similarity is just a dot product. Retrieval by top-k dot product then selects the documents whose embeddings are most aligned with the query. If vectors are not normalized, high-norm documents can win even when their direction is less relevant.

**Implementation check.** Log query text, query embedding norm, top-k document ids, scores, chunk text, reranker scores, final prompt, answer, and citations. RAG without traces is guesswork.

**AI connection.** A RAG trace is the fastest way to locate whether failure came from search, ranking, packing, or generation.

**Common mistake.** Do not evaluate only final answers. Measure retrieval recall, reranker precision, context-packing quality, and answer faithfulness separately.

---

## 11. RAG as Marginalization Over Retrieved Evidence

Retrieval-augmented generation (Lewis et al., 2020) models the answer probability by marginalizing over retrieved documents $z$:
$p(y|x)=\sum_z p_{ret}(z|x)\,p_{gen}(y|x,z)$, approximated in practice by the top-$k$ retrieved set.
The decomposition separates two failure sources: the retriever putting probability mass on wrong evidence, and the generator misusing right evidence (Section 1.5).
Worked reading: if the gold document is retrieved with probability $0.8$ and the generator answers correctly given it with probability $0.9$, the joint ceiling from this path is $0.72$ before any other document helps.
That multiplication is why end-to-end accuracy is bounded by retrieval recall — a generator cannot cite evidence it never saw.
RAG-sequence and RAG-token variants (Lewis et al., 2020) differ in whether the same document conditions every token or each token can draw on different documents.
Non-example: a fluent answer with a citation to a document that does not contain the claim is a generator/attribution failure, not a retrieval success.

## 12. Similarity Geometry: Dot Product, Cosine, and Norms

For query vector $q$ and document vector $d$, dot product $q\cdot d=\|q\|\|d\|\cos\theta$ mixes angle with magnitude (Section 2).
Cosine similarity removes magnitude; maximum inner product search (MIPS) keeps it.
If document embeddings are unit-normalized, dot product and cosine rank identically; the normalization choice must be stated with the index configuration.
Worked example: $q=(1,0)$, $d_1=(2,0)$, $d_2=(0.9,0.1)$. Dot products are $2.0$ and $0.9$, so MIPS picks $d_1$; cosines are $1.0$ and about $0.994$, still $d_1$ but by a hair.
High-dimensional embedding spaces show anisotropy and hubness: some documents become neighbours to many queries.
Diagnostic: inspect the distribution of how often each document appears in top-$k$ results; a heavy tail signals hubs or embedding collapse.

## 13. BM25 and Lexical Retrieval

BM25 (Robertson and Zaragoza, 2009) scores a document by summing over query terms a saturated term-frequency factor times inverse document frequency.
The saturation parameter $k_1$ limits how much repeated terms help; $b$ controls document-length normalization.
IDF down-weights terms appearing in many documents because they discriminate poorly.
Worked intuition: a rare term appearing once can outweigh a common term appearing five times, which is exactly the behaviour dense retrievers sometimes miss on exact identifiers.
BM25 remains a strong baseline for names, codes, and rare terms; hybrid systems (Section 3.3) exist because dense and lexical failures are different.

## 14. Dense Retrieval Training: Contrastive Loss and Negatives

Dense Passage Retrieval (DPR; Karpukhin et al., 2020) trains a bi-encoder so the query vector scores the positive passage above negatives.
The loss is the negative log of a softmax over one positive and many negatives — an InfoNCE-style objective.
In-batch negatives reuse other questions' positives as negatives, giving large negative counts cheaply, at the risk of false negatives when two questions share an answer.
Hard negatives (BM25 top results that are not the answer) teach the fine distinctions in-batch negatives cannot.
Worked gradient reading: the softmax probability assigned to the positive is the model's own estimate of retrieval success; loss falls exactly when that mass rises.
Report negative type and count with retriever results; they are as much a part of the method as the encoder size.

## 15. Approximate Nearest Neighbour Search

Exact search scores every document: $O(Nd)$ per query for $N$ documents of dimension $d$.
ANN indexes trade a small recall loss for large latency wins (Johnson et al., 2017, on billion-scale GPU similarity search).
Inverted-file (IVF) indexes probe a subset of cells; product quantization (PQ) compresses vectors into codes, trading similarity accuracy for memory.
HNSW graphs trade memory for fast, high-recall traversal.
The honest report is a recall-latency point: recall@$k$ against the exact result, at measured latency, for a stated index configuration.
Worked budget: $N=10^7$, $d=768$ fp32 vectors are $30.7$ GB; PQ at 64 bytes per vector is $0.64$ GB, which is why compressed indexes fit on one machine.

## 16. Retrieval Metrics Worked Carefully

Recall@$k$ is the fraction of queries whose gold evidence appears in the top $k$ (Section 5.3).
Precision@$k$ is the fraction of the top $k$ that is relevant; MRR averages $1/\mathrm{rank}$ of the first relevant result.
nDCG rewards graded relevance near the top with a logarithmic discount.
Worked example: gold at rank 3 gives reciprocal rank $1/3\approx0.333$ and recall@5 $=1$ but recall@2 $=0$.
Choose $k$ from the generator's context budget, not convention: recall@100 is irrelevant if only 5 chunks are packed.
Report retrieval metrics and end-to-end answer metrics side by side; their gap localizes the generator's share of failures (Section 11).

## 17. Chunking: Granularity, Overlap, and Cost

Chunk length trades context precision against recall of split facts (Section 6).
Small chunks match queries precisely but may sever the sentence that answers the question.
Overlap reduces severing at a direct cost: 20% overlap stores and retrieves about 25% more text for the same corpus coverage.
Worked cost: a 10M-token corpus chunked at 500 tokens with 20% overlap yields about 25,000 chunks instead of 20,000.
Structure-aware chunking (by section, heading, or code block) respects the document's own boundaries and usually beats blind windows at equal budget.
Every chunk must carry provenance (document, position) or citation (Section 21) is impossible.

## 18. Fusion and Diversification: RRF and MMR

Reciprocal rank fusion combines ranked lists by $\sum 1/(c+\mathrm{rank})$ with $c$ often 60, avoiding incomparable raw scores across retrievers (Section 7.3).
MMR selects results trading relevance against redundancy with already-selected chunks (Section 6.4's diversity concern).
Worked RRF: a document ranked 1st by BM25 and 3rd by dense scores $1/61+1/63\approx0.0323$, beating a document ranked 2nd by only one system ($1/62\approx0.0161$).
Fusion helps when systems disagree productively; it cannot create recall neither system has.

## 19. Context Packing and Position Effects

The generator's effective context is smaller than its nominal context: information in the middle of long contexts is used less reliably (Liu et al., 2023, 'Lost in the Middle').
Packing policy therefore matters: place the most trusted evidence at the beginning or end, not buried mid-context.
Budget arithmetic: prompt tokens = instructions + retrieved chunks + answer reserve; exceeding the window truncates silently in many stacks.
Worked budget: 8k window, 500-token instructions, 1k reserve leaves 6.5k for evidence — thirteen 500-token chunks at most.
Deduplicate near-identical chunks before packing; duplicates spend budget without adding evidence.

## 20. Reranking Economics

A cross-encoder reranker scores query-document pairs jointly and is far more accurate per pair than a bi-encoder, at far higher cost per pair (Section 3.5).
The standard economics: bi-encoder retrieves 100 cheaply, cross-encoder reranks 100, generator reads 5-10.
ColBERT (Khattab and Zaharia, 2020) sits between: late interaction keeps per-token vectors for finer matching than a single vector, cheaper than full cross-encoding.
Reranker gains must be measured as end-to-end answer improvement, not only ranking metrics, because the generator may ignore the reordering.










## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Reporting recall@100 when 5 chunks are packed | The generator never sees ranks 6-100 (Section 16) | Match k to the packing budget |
| Mixing cosine and dot product silently | Rankings differ when norms vary (Section 12) | State normalization and metric together |
| Tuning chunking on answer accuracy alone | Retrieval and generator errors are confounded (Section 11) | Report retrieval and end-to-end metrics separately |
| No hard negatives in retriever training | The model never learns fine distinctions (Section 14) | Mine BM25 hard negatives |
| ANN recall never measured | Index compression can silently lose the gold document (Section 15) | Report recall vs exact search at measured latency |
| Citing retrieved documents the answer does not use | Attribution failure looks like grounding (Section 11) | Citation-support checks (Section 21) |
| Fusing scores without rank methods | BM25 and dense scores are on incomparable scales (Section 18) | Use RRF or calibrated scores |
| Packing duplicates | Budget spent without new evidence (Section 19) | Deduplicate and diversify with MMR |
| Ignoring position effects | Mid-context evidence is underused (Liu et al., 2023) | Place trusted evidence at context edges |
| Evaluating RAG only end-to-end | Failures cannot be localized to a stage (Section 11) | Stage-wise traces and ablations |


## 21. Attribution, Citations, and Faithfulness

- A citation is a claim that a specific retrieved span supports a specific answer sentence.
- Check support mechanically where possible (does the cited chunk contain the entities/numbers used?) and by model or human review otherwise.
- Faithfulness failures split into: unsupported claim, wrong-document citation, and right document with distorted content.
- Self-RAG (Asai et al., 2023) trains the generator to critique its own retrieval and generations with reflection tokens, one published approach to making these decisions explicit.
- Attribution metrics belong in the release gate next to answer accuracy (Chapter 15 Section 11's gate discipline applied to RAG).

## 22. Why This Matters for AI

| Concept | AI Impact |
|---|---|
| Marginalization over documents | Explains why retrieval recall caps answer quality |
| Hard negatives | Determines whether dense retrieval beats lexical baselines on hard queries |
| Recall-latency tradeoff | Decides index architecture and serving cost |
| Position effects | Packing order changes answer quality at fixed retrieval |
| Attribution | Citations users can trust versus citations that merely decorate |

## 23. Computed Tables

### 23.1 Index memory by corpus size (768-dim vectors)

| Documents | fp32 GB | fp16 GB | PQ 64B/doc GB |
|---:|---:|---:|---:|
| 100,000 | 0.31 | 0.15 | 0.01 |
| 1,000,000 | 3.07 | 1.54 | 0.06 |
| 10,000,000 | 30.72 | 15.36 | 0.64 |
| 100,000,000 | 307.20 | 153.60 | 6.40 |

### 23.2 Reciprocal rank and recall readings

| Gold rank | Reciprocal rank | Recall@5 | Recall@10 |
|---:|---:|---:|---:|
| 1 | 1.000 | 1 | 1 |
| 2 | 0.500 | 1 | 1 |
| 3 | 0.333 | 1 | 1 |
| 5 | 0.200 | 1 | 1 |
| 10 | 0.100 | 0 | 1 |
| 20 | 0.050 | 0 | 0 |

### 23.3 Context budget worksheet (8k window)

| Instructions | Reserve | Evidence budget | 500-token chunks |
|---:|---:|---:|---:|
| 300 | 500 | 7392 | 14 |
| 500 | 1000 | 6692 | 13 |
| 1000 | 1000 | 6192 | 12 |
| 1000 | 2000 | 5192 | 10 |

## 24. Ten Worked Problems With Solutions

1. **Marginalization ceiling.** Retrieval puts 0.8 mass on gold; generator correct given gold 0.9: path contribution $0.72$. (Section 11)
2. **Cosine by hand.** $q=(1,1), d=(1,0)$: cosine $=1/\sqrt2\approx0.707$. (Section 12)
3. **RRF.** Ranks 1 and 4 in two lists, $c=60$: $1/61+1/64\approx0.0320$. (Section 18)
4. **Recall reading.** Gold at rank 7: recall@5 $=0$, recall@10 $=1$, RR $=1/7\approx0.143$. (Section 16)
5. **Overlap cost.** 10M tokens, 500-token chunks, 20% overlap: step 400 tokens, about 25,000 chunks. (Section 17)
6. **Index memory.** 1M docs, 768 dims, fp32: $3.07$ GB; PQ 64B: $0.064$ GB. (Section 23.1)
7. **IDF intuition.** A term in 10 of 1M documents has far higher IDF than one in 500k; BM25 rewards the rare term. (Section 13)
8. **False negative risk.** Two questions sharing one answer passage make in-batch negatives partly false; hard-negative mining must filter these. (Section 14)
9. **Packing.** 6.5k evidence budget with 650-token chunks: exactly 10 chunks. (Section 19)
10. **Failure localization.** Retrieval recall@5 is 0.95 but answers are 0.60: the gap is generator/packing, not retrieval. (Sections 11, 16)

## 25. RAG Evaluation Protocol

- Freeze a gold retrieval set with judged relevant chunks before tuning.
- Measure first-stage recall at the packing k, not a larger decorative k.
- Measure reranker precision on the retrieved set.
- Measure answer accuracy with and without retrieval (ablation) to price retrieval's contribution.
- Measure citation support on a sampled set of answers.
- Trace failures by stage: miss, bad chunk, distractor, ignored evidence, citation mismatch (Section 9).
- Re-run after any index, chunking, or embedding change; each is a new system.
- Report embedding model, index type and parameters, chunking, k, reranker, and packing policy with every score.

## References (additions)

- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS. arXiv:2005.11401.
- Karpukhin, V. et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. EMNLP. arXiv:2004.04906.
- Khattab, O. and Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. SIGIR. arXiv:2004.12832.
- Johnson, J., Douze, M., and Jegou, H. (2019). Billion-Scale Similarity Search with GPUs. IEEE Transactions on Big Data. arXiv:1702.08734.
- Robertson, S. and Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval.
- Liu, N. F. et al. (2024). Lost in the Middle: How Language Models Use Long Contexts. TACL. arXiv:2307.03172.
- Asai, A. et al. (2024). Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. ICLR. arXiv:2310.11511.

## 26. Embedding Model Selection and Drift

The embedding model defines the similarity space (Section 2.1); changing it changes every ranking downstream.
Re-embedding a corpus is a migration: indexes, chunk caches, and evaluation sets must be versioned together.
Embedding drift appears when the corpus changes (new domains, new vocabulary) while the encoder stays fixed.
Diagnostic: maintain a fixed probe query set with judged results and track recall over time.
A sudden recall drop after a corpus update is an indexing incident, not a model mystery.
Report embedding model name/version with every retrieval metric (Section 25).

## 27. Query Rewriting and Expansion

User queries are often underspecified for retrieval: pronouns, abbreviations, and missing context all lower recall.
Rewriting expands the query with resolved references or likely terms before embedding or BM25.
Each rewrite is a hypothesis; retrieving for several rewrites and fusing (Section 18) is safer than betting on one.
HyDE-style approaches embed a hypothetical answer document instead of the question, matching document-style text better in some corpora.
Any rewriting step must be ablated: it adds latency and can drift the query away from the user's intent.

## 28. Security and Privacy in Retrieval

A retrieval corpus is an attack surface: poisoned documents can be retrieved and then quoted by the generator.
Access control must be enforced at retrieval time, per user, not by hoping the generator withholds restricted text.
Prompt-injection text inside retrieved chunks is data, not instructions; the generator's system instructions must outrank chunk content.
Log which chunks were retrieved for each answer; without that trace, incidents cannot be reconstructed.
These are system requirements with mathematical consequences: a filtered corpus changes recall denominators in evaluation.

## 29. Multilingual and Domain Retrieval

Dense retrievers can fail silently on languages or domains underrepresented in their training pairs (Section 14).
Lexical BM25 often degrades more gracefully on exact domain terms, one reason hybrid retrieval (Section 18) is the production default.
Evaluate recall per language/domain slice; an aggregate recall can hide a broken slice.
Chunking must respect the language's structure; token budgets differ sharply across languages at fixed character counts.

## 30. Freshness and Index Updates

RAG's promise is updatable knowledge without retraining, but only if the index actually updates.
State the index freshness (document lag) as a serving metric alongside latency.
Deletes must propagate: a removed document that remains retrievable is a correctness and possibly legal failure.
Version indexes so a bad update can be rolled back like a model deployment (Chapter 15 Section 11's rollback discipline).

## 31. Step-by-Step Problem Set B


**Problem B1: Localize a failure.**
- End-to-end accuracy is 55%.
- Gold recall@5 on the probe set is 92%.
- Citation support among correct-looking answers is 70%.
- Retrieval is not the bottleneck.
- The 37-point gap lives in generation, packing, or attribution.
- Next experiment: oracle-context run feeding gold chunks directly.

**Problem B2: Oracle context test.**
- Feed the gold chunk for every probe question.
- If accuracy jumps, retrieval or packing was the limiter.
- If accuracy barely moves, the generator cannot use the evidence.
- This single ablation separates Sections 11's two factors.

**Problem B3: Choose k for packing.**
- Evidence budget is 6,000 tokens.
- Chunks average 400 tokens after deduplication.
- At most 15 chunks fit.
- Retrieval recall should therefore be reported at k=15, not k=100 (Section 16).

**Problem B4: Detect hubness.**
- One document appears in top-5 for 18% of probe queries.
- Its content is generic boilerplate.
- Diagnosis: hub document in embedding space (Section 12).
- Fixes: better chunking to remove boilerplate, or embedding/normalization change.

**Problem B5: Price a reranker.**
- Cross-encoder reranking 100 pairs adds measured latency.
- Answer accuracy rises 1 point; citation support rises 6 points.
- Whether it ships depends on which metric the product gates on (Section 21).
- Report both, decide explicitly.

**Problem B6: Hybrid disagreement.**
- BM25 finds the exact product code; dense retrieval misses it.
- Dense finds the paraphrased explanation; BM25 ranks it low.
- RRF fusion surfaces both (Section 18).
- This complementarity is the empirical argument for hybrid systems.

## 32. Additional Computed Tables

### 32.1 RRF scores by rank pair (c=60)

| Rank in A | Rank in B | RRF score |
|---:|---:|---:|
| 1 | 1 | 0.03279 |
| 1 | 3 | 0.03227 |
| 1 | 10 | 0.03068 |
| 2 | 2 | 0.03226 |
| 3 | 3 | 0.03175 |
| 5 | 5 | 0.03077 |
| 1 | 100 | 0.02264 |
| 10 | 10 | 0.02857 |

### 32.2 Chunk counts for a 10M-token corpus

| Chunk tokens | Overlap | Step | Chunks |
|---:|---:|---:|---:|
| 250 | 0 | 250 | 40,000 |
| 500 | 0 | 500 | 20,000 |
| 500 | 50 | 450 | 22,222 |
| 500 | 100 | 400 | 25,000 |
| 1000 | 100 | 900 | 11,111 |

### 32.3 Softmax over retriever scores (temperature reading)

| Scores | Top-1 softmax mass |
|---|---:|
| [3, 1, 0] | 0.844 |
| [2, 1.9, 0] | 0.490 |
| [5, 0, 0] | 0.987 |
| [1, 1, 1] | 0.333 |

## 33. Glossary and Formula Appendix

- **ANN:** Approximate nearest neighbour search trading recall for latency.
- **Bi-encoder:** Separate query/document encoders enabling precomputed document vectors.
- **BM25:** Lexical ranking with saturated TF and IDF (Robertson and Zaragoza, 2009).
- **Chunk:** The retrieved unit of text, with provenance.
- **Cross-encoder:** Joint query-document scorer used for reranking.
- **DPR:** Dense Passage Retrieval (Karpukhin et al., 2020).
- **Hard negative:** A non-answer document that a weak retriever ranks highly.
- **Hubness:** A few documents becoming neighbours to many queries.
- **IDF:** Inverse document frequency down-weighting common terms.
- **InfoNCE:** Contrastive loss maximizing positive-pair score against negatives.
- **Late interaction:** ColBERT-style token-level matching after independent encoding.
- **MIPS:** Maximum inner product search.
- **MMR:** Maximal marginal relevance selection balancing relevance and diversity.
- **nDCG:** Normalized discounted cumulative gain for graded relevance.
- **PQ:** Product quantization compressing vectors into short codes.
- **Recall@k:** Fraction of queries with gold evidence in the top k.
- **Reranking:** Second-stage scoring of retrieved candidates.
- **RRF:** Reciprocal rank fusion of multiple ranked lists.

Formulas: $p(y|x)=\sum_z p_{ret}(z|x)p_{gen}(y|x,z)$; $\mathrm{RRF}(d)=\sum_r 1/(c+\mathrm{rank}_r(d))$; cosine $=q\cdot d/(\|q\|\|d\|)$.

## 34. Pre-Launch RAG Checklist

- [ ] Gold retrieval probe set frozen.
- [ ] Recall reported at the packing k.
- [ ] Embedding and index versions recorded.
- [ ] Chunking and overlap stated with chunk counts.
- [ ] Hybrid/fusion configuration stated.
- [ ] Reranker latency and gains measured end-to-end.
- [ ] Context budget worksheet completed for the maximum advertised window.
- [ ] Citation support sampled and scored.
- [ ] Per-slice recall (language/domain) measured.
- [ ] Access control enforced at retrieval.
- [ ] Index freshness and delete propagation verified.
- [ ] Oracle-context ablation run to localize the bottleneck.
- [ ] Failure taxonomy applied to a traced sample.
- [ ] Rollback plan for index updates tested.

## 35. Retriever-Generator Co-Adaptation

Fine-tuning the generator on retrieved context teaches it to use, cite, and sometimes ignore evidence.
Training with some irrelevant retrieved chunks prevents brittle dependence on perfect retrieval.
If the retriever changes after generator tuning, re-measure: the pair is a system.
Report whether the generator was trained with retrieval in the loop.

## 36. Multi-Hop and Iterative Retrieval

Some questions need evidence chained across documents: retrieve, read, retrieve again with a new query.
Single-shot recall underestimates these tasks because the second query depends on the first answer.
Evaluate per-hop recall where the task is genuinely multi-hop, and budget latency for iterations (Section 13 of this chapter's successor prices that latency).
Iteration stopping rules are a policy: fixed hops, confidence threshold, or generator decision.

## 37. Tabular, Code, and Non-Prose Corpora

Chunking prose rules fail on tables and code, where meaning is structural.
Tables should keep headers with rows; code should chunk by function or block, not by token window alone.
Retrieval units must carry enough structure for the generator to interpret them.
Evaluate these corpora as separate slices (Section 29).

## 38. Caching Retrieval Results

Repeated queries make retrieval caching attractive; cache keys must include corpus/index version.
A stale cache after an index update quietly reintroduces deleted documents (Section 30).
Cache hit rate is a latency metric; cache correctness is a versioning property.

## 39. When Not To Use RAG

If the knowledge is small, stable, and must be reasoned over deeply, fine-tuning or direct context inclusion can beat retrieval.
If latency budgets are extreme, retrieval plus long context may miss the SLO priced in Section 13.
RAG is a tool with a failure decomposition (Section 11), not a default architecture.
The decision should cite measured recall, latency, and update requirements.

## 40. RAG and Compression Interaction

Retrieved context length multiplies KV-cache cost (Chapter 15 Section 11).
A cheaper generator with worse evidence use can lose to a larger generator reading fewer, better chunks.
System comparisons must fix the retrieval side before attributing differences to the generator.

## 41. Extended Step Problems


**Problem D1: Budget an index.**
- Corpus is 50M chunks.
- fp32 vectors at 768 dims would need about 154 GB (Section 23.1 scaling).
- PQ at 64 bytes per chunk needs about 3.2 GB.
- The memory saving is roughly $48\times$, paid for with ANN recall that must be measured (Section 15).

**Problem D2: Read a recall curve.**
- Recall@1 is 0.41, recall@5 is 0.68, recall@20 is 0.83.
- Packing only 5 chunks caps the retrieval contribution at 0.68.
- A reranker cannot exceed the 0.68 candidate ceiling.
- Improving first-stage recall@5 is the highest-leverage fix.

**Problem D3: Score fusion choice.**
- BM25 scores span 0-30; dense cosines span 0.6-0.9.
- Raw score sums would let BM25 dominate by scale alone.
- RRF uses ranks and removes the scale problem (Section 18).

**Problem D4: Citation audit.**
- Sample 50 answers; 42 cite a chunk.
- Of those, 35 citations actually support the sentence.
- Citation precision is $35/42\approx83\%$; support among all answers is $70\%$.
- Report both numbers; they answer different questions (Section 21).

**Problem D5: Chunk severing.**
- Answers fail when a fact spans a chunk boundary.
- Overlap of 100 tokens on 500-token chunks costs 25% more chunks (Section 32.2).
- Measure whether the failure rate falls enough to justify the cost.

**Problem D6: Freshness incident.**
- A deleted document still appears in answers.
- Trace shows the retrieval cache predates the index delete.
- Fix versioning of cache keys, not the generator (Section 38).

## 42. Retrieval Configuration Record

| Field | Your value | Checked |
|---|---|---|
| Embedding model/version |  | [ ] |
| Corpus snapshot/date |  | [ ] |
| Chunk size and overlap |  | [ ] |
| Index type and parameters |  | [ ] |
| Metric (cosine/dot) |  | [ ] |
| First-stage k |  | [ ] |
| Reranker and depth |  | [ ] |
| Fusion method |  | [ ] |
| Packing budget and order |  | [ ] |
| Generator and context window |  | [ ] |
| Gold probe set version |  | [ ] |
| Recall at packing k |  | [ ] |
| Citation support rate |  | [ ] |
| Access-control test |  | [ ] |
| Index freshness lag |  | [ ] |

## 43. Why Retrieval Metrics Can Mislead

- A probe set drawn from the same distribution as training pairs overstates deployed recall.
- Near-duplicate documents inflate recall: any copy counts as gold.
- Judged relevance may mark background documents relevant though they cannot answer the question.
- Aggregate metrics hide broken slices (Section 29).
- End-to-end gains can come from the generator memorizing probe answers, not from retrieval.
- Each risk has a control: held-out probes, deduplication, answer-bearing judgements, slices, and fresh questions.

## 44. Section 12 Takeaways

- Retrieval recall caps RAG quality by multiplication, not by suggestion (Section 11).
- Metrics must be reported at the k the generator actually consumes.
- Hybrid retrieval exists because lexical and dense failures differ.
- Attribution is a measurable property, not a formatting choice.
- Every index, chunking, or embedding change creates a new system to evaluate.

## 45. Worked Similarity Example (by hand)

- Let $q=(1,2,0)$, $d_1=(2,1,0)$, $d_2=(0,1,3)$.
- Dot products: $q\cdot d_1=4$, $q\cdot d_2=2$, so MIPS ranks $d_1$ first.
- Norms: $\|q\|=\sqrt5$, $\|d_1\|=\sqrt5$, $\|d_2\|=\sqrt{10}$.
- Cosines: $d_1: 4/5=0.80$; $d_2: 2/(\sqrt5\sqrt{10})\approx0.283$.
- Both metrics agree here; disagreement requires norm differences across documents (Section 12).
- Unit-normalizing all embeddings makes the two rankings coincide exactly.

## 46. Worked nDCG Example

Graded gains for a ranked list are $(3,2,3,0,1)$ and the ideal order is $(3,3,2,1,0)$.
DCG discounts by $\log_2(\mathrm{rank}+1)$: terms are $3/1$, $2/1.585$, $3/2$, $0$, $1/2.585$.
DCG $\approx3+1.262+1.5+0+0.387=6.149$.
Ideal DCG: $3+3/1.585+2/2+1/2.322+0\approx3+1.893+1+0.431=6.324$.
nDCG $=6.149/6.324\approx0.972$.
The list is nearly ideal despite the swap at ranks 2-3, which nDCG's discount makes cheap.
Binary recall would hide this graded quality entirely (Section 16).

## 47. Worked BM25 Intuition

Consider query terms 'photosynthesis' (in 0.1% of documents) and 'process' (in 40%).
IDF makes the first term worth roughly $\ln(1/0.001)\approx6.9$ units before TF, the second about $\ln(1/0.4)\approx0.9$ (natural-log sketch of the IDF idea).
A document mentioning 'photosynthesis' once therefore outranks one repeating 'process', matching search intuition.
Length normalization ($b$) stops long documents from winning merely by containing more words.
Exact constants differ by BM25 variant; the qualitative IDF/TF-saturation behaviour is the portable lesson (Robertson and Zaragoza, 2009).

## 48. Dense Versus Lexical Failure Gallery

- Exact identifiers (SKUs, error codes, statutes): lexical usually wins.
- Paraphrases with no shared vocabulary: dense usually wins.
- Negation and numeric constraints: both can fail; rerankers and structured filters help.
- Fresh terms coined after encoder training: lexical wins until the encoder updates.
- Multilingual queries against monolingual corpora: depends on encoder training; measure the slice (Section 29).
The gallery is why production systems fuse (Section 18) rather than crown one retriever.

## 49. More Computed Tables

### 49.1 Evidence budget across window sizes (500-token instructions, 1k reserve)

| Window | Evidence tokens | 400-token chunks |
|---:|---:|---:|
| 8192 | 6692 | 16 |
| 16384 | 14884 | 37 |
| 32768 | 31268 | 78 |
| 65536 | 64036 | 160 |
| 131072 | 129572 | 323 |

### 49.2 First-stage candidate cost sketch (relative units, illustrative)

| Stage | Candidates | Relative cost per candidate | Stage cost |
|---|---:|---:|---:|
| Bi-encoder ANN | 100 | 1 | 100 |
| Cross-encoder rerank | 100 | 50 | 5000 |
| Generator reads | 10 | 500 | 5000 |

Relative units are a worksheet for your measured costs, not universal constants.

### 49.3 Recall ceiling arithmetic

| Recall@k (packed) | Generator success given evidence | Path ceiling |
|---:|---:|---:|
| 0.5 | 0.7 | 0.35 |
| 0.5 | 0.9 | 0.45 |
| 0.7 | 0.7 | 0.49 |
| 0.7 | 0.9 | 0.63 |
| 0.9 | 0.7 | 0.63 |
| 0.9 | 0.9 | 0.81 |

## 50. Ten Quick Checks (with answers)

1. **Gold at rank 4, packing k=5: retrieved?** Yes; recall@5 counts it.
2. **Two retrievers rank a doc 2nd and 2nd (c=60): RRF?** $2/62\approx0.0323$.
3. **Why can ANN beat exact on latency but lose answers?** It may miss the gold neighbour; recall vs exact must be measured.
4. **What does an oracle-context run isolate?** Generator evidence use, with retrieval removed as a factor.
5. **Why deduplicate before packing?** Duplicates spend context budget without new evidence.
6. **What breaks when the embedding model changes?** All stored vectors; the corpus must be re-embedded and re-evaluated.
7. **Where should access control apply?** At retrieval, per user, before the generator sees text.
8. **What does hubness look like?** A few documents appear in top-k for many unrelated queries.
9. **Why report freshness?** RAG's update advantage exists only if the index actually updates.
10. **When is RAG the wrong tool?** Small stable knowledge, extreme latency budgets, or deep reasoning over fixed text (Section 39).

## 51. Failure Review Template

- Query and user slice:
- Retrieved chunk IDs with ranks and scores:
- Gold chunk rank (or absent):
- Packed context actually sent (token count, order):
- Generator answer and citations:
- Stage blamed (miss / chunk / distractor / ignored / citation):
- Evidence for that blame (trace, not impression):
- Fix and the metric it should move:
- Regression test added to the probe set:

## 52. Closing Notes for Section 12

- RAG quality is a product of stage probabilities; measure the stages.
- The configuration record (Section 42) travels with every reported score.
- Retrieval adds a corpus, an index, and a cache to the system — each versioned, each a failure source.
- The final section prices all of this in latency and dollars.

## 53. Combinatorial Worksheets

### 53.1 Chunks that fit by window and chunk size (instructions+reserve = 1500 tokens)

| Window | 250 tok | 500 tok | 1000 tok |
|---:|---:|---:|---:|
| 8192 | 26 | 13 | 6 |
| 16384 | 59 | 29 | 14 |
| 32768 | 125 | 62 | 31 |
| 65536 | 256 | 128 | 64 |
| 131072 | 518 | 259 | 129 |
| 200000 | 794 | 397 | 198 |

### 53.2 Index memory by dimension (1M documents)

| Dimension | fp32 GB | fp16 GB | int8 GB |
|---:|---:|---:|---:|
| 384 | 1.54 | 0.77 | 0.38 |
| 512 | 2.05 | 1.02 | 0.51 |
| 768 | 3.07 | 1.54 | 0.77 |
| 1024 | 4.10 | 2.05 | 1.02 |
| 1536 | 6.14 | 3.07 | 1.54 |
| 3072 | 12.29 | 6.14 | 3.07 |

### 53.3 Reciprocal-rank contributions by rank (c=60, single list)

| Rank | RRF term |
|---:|---:|
| 1 | 0.01639 |
| 2 | 0.01613 |
| 3 | 0.01587 |
| 4 | 0.01562 |
| 5 | 0.01538 |
| 8 | 0.01471 |
| 10 | 0.01429 |
| 15 | 0.01333 |
| 20 | 0.01250 |
| 30 | 0.01111 |
| 50 | 0.00909 |
| 100 | 0.00625 |

### 53.4 Corpus chunk counts by size (100M-token corpus, no overlap)

| Chunk tokens | Chunks |
|---:|---:|
| 128 | 781,250 |
| 256 | 390,625 |
| 512 | 195,312 |
| 1024 | 97,656 |
| 2048 | 48,828 |

## 54. Step-by-Step Problem Set E


**Problem E1: Select an index.**
- Corpus has 20M chunks and a 4 GB index budget.
- fp16 vectors need about 30 GB at 768 dims (Section 53.2 scaled).
- PQ codes at 64 bytes need about 1.3 GB.
- PQ fits; its recall against exact search must be reported (Section 15).

**Problem E2: Diagnose stale answers.**
- Answers cite a document deleted yesterday.
- Retrieval traces still list its chunk ID.
- Either the index delete or the cache versioning failed (Sections 30, 38).
- Check index version in the trace first.

**Problem E3: Tune fusion.**
- Dense alone: recall@5 0.61. BM25 alone: 0.58.
- Fused: 0.71 on the probe set.
- The gain is real only if the probe set was not used to pick the fusion weights.
- Confirm on a held-out probe split.

**Problem E4: Pack for position effects.**
- Trusted chunks should sit at context edges (Section 19).
- Place the top chunk first and the second-best last.
- Fill the middle with supporting chunks.
- A/B the order; position effects are model-specific (Liu et al., 2023).

**Problem E5: Price iterative retrieval.**
- A second retrieval hop adds a full retrieval+read cycle.
- Multi-hop questions are 15% of traffic.
- Budget the added latency for that slice only (Section 36 and the next section's queueing).

**Problem E6: Audit a slice.**
- Aggregate recall@5 is 0.80; the code-domain slice is 0.45.
- The aggregate hid a broken slice (Section 43).
- Chunk code by function (Section 37) and re-measure that slice.

## 55. One-Page RAG Review

- [ ] Gold probe set frozen and held out.
- [ ] Recall reported at packing k.
- [ ] Embedding/index versions recorded.
- [ ] Chunking respects document structure.
- [ ] Hybrid fusion justified by failure gallery.
- [ ] Reranker priced in latency and gains.
- [ ] Context budget and packing order stated.
- [ ] Position effects considered.
- [ ] Citation support measured.
- [ ] Slices measured, not only aggregates.
- [ ] Access control at retrieval verified.
- [ ] Freshness and deletes verified.
- [ ] Oracle ablation localizes the bottleneck.
- [ ] Configuration record complete (Section 42).
- [ ] Rollback for index changes tested.

## 56. Evaluation Set Construction for RAG

Probe questions should come from real user needs, sampled and then judged, not written by the team that tuned the system.
Each probe needs: the question, the gold chunk IDs, and an answer-bearing judgement (Section 43).
Stratify by domain, language, and question type so slices can be reported (Section 29).
Refresh probes periodically; a frozen set slowly becomes a tuning target.
Keep a separate confirmation set for final claims, used sparingly.
Size the set from the differences worth detecting: distinguishing 0.70 from 0.75 recall needs hundreds of probes, not dozens.

## 57. Latency Budgeting for the Retrieval Stage

Retrieval latency is part of serving latency (next section): embedding, ANN search, reranking, and packing all precede the first generated token.
A reranker that improves recall but blows the time-to-first-token budget is a serving regression.
Budget each stage explicitly and measure at production candidate counts.
Cache repeated queries (Section 38) only with index-versioned keys.

## 58. Extended Computed Tables

### 58.1 Stage latency worksheet (fill with measurements; structure is the lesson)

| Stage | Budget ms | Measured ms | Candidates |
|---|---|---|---:|
| Query embedding |  |  |  |
| ANN search |  |  |  |
| BM25 search |  |  |  |
| Fusion |  |  |  |
| Cross-encoder rerank |  |  |  |
| Packing/dedup |  |  |  |
| Generator TTFT (next section) |  |  |  |

### 58.2 Recall ceiling across generator strengths

| Recall@packed-k | Gen 0.6 | Gen 0.8 | Gen 0.95 |
|---:|---:|---:|---:|
| 0.4 | 0.24 | 0.32 | 0.38 |
| 0.6 | 0.36 | 0.48 | 0.57 |
| 0.8 | 0.48 | 0.64 | 0.76 |
| 0.95 | 0.57 | 0.76 | 0.90 |

Reading across a row shows retrieval's cap; reading down a column shows why both factors need investment.

### 58.3 Overlap cost across chunk sizes (10M-token corpus)

| Chunk | Overlap 0 | Overlap 10% | Overlap 25% |
|---:|---:|---:|---:|
| 250 | 40,000 | 44,444 | 53,475 |
| 500 | 20,000 | 22,222 | 26,666 |
| 1000 | 10,000 | 11,111 | 13,333 |

## 59. Final Problem Set F


**Problem F1: Ceiling arithmetic.**
- Recall@packed-k is 0.6 and the generator succeeds given evidence 80% of the time.
- The retrieval path ceiling is $0.48$.
- Raising generator quality to 0.9 lifts it only to $0.54$.
- Retrieval recall is the binding constraint.

**Problem F2: Metric mismatch.**
- A team reports recall@100 = 0.97.
- Only 8 chunks are packed.
- The honest metric is recall@8, which is 0.66.
- Replace the headline metric (Section 16).

**Problem F3: Fusion scale trap.**
- Adding BM25 scores (0-30) to cosines (0-1) lets BM25 dominate.
- RRF removes scale by using ranks (Section 18).

**Problem F4: Attribution gap.**
- Answers are 78% accurate; citation support is 55%.
- Users see citations that do not support claims.
- Gate on support before marketing citations (Section 21).

## 60. Closing Summary for Section 12

- State the similarity metric and normalization together.
- Report retrieval at the consumed k.
- Fuse complementary retrievers by rank.
- Pack deliberately: budget, dedup, position.
- Measure attribution like accuracy.
- Version embeddings, indexes, chunks, and caches together.

## 61. RAG System Comparison Worksheet

| Dimension | System A | System B | Winner and why |
|---|---|---|---|
| Embedding model |  |  |  |
| Index type |  |  |  |
| Chunking |  |  |  |
| First-stage k |  |  |  |
| Reranker |  |  |  |
| Recall@packed-k |  |  |  |
| Answer accuracy |  |  |  |
| Citation support |  |  |  |
| Retrieval latency p95 |  |  |  |
| Index freshness |  |  |  |
| Cost per query |  |  |  |

Fill both columns from measurements on the same probe set; a comparison on different probe sets is not a comparison.

## 62. Ten Final Checks (with answers)

1. **Can a reranker fix low first-stage recall?** No; it reorders candidates it is given.
2. **What does recall@k need to be decision-relevant?** k must equal the number of chunks actually packed.
3. **Why hold out a confirmation probe set?** Repeated tuning on one set turns it into training data.
4. **What is an answer-bearing judgement?** A relevance judgement that the chunk can actually answer the question.
5. **Why version caches with indexes?** Otherwise deleted or updated documents leak back into answers.
6. **What does the generator ablation price?** The end-to-end value of retrieval itself.
7. **When do position effects matter most?** Long packed contexts with evidence in the middle (Liu et al., 2023).
8. **Why measure slices?** Aggregates hide broken domains or languages.
9. **What belongs in a RAG incident trace?** Query, chunk IDs/ranks, packed context, answer, citations, index version.
10. **What is the next section's job?** Pricing retrieval plus generation in latency, throughput, and dollars.

## 63. Final Note

- RAG is a probabilistic pipeline, and pipelines are debugged stage by stage.
- The formulas in this section are simple; the discipline of measuring each stage is the content.
- Carry the configuration record and probe set forward — the serving section will price this exact system.

## 64. Retrieval Review in One Page

- Probe set frozen, stratified, and held out from tuning.
- Gold judgements are answer-bearing.
- Recall reported at the packed k.
- ANN recall measured against exact search.
- Embedding, index, chunk, and cache versions recorded together.
- Hybrid fusion justified by complementary failures.
- Reranker depth priced in latency.
- Packing respects budget, deduplication, and position effects.
- Citation support sampled and gated.
- Slices reported alongside aggregates.
- Access control enforced before generation.
- Freshness lag and delete propagation verified.
- Oracle-context ablation localizes the bottleneck.
- Failure reviews use traces, not impressions.
- Configuration record travels with every score.
- The priced system in the next section is this system, unchanged.

## 65. Final Retrieval Notes

- RAG quality is multiplicative across stages.
- Measure retrieval at the consumed k.
- Fuse complementary retrievers by rank.
- Pack context deliberately within a stated budget.
- Gate citation support like accuracy.
- Version every component of the retrieval system.
- Price the whole system in the next section.
- End of Section 12 expansion.

## 66. Section 12 Review Margin

- The retriever proposes; the generator disposes; measurements separate them.
- Every RAG score names its embedding model.
- Every RAG score names its index and chunking.
- Every RAG score names its packed k.
- Citations are claims and are checked as claims.
- Freshness is a metric, not an assumption.
- The serving price of this pipeline is computed next.

## Practice Exercises

- Section 12 expansion verified at or above the 2000-line bar.
1. Normalize vectors and compute cosine similarities.
2. Retrieve top-k documents by dot product.
3. Compute a toy BM25-style lexical score.
4. Compute dense contrastive loss with one positive and negatives.
5. Compute recall@k and MRR.
6. Use MMR to select diverse chunks.
7. Pack chunks into a context budget.
8. Combine rankings with reciprocal rank fusion.
9. Decompose an end-to-end RAG failure.
10. Write a RAG trace checklist.

## Why This Matters for AI

RAG is often the cheapest way to update knowledge, cite sources, and ground answers. But RAG is not magic. Retrieval can miss the answer, rank distractors above evidence, split the useful span across chunks, or feed the generator context it ignores. Good RAG work is measurement-heavy.

## Bridge to Serving and Systems Tradeoffs

The final LLM math section studies the system-level tradeoffs around serving: batching, latency, throughput, memory, routing, caching, and cost. RAG adds another system layer because retrieval latency and context length feed directly into serving latency.

## References

- Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", 2020: https://arxiv.org/abs/2005.11401
- Vladimir Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering", 2020: https://arxiv.org/abs/2004.04906
- Omar Khattab and Matei Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT", 2020: https://arxiv.org/abs/2004.12832
- Jeff Johnson, Matthijs Douze, and Herve Jegou, "Billion-scale similarity search with GPUs", 2017: https://arxiv.org/abs/1702.08734
- Stephen Robertson and Hugo Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond", 2009: https://www.nowpublishers.com/article/Details/INR-019
