[Home](../../README.md) | [Embedding Space Math →](../02-Embedding-Space-Math/notes.md)

---

# Tokenization Math

> _"A language model never sees text — it sees integer sequences produced by a tokenizer. The math behind that mapping determines what the model can and cannot learn."_

## Overview

Tokenization is the mathematical bridge between raw text and the integer sequences that LLMs process. This section covers the three dominant tokenization algorithms (BPE, Unigram, WordPiece), the information-theoretic foundations behind them, and the practical implications for model performance, cost, and multilingual fairness. Every formula is grounded in how it affects real LLM training and inference.

## Prerequisites

- Basic probability (conditional probability, Bayes' rule)
- Information theory fundamentals (entropy, bits)
- Dynamic programming basics
- Familiarity with Python and NumPy

## Companion Notebooks

| Notebook                           | Description                                                                      |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Full implementations of BPE, Unigram EM, WordPiece, entropy analysis, Viterbi DP |
| [exercises.ipynb](exercises.ipynb) | Practice problems: manual BPE, compression, fertility, Viterbi                   |

## Learning Objectives

After completing this section, you will:

- Implement BPE (character-level and word-level) from scratch and understand its greedy merge criterion
- Derive the Unigram Language Model's EM training (forward-backward, E-step, M-step, convergence guarantees)
- Explain WordPiece's PMI-based merge criterion and how it differs from BPE
- Calculate compression ratio, context window efficiency, and vocabulary parameter costs
- Apply the Viterbi algorithm for optimal tokenization and enumerate segmentation complexity
- Analyze tokenizer quality using Shannon entropy, Rényi entropy, and Zipf's law
- Quantify the multilingual "tokenization tax" and its impact on fairness and cost

## Table of Contents

- [Tokenization Math](#tokenization-math)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [What Is Tokenization?](#what-is-tokenization)
    - [Why Math Matters Here](#why-math-matters-here)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 Alphabet and Strings](#21-alphabet-and-strings)
    - [2.2 Vocabulary](#22-vocabulary)
    - [2.3 Tokenization Function](#23-tokenization-function)
    - [2.4 Token-to-Integer Mapping](#24-token-to-integer-mapping)
  - [3. Mathematical Formulation](#3-mathematical-formulation)
    - [3.1 Byte Pair Encoding (BPE)](#31-byte-pair-encoding-bpe)
      - [Algorithm](#algorithm)
      - [Frequency-Based Merge Selection](#frequency-based-merge-selection)
      - [Compression Analysis](#compression-analysis)
      - [Complexity](#complexity)
      - [Worked Example](#worked-example)
    - [3.2 Unigram Language Model (SentencePiece)](#32-unigram-language-model-sentencepiece)
      - [Probabilistic Framework](#probabilistic-framework)
      - [Optimal Segmentation](#optimal-segmentation)
      - [EM Training](#em-training)
      - [Loss of Removing a Token](#loss-of-removing-a-token)
    - [3.3 WordPiece (BERT)](#33-wordpiece-bert)
      - [Algorithm](#algorithm-1)
      - [PMI-Based Merge Selection](#pmi-based-merge-selection)
      - [Encoding with WordPiece](#encoding-with-wordpiece)
    - [3.4 Comparison](#34-comparison)
  - [4. AI/ML Applications](#4-aiml-applications)
    - [4.1 Context Window Efficiency](#41-context-window-efficiency)
    - [4.2 Vocabulary Size and Model Dimensions](#42-vocabulary-size-and-model-dimensions)
    - [4.3 Tokenization and Arithmetic](#43-tokenization-and-arithmetic)
      - [Why This Breaks Arithmetic](#why-this-breaks-arithmetic)
      - [The Inconsistency Problem](#the-inconsistency-problem)
    - [4.4 Tokenization Fertility](#44-tokenization-fertility)
    - [4.5 Special Tokens](#45-special-tokens)
      - [How Special Tokens Affect the Pipeline](#how-special-tokens-affect-the-pipeline)
  - [5. Information-Theoretic View](#5-information-theoretic-view)
    - [5.1 Tokenization as Compression](#51-tokenization-as-compression)
      - [Bits Per Character Analysis](#bits-per-character-analysis)
    - [5.2 Entropy of the Token Distribution](#52-entropy-of-the-token-distribution)
      - [Worked Example](#worked-example-1)
      - [Zipf's Law in Token Distributions](#zipfs-law-in-token-distributions)
    - [5.3 Rényi Entropy and Vocabulary Balance](#53-rényi-entropy-and-vocabulary-balance)
      - [Special Cases](#special-cases)
      - [Ordering and the Gap](#ordering-and-the-gap)
      - [Example](#example)
  - [6. Dynamic Programming: Optimal Segmentation](#6-dynamic-programming-optimal-segmentation)
    - [6.1 The Segmentation Problem](#61-the-segmentation-problem)
    - [6.2 Viterbi Algorithm for Tokenization](#62-viterbi-algorithm-for-tokenization)
      - [Worked Example](#worked-example-2)
    - [6.3 Number of Possible Segmentations](#63-number-of-possible-segmentations)
      - [Derivation (L = 2 case)](#derivation-l--2-case)
      - [General Case (max token length L)](#general-case-max-token-length-l)
      - [Growth Table](#growth-table)
  - [7. Common Mistakes](#7-common-mistakes)
  - [8. Exercises](#8-exercises)
    - [Exercise 1: Manual BPE (Pen and Paper)](#exercise-1-manual-bpe-pen-and-paper)
    - [Exercise 2: Compression Ratio](#exercise-2-compression-ratio)
    - [Exercise 3: Vocabulary Size Tradeoff](#exercise-3-vocabulary-size-tradeoff)
    - [Exercise 4: Fertility Analysis](#exercise-4-fertility-analysis)
    - [Exercise 5: Viterbi Segmentation](#exercise-5-viterbi-segmentation)
  - [9. Why This Matters for AI](#9-why-this-matters-for-ai)
    - [The Tokenization Tax](#the-tokenization-tax)
      - [Concrete Cost Calculation](#concrete-cost-calculation)
      - [Training Efficiency Impact](#training-efficiency-impact)
  - [10. Further Reading](#10-further-reading)
    - [Papers](#papers)
    - [Implementations](#implementations)
    - [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### What Is Tokenization?

Tokenization converts raw text into a sequence of integers that a neural network can process. It is the **first and last mathematical transformation** in every LLM pipeline:

```
"The cat sat" → tokenizer → [464, 3797, 3332] → model → [next_token_id] → detokenizer → "on"
```

Every character you read, every word GPT generates, passes through this mapping. The quality of this mapping directly affects:

- **Vocabulary efficiency**: How many tokens to represent a concept
- **Compression ratio**: How much text fits in a fixed context window
- **Cross-lingual ability**: Whether the model can handle multiple languages
- **Arithmetic ability**: Whether "123" is one token or three
- **Cost**: Tokens are what you pay for in API calls

### Why Math Matters Here

Tokenization seems like an engineering detail, but it is deeply mathematical:

| Mathematical Concept    | Tokenization Application           |
| ----------------------- | ---------------------------------- |
| Information theory      | Optimal compression bounds         |
| Probability / frequency | BPE merge decisions                |
| Graph theory            | Vocabulary as a directed graph     |
| Optimization            | Unigram model's EM algorithm       |
| Combinatorics           | Possible segmentations of a string |
| Linear algebra          | Token embeddings live in ℝᵈ        |

---

## 2. Formal Definitions

### 2.1 Alphabet and Strings

**Alphabet** Σ: A finite set of atomic symbols (typically UTF-8 bytes or Unicode codepoints).

$$\Sigma = \{a, b, c, \ldots\} \quad |\Sigma| = 256 \text{ (for byte-level)}$$

**String**: A finite sequence of symbols from Σ.

$$s = \sigma_1 \sigma_2 \cdots \sigma_n, \quad \sigma_i \in \Sigma$$

**Σ\***: The set of all finite strings over Σ (the Kleene closure).

### 2.2 Vocabulary

A **vocabulary** V is a finite set of strings (subwords) drawn from Σ\*:

$$V \subset \Sigma^*, \quad |V| = N \quad \text{(typically 32k–128k)}$$

Requirements:

- **Coverage**: Σ ⊆ V (every individual byte/char is in V, guaranteeing any string can be tokenized)
- **Finiteness**: |V| = N is fixed at training time
- **Prefix-freeness** (approximate): Ideally, no token is a prefix of another (not strictly enforced, but BPE merge order provides deterministic disambiguation)

Typical vocabulary sizes across real models:

| Model     | V       | Algorithm | Notes                            |
| --------- | ------- | --------- | -------------------------------- |
| GPT-2     | 50,257  | BPE       | Byte-level, no `<UNK>` needed    |
| BERT      | 30,522  | WordPiece | Character-level with `##` prefix |
| LLaMA-1/2 | 32,000  | BPE       | SentencePiece with byte fallback |
| LLaMA-3   | 128,000 | BPE       | 4× larger for multilingual       |
| GPT-4     | 100,277 | BPE       | cl100k_base encoding             |
| T5        | 32,100  | Unigram   | SentencePiece unigram mode       |

The **coverage guarantee** (Σ ⊆ V) is critical: because every byte is in V, any valid UTF-8 sequence can always be encoded — worst case, one byte per token. Byte-level BPE (GPT-2 onward) eliminated the need for `<UNK>` tokens entirely.

### 2.3 Tokenization Function

A **tokenizer** is a function that maps a string to a sequence of vocabulary tokens:

$$T: \Sigma^* \to V^*$$
$$T(s) = (t_1, t_2, \ldots, t_k) \quad \text{where } t_i \in V, \quad t_1 \circ t_2 \circ \cdots \circ t_k = s$$

where ∘ denotes string concatenation. The key constraint: **concatenating the tokens reconstructs the original string** (lossless).

### 2.4 Token-to-Integer Mapping

Each token in V is assigned a unique integer ID:

$$\text{encode}: V \to \{0, 1, \ldots, N-1\}$$
$$\text{decode}: \{0, 1, \ldots, N-1\} \to V$$

These are bijections. The full pipeline is:

$$\text{text} \xrightarrow{T} \text{tokens} \xrightarrow{\text{encode}} \text{integer IDs} \xrightarrow{\text{embedding}} \mathbb{R}^d$$

---

## 3. Mathematical Formulation

### 3.1 Byte Pair Encoding (BPE)

BPE is the dominant tokenization algorithm (used by GPT, LLaMA, Mistral). It is a **greedy compression algorithm** rooted in information theory.

#### Algorithm

```
Input:  corpus C, desired vocabulary size N
Output: vocabulary V, merge rules M

1. Initialize V = set of all individual bytes in C
2. While |V| < N:
   a. Count frequency of every adjacent pair (tᵢ, tᵢ₊₁) in tokenized corpus
   b. Find most frequent pair: (a, b) = argmax_{(x,y)} count(x, y)
   c. Create new token: t_new = concat(a, b)
   d. Add t_new to V
   e. Replace all occurrences of (a, b) with t_new in corpus
   f. Record merge rule: (a, b) → t_new
3. Return V, M
```

#### Frequency-Based Merge Selection

At each step, the pair with maximum frequency is merged:

$$(a^*, b^*) = \arg\max_{(a,b) \in V \times V} \sum_{i=1}^{|C|-1} \mathbb{1}[c_i = a \land c_{i+1} = b]$$

where C = c₁c₂...c\_|C| is the current tokenization of the corpus.

#### Compression Analysis

Each merge reduces the total token count. If pair (a, b) has frequency f:

$$|C_{\text{after}}| = |C_{\text{before}}| - f$$

The **compression ratio** after all merges:

$$\rho = \frac{\text{length in bytes}}{\text{length in tokens}}$$

Typical values: ρ ≈ 3.5–4.0 for English (each token represents ~4 characters on average).

#### Complexity

- Each merge step: O(|C|) to scan for pairs
- Total: O(N · |C|) for N merge operations
- In practice, optimized to O(|C| log |C|) with priority queues

#### Worked Example

Corpus: `"aabaab"`, initial vocabulary V = {a, b}.

**Initial state**: `[a, a, b, a, a, b]` — 6 tokens, |V| = 2

**Step 1** — Count all adjacent pairs:

| Pair   | Positions       | Count |
| ------ | --------------- | ----- |
| (a, a) | (0,1) and (3,4) | 2     |
| (a, b) | (1,2) and (4,5) | 2     |
| (b, a) | (2,3)           | 1     |

Tie between (a,a) and (a,b) at frequency 2. BPE breaks ties by implementation convention; here we choose (a,a). Create new token `"aa"` and replace all occurrences.

After merge: `[aa, b, aa, b]` — 4 tokens, V = {a, b, aa}

Token count reduced by f = 2 (the pair frequency), confirming |C_after| = |C_before| − f = 6 − 2 = 4.

**Step 2** — Count pairs in `[aa, b, aa, b]`:

| Pair    | Count |
| ------- | ----- |
| (aa, b) | 2     |
| (b, aa) | 1     |

Most frequent: (aa, b) at frequency 2. Merge into `"aab"`.

After merge: `[aab, aab]` — 2 tokens, V = {a, b, aa, aab}

**Result**:

- Final vocabulary: {a, b, aa, aab} with |V| = 4
- Compression ratio: ρ = 6 / 2 = **3.0×**
- Merge rules learned (in order): (a,a) → aa, then (aa,b) → aab
- To encode new text, apply merge rules in the exact order they were learned
- The original string is **losslessly recoverable**: concat("aab", "aab") = "aabaab" ✓

### 3.2 Unigram Language Model (SentencePiece)

The unigram model takes the **opposite approach** to BPE: start large, prune down.

#### Probabilistic Framework

Assign a probability to each token in vocabulary V:

$$P(t) \text{ for each } t \in V, \quad \sum_{t \in V} P(t) = 1$$

For a segmentation S = (t₁, t₂, ..., tₖ) of string s, the probability is:

$$P(S) = \prod_{i=1}^{k} P(t_i)$$

(assuming unigram independence — each token probability is independent).

#### Optimal Segmentation

The best tokenization **maximizes the product** (equivalently, minimizes negative log-likelihood):

$$S^* = \arg\max_{S \in \mathcal{S}(s)} \prod_{i=1}^{k} P(t_i) = \arg\min_{S \in \mathcal{S}(s)} \sum_{i=1}^{k} -\log P(t_i)$$

where $\mathcal{S}(s)$ is the set of all valid segmentations of s.

This is solved exactly via the **Viterbi algorithm** (dynamic programming) in O(n · max_token_length) time.

#### EM Training

Token probabilities are learned via Expectation-Maximization on the training corpus D.

**Objective**: Maximize the marginal log-likelihood over all training sentences:

$$\mathcal{L}(V) = \sum_{s \in D} \log P(s) = \sum_{s \in D} \log \left( \sum_{S \in \mathcal{S}(s)} \prod_{t \in S} P(t) \right)$$

Since the sum inside the log makes direct optimization intractable, EM iterates between two steps:

**E-step**: For each training sentence s, compute the **expected frequency** of each token t across all valid segmentations, weighted by segmentation probability:

$$E[\text{count}(t)] = \sum_{s \in D} \sum_{S \in \mathcal{S}(s)} P(S | s) \cdot \text{count}(t, S)$$

where $P(S | s) = \prod_{t_i \in S} P(t_i) \; / \; \sum_{S'} \prod_{t_j \in S'} P(t_j)$.

This is computed efficiently using the **forward-backward algorithm** on the segmentation lattice — a directed acyclic graph (DAG) where each edge corresponds to a valid token. The forward variable $\alpha[j]$ stores the total log-probability of all segmentations of s[0:j]:

$$\alpha[0] = 0, \quad \alpha[j] = \text{logaddexp}_{i: s[i:j] \in V}\left(\alpha[i] + \log P(s[i:j])\right)$$

The backward variable $\beta[i]$ stores the total log-probability of all segmentations of s[i:n]. The expected count of token s[i:j] is then:

$$E[\text{count}(s[i:j])] = \exp\left(\alpha[i] + \log P(s[i:j]) + \beta[j] - \alpha[n]\right)$$

Alternatively, **Viterbi EM** approximates by using only the single best segmentation per sentence in the E-step (replacing the sum with an argmax). This is faster but less accurate.

**M-step**: Update token probabilities using the expected counts:

$$P(t) \leftarrow \frac{E[\text{count}(t)]}{\sum_{t' \in V} E[\text{count}(t')]}$$

**Convergence**: EM guarantees $\mathcal{L}$ is non-decreasing at each iteration and converges to a local maximum. Typical convergence: 10–20 iterations.

**Pruning**: After EM converges, compute the marginal loss for each token — the change in corpus log-likelihood if that token were removed:

$$\Delta \mathcal{L}(t) = \mathcal{L}(V \setminus \{t\}) - \mathcal{L}(V) \geq 0$$

Remove the 10–20% of tokens with smallest ΔL (least impact on likelihood), then re-run EM. Repeat pruning cycles until |V| reaches the target size. This top-down approach (start large, prune down) contrasts directly with BPE's bottom-up approach.

#### Loss of Removing a Token

When considering pruning token t from V:

$$\Delta \mathcal{L}(t) = \mathcal{L}(V \setminus \{t\}) - \mathcal{L}(V)$$

Remove tokens with smallest ΔL (least impact on overall corpus likelihood).

### 3.3 WordPiece (BERT)

WordPiece is similar to BPE but uses a **likelihood-based criterion** for merges.

#### Algorithm

```
Input:  corpus C, desired vocabulary size N
Output: vocabulary V

1. Initialize V = set of all individual characters in C
2. While |V| < N:
   a. For every adjacent pair (a, b) in the tokenized corpus, compute:
      score(a, b) = count(ab) / (count(a) × count(b))
   b. Merge pair with highest score
   c. Add merged token to V
3. Return V
```

#### PMI-Based Merge Selection

The merge score is the **pointwise mutual information (PMI)**:

$$\text{score}(a, b) = \frac{P(ab)}{P(a) \cdot P(b)} = \frac{\text{count}(ab) / N}{\text{count}(a)/N \cdot \text{count}(b)/N}$$

Taking the logarithm:

$$\text{PMI}(a, b) = \log \frac{P(a, b)}{P(a)P(b)}$$

**Why PMI instead of frequency?** Consider two pairs:

- Pair ("t", "h") appears 1000 times, but "t" appears 5000 and "h" appears 4000 → PMI is low (they co-occur roughly as often as chance predicts)
- Pair ("q", "u") appears 200 times, but "q" appears 210 and "u" appears 3000 → PMI is high ("q" almost always precedes "u" — this is surprising)

BPE would merge ("t","h") first (higher frequency); WordPiece merges ("q","u") first (higher PMI). WordPiece captures **linguistic structure** better because it identifies pairs that are truly associated, not just common.

#### Encoding with WordPiece

WordPiece uses **greedy left-to-right longest match** (not Viterbi):

```
Input: "unbelievable"
1. Try "unbelievable" → not in V
2. Try "unbelievabl" → not in V
   ...
3. Try "un" → in V! Take it.
4. Try "believable" → not in V
5. Try "believabl" → not in V
   ...
6. Try "believe" → not in V
7. Try "believ" → not in V
   ...
8. Try "be" → in V! Take it (with ## prefix: "##be")
   ...
Result: ["un", "##believ", "##able"]
```

The `##` prefix marks continuation subwords (not word-initial). This is why BERT tokenization looks different from GPT tokenization.

> **Key difference from BPE**: BPE's greedy encoding applies merge rules in learned order. WordPiece's greedy encoding does longest-match at each position. Neither is globally optimal — only Unigram's Viterbi finds the true optimum.

### 3.4 Comparison

| Property                | BPE                    | Unigram                 | WordPiece              |
| ----------------------- | ---------------------- | ----------------------- | ---------------------- |
| Direction               | Bottom-up (grow V)     | Top-down (shrink V)     | Bottom-up              |
| Merge criterion         | Frequency              | Likelihood (EM)         | PMI / likelihood ratio |
| Segmentation            | Deterministic (greedy) | Probabilistic (Viterbi) | Greedy left-to-right   |
| Used by                 | GPT, LLaMA, Mistral    | T5, ALBERT, mBART       | BERT, DistilBERT       |
| Multiple segmentations? | No (1 canonical)       | Yes (can sample)        | No                     |

---

## 4. AI/ML Applications

### 4.1 Context Window Efficiency

The context window is measured in **tokens**, not characters. Compression ratio directly determines how much text fits:

$$\text{effective chars} = \text{context window (tokens)} \times \rho$$

| Compression Ratio      | 2K window | 4K window  | 8K window  | 32K window  | 128K window |
| ---------------------- | --------- | ---------- | ---------- | ----------- | ----------- |
| ρ = 2.0 (poor, CJK)    | ~4K chars | ~8K chars  | ~16K chars | ~64K chars  | ~256K chars |
| ρ = 3.0 (multilingual) | ~6K chars | ~12K chars | ~24K chars | ~96K chars  | ~384K chars |
| ρ = 3.7 (good EN)      | ~7K chars | ~15K chars | ~30K chars | ~118K chars | ~474K chars |
| ρ = 4.2 (optimized EN) | ~8K chars | ~17K chars | ~34K chars | ~134K chars | ~538K chars |

**Practical implication**: A 128K context window with ρ=4.0 holds roughly one average novel (~500K chars). With ρ=2.0 (for CJK), only half a novel fits — the user pays the same API cost for half the content.

**Multilingual penalty**: Languages like Chinese, Japanese, Korean have lower ρ because their characters are less frequent in English-dominated training data, resulting in more tokens per concept. This creates a compounding disadvantage:

1. **Context**: Less text fits in the window → model sees less relevant context
2. **Cost**: More tokens per query → higher API charges
3. **Latency**: More tokens → more autoregressive decode steps → slower generation
4. **Quality**: Shorter effective context → harder for the model to track long-range dependencies

Improving ρ from 2.0 to 4.0 is mathematically equivalent to **doubling the context window** at zero hardware cost.

### 4.2 Vocabulary Size and Model Dimensions

The vocabulary determines the size of two critical model components:

**Embedding matrix**: $E \in \mathbb{R}^{N \times d}$

$$\text{Embedding params} = N \times d$$

**Output projection (LM head)**: $W \in \mathbb{R}^{d \times N}$

$$\text{LM head params} = d \times N$$

**Weight tying**: Modern LLMs share the embedding and LM head matrices ($W = E^T$), cutting the vocabulary parameter cost in half:

$$\text{Vocab params (untied)} = 2Nd, \quad \text{Vocab params (tied)} = Nd$$

Vocabulary parameter costs across real models:

| Model        | d     | V       | Vocab Params (tied) | % of Total (tied) | % of Total (untied) |
| ------------ | ----- | ------- | ------------------- | ----------------- | ------------------- |
| GPT-2 Small  | 768   | 50,257  | 38.6M               | 31.1%             | 62.3%               |
| GPT-2 Large  | 1,280 | 50,257  | 64.3M               | 8.3%              | 16.6%               |
| LLaMA-7B     | 4,096 | 32,000  | 131.1M              | 1.9%              | 3.7%                |
| LLaMA-3 8B   | 4,096 | 128,000 | 524.3M              | 6.6%              | 13.1%               |
| GPT-4 (est.) | 8,192 | 100,277 | 821.5M              | 0.4%              | 0.8%                |

**Memory footprint** of the embedding table alone (FP16 = 2 bytes per param):

$$\text{Memory} = N \times d \times \text{bytes\_per\_param}$$

| V       | d=768 (FP16) | d=4096 (FP16) | d=8192 (FP16) |
| ------- | ------------ | ------------- | ------------- |
| 32,000  | 47 MB        | 250 MB        | 500 MB        |
| 50,257  | 74 MB        | 393 MB        | 786 MB        |
| 128,000 | 188 MB       | 1.0 GB        | 2.0 GB        |
| 256,000 | 375 MB       | 2.0 GB        | 4.0 GB        |

For edge deployment (phones, embedded devices), a 4 GB embedding table may exceed available memory. This motivates vocabulary pruning and quantized embeddings.

**The tradeoff**: Larger V → better compression (higher ρ) → fewer tokens per sequence → faster inference. But larger V → more parameters → more memory → rare tokens get fewer training examples. The sweet spot is typically 32K–128K tokens.

### 4.3 Tokenization and Arithmetic

Tokenizers create non-obvious number representations that directly undermine arithmetic reasoning:

```
GPT-4 tokenizer (cl100k_base):
  "1"      → ["1"]              (1 token)
  "12"     → ["12"]             (1 token)
  "123"    → ["123"]            (1 token)
  "1234"   → ["123", "4"]       (2 tokens — split!)
  "12345"  → ["123", "45"]      (2 tokens)
  "123456" → ["123", "456"]     (2 tokens)
```

#### Why This Breaks Arithmetic

For a human, the digit "4" in "1234" occupies the **ones place** (positional notation: $1 \times 10^3 + 2 \times 10^2 + 3 \times 10^1 + 4 \times 10^0$). But when the tokenizer splits "1234" into ["123", "4"], the model must somehow learn:

1. "4" after "123" means $4 \times 10^0$ (ones place)
2. "4" as a standalone in "42" means $4 \times 10^1$ (tens place)
3. "4" in "456" token means $4 \times 10^2$ (hundreds place, relative to the token)

The same token "4" carries **different positional value** depending on context — a mapping the model must learn implicitly from training data, without any architectural support.

#### The Inconsistency Problem

Consider addition: 1234 + 5678 = 6912

```
"1234" → ["123", "4"]        carries: (123 × 10) + 4
"5678" → ["567", "8"]        carries: (567 × 10) + 8
"6912" → ["691", "2"]        carries: (691 × 10) + 2
```

The "carry" boundaries don't align with token boundaries. The model must learn to carry across token splits that occur at different digit positions for different numbers.

**Mitigation**: Some models force single-digit tokenization (each digit = 1 token), ensuring consistent positional encoding at the cost of longer sequences for numbers.

### 4.4 Tokenization Fertility

**Fertility** measures how many tokens a word produces:

$$\text{fertility}(w) = |T(w)|$$

High fertility = computational cost + harder to learn. Rare/non-English words often have high fertility:

| Word           | GPT-4 Tokens         | Fertility |
| -------------- | -------------------- | --------- |
| "the"          | ["the"]              | 1         |
| "Bayesian"     | ["Bay", "esian"]     | 2         |
| "tokenization" | ["token", "ization"] | 2         |
| "münchen"      | ["m", "ün", "chen"]  | 3         |

### 4.5 Special Tokens

Special tokens carry structural information outside the text. They are **manually added** to the vocabulary (not learned by BPE/Unigram) and serve as control signals for the model:

| Token       | ID  | Purpose                  | When Used                                 |
| ----------- | --- | ------------------------ | ----------------------------------------- |
| `<\|bos\|>` | 1   | Beginning of sequence    | Prepended to every input; signals "start" |
| `<\|eos\|>` | 2   | End of sequence          | Model generates to signal completion      |
| `<\|pad\|>` | 0   | Padding for batching     | Fills shorter sequences to uniform length |
| `<\|unk\|>` | 3   | Unknown (not in V)       | Fallback; rare in byte-level BPE          |
| `<\|sep\|>` | —   | Segment separator (BERT) | Separates sentence A from sentence B      |

#### How Special Tokens Affect the Pipeline

**Sequence construction** (autoregressive LLM):

```
Raw text:  "Hello, how are you?"
Tokenized: ["Hello", ",", " how", " are", " you", "?"]
With special: [<bos>, "Hello", ",", " how", " are", " you", "?", <eos>]
IDs:       [1, 15496, 11, 703, 527, 499, 30, 2]
```

**Batching with padding** (training):

```
Seq 1: [<bos>, "Hello", <eos>, <pad>, <pad>]   attention_mask = [1, 1, 1, 0, 0]
Seq 2: [<bos>, "How", "are", "you", <eos>]     attention_mask = [1, 1, 1, 1, 1]
```

The attention mask ensures `<pad>` tokens are **ignored** in self-attention — the model never attends to them.

**Chat/instruction models** add additional special tokens:

```
<|system|>You are a helpful assistant.<|end|>
<|user|>What is 2+2?<|end|>
<|assistant|>4<|end|>
```

These special tokens partition the input into roles, enabling the model to distinguish instructions from user queries from its own responses.

---

## 5. Information-Theoretic View

### 5.1 Tokenization as Compression

BPE is closely related to the **Lempel-Ziv family** of compression algorithms. The optimal tokenizer would approach the **entropy rate** of the language:

$$H = -\sum_{s \in \Sigma^*} P(s) \log_2 P(s) \quad \text{(bits per character)}$$

English has an entropy of approximately **1.0–1.5 bits/character** (Shannon's experiments, 1951).

#### Bits Per Character Analysis

A tokenizer encodes text at a certain number of bits per character. If we use a uniform code for all tokens:

$$\text{bits/char (uniform)} = \frac{\log_2 |V|}{\rho}$$

If we use the actual token distribution (entropy-optimal coding like Huffman):

$$\text{bits/char (optimal)} = \frac{H_{\text{token}}}{\rho}$$

Comparison across tokenizer configurations:

| Tokenizer              | ρ   | Coding  | bits/token | bits/char | vs Shannon (1.3) |
| ---------------------- | --- | ------- | ---------- | --------- | ---------------- |
| ASCII (character)      | 1.0 | Uniform | 8.0        | 8.0       | 6.2×             |
| 26 letters + space     | 1.0 | Uniform | 4.75       | 4.75      | 3.7×             |
| BPE 32K (uniform code) | 3.7 | Uniform | 15.0       | 4.05      | 3.1×             |
| BPE 32K (Huffman)      | 3.7 | Entropy | ~10.0      | 2.70      | 2.1×             |
| BPE 100K (uniform)     | 4.0 | Uniform | 16.6       | 4.15      | 3.2×             |
| BPE 100K (Huffman)     | 4.0 | Entropy | ~11.0      | 2.75      | 2.1×             |

**Key insight**: The language model itself implicitly acts as an entropy coder. When GPT-4 achieves cross-entropy loss of ~1.5–2.0 bits/char on English text, it is approaching Shannon's bound — the tokenizer provides the vocabulary, and the Transformer learns the probability distribution to compress nearly optimally.

Practical tokenizers balance compression with vocabulary size constraints — a V=1M tokenizer would compress better but waste parameters on rare tokens.

### 5.2 Entropy of the Token Distribution

For a corpus tokenized by T, the token-level Shannon entropy is:

$$H_{\text{token}} = -\sum_{t \in V} P(t) \log_2 P(t)$$

where $P(t) = \text{count}(t) / \sum_{t'} \text{count}(t')$.

**Maximum entropy**: $H_{\max} = \log_2 N$ occurs when all N tokens are equally likely (uniform distribution).

**Efficiency**: The ratio $\eta = H_{\text{token}} / H_{\max}$ measures vocabulary **utilization**.

**Perplexity**: $\text{PPL} = 2^{H_{\text{token}}}$ gives the effective vocabulary size — the number of tokens in a hypothetical uniform distribution with the same entropy.

#### Worked Example

Suppose a BPE tokenizer with V = 50,000 produces the following token distribution on a corpus:

- 5,000 tokens appear frequently (P(t) ≈ 0.015 each) → contribute 75% of total mass
- 15,000 tokens appear occasionally (P(t) ≈ 0.0015 each) → contribute 22.5%
- 30,000 tokens appear rarely (P(t) ≈ 0.00008 each) → contribute 2.5%

Then:

$$H_{\text{token}} \approx 12.5 \text{ bits} \quad \text{vs} \quad H_{\max} = \log_2 50000 \approx 15.6 \text{ bits}$$

$$\eta = 12.5 / 15.6 = 80\% \quad \quad \text{PPL} = 2^{12.5} \approx 5792$$

Interpretation: Although V = 50,000, the **effective** vocabulary is only ~5,792 tokens. The remaining ~44,000 tokens are so rare they contribute almost nothing. This wasted capacity motivates vocabulary pruning — removing tokens that don't pay for their embedding parameters.

#### Zipf's Law in Token Distributions

Token frequencies follow a power law (Zipf's law):

$$f(r) \propto r^{-\alpha}, \quad \alpha \approx 1$$

where r is the rank (1 = most frequent, 2 = second most frequent, etc.). This means:

- The top 5% of tokens account for ~50% of all occurrences
- The bottom 50% of tokens together account for <5%
- A few vocabulary entries dominate; most are rare

On a log-log plot of frequency vs rank, Zipfian distributions appear linear with slope −α. This universal pattern holds across languages and tokenizer types.

### 5.3 Rényi Entropy and Vocabulary Balance

The Rényi entropy of order α generalizes Shannon entropy:

$$H_\alpha = \frac{1}{1-\alpha} \log_2 \left(\sum_{t \in V} P(t)^\alpha\right)$$

#### Special Cases

**Shannon entropy** (α → 1, by L'Hôpital's rule):

$$H_1 = -\sum_{t \in V} P(t) \log_2 P(t)$$

Measures the **average surprise** per token — the expected number of bits to encode a randomly drawn token.

**Collision entropy** (α = 2):

$$H_2 = -\log_2 \left(\sum_{t \in V} P(t)^2\right)$$

Measures the probability that two independently drawn tokens match. Lower $H_2$ means fewer tokens dominate the distribution. $\sum P(t)^2$ is also the **Herfindahl-Hirschman Index** (HHI) from economics — a measure of market concentration.

**Min-entropy** (α → ∞):

$$H_\infty = -\log_2 \left(\max_{t \in V} P(t)\right)$$

Dominated entirely by the most frequent token. Gives the **worst-case** surprise — the minimum number of bits needed per token.

#### Ordering and the Gap

For all distributions, Rényi entropies are non-increasing in α:

$$H_\infty \leq H_2 \leq H_1 \leq H_{0.5} \leq H_{\max} = \log_2 |V|$$

The **H₁ − H₂ gap** is particularly diagnostic:

- **Gap ≈ 0**: Distribution is nearly uniform — all tokens are well-utilized
- **Small gap (<0.5 bits)**: Mild skew — reasonable vocabulary
- **Large gap (>2 bits)**: Heavy tail of rarely-used tokens — vocabulary is wasteful

#### Example

For a BPE tokenizer with V = 50,000 on an English corpus:

| Metric            | Value     | Interpretation                                        |
| ----------------- | --------- | ----------------------------------------------------- |
| H_max             | 15.6 bits | Upper bound (uniform over V)                          |
| H₁ (Shannon)      | 12.5 bits | Average information per token                         |
| H₂ (Collision)    | 10.1 bits | Effective vocab for collision events ≈ 2^10.1 ≈ 1,097 |
| H_∞ (Min-entropy) | 6.0 bits  | Most frequent token has P ≈ 2^−6 = 1.6%               |
| H₁ − H₂ gap       | 2.4 bits  | Heavy tail — many tokens almost never appear          |

The 2.4-bit gap confirms that while the vocabulary has 50,000 entries, most text is covered by a much smaller "core" vocabulary. The long tail exists to handle rare words and multilingual content.

---

## 6. Dynamic Programming: Optimal Segmentation

### 6.1 The Segmentation Problem

Given string s of length n and vocabulary V with token scores (e.g., -log P(t)):

$$\text{cost}(S) = \sum_{i=1}^{k} w(t_i)$$

Find segmentation S\* minimizing total cost.

### 6.2 Viterbi Algorithm for Tokenization

Define dp[j] = minimum cost to tokenize s[1..j]:

```
dp[0] = 0  (empty prefix)
dp[j] = min over all i < j such that s[i+1..j] ∈ V:
        dp[i] + weight(s[i+1..j])
```

$$\text{dp}[j] = \min_{i: s[i+1..j] \in V} \left(\text{dp}[i] + w(s[i+1..j])\right)$$

**Complexity**: O(n · L) where L = max token length in V (typically 16–64 chars).

#### Worked Example

Vocabulary with costs: V = {"a": 1.0, "ab": 0.5, "b": 1.5, "ba": 0.8}

String to segment: s = "abab" (n = 4)

**Fill the DP table** (dp[j] = min cost to tokenize s[0:j]):

| j   | Candidates (s[i:j] ∈ V) | dp[i] + w(s[i:j])     | dp[j] | Back |
| --- | ----------------------- | --------------------- | ----- | ---- |
| 0   | (base case)             | —                     | 0.0   | —    |
| 1   | s[0:1]="a" (cost 1.0)   | dp[0] + 1.0 = **1.0** | 1.0   | 0    |
| 2   | s[0:2]="ab" (cost 0.5)  | dp[0] + 0.5 = **0.5** | 0.5   | 0    |
|     | s[1:2]="b" (cost 1.5)   | dp[1] + 1.5 = 2.5     |       |      |
| 3   | s[1:3]="ba" (cost 0.8)  | dp[1] + 0.8 = 1.8     | 1.5   | 2    |
|     | s[2:3]="a" (cost 1.0)   | dp[2] + 1.0 = **1.5** |       |      |
| 4   | s[2:4]="ab" (cost 0.5)  | dp[2] + 0.5 = **1.0** | 1.0   | 2    |
|     | s[3:4]="b" (cost 1.5)   | dp[3] + 1.5 = 3.0     |       |      |

**Backtrack**: dp[4] came from i=2 (token "ab") → dp[2] came from i=0 (token "ab")

**Optimal segmentation**: ["ab", "ab"] with total cost **1.0**

All 5 valid segmentations ranked by cost:

| Segmentation | Cost                        |
| ------------ | --------------------------- |
| [ab, ab]     | 0.5 + 0.5 = **1.0** ✓       |
| [ab, a, b]   | 0.5 + 1.0 + 1.5 = 3.0       |
| [a, b, ab]   | 1.0 + 1.5 + 0.5 = 3.0       |
| [a, ba, b]   | 1.0 + 0.8 + 1.5 = 3.3       |
| [a, b, a, b] | 1.0 + 1.5 + 1.0 + 1.5 = 5.0 |

The Viterbi algorithm found the global optimum in O(4 × 2) = 8 operations, versus 5 segmentations to check exhaustively. For real strings (n=1000, L=64), exhaustive search is astronomically intractable while Viterbi runs in O(64,000) — linear in practice.

### 6.3 Number of Possible Segmentations

For a string of length n with a vocabulary containing all substrings up to length L, how many valid segmentations exist?

#### Derivation (L = 2 case)

Let $C(n)$ = number of ways to segment a string of length n, where each segment has length 1 or 2.

At position 0, we can either:

- Take a **1-character** token, leaving $C(n-1)$ ways to segment the rest
- Take a **2-character** token, leaving $C(n-2)$ ways to segment the rest

This gives the recurrence:

$$C(n) = C(n-1) + C(n-2), \quad C(0) = 1, \quad C(1) = 1$$

This is exactly the **Fibonacci sequence**: $C(n) = F_{n+1}$.

#### General Case (max token length L)

With tokens up to length L:

$$C(n) = \sum_{k=1}^{\min(n, L)} C(n-k), \quad C(0) = 1$$

This is a generalized Fibonacci (tribonacci for L=3, etc.) with growth rate approaching $2^n$ as $L \to n$.

#### Growth Table

| String length n | L=2 (Fibonacci) | L=4        | L=16       | L=n (all substrings) |
| --------------- | --------------- | ---------- | ---------- | -------------------- |
| 5               | 8               | 16         | 16         | 16                   |
| 10              | 89              | 504        | 512        | 512                  |
| 20              | 10,946          | 283,953    | 524,288    | 524,288              |
| 50              | 2.0 × 10¹⁰      | 4.4 × 10¹⁴ | 5.6 × 10¹⁴ | 5.6 × 10¹⁴           |
| 100             | 5.7 × 10²⁰      | —          | 6.3 × 10²⁹ | 6.3 × 10²⁹           |

Even for a modest 50-character string, there are **billions** of possible segmentations. This is why:

- **BPE** uses greedy merging (never reconsiders past merges)
- **Unigram** uses Viterbi DP to find the single best segmentation in O(nL)
- **Exhaustive search** is computationally intractable for any real text

---

## 7. Common Mistakes

| Mistake                                     | Why It's Wrong                                                                                                    | Fix                                                                          |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| "Tokenization doesn't affect model quality" | Token boundaries determine what patterns the model can learn; bad tokenization = bad arithmetic, bad multilingual | Choose tokenizer carefully; evaluate fertility across languages              |
| "Larger vocabulary is always better"        | Larger V = larger embedding matrix = more parameters to train, and rare tokens get undertrained                   | Balance: 32k–128k is typical sweet spot                                      |
| "BPE is optimal compression"                | BPE is greedy — it doesn't find the globally optimal vocabulary                                                   | Unigram model with EM is closer to optimal; BPE is a practical approximation |
| "All tokenizers produce the same output"    | Different algorithms produce very different segmentations for the same text                                       | Always note which tokenizer was used when comparing models                   |
| "Tokens ≈ words"                            | Common tokens include subwords, spaces, punctuation; "I'm" might be 1 or 2 tokens                                 | Never assume word-level alignment                                            |
| "Re-tokenizing is cheap"                    | Changing the tokenizer invalidates ALL pretrained weights (embedding + LM head)                                   | Tokenizer is fixed for the lifetime of a model                               |

---

## 8. Exercises

See [exercises.ipynb](exercises.ipynb) for full implementations with scaffolds and solutions.

### Exercise 1: Manual BPE (Pen and Paper)

Starting with corpus "aabaabaab", initial vocabulary {a, b}:

1. Count all adjacent pairs
2. Perform 2 merge steps
3. What is the final vocabulary?
4. What is the compression ratio?

### Exercise 2: Compression Ratio

A tokenizer with V=32,000 tokenizes 1 million characters of English into 280,000 tokens.

1. What is the compression ratio ρ?
2. How many bits per character does this represent?
3. Compare to English entropy (~1.3 bits/char). Is the tokenizer near-optimal?

### Exercise 3: Vocabulary Size Tradeoff

For a model with d=4096:

1. Calculate embedding + LM head parameters for V = 32K, 64K, 128K, 256K
2. If total model params = 7B, what fraction is vocabulary for each?
3. At what point does vocabulary overhead become a concern?

### Exercise 4: Fertility Analysis

Using any tokenizer (tiktoken, sentencepiece):

1. Compute average fertility for English, Spanish, Chinese, Arabic text
2. Plot fertility distribution per language
3. What does this reveal about training data composition?

### Exercise 5: Viterbi Segmentation

Given vocabulary V = {"a": 1.0, "ab": 0.5, "b": 1.5, "ba": 0.8} with costs:

1. Draw the segmentation graph for "abab"
2. Find the optimal segmentation using DP
3. How many total segmentations exist?

### Exercise 6: Unigram Language Model & EM Training

1. Initialize vocabulary with character + bigram tokens
2. Implement E-step (Viterbi segmentation) and M-step (probability re-estimation)
3. Run EM for 10 iterations, track log-likelihood convergence
4. Compute ΔL for each token and prune the bottom 20%

### Exercise 7: WordPiece PMI Score Comparison

1. Compute PMI for all adjacent pairs in a corpus
2. Rank by frequency (BPE) vs PMI (WordPiece) — find disagreements
3. Implement the ## prefix encoding used by BERT's WordPiece

### Exercise 8: Context Window & Multilingual Penalty

1. Build effective context table for multiple window sizes and fertilities
2. Calculate the multilingual penalty (Thai vs English)
3. Cross-model, cross-language context capacity comparison

### Exercise 9: Digit Tokenization & Arithmetic

1. Train BPE on number-heavy corpus, observe inconsistent digit splits
2. Demonstrate the carry problem across token boundaries
3. Implement and compare a digit-aware tokenizer

### Exercise 10: Information-Theoretic Deep Dive

1. Compute bits-per-char under uniform vs entropy-optimal coding
2. Build the full Rényi spectrum: H_α for α ∈ {0.5, 1, 2, 5, 10, ∞}
3. Analyze H₁ − H₂ gap across uniform, Zipfian, and BPE distributions
4. Compute perplexity and vocabulary efficiency ratio η

### Exercise 11: Segmentation Counting & Fibonacci

1. Count segmentations using the generalized Fibonacci recurrence
2. Verify C(n) = F\_{n+1} for L=2
3. Build growth table and estimate growth rates for L = 2, 3, 4

### Exercise 12: Tokenization Cost Calculator

1. Build per-conversation cost calculator by language
2. Produce a fairness report card across languages and pricing tiers
3. Calculate annual savings from a perfect multilingual tokenizer

---

## 9. Why This Matters for AI

| Aspect                    | Impact                                                                                   |
| ------------------------- | ---------------------------------------------------------------------------------------- |
| **Cost**                  | API pricing is per-token; efficient tokenization saves money                             |
| **Context length**        | Better compression = more text in fixed context window                                   |
| **Multilingual fairness** | Poor tokenization of non-English = worse performance + higher cost                       |
| **Model capabilities**    | Number tokenization directly affects arithmetic ability                                  |
| **Training efficiency**   | Fewer tokens = fewer forward passes = faster training                                    |
| **Deployment**            | Vocabulary size determines embedding table memory (often the bottleneck on edge devices) |
| **Reproducibility**       | Tokenizer mismatch between training and inference = garbage output                       |

### The Tokenization Tax

Non-English languages pay a "tokenization tax" — the same semantic content requires more tokens:

```
English:  "Hello, how are you?"       → 5-6 tokens
Spanish:  "Hola, ¿cómo estás?"        → 7-9 tokens
Japanese: "こんにちは、お元気ですか？"    → 10-15 tokens
Thai:     "สวัสดี คุณเป็นอย่างไร?"      → 15-20 tokens
Amharic:  "ሰላም, እንዴት ነህ?"            → 20-30 tokens
```

#### Concrete Cost Calculation

Consider a customer-service chatbot processing 1M conversations/month, each averaging 500 characters of user input + 1000 characters of model output:

| Language | Fertility ρ | Input tokens | Output tokens | Total tokens/conv | Monthly tokens | Monthly cost (@$0.01/1K) |
| -------- | ----------- | ------------ | ------------- | ----------------- | -------------- | ------------------------ |
| English  | 4.0         | 125          | 250           | 375               | 375M           | **$3,750**               |
| Spanish  | 3.2         | 156          | 312           | 468               | 468M           | $4,680                   |
| Japanese | 1.6         | 312          | 625           | 937               | 937M           | $9,370                   |
| Thai     | 1.1         | 454          | 909           | 1,363             | 1,363M         | $13,630                  |

A Thai-language deployment costs **3.6× more** than English for identical semantic content.

#### Training Efficiency Impact

With fertility ρ, training on N characters requires N/ρ forward passes. If training a 7B model costs $2M on English text:

- Same content in Japanese: ~$5M (2.5× more forward passes)
- Same content in Thai: ~$7.3M (3.6× more forward passes)

This means non-English users:

- Get less context per dollar (pay more for the same conversation)
- Train slower (more tokens per sentence = more compute)
- See lower quality (each token carries less semantic content, so the model must compose more tokens to represent one concept)
- Hit context limits sooner (a 4K-token window holds ~16K English chars but only ~4.4K Thai chars)

This is an active area of research in fair and equitable AI. Approaches include training multilingual-balanced tokenizers, language-specific vocabulary allocation, and adaptor-based vocabulary extension.

---

## 10. Further Reading

### Papers

1. Sennrich et al. (2016) — "Neural Machine Translation of Rare Words with Subword Units" (original BPE for NLP)
2. Kudo (2018) — "Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates" (Unigram model)
3. Kudo & Richardson (2018) — "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
4. Radford et al. (2019) — GPT-2 paper (byte-level BPE)
5. Petrov et al. (2024) — "Language Model is All You Need: An Inequality on the Tokenization Tax"

### Implementations

- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast BPE tokenizer (Rust + Python)
- [SentencePiece](https://github.com/google/sentencepiece) — Google's BPE + Unigram tokenizer (C++)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) — Rust library supporting all algorithms
- [minbpe](https://github.com/karpathy/minbpe) — Karpathy's minimal BPE in pure Python (excellent for learning)

### Conceptual Bridge

Tokenization is the **input interface** between human language and the model's mathematical space. The next section, [Embedding Space Math](../02-Embedding-Space-Math/notes.md), covers what happens after tokenization: mapping token IDs into continuous vector representations in ℝᵈ where the model actually reasons.

```
Text → [Tokenization] → Token IDs → [Embedding] → Vectors in ℝᵈ → [Attention] → ...
       ^^^^^^^^^^^^^^                 ^^^^^^^^^^^
       THIS section                   NEXT section
```

---

[Home](../../README.md) | [Embedding Space Math →](../02-Embedding-Space-Math/notes.md)
