[← Positional Encodings](../04-Positional-Encodings/notes.md) | [Home](../../README.md) | [Training at Scale →](../06-Training-at-Scale/notes.md)

---

# Language Model Probability Math

> _"A language model is nothing more — and nothing less — than a probability distribution over sequences of tokens. Everything else is commentary."_

## Overview

Every large language model — GPT, LLaMA, Gemini, Claude — is fundamentally a probability estimator. Given a sequence of tokens, it produces a probability distribution over what comes next. The entire edifice of modern AI rests on this single operation: predict the next token. This section derives the mathematical foundations that make this possible: the chain rule decomposition that factorises sequence probability into autoregressive conditionals, the cross-entropy training objective and its gradient (which has the elegant form "predicted minus true"), information-theoretic measures (entropy, perplexity, KL divergence, bits-per-byte), the softmax function and its numerical stabilisation via log-sum-exp, all major decoding strategies (greedy, beam search, temperature, top-k, top-p, min-p, typical, contrastive, speculative), neural scaling laws (Kaplan, Chinchilla, inference-optimal, test-time compute), calibration and uncertainty quantification, conditional LMs (prompting, classifier-free guidance, RAG), RLHF/DPO/GRPO alignment objectives with full derivations, evaluation metrics (PPL, BLEU, ROUGE, BERTScore, LLM-as-judge), and implementation details for numerical stability at scale. This is the mathematical heart of every LLM.

## Prerequisites

- Probability theory: conditional probability, Bayes' theorem, expectation
- Information theory: entropy, cross-entropy, KL divergence (covered in [09-Information-Theory](../../09-Information-Theory/01-Entropy/notes.md))
- Calculus: partial derivatives, chain rule, gradient computation
- Completed: [01-Tokenization-Math](../01-Tokenization-Math/notes.md), [02-Embedding-Space-Math](../02-Embedding-Space-Math/notes.md), [03-Attention-Mechanism-Math](../03-Attention-Mechanism-Math/notes.md), and [04-Positional-Encodings](../04-Positional-Encodings/notes.md)

## Companion Notebooks

| Notebook                           | Description                                                                                              |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Softmax, cross-entropy, perplexity, decoding strategies, scaling laws, DPO, calibration, numerical demos |
| [exercises.ipynb](exercises.ipynb) | Chain rule decomposition, PPL calculation, temperature effects, top-p, CE gradient, scaling, DPO, ECE    |

## Learning Objectives

After completing this section, you will:

- State the chain rule decomposition and factorise any sequence probability into autoregressive conditionals
- Derive the softmax function from maximum-entropy principles and implement it with numerical stability
- Compute cross-entropy loss, its gradient with respect to logits, and explain why the gradient equals "predicted minus true"
- Define entropy, cross-entropy, KL divergence, perplexity, and bits-per-byte, and convert between them
- Implement and compare all major decoding strategies: greedy, beam search, temperature, top-k, top-p, min-p, typical
- Explain speculative decoding and contrastive decoding mathematically
- State the Kaplan and Chinchilla scaling laws, compute optimal model/data allocation for a given compute budget
- Define calibration, compute ECE, and apply temperature scaling for post-hoc calibration
- Derive the DPO loss from the RLHF objective and compute gradients by hand
- Explain GRPO and its advantage over PPO for reasoning model training
- Evaluate language models using perplexity, BLEU, ROUGE, and LLM-as-judge frameworks

## Table of Contents

- [Language Model Probability Math](#language-model-probability-math)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is a Language Model?](#11-what-is-a-language-model)
    - [1.2 The Prediction Game](#12-the-prediction-game)
    - [1.3 Why Probability?](#13-why-probability)
    - [1.4 Historical Timeline](#14-historical-timeline)
    - [1.5 Pipeline Position](#15-pipeline-position)
  - [2. Formal Definitions](#2-formal-definitions)
    - [2.1 Probability Over Sequences](#21-probability-over-sequences)
    - [2.2 Chain Rule Decomposition](#22-chain-rule-decomposition)
    - [2.3 Conditional Distribution](#23-conditional-distribution)
    - [2.4 The LM Head](#24-the-lm-head)
    - [2.5 Autoregressive Generation](#25-autoregressive-generation)
  - [3. Information Theory Foundations](#3-information-theory-foundations)
    - [3.1 Self-Information](#31-self-information)
    - [3.2 Entropy](#32-entropy)
    - [3.3 Cross-Entropy](#33-cross-entropy)
    - [3.4 KL Divergence](#34-kl-divergence)
    - [3.5 Perplexity](#35-perplexity)
    - [3.6 Bits Per Character / Bits Per Byte](#36-bits-per-character--bits-per-byte)
  - [4. Training Objective — Maximum Likelihood](#4-training-objective--maximum-likelihood)
    - [4.1 Log-Likelihood](#41-log-likelihood)
    - [4.2 Maximum Likelihood Estimation (MLE)](#42-maximum-likelihood-estimation-mle)
    - [4.3 Cross-Entropy Loss in Practice](#43-cross-entropy-loss-in-practice)
    - [4.4 Gradient of Cross-Entropy Loss](#44-gradient-of-cross-entropy-loss)
    - [4.5 Label Smoothing](#45-label-smoothing)
    - [4.6 Next-Token Prediction vs Masked LM](#46-next-token-prediction-vs-masked-lm)
  - [5. Decoding Strategies](#5-decoding-strategies)
    - [5.1 Greedy Decoding](#51-greedy-decoding)
    - [5.2 Beam Search](#52-beam-search)
    - [5.3 Temperature Sampling](#53-temperature-sampling)
    - [5.4 Top-k Sampling](#54-top-k-sampling)
    - [5.5 Top-p (Nucleus) Sampling](#55-top-p-nucleus-sampling)
    - [5.6 Min-p Sampling](#56-min-p-sampling)
    - [5.7 Typical Sampling](#57-typical-sampling)
    - [5.8 Contrastive / Speculative Methods](#58-contrastive--speculative-methods)
    - [5.9 Repetition and Frequency Penalties](#59-repetition-and-frequency-penalties)
  - [6. Scaling Laws](#6-scaling-laws)
    - [6.1 Neural Scaling Laws (Kaplan et al. 2020)](#61-neural-scaling-laws-kaplan-et-al-2020)
    - [6.2 Chinchilla Scaling Laws (Hoffmann et al. 2022)](#62-chinchilla-scaling-laws-hoffmann-et-al-2022)
    - [6.3 Inference-Optimal Scaling](#63-inference-optimal-scaling)
    - [6.4 Emergent Abilities and Phase Transitions](#64-emergent-abilities-and-phase-transitions)
    - [6.5 Test-Time Compute Scaling (2024–2026)](#65-test-time-compute-scaling-20242026)
  - [7. Calibration and Uncertainty](#7-calibration-and-uncertainty)
    - [7.1 What Is Calibration?](#71-what-is-calibration)
    - [7.2 Expected Calibration Error (ECE)](#72-expected-calibration-error-ece)
    - [7.3 Temperature Scaling for Calibration](#73-temperature-scaling-for-calibration)
    - [7.4 Overconfidence in LLMs](#74-overconfidence-in-llms)
    - [7.5 Epistemic vs Aleatoric Uncertainty](#75-epistemic-vs-aleatoric-uncertainty)
  - [8. Conditional Language Models](#8-conditional-language-models)
    - [8.1 Conditional vs Unconditional LM](#81-conditional-vs-unconditional-lm)
    - [8.2 Prompt as Conditioning](#82-prompt-as-conditioning)
    - [8.3 Bayes' Theorem in Decoding](#83-bayes-theorem-in-decoding)
    - [8.4 Classifier-Free Guidance (CFG)](#84-classifier-free-guidance-cfg)
    - [8.5 Retrieval-Augmented Generation (RAG)](#85-retrieval-augmented-generation-rag)
  - [9. RLHF and Reward-Based Probability](#9-rlhf-and-reward-based-probability)
    - [9.1 From Pretraining to Alignment](#91-from-pretraining-to-alignment)
    - [9.2 Reward Model](#92-reward-model)
    - [9.3 PPO Objective (RLHF)](#93-ppo-objective-rlhf)
    - [9.4 DPO — Direct Preference Optimisation](#94-dpo--direct-preference-optimisation)
    - [9.5 GRPO — Group Relative Policy Optimisation](#95-grpo--group-relative-policy-optimisation)
    - [9.6 The Alignment Tax](#96-the-alignment-tax)
  - [10. Evaluation Metrics](#10-evaluation-metrics)
    - [10.1 Perplexity (Revisited)](#101-perplexity-revisited)
    - [10.2 BLEU Score](#102-bleu-score)
    - [10.3 ROUGE](#103-rouge)
    - [10.4 BERTScore](#104-bertscore)
    - [10.5 LLM-as-Judge](#105-llm-as-judge)
    - [10.6 Benchmark Suites](#106-benchmark-suites)
  - [11. Numerical Stability and Implementation](#11-numerical-stability-and-implementation)
    - [11.1 Log-Sum-Exp Trick](#111-log-sum-exp-trick)
    - [11.2 Numerical Loss of Cross-Entropy](#112-numerical-loss-of-cross-entropy)
    - [11.3 Mixed Precision Training](#113-mixed-precision-training)
    - [11.4 Vocabulary Parallelism](#114-vocabulary-parallelism)
  - [12. Common Mistakes](#12-common-mistakes)
  - [13. Exercises](#13-exercises)
  - [14. Why This Matters for AI (2026 Perspective)](#14-why-this-matters-for-ai-2026-perspective)
  - [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is a Language Model?

A language model is a probability distribution over sequences of tokens. Given a vocabulary $V$, a language model assigns a probability to every possible finite string in $V^*$:

$$P: V^* \to [0, 1], \quad \sum_{\mathbf{t} \in V^*} P(\mathbf{t}) = 1$$

The core question is deceptively simple: **given what has come before, what comes next?**

Every LLM — GPT-4, LLaMA 3, Gemini, Claude — is fundamentally a probability estimator. The entire training process is about learning this distribution from data. Everything else — reasoning, coding, conversation, tool use — emerges from predicting the next token well enough.

### 1.2 The Prediction Game

| Component      | Description                                                                 |
| -------------- | --------------------------------------------------------------------------- |
| **Input**      | Sequence of tokens $(t_1, t_2, \ldots, t_n)$                                |
| **Output**     | Probability distribution over next token $P(t_{n+1} \mid t_1, \ldots, t_n)$ |
| **Generation** | Sample or argmax from this distribution, append, repeat                     |
| **Training**   | Adjust parameters so model's distribution matches data distribution         |

Everything else — reasoning, coding, conversation — emerges from this one operation: next-token prediction repeated until a stop condition.

### 1.3 Why Probability?

| Reason                | Explanation                                                                                       |
| --------------------- | ------------------------------------------------------------------------------------------------- |
| **Uncertainty**       | Language is fundamentally uncertain: many continuations are valid for any prefix                  |
| **Ranking**           | Probability allows ranking: "Paris" is more likely than "banana" after "The capital of France is" |
| **Calibration**       | A well-calibrated model knows what it doesn't know; uncertainty is informative                    |
| **Differentiability** | Probability is differentiable → enables gradient-based training via cross-entropy loss            |
| **Sampling**          | Probability enables controlled randomness: diversity, creativity, temperature scaling             |
| **Compositionality**  | Sequence probability decomposes into a product of conditionals via the chain rule                 |

### 1.4 Historical Timeline

| Year    | Milestone       | Key Contribution                                                                            |
| ------- | --------------- | ------------------------------------------------------------------------------------------- |
| 1948    | Shannon         | Information theory; entropy as measure of language uncertainty                              |
| 1951    | Shannon         | N-gram language models; bits-per-character experiments on English (≈1.0 BPC)                |
| 1980s   | Jelinek et al.  | N-gram LMs for speech recognition; smoothing methods (Kneser-Ney, Good-Turing)              |
| 2003    | Bengio et al.   | Neural probabilistic language model; learned embeddings replace discrete counts             |
| 2010    | Mikolov et al.  | RNN language models; arbitrary-length context via hidden state                              |
| 2013    | Graves          | Sequence generation with RNNs; handwriting and speech synthesis                             |
| 2017    | Vaswani et al.  | Transformer LM; self-attention replaces recurrence; parallelisable training                 |
| 2018–19 | Radford et al.  | GPT-1/2; large-scale autoregressive LM; zero-shot task transfer emerges                     |
| 2020    | Brown et al.    | GPT-3 (175B); in-context learning emerges from scale alone                                  |
| 2022    | Hoffmann et al. | Chinchilla scaling laws; optimal token/parameter ratio ≈ 20:1                               |
| 2023    | Touvron et al.  | LLaMA; open-source scaling; over-training for inference efficiency                          |
| 2023    | Rafailov et al. | DPO; direct preference optimisation bypasses reward model                                   |
| 2024    | OpenAI          | o1; test-time compute scaling for reasoning via chain-of-thought search                     |
| 2025    | DeepSeek        | R1; GRPO for reasoning; open-source test-time scaling                                       |
| 2025–26 | Community       | Mixture-of-experts LMs, 10M-token contexts, reasoning agents, speculative decoding standard |

### 1.5 Pipeline Position

```
Text → [Tokenizer] → IDs → [Embedding] → ℝᵈ → [Transformer Blocks] → hₙ → [LM Head] → logits → [Softmax] → P(next token)
                                                                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                                               THIS section
```

The transformer produces a hidden state $h_n \in \mathbb{R}^d$ for each position. This section covers everything that happens after: the LM head projects $h_n$ to logits, softmax converts to probabilities, cross-entropy measures the loss, and decoding strategies select the output token.

---

## 2. Formal Definitions

### 2.1 Probability Over Sequences

A language model defines a joint probability over token sequences:

$$P(t_1, t_2, \ldots, t_n) \geq 0, \quad \sum_{\mathbf{t} \in V^*} P(\mathbf{t}) = 1$$

This is a distribution over a **countably infinite set**: all finite strings over vocabulary $V$, including the empty string $\varepsilon$. The sum includes strings of every possible length.

For a vocabulary of size $|V| = 50{,}000$, the number of possible sequences of length $n$ is $50{,}000^n$. At $n = 100$, this exceeds $10^{469}$ — impossibly many to enumerate. The chain rule decomposition makes this tractable.

### 2.2 Chain Rule Decomposition

The chain rule of probability factorises the joint probability exactly:

$$\boxed{P(t_1, t_2, \ldots, t_n) = \prod_{i=1}^{n} P(t_i \mid t_1, t_2, \ldots, t_{i-1})}$$

Each factor is the conditional probability of the next token given all previous tokens. This is **exact** — no approximation. The chain rule is always valid for any joint distribution.

**Worked example**: P("the cat sat") with vocabulary tokens [the, cat, sat, ...]:

$$P(\text{the, cat, sat}) = \underbrace{P(\text{the})}_{\text{unigram}} \times \underbrace{P(\text{cat} \mid \text{the})}_{\text{bigram context}} \times \underbrace{P(\text{sat} \mid \text{the, cat})}_{\text{trigram context}}$$

We define $P(t_1 \mid \varepsilon) = P(t_1)$ where $\varepsilon$ is the empty context. In practice, $t_0 = \text{BOS}$ (beginning-of-sequence) token serves as the initial context.

**Key insight**: An autoregressive language model only needs to model one thing — $P(t_i \mid t_{<i})$ — and the chain rule gives us the probability of any sequence for free.

### 2.3 Conditional Distribution

At each step $i$, the model outputs a distribution over the full vocabulary $V$:

$$P(\cdot \mid t_1, \ldots, t_{i-1}): V \to [0, 1], \quad \sum_{t \in V} P(t \mid t_1, \ldots, t_{i-1}) = 1$$

This is a probability vector $\mathbf{p} \in \mathbb{R}^{|V|}$ with $|V|$ entries summing to 1. For GPT-2, $|V| = 50{,}257$; for LLaMA 3, $|V| = 128{,}256$.

**Implementation**: The conditional distribution is computed as:

$$\text{logits} \xrightarrow{\text{softmax}} \text{probability vector} \in \Delta^{|V|-1}$$

where $\Delta^{|V|-1}$ is the $(|V|-1)$-dimensional probability simplex.

### 2.4 The LM Head

The LM head converts hidden states to token probabilities:

| Component          | Shape          | Description                               |
| ------------------ | -------------- | ----------------------------------------- | ---------- | --------------------------------------------------------------------------------------------- |
| Hidden state $h_n$ | $\mathbb{R}^d$ | Final transformer output for position $n$ |
| Weight matrix $W$  | $\mathbb{R}^{  | V                                         | \times d}$ | Projection to vocabulary space (often tied to embedding matrix $E$)                           |
| Bias $b$           | $\mathbb{R}^{  | V                                         | }$         | Optional; often omitted in modern models                                                      |
| Logits $z$         | $\mathbb{R}^{  | V                                         | }$         | Raw unnormalised scores: $z = Wh_n + b$                                                       |
| Probabilities $p$  | $\Delta^{      | V                                         | -1}$       | $P(t \mid \text{context}) = \text{softmax}(z)_t = \frac{\exp(z_t)}{\sum_{v \in V} \exp(z_v)}$ |

**Weight tying**: In most modern LLMs, $W = E^\top$ — the LM head weight is the transpose of the token embedding matrix. This halves the parameter count of the vocabulary projection and provides a geometric interpretation: the probability of token $t$ is proportional to $\exp(h_n \cdot e_t)$, the exponentiated cosine-like similarity between the hidden state and the token embedding.

**Scale**: For LLaMA 3 8B ($d = 4096$, $|V| = 128{,}256$), the LM head is a $128{,}256 \times 4{,}096$ matrix — 525M parameters. With weight tying, these are shared with the embedding layer.

### 2.5 Autoregressive Generation

Generation proceeds token by token:

$$t_1 \sim P(\cdot \mid \text{BOS}), \quad t_2 \sim P(\cdot \mid \text{BOS}, t_1), \quad \ldots, \quad t_i \sim P(\cdot \mid t_1, \ldots, t_{i-1})$$

Stop when EOS token is sampled or maximum length is reached. Each step requires a full forward pass through the transformer (though KV caching makes this efficient).

This process defines a valid probability distribution over all possible generated sequences. The probability of any specific output sequence is the product of all per-step sampling probabilities.

---

## 3. Information Theory Foundations

### 3.1 Self-Information

The **surprise** (self-information) of observing token $t$ with probability $P(t)$:

$$I(t) = -\log_2 P(t) \quad \text{(bits)}$$

| $P(t)$ | $I(t)$ (bits) | Interpretation                       |
| ------ | ------------- | ------------------------------------ | ------- | --- | --- | ----------------------------------------- |
| 1.0    | 0             | Certain event: no surprise           |
| 0.5    | 1             | Fair coin flip: 1 bit of information |
| 0.25   | 2             | Two coin flips worth of surprise     |
| 0.01   | 6.64          | Rare event: high surprise            |
| 0.001  | 9.97          | Very rare: ~10 bits                  |
| $1/    | V             | $                                    | $\log_2 | V   | $   | Maximum surprise for uniform distribution |

**Key property**: Self-information is additive for independent events: $I(A \cap B) = I(A) + I(B)$.

### 3.2 Entropy

The **expected surprise** (average self-information) over a distribution:

$$\boxed{H(P) = -\sum_{t \in V} P(t) \log_2 P(t) = \mathbb{E}_{t \sim P}[I(t)]}$$

| Distribution                     | Entropy      | Interpretation                               |
| -------------------------------- | ------------ | -------------------------------------------- | ------- | --- | ------ | ------------------- |
| Point mass (one token has $P=1$) | 0 bits       | No uncertainty                               |
| Uniform over $                   | V            | $ tokens                                     | $\log_2 | V   | $ bits | Maximum uncertainty |
| English text (Shannon 1951)      | ≈1.0–1.5 BPC | Natural language has structure               |
| Typical LLM next-token           | ≈2–4 bits    | Model exploits context to reduce uncertainty |

**Shannon's source coding theorem**: Entropy is the minimum average number of bits needed to encode samples from $P$. No lossless compression can beat $H(P)$ bits per symbol on average.

### 3.3 Cross-Entropy

Expected surprise when using model distribution $Q$ to encode samples from true distribution $P$:

$$\boxed{H(P, Q) = -\sum_{t \in V} P(t) \log_2 Q(t) = \mathbb{E}_{t \sim P}[-\log_2 Q(t)]}$$

**Properties**:

- $H(P, Q) \geq H(P)$ always (Gibbs' inequality)
- $H(P, Q) = H(P)$ if and only if $P = Q$
- The gap $H(P, Q) - H(P) = D_{KL}(P \| Q) \geq 0$

**Training connection**: The LLM training objective is to minimise $H(P_{\text{data}}, P_{\text{model}})$ — make the model's distribution match the data distribution.

In practice, $P_{\text{data}}$ is empirical (one-hot at the true next token), so:

$$H(P_{\text{data}}, P_{\text{model}}) = -\log_2 P_{\text{model}}(t_{\text{true}} \mid \text{context})$$

This is simply the negative log-probability of the correct token — the cross-entropy loss.

### 3.4 KL Divergence

The **extra bits** needed to use $Q$ instead of $P$:

$$\boxed{D_{KL}(P \| Q) = \sum_{t \in V} P(t) \log \frac{P(t)}{Q(t)} = H(P, Q) - H(P)}$$

| Property       | Value                                                                                      |
| -------------- | ------------------------------------------------------------------------------------------ |
| Non-negative   | $D_{KL}(P \| Q) \geq 0$ always (Gibbs' inequality)                                         |
| Zero iff equal | $D_{KL}(P \| Q) = 0 \iff P = Q$                                                            |
| Not symmetric  | $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$ in general                                            |
| Not a metric   | Violates triangle inequality                                                               |
| Forward KL     | $D_{KL}(P_{\text{data}} \| P_{\text{model}})$: mean-seeking; MLE minimises this            |
| Reverse KL     | $D_{KL}(P_{\text{model}} \| P_{\text{data}})$: mode-seeking; used in variational inference |

**Training equivalence**: Minimising $D_{KL}(P_{\text{data}} \| P_{\text{model}})$ is equivalent to maximising log-likelihood, which is equivalent to minimising cross-entropy loss.

### 3.5 Perplexity

Exponentiated cross-entropy; the standard LM evaluation metric:

$$\boxed{\text{PPL} = 2^{H(P_{\text{data}}, P_{\text{model}})} = 2^{-\frac{1}{N}\sum_{i=1}^{N} \log_2 P(t_i \mid t_{<i})}}$$

Or equivalently using natural logarithm:

$$\text{PPL} = \exp\!\left(-\frac{1}{N} \sum_{i=1}^{N} \ln P(t_i \mid t_{<i})\right)$$

**Interpretation**: Perplexity = the effective vocabulary size the model is "choosing from" at each step.

| Model             | PPL (WikiText-103) | Effective branching factor      |
| ----------------- | ------------------ | ------------------------------- | ------ | -------------------------- |
| Uniform random ($ | V                  | =50K$)                          | 50,000 | Guessing from entire vocab |
| 5-gram LM         | ~70                | Better than random              |
| GPT-2 (1.5B)      | ~18                | Fairly confident                |
| GPT-3 (175B)      | ~10                | Often narrows to ~10 candidates |
| LLaMA 3 70B       | ~3–5               | Very confident; almost knows    |
| Perfect model     | 1                  | Always predicts correctly       |

Lower is better. Halving perplexity ≈ roughly doubling model quality in terms of predictive power.

### 3.6 Bits Per Character / Bits Per Byte

Normalise cross-entropy by characters or bytes instead of tokens:

$$\text{BPC} = \frac{-\sum_{i} \log_2 P(t_i \mid t_{<i})}{\text{total characters}}, \quad \text{BPB} = \frac{-\sum_{i} \log_2 P(t_i \mid t_{<i})}{\text{total bytes}}$$

| Metric     | Depends on tokeniser?                     | Use case                               |
| ---------- | ----------------------------------------- | -------------------------------------- |
| Perplexity | Yes — different tokenisers → incomparable | Model comparison within same tokeniser |
| BPC        | No — normalised by raw characters         | Cross-lingual comparison               |
| BPB        | No — normalised by raw bytes              | Tokeniser-independent comparison       |

English text entropy ≈ 1.0–1.5 BPC (Shannon 1951). Good LLMs in 2024–2026 achieve ~0.7–1.0 BPC on standard benchmarks.

**Conversion**: $\text{PPL}_{\text{token}} = 2^{\text{BPT} \times \text{avg chars per token}}$ where BPT is bits-per-token.

---

## 4. Training Objective — Maximum Likelihood

### 4.1 Log-Likelihood

Given corpus $\mathcal{D} = (t_1, t_2, \ldots, t_N)$, the log-likelihood under model parameters $\theta$:

$$\mathcal{L}(\theta) = \sum_{i=1}^{N} \log P_\theta(t_i \mid t_1, \ldots, t_{i-1})$$

Since each $\log P_\theta(t_i \mid t_{<i}) \leq 0$ (probabilities are at most 1), the log-likelihood is always non-positive. We want to maximise it (make it less negative).

Maximising $\mathcal{L}(\theta)$ = minimising negative log-likelihood (NLL) = minimising cross-entropy with the empirical distribution.

### 4.2 Maximum Likelihood Estimation (MLE)

$$\boxed{\theta^* = \arg\max_\theta \sum_{i=1}^{N} \log P_\theta(t_i \mid t_{<i})}$$

This is the standard training objective for all autoregressive LLMs.

| Property            | Description                                                                            |
| ------------------- | -------------------------------------------------------------------------------------- |
| **Unbiased**        | With infinite data, MLE recovers the true distribution                                 |
| **Consistent**      | As $N \to \infty$, $\hat{\theta} \to \theta^*$ in probability                          |
| **Efficient**       | Achieves the Cramér-Rao lower bound asymptotically                                     |
| **Teacher forcing** | During training, always condition on true previous tokens, not model's own predictions |

**Teacher forcing**: At training time, position $i$ receives the true tokens $t_1, \ldots, t_{i-1}$ as input, not the model's own predictions. This enables parallel training of all positions simultaneously but creates an **exposure bias**: at inference time, the model must condition on its own (possibly incorrect) predictions.

### 4.3 Cross-Entropy Loss in Practice

Per-token loss at position $i$:

$$\mathcal{L}_i = -\log P_\theta(t_i \mid t_{<i}) = -\log \text{softmax}(z_i)_{t_i} = -z_{t_i} + \log \sum_{v \in V} \exp(z_v)$$

The decomposition reveals two parts:

- $-z_{t_i}$: the logit for the correct token (want this large)
- $\log \sum_v \exp(z_v)$: the log partition function (normalisation constant)

Batch loss: mean over all tokens in batch

$$\mathcal{L} = \frac{1}{|B|} \sum_{i \in B} \mathcal{L}_i$$

**Computational cost**: The log-sum-exp term requires computing $\exp(z_v)$ for all $|V|$ tokens — for $|V| = 128K$, this is expensive. This is why vocabulary parallelism matters (§11.4).

### 4.4 Gradient of Cross-Entropy Loss

One of the most elegant results in machine learning. The gradient of the cross-entropy loss with respect to logits $z$:

$$\boxed{\frac{\partial \mathcal{L}_i}{\partial z_v} = P_\theta(v \mid t_{<i}) - \mathbb{1}[v = t_i] = \hat{p}_v - y_v}$$

where $\hat{p}_v = \text{softmax}(z)_v$ is the predicted probability and $y_v$ is the one-hot target.

**Interpretation**: The gradient = predicted probability − true probability.

| Token $v$           | $\hat{p}_v$ | $y_v$ | Gradient $\hat{p}_v - y_v$ | Effect                              |
| ------------------- | ----------- | ----- | -------------------------- | ----------------------------------- |
| Correct token $t_i$ | 0.7         | 1     | −0.3                       | Push logit **up** (decrease loss)   |
| Wrong token $A$     | 0.2         | 0     | +0.2                       | Push logit **down**                 |
| Wrong token $B$     | 0.1         | 0     | +0.1                       | Push logit **down** (less strongly) |

**Key properties**:

- Gradient is zero only when $\hat{p} = y$ (model perfectly predicts the target)
- Gradient magnitude is bounded: $|\hat{p}_v - y_v| \leq 1$
- Gradients sum to zero: $\sum_v (\hat{p}_v - y_v) = 1 - 1 = 0$
- This is why cross-entropy + softmax is the universal LM training loss: simple, elegant, numerically stable gradient

### 4.5 Label Smoothing

Soften one-hot targets to prevent overconfidence:

Instead of target $(0, \ldots, 1, \ldots, 0)$, use $\left(\frac{\varepsilon}{|V|}, \ldots, 1 - \varepsilon + \frac{\varepsilon}{|V|}, \ldots, \frac{\varepsilon}{|V|}\right)$

$$\mathcal{L}_{\text{smooth}} = (1 - \varepsilon) \cdot \mathcal{L}_{\text{CE}} + \varepsilon \cdot H_{\text{uniform}}$$

| Parameter           | Typical value        | Effect                                                         |
| ------------------- | -------------------- | -------------------------------------------------------------- |
| $\varepsilon = 0$   | Standard CE          | Model can become arbitrarily confident                         |
| $\varepsilon = 0.1$ | Standard choice      | Prevents logits from growing unboundedly; improves calibration |
| $\varepsilon = 0.2$ | Aggressive smoothing | May hurt accuracy on easy predictions                          |

Used in the original Transformer, T5, PaLM, and many modern models. Forces the model to maintain non-zero probability on all tokens, which acts as implicit regularisation.

### 4.6 Next-Token Prediction vs Masked LM

| Aspect              | Causal LM (GPT-style)                     | Masked LM (BERT-style)                                               |
| ------------------- | ----------------------------------------- | -------------------------------------------------------------------- |
| **Prediction**      | Predict each token from left context only | Predict randomly masked tokens (15%) from full bidirectional context |
| **Context**         | Unidirectional (causal mask)              | Bidirectional                                                        |
| **Training signal** | $N$ losses per sequence of length $N$     | ~$0.15N$ losses per sequence                                         |
| **Generation**      | Natural: sample left-to-right             | Unnatural: requires iterative refinement                             |
| **Examples**        | GPT-2/3/4, LLaMA, Claude, Gemini          | BERT, RoBERTa, DeBERTa                                               |
| **Formula**         | $\sum_i \log P(t_i \mid t_{<i})$          | $\sum_{i \in \text{masked}} \log P(t_i \mid t_{\setminus i})$        |

**UL2 (Tay et al. 2022)**: Unifies both with a mixture of denoising objectives (causal, prefix, span corruption). Used in PaLM 2 and Gemini.

---

## 5. Decoding Strategies

### 5.1 Greedy Decoding

$$t_i^* = \arg\max_{t \in V} P(t \mid t_{<i})$$

Always pick the highest-probability token. Fast, deterministic, but often suboptimal.

**Problem**: Locally optimal $\neq$ globally optimal. Greedy decoding can miss high-probability sequences because it commits to each choice without look-ahead.

**Example**: After "The most important thing is", greedy might pick "to" ($P=0.32$) but the sequence "the" → "fact" → "that" has higher joint probability.

### 5.2 Beam Search

Maintain $K$ candidate sequences (beams) at each step:

$$\text{score}(\mathbf{t}) = \frac{1}{|\mathbf{t}|^\alpha} \sum_{i=1}^{|\mathbf{t}|} \log P(t_i \mid t_{<i})$$

| Parameter                 | Description                          | Typical value |
| ------------------------- | ------------------------------------ | ------------- |
| $K$ (beam width)          | Number of candidates to maintain     | 4–10          |
| $\alpha$ (length penalty) | Prevents bias toward short sequences | 0.6–0.8       |
| $K = 1$                   | Reduces to greedy decoding           | —             |
| $K \to \infty$            | Exhaustive search (intractable)      | —             |

Beam search is standard for machine translation and summarisation. Less used for open-ended generation where diversity matters (beam search tends to produce generic, repetitive text).

### 5.3 Temperature Sampling

Divide logits by temperature $\tau$ before softmax:

$$\boxed{P_\tau(t \mid t_{<i}) = \frac{\exp(z_t / \tau)}{\sum_v \exp(z_v / \tau)}}$$

| Temperature $\tau$ | Effect                     | Entropy     | Use case                    |
| ------------------ | -------------------------- | ----------- | --------------------------- | --- | --------------- |
| $\tau \to 0$       | Approaches greedy (argmax) | $\to 0$     | Maximum confidence          |
| $\tau = 0.3$       | Sharp, focused             | Low         | Factual QA, code completion |
| $\tau = 1.0$       | Standard (no change)       | Original    | General purpose             |
| $\tau = 0.7$       | Slightly sharpened         | Moderate    | Creative writing, dialogue  |
| $\tau = 1.5$       | Flattened, more random     | High        | Brainstorming, exploration  |
| $\tau \to \infty$  | Approaches uniform random  | $\to \log_2 | V                           | $   | Pure randomness |

**Mathematical insight**: Temperature scales the entropy of the output distribution. If $H_1$ is the entropy at $\tau = 1$, then $H_\tau \approx H_1 / \tau$ for small perturbations.

### 5.4 Top-k Sampling

Sample only from the $k$ highest-probability tokens:

$$P_k(t) \propto P(t) \cdot \mathbb{1}[t \in \text{top-}k(P)]$$

Renormalise after filtering: probabilities of the top-$k$ tokens are rescaled to sum to 1.

**Problem**: Fixed $k$ ignores the shape of the distribution:

- When distribution is peaked: $k = 50$ includes many near-zero probability tokens (adds noise)
- When distribution is flat: $k = 50$ may exclude tokens with meaningful probability (truncates too aggressively)

Typical $k$: 40–100. Often combined with temperature.

### 5.5 Top-p (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability $\geq p$:

$$\mathcal{V}_p = \arg\min_{S \subseteq V} |S| \quad \text{s.t.} \quad \sum_{t \in S} P(t) \geq p$$

**Algorithm**:

1. Sort tokens by probability descending: $P(t_{(1)}) \geq P(t_{(2)}) \geq \ldots$
2. Find smallest $k^*$ such that $\sum_{j=1}^{k^*} P(t_{(j)}) \geq p$
3. Sample from $\{t_{(1)}, \ldots, t_{(k^*)}\}$ with renormalised probabilities

| $p$  | Effect                                            | Typical use              |
| ---- | ------------------------------------------------- | ------------------------ |
| 0.5  | Very focused; only most likely tokens             | Conservative generation  |
| 0.9  | Standard; good balance of diversity and coherence | General purpose          |
| 0.95 | Slightly more diverse                             | Creative tasks           |
| 1.0  | No filtering (full distribution)                  | Same as temperature-only |

**Advantage over top-k**: Adaptive vocabulary size. When the model is confident (peaked distribution), the nucleus is small. When uncertain (flat distribution), the nucleus is large. Outperforms top-k in practice (Holtzman et al. 2020).

### 5.6 Min-p Sampling

Filter tokens whose probability falls below a fraction of the top token's probability:

$$\mathcal{V}_{\text{min-p}} = \left\{t : P(t) \geq p_{\min} \cdot \max_v P(v)\right\}$$

**Example**: If $\max_v P(v) = 0.6$ and $p_{\min} = 0.1$, keep all tokens with $P(t) \geq 0.06$.

**Advantage over top-p**: Scales threshold relative to the model's confidence level. When the model is very confident ($\max P$ is high), the absolute threshold is high → tight filtering. When uncertain ($\max P$ is low), threshold is low → more diversity.

Increasingly adopted in 2024–2026; better than top-p for long generations where confidence varies substantially across positions.

### 5.7 Typical Sampling

Sample tokens whose self-information is close to the entropy of the distribution (Meister et al. 2023):

$$\mathcal{V}_{\text{typ}} = \left\{t : \left|-\log P(t) - H(P)\right| \leq \delta\right\}$$

**Intuition**: "Typical" tokens are neither too surprising (underconfident tail) nor too predictable (overconfident peak). Information theory says most samples from $P$ have self-information near $H(P)$ — this is the **asymptotic equipartition property**.

Avoids both the too-deterministic trap of greedy decoding and the too-random trap of high temperature. Strong results on open-ended generation.

### 5.8 Contrastive / Speculative Methods

**Contrastive decoding** (Li et al. 2022): Subtract logits of a smaller "amateur" model from the "expert" model:

$$z_{\text{CD}}(t) = \log P_{\text{expert}}(t) - \log P_{\text{amateur}}(t)$$

Amplifies what the large model knows that the small model doesn't. Reduces hallucination and repetition.

**Speculative decoding** (Leviathan et al. 2023): Use a small draft model to generate $K$ candidate tokens; verify all $K$ in parallel with the large target model.

| Aspect            | Detail                                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| **Draft**         | Small model generates $K$ tokens quickly (e.g., $K = 5$)                                             |
| **Verify**        | Large model evaluates all $K$ tokens in one forward pass                                             |
| **Accept/reject** | Accept token $k$ if $P_{\text{large}}(t_k) \geq r \cdot P_{\text{draft}}(t_k)$ where $r \sim U(0,1)$ |
| **Distribution**  | Produces exact same distribution as the target model                                                 |
| **Speedup**       | 2–3× inference speedup; standard in production 2024–2026                                             |

### 5.9 Repetition and Frequency Penalties

| Penalty            | Formula                                                            | Effect                                                    |
| ------------------ | ------------------------------------------------------------------ | --------------------------------------------------------- |
| Repetition penalty | $z_t \leftarrow z_t / r$ for previously seen $t$                   | Divide logit by $r > 1$; discourages exact repetition     |
| Frequency penalty  | $z_t \leftarrow z_t - \alpha \cdot \text{count}(t)$                | Subtract proportional to count; penalises frequent tokens |
| Presence penalty   | $z_t \leftarrow z_t - \beta \cdot \mathbb{1}[\text{count}(t) > 0]$ | Flat penalty if token appeared at all                     |

Used in OpenAI API, Anthropic API, and HuggingFace `generate()`. Typical values: $r = 1.1$, $\alpha = 0.5$, $\beta = 0.6$.

---

## 6. Scaling Laws

### 6.1 Neural Scaling Laws (Kaplan et al. 2020)

Loss scales as a power law in model parameters $N$, dataset size $D$, and compute $C$:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

| Finding           | Detail                                                                                                     |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| Power law         | Loss decreases as $N^{-0.076}$; $10\times$ parameters → loss decreases by factor $10^{0.076} \approx 1.19$ |
| No saturation     | No signs of diminishing returns at available scale                                                         |
| Smooth            | Loss curve is remarkably smooth; enables extrapolation                                                     |
| Kaplan allocation | Given fixed compute, allocate most to parameters, less to data                                             |

### 6.2 Chinchilla Scaling Laws (Hoffmann et al. 2022)

Reanalysis of Kaplan: **optimal $N$ and $D$ scale equally** with compute:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

**Chinchilla rule**: ~20 tokens per parameter for compute-optimal training.

| Model       | Parameters | Tokens trained | Tokens/param | Chinchilla?            |
| ----------- | ---------- | -------------- | ------------ | ---------------------- |
| GPT-3       | 175B       | 300B           | 1.7          | ✗ Under-trained        |
| Chinchilla  | 70B        | 1.4T           | 20           | ✓ Compute-optimal      |
| LLaMA 2 70B | 70B        | 2T             | 29           | Over-trained           |
| LLaMA 3 8B  | 8B         | 15T            | 1,875        | Massively over-trained |
| LLaMA 3 70B | 70B        | 15T            | 214          | Massively over-trained |

**2024–2026 shift**: "Over-training" is deliberate. Chinchilla optimises for training compute, but inference compute (deployment) is what matters in production. Smaller models trained on far more data achieve better loss per inference FLOP.

### 6.3 Inference-Optimal Scaling

For fixed inference budget, smaller model trained on more tokens is better:

- Inference cost $\propto N$ (parameters processed per token)
- Training cost $\propto N \times D$ (total FLOPs)
- Prefer smaller $N$ with larger $D$ to reach same loss with cheaper per-token inference

This is the **LLaMA philosophy**: train beyond Chinchilla-optimal for cheaper deployment. A LLaMA 3 8B trained on 15T tokens matches a 30B+ model trained at Chinchilla-optimal, but costs 4× less to serve.

### 6.4 Emergent Abilities and Phase Transitions

Some capabilities appear suddenly at scale; not predicted by smooth loss curves:

| Capability            | Emergence threshold (est.) | Note                                    |
| --------------------- | -------------------------- | --------------------------------------- |
| Arithmetic (3-digit)  | ~$10^{22}$ FLOPs           | Suddenly jumps from 0% to ~80% accuracy |
| Chain-of-thought      | ~$10^{23}$ FLOPs           | Requires explicit prompting             |
| Instruction following | ~$10^{22}$ FLOPs           | Enables zero-shot task performance      |
| Code generation       | ~$10^{23}$ FLOPs           | Requires substantial training data      |

**Debate** (Schaeffer et al. 2023): Emergence may be an artefact of discontinuous evaluation metrics (accuracy, exact match). The underlying loss curve is always smooth. When measured with continuous metrics (log-probability, Brier score), "emergence" appears gradual.

### 6.5 Test-Time Compute Scaling (2024–2026)

New scaling axis: scale inference compute, not just training compute.

$$L(C_{\text{test}}) \propto C_{\text{test}}^{-\alpha_{\text{test}}}$$

| Method                         | How it works                                        | Speedup/quality trade-off                |
| ------------------------------ | --------------------------------------------------- | ---------------------------------------- |
| Best-of-N                      | Generate $N$ responses; pick best (by reward model) | Linear compute cost; diminishing returns |
| Beam search over CoT           | Search over reasoning chains                        | Better for structured problems           |
| MCTS (Monte Carlo Tree Search) | Tree search over reasoning steps                    | Used by AlphaCode, o1                    |
| Verifier-guided search         | Use verifier to prune bad reasoning paths           | DeepSeek-R1 approach                     |

OpenAI o1 (2024) and DeepSeek-R1 (2025) demonstrated that test-time compute scaling can be as powerful as parameter scaling for reasoning tasks.

---

## 7. Calibration and Uncertainty

### 7.1 What Is Calibration?

A model is **calibrated** if its stated probabilities match empirical frequencies:

$$P(\text{correct} \mid P(t^*) = p) = p \quad \forall \, p \in [0, 1]$$

When the model says it's 70% confident, it should be right 70% of the time. LLMs are often **overconfident** (high probability on wrong answers).

### 7.2 Expected Calibration Error (ECE)

$$\boxed{\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|}$$

**Algorithm**:

1. Bin predictions by confidence level (e.g., $B = 10$ bins: $[0, 0.1), [0.1, 0.2), \ldots$)
2. For each bin $b$: compute average accuracy and average confidence
3. ECE = weighted average of $|\text{accuracy} - \text{confidence}|$ per bin

| ECE   | Interpretation          |
| ----- | ----------------------- |
| 0%    | Perfect calibration     |
| 1–3%  | Well calibrated         |
| 5–10% | Moderate miscalibration |
| > 10% | Poorly calibrated       |

### 7.3 Temperature Scaling for Calibration

Post-hoc calibration: find optimal temperature $\tau^*$ on validation set:

$$\tau^* = \arg\min_\tau \mathcal{L}_{\text{NLL}}\left(\text{softmax}(z / \tau), y\right)$$

Simple, effective, and does not change model accuracy — only adjusts confidence. One scalar parameter to optimise, typically via L-BFGS on a validation set.

Platt scaling: learn a linear transformation $z' = az + b$ for calibration. More flexible but risk of overfitting.

### 7.4 Overconfidence in LLMs

| Source                 | Mechanism                                                          |
| ---------------------- | ------------------------------------------------------------------ |
| MLE training           | Pushes model to maximise probability of training data; can overfit |
| Label smoothing        | Partially corrects by preventing logits from growing unboundedly   |
| RLHF                   | Can increase or decrease calibration depending on reward model     |
| Verbalized uncertainty | "I'm 95% sure" — often poorly calibrated in practice               |

### 7.5 Epistemic vs Aleatoric Uncertainty

| Type          | Source                        | Reducible?           | Example                                     |
| ------------- | ----------------------------- | -------------------- | ------------------------------------------- |
| **Aleatoric** | Genuine ambiguity in language | No                   | "The best programming language is \_\_\_"   |
| **Epistemic** | Lack of knowledge/data        | Yes (with more data) | "The population of Liechtenstein is \_\_\_" |

LLMs conflate both types; cannot cleanly separate in practice. **Conformal prediction** (2024): distribution-free calibration guarantees for LLMs — provides prediction sets with guaranteed coverage probability regardless of the model's internal calibration.

---

## 8. Conditional Language Models

### 8.1 Conditional vs Unconditional LM

| Type              | Distribution                 | Example                                                       |
| ----------------- | ---------------------------- | ------------------------------------------------------------- |
| **Unconditional** | $P(t_1, \ldots, t_n)$        | Distribution over all text; web crawl LM                      |
| **Conditional**   | $P(t_1, \ldots, t_n \mid c)$ | Distribution given context $c$: instruction, image, documents |

All instruction-tuned LLMs are conditional LMs: $P(\text{response} \mid \text{instruction})$.

### 8.2 Prompt as Conditioning

Prepend prompt tokens $p_1, \ldots, p_m$ to generation:

$$P(t_1, \ldots, t_n \mid p_1, \ldots, p_m) = \prod_{i=1}^{n} P(t_i \mid p_1, \ldots, p_m, t_1, \ldots, t_{i-1})$$

The prompt shifts the distribution — well-designed prompts steer the model toward high-probability desired outputs.

**Prompt engineering**: finding $\mathbf{p}$ that maximises $P(\text{desired output} \mid \mathbf{p})$. This is a combinatorial optimisation problem over the discrete space of token sequences.

### 8.3 Bayes' Theorem in Decoding

Noisy channel model reverses conditioning direction:

$$P(\text{output} \mid \text{input}) \propto P(\text{input} \mid \text{output}) \times P(\text{output})$$

Used in speech recognition (acoustic model × language model). Contrastive decoding uses Bayesian intuition: expert vs amateur models.

### 8.4 Classifier-Free Guidance (CFG)

Adapted from diffusion models to LLMs:

$$\log P_{\text{guided}}(t) = (1 + w) \log P(t \mid c) - w \log P(t)$$

| $w$     | Effect                                           |
| ------- | ------------------------------------------------ |
| $w = 0$ | Standard conditional generation                  |
| $w > 0$ | Amplifies conditioning signal; reduces diversity |
| $w < 0$ | Inverse guidance; increases diversity            |

Reduces diversity but increases adherence to prompt. Used in text-to-image (Stable Diffusion), some text generation systems. Requires two forward passes: one conditional, one unconditional.

### 8.5 Retrieval-Augmented Generation (RAG)

Condition on retrieved documents $\mathcal{D}$:

$$P(t_i \mid t_{<i}, \mathcal{D}) = P(t_i \mid t_{<i}, d_1, \ldots, d_k)$$

Documents are prepended to or interleaved with the context; the model attends to them during generation.

| Benefit                      | Mechanism                                              |
| ---------------------------- | ------------------------------------------------------ |
| Reduces hallucination        | Conditions on factual documents                        |
| Updatable knowledge          | Swap document index without retraining                 |
| Attributable                 | Can cite source documents                              |
| Probabilistic interpretation | Retrieved documents shift prior toward factual content |

---

## 9. RLHF and Reward-Based Probability

### 9.1 From Pretraining to Alignment

| Stage                            | Distribution                             | Objective                                      |
| -------------------------------- | ---------------------------------------- | ---------------------------------------------- |
| **Pretraining**                  | $P_{\text{PT}}(t \mid \text{context})$   | Model internet text distribution via MLE       |
| **SFT** (Supervised Fine-Tuning) | $P_{\text{SFT}}(t \mid \text{context})$  | Fine-tune on (instruction, response) pairs     |
| **RLHF**                         | $P_{\text{RLHF}}(t \mid \text{context})$ | Align with human preferences via reward signal |

### 9.2 Reward Model

Learn scalar reward $R(\text{prompt}, \text{response})$ from human preference pairs.

**Bradley-Terry model**: probability human prefers response $A$ over $B$:

$$P(A \succ B) = \frac{\exp(R(A))}{\exp(R(A)) + \exp(R(B))} = \sigma(R(A) - R(B))$$

where $\sigma$ is the sigmoid function. Train the reward model to maximise log-likelihood of observed human preferences:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(w, l) \sim \mathcal{D}}\left[\log \sigma(R(w) - R(l))\right]$$

### 9.3 PPO Objective (RLHF)

$$\boxed{\mathcal{L}_{\text{RLHF}} = \mathbb{E}\left[R(t_{1:n})\right] - \beta \cdot D_{KL}(P_\theta \| P_{\text{ref}})}$$

Maximise expected reward while staying close to the reference (SFT) model.

| Component            | Role                                                           |
| -------------------- | -------------------------------------------------------------- |
| $R(t_{1:n})$         | Reward for generated sequence; from reward model               |
| $\beta \cdot D_{KL}$ | KL penalty: prevents "reward hacking" (model drifting too far) |
| $P_{\text{ref}}$     | Reference policy = SFT model; anchor point                     |
| $\beta = 0$          | Pure RL; no constraint → reward hacking                        |
| $\beta \to \infty$   | No movement from SFT; ignores reward                           |
| $\beta$ typical      | 0.01–0.1; balances reward and distributional constraint        |

### 9.4 DPO — Direct Preference Optimisation

Rafailov et al. (2023) showed that the RLHF objective has a closed-form solution that bypasses the explicit reward model.

$$\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\!\left(\beta \log \frac{P_\theta(t_w)}{P_{\text{ref}}(t_w)} - \beta \log \frac{P_\theta(t_l)}{P_{\text{ref}}(t_l)}\right)\right]}$$

where $t_w$ = preferred (winner) response, $t_l$ = dispreferred (loser) response.

**Derivation sketch**:

1. Start from RLHF objective: $\max_\theta \mathbb{E}[R(t)] - \beta D_{KL}(P_\theta \| P_{\text{ref}})$
2. Optimal policy has closed form: $P^*(t) = \frac{P_{\text{ref}}(t) \exp(R(t)/\beta)}{Z}$
3. Invert: $R(t) = \beta \log \frac{P^*(t)}{P_{\text{ref}}(t)} + \beta \log Z$
4. Substitute into Bradley-Terry: the partition function $Z$ cancels
5. Result: DPO loss depends only on log-probability ratios — no reward model needed

| Property            | PPO (RLHF)                                 | DPO                                                       |
| ------------------- | ------------------------------------------ | --------------------------------------------------------- |
| Reward model        | Required (separate model)                  | Not needed                                                |
| Training loop       | RL loop: generate → score → update         | Standard supervised: forward-backward on preference pairs |
| Stability           | Often unstable; requires careful tuning    | Stable; standard training                                 |
| Memory              | 4 models: policy, reference, reward, value | 2 models: policy, reference                               |
| Dominance 2023–2026 | Still used at scale                        | Default method                                            |

### 9.5 GRPO — Group Relative Policy Optimisation

DeepSeek (2024): Removes the value function (critic) from PPO, reducing memory and compute.

**Key idea**: Estimate the baseline from a group of sampled responses rather than a learned value function:

$$A_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}$$

where $G$ is the group size (typically 8–64 samples per prompt) and $r_i$ is the reward for sample $i$.

| Advantage | Detail                                                             |
| --------- | ------------------------------------------------------------------ |
| No critic | Saves ~50% memory vs PPO                                           |
| Simple    | Group normalisation replaces learned value function                |
| Scalable  | Enables RL training on reasoning tasks at scale                    |
| Adoption  | Used by DeepSeek-R1; widely adopted 2025–2026 for reasoning models |

### 9.6 The Alignment Tax

Fine-tuning for alignment shifts $P_\theta$ away from $P_{\text{PT}}$:

| Concern                        | Reality (2024–2026)                                                       |
| ------------------------------ | ------------------------------------------------------------------------- |
| "Alignment reduces capability" | Tax much smaller than initially feared; careful RLHF preserves capability |
| "RLHF causes mode collapse"    | KL constraint prevents this; DPO is more stable                           |
| "Human feedback doesn't scale" | RLAIF (AI feedback), Constitutional AI scale better                       |
| "Alignment is fragile"         | Robustness improving; but adversarial attacks remain possible             |

---

## 10. Evaluation Metrics

### 10.1 Perplexity (Revisited)

Standard held-out test set perplexity. Caveats:

| Issue                | Detail                                                                                  |
| -------------------- | --------------------------------------------------------------------------------------- |
| Tokeniser dependence | Different tokenisers → incomparable PPL values                                          |
| Sliding window       | For long documents: stride $s < $ context length to avoid beginning-of-sequence effects |
| BPB alternative      | Bits-per-byte: tokeniser-independent; preferred for cross-model comparison              |

### 10.2 BLEU Score

Modified n-gram precision between generated and reference text:

$$\text{BLEU} = \text{BP} \cdot \exp\!\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

where BP = brevity penalty, $p_n$ = modified n-gram precision, $w_n = 1/N$.

| Component                | Formula                                                                     |
| ------------------------ | --------------------------------------------------------------------------- |
| Modified precision $p_n$ | Clip each n-gram count to max count in reference                            |
| Brevity penalty BP       | $\min(1, e^{1 - r/c})$ where $r$ = reference length, $c$ = candidate length |
| Typical $N$              | 4 (BLEU-4 most common)                                                      |

Standard for machine translation. Increasingly criticised for open-ended generation; does not correlate well with human judgement for LLMs.

### 10.3 ROUGE

Recall-oriented n-gram overlap; standard for summarisation:

$$\text{ROUGE-N} = \frac{\sum_{\text{ref}} \sum_{\text{gram}_n \in \text{ref}} \text{count\_match}(\text{gram}_n)}{\sum_{\text{ref}} \sum_{\text{gram}_n \in \text{ref}} \text{count}(\text{gram}_n)}$$

| Variant | What it measures           |
| ------- | -------------------------- |
| ROUGE-1 | Unigram recall             |
| ROUGE-2 | Bigram recall              |
| ROUGE-L | Longest common subsequence |

### 10.4 BERTScore

Compute cosine similarity between contextual embeddings of generated and reference tokens. Soft matching: semantically similar words score well even if not identical.

Better correlation with human judgement than BLEU/ROUGE. Sensitive to choice of BERT model and layer.

### 10.5 LLM-as-Judge (2023–2026)

Use a powerful LLM (GPT-4, Claude) to score or rank model outputs.

| Benchmark      | Setup                                      | Note                                |
| -------------- | ------------------------------------------ | ----------------------------------- |
| MT-Bench       | GPT-4 scores 1–10 on multi-turn dialogue   | First major LLM-as-judge benchmark  |
| AlpacaEval 2.0 | GPT-4 pairwise comparison vs reference     | Length-controlled variant preferred |
| Arena-Hard     | Derived from Chatbot Arena; automated      | Highly correlated with human Elo    |
| LiveBench      | Contamination-free; auto-updated questions | Launched 2024                       |

**Known biases**: position bias (prefers first response), verbosity bias (prefers longer text), self-preference (model favours its own outputs). Mitigation: randomise order, use length-controlled variants.

### 10.6 Benchmark Suites

| Benchmark      | Tests                  | Metric     | Top 2026 score |
| -------------- | ---------------------- | ---------- | -------------- |
| MMLU           | 57-subject knowledge   | Accuracy   | ~90%           |
| HellaSwag      | Commonsense completion | Accuracy   | ~97%           |
| HumanEval      | Code generation        | pass@1     | ~90%           |
| GSM8K          | Grade school math      | Accuracy   | ~97%           |
| MATH           | Competition math       | Accuracy   | ~75%           |
| ARC-Challenge  | Science QA             | Accuracy   | ~96%           |
| TruthfulQA     | Factual accuracy       | % truthful | ~75%           |
| BIG-Bench Hard | Reasoning              | Accuracy   | ~90%           |
| RULER          | Long context retrieval | Accuracy   | ~85%           |
| LiveBench      | Contamination-free     | Accuracy   | ~70%           |

---

## 11. Numerical Stability and Implementation

### 11.1 Log-Sum-Exp Trick

Direct computation of $\sum_v \exp(z_v)$ overflows for large $z_v$ (e.g., $z_v > 709$ in float64).

**Stable computation**:

$$\log \sum_v \exp(z_v) = a + \log \sum_v \exp(z_v - a), \quad a = \max(z)$$

Log-softmax (combining both):

$$\log \text{softmax}(z)_k = z_k - \left(a + \log \sum_v \exp(z_v - a)\right)$$

This shifts all exponents so the largest is 0, preventing overflow while maintaining precision.

### 11.2 Numerical Loss of Cross-Entropy

| Approach                          | Problem                                                                             | Solution                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| Compute $\log(\text{softmax}(z))$ | Underflow: softmax can produce values so small they round to 0; $\log(0) = -\infty$ | Use log-softmax directly                                                   |
| Separate softmax then log         | Two passes; numerical error compounds                                               | Fused log-softmax kernel                                                   |
| PyTorch practice                  | —                                                                                   | `F.cross_entropy(logits, targets)` = log-softmax + NLL; numerically stable |

**Rule**: Never compute `log(softmax(z))` in two steps. Always use `log_softmax(z)` directly.

### 11.3 Mixed Precision Training

| Precision    | Where used                                                        | Why                                                               |
| ------------ | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| FP32         | Softmax, loss computation, weight master copy                     | Numerical stability during normalisation                          |
| BF16         | Forward pass activations, gradients                               | Memory and speed efficiency; BF16 has same exponent range as FP32 |
| FP16         | Some activations (with loss scaling)                              | Higher precision than BF16 in mantissa but smaller exponent range |
| Loss scaling | Multiply loss by $2^{16}$ before backward; divide gradients after | Prevents FP16 underflow in small gradients                        |

### 11.4 Vocabulary Parallelism

For large $|V|$ (128K+): the LM head $W \in \mathbb{R}^{d \times |V|}$ is split across GPUs.

| Aspect             | Detail                                                                    |
| ------------------ | ------------------------------------------------------------------------- | --- | --------------------------------------------- | --- | --------- |
| Tensor parallel    | Each GPU holds $                                                          | V   | /k$ vocabulary; computes partial logits       |
| Partition function | Requires all-reduce across GPUs: $\log \sum_v \exp(z_v)$ needs all logits |
| Communication      | One all-reduce per forward pass for softmax normalisation                 |
| Memory saving      | $                                                                         | V   | \times d / k$ parameters per GPU instead of $ | V   | \times d$ |

---

## 12. Common Mistakes

| #   | Mistake                                 | Why It's Wrong                                                               | Fix                                                  |
| --- | --------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------- |
| 1   | "Higher probability = better output"    | Probability reflects training distribution, not quality; probable ≠ good     | Use reward model or human eval for quality           |
| 2   | "PPL is comparable across models"       | PPL depends on tokeniser; different vocab sizes → different PPL              | Use BPB for fair comparison                          |
| 3   | "Greedy decoding is optimal"            | Locally optimal ≠ globally optimal; greedy misses high-probability sequences | Use beam search for accuracy; sampling for diversity |
| 4   | "Temperature 1.0 is always best"        | Optimal $\tau$ is task-dependent; factual tasks → lower $\tau$               | Tune $\tau$ per task and domain                      |
| 5   | "BLEU correlates with quality"          | BLEU poorly correlates with human judgement for LLMs                         | Use LLM-as-judge or human eval                       |
| 6   | "MLE = learning the true distribution"  | MLE on finite data overfits; never sees all of $P_{\text{data}}$             | Regularise; label smoothing; diverse data            |
| 7   | "Scaling always improves all tasks"     | Some tasks saturate; some degrade (inverse scaling)                          | Evaluate task-specifically                           |
| 8   | "RLHF removes hallucination"            | RLHF improves perceived quality but may increase confident hallucination     | Combine with RAG and factuality training             |
| 9   | "Log probabilities are calibrated"      | LLMs are typically overconfident; $\log P \neq$ true probability             | Apply temperature scaling for calibration            |
| 10  | "Perplexity measures reasoning ability" | PPL measures next-token prediction only; reasoning needs separate benchmarks | Use task-specific benchmarks (GSM8K, MATH, etc.)     |

---

## 13. Exercises

1. **Chain rule decomposition** — Write $P(\text{"the cat sat"})$ as a product of conditional probabilities; identify what each factor means and what context it conditions on.
2. **Perplexity calculation** — Given log-probs $[-2.3, -1.1, -3.5, -0.8, -2.0]$ (nats), compute PPL; convert to bits-per-token; discuss what PPL means for this example.
3. **Temperature effect** — Given logits $[3.0, 1.0, 0.5, -1.0]$, compute softmax at $\tau = 0.5, 1.0, 2.0$; plot distributions; compare entropy at each temperature.
4. **Top-p sampling** — Sort token probabilities $[0.35, 0.25, 0.20, 0.12, 0.05, 0.03]$; find nucleus for $p = 0.9$; compute renormalised distribution.
5. **Cross-entropy gradient** — For logits $z = [2.0, 1.0, 0.5]$ and target $t^* = 0$: compute loss; compute $\partial \mathcal{L}/\partial z$; verify gradient formula $\hat{p} - y$.
6. **Scaling law extrapolation** — Given $L(N) = (N_c/N)^{0.076}$, if a 7B model achieves $L = 2.1$ nats, estimate loss for 70B and 700B models; plot the power law.
7. **DPO by hand** — For $\beta = 0.1$, $\log P_\theta(t_w)/P_{\text{ref}}(t_w) = 0.5$, $\log P_\theta(t_l)/P_{\text{ref}}(t_l) = -0.3$: compute DPO loss; interpret the gradient direction.
8. **Calibration check** — Given 100 predictions with confidence 0.9: if model is correct 72 times, compute ECE for this bin; determine if model is over- or underconfident.

---

## 14. Why This Matters for AI (2026 Perspective)

| Aspect                 | Impact                                                                                            |
| ---------------------- | ------------------------------------------------------------------------------------------------- |
| **Generation quality** | Decoding strategy directly determines output diversity, coherence, and factuality                 |
| **Alignment**          | RLHF and DPO reshape $P_\theta$ toward human-preferred outputs; probability is the lever          |
| **Hallucination**      | Hallucinations are high-probability but false outputs; calibration is the fix                     |
| **Reasoning models**   | Test-time scaling (o1, R1) uses probability to guide search over reasoning chains                 |
| **Evaluation**         | Perplexity, BPB, LLM-as-judge — all rooted in probability theory                                  |
| **Cost**               | Speculative decoding exploits probability structure for 2–3× inference speedup                    |
| **Safety**             | Probability of harmful outputs must be minimised; RLHF KL constraint bounds distribution shift    |
| **Interpretability**   | Logit lens, probing, activation patching all operate on probability outputs                       |
| **Multimodal**         | Same probabilistic framework extends to image, audio, video tokens                                |
| **Agents**             | Action selection in LLM agents is sampling from $P(\text{action} \mid \text{state}, \text{goal})$ |

---

## Conceptual Bridge

Language modelling probability is the **mathematical heart of every LLM**. Everything before this section — tokenisation, embedding, attention, positional encoding — produces a hidden state $h_n$. Everything after — decoding, alignment, evaluation — consumes the probability distribution $P(\cdot \mid \text{context})$ that this section defines.

The chain rule decomposition makes sequences tractable. Cross-entropy loss and its elegant gradient drive training. Decoding strategies control the quality-diversity trade-off. Scaling laws predict performance. And RLHF/DPO reshape the distribution toward human values.

**Next**: [Training at Scale](../06-Training-at-Scale/notes.md) — how gradient descent navigates the loss landscape to find $\theta^*$ that minimises cross-entropy, including optimisers, learning rate schedules, distributed training, and the mathematics of convergence at billion-parameter scale.

```
… → [Transformer] → hₙ → [LM Head] → logits → [Softmax] → P → [Cross-Entropy] → L → [Backprop] → ∇θ
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                        THIS section
```

---

[← Positional Encodings](../04-Positional-Encodings/notes.md) | [Home](../../README.md) | [Training at Scale →](../06-Training-at-Scale/notes.md)
