[<- Positional Encodings](../04-Positional-Encodings/notes.md) | [Home](../../README.md) | [Training at Scale ->](../06-Training-at-Scale/notes.md)

---

# Language Model Probability Math

An LLM is a machine for assigning probabilities to token continuations. The transformer stack builds a contextual vector; this section explains how that vector becomes a normalized next-token distribution, how training improves that distribution, and how decoding turns it back into text.

## Overview

Language-model probability is the center of the LLM learning loop. During training, the model receives a true prefix and is penalized when the observed next token has low probability. During evaluation, the same probabilities produce negative log-likelihood, cross-entropy, perplexity, calibration curves, and answer scores. During generation, the probabilities are transformed by decoding rules such as temperature, top-k, top-p, beam search, or constraints.

The most important idea is simple but deep:

$$
P(t_{1:n}) = \\prod_{i=1}^{n} P(t_i \\mid t_{<i}).
$$

This chain-rule factorization is exact. Autoregressive language modeling becomes a practical learning problem because a neural network can estimate each conditional distribution.

## Prerequisites

- Conditional probability and the product rule
- Logarithms, exponentials, and gradients
- Entropy, KL divergence, and cross-entropy
- Vector and matrix shapes from embeddings and attention
- The previous LLM sections on tokenization, embeddings, attention, and positions

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Builds the probability machinery with executable toy LMs, stable softmax, cross-entropy gradients, perplexity, calibration, and decoding demos. |
| [exercises.ipynb](exercises.ipynb) | Gives ten practice problems with scaffolds and complete solutions for sequence probability, loss, decoding, calibration, and conditional scoring. |

## Learning Objectives

After this section, you should be able to:

- Define a language model as a distribution over token sequences with an EOS convention.
- Derive the autoregressive factorization from the probability chain rule.
- Convert hidden states into logits, logits into probabilities, and probabilities into log-likelihoods.
- Implement numerically stable softmax and log-softmax.
- Derive the cross-entropy gradient $p-y$ with respect to logits.
- Compute entropy, KL divergence, cross-entropy, perplexity, bits per token, and length-normalized scores.
- Explain why decoding changes the sampling distribution without changing the trained model.
- Diagnose common probability bugs in LLM training and evaluation code.

## Table of Contents

1. [Language Models as Distributions](#1-language-models-as-distributions)
   - 1.1 [Finite vocabulary and EOS](#11-finite-vocabulary-and-eos)
   - 1.2 [Probability over finite strings](#12-probability-over-finite-strings)
   - 1.3 [Conditional next-token distributions](#13-conditional-nexttoken-distributions)
   - 1.4 [Support and impossible events](#14-support-and-impossible-events)
   - 1.5 [Why next-token prediction is enough](#15-why-nexttoken-prediction-is-enough)
2. [Autoregressive Factorization](#2-autoregressive-factorization)
   - 2.1 [Chain rule derivation](#21-chain-rule-derivation)
   - 2.2 [Teacher forcing](#22-teacher-forcing)
   - 2.3 [Causal masking](#23-causal-masking)
   - 2.4 [Sequence log probability](#24-sequence-log-probability)
   - 2.5 [Prefix scoring](#25-prefix-scoring)
3. [Logits, Softmax, and Log-Sum-Exp](#3-logits-softmax-and-logsumexp)
   - 3.1 [LM head](#31-lm-head)
   - 3.2 [Softmax normalization](#32-softmax-normalization)
   - 3.3 [Shift invariance](#33-shift-invariance)
   - 3.4 [Stable log-softmax](#34-stable-logsoftmax)
   - 3.5 [Temperature](#35-temperature)
4. [Maximum Likelihood and Cross-Entropy](#4-maximum-likelihood-and-crossentropy)
   - 4.1 [Dataset likelihood](#41-dataset-likelihood)
   - 4.2 [Negative log-likelihood](#42-negative-loglikelihood)
   - 4.3 [Cross-entropy](#43-crossentropy)
   - 4.4 [Gradient with respect to logits](#44-gradient-with-respect-to-logits)
   - 4.5 [Padding masks](#45-padding-masks)
5. [Entropy, KL, Perplexity, and Bits](#5-entropy-kl-perplexity-and-bits)
   - 5.1 [Entropy](#51-entropy)
   - 5.2 [Cross-entropy decomposition](#52-crossentropy-decomposition)
   - 5.3 [Perplexity](#53-perplexity)
   - 5.4 [Bits per token](#54-bits-per-token)
   - 5.5 [Tokenizer caveat](#55-tokenizer-caveat)
6. [Sequence Scoring and Calibration](#6-sequence-scoring-and-calibration)
   - 6.1 [Length bias](#61-length-bias)
   - 6.2 [Average log probability](#62-average-log-probability)
   - 6.3 [Calibration](#63-calibration)
   - 6.4 [Expected calibration error](#64-expected-calibration-error)
   - 6.5 [Temperature scaling](#65-temperature-scaling)
7. [Decoding as Distribution Transformation](#7-decoding-as-distribution-transformation)
   - 7.1 [Greedy decoding](#71-greedy-decoding)
   - 7.2 [Sampling](#72-sampling)
   - 7.3 [Top-k filtering](#73-topk-filtering)
   - 7.4 [Nucleus top-p filtering](#74-nucleus-topp-filtering)
   - 7.5 [Beam search and length penalty](#75-beam-search-and-length-penalty)
8. [From Count Models to Neural LMs](#8-from-count-models-to-neural-lms)
   - 8.1 [N-gram models](#81-ngram-models)
   - 8.2 [Smoothing](#82-smoothing)
   - 8.3 [Neural probabilistic LM](#83-neural-probabilistic-lm)
   - 8.4 [Recurrent LM](#84-recurrent-lm)
   - 8.5 [Transformer LM](#85-transformer-lm)
9. [Conditional Generation and Constraints](#9-conditional-generation-and-constraints)
   - 9.1 [Prompt conditioning](#91-prompt-conditioning)
   - 9.2 [RAG conditioning](#92-rag-conditioning)
   - 9.3 [Logit bias and constraints](#93-logit-bias-and-constraints)
   - 9.4 [Guidance by logit mixing](#94-guidance-by-logit-mixing)
   - 9.5 [Safety filters](#95-safety-filters)
10. [Diagnostics and Learning Practice](#10-diagnostics-and-learning-practice)
   - 10.1 [Normalization checks](#101-normalization-checks)
   - 10.2 [Likelihood sanity checks](#102-likelihood-sanity-checks)
   - 10.3 [Distribution shift](#103-distribution-shift)
   - 10.4 [Generation versus evaluation](#104-generation-versus-evaluation)
   - 10.5 [Bridge to training at scale](#105-bridge-to-training-at-scale)
11. [Conditional Entropy and Information](#11-conditional-entropy-and-information-between-context-and-token)
12. [Sampling Theory](#12-sampling-theory-tilting-truncation-renormalization)
13. [Sequence-Level Mismatch](#13-sequence-level-mismatch-exposure-bias-and-search-error)
14. [Large-Vocabulary Softmax](#14-large-vocabulary-softmax-approximations)
15. [Targets Beyond One-Hot](#15-targets-beyond-one-hot-label-smoothing-and-distillation-targets)
16. [Evaluation Beyond Perplexity](#16-evaluation-beyond-a-single-perplexity-number)
17. [Probability Under Tokenization](#17-probability-under-tokenization)
18. [Worked End-to-End Example](#18-worked-end-to-end-example)
19. [Derivations in Full](#19-derivations-in-full)
20. [Count Models Worked in Full](#20-count-models-worked-in-full)
21. [Decoding Transformations Compared](#21-decoding-transformations-compared-on-one-distribution)
22. [Numerical Cases](#22-numerical-cases-that-break-naive-implementations)
23. [Bridge Within Chapter 15](#23-from-this-section-to-the-rest-of-chapter-15)
24. [Ten Worked Problems](#24-ten-worked-problems-with-full-solutions)
25. [Diagnostic Protocol](#25-a-diagnostic-protocol-for-probability-bugs)
26. [Glossary](#26-glossary-of-probability-terms-used-in-this-chapter)
27. [Monte Carlo Estimation](#27-estimating-likelihood-quantities-by-monte-carlo)
28. [Comparing Two Language Models](#28-comparing-two-language-models-honestly)
29. [Historical Line](#29-historical-line-in-one-view)
30. [Softmax Jacobian Worked Numerically](#30-a-softmax-jacobian-worked-numerically)
31. [Choosing a Score for a Task](#31-choosing-a-score-for-a-task-decision-table)
32. [Formula Summary](#32-formula-summary)

---

## Notation

| Symbol | Meaning |
| --- | --- |
| $V$ | Vocabulary of ordinary tokens |
| $V_\\mathrm{ext}$ | Vocabulary plus special tokens such as EOS |
| $t_i$ | Token at position $i$ |
| $t_{<i}$ | Prefix before position $i$ |
| $h_i$ | Transformer hidden state at position $i$ |
| $z_i$ | Logit vector for the next-token distribution |
| $p_\\theta(\\cdot \\mid t_{<i})$ | Model distribution over the next token |
| $y_i$ | One-hot target vector |
| $m_i$ | Mask indicating whether a token contributes to loss |

The standard decoder-only training path is:

```text
tokens -> embeddings + positions -> causal transformer -> hidden state -> LM head -> logits -> log-softmax -> NLL
```

## 1. Language Models as Distributions

This part studies language models as distributions as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Finite vocabulary and EOS](#1-finite-vocabulary-and-eos) | make sequence probability normalize by ending strings explicitly | $V_\mathrm{ext}=V\cup\{\mathrm{EOS}\}$ |
| [Probability over finite strings](#1-probability-over-finite-strings) | a language model assigns mass to every complete token string | $P(t_{1:n},\mathrm{EOS})$ |
| [Conditional next-token distributions](#1-conditional-nexttoken-distributions) | generation uses one normalized categorical distribution at each prefix | $p_\theta(t_{i}\mid t_{<i})$ |
| [Support and impossible events](#1-support-and-impossible-events) | zero probability is a dangerous modeling choice | $p_\theta(v\mid h)>0$ after softmax |
| [Why next-token prediction is enough](#1-why-nexttoken-prediction-is-enough) | the chain rule converts local predictions into a full joint model | $P(t_{1:n})=\prod_i P(t_i\mid t_{<i})$ |

### 1.1 Finite vocabulary and EOS

**Main idea.** Make sequence probability normalize by ending strings explicitly.

The useful formula is:

$$V_\mathrm{ext}=V\cup\{\mathrm{EOS}\}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.2 Probability over finite strings

**Main idea.** A language model assigns mass to every complete token string.

The useful formula is:

$$P(t_{1:n},\mathrm{EOS})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.3 Conditional next-token distributions

**Main idea.** Generation uses one normalized categorical distribution at each prefix.

The useful formula is:

$$p_\theta(t_{i}\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.4 Support and impossible events

**Main idea.** Zero probability is a dangerous modeling choice.

The useful formula is:

$$p_\theta(v\mid h)>0$ after softmax$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 1.5 Why next-token prediction is enough

**Main idea.** The chain rule converts local predictions into a full joint model.

The useful formula is:

$$P(t_{1:n})=\prod_i P(t_i\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 2. Autoregressive Factorization

This part studies autoregressive factorization as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Chain rule derivation](#2-chain-rule-derivation) | no independence assumption is needed | $P(t_{1:n})=P(t_1)\prod_{i=2}^n P(t_i\mid t_{1:i-1})$ |
| [Teacher forcing](#2-teacher-forcing) | training conditions on the true prefix, not sampled model history | $-\log p_\theta(t_i^\star\mid t_{<i}^\star)$ |
| [Causal masking](#2-causal-masking) | attention must not leak future targets into the conditional | $M_{ij}=0$ for $j\le i$ and $-\infty$ otherwise |
| [Sequence log probability](#2-sequence-log-probability) | products become sums for stable scoring | $\log P(t_{1:n})=\sum_i \log p_\theta(t_i\mid t_{<i})$ |
| [Prefix scoring](#2-prefix-scoring) | prompt likelihood and answer likelihood are different objects | $\log p_\theta(y\mid x)=\sum_j \log p_\theta(y_j\mid x,y_{<j})$ |

### 2.1 Chain rule derivation

**Main idea.** No independence assumption is needed.

The useful formula is:

$$P(t_{1:n})=P(t_1)\prod_{i=2}^n P(t_i\mid t_{1:i-1})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.2 Teacher forcing

**Main idea.** Training conditions on the true prefix, not sampled model history.

The useful formula is:

$$-\log p_\theta(t_i^\star\mid t_{<i}^\star)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is why training can be massively parallel while generation is sequential.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.3 Causal masking

**Main idea.** Attention must not leak future targets into the conditional.

The useful formula is:

$$M_{ij}=0$ for $j\le i$ and $-\infty$ otherwise$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.4 Sequence log probability

**Main idea.** Products become sums for stable scoring.

The useful formula is:

$$\log P(t_{1:n})=\sum_i \log p_\theta(t_i\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 2.5 Prefix scoring

**Main idea.** Prompt likelihood and answer likelihood are different objects.

The useful formula is:

$$\log p_\theta(y\mid x)=\sum_j \log p_\theta(y_j\mid x,y_{<j})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 3. Logits, Softmax, and Log-Sum-Exp

This part studies logits, softmax, and log-sum-exp as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [LM head](#3-lm-head) | the hidden state becomes one score per vocabulary token | $z=Wh+b$ |
| [Softmax normalization](#3-softmax-normalization) | logits are unnormalized log probabilities | $p_i=\exp(z_i)/\sum_j\exp(z_j)$ |
| [Shift invariance](#3-shift-invariance) | adding a constant to every logit changes no probability | $\mathrm{softmax}(z+c\mathbf{1})=\mathrm{softmax}(z)$ |
| [Stable log-softmax](#3-stable-logsoftmax) | subtract the maximum before exponentiating | $\log p_i=z_i-\operatorname{LSE}(z)$ |
| [Temperature](#3-temperature) | divide logits before softmax to control entropy | $p_i(\tau)=\mathrm{softmax}(z_i/\tau)$ |

### 3.1 LM head

**Main idea.** The hidden state becomes one score per vocabulary token.

The useful formula is:

$$z=Wh+b$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.2 Softmax normalization

**Main idea.** Logits are unnormalized log probabilities.

The useful formula is:

$$p_i=\exp(z_i)/\sum_j\exp(z_j)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is the final probability gate between hidden-state geometry and token choice.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.3 Shift invariance

**Main idea.** Adding a constant to every logit changes no probability.

The useful formula is:

$$\mathrm{softmax}(z+c\mathbf{1})=\mathrm{softmax}(z)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.4 Stable log-softmax

**Main idea.** Subtract the maximum before exponentiating.

The useful formula is:

$$\log p_i=z_i-\operatorname{LSE}(z)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 3.5 Temperature

**Main idea.** Divide logits before softmax to control entropy.

The useful formula is:

$$p_i(\tau)=\mathrm{softmax}(z_i/\tau)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 4. Maximum Likelihood and Cross-Entropy

This part studies maximum likelihood and cross-entropy as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Dataset likelihood](#4-dataset-likelihood) | training maximizes probability assigned to observed tokens | $\max_\theta \sum_{(x,y)}\log p_\theta(y\mid x)$ |
| [Negative log-likelihood](#4-negative-loglikelihood) | loss is surprise under the model | $\ell=-\log p_\theta(y^\star\mid h)$ |
| [Cross-entropy](#4-crossentropy) | one-hot labels reduce cross-entropy to NLL | $H(q,p_\theta)=-\sum_i q_i\log p_{\theta,i}$ |
| [Gradient with respect to logits](#4-gradient-with-respect-to-logits) | the key derivative is predicted minus target | $\nabla_z \ell=p-y$ |
| [Padding masks](#4-padding-masks) | only real target tokens should contribute to the average loss | $L=\sum_i m_i\ell_i/\sum_i m_i$ |

### 4.1 Dataset likelihood

**Main idea.** Training maximizes probability assigned to observed tokens.

The useful formula is:

$$\max_\theta \sum_{(x,y)}\log p_\theta(y\mid x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.2 Negative log-likelihood

**Main idea.** Loss is surprise under the model.

The useful formula is:

$$\ell=-\log p_\theta(y^\star\mid h)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.3 Cross-entropy

**Main idea.** One-hot labels reduce cross-entropy to nll.

The useful formula is:

$$H(q,p_\theta)=-\sum_i q_i\log p_{\theta,i}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.4 Gradient with respect to logits

**Main idea.** The key derivative is predicted minus target.

The useful formula is:

$$\nabla_z \ell=p-y$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This derivative is the reason cross-entropy is so convenient for transformer training.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 4.5 Padding masks

**Main idea.** Only real target tokens should contribute to the average loss.

The useful formula is:

$$L=\sum_i m_i\ell_i/\sum_i m_i$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 5. Entropy, KL, Perplexity, and Bits

This part studies entropy, kl, perplexity, and bits as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Entropy](#5-entropy) | intrinsic uncertainty of a distribution | $H(p)=-\sum_x p(x)\log p(x)$ |
| [Cross-entropy decomposition](#5-crossentropy-decomposition) | model loss equals data entropy plus mismatch | $H(q,p)=H(q)+D_\mathrm{KL}(q\Vert p)$ |
| [Perplexity](#5-perplexity) | exponentiated average NLL | $\mathrm{PPL}=\exp\left(\frac{1}{N}\sum_i-\log p_i\right)$ |
| [Bits per token](#5-bits-per-token) | use base-2 logs for compression interpretation | $\mathrm{BPT}=\mathrm{NLL}_{\mathrm{nat}}/\log 2$ |
| [Tokenizer caveat](#5-tokenizer-caveat) | perplexity depends on the tokenization unit | $\mathrm{BPB}$ and $\mathrm{BPC}$ are more comparable |

### 5.1 Entropy

**Main idea.** Intrinsic uncertainty of a distribution.

The useful formula is:

$$H(p)=-\sum_x p(x)\log p(x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.2 Cross-entropy decomposition

**Main idea.** Model loss equals data entropy plus mismatch.

The useful formula is:

$$H(q,p)=H(q)+D_\mathrm{KL}(q\Vert p)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.3 Perplexity

**Main idea.** Exponentiated average nll.

The useful formula is:

$$\mathrm{PPL}=\exp\left(\frac{1}{N}\sum_i-\log p_i\right)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is the most common intrinsic language-model score, but it must be interpreted with tokenizer and dataset context.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.4 Bits per token

**Main idea.** Use base-2 logs for compression interpretation.

The useful formula is:

$$\mathrm{BPT}=\mathrm{NLL}_{\mathrm{nat}}/\log 2$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 5.5 Tokenizer caveat

**Main idea.** Perplexity depends on the tokenization unit.

The useful formula is:

$$\mathrm{BPB}$ and $\mathrm{BPC}$ are more comparable$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 6. Sequence Scoring and Calibration

This part studies sequence scoring and calibration as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Length bias](#6-length-bias) | raw log probability favors shorter strings | $\log p(y\mid x)$ decreases with length |
| [Average log probability](#6-average-log-probability) | length-normalized scores compare answers of different lengths | $\frac{1}{|y|}\sum_j \log p(y_j\mid x,y_{<j})$ |
| [Calibration](#6-calibration) | confidence should match empirical correctness | $P(\mathrm{correct}\mid \hat p=c)\approx c$ |
| [Expected calibration error](#6-expected-calibration-error) | bin confidence gaps to summarize miscalibration | $\mathrm{ECE}=\sum_b\frac{|B_b|}{n}|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$ |
| [Temperature scaling](#6-temperature-scaling) | post-hoc calibration rescales logits without changing class order | $z'=z/\tau$ |

### 6.1 Length bias

**Main idea.** Raw log probability favors shorter strings.

The useful formula is:

$$\log p(y\mid x)$ decreases with length$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.2 Average log probability

**Main idea.** Length-normalized scores compare answers of different lengths.

The useful formula is:

$$\frac{1}{|y|}\sum_j \log p(y_j\mid x,y_{<j})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.3 Calibration

**Main idea.** Confidence should match empirical correctness.

The useful formula is:

$$P(\mathrm{correct}\mid \hat p=c)\approx c$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.4 Expected calibration error

**Main idea.** Bin confidence gaps to summarize miscalibration.

The useful formula is:

$$\mathrm{ECE}=\sum_b\frac{|B_b|}{n}|\mathrm{acc}(B_b)-\mathrm{conf}(B_b)|$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 6.5 Temperature scaling

**Main idea.** Post-hoc calibration rescales logits without changing class order.

The useful formula is:

$$z'=z/\tau$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 7. Decoding as Distribution Transformation

This part studies decoding as distribution transformation as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Greedy decoding](#7-greedy-decoding) | choose the highest-probability next token | $t_i=\arg\max_v p(v\mid t_{<i})$ |
| [Sampling](#7-sampling) | draw from the categorical distribution | $t_i\sim p_\theta(\cdot\mid t_{<i})$ |
| [Top-k filtering](#7-topk-filtering) | keep only the k highest probability tokens | $S_k=\operatorname{TopK}(p,k)$ |
| [Nucleus top-p filtering](#7-nucleus-topp-filtering) | keep the smallest set whose mass exceeds p | $\sum_{v\in S}p(v)\ge p_\mathrm{nuc}$ |
| [Beam search and length penalty](#7-beam-search-and-length-penalty) | approximate sequence MAP with multiple partial hypotheses | $s(y)=\log p(y\mid x)/|y|^\alpha$ |

### 7.1 Greedy decoding

**Main idea.** Choose the highest-probability next token.

The useful formula is:

$$t_i=\arg\max_v p(v\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.2 Sampling

**Main idea.** Draw from the categorical distribution.

The useful formula is:

$$t_i\sim p_\theta(\cdot\mid t_{<i})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.3 Top-k filtering

**Main idea.** Keep only the k highest probability tokens.

The useful formula is:

$$S_k=\operatorname{TopK}(p,k)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.4 Nucleus top-p filtering

**Main idea.** Keep the smallest set whose mass exceeds p.

The useful formula is:

$$\sum_{v\in S}p(v)\ge p_\mathrm{nuc}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** This is the default practical compromise between deterministic decoding and unrestricted sampling.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 7.5 Beam search and length penalty

**Main idea.** Approximate sequence map with multiple partial hypotheses.

The useful formula is:

$$s(y)=\log p(y\mid x)/|y|^\alpha$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 8. From Count Models to Neural LMs

This part studies from count models to neural lms as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [N-gram models](#8-ngram-models) | approximate history by the last n minus one tokens | $P(t_i\mid t_{<i})\approx P(t_i\mid t_{i-n+1:i-1})$ |
| [Smoothing](#8-smoothing) | reserve probability for unseen events | $P_\alpha(w\mid h)=\frac{c(h,w)+\alpha}{c(h)+\alpha |V|}$ |
| [Neural probabilistic LM](#8-neural-probabilistic-lm) | learn distributed word representations and probability together | $e_w\in\mathbb{R}^d$ |
| [Recurrent LM](#8-recurrent-lm) | compress history into a state | $h_i=f(h_{i-1},x_i)$ |
| [Transformer LM](#8-transformer-lm) | causal self-attention computes context-dependent hidden states in parallel | $p_\theta(t_i\mid t_{<i})=\mathrm{softmax}(Wh_i+b)$ |

### 8.1 N-gram models

**Main idea.** Approximate history by the last n minus one tokens.

The useful formula is:

$$P(t_i\mid t_{<i})\approx P(t_i\mid t_{i-n+1:i-1})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.2 Smoothing

**Main idea.** Reserve probability for unseen events.

The useful formula is:

$$P_\alpha(w\mid h)=\frac{c(h,w)+\alpha}{c(h)+\alpha |V|}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.3 Neural probabilistic LM

**Main idea.** Learn distributed word representations and probability together.

The useful formula is:

$$e_w\in\mathbb{R}^d$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.4 Recurrent LM

**Main idea.** Compress history into a state.

The useful formula is:

$$h_i=f(h_{i-1},x_i)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 8.5 Transformer LM

**Main idea.** Causal self-attention computes context-dependent hidden states in parallel.

The useful formula is:

$$p_\theta(t_i\mid t_{<i})=\mathrm{softmax}(Wh_i+b)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 9. Conditional Generation and Constraints

This part studies conditional generation and constraints as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Prompt conditioning](#9-prompt-conditioning) | a prompt is the observed prefix in the conditional | $p_\theta(y\mid x)$ |
| [RAG conditioning](#9-rag-conditioning) | retrieved text changes the conditioning information | $p_\theta(y\mid x,r)$ |
| [Logit bias and constraints](#9-logit-bias-and-constraints) | decoding can restrict or reweight the support | $z'_v=z_v+\beta_v$ |
| [Guidance by logit mixing](#9-guidance-by-logit-mixing) | combine distributions in logit space when a control model exists | $z_\mathrm{guided}=z_\mathrm{base}+\lambda(z_\mathrm{cond}-z_\mathrm{base})$ |
| [Safety filters](#9-safety-filters) | post-processing policies are not the same as model probability | $\Pr(\mathrm{emit})$ may differ from $p_\theta$ |

### 9.1 Prompt conditioning

**Main idea.** A prompt is the observed prefix in the conditional.

The useful formula is:

$$p_\theta(y\mid x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.2 RAG conditioning

**Main idea.** Retrieved text changes the conditioning information.

The useful formula is:

$$p_\theta(y\mid x,r)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** Retrieval does not change the probability rules; it changes what information the conditional can see.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.3 Logit bias and constraints

**Main idea.** Decoding can restrict or reweight the support.

The useful formula is:

$$z'_v=z_v+\beta_v$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.4 Guidance by logit mixing

**Main idea.** Combine distributions in logit space when a control model exists.

The useful formula is:

$$z_\mathrm{guided}=z_\mathrm{base}+\lambda(z_\mathrm{cond}-z_\mathrm{base})$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 9.5 Safety filters

**Main idea.** Post-processing policies are not the same as model probability.

The useful formula is:

$$\Pr(\mathrm{emit})$ may differ from $p_\theta$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
## 10. Diagnostics and Learning Practice

This part studies diagnostics and learning practice as an operational object: something an LLM computes for every prefix during training, scoring, and generation. The formulas below are small enough to derive by hand, but they are also exactly the formulas used inside large decoder-only models.

| Subtopic | Core question | Formula |
| --- | --- | --- |
| [Normalization checks](#10-normalization-checks) | probabilities must sum to one at each prefix | $\sum_v p(v\mid h)=1$ |
| [Likelihood sanity checks](#10-likelihood-sanity-checks) | known continuations should score above random continuations | $\log p(y_\mathrm{good}\mid x)>\log p(y_\mathrm{bad}\mid x)$ |
| [Distribution shift](#10-distribution-shift) | low likelihood can indicate unfamiliar domain or bad tokenization | $p_\mathrm{train}\ne p_\mathrm{test}$ |
| [Generation versus evaluation](#10-generation-versus-evaluation) | sampling quality and held-out NLL measure different properties | $\arg\min \mathrm{NLL}$ is not always best user experience |
| [Bridge to training at scale](#10-bridge-to-training-at-scale) | the same loss drives distributed training and scaling laws | $L(C,N,D)$ |

### 10.1 Normalization checks

**Main idea.** Probabilities must sum to one at each prefix.

The useful formula is:

$$\sum_v p(v\mid h)=1$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.2 Likelihood sanity checks

**Main idea.** Known continuations should score above random continuations.

The useful formula is:

$$\log p(y_\mathrm{good}\mid x)>\log p(y_\mathrm{bad}\mid x)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.3 Distribution shift

**Main idea.** Low likelihood can indicate unfamiliar domain or bad tokenization.

The useful formula is:

$$p_\mathrm{train}\ne p_\mathrm{test}$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.4 Generation versus evaluation

**Main idea.** Sampling quality and held-out nll measure different properties.

The useful formula is:

$$\arg\min \mathrm{NLL}$ is not always best user experience$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.
### 10.5 Bridge to training at scale

**Main idea.** The same loss drives distributed training and scaling laws.

The useful formula is:

$$L(C,N,D)$$

The probability object should always be read with its conditioning context. A token is not likely or unlikely in isolation; it is likely or unlikely after a prefix, under a vocabulary, with a particular tokenizer and model state. This is why the same word can be high probability in one prompt and nearly impossible in another.

**Worked micro-example.** Suppose the current prefix is `The capital of France is`. A reasonable model should place more mass on `Paris` than on `banana`, but it should still maintain a normalized distribution over every token in the vocabulary. If the logits for three candidate tokens are $[5.0, 1.0, -2.0]$, the probability of the first token is large because exponentiation magnifies logit differences:

$$
p_1 = \frac{e^{5.0}}{e^{5.0}+e^{1.0}+e^{-2.0}}.
$$

The model is not storing a sentence list. It is using a conditional distribution whose parameters are generated from the prefix representation.

**Implementation check.** In code, check shapes and normalization. If logits have shape `(batch, time, vocab)`, then `softmax(logits, axis=-1)` should sum to one along the vocabulary axis for every batch and time position. When scoring labels, gather only the probability of the observed next token and mask padding positions before averaging.

**AI connection.** For LLMs, this is not decoration; it controls a concrete training, scoring, or decoding behavior.

**Common mistake.** Do not compare raw sequence probabilities across different output lengths without a length convention. Products of probabilities shrink as strings get longer, so a two-token answer often has a higher raw probability than a better twenty-token answer.

---

---

## 11. Conditional Entropy and Information Between Context and Token

### 11.1 Conditional entropy of the next token

**Definition.** For context $X=t_{<i}$ and next token $Y=t_i$,
$H(Y\mid X)=\mathbb{E}[-\log p(Y\mid X)]$.

- Example 1: after "The capital of France is", a good model puts most mass on " Paris", so the conditional entropy is near 0.1 nats.
- Example 2: after "She opened the", plausible continuations (door, book, window, letter) spread mass, so entropy is several nats.
- Example 3: for a uniform distribution over a 50,257-token vocabulary (GPT-2, Radford et al., 2019), $H=\ln 50257\approx10.82$ nats is the maximum possible value.
- Non-example 1: the marginal entropy $H(Y)$ of tokens in a corpus is not conditional entropy; it ignores context and is almost always larger.
- Non-example 2: the entropy of a hidden state $h_i$ is not defined by the formula above; $h_i$ is a deterministic vector, not the random variable $Y$.

**Theorem (conditioning reduces entropy).** $H(Y\mid X)\le H(Y)$, with equality iff $X$ and $Y$ are independent.
*Proof sketch.* $H(Y)-H(Y\mid X)=I(X;Y)$, and mutual information is a KL divergence $D_{\mathrm{KL}}(p(x,y)\|p(x)p(y))\ge0$ by Jensen's inequality applied to $-\log$. $\square$

**AI connection.** This is why context helps: every bit of mutual information between prefix and continuation is a bit the model need not "guess". Long-context models (e.g. RoPE extrapolation work following Su et al., 2021) are valuable exactly when distant context carries nonzero $I(X_{\text{far}};Y)$.

### 11.2 Mutual information as the value of context

**Definition.** $I(X;Y)=\sum_{x,y}p(x,y)\log\frac{p(x,y)}{p(x)p(y)}$.

Worked numbers: suppose $Y\in\{\text{yes},\text{no}\}$ is balanced ($H(Y)=0.693$ nats) and context $X$ predicts $Y$ with 90% accuracy symmetrically. Then $H(Y\mid X)\approx-0.9\ln0.9-0.1\ln0.1\approx0.325$ nats, so $I(X;Y)\approx0.368$ nats $\approx0.53$ bits: the context is worth about half a bit per token.

- Example: a retrieval passage that names the answer entity raises $I(X;Y)$ for the answer tokens in RAG (Lewis et al., 2020).
- Non-example: repeating the prompt back adds tokens but little mutual information about the *next* token beyond position 1.

### 11.3 Entropy rate of a text source

**Definition.** The entropy rate is $\mathcal{H}=\lim_{n\to\infty}\frac1n H(t_1,\dots,t_n)$ when the limit exists.
Shannon (1951) estimated the entropy rate of printed English at roughly 0.6–1.3 bits per character using human prediction experiments.
A model with bits-per-token $b$ on a tokenizer averaging $c$ characters per token is achieving $b/c$ bits per character, the tokenizer-free quantity of Section 15.5.

**Common mistake.** Reporting a lower per-token perplexity for a model with a *larger* vocabulary as if it were a better entropy rate. Larger vocabularies mechanically lower per-token NLL; bits per byte does not have this defect.

### 11.4 Pointwise mutual information and phrase association

$\mathrm{PMI}(x,y)=\log\frac{p(x,y)}{p(x)p(y)}$ measures how much more often $x,y$ co-occur than independence predicts.
Church and Hanks (1990) used PMI for word association; modern tokenizers inherit the idea: BPE merges (Sennrich et al., 2016) repeatedly join the most frequent adjacent pair, a count-based proxy for association.
PMI can be negative (repulsion), zero (independence), or positive (attraction); a softmax probability alone never shows the independence baseline, which is why PMI remains a useful diagnostic.

---

## 12. Sampling Theory: Tilting, Truncation, Renormalization

### 12.1 Ancestral sampling is exact

Sampling $t_i\sim p_\theta(\cdot\mid t_{<i})$ sequentially samples exactly from the joint $p_\theta(t_{1:n})$ by the chain rule of Section 2. There is no approximation in ancestral sampling itself; all distortion relative to the model comes from *modifying* each conditional before sampling.

### 12.2 Temperature as exponential tilting

Temperature sampling replaces $p_j\propto e^{z_j}$ by $p_j(\tau)\propto e^{z_j/\tau}=p_j^{1/\tau}/\sum_k p_k^{1/\tau}$, an exponential tilting (powering) of the base distribution.

- As $\tau\to0^+$, $p(\tau)$ concentrates on $\arg\max_j z_j$ (greedy decoding).
- As $\tau\to\infty$, $p(\tau)$ tends to uniform over the vocabulary.
- At $\tau=1$ the model distribution is unchanged.
Entropy $H(p(\tau))$ is monotonically non-decreasing in $\tau$ for $\tau>0$; Section 5 of the theory notebook verifies this numerically.

**Non-example.** Dividing *probabilities* (not logits) by $\tau$ and renormalizing is not temperature sampling; it is a much weaker linear rescaling.

### 12.3 Top-k and nucleus truncation, exactly

Top-$k$ (Fan et al., 2018) keeps the $k$ largest probabilities and renormalizes: $\tilde p_j=p_j/\sum_{i\in K}p_i$ for $j\in K$, else 0.
Nucleus/top-$p$ (Holtzman et al., 2020) keeps the smallest set $K$ with $\sum_{i\in K}p_i\ge p$.
Truncation always *lowers* entropy and always moves the sampling distribution away from $p_\theta$ in total variation by exactly the discarded mass $1-\sum_{i\in K}p_i$ before renormalization.

Worked example: for $p=(0.5,0.3,0.15,0.05)$, top-$p$ with $p=0.8$ keeps $\{0.5,0.3\}$ and renormalizes to $(0.625,0.375)$; the discarded mass is 0.20, so TV distance to the original is 0.20.

### 12.4 Why greedy text degenerates

Greedy decoding maximizes each conditional locally, which is neither joint MAP search nor typical sampling. Holtzman et al. (2020) showed human text has fluctuating per-token probability while greedy text loops into high-probability repetitions: once a phrase repeats, its continuation probability rises, a self-reinforcing fixed point of the greedy map.

---

## 13. Sequence-Level Mismatch: Exposure Bias and Search Error

### 13.1 Exposure bias

Teacher forcing trains on gold prefixes; generation conditions on the model's own prefixes. Bengio et al. (2015, scheduled sampling) named and addressed this train/test prefix-distribution mismatch by sometimes feeding model predictions during training.
Formally, training minimizes loss under $p_{\text{data}}(t_{<i})$ while decoding visits prefixes from $p_\theta(t_{<i})$; the gap compounds with length because each step can drift to prefixes never seen in training.

### 13.2 Search error versus model error

If the highest-probability sequence under $p_\theta$ is bad text, that is *model error*. If beam search fails to find that sequence, that is *search error*. Stahlberg and Byrne (2019) showed for neural machine translation that wider beams can find higher-probability but *worse* translations (often empty), proving the model's MAP sequence itself can be pathological — a model problem, not a search problem.

### 13.3 Minimum Bayes risk decoding

Instead of MAP, minimum Bayes risk decoding picks the candidate maximizing expected utility (e.g. similarity to other samples) under $p_\theta$. It uses the distribution as a distribution, not just its mode, and is the principled alternative when the mode is degenerate.

---

## 14. Large-Vocabulary Softmax Approximations

The LM head costs $O(|V|d)$ per token; with $|V|\approx 256{,}000$ (e.g. large modern vocabularies) this is a real fraction of compute.

### 14.1 Hierarchical softmax

Morin and Bengio (2005) factorize $p(y\mid h)$ over a binary tree over words, turning $O(|V|)$ normalization into $O(\log|V|)$ decisions. Each internal node is a binary classifier; the probability of a word is the product of path decisions, so the distribution remains exactly normalized.

### 14.2 Sampled softmax

Jean et al. (2015) train with the target plus a sampled subset of negatives, correcting with the sampling proposal probabilities. Training is cheaper, but the *evaluation* perplexity must still be computed with the full softmax or an unbiased correction, otherwise numbers are not comparable.

### 14.3 Adaptive softmax

Grave et al. (2017) give frequent words a full-size head and rare words smaller heads in frequency bands, exploiting Zipf's law: most tokens come from a small head vocabulary, so average cost falls while the model stays exactly normalized within each band's construction.

**Common mistake.** Benchmarking a sampled-softmax training loss against a full-softmax validation perplexity. They estimate different normalizers.

---

## 15. Targets Beyond One-Hot: Label Smoothing and Distillation Targets

### 15.1 Label smoothing as a changed target distribution

Label smoothing (Szegedy et al., 2016) replaces the one-hot target $y$ by $y'=(1-\varepsilon)y+\varepsilon/|V|$. The loss $-\sum_j y'_j\log p_j$ is still cross-entropy, but against a target with entropy floor $\varepsilon\log|V|$, so the optimum no longer drives logits to infinity.

### 15.2 What smoothing does to calibration

Müller et al. (2019) showed label smoothing improves expected calibration error while making the model's own probabilities a worse record of its knowledge for distillation: the teacher's dark knowledge is partially erased. **AI connection:** if you plan to distill (Hinton et al., 2015), train the teacher without (or with less) label smoothing.

### 15.3 Knowledge distillation targets

Distillation minimizes $D_{\mathrm{KL}}(p_{\text{teacher}}(\tau)\,\|\,p_{\text{student}}(\tau))$ at temperature $\tau>1$ (Hinton et al., 2015), transferring relative probabilities of *wrong* tokens — information a one-hot target cannot carry. This is Section 4's cross-entropy with a soft, model-generated target.

---

## 16. Evaluation Beyond a Single Perplexity Number

### 16.1 Bits per byte

Report $\text{bpb}=\frac{\sum_i -\log_2 p(t_i\mid t_{<i})}{\text{number of UTF-8 bytes}}$. It is tokenizer-independent and therefore the only likelihood number comparable across tokenizers (see Section 5.5).

### 16.2 Calibration for token predictions

A next-token model is calibrated at confidence $q$ if, among predictions made with max-probability $q$, the predicted token is correct a fraction $q$ of the time. Guo et al. (2017) showed modern neural networks are systematically overconfident and that a single temperature fitted on validation data (Section 6.5) fixes much of it.

### 16.3 Selective prediction

Report accuracy/coverage curves: abstain when max-probability is below a threshold. A model with slightly worse perplexity but better-ranked confidence can dominate on selective prediction, which is what deployed systems with a "I don't know" path actually use.

---

## 17. Probability Under Tokenization

### 17.1 Segmentation is a latent variable

A string $s$ can often be segmented several ways; a token-level model scores one deterministic segmentation $\tau(s)$, so $p_\theta(s):=p_\theta(\tau(s))$ depends on the tokenizer's greedy choices, not just the model. Two models with different tokenizers assign probabilities to *different event spaces*; only byte-normalized scores (16.1) are commensurable.

### 17.2 Boundary effects

Prompting with a partial word can force an unlikely first token because the continuation must begin mid-token-merge. This is a tokenization-boundary artifact of $p_\theta(\cdot\mid\tau(\text{prompt}))$, not evidence the model "doesn't know" the word; rephrasing the prompt to end at a token boundary changes the conditional distribution itself.

---

## 18. Worked End-to-End Example

Vocabulary $\{a,b,\text{EOS}\}$. Model conditionals: $p(a\mid\text{BOS})=0.6$, $p(b\mid\text{BOS})=0.3$, $p(\text{EOS}\mid\text{BOS})=0.1$; $p(b\mid a)=0.5$, $p(\text{EOS}\mid a)=0.5$; $p(\text{EOS}\mid b)=0.9$, $p(a\mid b)=0.1$.

1. $p(ab\,\text{EOS})=0.6\times0.5\times0.9=0.27$; $\log p=\ln0.27\approx-1.309$ nats.
2. Average NLL over the 3 tokens is $0.436$ nats, so token perplexity is $e^{0.436}\approx1.55$ and bits/token $=0.436/\ln2\approx0.63$.
3. Greedy from BOS picks $a$, then ties $b$/EOS at 0.5 — greedy needs a tie-breaking rule, illustrating that "the greedy sequence" is implementation-defined at ties.
4. Temperature $\tau=2$ at BOS: unnormalized $(\sqrt{0.6},\sqrt{0.3},\sqrt{0.1})=(0.775,0.548,0.316)$, normalized $\approx(0.473,0.334,0.193)$ — flatter, as 12.2 predicts.
5. Top-$p=0.8$ at BOS keeps $\{a,b\}$ (mass 0.9) and renormalizes to $(0.667,0.333)$.

Every number here is checkable in the theory notebook's checklist cell style: sums to one, logs match products, perplexity is $\exp(\text{mean NLL})$.

---


---

## 19. Derivations in Full

### 19.1 Softmax Jacobian

For $p=\mathrm{softmax}(z)$, $\frac{\partial p_i}{\partial z_j}=p_i(\delta_{ij}-p_j)$.
*Derivation.* Write $p_i=e^{z_i}/S$, $S=\sum_k e^{z_k}$. For $i=j$: $\partial p_i/\partial z_i=p_i-p_i^2=p_i(1-p_i)$. For $i\ne j$: $\partial p_i/\partial z_j=-e^{z_i}e^{z_j}/S^2=-p_ip_j$. Combining gives $p_i(\delta_{ij}-p_j)$. $\square$
Consequences: the Jacobian is symmetric, positive semidefinite, has rank $|V|-1$, and annihilates the all-ones vector — the infinitesimal version of shift invariance (Section 3.3).

### 19.2 Cross-entropy gradient, step by step

$L=-\log p_y=-z_y+\log\sum_k e^{z_k}$. Hence $\partial L/\partial z_j=-\delta_{jy}+p_j$, i.e. $\nabla_z L=p-y$.
This is the most-used gradient in LLM training: the error signal on every logit is exactly the probability error. A token already at $p_y=0.99$ contributes a gradient 100x smaller than one at $p_y=0.09$ with comparable other mass, which is why loss curves flatten even while rare-token errors persist.

### 19.3 Non-negativity of KL, with the equality condition

$D_{\mathrm{KL}}(p\|q)=\sum_j p_j\log(p_j/q_j)=-\sum_j p_j\log(q_j/p_j)\ge-\log\sum_j p_j(q_j/p_j)=-\log1=0$ by Jensen, since $-\log$ is convex. Equality holds iff $q_j/p_j$ is constant on the support of $p$, i.e. $p=q$. This one inequality underlies: conditioning reduces entropy (11.1), cross-entropy $\ge$ entropy (Section 5.2), and the fact that MLE minimizes an empirical KL (Section 4).

### 19.4 Why perplexity is monotone in mean NLL

Perplexity $=e^{\overline{\mathrm{NLL}}}$ is a strictly increasing reparametrization: it adds no information beyond mean NLL, it only changes units to "effective branching factor". Two models ranked by perplexity are ranked identically by mean NLL on the *same* tokenization — and by neither across tokenizations (Section 17.1).

### 19.5 Chain rule for entropy

$H(t_{1:n})=\sum_{i=1}^n H(t_i\mid t_{<i})$ by applying $H(X,Y)=H(X)+H(Y\mid X)$ inductively. Dividing by $n$ and taking limits gives the entropy rate of Section 11.3 as the limit of per-token conditional entropies — precisely the quantity a perfect LM's perplexity estimates.

---

## 20. Count Models Worked in Full

Corpus: `the cat sat`, `the cat slept`, `the dog sat` (with BOS/EOS).

| Bigram | Count | $p(\cdot\mid\text{the})$ | $p(\cdot\mid\text{cat})$ |
|---|---:|---:|---:|
| the->cat | 2 | 2/3 | - |
| the->dog | 1 | 1/3 | - |
| cat->sat | 1 | - | 1/2 |
| cat->slept | 1 | - | 1/2 |
| dog->sat | 1 | - | - |

$p(\text{the cat sat EOS}\mid\text{BOS})=1\times\frac23\times\frac12\times1=\frac13$ (taking $p(\text{EOS}\mid\text{sat})=1$ in this corpus).
The unseen sentence `the dog slept` gets probability $1\times\frac13\times0=0$ under the MLE bigram — the zero-probability failure that smoothing exists to fix.

### 20.1 Add-$\alpha$ smoothing on the same counts

With $|V|=6$ content/EOS outcomes and $\alpha=0.1$: $p_\alpha(\text{slept}\mid\text{dog})=\frac{0+0.1}{1+0.6}=0.0625$, while $p_\alpha(\text{sat}\mid\text{dog})=\frac{1.1}{1.6}=0.6875$. Every unseen continuation buys a small, explicit price paid by the seen ones.

### 20.2 Interpolation and backoff

Interpolated estimate: $\hat p(y\mid x)=\lambda_x p_{\text{bigram}}(y\mid x)+(1-\lambda_x)p_{\text{unigram}}(y)$. Kneser-Ney smoothing (Kneser and Ney, 1995) improves this by discounting counts and using continuation counts (in how many contexts does $y$ appear) for the lower order — still the reference smoothing in classical LM toolkits, and the conceptual ancestor of "reserve some mass for the unseen" in neural sampling filters.

### 20.3 Perplexity of the toy corpus

Under the MLE bigram, the held-out sentence `the cat slept EOS` has probability $1\times\frac23\times\frac12\times1=\frac13$ over 4 predicted tokens, so mean NLL $=\ln3/4\approx0.275$ nats and perplexity $=e^{0.275}\approx1.32$. Under add-0.1 smoothing the same sentence scores $\frac{2.1}{3.6}\times\frac{1.1}{2.6}\approx0.247$ joint, a *lower* likelihood here but a finite one for `the dog slept`, which MLE scores at exactly 0 (infinite perplexity). Smoothing trades likelihood on seen patterns for finiteness on unseen ones.

---

## 21. Decoding Transformations Compared on One Distribution

Base distribution over tokens A–E: $p=(0.40,0.25,0.20,0.10,0.05)$, entropy $H\approx1.42$ nats.

| Method | Resulting distribution | Entropy (nats) | Note |
|---|---|---:|---|
| Greedy | $(1,0,0,0,0)$ | 0.00 | Mode only; deterministic |
| Sample $\tau=1$ | $(0.40,0.25,0.20,0.10,0.05)$ | 1.42 | Exact model distribution |
| $\tau=0.5$ | $\propto p^2\approx(0.593,0.231,0.148,0.037,0.009)$ | 1.06 | Sharpened |
| $\tau=2$ | $\propto\sqrt p\approx(0.316,0.250,0.224,0.158,0.112)$ | 1.55 | Flattened toward uniform ($\ln5=1.61$) |
| Top-$k=2$ | $(0.615,0.385,0,0,0)$ | 0.67 | Discarded mass 0.35 |
| Top-$p=0.85$ | $(0.471,0.294,0.235,0,0)$ | 1.04 | Keeps A,B,C (mass 0.85) |

Reading the table: every filter trades entropy for concentration, and the *amount* of discarded mass — not the name of the method — sets the total-variation distance from the model. A top-$k$ that discards 0.35 distorts more than a top-$p$ that discards 0.15, whatever their labels.

### 21.1 Repetition penalties are logit surgery

A repetition penalty divides positive logits (or subtracts a penalty) for already-emitted tokens before softmax. It is therefore a *history-dependent logit bias* (Section 9.3), not a property of $p_\theta$; the same model with penalty 1.0 and 1.2 defines two different sequence distributions, and likelihood-based evaluation must use penalty 1.0.

### 21.2 Constrained decoding masks are conditional distributions

Setting logits of forbidden tokens to $-\infty$ and renormalizing samples from $p_\theta(\cdot\mid \text{allowed set})$ at each step. The resulting sequence distribution is *not* $p_\theta$ conditioned on the whole sequence being valid (that would require sequence-level renormalization); per-step masking can still walk into dead ends where every allowed continuation violates a global constraint — the standard argument for grammar-guided backtracking or constrained beam search in structured generation.

---

## 22. Numerical Cases That Break Naive Implementations

| Case | Naive result | Stable result | Lesson |
|---|---|---|---|
| $z=(1000,1001)$, softmax | $e^{1000}=\infty$, NaN | subtract max: $(0.269,0.731)$ | Always shift (Section 3.4) |
| $p_y=0$ in $-\log p_y$ | $\log0=-\infty$ loss | compute log-softmax directly, never $\log(\mathrm{softmax})$ | Loss must be fused |
| Mean over a padded batch of mostly pads | loss $\approx0$, looks "trained" | masked mean (Section 4.5) | Count real tokens in logs |
| bf16 logits rounded before loss | small-probability tokens collapse to ties | compute loss in fp32 | Keep the loss path in fp32 (cf. Section 8 of Training at Scale) |
| Summing probabilities then taking log for sequence score | underflow to 0 for long sequences | sum log-probabilities | Score in log space (Section 2.4) |

Each row is a real failure mode class seen in training-code reviews; the theory notebook's checklist cell encodes the first, second, and fifth as executable assertions.

---

## 23. From This Section to the Rest of Chapter 15

Tokenization (Section 01) chooses the event space of this section's distributions. Embeddings (02) parameterize $h_i$; attention (03) computes it; positional encodings (04) make it order-sensitive. This section turns $h_i$ into probability. Training at Scale (06) then optimizes exactly this cross-entropy at industrial scale, Fine-Tuning (07) changes the target distribution, and Scaling Laws (08) predict how its value falls with compute. If a later number in this chapter surprises you, translate it back to a statement about $p_\theta(t_i\mid t_{<i})$ here first.


---

## 24. Ten Worked Problems With Full Solutions

### Problem 1: chain rule by hand
Given $p(\text{the}\mid\text{BOS})=0.9$, $p(\text{cat}\mid\text{the})=0.4$, $p(\text{sat}\mid\text{cat})=0.5$, $p(\text{EOS}\mid\text{sat})=0.8$, find $p(\text{the cat sat})$.
Solution: include EOS: $0.9\times0.4\times0.5\times0.8=0.144$. In logs: $\ln0.9+\ln0.4+\ln0.5+\ln0.8=-0.105-0.916-0.693-0.223=-1.938$, and $e^{-1.938}=0.144$. The two computations must agree; checking this identity catches most hand-scoring errors.

### Problem 2: temperature limits
Logits $z=(2,1,0)$. At $\tau=1$: $p\propto(7.389,2.718,1)$, so $p=(0.665,0.245,0.090)$.
At $\tau=0.5$: $p\propto e^{(4,2,0)}=(54.6,7.39,1)$, so $p=(0.867,0.117,0.016)$.
At $\tau=4$: $p\propto e^{(0.5,0.25,0)}=(1.649,1.284,1)$, so $p=(0.419,0.326,0.254)$.
Entropy rises from 0.44 to 1.03 nats across these, monotone as Section 12.2 states.

### Problem 3: gradient magnitude
If $p_y=0.02$ for the target and another token has $p_j=0.90$, then $\nabla_z L$ has components $-0.98$ on $y$ and $+0.90$ on $j$. One gradient step on logits therefore mostly *removes* mass from the confident wrong token; this is why a few confident errors dominate early training loss.

### Problem 4: perplexity conversion
Mean NLL $=1.6$ nats/token. Perplexity $=e^{1.6}=4.95$. Bits/token $=1.6/\ln2=2.31$. If the tokenizer averages 4.2 bytes/token, bits/byte $=2.31/4.2=0.55$. Only the last number may be compared to a model using a different tokenizer.

### Problem 5: length normalization flips a ranking
Candidate A: 2 tokens, log-probs $(-0.2,-0.3)$, total $-0.5$, mean $-0.25$.
Candidate B: 5 tokens, each $-0.22$, total $-1.1$, mean $-0.22$.
Raw score picks A; mean score picks B. Neither is "correct" in isolation: the choice encodes whether the task rewards brevity (Section 6.1) or per-token quality.

### Problem 6: ECE by hand
Bins: (conf 0.9, acc 0.8, weight 0.5), (conf 0.6, acc 0.6, weight 0.3), (conf 0.4, acc 0.2, weight 0.2).
ECE $=0.5|0.8-0.9|+0.3|0.6-0.6|+0.2|0.2-0.4|=0.05+0+0.04=0.09$. The model is overconfident in both imperfect bins; a fitted temperature (Section 6.5) would soften the 0.9 bin.

### Problem 7: top-p by hand
$p=(0.45,0.25,0.15,0.10,0.05)$, $p_{\text{nucleus}}=0.9$: cumulative sorted mass is $0.45,0.70,0.85,0.95$, so keep the first four and renormalize by $0.95$: $(0.474,0.263,0.158,0.105)$. Discarded mass is exactly $0.05$.

### Problem 8: label smoothing changes the optimum
With $\varepsilon=0.1$, $|V|=4$, target class 1: $y'=(0.925,0.025,0.025,0.025)$. The loss-minimizing prediction is $p=y'$, not a one-hot: label smoothing (Section 15.1) caps justified confidence at $0.925$ for this setting, by construction.

### Problem 9: conditional scoring
Prompt log-probs sum to $-2.0$ over 8 tokens; answer log-probs sum to $-1.2$ over 3 tokens. Answer score (Section 9.1) is $-0.4$ per token. Including the prompt would give $-3.2/11=-0.29$, a *better-looking* number that mostly rewards the model for modelling the prompt the user wrote — a scoring bug, not a better answer.

### Problem 10: smoothing rescues a zero
In Section 20's corpus, $p(\text{slept}\mid\text{dog})=0$ under MLE, so any sentence containing "dog slept" has infinite perplexity. Add-0.1 gives $0.0625$ (Section 20.1): finite, small, and honestly priced. A neural LM with softmax output never assigns exact zero, but can assign $10^{-12}$, which is numerically the same cliff for log-loss.

---

## 25. A Diagnostic Protocol for Probability Bugs

Run these in order; each step isolates one layer of Sections 1–10.

1. **Vocabulary audit.** Print $|V|$, the BOS/EOS ids, and decode a batch back to text. Mismatched special-token ids corrupt every later number silently.
2. **Shift audit.** Confirm labels are inputs shifted by one position. An off-by-one here trains $p(t_i\mid t_{<i+1})$-style leakage or its mirror, and loss looks *suspiciously good* or *hopeless* rather than merely worse.
3. **Mask audit.** Sum the mask; confirm it equals the number of non-pad tokens, and that the reported loss divides by that sum (Section 4.5).
4. **Normalization audit.** Assert every row of $p_\theta(\cdot\mid t_{<i})$ sums to 1 within $10^{-5}$ (float tolerance), on a batch drawn from real data, not only on toy inputs.
5. **Log-space audit.** Recompute one sequence score as a product of probabilities and as a sum of log-probabilities; require agreement (Problem 1).
6. **Tokenizer audit.** Report tokens, bytes, and bits/byte together (Problem 4). A change in any likelihood claim that disappears in bits/byte was a tokenizer artifact.
7. **Decoding audit.** Log the discarded mass of every top-$k$/top-$p$ filter (Section 21). If you cannot state the discarded mass, you cannot state how far sampling is from the model.
8. **Calibration audit.** Bin confidences on validation data and compute ECE (Problem 6) before and after a fitted temperature. Fit the temperature on validation, never on test (Guo et al., 2017).
9. **Conditioning audit.** Score one answer with and without prompt tokens included (Problem 9) and confirm the pipeline uses the intended one.
10. **Reproducibility audit.** Fix the seed, sample 1,000 continuations, and record the empirical frequency of the most common first token; it should match the model's stated first-token probability within binomial sampling error. This single check exercises tokenization, conditioning, softmax, and the sampler together.

A probability stack that passes all ten is not guaranteed correct, but each historical failure class in Section 22 fails at least one of them.

---

## 26. Glossary of Probability Terms Used in This Chapter

| Term | Meaning in this chapter |
|---|---|
| Ancestral sampling | Sampling each token from its conditional, sequentially (12.1) |
| Bits per byte | Total NLL in bits divided by UTF-8 bytes; tokenizer-free (16.1) |
| Calibration | Predicted confidence equals empirical accuracy (6.3, 16.2) |
| Chain rule | Factorization of a joint into conditionals (2.1) |
| Cross-entropy | $-\sum y\log p$ against a target distribution (4.3) |
| ECE | Weighted mean gap between bin confidence and accuracy (6.4) |
| Entropy rate | Per-token entropy limit for long text (11.3) |
| Exposure bias | Gold-prefix training vs self-prefix generation (13.1) |
| Hierarchical softmax | Tree factorization of the vocabulary (14.1, Morin and Bengio 2005) |
| Label smoothing | Mixing the one-hot target with uniform mass (15.1) |
| Log-sum-exp | Stable $\log\sum e^{z_k}$ via max subtraction (3.4) |
| MAP sequence | Highest-probability complete sequence; rarely what you want (13.2) |
| Nucleus sampling | Top-$p$ truncation and renormalization (7.4, Holtzman et al. 2020) |
| Perplexity | $\exp(\text{mean NLL})$; same-tokenizer comparisons only (5.3) |
| PMI | Log ratio of joint to independent probabilities (11.4) |
| Sampled softmax | Training with a corrected negative subset (14.2, Jean et al. 2015) |
| Search error | Decoder fails to find the model's own best sequence (13.2) |
| Selective prediction | Abstaining below a confidence threshold (16.3) |
| Teacher forcing | Training on gold prefixes (2.2) |
| Temperature | Logit scaling $z/\tau$ before softmax (3.5, 12.2) |
| Tilting | Powering a distribution $p^{1/\tau}$ and renormalizing (12.2) |
| Top-$k$ | Keeping the $k$ largest probabilities (7.3, Fan et al. 2018) |

---

## 27. Estimating Likelihood Quantities by Monte Carlo

### 27.1 What sampling can and cannot estimate

Ancestral samples (12.1) estimate expectations under $p_\theta$: mean sequence length, frequency of a phrase, expected reward of a completion. They do *not* directly estimate the probability of a *specific* long sequence — that comes from the log-probability sum (2.4), which is exact for the model. Confusing these two is a category error: frequency estimation has Monte Carlo error $O(1/\sqrt{n})$; log-probability evaluation has none (up to float error).

### 27.2 Standard errors for sampled frequencies

If a phrase occurs in $\hat q$ of $n$ independent samples, the standard error is $\sqrt{\hat q(1-\hat q)/n}$. For $\hat q=0.02$ and $n=1000$, SE $\approx0.0044$, so a claimed difference between models of 0.003 in phrase frequency is inside noise at this sample size. Report $n$ with every sampled statistic — the reproducibility audit (25.10) uses exactly this check.

### 27.3 Importance sampling in one paragraph

To estimate $\mathbb{E}_{p}[f]$ using samples from $q$, weight each sample by $p(x)/q(x)$. In language modelling this appears when evaluating a model on continuations sampled from another (e.g. scoring human-preferred completions): the weights correct the sampling distribution, and they degenerate (one weight dominates) when $q$ is far from $p$ in sequence space — the sequence-level version of the support warning in Section 1.4.

---

## 28. Comparing Two Language Models Honestly

### 28.1 Paired comparison on the same documents

Evaluate both models on identical documents and form per-document NLL differences $d_k=\overline{\mathrm{NLL}}_A^{(k)}-\overline{\mathrm{NLL}}_B^{(k)}$. Pairing removes document difficulty from the comparison; the mean of $d_k$ with a bootstrap confidence interval over documents (resample documents, not tokens — tokens within a document are dependent) is the defensible headline number.

### 28.2 A worked sign test

Model A beats B (lower NLL) on 68 of 100 documents. Under the null of equal quality, wins are Binomial(100, 0.5), so 68 wins is about 3.6 standard deviations ($\sqrt{100\times0.25}=5$) above 50 — strong evidence A is better *on this corpus*, and no evidence at all about other corpora (Section 10.3 distribution shift).

### 28.3 Multiple comparisons

Comparing 20 checkpoints on the same validation set and reporting the best inflates the winner's apparent advantage (selection bias). Hold out a final test set touched once, or correct for the selection. This is the evaluation cousin of fitting temperature on test data (Problem-table, Common Mistakes): every adaptive choice spends the dataset it adapts to.

### 28.4 Checklist for a likelihood claim

- Same tokenizer, or bits per byte reported (16.1, 17.1).
- Same documents, paired differences, bootstrap over documents (28.1).
- EOS and padding handled identically for both models (1.2, 4.5).
- Number of documents and tokens stated.
- Decoding settings irrelevant to a likelihood claim — and therefore stated as unused, so generation settings are not smuggled into an evaluation result.
---

## 29. Historical Line in One View

| Year | Contribution | Probability idea this chapter uses |
|---:|---|---|
| 1948 | Shannon, "A Mathematical Theory of Communication" | Entropy $H$ and cross-entropy as coding length (Section 5) |
| 1951 | Shannon, "Prediction and Entropy of Printed English" | Entropy rate estimated by prediction (11.3) |
| 1989/1990 | Church and Hanks, word association via PMI | Association against an independence baseline (11.4) |
| 1995 | Kneser and Ney, improved backing-off | Discounting and continuation counts (20.2) |
| 2003 | Bengio et al., neural probabilistic LM | Distributed $h_i$ parameterizes the conditional (8.3) |
| 2005 | Morin and Bengio, hierarchical softmax | Tree factorization of a large vocabulary (14.1) |
| 2010 | Mikolov et al., recurrent LM | Recurrent conditional $p(t_i\mid t_{<i})$ at corpus scale (8.4) |
| 2015 | Bengio et al., scheduled sampling | Exposure bias named and mitigated (13.1) |
| 2015 | Hinton et al., distillation | Soft targets carry dark knowledge (15.3) |
| 2015 | Jean et al., sampled softmax for NMT | Corrected subset normalization in training (14.2) |
| 2016 | Szegedy et al., label smoothing | Non-one-hot targets (15.1) |
| 2017 | Grave et al., adaptive softmax | Frequency-banded exact normalization (14.3) |
| 2017 | Vaswani et al., Transformer | Attention computes $h_i$ for the LM head (8.5) |
| 2017 | Guo et al., calibration of modern networks | Temperature scaling on validation data (6.5, 16.2) |
| 2018 | Fan et al., hierarchical neural story generation | Top-$k$ sampling popularized (7.3, 12.3) |
| 2019 | Müller et al., label smoothing revisited | Smoothing helps calibration, hurts distillation (15.2) |
| 2019 | Stahlberg and Byrne, NMT search errors | MAP pathology is model error, not search error (13.2) |
| 2020 | Holtzman et al., neural text degeneration | Nucleus sampling; greedy degeneration (7.4, 12.4) |
| 2020 | Lewis et al., retrieval-augmented generation | Retrieved context raises $I(X;Y)$ for answers (9.2, 11.2) |
| 2020 | Brown et al., GPT-3 | Autoregressive LM probability at few-shot scale (9.1) |

## 30. A Softmax Jacobian Worked Numerically

For $z=(2,1,0)$, $p=(0.665,0.245,0.090)$ (Problem 2). The Jacobian $J_{ij}=p_i(\delta_{ij}-p_j)$ is:

| | $z_1$ | $z_2$ | $z_3$ |
|---|---:|---:|---:|
| $p_1$ | $0.665\times0.335=0.223$ | $-0.665\times0.245=-0.163$ | $-0.665\times0.090=-0.060$ |
| $p_2$ | $-0.163$ | $0.245\times0.755=0.185$ | $-0.022$ |
| $p_3$ | $-0.060$ | $-0.022$ | $0.090\times0.910=0.082$ |

Every row sums to 0 (to rounding): adding a constant to all logits changes nothing (Section 3.3), so the all-ones direction is in the null space. The cross-entropy gradient $p-y$ for target token 1 is $(-0.335,0.245,0.090)$; note it is exactly the first row of $J$ with sign pattern set by $y$ — the Jacobian and the loss gradient are the same object seen from two sides, which is why Section 19.1 and 19.2 belong in one chapter.

---

---

## 31. Choosing a Score for a Task: Decision Table

| Task | Score to report | Why | Section |
|---|---|---|---|
| Compare two models, same tokenizer | Mean NLL / perplexity on paired documents | Identical event space; pairing removes document difficulty | 5.3, 28.1 |
| Compare two models, different tokenizers | Bits per byte | Only tokenizer-free likelihood | 16.1 |
| Rank candidate answers for one prompt | Mean log-prob over answer tokens | Removes prompt modelling and length artefacts as far as possible | 6.2, 9.1 |
| Pick one generated answer to show a user | Sampled or MBR-selected completion, not MAP | The mode is often degenerate; utility is not likelihood | 12–13 |
| Decide whether to abstain | Calibrated max-probability with a threshold | Selective prediction needs calibrated confidence | 16.2–16.3 |
| Debug a loss that "looks too good" | Shift/mask audit numbers, then bits per byte | Leakage and mask bugs flatter NLL before they break it | 25 |
| Report generation quality | Human/task metric plus discarded-mass logs | Likelihood does not measure generation utility | 21, 25.7 |
| Compare a distilled student to its teacher | KL to teacher at distillation temperature, plus task NLL | One-hot NLL hides dark-knowledge transfer | 15.3 |

### 31.1 The one-sentence summary of this chapter

A language model is a family of conditional distributions $p_\theta(t_i\mid t_{<i})$; training minimizes their cross-entropy to data, evaluation measures that cross-entropy in tokenizer-fair units, and decoding deliberately distorts those conditionals — and almost every bug, misleading metric, or degenerate generation in LLM work is a confusion between those three operations.

### 31.2 Final self-check questions

1. Can you derive $p-y$ from $-\log\mathrm{softmax}(z)_y$ without looking (19.2)?
2. Can you state when perplexity comparisons are valid and what to use instead (5.5, 16.1)?
3. Can you compute the discarded mass of a top-$p$ filter and its TV distance (12.3)?
4. Can you explain why greedy text loops, in distribution terms (12.4)?
5. Can you score $p(y\mid x)$ without charging the model for the prompt (9.1)?
6. Can you fit a temperature without touching test data (6.5)?
7. Can you give a zero-probability failure and its smoothing fix with numbers (20–20.1)?

If any answer is no, the corresponding section above is the reread; the theory notebook turns each into an executable check.

---

## 32. Formula Summary

| Quantity | Formula | Units |
|---|---|---|
| Sequence probability | $p(t_{1:n})=\prod_i p(t_i\mid t_{<i})$ | probability |
| Sequence log-probability | $\sum_i \log p(t_i\mid t_{<i})$ | nats |
| Softmax | $p_j=e^{z_j}/\sum_k e^{z_k}$ | probability |
| Log-softmax | $z_j-\mathrm{logsumexp}(z)$ | nats |
| Temperature distribution | $p_j(\tau)\propto p_j^{1/\tau}$ | probability |
| Entropy | $H(p)=-\sum_j p_j\log p_j$ | nats or bits by log base |
| Cross-entropy | $H(p,q)=-\sum_j p_j\log q_j$ | nats |
| KL divergence | $D_{\mathrm{KL}}(p\|q)=H(p,q)-H(p)$ | nats, $\ge0$ |
| Perplexity | $\exp(\overline{\mathrm{NLL}})$ | effective branching factor |
| Bits per token | $\overline{\mathrm{NLL}}/\ln2$ | bits |
| Bits per byte | total NLL in bits / UTF-8 bytes | bits/byte |
| Conditional entropy | $H(Y\mid X)=\mathbb{E}[-\log p(Y\mid X)]$ | nats |
| Mutual information | $I(X;Y)=H(Y)-H(Y\mid X)$ | nats or bits |
| PMI | $\log\frac{p(x,y)}{p(x)p(y)}$ | nats |
| ECE | $\sum_b w_b\,|\mathrm{acc}_b-\mathrm{conf}_b|$ | probability |
| Softmax Jacobian | $\partial p_i/\partial z_j=p_i(\delta_{ij}-p_j)$ | — |
| CE gradient | $\nabla_z(-\log p_y)=p-y$ | — |
| Top-$p$ discarded mass | $1-\sum_{j\in K}p_j$ | probability (= TV distance) |
| Label-smoothed target | $(1-\varepsilon)y+\varepsilon/|V|$ | probability |
| Monte Carlo frequency SE | $\sqrt{\hat q(1-\hat q)/n}$ | probability |

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Comparing per-token perplexity across tokenizers | Different event spaces and token counts | Report bits per byte (Section 16.1) |
| Averaging loss over padding tokens | Padding is not data; it dilutes the mean | Mask pads (Section 4.5) and count real tokens |
| Exponentiating raw logits for softmax | Overflow for large logits | Subtract max / use log-sum-exp (Section 3.4) |
| Calling temperature on probabilities "temperature" | Tilting acts on logits / powers of p | Use $p^{1/\tau}$ renormalization (Section 12.2) |
| Scoring answers including prompt tokens | Measures prompt modelling, not answer quality | Average over answer tokens only (Section 9.1) |
| Treating greedy output as the model's MAP sequence | Greedy is a local heuristic; ties are arbitrary | Beam search for MAP approx.; sample for typical text |
| Reporting sampled-softmax loss as perplexity | Wrong normalizer | Full softmax at evaluation (Section 14.2) |
| Assuming lower entropy is always better | A confident wrong model has low entropy and high NLL | Judge by NLL/calibration, not entropy alone |
| Ignoring EOS in sequence probability | Strings of different lengths are then incomparable | Include the EOS factor (Section 1.2) |
| Fitting temperature on test data | Leaks the evaluation set into a parameter | Fit on validation only (Guo et al., 2017) |

---

## Practice Exercises

1. Factorize a three-token sentence probability using the chain rule.
2. Compute stable softmax probabilities from a logit vector by subtracting the maximum.
3. Derive the gradient of one-hot cross-entropy with respect to logits.
4. Compute a masked average loss for a padded batch.
5. Convert average NLL from nats to perplexity and bits per token.
6. Compare raw and length-normalized scores for two candidate answers.
7. Apply temperature, top-k, and top-p filtering to a toy distribution.
8. Compute expected calibration error for binned confidence values.
9. Score a conditional answer $p(y\\mid x)$ without including the prompt tokens in the average answer loss.
10. Write a short checklist for debugging an LM probability implementation.

## Why This Matters for AI

| Concept | AI Impact |
|---|---|
| Chain-rule factorization (Section 2) | Makes sequence probability computable one conditional at a time in GPT-style models (Brown et al., 2020) |
| Stable log-softmax (Section 3.4) | Prevents NaN losses in mixed-precision training at scale (Micikevicius et al., 2018) |
| Perplexity / bits per byte (Sections 5, 16) | Tokenizer-fair model comparison in scaling studies (Kaplan et al., 2020) |
| Nucleus sampling (Section 12) | Default open-ended generation in Holtzman et al. (2020) and successors |
| Calibration / ECE (Sections 6, 16) | Confidence gating and abstention in deployed assistants (Guo et al., 2017) |
| Label smoothing / distillation targets (Section 15) | Teacher-student compression (Hinton et al., 2015; Müller et al., 2019) |

The probability layer is the contract between representation learning and text behavior. If logits are unstable, loss is masked incorrectly, probabilities are compared across incompatible tokenizations, or decoding filters are misunderstood, the model may appear to improve while the measured probability object is wrong. Strong LLM work requires comfort with this layer because it appears in pretraining, supervised fine-tuning, preference optimization, retrieval evaluation, calibration, long-context tests, and deployment decoding.

## Bridge to Training at Scale

The next section studies what happens when the same cross-entropy objective is optimized with billions of tokens, large batches, distributed hardware, mixed precision, gradient accumulation, learning-rate schedules, and checkpointing. Nothing in the training-at-scale story changes the probability target. Scale changes how expensive and fragile it is to estimate the same conditional distributions.

## References

- Claude Shannon, "A Mathematical Theory of Communication", 1948.
- Claude Shannon, "Prediction and Entropy of Printed English", 1951.
- Yoshua Bengio, Rejean Ducharme, Pascal Vincent, and Christian Jauvin, "A Neural Probabilistic Language Model", JMLR, 2003: https://jmlr.csail.mit.edu/papers/v3/bengio03a.html
- Tomas Mikolov, Martin Karafiat, Lukas Burget, Jan Cernocky, and Sanjeev Khudanpur, "Recurrent neural network based language model", Interspeech, 2010: https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.html
- Dan Jurafsky and James H. Martin, "Speech and Language Processing", Chapter 3 on n-gram language models: https://web.stanford.edu/~jurafsky/slp3/
- Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu, "Exploring the Limits of Language Modeling", 2016: https://arxiv.org/abs/1602.02410
- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Ari Holtzman et al., "The Curious Case of Neural Text Degeneration", 2019: https://arxiv.org/abs/1904.09751
- Ofir Press et al., "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", 2021: https://arxiv.org/abs/2108.12409
- Kenneth Church and Patrick Hanks, "Word Association Norms, Mutual Information, and Lexicography", Computational Linguistics, 1990.
- Reinhard Kneser and Hermann Ney, "Improved Backing-off for M-gram Language Modeling", ICASSP, 1995.
- Frederic Morin and Yoshua Bengio, "Hierarchical Probabilistic Neural Network Language Model", AISTATS, 2005.
- Sebastien Jean et al., "On Using Very Large Target Vocabulary for Neural Machine Translation", ACL, 2015.
- Edouard Grave et al., "Efficient Softmax Approximation for GPUs", ICML, 2017.
- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, "Distilling the Knowledge in a Neural Network", 2015: https://arxiv.org/abs/1503.02531
- Christian Szegedy et al., "Rethinking the Inception Architecture for Computer Vision", CVPR, 2016 (label smoothing).
- Rafael Müller, Simon Kornblith, and Geoffrey Hinton, "When Does Label Smoothing Help?", NeurIPS, 2019.
- Chuan Guo et al., "On Calibration of Modern Neural Networks", ICML, 2017.
- Angela Fan, Mike Lewis, and Yann Dauphin, "Hierarchical Neural Story Generation", ACL, 2018 (top-k sampling).
- Felix Stahlberg and Bill Byrne, "On NMT Search Errors and Model Errors: Cat Got Your Tongue?", EMNLP, 2019.
- Samy Bengio et al., "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks", NeurIPS, 2015.
- Patrick Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", NeurIPS, 2020.
- Tom Brown et al., "Language Models are Few-Shot Learners", NeurIPS, 2020: https://arxiv.org/abs/2005.14165
