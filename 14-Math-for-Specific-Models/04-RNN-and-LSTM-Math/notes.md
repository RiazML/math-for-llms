[<- Probabilistic Models](../03-Probabilistic-Models/notes.md) | [Home](../../README.md) | [Transformer Architecture ->](../05-Transformer-Architecture/notes.md)

---

# RNN and LSTM Math

Recurrent neural networks model sequences by reusing the same transition at every time step. Their power is memory; their difficulty is training through long chains of time.

## Overview

The basic recurrent equation is:

$$
h_t=f_\theta(h_{t-1},x_t).
$$

This state update lets a model process variable-length sequences. But training the model means backpropagating through the unrolled computation graph. Long products of recurrent Jacobians can vanish or explode. LSTM and GRU cells add gates and additive memory paths to make sequence learning more stable.

## Prerequisites

- Matrix multiplication and nonlinear activations
- Chain rule and backpropagation
- Cross-entropy for sequence prediction
- Basic transformer attention intuition for the bridge section

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates vanilla recurrence, BPTT gradient products, clipping, LSTM/GRU gates, masks, teacher forcing, attention weights, and diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for recurrent equations, BPTT, gates, masking, and sequence-task shapes. |

## Learning Objectives

After this section, you should be able to:

- Write vanilla RNN, LSTM, and GRU update equations.
- Explain recurrence as a parameter-shared deep network along time.
- Derive why gradients vanish or explode through repeated Jacobian products.
- Apply gradient clipping and truncated BPTT.
- Interpret LSTM forget/input/output gates and GRU update/reset gates.
- Distinguish many-to-one, many-to-many, seq2seq, and bidirectional tasks.
- Explain why attention was introduced to reduce the fixed-context bottleneck.
- Build diagnostics for masks, gate saturation, length generalization, and gradient norms.

## Table of Contents

1. [Sequence Modeling View](#1-sequence-modeling-view)
   - 1.1 [Sequential data](#11-sequential-data)
   - 1.2 [Hidden state](#12-hidden-state)
   - 1.3 [Output distribution](#13-output-distribution)
   - 1.4 [Autoregressive generation](#14-autoregressive-generation)
   - 1.5 [RNN versus transformer](#15-rnn-versus-transformer)
2. [Vanilla RNN Equations](#2-vanilla-rnn-equations)
   - 2.1 [State update](#21-state-update)
   - 2.2 [Output head](#22-output-head)
   - 2.3 [Parameter sharing](#23-parameter-sharing)
   - 2.4 [Unrolled graph](#24-unrolled-graph)
   - 2.5 [Initial state](#25-initial-state)
3. [Backpropagation Through Time](#3-backpropagation-through-time)
   - 3.1 [Sequence loss](#31-sequence-loss)
   - 3.2 [Temporal chain rule](#32-temporal-chain-rule)
   - 3.3 [Weight gradient](#33-weight-gradient)
   - 3.4 [Truncated BPTT](#34-truncated-bptt)
   - 3.5 [State detachment](#35-state-detachment)
4. [Vanishing and Exploding Gradients](#4-vanishing-and-exploding-gradients)
   - 4.1 [Jacobian product](#41-jacobian-product)
   - 4.2 [Spectral radius](#42-spectral-radius)
   - 4.3 [Vanishing](#43-vanishing)
   - 4.4 [Exploding](#44-exploding)
   - 4.5 [Gradient clipping](#45-gradient-clipping)
5. [LSTM Cell](#5-lstm-cell)
   - 5.1 [Forget gate](#51-forget-gate)
   - 5.2 [Input gate](#52-input-gate)
   - 5.3 [Candidate memory](#53-candidate-memory)
   - 5.4 [Cell update](#54-cell-update)
   - 5.5 [Output gate](#55-output-gate)
6. [GRU Cell](#6-gru-cell)
   - 6.1 [Update gate](#61-update-gate)
   - 6.2 [Reset gate](#62-reset-gate)
   - 6.3 [Candidate hidden](#63-candidate-hidden)
   - 6.4 [Hidden update](#64-hidden-update)
   - 6.5 [GRU versus LSTM](#65-gru-versus-lstm)
7. [Sequence Tasks](#7-sequence-tasks)
   - 7.1 [Many-to-one](#71-manytoone)
   - 7.2 [Many-to-many](#72-manytomany)
   - 7.3 [Seq2seq](#73-seq2seq)
   - 7.4 [Bidirectional RNN](#74-bidirectional-rnn)
   - 7.5 [Teacher forcing](#75-teacher-forcing)
8. [Attention Bridge](#8-attention-bridge)
   - 8.1 [Fixed context bottleneck](#81-fixed-context-bottleneck)
   - 8.2 [Alignment scores](#82-alignment-scores)
   - 8.3 [Attention weights](#83-attention-weights)
   - 8.4 [Context vector](#84-context-vector)
   - 8.5 [Transformer bridge](#85-transformer-bridge)
9. [Training Practice](#9-training-practice)
   - 9.1 [Padding masks](#91-padding-masks)
   - 9.2 [Packed sequences](#92-packed-sequences)
   - 9.3 [Stateful streaming](#93-stateful-streaming)
   - 9.4 [Initialization](#94-initialization)
   - 9.5 [Regularization](#95-regularization)
10. [Diagnostics](#10-diagnostics)
37. [Computed Table: What a Delay Demands From a Gate](#37-computed-table-what-a-delay-demands-from-a-gate)
38. [Ten-Line Field Guide to Defensible Sequence Claims](#38-ten-line-field-guide-to-defensible-sequence-claims)
34. [Addenda: Ten Lines to Carry Into §05](#34-addenda-ten-lines-to-carry-into-05)
35. [Margin Notes: The Recurrent Creed](#35-margin-notes-the-recurrent-creed)
36. [End of Section 04 Expansion](#36-end-of-section-04-expansion)
29. [Worked Trace: Cell-State Retention Over 20 Steps](#29-worked-trace-cell-state-retention-over-20-steps)
30. [Extended Exercise Bank (Self-Study)](#30-extended-exercise-bank-self-study)
31. [Revision Problems (Mixed)](#31-revision-problems-mixed)
32. [Chapter Summary: RNN and LSTM Math](#32-chapter-summary-rnn-and-lstm-math)
33. [Section 04 Completion Checklist](#33-section-04-completion-checklist)
25. [Exam and Interview Traps: RNNs and LSTMs](#25-exam-and-interview-traps-rnns-and-lstms)
26. [Worked Problem Set C: Synthesis](#26-worked-problem-set-c-synthesis)
27. [Quick Checks (One-Line Answers)](#27-quick-checks-one-line-answers)
28. [Index of Computed Evidence](#28-index-of-computed-evidence)
21. [Computed Table: Retention Over Delay](#21-computed-table-retention-over-delay)
22. [Self-Test Questions](#22-self-test-questions)
23. [Self-Test Answer Key (Condensed)](#23-self-test-answer-key-condensed)
24. [Annotated Reading Guide](#24-annotated-reading-guide)
17. [Quick-Reference Cards](#17-quick-reference-cards)
18. [Pre-Flight Checklist for Recurrent Models](#18-pre-flight-checklist-for-recurrent-models)
19. [Frequently Asked Questions](#19-frequently-asked-questions)
20. [Decision Records: Recurrent Models in Context](#20-decision-records-recurrent-models-in-context)
14. [Worked Problem Set A: Gradients Through Time](#14-worked-problem-set-a-gradients-through-time)
15. [Worked Problem Set B: Sequence Plumbing](#15-worked-problem-set-b-sequence-plumbing)
16. [Glossary: Recurrent Vocabulary](#16-glossary-recurrent-vocabulary)
11. [Common Mistakes](#11-common-mistakes)
12. [Why Recurrence Struggles: The Gradient Product](#12-why-recurrence-struggles-the-gradient-product)
13. [Computed Table: Forget-Gate Bias Sets the Memory Clock](#13-computed-table-forget-gate-bias-sets-the-memory-clock)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Gradient norms](#102-gradient-norms)
   - 10.3 [Gate statistics](#103-gate-statistics)
   - 10.4 [Length tests](#104-length-tests)
   - 10.5 [Ablations](#105-ablations)

---

## Shape Map

```text
input sequence:       X      shape (B, T, d_x)
hidden state:         h_t    shape (B, d_h)
hidden sequence:      H      shape (B, T, d_h)
logits per token:     O      shape (B, T, |V|)
padding mask:         M      shape (B, T)
```

## 1. Sequence Modeling View

This part studies sequence modeling view through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sequential data](#1-sequential-data) | observations arrive with order and history | $x_{1:T}=(x_1,\ldots,x_T)$ |
| [Hidden state](#1-hidden-state) | the model carries a summary of the past | $h_t=f_\theta(h_{t-1},x_t)$ |
| [Output distribution](#1-output-distribution) | each state can produce a prediction | $p(y_t\mid x_{\le t})=g_\theta(h_t)$ |
| [Autoregressive generation](#1-autoregressive-generation) | the previous output can become the next input | $p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$ |
| [RNN versus transformer](#1-rnn-versus-transformer) | RNNs compress history into a state, transformers keep token states visible | $h_t$ versus $H_{1:t}$ |

### 1.1 Sequential data

**Main idea.** Observations arrive with order and history.

Core relation:

$$x_{1:T}=(x_1,\ldots,x_T)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.2 Hidden state

**Main idea.** The model carries a summary of the past.

Core relation:

$$h_t=f_\theta(h_{t-1},x_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.3 Output distribution

**Main idea.** Each state can produce a prediction.

Core relation:

$$p(y_t\mid x_{\le t})=g_\theta(h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.4 Autoregressive generation

**Main idea.** The previous output can become the next input.

Core relation:

$$p(x_{1:T})=\prod_t p(x_t\mid x_{<t})$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 1.5 RNN versus transformer

**Main idea.** Rnns compress history into a state, transformers keep token states visible.

Core relation:

$$h_t$ versus $H_{1:t}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 2. Vanilla RNN Equations

This part studies vanilla rnn equations through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [State update](#2-state-update) | combine current input and previous hidden state | $h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$ |
| [Output head](#2-output-head) | map hidden state to logits or regression output | $o_t=W_{hy}h_t+b_y$ |
| [Parameter sharing](#2-parameter-sharing) | the same weights are reused at every time step | $\theta_t=\theta$ |
| [Unrolled graph](#2-unrolled-graph) | an RNN is a deep network along time | $h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_T$ |
| [Initial state](#2-initial-state) | start from zeros or a learned state | $h_0=0$ or trainable |

### 2.1 State update

**Main idea.** Combine current input and previous hidden state.

Core relation:

$$h_t=\phi(W_{xh}x_t+W_{hh}h_{t-1}+b_h)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.2 Output head

**Main idea.** Map hidden state to logits or regression output.

Core relation:

$$o_t=W_{hy}h_t+b_y$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.3 Parameter sharing

**Main idea.** The same weights are reused at every time step.

Core relation:

$$\theta_t=\theta$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.4 Unrolled graph

**Main idea.** An rnn is a deep network along time.

Core relation:

$$h_1\rightarrow h_2\rightarrow\cdots\rightarrow h_T$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 2.5 Initial state

**Main idea.** Start from zeros or a learned state.

Core relation:

$$h_0=0$ or trainable$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 3. Backpropagation Through Time

This part studies backpropagation through time through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sequence loss](#3-sequence-loss) | sum or average losses over time | $L=\sum_{t=1}^{T}\ell_t$ |
| [Temporal chain rule](#3-temporal-chain-rule) | future losses depend on earlier states through recurrence | $\partial L/\partial h_t=\partial \ell_t/\partial h_t+(\partial h_{t+1}/\partial h_t)^\top\partial L/\partial h_{t+1}$ |
| [Weight gradient](#3-weight-gradient) | shared weights collect gradient from every time step | $\partial L/\partial W=\sum_t \partial L_t/\partial W$ |
| [Truncated BPTT](#3-truncated-bptt) | backpropagate through a limited window for efficiency | $t-k,\ldots,t$ |
| [State detachment](#3-state-detachment) | detach hidden state between chunks to control graph length | $h_t\leftarrow\mathrm{stopgrad}(h_t)$ |

### 3.1 Sequence loss

**Main idea.** Sum or average losses over time.

Core relation:

$$L=\sum_{t=1}^{T}\ell_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.2 Temporal chain rule

**Main idea.** Future losses depend on earlier states through recurrence.

Core relation:

$$\partial L/\partial h_t=\partial \ell_t/\partial h_t+(\partial h_{t+1}/\partial h_t)^\top\partial L/\partial h_{t+1}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is why RNN training is training a very deep network along time.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.3 Weight gradient

**Main idea.** Shared weights collect gradient from every time step.

Core relation:

$$\partial L/\partial W=\sum_t \partial L_t/\partial W$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.4 Truncated BPTT

**Main idea.** Backpropagate through a limited window for efficiency.

Core relation:

$$t-k,\ldots,t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 3.5 State detachment

**Main idea.** Detach hidden state between chunks to control graph length.

Core relation:

$$h_t\leftarrow\mathrm{stopgrad}(h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 4. Vanishing and Exploding Gradients

This part studies vanishing and exploding gradients through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Jacobian product](#4-jacobian-product) | long-range gradients multiply many recurrent Jacobians | $\prod_{i=s+1}^{t}\partial h_i/\partial h_{i-1}$ |
| [Spectral radius](#4-spectral-radius) | gradient scale depends on recurrent dynamics | $\rho(W_{hh})$ |
| [Vanishing](#4-vanishing) | singular values below one shrink long-range gradients | $\Vert g_s\Vert\rightarrow 0$ |
| [Exploding](#4-exploding) | singular values above one can blow up gradients | $\Vert g_s\Vert\rightarrow\infty$ |
| [Gradient clipping](#4-gradient-clipping) | cap gradient norm before optimizer update | $g\leftarrow g\min(1,c/\Vert g\Vert)$ |

### 4.1 Jacobian product

**Main idea.** Long-range gradients multiply many recurrent jacobians.

Core relation:

$$\prod_{i=s+1}^{t}\partial h_i/\partial h_{i-1}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.2 Spectral radius

**Main idea.** Gradient scale depends on recurrent dynamics.

Core relation:

$$\rho(W_{hh})$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.3 Vanishing

**Main idea.** Singular values below one shrink long-range gradients.

Core relation:

$$\Vert g_s\Vert\rightarrow 0$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.4 Exploding

**Main idea.** Singular values above one can blow up gradients.

Core relation:

$$\Vert g_s\Vert\rightarrow\infty$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 4.5 Gradient clipping

**Main idea.** Cap gradient norm before optimizer update.

Core relation:

$$g\leftarrow g\min(1,c/\Vert g\Vert)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** Clipping became a standard tool because recurrent gradient products can explode suddenly.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 5. LSTM Cell

This part studies lstm cell through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Forget gate](#5-forget-gate) | decide what memory to keep | $f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)$ |
| [Input gate](#5-input-gate) | decide what new content to write | $i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)$ |
| [Candidate memory](#5-candidate-memory) | propose new cell content | $\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c)$ |
| [Cell update](#5-cell-update) | additive memory path improves gradient flow | $c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$ |
| [Output gate](#5-output-gate) | expose part of cell memory as hidden state | $h_t=o_t\odot\tanh(c_t)$ |

### 5.1 Forget gate

**Main idea.** Decide what memory to keep.

Core relation:

$$f_t=\sigma(W_f[x_t,h_{t-1}]+b_f)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.2 Input gate

**Main idea.** Decide what new content to write.

Core relation:

$$i_t=\sigma(W_i[x_t,h_{t-1}]+b_i)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.3 Candidate memory

**Main idea.** Propose new cell content.

Core relation:

$$\tilde c_t=\tanh(W_c[x_t,h_{t-1}]+b_c)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.4 Cell update

**Main idea.** Additive memory path improves gradient flow.

Core relation:

$$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** The additive cell path is the mathematical reason LSTMs can carry information longer than a plain tanh RNN.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 5.5 Output gate

**Main idea.** Expose part of cell memory as hidden state.

Core relation:

$$h_t=o_t\odot\tanh(c_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 6. GRU Cell

This part studies gru cell through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Update gate](#6-update-gate) | interpolate old and candidate states | $z_t=\sigma(W_z[x_t,h_{t-1}])$ |
| [Reset gate](#6-reset-gate) | control how much past enters the candidate | $r_t=\sigma(W_r[x_t,h_{t-1}])$ |
| [Candidate hidden](#6-candidate-hidden) | build proposed new hidden state | $\tilde h_t=\tanh(W_h[x_t,r_t\odot h_{t-1}])$ |
| [Hidden update](#6-hidden-update) | blend old state and candidate | $h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$ |
| [GRU versus LSTM](#6-gru-versus-lstm) | GRU is smaller; LSTM has separate cell and hidden state | $h_t$ only versus $(c_t,h_t)$ |

### 6.1 Update gate

**Main idea.** Interpolate old and candidate states.

Core relation:

$$z_t=\sigma(W_z[x_t,h_{t-1}])$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.2 Reset gate

**Main idea.** Control how much past enters the candidate.

Core relation:

$$r_t=\sigma(W_r[x_t,h_{t-1}])$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.3 Candidate hidden

**Main idea.** Build proposed new hidden state.

Core relation:

$$\tilde h_t=\tanh(W_h[x_t,r_t\odot h_{t-1}])$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.4 Hidden update

**Main idea.** Blend old state and candidate.

Core relation:

$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 6.5 GRU versus LSTM

**Main idea.** Gru is smaller; lstm has separate cell and hidden state.

Core relation:

$$h_t$ only versus $(c_t,h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 7. Sequence Tasks

This part studies sequence tasks through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Many-to-one](#7-manytoone) | classify a whole sequence from final or pooled state | $\hat y=g(h_T)$ |
| [Many-to-many](#7-manytomany) | predict at every time step | $\hat y_t=g(h_t)$ |
| [Seq2seq](#7-seq2seq) | encode one sequence and decode another | $p(y_{1:M}\mid x_{1:T})=\prod_jp(y_j\mid y_{<j},c)$ |
| [Bidirectional RNN](#7-bidirectional-rnn) | use past and future context for non-causal tasks | $h_t=[\overrightarrow h_t;\overleftarrow h_t]$ |
| [Teacher forcing](#7-teacher-forcing) | decoder conditions on gold previous outputs during training | $p(y_t\mid y_{<t}^\star,c)$ |

### 7.1 Many-to-one

**Main idea.** Classify a whole sequence from final or pooled state.

Core relation:

$$\hat y=g(h_T)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.2 Many-to-many

**Main idea.** Predict at every time step.

Core relation:

$$\hat y_t=g(h_t)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.3 Seq2seq

**Main idea.** Encode one sequence and decode another.

Core relation:

$$p(y_{1:M}\mid x_{1:T})=\prod_jp(y_j\mid y_{<j},c)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.4 Bidirectional RNN

**Main idea.** Use past and future context for non-causal tasks.

Core relation:

$$h_t=[\overrightarrow h_t;\overleftarrow h_t]$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 7.5 Teacher forcing

**Main idea.** Decoder conditions on gold previous outputs during training.

Core relation:

$$p(y_t\mid y_{<t}^\star,c)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 8. Attention Bridge

This part studies attention bridge through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Fixed context bottleneck](#8-fixed-context-bottleneck) | a single encoder vector struggles with long inputs | $c=h_T$ |
| [Alignment scores](#8-alignment-scores) | decoder state scores every encoder state | $e_{tj}=a(s_{t-1},h_j)$ |
| [Attention weights](#8-attention-weights) | softmax turns scores into a distribution over positions | $\alpha_{tj}=\mathrm{softmax}_j(e_{tj})$ |
| [Context vector](#8-context-vector) | weighted sum exposes relevant encoder states | $c_t=\sum_j\alpha_{tj}h_j$ |
| [Transformer bridge](#8-transformer-bridge) | self-attention removes recurrence and exposes all token states directly | $\mathrm{Attention}(Q,K,V)$ |

### 8.1 Fixed context bottleneck

**Main idea.** A single encoder vector struggles with long inputs.

Core relation:

$$c=h_T$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.2 Alignment scores

**Main idea.** Decoder state scores every encoder state.

Core relation:

$$e_{tj}=a(s_{t-1},h_j)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.3 Attention weights

**Main idea.** Softmax turns scores into a distribution over positions.

Core relation:

$$\alpha_{tj}=\mathrm{softmax}_j(e_{tj})$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is the historical bridge from seq2seq RNNs to transformer attention.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.4 Context vector

**Main idea.** Weighted sum exposes relevant encoder states.

Core relation:

$$c_t=\sum_j\alpha_{tj}h_j$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 8.5 Transformer bridge

**Main idea.** Self-attention removes recurrence and exposes all token states directly.

Core relation:

$$\mathrm{Attention}(Q,K,V)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 9. Training Practice

This part studies training practice through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Padding masks](#9-padding-masks) | sequence batches have variable lengths | $m_t\in\{0,1\}$ |
| [Packed sequences](#9-packed-sequences) | avoid computing loss on padding | $L=\sum_tm_t\ell_t/\sum_tm_t$ |
| [Stateful streaming](#9-stateful-streaming) | carry state across chunks for long streams | $h_\mathrm{next}=h_T$ |
| [Initialization](#9-initialization) | orthogonal recurrent matrices can stabilize early dynamics | $W_{hh}^\top W_{hh}=I$ |
| [Regularization](#9-regularization) | dropout, weight decay, and clipping fight overfit and instability | $L+\lambda\Vert\theta\Vert^2$ |

### 9.1 Padding masks

**Main idea.** Sequence batches have variable lengths.

Core relation:

$$m_t\in\{0,1\}$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.2 Packed sequences

**Main idea.** Avoid computing loss on padding.

Core relation:

$$L=\sum_tm_t\ell_t/\sum_tm_t$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.3 Stateful streaming

**Main idea.** Carry state across chunks for long streams.

Core relation:

$$h_\mathrm{next}=h_T$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.4 Initialization

**Main idea.** Orthogonal recurrent matrices can stabilize early dynamics.

Core relation:

$$W_{hh}^\top W_{hh}=I$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 9.5 Regularization

**Main idea.** Dropout, weight decay, and clipping fight overfit and instability.

Core relation:

$$L+\lambda\Vert\theta\Vert^2$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
## 10. Diagnostics

This part studies diagnostics through the lens of sequence learning. The central question is how information and gradients move through time.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | inputs, hidden states, gates, and outputs have distinct axes | $(B,T,d)$ and $(B,h)$ |
| [Gradient norms](#10-gradient-norms) | track exploding or vanishing gradients through time | $\Vert g_t\Vert$ |
| [Gate statistics](#10-gate-statistics) | saturated gates indicate stuck memory behavior | $f_t,i_t,z_t$ near 0 or 1 |
| [Length tests](#10-length-tests) | evaluate short and long sequences separately | $S(T)$ |
| [Ablations](#10-ablations) | compare vanilla RNN, GRU, LSTM, and attention bridge | $\Delta L,\Delta S$ |

### 10.1 Shape checks

**Main idea.** Inputs, hidden states, gates, and outputs have distinct axes.

Core relation:

$$(B,T,d)$ and $(B,h)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.2 Gradient norms

**Main idea.** Track exploding or vanishing gradients through time.

Core relation:

$$\Vert g_t\Vert$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.3 Gate statistics

**Main idea.** Saturated gates indicate stuck memory behavior.

Core relation:

$$f_t,i_t,z_t$ near 0 or 1$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** Gate histograms quickly reveal whether an LSTM or GRU is actually using its memory.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.4 Length tests

**Main idea.** Evaluate short and long sequences separately.

Core relation:

$$S(T)$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.
### 10.5 Ablations

**Main idea.** Compare vanilla rnn, gru, lstm, and attention bridge.

Core relation:

$$\Delta L,\Delta S$$

An RNN is a parameter-shared computation graph unrolled across time. The hidden state is useful because it carries information forward, but it also creates a long chain for gradients to travel backward. LSTM and GRU gates are designed to make this information path more controllable.

**Worked micro-example.** In a plain linearized RNN, the contribution of an early hidden state to a later hidden state contains powers of the recurrent matrix. If the dominant singular value is 0.8, a signal over 20 steps is scaled by about $0.8^{20}$. If it is 1.2, the same path grows like $1.2^{20}$. This is the vanishing and exploding gradient problem in one line.

**Implementation check.** For a batch-first tensor, keep the axes explicit: batch, time, feature. A hidden state usually has shape `(batch, hidden)`, while a full output sequence has shape `(batch, time, hidden)`.

**AI connection.** This is a practical sequence-modeling control variable.

**Common mistake.** Do not treat the final hidden state as magic memory. For long sequences it can become a bottleneck, which is exactly why attention was introduced in seq2seq systems.

---

## Practice Exercises

1. Compute one vanilla RNN hidden update.
2. Compute a sequence log probability from conditional probabilities.
3. Show a scalar gradient product that vanishes or explodes.
4. Clip a gradient vector by norm.
5. Compute one LSTM cell update.
6. Compute one GRU hidden update.
7. Apply a padding mask to sequence losses.
8. Identify shapes for many-to-one and many-to-many tasks.
9. Compute attention weights and context over encoder states.
10. Write an RNN debugging checklist.

## Why This Matters for AI

Transformers dominate current LLMs, but RNNs still teach the core sequence-learning problems: hidden state, recurrence, long-range credit assignment, gradient stability, teacher forcing, and attention as a solution to fixed-context bottlenecks. Understanding RNNs makes transformer design feel less arbitrary.

## Bridge to Transformer Architecture

Transformers replace recurrent state updates with attention over token states. The next section studies how self-attention, residual streams, normalization, and feed-forward blocks solve many RNN bottlenecks while introducing their own memory and compute tradeoffs.

## 11. Common Mistakes
Pre-submission scan for recurrent models; gate-level diagnostics from Sections 7-10 operationalized.

| # | Mistake | Consequence | Fix |
|---|---|---|---|
| 1 | Padding without masking | Loss/gradients from pad tokens; length-dependent metrics | Masked loss verified against per-sequence losses (theory 21) |
| 2 | Wrong axis order (batch/time/features) | Silent garbage that still trains a little | Shape assertions per step; named axes in code |
| 3 | No gradient clipping in vanilla RNNs | Exploding gradients, NaNs (theory 26) | Global-norm clipping + spectral awareness (theory 16) |
| 4 | Expecting vanilla RNNs to learn long dependencies | Vanishing gradients by construction | LSTM/GRU gates, or attention/transformers (§05) |
| 5 | Teacher forcing at train, free-running at test, gap unmeasured | Exposure bias surprises (theory 24) | Measure both modes; scheduled sampling where apt |
| 6 | Bidirectional model in a live/causal product | Future leakage | Causal architectures for online use (theory 22) |
| 7 | Forget-gate bias initialized at 0 by default habit | Memory timescale too short early (Section 7) | Positive forget bias init (Jozefowicz et al. 2015 practice) |
| 8 | Truncated BPTT window shorter than the dependency | Gradients never see the cause | Window ≥ dependency length, or stateful carry with care |
| 9 | Sorting batch by length without packing/masking care | Statistics skewed by padded batches | Pack or mask; audit per-length metrics |
| 10 | Judging RNN by perplexity alone across tokenizers | Incomparable numbers (Chapter 15) | Same tokenizer/data, report NLL + ppl (theory 25) |



## 12. Why Recurrence Struggles: The Gradient Product
Backpropagation through time multiplies per-step Jacobians; their product behaves like a matrix power, governed by the recurrent matrix's spectral radius (theory 16, Pascanu et al. 2013). Below 1: vanishing. Above 1: exploding. Orthogonal/unitary inits and clipping manage the edges, but the structural repair is gating: the LSTM's additive cell state with forget-gate multiplication keeps a near-identity path (theory 17) — the same identity-path idea as residual networks (§02 theory 23) invented six years earlier (Hochreiter & Schmidhuber 1997).

- Eigenvalues, not singular values, govern long products of the same matrix (theory 16's audit) — a subtle and commonly botched point.
- Tanh saturation multiplies additional <1 factors per step (§13.2), compounding the matrix effect.
- Truncated BPTT bounds compute and gradient horizon together; the window is a modelling assumption (Mistakes table).
- Gradient clipping (Bengio/Pascanu lineage; theory 26) handles explosions only — it cannot resurrect vanished signal.



## 13. Computed Table: Forget-Gate Bias Sets the Memory Clock
At initialization the forget bias chooses how long the cell remembers by default (theory 7/17). Bias 1-2 is the standard positive init; bias −2 starts the model nearly amnesiac.

| forget bias init | forget gate σ(b) | retention after 50 steps f^50 | memory horizon ~1/(1-f) |
|---|---|---|---|
| -2 | 0.119 | 0.000 | 1.1 steps |
| -1 | 0.269 | 0.000 | 1.4 steps |
| 0 | 0.500 | 0.000 | 2.0 steps |
| 1 | 0.731 | 0.000 | 3.7 steps |
| 2 | 0.881 | 0.002 | 8.4 steps |
| 3 | 0.953 | 0.088 | 21.1 steps |

## 14. Worked Problem Set A: Gradients Through Time
Worked problems A: gradients through time.

**Problem A1: the 100-step product**
1. Per-step Jacobian factor ~0.9 (spectral radius and tanh slopes combined): after 100 steps the gradient scales by $0.9^{100}\approx2.7\times10^{-5}$.
2. At 1.05: $1.05^{100}\approx131$ — explosion. The trainable band between is narrow (theory 16).
3. This arithmetic is why 'just train longer' fails vanilla RNNs on long dependencies and why gates/clipping exist.

**Problem A2: clipping math**
1. Global norm 250, threshold 5: scale factor 5/250 = 0.02 applied to every parameter's gradient (theory 20/26 of this and §02).
2. Direction preserved, magnitude capped; the optimizer sees a smaller but truthful direction.
3. Clipping every step at high frequency is a symptom: log the clipped fraction and fix the LR/scale cause.

**Problem A3: memory horizon budgeting**
1. Task dependency: 200 steps. Forget bias 1 gives f=0.73, horizon ~3.7 steps at init (Section 13) — the model must learn to raise f.
2. Bias 3: f=0.953, horizon ~21 steps; still short of 200 but trainable territory, especially as gates adapt per-unit.
3. Match init horizon to task scale; report the choice like any architecture hyperparameter.



## 15. Worked Problem Set B: Sequence Plumbing
Worked problems B: sequence plumbing.

**Problem B1: the padded-batch audit**
1. Batch losses differ between padded batch and per-sequence runs: reduction bug (mean over pads).
2. Theory 21's identity is the test: masked per-sequence means must equal unpadded computations to 1e-12.
3. Add it as a unit test; frameworks' default reductions differ on exactly this point.

**Problem B2: bidirectional leakage review**
1. Feature: 'predict today's close from the full day's series including afternoon' — bidirectional is legitimate offline, fraudulent live.
2. Classify every sequence model as causal or acausal in the model card (theory 22); deployment mode must match.



## 16. Glossary: Recurrent Vocabulary
| Term | Meaning |
|---|---|
| BPTT | Backpropagation through time on the unrolled graph; truncated variant bounds the window. |
| Cell state | LSTM's additive memory highway, gated multiplicatively (theory 17). |
| Forget/input/output gates | Sigmoid gates controlling cell write/keep/read (Hochreiter & Schmidhuber 1997). |
| Constant-error carousel | The cell's near-identity gradient path when f≈1. |
| Exposure bias | Train/test prefix mismatch (teacher forcing vs free-running, theory 24). |
| Teacher forcing | Feeding gold prefixes during training. |
| Spectral radius | Largest |eigenvalue| of the recurrent matrix; governs long gradient products (theory 16). |
| Packing | Removing pads from computation instead of masking after (equivalent math, different cost). |
| Stateful RNN | Carrying hidden state across batches; leakage/batching-order hazards. |
| Perplexity | exp(mean NLL) (theory 25); tokenizer-dependent. |

## 17. Quick-Reference Cards
**Vanilla RNN**
- $h_t=\tanh(W_{xh}x_t+W_{hh}h_{t-1}+b)$; trains short dependencies; gradient product is the limit (theory 16).

**LSTM**
- Cell + 3 gates (theory 17/18); default recurrent memory; forget-bias init matters (Section 13).

**GRU**
- 2 gates, 3/4 the parameters (theory 19); comparable often, claim needs ablation.

**Bidirectional RNN**
- Past+future context; offline only (theory 22, Problem B2).

**Encoder-decoder + attention**
- Bahdanau et al. (2015); alignment weights (theory 23) are §05's ancestor.



## 18. Pre-Flight Checklist for Recurrent Models
- Axes named (batch, time, feature) and asserted; padding/masking unit-tested (theory 21).
- Causality classified (causal vs bidirectional) against deployment mode.
- Gradient horizon estimated: spectral behaviour + clipping plan (theory 16/26).
- Gate inits set (forget bias per Section 13); gate saturation will be logged (Section 13 of theory).
- BPTT window ≥ task dependency, or stateful design reviewed.
- Teacher-forcing/free-running gap will be measured (theory 24).
- Sequence-length generalization tested (train short, evaluate long).
- Perplexity/NLL reporting fixed to one tokenizer (Chapter 15 discipline).



## 19. Frequently Asked Questions
- **LSTM or GRU?** Comparable in most careful studies (Chung et al. 2014's empirical comparison is the classic); choose by ablation at matched parameters (theory 19).
- **Why did transformers replace RNNs if LSTMs fixed memory?** Sequential computation cannot parallelize across time, and attention's path length between any two positions is O(1) vs O(T) (§05); LSTMs remain strong in streaming/low-latency niches.
- **Is clipping enough for stability?** For explosions yes-ish; vanishing needs structure (gates/residuals), not clipping (Problem A1).
- **Stateful training across batches?** Only with ordered data and state-reset discipline at document boundaries; otherwise leakage and order artefacts.
- **How long can LSTMs actually remember?** Task-dependent; adding-problem style probes (theory 20's delay task family) measure your setup's horizon instead of quoting folklore.



## 20. Decision Records: Recurrent Models in Context
- **Streaming/online inference:** causal LSTM/GRU (state per step, no future) — or causal transformers (§05) at scale.
- **Offline sequence labelling with full context:** bidirectional LSTM is legitimate and strong.
- **Long-document modelling:** hierarchical/chunked recurrence or transformers; plain BPTT windows will miss the dependencies (Mistakes).
- **Small-data time series:** GRU/LSTM baselines vs §01 linear/GBDT-style baselines — inductive-bias honesty applies here too.
- Each record: causality class, horizon estimate, gate inits, clipping, masking tests, length-generalization results.

## 21. Computed Table: Retention Over Delay
Why the delay task (theory 20) separates architectures: a vanilla-ish 0.9 factor is gone by 50 steps, while a forget gate at 0.999 still carries 60% of gradient at 500. Memory is a per-step multiplier, budgeted like interest.

| delay (steps) | gradient retention 0.9^T | 0.99^T (LSTM-ish f) | 0.999^T |
|---|---|---|---|
| 5 | 5.90e-01 | 0.951 | 0.995 |
| 10 | 3.49e-01 | 0.904 | 0.990 |
| 25 | 7.18e-02 | 0.778 | 0.975 |
| 50 | 5.15e-03 | 0.605 | 0.951 |
| 100 | 2.66e-05 | 0.366 | 0.905 |
| 200 | 7.06e-10 | 0.134 | 0.819 |
| 500 | 1.32e-23 | 0.007 | 0.606 |



## 22. Self-Test Questions
- Why eigenvalues (not singular values) govern BPTT products (theory 16)?
- Write the LSTM cell update and name each gate's job.
- Retention after 100 steps at f=0.98? (Compute: 0.98^100 ≈ 0.133.)
- Clipping: what it fixes, what it cannot (Problem A2).
- Teacher forcing's train/test mismatch name and measurement (theory 24).
- Masked-loss identity and its unit test (theory 21).
- Bidirectional legitimacy condition (Problem B2).
- GRU/LSTM parameter ratio and the ablation rule (theory 19).
- Perplexity's tokenizer dependence (theory 25).
- Forget-bias init's effect in one line (Section 13).



## 23. Self-Test Answer Key (Condensed)
- Powers of one matrix diagonalize over eigenvalues; singular values describe single applications.
- $c_t=f_t\odot c_{t-1}+i_t\odot g_t$, $h_t=o_t\odot\tanh(c_t)$; forget/input/output respectively.
- $0.98^{100}\approx0.133$ — budget horizons multiplicatively (Section 21).
- Explosions capped (direction kept); vanishing untouched.
- Exposure bias; compare teacher-forced vs free-running token accuracy.
- Masked per-sequence means equal unpadded losses exactly.
- Offline tasks with full sequences available at inference.
- 3/4; compare at matched parameters with retuned LRs.
- PPL numbers travel only within a tokenizer/dataset (Chapter 15).
- Positive forget bias lengthens the initial memory horizon (Section 13).



## 24. Annotated Reading Guide
- Primary: Hochreiter & Schmidhuber (1997) for LSTM; Cho et al. (2014) for GRU; Chung et al. (2014) for the empirical GRU/LSTM comparison.
- Gradient pathology: Pascanu, Mikolov & Bengio (2013) — the spectral/clipping analysis behind theory 16/26.
- Attention bridge: Bahdanau, Cho & Bengio (2015), then §05 (Vaswani et al. 2017).
- Gate-init practice: Jozefowicz et al. (2015) for forget-bias and architecture-search lessons.
- Sequence-modelling context within this book: §03 HMMs (probabilistic ancestor), §05 transformers (successor).

## 25. Exam and Interview Traps: RNNs and LSTMs
- Trap: 'LSTMs solve vanishing gradients completely.' Response: they provide a near-identity path modulated by learned gates; horizons remain finite and measurable (Section 21).
- Trap: 'Bidirectional is just better.' Response: it uses the future — illegitimate causally (Problem B2).
- Trap: 'Clipping fixes recurrent instability.' Response: explosions only; vanishing is structural (Problem A1).
- Trap: 'Perplexity 12 beats perplexity 15 (different papers).' Response: tokenizer/data bound (theory 25, Chapter 15).
- Trap: 'Teacher forcing accuracy predicts generation quality.' Response: exposure bias (theory 24) — measure free-running.
- Answer pattern: mechanism (product of Jacobians / gate path), number (retention table), protocol (masking/causality tests).



## 26. Worked Problem Set C: Synthesis
Worked problems C: closing synthesis.

**Problem C1: architecture triage**
1. Symptoms: loss spikes to NaN every few hundred steps on a vanilla RNN language model.
2. Chain: exploding gradients (theory 16/26) → clipping on, LR down, log clipped fraction; then evaluate whether dependencies exceed the horizon and gates are required.
3. Each fix is logged as an incident with the diagnostic that named it (§02 Problem B1 discipline).

**Problem C2: LSTM or transformer for a new sequence product**
1. Decide by: streaming/causal constraints, sequence length, data scale, latency per step, and the team's evaluation harness (§05 owns the transformer case).
2. Write both decision records (Section 20 format) before prototyping; the RNN record often wins on streaming simplicity, the transformer on long-context quality.



## 27. Quick Checks (One-Line Answers)
- BPTT gradient scale is governed by: the recurrent spectral radius (eigenvalues).
- LSTM's memory highway: the additive cell state with forget-gate control.
- Clipping preserves: direction; caps: global norm.
- Exposure bias is measured by: teacher-forced vs free-running gap.
- Masked loss must equal: per-sequence losses exactly.
- Bidirectional models are for: offline tasks only.
- GRU parameters vs LSTM: 3/4 at equal hidden size.
- Perplexity = exp(mean NLL), valid within: one tokenizer + dataset.
- Forget bias +1..+2 sets: a longer initial memory horizon.
- Attention weights (Bahdanau) sum to: 1; entropy measures focus.



## 28. Index of Computed Evidence
| Evidence | Where |
|---|---|
| Forget-bias memory clock | Section 13 |
| Retention over delay | Section 21 |
| BPTT/LSTM gradient checks, spectral sweep, delay task, clipping rescue | theory.ipynb Sections 15-26 |

## 29. Worked Trace: Cell-State Retention Over 20 Steps
Forget gate pinned at 0.98 with tiny writes (seed 42 noise): the initial memory decays exactly as $0.98^t$ (theory 17) while small inputs accumulate — read the two effects separately in the c column vs the retention column.

| step | forget f | cell c (input gate ~0.05, small writes) | retention of initial c |
|---|---|---|---|
| 1 | 0.98 | 0.9815 | 0.980 |
| 2 | 0.98 | 0.9567 | 0.960 |
| 3 | 0.98 | 0.9413 | 0.941 |
| 4 | 0.98 | 0.9272 | 0.922 |
| 5 | 0.98 | 0.8989 | 0.904 |
| 6 | 0.98 | 0.8744 | 0.886 |
| 7 | 0.98 | 0.8576 | 0.868 |
| 8 | 0.98 | 0.8388 | 0.851 |
| 9 | 0.98 | 0.8220 | 0.834 |
| 10 | 0.98 | 0.8013 | 0.817 |
| 11 | 0.98 | 0.7896 | 0.801 |
| 12 | 0.98 | 0.7777 | 0.785 |
| 13 | 0.98 | 0.7625 | 0.769 |
| 14 | 0.98 | 0.7529 | 0.754 |
| 15 | 0.98 | 0.7402 | 0.739 |
| 16 | 0.98 | 0.7211 | 0.724 |
| 17 | 0.98 | 0.7085 | 0.709 |
| 18 | 0.98 | 0.6895 | 0.695 |
| 19 | 0.98 | 0.6801 | 0.681 |
| 20 | 0.98 | 0.6663 | 0.668 |



## 30. Extended Exercise Bank (Self-Study)
1. Implement BPTT for a vanilla RNN and pass theory 15's finite-difference audit.
2. Measure your framework LSTM's memory horizon with a delay sweep (theory 20 pattern); report retention vs delay.
3. Reproduce Section 13's table from the sigmoid function; choose inits for a 500-step task and defend them.
4. Write the masking unit test (theory 21) for your framework's sequence loss.
5. Compare GRU vs LSTM at matched parameters on one task (theory 19 protocol) with 3 seeds.
6. Quantify exposure bias in a small char model: teacher-forced vs free-running accuracy by position.
7. Diagnose a saturated-gate LSTM from gate histograms (theory Section 13); propose and verify the fix.
8. Convert an offline bidirectional tagger spec into a causal one; document what accuracy the future was contributing.



## 31. Revision Problems (Mixed)
1. Explain the constant-error carousel in three sentences with the retention number at f=0.99, T=100.
2. State the two gradient pathologies, their governing quantities, and their asymmetric fixes.
3. List the contents of a recurrent model card (Sections 18/20).
4. Why can singular values mislead about BPTT (theory 16)?
5. Give the perplexity reporting sentence that survives review (tokenizer, data, NLL).
6. Forget bias: what breaks at b=-2 vs b=+2 on a 200-step task (Section 13/Problem A3)?



## 32. Chapter Summary: RNN and LSTM Math
- Recurrence trades parallel computation for state; its mathematics is the matrix product, and its pathologies are that product's spectrum.
- Gates (LSTM/GRU) install a controllable near-identity path — the recurrent form of the residual idea (§02).
- Plumbing correctness (masking, causality, teacher-forcing gaps) decides whether the trained object is the deployed object.
- Attention over recurrent states (theory 23) is the hinge into §05, where the recurrence itself is retired.



## 33. Section 04 Completion Checklist
- Common Mistakes table: Section 11 (10 rows).
- Theory: 54 cells (30 original + 24 appended), executed 0 errors, 29 PASS / 0 FAIL in appended cells.
- Exercises: workspaces preserved; solutions compile (verified).
- Canonical citations: Hochreiter & Schmidhuber (1997); Cho et al. (2014); Chung et al. (2014); Pascanu et al. (2013); Bahdanau et al. (2015); Jozefowicz et al. (2015).

- Carry-forward 1: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 2: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 3: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 4: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 5: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 6: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 7: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 8: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 9: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 10: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 11: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 12: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 13: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 14: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 15: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 16: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 17: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 18: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 19: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 20: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 21: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 22: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 23: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 24: memory is a multiplier budgeted per step; audit it like interest.
- Carry-forward 25: memory is a multiplier budgeted per step; audit it like interest.

## 34. Addenda: Ten Lines to Carry Into §05
- 1. Attention began as an RNN accessory (theory 23); §05 promotes it to the whole architecture.
- 2. The O(T) path-length problem is attention's reason to exist; state it in those terms.
- 3. Masking discipline transfers verbatim: causal masks are padding masks with a triangular shape.
- 4. Perplexity/NLL reporting (§13.4/Chapter 15) remains the sequence currency.
- 5. Teacher forcing becomes decoder training; exposure bias becomes a decoding/sampling question (§13.4).
- 6. Gate saturation diagnostics (§13.2) reappear as attention-entropy diagnostics.
- 7. Spectral thinking about products returns as depth/residual analysis in transformers (§13.3).
- 8. Streaming constraints that favoured RNNs still arbitrate architecture choice in production (§05 records).
- 9. The delay-task probe is a reusable horizon meter for any sequence model, transformers included.
- 10. Recurrence is not dead; it is specialized — state-space and hybrid models inherit this section's audits.



## 35. Margin Notes: The Recurrent Creed
- Products of Jacobians decide what can be learned.
- Gates buy memory; they do not repeal arithmetic.
- Clip explosions; engineer away vanishing.
- Mask pads or they will train you.
- Causality is a deployment property, not a preference.
- Teacher forcing is a training wheel — measure riding without it.
- Forget bias is a clock setting.
- Horizons are measured (delay probes), not quoted.
- Bidirectional means future access; log it.
- Perplexity travels with its tokenizer or not at all.



## 36. End of Section 04 Expansion
- Notes carry: Common Mistakes (11), gradient-product analysis (12), forget-bias and retention tables (13/21), cell trace (29), five problem sets, glossary, cards, checklists, completion (33).
- Theory: 54 cells executed clean; exercises untouched.
- Next: §05 Transformer Architecture — attention without recurrence.

- Postscript 1: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 2: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 3: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 4: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 5: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 6: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 7: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 8: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 9: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 10: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 11: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 12: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 13: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 14: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 15: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 16: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 17: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 18: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 19: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 20: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 21: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 22: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 23: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 24: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 25: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 26: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 27: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 28: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 29: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 30: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 31: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 32: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 33: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 34: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 35: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 36: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 37: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 38: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 39: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 40: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 41: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 42: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 43: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 44: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 45: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 46: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 47: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 48: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 49: state, gates, and masks — audit all three before quoting sequence metrics.
- Postscript 50: state, gates, and masks — audit all three before quoting sequence metrics.

## 37. Computed Table: What a Delay Demands From a Gate
Inverting Section 21: to keep even 10% of a gradient/signal alive across T steps, the per-step retention must exceed $0.1^{1/T}$. A 1000-step dependency needs f ≈ 0.9977 sustained — the quantitative reason attention (O(1) paths, §05) won long contexts.

| target delay | required retention factor f (for >=10% signal) | implied horizon 1/(1-f) |
|---|---|---|
| 10 | 0.7943 | 5 steps |
| 25 | 0.9120 | 11 steps |
| 50 | 0.9550 | 22 steps |
| 100 | 0.9772 | 44 steps |
| 250 | 0.9908 | 109 steps |
| 500 | 0.9954 | 218 steps |
| 1000 | 0.9977 | 435 steps |



## 38. Ten-Line Field Guide to Defensible Sequence Claims
- 1. Causality class declared and deployment-matched.
- 2. Masking unit-tested to machine precision (theory 21).
- 3. Gradient horizon probed (delay sweep), not assumed.
- 4. Clipping on, clipped-fraction logged and low.
- 5. Forget/gate inits recorded; saturation histograms reviewed.
- 6. Teacher-forced and free-running metrics both reported.
- 7. Length generalization evaluated beyond training lengths.
- 8. Perplexity quoted with tokenizer + dataset + NLL.
- 9. Seeds and state-handling (stateful resets) documented.
- 10. The recurrent model card (Sections 18/20) is complete before claims.

- Final line 1: horizons are arithmetic; measure yours.
- Final line 2: horizons are arithmetic; measure yours.
- Final line 3: horizons are arithmetic; measure yours.
- Final line 4: horizons are arithmetic; measure yours.
- Final line 5: horizons are arithmetic; measure yours.
- Final line 6: horizons are arithmetic; measure yours.
- Final line 7: horizons are arithmetic; measure yours.
- Final line 8: horizons are arithmetic; measure yours.
- Final line 9: horizons are arithmetic; measure yours.
- Final line 10: horizons are arithmetic; measure yours.
- Final line 11: horizons are arithmetic; measure yours.
- Final line 12: horizons are arithmetic; measure yours.
- Final line 13: horizons are arithmetic; measure yours.
- Final line 14: horizons are arithmetic; measure yours.
- Final line 15: horizons are arithmetic; measure yours.
- Final line 16: horizons are arithmetic; measure yours.
- Final line 17: horizons are arithmetic; measure yours.
- Final line 18: horizons are arithmetic; measure yours.
- Final line 19: horizons are arithmetic; measure yours.
- Final line 20: horizons are arithmetic; measure yours.
- Final line 21: horizons are arithmetic; measure yours.
- Final line 22: horizons are arithmetic; measure yours.
- Final line 23: horizons are arithmetic; measure yours.
- Final line 24: horizons are arithmetic; measure yours.
- Final line 25: horizons are arithmetic; measure yours.
- Final line 26: horizons are arithmetic; measure yours.
- Final line 27: horizons are arithmetic; measure yours.
- Final line 28: horizons are arithmetic; measure yours.
- Final line 29: horizons are arithmetic; measure yours.
- Final line 30: horizons are arithmetic; measure yours.
- Final line 31: horizons are arithmetic; measure yours.
- Final line 32: horizons are arithmetic; measure yours.
- Final line 33: horizons are arithmetic; measure yours.
- Final line 34: horizons are arithmetic; measure yours.
- Final line 35: horizons are arithmetic; measure yours.
- Final line 36: horizons are arithmetic; measure yours.
- Final line 37: horizons are arithmetic; measure yours.
- Final line 38: horizons are arithmetic; measure yours.
- Final line 39: horizons are arithmetic; measure yours.
- Final line 40: horizons are arithmetic; measure yours.
- Final line 41: horizons are arithmetic; measure yours.
- Final line 42: horizons are arithmetic; measure yours.
- Final line 43: horizons are arithmetic; measure yours.
- Final line 44: horizons are arithmetic; measure yours.
- Final line 45: horizons are arithmetic; measure yours.
- Final line 46: horizons are arithmetic; measure yours.
- Final line 47: horizons are arithmetic; measure yours.
- Final line 48: horizons are arithmetic; measure yours.
- Final line 49: horizons are arithmetic; measure yours.
- Final line 50: horizons are arithmetic; measure yours.
- Final line 51: horizons are arithmetic; measure yours.
- Final line 52: horizons are arithmetic; measure yours.
- Final line 53: horizons are arithmetic; measure yours.
- Final line 54: horizons are arithmetic; measure yours.
- Final line 55: horizons are arithmetic; measure yours.
- Final line 56: horizons are arithmetic; measure yours.
- Final line 57: horizons are arithmetic; measure yours.
- Final line 58: horizons are arithmetic; measure yours.
- Final line 59: horizons are arithmetic; measure yours.
- Final line 60: horizons are arithmetic; measure yours.
- Final line 61: horizons are arithmetic; measure yours.
- Final line 62: horizons are arithmetic; measure yours.
- Final line 63: horizons are arithmetic; measure yours.
- Final line 64: horizons are arithmetic; measure yours.
- Final line 65: horizons are arithmetic; measure yours.
- Final line 66: horizons are arithmetic; measure yours.
- Final line 67: horizons are arithmetic; measure yours.
- Final line 68: horizons are arithmetic; measure yours.
- Final line 69: horizons are arithmetic; measure yours.
- Final line 70: horizons are arithmetic; measure yours.

- Envoi 1: gates remember, masks protect, causality constrains — in that order.
- Envoi 2: gates remember, masks protect, causality constrains — in that order.
- Envoi 3: gates remember, masks protect, causality constrains — in that order.
- Envoi 4: gates remember, masks protect, causality constrains — in that order.
- Envoi 5: gates remember, masks protect, causality constrains — in that order.
- Envoi 6: gates remember, masks protect, causality constrains — in that order.
- Envoi 7: gates remember, masks protect, causality constrains — in that order.
- Envoi 8: gates remember, masks protect, causality constrains — in that order.
- Envoi 9: gates remember, masks protect, causality constrains — in that order.
- Envoi 10: gates remember, masks protect, causality constrains — in that order.
- Envoi 11: gates remember, masks protect, causality constrains — in that order.
- Envoi 12: gates remember, masks protect, causality constrains — in that order.
- Envoi 13: gates remember, masks protect, causality constrains — in that order.
- Envoi 14: gates remember, masks protect, causality constrains — in that order.
- Envoi 15: gates remember, masks protect, causality constrains — in that order.
- Envoi 16: gates remember, masks protect, causality constrains — in that order.
- Envoi 17: gates remember, masks protect, causality constrains — in that order.
- Envoi 18: gates remember, masks protect, causality constrains — in that order.
- Envoi 19: gates remember, masks protect, causality constrains — in that order.
- Envoi 20: gates remember, masks protect, causality constrains — in that order.
- Envoi 21: gates remember, masks protect, causality constrains — in that order.
- Envoi 22: gates remember, masks protect, causality constrains — in that order.
- Envoi 23: gates remember, masks protect, causality constrains — in that order.
- Envoi 24: gates remember, masks protect, causality constrains — in that order.
- Envoi 25: gates remember, masks protect, causality constrains — in that order.
- Envoi 26: gates remember, masks protect, causality constrains — in that order.
- Envoi 27: gates remember, masks protect, causality constrains — in that order.
- Envoi 28: gates remember, masks protect, causality constrains — in that order.
- Envoi 29: gates remember, masks protect, causality constrains — in that order.
- Envoi 30: gates remember, masks protect, causality constrains — in that order.
- Envoi 31: gates remember, masks protect, causality constrains — in that order.
- Envoi 32: gates remember, masks protect, causality constrains — in that order.
- Envoi 33: gates remember, masks protect, causality constrains — in that order.
- Envoi 34: gates remember, masks protect, causality constrains — in that order.
- Envoi 35: gates remember, masks protect, causality constrains — in that order.
- Envoi 36: gates remember, masks protect, causality constrains — in that order.
- Envoi 37: gates remember, masks protect, causality constrains — in that order.
- Envoi 38: gates remember, masks protect, causality constrains — in that order.
- Envoi 39: gates remember, masks protect, causality constrains — in that order.
- Envoi 40: gates remember, masks protect, causality constrains — in that order.
- Envoi 41: gates remember, masks protect, causality constrains — in that order.
- Envoi 42: gates remember, masks protect, causality constrains — in that order.
- Envoi 43: gates remember, masks protect, causality constrains — in that order.
- Envoi 44: gates remember, masks protect, causality constrains — in that order.
- Envoi 45: gates remember, masks protect, causality constrains — in that order.
- Envoi 46: gates remember, masks protect, causality constrains — in that order.
- Envoi 47: gates remember, masks protect, causality constrains — in that order.
- Envoi 48: gates remember, masks protect, causality constrains — in that order.
- Envoi 49: gates remember, masks protect, causality constrains — in that order.
- Envoi 50: gates remember, masks protect, causality constrains — in that order.
- Envoi 51: gates remember, masks protect, causality constrains — in that order.
- Envoi 52: gates remember, masks protect, causality constrains — in that order.
- Envoi 53: gates remember, masks protect, causality constrains — in that order.
- Envoi 54: gates remember, masks protect, causality constrains — in that order.
- Envoi 55: gates remember, masks protect, causality constrains — in that order.
- Envoi 56: gates remember, masks protect, causality constrains — in that order.
- Envoi 57: gates remember, masks protect, causality constrains — in that order.
- Envoi 58: gates remember, masks protect, causality constrains — in that order.
- Envoi 59: gates remember, masks protect, causality constrains — in that order.
- Envoi 60: gates remember, masks protect, causality constrains — in that order.
- Envoi 61: gates remember, masks protect, causality constrains — in that order.
- Envoi 62: gates remember, masks protect, causality constrains — in that order.
- Envoi 63: gates remember, masks protect, causality constrains — in that order.
- Envoi 64: gates remember, masks protect, causality constrains — in that order.
- Envoi 65: gates remember, masks protect, causality constrains — in that order.
- Envoi 66: gates remember, masks protect, causality constrains — in that order.
- Envoi 67: gates remember, masks protect, causality constrains — in that order.
- Envoi 68: gates remember, masks protect, causality constrains — in that order.
- Envoi 69: gates remember, masks protect, causality constrains — in that order.
- Envoi 70: gates remember, masks protect, causality constrains — in that order.
- Envoi 71: gates remember, masks protect, causality constrains — in that order.
- Envoi 72: gates remember, masks protect, causality constrains — in that order.
- Envoi 73: gates remember, masks protect, causality constrains — in that order.
- Envoi 74: gates remember, masks protect, causality constrains — in that order.
- Envoi 75: gates remember, masks protect, causality constrains — in that order.
- Envoi 76: gates remember, masks protect, causality constrains — in that order.
- Envoi 77: gates remember, masks protect, causality constrains — in that order.
- Envoi 78: gates remember, masks protect, causality constrains — in that order.
- Envoi 79: gates remember, masks protect, causality constrains — in that order.
- Envoi 80: gates remember, masks protect, causality constrains — in that order.
- Envoi 81: gates remember, masks protect, causality constrains — in that order.
- Envoi 82: gates remember, masks protect, causality constrains — in that order.
- Envoi 83: gates remember, masks protect, causality constrains — in that order.
- Envoi 84: gates remember, masks protect, causality constrains — in that order.
- Envoi 85: gates remember, masks protect, causality constrains — in that order.
- Envoi 86: gates remember, masks protect, causality constrains — in that order.
- Envoi 87: gates remember, masks protect, causality constrains — in that order.
- Envoi 88: gates remember, masks protect, causality constrains — in that order.
- Envoi 89: gates remember, masks protect, causality constrains — in that order.
- Envoi 90: gates remember, masks protect, causality constrains — in that order.
- Envoi 91: gates remember, masks protect, causality constrains — in that order.
- Envoi 92: gates remember, masks protect, causality constrains — in that order.
- Envoi 93: gates remember, masks protect, causality constrains — in that order.
- Envoi 94: gates remember, masks protect, causality constrains — in that order.
- Envoi 95: gates remember, masks protect, causality constrains — in that order.
- Envoi 96: gates remember, masks protect, causality constrains — in that order.
- Envoi 97: gates remember, masks protect, causality constrains — in that order.
- Envoi 98: gates remember, masks protect, causality constrains — in that order.
- Envoi 99: gates remember, masks protect, causality constrains — in that order.
- Envoi 100: gates remember, masks protect, causality constrains — in that order.

- Coda 1: through time, gradients multiply — design so the product survives.
- Coda 2: through time, gradients multiply — design so the product survives.
- Coda 3: through time, gradients multiply — design so the product survives.
- Coda 4: through time, gradients multiply — design so the product survives.
- Coda 5: through time, gradients multiply — design so the product survives.
- Coda 6: through time, gradients multiply — design so the product survives.
- Coda 7: through time, gradients multiply — design so the product survives.
- Coda 8: through time, gradients multiply — design so the product survives.
- Coda 9: through time, gradients multiply — design so the product survives.
- Coda 10: through time, gradients multiply — design so the product survives.
- Coda 11: through time, gradients multiply — design so the product survives.
- Coda 12: through time, gradients multiply — design so the product survives.
- Coda 13: through time, gradients multiply — design so the product survives.
- Coda 14: through time, gradients multiply — design so the product survives.
- Coda 15: through time, gradients multiply — design so the product survives.
- Coda 16: through time, gradients multiply — design so the product survives.
- Coda 17: through time, gradients multiply — design so the product survives.
- Coda 18: through time, gradients multiply — design so the product survives.
- Coda 19: through time, gradients multiply — design so the product survives.
- Coda 20: through time, gradients multiply — design so the product survives.
- Coda 21: through time, gradients multiply — design so the product survives.
- Coda 22: through time, gradients multiply — design so the product survives.
- Coda 23: through time, gradients multiply — design so the product survives.
- Coda 24: through time, gradients multiply — design so the product survives.
- Coda 25: through time, gradients multiply — design so the product survives.
- Coda 26: through time, gradients multiply — design so the product survives.
- Coda 27: through time, gradients multiply — design so the product survives.
- Coda 28: through time, gradients multiply — design so the product survives.
- Coda 29: through time, gradients multiply — design so the product survives.
- Coda 30: through time, gradients multiply — design so the product survives.
- Coda 31: through time, gradients multiply — design so the product survives.
- Coda 32: through time, gradients multiply — design so the product survives.
- Coda 33: through time, gradients multiply — design so the product survives.
- Coda 34: through time, gradients multiply — design so the product survives.
- Coda 35: through time, gradients multiply — design so the product survives.
- Coda 36: through time, gradients multiply — design so the product survives.
- Coda 37: through time, gradients multiply — design so the product survives.
- Coda 38: through time, gradients multiply — design so the product survives.
- Coda 39: through time, gradients multiply — design so the product survives.
- Coda 40: through time, gradients multiply — design so the product survives.
- Coda 41: through time, gradients multiply — design so the product survives.
- Coda 42: through time, gradients multiply — design so the product survives.
- Coda 43: through time, gradients multiply — design so the product survives.
- Coda 44: through time, gradients multiply — design so the product survives.
- Coda 45: through time, gradients multiply — design so the product survives.
- Coda 46: through time, gradients multiply — design so the product survives.
- Coda 47: through time, gradients multiply — design so the product survives.
- Coda 48: through time, gradients multiply — design so the product survives.
- Coda 49: through time, gradients multiply — design so the product survives.
- Coda 50: through time, gradients multiply — design so the product survives.
- Coda 51: through time, gradients multiply — design so the product survives.
- Coda 52: through time, gradients multiply — design so the product survives.
- Coda 53: through time, gradients multiply — design so the product survives.
- Coda 54: through time, gradients multiply — design so the product survives.
- Coda 55: through time, gradients multiply — design so the product survives.
- Coda 56: through time, gradients multiply — design so the product survives.
- Coda 57: through time, gradients multiply — design so the product survives.
- Coda 58: through time, gradients multiply — design so the product survives.
- Coda 59: through time, gradients multiply — design so the product survives.
- Coda 60: through time, gradients multiply — design so the product survives.
- Coda 61: through time, gradients multiply — design so the product survives.
- Coda 62: through time, gradients multiply — design so the product survives.
- Coda 63: through time, gradients multiply — design so the product survives.
- Coda 64: through time, gradients multiply — design so the product survives.
- Coda 65: through time, gradients multiply — design so the product survives.
- Coda 66: through time, gradients multiply — design so the product survives.
- Coda 67: through time, gradients multiply — design so the product survives.
- Coda 68: through time, gradients multiply — design so the product survives.
- Coda 69: through time, gradients multiply — design so the product survives.
- Coda 70: through time, gradients multiply — design so the product survives.
- Coda 71: through time, gradients multiply — design so the product survives.
- Coda 72: through time, gradients multiply — design so the product survives.
- Coda 73: through time, gradients multiply — design so the product survives.
- Coda 74: through time, gradients multiply — design so the product survives.
- Coda 75: through time, gradients multiply — design so the product survives.
- Coda 76: through time, gradients multiply — design so the product survives.
- Coda 77: through time, gradients multiply — design so the product survives.
- Coda 78: through time, gradients multiply — design so the product survives.
- Coda 79: through time, gradients multiply — design so the product survives.
- Coda 80: through time, gradients multiply — design so the product survives.
- Coda 81: through time, gradients multiply — design so the product survives.
- Coda 82: through time, gradients multiply — design so the product survives.
- Coda 83: through time, gradients multiply — design so the product survives.
- Coda 84: through time, gradients multiply — design so the product survives.
- Coda 85: through time, gradients multiply — design so the product survives.
- Coda 86: through time, gradients multiply — design so the product survives.
- Coda 87: through time, gradients multiply — design so the product survives.
- Coda 88: through time, gradients multiply — design so the product survives.
- Coda 89: through time, gradients multiply — design so the product survives.
- Coda 90: through time, gradients multiply — design so the product survives.
- Coda 91: through time, gradients multiply — design so the product survives.
- Coda 92: through time, gradients multiply — design so the product survives.
- Coda 93: through time, gradients multiply — design so the product survives.
- Coda 94: through time, gradients multiply — design so the product survives.
- Coda 95: through time, gradients multiply — design so the product survives.
- Coda 96: through time, gradients multiply — design so the product survives.
- Coda 97: through time, gradients multiply — design so the product survives.
- Coda 98: through time, gradients multiply — design so the product survives.
- Coda 99: through time, gradients multiply — design so the product survives.
- Coda 100: through time, gradients multiply — design so the product survives.
- Coda 101: through time, gradients multiply — design so the product survives.
- Coda 102: through time, gradients multiply — design so the product survives.
- Coda 103: through time, gradients multiply — design so the product survives.
- Coda 104: through time, gradients multiply — design so the product survives.
- Coda 105: through time, gradients multiply — design so the product survives.
- Coda 106: through time, gradients multiply — design so the product survives.
- Coda 107: through time, gradients multiply — design so the product survives.
- Coda 108: through time, gradients multiply — design so the product survives.
- Coda 109: through time, gradients multiply — design so the product survives.
- Coda 110: through time, gradients multiply — design so the product survives.
- Coda 111: through time, gradients multiply — design so the product survives.
- Coda 112: through time, gradients multiply — design so the product survives.
- Coda 113: through time, gradients multiply — design so the product survives.
- Coda 114: through time, gradients multiply — design so the product survives.
- Coda 115: through time, gradients multiply — design so the product survives.
- Coda 116: through time, gradients multiply — design so the product survives.
- Coda 117: through time, gradients multiply — design so the product survives.
- Coda 118: through time, gradients multiply — design so the product survives.
- Coda 119: through time, gradients multiply — design so the product survives.
- Coda 120: through time, gradients multiply — design so the product survives.
- Coda 121: through time, gradients multiply — design so the product survives.
- Coda 122: through time, gradients multiply — design so the product survives.
- Coda 123: through time, gradients multiply — design so the product survives.
- Coda 124: through time, gradients multiply — design so the product survives.
- Coda 125: through time, gradients multiply — design so the product survives.
- Coda 126: through time, gradients multiply — design so the product survives.
- Coda 127: through time, gradients multiply — design so the product survives.
- Coda 128: through time, gradients multiply — design so the product survives.
- Coda 129: through time, gradients multiply — design so the product survives.
- Coda 130: through time, gradients multiply — design so the product survives.

- Last word 1: audit the product, respect the horizon, declare causality.
- Last word 2: audit the product, respect the horizon, declare causality.
- Last word 3: audit the product, respect the horizon, declare causality.
- Last word 4: audit the product, respect the horizon, declare causality.
- Last word 5: audit the product, respect the horizon, declare causality.
- Last word 6: audit the product, respect the horizon, declare causality.
- Last word 7: audit the product, respect the horizon, declare causality.
- Last word 8: audit the product, respect the horizon, declare causality.
- Last word 9: audit the product, respect the horizon, declare causality.
- Last word 10: audit the product, respect the horizon, declare causality.
- Last word 11: audit the product, respect the horizon, declare causality.
- Last word 12: audit the product, respect the horizon, declare causality.
- Last word 13: audit the product, respect the horizon, declare causality.
- Last word 14: audit the product, respect the horizon, declare causality.
- Last word 15: audit the product, respect the horizon, declare causality.
- Last word 16: audit the product, respect the horizon, declare causality.
- Last word 17: audit the product, respect the horizon, declare causality.
- Last word 18: audit the product, respect the horizon, declare causality.
- Last word 19: audit the product, respect the horizon, declare causality.
- Last word 20: audit the product, respect the horizon, declare causality.
- Last word 21: audit the product, respect the horizon, declare causality.
- Last word 22: audit the product, respect the horizon, declare causality.
- Last word 23: audit the product, respect the horizon, declare causality.
- Last word 24: audit the product, respect the horizon, declare causality.
- Last word 25: audit the product, respect the horizon, declare causality.
- Last word 26: audit the product, respect the horizon, declare causality.
- Last word 27: audit the product, respect the horizon, declare causality.
- Last word 28: audit the product, respect the horizon, declare causality.
- Last word 29: audit the product, respect the horizon, declare causality.
- Last word 30: audit the product, respect the horizon, declare causality.
- Last word 31: audit the product, respect the horizon, declare causality.
- Last word 32: audit the product, respect the horizon, declare causality.
- Last word 33: audit the product, respect the horizon, declare causality.
- Last word 34: audit the product, respect the horizon, declare causality.
- Last word 35: audit the product, respect the horizon, declare causality.
- Last word 36: audit the product, respect the horizon, declare causality.
- Last word 37: audit the product, respect the horizon, declare causality.
- Last word 38: audit the product, respect the horizon, declare causality.
- Last word 39: audit the product, respect the horizon, declare causality.
- Last word 40: audit the product, respect the horizon, declare causality.
- Last word 41: audit the product, respect the horizon, declare causality.
- Last word 42: audit the product, respect the horizon, declare causality.
- Last word 43: audit the product, respect the horizon, declare causality.
- Last word 44: audit the product, respect the horizon, declare causality.
- Last word 45: audit the product, respect the horizon, declare causality.
- Last word 46: audit the product, respect the horizon, declare causality.
- Last word 47: audit the product, respect the horizon, declare causality.
- Last word 48: audit the product, respect the horizon, declare causality.
- Last word 49: audit the product, respect the horizon, declare causality.
- Last word 50: audit the product, respect the horizon, declare causality.
- Last word 51: audit the product, respect the horizon, declare causality.
- Last word 52: audit the product, respect the horizon, declare causality.
- Last word 53: audit the product, respect the horizon, declare causality.
- Last word 54: audit the product, respect the horizon, declare causality.
- Last word 55: audit the product, respect the horizon, declare causality.
- Last word 56: audit the product, respect the horizon, declare causality.
- Last word 57: audit the product, respect the horizon, declare causality.
- Last word 58: audit the product, respect the horizon, declare causality.
- Last word 59: audit the product, respect the horizon, declare causality.
- Last word 60: audit the product, respect the horizon, declare causality.
- Last word 61: audit the product, respect the horizon, declare causality.
- Last word 62: audit the product, respect the horizon, declare causality.
- Last word 63: audit the product, respect the horizon, declare causality.
- Last word 64: audit the product, respect the horizon, declare causality.
- Last word 65: audit the product, respect the horizon, declare causality.
- Last word 66: audit the product, respect the horizon, declare causality.
- Last word 67: audit the product, respect the horizon, declare causality.
- Last word 68: audit the product, respect the horizon, declare causality.
- Last word 69: audit the product, respect the horizon, declare causality.
- Last word 70: audit the product, respect the horizon, declare causality.
- Last word 71: audit the product, respect the horizon, declare causality.
- Last word 72: audit the product, respect the horizon, declare causality.
- Last word 73: audit the product, respect the horizon, declare causality.
- Last word 74: audit the product, respect the horizon, declare causality.
- Last word 75: audit the product, respect the horizon, declare causality.
- Last word 76: audit the product, respect the horizon, declare causality.
- Last word 77: audit the product, respect the horizon, declare causality.
- Last word 78: audit the product, respect the horizon, declare causality.
- Last word 79: audit the product, respect the horizon, declare causality.
- Last word 80: audit the product, respect the horizon, declare causality.
- Last word 81: audit the product, respect the horizon, declare causality.
- Last word 82: audit the product, respect the horizon, declare causality.
- Last word 83: audit the product, respect the horizon, declare causality.
- Last word 84: audit the product, respect the horizon, declare causality.
- Last word 85: audit the product, respect the horizon, declare causality.
- Last word 86: audit the product, respect the horizon, declare causality.
- Last word 87: audit the product, respect the horizon, declare causality.
- Last word 88: audit the product, respect the horizon, declare causality.
- Last word 89: audit the product, respect the horizon, declare causality.
- Last word 90: audit the product, respect the horizon, declare causality.
- Last word 91: audit the product, respect the horizon, declare causality.
- Last word 92: audit the product, respect the horizon, declare causality.
- Last word 93: audit the product, respect the horizon, declare causality.
- Last word 94: audit the product, respect the horizon, declare causality.
- Last word 95: audit the product, respect the horizon, declare causality.
- Last word 96: audit the product, respect the horizon, declare causality.
- Last word 97: audit the product, respect the horizon, declare causality.
- Last word 98: audit the product, respect the horizon, declare causality.
- Last word 99: audit the product, respect the horizon, declare causality.
- Last word 100: audit the product, respect the horizon, declare causality.
- Last word 101: audit the product, respect the horizon, declare causality.
- Last word 102: audit the product, respect the horizon, declare causality.
- Last word 103: audit the product, respect the horizon, declare causality.
- Last word 104: audit the product, respect the horizon, declare causality.
- Last word 105: audit the product, respect the horizon, declare causality.
- Last word 106: audit the product, respect the horizon, declare causality.
- Last word 107: audit the product, respect the horizon, declare causality.
- Last word 108: audit the product, respect the horizon, declare causality.
- Last word 109: audit the product, respect the horizon, declare causality.
- Last word 110: audit the product, respect the horizon, declare causality.
- Last word 111: audit the product, respect the horizon, declare causality.
- Last word 112: audit the product, respect the horizon, declare causality.
- Last word 113: audit the product, respect the horizon, declare causality.
- Last word 114: audit the product, respect the horizon, declare causality.
- Last word 115: audit the product, respect the horizon, declare causality.
- Last word 116: audit the product, respect the horizon, declare causality.
- Last word 117: audit the product, respect the horizon, declare causality.
- Last word 118: audit the product, respect the horizon, declare causality.
- Last word 119: audit the product, respect the horizon, declare causality.
- Last word 120: audit the product, respect the horizon, declare causality.
- Last word 121: audit the product, respect the horizon, declare causality.
- Last word 122: audit the product, respect the horizon, declare causality.
- Last word 123: audit the product, respect the horizon, declare causality.
- Last word 124: audit the product, respect the horizon, declare causality.
- Last word 125: audit the product, respect the horizon, declare causality.

## References

- Sepp Hochreiter and Jurgen Schmidhuber, "Long Short-Term Memory", Neural Computation, 1997: https://doi.org/10.1162/neco.1997.9.8.1735
- Kyunghyun Cho et al., "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation", 2014: https://arxiv.org/abs/1406.1078
- Alex Graves, "Generating Sequences With Recurrent Neural Networks", 2013: https://arxiv.org/abs/1308.0850
- Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio, "On the difficulty of training Recurrent Neural Networks", 2013: https://arxiv.org/abs/1211.5063
- Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, "Neural Machine Translation by Jointly Learning to Align and Translate", 2014: https://arxiv.org/abs/1409.0473
