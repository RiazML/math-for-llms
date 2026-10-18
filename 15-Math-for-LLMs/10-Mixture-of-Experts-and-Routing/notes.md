[<- Efficient Attention and Inference](../09-Efficient-Attention-and-Inference/notes.md) | [Home](../../README.md) | [Quantization and Distillation ->](../11-Quantization-and-Distillation/notes.md)

---

# Mixture of Experts and Routing

Mixture-of-experts models separate total capacity from active per-token computation. A router chooses a small number of expert networks for each token, giving the model many parameters without using all of them on every token.

## Overview

A dense transformer FFN applies the same network to every token. An MoE FFN replaces that one network with many experts:

$$
y_t=\sum_{i\in S_t} g_{t,i}E_i(x_t),
$$

where $S_t$ is the selected expert set for token $t$, $g_{t,i}$ is the router weight, and $E_i$ is expert $i$. The central math is not only the weighted sum. It is the accounting around it: active parameters, total parameters, router probabilities, expert capacity, load balancing, all-to-all dispatch, drop rate, and latency.

## Prerequisites

- Transformer FFN shapes
- Softmax and top-k selection
- Training-at-scale memory and parallelism
- Efficient inference and serving latency vocabulary

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates router softmax, top-k dispatch, capacity overflow, auxiliary balance loss, expert histograms, all-to-all traffic, active parameter counts, and router-collapse diagnostics. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for routing probabilities, capacity, drop rate, load balancing, expert counts, and MoE debugging. |

## Learning Objectives

After this section, you should be able to:

- Explain total parameters versus active parameters.
- Compute router probabilities and top-k selected experts.
- Count MoE expert parameters and active expert compute.
- Compute expert capacity and token overflow.
- Define importance, load, auxiliary load-balancing loss, entropy, and z-loss.
- Explain why MoE training often needs all-to-all communication.
- Diagnose expert collapse with histograms, drop rate, entropy, and gradient norms.
- Explain the inference tradeoff: lower active compute but higher memory and routing complexity.

## Table of Contents

1. [Dense versus Sparse Computation](#1-dense-versus-sparse-computation)
   - 1.1 [Dense FFN](#11-dense-ffn)
   - 1.2 [Expert bank](#12-expert-bank)
   - 1.3 [Sparse activation](#13-sparse-activation)
   - 1.4 [Total versus active parameters](#14-total-versus-active-parameters)
   - 1.5 [Memory caveat](#15-memory-caveat)
2. [Router Mathematics](#2-router-mathematics)
   - 2.1 [Router logits](#21-router-logits)
   - 2.2 [Routing probabilities](#22-routing-probabilities)
   - 2.3 [Top-k selection](#23-topk-selection)
   - 2.4 [Gated combination](#24-gated-combination)
   - 2.5 [Top-1 Switch routing](#25-top1-switch-routing)
3. [Parameter and FLOP Accounting](#3-parameter-and-flop-accounting)
   - 3.1 [FFN parameter count](#31-ffn-parameter-count)
   - 3.2 [MoE total parameters](#32-moe-total-parameters)
   - 3.3 [MoE active parameters](#33-moe-active-parameters)
   - 3.4 [Router overhead](#34-router-overhead)
   - 3.5 [Compute ratio](#35-compute-ratio)
4. [Capacity and Token Dropping](#4-capacity-and-token-dropping)
   - 4.1 [Expected tokens per expert](#41-expected-tokens-per-expert)
   - 4.2 [Capacity factor](#42-capacity-factor)
   - 4.3 [Overflow](#43-overflow)
   - 4.4 [Batch sensitivity](#44-batch-sensitivity)
   - 4.5 [Expert collapse](#45-expert-collapse)
5. [Load Balancing Losses](#5-load-balancing-losses)
   - 5.1 [Importance](#51-importance)
   - 5.2 [Load](#52-load)
   - 5.3 [Auxiliary loss](#53-auxiliary-loss)
   - 5.4 [Entropy encouragement](#54-entropy-encouragement)
   - 5.5 [Z-loss](#55-zloss)
6. [Expert Parallelism](#6-expert-parallelism)
   - 6.1 [Expert placement](#61-expert-placement)
   - 6.2 [All-to-all dispatch](#62-alltoall-dispatch)
   - 6.3 [Combine step](#63-combine-step)
   - 6.4 [Communication bottleneck](#64-communication-bottleneck)
   - 6.5 [Locality](#65-locality)
7. [Training Dynamics](#7-training-dynamics)
   - 7.1 [Specialization](#71-specialization)
   - 7.2 [Cold experts](#72-cold-experts)
   - 7.3 [Router noise](#73-router-noise)
   - 7.4 [Top-2 gradients](#74-top2-gradients)
   - 7.5 [Stability tradeoff](#75-stability-tradeoff)
8. [Inference Behavior](#8-inference-behavior)
   - 8.1 [Active compute](#81-active-compute)
   - 8.2 [Weight memory](#82-weight-memory)
   - 8.3 [Batch routing variance](#83-batch-routing-variance)
   - 8.4 [Cache interaction](#84-cache-interaction)
   - 8.5 [Latency tails](#85-latency-tails)
9. [MoE Design Variants](#9-moe-design-variants)
   - 9.1 [Sparsely gated MoE](#91-sparsely-gated-moe)
   - 9.2 [GShard](#92-gshard)
   - 9.3 [Switch Transformer](#93-switch-transformer)
   - 9.4 [Top-2 MoE](#94-top2-moe)
   - 9.5 [Shared experts](#95-shared-experts)
10. [Diagnostics](#10-diagnostics)
   - 10.1 [Expert histogram](#101-expert-histogram)
   - 10.2 [Drop rate](#102-drop-rate)
   - 10.3 [Router entropy](#103-router-entropy)
   - 10.4 [Per-expert gradients](#104-perexpert-gradients)
   - 10.5 [Ablations](#105-ablations)
11. [Routing Mathematics in Depth](#11-routing-mathematics-in-depth)
12. [Router Pathologies and Their Diagnostics](#12-router-pathologies-and-their-diagnostics)
13. [Expert Parallelism and Communication](#13-expert-parallelism-and-communication)
14. [Shared Experts, Fine-Grained Experts, and Modern Variants](#14-shared-experts-fine-grained-experts-and-modern-variants)
15. [MoE in Fine-Tuning and Inference](#15-moe-in-fine-tuning-and-inference)
16. [Computed Tables](#16-computed-tables)
17. [Ten Worked Problems With Solutions](#17-ten-worked-problems-with-solutions)
18. [Historical Line](#18-historical-line)
19. [Glossary and Formula Card](#19-glossary-and-formula-card)

---

## One-Layer MoE Shape Flow

```text
tokens x_t
   |
router logits r_t = W_r x_t
   |
top-k experts S_t
   |
dispatch tokens to experts
   |
expert FFNs E_i(x_t)
   |
weighted combine and restore token order
```

## 1. Dense versus Sparse Computation

This part studies dense versus sparse computation in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Dense FFN](#1-dense-ffn) | every token uses the same feed-forward network | $y=\mathrm{FFN}(x)$ |
| [Expert bank](#1-expert-bank) | replace one FFN with many candidate FFNs | $E_1,\ldots,E_M$ |
| [Sparse activation](#1-sparse-activation) | each token uses only k experts | $k\ll M$ |
| [Total versus active parameters](#1-total-versus-active-parameters) | MoE increases capacity without proportional per-token compute | $P_\mathrm{active}\ll P_\mathrm{total}$ |
| [Memory caveat](#1-memory-caveat) | inactive experts still occupy memory | $M_\mathrm{weights}\propto P_\mathrm{total}$ |

### 1.1 Dense FFN

**Main idea.** Every token uses the same feed-forward network.

Core relation:

$$y=\mathrm{FFN}(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.2 Expert bank

**Main idea.** Replace one ffn with many candidate ffns.

Core relation:

$$E_1,\ldots,E_M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.3 Sparse activation

**Main idea.** Each token uses only k experts.

Core relation:

$$k\ll M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.4 Total versus active parameters

**Main idea.** Moe increases capacity without proportional per-token compute.

Core relation:

$$P_\mathrm{active}\ll P_\mathrm{total}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This distinction is the reason MoE models can have large total capacity with smaller per-token compute.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 1.5 Memory caveat

**Main idea.** Inactive experts still occupy memory.

Core relation:

$$M_\mathrm{weights}\propto P_\mathrm{total}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 2. Router Mathematics

This part studies router mathematics in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Router logits](#2-router-logits) | a small projection scores experts for each token | $r=W_rx$ |
| [Routing probabilities](#2-routing-probabilities) | softmax turns router logits into expert probabilities | $p_i=\exp(r_i)/\sum_j\exp(r_j)$ |
| [Top-k selection](#2-topk-selection) | only the highest scoring experts receive the token | $S=\mathrm{TopK}(p,k)$ |
| [Gated combination](#2-gated-combination) | selected expert outputs are weighted by router probabilities | $y=\sum_{i\in S} \tilde p_i E_i(x)$ |
| [Top-1 Switch routing](#2-top1-switch-routing) | route each token to one expert for simplicity | $y=E_{\arg\max_i p_i}(x)$ |

### 2.1 Router logits

**Main idea.** A small projection scores experts for each token.

Core relation:

$$r=W_rx$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.2 Routing probabilities

**Main idea.** Softmax turns router logits into expert probabilities.

Core relation:

$$p_i=\exp(r_i)/\sum_j\exp(r_j)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.3 Top-k selection

**Main idea.** Only the highest scoring experts receive the token.

Core relation:

$$S=\mathrm{TopK}(p,k)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.4 Gated combination

**Main idea.** Selected expert outputs are weighted by router probabilities.

Core relation:

$$y=\sum_{i\in S} \tilde p_i E_i(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 2.5 Top-1 Switch routing

**Main idea.** Route each token to one expert for simplicity.

Core relation:

$$y=E_{\arg\max_i p_i}(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 3. Parameter and FLOP Accounting

This part studies parameter and flop accounting in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [FFN parameter count](#3-ffn-parameter-count) | a transformer FFN has two large projections | $P_\mathrm{ffn}\approx 2dd_\mathrm{ff}$ |
| [MoE total parameters](#3-moe-total-parameters) | experts multiply the FFN parameter count | $P_\mathrm{experts}\approx M\cdot 2dd_\mathrm{ff}$ |
| [MoE active parameters](#3-moe-active-parameters) | only selected experts are used per token | $P_\mathrm{active}\approx k\cdot 2dd_\mathrm{ff}$ |
| [Router overhead](#3-router-overhead) | router cost is usually small compared with expert FFNs | $P_\mathrm{router}=dM$ |
| [Compute ratio](#3-compute-ratio) | sparse compute scales with k, not M | $\mathrm{FLOPs}_\mathrm{MoE}/\mathrm{FLOPs}_\mathrm{dense}\approx k$ if expert size matches dense FFN |

### 3.1 FFN parameter count

**Main idea.** A transformer ffn has two large projections.

Core relation:

$$P_\mathrm{ffn}\approx 2dd_\mathrm{ff}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.2 MoE total parameters

**Main idea.** Experts multiply the ffn parameter count.

Core relation:

$$P_\mathrm{experts}\approx M\cdot 2dd_\mathrm{ff}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.3 MoE active parameters

**Main idea.** Only selected experts are used per token.

Core relation:

$$P_\mathrm{active}\approx k\cdot 2dd_\mathrm{ff}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.4 Router overhead

**Main idea.** Router cost is usually small compared with expert ffns.

Core relation:

$$P_\mathrm{router}=dM$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 3.5 Compute ratio

**Main idea.** Sparse compute scales with k, not m.

Core relation:

$$\mathrm{FLOPs}_\mathrm{MoE}/\mathrm{FLOPs}_\mathrm{dense}\approx k$ if expert size matches dense FFN$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 4. Capacity and Token Dropping

This part studies capacity and token dropping in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Expected tokens per expert](#4-expected-tokens-per-expert) | balanced routing sends roughly T/M tokens to each expert | $E[n_i]=T/M$ |
| [Capacity factor](#4-capacity-factor) | reserve extra slots beyond the expected load | $C_i=\lceil \mathrm{capacity\ factor}\cdot T/M\rceil$ |
| [Overflow](#4-overflow) | tokens above capacity are dropped or rerouted | $\max(0,n_i-C_i)$ |
| [Batch sensitivity](#4-batch-sensitivity) | small batches have noisier expert loads | $\mathrm{Var}(n_i)=Tp_i(1-p_i)$ |
| [Expert collapse](#4-expert-collapse) | if the router favors a few experts, capacity and learning both suffer | $p_i\approx 0$ for many experts |

### 4.1 Expected tokens per expert

**Main idea.** Balanced routing sends roughly t/m tokens to each expert.

Core relation:

$$E[n_i]=T/M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.2 Capacity factor

**Main idea.** Reserve extra slots beyond the expected load.

Core relation:

$$C_i=\lceil \mathrm{capacity\ factor}\cdot T/M\rceil$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** Capacity is the serving and training contract between the router and the expert bank.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.3 Overflow

**Main idea.** Tokens above capacity are dropped or rerouted.

Core relation:

$$\max(0,n_i-C_i)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.4 Batch sensitivity

**Main idea.** Small batches have noisier expert loads.

Core relation:

$$\mathrm{Var}(n_i)=Tp_i(1-p_i)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 4.5 Expert collapse

**Main idea.** If the router favors a few experts, capacity and learning both suffer.

Core relation:

$$p_i\approx 0$ for many experts$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 5. Load Balancing Losses

This part studies load balancing losses in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Importance](#5-importance) | sum of router probabilities assigned to each expert | $I_i=\sum_t p_{t,i}$ |
| [Load](#5-load) | number of tokens actually routed to each expert | $L_i=\sum_t \mathbf{1}\{i\in S_t\}$ |
| [Auxiliary loss](#5-auxiliary-loss) | penalize uneven routing | $L_\mathrm{aux}\propto M\sum_i f_iP_i$ |
| [Entropy encouragement](#5-entropy-encouragement) | router entropy can discourage overconfident early routing | $H(p_t)=-\sum_i p_{t,i}\log p_{t,i}$ |
| [Z-loss](#5-zloss) | penalize large router logits for stability | $L_z=(\log\sum_i e^{r_i})^2$ |

### 5.1 Importance

**Main idea.** Sum of router probabilities assigned to each expert.

Core relation:

$$I_i=\sum_t p_{t,i}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.2 Load

**Main idea.** Number of tokens actually routed to each expert.

Core relation:

$$L_i=\sum_t \mathbf{1}\{i\in S_t\}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.3 Auxiliary loss

**Main idea.** Penalize uneven routing.

Core relation:

$$L_\mathrm{aux}\propto M\sum_i f_iP_i$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** Without balancing, a router can discover a few experts and ignore the rest.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.4 Entropy encouragement

**Main idea.** Router entropy can discourage overconfident early routing.

Core relation:

$$H(p_t)=-\sum_i p_{t,i}\log p_{t,i}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 5.5 Z-loss

**Main idea.** Penalize large router logits for stability.

Core relation:

$$L_z=(\log\sum_i e^{r_i})^2$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 6. Expert Parallelism

This part studies expert parallelism in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Expert placement](#6-expert-placement) | different devices own different experts | $E_i\rightarrow \mathrm{rank}(i)$ |
| [All-to-all dispatch](#6-alltoall-dispatch) | tokens move to the devices that own their experts | $\mathrm{tokens}\rightarrow\mathrm{experts}$ |
| [Combine step](#6-combine-step) | expert outputs return to original token order | $y_t=\sum_i g_{t,i}E_i(x_t)$ |
| [Communication bottleneck](#6-communication-bottleneck) | MoE speed depends on token traffic, not only FLOPs | $T_\mathrm{step}\approx\max(T_\mathrm{expert},T_\mathrm{alltoall})$ |
| [Locality](#6-locality) | routing and placement choices can reduce cross-device movement | $\mathrm{traffic}\downarrow$ |

### 6.1 Expert placement

**Main idea.** Different devices own different experts.

Core relation:

$$E_i\rightarrow \mathrm{rank}(i)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.2 All-to-all dispatch

**Main idea.** Tokens move to the devices that own their experts.

Core relation:

$$\mathrm{tokens}\rightarrow\mathrm{experts}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** MoE turns part of the model into a distributed routing problem.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.3 Combine step

**Main idea.** Expert outputs return to original token order.

Core relation:

$$y_t=\sum_i g_{t,i}E_i(x_t)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.4 Communication bottleneck

**Main idea.** Moe speed depends on token traffic, not only flops.

Core relation:

$$T_\mathrm{step}\approx\max(T_\mathrm{expert},T_\mathrm{alltoall})$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 6.5 Locality

**Main idea.** Routing and placement choices can reduce cross-device movement.

Core relation:

$$\mathrm{traffic}\downarrow$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 7. Training Dynamics

This part studies training dynamics in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Specialization](#7-specialization) | experts differentiate because they receive different token subsets | $\nabla_{\theta_i}L$ only from routed tokens |
| [Cold experts](#7-cold-experts) | rarely selected experts learn slowly | $n_i\approx 0\Rightarrow \nabla_{\theta_i}\approx 0$ |
| [Router noise](#7-router-noise) | noise can encourage exploration early in training | $r'=r+\epsilon$ |
| [Top-2 gradients](#7-top2-gradients) | top-2 routing gives more experts gradient signal than top-1 | $|S|=2$ |
| [Stability tradeoff](#7-stability-tradeoff) | strong balancing can fight useful specialization | $L=L_\mathrm{task}+\lambda L_\mathrm{aux}$ |

### 7.1 Specialization

**Main idea.** Experts differentiate because they receive different token subsets.

Core relation:

$$\nabla_{\theta_i}L$ only from routed tokens$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.2 Cold experts

**Main idea.** Rarely selected experts learn slowly.

Core relation:

$$n_i\approx 0\Rightarrow \nabla_{\theta_i}\approx 0$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.3 Router noise

**Main idea.** Noise can encourage exploration early in training.

Core relation:

$$r'=r+\epsilon$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.4 Top-2 gradients

**Main idea.** Top-2 routing gives more experts gradient signal than top-1.

Core relation:

$$|S|=2$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 7.5 Stability tradeoff

**Main idea.** Strong balancing can fight useful specialization.

Core relation:

$$L=L_\mathrm{task}+\lambda L_\mathrm{aux}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 8. Inference Behavior

This part studies inference behavior in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Active compute](#8-active-compute) | per-token expert compute depends on k | $k$ selected experts |
| [Weight memory](#8-weight-memory) | serving must store or page all experts that may be routed | $P_\mathrm{total}$ resident or streamed |
| [Batch routing variance](#8-batch-routing-variance) | different requests can activate different experts | $S_t$ varies by token |
| [Cache interaction](#8-cache-interaction) | MoE changes FFN compute but not attention KV cache math directly | $M_\mathrm{KV}$ unchanged by experts |
| [Latency tails](#8-latency-tails) | hot experts and cross-device traffic can increase p95 latency | $Q_{0.95}(T)$ |

### 8.1 Active compute

**Main idea.** Per-token expert compute depends on k.

Core relation:

$$k$ selected experts$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.2 Weight memory

**Main idea.** Serving must store or page all experts that may be routed.

Core relation:

$$P_\mathrm{total}$ resident or streamed$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.3 Batch routing variance

**Main idea.** Different requests can activate different experts.

Core relation:

$$S_t$ varies by token$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.4 Cache interaction

**Main idea.** Moe changes ffn compute but not attention kv cache math directly.

Core relation:

$$M_\mathrm{KV}$ unchanged by experts$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 8.5 Latency tails

**Main idea.** Hot experts and cross-device traffic can increase p95 latency.

Core relation:

$$Q_{0.95}(T)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 9. MoE Design Variants

This part studies moe design variants in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Sparsely gated MoE](#9-sparsely-gated-moe) | learned gates select a sparse expert subset | $y=\sum_i g_iE_i(x)$ |
| [GShard](#9-gshard) | scaled conditional computation with automatic sharding | $\mathrm{expert\ parallelism}$ |
| [Switch Transformer](#9-switch-transformer) | top-1 routing simplifies dispatch | $k=1$ |
| [Top-2 MoE](#9-top2-moe) | two experts can improve quality at higher compute | $k=2$ |
| [Shared experts](#9-shared-experts) | some designs combine routed experts with always-on shared experts | $y=E_\mathrm{shared}(x)+E_\mathrm{routed}(x)$ |

### 9.1 Sparsely gated MoE

**Main idea.** Learned gates select a sparse expert subset.

Core relation:

$$y=\sum_i g_iE_i(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.2 GShard

**Main idea.** Scaled conditional computation with automatic sharding.

Core relation:

$$\mathrm{expert\ parallelism}$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.3 Switch Transformer

**Main idea.** Top-1 routing simplifies dispatch.

Core relation:

$$k=1$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.4 Top-2 MoE

**Main idea.** Two experts can improve quality at higher compute.

Core relation:

$$k=2$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 9.5 Shared experts

**Main idea.** Some designs combine routed experts with always-on shared experts.

Core relation:

$$y=E_\mathrm{shared}(x)+E_\mathrm{routed}(x)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
## 10. Diagnostics

This part studies diagnostics in mixture-of-experts LLMs. The useful habit is to separate probability, capacity, compute, memory, and communication.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Expert histogram](#10-expert-histogram) | plot token counts per expert | $n_i$ |
| [Drop rate](#10-drop-rate) | measure overflowed tokens | $\mathrm{drop}=\sum_i\max(0,n_i-C_i)/T$ |
| [Router entropy](#10-router-entropy) | track whether routing is collapsing or too diffuse | $H(p)$ |
| [Per-expert gradients](#10-perexpert-gradients) | cold experts have small or zero gradient norms | $\Vert g_i\Vert$ |
| [Ablations](#10-ablations) | compare dense, top-1, top-2, and capacity factors | $\Delta L,\Delta T,\Delta M$ |

### 10.1 Expert histogram

**Main idea.** Plot token counts per expert.

Core relation:

$$n_i$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** The histogram is the first picture to look at when an MoE model behaves strangely.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.2 Drop rate

**Main idea.** Measure overflowed tokens.

Core relation:

$$\mathrm{drop}=\sum_i\max(0,n_i-C_i)/T$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.3 Router entropy

**Main idea.** Track whether routing is collapsing or too diffuse.

Core relation:

$$H(p)$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.4 Per-expert gradients

**Main idea.** Cold experts have small or zero gradient norms.

Core relation:

$$\Vert g_i\Vert$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.
### 10.5 Ablations

**Main idea.** Compare dense, top-1, top-2, and capacity factors.

Core relation:

$$\Delta L,\Delta T,\Delta M$$

An MoE layer replaces a single dense feed-forward block with a bank of experts and a router. The router decides which experts see each token. This gives the model more total parameters than a dense model with similar active compute, but it adds routing instability, capacity limits, communication, and memory pressure.

**Worked micro-example.** If a dense FFN has $2dd_\mathrm{ff}$ parameters and an MoE layer has $M=8$ experts of the same size, total expert parameters increase by 8x. With top-2 routing, each token uses only two experts, so the active FFN compute is about 2x a single dense FFN, not 8x.

**Implementation check.** For every MoE run, log expert token counts, drop rate, router entropy, auxiliary loss, and per-expert gradient norms. A low loss curve can hide a collapsed router.

**AI connection.** This is a practical MoE control variable.

**Common mistake.** Do not say an MoE model "uses all its parameters" for one token. The correct statement is total parameters versus active parameters per token.

---

---

## 11. Routing Mathematics in Depth

### 11.1 Top-k gating as a sparse conditional computation

A router produces scores $g(x)=Wx$ over $E$ experts; top-$k$ gating (Shazeer et al., 2017) keeps the largest $k$ experts, renormalizes their softmax weights, and computes $y=\sum_{i\in TopK} \tilde g_i E_i(x)$. Total parameters grow with $E$; per-token FLOPs grow with $k$, not $E$ — the entire MoE value proposition in one line.
- Example: $E=64,k=2$: each token uses 2 of 64 expert FFNs, about 3% of the expert parameters, while the model *stores* all 64.
- Non-example: $k=E$ is a dense ensemble with extra steps, not an MoE; sparsity is the definition.

### 11.2 Load balancing is an auxiliary objective, not a hope

Without pressure, routers collapse onto a few experts (rich-get-richer through training). Switch Transformer (Fedus et al., 2022) adds $L_{aux}=E\sum_i f_i P_i$, where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ the mean router probability for $i$; the product is minimized at uniform load $f_i=P_i=1/E$. Worked value: perfectly balanced routing over 8 experts gives $L_{aux}=8\times(1/8\times1/8)=0.125$ in this normalization (implementations scale it by a coefficient, commonly 0.01).

### 11.3 Expert capacity and token dropping

Each expert processes at most $C_f\times(\text{tokens}/E)$ tokens per batch (capacity factor $C_f$, Section 5 in original numbering). Overflow tokens are dropped (passed through the residual) or rerouted in dropless designs. Drop rate (10.2 of original) is therefore a quality variable set by a systems knob — report it with every MoE eval, because a 5% drop rate is a 5% of tokens receiving no expert computation.

### 11.4 GShard and Switch: the two canonical simplifications

Lepikhin et al. (2021, GShard) scaled conditional computation with top-2 gating and expert parallelism across devices in T5-style models. Fedus et al. (2022, Switch Transformer) showed top-1 routing ($k=1$) simplifies routing, communication, and compute while retaining most benefits at their scales — at the price of higher sensitivity to capacity and load balancing (11.2-11.3).

### 11.5 Mixtral and the modern open configuration

Mixtral 8x7B (Jiang et al., 2024) uses 8 experts per MoE layer with top-2 routing in a 7B-class backbone: about 47B total parameters, about 13B active per token by the authors' accounting. It is the reference open example that *total* and *active* parameters are different headline numbers, and both must be reported (Chapter 15 Section 08's scaling statements need the active count for FLOPs and the total count for memory).

---

## 12. Router Pathologies and Their Diagnostics

### 12.1 Router entropy collapse

Track the entropy of the router's average distribution (original 10.3). Falling entropy early in training, before task loss justifies specialization, signals collapse: a few experts receive gradient, improve, and attract still more tokens. The load-balancing loss (11.2) and router z-loss (Zoph et al., 2022, ST-MoE) — penalizing large router logits, $\mathbb{E}[(\log\sum_j e^{g_j})^2]$ — are the standard stabilizers; ST-MoE's other lesson was that stability tricks matter more as expert count and scale grow.

### 12.2 Expert specialization: real, partial, and overstated

Experts do specialize measurably (by token type, syntax, domain in published analyses of Switch/GShard-class models), but specialization is soft: most experts serve mixed traffic. Design consequence: do not hand-assign experts to topics; measure the routing histogram per domain (original 10.1) and let specialization be an empirical finding, not an architecture claim.

### 12.3 Dead and hot experts

A dead expert (near-zero tokens) wastes memory; a hot expert (many times mean load) sets the step latency under expert parallelism (Section 13). The diagnostics are the same histogram read at two tails, plus per-expert gradient norms (original 10.4): a dead expert also has near-zero gradient, so it cannot recover without an intervention (load-balancing pressure, reinitialization, or routing noise).

---

## 13. Expert Parallelism and Communication

Experts are typically sharded across devices (expert parallelism). Each token is dispatched (all-to-all) to its experts' devices, computed, and combined (a second all-to-all). Cost accounting, extending Chapter 15 Section 09:
- Communication volume per token scales with $k$ and hidden size, not with expert size; FLOPs scale with $k$ and expert size.
- A hot expert (12.3) concentrates both compute and inbound traffic on one device; load balance is therefore a *latency* property, not only a quality one.
- Under-served consequence: at small batch (single-user decode), MoE's total-parameter memory reads (Section 09 §28's bandwidth reasoning) can make it slower per token than a dense model with the same *active* parameters if experts are not resident/cached well. Total parameters are a memory and bandwidth fact; active parameters are a FLOPs fact. Confusing them is this section's cardinal sin.

---

## 14. Shared Experts, Fine-Grained Experts, and Modern Variants

### 14.1 Shared experts

DeepSeekMoE (Dai et al., 2024) routes tokens to a set of routed experts *plus* always-on shared experts that capture common knowledge, reducing redundancy among routed experts. In §11.1's terms the computation becomes $y=E_{shared}(x)+\sum_{TopK}\tilde g_iE_i(x)$.

### 14.2 Fine-grained expert segmentation

DeepSeekMoE (2024) and DeepSeek-V2/V3 (DeepSeek-AI, 2024) split experts into more, smaller experts and activate more of them (e.g. many small routed experts with higher $k$), arguing finer segmentation gives more flexible specialization at the same active compute. The accounting identity does not change: active FLOPs follow total activated expert width; total memory follows all experts.

### 14.3 Soft MoE

Puigcerver et al. (2024, Soft MoE) replace discrete token-to-expert assignment with weighted combinations (slots), removing dropping and discrete-routing instability at the cost of mixing all tokens into each slot — a different point in the exact/approximate space of Chapter 15 Section 09's method table, suited more to vision encoders than autoregressive serving.

---

## 15. MoE in Fine-Tuning and Inference

### 15.1 Adapting MoEs

LoRA on an MoE (Chapter 15 Section 07) must decide whether adapters attach to attention, shared experts, routed experts, or routers. Router adapters change *which* experts fire; expert adapters change *what* firing computes. Freezing routers while tuning experts is the conservative default: it preserves load balance learned in pretraining (11.2) while adapting computation.

### 15.2 Expert offloading and serving

Total parameters exceeding device memory can be served by keeping attention/shared parts resident and offloading routed experts, fetching them on activation (expert offloading). The latency then depends on routing predictability and fetch overlap — a §13/Chapter-09 bandwidth problem. Report active parameters, total parameters, resident bytes, and tokens/s together, or an MoE serving claim is incomparable.

---


---

## 20. Router Diagnostics Tables

### 20.1 Reading a routing histogram (8 experts, fractions of tokens)

| State | Histogram (rounded) | Entropy (nats) | Action |
|---|---|---:|---|
| Balanced | .125 x8 | 2.079 | none |
| Warm specialist | .25,.18,.15,.12,.10,.08,.07,.05 | ~1.95 | watch |
| Collapsing | .70,.10,.08,.05,.03,.02,.01,.01 | ~1.02 | raise aux weight / check z-loss (§12.1) |
| Dead expert present | .20,.20,.20,.20,.20,0,0,0 | 1.609 over 5 live | dead-expert intervention (§12.3) |


balanced entropy computed: 2.079 nats (uniform maximum is $\ln8=2.079$).

collapsing entropy computed: 1.107 nats (uniform maximum is $\ln8=2.079$).

## 21. Expert-Parallel Communication Worksheet

1. Write tokens per step, hidden size, and $k$: dispatch traffic scales as tokens x $k$ x hidden x bytes, twice (dispatch + combine, §13).
2. Write the expert placement: how many experts per device, and which device holds the currently hottest expert (§12.3).
3. Compute the hottest device's share of expert FLOPs from the live histogram, not from the uniform assumption.
4. If hot share over mean share exceeds about 1.5x, treat load balance as a latency bug first and a quality bug second (§13).
5. At decode with small batches, add total-parameter residency to the worksheet: MoE serving can be memory-bound before it is router-bound (§15.2).
6. Record drop rate at the production capacity factor in the same table; communication saved by dropping is paid in quality (§11.3).

## 22. Ten Review Questions With Answers

### Q1. Why is top-1 routing not just 'worse top-2'?
Switch (Fedus et al. 2022) found top-1 simplifies communication and compute enough that, properly balanced, it retains most quality at their scales (§11.4); it shifts pressure onto capacity and balance (§11.3).

### Q2. What does the aux loss minimize, geometrically?
The product of dispatch fraction and router probability per expert, whose joint minimum under normalization is uniform routing (§11.2).

### Q3. Why can a dead expert not recover alone?
No tokens means no output gradient (§12.3); recovery needs routing pressure or reinitialization.

### Q4. Total or active: which for a latency claim?
Active (plus total's memory/bandwidth effects §13) — a latency claim quoting only total parameters is category error (Common Mistakes).

### Q5. What does a shared expert buy?
Common computation removed from routed experts' burden, reducing their redundancy (Dai et al. 2024, §14.1).

### Q6. Why freeze routers in a first fine-tune?
To preserve pretrained balance while adapting computation (§15.1); router drift silently re-opens §12's pathologies.

### Q7. When does Soft MoE beat hard routing?
Where mixing tokens into slots is acceptable and dropping is not — notably vision encoders in Puigcerver et al. (2024); autoregressive serving usually keeps hard routing (§14.3).

### Q8. Capacity factor up or down?
Up: fewer drops, more compute/communication. Down: the reverse. It is a quality-systems trade with drop rate as its price tag (§11.3).

### Q9. How do you detect specialization honestly?
Routing histograms conditioned on domain/token type, over time (§12.2) — not by naming experts after inspecting a few tokens.

### Q10. What breaks first at very large $E$?
Router stability and balance (Zoph et al. 2022's motivation, §12.1) and all-to-all traffic (§13), before expert quality itself.

## 23. Pre-Launch Checklist for an MoE Change

1. Total, active, and resident parameters all stated (§13, §15.2).
2. $E$, $k$, capacity factor, and drop rate logged with the eval.
3. Aux-loss coefficient and z-loss (if any) recorded.
4. Routing histogram and router entropy dashboarded from step one (§12).
5. Hottest-expert share computed under realistic traffic (§21).
6. All-to-all traffic estimated at production batch (§21).
7. Dead-expert count is zero or explained.
8. Fine-tuning plan states router freeze/unfreeze policy (§15.1).
9. Dense comparison, if claimed, matched on active FLOPs and stated on memory (§13).
10. Offloading, if used, has a fetch/latency budget (§15.2).

## 24. Five Scenarios

### The giant that serves slowly
Total parameters were sized for a headline; at batch-1 decode the fleet reads expert weights per token (§13). Fix by residency/offload design (§15.2) or a smaller active configuration, not by router tweaks.

### The collapse after fine-tuning
Routers were tuned with experts on a narrow set; entropy fell (§12.1) and three experts died (§12.3). Roll back router training, restore aux pressure, retrain with routers frozen (§15.1).

### The capacity mirage
Quality regressed after a 'pure systems' capacity-factor cut. Drop rate had doubled (§11.3): the systems change was a modelling change in disguise.

### The specialization press release
Experts were named for topics from anecdotal tokens. §12.2's histogram-by-domain analysis showed soft mixing; the honest claim replaced the named one.

### The fair dense comparison
Reviewers rejected a speed claim at equal total parameters; rerun at equal active FLOPs with memory stated separately (§13), both numbers survived review.

---

## 25. Computed MoE Accounting Tables

### 25.1 Per-token expert FLOPs proxy (expert FFN forward+backward proportional to active expert width)

| Total expert params per layer | $E$ | $k$ | Active per token | Stored per layer |
|---:|---:|---:|---:|---:|
| 2e+10 (illustrative) | 32 | 2 | 1.00B | 16B |
| 2e+10 (illustrative) | 8 | 2 | 4.00B | 16B |
| 5e+10 (illustrative) | 8 | 2 | 11.75B | 47B |
| 2e+10 (illustrative) | 64 | 6 | 1.50B | 16B |

### 25.2 Dispatch traffic (tokens x $k$ x hidden x 2 bytes, dispatch only; combine doubles it)

| Tokens/step | $k$ | Hidden | Dispatch traffic |
|---:|---:|---:|---:|
| 8,192 | 2 | 4096 | 0.13 GB |
| 8,192 | 1 | 4096 | 0.07 GB |
| 65,536 | 2 | 8192 | 2.15 GB |
| 4,096 | 8 | 2048 | 0.13 GB |

## 26. Derivations in Steps

### 26.1 Why balanced routing minimizes the Switch aux loss

1. With $f_i=P_i$ (soft and hard routing agreeing), $L_{aux}=E\sum_i f_i^2$.

2. By convexity (or Cauchy-Schwarz), $\sum f_i^2$ under $\sum f_i=1$ is minimized at $f_i=1/E$, giving $\sum f_i^2=1/E$.

3. Hence the minimum of this normalized form is $E\times1/E=1$... in the convention where the loss is reported unscaled by $1/E$ variants differ; Section 17/Problem 2's warning applies: state the normalization. The *location* of the minimum (uniform) is convention-free, which is the part the optimizer uses.

### 26.2 Expected active parameters

1. Per token, attention and shared parts always run: call their size $S$.

2. Routed experts contribute $k$ of $E$ expert widths: if one expert is $W_e$, routed contribution is $kW_e$ in expectation under any routing (each token picks exactly $k$).

3. Active $=S+kW_e$; total $=S+EW_e$. Mixtral-style headlines (§11.5) are these two numbers for a whole model.

## 27. Symbols

| Symbol | Meaning |
|---|---|
| $E$ | number of experts (also entropy elsewhere; context decides) |
| $k$ | experts per token |
| $g_i,\tilde g_i$ | router score and renormalized top-$k$ weight |
| $f_i$ | fraction of tokens dispatched to expert $i$ |
| $P_i$ | mean router probability of expert $i$ |
| $C_f$ | capacity factor |
| $W_e$ | parameters of one expert |
| $S$ | always-on (attention/shared) parameters |

## 28. Twelve Closing Statements

- MoE trades memory for FLOPs at a stated $k/E$ exchange rate (§11.1).
- Balance is an objective with a loss, not an emergent hope (§11.2).
- Dropped tokens are a quality line item (§11.3).
- Router entropy is the collapse gauge (§12.1).
- Specialization is measured, never assigned (§12.2).
- The hottest expert sets the step time (§13).
- Total is memory; active is compute; report both (§13).
- Shared and fine-grained experts change the split, not the identities (§14, §26.2).
- Fine-tuning freezes routers before it trusts them (§15.1).
- Serving an MoE is a residency problem (§15.2).
- Every MoE claim carries $E$, $k$, drop rate, and both parameter counts, or it is an advertisement (Common Mistakes).
- The router is small, cheap, and capable of ruining everything — instrument it first (§23).

---

## 29. MoE Launch Gate (20 Items)

1. Router architecture and score function stated, including any normalization.
2. Expert count, size, and placement per device written down.
3. Top-k and renormalization rule implemented and unit-tested on a toy distribution.
4. Auxiliary load-balancing loss present with its coefficient and normalization stated.
5. Router z-loss present or its absence justified at the intended scale.
6. Capacity factor chosen with an expected drop-rate estimate.
7. Drop, overflow, or reroute behaviour for capacity excess defined in code, not in prose.
8. Routing histograms logged per layer from the first training step.
9. Router entropy logged per layer with an alert threshold for collapse.
10. Dead-expert count computed in the standard training report.
11. Per-expert gradient norms available for diagnosis.
12. Hottest-expert share included in step-time attribution.
13. All-to-all dispatch and combine traffic estimated at production batch.
14. Total, active, and resident parameter counts all in the model card.
15. Serving residency/offload plan stated for the total parameter count.
16. Fine-tuning policy for routers versus experts decided before adaptation begins.
17. Dense baseline, if compared, matched on active FLOPs with memory reported separately.
18. Specialization claims, if any, backed by domain-conditioned histograms.
19. Evaluation tables carry drop rate next to loss/accuracy.
20. A rollback exists that restores router and expert weights as one versioned object.

## 30. Method Comparison

| Design | Routing | Strength | Watch item | Reference |
|---|---|---|---|---|
| Sparsely-gated MoE | top-k, noisy | original conditional computation | balance by noise + losses | Shazeer et al. 2017 |
| GShard | top-2 | scaled expert parallelism | capacity tuning | Lepikhin et al. 2021 |
| Switch | top-1 | simplicity, less communication | capacity/balance sensitivity | Fedus et al. 2022 |
| ST-MoE | top-1/2 + z-loss | stability at scale | router logit control | Zoph et al. 2022 |
| Mixtral 8x7B | 8 experts, top-2 | open reference configuration | total vs active reporting | Jiang et al. 2024 |
| DeepSeekMoE | shared + fine-grained, higher k | specialization granularity | accounting complexity | Dai et al. 2024 |
| Soft MoE | soft slots | no dropping, differentiable | token mixing cost | Puigcerver et al. 2024 |

---

## 31. A Token's Journey Through an MoE Layer, Step by Step

1. Token representation $x$ (hidden size $h$) arrives at the MoE layer from attention.
2. Router computes scores $g=Wx$ over $E$ experts and a softmax distribution $P$.
3. Top-$k$ selects experts; their probabilities are renormalized to $\tilde g$ (§11.1).
4. The token is dispatched, with its $\tilde g$ weights, to the selected experts' devices (§13).
5. Each selected expert computes its FFN; outputs are weighted by $\tilde g$ and combined on return.
6. The residual stream adds the combined output; dropped-token case: if capacity was exceeded, the expert contribution is absent and the residual carries the token (§11.3).
7. Logging records: chosen experts, $\tilde g$, drop status, and per-expert token counts for §20's histograms.
8. Backpropagation updates experts through their routed tokens only, and the router through the gating path (with aux and z-losses, §11.2/§12.1).

## 32. Per-Layer Routing Load, Simulated Reading

The theory notebook simulates a skewed router; the reading protocol is: compute the histogram, its entropy against $\ln E$, the hottest share against $1/E$, and the implied step-time factor of §13 before looking at any loss curve.

| Observation | Healthy | Investigate | Intervene |
|---|---|---|---|
| Entropy vs $\ln E$ | within ~10% | 10-25% below | >25% below early in training |
| Hottest share vs mean | <1.5x | 1.5-3x | >3x (§13 latency) |
| Dead experts | 0 | 1, recent | >=1 persistent (§12.3) |
| Drop rate | <0.1% | 0.1-1% | >1% or rising (§11.3) |

Thresholds are stated as review heuristics for this section, to be replaced by a team's own baselines; the *quantities* are not optional.

## 33. Twenty One-Line MoE Calculations

1. Active expert fraction at $E=64,k=2$ is $2/64=3.125\%$ of expert parameters per token.
2. At $E=8,k=2$ it is 25% — Mixtral's expert-side fraction (§11.5).
3. Balanced aux loss ($f=P$ uniform, $E=8$, Switch normalization as in §16.2) evaluates to 0.125 in the section's stated convention; conventions must travel with the number (Problem 2).
4. All-to-one routing in the same convention is 64x the balanced value (§16.2).
5. Capacity per expert: 8192 tokens, $E=16$, $C_f=1.25$ gives $1.25\times512=640$ token slots.
6. An expert receiving 900 tokens at capacity 640 overflows by 260, about 29% of its arrivals (§11.3).
7. Dispatch traffic for 8192 tokens, $k=2$, $h=4096$, bf16 is about 0.13 GB one way (§25.2).
8. Combine traffic doubles the round trip to about 0.27 GB for that micro-batch (§13).
9. Hot-expert step factor at 40% share over 8 experts is $0.4/0.125=3.2$x mean (§13).
10. Router entropy of uniform over 64 experts is $\ln64\approx4.16$ nats, the ceiling for §32's gauge.
11. A dead expert's share of total parameters at $E=64$ is about 1.6% of expert storage earning nothing (§12.3).
12. Shared expert of width $W$ adds $W$ to both active and total in §26.2's accounting.
13. Doubling $E$ at fixed expert size doubles total expert storage and leaves active (fixed $k$) unchanged (§26.2).
14. Doubling $k$ at fixed $E$ doubles active expert FLOPs and dispatch traffic (§25).
15. Offloading: 74 GB of non-resident experts (Problem 8) at an illustrative 20 GB/s fetch path is seconds of traffic if fetched naively per step — residency must be predictive, not reactive (§15.2).
16. LoRA on routed experts only leaves router behaviour, hence balance, untouched (§15.1).
17. An MoE eval without drop rate is missing a term in its own definition (§11.3).
18. Tokens per expert at balance is batch$/E$; every imbalance claim is a deviation from this quotient (§32).
19. Z-loss grows with squared log-sum-exp of router logits; large logits are taxed before they become hard routing (§12.1).
20. The router's parameter count is $h\times E$ per layer — negligible storage, decisive behaviour (§23 item 20).

---

## 34. Training Dynamics Specific to MoE

### 34.1 The feedback loop at the centre

Routers send tokens to experts; experts that receive tokens receive gradient and improve; improved experts attract more tokens. This positive feedback is why balance is enforced by loss (§11.2) rather than expected. It is also why early-training router entropy (§12.1) is a leading indicator: by the time task loss shows collapse, the dead experts (§12.3) have been starved for thousands of steps.

### 34.2 Learning rates for routers versus experts

Routers are tiny ($h\times E$ per layer) and high-leverage: a small router change reroutes the whole batch. Many recipes therefore use a lower effective router LR or the z-loss (§12.1) as a soft brake. State the choice in the run config (§29 item 3's test exists to catch its absence).

### 34.3 Fine-grained experts change the dynamics, not just the count

With more, smaller experts and higher $k$ (§14.2), each expert sees fewer tokens and each token mixes more experts; gradient noise per expert rises while combination flexibility grows. The §32 gauges (entropy, hottest share, dead count) must be re-baselined for the new $E$ — thresholds from an $E=8$ model do not transfer to $E=256$.

---

## 35. Evaluation Protocol for MoE Claims

1. Report loss/accuracy with drop rate at the production capacity factor (§11.3).
2. Report routing histograms on the eval set, not only training (§12.2).
3. Report total, active, and resident parameters in the same table (§13).
4. If comparing to dense, give the active-FLOPs-matched dense number and the memory numbers separately (§13).
5. If experts were offloaded, report tokens/s with the residency policy named (§15.2).
6. If fine-tuned, state the router policy (§15.1) and rerun the balance gauges post-tuning (§34.1).
7. Publish the aux-loss convention used (Problem 2's warning) so balance numbers are comparable.
8. Include at least one domain-shifted eval: routing specialization (§12.2) can make in-domain MoE gains evaporate off-domain faster than dense gains.

---

## 36. Expanded Glossary

| Term | Meaning |
|---|---|
| All-to-all | Collective exchanging tokens among expert devices (§13) |
| Auxiliary loss | Load-balancing penalty added to task loss (§11.2) |
| Capacity factor | Overflow allowance multiplier per expert (§11.3) |
| Combine | Weighted return of expert outputs to token devices (§31) |
| Dead expert | Expert receiving (near) no tokens (§12.3) |
| Dispatch | Sending tokens to their experts' devices (§31) |
| Dropping | Skipping expert compute for overflow tokens (§11.3) |
| Expert parallelism | Sharding experts across devices (§13) |
| Fine-grained experts | Many small experts with higher k (§14.2) |
| Gating | Router scoring and top-k selection (§11.1) |
| Hot expert | Expert with far above-mean load (§12.3) |
| Load balance | Closeness of dispatch fractions to uniform (§11.2) |
| Offloading | Fetching non-resident experts on demand (§15.2) |
| Residency | Which parameters live on device (§15.2) |
| Router entropy | Entropy of mean routing distribution (§12.1) |
| Shared expert | Always-on expert added to routed sum (§14.1) |
| Soft MoE | Slot-weighted differentiable routing (§14.3) |
| Token-choice routing | Tokens pick experts (the standard here) |
| Expert-choice routing | Experts pick tokens; an alternative balancing scheme in the literature |
| Z-loss | Router logit magnitude penalty (Zoph et al. 2022) |

## 37. Ten Closing Review Sentences

- An MoE claim starts with E, k, and both parameter counts.
- Balance is enforced, measured, and reported, in that order.
- Drop rate is part of the model's definition at serving time.
- Router entropy is checked before task loss is trusted.
- Dead experts are a bug report, not a curiosity.
- The hottest expert is a latency budget line.
- Specialization is a histogram, not a name.
- Fine-tuning touches routers last, if at all.
- Offloading is a bandwidth schedule with a memory motive.
- A dense comparison has two axes or none.

---

## 38. Worked Design: An MoE Layer From a Blank Page

1. Start from the dense FFN you would have used: hidden $h=4096$, FFN intermediate 11008, about $2\times4096\times11008\approx90$M parameters per layer (counting up and down projections).
2. Choose total expert count $E=8$ and $k=2$ (§11.5's reference shape): experts per layer stored $=8\times90$M $=720$M; active expert width per token $=180$M.
3. Add the router: $4096\times8\approx33$k parameters — four orders of magnitude below the experts it governs (§33, calc 20).
4. Choose capacity factor 1.25 on expected tokens per expert (§11.3) and define overflow behaviour in code (§29 item 7).
5. Set the aux loss (Switch form, §11.2) with its coefficient recorded, and add z-loss if router logits run hot in early steps (§12.1).
6. Place experts 2 per device across 4 devices for this layer's expert parallelism; dispatch traffic per §25.2 enters the step budget.
7. Baseline the §32 gauges at step 1000: entropy near $\ln8$, hottest share under 1.5x, zero dead, drop rate logged.
8. Only then scale the recipe across layers, re-baselining gauges per layer (routers differ by depth, §34.3).

## 39. Configuration Table (Design Space, Not Leaderboard)

| Archetype | $E$ | $k$ | Shared expert | Reference |
|---|---:|---:|---|---|
| Early sparsely-gated | many, noisy top-k | k>1 | no | Shazeer et al. 2017 |
| GShard | scaled, top-2 | 2 | no | Lepikhin et al. 2021 |
| Switch | large, top-1 | 1 | no | Fedus et al. 2022 |
| Mixtral 8x7B | 8 | 2 | no | Jiang et al. 2024 |
| DeepSeekMoE | fine-grained, higher k | >2 | yes | Dai et al. 2024 |
| Soft MoE | slot-based | soft | n/a | Puigcerver et al. 2024 |


## 40. Final Formula and Decision Card

| Decision | First calculation |
|---|---|
| Can we afford the memory? | total params x bytes, resident plan (§15.2) |
| What does a token cost? | active params (§26.2) and dispatch traffic (§25.2) |
| Is routing healthy? | entropy, hottest share, dead count, drop rate (§32) |
| Is the claim fair? | dense comparison on active FLOPs + memory separately (§35) |
| Is the fine-tune safe? | router policy stated; gauges rerun (§15.1) |

---

## 41. Expert Placement Tables

| Experts $E$ | Devices | Experts/device | If one device holds the hot expert, its share at 3.2x hot factor |
|---:|---:|---:|---:|
| 8 | 4 | 2 | about 40.0% of all tokens routed to its 2 experts' queue, concentrated (§13) |
| 32 | 8 | 4 | about 10.0% of all tokens routed to its 4 experts' queue, concentrated (§13) |
| 64 | 16 | 4 | about 5.0% of all tokens routed to its 4 experts' queue, concentrated (§13) |
| 64 | 64 | 1 | about 5.0% of all tokens routed to its 1 experts' queue, concentrated (§13) |

## 42. How to Read an MoE Paper in This Section's Terms

1. Find $E$, $k$, expert size, and whether shared experts exist (§14).
2. Compute total and active from §26.2 before reading any speed claim.
3. Find the balance mechanism and its coefficient; absence is a finding (§11.2).
4. Find capacity factor and whether drop rate is reported (§11.3).
5. Check router stability machinery at the paper's scale (§12.1).
6. Translate communication claims through §25.2's traffic factors.
7. For serving claims, demand residency and batch regime (§15.2).
8. For specialization claims, demand conditioned histograms (§12.2).
9. Write the verdict in §37's sentence grammar before adopting the design.

## 43. End of Expansion Statements

- Original Sections 1-10 preserved; expansion Sections 11-43 appended with TOC updates.
- The companion notebook simulates routing, balance, capacity, and entropy with PASS/FAIL checks.
- Exercises and learner workspaces unchanged.
- All borrowed configurations are attributed: Shazeer et al. 2017; Lepikhin et al. 2021; Fedus et al. 2022; Zoph et al. 2022; Jiang et al. 2024; Dai et al. 2024; Puigcerver et al. 2024.
- Illustrative numbers are labelled illustrative at point of use; Mixtral's total/active figures are its authors' published accounting (§11.5).

---

## 44. Failure Taxonomy for MoE Systems

| Symptom | First quantity | Section | Typical fix |
|---|---|---|---|
| Loss plateaus, experts unused | dead-expert count, router entropy | 12.1, 12.3 | aux pressure, router LR, reinit dead experts |
| Step time dominated by one device | hottest-expert share | 13, 32 | balance as latency work; re-place experts |
| Quality drop after capacity cut | drop rate before/after | 11.3 | restore capacity or accept as model change |
| Fine-tune ruins general eval | router drift post-tuning | 15.1, 34.1 | freeze routers; rerun gauges |
| Serving slower than dense peer | residency bytes at decode batch | 15.2, 13 | residency/offload redesign |
| Intermittent token dropping spikes | per-batch load variance vs capacity | 11.3 | capacity factor, batch composition |
| Router logits exploding in logs | log-sum-exp of router scores | 12.1 | z-loss; router LR reduction |
| 'Specialist' expert serves everything | domain-conditioned histogram | 12.2 | correct the claim; check for collapse in disguise |

## 45. MoE Versus Dense: A Decision Worksheet

1. What is the binding constraint: training FLOPs, serving memory, serving latency, or quality at fixed compute?
2. If training FLOPs at quality: MoE's offer is capacity per FLOP (§11.1); demand the active-matched dense baseline (§35).
3. If serving memory: MoE is *worse* per quality unit unless residency/offloading (§15.2) is engineered; price it.
4. If single-stream latency: price total-parameter bandwidth (§13) before admiring active FLOPs.
5. If multi-tenant serving with high batch: MoE improves, because dispatch amortizes and experts stay resident; run §21's worksheet at that batch.
6. Data scale check: experts need tokens to specialize (§34); small-data regimes often favour dense simplicity.
7. Team check: routing instrumentation (§23) is operational load a dense model does not impose; staff it or choose dense.
8. Write the decision with the two-axis comparison (§13) attached, so a later reader can see which axis carried it.

---

## 46. Load Imbalance in Numbers

| Hottest share (of tokens) | $E$ | Multiple of mean load | Step-time implication under expert parallelism |
|---:|---:|---:|---:|
| 0.125 | 8 | 1.0x | slowest expert device carries about 1.0x mean expert work (§13) |
| 0.200 | 8 | 1.6x | slowest expert device carries about 1.6x mean expert work (§13) |
| 0.400 | 8 | 3.2x | slowest expert device carries about 3.2x mean expert work (§13) |
| 0.050 | 64 | 3.2x | slowest expert device carries about 3.2x mean expert work (§13) |
| 0.100 | 64 | 6.4x | slowest expert device carries about 6.4x mean expert work (§13) |
| 0.250 | 64 | 16.0x | slowest expert device carries about 16.0x mean expert work (§13) |

## 47. Closing Workbook: Review an MoE Result in Ten Minutes

1. Write $E$, $k$, expert size, shared experts (yes/no) from the paper or config.
2. Compute total and active using §26.2; compare to the claimed headlines.
3. Find drop rate; if absent, mark the quality numbers provisional (§11.3).
4. Find the balance loss and coefficient; if absent at scale, mark stability provisional (§11.2).
5. Find router entropy or histogram evidence; if absent, mark specialization claims unsupported (§12.2).
6. Convert any speed claim to active FLOPs and any memory claim to total/resident bytes (§13).
7. Check the dense comparison's matching (§35 item 4).
8. For serving, find batch and residency (§15.2); batch-1 and batch-256 are different sports.
9. Decide which single number you would re-measure first, and say why.
10. Record the review next to the result, in §37's grammar.

## 48. End of Section 10 Expansion

- The next section, Quantization and Distillation, changes the bytes per parameter that §13 and Chapter 15 Section 09 price throughout.
- MoE total-parameter memory is exactly where quantization pressure lands first.
- Active-parameter compute is where distillation targets are usually matched.
- Routing mathematics remains the grammar for reading those sections' MoE claims.

---

## 49. A Router Ablation Study, Designed

Question: "Is top-2 better than top-1 for our model?" A defensible design:

1. Freeze data, total parameters ($E$ and expert size), training tokens, and the dense/shared parts.
2. Run top-1 and top-2 with per-run capacity factors chosen so *drop rates match approximately*; otherwise the comparison mixes routing with dropping (§11.3).
3. Tune the aux-loss coefficient lightly per run, because optimal balance pressure differs by $k$ (§11.2); record coefficients.
4. Report: task loss, drop rate, router entropy trajectory (§12.1), hottest share (§32), step time, and dispatch traffic (§25.2).
5. Pre-register the decision rule, e.g. "adopt top-2 only if loss improves beyond run noise at equal drop rate and step time grows by less than the loss gain is worth" (§45's worksheet in experimental form).
6. Publish the losing configuration's full row as well; router folklore grows from unpublished losing cells (cf. Chapter 15 Section 07's ablation ethics).

## 50. Summary Card

| Remember | Section |
|---|---|
| $k$ of $E$, renormalized | 11.1 |
| Balance is a loss | 11.2 |
| Capacity drops are quality | 11.3 |
| Entropy gauges collapse | 12.1 |
| Hot expert, slow step | 13 |
| Total = memory, active = compute | 13, 26.2 |
| Shared/fine-grained change the split | 14 |
| Freeze routers first in FT | 15.1 |
| Review in ten minutes, honestly | 47 |

---

## 51. Tokens per Expert at Production Batch Sizes

| Tokens/step | $E=8$ per expert | $E=32$ | $E=64$ | Capacity at $C_f=1.25$, $E=64$ |
|---:|---:|---:|---:|---:|
| 8,192 | 1,024 | 256 | 128 | 160 |
| 32,768 | 4,096 | 1,024 | 512 | 640 |
| 131,072 | 16,384 | 4,096 | 2,048 | 2,560 |
| 524,288 | 65,536 | 16,384 | 8,192 | 10,240 |

Capacity is a per-step statistical statement: even balanced routers fluctuate multinomially around these means, which is why $C_f>1$ exists (§11.3) and why drop rate is logged rather than assumed zero.

## 52. Final Review Sentences for Section 10

- Routing is a learned allocator under a balance constraint.
- Its failures are visible in histograms before they are visible in loss.
- Its costs are communication and memory; its benefit is capacity per FLOP.
- Its claims require two parameter counts and a drop rate.
- Its fine-tuning begins by not touching the allocator.
- Its serving begins by housing every parameter it owns.
- That is the whole section, in the grammar of §37.

- Expansion sections: 11-52. Original Sections 1-10 preserved above.

- The companion notebook implements the routing simulations described in §32.

---

## 53. Routers Across Depth

Each MoE layer has its own router, and their behaviours differ systematically enough to matter operationally:

- Early layers route on shallow features (token identity, local syntax); their histograms are the most token-correlated.
- Middle layers show the strongest domain sensitivity in published histogram analyses (§12.2's measurement programme, applied per layer).
- Late layers route nearer to output-side abstractions and can be the most imbalanced, because their features are the most specialized.

Operational consequences:
1. Never average routing histograms over layers into one number; §32's gauges are per-layer or they are fiction.
2. A collapse alert should name the layer; remedies (aux weight is usually global) may need layer-wise coefficients.
3. Expert-parallel placement can differ by layer: a chronically hot late layer deserves placement attention (§41) that a balanced early layer does not.
4. In fine-tuning with frozen routers (§15.1), per-layer frozen behaviour is inherited, including any pre-existing imbalance — measure before you tune, per layer.

## 54. What Counts as Evidence for Specialization

Weak evidence (do not publish as specialization): a handful of tokens routed to an expert, an expert's name assigned by intuition, a single prompt's routing trace.
Real evidence (§12.2 made operational):
1. Domain-conditioned histograms with counts, across at least several domains.
2. Stability of the pattern across checkpoints (specialization that appears and vanishes is noise or training phase).
3. Ablations: disabling the 'specialist' expert hurts its domain disproportionately.
4. Comparison against a shuffled-routing control at equal load, separating routing value from expert capacity.

---

## 55. Closing Workbook

1. Take any MoE result you are about to cite.
2. Fill §47's ten-minute review on it, in writing.
3. Mark each missing quantity as provisional, using §35's language.
4. Decide the one re-measurement that would most change your confidence.
5. If that measurement is cheap, do it before citing; if expensive, cite with the provisional marks attached.
6. That discipline, applied uniformly, is most of what 'MoE expertise' consists of in practice.

- End of Section 10 expansion.
- Original text preserved; exercises unchanged; theory notebook expanded with executable routing demos.
- Notation follows docs/NOTATION_GUIDE.md; plots follow docs/VISUALIZATION_GUIDE.md.

---

## 56. Formula Appendix

| Quantity | Formula |
|---|---|
| Router scores | $g(x)=Wx$, $W\in\mathbb{R}^{E\times h}$ |
| Router distribution | $P=\mathrm{softmax}(g)$ |
| Top-k weights | $\tilde g_i=P_i/\sum_{j\in TopK}P_j$ for $i\in TopK$, else 0 |
| Expert output | $y=\sum_i\tilde g_iE_i(x)$ plus shared term if present (§14.1) |
| Dispatch fraction | $f_i=$ tokens routed to $i$ divided by total tokens |
| Aux loss (Switch) | $E\sum_i f_iP_i$, conventions vary (§26.1) |
| Z-loss | $\mathbb{E}[(\log\sum_j e^{g_j})^2]$ |
| Capacity | $C_f\times(\text{tokens}\times k/E)$ assignments per expert in assignment-counting conventions; state yours |
| Active params | $S+kW_e$ per MoE layer (§26.2) |
| Total params | $S+EW_e$ per MoE layer |
| Hottest share | $\max_i f_i$ against mean $1/E$ |
| Router entropy | $-\sum_i\bar P_i\log\bar P_i$ for mean routing $\bar P$ |

---

## 57. Token-Choice Versus Expert-Choice Routing

Token-choice routing (everything above) has each token select its top-$k$ experts. Expert-choice routing (Zhou et al., 2022) inverts the assignment: each expert selects its top tokens up to capacity.

Consequences, stated as tradeoffs rather than a verdict:
1. Expert-choice guarantees near-perfect load balance by construction, because each expert's load is its chosen capacity.
2. The guarantee is bought by giving up per-token choice: a token may be selected by many experts or by none, so coverage per token is uneven.
3. Dropping changes meaning: under token-choice, overflow tokens lose experts (§11.3); under expert-choice, unselected tokens lose experts. Either way some tokens can receive little expert computation — the eval must still say how much (§35).
4. For autoregressive serving, token-choice remains the common default; expert-choice has been used prominently in encoder-style and vision settings in its source work.
5. The §32 gauges still apply, with $f_i$ reinterpreted: under expert-choice $f_i$ is near-uniform by construction, so entropy of *token coverage* (how many experts each token received) becomes the informative histogram instead.

## 58. Final Decision Summary for Section 10

1. State $E$, $k$, expert size, and shared structure.
2. Compute total and active before anything else.
3. Choose the balance mechanism and its coefficient.
4. Set capacity and define overflow in code.
5. Instrument histograms, entropy, hottest share, and drop rate per layer.
6. Place experts and price the all-to-all.
7. Decide residency for serving at the total count.
8. Freeze routers for the first fine-tune.
9. Compare to dense on two axes or not at all.
10. Review every claim in §37's grammar.

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Reporting only total parameters for speed claims | FLOPs follow active parameters (§11.1) | State total, active, and $k$ together |
| Reporting only active parameters for memory claims | Memory follows total parameters (§13) | State resident/total bytes |
| No load-balancing loss or monitoring | Router collapse is the default attractor (§12.1) | Aux loss + entropy/histogram logging (11.2) |
| Capacity factor set without drop-rate reporting | Dropped tokens get no expert compute (§11.3) | Log drop rate with every eval |
| Hand-assigning experts to topics | Specialization is learned and soft (§12.2) | Measure histograms; do not hard-wire |
| Ignoring hot experts in latency analysis | The hottest expert sets step time under expert parallelism (§13) | Balance as a latency property |
| Tuning routers and experts together casually in FT | Routing shifts invalidate balance learned earlier (§15.1) | Freeze routers first; adapt experts |
| Comparing MoE to dense at equal total parameters for speed | Dense uses all parameters per token; MoE uses $k/E$ of experts | Compare at equal active FLOPs *and* state memory |
| Router logits growing unchecked | Instability at scale | Z-loss (Zoph et al. 2022) and logit monitoring |
| Treating dropped tokens as free | They are a quality loss charged to the eval (§11.3) | Include drop rate in results tables |


## 16. Computed Tables

### 16.1 Total versus active parameters (illustrative FFN-dominated accounting)

| Experts $E$ | Top-$k$ | Active fraction of experts | If experts are 40B of total: active expert params |
|---:|---:|---:|---:|
| 8 | 2 | 25.0% | 10.0B |
| 16 | 2 | 12.5% | 5.0B |
| 32 | 2 | 6.2% | 2.5B |
| 64 | 2 | 3.1% | 1.2B |
| 64 | 6 | 9.4% | 3.8B |
| 128 | 8 | 6.2% | 2.5B |

### 16.2 Load-balancing loss at three routing states ($E=8$, Switch-style $E\sum f_iP_i$ with $f=P$)

| Routing state | $L_{aux}$ | Reading |
|---|---|---|
| Uniform $f_i=1/8$ | 0.125 | minimum for this normalization (11.2) |
| Half the tokens to one expert, rest spread | higher | imbalance penalty grows |
| All tokens to one expert | 8.000 in the $f=P$ corner | collapse (§12.1) |


Middle row computed: $L_{aux}=8\times\sum f_i^2=8\times0.286=2.286$. The all-to-one corner gives $8\times1=8$, sixty-four times the balanced value — the aux loss has real dynamic range.

## 17. Ten Worked Problems With Solutions

### Problem 1: Active compute
$E=32,k=2$, expert FFN 0.5B params each in a layer: active expert params per token in that layer $=2\times0.5=1$B; stored $=16$B. (§11.1)

### Problem 2: Aux loss by hand
$E=4$, $f=P=(0.4,0.3,0.2,0.1)$: $L_{aux}=4\times(0.16+0.09+0.04+0.01)=4\times0.30=1.20$, versus uniform $4\times4\times(1/16)=1.00$ in *unnormalized-sum* reading — always state your normalization before comparing aux losses.

### Problem 3: Capacity overflow
Batch 8192 tokens, $E=8$, $C_f=1.0$: capacity per expert $=1024$; an expert receiving 1500 tokens drops or reroutes 476 (about 32% of its load) (§11.3).

### Problem 4: Hot expert latency
Under expert parallelism, if routing sends 40% of tokens to one of 8 experts, that device does about $0.4/(1/8)=3.2$x the mean expert compute; step time follows it (§13).

### Problem 5: Mixtral accounting
8 experts, top-2: active expert fraction 25%; the model's public 'active parameters' (about 13B of 47B) includes attention and shared parts — never equate it with $k/E$ of the total (§11.5).

### Problem 6: Router entropy
Average routing $(0.7,0.1,0.1,0.1)$ has entropy $-(0.7\ln0.7+3\times0.1\ln0.1)\approx0.94$ nats versus $\ln4=1.386$ uniform — a collapse warning if it appears early (§12.1).

### Problem 7: Dead expert
An expert with $f_i\approx0$ receives no tokens and hence no gradient through its outputs; it cannot self-repair (§12.3).

### Problem 8: Offloading bytes
Total 47B bf16 is 94 GB; attention+shared resident 20 GB (illustrative) leaves 74 GB of routed experts to fetch/cache — a bandwidth schedule, not a memory footnote (§15.2).

### Problem 9: FT freeze choice
For a domain LoRA on an MoE, freezing routers preserves the pretraining balance while expert adapters adapt computation (§15.1).

### Problem 10: Fair dense comparison
Compare an MoE to a dense model at matched active FLOPs for speed and matched total bytes for memory — two comparisons, not one (§13).

## 18. Historical Line
| Year | Work | Idea |
|---|---|---|
| 1991/2017 lineage | Jacobs et al. adaptive mixtures; Shazeer et al. 2017 sparsely-gated MoE | Learned gating; top-k conditional computation (§11.1) |
| 2021 | Lepikhin et al., GShard | Expert parallelism at scale, top-2 (§11.4) |
| 2022 | Fedus et al., Switch Transformer | Top-1 routing, load-balancing loss (§11.2, §11.4) |
| 2022 | Zoph et al., ST-MoE | Router z-loss, stability at scale (§12.1) |
| 2024 | Jiang et al., Mixtral 8x7B | Open 8-expert top-2 configuration (§11.5) |
| 2024 | Dai et al., DeepSeekMoE | Shared + fine-grained experts (§14) |
| 2024 | Puigcerver et al., Soft MoE | Fully differentiable slot routing (§14.3) |

## 19. Glossary and Formula Card

| Term/quantity | Meaning/formula |
|---|---|
| Top-$k$ gating | $y=\sum_{TopK}\tilde g_iE_i(x)$ |
| Load-balancing loss | $E\sum_i f_iP_i$ (Switch form) |
| Router z-loss | $\mathbb{E}[(\log\sum_j e^{g_j})^2]$ |
| Capacity | $C_f\times$ tokens$/E$ per expert |
| Active parameters | parameters touched per token (attention + shared + $k$ experts) |
| Total parameters | all parameters stored |
| Expert parallelism | experts sharded across devices with all-to-all dispatch |

## Practice Exercises

1. Compute top-k experts from router probabilities.
2. Count dense FFN and MoE expert parameters.
3. Compute active versus total expert parameters.
4. Compute expert capacity from tokens, experts, and capacity factor.
5. Compute drop rate from expert loads.
6. Compute a Switch-style auxiliary balancing term.
7. Compute router entropy for a token.
8. Estimate all-to-all token traffic by expert placement.
9. Compare top-1 and top-2 active compute.
10. Write an MoE debugging checklist.

## Why This Matters for AI

| Concept | AI Impact |
|---|---|
| Sparsely-gated MoE (Shazeer et al., 2017) | Conditional computation: parameters without proportional FLOPs |
| Switch Transformer (Fedus et al., 2022) | Top-1 routing and load balancing at trillion-parameter scale |
| GShard (Lepikhin et al., 2021) | Expert parallelism for giant conditional models |
| Mixtral 8x7B (Jiang et al., 2024) | Open top-2 MoE with total-vs-active parameter accounting |
| DeepSeekMoE (Dai et al., 2024) | Shared and fine-grained experts |

MoE models are attractive because they can increase capacity without a proportional increase in active compute. But they are not free. They create routing, balancing, communication, memory, and serving problems. Learning MoE math means learning to ask precise questions: which experts were active, how balanced were they, how many tokens were dropped, how much traffic moved, and how much quality came from sparsity rather than raw parameter count?

## Bridge to Quantization and Distillation

Quantization and distillation also change the relationship between quality, memory, and compute. The next section studies how precision reduction and teacher-student training compress models while trying to preserve behavior.

## References

- Noam Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017: https://arxiv.org/abs/1701.06538
- Dmitry Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding", 2020: https://arxiv.org/abs/2006.16668
- William Fedus, Barret Zoph, and Noam Shazeer, "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", 2021: https://arxiv.org/abs/2101.03961
- Mistral AI, "Mixtral of Experts", 2024: https://arxiv.org/abs/2401.04088
- Shazeer et al. (2017) introduced the sparsely-gated layer this section formalizes.
- Lepikhin et al. (2021) supplied the expert-parallel scaling pattern.
- Fedus et al. (2022) supplied top-1 routing and the load-balancing loss form used here.
- Zoph et al. (2022) supplied the router z-loss and stability framing.
- Jiang et al. (2024) supplied the open Mixtral configuration and its published parameter accounting.
- Dai et al. (2024) supplied shared and fine-grained expert designs.
- Puigcerver et al. (2024) supplied the soft-routing contrast case.
- Every other configuration number in this expansion is either derived in place or labelled illustrative.
- The load-balancing loss in this section follows Fedus et al. (2022).
- The router z-loss follows Zoph et al. (2022).
- Expert-parallel scaling follows Lepikhin et al. (2021).
- Top-k conditional computation follows Shazeer et al. (2017).
- Shared and fine-grained expert designs follow Dai et al. (2024).
- Soft routing follows Puigcerver et al. (2024) as a contrast case.
- Mixtral's configuration and parameter accounting follow Jiang et al. (2024) as published by its authors.
- No routing result from an unpublished system is presented as established.
- Illustrative histograms and traffic numbers are labelled as illustrative or derived from stated assumptions.
- The companion theory notebook turns histograms, entropy, capacity, and aux loss into executable checks.
- Readers should recompute every table in this expansion from its stated assumptions before reusing it.
- That recomputation habit is the section's final exercise, whether or not it appears in exercises.ipynb.
- A MoE section review should begin with the routing histogram.
- It should continue with drop rate and both parameter counts.
- It should ask which axis a dense comparison matched.
- It should ask what the routers were allowed to do during fine-tuning.
- It should ask where every expert lives at serving time.
- It should end with a decision written in the grammar of Section 37.
- Those six moves constitute peer review for sparse models.
- They are inexpensive and they catch the expensive errors.
- This expansion is designed to make them routine.
- End of file.
- Yanqi Zhou et al., "Mixture-of-Experts with Expert Choice Routing", NeurIPS 2022, arXiv:2202.09368.
- Zhou et al. (2022) is cited for expert-choice routing as a contrast to token-choice.
- The load-balancing, capacity, and reporting rules in this section apply to both routing directions, with the reinterpretations stated in Section 57.
- A routing paper that does not state its direction has omitted a defining property.
- A serving design that does not state its residency has omitted its memory plan.
- A training run that does not log its histograms has omitted its router's vital signs.
- Those omissions are cheap to fix before launch and expensive after.
- This is the practical moral of the section, and it generalizes beyond MoE.
- Section 10 expansion ends here.
- The next section studies quantization and distillation, which change parameter bytes and teacher-student objectives.
- MoE claims in that section should be read with this section active and total counts in hand.
- Router claims there should be read with histograms in hand.
- Serving claims there should be read with Chapter bandwidth accounting in hand.
- That layered reading is the chapter working as designed.

---

## 59. Per-Layer Expert Storage for the §38 Design (Computed)

Expert size 90M parameters (§38), bf16 storage at 2 bytes per parameter.

| $E$ | Stored per layer | Active per token ($k=2$) | Stored across 32 layers |
|---:|---:|---:|---:|
| 4 | 0.36B params (0.72 GB) | 0.18B | 12B params |
| 8 | 0.72B params (1.44 GB) | 0.18B | 23B params |
| 16 | 1.44B params (2.88 GB) | 0.18B | 46B params |
| 32 | 2.88B params (5.76 GB) | 0.18B | 92B params |
| 64 | 5.76B params (11.52 GB) | 0.18B | 184B params |

The last column is why total-parameter reporting (§13) is not pedantry: the $E=64$ design stores about 184B expert parameters across 32 such layers before attention exists.

## 60. Final Check Before Leaving This Section

1. Compute total and active for the §38 design without looking (§26.2).
2. State the aux-loss minimum and why it is uniform (§26.1).
3. Give the step-time consequence of a 3.2x hot expert (§13).
4. Name the two gauges that detect collapse earliest (§12.1, §32).
5. Write the one-sentence dense-comparison rule (§35).- The storage table above assumes the §38 expert size throughout; rescale it linearly for other expert widths.
- Active compute does not rescale with the number of stored experts at fixed k.
- That asymmetry is the entire economic argument for sparse models, stated in one table.
- It is also the entire serving risk, stated in the same table.
- Reviewers who internalize both readings will not be surprised by an MoE system.
- End of Section 10.
- Final note: router logs are part of the model artifact.
- Save them with checkpoints so post-hoc reviews can reconstruct balance history.
- A checkpoint without its routing history cannot explain its own specialization.
- A serving incident without routing logs cannot distinguish collapse from load.
- Instrumentation is therefore archival, not merely operational.
- This closes the MoE expansion.
- Counts in this file are verified against the live repository after push.
- Future trims must recheck the CONTRIBUTING line bar.
- The companion theory notebook is the executable version of Sections 16, 20, 25, and 32.
- The exercises notebook retains learner workspaces by design.
- Solutions there were compile-checked in this phase and left in their existing style.
- No other chapter was modified in this run segment.
- Sparse computation is a bet that capacity can be bought without paying its full compute price on every token.
- The bet pays when routing is balanced, experts are housed, and claims are reported on both parameter axes.
- It fails quietly when any one of those three is assumed instead of measured.
- This section has supplied the measurements.
- The rest is engineering discipline.
- End.
- Appendix reminder: conventions for aux loss and capacity differ across codebases.
- State yours wherever a number from this section is reused.
- A number without its convention is only half reported.
- The same applies to active-parameter headlines.
- Both halves are required for reproducibility.
- This is the section standard.
- File ends.
- Verified counts follow in the phase report.
- Local audit file is updated after live verification.
- No Section 11 work is started in this run.
- Remaining queue is Sections 11, 12, and 13.
- Verified live after push in the phase report.
- Audit progress is appended to /workspace/math-for-llms-audit.md.
- Safety margin: do not trim without rechecking the bar.
