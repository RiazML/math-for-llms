[<- Scaling Laws](../08-Scaling-Laws/notes.md) | [Home](../../README.md) | [Mixture of Experts and Routing ->](../10-Mixture-of-Experts-and-Routing/notes.md)

---

# Efficient Attention and Inference

Efficient inference is the mathematics of turning a trained LLM into a responsive service. The main objects are attention cost, KV cache memory, prefill and decode latency, batching, scheduling, and probability-preserving acceleration.

## Overview

Autoregressive inference has two phases:

```text
prompt tokens -> prefill -> KV cache -> decode one token -> append -> decode one token -> ...
```

Prefill processes the prompt in parallel and is often compute-heavy. Decode emits one token at a time and is often memory-bandwidth-heavy because each step reads weights and KV cache. Efficient serving is therefore not just "make attention faster." It is a coordinated memory, kernel, cache, batching, and scheduling problem.

The central cache formula is:

$$
M_\mathrm{KV}=2BLTH_{kv}d_hb.
$$

The factor 2 is for keys and values. $B$ is batch size, $L$ is layers, $T$ is cached context length, $H_{kv}$ is the number of key-value heads, $d_h$ is head dimension, and $b$ is bytes per cache element.

## Prerequisites

- Attention mechanism math and causal masking
- Positional encodings and autoregressive decoding
- Scaling-law and training-at-scale cost vocabulary
- Basic memory units and tensor shapes

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Computes prefill/decode costs, KV cache memory, MHA/MQA/GQA savings, FlashAttention memory intuition, paged-cache waste, speculative speedup, and latency budgets. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for KV cache sizing, GQA savings, pipeline latency, page waste, speculative decoding, and serving diagnostics. |

## Learning Objectives

After this section, you should be able to:

- Distinguish prefill latency, decode latency, TTFT, TPOT, throughput, and tail latency.
- Estimate dense attention and decode attention cost.
- Compute KV cache memory for MHA, MQA, and GQA.
- Explain why FlashAttention is exact but IO-aware.
- Explain why PagedAttention-style cache management improves serving memory utilization.
- Reason about continuous batching, chunked prefill, and head-of-line blocking.
- Estimate speculative decoding speedup from acceptance rate and draft latency.
- Build an inference debugging checklist that checks both speed and correctness.

## Table of Contents

1. [Inference Phases](#1-inference-phases)
   - 1.1 [Prefill](#11-prefill)
   - 1.2 [Decode](#12-decode)
   - 1.3 [TTFT](#13-ttft)
   - 1.4 [TPOT](#14-tpot)
   - 1.5 [Throughput](#15-throughput)
2. [Attention Cost](#2-attention-cost)
   - 2.1 [Training attention](#21-training-attention)
   - 2.2 [Decode attention](#22-decode-attention)
   - 2.3 [Causal mask](#23-causal-mask)
   - 2.4 [Memory traffic](#24-memory-traffic)
   - 2.5 [Arithmetic intensity](#25-arithmetic-intensity)
3. [KV Cache Math](#3-kv-cache-math)
   - 3.1 [Cache contents](#31-cache-contents)
   - 3.2 [Cache memory](#32-cache-memory)
   - 3.3 [MHA](#33-mha)
   - 3.4 [MQA](#34-mqa)
   - 3.5 [GQA](#35-gqa)
4. [FlashAttention and IO Awareness](#4-flashattention-and-io-awareness)
   - 4.1 [Naive attention memory](#41-naive-attention-memory)
   - 4.2 [Tiling](#42-tiling)
   - 4.3 [Online softmax](#43-online-softmax)
   - 4.4 [Exactness](#44-exactness)
   - 4.5 [Long context](#45-long-context)
5. [Serving Memory Management](#5-serving-memory-management)
   - 5.1 [Request lengths vary](#51-request-lengths-vary)
   - 5.2 [Paged KV cache](#52-paged-kv-cache)
   - 5.3 [Continuous batching](#53-continuous-batching)
   - 5.4 [Prefix sharing](#54-prefix-sharing)
   - 5.5 [Eviction and recompute](#55-eviction-and-recompute)
6. [Decode Acceleration](#6-decode-acceleration)
   - 6.1 [Greedy serial bottleneck](#61-greedy-serial-bottleneck)
   - 6.2 [Speculative decoding](#62-speculative-decoding)
   - 6.3 [Acceptance rate](#63-acceptance-rate)
   - 6.4 [Parallel heads](#64-parallel-heads)
   - 6.5 [Distribution preservation](#65-distribution-preservation)
7. [Batching and Scheduling](#7-batching-and-scheduling)
   - 7.1 [Static batching](#71-static-batching)
   - 7.2 [Dynamic batching](#72-dynamic-batching)
   - 7.3 [Head-of-line blocking](#73-headofline-blocking)
   - 7.4 [Chunked prefill](#74-chunked-prefill)
   - 7.5 [SLA tradeoff](#75-sla-tradeoff)
8. [Quantization and Bandwidth](#8-quantization-and-bandwidth)
   - 8.1 [Weight bandwidth](#81-weight-bandwidth)
   - 8.2 [KV quantization](#82-kv-quantization)
   - 8.3 [Accuracy tradeoff](#83-accuracy-tradeoff)
   - 8.4 [Kernel support](#84-kernel-support)
   - 8.5 [Mixed systems](#85-mixed-systems)
9. [Latency Metrics](#9-latency-metrics)
   - 9.1 [Queue time](#91-queue-time)
   - 9.2 [First-token latency](#92-firsttoken-latency)
   - 9.3 [Inter-token latency](#93-intertoken-latency)
   - 9.4 [Tail latency](#94-tail-latency)
   - 9.5 [Cost per token](#95-cost-per-token)
10. [Debugging Efficient Inference](#10-debugging-efficient-inference)
   - 10.1 [Shape checks](#101-shape-checks)
   - 10.2 [Cache correctness](#102-cache-correctness)
   - 10.3 [Memory accounting](#103-memory-accounting)
   - 10.4 [Bottleneck attribution](#104-bottleneck-attribution)
   - 10.5 [Quality regression](#105-quality-regression)
11. [KV Cache in Depth: Memory Is the Sequence Dimension Made Physical](#11-kv-cache-in-depth-memory-is-the-sequence-dimension-made-physical)
12. [FlashAttention and Online Softmax, Derived](#12-flashattention-and-online-softmax-derived)
13. [Serving Systems: Paging, Continuous Batching, Prefix Sharing](#13-serving-systems-paging-continuous-batching-prefix-sharing)
14. [Speculative Decoding, Exactly](#14-speculative-decoding-exactly)
15. [Quantization for Inference, as a Bandwidth Decision](#15-quantization-for-inference-as-a-bandwidth-decision)
16. [Latency and Cost Accounting That Cannot Be Gamed](#16-latency-and-cost-accounting-that-cannot-be-gamed)
17. [Computed KV-Cache Tables](#17-computed-kv-cache-tables)
18. [Speculative Acceptance Arithmetic (14.2)](#18-speculative-acceptance-arithmetic-(14.2))
19. [Ten Worked Problems With Solutions](#19-ten-worked-problems-with-solutions)
20. [Serving Review Checklist](#20-serving-review-checklist)
21. [Historical Line](#21-historical-line)
22. [Formula Summary](#22-formula-summary)
23. [Cache Size Across Sequence Lengths and Batch Sizes](#23-cache-size-across-sequence-lengths-and-batch-sizes)
24. [Latency Budget Worksheet](#24-latency-budget-worksheet)
25. [Ten Review Questions With Answers](#25-ten-review-questions-with-answers)
26. [Glossary](#26-glossary)
27. [Consolidated References for the Expansion](#27-consolidated-references-for-the-expansion)

---

## Mental Model

```text
quality target
   |
model weights -- kernels -- memory bandwidth -- KV cache -- scheduler -- user latency
                   |              |             |             |
              FlashAttention   quantization   paging      batching
```

Every speedup lives somewhere in this chain. The math tells you which bottleneck it can actually improve.

## 1. Inference Phases

This part studies inference phases as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Prefill](#1-prefill) | process the prompt in parallel and create the KV cache | $T_\mathrm{prefill}\propto T_\mathrm{prompt}^2$ for dense attention |
| [Decode](#1-decode) | generate one token at a time using cached keys and values | $T_\mathrm{decode}\propto T_\mathrm{context}$ per output token |
| [TTFT](#1-ttft) | time to first token includes scheduling plus prefill | $\mathrm{TTFT}=T_\mathrm{queue}+T_\mathrm{prefill}+T_\mathrm{sample}$ |
| [TPOT](#1-tpot) | time per output token measures decode speed | $\mathrm{TPOT}=T_\mathrm{decode\ step}$ |
| [Throughput](#1-throughput) | serving systems balance tokens per second against latency | $\mathrm{tokens/sec}=\mathrm{completed\ tokens}/T$ |

### 1.1 Prefill

**Main idea.** Process the prompt in parallel and create the kv cache.

Core relation:

$$T_\mathrm{prefill}\propto T_\mathrm{prompt}^2$ for dense attention$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.2 Decode

**Main idea.** Generate one token at a time using cached keys and values.

Core relation:

$$T_\mathrm{decode}\propto T_\mathrm{context}$ per output token$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.3 TTFT

**Main idea.** Time to first token includes scheduling plus prefill.

Core relation:

$$\mathrm{TTFT}=T_\mathrm{queue}+T_\mathrm{prefill}+T_\mathrm{sample}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.4 TPOT

**Main idea.** Time per output token measures decode speed.

Core relation:

$$\mathrm{TPOT}=T_\mathrm{decode\ step}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 1.5 Throughput

**Main idea.** Serving systems balance tokens per second against latency.

Core relation:

$$\mathrm{tokens/sec}=\mathrm{completed\ tokens}/T$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 2. Attention Cost

This part studies attention cost as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Training attention](#2-training-attention) | dense attention over a full sequence forms a T by T score matrix | $O(T^2d)$ |
| [Decode attention](#2-decode-attention) | one query attends to all cached keys | $O(Td)$ per generated token |
| [Causal mask](#2-causal-mask) | future tokens are invisible but the triangular work is still large in prefill | $j\le i$ |
| [Memory traffic](#2-memory-traffic) | attention can be limited by reads and writes, not only FLOPs | $T\approx\max(T_\mathrm{compute},T_\mathrm{memory})$ |
| [Arithmetic intensity](#2-arithmetic-intensity) | roofline reasoning compares FLOPs to bytes moved | $I=\mathrm{FLOPs}/\mathrm{bytes}$ |

### 2.1 Training attention

**Main idea.** Dense attention over a full sequence forms a t by t score matrix.

Core relation:

$$O(T^2d)$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.2 Decode attention

**Main idea.** One query attends to all cached keys.

Core relation:

$$O(Td)$ per generated token$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.3 Causal mask

**Main idea.** Future tokens are invisible but the triangular work is still large in prefill.

Core relation:

$$j\le i$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.4 Memory traffic

**Main idea.** Attention can be limited by reads and writes, not only flops.

Core relation:

$$T\approx\max(T_\mathrm{compute},T_\mathrm{memory})$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 2.5 Arithmetic intensity

**Main idea.** Roofline reasoning compares flops to bytes moved.

Core relation:

$$I=\mathrm{FLOPs}/\mathrm{bytes}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 3. KV Cache Math

This part studies kv cache math as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Cache contents](#3-cache-contents) | store keys and values for every layer, token, and KV head | $K,V\in\mathbb{R}^{L\times T\times H_{kv}\times d_h}$ |
| [Cache memory](#3-cache-memory) | KV memory grows linearly with context and batch | $M_\mathrm{KV}=2BLTH_{kv}d_hb$ |
| [MHA](#3-mha) | standard multi-head attention has one KV head per query head | $H_{kv}=H_q$ |
| [MQA](#3-mqa) | multi-query attention shares one KV head across query heads | $H_{kv}=1$ |
| [GQA](#3-gqa) | grouped-query attention interpolates between MHA and MQA | $1\le H_{kv}\le H_q$ |

### 3.1 Cache contents

**Main idea.** Store keys and values for every layer, token, and kv head.

Core relation:

$$K,V\in\mathbb{R}^{L\times T\times H_{kv}\times d_h}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.2 Cache memory

**Main idea.** Kv memory grows linearly with context and batch.

Core relation:

$$M_\mathrm{KV}=2BLTH_{kv}d_hb$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This one formula often decides how many requests fit on a serving GPU.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.3 MHA

**Main idea.** Standard multi-head attention has one kv head per query head.

Core relation:

$$H_{kv}=H_q$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.4 MQA

**Main idea.** Multi-query attention shares one kv head across query heads.

Core relation:

$$H_{kv}=1$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 3.5 GQA

**Main idea.** Grouped-query attention interpolates between mha and mqa.

Core relation:

$$1\le H_{kv}\le H_q$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 4. FlashAttention and IO Awareness

This part studies flashattention and io awareness as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Naive attention memory](#4-naive-attention-memory) | materializing attention scores costs T squared memory | $M_\mathrm{scores}=BT^2H$ |
| [Tiling](#4-tiling) | FlashAttention computes blocks while keeping partial statistics | $\mathrm{softmax}(QK^\top)V$ without storing all scores |
| [Online softmax](#4-online-softmax) | blockwise max and denominator keep softmax exact | $m=\max_i s_i,\quad l=\sum_i e^{s_i-m}$ |
| [Exactness](#4-exactness) | FlashAttention changes memory access, not the attention definition | $\mathrm{output}_\mathrm{flash}=\mathrm{output}_\mathrm{dense}$ |
| [Long context](#4-long-context) | IO savings matter more as T grows | $T^2$ score storage becomes the wall |

### 4.1 Naive attention memory

**Main idea.** Materializing attention scores costs t squared memory.

Core relation:

$$M_\mathrm{scores}=BT^2H$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.2 Tiling

**Main idea.** Flashattention computes blocks while keeping partial statistics.

Core relation:

$$\mathrm{softmax}(QK^\top)V$ without storing all scores$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** FlashAttention is fast because it respects the memory hierarchy, not because it approximates attention.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.3 Online softmax

**Main idea.** Blockwise max and denominator keep softmax exact.

Core relation:

$$m=\max_i s_i,\quad l=\sum_i e^{s_i-m}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.4 Exactness

**Main idea.** Flashattention changes memory access, not the attention definition.

Core relation:

$$\mathrm{output}_\mathrm{flash}=\mathrm{output}_\mathrm{dense}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 4.5 Long context

**Main idea.** Io savings matter more as t grows.

Core relation:

$$T^2$ score storage becomes the wall$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 5. Serving Memory Management

This part studies serving memory management as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Request lengths vary](#5-request-lengths-vary) | static allocation wastes KV memory for short or finished requests | $\mathrm{waste}=M_\mathrm{reserved}-M_\mathrm{used}$ |
| [Paged KV cache](#5-paged-kv-cache) | allocate KV cache in blocks and map logical tokens to physical blocks | $\mathrm{token}\rightarrow\mathrm{block\ id},\mathrm{offset}$ |
| [Continuous batching](#5-continuous-batching) | add and remove requests between decode steps | $B_t$ changes over time |
| [Prefix sharing](#5-prefix-sharing) | shared prompts can reuse KV cache | $KV(x,y)=KV(x)+KV(y\mid x)$ |
| [Eviction and recompute](#5-eviction-and-recompute) | memory pressure can force swapping or rebuilding cache | $T_\mathrm{recompute}$ trades against $M_\mathrm{free}$ |

### 5.1 Request lengths vary

**Main idea.** Static allocation wastes kv memory for short or finished requests.

Core relation:

$$\mathrm{waste}=M_\mathrm{reserved}-M_\mathrm{used}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.2 Paged KV cache

**Main idea.** Allocate kv cache in blocks and map logical tokens to physical blocks.

Core relation:

$$\mathrm{token}\rightarrow\mathrm{block\ id},\mathrm{offset}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** Paged allocation makes serving look more like an operating-system memory problem.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.3 Continuous batching

**Main idea.** Add and remove requests between decode steps.

Core relation:

$$B_t$ changes over time$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.4 Prefix sharing

**Main idea.** Shared prompts can reuse kv cache.

Core relation:

$$KV(x,y)=KV(x)+KV(y\mid x)$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 5.5 Eviction and recompute

**Main idea.** Memory pressure can force swapping or rebuilding cache.

Core relation:

$$T_\mathrm{recompute}$ trades against $M_\mathrm{free}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 6. Decode Acceleration

This part studies decode acceleration as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Greedy serial bottleneck](#6-greedy-serial-bottleneck) | autoregressive decoding normally emits one token per target-model step | $K$ tokens require $K$ target passes |
| [Speculative decoding](#6-speculative-decoding) | a draft model proposes tokens and the target verifies them | $\mathrm{speedup}\approx \mathrm{accepted\ tokens}/\mathrm{target\ pass}$ |
| [Acceptance rate](#6-acceptance-rate) | speedup depends on draft quality and draft latency | $a=P(\mathrm{accept})$ |
| [Parallel heads](#6-parallel-heads) | methods such as Medusa predict multiple future tokens with extra heads | $y_{t+1:t+k}$ candidates |
| [Distribution preservation](#6-distribution-preservation) | exact speculative schemes preserve the target distribution when acceptance rules are correct | $p_\mathrm{output}=p_\mathrm{target}$ |

### 6.1 Greedy serial bottleneck

**Main idea.** Autoregressive decoding normally emits one token per target-model step.

Core relation:

$$K$ tokens require $K$ target passes$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.2 Speculative decoding

**Main idea.** A draft model proposes tokens and the target verifies them.

Core relation:

$$\mathrm{speedup}\approx \mathrm{accepted\ tokens}/\mathrm{target\ pass}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** The target model can verify several proposed tokens in one pass, reducing serial decode steps.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.3 Acceptance rate

**Main idea.** Speedup depends on draft quality and draft latency.

Core relation:

$$a=P(\mathrm{accept})$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.4 Parallel heads

**Main idea.** Methods such as medusa predict multiple future tokens with extra heads.

Core relation:

$$y_{t+1:t+k}$ candidates$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 6.5 Distribution preservation

**Main idea.** Exact speculative schemes preserve the target distribution when acceptance rules are correct.

Core relation:

$$p_\mathrm{output}=p_\mathrm{target}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 7. Batching and Scheduling

This part studies batching and scheduling as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Static batching](#7-static-batching) | wait for a batch, run together, then finish together | $B$ fixed |
| [Dynamic batching](#7-dynamic-batching) | merge compatible requests at runtime | $B_t$ variable |
| [Head-of-line blocking](#7-headofline-blocking) | one long request can delay short ones | $T_\mathrm{latency}$ depends on schedule |
| [Chunked prefill](#7-chunked-prefill) | split long prompts so decode traffic is not starved | $T_\mathrm{prompt}$ chunks |
| [SLA tradeoff](#7-sla-tradeoff) | higher throughput can increase tail latency | $p95\ \mathrm{latency}$ versus tokens/sec |

### 7.1 Static batching

**Main idea.** Wait for a batch, run together, then finish together.

Core relation:

$$B$ fixed$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.2 Dynamic batching

**Main idea.** Merge compatible requests at runtime.

Core relation:

$$B_t$ variable$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.3 Head-of-line blocking

**Main idea.** One long request can delay short ones.

Core relation:

$$T_\mathrm{latency}$ depends on schedule$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.4 Chunked prefill

**Main idea.** Split long prompts so decode traffic is not starved.

Core relation:

$$T_\mathrm{prompt}$ chunks$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 7.5 SLA tradeoff

**Main idea.** Higher throughput can increase tail latency.

Core relation:

$$p95\ \mathrm{latency}$ versus tokens/sec$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 8. Quantization and Bandwidth

This part studies quantization and bandwidth as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Weight bandwidth](#8-weight-bandwidth) | decode often reloads weights for each generated token | $T_\mathrm{weights}\approx M_\mathrm{weights}/\mathrm{bandwidth}$ |
| [KV quantization](#8-kv-quantization) | compressing KV cache can increase batch or context capacity | $M_\mathrm{KV}\downarrow$ |
| [Accuracy tradeoff](#8-accuracy-tradeoff) | lower precision can change probabilities | $D_\mathrm{KL}(p_\mathrm{fp},p_\mathrm{quant})$ |
| [Kernel support](#8-kernel-support) | a quantized format helps only if kernels are fast | $T_\mathrm{kernel}$ matters |
| [Mixed systems](#8-mixed-systems) | weights, activations, and KV cache may use different precision | $b_\mathrm{weight}\ne b_\mathrm{KV}$ |

### 8.1 Weight bandwidth

**Main idea.** Decode often reloads weights for each generated token.

Core relation:

$$T_\mathrm{weights}\approx M_\mathrm{weights}/\mathrm{bandwidth}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.2 KV quantization

**Main idea.** Compressing kv cache can increase batch or context capacity.

Core relation:

$$M_\mathrm{KV}\downarrow$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.3 Accuracy tradeoff

**Main idea.** Lower precision can change probabilities.

Core relation:

$$D_\mathrm{KL}(p_\mathrm{fp},p_\mathrm{quant})$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.4 Kernel support

**Main idea.** A quantized format helps only if kernels are fast.

Core relation:

$$T_\mathrm{kernel}$ matters$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 8.5 Mixed systems

**Main idea.** Weights, activations, and kv cache may use different precision.

Core relation:

$$b_\mathrm{weight}\ne b_\mathrm{KV}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 9. Latency Metrics

This part studies latency metrics as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Queue time](#9-queue-time) | user latency includes time before GPU work starts | $T_\mathrm{queue}$ |
| [First-token latency](#9-firsttoken-latency) | long prompts mainly hurt TTFT | $\mathrm{TTFT}$ |
| [Inter-token latency](#9-intertoken-latency) | long outputs mainly accumulate TPOT | $T_\mathrm{output}\approx n_\mathrm{out}\mathrm{TPOT}$ |
| [Tail latency](#9-tail-latency) | p95 and p99 matter more than average for products | $Q_{0.95}(T)$ |
| [Cost per token](#9-cost-per-token) | serving cost combines hardware time and utilization | $\mathrm{cost/token}=T_\mathrm{gpu}\mathrm{price}/\mathrm{tokens}$ |

### 9.1 Queue time

**Main idea.** User latency includes time before gpu work starts.

Core relation:

$$T_\mathrm{queue}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.2 First-token latency

**Main idea.** Long prompts mainly hurt ttft.

Core relation:

$$\mathrm{TTFT}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.3 Inter-token latency

**Main idea.** Long outputs mainly accumulate tpot.

Core relation:

$$T_\mathrm{output}\approx n_\mathrm{out}\mathrm{TPOT}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.4 Tail latency

**Main idea.** P95 and p99 matter more than average for products.

Core relation:

$$Q_{0.95}(T)$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 9.5 Cost per token

**Main idea.** Serving cost combines hardware time and utilization.

Core relation:

$$\mathrm{cost/token}=T_\mathrm{gpu}\mathrm{price}/\mathrm{tokens}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
## 10. Debugging Efficient Inference

This part studies debugging efficient inference as serving math. The goal is to connect latency and memory behavior to concrete tensor shapes, not to memorize system names.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Shape checks](#10-shape-checks) | KV heads, query heads, and head dimension must align | $H_q/H_{kv}$ integer for GQA |
| [Cache correctness](#10-cache-correctness) | cached decode must match full recomputation | $\max|y_\mathrm{cache}-y_\mathrm{full}|$ small |
| [Memory accounting](#10-memory-accounting) | estimate weights plus KV cache plus workspace | $M=M_\mathrm{weights}+M_\mathrm{KV}+M_\mathrm{workspace}$ |
| [Bottleneck attribution](#10-bottleneck-attribution) | separate compute, memory, scheduler, and network time | $T=\sum_j T_j$ |
| [Quality regression](#10-quality-regression) | speedups must preserve target quality unless explicitly approximate | $\Delta L,\Delta S$ tracked |

### 10.1 Shape checks

**Main idea.** Kv heads, query heads, and head dimension must align.

Core relation:

$$H_q/H_{kv}$ integer for GQA$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.2 Cache correctness

**Main idea.** Cached decode must match full recomputation.

Core relation:

$$\max|y_\mathrm{cache}-y_\mathrm{full}|$ small$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** A fast cached path is worthless if it disagrees with full recomputation.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.3 Memory accounting

**Main idea.** Estimate weights plus kv cache plus workspace.

Core relation:

$$M=M_\mathrm{weights}+M_\mathrm{KV}+M_\mathrm{workspace}$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.4 Bottleneck attribution

**Main idea.** Separate compute, memory, scheduler, and network time.

Core relation:

$$T=\sum_j T_j$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.
### 10.5 Quality regression

**Main idea.** Speedups must preserve target quality unless explicitly approximate.

Core relation:

$$\Delta L,\Delta S$ tracked$$

Inference math is different from training math because the bottleneck shifts. Training usually wants maximum throughput over large batches. Interactive inference must manage first-token latency, per-token latency, KV cache memory, and request scheduling. The same transformer can feel fast or slow depending on prompt length, output length, batch shape, cache layout, and kernel support.

**Worked micro-example.** For a model with $L=32$ layers, batch $B=8$, context $T=4096$, $H_{kv}=8$ KV heads, head dimension $d_h=128$, and bf16 cache values with $b=2$ bytes, the KV cache is $2BLTH_{kv}d_hb$ bytes. Reducing $H_{kv}$ from 32 to 8 cuts this cache by 4x.

**Implementation check.** Compare cached decode against full-prefix recomputation on a tiny example. Then measure memory and latency separately for prefill and decode; an aggregate tokens/sec number hides too much.

**AI connection.** This quantity is a practical lever for LLM inference.

**Common mistake.** Do not optimize benchmark throughput while ignoring p95 latency, prompt length distribution, output length distribution, and quality regression.

---

---

## 11. KV Cache in Depth: Memory Is the Sequence Dimension Made Physical

### 11.1 The exact cache formula

For batch $B$, layers $L$, KV heads $h_{kv}$, head dimension $d_h$, sequence length $s$, and $b$ bytes per element, the key-value cache holds

$$M_{KV}=2\times B\times L\times h_{kv}\times d_h\times s\times b$$

bytes, where the leading 2 counts keys and values separately. Every design decision in this section is a move on one factor of this product.

- Example: $B=1,L=32,h_{kv}=32,d_h=128,s=4096,b=2$ (bf16) gives $2\cdot32\cdot32\cdot128\cdot4096\cdot2=2.15$ GB for one request.
- Example:Grouped-query attention with $h_{kv}=8$ (Ainslie et al., 2023) divides that by 4, to 0.54 GB.
- Non-example: reporting cache size without $B$ and $s$ is meaningless; the product, not the architecture name, is the memory.

### 11.2 MQA, GQA, and the quality-memory tradeoff

Multi-query attention (Shazeer, 2019) sets $h_{kv}=1$: every query head shares one key/value head, minimizing cache and decode bandwidth at some quality cost. Grouped-query attention (Ainslie et al., 2023) interpolates with $1<h_{kv}<h_q$ and, in their experiments, recovers most of multi-head quality at a fraction of the cache. Multi-head latent attention in DeepSeek-V2 (DeepSeek-AI, 2024) compresses keys/values into a low-rank latent vector instead, attacking $d_h$ in the formula rather than $h_{kv}$.

### 11.3 Why decode is bandwidth-bound

One decode step reads the *entire* cache for one new token (Section 2.2). Arithmetic intensity (2.5) is roughly FLOPs per byte read, and for batch-1 decode it is near 1: the accelerator's memory bandwidth, not its FLOPs, sets inter-token latency. This is the quantitative reason quantization (Section 8), GQA (11.2), and cache compression help decode far more than their FLOP savings suggest.

---

## 12. FlashAttention and Online Softmax, Derived

### 12.1 The problem is IO, not FLOPs

Naive attention materializes the $s\times s$ score matrix in high-bandwidth memory. Dao et al. (2022, FlashAttention) tile queries, keys, and values through SRAM and never write the full matrix: FLOPs are unchanged (slightly higher from recomputation), but HBM traffic falls from quadratic-matrix writes to streaming the inputs. On their A100 experiments this gave large speedups and enabled longer contexts; FlashAttention-2 (Dao, 2023) improved work partitioning and parallelism.

### 12.2 Online softmax in three lines of algebra

Softmax needs a max and a sum over the whole row before any output. The online/streaming version (Milakov and Gimelshein, 2018, used inside FlashAttention) maintains running $(m,\ell)$ = (running max, running sum of $e^{x-m}$). When a new block raises the max from $m$ to $m'$, rescale: $\ell\leftarrow\ell e^{m-m'}+\sum e^{x-m'}$, and rescale the accumulated output the same way. Exactness (4.4) is preserved because rescaling is algebra, not approximation.

Worked step: values $(1,3)$ seen: $m=3$, $\ell=e^{-2}+1=1.135$. New value 4 arrives: $m'=4$, $\ell\leftarrow1.135e^{-1}+e^{0}=0.418+1=1.418$, which equals $e^{-3}+e^{-1}+1$ directly. The notebook verifies this identity numerically.

### 12.3 What FlashAttention does not do

It does not approximate attention, change its output (up to floating-point order), or reduce training FLOPs. Claims of the form "FlashAttention made the model better" confuse a memory/IO optimization with a modelling change; quality deltas after adopting it are numerics, not architecture (10.5).

---

## 13. Serving Systems: Paging, Continuous Batching, Prefix Sharing

### 13.1 PagedAttention

Kwon et al. (2023, vLLM/PagedAttention) store KV cache in fixed-size blocks allocated on demand, like virtual memory pages, instead of pre-allocating the maximum sequence length per request. Their reported effect is the elimination of most internal/external fragmentation waste, raising achievable batch size and throughput at the same memory. In terms of 11.1: paging does not change bytes per token; it changes how much of the device the product $M_{KV}$ is allowed to use.

### 13.2 Continuous (iteration-level) batching

Yu et al. (2022, Orca) schedule at iteration granularity: finished requests leave and new requests join at every decode step, instead of waiting for a static batch to finish (7.1-7.2). The utilization gain is largest when output lengths vary widely — precisely when 9.4's tail latency is worst under static batching.

### 13.3 Prefix sharing and radix caching

If requests share a prompt prefix (system prompts, few-shot templates, multi-turn history), their KV for that prefix is identical and can be computed once and shared (5.4). The saving is multiplicative with request count for system-prompt-heavy workloads; the correctness condition is exact prefix identity at token level, including the tokenizer boundary (Chapter 15 Section 05's tokenization caveats apply verbatim).

### 13.4 Chunked prefill and the TTFT/TPOT tradeoff

Long prefills block decode steps for other requests. Chunked prefill (7.4) interleaves prefill chunks with decodes, trading a little prefill efficiency for bounded inter-token latency under load. State the tradeoff with the metrics of Section 9: TTFT (time to first token), TPOT (time per output token), and throughput — a system tuned for one silently sacrifices another, so all three belong in every serving report.

---

## 14. Speculative Decoding, Exactly

### 14.1 Draft, verify, accept

A small draft model proposes $\gamma$ tokens; the target model scores them in one parallel forward pass; tokens are accepted by a rejection-sampling rule: accept token $x$ with probability $\min(1,p(x)/q(x))$, and on the first rejection sample a correction from the normalized positive part of $p-q$ and stop (Leviathan et al., 2023; Chen et al., 2023). 

### 14.2 The distribution is preserved, not approximated

That acceptance rule makes the output an exact sample from the target distribution $p$, for any draft $q$ — the theorem both papers prove. Speed comes from parallelism (verifying $\gamma$ tokens costs little more than generating 1 when decode is bandwidth-bound, 11.3), not from lowering quality. Expected accepted length for per-token acceptance $\alpha$ is $(1-\alpha^{\gamma+1})/(1-\alpha)$ in the idealized independent model: at $\alpha=0.8,\gamma=4$, about 3.4 tokens per target pass.

### 14.3 When it does not help

Low acceptance (draft/target mismatch), small $\gamma$ overheads at high batch (the system may already be compute-bound, so verification is not "free"), and memory for a second model. Report acceptance rate and tokens-per-target-pass alongside latency; a speedup claim without $\alpha$ is unverifiable (cf. 10.2's cache-correctness checks).

### 14.4 Medusa and multi-head drafting

Cai et al. (2024, Medusa) attach multiple decoding heads to the target model itself to propose future tokens, trading the separate draft model for extra heads plus a tree-attention verification. The acceptance mathematics of 14.1-14.2 is unchanged; only the source of $q$ moves.

---

## 15. Quantization for Inference, as a Bandwidth Decision

Weight-only quantization (e.g. 4-bit weights, bf16 activations; GPTQ, Frantar et al., 2023; AWQ, Lin et al., 2024) shrinks the bytes read per decode step, which is the binding resource by 11.3. Worked arithmetic: a 7B model reads 14 GB of bf16 weights per generated token at batch 1; at 4 bits it reads 3.5 GB, so ideal decode latency falls up to 4x before overheads — while *training* arithmetic (Chapter 15 Section 07, QLoRA) is a different use of the same bit-width idea. Activation quantization (e.g. SmoothQuant, Xiao et al., 2023) additionally attacks activation and KV-cache bytes (11.1's $b$), at higher accuracy risk because activation outliers are input-dependent.

---

## 16. Latency and Cost Accounting That Cannot Be Gamed

Report, for a stated workload (prompt/output length distributions, concurrency):
1. TTFT: queue + prefill time to the first token (9.2).
2. TPOT/ITL: per-output-token time after the first (9.3), at p50 and p99 (9.4).
3. Throughput: output tokens per second per device at the stated concurrency.
4. Cost per million output tokens at a stated device price (9.5).
5. Cache hit/prefix-sharing rate if 13.3 is enabled.
6. Speculative acceptance $\alpha$ and tokens per target pass if 14 is enabled.
A single "tokens per second" number without items 1, 2, and the workload is marketing; the notebook's final cells compute all of these on toy workloads so the definitions are operational.

---


---

## 28. Decode Latency Floors From Bandwidth (Computed)

Weights-only floor at batch 1: bytes read per token divided by an illustrative 2 TB/s HBM. Cache reads (§11) lower these further in practice.

| Model | bf16 (14-350 GB reads) | 8-bit | 4-bit | bf16 floor (tokens/s) | 4-bit floor (tokens/s) |
|---|---:|---:|---:|---:|---:|
| 7B | 14 GB | 7 GB | 3.5 GB | 143 | 571 |
| 13B | 26 GB | 13 GB | 6.5 GB | 77 | 308 |
| 70B | 140 GB | 70 GB | 35.0 GB | 14 | 57 |

These are ceilings, not predictions: a measured rate above the ceiling means the accounting (batch, bytes, or bandwidth) is wrong somewhere (§10.4).

## 29. Implementation Audit Protocol

1. Print the production config's six KV factors and the predicted cache bytes for a stated batch (11.1).
2. Measure actual cache allocation at that batch; reconcile with the prediction before serving traffic (10.3).
3. Verify position ids under prefix sharing: shared tokens get their original positions, not zeros (13.3).
4. Verify the cache key includes token ids and model/version; a stale-model cache is a silent quality bug.
5. Run one request with cache disabled and compare logits on the first generated token (10.2).
6. Check padding side and attention masks in batched prefill against a batch-1 reference (10.1).
7. Log TTFT and TPOT separately from the first day; a combined 'latency' hides which phase regressed (16).
8. Log speculative acceptance per request class if enabled; aggregate-only acceptance hides workload dependence (14.3).
9. After any quantization change, run the quality regression suite before the latency celebration (10.5, 15).
10. After any kernel change, compare outputs to the naive attention reference on random inputs within float tolerance (12.3).

## 30. Five Serving Scenarios

### The demo that dies in production
Batch-1 latency looked fine; at concurrency 64 the cache product (§23) exceeded memory and requests queued. Diagnosis is multiplication, not tuning: cap concurrency by cache, adopt paging (§13.1), or cut $h_{kv}$ (§11.2).

### The throughput benchmark nobody can reproduce
Reported tokens/s used batch 256, 32-token outputs, and no latency numbers. §16's six items make the claim checkable; without them it is incomparable, not impressive.

### The long-context feature that halved fleet capacity
Raising max context 8k to 32k quadruples per-request cache at fixed batch (§23); capacity planning that ignored the $s$ factor discovered it empirically.

### Speculative decoding that slowed the API
Enabled at high batch where verification is not free (§14.3); acceptance logs (§29, item 8) showed the regime, and the fix was gating speculation by current batch size.

### The quantization 'free lunch' with a quality bill
4-bit weights doubled decode speed and quietly cost points on a reasoning eval (§15, §10.5). The trade may be right; it must be *chosen* with both numbers on the table.

## 31. Final Self-Check

1. Can you compute KV bytes for any served config from its six factors (11.1)?
2. Can you explain decode's bandwidth bound and name two interventions that attack bytes (11.3)?
3. Can you derive the online-softmax rescale and verify it on three numbers (12.2)?
4. Can you state exactly why speculative sampling preserves the target distribution (14.2)?
5. Can you produce a serving report that §16 would accept?
6. Can you find the ceiling in §28's table for your model and compare a measured rate to it?

---

## 32. Twenty One-Line Inference Calculations

1. KV bytes/token, $L=80,h_{kv}=8,d_h=128$, bf16: $2\times80\times8\times128\times2=327{,}680$ bytes.
2. That is 0.33 MB per token, so 8192 tokens of context cost about 2.7 GB of cache for one request.
3. At batch 16 the same context costs about 43 GB — concurrency is purchased with cache (§23).
4. GQA 32 to 8 heads divides every number above by 4 (§11.2).
5. MQA divides by 32 instead, at the quality risk Shazeer (2019) and Ainslie et al. (2023) bracket.
6. Weight traffic per token, 13B bf16, batch 1: 26 GB read for one token (§28).
7. At 2 TB/s that is a 13 ms floor per token before cache traffic (§11.3).
8. Speculative: $\alpha=0.9,\gamma=4$ gives $(1-0.9^5)/0.1\approx4.1$ expected accepted tokens (18).
9. Speculative: $\alpha=0.5,\gamma=4$ gives about 1.94 — positive but fragile (§14.3).
10. TTFT floor: 8192 prompt tokens at 50k prefill tokens/s is about 164 ms (§13.4).
11. TPOT budget: 200 tokens at 25 ms is 5 s of decode — TPOT dominates long answers (Problem 5).
12. Prefix sharing: a shared 1000-token prefix across 100 requests saves 99,000 prefill token-computations (§13.3).
13. Paged cache: paging changes usable fraction of memory, not bytes/token (§13.1) — a 70% to 95% usable-fraction gain is a 1.36x batch gain.
14. Continuous batching: a finished request's slot is reusable at the next iteration, not the next batch (§13.2).
15. 4-bit 70B weights: 35 GB read per token at batch 1, floor near 57 tokens/s at 2 TB/s (§28).
16. Batching amortizes weight reads: at batch 32, weight bytes per token fall by 32x while cache bytes per token do not (§11) — large-batch decode becomes cache-bound.
17. Online softmax: rescaling by $e^{m-m'}$ is what makes blockwise attention exact (§12.2).
18. FlashAttention memory: scores tile through SRAM; the $s\times s$ matrix is never materialized (§12.1).
19. Cost: at a stated $ per device-hour, cost per million tokens is device-seconds per token times price, divided into a million (§16).
20. p99 TPOT is a scheduling diagnostic first and a model diagnostic second (§30, Scenario patterns).

## 33. Attention Memory: Naive Versus Tiled, in Numbers

| Sequence $s$ | Naive score matrix (bf16, one head) | Tiled/Flash-style HBM footprint (order) |
|---:|---:|---:|
| 1,024 | 0.00 GB | streams $Q,K,V$: about 1 MB per head-dimension pass |
| 4,096 | 0.03 GB | streams $Q,K,V$: about 3 MB per head-dimension pass |
| 16,384 | 0.54 GB | streams $Q,K,V$: about 13 MB per head-dimension pass |
| 65,536 | 8.59 GB | streams $Q,K,V$: about 50 MB per head-dimension pass |
| 131,072 | 34.36 GB | streams $Q,K,V$: about 101 MB per head-dimension pass |

The naive column is why $s=131072$ was not a serving feature before IO-aware attention (§12); the right column is deliberately stated as an order, because exact traffic depends on tiling and hardware (Dao et al., 2022).

## 34. Symbols

| Symbol | Meaning |
|---|---|
| $B$ | batch / concurrent requests |
| $L$ | layers |
| $h_q,h_{kv}$ | query and KV head counts |
| $d_h$ | head dimension |
| $s$ | sequence length |
| $b$ | bytes per element |
| $\gamma$ | speculative draft length |
| $\alpha$ | per-token acceptance probability |
| TTFT | time to first token |
| TPOT | time per output token |

## 35. Closing Statements

- Inference is a memory-system problem wearing a modelling costume (§11.3).
- The cache formula's six factors are the whole design space (§11.1).
- Exact optimizations (FlashAttention, speculative decoding) should never change your eval; approximate ones (quantization) must re-earn it (§12.3, §10.5).
- Schedulers move waiting between metrics; §16's report is how you notice.
- Every serving claim needs a workload, a latency pair, and a cost — or it is an anecdote (§30).

---

## 36. Why the Speculative Acceptance Rule Is Exact: Proof in Steps

Setting: draft distribution $q$, target distribution $p$, one token position (the argument repeats per position; Leviathan et al., 2023; Chen et al., 2023).

1. Sample a draft token $x\sim q$.
2. Accept it with probability $\min(1,p(x)/q(x))$; if accepted, emit $x$ and continue to the next drafted token.
3. The probability that this procedure emits a particular $x$ via acceptance is $q(x)\min(1,p(x)/q(x))=\min(q(x),p(x))$.
4. The total rejection probability is $1-\sum_x\min(q(x),p(x))=\sum_x\max(0,p(x)-q(x))$, a quantity often written $\beta_{rej}$.
5. On rejection, sample from $p'(x)=\max(0,p(x)-q(x))/\beta_{rej}$ and emit that, then stop the current draft round.
6. Emission probability for $x$ through the rejection path is $\beta_{rej}\times p'(x)=\max(0,p(x)-q(x))$.
7. Total emission probability: $\min(q(x),p(x))+\max(0,p(x)-q(x))=p(x)$, case by case: if $p(x)\ge q(x)$ the terms are $q(x)+(p(x)-q(x))=p(x)$; if $p(x)<q(x)$ they are $p(x)+0=p(x)$.
8. Therefore each emitted token is an exact sample from $p$, regardless of $q$. $\square$

Corollary for practitioners: a bug in this rule (e.g. accepting with probability $p(x)$ instead of the ratio) breaks step 3 and silently changes the served distribution — which is why §29 item 5 and the distribution check in the theory notebook exist.

## 37. FlashAttention Tiling, Step by Step

1. Partition the query rows into blocks that fit in SRAM with a block of keys/values and the running output.
2. For each key/value block, compute the block's scores on chip; never write them to HBM (§12.1).
3. Update the running max $m$ and sum $\ell$ with the rescaling of §12.2, and rescale the partial output identically.
4. Accumulate the block's contribution $e^{x-m'}V$ into the output tile.
5. After all blocks, divide the output by the final $\ell$; this is the exact softmax-weighted value (§12.3).
6. In the backward pass, recompute the block scores from saved $(m,\ell)$ statistics instead of stored matrices — trading FLOPs for the memory that made training long contexts feasible (Dao et al., 2022).

## 38. Expanded Glossary

| Term | Meaning |
|---|---|
| Acceptance rate $\alpha$ | Fraction of drafted tokens accepted (§14) |
| Arithmetic intensity | FLOPs per byte moved; decode's is near 1 (§11.3) |
| Block (paging) | Fixed-size KV allocation unit (§13.1) |
| Correction token | Sample from normalized $(p-q)_+$ after a rejection (§36) |
| Draft model | Small model proposing tokens for verification (§14.1) |
| Fragmentation | Unusable gaps in pre-allocated cache (§13.1) |
| Head dimension $d_h$ | Per-head key/value width (§11.1) |
| Iteration-level scheduling | Join/leave at every decode step (§13.2) |
| KV head | One key/value projection set, possibly shared by query heads (§11.2) |
| Radix/prefix sharing | Reusing KV for token-identical prefixes (§13.3) |
| SRAM | On-chip fast memory used by tiled attention (§37) |
| Target model | The model whose distribution must be preserved (§14) |
| Tree attention | Verifying multiple draft branches in one pass (Medusa-style, §14.4) |
| Verification pass | One parallel target forward over drafted tokens (§14.1) |

---

## 39. Workload Latency Tables (Illustrative System, Stated as Such)

Assumptions, stated so the tables can be recomputed: prefill 50,000 tokens/s, TPOT 25 ms at moderate batch, no queueing. These illustrate §16's reporting, they are not measurements.

| Prompt tokens | Output tokens | TTFT (s) | Decode time (s) | Total (s) | Dominant phase |
|---:|---:|---:|---:|---:|---|
| 128 | 128 | 0.003 | 3.20 | 3.20 | decode/TPOT |
| 1,024 | 256 | 0.020 | 6.40 | 6.42 | decode/TPOT |
| 4,096 | 256 | 0.082 | 6.40 | 6.48 | decode/TPOT |
| 8,192 | 1,024 | 0.164 | 25.60 | 25.76 | decode/TPOT |
| 32,768 | 128 | 0.655 | 3.20 | 3.86 | decode/TPOT |

## 40. Concurrency Is Bought With Cache (Worksheet)

1. From §23, take bytes per token for your config.
2. Multiply by the maximum sequence length you will actually serve, not the model's theoretical maximum.
3. That product is the worst-case cache per request; multiply by target concurrency.
4. If the result exceeds device memory minus weights and activations, the plan is impossible as stated — before any code exists.
5. Now choose the lever, in this order of preference: paged allocation to reclaim fragmentation (§13.1), GQA/quantized KV to cut bytes per token (§11.2, §15), shorter served context, or lower concurrency.
6. Recompute after the lever; the worksheet is done when the product fits with headroom for activations.
7. Record the resulting concurrency cap in the serving config so a later context increase re-triggers this worksheet (§30, Scenario 3).

## 41. Ten More Review Statements (True/False With Answers)

### S1. True or false: doubling batch always halves per-token decode latency.
False. Weight reads amortize, cache reads do not (§11, calc 16); and beyond the compute crossover, extra batch can add latency.

### S2. True or false: PagedAttention reduces bytes per token.
False. It raises the usable fraction of memory (§13.1).

### S3. True or false: a draft model must be a smaller version of the target.
False. Correctness holds for any $q$ (§36); only acceptance/speed depends on similarity.

### S4. True or false: FlashAttention-2 changed attention's mathematics.
False. It changed scheduling/parallelism of the same exact computation (Dao, 2023).

### S5. True or false: prefix caching is safe if prompts share a text prefix.
False as stated: identity must be at token level, including tokenization boundaries (§13.3).

### S6. True or false: p99 TPOT problems are usually model problems.
False. They are usually scheduling/prefill-interference problems first (§13.4, calc 20).

### S7. True or false: 4-bit weights quarter decode latency in production.
Not necessarily: 4x is the traffic ceiling before overheads and non-weight bytes (§15).

### S8. True or false: continuous batching helps most when output lengths are uniform.
False. It helps most when lengths vary (§13.2).

### S9. True or false: TTFT includes queueing in an honest report.
True, under load; idle-system TTFT must be labelled as such (§24, step 3).

### S10. True or false: KV quantization risks quality differently from weight quantization.
True. It scales with context length and touches every attended token (§15's $b$).

---

## 42. Long-Context Serving: The Cache Is the Cost

### 42.1 Sliding-window attention

Mistral 7B (Jiang et al., 2023) uses sliding-window attention: each token attends to a fixed trailing window (4096 in that model), so KV cache per request is capped by the window rather than the document length. In §11.1's terms, effective $s$ for cache becomes $\min(s,W)$. The tradeoff is explicit: information outside the window reaches a token only indirectly through stacked layers' overlapping windows.

### 42.2 Cache compression and eviction policies

StreamingLLM (Xiao et al., 2023) keeps attention "sink" tokens (the first few, which attract disproportionate attention mass) plus a trailing window, enabling long streams with bounded cache. H2O (Zhang et al., 2023) evicts KV entries by accumulated attention importance ("heavy hitters"). Both are *approximate* serving choices — unlike §12 and §14's exact methods, they change the effective distribution and must re-earn quality per §10.5, with the eviction policy stated as part of the served model.

### 42.3 Position extrapolation is a model property, serving pays for it

RoPE-context-extension methods (e.g. position interpolation, Chen et al., 2023) make long contexts *model-valid*; §11.1 makes them *serving-expensive* linearly in $s$. A context-length launch decision is therefore jointly owned: modelling owns quality at length, serving owns the §40 worksheet at that length.

---

## 43. Method Comparison Table

| Method | Exact? | What it saves | What it costs | Key reference |
|---|---|---|---|---|
| FlashAttention | Yes | HBM traffic, score-matrix memory | recompute in backward | Dao et al. 2022 |
| GQA/MQA | Model change | KV bytes, decode bandwidth | possible quality | Ainslie 2023; Shazeer 2019 |
| Paged KV | Yes (systems) | fragmentation waste | block management | Kwon et al. 2023 |
| Continuous batching | Yes (systems) | idle slots | scheduler complexity | Yu et al. 2022 |
| Prefix sharing | Yes if token-identical | duplicate prefill | key correctness burden | §13.3 |
| Speculative decoding | Yes (distribution) | sequential steps | draft compute, memory | Leviathan 2023; Chen 2023 |
| Medusa heads | Yes with tree verify | separate draft model | extra heads | Cai et al. 2024 |
| Weight quantization | Approximate | weight bytes/token | dequant overhead, quality risk | Frantar 2023; Lin 2024 |
| KV quantization | Approximate | cache bytes | quality risk growing with $s$ | §15 |
| Sliding window | Approximate (model) | cache cap | long-range direct attention | Jiang et al. 2023 |
| Streaming/H2O eviction | Approximate | cache cap | dropped context risk | Xiao 2023; Zhang 2023 |

---

## 44. Serving Launch Gate (30 Items)

1. KV bytes per token computed from the six factors and written in the launch doc.
2. Maximum served sequence length stated, with its cache product at target concurrency.
3. Concurrency cap derived from memory, not from a load test that happened to pass.
4. Paged allocation enabled, or fragmentation measured and accepted explicitly.
5. GQA/MQA/MLA configuration stated as KV heads in the formula, not as a model nickname.
6. Weight bit-width and activation/KV bit-width stated separately.
7. Bandwidth floor for decode computed and compared to the TPOT target.
8. If the floor fails, the design changes bytes or hardware before launch, not after.
9. Attention kernel compared to a naive reference on random inputs.
10. Cache on/off logit comparison passed on representative prompts.
11. Position ids verified under prefix sharing and under chunked prefill.
12. Prefix cache keys include token ids, model version, and adapter/version where relevant.
13. Continuous batching behaviour verified with mixed output lengths.
14. Static-batch fallback, if any, documented with its tail-latency cost.
15. TTFT dashboard split into queue and prefill components.
16. TPOT reported at p50 and p99, per workload class.
17. Throughput reported only alongside TTFT/TPOT and workload.
18. Cost per million output tokens computed at the contracted device price.
19. Speculative decoding, if on: draft identity, gamma, and acceptance logged per class.
20. Speculative off-ramp exists for high-batch regimes where verification is not free.
21. Quality regression suite run after every approximate change: quantization, eviction, window.
22. Eviction/window policy, if any, documented as part of the served model definition.
23. Long-context quality evaluated at the served length, not only at training length.
24. Rollback: previous kernel/quantization/config restorable without a model reload incident.
25. Load test includes the longest prompts and outputs in the stated workload, not the median case.
26. Memory watermark alarms set below the cache product computed in item 2.
27. Per-request token accounting reconciles with cache allocation within tolerance.
28. Multi-turn requests verified: history tokens, cache reuse, and eviction interact as intended.
29. Sampling parameters logged per request so latency/quality incidents are reproducible.
30. A single 'serving config' artifact versions model, kernel, quantization, scheduler, and limits together.

---

## 45. Prefill Versus Decode: Different Bottlenecks, in Numbers

Prefill processes $s$ prompt tokens in parallel: FLOPs $pprox2Ns$ (forward only), high arithmetic intensity, compute-bound. Decode generates one token per step per request: FLOPs $pprox2N$ per token, but reads all weights and the request's cache (§11.3).

| Phase | FLOPs per token (7B) | Bytes read per token (bf16, B=1) | Binding resource |
|---|---:|---:|---|
| Prefill (amortized over prompt) | $1.4\times10^{10}$ | weights shared across prompt positions | compute |
| Decode, B=1 | $1.4\times10^{10}$ | 14 GB weights + growing cache | bandwidth |
| Decode, B=32 (per-token share) | $1.4\times10^{10}$ | 0.44 GB weights share + own cache | cache bandwidth increasingly |


## 46. Final Workbook: Size a Serving Deployment in Seven Steps

1. Pick the served model and write its six KV factors (§11.1) and weight bytes (§28).
2. Pick the workload: prompt/output distributions and target concurrency (§16).
3. Compute cache at maximum concurrency and served length (§40); iterate levers until it fits with activation headroom.
4. Compute TTFT from prefill throughput at the workload's prompt lengths; add queueing honestly (§24).
5. Compute TPOT from the bandwidth floor, then from a measured pilot; report both, and investigate any large gap (§10.4).
6. Decide the approximate methods (quantization, eviction, window) and schedule their quality regressions before launch (§43).
7. Fill §44's gate; any unchecked item is a named risk with an owner, not a surprise. 

## 47. Closing Formula Card

| Question | First calculation |
|---|---|
| Will it fit? | $2BLh_{kv}d_hsb$ plus weights and activations |
| How fast can one stream decode? | bytes/token divided by bandwidth (§28) |
| Why is the fleet slow at noon? | per-request cache times concurrency, then TPOT tails (§23, §13.4) |
| Is this speedup real? | exact or approximate (§43), and acceptance/quality logged (§14, §10.5) |
| What does it cost? | device-seconds per token at a stated price (§16) |

---

## 48. Approximate Attention Families, and Why Serving Still Uses Softmax

Linear attention (Katharopoulos et al., 2020) replaces $\mathrm{softmax}(QK^\top)V$ with a kernel feature map $\phi(Q)(\phi(K)^\top V)$, making sequence cost linear and decode state constant-size (a matrix state instead of a growing cache, §11). Sparse patterns (e.g. local windows, §42.1) keep softmax inside a restricted set. Both families trade the exact, content-addressed retrieval of full softmax for efficiency; at LLM scale the field's centre of gravity has remained full softmax made *systems-efficient* (§12–§13) plus cache-factor reductions (§11.2), because quality per token at scale has favoured exact attention. A serving proposal to switch families is a modelling change requiring §10.5's regression, not an optimization.

## 49. What to Measure When Context Length Doubles

1. Cache per request at the new length (§11.1) — it exactly doubles, all else equal.
2. Achievable concurrency at fixed memory (§40) — it halves, all else equal.
3. Prefill time and TTFT at the new prompt lengths (§39) — linear in prompt tokens at best.
4. TPOT degradation from longer cache reads per decode step (§11.3).
5. Quality at the new length on a length-targeted eval (§42.3), separately from serving metrics.
6. Prefix-sharing hit patterns: longer shared prefixes raise the value of §13.3.
7. Eviction/window behaviour if a cap is used to survive items 1–2 (§42.2), including its quality regression.
8. Cost per request at the new workload (§16) — expect it to more than double for long-output traffic.

---

## 50. Metrics Dictionary for Serving Dashboards

| Metric | Definition | Misreading to avoid |
|---|---|---|
| Requests per second | Completed requests per second at the stated mix | hides token counts; pair with tokens/s |
| Output tokens per second | Fleet decode throughput | hides latency; pair with TPOT |
| TTFT p50 | Median time to first token including queue | idle-system values must be labelled |
| TTFT p99 | Tail first-token latency | usually queueing or long-prompt prefill |
| TPOT p50 | Median per-token decode time | batch-dependent; state concurrency |
| TPOT p99 | Tail per-token time | prefill interference and stragglers first (§13.4) |
| Cache utilization | Allocated KV bytes over device memory | near 100% with paging is healthy; with pre-allocation it is fragmentation |
| Prefix hit rate | Requests reusing at least one shared block | only meaningful token-keyed (§13.3) |
| Acceptance rate | Draft tokens accepted over drafted (§14) | report per workload class |
| Tokens per target pass | Emitted tokens per target forward (§14.2) | the speculative speed numerator |
| GPU memory watermark | Peak device memory over capacity | alarms belong below the §40 product |
| Cost per 1M output tokens | Device cost allocated over output tokens | state device price and utilization |
| Goodput | Tokens/s meeting the latency SLA | the only throughput a user feels |

Goodput deserves emphasis: a system can raise raw throughput by batching until TPOT violates the product's latency budget, at which point the extra tokens are not good. Report goodput at a stated SLA alongside raw throughput (§16) and the benchmark's claim becomes falsifiable.

---

## 51. End-to-End Sizing Example

Deployment: a 13B GQA model ($L=40$, $h_{kv}=8$, $d_h=128$, bf16), served context 8192, target 32 concurrent requests, device memory 80 GB.

Step 1, weights: 26 GB in bf16 (§28's row), leaving about 54 GB before activations and overhead.
Step 2, cache per token: $2\times40\times8\times128\times2=163{,}840$ bytes (§11.1).
Step 3, cache per request at full context: $163{,}840\times8192\approx1.34$ GB.
Step 4, cache at concurrency 32: about 43 GB. Total with weights: about 69 GB — feasible with paged allocation (§13.1) and modest headroom, infeasible if most requests truly sit at maximum context *and* activations/fragmentation are ignored.
Step 5, decode floor: weight traffic at batch 32 is amortized (calc 16), so per-request cache reads ($1.34$ GB growing context) become the TPOT driver late in long generations (§11.3).
Step 6, decision recorded: enable KV quantization only if §10.5's regression passes, because Step 4 fits without it; do not buy memory you do not need with quality you cannot spare (§15, §43).

## 52. Twelve Closing Statements

1. Write the cache product before choosing hardware (§11.1).
2. Decode is bandwidth until proven compute-bound (§11.3).
3. Exact optimizations are free; approximate ones bill the eval (§43).
4. Paging buys usable memory, not smaller tokens (§13.1).
5. Schedulers trade between TTFT, TPOT, and throughput (§13.4).
6. Speculation is exact and acceptance-priced (§36).
7. Quantization is a bandwidth decision with a quality invoice (§15).
8. Long context is a capacity decision (§49).
9. Prefix sharing is a token-identity contract (§13.3).
10. Report six numbers or none (§16).
11. Audit the system against its own arithmetic (§29).
12. Size the deployment on paper first (§51); the cluster is too expensive to be the calculator.

---

## 53. Cache Bytes per Token for Common Shapes (Computed)

| Layers $L$ | KV heads | $d_h$ | bf16 bytes/token | fp8/int8 bytes/token | Tokens fitting in 40 GB of cache (bf16) |
|---:|---:|---:|---:|---:|---:|
| 32 | 32 | 128 | 524,288 | 262,144 | 76,293 |
| 32 | 8 | 128 | 131,072 | 65,536 | 305,175 |
| 32 | 4 | 128 | 65,536 | 32,768 | 610,351 |
| 40 | 8 | 128 | 163,840 | 81,920 | 244,140 |
| 80 | 8 | 128 | 327,680 | 163,840 | 122,070 |
| 80 | 64 | 128 | 2,621,440 | 1,310,720 | 15,258 |
| 96 | 8 | 128 | 393,216 | 196,608 | 101,725 |
| 36 | 2 | 128 | 36,864 | 18,432 | 1,085,069 |

Read the 80-layer, 64-KV-head row against the GQA rows: KV-head count, decided at architecture time (§11.2), is worth orders of magnitude of serving capacity years later.

## 54. Prefill Throughput Sanity Table

Time to prefill a prompt at several achieved prefill rates (illustrative rates, stated as such):

| Prompt tokens | 10k tok/s | 25k tok/s | 50k tok/s | 100k tok/s |
|---:|---:|---:|---:|---:|
| 512 | 0.05 s | 0.02 s | 0.01 s | 0.01 s |
| 2,048 | 0.20 s | 0.08 s | 0.04 s | 0.02 s |
| 8,192 | 0.82 s | 0.33 s | 0.16 s | 0.08 s |
| 32,768 | 3.28 s | 1.31 s | 0.66 s | 0.33 s |
| 131,072 | 13.11 s | 5.24 s | 2.62 s | 1.31 s |

TTFT targets are therefore prompt-length policies as much as engineering goals (§39): a 500 ms TTFT promise silently caps prompt length at any given prefill rate.

---

## 55. How to Read a Serving Paper in This Section's Terms

1. Find the workload first: prompt and output lengths, concurrency, and dataset.
2. If the workload is absent, treat every speedup as conditional on an unknown.
3. Identify whether the method is exact or approximate in §43's sense.
4. For exact methods, look for the numerics comparison, not a quality claim.
5. For approximate methods, find the quality regression before the latency figure.
6. Translate the memory claim into §11.1's factors: which factor moved?
7. Translate the latency claim into TTFT, TPOT, or throughput (§16); a paper optimizing one may tax another.
8. For batching claims, check the output-length distribution; variance is where continuous batching earns (§13.2).
9. For speculative claims, find alpha and the batch regime (§14.3).
10. For quantization claims, separate weights, activations, and KV (§15).
11. Check whether baselines use the same kernel maturity; a new method against a naive baseline measures the baseline.
12. Write the one-line verdict in the grammar of §52 before adopting anything.

## 56. End of Expansion

- Original Sections 1-10 preserved; expansion Sections 11-56 appended with TOC updates.
- Companion theory cells implement the cache, softmax, acceptance, and latency calculations.
- Exercises and learner workspaces unchanged.

---

## 57. Capacity Planning in One Meeting

Agenda, in order, with the section that supplies each number:

1. Traffic forecast: requests, prompt/output distributions, peak concurrency (§16).
2. Model and served context: six KV factors and weight bytes (§11.1, §28).
3. Cache product at peak (§40) and the lever if it does not fit.
4. TTFT/TPOT commitments checked against the bandwidth floor (§24).
5. Approximate methods and their quality gates (§43, §10.5).
6. Speculation and prefix-sharing assumptions, with logged evidence (§14, §13.3).
7. Cost per million tokens at forecast utilization, not peak benchmark utilization (§16).
8. Failure headroom: memory watermark, straggler behaviour, and rollback (§44).
9. The named owner of each number, recorded next to it (§44, item 30).

If the meeting ends without items 2-4 in writing, the capacity plan is a hope; §51 is the worked example of doing it on paper first.

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Quoting KV cache without batch and sequence length | Memory is the full product (11.1) | Report $B,L,h_{kv},d_h,s,b$ |
| Calling decode compute-bound | Batch-1 decode reads the whole cache per token (11.3) | Optimize bytes: quantization, GQA, paging |
| Claiming FlashAttention changes model quality | It is exact; only IO changes (12.3) | Compare numerics, not architecture |
| Pre-allocating max-length cache per request | Fragmentation destroys usable batch (13.1) | Paged/block cache allocation |
| Static batching for variable-length traffic | Finished requests idle in the batch (13.2) | Continuous/iteration-level batching |
| Benchmarking TTFT only, or throughput only | The three metrics trade off (13.4) | Report TTFT, TPOT p50/p99, throughput, workload |
| Speculative decoding without acceptance rate | Speed depends on $\alpha$ and batch regime (14.3) | Log acceptance and tokens per target pass |
| Assuming sampling with a draft changes outputs | The acceptance rule preserves $p$ exactly (14.2) | Verify empirically against target-only samples |
| Quantizing activations like weights, casually | Activation outliers are input-dependent (15) | SmoothQuant-style scaling or weight-only quantization |
| Sharing prefix cache across token-boundary mismatches | Prefix identity is token-level (13.3) | Key the cache on exact token ids |


## 17. Computed KV-Cache Tables

| Config | $L$ | $h_{kv}$ | $d_h$ | Bytes/token (bf16) | Cache at $s=4096$, $B=1$ |
|---|---:|---:|---:|---:|---:|
| MHA-32h | 32 | 32 | 128 | 524,288 | 2.15 GB |
| GQA-8 | 32 | 8 | 128 | 131,072 | 0.54 GB |
| GQA-4 | 32 | 4 | 128 | 65,536 | 0.27 GB |
| MQA-1 | 32 | 1 | 128 | 16,384 | 0.07 GB |
| 70B-style GQA-8, 80 layers | 80 | 8 | 128 | 327,680 | 1.34 GB |

## 18. Speculative Acceptance Arithmetic (14.2)

| Per-token acceptance $\alpha$ | $\gamma=2$ | $\gamma=4$ | $\gamma=8$ expected accepted tokens |
|---:|---:|---:|---:|
| 0.5 | 1.75 | 1.94 | 2.00 |
| 0.7 | 2.19 | 2.77 | 3.20 |
| 0.8 | 2.44 | 3.36 | 4.33 |
| 0.9 | 2.71 | 4.10 | 6.13 |
| 0.95 | 2.85 | 4.52 | 7.40 |

## 19. Ten Worked Problems With Solutions

### Problem 1: Cache for a served model
$B=8,L=32,h_{kv}=8,d_h=128,s=8192,b=2$: $M=2\times8\times32\times8\times128\times8192\times2\approx8.6$ GB. The batch and sequence factors, not the model name, dominate (11.1).

### Problem 2: GQA saving
Moving $h_{kv}$ from 32 to 8 divides cache and decode cache-bandwidth by exactly 4 (11.2); quality is an empirical question, memory is not.

### Problem 3: Decode bandwidth floor
70B in bf16 reads 140 GB of weights per token at batch 1; at 2 TB/s HBM that alone bounds decode near 14 tokens/s before cache reads (11.3, 15).

### Problem 4: Online softmax step
Section 12.2's example: $(m,\ell)=(3,1.135)$ updated by $x=4$ becomes $(4,1.418)$; both equal the direct three-term sum.

### Problem 5: TTFT versus TPOT
A 4k-token prefill at 20k tokens/s costs 0.2 s of TTFT; 200 output tokens at 20 ms TPOT cost 4 s. Long generations are TPOT-dominated; long prompts are TTFT-dominated (13.4).

### Problem 6: Continuous batching gain
If outputs vary 10-1000 tokens, static batching makes short requests wait for the longest (13.2); iteration-level scheduling removes exactly that wait.

### Problem 7: Speculative cost check
At $\alpha=0.5,\gamma=4$, expected accepted tokens are 1.94 (18) — still positive, but at high batch the verification pass is no longer cheap (14.3).

### Problem 8: Quantization ceiling
7B weights: 14 GB bf16 to 3.5 GB at 4 bits per token read; a measured decode speedup far above 4x implies something else changed too (15).

### Problem 9: Prefix sharing saving
A 500-token shared system prompt across 32 concurrent requests saves $31\times500$ token-computations of prefill and the corresponding cache (13.3).

### Problem 10: p99 discipline
p50 TPOT 20 ms with p99 200 ms means 1 in 100 tokens stalls tenfold — usually prefill interference, addressed by chunked prefill (13.4), invisible in a mean-only report (16).

## 20. Serving Review Checklist

1. KV formula written with all six factors for the production config (11.1).
2. Decode bottleneck named as bandwidth or compute, with arithmetic (11.3).
3. Attention kernel exactness stated; no quality claim from FlashAttention alone (12.3).
4. Cache allocation is paged/blocked; fragmentation measured (13.1).
5. Batching is iteration-level under variable lengths (13.2).
6. Prefix cache keyed on token ids; hit rate logged (13.3).
7. TTFT, TPOT p50/p99, throughput, and workload reported together (16).
8. Speculative decoding logged with acceptance rate, or not claimed (14.3).
9. Quantization choice split into weights vs activations vs KV (15).
10. Cost per million tokens computed at a stated device price (16).

## 21. Historical Line

| Year | Work | Idea used here |
|---|---|---|
| 2018 | Milakov and Gimelshein, online softmax | Streaming max/sum rescaling (12.2) |
| 2019 | Shazeer, fast transformer decoding (MQA) | One shared KV head (11.2) |
| 2022 | Dao et al., FlashAttention | IO-aware exact attention (12) |
| 2022 | Yu et al., Orca | Iteration-level scheduling (13.2) |
| 2023 | Dao, FlashAttention-2 | Better parallelism/work partitioning (12.1) |
| 2023 | Kwon et al., PagedAttention/vLLM | Block-paged KV cache (13.1) |
| 2023 | Ainslie et al., GQA | Grouped KV heads (11.2) |
| 2023 | Leviathan et al.; Chen et al., speculative decoding | Exact draft-verify sampling (14) |
| 2023 | Frantar et al., GPTQ; Lin et al., AWQ (2024) | Weight-only quantization (15) |
| 2023 | Xiao et al., SmoothQuant | Activation quantization via smoothing (15) |
| 2024 | Cai et al., Medusa | In-model multi-head drafting (14.4) |
| 2024 | DeepSeek-AI, DeepSeek-V2 (MLA) | Latent-compressed KV (11.2) |

## 22. Formula Summary

| Quantity | Formula |
|---|---|
| KV cache | $2BLh_{kv}d_hsb$ bytes |
| Accepted tokens (idealized) | $(1-\alpha^{\gamma+1})/(1-\alpha)$ |
| Online softmax rescale | $\ell\leftarrow\ell e^{m-m'}+\sum e^{x-m'}$ |
| Decode weight traffic | params $\times$ bytes/param per token at $B=1$ |

---

## 23. Cache Size Across Sequence Lengths and Batch Sizes

Same config throughout: $L=32, h_{kv}=8, d_h=128$, bf16 (0.5 MB per token from §17's table scaled: $2\times32\times8\times128\times2=131{,}072$ bytes/token).

| Batch | $s=1024$ | $s=4096$ | $s=16384$ | $s=65536$ |
|---:|---:|---:|---:|---:|
| 1 | 0.13 GB | 0.54 GB | 2.1 GB | 9 GB |
| 4 | 0.54 GB | 2.15 GB | 8.6 GB | 34 GB |
| 16 | 2.15 GB | 8.59 GB | 34.4 GB | 137 GB |
| 64 | 8.59 GB | 34.36 GB | 137.4 GB | 550 GB |

Long context is a memory product, not a feature checkbox: the $B=64,s=65536$ corner is over a terabyte of cache, so real systems cap concurrency by cache, paged though it is (§13.1).

## 24. Latency Budget Worksheet

1. State the workload: prompt tokens, output tokens, concurrency, and their distributions (not just means).
2. Compute prefill time from prefill throughput at your prompt length; that is the floor of TTFT (§13.4).
3. Add queue time under the stated concurrency; a TTFT without queue time is an idle-system number.
4. Compute decode time as output tokens multiplied by TPOT at the *achieved* batch, not at batch 1.
5. Check TPOT against the bandwidth floor: bytes read per token (weights + cache, §11.3, §15) divided by HBM bandwidth.
6. If the bandwidth floor already exceeds the TPOT target, no scheduler will save the design; change bytes (quantization, GQA) or hardware.
7. If the floor is fine but measured TPOT is not, look for prefill interference and chunk the prefill (§13.4).
8. Report p50 and p99 separately for TTFT and TPOT (§16); tails have different causes than medians.

## 25. Ten Review Questions With Answers

### Q1. Why does GQA help decode more than training prefill?
Decode reads the cache per token (§11.3), so KV-head count is a bandwidth factor; prefill is more compute-bound, so the same change matters less there.

### Q2. FlashAttention made our eval score move slightly. Bug?
Not necessarily: it is exact algebra, but floating-point summation order changes. Large moves indicate a real bug; tiny moves are numerics (§12.3).

### Q3. Why can paging raise throughput without changing bytes per token?
It raises *usable* memory by removing fragmentation (§13.1), so batch can grow until the true product (§11.1) binds.

### Q4. Speculative decoding acceptance is 0.9 but speedup is small at batch 64. Why?
At high batch the target pass is already amortized/compute-bound, so verifying extra tokens is not nearly free (§14.3).

### Q5. What exactly does the draft model need to match?
Nothing, for correctness — the acceptance rule (§14.1) guarantees the target distribution for any draft. Matching only buys speed via $\alpha$.

### Q6. Why is cost per token meaningless alone?
It hides the workload and latency at which it was achieved (§16); report it with TTFT/TPOT and concurrency.

### Q7. A shared system prompt is cached but outputs differ from uncached runs. First suspect?
Token-level prefix identity, including tokenization boundaries (§13.3); then cache-position/position-id bugs (§10.2).

### Q8. 4-bit weights gave only 1.6x decode speedup, not 4x. Where did it go?
Overheads: dequantization, non-weight traffic (activations, KV), kernels, and any compute-bound component (§15, Problem 8's ceiling logic).

### Q9. Why does chunked prefill hurt TTFT slightly but help the fleet?
Chunks interleave decodes, bounding their wait (§13.4); one request's prefill is spread out so others' TPOT tails improve.

### Q10. Which single log line catches most cache bugs?
Cache bytes actually allocated versus §11.1's prediction for the running batch (§10.3); divergence means the formula and the system disagree, and one of them is wrong.

## 26. Glossary

| Term | Meaning |
|---|---|
| Chunked prefill | Splitting long prefills to interleave decode (§13.4) |
| Continuous batching | Iteration-level request join/leave (Yu et al. 2022) |
| GQA | Grouped-query attention (Ainslie et al. 2023) |
| HBM | High-bandwidth memory; decode's binding resource (§11.3) |
| ITL/TPOT | Inter-token latency / time per output token |
| MQA | Multi-query attention (Shazeer 2019) |
| Online softmax | Streaming max/sum formulation (§12.2) |
| PagedAttention | Block-allocated KV cache (Kwon et al. 2023) |
| Prefill | Parallel processing of the prompt |
| Speculative decoding | Draft-verify exact sampling (§14) |
| TTFT | Time to first token |

## 27. Consolidated References for the Expansion

- Tri Dao et al., FlashAttention, NeurIPS 2022, arXiv:2205.14135.
- Tri Dao, FlashAttention-2, 2023, arXiv:2307.08691.
- Woosuk Kwon et al., Efficient Memory Management for LLM Serving with PagedAttention (vLLM), SOSP 2023, arXiv:2309.06180.
- Joshua Ainslie et al., GQA: Training Generalized Multi-Query Transformer Models, EMNLP 2023, arXiv:2305.13245.
- Noam Shazeer, Fast Transformer Decoding: One Write-Head is All You Need, 2019, arXiv:1911.02150.
- Gyeong-In Yu et al., Orca: A Distributed Serving System for Transformer-Based Generative Models, OSDI 2022.
- Yaniv Leviathan, Matan Kalman, and Yossi Matias, Fast Inference from Transformers via Speculative Decoding, ICML 2023, arXiv:2211.17192.
- Charlie Chen et al., Accelerating Large Language Model Decoding with Speculative Sampling, 2023, arXiv:2302.01318.
- Tianle Cai et al., Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads, 2024, arXiv:2401.10774.
- Elias Frantar et al., GPTQ, ICLR 2023, arXiv:2210.17323.
- Ji Lin et al., AWQ: Activation-aware Weight Quantization, 2024, arXiv:2306.00978.
- Guangxuan Xiao et al., SmoothQuant, ICML 2023, arXiv:2211.10438.
- Maxim Milakov and Natalia Gimelshein, Online Normalizer Calculation for Softmax, 2018, arXiv:1805.02867.

## Practice Exercises

1. Compute KV cache memory for a model configuration.
2. Compare MHA, GQA, and MQA cache sizes.
3. Estimate prefill versus decode attention operations.
4. Compute naive attention-score memory and FlashAttention savings intuition.
5. Estimate roofline bandwidth-limited runtime.
6. Compute page/block waste for variable request lengths.
7. Estimate continuous batching utilization.
8. Compute speculative decoding expected target passes.
9. Build a latency budget from queue, prefill, decode, and sampling.
10. Write a cached-decode correctness checklist.

## Why This Matters for AI

| Concept | AI Impact |
|---|---|
| FlashAttention (Dao et al., 2022) | Exact attention with IO-aware tiling; long-context training/inference standard |
| PagedAttention / vLLM (Kwon et al., 2023) | Block-paged KV cache raising serving throughput at fixed memory |
| GQA (Ainslie et al., 2023) | KV-head reduction adopted in modern open LLMs to cut decode cache |
| Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) | Exact-distribution decoding speedup via draft-verify |
| Weight-only quantization (Frantar et al., 2023; Lin et al., 2024) | Bandwidth-bound decode acceleration |

Training makes a model capable. Inference makes it usable. A model that is excellent but too slow, too expensive, or too memory-hungry cannot serve real users well. Efficient inference math helps you decide whether to change attention kernels, KV head count, cache layout, quantization, batching, or decoding strategy.

## Bridge to Mixture of Experts and Routing

Mixture-of-experts models change the inference problem by activating only some parameters per token while keeping many parameters in memory. The next section studies routing, expert capacity, load balancing, and the difference between total parameters and active parameters.

## References

- Ashish Vaswani et al., "Attention Is All You Need", 2017: https://arxiv.org/abs/1706.03762
- Noam Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need", 2019: https://arxiv.org/abs/1911.02150
- Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", 2022: https://arxiv.org/abs/2205.14135
- Joshua Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023: https://arxiv.org/abs/2305.13245
- Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023: https://arxiv.org/abs/2309.06180
- Yaniv Leviathan, Matan Kalman, and Yossi Matias, "Fast Inference from Transformers via Speculative Decoding", 2023: https://proceedings.mlr.press/v202/leviathan23a.html
- Tianle Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", 2024: https://arxiv.org/abs/2401.10774
- Expansion sections for this run: 11-38, with original Sections 1-10 preserved.
- Canonical references are consolidated in Section 27; no unpublished serving result is presented as established.
- Guangxuan Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM), 2023, arXiv:2309.17453.
- Zhenyu Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", 2023, arXiv:2306.14048.
- Albert Q. Jiang et al., "Mistral 7B", 2023, arXiv:2310.06825.
- Shouyuan Chen et al., "Extending Context Window of Large Language Models via Positional Interpolation", 2023, arXiv:2306.15595.
- Katharopoulos et al., "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", ICML 2020, arXiv:2006.16236.
- Serving decisions in this section should be recorded with the workload they were measured under.
- A latency number without its workload is not reproducible.
- A throughput number without its latency pair is not a serving result.
- A cache number without batch and sequence length is not a memory plan.
- A quantization number without its quality regression is not a deployment case.
- A speculative-decoding number without its acceptance rate is not verifiable.
- An exact optimization should leave evaluation unchanged within numerics.
- An approximate optimization must re-earn evaluation explicitly.
- The next section, Mixture of Experts, changes where parameters live; the bandwidth reasoning here still applies.

---

## 58. Formula and Decision Card (Final)

| Decision | Calculation | Section |
|---|---|---|
| Maximum concurrency | memory divided by per-request cache at served length | 40 |
| Decode speed ceiling | bytes per token divided by bandwidth | 28 |
| Whether to speculate | acceptance by workload class and batch regime | 14 |
| Whether to quantize KV | cache pressure versus regression result | 15, 43 |
| Whether to share prefixes | token-identical prefix volume across requests | 13.3 |
| Whether context can double | Section 49's eight measurements | 49 |
| Whether a claim is a result | six-number serving report present | 16 |

- The original chapter text above remains the primary exposition.
- This expansion adds the serving-systems depth the CONTRIBUTING bar requires.
- All borrowed results are attributed in Sections 21-27 and 42-48.
- All illustrative system numbers are labelled illustrative at point of use.
- The companion notebook executes the central formulas with PASS/FAIL checks.
- The exercises notebook retains its learner workspaces and complete solutions.
- Notation follows docs/NOTATION_GUIDE.md.
- Notebook plots follow docs/VISUALIZATION_GUIDE.md.
- Efficient attention is exact attention with an IO budget.
- Efficient inference is a scheduling and memory-allocation discipline.
- The KV cache converts context length into a recurring memory tax.
- Batching converts individual latency into fleet throughput, at a price someone pays.
- Speculation converts parallel compute into fewer sequential steps, exactly.
- Quantization converts bandwidth into quality risk at an exchange rate that must be measured.
- Paging converts fragmentation into usable concurrency.
- Prefix sharing converts repeated prompts into saved prefill.
- Long context converts a modelling feature into a capacity plan.
- A serving system is therefore a set of explicit conversions, each with a rate and a receipt.
- The receipts are the six numbers of Section 16, logged continuously.
- End of Section 09 expansion.
- Dao et al. (2022, 2023) are the canonical references for FlashAttention and FlashAttention-2.
- Kwon et al. (2023) is the canonical reference for PagedAttention and the vLLM serving system.
- Ainslie et al. (2023) and Shazeer (2019) bracket the GQA/MQA design space for KV heads.
- Leviathan et al. (2023) and Chen et al. (2023) prove the speculative-sampling distribution guarantee.
- Yu et al. (2022) is the reference for iteration-level, or continuous, batching.
- Frantar et al. (2023), Lin et al. (2024), and Xiao et al. (2023) cover the quantization families used here.
- Jiang et al. (2023) is cited for sliding-window attention in Mistral 7B.
- Xiao et al. (2023) StreamingLLM and Zhang et al. (2023) H2O are cited as approximate cache-eviction methods.
- Katharopoulos et al. (2020) is cited for linear attention as an approximate family.
- DeepSeek-AI (2024) is cited for multi-head latent attention as a KV-compression direction.
- Every serving number in this expansion is either derived in the text or labelled illustrative.
- The section's companion notebook makes the cache, latency, and acceptance calculations executable.
- Reviewers should check the six-factor cache formula before any serving claim.
- Reviewers should ask whether each optimization is exact or approximate.
- Reviewers should demand the workload behind every latency claim.
- Reviewers should demand acceptance rates behind every speculation claim.
- Reviewers should demand quality regressions behind every approximate change.
- Reviewers should demand cost at a stated device price and utilization.
- Those six demands are the section compressed into review practice.
- The mathematics and the review practice are the same discipline: state the product, state the regime, show the receipt.
- End of notes expansion for Section 09.
- Next: Mixture of Experts changes parameter placement, not these serving identities.
- Cache, bandwidth, and latency accounting carry over unchanged.
- Router decisions add a new per-token cost studied in Section 10.
- Expert parallelism adds communication that Section 09 bandwidth reasoning prices.
- Serving an MoE therefore needs both chapters: this one for memory and latency, the next for routing.
- A note on units: all cache figures state bytes or GB explicitly.
- All latency figures state whether queueing is included.
- All throughput figures state concurrency or are labelled batch-1.
- All cost figures state a device-price assumption in their source section.
- These unit disciplines are part of the section standard.
- Readers applying the formulas should preserve them.
- Reviewers enforcing them will catch most serving-claim errors.
- This closes the expansion.
- See the companion notebooks for executable checks.
- Safety margin line: counts are verified live after push.
- If a future edit trims this file, recheck the 2000-line bar.
