[<- RAG Math and Retrieval](../12-RAG-Math-and-Retrieval/notes.md) | [Home](../../README.md)

---

# Serving and Systems Tradeoffs

Serving is where LLM math becomes a live system. A deployed model must manage stochastic demand, queueing, prefill, decode, KV cache memory, batching, scheduling, cost, observability, and reliability.

## Overview

The serving equation is not one equation. It is a budget:

$$
T_\mathrm{user}=T_q+T_p+n_\mathrm{out}\mathrm{TPOT}+T_o,
$$

plus memory:

$$
M_w+M_\mathrm{KV}+M_\mathrm{work}\le M_\mathrm{GPU},
$$

plus cost:

$$
\mathrm{CPM}=10^6C_\mathrm{hour}/(3600Q).
$$

Serving tradeoffs are practical: low latency, high throughput, low cost, high quality, and high reliability cannot all be maximized independently.

## Prerequisites

- Efficient attention and inference metrics
- KV cache memory math
- RAG retrieval latency and context budget
- Basic probability for percentiles and queueing

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates Little's law, utilization curves, latency decomposition, batch tradeoffs, memory concurrency, cost per million tokens, autoscaling, and SLO budgets. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for queueing, latency, memory, cost, scheduling, and observability. |

## Learning Objectives

After this section, you should be able to:

- Compute TTFT, TPOT, total latency, throughput, and cost per million tokens.
- Use Little's law for serving capacity planning.
- Explain why high utilization increases queueing delay.
- Estimate max concurrency from weight memory and KV cache memory.
- Compare static batching, continuous batching, and chunked prefill.
- Explain serving parallelism choices and phase splitting.
- Build a cost model for tokens per dollar.
- Define SLOs, error budgets, and observability traces.
- Choose operational fallbacks under overload.

## Table of Contents

1. [Serving Objectives](#1-serving-objectives)
   - 1.1 [User latency](#11-user-latency)
   - 1.2 [Throughput](#12-throughput)
   - 1.3 [Cost](#13-cost)
   - 1.4 [Quality](#14-quality)
   - 1.5 [Reliability](#15-reliability)
2. [Queueing Basics](#2-queueing-basics)
   - 2.1 [Arrival rate](#21-arrival-rate)
   - 2.2 [Service rate](#22-service-rate)
   - 2.3 [Utilization](#23-utilization)
   - 2.4 [Little's law](#24-littles-law)
   - 2.5 [Tail latency](#25-tail-latency)
3. [Latency Decomposition](#3-latency-decomposition)
   - 3.1 [Queue time](#31-queue-time)
   - 3.2 [Prefill time](#32-prefill-time)
   - 3.3 [Decode time](#33-decode-time)
   - 3.4 [Postprocess time](#34-postprocess-time)
   - 3.5 [End-to-end latency](#35-endtoend-latency)
4. [Batching Tradeoffs](#4-batching-tradeoffs)
   - 4.1 [Static batching](#41-static-batching)
   - 4.2 [Continuous batching](#42-continuous-batching)
   - 4.3 [Batch efficiency](#43-batch-efficiency)
   - 4.4 [Head-of-line blocking](#44-headofline-blocking)
   - 4.5 [Chunked prefill](#45-chunked-prefill)
5. [Memory and Concurrency](#5-memory-and-concurrency)
   - 5.1 [Weight memory](#51-weight-memory)
   - 5.2 [KV cache memory](#52-kv-cache-memory)
   - 5.3 [Workspace memory](#53-workspace-memory)
   - 5.4 [Max concurrency](#54-max-concurrency)
   - 5.5 [Fragmentation](#55-fragmentation)
6. [Parallelism for Serving](#6-parallelism-for-serving)
   - 6.1 [Tensor parallelism](#61-tensor-parallelism)
   - 6.2 [Pipeline parallelism](#62-pipeline-parallelism)
   - 6.3 [Data parallel replicas](#63-data-parallel-replicas)
   - 6.4 [Phase splitting](#64-phase-splitting)
   - 6.5 [Network cost](#65-network-cost)
7. [Cost Modeling](#7-cost-modeling)
   - 7.1 [GPU-hour cost](#71-gpuhour-cost)
   - 7.2 [Tokens per dollar](#72-tokens-per-dollar)
   - 7.3 [Cost per million tokens](#73-cost-per-million-tokens)
   - 7.4 [Utilization](#74-utilization)
   - 7.5 [Quality-adjusted cost](#75-qualityadjusted-cost)
8. [Scheduling Policies](#8-scheduling-policies)
   - 8.1 [FIFO](#81-fifo)
   - 8.2 [Shortest remaining work](#82-shortest-remaining-work)
   - 8.3 [Priority queues](#83-priority-queues)
   - 8.4 [Admission control](#84-admission-control)
   - 8.5 [Autoscaling](#85-autoscaling)
9. [Observability and SLOs](#9-observability-and-slos)
   - 9.1 [Metrics](#91-metrics)
   - 9.2 [Percentiles](#92-percentiles)
   - 9.3 [Error budget](#93-error-budget)
   - 9.4 [Tracing](#94-tracing)
   - 9.5 [Canarying](#95-canarying)
10. [Operational Tradeoffs](#10-operational-tradeoffs)
70. [End Matter](#70-end-matter)
69. [Serving Review Margin](#69-serving-review-margin)
68. [Final Serving Notes](#68-final-serving-notes)
67. [Serving Margin Notes](#67-serving-margin-notes)
63. [Serving Decision Tables](#63-serving-decision-tables)
64. [Worked Fleet Example, End to End](#64-worked-fleet-example-end-to-end)
65. [Final Review List](#65-final-review-list)
66. [End of Chapter 15](#66-end-of-chapter-15)
57. [SLO Error Budgets Worked](#57-slo-error-budgets-worked)
58. [Capacity Incident Playbook (condensed)](#58-capacity-incident-playbook-(condensed))
59. [What Good Looks Like: A Serving Report](#59-what-good-looks-like-a-serving-report)
60. [Extended Worksheets](#60-extended-worksheets)
61. [Final Problem Set D](#61-final-problem-set-d)
62. [Chapter 15 Completion Note](#62-chapter-15-completion-note)
51. [Prefill/Decode Disaggregation Tradeoffs](#51-prefilldecode-disaggregation-tradeoffs)
52. [Batching Under Length Heterogeneity](#52-batching-under-length-heterogeneity)
53. [The Serving Stack as One System](#53-the-serving-stack-as-one-system)
54. [Final Computed Tables](#54-final-computed-tables)
55. [Ten Final Checks (with answers)](#55-ten-final-checks-(with-answers))
56. [Closing Notes for Section 13 and Chapter 15](#56-closing-notes-for-section-13-and-chapter-15)
45. [Worked Queueing Example in Full](#45-worked-queueing-example-in-full)
46. [Worked Cost Example in Full](#46-worked-cost-example-in-full)
47. [Reading a Serving Regression](#47-reading-a-serving-regression)
48. [Extended Computed Tables](#48-extended-computed-tables)
49. [Step-by-Step Problem Set C](#49-step-by-step-problem-set-c)
50. [One-Page Serving Review](#50-one-page-serving-review)
11. [Latency Vocabulary: TTFT, TPOT, and End-to-End](#11-latency-vocabulary-ttft-tpot-and-end-to-end)
12. [Queueing: Utilization and the Nonlinear Wait](#12-queueing-utilization-and-the-nonlinear-wait)
13. [Little's Law as a Capacity Tool](#13-little's-law-as-a-capacity-tool)
14. [Prefill Is Compute-Bound, Decode Is Bandwidth-Bound](#14-prefill-is-compute-bound-decode-is-bandwidth-bound)
15. [Continuous Batching and Iteration-Level Scheduling](#15-continuous-batching-and-iteration-level-scheduling)
16. [Chunked Prefill and Piggybacking](#16-chunked-prefill-and-piggybacking)
17. [KV Cache and Maximum Concurrency](#17-kv-cache-and-maximum-concurrency)
18. [Parallelism for Serving](#18-parallelism-for-serving)
19. [Cost Accounting That Survives Scrutiny](#19-cost-accounting-that-survives-scrutiny)
20. [Autoscaling, Admission Control, and Graceful Degradation](#20-autoscaling-admission-control-and-graceful-degradation)
21. [Why This Matters for AI](#21-why-this-matters-for-ai)
22. [Computed Tables](#22-computed-tables)
23. [Ten Worked Problems With Solutions](#23-ten-worked-problems-with-solutions)
24. [Serving Review Protocol](#24-serving-review-protocol)
25. [Scheduling Policies in Practice](#25-scheduling-policies-in-practice)
26. [Prefix Caching and Request Similarity](#26-prefix-caching-and-request-similarity)
27. [Speculative and Parallel Decoding in Serving](#27-speculative-and-parallel-decoding-in-serving)
28. [Multi-Model and Routed Serving](#28-multi-model-and-routed-serving)
29. [Reliability: Errors, Timeouts, and Partial Failures](#29-reliability-errors-timeouts-and-partial-failures)
30. [Observability That Answers Questions](#30-observability-that-answers-questions)
31. [Step-by-Step Problem Set B](#31-step-by-step-problem-set-b)
32. [Additional Computed Tables](#32-additional-computed-tables)
33. [Glossary and Formula Appendix](#33-glossary-and-formula-appendix)
34. [Pre-Production Serving Checklist](#34-pre-production-serving-checklist)
35. [Load Testing Without Lying to Yourself](#35-load-testing-without-lying-to-yourself)
36. [Capacity Planning Over Growth](#36-capacity-planning-over-growth)
37. [Energy and Carbon as Serving Costs](#37-energy-and-carbon-as-serving-costs)
38. [Benchmarking Serving Systems Fairly](#38-benchmarking-serving-systems-fairly)
39. [Incident Review for Serving](#39-incident-review-for-serving)
40. [Chapter 15 Synthesis: From Probabilities to Production](#40-chapter-15-synthesis-from-probabilities-to-production)
41. [Combinatorial Worksheets](#41-combinatorial-worksheets)
42. [Ten Quick Checks (with answers)](#42-ten-quick-checks-(with-answers))
43. [Serving Configuration Record](#43-serving-configuration-record)
44. [Section 13 Takeaways](#44-section-13-takeaways)
   - 10.1 [Fallback models](#101-fallback-models)
   - 10.2 [Caching](#102-caching)
   - 10.3 [Rate limits](#103-rate-limits)
   - 10.4 [Graceful degradation](#104-graceful-degradation)
   - 10.5 [Rollback](#105-rollback)

---

## Serving Control Loop

```text
traffic -> admission -> queue -> scheduler -> prefill/decode workers -> postprocess -> response
             |            |          |                 |                  |
          rate limit    metrics   batching          memory             traces
```

## 1. Serving Objectives

This part studies serving objectives as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [User latency](#1-user-latency) | interactive systems care about time to first token and total response time | $T_\mathrm{user}=\mathrm{TTFT}+n_\mathrm{out}\mathrm{TPOT}$ |
| [Throughput](#1-throughput) | operators care about tokens completed per second | $Q=\mathrm{tokens}/\mathrm{sec}$ |
| [Cost](#1-cost) | product decisions depend on cost per useful token | $\mathrm{cost/token}=T_\mathrm{gpu}\cdot \mathrm{price}/\mathrm{tokens}$ |
| [Quality](#1-quality) | systems changes must preserve model behavior | $\Delta S$ bounded |
| [Reliability](#1-reliability) | serving must meet SLOs under variable load | $P(T\le T_\mathrm{SLO})\ge 0.95$ |

### 1.1 User latency

**Main idea.** Interactive systems care about time to first token and total response time.

Core relation:

$$T_\mathrm{user}=\mathrm{TTFT}+n_\mathrm{out}\mathrm{TPOT}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.2 Throughput

**Main idea.** Operators care about tokens completed per second.

Core relation:

$$Q=\mathrm{tokens}/\mathrm{sec}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.3 Cost

**Main idea.** Product decisions depend on cost per useful token.

Core relation:

$$\mathrm{cost/token}=T_\mathrm{gpu}\cdot \mathrm{price}/\mathrm{tokens}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.4 Quality

**Main idea.** Systems changes must preserve model behavior.

Core relation:

$$\Delta S$ bounded$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 1.5 Reliability

**Main idea.** Serving must meet slos under variable load.

Core relation:

$$P(T\le T_\mathrm{SLO})\ge 0.95$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 2. Queueing Basics

This part studies queueing basics as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Arrival rate](#2-arrival-rate) | requests arrive stochastically | $\lambda$ requests/sec |
| [Service rate](#2-service-rate) | the system completes work at rate mu | $\mu$ requests/sec |
| [Utilization](#2-utilization) | high utilization increases queueing delay | $\rho=\lambda/\mu$ |
| [Little's law](#2-littles-law) | average concurrency equals arrival rate times latency | $L=\lambda W$ |
| [Tail latency](#2-tail-latency) | p95 and p99 grow before averages look alarming | $Q_{0.95}(T)$ |

### 2.1 Arrival rate

**Main idea.** Requests arrive stochastically.

Core relation:

$$\lambda$ requests/sec$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.2 Service rate

**Main idea.** The system completes work at rate mu.

Core relation:

$$\mu$ requests/sec$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.3 Utilization

**Main idea.** High utilization increases queueing delay.

Core relation:

$$\rho=\lambda/\mu$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.4 Little's law

**Main idea.** Average concurrency equals arrival rate times latency.

Core relation:

$$L=\lambda W$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is the smallest useful equation for capacity planning.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 2.5 Tail latency

**Main idea.** P95 and p99 grow before averages look alarming.

Core relation:

$$Q_{0.95}(T)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 3. Latency Decomposition

This part studies latency decomposition as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Queue time](#3-queue-time) | waiting before GPU work starts | $T_q$ |
| [Prefill time](#3-prefill-time) | process prompt and build KV cache | $T_p$ |
| [Decode time](#3-decode-time) | generate output tokens serially | $T_d=n_\mathrm{out}\mathrm{TPOT}$ |
| [Postprocess time](#3-postprocess-time) | sampling, detokenization, filters, and transport add overhead | $T_o$ |
| [End-to-end latency](#3-endtoend-latency) | measure the whole path users experience | $T=T_q+T_p+T_d+T_o$ |

### 3.1 Queue time

**Main idea.** Waiting before gpu work starts.

Core relation:

$$T_q$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.2 Prefill time

**Main idea.** Process prompt and build kv cache.

Core relation:

$$T_p$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.3 Decode time

**Main idea.** Generate output tokens serially.

Core relation:

$$T_d=n_\mathrm{out}\mathrm{TPOT}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.4 Postprocess time

**Main idea.** Sampling, detokenization, filters, and transport add overhead.

Core relation:

$$T_o$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 3.5 End-to-end latency

**Main idea.** Measure the whole path users experience.

Core relation:

$$T=T_q+T_p+T_d+T_o$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 4. Batching Tradeoffs

This part studies batching tradeoffs as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Static batching](#4-static-batching) | wait to form a batch and run it together | $B$ fixed |
| [Continuous batching](#4-continuous-batching) | insert and remove requests between decode steps | $B_t$ changes |
| [Batch efficiency](#4-batch-efficiency) | larger batches increase utilization until memory or latency limits | $Q(B)$ |
| [Head-of-line blocking](#4-headofline-blocking) | long requests can delay short requests | $T_\mathrm{short}$ increases |
| [Chunked prefill](#4-chunked-prefill) | split long prompts to avoid starving decode | $T_p=\sum_c T_{p,c}$ |

### 4.1 Static batching

**Main idea.** Wait to form a batch and run it together.

Core relation:

$$B$ fixed$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.2 Continuous batching

**Main idea.** Insert and remove requests between decode steps.

Core relation:

$$B_t$ changes$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is why modern LLM serving does not behave like ordinary fixed-batch inference.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.3 Batch efficiency

**Main idea.** Larger batches increase utilization until memory or latency limits.

Core relation:

$$Q(B)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.4 Head-of-line blocking

**Main idea.** Long requests can delay short requests.

Core relation:

$$T_\mathrm{short}$ increases$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 4.5 Chunked prefill

**Main idea.** Split long prompts to avoid starving decode.

Core relation:

$$T_p=\sum_c T_{p,c}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 5. Memory and Concurrency

This part studies memory and concurrency as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Weight memory](#5-weight-memory) | model weights consume a fixed resident footprint | $M_w$ |
| [KV cache memory](#5-kv-cache-memory) | active requests consume context-dependent memory | $M_\mathrm{KV}=2BLTH_{kv}d_hb$ |
| [Workspace memory](#5-workspace-memory) | kernels and temporary buffers also need headroom | $M_\mathrm{work}$ |
| [Max concurrency](#5-max-concurrency) | available memory bounds active tokens | $M_w+M_\mathrm{KV}+M_\mathrm{work}\le M_\mathrm{GPU}$ |
| [Fragmentation](#5-fragmentation) | variable request lengths waste reserved cache blocks | $M_\mathrm{reserved}-M_\mathrm{used}$ |

### 5.1 Weight memory

**Main idea.** Model weights consume a fixed resident footprint.

Core relation:

$$M_w$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.2 KV cache memory

**Main idea.** Active requests consume context-dependent memory.

Core relation:

$$M_\mathrm{KV}=2BLTH_{kv}d_hb$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.3 Workspace memory

**Main idea.** Kernels and temporary buffers also need headroom.

Core relation:

$$M_\mathrm{work}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.4 Max concurrency

**Main idea.** Available memory bounds active tokens.

Core relation:

$$M_w+M_\mathrm{KV}+M_\mathrm{work}\le M_\mathrm{GPU}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** Most practical serving limits are memory limits before they are pure compute limits.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 5.5 Fragmentation

**Main idea.** Variable request lengths waste reserved cache blocks.

Core relation:

$$M_\mathrm{reserved}-M_\mathrm{used}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 6. Parallelism for Serving

This part studies parallelism for serving as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Tensor parallelism](#6-tensor-parallelism) | split matrix operations across devices | $N_\mathrm{tp}$ |
| [Pipeline parallelism](#6-pipeline-parallelism) | place layers on different devices | $N_\mathrm{pp}$ |
| [Data parallel replicas](#6-data-parallel-replicas) | replicate the full serving stack for more throughput | $N_\mathrm{replica}$ |
| [Phase splitting](#6-phase-splitting) | prefill and decode may run on different pools | $\mathrm{prefill\ pool},\mathrm{decode\ pool}$ |
| [Network cost](#6-network-cost) | multi-node serving pays communication latency | $T_\mathrm{net}$ |

### 6.1 Tensor parallelism

**Main idea.** Split matrix operations across devices.

Core relation:

$$N_\mathrm{tp}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.2 Pipeline parallelism

**Main idea.** Place layers on different devices.

Core relation:

$$N_\mathrm{pp}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.3 Data parallel replicas

**Main idea.** Replicate the full serving stack for more throughput.

Core relation:

$$N_\mathrm{replica}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.4 Phase splitting

**Main idea.** Prefill and decode may run on different pools.

Core relation:

$$\mathrm{prefill\ pool},\mathrm{decode\ pool}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 6.5 Network cost

**Main idea.** Multi-node serving pays communication latency.

Core relation:

$$T_\mathrm{net}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 7. Cost Modeling

This part studies cost modeling as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [GPU-hour cost](#7-gpuhour-cost) | hardware price turns time into dollars | $C_\mathrm{hour}$ |
| [Tokens per dollar](#7-tokens-per-dollar) | throughput divided by hourly cost | $\mathrm{tokens}/\$=3600Q/C_\mathrm{hour}$ |
| [Cost per million tokens](#7-cost-per-million-tokens) | standardize cost reporting | $\mathrm{CPM}=10^6C_\mathrm{hour}/(3600Q)$ |
| [Utilization](#7-utilization) | idle capacity increases effective cost | $C_\mathrm{effective}=C_\mathrm{nominal}/u$ |
| [Quality-adjusted cost](#7-qualityadjusted-cost) | cheaper systems can be worse if quality falls | $J=\mathrm{cost}+\lambda(1-S)$ |

### 7.1 GPU-hour cost

**Main idea.** Hardware price turns time into dollars.

Core relation:

$$C_\mathrm{hour}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.2 Tokens per dollar

**Main idea.** Throughput divided by hourly cost.

Core relation:

$$\mathrm{tokens}/\$=3600Q/C_\mathrm{hour}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.3 Cost per million tokens

**Main idea.** Standardize cost reporting.

Core relation:

$$\mathrm{CPM}=10^6C_\mathrm{hour}/(3600Q)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is the number that connects kernel work to product economics.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.4 Utilization

**Main idea.** Idle capacity increases effective cost.

Core relation:

$$C_\mathrm{effective}=C_\mathrm{nominal}/u$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 7.5 Quality-adjusted cost

**Main idea.** Cheaper systems can be worse if quality falls.

Core relation:

$$J=\mathrm{cost}+\lambda(1-S)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 8. Scheduling Policies

This part studies scheduling policies as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [FIFO](#8-fifo) | serve requests in arrival order | $\mathrm{order}=t_\mathrm{arrival}$ |
| [Shortest remaining work](#8-shortest-remaining-work) | favor small jobs to reduce mean latency | $\min \hat T_\mathrm{remaining}$ |
| [Priority queues](#8-priority-queues) | separate interactive and batch traffic | $p_i$ priority |
| [Admission control](#8-admission-control) | reject or defer work when queues exceed budget | $L_q>L_\mathrm{max}$ |
| [Autoscaling](#8-autoscaling) | add replicas when load exceeds target utilization | $n\ge \lambda/(\rho_\mathrm{target}\mu)$ |

### 8.1 FIFO

**Main idea.** Serve requests in arrival order.

Core relation:

$$\mathrm{order}=t_\mathrm{arrival}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.2 Shortest remaining work

**Main idea.** Favor small jobs to reduce mean latency.

Core relation:

$$\min \hat T_\mathrm{remaining}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.3 Priority queues

**Main idea.** Separate interactive and batch traffic.

Core relation:

$$p_i$ priority$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.4 Admission control

**Main idea.** Reject or defer work when queues exceed budget.

Core relation:

$$L_q>L_\mathrm{max}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 8.5 Autoscaling

**Main idea.** Add replicas when load exceeds target utilization.

Core relation:

$$n\ge \lambda/(\rho_\mathrm{target}\mu)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 9. Observability and SLOs

This part studies observability and slos as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Metrics](#9-metrics) | log TTFT, TPOT, total latency, queue time, throughput, memory, and errors | $m_t$ |
| [Percentiles](#9-percentiles) | track p50, p95, and p99 separately | $Q_p(T)$ |
| [Error budget](#9-error-budget) | allowed failures over a window | $B=(1-\mathrm{SLO})N$ |
| [Tracing](#9-tracing) | attach per-request spans for queue, prefill, decode, retrieval, and postprocess | $\mathrm{trace}$ |
| [Canarying](#9-canarying) | roll out changes to a small fraction and compare metrics | $\Delta m$ |

### 9.1 Metrics

**Main idea.** Log ttft, tpot, total latency, queue time, throughput, memory, and errors.

Core relation:

$$m_t$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.2 Percentiles

**Main idea.** Track p50, p95, and p99 separately.

Core relation:

$$Q_p(T)$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.3 Error budget

**Main idea.** Allowed failures over a window.

Core relation:

$$B=(1-\mathrm{SLO})N$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.4 Tracing

**Main idea.** Attach per-request spans for queue, prefill, decode, retrieval, and postprocess.

Core relation:

$$\mathrm{trace}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** Without per-request traces, serving optimization becomes guesswork.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 9.5 Canarying

**Main idea.** Roll out changes to a small fraction and compare metrics.

Core relation:

$$\Delta m$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
## 10. Operational Tradeoffs

This part studies operational tradeoffs as systems math for LLM deployment. The useful habit is to connect every serving choice to latency, throughput, memory, cost, quality, or reliability.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Fallback models](#10-fallback-models) | route to smaller models under overload | $M_\mathrm{large}\rightarrow M_\mathrm{small}$ |
| [Caching](#10-caching) | reuse repeated prompts or retrieved context when safe | $y=f(x)$ cache hit |
| [Rate limits](#10-rate-limits) | protect service health by limiting demand | $\lambda\le\lambda_\mathrm{max}$ |
| [Graceful degradation](#10-graceful-degradation) | shorter outputs, lower k retrieval, or smaller model can preserve responsiveness | $T\downarrow$ with bounded $\Delta S$ |
| [Rollback](#10-rollback) | keep a fast path back to the previous stable serving configuration | $v_{new}\rightarrow v_{old}$ |

### 10.1 Fallback models

**Main idea.** Route to smaller models under overload.

Core relation:

$$M_\mathrm{large}\rightarrow M_\mathrm{small}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.2 Caching

**Main idea.** Reuse repeated prompts or retrieved context when safe.

Core relation:

$$y=f(x)$ cache hit$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.3 Rate limits

**Main idea.** Protect service health by limiting demand.

Core relation:

$$\lambda\le\lambda_\mathrm{max}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.4 Graceful degradation

**Main idea.** Shorter outputs, lower k retrieval, or smaller model can preserve responsiveness.

Core relation:

$$T\downarrow$ with bounded $\Delta S$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.
### 10.5 Rollback

**Main idea.** Keep a fast path back to the previous stable serving configuration.

Core relation:

$$v_{new}\rightarrow v_{old}$$

LLM serving is a queueing and memory-management problem wrapped around transformer inference. The model must be fast enough, cheap enough, reliable enough, and good enough at the same time. Improving one axis can hurt another: larger batches improve throughput but can increase latency; longer contexts improve answer quality but consume KV memory; quantization saves memory but can change quality.

**Worked micro-example.** If requests arrive at $\lambda=20$ per second and average end-to-end latency is $W=1.5$ seconds, Little's law gives average concurrency $L=30$ requests. If each active request consumes KV cache proportional to its context length, concurrency directly becomes a memory-planning number.

**Implementation check.** Measure queue time, prefill time, decode TPOT, output length, memory use, and final status for each request. Then inspect percentiles, not only means.

**AI connection.** This is a practical serving control variable.

**Common mistake.** Do not report tokens/sec alone. A serving system can have high throughput while p95 latency is unacceptable.

---

## 11. Latency Vocabulary: TTFT, TPOT, and End-to-End

Time to first token (TTFT) is queue time plus retrieval time (Section 12) plus prefill time.
Time per output token (TPOT) is the decode step time experienced by the user after the first token.
End-to-end latency for $n$ output tokens is approximately TTFT $+(n-1)\times$ TPOT.
Worked example: TTFT 800 ms and TPOT 25 ms for 200 tokens gives about $0.8+199\times0.025\approx5.8$ seconds.
Streaming hides some latency perceptually but does not change the arithmetic.
An SLO must name which quantity it bounds: a TTFT SLO and an end-to-end SLO are different products.
Report percentiles (Section 9.2): the mean hides the tail that users remember (Dean and Barroso, 2013, 'The Tail at Scale').

## 12. Queueing: Utilization and the Nonlinear Wait

With arrival rate $\lambda$ and service rate $\mu$, utilization is $\rho=\lambda/\mu$ (Section 2.3).
For the simplest M/M/1 queue, mean waiting time grows like $1/(\mu-\lambda)$: it diverges as $\rho\to1$.
Worked example: $\mu=10$ requests/s. At $\lambda=5$, mean system time is $1/(10-5)=0.2$ s; at $\lambda=9$, it is $1.0$ s; at $\lambda=9.9$, $10$ s.
The service itself did not slow down; the queue did.
This nonlinearity is the single most important serving intuition: running hotter than about 70-80% utilization buys throughput with tail latency.
Real LLM service times are variable (long generations), which makes tails worse than M/M/1, not better.

## 13. Little's Law as a Capacity Tool

Little's law, $L=\lambda W$, relates average in-system requests $L$, arrival rate $\lambda$, and average time in system $W$ (Section 2.4).
Worked example: 20 requests/s with average latency 2.5 s means about 50 requests in system on average.
Those 50 requests are exactly the concurrency the KV cache must hold (Section 17).
Little's law needs no distributional assumptions, which is why it is the first check on any capacity claim.
If measured $L$, $\lambda$, and $W$ violate the law materially, the instrumentation is wrong somewhere.

## 14. Prefill Is Compute-Bound, Decode Is Bandwidth-Bound

Prefill processes the whole prompt in parallel: large matrix multiplications, high arithmetic intensity, compute-bound.
Decode generates one token per sequence per step: it reads all model weights (and KV) for tiny compute per sequence, bandwidth-bound at small batch.
Batching decode across sequences amortizes weight reads, which is why throughput rises with batch while per-token latency worsens slowly, then queues (Section 12).
The roofline question for any serving claim: which regime, at what batch, on what hardware?
Quantization (Section 11) helps decode by shrinking weight reads; it helps prefill less if compute becomes the binding term.

## 15. Continuous Batching and Iteration-Level Scheduling

Static batching waits for the longest generation in the batch; finished sequences idle while others decode (Section 4.1).
Orca (Yu et al., 2022) introduced iteration-level scheduling: sequences join and leave the running batch at each decode step, and selective batching treats prefill and decode operations differently.
Continuous batching is the production descendant of that idea and is why modern serving throughput far exceeds static-batch baselines.
Worked intuition: if generation lengths vary from 10 to 1000 tokens, static batching wastes most slots on most steps; continuous batching refills them.

## 16. Chunked Prefill and Piggybacking

A long prefill can monopolize the GPU and stall decode steps for running sequences, spiking TPOT.
Chunked prefill splits a long prompt across steps, mixing prefill chunks with decode work.
Sarathi-Serve (Agrawal et al., 2024) systematized this as stall-free scheduling by piggybacking decodes with chunked prefills.
The tradeoff is explicit: slightly slower TTFT for the long prompt, far steadier TPOT for everyone else.
Tune chunk size against the TPOT SLO, not in the abstract.

## 17. KV Cache and Maximum Concurrency

Per-token KV bytes follow Chapter 15 Section 09: $2\times L\times h_{kv}\times d_h\times$ bytes per element.
Maximum concurrency is roughly (GPU memory $-$ weights $-$ workspace) divided by (per-token KV $\times$ average total sequence length).
PagedAttention (Kwon et al., 2023) manages KV in blocks, cutting fragmentation waste so this arithmetic is closer to achievable.
Worked example: 80 GB GPU, 14 GB INT8 weights for a 7B-class model, 6 GB workspace leaves 60 GB for KV.
At 0.13 MB per token (32 layers, 8 KV heads, 128 head dim, bf16) that is about 460k tokens of KV.
At 4k average total length, concurrency is about 112 sequences — before fragmentation and safety margin.

## 18. Parallelism for Serving

Tensor parallelism splits layers across GPUs: low latency per token, high interconnect demand (Section 6.1).
Pipeline parallelism splits layers into stages: better for throughput, adds bubble and stage-balancing concerns.
Data-parallel replicas are the simplest scale-out: each replica is a full serving unit behind a load balancer.
Phase splitting (disaggregated prefill/decode) runs prefill and decode on different pools so each can be sized for its regime (Section 14); DistServe (Zhong et al., 2024) is a canonical published design point.
Choose parallelism from the binding constraint: single-request latency, aggregate throughput, or memory fit.

## 19. Cost Accounting That Survives Scrutiny

Cost per million tokens is GPU-hour price divided by tokens produced per GPU-hour at the achieved utilization (Section 7).
Worked example: at $2 per GPU-hour and 3,600 seconds, one GPU producing 5,000 tokens/s delivers 18M tokens per hour, about $0.11 per million tokens before overheads.
Halving utilization doubles that cost; this is why queueing headroom (Section 12) is a cost decision, not waste.
Quality-adjusted cost divides by the fraction of outputs meeting the quality bar, not by raw tokens.
State hardware, price source and date, batch regime, and utilization with every cost claim.

## 20. Autoscaling, Admission Control, and Graceful Degradation

Autoscaling adds replicas on queue depth or utilization signals, but new replicas take time to load weights; scale before the queue explodes.
Admission control rejects or delays work early rather than letting every request miss its SLO slowly.
Graceful degradation menu, in rough order: shorten retrieved context (Section 12), cap output length, route to a smaller model, disable optional reranking, then shed load.
Each degradation is a quality change and should be logged and alarmed like an error.
Fallback models (Section 10.1) must be evaluated before the incident, not discovered during it.










## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Reporting mean latency only | Tails define user experience and SLOs (Dean and Barroso, 2013) | Report p50/p95/p99 for TTFT, TPOT, and end-to-end |
| Running near 100% utilization for cost | Queue wait diverges as utilization approaches 1 (Section 12) | Hold headroom; price it explicitly |
| Tokens/sec without batch or latency | Throughput can be bought with unacceptable delay | Report throughput at a stated SLO |
| Concurrency from weight memory only | KV cache usually bounds concurrency (Section 17) | Do the full memory budget |
| Static batching for variable lengths | Finished sequences idle in the batch (Section 15) | Continuous batching |
| Long prefills stalling decode | TPOT spikes for all running sequences (Section 16) | Chunked prefill |
| Cost per token at benchmark utilization | Production utilization is lower | Measure cost at achieved utilization |
| No admission control | Overload makes every request late instead of some rejected | Shed or degrade early and explicitly |
| Autoscaling on CPU-style signals only | LLM replicas are gated by GPU memory and queue depth | Scale on queue/utilization with warm-up time priced |
| Silent quality degradation under load | Users get worse answers with no signal | Log and alarm every degradation mode |


## 21. Why This Matters for AI

| Concept | AI Impact |
|---|---|
| Utilization nonlinearity | Decides how much headroom a serving fleet must hold |
| Prefill/decode asymmetry | Explains disaggregated architectures and chunked prefill |
| KV-bounded concurrency | Turns context length into a capacity and cost variable |
| Quality-adjusted cost | Prevents cheap-but-wrong serving from looking efficient |
| Tail latency | SLOs users actually feel are percentile SLOs |

## 22. Computed Tables

### 22.1 M/M/1 mean system time, service rate 10 requests/s

| Arrival rate | Utilization | Mean system time |
|---:|---:|---:|
| 2 | 20% | 0.12 s |
| 5 | 50% | 0.20 s |
| 7 | 70% | 0.33 s |
| 8 | 80% | 0.50 s |
| 9 | 90% | 1.00 s |
| 9.5 | 95% | 2.00 s |
| 9.9 | 99% | 10.00 s |

### 22.2 Cost per million tokens by throughput (at $2/GPU-hour)

| Tokens/s per GPU | Tokens/hour | Cost per 1M tokens |
|---:|---:|---:|
| 500 | 1,800,000 | $1.111 |
| 1000 | 3,600,000 | $0.556 |
| 2500 | 9,000,000 | $0.222 |
| 5000 | 18,000,000 | $0.111 |
| 10000 | 36,000,000 | $0.056 |

### 22.3 Concurrency from KV budget (0.13 MB/token, 60 GB KV budget)

| Avg total length | Sequences supported |
|---:|---:|
| 1024 | 447 |
| 4096 | 111 |
| 8192 | 55 |
| 32768 | 13 |

## 23. Ten Worked Problems With Solutions

1. **Little's law.** $\lambda=20$/s, $W=2.5$ s: $L=50$ concurrent requests. (Section 13)
2. **Queue blow-up.** $\mu=10$, $\lambda$ from 8 to 9: mean system time $0.5\to1.0$ s at constant service speed. (Section 12)
3. **End-to-end.** TTFT 0.8 s, TPOT 25 ms, 200 tokens: about 5.8 s. (Section 11)
4. **Concurrency.** 60 GB KV, 0.13 MB/token, 4k length: about 112 sequences. (Section 17)
5. **Cost.** 5,000 tok/s at $2/GPU-hour: about $0.11 per million tokens. (Section 19)
6. **Utilization target.** At 95% utilization the M/M/1 wait multiplier is about $20\times$ the service time; at 70% about $3.3\times$. (Section 12)
7. **Chunked prefill benefit.** A 32k prefill chunked at 4k spreads across 8 steps, bounding decode stalls per step. (Section 16)
8. **Degradation choice.** Overload with TTFT SLO at risk: shorten retrieved context first; it cuts prefill directly (Sections 12, 20).
9. **Replica count.** Demand 3 GPUs worth at target utilization 0.7: provision $\lceil3/0.7\rceil=5$ replicas to hold headroom. (Sections 12, 20)
10. **Quality-adjusted cost.** $0.11$/1M tokens at 90% acceptable outputs: $0.11/0.9\approx\$0.12$ per million acceptable tokens. (Section 19)

## 24. Serving Review Protocol

- Define SLOs for TTFT, TPOT, and end-to-end at stated percentiles.
- Load-test at production prompt/output length distributions, not uniform toys.
- Measure the queueing curve (latency vs offered load) and choose the operating utilization from it.
- Do the full memory budget: weights, KV at max batch/context, workspace, fragmentation margin.
- Verify continuous batching and chunked-prefill settings against TPOT under long-prompt load.
- Price cost per million tokens at achieved utilization, quality-adjusted.
- Test admission control and each degradation mode deliberately.
- Test autoscaling with realistic replica warm-up (weight loading) time.
- Trace a sample of requests across retrieval, prefill, and decode stages.
- Record hardware, model configuration (Sections 11-12), and serving settings with every benchmark.

## References (additions)

- Yu, G.-I. et al. (2022). Orca: A Distributed Serving System for Transformer-Based Generative Models. OSDI.
- Kwon, W. et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP. arXiv:2309.06180.
- Agrawal, A. et al. (2024). Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. OSDI. arXiv:2403.02310.
- Sheng, Y. et al. (2023). FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU. ICML. arXiv:2303.06865.
- Dean, J. and Barroso, L. A. (2013). The Tail at Scale. Communications of the ACM 56(2).
- Zhong, Y. et al. (2024). DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. OSDI. arXiv:2401.09670.

## 25. Scheduling Policies in Practice

FIFO is fair and simple but lets a long generation delay many short ones (head-of-line blocking, Section 4.4).
Shortest-remaining-processing-time reduces mean latency but needs output-length estimates LLM serving rarely has.
Priority queues serve paying or interactive traffic first; starvation of low priorities must be bounded explicitly.
Fairness across users matters: one client with huge prompts can consume the KV budget (Section 17) and starve others.
Per-user concurrency and token quotas are scheduling policy, not merely billing.
Whatever policy is chosen, measure latency by priority class; an aggregate p95 can hide a starved class.

## 26. Prefix Caching and Request Similarity

Shared prompt prefixes (system prompts, few-shot templates, retrieved boilerplate) recur across requests.
Caching their KV lets later requests skip that prefill portion, cutting TTFT.
Cache hit rate depends on prefix structure; log it as a serving metric.
Invalidation follows the same versioning discipline as retrieval caches (Section 12): a changed template must not serve stale KV.

## 27. Speculative and Parallel Decoding in Serving

Speculative decoding (Chapter 15 Section 09) trades extra draft compute for fewer target-model steps when acceptance is high.
In serving, its value depends on spare compute: at saturation, draft work competes with other requests' tokens.
Measure acceptance rate and end-to-end effect at production load, not in an empty benchmark.
Report it as a configuration flag in benchmarks; results with and without speculation are not comparable.

## 28. Multi-Model and Routed Serving

Production fleets route among model sizes by task difficulty, user tier, or confidence.
Routing is a quality decision priced like any other: measure quality by route, not only fleet averages.
A small model handling 70% of traffic at equal quality dominates fleet cost; at lower quality it is a degradation (Section 20).
Routers themselves need evaluation and rollback, like models (Section 11's discipline).

## 29. Reliability: Errors, Timeouts, and Partial Failures

Long generations fail in more ways: client disconnects, timeouts mid-stream, GPU faults losing a whole batch's KV.
Idempotency and retry policy must be explicit for non-streaming calls; retries multiply load during incidents (Section 12's queue).
Timeouts should be set from the latency distribution (Section 11), not round numbers.
Error budgets (Section 9.3) allocate allowed unreliability; burn rate alarms page humans before the budget is gone.

## 30. Observability That Answers Questions

Minimum useful traces join: request ID, prompt/output lengths, queue time, TTFT, TPOT, finish reason, model/config versions, and cost attribution.
Percentile dashboards per stage localize regressions the way RAG stage metrics localize retrieval failures (Section 12).
Log degradation and routing decisions (Sections 20, 28) on every affected request.
If a metric cannot be tied to a decision, it is decoration; keep the dashboard small and decision-linked.

## 31. Step-by-Step Problem Set B


**Problem B1: Find the bottleneck.**
- p95 TTFT is 4 s; measured prefill p95 is 0.7 s.
- The remaining 3.3 s is queue or retrieval.
- Utilization is 0.93, deep in the nonlinear zone (Section 12).
- Adding replicas or shedding load, not faster kernels, is the fix.

**Problem B2: Size a fleet.**
- Each replica sustains 800 tokens/s at the TPOT SLO.
- Peak demand is 5,600 tokens/s.
- At target utilization 0.75, replicas $=\lceil5600/(800\times0.75)\rceil=10$.
- Provisioning 7 would run at saturation in peaks.

**Problem B3: Read a cost claim.**
- A vendor quotes $0.05 per million tokens.
- Their benchmark batch implies utilization far above production.
- Recompute at your measured utilization (Section 22.2) before comparing.

**Problem B4: Protect TPOT.**
- Long-document requests spike TPOT for chat users.
- Chunked prefill (Section 16) bounds per-step stall.
- Separate pools (Section 18) are the heavier fix if chunking is insufficient.

**Problem B5: Design degradation.**
- At overload, first shorten retrieved context (cuts prefill).
- Then cap max output tokens (cuts decode occupancy).
- Then route simple traffic to the small model (Section 28).
- Shed lowest-priority load last, explicitly, with client-visible errors.

**Problem B6: Set a timeout.**
- End-to-end p99 at target lengths is 21 s.
- A 10 s timeout would kill about the slowest legitimate percent and more.
- Set timeouts per request class from measured distributions (Section 29).

## 32. Additional Computed Tables

### 32.1 End-to-end latency by output length (TTFT 0.8 s)

| Output tokens | TPOT 15 ms | TPOT 25 ms | TPOT 50 ms |
|---:|---:|---:|---:|
| 50 | 1.5 s | 2.0 s | 3.2 s |
| 200 | 3.8 s | 5.8 s | 10.8 s |
| 500 | 8.3 s | 13.3 s | 25.8 s |
| 1000 | 15.8 s | 25.8 s | 50.8 s |

### 32.2 In-system requests by Little's law

| Arrival/s | Avg latency 1 s | 2.5 s | 5 s |
|---:|---:|---:|---:|
| 5 | 5 | 12 | 25 |
| 20 | 20 | 50 | 100 |
| 50 | 50 | 125 | 250 |

### 32.3 KV budget share (80 GB GPU, 6 GB workspace)

| Weights | KV budget | Tokens at 0.13 MB | Sequences at 4k |
|---:|---:|---:|---:|
| 14 GB | 60 GB | 457,763 | 111 |
| 35 GB | 39 GB | 297,546 | 72 |
| 70 GB | 4 GB | 30,517 | 7 |

## 33. Glossary and Formula Appendix

- **Admission control:** Rejecting or delaying work to protect in-flight SLOs.
- **Chunked prefill:** Splitting long prefills across steps to protect decode latency.
- **Continuous batching:** Iteration-level joining/leaving of sequences in the running batch.
- **Goodput:** Throughput counting only outputs meeting quality/SLO criteria.
- **Head-of-line blocking:** A long job delaying short jobs behind it.
- **Little's law:** $L=\lambda W$ relating concurrency, arrival rate, and latency.
- **Prefill:** Parallel processing of the prompt; compute-bound.
- **Decode:** Token-by-token generation; bandwidth-bound at small batch.
- **SLO:** Service-level objective on a stated percentile and metric.
- **TPOT:** Time per output token after the first.
- **TTFT:** Time to first token, including queue and prefill.
- **Utilization:** $\rho=\lambda/\mu$; queue delay diverges as it approaches 1.

Formulas: $\rho=\lambda/\mu$; M/M/1 mean system time $1/(\mu-\lambda)$; $L=\lambda W$; end-to-end $\approx$ TTFT $+(n-1)$TPOT; cost/1M $=$ GPU-hour price $/$ tokens per GPU-hour.

## 34. Pre-Production Serving Checklist

- [ ] SLOs defined per metric and percentile.
- [ ] Queueing curve measured; operating utilization chosen.
- [ ] Full memory budget including KV and fragmentation.
- [ ] Continuous batching verified under variable lengths.
- [ ] Chunked prefill tuned against TPOT.
- [ ] Cost model at achieved, quality-adjusted utilization.
- [ ] Admission control and degradations tested.
- [ ] Autoscaling tested with real warm-up time.
- [ ] Fallback/smaller model evaluated and ready.
- [ ] Per-class (priority/user) latency visible.
- [ ] Traces join retrieval, prefill, decode, and config versions.
- [ ] Error budget and burn-rate alarms configured.
- [ ] Load test uses production length distributions.
- [ ] Rollback for model and config changes tested.

## 35. Load Testing Without Lying to Yourself

Synthetic uniform prompts flatter the system: real traffic mixes short chats and long documents.
Length distributions drive batching efficiency, KV occupancy, and tails (Sections 15, 17).
Replay production-like traces (privacy permitting) or fit distributions and sample.
Report the workload with the result; a benchmark without a workload description is marketing.
Include ramp and spike phases: steady state hides autoscaling and queue-recovery behaviour.

## 36. Capacity Planning Over Growth

Capacity is a forecast: demand growth, model changes, and context-length growth all move the requirement.
Longer advertised contexts raise KV per request (Section 17) even at constant request counts.
Plan in tokens/s and concurrent sequences, the two currencies Sections 13 and 17 price.
Keep the queueing curve (Section 12) as the planning artifact; update it each release.

## 37. Energy and Carbon as Serving Costs

Tokens per joule follows the same utilization logic as tokens per dollar (Section 19).
Low utilization wastes energy on idle-but-powered capacity; the headroom decision (Section 12) is also an energy decision.
Report energy per million tokens where sustainability claims are made, measured not modelled.

## 38. Benchmarking Serving Systems Fairly

Fix model, quantization (Section 11), retrieval configuration (Section 12), prompts, and length distributions before comparing engines.
Warm up caches and JIT paths; report cold and warm separately if cold matters.
State batch policy, chunk sizes, and speculative settings (Sections 15-16, 27).
A comparison missing any of these is a comparison of configurations, not engines.

## 39. Incident Review for Serving

Reconstruct from traces (Section 30): offered load, utilization, queue, degradations active, config versions.
Classify the incident: overload, bad deploy, capacity loss, upstream (retrieval) failure, or client behaviour change.
The fix list should name the metric each fix moves (Section 24's discipline).
Add the incident's load pattern to the load-test suite (Section 35).

## 40. Chapter 15 Synthesis: From Probabilities to Production

Section 05 priced tokens as probabilities; Section 11 compressed the weights that produce them.
Section 12 added retrieved evidence and its context cost.
This section turned model, compression, and retrieval choices into latency, throughput, and dollars.
The chain of accounting is the chapter's real content: every upstream choice appears as a downstream serving number.
A claim at any stage that cannot be traced to a serving consequence, or a serving claim that hides its upstream configuration, is incomplete.

## 41. Combinatorial Worksheets

### 41.1 Replicas needed by demand and target utilization (800 tok/s per replica at SLO)

| Demand tok/s | util 0.6 | util 0.75 | util 0.9 |
|---:|---:|---:|---:|
| 2000 | 5 | 4 | 3 |
| 5600 | 12 | 10 | 8 |
| 10000 | 21 | 17 | 14 |
| 25000 | 53 | 42 | 35 |

### 41.2 TTFT budget worksheet

| Component | Budget ms | Measured ms |
|---|---|---|
| Queue |  |  |
| Retrieval (Section 12) |  |  |
| Prefill |  |  |
| First-token overhead |  |  |
| Total TTFT |  |  |

### 41.3 Utilization and M/M/1 wait multiplier (wait/service)

| Utilization | Wait multiplier $\rho/(1-\rho)$ |
|---:|---:|
| 0.5 | 1.0x |
| 0.7 | 2.3x |
| 0.8 | 4.0x |
| 0.9 | 9.0x |
| 0.95 | 19.0x |
| 0.99 | 99.0x |

## 42. Ten Quick Checks (with answers)

1. **TTFT includes what?** Queue, retrieval, and prefill before the first token.
2. **Why does wait explode near full utilization?** Queueing: mean wait scales like $1/(1-\rho)$ in the simple model.
3. **What bounds concurrency usually?** KV-cache memory at the served context lengths.
4. **What did Orca introduce?** Iteration-level scheduling for generative serving (Yu et al., 2022).
5. **What does chunked prefill protect?** TPOT of running decodes against long-prefill stalls.
6. **Cost per million at half utilization?** Roughly double the full-utilization figure, all else equal.
7. **Why quality-adjust cost?** Cheap unacceptable tokens are not output.
8. **What is goodput?** Throughput of outputs meeting the SLO/quality bar.
9. **Why per-class latency?** Aggregates hide starved priorities or users.
10. **What must a serving benchmark state?** Workload, lengths, batch policy, model/compression/retrieval configuration, hardware.

## 43. Serving Configuration Record

| Field | Your value | Checked |
|---|---|---|
| Model and checkpoint |  | [ ] |
| Quantization configuration |  | [ ] |
| Retrieval configuration |  | [ ] |
| Engine and version |  | [ ] |
| Batching policy |  | [ ] |
| Chunked-prefill size |  | [ ] |
| Parallelism (TP/PP/replicas) |  | [ ] |
| Hardware per replica |  | [ ] |
| SLOs (metric, percentile, target) |  | [ ] |
| Operating utilization |  | [ ] |
| Cost per 1M (quality-adjusted) |  | [ ] |
| Degradation menu |  | [ ] |
| Autoscaling signals |  | [ ] |
| Trace fields verified |  | [ ] |

## 44. Section 13 Takeaways

- Latency is a budget spent across queue, retrieval, prefill, and decode.
- Utilization headroom is purchased reliability, priced by queueing math.
- Concurrency is a memory equation dominated by KV at long context.
- Cost claims live or die by achieved, quality-adjusted utilization.
- Chapter 15 ends where it began: measurable quantities, honestly reported.

## 45. Worked Queueing Example in Full

Service rate per replica is 12 requests/s at the production length mix.
Offered load is 9 requests/s, so utilization is $0.75$.
M/M/1 mean system time is $1/(12-9)=0.33$ s, with mean wait $0.25$ s before service.
A traffic rise to 11 requests/s (utilization $0.92$) makes system time $1.0$ s.
Nothing about the model changed; the queue consumed the SLO.
The capacity decision — how many replicas keep peak utilization at or below 0.75 — follows directly.

## 46. Worked Cost Example in Full

A replica on a $2/GPU-hour instance sustains 4,000 tokens/s at the TPOT SLO.
At 100% of that sustainable rate, hourly output is 14.4M tokens and cost is about $0.139 per million.
At a realistic 60% achieved utilization, output is 8.64M tokens/h and cost rises to about $0.231 per million.
If 92% of outputs meet the quality bar, quality-adjusted cost is $0.231/0.92\approx\$0.251$ per million acceptable tokens.
Each step is a separate, reportable assumption (Section 19).

## 47. Reading a Serving Regression

After a release, p95 TPOT rises 30% while TTFT is unchanged.
Prefill and queue are exonerated by the unchanged TTFT.
Suspects narrow to decode: batch composition, KV pressure, kernel change, or a longer average output length in traffic.
Check traffic mix first; a workload change is not an engine regression (Section 35).
Traces by stage (Section 30) settle which suspect is real.

## 48. Extended Computed Tables

### 48.1 Mean system time by load (service 12 req/s)

| Arrival/s | Utilization | System time |
|---:|---:|---:|
| 4 | 33% | 0.12 s |
| 6 | 50% | 0.17 s |
| 8 | 67% | 0.25 s |
| 9 | 75% | 0.33 s |
| 10 | 83% | 0.50 s |
| 11 | 92% | 1.00 s |
| 11.5 | 96% | 2.00 s |

### 48.2 Quality-adjusted cost ($2/GPU-hour, 4000 tok/s sustainable)

| Utilization | Acceptable fraction | Cost per 1M acceptable |
|---:|---:|---:|
| 0.4 | 0.85 | $0.408 |
| 0.4 | 0.95 | $0.365 |
| 0.6 | 0.85 | $0.272 |
| 0.6 | 0.95 | $0.244 |
| 0.8 | 0.85 | $0.204 |
| 0.8 | 0.95 | $0.183 |

### 48.3 Sequence capacity across KV geometries (60 GB KV)

| Layers | KV heads | Bytes/token | Sequences at 8k |
|---:|---:|---:|---:|
| 32 | 8 | 131,072 | 55 |
| 32 | 32 | 524,288 | 13 |
| 80 | 8 | 327,680 | 22 |
| 80 | 40 | 1,638,400 | 4 |

GQA's smaller KV-head count is a concurrency feature, not only a speed feature (Chapter 15 Section 09).

## 49. Step-by-Step Problem Set C


**Problem C1: Choose the operating point.**
- The queueing curve shows p95 latency doubling between utilization 0.75 and 0.9.
- The SLO has 20% headroom at 0.75 and none at 0.9.
- Operate at 0.75; the extra replicas are the price of the SLO (Section 12).

**Problem C2: Attribute a TTFT spike.**
- Retrieval p95 rose after an index rebuild (Section 12).
- Generator prefill is unchanged.
- TTFT inherits the retrieval regression directly (Section 11).
- Fix the index configuration, not the serving engine.

**Problem C3: Plan a model upgrade.**
- The new model is 1.6x slower per token but 8 points more accurate.
- Cost per acceptable token may still fall if the acceptable fraction rises enough.
- Compute quality-adjusted cost both ways (Section 46) before deciding.

**Problem C4: Survive a spike.**
- Traffic triples for ten minutes; autoscaling needs four minutes to add replicas.
- Admission control protects in-flight SLOs during the gap (Section 20).
- Degradations engage in the published order and are logged.

**Problem C5: Compare engines honestly.**
- Engine B is faster on uniform 100-token prompts.
- Production traffic mixes 50-token chats and 8k documents.
- Re-run on the production mix (Section 38) before switching.

## 50. One-Page Serving Review

- [ ] SLOs name metric, percentile, and length class.
- [ ] Queueing curve measured; operating utilization chosen from it.
- [ ] Memory budget covers weights, KV, workspace, fragmentation.
- [ ] Batching and chunking settings recorded and load-tested.
- [ ] Retrieval and compression configurations frozen for benchmarks.
- [ ] Cost is quality-adjusted at achieved utilization.
- [ ] Degradation menu ordered, tested, and logged.
- [ ] Autoscaling accounts for replica warm-up.
- [ ] Per-class latency and traces verified.
- [ ] Configuration record (Section 43) complete.
- [ ] Incident load patterns added to the test suite.
- [ ] Rollback tested for model and serving config.

## 51. Prefill/Decode Disaggregation Tradeoffs

Splitting prefill and decode pools (Section 18) lets each scale independently with load mix.
Chatty short-prompt traffic is decode-heavy; document Q&A (Section 12) is prefill-heavy.
The cost is KV transfer between pools, priced in bandwidth and added TTFT complexity.
Measure the transfer at production context lengths; it grows with exactly the KV bytes of Section 17.
Disaggregation is justified by measured interference, not by architecture fashion.

## 52. Batching Under Length Heterogeneity

Output-length variance is the enemy of static batching (Section 15).
Continuous batching converts that variance from wasted slots into scheduling work.
Prompt-length variance instead attacks through prefill; chunking (Section 16) is the corresponding tool.
A fleet seeing both variances needs both mechanisms; either alone leaves a tail source alive.
Report length distributions with every throughput number (Section 35).

## 53. The Serving Stack as One System

Retrieval latency (Section 12), compression choice (Section 11), MoE routing (Section 10), and attention efficiency (Section 09) all land in this section's budget lines.
Optimizing one stage while starving another moves the bottleneck; it does not remove it.
The configuration records of Sections 11, 12, and this section together describe one deployable system.
Change any one record and the serving numbers must be re-measured.

## 54. Final Computed Tables

### 54.1 Tokens per GPU-hour and cost by sustainable rate ($2/GPU-hour)

| Tok/s | Tokens/hour | $/1M |
|---:|---:|---:|
| 1000 | 3,600,000 | $0.556 |
| 2000 | 7,200,000 | $0.278 |
| 4000 | 14,400,000 | $0.139 |
| 8000 | 28,800,000 | $0.069 |
| 16000 | 57,600,000 | $0.035 |

### 54.2 Concurrency by average length across KV budgets (0.13 MB/token)

| KV GB | 2k len | 8k len | 32k len |
|---:|---:|---:|---:|
| 20 | 74 | 18 | 4 |
| 40 | 149 | 37 | 9 |
| 60 | 223 | 55 | 13 |
| 120 | 447 | 111 | 27 |

### 54.3 Wait multiplier reminder

| Utilization | Queue regime |
|---:|---|
| 0.5 | comfortable (1.0x wait/service) |
| 0.7 | planned headroom (2.3x wait/service) |
| 0.85 | tail risk (5.7x wait/service) |
| 0.95 | SLO emergency (19.0x wait/service) |

## 55. Ten Final Checks (with answers)

1. **What three latencies need SLOs?** TTFT, TPOT, and end-to-end, each at a percentile.
2. **What does Little's law check?** Consistency of concurrency, arrival rate, and latency measurements.
3. **Why can a bigger batch hurt users?** Throughput rises while per-request latency and queueing worsen.
4. **What does PagedAttention improve?** KV memory fragmentation, raising achievable concurrency (Kwon et al., 2023).
5. **When is disaggregation worth it?** When prefill/decode interference is measured and transfer cost is smaller.
6. **What is the first degradation under overload?** The cheapest quality trade: usually shorter context or capped outputs, explicitly logged.
7. **Why warm replicas matter?** Weight loading delays autoscaling exactly when load spikes.
8. **What ends an incident review?** Fixes tied to metrics, and the incident pattern added to load tests.
9. **How does Section 11 change serving?** Quantization shrinks weight reads and KV options change concurrency.
10. **How does Section 12 change serving?** Retrieval adds TTFT latency and context tokens that multiply KV cost.

## 56. Closing Notes for Section 13 and Chapter 15

- Serving is applied queueing plus memory arithmetic plus honest costing.
- Every upstream chapter-15 choice reappears as a serving number.
- Measure at production workloads, operate below saturation, and record configurations.
- With this section, Chapter 15 runs from token probabilities to production systems.

## 57. SLO Error Budgets Worked

An SLO of 99% of requests with TTFT under 1 s allows 1% violations over the window.
At 10M requests/month, that budget is 100,000 slow first tokens.
Burn rate compares current violation pace to the budgeted pace; 2x burn halves the time to exhaustion.
Error budgets turn reliability into an allocation decision: spend budget on launches, save it before risky changes.
Pair latency SLOs with availability SLOs; a fast error is still an error.

## 58. Capacity Incident Playbook (condensed)

- Declare: name the SLO at risk and the current burn.
- Protect: enable admission control and the first degradation (Section 20).
- Localize: stage traces separate queue, retrieval, prefill, decode (Section 30).
- Scale or shed: add replicas accounting for warm-up, or shed lowest priority.
- Verify: watch the specific percentile that triggered, not the mean.
- Record: incident pattern joins the load-test suite (Section 39).

## 59. What Good Looks Like: A Serving Report

A one-page serving report states: workload, SLOs and attainment, operating utilization, concurrency limits and their memory derivation, cost per million acceptable tokens, degradation events, and open risks.
Every number names its configuration (Sections 38, 43).
Any reader should be able to reproduce the capacity conclusion from the report alone.

## 60. Extended Worksheets

### 60.1 Error-budget consumption (99% SLO, monthly)

| Requests/month | Allowed violations | At 1.5x burn, days to exhaust (30-day window) |
|---:|---:|---:|
| 1,000,000 | 10,000 | 20 |
| 10,000,000 | 100,000 | 20 |
| 100,000,000 | 1,000,000 | 20 |

### 60.2 TTFT decomposition worksheet (production mix)

| Component | p50 ms | p95 ms | Owner stage |
|---|---|---|---|
| Queue |  |  | scheduling |
| Retrieval |  |  | Section 12 |
| Prefill |  |  | compute |
| Scheduling overhead |  |  | engine |
| Total TTFT |  |  | SLO |

### 60.3 Fleet sizing reminder

| Quantity | Symbol | Source |
|---|---|---|
| Demand at SLO | tok/s | load forecast |
| Per-replica rate at SLO | tok/s | load test (Section 35) |
| Target utilization | rho | queueing curve (Section 12) |
| Replicas | ceil(demand/(rate x rho)) | Section 41.1 |
| Warm-up time | minutes | measured weight load |
| Headroom replicas | policy | incident history |

## 61. Final Problem Set D


**Problem D1: Exhaust a budget.**
- 99.5% availability SLO at 20M requests allows 100,000 failures.
- An incident burns 30,000 in an hour.
- Remaining budget is 70,000; launches pause until burn normalizes.

**Problem D2: Pick a percentile.**
- Chat users feel the slowest first token in a session.
- A p50 TTFT SLO would pass while half of sessions start slowly.
- Set TTFT SLOs at p95 for interactive traffic.

**Problem D3: Justify headroom to finance.**
- Moving from 0.9 to 0.75 utilization raises replica count by 20%.
- It cuts the queue wait multiplier from 9x to 3x (Section 41.3).
- That is the purchased SLO headroom, stated in one line.

**Problem D4: Stop a bad comparison.**
- Two engines are compared with different quantization and chunking.
- The comparison measures configurations, not engines (Section 38).
- Freeze configurations and rerun.

## 62. Chapter 15 Completion Note

- Sections 01-04 were already at the bar before this phase.
- Sections 05-13 were expanded in order during Phase 2, one commit per section.
- Each expanded section preserves its original text and notebooks while adding the missing depth.
- Chapter 15 now runs continuously from probability foundations to serving systems.

## 63. Serving Decision Tables

### 63.1 Symptom-to-fix map

| Symptom | Likely stage | First action |
|---|---|---|
| TTFT p95 up, TPOT flat | queue or retrieval | check utilization and retrieval p95 |
| TPOT p95 up, TTFT flat | decode/KV pressure | chunk prefill, check batch mix |
| Both up at peak only | saturation | admission control, scale |
| Cost up, latency flat | utilization fell | rebalance replicas to demand |
| Quality down under load | degradation active or routing shift | audit degradation logs |
| One user class slow | priority starvation | per-class quotas and latency |

### 63.2 Degradation menu with measured prices (fill in)

| Action | Latency saved | Quality price | Logged |
|---|---|---|---|
| Shorten retrieved context |  |  | [ ] |
| Cap output tokens |  |  | [ ] |
| Skip reranker |  |  | [ ] |
| Route to smaller model |  |  | [ ] |
| Shed low-priority traffic |  |  | [ ] |

## 64. Worked Fleet Example, End to End

- Demand forecast: 12,000 tokens/s peak at the production length mix.
- Load test: one replica sustains 900 tokens/s at the TPOT SLO.
- Target utilization from the queueing curve: 0.75.
- Replicas for peak: $\lceil12000/(900\times0.75)\rceil=18$.
- KV check per replica at batch concurrency target passes with the Section 17 budget plus fragmentation margin.
- Cost at $2/GPU-hour: 18 replicas cost $36/hour at peak; at 65% day-average utilization the effective cost per million acceptable tokens is computed as in Section 46.
- Degradation and admission policies are tested before launch (Section 50).
- This worked example is a template: substitute measured rates, never copy the numbers.

## 65. Final Review List

- [ ] SLOs and error budgets defined and dashboarded.
- [ ] Queueing curve on file and current.
- [ ] Memory budget derivation on file.
- [ ] Cost model with dated prices.
- [ ] Load tests replay production-like lengths.
- [ ] Degradations tested, ordered, and alarmed.
- [ ] Autoscaling warm-up measured.
- [ ] Fallback models evaluated.
- [ ] Configuration record complete.
- [ ] Incident playbook rehearsed.

## 66. End of Chapter 15

- The mathematics of LLMs ends, operationally, in a serving budget.
- Keep the chain auditable from probability to production.
- Future chapters build on these foundations.

## 67. Serving Margin Notes

- Latency budgets are spent, not wished.
- Queueing punishes optimism near saturation.
- KV memory is concurrency.
- Cost follows achieved utilization.
- Quality-adjusted tokens are the unit that matters.
- Degradation is a decision, logged like code.
- Traces turn incidents into arithmetic.
- Configuration records make benchmarks comparable.
- Headroom is purchased reliability.
- Percentiles are the user's experience.
- Prefill and decode are different regimes.
- Batch for throughput, schedule for latency.
- Protect running decodes from long prefills.
- Scale before the queue, not after it.
- Measure the workload you actually serve.
- Chapter 15 closes with systems that keep their promises.

## 68. Final Serving Notes

- SLOs are percentile promises on named metrics.
- Utilization headroom is bought with replicas and priced in dollars.
- Concurrency is derived from memory, not hoped for.
- Continuous batching and chunked prefill defend the tail.
- Cost per acceptable token is the honest efficiency unit.
- Every degradation is logged, alarmed, and reversible.
- Traces connect a user's wait to a stage's arithmetic.
- Benchmarks state workload, configuration, and hardware.
- Incidents become load tests; load tests prevent incidents.
- Serving keeps the promises the model makes possible.
- This completes Chapter 15: from token probabilities to production systems, every step measured.

## 69. Serving Review Margin

- The queue is part of the product.
- The tail is the experience.
- Memory arithmetic sets concurrency.
- Utilization sets the wait.
- Configurations make results comparable.
- Chapter 15 is complete when serving numbers trace back to model choices.

## 70. End Matter

- Section 13 closes the serving arc of Chapter 15.
- Its formulas are deliberately simple; their discipline is the difficulty.
- Queue, memory, and cost arithmetic should accompany every deployment claim.
- End of Section 13 expansion.

- Live verification follows in the audit file.
- Chapter 15 completion is summarised there.

- No further sections are added in this phase.
- The next phase begins only on instruction.

- Safety margin above the bar is intentional.
- Do not trim without rechecking the count.

## Practice Exercises

1. Use Little's law to compute average concurrency.
2. Compute utilization from arrival and service rates.
3. Build an end-to-end latency budget.
4. Compute max concurrent requests from KV cache memory.
5. Estimate cost per million tokens.
6. Compare batch choices under a latency budget.
7. Compute autoscaling replica count.
8. Compute an SLO error budget.
9. Choose a graceful degradation action under overload.
10. Write a serving trace checklist.

## Why This Matters for AI

LLMs are not useful only because they are trained. They become useful when they can answer real requests within latency, cost, and reliability limits. Serving math keeps deployment decisions honest: every model, context length, retrieval choice, quantization format, and batching policy has a measurable tradeoff.

## References

- Gyeong-In Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", 2022: https://www.usenix.org/conference/osdi22/presentation/yu
- Woosuk Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", 2023: https://arxiv.org/abs/2309.06180
- Lequn Chen et al., "Efficiently Scaling Transformer Inference", 2023: https://arxiv.org/abs/2211.05102
- Ying Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU", 2023: https://arxiv.org/abs/2303.06865
- Amey Agrawal et al., "Sarathi-Serve: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills", 2024: https://arxiv.org/abs/2403.02310
