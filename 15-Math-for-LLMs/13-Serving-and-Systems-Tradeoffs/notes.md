# Serving and Systems Tradeoffs

[← RAG Math and Retrieval](../12-RAG-Math-and-Retrieval/notes.md) | [Home](../../README.md)

---

## Table of Contents

1. [Intuition: What Makes LLM Serving Hard](#1-intuition-what-makes-llm-serving-hard)
2. [Formal Definitions and Metrics](#2-formal-definitions-and-metrics)
3. [Memory Hierarchy and Bandwidth](#3-memory-hierarchy-and-bandwidth)
4. [Batching Strategies](#4-batching-strategies)
5. [Scheduling Algorithms](#5-scheduling-algorithms)
6. [KV Cache Management](#6-kv-cache-management)
7. [Multi-GPU and Multi-Node Serving](#7-multi-gpu-and-multi-node-serving)
8. [Speculative Decoding Systems](#8-speculative-decoding-systems)
9. [Inference Optimisation](#9-inference-optimisation)
10. [Cost Modelling and Economics](#10-cost-modelling-and-economics)
11. [Distributed Systems for Serving](#11-distributed-systems-for-serving)
12. [Hardware Selection](#12-hardware-selection)
13. [Serving Frameworks](#13-serving-frameworks)
14. [Advanced Serving Patterns](#14-advanced-serving-patterns)
15. [Emerging Architectures](#15-emerging-architectures)
16. [Common Mistakes and Pitfalls](#16-common-mistakes-and-pitfalls)
17. [Exercises](#17-exercises)
18. [Why This Matters](#18-why-this-matters)

---

## 1. Intuition: What Makes LLM Serving Hard

### What Is "Serving" and Why Is It Different from Training?

Training processes a _fixed_ dataset in large batches over hours or days — throughput is king.
Serving answers _live_ user requests with strict latency constraints — the regime is entirely different:

| Dimension       | Training                        | Serving                        |
| --------------- | ------------------------------- | ------------------------------ |
| Objective       | Maximise throughput (tokens/s)  | Minimise latency _per request_ |
| Batch control   | You choose batch size           | Users arrive stochastically    |
| Sequence length | Fixed or bucketed               | Unpredictable output length    |
| Memory pattern  | Gradient + activations dominate | KV cache dominates             |
| Failure cost    | Re-run the batch                | User sees an error / timeout   |
| Hardware util.  | Near 100 % MFU achievable       | Typically 30–60 % MFU          |

### The Fundamental Tension

Every serving system must navigate three competing objectives:

```
                    Latency
                   ╱       ╲
                  ╱  Choose  ╲
                 ╱    Two     ╲
                ╱               ╲
          Cost ─────────────── Throughput
```

**Pick any two:**

- **Low latency + high throughput** → Expensive (many GPUs, low utilisation).
- **Low latency + low cost** → Low throughput (small batches, one user at a time).
- **High throughput + low cost** → High latency (large batches, users wait).

The entire chapter quantifies how to push this Pareto frontier.

### Why Standard Web-Serving Intuition Breaks

Traditional web APIs serve _stateless_ requests that take microseconds.
LLM inference is **stateful** (KV cache persists across tokens), **autoregressive**
(each token depends on the previous one), and **memory-bound** during decode
(arithmetic intensity ≈ 1 FLOP/byte).

### Who Cares?

| Stakeholder     | Primary metric                | Why?                          |
| --------------- | ----------------------------- | ----------------------------- |
| End user        | Time-to-first-token (TTFT)    | Perceived responsiveness      |
| Product manager | P99 end-to-end latency        | SLA compliance                |
| ML engineer     | Throughput (tok/s)            | Model iteration speed         |
| CFO             | Cost per million tokens (CPM) | Unit economics of the product |
| Ops engineer    | GPU utilisation / uptime      | Fleet efficiency              |

### Timeline: Serving Innovation 2020–2026

```
2020   FasterTransformer (NVIDIA)       — fused kernels, first optimised decoder
2022   ORCA (continuous batching)       — broke the static-batch paradigm
2022   FlashAttention v1                — IO-aware exact attention
       FasterTransformer + Triton Int8  — quantised serving at scale
2023   vLLM / PagedAttention            — virtual-memory KV cache
       TensorRT-LLM GA                  — NVIDIA's production engine
       FlashAttention v2                — 2× faster, broader GPU support
       Medusa / SpecInfer               — speculative decoding systems
2024   SGLang / RadixAttention          — prefix-tree KV sharing
       FlashAttention v3 (Hopper)       — H100 warp specialisation
       DeepSeek-V2 MLA                  — latent KV compression
       Chunked prefill (Sarathi-Serve)  — prefill-decode disaggregation
2025   FlashMLA / FlashInfer            — batched paged MLA kernels
       KV disaggregation (Mooncake)     — CPU-DRAM-SSD tiered KV
       Hybrid SSM-attention models      — sub-quadratic serving
2026+  Photonic / neuromorphic accel.   — research frontier
```

### The Serving Stack

```
┌─────────────────────────────────────────────┐
│              Client / API Gateway           │
├─────────────────────────────────────────────┤
│           Load Balancer / Router            │
├─────────────────────────────────────────────┤
│         Scheduler & Request Queue           │
├─────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Worker 0│  │ Worker 1│  │ Worker N│    │
│  │ ┌─────┐ │  │ ┌─────┐ │  │ ┌─────┐ │    │
│  │ │Model│ │  │ │Model│ │  │ │Model│ │    │
│  │ │ + KV│ │  │ │ + KV│ │  │ │ + KV│ │    │
│  │ │Cache│ │  │ │Cache│ │  │ │Cache│ │    │
│  │ └─────┘ │  │ └─────┘ │  │ └─────┘ │    │
│  └─────────┘  └─────────┘  └─────────┘    │
├─────────────────────────────────────────────┤
│       GPU Cluster / Hardware Layer          │
└─────────────────────────────────────────────┘
```

---

## 2. Formal Definitions and Metrics

### 2.1 Latency Metrics

**Time-to-First-Token (TTFT):** The wall-clock time from when a request arrives to
when the first output token is generated.

$$\text{TTFT} = t_{\text{queue}} + t_{\text{prefill}}$$

where $t_{\text{queue}}$ is queuing delay and $t_{\text{prefill}}$ is the time to
process all prompt tokens through the model.

**Time-Per-Output-Token (TPOT):** The average inter-token latency during the decode
(autoregressive) phase.

$$\text{TPOT} = \frac{t_{\text{decode}}}{n_{\text{output}} - 1}$$

**End-to-End Latency (E2E):**

$$T_{\text{E2E}} = \text{TTFT} + (n_{\text{output}} - 1) \times \text{TPOT}$$

> **Example:** LLaMA-3 8B on H100. Prompt = 512 tokens, output = 128 tokens.
> TTFT = 45 ms, TPOT = 12 ms.
> $T_{\text{E2E}} = 45 + 127 \times 12 = 1{,}569$ ms ≈ 1.6 s.

### 2.2 Throughput Metrics

**Throughput (tokens/s):** Total tokens generated across all requests per second.

$$\Theta = \frac{\sum_{i} n_{\text{output},i}}{T_{\text{wall}}}$$

**Goodput:** Throughput of _useful_ tokens — excludes wasted speculative drafts, padding, and tokens from requests that were preempted or timed out.

$$\Theta_{\text{good}} = \frac{\sum_{i \in \text{completed}} n_{\text{output},i}}{T_{\text{wall}}}$$

### 2.3 GPU Utilisation

**Model FLOPs Utilisation (MFU):**

$$\text{MFU} = \frac{\text{Achieved FLOP/s}}{\text{Peak FLOP/s of hardware}}$$

| Phase                 | Typical MFU | Bottleneck       |
| --------------------- | ----------- | ---------------- |
| Prefill               | 40–70 %     | Compute-bound    |
| Decode                | 1–5 %       | Memory-bandwidth |
| Batched decode (B=64) | 15–30 %     | Transitioning    |

**Model Bandwidth Utilisation (MBU):**

$$\text{MBU} = \frac{\text{Achieved bandwidth (GB/s)}}{\text{Peak memory bandwidth}}$$

During decode at small batch sizes, MBU ≈ 80–95 % while MFU ≈ 1–5 %. The GPU is
_busy_ moving data, not computing — this is the memory-wall problem.

### 2.4 Cost Per Million Tokens (CPM)

$$\text{CPM} = \frac{\text{GPU-hour cost} \times 10^6}{\Theta \times 3600}$$

> **Example:** H100 at \$3.00/hr, throughput = 8 000 tok/s.
> CPM = $\frac{3.00 \times 10^6}{8000 \times 3600} = \$0.104$ per million output tokens.

### 2.5 SLOs and SLAs

A **Service-Level Objective (SLO)** is a target metric:

| SLO          | Typical Target |
| ------------ | -------------- |
| TTFT P50     | < 200 ms       |
| TTFT P99     | < 500 ms       |
| TPOT P50     | < 30 ms        |
| E2E P99      | < 5 s          |
| Availability | 99.9 %         |
| Error rate   | < 0.1 %        |

A **Service-Level Agreement (SLA)** is a contractual commitment with penalties:

$$\text{SLA penalty} = \max\!\bigl(0,\; \alpha \cdot (p_{99}^{\text{observed}} - p_{99}^{\text{target}})\bigr)$$

### 2.6 Queuing Theory Foundations

**M/M/1 Queue** (single server, Poisson arrivals, exponential service):

| Symbol                 | Meaning                     |
| ---------------------- | --------------------------- |
| $\lambda$              | Arrival rate (requests / s) |
| $\mu$                  | Service rate (requests / s) |
| $\rho = \lambda / \mu$ | Server utilisation          |

Key results (stable when $\rho < 1$):

$$W_q = \frac{\rho}{\mu(1 - \rho)} \qquad L_q = \frac{\rho^2}{1 - \rho}$$

where $W_q$ is mean waiting time and $L_q$ is mean queue length.

**M/M/c Queue** (c identical servers):

$$P_0 = \left[\sum_{k=0}^{c-1}\frac{(c\rho)^k}{k!} + \frac{(c\rho)^c}{c!(1-\rho)}\right]^{-1}$$

$$W_q = \frac{P_0 (c\rho)^c}{c!\,c\mu(1-\rho)^2}$$

> **Why this matters:** With $c = 4$ GPUs, $\lambda = 10$ req/s, $\mu = 3$ req/s per GPU:
> $\rho = 10 / (4 \times 3) = 0.833$. Plugging in gives $W_q \approx 0.17$ s.
> Dropping to $c = 3$ gives $\rho = 1.11 > 1$ → queue is **unstable**.

---

## 3. Memory Hierarchy and Bandwidth

### 3.1 GPU Memory Hierarchy

Modern GPU serving is dominated by the speed of moving bytes, not computing FLOPs.

```
┌─────────────────────────────────────────────────────────────────┐
│  Level         │ Capacity        │ Bandwidth       │ Latency    │
├─────────────────────────────────────────────────────────────────┤
│  Registers     │  ~256 KB/SM     │  ∞ (local)      │  0 cycles  │
│  Shared/L1     │  228 KB/SM      │  ~33 TB/s*      │  ~20 ns    │
│  L2 Cache      │  50 MB          │  ~12 TB/s       │  ~200 ns   │
│  HBM3e (H100)  │  80 GB          │  3.35 TB/s      │  ~400 ns   │
│  NVLink 4.0    │  (inter-GPU)    │  900 GB/s       │  ~1 µs     │
│  PCIe Gen5     │  (CPU↔GPU)      │  128 GB/s bidi  │  ~2 µs     │
│  CPU DRAM      │  ~2 TB          │  ~400 GB/s      │  ~80 ns    │
│  NVMe SSD      │  ~30 TB         │  ~14 GB/s       │  ~10 µs    │
│  Network (IB)  │  (inter-node)   │  400 Gb/s       │  ~5 µs     │
└─────────────────────────────────────────────────────────────────┘
  * aggregate across all SMs
```

### 3.2 Interconnect Technologies

**NVLink** connects GPUs within a node:

| Generation        | Per-link BW | Links/GPU | Total BW/GPU |
| ----------------- | ----------- | --------- | ------------ |
| NVLink 3.0 (A100) | 50 GB/s     | 12        | 600 GB/s     |
| NVLink 4.0 (H100) | 50 GB/s     | 18        | 900 GB/s     |
| NVLink 5.0 (B200) | 100 GB/s    | 18        | 1,800 GB/s   |

**NVSwitch** provides all-to-all GPU connectivity within a node (DGX):

$$\text{Bisection BW} = \frac{N_{\text{GPU}} \times \text{BW}_{\text{NVLink}}}{2}$$

For 8× H100: bisection BW = $\frac{8 \times 900}{2} = 3{,}600$ GB/s.

**InfiniBand** connects nodes:

| Generation | Per-port BW | Typical config         |
| ---------- | ----------- | ---------------------- |
| HDR        | 200 Gb/s    | 8× per node = 200 GB/s |
| NDR        | 400 Gb/s    | 8× per node = 400 GB/s |
| XDR        | 800 Gb/s    | next-gen               |

### 3.3 Arithmetic Intensity and the Roofline Model

**Arithmetic intensity** (operational intensity):

$$I = \frac{\text{FLOPs}}{\text{Bytes transferred}}$$

The **roofline** model bounds achievable performance:

$$\text{Perf}(I) = \min\!\bigl(\text{Peak FLOP/s},\;\; I \times \text{Peak BW}\bigr)$$

```
                         ┌─────────── Peak FLOP/s (989 TFLOP/s BF16)
  log(FLOP/s)           │
       │           ╱─────
       │         ╱
       │       ╱  ← slope = Peak BW (3.35 TB/s)
       │     ╱
       │   ╱
       │ ╱
       └──────────────── log(Arithmetic Intensity)
             │
        Ridge point = Peak FLOP/s ÷ Peak BW
                    = 989 / 3.35 ≈ 295 FLOP/byte
```

**Decode-phase arithmetic intensity:**

For a single-token decode step with model parameters $P$ in $b$ bytes each:

$$I_{\text{decode}} = \frac{2P}{P \cdot b} = \frac{2}{b}$$

| Precision | Bytes/param ($b$) | $I_{\text{decode}}$ |
| --------- | ----------------- | ------------------- |
| FP32      | 4                 | 0.5                 |
| BF16      | 2                 | 1.0                 |
| INT8      | 1                 | 2.0                 |
| INT4      | 0.5               | 4.0                 |

All values are **far below** the H100 ridge point (295), confirming decode is
**always memory-bandwidth-bound** at batch size 1.

### 3.4 Bandwidth-Bound Decode Speed Formula

At batch size $B$ with model parameters $P$ at $b$ bytes/param, each decode step
must read the model weights from HBM once (amortised across the batch):

$$\text{time per step} = \frac{P \cdot b}{\text{BW}_{\text{HBM}}}$$

$$\text{tokens/s} = \frac{B}{\text{time per step}} = \frac{B \times \text{BW}_{\text{HBM}}}{P \cdot b}$$

> **Worked example — LLaMA-3 8B, BF16, H100:**
>
> $P = 8 \times 10^9$, $b = 2$, BW = 3.35 TB/s = $3.35 \times 10^{12}$ B/s.
>
> - $B = 1$: $\frac{1 \times 3.35 \times 10^{12}}{8 \times 10^9 \times 2} = 209$ tok/s → TPOT ≈ 4.8 ms
> - $B = 8$: $209 \times 8 = 1{,}675$ tok/s → TPOT ≈ 4.8 ms (same per-user latency)
> - $B = 32$: $209 \times 32 = 6{,}700$ tok/s (if KV cache fits in memory)
> - $B = 128$: starts hitting compute bound; need roofline analysis

The **batch-size crossover** from bandwidth-bound to compute-bound:

$$B^* = \frac{\text{Peak FLOP/s} \times b}{2 \times \text{BW}_{\text{HBM}}}$$

For H100 BF16: $B^* = \frac{989 \times 10^{12} \times 2}{2 \times 3.35 \times 10^{12}} \approx 295$.

At $B > B^*$, arithmetic intensity exceeds the ridge point and the system becomes
compute-bound — latency starts increasing with batch size.

### 3.5 Memory Capacity Constraints

Total GPU memory must hold:

$$M_{\text{total}} = M_{\text{model}} + M_{\text{KV}} + M_{\text{activations}} + M_{\text{overhead}}$$

**Model weights:**

$$M_{\text{model}} = P \times b$$

LLaMA-3 70B in BF16: $70 \times 10^9 \times 2 = 140$ GB → needs ≥ 2 H100s (TP=2).

**KV cache per token per layer:**

$$M_{\text{KV/tok/layer}} = 2 \times n_{\text{heads}} \times d_{\text{head}} \times b_{\text{KV}}$$

The factor 2 accounts for both K and V. For grouped-query attention (GQA) with $n_{\text{kv\_heads}}$ KV heads:

$$M_{\text{KV/tok/layer}} = 2 \times n_{\text{kv\_heads}} \times d_{\text{head}} \times b_{\text{KV}}$$

**Total KV cache:**

$$M_{\text{KV}} = n_{\text{layers}} \times M_{\text{KV/tok/layer}} \times \sum_{i} s_i$$

where $s_i$ is the current sequence length (prompt + generated) for request $i$.

> **Worked example — LLaMA-3 70B, BF16, TP=2:**
>
> - Layers = 80, KV heads = 8 (GQA), $d_{\text{head}}$ = 128, $b_{\text{KV}}$ = 2
> - Per token per layer: $2 \times 8 \times 128 \times 2 = 4{,}096$ bytes = 4 KB
> - Per token (all layers): $80 \times 4{,}096 = 327{,}680$ bytes ≈ 320 KB
> - Per token per GPU (TP=2): 160 KB
> - Available KV memory per GPU: 80 GB − 70 GB model / 2 = 45 GB
> - Max concurrent tokens: $45 \times 10^9 / (160 \times 10^3) \approx 281{,}250$ tokens
> - At 2048 tokens/request: ~137 concurrent requests per GPU pair

---

## 4. Batching Strategies

### 4.1 Static Batching

The simplest approach: collect $B$ requests, pad all sequences to the _longest_,
run the batch, return all results together.

$$\text{Waste}_{\text{static}} = \sum_{i=1}^{B} (s_{\max} - s_i)$$

```
Request 0:  [████████████████████·····]  ← 20 tokens, padded to 25
Request 1:  [████████████·············]  ← 12 tokens, padded to 25
Request 2:  [█████████████████████████]  ← 25 tokens (longest)
Request 3:  [████████·················]  ←  8 tokens, padded to 25
                                           Waste: 5+13+0+17 = 35 of 100 slots
```

**Problem:** A request that finishes early still occupies its batch slot.
GPU utilisation plummets as requests complete at different times.

### 4.2 Dynamic Batching

Group requests by similar length. When a request completes, its slot is freed but
**a new request cannot join** until the entire batch finishes. Better than static
but still wastes tail-end capacity.

### 4.3 Continuous Batching (ORCA)

**Key insight** (Yu et al., 2022): Treat each **iteration** (one decode step) as
the scheduling unit, not each request.

```
Iteration 1:  [A  B  C  D]     ← 4 active requests
Iteration 2:  [A  B  C  D]
Iteration 3:  [A  B  ·  D]     ← C finished; slot freed
Iteration 4:  [A  B  E  D]     ← E joins immediately
Iteration 5:  [A  ·  E  D]     ← B finished
Iteration 6:  [A  F  E  D]     ← F joins
```

**Throughput improvement over static:**

$$\text{Speedup} = \frac{\mathbb{E}[s_{\max}]}{\mathbb{E}[s]} \times \frac{1}{1 + \text{scheduling overhead}}$$

For typical output-length distributions (geometric with mean 100, max 500):
speedup ≈ 2–5× over static batching.

### 4.4 Chunked Prefill (Sarathi-Serve)

**Problem with continuous batching:** A long prefill (e.g., 8192 tokens) blocks
all decode steps, causing latency spikes (stalls) for in-flight requests.

**Solution:** Split prefill into fixed-size chunks (e.g., 512 tokens) and
interleave them with decode steps:

```
Step 1:  [decode₁ decode₂ decode₃ prefill_A_chunk1]
Step 2:  [decode₁ decode₂ decode₃ prefill_A_chunk2]
Step 3:  [decode₁ decode₂ decode₃ prefill_A_chunk3]
```

Each step has predictable latency ≈ $\text{chunk\_size} \times t_{\text{per\_token}}$.

**TTFT increase** (tradeoff): Prefill takes more steps, so:

$$\text{TTFT}_{\text{chunked}} = \left\lceil \frac{s_{\text{prompt}}}{\text{chunk\_size}} \right\rceil \times t_{\text{step}}$$

### 4.5 Prefill-Decode Disaggregation

Prefill is **compute-bound** (high arithmetic intensity).
Decode is **memory-bandwidth-bound** (low arithmetic intensity).
Running them on the _same_ GPU forces suboptimal resource sharing.

**Splitwise / DistServe** approach:

```
┌─────────────────────┐        ┌──────────────────────┐
│   Prefill GPUs      │──KV──→│   Decode GPUs         │
│  (compute-optimised)│  xfer  │  (bandwidth-optimised)│
│  High batch size    │        │  Low batch size        │
│  Short-lived        │        │  Long-lived            │
└─────────────────────┘        └──────────────────────┘
```

KV cache is transferred via NVLink or network. The overhead is:

$$t_{\text{KV\_xfer}} = \frac{M_{\text{KV/request}}}{\text{BW}_{\text{interconnect}}}$$

**Optimal prefill-to-decode GPU ratio:**

$$r = \frac{N_{\text{prefill}}}{N_{\text{decode}}} = \frac{t_{\text{prefill/req}}}{t_{\text{decode/req}}} \times \frac{\Theta_{\text{decode/GPU}}}{\Theta_{\text{prefill/GPU}}}$$

---

## 5. Scheduling Algorithms

### 5.1 FCFS (First-Come, First-Served)

The simplest policy: requests are served in arrival order. Fair but suffers from
**head-of-line blocking** — a long request delays all subsequent ones.

$$W_{\text{FCFS}} = \frac{\lambda \mathbb{E}[S^2]}{2(1 - \rho)}$$

where $S$ is the service time random variable (Pollaczek-Khinchine formula).

### 5.2 SJF (Shortest-Job-First) and SRTF

Minimises mean waiting time by serving shorter requests first.

**Problem:** Output length is unknown at request arrival time.

**Solution — output length prediction:**

Predictors use lightweight models or heuristics:

$$\hat{n}_{\text{output}} = f_\theta(\text{prompt tokens}, \text{task type}, \text{history})$$

Classification into buckets (short/medium/long) often suffices:

| Predictor Type          | Accuracy | Overhead |
| ----------------------- | -------- | -------- |
| Prompt-length heuristic | 45–55 %  | ~0 ms    |
| Logistic regression     | 60–70 %  | < 1 ms   |
| Small transformer       | 75–85 %  | 2–5 ms   |
| First-K-token pilot     | 80–90 %  | K × TPOT |

### 5.3 Priority Scheduling and Weighted Fair Queuing (WFQ)

Assign requests weights based on SLO tier:

$$\text{Virtual finish time}_i = \frac{s_i}{w_i} + \text{start time}$$

Schedule the request with the smallest virtual finish time.

```
Premium (w=4):  ████████░░░░░░░░░░░░░░░░░░  ← gets 4× the scheduling share
Standard (w=2): ░░░░████████░░░░░░░░░░░░░░
Free (w=1):     ░░░░░░░░░░░░████████░░░░░░
```

### 5.4 Preemption and Swapping

When a higher-priority request arrives and batch is full:

1. **Swap** the lowest-priority request's KV cache to CPU memory
2. Serve the higher-priority request
3. When a slot opens, **swap back** and resume

**Swap cost:**

$$t_{\text{swap}} = \frac{M_{\text{KV/request}}}{\text{BW}_{\text{PCIe}}}$$

> LLaMA-3 70B, 2048-token request: KV = 640 MB, PCIe gen5 = 64 GB/s →
> $t_{\text{swap}} = 640 / 64{,}000 \approx 10$ ms.

**Recompute** (alternative): Discard KV and re-prefill when resuming.
Cheaper if swap bandwidth is limited; wasteful if prompt is long.

### 5.5 Work-Stealing and Gang Scheduling

**Work-stealing:** Idle workers steal requests from busy workers' queues.
Useful when tensor-parallel groups have uneven load.

**Gang scheduling:** In tensor-parallel setups, all GPUs in a TP group must
execute the same request simultaneously. The scheduler must treat TP groups
as atomic units.

---

## 6. KV Cache Management

### 6.1 The KV Cache Problem

The KV cache is the dominant memory consumer at serving time:

```
┌────────────────────────────────────────────────────────┐
│                    80 GB HBM (H100)                    │
│                                                        │
│  ┌──────────────┐  ┌──────────────────────────────┐   │
│  │  Model Weights│  │         KV Cache              │   │
│  │   16 GB (8B)  │  │     Up to 60+ GB              │   │
│  └──────────────┘  └──────────────────────────────┘   │
│  ┌──────┐  ┌───────┐                                  │
│  │ Act. │  │Overhead│                                  │
│  │ 1 GB │  │  2 GB  │                                  │
│  └──────┘  └───────┘                                  │
└────────────────────────────────────────────────────────┘
```

**Naive allocation:** Pre-allocate max-sequence-length for each request.

$$\text{Waste}_{\text{naive}} = \sum_{i=1}^{B} (s_{\max} - s_i^{\text{current}}) \times M_{\text{KV/tok}}$$

For 100 requests with $s_{\max} = 4096$, avg $s_i = 1000$: waste ≈ 70 % of KV memory.

### 6.2 PagedAttention (vLLM)

**Key insight:** Borrow **virtual memory** concepts from operating systems.

KV cache is divided into fixed-size **blocks** (pages), e.g., 16 tokens per block.
A **block table** maps logical block indices to physical block locations.

```
Request A (seq_len=35):
  Logical:   [Block 0] [Block 1] [Block 2]   (3 blocks, last partially filled)
  Physical:  [  #7   ] [  #23  ] [  #41  ]   (non-contiguous in HBM)

Request B (seq_len=18):
  Logical:   [Block 0] [Block 1]
  Physical:  [  #3   ] [  #15  ]

Free blocks: #1, #2, #5, #8, #9, ...
```

**Internal fragmentation:** Only the last block of each request wastes space.

$$\text{Waste}_{\text{paged}} \leq B \times (\text{block\_size} - 1) \times M_{\text{KV/tok}}$$

**Fragmentation comparison:**

$$\text{Waste ratio} = \frac{\text{Waste}_{\text{paged}}}{\text{Waste}_{\text{naive}}} = \frac{B \times (\text{block\_size} - 1)}{B \times (s_{\max} - \bar{s})} = \frac{\text{block\_size} - 1}{s_{\max} - \bar{s}}$$

> With block*size = 16, $s*{\max}$ = 4096, $\bar{s}$ = 1000:
> ratio = $15 / 3096 \approx 0.5\%$ → **paged wastes ~200× less** than naive.

### 6.3 Prefix Caching and RadixAttention (SGLang)

Many requests share a common **system prompt** (e.g., 1000 tokens).
RadixAttention stores KV blocks in a **radix tree** indexed by token sequences.

```
                    [system_prompt_tokens]
                   /                      \
          [user_A_prefix]           [user_B_prefix]
         /              \                    |
   [continuation_1] [continuation_2]  [continuation_3]
```

**Memory savings:**

$$\text{Savings} = (N_{\text{requests}} - 1) \times s_{\text{shared}} \times M_{\text{KV/tok}}$$

For 100 requests sharing a 1000-token system prompt with LLaMA-3 8B (640 bytes/token):
savings = $99 \times 1000 \times 640 \approx 60$ MB.

For longer shared prefixes (few-shot examples, RAG context): savings scale linearly.

### 6.4 Eviction Policies

When KV memory is full, which blocks to evict?

| Policy   | Description            | Pros                   | Cons                     |
| -------- | ---------------------- | ---------------------- | ------------------------ |
| LRU      | Least recently used    | Simple, cache-friendly | May evict active request |
| LFU      | Least frequently used  | Keeps popular prefixes | Slow to adapt            |
| FIFO     | First in, first out    | Simplest               | Ignores access pattern   |
| Priority | Based on request SLO   | SLO-aware              | Starvation risk          |
| Size     | Evict largest KV first | Frees most memory      | Penalises long contexts  |

**Optimal eviction** is related to the **Belady's algorithm** (MIN): evict the block
whose next access is furthest in the future. This is optimal but requires future
knowledge — approximated by predicting remaining output length.

### 6.5 KV Cache Offloading

Tiered storage: GPU HBM → CPU DRAM → NVMe SSD.

$$t_{\text{retrieve}} = \frac{M_{\text{KV/seq}}}{\text{BW}_{\text{tier}}} + t_{\text{latency,tier}}$$

| Tier     | BW        | Retrieve 320 KB (1 token, 70B) | Retrieve 640 MB (2K seq) |
| -------- | --------- | ------------------------------ | ------------------------ |
| HBM      | 3.35 TB/s | ~0.1 µs                        | ~0.2 ms                  |
| CPU DRAM | 400 GB/s  | ~0.8 µs                        | ~1.6 ms                  |
| NVMe     | 14 GB/s   | ~23 µs                         | ~46 ms                   |

**Prefetching:** Predict which KV blocks will be needed and start the transfer
$t_{\text{retrieve}}$ ahead of time — hides latency behind computation.

---

## 7. Multi-GPU and Multi-Node Serving

### 7.1 Tensor Parallelism (TP) for Serving

Split each layer's weight matrices across $T$ GPUs. Every decode step requires
an **all-reduce** for the output projection.

**Communication cost per step:**

$$t_{\text{allreduce}} = 2 \times \frac{(T - 1)}{T} \times \frac{d_{\text{model}} \times b \times B}{\text{BW}_{\text{NVLink}}}$$

For $T = 8$, $d_{\text{model}} = 8192$, $b = 2$, $B = 1$, NVLink = 900 GB/s:

$$t_{\text{allreduce}} = 2 \times \frac{7}{8} \times \frac{8192 \times 2 \times 1}{900 \times 10^9} \approx 32\text{ ns}$$

For $n_{\text{layers}} = 80$: total comm = ~2.5 µs ≪ compute time. TP within a node
is almost free for small batches.

### 7.2 Pipeline Parallelism (PP) for Serving

Split layers across $P$ GPUs sequentially. Each stage communicates activations,
not weights.

**Latency penalty:** Each pipeline stage adds a communication hop:

$$T_{\text{PP}} = P \times t_{\text{stage}} + (P - 1) \times t_{\text{comm}}$$

**Pipeline bubbles** amplify during decode because each step generates only one
token — microbatching cannot fill the pipeline.

**Recommendation:** Prefer TP over PP for latency-sensitive serving.
Use PP only when model doesn't fit in TP-reachable GPUs.

### 7.3 Data Parallelism (DP) for Serving

Replicate the entire model on $D$ GPU groups. Each group independently serves requests.
A load balancer distributes incoming requests.

$$\Theta_{\text{total}} = D \times \Theta_{\text{single}}$$

No inter-group communication at inference time — linear scaling.

### 7.4 Parallelism Strategy Decision Table

| Model Size | GPUs Avail. | Strategy         | Reasoning                          |
| ---------- | ----------- | ---------------- | ---------------------------------- |
| 8B         | 1           | None             | Fits on single GPU                 |
| 8B         | 8           | DP=8             | Replicate for throughput           |
| 70B        | 2           | TP=2             | Model doesn't fit on 1 GPU         |
| 70B        | 8           | TP=2, DP=4       | 4 replicas for throughput          |
| 70B        | 16          | TP=4, DP=4       | Lower TP latency + high throughput |
| 405B       | 8           | TP=8             | Minimum to fit model               |
| 405B       | 32          | TP=8, DP=4       | Max throughput within SLO          |
| 405B       | 64          | TP=8, PP=2, DP=4 | Cross-node, pipeline needed        |

### 7.5 Expert Parallelism for MoE

Mixture-of-Experts models (e.g., Mixtral, DeepSeek-V2) have sparse FFN layers:

$$\text{FFN}_{\text{MoE}}(x) = \sum_{i \in \text{TopK}} g_i \cdot \text{Expert}_i(x)$$

**Expert parallelism (EP):** Distribute experts across GPUs:

- 8 experts, 8 GPUs: 1 expert per GPU
- Each token is routed to $K = 2$ experts → all-to-all communication

**Communication cost (all-to-all):**

$$t_{\text{a2a}} = \frac{B \times d_{\text{model}} \times b \times K}{N_{\text{GPU}} \times \text{BW}_{\text{interconnect}}}$$

### 7.6 Disaggregated Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     API Gateway                         │
│                         │                               │
│              ┌──────────┴──────────┐                    │
│              ▼                     ▼                    │
│     ┌────────────────┐   ┌────────────────┐            │
│     │ Prefill Cluster │   │ Decode Cluster  │            │
│     │  (H100 SXM)     │   │  (H100 SXM)     │            │
│     │  TP=4, DP=2     │   │  TP=2, DP=4     │            │
│     │  High compute   │   │  High BW focus  │            │
│     └───────┬─────────┘   └───────┬─────────┘            │
│             │    KV Transfer      │                      │
│             └──────────┬──────────┘                      │
│                        ▼                                 │
│              ┌────────────────┐                          │
│              │  KV Cache Store │                          │
│              │  (DRAM / NVMe) │                          │
│              └────────────────┘                          │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Speculative Decoding Systems

### 8.1 Core Architecture

Use a small **draft model** $M_q$ to propose $K$ tokens cheaply, then have the
large **target model** $M_p$ verify them all in one parallel forward pass.

```
Draft model (8B):   generates K=4 tokens:  [t₁, t₂, t₃, t₄]   — fast, cheap
Target model (70B): verifies in 1 pass:    [✓,  ✓,  ✓,  ✗ ]   — expensive but parallel
                    accepts 3 tokens + samples 1 corrected token
                    Net: 4 tokens in ~1 target-model step
```

### 8.2 Acceptance Probability and Token Distribution

Draft token $t_k$ sampled from $q(t_k | t_{<k})$ is accepted with probability:

$$\alpha_k = \min\!\left(1, \frac{p(t_k | t_{<k})}{q(t_k | t_{<k})}\right)$$

where $p$ is the target distribution and $q$ is the draft distribution.

**Expected accepted tokens per draft cycle:**

$$\mathbb{E}[\text{accepted}] = \sum_{k=1}^{K} \prod_{j=1}^{k} \alpha_j$$

If $\alpha$ is constant:

$$\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{K+1}}{1 - \alpha} - 1$$

> **Example:** $\alpha = 0.75$, $K = 4$:
>
> $\mathbb{E}[\text{accepted}] = \frac{1 - 0.75^5}{1 - 0.75} - 1 = \frac{1 - 0.2373}{0.25} - 1 = 3.051 - 1 = 2.05$
>
> Plus 1 corrected token → **3.05 tokens per cycle** on average.

### 8.3 Memory Cost of Speculative Decoding

Must hold both models simultaneously:

$$M_{\text{spec}} = M_{\text{target}} + M_{\text{draft}} + M_{\text{KV,target}} + M_{\text{KV,draft}}$$

> 70B target (BF16) + 8B draft (BF16) = 140 + 16 = 156 GB model weights alone.
> Requires ≥ 3 H100-80GB with TP.

### 8.4 Throughput Speed-Up Model

Let $t_d$ = draft model time per token, $t_v$ = target verification time for $K$ tokens.

**Time per speculative cycle:**

$$t_{\text{cycle}} = K \cdot t_d + t_v$$

**Tokens per cycle:**

$$n_{\text{cycle}} = \mathbb{E}[\text{accepted}] + 1$$

**Speed-up over standard autoregressive:**

$$S = \frac{n_{\text{cycle}} \times t_{\text{AR}}}{t_{\text{cycle}}} = \frac{n_{\text{cycle}} \times t_{\text{AR}}}{K \cdot t_d + t_v}$$

Approximating $t_v \approx t_{\text{AR}}$ (single target forward pass) and $t_d \approx \gamma \cdot t_{\text{AR}}$ where $\gamma \ll 1$ (draft is much cheaper):

$$S \approx \frac{n_{\text{cycle}}}{K\gamma + 1}$$

> With $n_{\text{cycle}} = 3.05$, $K = 4$, $\gamma = 0.1$: $S = \frac{3.05}{0.4 + 1} = 2.18\times$ speed-up.

### 8.5 Dynamic Draft Length

**Problem:** Fixed $K$ is suboptimal — easy tokens need fewer drafts, hard tokens
need more.

**Adaptive strategy:** Stop drafting early if draft model uncertainty exceeds a threshold:

$$\text{Stop if } H(q(\cdot | t_{<k})) > \tau$$

where $H$ is the entropy of the draft distribution.

### 8.6 Batch Speculation

Speculative decoding with batched requests is complex:

- Different requests may accept different numbers of tokens
- The target model verification batch has **variable** sequence lengths
- Memory fragmentation increases

**Token tree verification** (SpecInfer, Sequoia): Instead of a single draft
sequence, explore a _tree_ of token continuations:

```
        t₁ ─── t₂ ─── t₃
         │       └──── t₃'
         └──── t₂'─── t₃''
```

More tokens verified per target call but memory grows with tree width.

### 8.7 Medusa and Draft-Head Approaches

Instead of a separate draft model, add **extra prediction heads** to the target model:

$$\hat{t}_{k+j} = \text{Head}_j(h_k) \qquad j = 1, 2, \ldots, K$$

**Advantages:** No separate model weight memory; shared KV cache.
**Disadvantages:** Lower acceptance rate than a full draft model; extra training required.

---

## 9. Inference Optimisation

### 9.1 Kernel Fusion

Combine multiple GPU operations into a single kernel launch:

**Before (3 kernel launches):**

```
kernel_1: LayerNorm
kernel_2: QKV projection
kernel_3: Bias add
```

**After (1 fused kernel):**

```
kernel_fused: LayerNorm + QKV + Bias
```

**Savings:** Reduces kernel launch overhead (~5 µs each), eliminates intermediate
HBM reads/writes, and keeps data in registers/shared memory.

$$\text{Speedup}_{\text{fusion}} \approx \frac{n_{\text{kernels}} \times (t_{\text{launch}} + t_{\text{HBM\_roundtrip}})}{t_{\text{launch}} + t_{\text{compute}}}$$

### 9.2 CUDA Graphs

**Problem:** CPU-side kernel scheduling overhead dominates when GPU kernels are tiny
(decode with small batch).

**Solution:** Record a sequence of CUDA operations into a **graph**, then replay it
with minimal CPU overhead.

Savings: CPU scheduling time drops from ~100 µs to ~10 µs per step.

**Limitation:** Graph topology must be fixed → can't handle dynamic shapes
(variable batch, variable seq length) without re-recording.

### 9.3 Tensor-Parallel Communication Overlap

Overlap all-reduce communication with computation of the next layer:

```
Time →
GPU 0: [FFN layer L] [allreduce L] [Attn layer L+1] [allreduce L+1]
                      ↑ overlap ↑
GPU 0: [FFN layer L] [allreduce L ∥ Attn layer L+1] [allreduce L+1]
```

$$t_{\text{overlap}} = \max(t_{\text{compute}}, t_{\text{comm}}) \quad \text{vs} \quad t_{\text{sequential}} = t_{\text{compute}} + t_{\text{comm}}$$

### 9.4 In-Flight Request Management

Track per-request state across iterations:

```python
# Per-request state machine
STATES = {
    "QUEUED":     "Waiting for batch slot",
    "PREFILLING": "Processing prompt tokens",
    "DECODING":   "Generating output tokens",
    "SWAPPED":    "KV cache offloaded to CPU",
    "FINISHED":   "EOS or max_length reached",
    "TIMEOUT":    "Exceeded SLO deadline",
}
```

### 9.5 Structured Output and FSM Constraints

When output must be valid JSON, SQL, or match a regex, use a **finite-state machine (FSM)** or **context-free grammar (CFG)** to constrain decoding:

$$p_{\text{constrained}}(t_k) = \frac{p(t_k) \cdot \mathbf{1}[t_k \in \mathcal{V}_{\text{valid}}]}{\sum_{t \in \mathcal{V}_{\text{valid}}} p(t)}$$

**Performance impact:** Token masking adds ~1 % overhead; the real gain is
**zero retries** from invalid output → effective throughput increases significantly.

### 9.6 Attention Sinks

Observation: The first few tokens in the KV cache receive disproportionate attention
scores regardless of content (Xiao et al., 2023).

**StreamingLLM** keeps a fixed **attention sink** window (first 4 tokens) + a
recent sliding window:

$$\text{KV}_{\text{active}} = \text{KV}[0:4] \cup \text{KV}[t - w : t]$$

Memory: $O(w)$ instead of $O(t)$ — enables infinite-length streaming.

---

## 10. Cost Modelling and Economics

### 10.1 Total Cost of Ownership (TCO)

$$\text{TCO}_{\text{annual}} = C_{\text{hardware}} / L + C_{\text{power}} + C_{\text{cooling}} + C_{\text{network}} + C_{\text{ops}} + C_{\text{space}}$$

where $L$ is hardware lifetime (typically 3–5 years).

| Component     | H100 DGX (8-GPU) | A100 DGX (8-GPU) |
| ------------- | ---------------- | ---------------- |
| Hardware      | ~\$300K          | ~\$200K          |
| Lifetime      | 4 years          | 4 years          |
| Annual depr.  | \$75K            | \$50K            |
| Power (8 kW)  | \$14K/yr         | \$10K/yr         |
| Cooling       | \$4K/yr          | \$3K/yr          |
| Ops (0.5 FTE) | \$75K/yr         | \$75K/yr         |
| **TCO/yr**    | **~\$168K**      | **~\$138K**      |

### 10.2 CPM Formula with Worked Examples

$$\text{CPM} = \frac{\text{GPU cost/hour} \times 10^6}{\text{throughput (tok/s)} \times 3600}$$

**Equivalently, for a fleet:**

$$\text{CPM} = \frac{N_{\text{GPU}} \times c_{\text{GPU/hr}} \times 10^6}{\Theta_{\text{total}} \times 3600}$$

| Setup                      | GPUs | Cost/hr | Tok/s  | CPM     |
| -------------------------- | ---- | ------- | ------ | ------- |
| LLaMA-3 8B, BF16, 1×H100   | 1    | \$3.00  | 8,000  | \$0.10  |
| LLaMA-3 8B, INT4, 1×H100   | 1    | \$3.00  | 15,000 | \$0.056 |
| LLaMA-3 70B, BF16, 2×H100  | 2    | \$6.00  | 2,500  | \$0.67  |
| LLaMA-3 70B, INT4, 2×H100  | 2    | \$6.00  | 5,500  | \$0.30  |
| LLaMA-3 405B, BF16, 8×H100 | 8    | \$24.00 | 1,200  | \$5.56  |

### 10.3 Throughput-Latency Pareto Curve

As batch size increases, throughput rises but so does latency:

```
Throughput │                     ╭──────── max throughput (compute-bound)
(tok/s)    │                ╭───╯
           │           ╭───╯
           │      ╭───╯
           │  ╭──╯
           │╭╯
           └──────────────────────────── Batch size

Latency    │                          ╭── latency explodes
(ms)       │                     ╭───╯
           │                ╭───╯
           │ ───────────────╯ ← latency flat (BW-bound → compute-bound transition)
           └──────────────────────────── Batch size
```

**SLO-constrained optimal batch size:**

$$B^* = \max\left\{B : \text{TPOT}(B) \leq \text{SLO}_{\text{TPOT}}\right\}$$

### 10.4 Capacity Planning

Given expected traffic $\lambda$ (requests/s) with average $n_{\text{output}}$ tokens:

$$N_{\text{GPU,min}} = \left\lceil \frac{\lambda \times n_{\text{output}}}{\Theta_{\text{GPU}} \times (1 - \text{headroom})} \right\rceil$$

> **Example:** $\lambda = 50$ req/s, avg 200 tokens, per-GPU throughput = 5{,}000 tok/s, 30 % headroom:
>
> $N = \lceil \frac{50 \times 200}{5000 \times 0.7} \rceil = \lceil 2.86 \rceil = 3$ GPUs minimum.

**Burst capacity:** Size for P99 arrival rate, not mean:

$$N_{\text{burst}} = \left\lceil \frac{\lambda_{P99} \times n_{\text{output}}}{\Theta_{\text{GPU}} \times (1 - \text{headroom})} \right\rceil$$

### 10.5 Pricing Strategy

| Pricing Model      | Description                             | When to use                |
| ------------------ | --------------------------------------- | -------------------------- |
| Per-token (input)  | Charge per input token                  | Variable prompt lengths    |
| Per-token (output) | Charge per output token (higher rate)   | Output is more expensive   |
| Per-request        | Flat fee per API call                   | Predictable workloads      |
| Per-minute         | Charge for wall-clock compute time      | Streaming / long sessions  |
| Commitment         | Discounted rate for reserved throughput | Steady, predictable demand |

**Margin calculation:**

$$\text{Margin} = 1 - \frac{\text{CPM}_{\text{cost}}}{\text{CPM}_{\text{price}}}$$

> If CPM cost = \$0.30 and price = \$1.00: margin = 70 %.

### 10.6 Fleet Utilisation

$$U_{\text{fleet}} = \frac{\sum_t \text{active GPUs}(t)}{\text{total GPUs} \times T}$$

Target: $U_{\text{fleet}} > 0.7$ for cost efficiency. Below 0.5 indicates over-provisioning.

**Strategies to improve utilisation:**

- **Autoscaling:** Scale GPU count with demand.
- **Mixed workloads:** Run batch jobs (fine-tuning, evals) on idle serving GPUs.
- **Spot instances:** Offload low-priority traffic to cheaper spot GPUs.

---

## 11. Distributed Systems for Serving

### 11.1 CAP Theorem Implications

In LLM serving, the CAP tradeoff manifests as:

| Property            | In LLM serving context                    |
| ------------------- | ----------------------------------------- |
| Consistency         | All replicas serve the same model version |
| Availability        | Every request gets a response             |
| Partition tolerance | System handles network failures           |

Most LLM serving systems choose **AP** (availability + partition tolerance) with
**eventual consistency** for model updates — a stale model version is acceptable
for a short window during rollout.

### 11.2 Load Balancing

**Strategies:**

| Algorithm              | Description                           | Best for                |
| ---------------------- | ------------------------------------- | ----------------------- |
| Round-robin            | Cycle through backends                | Homogeneous fleet       |
| Least-connections      | Route to least busy backend           | Variable request length |
| Least-tokens-in-flight | Route based on KV cache occupancy     | LLM-specific            |
| Prefix-aware           | Route to backend with cached prefix   | Shared system prompts   |
| Consistent hashing     | Hash-based routing for cache locality | Prefix caching          |

**Prefix-aware routing** maximises cache hits:

$$\text{Backend} = \arg\max_{b} |\text{prefix}(r) \cap \text{cached\_prefixes}(b)|$$

### 11.3 Fault Tolerance

**Failure modes:**

| Failure           | Impact                        | Mitigation                   |
| ----------------- | ----------------------------- | ---------------------------- |
| GPU OOM           | Single request or batch fails | KV eviction + retry          |
| GPU hang          | Worker becomes unresponsive   | Heartbeat + restart          |
| Node crash        | All requests on node lost     | Replica failover, re-prefill |
| Network partition | Split-brain serving           | Quorum-based routing         |
| Model corruption  | Silent wrong outputs          | Checksum verification        |

**Request retry:** On failure, re-route to another replica. Since decode is stateful,
the new replica must re-prefill from the prompt — no state to transfer.

### 11.4 Model Rollout

**Blue-green deployment:**

```
Traffic: 100% ──→ [Blue: Model v1]    0% ──→ [Green: Model v2]
         ↓ gradual shift
Traffic:  50% ──→ [Blue: Model v1]   50% ──→ [Green: Model v2]
         ↓
Traffic:   0% ──→ [Blue: Model v1]  100% ──→ [Green: Model v2]
```

**Canary deployment:** Route X % of traffic to new model, monitor metrics,
auto-rollback if quality drops.

**Shadow deployment:** Send copies of real traffic to new model but discard
responses — measure latency and quality without impacting users.

### 11.5 Observability

Essential metrics to monitor:

```
┌─────────────────────────────────────────────────────────────┐
│                    Observability Stack                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Request metrics                                   │
│    - TTFT / TPOT / E2E latency (P50, P95, P99)             │
│    - Throughput (tok/s), QPS                                │
│    - Error rate, timeout rate                               │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: System metrics                                    │
│    - GPU utilisation, memory usage                          │
│    - KV cache occupancy (%)                                 │
│    - Queue depth, batch size distribution                   │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Business metrics                                  │
│    - CPM (cost per million tokens)                          │
│    - SLO attainment (%)                                     │
│    - Revenue per GPU-hour                                   │
└─────────────────────────────────────────────────────────────┘
```

### 11.6 Network Topology

**Fat-tree topology** (common in GPU clusters):

```
          ┌─────────┐
          │  Spine   │
          │ Switches │
          └─┬──┬──┬─┘
         ╱  │  │  │  ╲
   ┌────┐  ┌────┐  ┌────┐
   │Leaf│  │Leaf│  │Leaf│   ← Top-of-Rack switches
   └─┬──┘  └─┬──┘  └─┬──┘
     │        │        │
   [Nodes]  [Nodes]  [Nodes]   ← GPU servers
```

**Bisection bandwidth:** The minimum bandwidth to partition the network into two
equal halves. Must be sufficient for all-reduce in TP across nodes.

---

## 12. Hardware Selection

### 12.1 GPU Comparison Table

| Spec              | A100 SXM      | H100 SXM     | H200 SXM       | B200 SXM       |
| ----------------- | ------------- | ------------ | -------------- | -------------- |
| Architecture      | Ampere        | Hopper       | Hopper         | Blackwell      |
| FP16/BF16 TFLOP/s | 312           | 989          | 989            | 2,250          |
| INT8 TOPS         | 624           | 1,979        | 1,979          | 4,500          |
| FP4 TOPS          | —             | —            | —              | 9,000          |
| HBM Capacity      | 80 GB (HBM2e) | 80 GB (HBM3) | 141 GB (HBM3e) | 192 GB (HBM3e) |
| HBM Bandwidth     | 2.0 TB/s      | 3.35 TB/s    | 4.8 TB/s       | 8.0 TB/s       |
| TDP               | 400 W         | 700 W        | 700 W          | 1,000 W        |
| NVLink BW         | 600 GB/s      | 900 GB/s     | 900 GB/s       | 1,800 GB/s     |

### 12.2 Inference-Optimised Architectures

**TPU v5e/v6e (Google):**

- Optimised for large-batch inference
- 256-chip pods with high-bandwidth ICI interconnect
- BF16 compute with INT8 quantisation support
- Better cost-efficiency for high-throughput, latency-tolerant workloads

**Groq LPU (Language Processing Unit):**

- SRAM-only architecture (no HBM) → deterministic latency
- 230 MB on-chip SRAM → supports only smaller models natively
- > 500 tok/s per user for LLaMA-3 70B (multi-chip)
- Excellent for low-latency, moderate-throughput use cases

**AWS Inferentia2 / Trainium:**

- Custom interconnect (NeuronLink)
- Optimised for AWS ecosystem
- Lower cost/tok but less flexible than GPUs

**Cerebras CS-3 (Wafer-Scale):**

- Single 900,000-core chip
- 44 GB on-chip SRAM, 21 PB/s internal bandwidth
- Model must fit on-chip → limited to ~13B params
- Eliminates memory bandwidth bottleneck entirely

### 12.3 Edge and On-Device Hardware

| Device              | Compute       | Memory        | Use case              |
| ------------------- | ------------- | ------------- | --------------------- |
| Apple M4 Max        | 38 TOPS (ANE) | 128 GB shared | Desktop LLM inference |
| NVIDIA Jetson Orin  | 275 TOPS INT8 | 64 GB         | Robotics, edge AI     |
| Qualcomm Snapdragon | 45 TOPS       | 16 GB         | Mobile LLM inference  |
| Google Edge TPU     | 4 TOPS        | —             | IoT, tiny models      |

### 12.4 Hardware Selection Framework

```
                  ┌─────────────────────┐
                  │ What is your SLO?   │
                  └───────┬─────────────┘
                          │
            ┌─────────────┴─────────────┐
            │                           │
     TTFT < 100ms               TTFT < 1s
     (interactive)               (batch-ok)
            │                           │
   ┌────────┴────────┐         ┌───────┴────────┐
   │Model < 13B?     │         │ Cloud or On-prem│
   │                 │         │                 │
  Yes               No        Cloud          On-prem
   │                 │         │                 │
Groq/Edge     H100/B200      TPU/Inf2      H100/A100
```

---

## 13. Serving Frameworks

### Framework Comparison Table

| Feature             | vLLM            | TensorRT-LLM    | SGLang       | DeepSpeed-FastGen | Ollama       |
| ------------------- | --------------- | --------------- | ------------ | ----------------- | ------------ |
| Continuous batching | ✓               | ✓               | ✓            | ✓                 | ✗            |
| PagedAttention      | ✓               | ✓ (variant)     | ✓            | ✓                 | ✗            |
| Prefix caching      | ✓ (APC)         | ✗               | ✓ (Radix)    | ✗                 | ✗            |
| Speculative decode  | ✓               | ✓               | ✓            | ✓                 | ✗            |
| Multi-LoRA          | ✓               | ✗               | ✓            | ✗                 | partial      |
| Quantisation        | AWQ, GPTQ, FP8  | FP8, INT8, INT4 | AWQ, GPTQ    | INT8              | GGUF (Q4-Q8) |
| Structured output   | ✓               | limited         | ✓ (native)   | ✗                 | ✗            |
| TP support          | ✓               | ✓               | ✓            | ✓                 | ✗            |
| PP support          | ✓               | ✓               | partial      | ✓                 | ✗            |
| CUDA graphs         | ✓               | ✓               | ✓            | ✓                 | ✗            |
| Production-ready    | ✓ (v0.4+)       | ✓               | emerging     | ✓                 | desktop      |
| Primary language    | Python/C++      | C++/Python      | Python/C++   | Python/C++        | Go/C++       |
| Best for            | General serving | Max throughput  | Prefix-heavy | DeepSpeed users   | Local / dev  |

### When to Use What

| Scenario                        | Recommended    | Why                             |
| ------------------------------- | -------------- | ------------------------------- |
| General production serving      | vLLM / TRT-LLM | Battle-tested, feature-complete |
| Many users share system prompt  | SGLang         | RadixAttention prefix caching   |
| Maximum throughput, NVIDIA GPUs | TensorRT-LLM   | Best kernel optimisation        |
| Local development / prototyping | Ollama         | Simplest setup                  |
| Multi-LoRA serving              | vLLM / SGLang  | Native S-LoRA support           |
| Structured JSON output          | SGLang         | Native grammar-guided decoding  |

---

## 14. Advanced Serving Patterns

### 14.1 Cascaded Serving

Route easy queries to a small model; escalate hard ones to a large model.

```
Request → [Router] ──easy──→ [8B model]  → Response
              │
              └──hard──→ [70B model] → Response
```

**Router decision function:**

$$\text{route}(x) = \begin{cases} \text{small} & \text{if } \text{confidence}(x) > \tau \\ \text{large} & \text{otherwise} \end{cases}$$

**Expected cost:**

$$\text{CPM}_{\text{cascade}} = f_{\text{easy}} \cdot \text{CPM}_{\text{small}} + (1 - f_{\text{easy}}) \cdot (\text{CPM}_{\text{small}} + \text{CPM}_{\text{large}})$$

If 80 % of queries are easy: CPM = $0.8 \times 0.10 + 0.2 \times (0.10 + 0.67) = 0.08 + 0.154 = \$0.234$ vs. \$0.67 for always-large.

### 14.2 Multi-LoRA Serving (S-LoRA)

Serve many fine-tuned LoRA adapters from a single base-model instance.

```
Base Model (70B, shared):  [=================================]
LoRA Adapter A (rank 16):  [··]  ← ~50 MB
LoRA Adapter B (rank 32):  [····] ← ~100 MB
LoRA Adapter C (rank 16):  [··]  ← ~50 MB
```

**Weight computation:**

$$W_{\text{adapted}} = W_{\text{base}} + B_i A_i$$

where $A_i \in \mathbb{R}^{r \times d}$, $B_i \in \mathbb{R}^{d \times r}$, and
$r$ is the LoRA rank.

**Memory:** Base model loaded once; each adapter adds $2 \times n_{\text{layers}} \times r \times d \times b$ bytes.

For 70B model with rank-16 adapters: ~50 MB each → can serve 1000+ adapters with
50 GB adapter memory.

### 14.3 Streaming and Server-Sent Events (SSE)

Token-by-token streaming delivers better user experience:

```
Client ─── HTTP POST /v1/chat/completions (stream=true) ───→ Server
Client ←── data: {"token": "The"}                          ←── Server
Client ←── data: {"token": " answer"}                      ←── Server
Client ←── data: {"token": " is"}                          ←── Server
Client ←── data: [DONE]                                    ←── Server
```

**Latency perception:** Users perceive TTFT as responsiveness, not E2E.
Streaming reduces _perceived_ latency even if _total_ latency is unchanged.

### 14.4 Prompt Compression (LLMLingua)

Reduce prompt length to lower TTFT and KV cache memory:

$$\text{compressed} = \text{LLMLingua}(\text{prompt}, \text{ratio}=0.5)$$

**Information-theoretic basis:** Remove tokens with high conditional probability
(low information content):

$$\text{keep}(t_i) = \mathbf{1}\!\left[-\log p(t_i | t_{<i}) > \tau\right]$$

**Tradeoff:** 2× prompt compression → ~50 % TTFT reduction, ~1–3 % quality loss.

### 14.5 Test-Time Compute Scaling

Allocate more compute at inference time for harder problems:

- **Best-of-N sampling:** Generate $N$ completions, pick the best via a verifier.
- **Chain-of-thought:** Longer reasoning → more tokens → more compute.
- **Tree search (MCTS):** Explore solution space with backtracking.

**Cost model:**

$$\text{CPM}_{\text{TTC}} = N \times \text{CPM}_{\text{base}} + \text{CPM}_{\text{verifier}}$$

### 14.6 Multimodal Serving

Images and video add prefill compute without proportional output:

| Modality | Tokens per input    | Prefill overhead    |
| -------- | ------------------- | ------------------- |
| Text     | 1 per token         | Baseline            |
| Image    | 576–2048 per image  | 10–50× more prefill |
| Video    | 1000–10000 per clip | 100× more prefill   |
| Audio    | 25 per second       | 5× per second       |

**Implication:** Multimodal serving is even more prefill-dominated → disaggregation
even more beneficial.

---

## 15. Emerging Architectures

### 15.1 Hybrid SSM-Attention Models

Models like Jamba, Zamba mix Mamba (SSM) layers with sparse attention layers:

**SSM decode complexity:** $O(d \cdot s)$ per layer (state-space model — no KV cache).
**Attention decode complexity:** $O(d \cdot n)$ per layer (needs KV cache of length $n$).

**Hybrid benefit:** If only 1/8 layers use attention:

$$M_{\text{KV,hybrid}} = \frac{n_{\text{attn\_layers}}}{n_{\text{total\_layers}}} \times M_{\text{KV,full}}$$

For 32 layers, 4 attention: KV memory is 12.5 % of a full-attention model.

### 15.2 Diffusion Language Models

Instead of autoregressive generation, generate entire sequence in parallel
via diffusion denoising:

$$x_0 = \text{denoise}(x_T, T \text{ steps})$$

**Latency:** Fixed $T$ steps regardless of sequence length → no autoregressive bottleneck.
**Problem:** Quality currently lags autoregressive models; active research area.

### 15.3 KV Cache Disaggregation (Mooncake)

Separate KV cache storage from compute:

```
┌────────────┐     ┌─────────────────┐
│ GPU Workers │────→│ KV Cache Pool   │
│ (compute)   │←────│ (CPU DRAM + SSD)│
└────────────┘     └─────────────────┘
```

**Benefits:** KV cache lifetime decoupled from GPU lifetime; enables session
persistence, KV sharing across replicas, and massive context windows.

### 15.4 Neuromorphic and Photonic Accelerators

| Technology   | Status   | Promise                             | Limitation         |
| ------------ | -------- | ----------------------------------- | ------------------ |
| Neuromorphic | Research | Event-driven, ultra-low power       | Programming model  |
| Photonic     | Research | Speed-of-light matrix multiply      | Precision, scale   |
| Analog       | Research | In-memory compute, no data movement | Noise, calibration |

### 15.5 Serverless LLM Inference

**Challenge:** Model loading time (cold start) dominates:

$$t_{\text{cold}} = \frac{M_{\text{model}}}{\text{BW}_{\text{storage}}}$$

70B BF16 from NVMe (14 GB/s): $t_{\text{cold}} = 140 / 14 = 10$ s.

**Solutions:**

- **Checkpoint streaming:** Start inference before full model is loaded.
- **Model caching:** Keep warm models in a shared GPU pool.
- **Speculative loading:** Pre-load likely-needed models based on traffic patterns.

---

## 16. Common Mistakes and Pitfalls

| #   | Mistake                                    | Why It's Wrong                                          | Fix                                                |
| --- | ------------------------------------------ | ------------------------------------------------------- | -------------------------------------------------- |
| 1   | Optimising only for throughput             | Ignoring latency violates user-facing SLOs              | Set SLO first, then maximise throughput within SLO |
| 2   | Using static batching in production        | Wastes 40–70 % of GPU capacity                          | Use continuous batching (vLLM, TRT-LLM, SGLang)    |
| 3   | Ignoring queuing theory                    | Under-provisioning causes latency to explode at ρ > 0.8 | Provision for ρ < 0.7; use M/M/c models            |
| 4   | Pre-allocating max-length KV cache         | Wastes 70 %+ of KV memory                               | Use PagedAttention with dynamic allocation         |
| 5   | TP across nodes for latency-sensitive apps | Inter-node latency (5+ µs) kills decode speed           | TP within node only; use PP or DP across nodes     |
| 6   | Ignoring the prefill-decode asymmetry      | Mixing compute-bound and BW-bound phases is suboptimal  | Consider chunked prefill or disaggregation         |
| 7   | Benchmarking at batch=1 only               | Real-world serving has varying batch sizes              | Measure throughput across batch sizes; find Pareto |
| 8   | Not monitoring KV cache occupancy          | Silent OOM → requests dropped silently                  | Alert on KV occupancy > 80 %; implement eviction   |
| 9   | Using FP32 for inference                   | 2× memory, ~0 quality difference vs BF16                | Use BF16 minimum; INT8/INT4 with calibration       |
| 10  | Ignoring cold start for serverless         | 10+ second model load time destroys latency             | Model caching, speculative loading, warm pools     |

---

## 17. Exercises

### Exercise 1: Latency Modelling

Given LLaMA-3 8B (8 × 10⁹ params) in BF16 on a single H100 (3.35 TB/s HBM BW,
989 TFLOP/s BF16):

1. Compute the bandwidth-bound TPOT at batch sizes B = 1, 8, 32, 128.
2. Compute the critical batch size $B^*$ where the system transitions from bandwidth-bound to compute-bound.
3. At what batch size does per-user TPOT start increasing?

### Exercise 2: KV Cache Budget

For LLaMA-3 70B (80 layers, GQA with 8 KV heads, $d_{\text{head}}$ = 128) in BF16
on 2× H100-80GB with TP=2:

1. Compute KV cache bytes per token per layer.
2. Compute KV cache bytes per token (all layers) per GPU.
3. Compute available KV memory per GPU (after model weights).
4. Find maximum concurrent tokens and maximum concurrent 2048-token requests.

### Exercise 3: Continuous vs. Static Batching Throughput

10 requests with output lengths: [50, 75, 100, 125, 150, 175, 200, 250, 300, 400].
Batch size = 10. TPOT = 10 ms.

1. Compute total time under static batching.
2. Compute total time under continuous batching (assume new requests arrive as slots free).
3. Calculate the throughput improvement.

### Exercise 4: PagedAttention Fragmentation

16 requests with sequence lengths drawn from Uniform(100, 2000). Block size = 16 tokens.
KV bytes per token = 320 KB. $s_{\max}$ = 4096.

1. Compute expected naive waste (pre-allocated max length) in GB.
2. Compute expected paged waste (only last-block fragmentation) in GB.
3. How many additional requests can you serve with the saved memory?

### Exercise 5: Speculative Decoding

Target model: 70B (BF16), draft model: 8B (BF16). Draft length $K = 4$.
Mean acceptance rate $\alpha = 0.75$. Draft model is 10× faster than target per token.

1. Compute expected tokens per speculative cycle.
2. Compute speed-up over standard autoregressive decoding.
3. What value of $\alpha$ makes speculation break even ($S = 1$)?

### Exercise 6: Prefill-Decode Disaggregation

A serving system processes requests with avg 500-token prompts and 200-token outputs.
Prefill throughput per GPU: 50,000 input tok/s. Decode throughput per GPU: 5,000 output tok/s.
Traffic: 100 req/s.

1. Compute GPU-seconds per request for prefill and decode separately.
2. Find optimal prefill:decode GPU ratio.
3. For 20 total GPUs, how many should be prefill vs. decode?

### Exercise 7: Cost Optimisation

LLaMA-3 70B on 2× H100 at \$3.00/GPU/hr. BF16 throughput: 2,500 tok/s.
INT4 throughput: 5,500 tok/s (with quantisation).

1. Compute CPM for BF16 and INT4.
2. For 1 billion tokens/day, compute daily and monthly cost savings of INT4 vs BF16.
3. At what quality-loss threshold (tokens needing regeneration) does INT4 stop being worth it?

### Exercise 8: Queue Stability

Poisson arrivals at $\lambda = 20$ req/s. Each request takes avg 2 seconds to serve.

1. What is the minimum number of GPUs for a stable queue?
2. For the queue to have $\rho < 0.7$, how many GPUs are needed?
3. With the number from (2), compute expected wait time $W_q$ using M/M/c.

---

## 18. Why This Matters

Serving is where the **rubber meets the road** in production ML. A model that runs at
1 tok/s costs 100× more per token than one optimised to run at 100 tok/s — and the
math in this chapter is how you close that gap.

Every section maps to a direct engineering lever:

| Concept               | Engineering Decision            | Impact               |
| --------------------- | ------------------------------- | -------------------- |
| Roofline model        | Choose batch size, quantisation | 2–10× throughput     |
| Queuing theory        | Capacity planning, GPU count    | SLO compliance       |
| PagedAttention math   | KV cache efficiency             | 2–4× more requests   |
| Speculative decoding  | Latency vs. memory tradeoff     | 2–3× lower latency   |
| Cost modelling        | Pricing, hardware selection     | 50–80 % cost savings |
| Scheduling algorithms | Fairness, tail latency          | P99 latency control  |

**The serving stack is the bridge between model quality and user experience.**
Without the mathematics of this chapter, you cannot answer the most basic production
question: "How many GPUs do I need, and how much will it cost?"

### Conceptual Bridge

This chapter completes the LLM lifecycle:

```
[Pre-training math] → [Architecture math] → [Training math] → [Fine-tuning math]
       ↓                     ↓                    ↓                    ↓
[Scaling laws]  →  [Attention/FFN]  →  [Optimisers]  →  [LoRA/RLHF]
                                                              ↓
                                                    [Inference & Serving] ← YOU ARE HERE
                                                              ↓
                                                    [Production systems]
```

The mathematics of serving — bandwidth analysis, queuing theory, cost modelling,
and systems optimisation — transform a trained model into a product that real
users can interact with at scale.

---

[← RAG Math and Retrieval](../12-RAG-Math-and-Retrieval/notes.md) | [Home](../../README.md)
