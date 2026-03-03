[← Language Model Probability](../05-Language-Model-Probability/notes.md) | [Home](../../README.md) | [Fine-Tuning Math →](../07-Fine-Tuning-Math/notes.md)

---

# Training at Scale

> _"Training a large language model is an exercise in distributed numerical optimisation under extreme constraints — memory, bandwidth, precision, and budget all compete, and the mathematics of each determines what is possible."_

## Overview

Training at scale means optimising billions to trillions of parameters across thousands of accelerators simultaneously. This is not just "bigger gradient descent" — fundamentally different engineering and mathematical challenges emerge at every order of magnitude. A single 70B-parameter model in bf16 requires ~140 GB of memory for weights alone; adding gradient buffers, optimiser states (Adam m and v), and activations pushes the total to 840+ GB — far exceeding any single GPU. The solution is a layered stack of mathematical techniques: mixed-precision arithmetic, gradient accumulation, learning rate scheduling, data/tensor/pipeline/sequence parallelism, memory-efficient optimisers, and careful numerical stabilisation. Every design decision that works at 1M parameters may catastrophically fail at 100B. This section derives the mathematics behind each technique, from the Adam optimiser update equations and their bias corrections, through ZeRO memory sharding algebra, to pipeline bubble fractions, MoE load balancing losses, and μP scaling rules that enable hyperparameter transfer across model sizes.

## Prerequisites

- Calculus: partial derivatives, chain rule, gradient computation
- Linear algebra: matrix multiplication, norms, decompositions
- Probability: expectation, variance, stochastic estimation
- Completed: [05-Language-Model-Probability](../05-Language-Model-Probability/notes.md) — cross-entropy loss, softmax, gradient derivation

## Companion Notebooks

| Notebook                           | Description                                                                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Adam optimiser, gradient clipping, LR schedules, parallelism simulation, ZeRO memory, pipeline bubbles, MoE routing, MFU calculation, LoRA parameter counting |
| [exercises.ipynb](exercises.ipynb) | Adam by hand, gradient clipping, ZeRO memory, pipeline bubble, MFU, LoRA params, critical batch size, cosine schedule                                         |

## Learning Objectives

After completing this section, you will:

- Derive the Adam optimiser update equations including bias correction and explain why AdamW decouples weight decay
- Compute gradient clipping operations and explain their role in preventing training divergence
- Implement and compare learning rate schedules: linear warmup, cosine decay, WSD (trapezoidal)
- Calculate memory requirements for training any model with mixed-precision and ZeRO sharding
- Explain data, tensor, pipeline, and sequence parallelism and compute their communication costs
- Derive the pipeline bubble fraction and determine optimal micro-batch count
- Explain MoE load balancing, compute the auxiliary loss, and analyse expert capacity overflow
- Calculate Model FLOPs Utilisation (MFU) and identify training bottlenecks
- Count LoRA/QLoRA trainable parameters and compare to full fine-tuning
- Apply μP scaling rules to transfer hyperparameters from small proxy to large target model

## Table of Contents

- [Training at Scale](#training-at-scale)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Is Training at Scale?](#11-what-is-training-at-scale)
    - [1.2 Why Scale Is Hard](#12-why-scale-is-hard)
    - [1.3 The Three Walls](#13-the-three-walls)
    - [1.4 Historical Scale Milestones](#14-historical-scale-milestones)
    - [1.5 Pipeline Position](#15-pipeline-position)
  - [2. Optimisation Foundations](#2-optimisation-foundations)
    - [2.1 Gradient Descent](#21-gradient-descent)
    - [2.2 Stochastic Gradient Descent](#22-stochastic-gradient-descent)
    - [2.3 Momentum](#23-momentum)
    - [2.4 Adam Optimiser](#24-adam-optimiser)
    - [2.5 AdamW](#25-adamw)
    - [2.6 Gradient Clipping](#26-gradient-clipping)
    - [2.7 Loss Landscape Geometry](#27-loss-landscape-geometry)
  - [3. Learning Rate Scheduling](#3-learning-rate-scheduling)
    - [3.1 Why Scheduling Matters](#31-why-scheduling-matters)
    - [3.2 Linear Warmup](#32-linear-warmup)
    - [3.3 Cosine Decay](#33-cosine-decay)
    - [3.4 WSD (Trapezoidal)](#34-wsd-trapezoidal)
    - [3.5 Cooldown and LR Rewinding](#35-cooldown-and-lr-rewinding)
    - [3.6 LR and Batch Size Interaction](#36-lr-and-batch-size-interaction)
  - [4. Parallelism Strategies](#4-parallelism-strategies)
    - [4.1 Why Single-GPU Fails](#41-why-single-gpu-fails)
    - [4.2 Data Parallelism (DP)](#42-data-parallelism-dp)
    - [4.3 FSDP / ZeRO](#43-fsdp--zero)
    - [4.4 Tensor Parallelism (TP)](#44-tensor-parallelism-tp)
    - [4.5 Pipeline Parallelism (PP)](#45-pipeline-parallelism-pp)
    - [4.6 Sequence Parallelism (SP)](#46-sequence-parallelism-sp)
    - [4.7 3D / 4D Parallelism](#47-3d--4d-parallelism)
    - [4.8 Communication Primitives](#48-communication-primitives)
  - [5. Memory Management](#5-memory-management)
    - [5.1 Memory Breakdown](#51-memory-breakdown)
    - [5.2 Activation Memory](#52-activation-memory)
    - [5.3 Gradient Checkpointing](#53-gradient-checkpointing)
    - [5.4 Mixed Precision (BF16/FP16)](#54-mixed-precision-bf16fp16)
    - [5.5 FP8 Training (2024–2026)](#55-fp8-training-20242026)
    - [5.6 Offloading](#56-offloading)
  - [6. Distributed Optimisation](#6-distributed-optimisation)
    - [6.1 Synchronous vs Asynchronous SGD](#61-synchronous-vs-asynchronous-sgd)
    - [6.2 Gradient Accumulation](#62-gradient-accumulation)
    - [6.3 Gradient All-Reduce Mathematics](#63-gradient-all-reduce-mathematics)
    - [6.4 Overlap Compute and Communication](#64-overlap-compute-and-communication)
    - [6.5 ZeRO Sharding Mathematics](#65-zero-sharding-mathematics)
  - [7. Numerical Stability at Scale](#7-numerical-stability-at-scale)
    - [7.1 Why Stability Is Harder at Scale](#71-why-stability-is-harder-at-scale)
    - [7.2 Loss Spikes](#72-loss-spikes)
    - [7.3 Gradient Explosion and Vanishing](#73-gradient-explosion-and-vanishing)
    - [7.4 Initialisation at Scale](#74-initialisation-at-scale)
    - [7.5 Attention Logit Growth](#75-attention-logit-growth)
    - [7.6 μP — Maximal Update Parametrisation](#76-μp--maximal-update-parametrisation)
  - [8. Mixture of Experts Training](#8-mixture-of-experts-training)
    - [8.1 MoE Architecture Basics](#81-moe-architecture-basics)
    - [8.2 Load Balancing Problem](#82-load-balancing-problem)
    - [8.3 Expert Capacity](#83-expert-capacity)
    - [8.4 Token Choice vs Expert Choice Routing](#84-token-choice-vs-expert-choice-routing)
    - [8.5 Expert Parallelism](#85-expert-parallelism)
    - [8.6 MoE Memory and Compute](#86-moe-memory-and-compute)
  - [9. Checkpointing and Fault Tolerance](#9-checkpointing-and-fault-tolerance)
    - [9.1 Why Fault Tolerance Matters](#91-why-fault-tolerance-matters)
    - [9.2 Checkpoint Contents](#92-checkpoint-contents)
    - [9.3 Checkpoint Frequency](#93-checkpoint-frequency)
    - [9.4 Elastic Training and Reproducibility](#94-elastic-training-and-reproducibility)
  - [10. Distributed Data Processing](#10-distributed-data-processing)
    - [10.1 Data Pipeline](#101-data-pipeline)
    - [10.2 Key Techniques](#102-key-techniques)
  - [11. Efficient Architectures](#11-efficient-architectures)
    - [11.1 Parameter Efficiency](#111-parameter-efficiency)
    - [11.2 FlashAttention](#112-flashattention)
    - [11.3 Activation Functions](#113-activation-functions)
    - [11.4 RMSNorm vs LayerNorm](#114-rmsnorm-vs-layernorm)
    - [11.5 Hardware-Aware Design](#115-hardware-aware-design)
  - [12. Monitoring and Diagnostics](#12-monitoring-and-diagnostics)
    - [12.1 Key Training Metrics](#121-key-training-metrics)
    - [12.2 Model FLOPs Utilisation (MFU)](#122-model-flops-utilisation-mfu)
    - [12.3 Hardware FLOPs](#123-hardware-flops)
  - [13. Post-Training and Fine-Tuning](#13-post-training-and-fine-tuning)
    - [13.1 Supervised Fine-Tuning (SFT)](#131-supervised-fine-tuning-sft)
    - [13.2 Parameter-Efficient Fine-Tuning (PEFT)](#132-parameter-efficient-fine-tuning-peft)
    - [13.3 Continual Pretraining](#133-continual-pretraining)
    - [13.4 RLHF at Scale](#134-rlhf-at-scale)
  - [14. Common Mistakes](#14-common-mistakes)
  - [15. Exercises](#15-exercises)
  - [16. Why This Matters](#16-why-this-matters)
    - [Conceptual Bridge](#conceptual-bridge)

---

## 1. Intuition

### 1.1 What Is Training at Scale?

Training at scale means optimising billions to trillions of parameters across thousands of accelerators simultaneously. This is not just "bigger gradient descent" — fundamentally different engineering and mathematical challenges emerge at every order of magnitude. Three interacting axes drive scale:

- **Model size** (parameters $N$): determines capacity and memory requirements
- **Data size** (tokens $D$): determines how well the model generalises
- **Compute** (FLOPs $C$): determines wall-clock time and cost; $C \approx 6ND$

Every decision that works at 1M parameters may catastrophically fail at 100B. Scale introduces new phenomena: emergent capabilities, sharp loss transitions, gradient pathologies, and memory walls.

### 1.2 Why Scale Is Hard

A single 70B model in bf16 requires ~140 GB memory — more than any single GPU. Forward + backward pass memory grows to 3–6× model weights (activations, gradients, optimiser states). Communication between thousands of GPUs introduces latency that can dominate compute time. Numerical precision: small errors compound across billions of operations. A training run of $10^{23}$ FLOPs costs millions of dollars — bugs are catastrophically expensive. Reproducibility is challenged by stochastic operations, hardware variation, and floating-point non-associativity.

### 1.3 The Three Walls

| Wall                   | Constraint                                                                 | Root Cause                                                      |
| ---------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **Memory wall**        | Model + activations + gradients + optimiser states don't fit in GPU memory | GPU HBM capacity (80 GB H100) ≪ model state (840 GB for 70B)    |
| **Compute wall**       | FLOPs required grow with $6ND$; hardware improvements lag demand           | Exponential scaling of compute vs linear hardware improvement   |
| **Communication wall** | Inter-GPU bandwidth orders of magnitude slower than compute                | NVLink ~600 GB/s vs compute ~312 TFLOPS; ratio worsens at scale |

### 1.4 Historical Scale Milestones

| Year | Model           | Parameters        | Hardware          | Key Innovation         |
| ---- | --------------- | ----------------- | ----------------- | ---------------------- |
| 2018 | BERT-Large      | 340M              | 64 TPUv2          | MLM pretraining        |
| 2019 | GPT-2           | 1.5B              | V100 GPUs         | Autoregressive scaling |
| 2020 | GPT-3           | 175B              | ~10K V100         | Few-shot emergence     |
| 2021 | Megatron-Turing | 530B              | 4480 A100         | 3D parallelism         |
| 2022 | PaLM            | 540B              | 6144 TPUv4        | Pathways system        |
| 2023 | LLaMA           | 65B               | 2048 A100         | Open + efficient       |
| 2024 | LLaMA-3         | 405B              | 16K H100          | GQA + long context     |
| 2024 | DeepSeek-V3     | 671B (37B active) | 2048 H800         | MoE + MLA              |
| 2025 | Frontier models | ~1–2T (est.)      | H200/B200         | FP8 + MoE              |
| 2026 | Next frontier   | Unknown           | Blackwell cluster | —                      |

### 1.5 Pipeline Position

```
Data → [Preprocessing] → Tokens → [Distributed Training] → θ* → [Evaluation] → Deployed Model
                                    ^^^^^^^^^^^^^^^^^^^^^
                                       THIS section
```

---

## 2. Optimisation Foundations

### 2.1 Gradient Descent

The fundamental update rule:

$$\boxed{\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}(\theta)}$$

where $\eta$ is the learning rate and $\nabla_\theta \mathcal{L}$ is the gradient of the loss with respect to all parameters.

| Variant        | Gradient Source | Pro            | Con                            |
| -------------- | --------------- | -------------- | ------------------------------ |
| Full-batch GD  | All data        | Exact gradient | Impractical for large datasets |
| SGD            | Single sample   | Fast per step  | Very noisy                     |
| Mini-batch SGD | $B$ samples     | Balanced       | Standard choice                |

### 2.2 Stochastic Gradient Descent

Mini-batch gradient estimator:

$$\hat{g} = \frac{1}{B} \sum_{i \in \text{batch}} \nabla_\theta \mathcal{L}_i(\theta)$$

**Properties:**

- **Unbiased**: $\mathbb{E}[\hat{g}] = \nabla_\theta \mathcal{L}(\theta)$
- **Variance**: $\text{Var}(\hat{g}) = \sigma^2 / B$ — larger batch → lower variance
- **Gradient noise as regularisation**: noise prevents convergence to sharp minima; improves generalisation
- **Linear scaling rule** (Goyal et al. 2017): if batch size increases $k\times$, scale learning rate $k\times$

### 2.3 Momentum

Accumulate exponential moving average of gradients:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$\theta_t = \theta_{t-1} - \eta m_t$$

$\beta_1 = 0.9$ typical. Momentum dampens oscillation and accelerates convergence along consistent gradient directions. Physical analogy: a ball rolling with inertia.

**Nesterov momentum**: compute gradient at the "lookahead" position $\theta + \beta m$; slightly better convergence guarantees.

### 2.4 Adam Optimiser

**Adam** (Kingma & Ba 2014) — the dominant optimiser for LLM training:

$$\boxed{m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(first moment)}}$$

$$\boxed{v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(second moment)}}$$

$$\boxed{\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \qquad \text{(bias correction)}}$$

$$\boxed{\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}}$$

| Hyperparameter | Value                        | Role                              |
| -------------- | ---------------------------- | --------------------------------- |
| $\beta_1$      | 0.9                          | First moment decay (momentum)     |
| $\beta_2$      | 0.999                        | Second moment decay (adaptive LR) |
| $\epsilon$     | $10^{-8}$                    | Numerical stability in division   |
| $\eta$         | $3 \times 10^{-4}$ (typical) | Base learning rate                |

**Why bias correction?** At $t=1$: $m_1 = (1-\beta_1) g_1 = 0.1 g_1$ (biased toward 0). Correction: $\hat{m}_1 = m_1 / (1-0.9^1) = g_1$ (unbiased).

**Adaptive per-parameter learning rates**: parameters with large historical gradients get smaller effective LR — automatic scaling.

### 2.5 AdamW

Standard Adam with L2 regularisation adds $\lambda\theta$ to the gradient, which interacts with the adaptive scaling. **AdamW** (Loshchilov & Hutter 2017) decouples weight decay:

$$\boxed{\theta_t = \theta_{t-1} - \eta\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} - \eta\lambda\theta_{t-1}}$$

Weight decay is applied directly to parameters, not through the gradient. Standard for all modern LLM training. Typical $\lambda = 0.01$–$0.1$.

### 2.6 Gradient Clipping

Clip gradient norm before update to prevent gradient explosion:

$$\boxed{\hat{g} = g \cdot \min\!\left(1, \frac{\tau}{\|g\|_2}\right)}$$

| Property                 | Detail                                                    |
| ------------------------ | --------------------------------------------------------- |
| Threshold $\tau$         | 1.0 typical                                               |
| When $\|g\|_2 \leq \tau$ | No clipping; $\hat{g} = g$                                |
| When $\|g\|_2 > \tau$    | Scale down: $\|\hat{g}\|_2 = \tau$; preserves direction   |
| Scope                    | Global norm clipping (across all parameters jointly)      |
| Critical for             | Preventing divergence from outlier batches or loss spikes |

### 2.7 Loss Landscape Geometry

The loss surface $\mathcal{L}(\theta)$ is a high-dimensional non-convex function with:

- **Local minima**: approximately equally good at scale (loss-wise)
- **Saddle points**: dominant critical points in high dimensions; SGD escapes naturally
- **Sharp minima**: generalise worse than flat minima; SGD noise biases toward flat minima
- **Hessian** $H = \nabla^2 \mathcal{L}$: eigenvalue spectrum reveals curvature; large eigenvalues = sharp directions

---

## 3. Learning Rate Scheduling

### 3.1 Why Scheduling Matters

Constant LR is a compromise: too large → divergence; too small → slow convergence. The optimal strategy uses large LR early (fast progress when far from minimum) and small LR late (fine-grained convergence near minimum). Schedule shape often matters as much as the LR value itself.

### 3.2 Linear Warmup

Increase LR linearly from 0 to $\eta_{\max}$ over $W$ steps:

$$\boxed{\eta_t = \eta_{\max} \cdot \frac{t}{W}, \quad t \leq W}$$

Without warmup, large random initial gradients combined with large LR cause divergence. $W = 1000$–$4000$ steps typical. Warmup allows optimiser moments (Adam $m$ and $v$) to stabilise before full LR is applied.

### 3.3 Cosine Decay

$$\boxed{\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\!\left(\frac{\pi(t-W)}{T-W}\right)\right)}$$

Smooth decay from $\eta_{\max}$ to $\eta_{\min}$ over $T$ total steps. $\eta_{\min}$ typically $\eta_{\max}/10$ or 0. Most common LLM schedule: **linear warmup + cosine decay**. Used by GPT-3, LLaMA, Mistral, Gemma, Falcon, and all major models.

### 3.4 WSD (Trapezoidal)

Warmup-Stable-Decay — three phases:

$$\eta_t = \begin{cases} \eta_{\max} \cdot t/W & t \leq W \quad \text{(warmup)} \\ \eta_{\max} & W < t \leq T-D \quad \text{(stable)} \\ \eta_{\max} \cdot (T-t)/D & T-D < t \leq T \quad \text{(decay)} \end{cases}$$

MiniCPM (2024): WSD enables continual training — add new data in stable phase, then decay to consolidate. Advantage over cosine: can extend training without restarting LR schedule. Becoming standard for models trained on growing data streams (2024–2026).

### 3.5 Cooldown and LR Rewinding

- **Cooldown**: brief LR reduction at end of training; sharpens convergence
- **LR rewinding** (Frankle & Carlin 2019): reset LR to initial value and retrain subnetwork
- **Cyclical LR** (Smith 2017): oscillate LR between bounds; explores multiple basins
- **1-Cycle LR**: one cycle with warmup + decay; effective for fine-tuning

### 3.6 LR and Batch Size Interaction

| Rule                | Formula                                                        | When                                          |
| ------------------- | -------------------------------------------------------------- | --------------------------------------------- |
| Linear scaling      | $\eta \propto B$                                               | Default; double batch → double LR             |
| Square root         | $\eta \propto \sqrt{B}$                                        | More conservative; small models               |
| Critical batch size | $B_{\text{crit}} \approx B_{\text{noise}} / B_{\text{signal}}$ | Beyond $B_{\text{crit}}$: diminishing returns |

---

## 4. Parallelism Strategies

### 4.1 Why Single-GPU Fails

| Model        | bf16 Weights | + Adam States | Largest GPU |
| ------------ | ------------ | ------------- | ----------- |
| LLaMA-3 8B   | 16 GB        | 96 GB         | H100 80 GB  |
| LLaMA-3 70B  | 140 GB       | 840 GB        | H100 80 GB  |
| LLaMA-3 405B | 810 GB       | 4.9 TB        | H100 80 GB  |

Must distribute across multiple GPUs. Multiple parallelism strategies needed simultaneously.

### 4.2 Data Parallelism (DP)

Replicate full model on each GPU; split data across GPUs:

1. Each GPU computes gradient on its batch shard
2. **All-reduce** gradients across GPUs → each GPU gets the average gradient
3. All GPUs apply identical update → stay synchronised

Scales linearly in throughput. Does **not** reduce per-GPU memory (full model on each GPU). All-reduce cost: $O\!\left(\frac{2(K-1)}{K} \times |W|\right)$ for $K$ GPUs.

### 4.3 FSDP / ZeRO

**ZeRO** (Rajbhandari et al. 2020) — Zero Redundancy Optimiser. Three stages of sharding:

| Stage         | What's Sharded   | Memory per GPU                 | Communication                      |
| ------------- | ---------------- | ------------------------------ | ---------------------------------- |
| ZeRO-1        | Optimiser states | $\frac{4N+4N+4N}{K} + 2N + 2N$ | Same as DP                         |
| ZeRO-2        | + Gradients      | $\frac{4N+4N+4N+2N}{K} + 2N$   | Same as DP                         |
| ZeRO-3 / FSDP | + Parameters     | $\frac{16N}{K}$                | $3\times \lvert W \rvert$ per step |

All-gather parameters before each layer's forward/backward; re-shard after. Trade: more communication for much less memory.

### 4.4 Tensor Parallelism (TP)

Split individual weight matrices across GPUs along a dimension. For linear layer $Y = XW$: split $W$ into $[W_1 \mid W_2]$ along columns → $Y = [XW_1 \mid XW_2]$.

- Megatron-LM (Shoeybi et al. 2019): tensor parallel for attention and FFN
- **Attention**: split heads across GPUs (each GPU owns $H/k$ heads)
- **FFN**: split hidden dimension across GPUs
- Requires high-bandwidth intra-node interconnect (NVLink)
- Typical: $\text{TP}=8$ within a single node (8× A100/H100)

### 4.5 Pipeline Parallelism (PP)

Split model layers into stages; assign each stage to different GPUs:

- GPU 0: layers 1–8; GPU 1: layers 9–16; etc.
- Pipeline micro-batching: split mini-batch into micro-batches to fill pipeline

**Pipeline bubble fraction:**

$$\boxed{\text{bubble} = \frac{p-1}{p + m - 1}}$$

$p$ = pipeline stages; $m$ = micro-batches. Larger $m$ → smaller bubble.

| $p$ | $m$ | Bubble |
| --- | --- | ------ |
| 4   | 4   | 42.9%  |
| 4   | 8   | 27.3%  |
| 4   | 16  | 15.8%  |
| 4   | 32  | 8.6%   |

**1F1B schedule**: interleave forward and backward passes; reduces activation memory.

### 4.6 Sequence Parallelism (SP)

Split the sequence dimension across GPUs:

- Attention is the bottleneck: $O(n^2)$ memory in attention matrix
- **Ring attention** (Liu et al. 2023): each GPU holds $n/k$ tokens; send K, V around ring for full attention
- Flash-Decoding: parallelise across sequence for faster decoding
- Enables training on very long sequences (1M+ tokens)

### 4.7 3D / 4D Parallelism

Combine DP + TP + PP simultaneously: "3D parallelism" (Megatron-DeepSpeed). Add SP for long sequences: "4D parallelism".

| Component | Scope               | Typical Value     | Constraint       |
| --------- | ------------------- | ----------------- | ---------------- |
| TP        | Intra-node          | 8                 | NVLink bandwidth |
| PP        | Inter-node          | 8–16              | Pipeline bubble  |
| DP        | Remaining GPUs      | Total / (TP × PP) | Batch size       |
| EP        | Expert distribution | N_experts / k     | MoE all-to-all   |

LLaMA-3 405B training: TP=8, PP=16, DP=many; across 16K H100s.

### 4.8 Communication Primitives

| Primitive          | Operation                                       | Use Case                    |
| ------------------ | ----------------------------------------------- | --------------------------- |
| **All-reduce**     | Sum/average tensor; result on all GPUs          | Data-parallel gradient sync |
| **All-gather**     | Collect sharded tensors; full tensor everywhere | FSDP parameter retrieval    |
| **Reduce-scatter** | Sum across GPUs; distribute shards              | FSDP gradient sync          |
| **All-to-all**     | Each GPU sends different data to each other     | MoE expert routing          |

Ring algorithm for all-reduce: $O\!\left(\frac{2(K-1)}{K} \times \text{size}\right)$; near-optimal bandwidth utilisation. NCCL: standard collective communication library.

---

## 5. Memory Management

### 5.1 Memory Breakdown

| Component                   | Size              | Notes                             |
| --------------------------- | ----------------- | --------------------------------- |
| Model weights               | $2N$ bytes (bf16) | $N$ = parameter count             |
| Gradient buffer             | $2N$ bytes (bf16) | Same size as weights              |
| FP32 master weights         | $4N$ bytes (fp32) | Required for mixed-precision      |
| Adam $m$ (momentum)         | $4N$ bytes (fp32) | First moment; fp32 for stability  |
| Adam $v$ (variance)         | $4N$ bytes (fp32) | Second moment; fp32               |
| **Total (mixed precision)** | **$16N$ bytes**   | **Rule of thumb**                 |
| Activations                 | Variable          | $O(B \times n \times d \times L)$ |
| Peak: forward + backward    | +$2$–$3N$         | Temporary buffers                 |

**Examples:**

- LLaMA-3 8B: $16 \times 8\text{B} = 128$ GB minimum (before activations)
- LLaMA-3 70B: $16 \times 70\text{B} = 1{,}120$ GB

### 5.2 Activation Memory

Activations stored during forward pass for use in backward pass:

- Per-layer: $O(B \times n \times d)$ for batch $B$, sequence length $n$, dimension $d$
- Full model: $O(B \times n \times d \times L)$ for $L$ layers
- Dominates for large batch sizes and long sequences

### 5.3 Gradient Checkpointing

Don't store all activations; recompute during backward pass:

| Approach                                   | Memory                                   | Extra Compute |
| ------------------------------------------ | ---------------------------------------- | ------------- |
| No checkpointing                           | $O(L \times B \times n \times d)$        | 0%            |
| Full checkpointing ($\sqrt{L}$ boundaries) | $O(\sqrt{L} \times B \times n \times d)$ | ~33%          |
| Selective recomputation                    | Between                                  | ~10–15%       |

Selective recomputation: only recompute cheap ops (layernorm, activation functions); store expensive ones (attention). Essential for training large models; universally used.

### 5.4 Mixed Precision (BF16/FP16)

| Component        | Precision | Memory |
| ---------------- | --------- | ------ |
| Master weights   | FP32      | $4N$   |
| Forward/backward | BF16      | $2N$   |
| Gradients        | BF16      | $2N$   |
| Adam states      | FP32      | $8N$   |

- **BF16** preferred over FP16: same exponent range as FP32 (8 bits); no loss scaling needed
- **FP16**: narrower dynamic range; requires loss scaling to prevent gradient underflow (min ≈ $6 \times 10^{-8}$)
- All modern LLM training uses BF16 (A100, H100, TPUv4+)

### 5.5 FP8 Training (2024–2026)

| Format | Exponent | Mantissa | Use                       |
| ------ | -------- | -------- | ------------------------- |
| E4M3   | 4 bits   | 3 bits   | Forward pass (range)      |
| E5M2   | 5 bits   | 2 bits   | Backward pass (precision) |

- H100 FP8 tensor cores: up to 2× throughput vs BF16
- Requires per-tensor or per-block scaling to avoid precision loss
- DeepSeek-V3 (2024): FP8 training throughout; significant cost reduction
- FlashAttention-3: FP8 Q, K, V with FP32 softmax accumulation
- 2025–2026: FP8 training becoming standard for frontier models

### 5.6 Offloading

- **CPU offloading**: move optimiser states to CPU RAM; PCIe bandwidth bottleneck (64 GB/s vs 2 TB/s NVLink)
- **ZeRO-Infinity**: offload to CPU + NVMe SSD; enables training on limited GPU memory
- Used for fine-tuning very large models on limited hardware (LoRA + CPU offload)

---

## 6. Distributed Optimisation

### 6.1 Synchronous vs Asynchronous SGD

| Property             | Synchronous                     | Asynchronous             |
| -------------------- | ------------------------------- | ------------------------ |
| Gradient freshness   | All current                     | Some stale               |
| Straggler problem    | Yes (slowest determines speed)  | No                       |
| Convergence analysis | Standard                        | Harder (stale gradients) |
| Used for LLMs        | ✓ (with micro-batch pipelining) | Rarely                   |

### 6.2 Gradient Accumulation

Simulate larger batch by accumulating gradients over $G$ steps before updating:

$$\boxed{g_{\text{effective}} = \frac{1}{G}\sum_{k=1}^{G} g_k}$$

Effective batch size = $G \times B \times K_{\text{DP}}$ (gradient accumulation steps × micro-batch × DP replicas).

No communication overhead vs actual large batch; mathematically identical result.

### 6.3 Gradient All-Reduce Mathematics

Ring all-reduce algorithm:

1. **Reduce-scatter**: each GPU accumulates partial sum for its shard
2. **All-gather**: each GPU broadcasts its shard; everyone gets full result

Total data transmitted: $\frac{2(K-1)}{K} \times |W| \approx 2 \times |W|$.

| Component               | Bandwidth | LLaMA-3 8B (bf16 = 16 GB) |
| ----------------------- | --------- | ------------------------- |
| NVLink (intra-node)     | ~600 GB/s | 53 ms                     |
| InfiniBand (inter-node) | ~50 GB/s  | 640 ms                    |

### 6.4 Overlap Compute and Communication

Gradient all-reduce can overlap with backward pass computation:

- While computing gradients for layer $l$, all-reduce layer $l+1$ gradients
- **Bucket-based all-reduce**: group small tensors into buckets; reduce bucket when full
- PyTorch DDP: automatic bucketing and overlap

### 6.5 ZeRO Sharding Mathematics

$N$ parameters, $K$ GPUs, ZeRO-3:

$$\text{Memory per GPU} = \frac{(2 + 2 + 4 + 4 + 4) \times N}{K} = \frac{16N}{K}$$

| Metric                      | Standard DP                | ZeRO-3                     |
| --------------------------- | -------------------------- | -------------------------- |
| Memory per GPU              | $16N$                      | $16N/K$                    |
| Communication per step      | $2 \times \lvert W \rvert$ | $3 \times \lvert W \rvert$ |
| Memory reduction ($K=1000$) | 1×                         | 1000×                      |

---

## 7. Numerical Stability at Scale

### 7.1 Why Stability Is Harder at Scale

More layers = more gradient vanishing/explosion opportunities. More parameters = more loss landscape pathologies. Mixed precision = reduced dynamic range. Distributed computation = floating-point non-associativity (different GPU orderings → different results).

### 7.2 Loss Spikes

Sudden increase in loss during training; common at large scale.

| Aspect     | Detail                                                         |
| ---------- | -------------------------------------------------------------- |
| Causes     | Outlier data batches, gradient explosion, numerical overflow   |
| Detection  | Monitor loss, gradient norm, weight norm per step              |
| Recovery   | Rollback to checkpoint; skip bad batch; reduce LR temporarily  |
| Prevention | Gradient clipping, careful data preprocessing, spike detection |

### 7.3 Gradient Explosion and Vanishing

| Problem   | Symptom            | Solutions                                               |
| --------- | ------------------ | ------------------------------------------------------- |
| Explosion | $\|g\| \to \infty$ | Gradient clipping, careful LR schedule                  |
| Vanishing | $\|g\| \to 0$      | Residual connections, layer normalisation, careful init |

RMSNorm + pre-norm architecture largely solves vanishing at LLM scale.

### 7.4 Initialisation at Scale

| Method                 | Distribution                                                                                                            | Designed For                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Xavier / Glorot        | $W \sim U\!\left(-\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}, +\sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$ | Linear layers; variance preservation |
| He                     | $W \sim \mathcal{N}(0, 2/n_{\text{in}})$                                                                                | ReLU activations                     |
| GPT-2 residual scaling | Scale residual branch by $1/\sqrt{2L}$                                                                                  | Deep transformers                    |

GPT-2 residual scaling: each residual addition $\text{Var}(x + f(x)) = \text{Var}(x) + \text{Var}(f(x))$; without scaling, variance grows as $L$. With $1/\sqrt{2L}$ scaling, total variance ≈ constant regardless of depth.

### 7.5 Attention Logit Growth

Without stabilisation, attention logits $\|QK^\top\|$ grow with depth and sequence length. Causes: weight norms increase during training; logits saturate softmax.

**QK-Norm** (2023): apply L2 normalisation to $Q$ and $K$ before dot product:
$(Q/\|Q\|) \cdot (K/\|K\|)$ bounded in $[-1, 1] \times d_k$. Prevents softmax saturation. Used by Gemma-2, LLaMA-3.1 variants, many 2024–2026 models.

### 7.6 μP — Maximal Update Parametrisation

(Yang et al. 2022) Standard parametrisation: feature learning decreases as width increases. μP rescales weights to maintain constant feature learning at any width.

| Component       | Standard                            | μP                                                      |
| --------------- | ----------------------------------- | ------------------------------------------------------- |
| Input embedding | $W \sim \mathcal{N}(0, 1/d)$        | $W \sim \mathcal{N}(0, 1)$                              |
| Hidden layers   | $W \sim \mathcal{N}(0, 1/\sqrt{d})$ | $W \sim \mathcal{N}(0, 1/d)$                            |
| Output layer    | $W \sim \mathcal{N}(0, 1/\sqrt{d})$ | $W \sim \mathcal{N}(0, 1/d)$; scale by $1/\text{width}$ |
| Learning rate   | Same for all widths                 | $\eta \propto 1/\text{width}$                           |

Main benefit: **hyperparameters transfer from small proxy model to large model**. Tune LR, $\beta$, etc. on 10M model → apply directly to 10B model.

---

## 8. Mixture of Experts Training

### 8.1 MoE Architecture Basics

Replace dense FFN with $N$ expert FFNs; route each token to top-$k$ experts:

- **Router**: learned linear projection → softmax → select top-$k$ experts
- Only $k/N$ fraction of parameters active per token → sparse computation
- Total params: $N \times$ (FFN size); active params: $k \times$ (FFN size) per token

### 8.2 Load Balancing Problem

Without constraints, the router collapses — sends all tokens to the same expert(s). Load balancing auxiliary loss:

$$\boxed{\mathcal{L}_{\text{aux}} = \alpha \sum_{i=1}^{N} f_i \cdot P_i}$$

$f_i$ = fraction of tokens routed to expert $i$; $P_i$ = average router probability for expert $i$. Minimising this correlation encourages uniform distribution.

### 8.3 Expert Capacity

| Parameter          | Definition                      | Typical                 |
| ------------------ | ------------------------------- | ----------------------- |
| Capacity $C$       | Max tokens per expert per batch | $\text{CF} \times T/N$  |
| Capacity factor CF | Slack for imbalance             | 1.25                    |
| Overflow           | Tokens exceeding $C$            | Dropped (not processed) |

### 8.4 Token Choice vs Expert Choice Routing

| Strategy                         | Mechanism                                          | Load Balance             | Dropped Tokens                       |
| -------------------------------- | -------------------------------------------------- | ------------------------ | ------------------------------------ |
| Token choice                     | Each token picks top-$k$ experts                   | May be imbalanced        | Possible                             |
| Expert choice (Zhou et al. 2022) | Each expert picks top-$C$ tokens                   | Perfect by construction  | None (but some tokens get 0 experts) |
| DeepSeek MoE (2024)              | Fine-grained (256 experts, top-8) + shared experts | Good with auxiliary loss | Minimal                              |

### 8.5 Expert Parallelism

Distribute experts across GPUs: each GPU owns $N/k$ experts.

- Forward: all-to-all to route tokens to correct GPU
- Backward: all-to-all to return gradients
- Communication: $O(N \times \text{token\_size})$ per layer
- DeepSeek-V3: EP=64; 64 GPUs share 256 experts; 4 per GPU

### 8.6 MoE Memory and Compute

- **Memory**: all expert parameters loaded even if not active (during training)
- **Active compute**: $k/N \times$ dense compute; but all $N$ experts updated
- **Gradient sparsity**: expert $i$ receives gradient only from tokens routed to it
- 2024–2026 MoE models: DeepSeek-V3 (671B/37B active), Mixtral 8×22B, Grok-1

---

## 9. Checkpointing and Fault Tolerance

### 9.1 Why Fault Tolerance Matters

LLaMA-3 405B: ~90 days training; thousands of GPUs. GPU failure rate: ~0.1–1% per GPU per day; 10K GPUs → 10–100 failures/day. Without fault tolerance: restart from scratch = days of lost work.

### 9.2 Checkpoint Contents

| Component         | Size                 | Purpose                           |
| ----------------- | -------------------- | --------------------------------- |
| Model weights     | $2N$ bytes           | All shards                        |
| Optimiser states  | $8N$ bytes           | Adam $m$, $v$ (largest component) |
| LR schedule state | Tiny                 | Current step, warmup phase        |
| RNG states        | Tiny                 | For reproducibility               |
| Data loader state | Small                | Which samples seen                |
| **Total**         | **$\\sim10N$ bytes** | Per checkpoint (weights + Adam)   |

### 9.3 Checkpoint Frequency

More frequent → less work lost, more I/O overhead, more storage. Typical: every 500–1000 steps. **Async checkpointing**: snapshot to CPU RAM; write to storage in background.

### 9.4 Elastic Training and Reproducibility

- **Elastic training**: dynamically change worker count without restarting (cloud spot instances)
- **Determinism**: floating-point non-associativity means $(a+b)+c \neq a+(b+c)$; all-reduce order varies
- **Practical reproducibility**: same loss curve shape even if not bit-exact

---

## 10. Distributed Data Processing

### 10.1 Data Pipeline

Data loading must never let GPUs sit idle. A100 peak: ~312 TFLOPS. Solution: prefetch data in parallel CPU processes.

### 10.2 Key Techniques

| Technique           | Description                                                          |
| ------------------- | -------------------------------------------------------------------- |
| Dataset sharding    | Split dataset into shards; each worker processes its shard           |
| Data mixing         | Control proportion of each domain (web, code, books, math) per batch |
| Curriculum learning | Start with easier/cleaner data; introduce harder data later          |
| Exact deduplication | Hash-based removal of exact duplicate documents                      |
| Fuzzy deduplication | MinHash + LSH for near-duplicate removal                             |
| Packed sequences    | Concatenate documents with EOS separator; no wasted padding          |
| Pre-tokenisation    | Tokenise entire dataset before training; store token IDs on disk     |

Storage: 15T tokens × 2 bytes/token (int16) = 30 TB raw token storage.

---

## 11. Efficient Architectures

### 11.1 Parameter Efficiency

| Technique                         | Savings             | Trade-off                |
| --------------------------------- | ------------------- | ------------------------ |
| Weight tying (embed ↔ LM head)    | Saves $2Nd$ bytes   | Slight quality reduction |
| Layer sharing (ALBERT)            | $L\times$ reduction | Quality reduction        |
| Low-rank factors ($W \approx UV$) | Rank reduction      | Approximation error      |

### 11.2 FlashAttention

| Version    | Key Feature                                                                       | Speedup      |
| ---------- | --------------------------------------------------------------------------------- | ------------ |
| FA1 (2022) | Tiled attention; never materialise $A \in \mathbb{R}^{n \times n}$; $O(n)$ memory | 2–4×         |
| FA2 (2023) | Better tiling; full backward pass                                                 | 2× over FA1  |
| FA3 (2024) | H100 optimised; FP8 support                                                       | ~3× over FA2 |

### 11.3 Activation Functions

| Function | Formula                               | Used By                                          |
| -------- | ------------------------------------- | ------------------------------------------------ |
| ReLU     | $\max(0, x)$                          | Early models; dead neuron problem                |
| GELU     | $x \Phi(x)$                           | BERT, GPT-2/3                                    |
| SwiGLU   | $\text{Swish}(xW + b) \odot (xV + c)$ | LLaMA, PaLM, Gemma, Mistral (2023–2026 standard) |

SwiGLU requires 3 matrices (up, gate, down) vs 2 for standard FFN; increases params by 50%. Compensate by reducing hidden dim: $4d \to 8d/3 \approx 2.67d$.

### 11.4 RMSNorm vs LayerNorm

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta \qquad \text{(2 params per dim)}$$

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma \qquad \text{(1 param per dim)}$$

RMSNorm: ~10% faster; similar quality; used by LLaMA, Gemma, Mistral, Qwen. Pre-norm placement (before attention/FFN) more stable than post-norm.

### 11.5 Hardware-Aware Design

| Alignment         | Value           | Why                    |
| ----------------- | --------------- | ---------------------- |
| Weight dimensions | Multiple of 128 | Tensor core tile size  |
| Head dimension    | 64 or 128       | FlashAttention optimal |
| FFN hidden dim    | Multiple of 256 | Memory efficiency      |
| Vocabulary size   | Nearest 64      | Padding alignment      |

---

## 12. Monitoring and Diagnostics

### 12.1 Key Training Metrics

| Metric           | What It Measures         | Healthy Range       |
| ---------------- | ------------------------ | ------------------- |
| Loss curve       | Training/validation loss | Smooth descent      |
| Gradient norm    | $\|g\|_2$ per step       | Stable ~1.0         |
| Learning rate    | Effective LR             | Matches schedule    |
| Throughput       | Tokens/second            | >50% MFU            |
| Weight norms     | Per-layer $\|W\|$        | Stable; not growing |
| Activation norms | Per-layer $\|h\|$        | Signal propagation  |
| Loss spikes      | Sudden jumps             | Should be rare      |

### 12.2 Model FLOPs Utilisation (MFU)

$$\boxed{\text{MFU} = \frac{\text{actual FLOPs/sec}}{\text{theoretical peak FLOPs/sec}}}$$

| Model       | Reported MFU      | Hardware    |
| ----------- | ----------------- | ----------- |
| PaLM        | 46%               | TPUv4       |
| LLaMA-3     | ~38%              | H100        |
| Good target | 35–55%            | —           |
| Below 20%   | Severe bottleneck | Investigate |

### 12.3 Hardware FLOPs

For transformer: FLOPs per step ≈ $6ND$ ($N$ = parameters, $D$ = tokens in batch). Factor 6: 2 for multiply-add, 3 for backward (2× gradient computation).

More precise per layer:

- Attention: $4Bnd^2 + 2Bn^2 d$
- FFN: $16Bnd^2$ (assuming $4d$ hidden)

---

## 13. Post-Training and Fine-Tuning

### 13.1 Supervised Fine-Tuning (SFT)

Train on (instruction, response) pairs. Much smaller dataset (10K–1M examples). Lower LR ($10^{-5}$ to $10^{-6}$). Short training (1–3 epochs).

### 13.2 Parameter-Efficient Fine-Tuning (PEFT)

**LoRA** (Hu et al. 2021):

$$\boxed{\Delta W = BA, \quad B \in \mathbb{R}^{m \times r}, \; A \in \mathbb{R}^{r \times n}, \; r \ll \min(m, n)}$$

Freeze $W$; train only $B$ and $A$. Trainable parameters: $r(m+n)$ vs $mn$ for full fine-tune. Typically 0.1–1% of total parameters.

| Method                | Key Idea                        | Memory Savings              |
| --------------------- | ------------------------------- | --------------------------- |
| LoRA                  | Low-rank weight updates         | 10–100× fewer params        |
| QLoRA (Dettmers 2023) | 4-bit NF4 base + LoRA in BF16   | 4× memory reduction         |
| DoRA (Liu 2024)       | Decompose magnitude + direction | Better quality at same rank |

**QLoRA**: fine-tune LLaMA-3 70B on single A100 80GB. NF4: optimal quantisation for normally-distributed weights. Double quantisation: quantise the quantisation constants (~0.37 bits/param saved).

### 13.3 Continual Pretraining

Continue on new domain data. Risk: catastrophic forgetting. Mitigation: replay buffer (mix new + original data); EWC regularisation; WSD schedule.

### 13.4 RLHF at Scale

| Method               | Models in Memory               | Complexity                |
| -------------------- | ------------------------------ | ------------------------- |
| PPO                  | 4 (policy, ref, reward, value) | High; 4× model size       |
| DPO                  | 2 (policy, reference)          | Lower; standard 2024–2026 |
| GRPO (DeepSeek 2025) | 1 + group samples              | Lowest; no value model    |

---

## 14. Common Mistakes

| #   | Mistake                                      | Why It's Wrong                                          | Fix                                               |
| --- | -------------------------------------------- | ------------------------------------------------------- | ------------------------------------------------- |
| 1   | "Bigger batch = faster training"             | Beyond $B_{\text{crit}}$, diminishing gradient signal   | Find $B_{\text{crit}}$; use gradient accumulation |
| 2   | "Linear scaling rule always holds"           | Breaks down beyond $B_{\text{crit}}$                    | Test empirically; monitor loss per compute        |
| 3   | "Just add more GPUs"                         | Communication overhead dominates with wrong parallelism | Profile compute vs communication; tune strategy   |
| 4   | "FP16 is fine for LLMs"                      | Dynamic range too small; needs loss scaling             | Use BF16; switch to FP8 with scaling              |
| 5   | "Checkpoint every N steps is enough"         | Silent GPU errors produce wrong weights                 | Validate checkpoint quality; compare loss         |
| 6   | "MoE is always more efficient"               | All-to-all communication + load imbalance               | Profile communication; implement load balancing   |
| 7   | "LoRA rank doesn't matter much"              | Too low: underfits; too high: approaches full-FT cost   | Search $r \in \{4,8,16,32,64\}$                   |
| 8   | "Gradient clipping prevents all instability" | Clips norm not direction                                | Combine with μP init and QK-Norm                  |
| 9   | "Same hyperparams work at every scale"       | Optimal LR, batch, $\beta$ change with size             | Use μP proxy tuning                               |
| 10  | "Training loss = model quality"              | Loss measures next-token prediction only                | Evaluate on diverse benchmarks                    |

---

## 15. Exercises

1. **Adam update by hand** — given $g = [0.5, -1.2, 0.3]$, $\beta_1=0.9$, $\beta_2=0.999$, $\eta=0.001$, $t=1$: compute $m_1$, $v_1$, $\hat{m}_1$, $\hat{v}_1$, $\Delta\theta$
2. **Gradient clipping** — gradient vector $g = [3.0, 4.0, 0.0]$, $\tau=1.0$: compute $\|g\|$; clipped gradient; verify $\|\hat{g}\| = \tau$
3. **ZeRO memory** — 7B model, 8 GPUs, ZeRO-3, bf16 + fp32 Adam: compute memory per GPU
4. **Pipeline bubble** — $p=4$ stages, $m=8$ micro-batches: compute bubble fraction; compare to $m=16$
5. **MFU calculation** — 1024 A100s, 312 TFLOPS each, 7B model, 2M tokens/step, 1.2s/step: compute MFU
6. **LoRA parameter count** — $d=4096$, $r=16$, Q and V projections, 32-layer model: compute trainable params
7. **Critical batch size** — $B_{\text{noise}} = 10^6$, $B_{\text{signal}} = 2 \times 10^4$: compute $B_{\text{crit}}$
8. **Cosine schedule** — $\eta_{\max}=3 \times 10^{-4}$, $\eta_{\min}=3 \times 10^{-5}$, $W=2000$, $T=100\text{K}$: compute $\eta$ at $t = 1000, 5000, 50000, 99000$

---

## 16. Why This Matters

| Aspect            | Impact                                                                            |
| ----------------- | --------------------------------------------------------------------------------- |
| **Capability**    | Scale is the primary driver of capability; training at scale = frontier models    |
| **Cost**          | LLaMA-3 405B: ~$10M training run; wrong parallelism strategy = 2× cost            |
| **Accessibility** | QLoRA enables fine-tuning 70B on a single GPU; democratises adaptation            |
| **Efficiency**    | FP8 + FlashAttention-3 + GQA = 3–5× cost reduction vs 2022                        |
| **Reasoning**     | Test-time compute scaling (o1, R1) requires efficient inference infrastructure    |
| **Safety**        | RLHF/DPO at scale = how alignment is operationalised; training stability critical |
| **Open source**   | Training efficiency determines who can train frontier models; efficiency = access |
| **Environment**   | Frontier runs consume megawatts; efficiency = sustainability                      |
| **Speed**         | Faster training = faster iteration = faster capability development                |
| **MoE**           | Sparse MoE enables trillion-param models at same compute as 30B dense             |

---

### Conceptual Bridge

Training at scale is where all the mathematics comes together. Probability theory → loss function; linear algebra → gradient computation; distributed systems → parallelism; numerical analysis → stability.

**Next**: [Fine-Tuning Math](../07-Fine-Tuning-Math/notes.md) — how to adapt pretrained models to specific tasks efficiently with LoRA, QLoRA, and alignment techniques.

```
Data → [Distributed Training] → θ* → [Fine-Tuning] → θ_task
        ^^^^^^^^^^^^^^^^^^^^^
           THIS section
```

---

[← Language Model Probability](../05-Language-Model-Probability/notes.md) | [Home](../../README.md) | [Fine-Tuning Math →](../07-Fine-Tuning-Math/notes.md)
