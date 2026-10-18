[<- Language Model Probability](../05-Language-Model-Probability/notes.md) | [Home](../../README.md) | [Fine-Tuning Math ->](../07-Fine-Tuning-Math/notes.md)

---

# Training at Scale

Training at scale is the mathematics of keeping the same language-model objective useful when the model, data, batch size, memory footprint, communication pattern, and wall-clock budget are all large enough to break naive code.

## Overview

The previous section derived the next-token probability objective. This section asks what changes when that objective is optimized over billions of parameters and trillions of token updates. The answer is not a new loss. The answer is a stack of constraints:

```text
loss math -> optimizer state -> batch schedule -> memory budget -> parallelism -> communication -> checkpoint/restart -> validation signal
```

The central skill is quantitative accounting. You should be able to estimate memory before launching, compute effective batch size from the training configuration, reason about why ZeRO/FSDP helps, explain the pipeline bubble, compute approximate training FLOPs, and debug a run whose loss or throughput looks wrong.

## Prerequisites

- Cross-entropy and next-token probability from [05-Language-Model-Probability](../05-Language-Model-Probability/notes.md)
- Gradients, vector norms, and stochastic optimization
- Matrix multiplication shapes from attention and MLP blocks
- Basic probability for mini-batch gradient variance
- Familiarity with GPU memory and distributed workers is useful, but not required

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Executable demos for AdamW, clipping, schedules, memory accounting, ZeRO stages, pipeline bubbles, FLOPs, MFU, and checkpoint overhead. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems that make you compute the quantities used in real training-plan reviews. |

## Learning Objectives

After this section, you should be able to:

- Explain why training memory is larger than inference memory.
- Derive Adam and AdamW updates with bias correction.
- Compute effective batch size under data parallelism and gradient accumulation.
- Build warmup and cosine learning-rate schedules.
- Estimate parameter, gradient, optimizer, and activation memory.
- Compare data, tensor, pipeline, sequence, and sharded parallelism.
- Compute pipeline bubble fraction and approximate all-reduce cost.
- Estimate training FLOPs with $C\approx 6ND$ and interpret MFU.
- Explain the practical meaning of Kaplan-style and Chinchilla-style scaling laws.
- Build a debugging checklist for loss spikes, bad resumes, low throughput, and masking errors.

## Table of Contents

1. [Scale as a Constraint Problem](#1-scale-as-a-constraint-problem)
   - 1.1 [The same loss at larger cost](#11-the-same-loss-at-larger-cost)
   - 1.2 [Four limiting resources](#12-four-limiting-resources)
   - 1.3 [Parameter, optimizer, and activation memory](#13-parameter-optimizer-and-activation-memory)
   - 1.4 [Throughput versus convergence](#14-throughput-versus-convergence)
   - 1.5 [Failure modes](#15-failure-modes)
2. [Optimization Core](#2-optimization-core)
   - 2.1 [Mini-batch gradient](#21-minibatch-gradient)
   - 2.2 [Adam moments](#22-adam-moments)
   - 2.3 [Bias correction](#23-bias-correction)
   - 2.4 [AdamW](#24-adamw)
   - 2.5 [Gradient clipping](#25-gradient-clipping)
3. [Batching and Schedules](#3-batching-and-schedules)
   - 3.1 [Effective batch size](#31-effective-batch-size)
   - 3.2 [Gradient accumulation](#32-gradient-accumulation)
   - 3.3 [Linear warmup](#33-linear-warmup)
   - 3.4 [Cosine decay](#34-cosine-decay)
   - 3.5 [Critical batch intuition](#35-critical-batch-intuition)
4. [Memory Accounting](#4-memory-accounting)
   - 4.1 [Bytes per parameter](#41-bytes-per-parameter)
   - 4.2 [Activation memory](#42-activation-memory)
   - 4.3 [Activation checkpointing](#43-activation-checkpointing)
   - 4.4 [Optimizer state sharding](#44-optimizer-state-sharding)
   - 4.5 [Offload boundary](#45-offload-boundary)
5. [Parallelism Strategies](#5-parallelism-strategies)
   - 5.1 [Data parallelism](#51-data-parallelism)
   - 5.2 [Tensor parallelism](#52-tensor-parallelism)
   - 5.3 [Pipeline parallelism](#53-pipeline-parallelism)
   - 5.4 [Sequence parallelism](#54-sequence-parallelism)
   - 5.5 [Parallelism product](#55-parallelism-product)
6. [Communication Math](#6-communication-math)
   - 6.1 [All-reduce cost](#61-allreduce-cost)
   - 6.2 [Reduce-scatter and all-gather](#62-reducescatter-and-allgather)
   - 6.3 [Overlap](#63-overlap)
   - 6.4 [Bandwidth hierarchy](#64-bandwidth-hierarchy)
   - 6.5 [Straggler sensitivity](#65-straggler-sensitivity)
7. [Compute and Scaling Laws](#7-compute-and-scaling-laws)
   - 7.1 [Training FLOPs estimate](#71-training-flops-estimate)
   - 7.2 [Kaplan-style power laws](#72-kaplanstyle-power-laws)
   - 7.3 [Compute-optimal tradeoff](#73-computeoptimal-tradeoff)
   - 7.4 [MFU](#74-mfu)
   - 7.5 [Inference-aware training](#75-inferenceaware-training)
8. [Numerical Stability](#8-numerical-stability)
   - 8.1 [Mixed precision](#81-mixed-precision)
   - 8.2 [Loss scaling](#82-loss-scaling)
   - 8.3 [Attention stability](#83-attention-stability)
   - 8.4 [Loss spikes](#84-loss-spikes)
   - 8.5 [Resume correctness](#85-resume-correctness)
9. [Data and Checkpoint Operations](#9-data-and-checkpoint-operations)
   - 9.1 [Token budget](#91-token-budget)
   - 9.2 [Packing](#92-packing)
   - 9.3 [Deduplication and filtering](#93-deduplication-and-filtering)
   - 9.4 [Checkpoint frequency](#94-checkpoint-frequency)
   - 9.5 [Validation cadence](#95-validation-cadence)
10. [Operational Debugging](#10-operational-debugging)
   - 10.1 [Shape and mask checks](#101-shape-and-mask-checks)
   - 10.2 [Gradient norm traces](#102-gradient-norm-traces)
   - 10.3 [Learning-rate traces](#103-learningrate-traces)
   - 10.4 [Throughput decomposition](#104-throughput-decomposition)
   - 10.5 [Reproducible small run](#105-reproducible-small-run)
11. [Gradient Noise, Critical Batch Size, and Data Efficiency](#11-gradient-noise-critical-batch-size-and-data-efficiency)
12. [Learning-Rate Scaling When Batch Size Changes](#12-learning-rate-scaling-when-batch-size-changes)
13. [Memory in Detail](#13-memory-in-detail)
14. [Parallelism Worked Example](#14-parallelism-worked-example)
15. [Reliability Mathematics](#15-reliability-mathematics)
16. [Energy and Cost Accounting](#16-energy-and-cost-accounting)
17. [Numerical Stability in Practice](#17-numerical-stability-in-practice)
18. [End-to-End Launch Budget](#18-end-to-end-launch-budget)
19. [Memory Worksheets](#19-memory-worksheets)
20. [Learning-Rate Schedule Worksheets](#20-learning-rate-schedule-worksheets)
21. [Communication Worksheets](#21-communication-worksheets)
22. [Data Pipeline Mathematics](#22-data-pipeline-mathematics)
23. [Debugging Playbook](#23-debugging-playbook)
24. [Five End-to-End Scenarios](#24-five-end-to-end-scenarios)
25. [Formula Summary](#25-formula-summary)
26. [Pre-Launch Checklist](#26-pre-launch-checklist)
27. [Glossary](#27-glossary)
28. [Historical Line](#28-historical-line)
29. [Computed Tables](#29-computed-tables-you-can-reproduce)
30. [AdamW Trace](#30-adamw-trace-you-can-verify-by-hand)
31. [Activation Memory by Layer Count](#31-activation-memory-by-layer-count)
32. [Launch Decision Worksheet](#32-launch-decision-worksheet-worked)
33. [What Good Looks Like at Three Scales](#33-what-good-looks-like-at-three-scales)
34. [Ten Review Questions](#34-ten-review-questions-with-worked-answers)
35. [Failure Taxonomy](#35-failure-taxonomy-for-large-runs)

---

## A Quick Scale Budget

Before a large training run, write a one-page budget:

| Quantity | Example question |
| --- | --- |
| Parameters | How many trainable parameters are updated? |
| Tokens | How many training tokens are consumed? |
| Precision | Which tensors are bf16, fp16, fp32, or quantized? |
| Batch | What is the effective global batch in tokens? |
| Parallelism | Which axes are data, tensor, pipeline, sequence, and sharded? |
| Optimizer | Which states are stored, sharded, or offloaded? |
| Checkpoint | What must be saved to resume exactly? |
| Validation | Which held-out loss confirms learning? |

The point is not paperwork. The point is to make hidden assumptions explicit before they become expensive failures.

## 1. Scale as a Constraint Problem

This part focuses on scale as a constraint problem as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [The same loss at larger cost](#1-the-same-loss-at-larger-cost) | training at scale still minimizes next-token cross-entropy | $L(\theta)=-\sum_i \log p_\theta(t_i\mid t_{<i})$ |
| [Four limiting resources](#1-four-limiting-resources) | memory, compute, bandwidth, and data quality each become a bottleneck | $\mathrm{time}\approx\max(T_\mathrm{compute},T_\mathrm{comm},T_\mathrm{input})$ |
| [Parameter, optimizer, and activation memory](#1-parameter-optimizer-and-activation-memory) | weights are only one part of training memory | $M=M_\mathrm{params}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$ |
| [Throughput versus convergence](#1-throughput-versus-convergence) | fast tokens per second are useful only if loss improves | $\mathrm{tokens/sec}$ must be read with $L(\mathrm{tokens})$ |
| [Failure modes](#1-failure-modes) | large training fails by divergence, stalls, bad data, communication bottlenecks, or checkpoint loss | $\Delta L>0$ for many steps is a symptom, not a diagnosis |

### 1.1 The same loss at larger cost

**Main idea.** Training at scale still minimizes next-token cross-entropy.

Core relation:

$$L(\theta)=-\sum_i \log p_\theta(t_i\mid t_{<i})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.2 Four limiting resources

**Main idea.** Memory, compute, bandwidth, and data quality each become a bottleneck.

Core relation:

$$\mathrm{time}\approx\max(T_\mathrm{compute},T_\mathrm{comm},T_\mathrm{input})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.3 Parameter, optimizer, and activation memory

**Main idea.** Weights are only one part of training memory.

Core relation:

$$M=M_\mathrm{params}+M_\mathrm{grads}+M_\mathrm{opt}+M_\mathrm{act}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.4 Throughput versus convergence

**Main idea.** Fast tokens per second are useful only if loss improves.

Core relation:

$$\mathrm{tokens/sec}$ must be read with $L(\mathrm{tokens})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 1.5 Failure modes

**Main idea.** Large training fails by divergence, stalls, bad data, communication bottlenecks, or checkpoint loss.

Core relation:

$$\Delta L>0$ for many steps is a symptom, not a diagnosis$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 2. Optimization Core

This part focuses on optimization core as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Mini-batch gradient](#2-minibatch-gradient) | distributed workers estimate the same gradient with different data shards | $g_t=\frac{1}{B}\sum_{i=1}^{B}\nabla_\theta \ell_i$ |
| [Adam moments](#2-adam-moments) | first and second moment estimates adapt update scale | $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$ |
| [Bias correction](#2-bias-correction) | early moments are corrected because they start at zero | $\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t)$ |
| [AdamW](#2-adamw) | weight decay is applied outside the adaptive gradient ratio | $\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)-\eta\lambda\theta_t$ |
| [Gradient clipping](#2-gradient-clipping) | cap update norm when rare batches produce spikes | $g\leftarrow g\min(1,c/\Vert g\Vert_2)$ |

### 2.1 Mini-batch gradient

**Main idea.** Distributed workers estimate the same gradient with different data shards.

Core relation:

$$g_t=\frac{1}{B}\sum_{i=1}^{B}\nabla_\theta \ell_i$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.2 Adam moments

**Main idea.** First and second moment estimates adapt update scale.

Core relation:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.3 Bias correction

**Main idea.** Early moments are corrected because they start at zero.

Core relation:

$$\hat m_t=m_t/(1-\beta_1^t),\quad \hat v_t=v_t/(1-\beta_2^t)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.4 AdamW

**Main idea.** Weight decay is applied outside the adaptive gradient ratio.

Core relation:

$$\theta_{t+1}=\theta_t-\eta\hat m_t/(\sqrt{\hat v_t}+\epsilon)-\eta\lambda\theta_t$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 2.5 Gradient clipping

**Main idea.** Cap update norm when rare batches produce spikes.

Core relation:

$$g\leftarrow g\min(1,c/\Vert g\Vert_2)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** Clipping is not a cure for a bad run, but it can prevent one rare batch from destroying useful optimizer state.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 3. Batching and Schedules

This part focuses on batching and schedules as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Effective batch size](#3-effective-batch-size) | global batch combines devices and accumulation steps | $B_\mathrm{eff}=B_\mathrm{device}G_\mathrm{accum}N_\mathrm{dp}$ |
| [Gradient accumulation](#3-gradient-accumulation) | several micro-batches approximate one larger batch | $g=\frac{1}{K}\sum_{k=1}^{K}g_k$ |
| [Linear warmup](#3-linear-warmup) | the learning rate starts small to avoid early instability | $\eta_t=\eta_\max t/T_\mathrm{warmup}$ |
| [Cosine decay](#3-cosine-decay) | the learning rate anneals smoothly after warmup | $\eta_t=\eta_\min+\frac{1}{2}(\eta_\max-\eta_\min)(1+\cos(\pi s))$ |
| [Critical batch intuition](#3-critical-batch-intuition) | past a point, larger batches waste compute rather than reducing noise usefully | $\mathrm{noise}\propto 1/B$ only in the useful regime |

### 3.1 Effective batch size

**Main idea.** Global batch combines devices and accumulation steps.

Core relation:

$$B_\mathrm{eff}=B_\mathrm{device}G_\mathrm{accum}N_\mathrm{dp}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.2 Gradient accumulation

**Main idea.** Several micro-batches approximate one larger batch.

Core relation:

$$g=\frac{1}{K}\sum_{k=1}^{K}g_k$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.3 Linear warmup

**Main idea.** The learning rate starts small to avoid early instability.

Core relation:

$$\eta_t=\eta_\max t/T_\mathrm{warmup}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.4 Cosine decay

**Main idea.** The learning rate anneals smoothly after warmup.

Core relation:

$$\eta_t=\eta_\min+\frac{1}{2}(\eta_\max-\eta_\min)(1+\cos(\pi s))$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 3.5 Critical batch intuition

**Main idea.** Past a point, larger batches waste compute rather than reducing noise usefully.

Core relation:

$$\mathrm{noise}\propto 1/B$ only in the useful regime$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 4. Memory Accounting

This part focuses on memory accounting as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Bytes per parameter](#4-bytes-per-parameter) | bf16 weights use 2 bytes but Adam states are often fp32 | $M_\mathrm{Adam}\approx 2P + 2P + 8P$ bytes |
| [Activation memory](#4-activation-memory) | stored forward activations can dominate at long context | $M_\mathrm{act}\propto BTLd$ |
| [Activation checkpointing](#4-activation-checkpointing) | save memory by recomputing intermediate activations | $M_\mathrm{act}\downarrow,\quad T_\mathrm{compute}\uparrow$ |
| [Optimizer state sharding](#4-optimizer-state-sharding) | ZeRO/FSDP shard model states across data-parallel ranks | $M_\mathrm{per\ rank}\approx M/N$ for fully sharded states |
| [Offload boundary](#4-offload-boundary) | CPU or NVMe offload trades memory for bandwidth and latency | $T_\mathrm{step}$ can become transfer-bound |

### 4.1 Bytes per parameter

**Main idea.** Bf16 weights use 2 bytes but adam states are often fp32.

Core relation:

$$M_\mathrm{Adam}\approx 2P + 2P + 8P$ bytes$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.2 Activation memory

**Main idea.** Stored forward activations can dominate at long context.

Core relation:

$$M_\mathrm{act}\propto BTLd$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.3 Activation checkpointing

**Main idea.** Save memory by recomputing intermediate activations.

Core relation:

$$M_\mathrm{act}\downarrow,\quad T_\mathrm{compute}\uparrow$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.4 Optimizer state sharding

**Main idea.** Zero/fsdp shard model states across data-parallel ranks.

Core relation:

$$M_\mathrm{per\ rank}\approx M/N$ for fully sharded states$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This is why a model that cannot fit on one accelerator can still be trained across many accelerators.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 4.5 Offload boundary

**Main idea.** Cpu or nvme offload trades memory for bandwidth and latency.

Core relation:

$$T_\mathrm{step}$ can become transfer-bound$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 5. Parallelism Strategies

This part focuses on parallelism strategies as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Data parallelism](#5-data-parallelism) | replicate the model and all-reduce gradients | $g=\frac{1}{N}\sum_{r=1}^{N}g_r$ |
| [Tensor parallelism](#5-tensor-parallelism) | split matrix multiplications across devices | $Y=X[W_1\ W_2]$ or $Y=XW_1+XW_2$ depending on layout |
| [Pipeline parallelism](#5-pipeline-parallelism) | place layers on stages and stream micro-batches | $\mathrm{bubble}\approx(P-1)/(M+P-1)$ |
| [Sequence parallelism](#5-sequence-parallelism) | split sequence-length work when activations are too large | $T$ is partitioned across ranks |
| [Parallelism product](#5-parallelism-product) | large jobs combine data, tensor, pipeline, and sometimes sequence parallelism | $N_\mathrm{total}=N_\mathrm{dp}N_\mathrm{tp}N_\mathrm{pp}N_\mathrm{sp}$ |

### 5.1 Data parallelism

**Main idea.** Replicate the model and all-reduce gradients.

Core relation:

$$g=\frac{1}{N}\sum_{r=1}^{N}g_r$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.2 Tensor parallelism

**Main idea.** Split matrix multiplications across devices.

Core relation:

$$Y=X[W_1\ W_2]$ or $Y=XW_1+XW_2$ depending on layout$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.3 Pipeline parallelism

**Main idea.** Place layers on stages and stream micro-batches.

Core relation:

$$\mathrm{bubble}\approx(P-1)/(M+P-1)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.4 Sequence parallelism

**Main idea.** Split sequence-length work when activations are too large.

Core relation:

$$T$ is partitioned across ranks$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 5.5 Parallelism product

**Main idea.** Large jobs combine data, tensor, pipeline, and sometimes sequence parallelism.

Core relation:

$$N_\mathrm{total}=N_\mathrm{dp}N_\mathrm{tp}N_\mathrm{pp}N_\mathrm{sp}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 6. Communication Math

This part focuses on communication math as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [All-reduce cost](#6-allreduce-cost) | gradient synchronization costs latency plus bandwidth | $T\approx \alpha\log N+\beta S$ |
| [Reduce-scatter and all-gather](#6-reducescatter-and-allgather) | sharded training replaces one all-reduce with state movement primitives | $\mathrm{allreduce}=\mathrm{reduce\ scatter}+\mathrm{all\ gather}$ |
| [Overlap](#6-overlap) | hide communication under backward computation when dependencies allow it | $T_\mathrm{step}\approx\max(T_\mathrm{compute},T_\mathrm{comm})$ |
| [Bandwidth hierarchy](#6-bandwidth-hierarchy) | intra-node links are much faster than inter-node links | $T_\mathrm{inter}>T_\mathrm{intra}$ for the same payload |
| [Straggler sensitivity](#6-straggler-sensitivity) | synchronous steps wait for the slowest rank | $T_\mathrm{step}=\max_r T_r$ |

### 6.1 All-reduce cost

**Main idea.** Gradient synchronization costs latency plus bandwidth.

Core relation:

$$T\approx \alpha\log N+\beta S$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.2 Reduce-scatter and all-gather

**Main idea.** Sharded training replaces one all-reduce with state movement primitives.

Core relation:

$$\mathrm{allreduce}=\mathrm{reduce\ scatter}+\mathrm{all\ gather}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.3 Overlap

**Main idea.** Hide communication under backward computation when dependencies allow it.

Core relation:

$$T_\mathrm{step}\approx\max(T_\mathrm{compute},T_\mathrm{comm})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.4 Bandwidth hierarchy

**Main idea.** Intra-node links are much faster than inter-node links.

Core relation:

$$T_\mathrm{inter}>T_\mathrm{intra}$ for the same payload$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 6.5 Straggler sensitivity

**Main idea.** Synchronous steps wait for the slowest rank.

Core relation:

$$T_\mathrm{step}=\max_r T_r$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 7. Compute and Scaling Laws

This part focuses on compute and scaling laws as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Training FLOPs estimate](#7-training-flops-estimate) | dense transformer training is often approximated by six times parameters times tokens | $C\approx 6ND$ |
| [Kaplan-style power laws](#7-kaplanstyle-power-laws) | loss follows predictable power trends over model, data, and compute in a range | $L(C)=L_\infty+aC^{-\alpha}$ |
| [Compute-optimal tradeoff](#7-computeoptimal-tradeoff) | for a fixed budget, model size and token count must be balanced | $C\approx 6ND$ with both $N$ and $D$ chosen |
| [MFU](#7-mfu) | model FLOPs utilization compares achieved useful FLOPs to hardware peak | $\mathrm{MFU}=\mathrm{model\ FLOPs/sec}/\mathrm{peak\ FLOPs/sec}$ |
| [Inference-aware training](#7-inferenceaware-training) | overtraining a smaller model can reduce serving cost even if it is not pure compute-optimal pretraining | $\mathrm{train\ cost}+\mathrm{serve\ cost}$ matters |

### 7.1 Training FLOPs estimate

**Main idea.** Dense transformer training is often approximated by six times parameters times tokens.

Core relation:

$$C\approx 6ND$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This simple estimate is often the first line in a training-budget spreadsheet.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.2 Kaplan-style power laws

**Main idea.** Loss follows predictable power trends over model, data, and compute in a range.

Core relation:

$$L(C)=L_\infty+aC^{-\alpha}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.3 Compute-optimal tradeoff

**Main idea.** For a fixed budget, model size and token count must be balanced.

Core relation:

$$C\approx 6ND$ with both $N$ and $D$ chosen$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.4 MFU

**Main idea.** Model flops utilization compares achieved useful flops to hardware peak.

Core relation:

$$\mathrm{MFU}=\mathrm{model\ FLOPs/sec}/\mathrm{peak\ FLOPs/sec}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This separates a slow model because it is mathematically large from a slow run because the system is wasting hardware.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 7.5 Inference-aware training

**Main idea.** Overtraining a smaller model can reduce serving cost even if it is not pure compute-optimal pretraining.

Core relation:

$$\mathrm{train\ cost}+\mathrm{serve\ cost}$ matters$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 8. Numerical Stability

This part focuses on numerical stability as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Mixed precision](#8-mixed-precision) | bf16/fp16 reduce memory and increase throughput but require stable reductions | $\theta$ may be bf16 while optimizer states stay fp32 |
| [Loss scaling](#8-loss-scaling) | fp16 may need scaling to avoid underflow | $\tilde L=sL,\quad \tilde g=sg$ |
| [Attention stability](#8-attention-stability) | score scaling and stable softmax matter more at long sequence lengths | $QK^\top/\sqrt d$ |
| [Loss spikes](#8-loss-spikes) | spikes can come from data, optimizer state, numerical overflow, or synchronization problems | $L_t\gg\mathrm{median}(L_{t-k:t})$ |
| [Resume correctness](#8-resume-correctness) | checkpoint reload must restore model, optimizer, scheduler, RNG, and dataloader state | $\theta,m,v,t,\mathrm{rng}$ all matter |

### 8.1 Mixed precision

**Main idea.** Bf16/fp16 reduce memory and increase throughput but require stable reductions.

Core relation:

$$\theta$ may be bf16 while optimizer states stay fp32$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.2 Loss scaling

**Main idea.** Fp16 may need scaling to avoid underflow.

Core relation:

$$\tilde L=sL,\quad \tilde g=sg$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.3 Attention stability

**Main idea.** Score scaling and stable softmax matter more at long sequence lengths.

Core relation:

$$QK^\top/\sqrt d$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.4 Loss spikes

**Main idea.** Spikes can come from data, optimizer state, numerical overflow, or synchronization problems.

Core relation:

$$L_t\gg\mathrm{median}(L_{t-k:t})$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 8.5 Resume correctness

**Main idea.** Checkpoint reload must restore model, optimizer, scheduler, rng, and dataloader state.

Core relation:

$$\theta,m,v,t,\mathrm{rng}$ all matter$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** A bad resume can silently fork the training trajectory even when the checkpoint file loads.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 9. Data and Checkpoint Operations

This part focuses on data and checkpoint operations as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Token budget](#9-token-budget) | data is counted in tokens, not documents | $D=\sum_\mathrm{docs}\mathrm{tokens(doc)}$ |
| [Packing](#9-packing) | short examples are packed to reduce padding waste | $\mathrm{utilization}=\mathrm{real\ tokens}/\mathrm{allocated\ tokens}$ |
| [Deduplication and filtering](#9-deduplication-and-filtering) | bad repeated data can improve train loss while hurting generalization | $p_\mathrm{train}$ can drift from desired $p_\mathrm{deploy}$ |
| [Checkpoint frequency](#9-checkpoint-frequency) | the optimal interval balances lost work and checkpoint overhead | $\mathrm{overhead}\approx T_\mathrm{ckpt}/K+\mathrm{failure\ loss}(K)$ |
| [Validation cadence](#9-validation-cadence) | held-out loss catches overfitting, data bugs, and regression after resume | $L_\mathrm{val}$ is the early warning signal |

### 9.1 Token budget

**Main idea.** Data is counted in tokens, not documents.

Core relation:

$$D=\sum_\mathrm{docs}\mathrm{tokens(doc)}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.2 Packing

**Main idea.** Short examples are packed to reduce padding waste.

Core relation:

$$\mathrm{utilization}=\mathrm{real\ tokens}/\mathrm{allocated\ tokens}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.3 Deduplication and filtering

**Main idea.** Bad repeated data can improve train loss while hurting generalization.

Core relation:

$$p_\mathrm{train}$ can drift from desired $p_\mathrm{deploy}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.4 Checkpoint frequency

**Main idea.** The optimal interval balances lost work and checkpoint overhead.

Core relation:

$$\mathrm{overhead}\approx T_\mathrm{ckpt}/K+\mathrm{failure\ loss}(K)$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 9.5 Validation cadence

**Main idea.** Held-out loss catches overfitting, data bugs, and regression after resume.

Core relation:

$$L_\mathrm{val}$ is the early warning signal$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
## 10. Operational Debugging

This part focuses on operational debugging as a practical mathematical constraint in LLM training. The goal is not to memorize infrastructure names, but to understand the formulas that determine whether a run fits, learns, communicates, and resumes.

| Subtopic | Operational question | Formula |
| --- | --- | --- |
| [Shape and mask checks](#10-shape-and-mask-checks) | wrong labels or masks can produce plausible but meaningless loss | $\mathrm{target}_i=\mathrm{input}_{i+1}$ |
| [Gradient norm traces](#10-gradient-norm-traces) | track global norms before and after clipping | $\Vert g\Vert_2$ |
| [Learning-rate traces](#10-learningrate-traces) | optimizer behavior must match the intended schedule | $\eta_t$ |
| [Throughput decomposition](#10-throughput-decomposition) | separate dataloader, forward, backward, communication, optimizer, and checkpoint time | $T_\mathrm{step}=\sum_j T_j$ |
| [Reproducible small run](#10-reproducible-small-run) | scale only after a small deterministic run learns and resumes correctly | $L_{100}<L_0$ is a smoke test |

### 10.1 Shape and mask checks

**Main idea.** Wrong labels or masks can produce plausible but meaningless loss.

Core relation:

$$\mathrm{target}_i=\mathrm{input}_{i+1}$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.2 Gradient norm traces

**Main idea.** Track global norms before and after clipping.

Core relation:

$$\Vert g\Vert_2$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.3 Learning-rate traces

**Main idea.** Optimizer behavior must match the intended schedule.

Core relation:

$$\eta_t$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.4 Throughput decomposition

**Main idea.** Separate dataloader, forward, backward, communication, optimizer, and checkpoint time.

Core relation:

$$T_\mathrm{step}=\sum_j T_j$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.
### 10.5 Reproducible small run

**Main idea.** Scale only after a small deterministic run learns and resumes correctly.

Core relation:

$$L_{100}<L_0$ is a smoke test$$

At small scale, this relation may feel like bookkeeping. At LLM scale, it becomes a hard constraint. A missing factor of two in a memory estimate can decide whether the job starts. A wrong batch-size convention can change the optimization regime. A poor communication plan can leave expensive accelerators idle.

**Worked micro-example.** Suppose a dense model has $P=7$ billion parameters. bf16 weights alone require about $2P$ bytes, or roughly 14 GB. Training with Adam usually also needs gradients and two optimizer moment tensors. If the moments are fp32, the optimizer state adds about $8P$ bytes, before activations. That is why "weights fit" is not the same as "training fits."

**Implementation check.** Write down the unit. Is the number per parameter, per token, per device, per data-parallel rank, per step, or per full run? Most scale-training bugs are not exotic math errors; they are unit and axis errors.

**AI connection.** This formula is part of the control surface for a large training run.

**Common mistake.** Do not optimize one metric in isolation. More tokens per second can be bad if validation loss stops improving, and lower memory can be bad if recomputation makes the step too slow.

---

---

## 11. Gradient Noise, Critical Batch Size, and Data Efficiency

### 11.1 The gradient noise scale

McCandlish et al. (2018) define the gradient noise scale $\mathcal{B}_{\text{noise}}=\frac{\mathrm{tr}(\Sigma)}{|\mathbf{g}|^2}$, where $\Sigma$ is the per-example gradient covariance and $\mathbf{g}$ the true gradient. It estimates the batch size at which doubling the batch stops halving gradient variance usefully.

- Early in training $|\mathbf{g}|$ is large, so $\mathcal{B}_{\text{noise}}$ is small (tens of thousands of tokens can suffice).
- Late in training $|\mathbf{g}|$ shrinks toward the noise floor, so $\mathcal{B}_{\text{noise}}$ grows — one principled reason large runs increase batch size over time rather than fixing it.
- Non-example: tokens per batch is not $\mathcal{B}_{\text{noise}}$; one is a chosen hyperparameter, the other a measured property of the loss surface and data.

### 11.2 Worked critical-batch estimate

Suppose per-token gradient variance trace is $\mathrm{tr}(\Sigma)=4.0$ (in normalized units) and the full-batch gradient norm squared is $|\mathbf{g}|^2=10^{-4}$ late in training. Then $\mathcal{B}_{\text{noise}}\approx40{,}000$ examples-equivalent; a batch far below this is noise-dominated (more steps at smaller batch train faster per token), far above it is waste (extra tokens per step buy almost no variance reduction). McCandlish et al. (2018) verified this predicts the largest *efficient* batch across tasks.

### 11.3 Implication for the token budget

Training tokens $D$, batch $B$ (tokens), and steps $S$ obey $S=D/B$. At $D=300$B tokens: $B=0.5$M tokens gives $600{,}000$ steps; $B=4$M gives $75{,}000$ steps. Fewer, larger steps parallelize better but take fewer optimizer updates — the steps-versus-tokens tradeoff behind the batch ramps used in large runs (e.g. the GPT-3 schedule, Brown et al., 2020).

---

## 12. Learning-Rate Scaling When Batch Size Changes

### 12.1 Linear scaling and its limits

Goyal et al. (2017) scaled learning rate linearly with batch size (with warmup) for ResNets; the heuristic keeps the per-example update contribution roughly constant while gradients are not yet noise-dominated. It fails when $B\gg\mathcal{B}_{\text{noise}}$ (Section 11): then extra examples add little signal, and linear LR growth mostly adds instability.

### 12.2 Square-root scaling

The alternative $\eta\propto\sqrt{B}$ comes from keeping update *variance* constant: averaging $B$ gradients shrinks noise as $1/\sqrt{B}$, so a $\sqrt{B}$ LR keeps the stochastic part of the update comparable. Which rule is closer depends on whether the run is in the noise-dominated or curvature-dominated regime — measurable via Section 11, not a matter of taste.

### 12.3 Layer-wise adaptivity

LAMB (You et al., 2020) scales each layer's update by the ratio of weight norm to update norm, enabling very large batches for BERT-style training. It is Adam (Kingma and Ba, 2015) plus a trust-ratio step, not a new objective: the loss of Section 2 is unchanged, only the geometry of the step.

---

## 13. Memory in Detail

### 13.1 Bytes per parameter for AdamW training

| State | Precision | Bytes/param |
|---|---|---:|
| Weights (master copy) | fp32 | 4 |
| Weights (compute copy, mixed precision) | bf16 | 2 |
| Gradients | bf16 or fp32 | 2–4 |
| Adam $m$ | fp32 | 4 |
| Adam $v$ | fp32 | 4 |
| **Total (typical mixed-precision AdamW)** | | **16–18** |

So a 7B model needs roughly $7\times10^9\times16\approx112$ GB before activations — more than one 80 GB accelerator, hence sharding (Section 5.4) even at "small" LLM scale. This table is the expansion of Section 4.1's estimate.

### 13.2 Activation memory per layer

For batch tokens $T$, hidden size $h$, the saved activations of one transformer layer scale as $O(T\cdot h)$ per stored tensor, and attention additionally stores $O(T\cdot s)$ for sequence length $s$ unless memory-efficient attention (Dao et al., 2022, FlashAttention) recomputes softmax statistics instead of materializing the $s\times s$ matrix. Activation checkpointing (Chen et al., 2016) stores only layer-boundary activations and recomputes interiors in the backward pass: memory falls from $O(L)$ layer activations to $O(\sqrt{L})$ segments at roughly one extra forward pass of compute (~30–40% slower in the classic tradeoff, cf. Section 4.3).

### 13.3 ZeRO stages restated as a memory equation

With $P$ parameters, $\Psi$ bytes of optimizer+master state per parameter, and $N$ data-parallel ranks: Stage 1 shards optimizer states ($\Psi/N$ per rank), Stage 2 also shards gradients, Stage 3 also shards the compute weights themselves (Rajbhandari et al., 2020). The price of Stage 3 is extra all-gather communication per layer — memory is bought with bandwidth, the recurring trade of Section 6.

---

## 14. Parallelism Worked Example: Placing a 70B Model

Target: 70B parameters, AdamW mixed precision at 16 B/param $\approx1.12$ TB of training state.

- Pure data parallelism on 80 GB devices is impossible: one replica alone exceeds device memory by ~14x.
- Tensor parallelism (Shoeybi et al., 2019) of degree 8 within a node shards each layer's matrices; pipeline parallelism (Huang et al., 2019, GPipe; Narayanan et al., 2021, Megatron-LM schedules) of degree 8 across nodes shards layers; data parallelism of degree 16 across the resulting model replicas gives $8\times8\times16=1024$ devices and $1.12\text{ TB}/64\approx17.5$ GB of static state per device before activations — feasible, with activation checkpointing (13.2) for the remainder.
- Rule used above, from Section 5.5: total devices $=$ TP $\times$ PP $\times$ DP, and each degree is placed where its communication is cheapest: TP on the fastest interconnect (intra-node), PP next, DP (gradient all-reduce, once per step) across the slowest links.

**Non-example.** Choosing TP degree 64 across nodes because "more sharding is safer": all-reduce in every layer over a slow interconnect makes communication, not compute, the step time (Section 6.1).

---

## 15. Reliability Mathematics

### 15.1 Failure probability over a long run

If one device fails as a Poisson process with mean time between failures $M$ hours, a $N$-device job sees failures at rate $N/M$, so $P(\text{no failure in }T\text{ hours})=e^{-NT/M}$. With $N=1024$, $M=50{,}000$ h (illustrative), $T=24$ h: exponent $=0.49$, so a failure-free day has probability only $\approx61\%$. Checkpointing (Section 9.4) is therefore not paranoia; it is the expected case.

### 15.2 Optimal checkpoint interval

Young's approximation (Young, 1974; Daly, 2006 refinement): with checkpoint cost $C$ seconds and job MTBF $\mu$ seconds, the waste-minimizing interval is $\approx\sqrt{2C\mu}$. Example: $C=180$ s, $\mu=48$ h $=172{,}800$ s gives $\sqrt{2\times180\times172800}\approx7{,}900$ s $\approx2.2$ h between checkpoints. Checkpointing every 100 steps "to be safe" can waste more time than the failures it protects against — Section 9.4's overhead formula quantifies the cost side.

### 15.3 Resume correctness is a state problem

A resumable checkpoint must contain weights, optimizer moments, step count, LR schedule position, and data-order/RNG state. Missing the Adam moments (13.1) does not crash; it silently restarts the optimizer geometry and can look like a loss spike (Section 8.4) — which is why Section 8.5 insists on a small-run resume test before launch.

---

## 16. Energy and Cost Accounting

Training energy $\approx$ devices $\times$ hours $\times$ device power $\times$ PUE. For the 14-day, 7B example of theory notebook Section 10 scaled to 128 devices at 0.4 kW average draw and PUE 1.2: $128\times336\times0.4\times1.2\approx20{,}600$ kWh. Reporting this alongside tokens and FLOPs (Sections 7.1, 7.4) is becoming standard practice in model cards; cost per *unit of validation-loss reduction*, not per token, is the efficiency number that survives comparison across runs.

---

## 17. Numerical Stability in Practice

### 17.1 Why bf16 needs an fp32 master and loss scaling history

bf16 has ~3 decimal digits of precision; a weight update $\eta g$ smaller than the bf16 rounding unit of the weight is lost. Mixed precision (Micikevicius et al., 2018) therefore keeps fp32 master weights (13.1) and, for fp16, a dynamic loss scale; bf16's wider exponent removes most overflow need but not the master-weight requirement.

### 17.2 Attention logit growth

Attention logits $q\cdot k/\sqrt d$ grow with hidden size and activation outliers; Section 8.3's stability concern became concrete in large runs where attention-entropy collapse preceded loss spikes. Monitoring max attention logit and gradient norm (Section 10.2) gives warning *before* the loss moves — the operational reason those traces are in the launch checklist.

### 17.3 Loss spikes: rollback, do not ride through

Standard practice in large runs (e.g. the OPT logbook, Zhang et al., 2022) is: detect spike, roll back to the last healthy checkpoint, skip the offending data batch or lower LR, resume. Riding through a spike wastes tokens in a damaged basin and can permanently raise the achievable loss.

---

## 18. End-to-End Launch Budget for a 7B Run

| Quantity | Value | Source section |
|---|---:|---|
| Parameters $N$ | $7\times10^9$ | — |
| Tokens $D$ | $3\times10^{11}$ | 9.1 |
| Training FLOPs $\approx6ND$ | $1.26\times10^{22}$ | 7.1 |
| Static training memory at 16 B/param | 112 GB | 13.1 |
| Batch $B$ | 4M tokens -> 75,000 steps | 11.3 |
| At 2.0 s/step | 41.7 h of wall time (ideal) | 9.4/10.4 |
| Checkpoint every ~2.2 h | $\approx$19 checkpoints, 180 s each $\approx$1.6% overhead | 15.2 |
| Failure-free probability per day (1024-device illustrative rate, scaled down here) | compute per 15.1 for your $N$ | 15.1 |

Kaplan et al. (2020) and Hoffmann et al. (2022) disagree on the compute-optimal $D/N$ ratio (Chinchilla argues for far more tokens per parameter than early practice); the budget above uses $D/N\approx43$, between the two, and the theory notebook's Section 11 surface lets you re-derive the loss proxy at other ratios instead of trusting either paper blindly.

---


---

## 19. Memory Worksheets for Four Model Sizes

Using Section 13.1 at 16 bytes per parameter (bf16 compute weights 2 + fp32 master 4 + bf16 gradients 2 + Adam moments 8):

| Parameters | Weights bf16 | Master fp32 | Gradients bf16 | Adam $m+v$ | Total static |
|---:|---:|---:|---:|---:|---:|
| 1B | 2 GB | 4 GB | 2 GB | 8 GB | 16 GB |
| 7B | 14 GB | 28 GB | 14 GB | 56 GB | 112 GB |
| 13B | 26 GB | 52 GB | 26 GB | 104 GB | 208 GB |
| 70B | 140 GB | 280 GB | 140 GB | 560 GB | 1,120 GB |
| 175B | 350 GB | 700 GB | 350 GB | 1,400 GB | 2,800 GB |

Reading the table column by column is the fastest launch review there is:

- If the Adam column alone exceeds device memory, no activation trick will save a data-parallel-only plan; shard the optimizer (13.3).
- The master-weights column is the one most often forgotten in hallway estimates, because inference only needs the bf16 column.
- The gradients column can be halved in time (not space) by bucketed all-reduce, but its *space* only falls with ZeRO Stage 2+ sharding.
- Activations are *additional* to every number above and scale with tokens per device, not parameters (13.2).

### 19.1 Worked ZeRO arithmetic for the 7B row

World size 8, Stage 1: per device $=14+28+14+56/8=63$ GB — still too large for a 40 GB device once activations are added.
Stage 2: gradients also sharded: $14+28+14/8+56/8\approx50.8$ GB.
Stage 3: everything sharded: $(14+28+14+56)/8=14$ GB per device, at the price of all-gathering weights layer by layer (6.2).
The three numbers, computed before any code runs, already dictate the parallelism plan of Section 14 — this is why memory accounting is Section 4 and not an appendix.

### 19.2 What changes with 8-bit optimizers

8-bit Adam (Dettmers et al., 2022) stores each moment in 1 byte with blockwise quantization: the Adam column falls from 8 to about 2 bytes per parameter, so the 7B total falls from 112 GB to about 70 GB. The tradeoff is quantization error in the second moment, which matters most when $v$ spans many orders of magnitude across coordinates — a testable small-run comparison, not a faith decision.

---

## 20. Learning-Rate Schedule Worksheets

### 20.1 A cosine schedule you can audit by hand

Peak $\eta_{\max}=3\times10^{-4}$, floor $\eta_{\min}=3\times10^{-5}$, warmup 100 steps, total 1,000 steps (theory notebook Section 3 computes the curve).

| Step | Phase | $\eta$ (approx.) | Why |
|---:|---|---:|---|
| 0 | warmup | $0$ | No trust in initial gradients yet (12.1) |
| 50 | warmup | $1.5\times10^{-4}$ | Linear ramp |
| 100 | peak | $3.0\times10^{-4}$ | Warmup complete |
| 300 | cosine | $2.6\times10^{-4}$ | Slow early decay preserves exploration |
| 550 | cosine | $1.65\times10^{-4}$ | Midpoint of the run, half-decayed in cosine phase |
| 800 | cosine | $7.5\times10^{-5}$ | Late training polishes |
| 1,000 | floor | $3.0\times10^{-5}$ | Nonzero floor avoids a hard stop |

### 20.2 Warmup-stable-decay as an alternative

The warmup-stable-decay schedule holds $\eta_{\max}$ constant after warmup and decays only in the final fraction of training. Its advantage is operational: the decay phase can be started from any stable-phase checkpoint, so the total token budget need not be fixed at launch — a direct response to the budget uncertainty in Section 18's table. Its risk is the mirror image: most of training runs at peak LR, so stability monitoring (10.2–10.3) carries more weight.

### 20.3 Schedule bugs that look like model bugs

- Decay indexed by *micro-step* instead of optimizer step under gradient accumulation: the schedule runs $k$ times too fast for accumulation factor $k$ (3.2). Loss plateaus early and is blamed on data.
- Warmup counted in tokens in one config and steps in another after a batch-size change: the effective warmup silently scales with batch.
- Cosine $T$ set to the *planned* run length, then the run is extended: LR is already at the floor and extended training barely moves. Restarting the schedule's decay phase (20.2) is the cleaner fix.

Each bug is a units error, and each is caught by logging $\eta$ against optimizer step at launch (10.3) rather than by watching the loss for a week.

---

## 21. Communication Worksheets

### 21.1 All-reduce time for the gradient bucket

A ring all-reduce of $M$ bytes over $N$ ranks moves about $2M(N-1)/N$ bytes per rank. For a 2 GB gradient bucket at 25 GB/s effective interconnect bandwidth: time $\approx2\times2/25\approx0.16$ s for large $N$ — comparable to a whole optimizer step's compute in small models, which is why gradient buckets are overlapped with backward computation (6.3) instead of sent at the end.

### 21.2 Bandwidth hierarchy, stated as a design rule

| Link | Typical role | Parallelism placed there |
|---|---|---|
| Intra-device (registers/shared memory) | attention, matmul tiles | kernel fusion |
| Intra-node (e.g. NVLink-class) | tensor-parallel all-reduce every layer | TP (14) |
| Inter-node within a job (e.g. InfiniBand-class) | pipeline sends, ZeRO gathers | PP, ZeRO |
| Across jobs / to storage | checkpoints, data loading | asynchronous copy only |

The rule from Section 14 in operational form: *the more often a collective fires per step, the faster the link it must live on.* Tensor parallelism fires per layer; data parallelism fires once per step. Swapping their placement is the single most common cluster-design error this chapter can prevent.

### 21.3 Stragglers quantified

If step time is set by the slowest of $N$ ranks and per-rank step times have standard deviation $\sigma$ around mean $\mu$, the expected maximum of $N$ roughly Gaussian times grows like $\mu+\sigma\sqrt{2\ln N}$. At $N=1024$ that is $\mu+3.7\sigma$: a modest 5% per-rank jitter becomes an ~18% fleet slowdown. This is Section 6.5 with the arithmetic shown, and it is why "average GPU utilization" overstates delivered MFU (7.4).

---

## 22. Data Pipeline Mathematics

### 22.1 Epochs, tokens, and the repetition question

Tokens seen $D$ over a corpus of $U$ unique tokens is $D/U$ epochs. Muennighoff et al. (2023, data-constrained scaling) found repeated data is nearly as good as fresh data for up to about 4 epochs, with diminishing returns thereafter — so a data-constrained run should grow epochs toward 4 before it grows parameters past the compute-optimal point of Section 7.3.

### 22.2 Packing efficiency

Packing documents into fixed blocks of length $s$: efficiency is real tokens / padded tokens. For document lengths $(128,512,900,64,256,700,1024,300)$ packed naively one-per-block at $s=1024$: real $=3884$, padded $=8192$, efficiency $47\%$. Sorted/greedy bin packing of the same documents recovers most of the gap; the theory notebook Section 12 computes this example. Cross-document attention inside a packed block must be masked or position-reset (20.3's cousin, Section 9.2), or packing buys tokens and pays in leakage.

### 22.3 Filtering is a distribution change

Deduplication and quality filtering (9.3) change $p_{\text{data}}$, hence change the optimum of Section 2's objective, not merely its cost. Removing 30% of a corpus as near-duplicates can *improve* loss per token while *raising* loss per wall-clock hour if the filter is slow; report both, and version the filter with the dataset (Chapter 19's lineage rules) so a loss change can be attributed to data or code.

---

## 23. Debugging Playbook With Decision Points

### 23.1 Loss is NaN within the first hundred steps

Check, in order: (1) a learning-rate or warmup units bug (20.3); (2) an unscaled fp16 gradient overflowing (8.2); (3) a mask that makes some rows all-pad, producing a 0/0 mean (Chapter 15 Section 05, Problem-table); (4) logits entering softmax in bf16 with extreme values (17.1). Each check is a print statement, not a retrain.

### 23.2 Loss plateaus above the scaling-law prediction

Compare achieved loss to the Section 7.2 power-law expectation at the current FLOP count. If FLOPs are on plan but loss is not, suspect data (22.3) or a silent LR schedule bug (20.3) before architecture: both shift the whole curve, while architecture changes mostly shift its asymptote.

### 23.3 Throughput decays over hours

A slow decline in tokens/sec with stable step loss suggests thermal, fragmentation, or data-loader drift, not optimization. Decompose step time (10.4) into compute, communication, and data wait; the straggler arithmetic of 21.3 says a single degrading rank taxes the whole fleet, so per-rank step-time histograms — not fleet averages — are the diagnostic.

### 23.4 The small-run contract

Before any large launch, a small run must demonstrate: loss decreases, gradients have finite norms, a checkpoint resumes to a bitwise-continuing trajectory (15.3), and the Section 19 memory estimate matches measured device memory within ~10%. Four properties, each cheap at small scale and each ruinous to discover at large scale. This is Section 10.5 operationalized as a gate.

---

## 24. Five End-to-End Scenarios

### Scenario A: the hallway estimate that fails
"We have 8x80 GB, let's data-parallel a 13B." Section 19: static state alone is 208 GB *per replica*. Verdict before code: impossible; needs Stage-3 sharding or model parallelism (14).

### Scenario B: the batch-size raise that stalls training
Batch raised 8x, LR raised 8x linearly, loss diverges in warmup. Sections 11–12: the run crossed $\mathcal{B}_{\text{noise}}$; linear scaling no longer applies. Fix: $\sqrt{}$ scaling or a measured noise-scale batch, plus longer warmup.

### Scenario C: the checkpoint that ate the run
Checkpoint every 100 steps at 180 s per checkpoint and 2 s steps: overhead $=180/(100\times2)=90\%$ of step time spent checkpointing. Section 15.2's interval ($\approx$2.2 h at these numbers) reduces this to ~1–2%.

### Scenario D: the resume that spiked the loss
After a restart, loss jumps 0.4 and slowly recovers. Section 15.3: optimizer moments were not restored. The spike is the signature, the missing $m,v$ state is the cause, the small-run resume gate (23.4) is the prevention.

### Scenario E: the bigger cluster that got slower
Doubling devices by doubling tensor parallelism across nodes made steps slower. Sections 14 and 21.2: TP collectives now fire per layer over the inter-node link. Rebalance toward data/pipeline parallelism on the slow link instead.

Each scenario is the same method applied five times: convert the symptom to the quantity this chapter assigns to it (bytes, noise scale, interval, state, bandwidth), compute, then act.


---

## 25. Formula Summary for Training at Scale

| Quantity | Formula | Section |
|---|---|---|
| Training FLOPs | $\approx 6ND$ | 7.1 |
| MFU | achieved FLOPs/s / peak FLOPs/s | 7.4 |
| Tokens per step | micro-batch $\times$ seq len $\times$ DP $\times$ accum | 3.1 |
| Steps | $S=D/B$ | 11.3 |
| Static memory | params $\times$ bytes/param (16–18 for AdamW) | 13.1 |
| Gradient noise scale | $\mathcal{B}_{\text{noise}}=\mathrm{tr}(\Sigma)/|\mathbf{g}|^2$ | 11.1 |
| Ring all-reduce bytes/rank | $2M(N-1)/N$ | 21.1 |
| Pipeline bubble | $(p-1)/(m+p-1)$ for $p$ stages, $m$ micro-batches | 5.3 |
| Failure-free probability | $e^{-NT/M}$ | 15.1 |
| Young checkpoint interval | $\sqrt{2C\mu}$ | 15.2 |
| Linear LR scaling | $\eta\propto B$ (small-batch regime) | 12.1 |
| Square-root LR scaling | $\eta\propto\sqrt{B}$ (variance regime) | 12.2 |
| Adam update | $\eta\,\hat m/(\sqrt{\hat v}+\epsilon)$ with bias correction | 2.2–2.3 |
| Gradient clipping | $\mathbf{g}\leftarrow\mathbf{g}\min(1,c/\|\mathbf{g}\|)$ | 2.5 |
| Packing efficiency | real tokens / padded tokens | 22.2 |
| Energy | devices $\times$ hours $\times$ power $\times$ PUE | 16 |

## 26. Pre-Launch Checklist (Gate for Section 23.4)

Memory and parallelism:

1. Static memory per device computed from Section 19's table, not from parameter count alone.
2. Activation memory measured on one real batch at target sequence length.
3. TP degree placed on the intra-node interconnect only (21.2).
4. PP bubble fraction computed for the chosen micro-batch count (5.3).
5. DP all-reduce bucket size and overlap verified in a profile (21.1).
6. ZeRO stage chosen from the arithmetic of 19.1, and its extra communication accepted explicitly.

Optimization:

7. Batch size in tokens, steps $S=D/B$, and $\mathcal{B}_{\text{noise}}$ regime stated (11).
8. LR rule (linear vs square-root vs layer-wise) justified for that regime (12).
9. Warmup counted in optimizer steps, and schedule indexed by optimizer step under accumulation (20.3).
10. Gradient clipping threshold set and norm logging enabled (2.5, 10.2).
11. Loss computed in fp32; master weights in fp32 (17.1).

Data:

12. Unique tokens $U$, epochs $D/U$, and the repetition argument recorded (22.1).
13. Packing efficiency measured; document masks or position resets verified (22.2).
14. Dedup/filter version pinned to the dataset version (22.3).

Reliability:

15. Checkpoint interval from Young's approximation, overhead percentage computed (15.2).
16. Checkpoint contains weights, optimizer, step, schedule position, data order, RNG (15.3).
17. Small-run resume test passed before launch (23.4).
18. Spike policy agreed in advance: roll back, skip batch or lower LR, resume (17.3).

Reporting:

19. Tokens/sec reported together with MFU (7.4).
20. Loss plotted against FLOPs as well as steps (23.2).
21. Energy estimate recorded with the run configuration (16).
22. Any mid-run change (LR, data, batch) logged as a new run segment, not edited history.

## 27. Glossary

| Term | Meaning |
|---|---|
| Activation checkpointing | Recompute activations in backward to save memory (13.2, Chen et al. 2016) |
| Bubble | Idle pipeline time between stages (5.3) |
| Critical batch size | Batch beyond which tokens/step stops buying efficiency (11) |
| Data parallelism | Replicated model, sharded data, gradient all-reduce (5.1) |
| FlashAttention | IO-aware exact attention avoiding $s\times s$ materialization (Dao et al. 2022) |
| Gradient accumulation | Several micro-batches per optimizer step (3.2) |
| Gradient noise scale | $\mathrm{tr}(\Sigma)/|\mathbf{g}|^2$ (11.1, McCandlish et al. 2018) |
| LAMB | Layer-wise trust-ratio Adam variant for huge batches (12.3, You et al. 2020) |
| Loss scaling | Multiplying loss to keep fp16 gradients in range (8.2) |
| MFU | Model FLOPs utilization (7.4) |
| Pipeline parallelism | Layers split across stages (5.3) |
| Tensor parallelism | Individual matrices split across devices (5.2, Shoeybi et al. 2019) |
| Warmup | Initial LR ramp (3.3, 20) |
| WSD schedule | Warmup-stable-decay (20.2) |
| ZeRO | Sharded optimizer/gradient/weight states (13.3, Rajbhandari et al. 2020) |

## 28. Historical Line

| Year | Work | Idea used here |
|---:|---|---|
| 1974 | Young, checkpoint interval approximation | $\sqrt{2C\mu}$ rule (15.2) |
| 2015 | Kingma and Ba, Adam | Moment estimates and bias correction (2.2) |
| 2016 | Chen et al., activation checkpointing | Memory/compute tradeoff (13.2) |
| 2017 | Goyal et al., large-minibatch SGD | Linear LR scaling with warmup (12.1) |
| 2018 | Micikevicius et al., mixed precision | fp32 master weights, loss scaling (17.1) |
| 2018 | McCandlish et al., gradient noise scale | Critical batch size (11) |
| 2019 | Shoeybi et al., Megatron-LM | Tensor parallelism (14) |
| 2019 | Huang et al., GPipe | Pipeline parallelism and bubble (5.3) |
| 2019 | Loshchilov and Hutter, AdamW | Decoupled weight decay (2.4) |
| 2020 | Kaplan et al., neural scaling laws | Power-law loss vs compute (7.2) |
| 2020 | Rajbhandari et al., ZeRO | Sharded training state (13.3) |
| 2020 | You et al., LAMB | Very-large-batch training (12.3) |
| 2021 | Narayanan et al., Megatron-LM training at scale | Combined 3D parallelism in practice (14) |
| 2022 | Hoffmann et al., Chinchilla | Compute-optimal tokens/parameter (7.3, 18) |
| 2022 | Dao et al., FlashAttention | Memory-efficient exact attention (13.2) |
| 2022 | Dettmers et al., 8-bit optimizers | Quantized Adam states (19.2) |
| 2022 | Zhang et al., OPT logbook | Spike rollback practice in the open (17.3) |
| 2023 | Muennighoff et al., data-constrained scaling | Value of repeated epochs (22.1) |

---

## 29. Computed Tables You Can Reproduce

### 29.1 Cosine schedule values (peak 3e-4, floor 3e-5, warmup 100, total 1000)

| Step | Learning rate | Phase |
|---:|---:|---|
| 0 | 0.000e+00 | warmup |
| 50 | 1.500e-04 | warmup |
| 100 | 3.000e-04 | cosine decay |
| 150 | 2.979e-04 | cosine decay |
| 200 | 2.919e-04 | cosine decay |
| 250 | 2.819e-04 | cosine decay |
| 300 | 2.684e-04 | cosine decay |
| 350 | 2.518e-04 | cosine decay |
| 400 | 2.325e-04 | cosine decay |
| 450 | 2.112e-04 | cosine decay |
| 500 | 1.884e-04 | cosine decay |
| 550 | 1.650e-04 | cosine decay |
| 600 | 1.416e-04 | cosine decay |
| 650 | 1.188e-04 | cosine decay |
| 700 | 9.750e-05 | cosine decay |
| 750 | 7.822e-05 | cosine decay |
| 800 | 6.158e-05 | cosine decay |
| 850 | 4.809e-05 | cosine decay |
| 900 | 3.814e-05 | cosine decay |
| 950 | 3.205e-05 | cosine decay |
| 1000 | 3.000e-05 | cosine decay |

### 29.2 Static memory per device under ZeRO stages (7B, 112 GB total, Section 19.1 generalized)

| World size | Stage 1 GB/device | Stage 2 GB/device | Stage 3 GB/device |
|---:|---:|---:|---:|
| 1 | 112.0 | 112.0 | 112.0 |
| 2 | 84.0 | 77.0 | 56.0 |
| 4 | 70.0 | 59.5 | 28.0 |
| 8 | 63.0 | 50.8 | 14.0 |
| 16 | 59.5 | 46.4 | 7.0 |
| 32 | 57.8 | 44.2 | 3.5 |
| 64 | 56.9 | 43.1 | 1.8 |

These rows assume the 14 GB bf16 compute weights and 28 GB fp32 master are replicated in Stages 1-2 and sharded only in Stage 3, matching Section 13.3's definitions.

### 29.3 Checkpoint interval from Young's rule (checkpoint cost 180 s)

| Job MTBF (hours) | Optimal interval (hours) | Overhead at optimum (approx.) |
|---:|---:|---:|
| 12 | 1.10 | 4.56% of step time |
| 24 | 1.55 | 3.23% of step time |
| 48 | 2.19 | 2.28% of step time |
| 96 | 3.10 | 1.61% of step time |
| 192 | 4.38 | 1.14% of step time |
| 384 | 6.20 | 0.81% of step time |

### 29.4 Failure-free probability for a 24-hour window (Section 15.1)

| Devices | Assumed device MTBF (hours) | P(no failure in 24 h) |
|---:|---:|---:|
| 8 | 50000 | 0.996 |
| 64 | 50000 | 0.970 |
| 256 | 50000 | 0.884 |
| 1024 | 50000 | 0.612 |
| 4096 | 50000 | 0.140 |

### 29.5 Tokens, steps, and wall time at 2.0 s/step (300B tokens)

| Batch (M tokens) | Optimizer steps | Wall time at 2.0 s/step (days) |
|---:|---:|---:|
| 0.5 | 600,000 | 13.9 |
| 1 | 300,000 | 6.9 |
| 2 | 150,000 | 3.5 |
| 4 | 75,000 | 1.7 |
| 8 | 37,500 | 0.9 |

Reading 29.5 with Section 11: the 8M-token row is only a good idea if the measured gradient noise scale says those extra tokens per step still buy variance reduction.

### 29.6 MFU arithmetic for one configuration

Peak per device 300 TFLOPs (illustrative bf16 dense), 128 devices: peak $=38.4$ PFLOPs/s. A 7B model training on 4M tokens per 2.0 s step performs $6\times7\times10^9\times4\times10^6/2.0=8.4\times10^{16}$ FLOPs/s $=84$ PFLOPs/s of *required* useful compute at that token rate — above peak, so the configuration is impossible as stated; either step time rises or MFU accounting exposes an error. This kind of impossibility check (7.4) catches misreported throughput before it reaches a launch review: useful FLOPs/s can never exceed devices $\times$ peak.

---

## 30. AdamW Trace You Can Verify by Hand

Scalar $	heta_0=1.5$, constant gradient $g=0.30$, $eta_1=0.9$, $eta_2=0.999$, $\eta=10^{-3}$, weight decay $0.1$ (AdamW, Section 2.4). Each row is one optimizer step; the theory notebook's first cell performs the first of these.

| Step | $m$ | $v$ | $\hat m$ | $\hat v$ | Update size | $	heta$ after step |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.03000 | 0.000090 | 0.30000 | 0.090000 | 0.001000 | 1.49885 |
| 2 | 0.05700 | 0.000180 | 0.30000 | 0.090000 | 0.001000 | 1.49770 |
| 3 | 0.08130 | 0.000270 | 0.30000 | 0.090000 | 0.001000 | 1.49655 |
| 4 | 0.10317 | 0.000359 | 0.30000 | 0.090000 | 0.001000 | 1.49540 |
| 5 | 0.12285 | 0.000449 | 0.30000 | 0.090000 | 0.001000 | 1.49425 |
| 6 | 0.14057 | 0.000539 | 0.30000 | 0.090000 | 0.001000 | 1.49310 |
| 7 | 0.15651 | 0.000628 | 0.30000 | 0.090000 | 0.001000 | 1.49195 |
| 8 | 0.17086 | 0.000717 | 0.30000 | 0.090000 | 0.001000 | 1.49080 |
| 9 | 0.18377 | 0.000807 | 0.30000 | 0.090000 | 0.001000 | 1.48965 |
| 10 | 0.19540 | 0.000896 | 0.30000 | 0.090000 | 0.001000 | 1.48851 |
| 11 | 0.20586 | 0.000985 | 0.30000 | 0.090000 | 0.001000 | 1.48736 |
| 12 | 0.21527 | 0.001074 | 0.30000 | 0.090000 | 0.001000 | 1.48621 |
| 13 | 0.22374 | 0.001163 | 0.30000 | 0.090000 | 0.001000 | 1.48506 |
| 14 | 0.23137 | 0.001252 | 0.30000 | 0.090000 | 0.001000 | 1.48391 |
| 15 | 0.23823 | 0.001341 | 0.30000 | 0.090000 | 0.001000 | 1.48276 |
| 16 | 0.24441 | 0.001429 | 0.30000 | 0.090000 | 0.001000 | 1.48161 |
| 17 | 0.24997 | 0.001518 | 0.30000 | 0.090000 | 0.001000 | 1.48047 |
| 18 | 0.25497 | 0.001606 | 0.30000 | 0.090000 | 0.001000 | 1.47932 |
| 19 | 0.25947 | 0.001695 | 0.30000 | 0.090000 | 0.001000 | 1.47817 |
| 20 | 0.26353 | 0.001783 | 0.30000 | 0.090000 | 0.001000 | 1.47702 |

Two facts are visible only because the trace is printed: the bias corrections (2.3) make the first update already of size $pprox\eta$ rather than $\eta/10$, and with a constant gradient Adam's step size saturates at $pprox\eta$ — Adam normalizes gradient *scale* away, which is why gradient magnitude information must be monitored separately (10.2).

## 31. Activation Memory by Layer Count

Assumptions stated so you can rescale them: 0.35 GB of saved activations per layer at the chosen micro-batch and sequence length (theory notebook Section 7 uses the same figure), checkpointing to layer boundaries only.

| Layers | Without checkpointing (GB) | Boundary-only checkpointing (GB) | Saving |
|---:|---:|---:|---:|
| 12 | 4.2 | 1.2 | 71% |
| 24 | 8.4 | 1.7 | 80% |
| 32 | 11.2 | 2.0 | 82% |
| 48 | 16.8 | 2.4 | 86% |
| 80 | 28.0 | 3.1 | 89% |
| 96 | 33.6 | 3.4 | 90% |

The checkpointed column grows as $\sqrt{L}$ segments (Chen et al., 2016, Section 13.2): the saving is bought with roughly one extra forward pass, so at 80 layers the question is never 'checkpoint or not' but 'how much recompute can the MFU budget (7.4) afford'.

## 32. Launch Decision Worksheet (Worked)

A complete pass for a hypothetical 13B run on 64 devices of 80 GB. Each line is a decision, its source section, and its number.

1. Static state: $13\times10^9\times16$ B $=208$ GB total (13.1, 19).
2. Per-device static under ZeRO-3 at world 64: $208/64=3.25$ GB — ample room for activations.
3. But ZeRO-3 gathers weights per layer (13.3); if interconnect profiling (21.1) shows gather-bound steps, prefer TP8 x PP2 x DP4 (=64) instead (14).
4. Under TP8xPP2: model state per device $=208/(8\times2)=13$ GB before DP sharding choices.
5. Activations: choose micro-batch so measured activation memory (26.2) leaves at least 15% device headroom for fragmentation.
6. Activation checkpointing on (31): at 40 layers it cuts activation memory by about 84% for ~30% recompute.
7. Tokens: target $D=260$B tokens ($D/N=20$, the Chinchilla-side ratio of 7.3/18).
8. FLOPs: $6ND=6\times13\times10^9\times260\times10^9\approx2.0\times10^{22}$ (7.1).
9. Batch: start at 2M tokens/step while the noise scale is small (11), ramp to 4M late in training.
10. Steps at an average 3M tokens/step: $260\text{B}/3\text{M}\approx86{,}700$ optimizer steps (11.3).
11. LR: peak $2\times10^{-4}$ with 2,000-step warmup, cosine to 10% (20.1), schedule indexed by optimizer step (20.3).
12. Clipping at global norm 1.0, norm logged every step (2.5, 10.2).
13. Checkpoint cost measured, then interval from Young's rule (15.2, 29.3) — not copied from another run.
14. Data: unique tokens and epochs stated (22.1); packing efficiency measured above 90% or packing is reworked (22.2).
15. Small-run gate (23.4, 26) signed off: learns, resumes exactly, memory within 10% of line 2–5 estimates.
16. Reporting: MFU, tokens/sec, loss-vs-FLOPs, and energy estimate all on the run dashboard from step one (16, 26.19–26.22).

If any line cannot be filled with a number, that line — not the cluster — is what blocks the launch.

## 33. What 'Good' Looks Like at Three Scales

| | 1B on 8 devices | 13B on 64 devices | 70B on 1024 devices |
|---|---|---|---|
| Static state (19) | 16 GB | 208 GB | 1,120 GB |
| Simplest viable plan | Data parallel + ZeRO-1 | TP/PP/DP mix or ZeRO-3 (32) | 3D parallelism, TP intra-node (14) |
| Dominant risk | Under-using devices (low MFU) | Communication balance (21) | Failures are routine (15.1) |
| Checkpoint logic | Minutes apart is cheap | Young's rule with measured cost (29.3) | Interval, format, and restore time are design inputs |
| Batch logic | Noise scale is small; modest batch (11) | Ramp with training progress (11.1) | Very large batch; LR rule must be justified (12) |
| First diagnostic when wrong | Loss and grad-norm traces (10.2) | Step-time decomposition (23.3) | Per-rank histograms; stragglers (21.3) |


The pattern across the row is the chapter in one line: as scale grows, the binding constraint moves from optimization, to communication, to reliability — and the mathematics that matters moves with it (Sections 2, 6, 15 respectively).

---

## 34. Ten Review Questions With Worked Answers

### Q1. A 3B model trains with AdamW in mixed precision. How much static memory must the plan hold?
Answer: at 16 B/param (13.1), $3\times10^9\times16=48$ GB, before activations.
Step 1: identify the components — bf16 weights 6 GB, fp32 master 12 GB, gradients 6 GB, Adam moments 24 GB.
Step 2: notice Adam moments are half the total; sharding them first (ZeRO-1) is the highest-value memory move.
Conclusion: a single 80 GB device can hold the static state; whether it can hold activations at the target sequence length is a separate, measured question (26.2).

### Q2. Tokens double from 150B to 300B at fixed batch. What happens to steps and to the LR schedule?
Answer: steps double (11.3, $S=D/B$).
The schedule's cosine horizon $T$ must double too (20.1); leaving $T$ at the old value parks the LR at its floor for the second half of training (20.3).
Conclusion: token budget, step count, and schedule are one decision, not three.

### Q3. Batch rises from 1M to 4M tokens. Which LR changes are defensible?
Answer: linear scaling ($\times4$) is defensible only while the batch remains near or below the measured gradient noise scale (11–12.1).
Square-root scaling ($\times2$) is the variance-regime choice (12.2).
Doing nothing is also a choice: it spends the extra tokens on variance reduction instead of step size.
Conclusion: measure $\mathcal{B}_{\text{noise}}$; the regime chooses the rule.

### Q4. Why can data-parallel replicas not simply average their *weights* after training independently?
Answer: independently trained replicas land in different loss basins related by permutation symmetries; naive weight averaging mixes mismatched coordinates and typically destroys the model.
Gradient averaging each step (5.1) keeps replicas in the *same* basin by construction.
Conclusion: synchronization during training is the mechanism, not a post-hoc merge.

### Q5. A run's MFU falls when devices double. Name the first three suspects.
Suspect 1: a parallelism degree was placed on a slower link (21.2).
Suspect 2: stragglers — the expected slowest-rank penalty grows as $\sqrt{2\ln N}$ (21.3).
Suspect 3: communication that used to overlap compute no longer does at the new bucket/step geometry (6.3, 21.1).
Conclusion: decompose step time (23.3) before changing the model or the data.

### Q6. Checkpoint cost is 300 s, job MTBF is 24 h. How often should you checkpoint?
Young's rule (15.2): $\sqrt{2\times300\times86\,400}\approx7{,}200$ s, about every 2 hours.
Checkpointing every 15 minutes would spend $300/900=33\%$ of wall time writing checkpoints.
Conclusion: more frequent is not safer; it is slower, measurably.

### Q7. After a restart the loss spikes and recovers over thousands of steps. What was most likely missing?
Answer: the Adam moments $m,v$ (15.3).
Without them the effective per-coordinate step sizes are wrong until the moments re-accumulate.
Weights-only checkpoints do not crash; they silently change the trajectory.
Conclusion: the checkpoint manifest (26.16) is part of the model.

### Q8. Packing raises tokens/sec 40% but validation loss per token worsens slightly. How do you decide?
Answer: check the document-mask/position handling first (22.2): cross-document leakage is a correctness bug, not a tradeoff.
If masking is correct, compare loss *per wall-clock hour*, since packing's benefit is throughput (22.3's reporting rule).
Conclusion: correctness first, then the rate-normalized metric.

### Q9. Two runs use the same tokens but different $D/N$. Which loss comparison is fair?
Answer: loss against FLOPs ($6ND$, 7.1), not loss against tokens (Common Mistakes table).
The run with more parameters does more compute per token by construction.
Conclusion: Kaplan et al. (2020) and Hoffmann et al. (2022) comparisons live on the FLOP axis for this reason.

### Q10. The gradient norm trace rises steadily for days before any loss spike. What do you do?
Answer: treat it as the early warning it is (10.2, 17.2): inspect attention logits and the data stream, prepare a rollback point, and consider clipping or a modest LR reduction *before* the spike.
Riding through (17.3) spends tokens in a degrading basin.
Conclusion: norm traces exist so that the spike is a confirmation, not a surprise.

---

## 35. Failure Taxonomy for Large Runs

| Symptom | First quantity to compute | Section | Typical fix |
|---|---|---|---|
| OOM at first step | Static bytes/param and activation estimate | 13, 19 | Shard state; checkpoint activations |
| OOM after hours | Activation growth from longer sequences in the data mix | 13.2 | Cap sequence bucket; dynamic batching |
| Loss NaN immediately | LR/warmup units; mask 0/0 rows | 20.3, 23.1 | Fix units; guard empty rows |
| Loss NaN mid-run | Gradient norm trace before the event | 10.2, 17.3 | Roll back; skip batch; clip |
| Loss plateaus early | Schedule horizon vs actual steps | 20.3, Q2 | Re-index schedule to optimizer steps |
| Loss above scaling prediction | Loss-vs-FLOPs gap | 23.2 | Audit data and LR before architecture |
| Throughput low from step one | Step-time decomposition; link placement | 21.2, 23.3 | Move TP intra-node; overlap buckets |
| Throughput decays over hours | Per-rank step-time histograms | 21.3 | Replace/repair the straggler rank |
| Resume changes the loss curve | Checkpoint manifest diff | 15.3, Q7 | Restore optimizer/RNG/schedule state |
| Checkpointing dominates runtime | Checkpoint cost vs Young's interval | 15.2, Q6 | Lengthen interval to the optimum |
| Bigger batch, worse training | Gradient noise scale vs new batch | 11, Q3 | Change LR rule to the regime |
| Packing hurts quality | Mask/position audit inside blocks | 22.2, Q8 | Document masks or position resets |

---

## 36. Derivations in Steps

### 36.1 Where $6ND$ comes from

1. A matrix multiply of an $N$-parameter model on one token costs about $2N$ FLOPs in the forward pass (one multiply and one add per parameter).
2. The backward pass costs about twice the forward pass, $4N$, because it computes gradients with respect to both activations and weights.
3. Forward plus backward is therefore about $6N$ FLOPs per token.
4. Over $D$ tokens, total training compute is $C\approx6ND$ (Kaplan et al., 2020).
5. Attention's $O(s^2)$ terms, embeddings, and optimizer math are omitted; at long sequence lengths they must be added back explicitly.
6. Every MFU, budget, and scaling-law number in this chapter is this derivation with different symbols solved for.

### 36.2 Ring all-reduce cost

1. Split the gradient tensor of size $M$ into $N$ chunks, one per rank.
2. Reduce-scatter: after $N-1$ send/receive rounds, each rank holds the summed version of one chunk; each rank has sent $(N-1)M/N$ bytes.
3. All-gather: another $N-1$ rounds distribute the summed chunks; another $(N-1)M/N$ bytes per rank.
4. Total per rank: $2M(N-1)/N$, which tends to $2M$ for large $N$ (21.1).
5. Consequence: all-reduce cost per step is nearly independent of world size but proportional to gradient size — hence gradient *size* (parameters) and step *frequency* set the data-parallel communication bill.

### 36.3 Pipeline bubble fraction

1. With $p$ stages and $m$ micro-batches, the first micro-batch needs $p-1$ stage-steps before the last stage is busy.
2. The same drain happens at the end, so idle stage-steps total about $2(p-1)$ against useful work proportional to $m+p-1$ in the standard GPipe accounting (Huang et al., 2019).
3. The bubble fraction quoted in Section 5.3 is $(p-1)/(m+p-1)$.
4. Doubling micro-batches roughly halves the bubble but raises activation memory (13.2), because more micro-batches are in flight.
5. Interleaved schedules (Narayanan et al., 2021) reduce the bubble further by assigning multiple layer chunks per device, at the price of more communication.

### 36.4 Young's checkpoint interval, intuitively

1. Checkpoint too rarely, and each failure destroys, on average, half an interval of work.
2. Checkpoint too often, and the fixed cost $C$ per checkpoint dominates (Q6's 33% example).
3. Expected waste per unit time is roughly $C/\tau+\tau/(2\mu)$ for interval $\tau$ and job MTBF $\mu$: checkpoint cost rate plus expected lost-work rate.
4. Setting the derivative to zero, $-C/\tau^2+1/(2\mu)=0$, gives $\tau=\sqrt{2C\mu}$ (Young, 1974).
5. Daly (2006) refines the constants; the square-root *shape* — interval grows slowly with reliability — is the durable lesson.

---

## 37. Data Order, Shuffling, and Repetition

### 37.1 Why order is a mathematical choice

SGD's convergence story assumes gradient estimates that are (nearly) unbiased samples of the full-data gradient (Section 2.1).
A data order that presents all documents from one source, then all from another, violates that assumption block by block: each block's updates pull toward that source's optimum.
Shuffling is therefore not housekeeping; it is what makes the mini-batch gradient an estimate of the intended objective $p_{\text{data}}$ (22.3).
Curriculum orderings (easy before hard) are deliberate, *declared* violations of the shuffle assumption and should be reported as part of the method when used.

### 37.2 Repetition with reshuffling

At $D/U$ epochs (22.1), each epoch should reshuffle and, ideally, re-pack (22.2), so the model does not memorize a fixed neighbour structure inside packed blocks.
Muennighoff et al. (2023) quantify the value decay of repetition; the operational rule is: repeat up to a few epochs before preferring more unique data, and treat epoch count as a scaling variable alongside $N$ and $D$ in Section 7's laws.

### 37.3 Validation must not leak into the shuffle

A validation document that enters the training shuffle — through a dedup failure (9.3) or a shared packed block (22.2) — flatters validation loss and corrupts every scaling comparison built on it (23.2).
Decontamination (Chapter 16) is therefore part of *this* chapter's mathematics too: the loss is only an estimate of the intended quantity if the estimator's data hygiene holds.

---

## 38. The Chapter in One Decision Sequence

1. State $N$, $D$, and the ratio $D/N$ you are targeting, and why (7.3, 18).
2. Compute $6ND$ and convert it to device-days at an honest MFU, not at peak (36.1, 7.4).
3. Compute static memory at 16–18 B/param and choose the sharding/parallelism plan that fits it (13, 19, 14).
4. Place each parallelism degree on the fastest link its communication frequency allows (21.2).
5. Measure or estimate the gradient noise scale and set batch and LR rule for that regime (11–12).
6. Index the LR schedule by optimizer step and count warmup in the same units (20.3).
7. Set the checkpoint interval from measured cost and expected MTBF; write the checkpoint manifest (15).
8. Verify data: unique tokens, epochs, packing efficiency, masks, decontamination (22, 37).
9. Pass the small-run gate: learns, resumes exactly, memory matches estimates (23.4).
10. Launch with dashboards for loss-vs-FLOPs, gradient norms, per-rank step times, and MFU from step one (10, 23).
11. When something deviates, convert the symptom to its quantity in this chapter and compute before acting (24, 35).
12. Report tokens, FLOPs, MFU, energy, and the data version together, so the next run can actually be compared to this one (16, 22.3).

---

## 39. Twenty One-Line Calculations Every LLM Engineer Should Do Fluently

1. FLOPs for 7B on 1T tokens: $6\times7\times10^9\times10^{12}=4.2\times10^{22}$.
2. Steps for 1T tokens at 4M tokens/step: $10^{12}/(4\times10^6)=250{,}000$ steps.
3. Static memory for 1B AdamW: $10^9\times16$ B $=16$ GB.
4. Adam moments alone for 70B: $70\times10^9\times8$ B $=560$ GB.
5. ZeRO-3 static per device for 70B on 512 devices: $1120/512\approx2.2$ GB.
6. Pipeline bubble at $p=8$, $m=32$: $7/39\approx18\%$.
7. Pipeline bubble at $p=8$, $m=128$: $7/135\approx5\%$ — micro-batches buy back the bubble (36.3).
8. Ring all-reduce traffic for a 14 GB gradient at large $N$: about $2\times14=28$ GB per rank (36.2).
9. At 25 GB/s effective bandwidth, that all-reduce costs about $28/25\approx1.1$ s if not overlapped (21.1).
10. Gradient accumulation: micro-batch 0.5M tokens, 8 accumulation steps gives a 4M-token optimizer batch (3.2).
11. Warmup of 2,000 steps at 4M tokens/step consumes 8B tokens before peak LR is reached (20.1).
12. Cosine midpoint: at half the post-warmup steps, LR is about halfway between peak and floor on the cosine curve (20.1).
13. Young interval for $C=60$ s, $\mu=12$ h: $\sqrt{2\times60\times43\,200}\approx2{,}277$ s, about 38 minutes (36.4).
14. Failure-free 24 h at $N=256$, device MTBF 50,000 h: $e^{-256\times24/50\,000}\approx e^{-0.123}\approx0.884$ (29.4).
15. Epochs: 300B tokens seen over a 100B-token corpus is 3 epochs (22.1).
16. Packing: 3,884 real tokens in 8,192 padded slots is 47% efficiency (22.2).
17. Energy: 64 devices, 100 h, 0.4 kW, PUE 1.2 gives $64\times100\times0.4\times1.2\approx3{,}072$ kWh (Section 16).
18. MFU check: required useful FLOPs/s above devices-times-peak is impossible, whatever the dashboard says (29.6).
19. $D/N$ for 13B on 260B tokens is 20 tokens per parameter (32, step 7).
20. Clipping: a gradient of norm 13 clipped to 5 is scaled by $5/13\approx0.385$; direction unchanged, norm exactly 5 (2.5).

If any of these takes more than a minute with a calculator, the corresponding section is the reread — the theory notebook turns the same calculations into executable checks.

---

## 40. A Scaling-Law Fit You Can Do on a Napkin

Kaplan et al. (2020) model test loss as a power law in compute, $L(C)\approx (C_c/C)^{\alpha_C}$, over the regime before irreducible entropy dominates. The numbers below are an *illustrative* fit with $\alpha_C=0.05$ and $L=2.0$ at $C=10^{21}$ FLOPs, included so the mechanics — not the constants — can be checked by hand.

| Compute $C$ (FLOPs) | Predicted loss $L(C)$ | Loss drop per 10x compute |
|---:|---:|---:|
| $10^{19}$ | 2.518 | — |
| $10^{20}$ | 2.244 | 0.274 |
| $10^{21}$ | 2.000 | 0.244 |
| $10^{22}$ | 1.783 | 0.217 |
| $10^{23}$ | 1.589 | 0.194 |
| $10^{24}$ | 1.416 | 0.173 |

Three lessons survive replacing the illustrative constants with fitted ones (Hoffmann et al., 2022, fit the joint $N,D$ form instead):

1. Equal *ratios* of compute buy roughly equal *absolute* loss drops in the power-law regime — progress is logarithmic in spending.

2. Extrapolating a fit beyond the compute range it was fitted on is the standard way scaling claims go wrong; the fit is an interpolation tool with modest extrapolation rights.

3. The Chinchilla correction (Hoffmann et al., 2022) was not 'Kaplan was wrong' but 'the joint optimum over $N$ and $D$ at fixed $C=6ND$ sits at larger $D/N$ than early runs used' — exactly the tradeoff surface the theory notebook plots in its Section 11.

### 40.1 Using the table in a planning meeting

- Question: 'What does 10x more compute buy?' Answer from the table: about 0.11 nats at this illustrative exponent, wherever in the table you start.

- Question: 'Should the 10x go to parameters or tokens?' The one-variable table cannot answer; that is the joint $L(N,D)$ fit of Hoffmann et al. (2022), and Section 18's budget states its assumed $D/N$ for this reason.

- Question: 'When do we stop scaling this recipe?' When measured loss departs from the fitted curve beyond noise (23.2), the recipe — data, architecture, or schedule — not the budget, is the first suspect.

---

## 41. Final Self-Check

Answer each in one line with a number, not an adjective.

1. What is $6ND$ for your next run?
2. What MFU are you assuming, and what evidence supports it?
3. How many bytes per parameter does your memory plan hold?
4. Which ZeRO stage or parallelism mix follows from that number?
5. What is your tokens-per-step, and how many optimizer steps follow?
6. Which LR scaling regime are you in, by measured noise scale?
7. In what units are warmup and schedule indexed?
8. What is your checkpoint cost, MTBF assumption, and resulting interval?
9. Does your checkpoint restore optimizer, schedule, data order, and RNG?
10. How many unique tokens and epochs does the run see?
11. What is your measured packing efficiency?
12. Which link carries tensor parallelism, and how often does it fire per step?
13. What does your gradient-norm trace look like on a healthy day?
14. What is the agreed action at the first loss spike?
15. Which small-run gate results are recorded, and where?

A run that can answer all fifteen is ready to be expensive.
A run that cannot is ready to be interesting in the wrong way.
That distinction is the practical content of this section.
The mathematics is not decoration on the launch process.
It is the launch process, written in a form that can be checked.

---

## 42. Symbols Used in the Expansion Sections

| Symbol | Meaning |
|---|---|
| $N$ | Parameter count |
| $D$ | Training tokens seen |
| $U$ | Unique tokens in the corpus |
| $C$ | Training compute in FLOPs, $pprox6ND$ |
| $B$ | Tokens per optimizer step |
| $S$ | Optimizer steps, $D/B$ |
| $\mathcal{B}_{\text{noise}}$ | Gradient noise scale (11.1) |
| $\eta$ | Learning rate |
| $m, v$ | Adam first and second moment estimates |
| $\hat m, \hat v$ | Bias-corrected Adam moments |
| $p$ | Pipeline stages in bubble formulas |
| $m_b$ | Micro-batches per pipeline step (to distinguish from Adam $m$) |
| $M$ | Bytes in a gradient all-reduce; also device MTBF in hours in 15.1 by local convention |
| $\mu$ | Job mean time between failures in Young's rule |
| $C_{\text{ckpt}}$ | Checkpoint write cost in seconds (written $C$ in 15.2) |
| PUE | Power usage effectiveness of the facility |
| MFU | Model FLOPs utilization: useful FLOPs/s divided by peak FLOPs/s |

## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Estimating memory as weights only | Adam moments + master weights dominate (13.1) | Budget 16–18 B/param before activations |
| Scaling LR linearly at huge batch without warmup | Noise scale is exceeded; early steps diverge (11–12) | Warmup + measure $\mathcal{B}_{\text{noise}}$ |
| Tensor parallelism across slow interconnects | Per-layer all-reduce dominates step time | TP intra-node only (Section 14) |
| Checkpointing weights but not optimizer/RNG | Resume silently changes the optimization trajectory | Save full state; test resume on a small run (15.3) |
| Checkpointing as often as possible | Checkpoint cost can exceed failure waste | Young's interval $\sqrt{2C\mu}$ (15.2) |
| Riding through a loss spike | Damaged basin; wasted tokens | Roll back, skip batch / lower LR (17.3) |
| Reporting tokens/sec without MFU | Hides communication and recompute waste | Report MFU with tokens/sec (Section 7.4) |
| Comparing runs by tokens when $D/N$ differs wildly | Different points on the scaling surface | Compare loss vs FLOPs (Kaplan 2020; Hoffmann 2022) |
| Loss in bf16 | Small-probability errors; lost updates | fp32 loss path and master weights (17.1) |
| Packing sequences without position/mask care | Cross-document attention leaks | Document masks / position resets (Section 9.2) |

## Practice Exercises

1. Compute one AdamW update by hand for a scalar parameter.
2. Clip a gradient vector to a target norm.
3. Build a warmup plus cosine learning-rate schedule.
4. Compute effective batch size in tokens.
5. Estimate memory for Adam training with and without sharding.
6. Compute a pipeline bubble fraction.
7. Determine tensor-parallel shard shapes for a linear layer.
8. Estimate training FLOPs from parameter and token counts.
9. Compute model FLOPs utilization from achieved throughput.
10. Create a launch checklist for a small reproducible training run.

## Why This Matters for AI

| Concept | AI Impact |
|---|---|
| $6ND$ FLOPs accounting (7.1) | Budgeting GPT-3/Chinchilla-class runs (Brown et al. 2020; Hoffmann et al. 2022) |
| ZeRO sharding (13.3) | Training trillion-parameter models (Rajbhandari et al. 2020) |
| Tensor/pipeline parallelism (14) | Megatron-LM training at scale (Shoeybi et al. 2019; Narayanan et al. 2021) |
| Gradient noise scale (11) | Choosing efficient batch sizes (McCandlish et al. 2018) |
| Mixed precision (17.1) | Standard LLM training numerics (Micikevicius et al. 2018) |

Good LLM training is not only about choosing a model architecture. The optimizer can diverge, the memory plan can be impossible, the communication plan can waste the cluster, the data stream can repeat contaminated text, and the checkpoint can fail to restore optimizer state. The mathematics in this section lets you reason about those failures before the run burns budget.

## Bridge to Fine-Tuning Math

Fine-tuning keeps many of the same scale constraints, but usually changes the parameter-update surface. The next section studies full fine-tuning, adapters, LoRA-style low-rank updates, prompt tuning, and preference-oriented objectives. The training-at-scale accounting here remains useful because every fine-tuning method still has memory, optimizer, throughput, and evaluation budgets.

## References

- Jared Kaplan et al., "Scaling Laws for Neural Language Models", 2020: https://arxiv.org/abs/2001.08361
- Jordan Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022: https://arxiv.org/abs/2203.15556
- Mohammad Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", 2019: https://arxiv.org/abs/1909.08053
- Samyam Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", 2019: https://arxiv.org/abs/1910.02054
- Priya Goyal et al., "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour", 2017: https://arxiv.org/abs/1706.02677
- Noam Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", 2017: https://arxiv.org/abs/1701.06538
- OpenAI, "AI and Compute", 2018: https://openai.com/research/ai-and-compute
- Diederik Kingma and Jimmy Ba, "Adam: A Method for Stochastic Optimization", ICLR, 2015.
- Ilya Loshchilov and Frank Hutter, "Decoupled Weight Decay Regularization", ICLR, 2019.
- Paulius Micikevicius et al., "Mixed Precision Training", ICLR, 2018.
- Sam McCandlish et al., "An Empirical Model of Large-Batch Training", 2018: https://arxiv.org/abs/1812.06162
- Yang You et al., "Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes", ICLR, 2020.
- Tianqi Chen et al., "Training Deep Nets with Sublinear Memory Cost", 2016: https://arxiv.org/abs/1604.06174
- Tri Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS, 2022.
- Tim Dettmers et al., "LLM.int8() / 8-bit Optimizers via Block-wise Quantization", 2021-2022.
- Susan Zhang et al., "OPT: Open Pre-trained Transformer Language Models", 2022: https://arxiv.org/abs/2205.01068
- Hugo Muennighoff et al., "Scaling Data-Constrained Language Models", NeurIPS, 2023.
- Deepak Narayanan et al., "Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM", SC, 2021.
- John W. Young, "A First Order Approximation to the Optimum Checkpoint Interval", Communications of the ACM, 1974.
- John T. Daly, "A Higher Order Estimate of the Optimum Checkpoint Interval for Restart Dumps", Future Generation Computer Systems, 2006.
- Tom Brown et al., "Language Models are Few-Shot Learners", NeurIPS, 2020: https://arxiv.org/abs/2005.14165
- Mohammad Shoeybi et al. and Samyam Rajbhandari et al. are cited in Sections 13-14; full entries appear in the original References list above.
- For the joint scaling fit: Jordan Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022: https://arxiv.org/abs/2203.15556
- For data repetition: Hugo Muennighoff et al., "Scaling Data-Constrained Language Models", 2023: https://arxiv.org/abs/2305.16264
