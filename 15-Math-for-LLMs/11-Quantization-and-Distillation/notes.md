[<- Mixture of Experts and Routing](../10-Mixture-of-Experts-and-Routing/notes.md) | [Home](../../README.md) | [RAG Math and Retrieval ->](../12-RAG-Math-and-Retrieval/notes.md)

---

# Quantization and Distillation

Quantization compresses tensors by changing their numeric representation. Distillation compresses behavior by training a smaller or cheaper model to imitate a stronger teacher. Both are central to making LLMs cheaper to store, serve, and deploy.

## Overview

The simplest uniform quantizer is:

$$
q=\mathrm{round}(x/s)+z,\qquad \hat x=s(q-z).
$$

Here $s$ is the scale and $z$ is the zero point. Quantization asks how much error this approximation introduces and whether the hardware can exploit the smaller representation.

Distillation uses a teacher distribution:

$$
L_\mathrm{KD}=\tau^2D_\mathrm{KL}(p_T^{(\tau)}\Vert p_S^{(\tau)}).
$$

The student learns from teacher probabilities, generated sequences, features, or preferences. Quantization and distillation can be combined: distill to a smaller model, then quantize it for deployment.

## Prerequisites

- Logits, softmax, KL divergence, and cross-entropy
- Matrix multiplication and tensor shapes
- Inference memory and bandwidth constraints
- Fine-tuning and LoRA basics

## Companion Notebooks

| Notebook | Purpose |
| --- | --- |
| [theory.ipynb](theory.ipynb) | Demonstrates uniform quantization, clipping, group-wise scales, error versus bits, SmoothQuant intuition, distillation temperature, KL loss, and memory savings. |
| [exercises.ipynb](exercises.ipynb) | Ten practice problems for quantizer formulas, memory savings, clipping, KL distillation, and deployment checks. |

## Learning Objectives

After this section, you should be able to:

- Implement affine quantization and dequantization.
- Explain scale, zero point, clipping, and quantization error.
- Compare per-tensor, per-channel, and group-wise quantization.
- Explain PTQ, QAT, GPTQ intuition, AWQ intuition, and QLoRA.
- Compute memory savings from lower bit widths.
- Compute distillation KL loss with temperature.
- Explain logit, sequence, feature, and preference distillation.
- Build a quality and latency evaluation checklist for compressed LLMs.

## Table of Contents

1. [Compression Goals](#1-compression-goals)
   - 1.1 [Memory reduction](#11-memory-reduction)
   - 1.2 [Bandwidth reduction](#12-bandwidth-reduction)
   - 1.3 [Compute support](#13-compute-support)
   - 1.4 [Quality preservation](#14-quality-preservation)
   - 1.5 [Pareto frontier](#15-pareto-frontier)
2. [Uniform Quantization](#2-uniform-quantization)
   - 2.1 [Affine quantizer](#21-affine-quantizer)
   - 2.2 [Dequantization](#22-dequantization)
   - 2.3 [Scale](#23-scale)
   - 2.4 [Zero point](#24-zero-point)
   - 2.5 [Quantization error](#25-quantization-error)
3. [Granularity](#3-granularity)
   - 3.1 [Per-tensor](#31-pertensor)
   - 3.2 [Per-channel](#32-perchannel)
   - 3.3 [Group-wise](#33-groupwise)
   - 3.4 [Activation quantization](#34-activation-quantization)
   - 3.5 [KV-cache quantization](#35-kvcache-quantization)
4. [Post-Training Quantization](#4-posttraining-quantization)
   - 4.1 [Calibration data](#41-calibration-data)
   - 4.2 [Clipping](#42-clipping)
   - 4.3 [Weighted error](#43-weighted-error)
   - 4.4 [GPTQ intuition](#44-gptq-intuition)
   - 4.5 [AWQ intuition](#45-awq-intuition)
5. [Quantization-Aware Training and QLoRA](#5-quantizationaware-training-and-qlora)
   - 5.1 [Fake quantization](#51-fake-quantization)
   - 5.2 [Straight-through estimator](#52-straightthrough-estimator)
   - 5.3 [QLoRA pattern](#53-qlora-pattern)
   - 5.4 [Optimizer memory](#54-optimizer-memory)
   - 5.5 [Dequantization path](#55-dequantization-path)
6. [Distillation Basics](#6-distillation-basics)
   - 6.1 [Teacher and student](#61-teacher-and-student)
   - 6.2 [Soft targets](#62-soft-targets)
   - 6.3 [Temperature](#63-temperature)
   - 6.4 [KL distillation loss](#64-kl-distillation-loss)
   - 6.5 [Hard-label mixture](#65-hardlabel-mixture)
7. [LLM Distillation Types](#7-llm-distillation-types)
   - 7.1 [Logit distillation](#71-logit-distillation)
   - 7.2 [Sequence distillation](#72-sequence-distillation)
   - 7.3 [Feature distillation](#73-feature-distillation)
   - 7.4 [Preference distillation](#74-preference-distillation)
   - 7.5 [Reasoning trace distillation](#75-reasoning-trace-distillation)
8. [Error and Evaluation](#8-error-and-evaluation)
   - 8.1 [Perplexity shift](#81-perplexity-shift)
   - 8.2 [Task score shift](#82-task-score-shift)
   - 8.3 [Calibration shift](#83-calibration-shift)
   - 8.4 [Layer sensitivity](#84-layer-sensitivity)
   - 8.5 [Outlier channels](#85-outlier-channels)
9. [Deployment Choices](#9-deployment-choices)
   - 9.1 [Weight-only quantization](#91-weightonly-quantization)
   - 9.2 [Weight-activation quantization](#92-weightactivation-quantization)
   - 9.3 [KV-cache quantization](#93-kvcache-quantization)
   - 9.4 [Distill then quantize](#94-distill-then-quantize)
   - 9.5 [Hardware format](#95-hardware-format)
10. [Debugging Checklist](#10-debugging-checklist)
61. [Final Note](#61-final-note)
58. [Final Computed Sweep: Effective Memory Including Embeddings Kept in bf16](#58-final-computed-sweep-effective-memory-including-embeddings-kept-in-bf16)
59. [Closing Notes for Section 11](#59-closing-notes-for-section-11)
55. [Large Computed Reference Tables](#55-large-computed-reference-tables)
56. [Step-by-Step Problem Set C](#56-step-by-step-problem-set-c)
57. [Section 11 Takeaways](#57-section-11-takeaways)
51. [Ten More Worked Derivations](#51-ten-more-worked-derivations)
52. [What To Log During PTQ](#52-what-to-log-during-ptq)
53. [Compression Failure Taxonomy](#53-compression-failure-taxonomy)
54. [Final Reading Guide for Section 11](#54-final-reading-guide-for-section-11)
11. [Affine Quantization Error Budget](#11-affine-quantization-error-budget)
12. [Granularity as a Bias-Variance Tradeoff](#12-granularity-as-a-bias-variance-tradeoff)
13. [GPTQ, AWQ, and SmoothQuant as Error Placement](#13-gptq-awq-and-smoothquant-as-error-placement)
14. [Quantization-Aware Training and the Straight-Through Estimator](#14-quantization-aware-training-and-the-straight-through-estimator)
15. [Distillation Loss, Temperature, and Gradient Scale](#15-distillation-loss-temperature-and-gradient-scale)
16. [Sequence, Feature, and Preference Distillation in LLMs](#16-sequence-feature-and-preference-distillation-in-llms)
17. [Distill-Then-Quantize Versus Quantize-Then-Distill](#17-distill-then-quantize-versus-quantize-then-distill)
18. [Evaluation Gates for Compressed Models](#18-evaluation-gates-for-compressed-models)
19. [Computed Tables](#19-computed-tables)
20. [Why This Matters for AI](#20-why-this-matters-for-ai)
21. [Ten Worked Problems With Solutions](#21-ten-worked-problems-with-solutions)
22. [Diagnostic Protocol and Launch Gate](#22-diagnostic-protocol-and-launch-gate)
23. [Glossary and Formula Appendix](#23-glossary-and-formula-appendix)
24. [Calibration Set Design](#24-calibration-set-design)
25. [Clipping and Range Estimation](#25-clipping-and-range-estimation)
26. [KV-Cache Quantization Budget](#26-kv-cache-quantization-budget)
27. [Activation Quantization and Outliers](#27-activation-quantization-and-outliers)
28. [Student Capacity Gap and Teacher Choice](#28-student-capacity-gap-and-teacher-choice)
29. [Computed Worksheets](#29-computed-worksheets)
30. [Extended Solved Problems (steps shown)](#30-extended-solved-problems-(steps-shown))
31. [Pre-Release Checklist (expanded)](#31-pre-release-checklist-(expanded))
32. [Questions and Answers](#32-questions-and-answers)
33. [Sensitive Layers and Mixed Precision Allocation](#33-sensitive-layers-and-mixed-precision-allocation)
34. [Quantization and Long-Context Behaviour](#34-quantization-and-long-context-behaviour)
35. [Distillation Data Governance](#35-distillation-data-governance)
36. [Compression and Safety Behaviour](#36-compression-and-safety-behaviour)
37. [Choosing Bit-Width From the Constraint, Not Habit](#37-choosing-bit-width-from-the-constraint-not-habit)
38. [Recovery Fine-Tuning After PTQ](#38-recovery-fine-tuning-after-ptq)
39. [Combinatorial Computed Tables](#39-combinatorial-computed-tables)
40. [Layer-Sensitivity Worksheet (template with example entries)](#40-layer-sensitivity-worksheet-(template-with-example-entries))
41. [Ten Additional Practice Checks (with answers)](#41-ten-additional-practice-checks-(with-answers))
42. [Summary Decision Table](#42-summary-decision-table)
43. [Worked Example: Quantizing a Small Tensor End to End](#43-worked-example-quantizing-a-small-tensor-end-to-end)
44. [Worked Example: Reading a Distillation Curve](#44-worked-example-reading-a-distillation-curve)
45. [Error Propagation Across Layers](#45-error-propagation-across-layers)
46. [Compression Reporting Standard](#46-compression-reporting-standard)
47. [Formula Appendix (expanded)](#47-formula-appendix-(expanded))
48. [Extended Glossary](#48-extended-glossary)
49. [Audit Worksheet: Configuration Record](#49-audit-worksheet-configuration-record)
50. [Chapter-15 Integration](#50-chapter-15-integration)
   - 10.1 [Calibration representativeness](#101-calibration-representativeness)
   - 10.2 [Layer-by-layer error](#102-layerbylayer-error)
   - 10.3 [Reference comparisons](#103-reference-comparisons)
   - 10.4 [Latency measurement](#104-latency-measurement)
   - 10.5 [Quality gates](#105-quality-gates)

---

## Compression Map

| Method | Changes | Needs data? | Main benefit | Main risk |
| --- | --- | --- | --- | --- |
| PTQ | Numeric format after training | Calibration data | Fast compression | Calibration mismatch |
| QAT | Training simulates quantization | Training data | Better low-bit robustness | More compute |
| QLoRA | Quantized frozen base plus LoRA | Fine-tune data | Cheap adaptation | Activation memory remains |
| Logit distillation | Student matches teacher probabilities | Teacher outputs | Smaller model behavior transfer | Teacher errors transfer |
| Sequence distillation | Student trains on teacher completions | Teacher generations | Simple data pipeline | Diversity loss |

## 1. Compression Goals

This part studies compression goals as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Memory reduction](#1-memory-reduction) | store fewer bytes per parameter or cache entry | $M=P\cdot b/8$ |
| [Bandwidth reduction](#1-bandwidth-reduction) | read fewer bytes per generated token | $T_\mathrm{read}\approx M/\mathrm{bandwidth}$ |
| [Compute support](#1-compute-support) | low precision helps only when kernels and hardware support it | $T_\mathrm{kernel}$ |
| [Quality preservation](#1-quality-preservation) | compressed outputs should stay close to reference outputs | $D_\mathrm{KL}(p_\mathrm{ref}\Vert p_\mathrm{comp})$ |
| [Pareto frontier](#1-pareto-frontier) | compression is a quality-cost tradeoff | $\mathrm{quality}$ versus $\mathrm{memory},\mathrm{latency},\mathrm{cost}$ |

### 1.1 Memory reduction

**Main idea.** Store fewer bytes per parameter or cache entry.

Core relation:

$$M=P\cdot b/8$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.2 Bandwidth reduction

**Main idea.** Read fewer bytes per generated token.

Core relation:

$$T_\mathrm{read}\approx M/\mathrm{bandwidth}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.3 Compute support

**Main idea.** Low precision helps only when kernels and hardware support it.

Core relation:

$$T_\mathrm{kernel}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.4 Quality preservation

**Main idea.** Compressed outputs should stay close to reference outputs.

Core relation:

$$D_\mathrm{KL}(p_\mathrm{ref}\Vert p_\mathrm{comp})$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 1.5 Pareto frontier

**Main idea.** Compression is a quality-cost tradeoff.

Core relation:

$$\mathrm{quality}$ versus $\mathrm{memory},\mathrm{latency},\mathrm{cost}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 2. Uniform Quantization

This part studies uniform quantization as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Affine quantizer](#2-affine-quantizer) | map real values to integer grid points | $q=\mathrm{round}(x/s)+z$ |
| [Dequantization](#2-dequantization) | recover an approximate real value | $\hat x=s(q-z)$ |
| [Scale](#2-scale) | the step size controls resolution | $s=(x_\max-x_\min)/(q_\max-q_\min)$ |
| [Zero point](#2-zero-point) | asymmetric quantization uses an integer offset | $z=q_\min-\mathrm{round}(x_\min/s)$ |
| [Quantization error](#2-quantization-error) | rounding error is bounded by half a step before clipping | $|x-\hat x|\le s/2$ |

### 2.1 Affine quantizer

**Main idea.** Map real values to integer grid points.

Core relation:

$$q=\mathrm{round}(x/s)+z$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This tiny formula is the bridge between real model weights and integer storage.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.2 Dequantization

**Main idea.** Recover an approximate real value.

Core relation:

$$\hat x=s(q-z)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.3 Scale

**Main idea.** The step size controls resolution.

Core relation:

$$s=(x_\max-x_\min)/(q_\max-q_\min)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.4 Zero point

**Main idea.** Asymmetric quantization uses an integer offset.

Core relation:

$$z=q_\min-\mathrm{round}(x_\min/s)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 2.5 Quantization error

**Main idea.** Rounding error is bounded by half a step before clipping.

Core relation:

$$|x-\hat x|\le s/2$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 3. Granularity

This part studies granularity as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Per-tensor](#3-pertensor) | one scale for the whole tensor | $s$ shared by all entries |
| [Per-channel](#3-perchannel) | one scale per output channel or column | $s_c$ |
| [Group-wise](#3-groupwise) | one scale per block of weights | $s_g$ |
| [Activation quantization](#3-activation-quantization) | activation ranges depend on input data | $x=x(\mathrm{batch})$ |
| [KV-cache quantization](#3-kvcache-quantization) | cache precision affects long-context memory and attention quality | $M_\mathrm{KV}\propto b_\mathrm{KV}$ |

### 3.1 Per-tensor

**Main idea.** One scale for the whole tensor.

Core relation:

$$s$ shared by all entries$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.2 Per-channel

**Main idea.** One scale per output channel or column.

Core relation:

$$s_c$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.3 Group-wise

**Main idea.** One scale per block of weights.

Core relation:

$$s_g$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** Group scales are one reason modern low-bit LLM quantization can work better than one global scale.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.4 Activation quantization

**Main idea.** Activation ranges depend on input data.

Core relation:

$$x=x(\mathrm{batch})$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 3.5 KV-cache quantization

**Main idea.** Cache precision affects long-context memory and attention quality.

Core relation:

$$M_\mathrm{KV}\propto b_\mathrm{KV}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 4. Post-Training Quantization

This part studies post-training quantization as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Calibration data](#4-calibration-data) | estimate activation or weight ranges from representative examples | $D_\mathrm{cal}$ |
| [Clipping](#4-clipping) | smaller range improves resolution but clips outliers | $x\leftarrow\mathrm{clip}(x,-c,c)$ |
| [Weighted error](#4-weighted-error) | some weights matter more because activations amplify them | $\Vert X(W-\hat W)\Vert^2$ |
| [GPTQ intuition](#4-gptq-intuition) | quantize weights while compensating error using approximate second-order information | $\Delta L\approx \frac12\Delta w^\top H\Delta w$ |
| [AWQ intuition](#4-awq-intuition) | protect activation-salient channels during weight quantization | $|x_j|$ indicates sensitivity |

### 4.1 Calibration data

**Main idea.** Estimate activation or weight ranges from representative examples.

Core relation:

$$D_\mathrm{cal}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.2 Clipping

**Main idea.** Smaller range improves resolution but clips outliers.

Core relation:

$$x\leftarrow\mathrm{clip}(x,-c,c)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.3 Weighted error

**Main idea.** Some weights matter more because activations amplify them.

Core relation:

$$\Vert X(W-\hat W)\Vert^2$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** Quantizing a weight is more harmful when common activations magnify that weight's error.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.4 GPTQ intuition

**Main idea.** Quantize weights while compensating error using approximate second-order information.

Core relation:

$$\Delta L\approx \frac12\Delta w^\top H\Delta w$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 4.5 AWQ intuition

**Main idea.** Protect activation-salient channels during weight quantization.

Core relation:

$$|x_j|$ indicates sensitivity$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 5. Quantization-Aware Training and QLoRA

This part studies quantization-aware training and qlora as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Fake quantization](#5-fake-quantization) | simulate quantization during training while keeping gradients useful | $\hat W=Q(W)$ in forward |
| [Straight-through estimator](#5-straightthrough-estimator) | treat rounding as identity in backward | $\partial \mathrm{round}(x)/\partial x\approx 1$ |
| [QLoRA pattern](#5-qlora-pattern) | freeze a quantized base and train low-rank adapters | $W\approx Q(W_0)+(\alpha/r)BA$ |
| [Optimizer memory](#5-optimizer-memory) | optimizer states are needed for trainable adapters, not frozen base weights | $M_\mathrm{opt}\propto P_\mathrm{trainable}$ |
| [Dequantization path](#5-dequantization-path) | many kernels dequantize blocks on the fly for matmul | $q,s\rightarrow \hat W$ |

### 5.1 Fake quantization

**Main idea.** Simulate quantization during training while keeping gradients useful.

Core relation:

$$\hat W=Q(W)$ in forward$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.2 Straight-through estimator

**Main idea.** Treat rounding as identity in backward.

Core relation:

$$\partial \mathrm{round}(x)/\partial x\approx 1$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.3 QLoRA pattern

**Main idea.** Freeze a quantized base and train low-rank adapters.

Core relation:

$$W\approx Q(W_0)+(\alpha/r)BA$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.4 Optimizer memory

**Main idea.** Optimizer states are needed for trainable adapters, not frozen base weights.

Core relation:

$$M_\mathrm{opt}\propto P_\mathrm{trainable}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 5.5 Dequantization path

**Main idea.** Many kernels dequantize blocks on the fly for matmul.

Core relation:

$$q,s\rightarrow \hat W$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 6. Distillation Basics

This part studies distillation basics as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Teacher and student](#6-teacher-and-student) | train a smaller model to match a larger model | $p_T(y\mid x),p_S(y\mid x)$ |
| [Soft targets](#6-soft-targets) | teacher probabilities contain similarity information beyond one-hot labels | $p_T$ |
| [Temperature](#6-temperature) | soften probability distributions | $p_i^{(\tau)}=\mathrm{softmax}(z_i/\tau)$ |
| [KL distillation loss](#6-kl-distillation-loss) | minimize divergence from teacher to student | $L_\mathrm{KD}=\tau^2D_\mathrm{KL}(p_T^{(\tau)}\Vert p_S^{(\tau)})$ |
| [Hard-label mixture](#6-hardlabel-mixture) | combine task loss with distillation loss | $L=\alpha L_\mathrm{CE}+(1-\alpha)L_\mathrm{KD}$ |

### 6.1 Teacher and student

**Main idea.** Train a smaller model to match a larger model.

Core relation:

$$p_T(y\mid x),p_S(y\mid x)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.2 Soft targets

**Main idea.** Teacher probabilities contain similarity information beyond one-hot labels.

Core relation:

$$p_T$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.3 Temperature

**Main idea.** Soften probability distributions.

Core relation:

$$p_i^{(\tau)}=\mathrm{softmax}(z_i/\tau)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.4 KL distillation loss

**Main idea.** Minimize divergence from teacher to student.

Core relation:

$$L_\mathrm{KD}=\tau^2D_\mathrm{KL}(p_T^{(\tau)}\Vert p_S^{(\tau)})$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** Distillation trains the student on the teacher's distribution, not only the final answer.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 6.5 Hard-label mixture

**Main idea.** Combine task loss with distillation loss.

Core relation:

$$L=\alpha L_\mathrm{CE}+(1-\alpha)L_\mathrm{KD}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 7. LLM Distillation Types

This part studies llm distillation types as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Logit distillation](#7-logit-distillation) | match next-token distributions | $D_\mathrm{KL}(p_T(\cdot\mid x)\Vert p_S(\cdot\mid x))$ |
| [Sequence distillation](#7-sequence-distillation) | train on teacher-generated completions | $y\sim p_T(\cdot\mid x)$ |
| [Feature distillation](#7-feature-distillation) | match hidden states or attention maps | $\Vert h_T-h_S\Vert^2$ |
| [Preference distillation](#7-preference-distillation) | transfer teacher comparisons or judge preferences | $y^+\succ_T y^-$ |
| [Reasoning trace distillation](#7-reasoning-trace-distillation) | train on teacher-produced intermediate reasoning when appropriate | $p_S(r,y\mid x)$ |

### 7.1 Logit distillation

**Main idea.** Match next-token distributions.

Core relation:

$$D_\mathrm{KL}(p_T(\cdot\mid x)\Vert p_S(\cdot\mid x))$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.2 Sequence distillation

**Main idea.** Train on teacher-generated completions.

Core relation:

$$y\sim p_T(\cdot\mid x)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.3 Feature distillation

**Main idea.** Match hidden states or attention maps.

Core relation:

$$\Vert h_T-h_S\Vert^2$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.4 Preference distillation

**Main idea.** Transfer teacher comparisons or judge preferences.

Core relation:

$$y^+\succ_T y^-$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 7.5 Reasoning trace distillation

**Main idea.** Train on teacher-produced intermediate reasoning when appropriate.

Core relation:

$$p_S(r,y\mid x)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 8. Error and Evaluation

This part studies error and evaluation as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Perplexity shift](#8-perplexity-shift) | quantization can be measured by held-out NLL change | $\Delta L=L_\mathrm{quant}-L_\mathrm{base}$ |
| [Task score shift](#8-task-score-shift) | compression should be checked on downstream tasks | $\Delta S=S_\mathrm{comp}-S_\mathrm{base}$ |
| [Calibration shift](#8-calibration-shift) | probabilities may become miscalibrated | $\mathrm{ECE}$ |
| [Layer sensitivity](#8-layer-sensitivity) | some layers or projections tolerate fewer bits poorly | $\Delta L_\ell$ |
| [Outlier channels](#8-outlier-channels) | activation outliers often dominate low-bit error | $\max |x_j|$ |

### 8.1 Perplexity shift

**Main idea.** Quantization can be measured by held-out nll change.

Core relation:

$$\Delta L=L_\mathrm{quant}-L_\mathrm{base}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.2 Task score shift

**Main idea.** Compression should be checked on downstream tasks.

Core relation:

$$\Delta S=S_\mathrm{comp}-S_\mathrm{base}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.3 Calibration shift

**Main idea.** Probabilities may become miscalibrated.

Core relation:

$$\mathrm{ECE}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.4 Layer sensitivity

**Main idea.** Some layers or projections tolerate fewer bits poorly.

Core relation:

$$\Delta L_\ell$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 8.5 Outlier channels

**Main idea.** Activation outliers often dominate low-bit error.

Core relation:

$$\max |x_j|$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 9. Deployment Choices

This part studies deployment choices as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Weight-only quantization](#9-weightonly-quantization) | reduce weight bandwidth while leaving activations higher precision | $W_q,\ x_\mathrm{fp}$ |
| [Weight-activation quantization](#9-weightactivation-quantization) | quantize both weights and activations for more kernel speed | $W_q,\ x_q$ |
| [KV-cache quantization](#9-kvcache-quantization) | increase context or batch capacity | $b_\mathrm{KV}\downarrow$ |
| [Distill then quantize](#9-distill-then-quantize) | a smaller student can also be quantized | $S\rightarrow Q(S)$ |
| [Hardware format](#9-hardware-format) | INT4, INT8, FP8, and NF4 need matching kernels | $\mathrm{format}\rightarrow\mathrm{kernel}$ |

### 9.1 Weight-only quantization

**Main idea.** Reduce weight bandwidth while leaving activations higher precision.

Core relation:

$$W_q,\ x_\mathrm{fp}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.2 Weight-activation quantization

**Main idea.** Quantize both weights and activations for more kernel speed.

Core relation:

$$W_q,\ x_q$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.3 KV-cache quantization

**Main idea.** Increase context or batch capacity.

Core relation:

$$b_\mathrm{KV}\downarrow$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.4 Distill then quantize

**Main idea.** A smaller student can also be quantized.

Core relation:

$$S\rightarrow Q(S)$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 9.5 Hardware format

**Main idea.** Int4, int8, fp8, and nf4 need matching kernels.

Core relation:

$$\mathrm{format}\rightarrow\mathrm{kernel}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
## 10. Debugging Checklist

This part studies debugging checklist as compression math. The useful habit is to separate storage format, dequantized computation, approximation error, and evaluation.

| Subtopic | Question | Formula |
| --- | --- | --- |
| [Calibration representativeness](#10-calibration-representativeness) | calibration data should match deployment prompts | $D_\mathrm{cal}\approx D_\mathrm{deploy}$ |
| [Layer-by-layer error](#10-layerbylayer-error) | inspect where compression hurts | $\Vert X(W-\hat W)\Vert$ |
| [Reference comparisons](#10-reference-comparisons) | compare logits before and after compression | $\max|z-\hat z|$ |
| [Latency measurement](#10-latency-measurement) | confirm the chosen format is actually faster | $T_\mathrm{comp}<T_\mathrm{base}$ |
| [Quality gates](#10-quality-gates) | do not ship a compressed model without task and safety checks | $\Delta S$ bounded |

### 10.1 Calibration representativeness

**Main idea.** Calibration data should match deployment prompts.

Core relation:

$$D_\mathrm{cal}\approx D_\mathrm{deploy}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.2 Layer-by-layer error

**Main idea.** Inspect where compression hurts.

Core relation:

$$\Vert X(W-\hat W)\Vert$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.3 Reference comparisons

**Main idea.** Compare logits before and after compression.

Core relation:

$$\max|z-\hat z|$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.4 Latency measurement

**Main idea.** Confirm the chosen format is actually faster.

Core relation:

$$T_\mathrm{comp}<T_\mathrm{base}$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** A smaller file is not automatically a faster model if kernels do not support the format well.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.
### 10.5 Quality gates

**Main idea.** Do not ship a compressed model without task and safety checks.

Core relation:

$$\Delta S$ bounded$$

Quantization changes the numerical representation of tensors. Distillation changes the training signal so a smaller or cheaper model imitates a stronger teacher. Both are compression tools, but they fail in different ways: quantization can introduce numerical error, while distillation can omit teacher capabilities that are not present in the distillation data.

**Worked micro-example.** If weights lie in $[-1,1]$ and we use signed 4-bit integers with values from -8 to 7, a symmetric step size is roughly $s=1/7$. A real weight $0.33$ maps to integer $\mathrm{round}(0.33/s)$ and dequantizes back to the nearest grid point. Smaller $s$ improves resolution near zero but clips large values if the range is too narrow.

**Implementation check.** Always compare base and compressed logits on the same inputs. Then check held-out loss, task quality, calibration, memory, and latency. Compression is successful only if the target tradeoff improves.

**AI connection.** This is a practical compression control variable.

**Common mistake.** Do not report "4-bit" without saying what is quantized, the granularity, the calibration data, and the serving kernel.

---

## 11. Affine Quantization Error Budget

The affine quantizer from Section 2 maps a real weight $w$ to $q=\mathrm{clip}(\mathrm{round}(w/s)+z,q_{min},q_{max})$ and back to $\hat w=s(q-z)$.
Three distinct error sources must be reported separately, because they have different fixes.
**Rounding error** is bounded by $s/2$ for values inside the representable range.
**Clipping error** is unbounded in principle: every value outside $[s(q_{min}-z),s(q_{max}-z)]$ collapses to an endpoint.
**Scale error** appears when the calibration range that set $s$ does not match the deployed distribution.
For symmetric signed INT$b$ quantization, $s=\max|w|/(2^{b-1}-1)$ and the step doubles roughly when one bit is removed.
A useful first diagnostic is the signal-to-quantization-noise ratio, $\mathrm{SQNR}=10\log_{10}(\mathrm{Var}(w)/\mathrm{MSE}(w,\hat w))$ in decibels.
Worked example: weights uniform on $[-1,1]$ have variance $1/3$.
With INT8 symmetric quantization, $s=1/127\approx0.00787$, so ideal rounding MSE is about $s^2/12\approx5.2\times10^{-6}$.
That gives SQNR near $48$ dB, close to the textbook $6.02b+1.76$ dB rule for a full-range uniform signal.
With INT4, $s=1/7\approx0.143$, ideal rounding MSE is about $1.7\times10^{-3}$, and SQNR falls to about $23$ dB.
The 24 dB gap is why INT4 weight-only methods need grouping, calibration, and error compensation rather than naive rounding.
Non-example: reporting only the maximum absolute error hides whether the error is concentrated in a few outlier weights or spread across the tensor.
Example: a tensor can have tiny MSE yet fail if the largest errors land on the few weights multiplied by the largest activations (Section 8.5).
The deployment rule is therefore two-part: report MSE or SQNR **and** a downstream quality metric on a fixed evaluation set.

## 12. Granularity as a Bias-Variance Tradeoff

Granularity chooses how many scales are stored: one per tensor, one per output channel, or one per group of $g$ consecutive weights (Section 3).
Finer granularity tracks local weight ranges better, reducing rounding error where ranges vary across channels.
Finer granularity also stores more scales and zero-points, and can make kernels slower if groups are too small.
For group size $g$ with one fp16 scale per group, scale storage is $16/g$ bits per weight before any zero-point cost.
At $g=128$ that is $0.125$ bits per weight; at $g=32$ it is $0.5$ bits per weight, a real cost on top of a 4-bit payload.
Worked example: a 7B-parameter model at 4 bits needs $3.5$ GB of payload.
With $g=128$ fp16 scales it adds about $0.11$ GB; with $g=32$ it adds about $0.44$ GB.
Neither overhead changes the headline conclusion, but ignoring the $g=32$ overhead overstates compression by more than $10\%$ of the added scale bytes.
Per-channel quantization is the natural minimum for linear layers because output channels have visibly different ranges.
Per-tensor quantization is usually reserved for activations or for hardware that only supports one scale.
Diagnostic: plot the per-channel max-abs distribution before choosing granularity; a wide spread argues for per-channel or grouped scales.

## 13. GPTQ, AWQ, and SmoothQuant as Error Placement

Post-training methods differ mainly in **where they allow the error to live**.
GPTQ (Frantar et al., 2022) quantizes weights one column at a time and updates the remaining unquantized weights to compensate, using approximate second-order (Hessian) information from calibration activations.
The objective is layer output error $\|WX-\hat WX\|$, not weight error $\|W-\hat W\|$, which is why calibration activations matter.
AWQ (Lin et al., 2023) observes that a small fraction of weight channels are salient because their activations are large, and scales those channels up before quantization so relative error there falls.
SmoothQuant (Xiao et al., 2022) migrates difficulty from activations to weights with a per-channel smoothing factor, making both sides easier to quantize for W8A8 inference.
LLM.int8() (Dettmers et al., 2022) keeps outlier activation features in higher precision and quantizes the rest, an early production example of mixed-precision decomposition.
The common lesson is identical to Chapter 15 Section 08's: minimize the error the **task** sees, weighted by activations, not the error a weight histogram sees.
Worked comparison on one layer output $y=Wx$: if one input coordinate has activation magnitude $10\times$ the median, a fixed weight error on that column costs $10\times$ more output error.
Therefore a method that spends precision on high-activation columns can beat a method with lower unweighted weight MSE.
Non-example: choosing a PTQ method by weight MSE alone can select exactly the wrong method for activation-heavy layers.

## 14. Quantization-Aware Training and the Straight-Through Estimator

QAT inserts fake quantization into the forward pass (Section 5.1) so training optimizes the deployed function, not a float proxy.
The rounding function has derivative zero almost everywhere, so backpropagation uses the straight-through estimator (STE): pass the gradient through as if rounding were identity inside the clipping range.
STE is biased, but it gives the scale and clipping parameters a usable learning signal.
Learnable clipping (as in PACT-style methods) often matters more than learnable weights once bit-width is very low.
QAT cost is justified when PTQ quality loss exceeds the product threshold and enough training budget exists; it is not the default for a one-off model release.
QLoRA (Dettmers et al., 2023) is not full QAT: the base model stays frozen in 4-bit NormalFloat (NF4), and LoRA adapters train in higher precision.
NF4 is information-theoretically motivated for approximately normally distributed weights: quantile bins give roughly equal counts per bin.
QLoRA's memory claim must include adapters, gradients for adapters, optimizer states for adapters, and activations — not only the 4-bit base.
Worked memory sketch for a 65B base in NF4: base payload about $32.5$ GB at 4 bits plus scales, which is why the method fit large-model fine-tuning on a single high-memory GPU in the paper's setting.

## 15. Distillation Loss, Temperature, and Gradient Scale

Hinton et al. (2015) soften teacher and student logits with temperature $T$: $p_i=\mathrm{softmax}(z_i/T)$.
Raising $T$ reveals dark knowledge: the teacher's relative preferences among wrong answers, which carry class-similarity structure.
The standard loss is $\mathcal{L}=\alpha T^2\,\mathrm{KL}(p_T^{teacher}\|p_T^{student})+(1-\alpha)\,\mathrm{CE}(y,p^{student})$ (Section 6.5).
The $T^2$ factor compensates the $1/T^2$ shrinkage of soft-target gradients so the two terms keep comparable scale as $T$ changes.
Worked example: teacher logits $(4,2,0)$ at $T=1$ give probabilities about $(0.84,0.11,0.02)$ after normalization with the remaining mass pattern made explicit in the theory notebook.
At $T=4$, the same logits soften to a much flatter distribution, increasing the gradient signal on the second-best class.
Too small $T$ wastes the teacher's ranking information; too large $T$ makes targets approach uniform and the KL term approaches a matching of logits rather than decisions.
Practical sweep: $T\in\{1,2,4\}$ and $\alpha\in\{0.5,0.9\}$ on a validation set is a defensible starter grid, not folklore.
Non-example: distilling only the argmax label is hard-label training with extra steps; it discards the soft structure that motivates distillation.

## 16. Sequence, Feature, and Preference Distillation in LLMs

Logit distillation matches per-token distributions and needs teacher and student to share a vocabulary or a mapping between vocabularies.
Sequence distillation trains the student on teacher-generated text, turning distribution matching into supervised fine-tuning on synthetic data.
That makes data licensing, teacher terms of use, and contamination checks part of the math problem, because the training distribution **is** the teacher's output distribution.
Feature distillation matches hidden states or attention patterns, adding an alignment problem when depths differ.
Preference distillation transfers rankings or chosen/rejected pairs, connecting distillation to the preference objectives of Chapter 15 Section 07.
Reasoning-trace distillation trains on teacher rationales; evaluate final answers and trace faithfulness separately, because fluent traces can rationalize wrong answers.
Self-distillation uses the same or an earlier checkpoint as teacher and can improve calibration or compress ensembling gains into one model.
Failure check: if the student improves on teacher-generated validation data but not on human or held-out data, it has learned teacher style, not task competence.

## 17. Distill-Then-Quantize Versus Quantize-Then-Distill

Pipeline order changes the error budget.
Distill-then-quantize trains a clean student, then accepts a second, independent PTQ error on top of the distillation gap.
Quantize-then-distill (distillation into an already-quantized student, or QAT-style recovery) lets training absorb quantization error, usually at higher training cost.
A useful accounting model adds gaps on a validation metric: total gap $\approx$ distillation gap $+$ quantization gap $+$ interaction term.
The interaction term is why the two-step estimate must be verified end-to-end (Section 9.4).
Decision rule: if PTQ alone already meets the quality gate, do not distill; if a smaller student is required for latency, distill first and quantize with a recovery fine-tune reserved in the budget.

## 18. Evaluation Gates for Compressed Models

A compressed model is a new model for evaluation purposes.
Gate 1, capability: fixed task suite compared against the uncompressed baseline with a pre-registered tolerance.
Gate 2, probability quality: perplexity or log-loss shift on held-out text, because task accuracy can hide degraded probabilities.
Gate 3, calibration: expected calibration error or reliability curves when probabilities drive decisions.
Gate 4, robustness slices: long-context, rare-token, numeric, and safety slices where quantization outliers often appear first.
Gate 5, systems: measured latency, memory, and throughput on the target hardware, not a parameter-count proxy.
Any gate can veto release; average improvement across gates is not a release criterion.
Report the compression configuration with the score: bits, granularity, group size, calibration set size, method, and kernel.
A score without that configuration is not reproducible (Chapter 15 Section 08's reporting rule applied to compression).

## 19. Computed Tables

### 19.1 Weight memory by model size and precision (payload only, decimal GB)

| Parameters | bf16 (16-bit) | INT8 | INT4 |
|---:|---:|---:|---:|
| 1B | 2 GB | 1 GB | 0.5 GB |
| 7B | 14 GB | 7 GB | 3.5 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 34B | 68 GB | 34 GB | 17.0 GB |
| 70B | 140 GB | 70 GB | 35.0 GB |
| 175B | 350 GB | 175 GB | 87.5 GB |

Reading: INT4 quarters bf16 payload, but scales, zero-points, embeddings kept in higher precision, and KV cache are extra (Sections 3, 9).

### 19.2 Ideal uniform-quantization SQNR versus bits (full-range uniform signal, $6.02b+1.76$ dB rule)

| Bits | Levels | Ideal SQNR (dB) | Practical reading |
|---:|---:|---:|---|
| 2 | 4 | 13.8 | research / extreme; needs QAT or grouping |
| 3 | 8 | 19.8 | aggressive weight-only with grouping |
| 4 | 16 | 25.8 | mainstream weight-only PTQ zone (GPTQ/AWQ) |
| 8 | 256 | 49.9 | low-loss default for weights and many activations |
| 16 | 65536 | 98.1 | bf16 reference |

The rule assumes a full-range uniform signal and no clipping; real tensors lose several dB to range mismatch and outliers.

### 19.3 Group-size scale overhead (one fp16 scale per group)

| Group size | Scale bits / weight | Added GB for 7B weights | Added GB for 70B weights |
|---:|---:|---:|---:|
| 32 | 0.500 | 0.44 | 4.38 |
| 64 | 0.250 | 0.22 | 2.19 |
| 128 | 0.125 | 0.11 | 1.09 |
| 256 | 0.062 | 0.05 | 0.55 |

Zero-points, if stored per group, add further bits; asymmetric schemes must count them.

### 19.4 Distillation temperature on teacher logits $(4,2,0)$

| $T$ | $p_1$ | $p_2$ | $p_3$ | Reading |
|---:|---:|---:|---:|---|
| 1 | 0.867 | 0.117 | 0.016 | near-hard |
| 2 | 0.665 | 0.245 | 0.090 | softer dark knowledge |
| 4 | 0.506 | 0.307 | 0.186 | softer dark knowledge |
| 10 | 0.402 | 0.329 | 0.269 | near uniform |







## Common Mistakes

| Mistake | Why Wrong | Fix |
|---|---|---|
| Reporting compression ratio from bits alone | Scales, zero-points, embeddings, and KV cache are ignored (Sections 12, 19.3) | Report total deployed bytes |
| Choosing PTQ by weight MSE | Task error is activation-weighted (Section 13) | Evaluate layer-output and downstream metrics |
| Per-tensor scales for wide-range channels | One scale cannot fit all channels (Section 12) | Per-channel or grouped scales |
| No calibration-set provenance | GPTQ/AWQ quality depends on calibration data (Section 13) | Record source, size, and date of calibration data |
| Distilling without the $T^2$ factor carelessly | Soft-target gradients shrink as $1/T^2$ (Section 15) | Use the standard scaled loss or retune deliberately |
| Hard-label-only 'distillation' | Discards dark knowledge (Section 15) | Match soft distributions or sequences explicitly |
| Quantizing after distillation with no recovery budget | Errors add with an interaction term (Section 17) | Reserve recovery fine-tuning and re-gate |
| Accuracy-only release gates | Calibration and slices degrade first (Section 18) | Five-gate evaluation including calibration and slices |
| Comparing latency across different kernels | Bits do not guarantee speed without kernel support (Sections 2, 9) | Measure on target hardware and kernel |
| Ignoring KV-cache bytes at long context | KV can exceed weight savings (Chapter 15 Section 09) | Budget weights and KV together |


## 20. Why This Matters for AI

| Concept | AI Impact |
|---|---|
| Activation-weighted error | Decides whether a 4-bit model keeps task quality or only looks good on weight MSE |
| Temperature-scaled KL | Transfers teacher ranking structure, enabling smaller students than hard labels allow |
| Grouped scales | Makes INT4 practical by tracking local weight ranges at small storage cost |
| Calibration data | PTQ quality is a data problem as much as a numerics problem |
| Compression gates | Prevent silent calibration, long-context, and safety regressions at release |

## 21. Ten Worked Problems With Solutions

1. **INT8 scale.** Weights in $[-0.8,0.8]$, symmetric INT8: $s=0.8/127=0.00630$; $w=0.5\to q=\mathrm{round}(79.4)=79$, $\hat w=0.4976$, error $0.0024<s/2$. (Section 11)
2. **INT4 error bound.** Same range at INT4: $s=0.8/7=0.114$; worst rounding error $s/2=0.057$, about $18\times$ the INT8 bound. (Section 19.2)
3. **Memory.** 13B model bf16 $=26$ GB; INT4 payload $=6.5$ GB plus $g=128$ scales $13\times0.125/8\approx0.20$ GB. (Sections 12, 19.1)
4. **SQNR.** MSE $=10^{-4}$ on weights with variance $0.04$: SQNR $=10\log_{10}(0.04/10^{-4})=26.0$ dB. (Section 11)
5. **Temperature.** Teacher logits $(4,2,0)$, $T=2$: probabilities about $(0.665,0.245,0.090)$; the second class keeps a quarter of the mass for the student to learn from. (Section 19.4)
6. **KL by hand.** Teacher $(0.7,0.2,0.1)$, student $(0.6,0.3,0.1)$: KL $=0.7\ln(0.7/0.6)+0.2\ln(0.2/0.3)+0.1\ln(1)\approx0.027$ nats. (Section 15)
7. **Mixed loss.** $\alpha=0.9,T=2$, KL $=0.027$, CE $=0.40$: $\mathcal L=0.9\times4\times0.027+0.1\times0.40=0.137$. (Section 15)
8. **Group overhead.** 70B weights, $g=32$: scale bytes $=70\times0.5/8=4.38$ GB on top of $35$ GB INT4 payload — over $12\%$ extra. (Section 19.3)
9. **Pipeline gap.** Distillation gap $1.0$ point, PTQ gap $0.4$, measured total $1.7$: interaction term $0.3$ points must be reported, not hidden. (Section 17)
10. **Gate decision.** INT4 model passes accuracy ($-0.3$ pts) but calibration error doubles: Gate 3 vetoes release until fixed. (Section 18)

## 22. Diagnostic Protocol and Launch Gate

- Fix the uncompressed baseline, evaluation sets, and tolerances before compressing.
- Quantize a copy; never overwrite the only full-precision checkpoint.
- Run layer-by-layer output-error probes to find sensitive layers (Section 8.4).
- Sweep granularity and clipping on a calibration split, select on validation, report on test.
- Evaluate all five gates (Section 18) including long-context and calibration slices.
- Measure latency and memory on target hardware with the production kernel.
- Record bits, group size, method, calibration provenance, and kernel version with every score.
- Keep a full-precision fallback path for the first production window.

## 23. Glossary and Formula Appendix

- **Affine quantization:** $\hat w=s(q-z)$ with scale $s$ and zero-point $z$.
- **Symmetric quantization:** Zero-point fixed at 0; range is $[-a,a]$.
- **PTQ:** Post-training quantization calibrated without full retraining.
- **QAT:** Quantization-aware training with fake quantization and STE.
- **NF4:** 4-bit NormalFloat quantile format used by QLoRA (Dettmers et al., 2023).
- **Dark knowledge:** Teacher's relative probabilities over non-target classes (Hinton et al., 2015).
- **Temperature:** Logit divisor $T$ softening teacher/student distributions.
- **STE:** Straight-through estimator passing gradients through rounding.
- **SQNR:** Signal-to-quantization-noise ratio in dB.
- **Calibration set:** Data used to set scales or Hessian statistics in PTQ.

Key formulas: $s=(w_{max}-w_{min})/(2^b-1)$ (asymmetric), $s=\max|w|/(2^{b-1}-1)$ (symmetric), $\mathcal L_{distill}=\alpha T^2 KL(p_T^t\|p_T^s)+(1-\alpha)CE$.

## References (additions)

- Hinton, G., Vinyals, O., and Dean, J. (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531.
- Jacob, B. et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR.
- Frantar, E. et al. (2023). GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. ICLR.
- Lin, J. et al. (2024). AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. MLSys.
- Xiao, G. et al. (2023). SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. ICML.
- Dettmers, T. et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS.
- Dettmers, T. et al. (2022). LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale. NeurIPS.

## 24. Calibration Set Design

Calibration data determines scales, clipping ranges, and GPTQ Hessian statistics.
It should match the deployed input distribution in language, domain, sequence length, and formatting.
A calibration set of a few hundred to a few thousand sequences is typical for PTQ statistics; larger is not automatically better if it drifts off-distribution.
Worked failure: calibrating a code model only on English prose underestimates activation ranges for code tokens.
The fix is stratified calibration: sample each deployed slice in proportion, then check per-slice ranges.
- Record source and licence of calibration data.
- Record sequence length and number of sequences.
- Record date, because deployed traffic drifts.
- Hold out a separate validation split for method selection.
- Never tune clipping on the test split.
Calibration is a dataset, and datasets need provenance (Chapter 15 Section 08 reporting rule).

## 25. Clipping and Range Estimation

Min-max ranges are fragile: one outlier token can double a scale and waste resolution on all other values.
Percentile clipping (for example 99.9%) trades a small clipping error for a much smaller rounding step.
MSE-optimal clipping searches the range minimizing $\|w-\hat w\|$ on calibration weights; entropy-based methods match distributions.
The search must be per granularity unit (Section 12), not global, when channel ranges differ.
Worked example: weights with 99.9% of mass in $[-0.5,0.5]$ and outliers to $2.0$.
Min-max INT8 gives $s=2/127\approx0.0157$; clipping at $0.5$ gives $s\approx0.0039$, a $4\times$ finer step for 99.9% of weights.
The clipped outliers pay endpoint error, which is exactly the trade percentile methods price in.

## 26. KV-Cache Quantization Budget

Weight compression does not bound total memory at long context; the KV cache grows with sequence length (Chapter 15 Section 09).
Per-token KV bytes are $2\times L\times h_{kv}\times d_h\times b$ for $L$ layers, $h_{kv}$ KV heads, head dim $d_h$, and $b$ bytes per element.
Worked example: $L=32$, $h_{kv}=8$ (GQA), $d_h=128$, bf16: $2\times32\times8\times128\times2=131{,}072$ bytes per token, about $0.13$ MB.
At 32k context that is about $4.3$ GB per sequence before weights.
INT8 KV halves that; INT4 quarters it, with quality risk concentrated at long-range retrieval positions.
Therefore a serving claim must state weight bits **and** KV bits at the advertised context length (Section 18, Gate 5).

## 27. Activation Quantization and Outliers

Activations are harder than weights: ranges vary per token and a few channels carry extreme magnitudes (Dettmers et al., 2022).
Per-token dynamic scales adapt to each token's range but add runtime scale computation.
SmoothQuant (Xiao et al., 2022) shifts activation difficulty into weights with a migration factor chosen per channel.
Outlier-aware mixed precision keeps a small outlier subspace in fp16/int8-high-precision while quantizing the bulk.
Diagnostic: measure per-channel activation kurtosis; heavy tails predict W8A8 failure before any benchmark is run.

## 28. Student Capacity Gap and Teacher Choice

A student far smaller than its teacher may be unable to represent the teacher's function regardless of loss design.
Born-again and self-distillation results show equal-capacity students can still gain, so the gap is about fit, not a hard law.
Choose the smallest teacher that is reliably better than the student on the target task; a much larger teacher can produce worse students when the capacity gap dominates.
Intermediate-layer hints help when the student is deep but narrow, because logits alone under-specify the computation.
Report teacher, student, data, temperature, and $\alpha$ with every distilled score — the same reproducibility rule as quantization (Section 18).

## 29. Computed Worksheets

### 29.1 KV-cache GB per sequence by context (32 layers, 8 KV heads, 128 head dim)

| Context tokens | bf16 GB | INT8 GB | INT4 GB |
|---:|---:|---:|---:|
| 1024 | 0.13 | 0.07 | 0.03 |
| 4096 | 0.54 | 0.27 | 0.13 |
| 8192 | 1.07 | 0.54 | 0.27 |
| 16384 | 2.15 | 1.07 | 0.54 |
| 32768 | 4.29 | 2.15 | 1.07 |
| 65536 | 8.59 | 4.29 | 2.15 |
| 131072 | 17.18 | 8.59 | 4.29 |

### 29.2 Total deployed GB: INT4 weights + bf16 KV (one sequence, 8k context, same KV geometry)

| Model | INT4 weights (GB) | KV at 8k (GB) | Total (GB) | KV share |
|---:|---:|---:|---:|
| 7B | 3.61 | 1.07 | 4.68 | 23% |
| 13B | 6.70 | 1.07 | 7.78 | 14% |
| 34B | 17.53 | 1.07 | 18.60 | 6% |
| 70B | 36.09 | 1.07 | 37.17 | 3% |

Small models at long context are KV-dominated; large models are weight-dominated. Compression priorities follow the larger term.

### 29.3 Ideal rounding MSE by bits for range $[-1,1]$ symmetric ($s^2/12$)

| Bits | Scale $s$ | Ideal MSE | Max rounding error |
|---:|---:|---:|---:|
| 2 | 1.00000 | 8.33e-02 | 0.50000 |
| 3 | 0.33333 | 9.26e-03 | 0.16667 |
| 4 | 0.14286 | 1.70e-03 | 0.07143 |
| 5 | 0.06667 | 3.70e-04 | 0.03333 |
| 6 | 0.03226 | 8.67e-05 | 0.01613 |
| 8 | 0.00787 | 5.17e-06 | 0.00394 |

## 30. Extended Solved Problems (steps shown)


**Problem 1: Per-channel beats per-tensor.**
- Two channels have ranges $[-0.1,0.1]$ and $[-1,1]$.
- Per-tensor INT8 uses $s=1/127$ for both.
- The small channel then spans only about $25$ levels effectively.
- Per-channel gives the small channel $s=0.1/127$, using all 255 levels.
- Conclusion: granularity follows range heterogeneity (Section 12).

**Problem 2: NF4 intuition.**
- Normal weights concentrate near zero.
- Equal-width bins would put most weights in a few central bins.
- Quantile (NF4) bins place bin edges so counts are roughly equal.
- Each 4-bit code is therefore used about equally often.
- That equal use is the information-theoretic point of NF4 (Dettmers et al., 2023).

**Problem 3: When PTQ is enough.**
- Baseline task score 82.0; tolerance is $-0.5$ points.
- INT8 PTQ scores 81.8; INT4 GPTQ scores 81.2.
- INT8 passes the gate; INT4 does not.
- No QAT budget is spent, because a passing configuration exists (Section 18).

**Problem 4: When QAT is justified.**
- INT4 PTQ scores $2.1$ points below baseline, outside tolerance.
- The product requires INT4 memory for on-device deployment.
- QAT/recovery fine-tuning is therefore justified by a failed gate, not by fashion.

**Problem 5: Distillation data check.**
- Student improves on teacher-generated validation by 5 points.
- On held-out human data it improves by 0.2 points.
- Diagnosis: style transfer, not capability transfer (Section 16).
- Fix: evaluate and mix in human/held-out supervision.

## 31. Pre-Release Checklist (expanded)

- [ ] Baseline checkpoint and scores frozen and archived.
- [ ] Quantization configuration fully recorded (bits, granularity, group size, zero-points).
- [ ] Calibration provenance recorded.
- [ ] Layer sensitivity probe run; sensitive layers identified.
- [ ] Clipping sweep done per granularity unit.
- [ ] Five evaluation gates passed or formally waived with named risk.
- [ ] Calibration (ECE) compared to baseline.
- [ ] Long-context slice evaluated at advertised maximum length.
- [ ] Safety/regression slices evaluated.
- [ ] Latency measured on production kernel and hardware.
- [ ] Memory measured including KV at maximum batch and context.
- [ ] Fallback to full precision tested.
- [ ] Distillation teacher/data/terms documented if applicable.
- [ ] Canary plan with automatic rollback threshold defined.

## 32. Questions and Answers


**Q: Why does INT4 need grouping more than INT8?**
A: At 4 bits the step is coarse; grouping lets the scale track local ranges, recovering several dB (Sections 12, 19.2).

**Q: Is QLoRA quantization-aware training?**
A: No. The base is frozen in NF4; LoRA adapters train in higher precision (Dettmers et al., 2023).

**Q: What does temperature reveal?**
A: Relative teacher preferences among non-target classes — dark knowledge (Hinton et al., 2015).

**Q: Why weight MSE misleads?**
A: Output error weights weight error by activation magnitude (Section 13).

**Q: When is distillation pointless?**
A: When PTQ alone meets gates, or the student cannot fit the teacher's function (Sections 17-18, 28).

**Q: What must a compression claim state?**
A: Bits, granularity, group size, calibration provenance, kernel, measured memory/latency, and gate scores.

## 33. Sensitive Layers and Mixed Precision Allocation

Most networks are not uniformly sensitive to quantization.
Early embedding-adjacent layers and final output-adjacent layers often show higher sensitivity in published PTQ practice, but this must be measured per model.
The measurement is a leave-one-layer-in experiment: quantize one layer at a time and record validation loss change.
Layers are then ranked, and the precision budget is spent where the loss slope is steepest.
Mixed precision is therefore an allocation problem: minimize total loss increase subject to a byte budget.
A simple greedy rule assigns higher precision to the layer with the largest loss increase per extra byte saved.
Worked sketch: if quantizing layer A costs $0.30$ loss points and layer B costs $0.02$, B is quantized first at equal byte savings.
Non-example: uniformly quantizing every layer to INT4 because the average sensitivity is low hides the one layer that breaks the model.
Report the sensitivity ranking with the release; it is part of the compression configuration (Section 18).

## 34. Quantization and Long-Context Behaviour

Long-context quality depends on precise attention logits over many positions.
KV quantization error compounds across positions because every attended value carries error.
Retrieval-style long-context probes (finding a fact placed at a known depth) are sensitive early indicators.
A model can pass short benchmarks and fail these probes at 32k context after KV INT4 quantization.
Gate 4 (Section 18) therefore includes the maximum advertised context, not a convenient shorter one.
If KV quantization is used, state its bits alongside weight bits in every serving claim.

## 35. Distillation Data Governance

Sequence distillation makes teacher outputs into training data.
That data inherits the teacher's errors, biases, and style; filtering is part of the algorithm, not an afterthought.
Deduplicate synthetic data against evaluation sets to avoid contamination inflating student scores.
Record teacher identity and generation settings (temperature, prompts) as training provenance.
Where teacher terms restrict output use, that restriction is a hard constraint on the pipeline, not a legal footnote.

## 36. Compression and Safety Behaviour

Compression changes logits, and small logit changes can flip borderline refusals or safety classifications.
Safety slices belong in Gate 4 for exactly this reason.
Evaluate the compressed model, not the baseline, on the safety suite used for release.
A compressed checkpoint is a new model for safety sign-off (Section 18).

## 37. Choosing Bit-Width From the Constraint, Not Habit

Start from the binding constraint: device memory, bandwidth, latency, or cost.
If the constraint is memory at short context, weight bits dominate (Section 29.2).
If the constraint is long-context serving, KV bits and batch dominate.
Choose the least aggressive compression that satisfies the constraint and passes the gates.
Extra compression beyond the constraint buys risk without product value.

## 38. Recovery Fine-Tuning After PTQ

When PTQ misses the gate by a small margin, a short recovery fine-tune on the quantized weights (or adapters on top) often closes much of the gap.
Recovery data should match the deployment distribution, like calibration data (Section 24).
Re-run all gates after recovery; recovery can fix accuracy while shifting calibration.
Budget recovery in the original plan whenever INT4 is the target (Section 17).

## 39. Combinatorial Computed Tables

### 39.1 KV-cache GB by batch and context (bf16, 32 layers, 8 KV heads, 128 head dim)

| Batch | 4k ctx | 8k ctx | 32k ctx | 128k ctx |
|---:|---:|---:|---:|---:|
| 1 | 0.54 | 1.07 | 4.29 | 17.18 |
| 4 | 2.15 | 4.29 | 17.18 | 68.72 |
| 8 | 4.29 | 8.59 | 34.36 | 137.44 |
| 16 | 8.59 | 17.18 | 68.72 | 274.88 |
| 32 | 17.18 | 34.36 | 137.44 | 549.76 |
| 64 | 34.36 | 68.72 | 274.88 | 1099.51 |

Batch multiplies KV exactly; concurrency claims that ignore context length are incomplete.

### 39.2 INT4 total weight GB including fp16 group scales

| Model | Payload | g=32 scales | g=128 scales | Total (g=128) |
|---:|---:|---:|---:|---:|
| 1B | 0.50 | 0.06 | 0.02 | 0.52 |
| 3B | 1.50 | 0.19 | 0.05 | 1.55 |
| 7B | 3.50 | 0.44 | 0.11 | 3.61 |
| 13B | 6.50 | 0.81 | 0.20 | 6.70 |
| 34B | 17.00 | 2.12 | 0.53 | 17.53 |
| 70B | 35.00 | 4.38 | 1.09 | 36.09 |
| 175B | 87.50 | 10.94 | 2.73 | 90.23 |

### 39.3 Temperature softening for teacher logits $(3,1,0,-1)$

| $T$ | $p_1$ | $p_2$ | $p_3$ | $p_4$ |
|---:|---:|---:|---:|---:|
| 0.5 | 0.979 | 0.018 | 0.002 | 0.000 |
| 1 | 0.831 | 0.112 | 0.041 | 0.015 |
| 2 | 0.579 | 0.213 | 0.129 | 0.078 |
| 4 | 0.409 | 0.248 | 0.193 | 0.150 |
| 8 | 0.325 | 0.253 | 0.224 | 0.197 |

As $T$ rises the tail classes gain mass; that tail mass is the dark knowledge the student matches.

## 40. Layer-Sensitivity Worksheet (template with example entries)

| Layer group | Probe | Example loss delta | Decision |
|---|---|---|
| Embeddings | quantize-alone probe | 0.00 pts (illustrative) | Embeddings policy |
| Attention QKV, early | quantize-alone probe | 0.04 pts (illustrative) | Attention QKV policy |
| Attention out, mid | quantize-alone probe | 0.03 pts (illustrative) | Attention out policy |
| MLP up, mid | quantize-alone probe | 0.02 pts (illustrative) | MLP up policy |
| MLP down, late | quantize-alone probe | 0.09 pts (illustrative) | MLP down policy |
| Final norm + head | quantize-alone probe | 0.00 pts (illustrative) | Final norm + head policy |

Entries marked illustrative are a worksheet format, not measurements of a named model; replace them with your probe results.

## 41. Ten Additional Practice Checks (with answers)

1. **Symmetric INT8 scale for range 1.27?** $s=1.27/127=0.01$ exactly by construction.
2. **Bits for 70B in 40 GB (payload only)?** $40/70\times8\approx4.6$ bits/weight, so INT4 plus scales fits; INT8 does not.
3. **KV per token for 80 layers, 8 KV heads, 128 dim, bf16?** $2\times80\times8\times128\times2=327{,}680$ bytes.
4. **Why can INT4 be slower than INT8?** If the kernel dequantizes inefficiently or groups are too small, overhead eats the bandwidth win.
5. **KL direction in distillation?** Forward KL from teacher to student, $KL(p_t\|p_s)$, matching teacher mass.
6. **Effect of doubling group size?** Scale storage halves; range tracking coarsens, usually raising MSE.
7. **What breaks first at W4A16 usually?** Layers with outlier activations or sensitive final layers; measure, do not assume.
8. **Why archive the fp16 checkpoint?** Rollback and re-quantization with better methods require the source.
9. **Temperature $T\to\infty$ limit?** Targets approach uniform; gradient signal becomes logit matching.
10. **Gate that catches calibration drift?** Gate 3 (calibration), Section 18.

## 42. Summary Decision Table

| Situation | First choice | Escalation |
|---|---|---|
| Fits in memory, need low risk | INT8 PTQ | INT4 grouped PTQ |
| Memory-bound on device | INT4 grouped PTQ (GPTQ/AWQ-class) | Recovery fine-tune, then QAT |
| Latency-bound, long context | KV quantization + weight INT8 | Mixed precision by sensitivity |
| Need smaller architecture | Distillation | Distill then quantize with recovery |
| Quality gate fails by wide margin | Do not ship compressed | QAT or larger student |

## 43. Worked Example: Quantizing a Small Tensor End to End

Take weights $w=(-0.62,-0.21,0.05,0.34,0.91)$ and symmetric INT8 quantization.
Step 1: range $a=\max|w|=0.91$.
Step 2: scale $s=0.91/127\approx0.007165$.
Step 3: divide and round: $q=(-87,-29,7,47,127)$.
Step 4: dequantize: $\hat w=s q\approx(-0.623,-0.208,0.050,0.337,0.910)$.
Step 5: errors are about $(-0.003,0.002,0.000,-0.003,0.000)$, each within $s/2\approx0.0036$.
Step 6: MSE is about $4.4\times10^{-6}$, consistent with the $s^2/12$ ideal (Section 29.3).
At INT4 the same tensor uses $s=0.91/7=0.13$ and $q=(-5,-2,0,3,7)$.
Dequantized values $(-0.65,-0.26,0.00,0.39,0.91)$ show errors up to $0.05$, near the $s/2=0.065$ bound.
The example is small enough to check by hand, which is why it belongs in a unit test for any quantization code.

## 44. Worked Example: Reading a Distillation Curve

A student trained with $\alpha=0.9$, $T=2$ shows KL falling while hard-label accuracy stalls.
Interpretation 1: the student matches teacher probabilities on training data but cannot convert them into decisions; capacity may be the limit (Section 28).
Interpretation 2: the teacher's soft labels are noisy on this slice; check a human-labelled validation slice (Section 16).
Interpretation 3: temperature is too high, so the KL term rewards matching a near-uniform distribution (Section 39.3).
The discriminating experiment is a $T$ sweep on validation, not more epochs at the same setting.
Record which interpretation the sweep supports; that note is part of the training log.

## 45. Error Propagation Across Layers

Quantizing one layer perturbs its output; the next layer amplifies or damps that perturbation depending on its Jacobian.
A first-order model multiplies per-layer relative errors by layer gains, so early errors can compound.
This is why end-to-end loss after quantizing all layers is usually worse than the sum of single-layer probes, and why joint methods (GPTQ-style compensation, recovery tuning) exist.
Practical rule: single-layer probes rank layers (Section 33); only the full-model gate decides release (Section 18).

## 46. Compression Reporting Standard

Every compressed-model result should be reportable in one block:
- Base model and checkpoint identifier.
- Method (PTQ/QAT/distillation) and implementation version.
- Bits for weights, activations, and KV cache separately.
- Granularity and group size; zero-point scheme.
- Calibration or recovery data provenance.
- Evaluation gates with tolerances and outcomes.
- Measured memory and latency on named hardware and kernel.
- Known failing slices, if any, with the waiver decision.
Missing any line makes the result hard to reproduce or compare (Section 18).

## 47. Formula Appendix (expanded)

- **Asymmetric scale:** $s=(w_{max}-w_{min})/(2^b-1)$
- **Zero-point:** $z=\mathrm{round}(q_{min}-w_{min}/s)$
- **Dequantization:** $\hat w=s(q-z)$
- **Symmetric scale:** $s=\max|w|/(2^{b-1}-1)$
- **Rounding bound:** $|w-\hat w|\le s/2$ inside range
- **Ideal rounding MSE:** $s^2/12$ for uniform quantization noise
- **SQNR:** $10\log_{10}(\mathrm{Var}(w)/\mathrm{MSE})$ dB
- **Softmax temperature:** $p_i= e^{z_i/T}/\sum_j e^{z_j/T}$
- **Distillation loss:** $\alpha T^2 KL(p_T^t\|p_T^s)+(1-\alpha)CE(y,p_s)$
- **KL divergence:** $\sum_i p_i\log(p_i/q_i)$
- **KV bytes per token:** $2 L h_{kv} d_h b$
- **Effective bits:** $(\text{payload bits}+\text{scale bits}+\text{zero-point bits})/\text{weight}$

## 48. Extended Glossary

- **Calibration:** Setting quantization parameters from representative data without full training.
- **Clipping range:** The real-valued interval mapped onto the integer grid.
- **Codebook:** The set of representable values in non-uniform quantization.
- **Dark knowledge:** Information in teacher probabilities over wrong classes.
- **Dequantization:** Mapping integers back to approximate reals for compute.
- **Fake quantization:** Simulating quantization in the forward pass during QAT.
- **Granularity:** The tensor unit sharing one scale: tensor, channel, or group.
- **Hessian compensation:** GPTQ-style update of remaining weights using second-order statistics.
- **Mixed precision:** Different layers or tensors at different bit-widths.
- **NF4:** NormalFloat 4-bit quantile codebook (Dettmers et al., 2023).
- **Outlier channel:** Activation channel with extreme magnitude harming quantization.
- **Recovery tuning:** Short fine-tune after PTQ to close the quality gap.
- **STE:** Straight-through estimator for gradients through rounding.
- **Student:** The smaller model trained to imitate a teacher.
- **Teacher:** The larger/reference model providing targets.
- **Weight-only quantization:** Only weights quantized; activations stay higher precision.

## 49. Audit Worksheet: Configuration Record

| Field | Your value | Checked |
|---|---|---|
| Base checkpoint |  | [ ] |
| Method and version |  | [ ] |
| Weight bits |  | [ ] |
| Activation bits |  | [ ] |
| KV bits |  | [ ] |
| Granularity |  | [ ] |
| Group size |  | [ ] |
| Zero-point scheme |  | [ ] |
| Calibration source |  | [ ] |
| Calibration size |  | [ ] |
| Recovery data |  | [ ] |
| Gate tolerances |  | [ ] |
| Hardware |  | [ ] |
| Kernel |  | [ ] |
| Latency (measured) |  | [ ] |
| Memory (measured) |  | [ ] |
| Failing slices |  | [ ] |
| Rollback plan |  | [ ] |

## 50. Chapter-15 Integration

- Section 05 supplies the probability language (KL, temperature) that distillation optimizes.
- Section 07 supplies LoRA/QLoRA fine-tuning used for recovery and adaptation of compressed models.
- Section 08 supplies the scaling and reporting discipline: state active versus stored quantities honestly.
- Section 09 supplies KV-cache and bandwidth math that weight compression alone does not settle.
- Section 10 supplies the MoE total-versus-active distinction that compression accounting mirrors.
- Section 12 (next) adds retrieval, whose context length multiplies the KV term priced here.
- Section 13 (final) turns these per-model numbers into serving cost and latency decisions.
- Compression is therefore not an isolated trick; it is one lever in the serving system priced by Sections 12-13.

## 51. Ten More Worked Derivations


**Derivation 1: Effective bits with scales.**
- INT4 payload is 4 bits per weight.
- Group size 128 with fp16 scale adds $16/128=0.125$ bits.
- Asymmetric zero-point per group adds another $0.125$ bits if stored at fp16-equivalent cost in this accounting.
- Effective rate is therefore $4.25$ bits per weight, not $4.00$.
- For 70B weights that is $70\times4.25/8=37.2$ GB.
- Reporting 35.0 GB understates memory by 2.2 GB.

**Derivation 2: Bandwidth saving.**
- A 70B model in bf16 reads 140 GB of weights per token in the naive single-sequence limit.
- INT4 reads about 37 GB including scales (previous derivation).
- The ideal bandwidth saving is about $3.8\times$, before kernel efficiency.
- Measured speedup below that implicates the kernel or dequantization path, not the arithmetic (Section 13).

**Derivation 3: Calibration range drift.**
- Calibration max-abs for a channel is 6.0; deployed traffic reaches 9.0.
- Values above the clipped range collapse to the endpoint.
- The symptom is a slice-specific quality drop with unchanged average benchmarks.
- Fix: recalibrate with current traffic and add range monitoring (Section 24).

**Derivation 4: Student-teacher KL budget.**
- If teacher-student KL stays high while CE falls, the student fits labels but not teacher structure.
- Check capacity (Section 28) before adding epochs.
- A wider student or intermediate hints address a capacity limit; more of the same data does not.

**Derivation 5: Gate arithmetic.**
- Baseline 84.0, tolerance $-0.5$, INT4 result 83.7: passes Gate 1.
- Perplexity shift $+0.9$ against a $+0.3$ tolerance: fails Gate 2.
- Decision: do not release, even though the headline accuracy passed (Section 18).

## 52. What To Log During PTQ

- Per-layer output MSE before and after quantization.
- Per-channel activation max-abs and kurtosis.
- Clipping ranges chosen per granularity unit.
- Calibration batch identifiers for reproducibility.
- Validation loss after each stage (weights, activations, KV).
- Final gate table with tolerances.
These logs turn a failed compression from a mystery into a localized layer or slice.

## 53. Compression Failure Taxonomy

| Failure | Signature | First fix |
|---|---|---|
| Range failure | Slice-specific collapse, outliers clipped | Recalibrate, percentile clipping |
| Granularity failure | One channel group dominates error | Finer groups, per-channel scales |
| Sensitivity failure | One layer's probe delta dwarfs others | Mixed precision for that layer |
| Calibration failure | Compressed model degrades on new domain only | Stratified recalibration |
| Kernel failure | Bytes fall but latency does not | Change kernel or group size |
| Distillation failure | Student matches style, not answers | Held-out human evaluation, capacity check |
| KV failure | Short context fine, long context retrieval fails | Raise KV bits, long-context gate |

## 54. Final Reading Guide for Section 11

Read Sections 11-14 for the error budget and where methods place error.
Read Sections 15-17 for distillation mechanics and pipeline ordering.
Read Sections 18 and 46 before any release decision; they are the gate and the reporting standard.
Use Sections 19, 29, and 39 as worksheets: replace the illustrative geometry with your model's.
Keep the configuration record (Section 49) with the checkpoint it describes.

## 55. Large Computed Reference Tables

### 55.1 Scale and step for symmetric quantization, range $[-a,a]$

| Range $a$ | Bits | Scale $s$ | Max rounding error |
|---:|---:|---:|---:|
| 0.25 | 4 | 0.035714 | 0.017857 |
| 0.25 | 8 | 0.001969 | 0.000984 |
| 0.5 | 4 | 0.071429 | 0.035714 |
| 0.5 | 8 | 0.003937 | 0.001969 |
| 1.0 | 4 | 0.142857 | 0.071429 |
| 1.0 | 8 | 0.007874 | 0.003937 |
| 2.0 | 4 | 0.285714 | 0.142857 |
| 2.0 | 8 | 0.015748 | 0.007874 |

### 55.2 KV GB for a 80-layer GQA model (8 KV heads, 128 dim, bf16) by batch and context

| Batch | 8k | 32k | 128k |
|---:|---:|---:|---:|
| 1 | 2.7 | 10.7 | 42.9 |
| 2 | 5.4 | 21.5 | 85.9 |
| 4 | 10.7 | 42.9 | 171.8 |
| 8 | 21.5 | 85.9 | 343.6 |
| 16 | 42.9 | 171.8 | 687.2 |
| 32 | 85.9 | 343.6 | 1374.4 |

### 55.3 Teacher softening, logits $(5,3,1,0)$

| $T$ | $p_1$ | $p_2$ | $p_3$ | $p_4$ |
|---:|---:|---:|---:|---:|
| 1 | 0.862 | 0.117 | 0.016 | 0.006 |
| 2 | 0.631 | 0.232 | 0.085 | 0.052 |
| 3 | 0.509 | 0.261 | 0.134 | 0.096 |
| 5 | 0.402 | 0.269 | 0.181 | 0.148 |
| 10 | 0.323 | 0.264 | 0.217 | 0.196 |

## 56. Step-by-Step Problem Set C


**Problem C1: Choose granularity.**
- Channel ranges vary from 0.05 to 3.0 across a layer.
- A single per-tensor scale fits the largest channel.
- Small channels would use only a few integer levels.
- Per-channel scales restore resolution per channel.
- Grouped scales go further if ranges vary inside channels.
- Decision: start per-channel; measure before refining.

**Problem C2: Diagnose a slow INT4 model.**
- Weight bytes fell by $3.8\times$ (Section 51) but latency fell only $1.2\times$.
- The arithmetic predicts a bandwidth win, so suspect implementation.
- Check kernel support for the chosen group size.
- Check dequantization overhead in the profile.
- Change one variable at a time and re-measure.

**Problem C3: Plan a distillation.**
- Teacher 70B, target device holds an 8B student at INT4.
- Distill 70B to 8B in bf16 with temperature sweep (Section 44).
- Gate the 8B student against the teacher.
- Quantize the student with grouped INT4 and recovery tuning.
- Gate again, including safety slices (Section 36).
- Ship only if both gate rounds pass.

**Problem C4: Price KV versus weights.**
- A 7B INT4 model carries about 3.7 GB of weights.
- At batch 32 and 32k context, the bf16 KV cache (Section 39.1 geometry scaled) dwarfs that.
- Serving cost is therefore set by KV, not weights.
- KV INT8 or a smaller batch changes cost more than further weight compression.

**Problem C5: Read an aux result.**
- Weight MSE improved after switching PTQ methods.
- Downstream accuracy worsened on activation-heavy slices.
- Activation weighting explains the reversal (Section 13).
- Method selection must use output/task error, not weight MSE.

**Problem C6: Set a tolerance.**
- Baseline accuracy 80.0 with run-to-run noise $\pm0.3$.
- A tolerance tighter than noise cannot be evaluated.
- Set tolerance at least as wide as measured baseline noise.
- Otherwise gates reject (or accept) noise, not compression effects.

## 57. Section 11 Takeaways

- Quantization error is priced where activations spend it, not where weights sit.
- Distillation transfers distributions; temperature controls how much structure is visible.
- Pipeline order (distill/quantize) changes the total error budget.
- A compressed model is a new model: gate it, measure it, and record its configuration.
- Bits are a means; the binding constraint (memory, bandwidth, latency, cost) chooses them.

## 58. Final Computed Sweep: Effective Memory Including Embeddings Kept in bf16

Assume embeddings and final head are 5% of parameters kept in bf16, the rest INT4 grouped (g=128). Effective GB:

| Model | INT4 rest | bf16 embed/head | Scales | Total | vs bf16 |
|---:|---:|---:|---:|---:|
| 1B | 0.47 | 0.10 | 0.01 | 0.59 | 3.39x smaller |
| 3B | 1.42 | 0.30 | 0.04 | 1.77 | 3.39x smaller |
| 7B | 3.32 | 0.70 | 0.10 | 4.13 | 3.39x smaller |
| 13B | 6.17 | 1.30 | 0.19 | 7.67 | 3.39x smaller |
| 34B | 16.15 | 3.40 | 0.50 | 20.05 | 3.39x smaller |
| 70B | 33.25 | 7.00 | 1.04 | 41.29 | 3.39x smaller |
| 175B | 83.12 | 17.50 | 2.60 | 103.22 | 3.39x smaller |

Keeping a small fraction in bf16 is cheap insurance for sensitive tensors (Section 33) and barely moves total bytes.

## 59. Closing Notes for Section 11

- Compression decisions should be reversible: keep the source checkpoint.
- Every number in this section can be recomputed from the stated formulas; do the same for your model.
- Where a table says illustrative, replace it with your measurement before acting on it.
- The next section adds retrieval; its context costs are priced with the KV formulas used here.

## 60. One-Page Compression Review

- Constraint identified (memory, bandwidth, latency, or cost).
- Baseline frozen and archived.
- Method chosen from the constraint, not habit.
- Granularity chosen from measured channel ranges.
- Calibration provenance recorded.
- Layer sensitivity ranked by probe.
- Distillation, if any, has teacher, data, temperature, and alpha recorded.
- Pipeline order and interaction term considered.
- Five gates passed on the compressed checkpoint itself.
- Safety and long-context slices included.
- Measured latency and memory on target hardware.
- Configuration record completed (Section 49).
- Rollback path tested.
- Failing slices, if waived, named with an owner.
- Result reported with bits for weights, activations, and KV separately.
- Effective bits include scales and zero-points.
- Comparison models measured with the same kernel and hardware.
- Student/teacher capacity gap considered before more distillation epochs.
- Recovery tuning budgeted when INT4 is the target.
- Release notes state the compression configuration in full.

## 61. Final Note

- Compression is an experiment with a configuration, a dataset, and a gate.
- Treat it with the same rigour as training.
- The checkpoint you ship is the one you measured.

## Practice Exercises

1. Quantize and dequantize scalar values with an affine quantizer.
2. Compute symmetric INT4 scale and error.
3. Compare per-tensor and per-channel quantization error.
4. Compute memory reduction from bf16 to 4-bit weights.
5. Sweep clipping ranges and choose the lowest MSE.
6. Compute distillation probabilities at temperature.
7. Compute KL distillation loss for teacher and student distributions.
8. Combine hard-label CE and distillation loss.
9. Estimate QLoRA optimizer-state memory.
10. Write a compression deployment checklist.

## Why This Matters for AI

Compression determines who can run a model, how much serving costs, and which devices can host useful AI locally. The math matters because bad compression can keep a model small but silently damage probabilities, calibration, long-context behavior, or safety behavior.

## Bridge to RAG Math and Retrieval

Compression makes a model cheaper. Retrieval can make a model more informed without changing all its weights. The next section studies embedding retrieval, similarity search, ranking, context packing, and how retrieval changes the conditional distribution used by an LLM.

## References

- Geoffrey Hinton, Oriol Vinyals, and Jeff Dean, "Distilling the Knowledge in a Neural Network", 2015: https://arxiv.org/abs/1503.02531
- Benoit Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", 2017: https://arxiv.org/abs/1712.05877
- Tim Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", 2022: https://arxiv.org/abs/2208.07339
- Elias Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", 2022: https://arxiv.org/abs/2210.17323
- Guangxuan Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models", 2022: https://arxiv.org/abs/2211.10438
- Ji Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration", 2023: https://arxiv.org/abs/2306.00978
- Tim Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023: https://arxiv.org/abs/2305.14314
