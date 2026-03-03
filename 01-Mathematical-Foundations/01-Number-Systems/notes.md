[← Back to Mathematical Foundations](../README.md) | [Next: Sets and Logic →](../02-Sets-and-Logic/notes.md)

---

# Number Systems

> _"Every weight, every gradient, every activation in a neural network is ultimately a finite string of bits. The choice of how to interpret those bits — the number system — determines what a model can represent, how fast it trains, and whether it trains at all."_

## Overview

A number system is a formal framework for representing, storing, and computing with numerical quantities using a finite set of symbols. Every number stored in a computer is an approximation: real numbers are infinite; bits are finite. The choice of number system determines what values can be represented, how precisely, at what memory cost, and how fast arithmetic can be performed.

For AI and LLMs specifically, the number system used for weights, activations, and gradients is one of the most consequential engineering decisions. A seemingly small change — from FP32 to BF16 — can halve memory, double throughput, and change training dynamics in ways that took the research community years to fully understand.

This chapter covers number systems from first principles through to the frontier formats used in 2026-era LLM training and inference. Every concept is grounded in how it affects real neural network computation.

## Prerequisites

- Basic arithmetic operations and algebra
- Binary number basics (what bits are)
- Familiarity with Python/NumPy
- Awareness of neural network forward/backward passes (helpful but not required)

## Companion Notebooks

| Notebook                           | Description                                                                                  |
| ---------------------------------- | -------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Interactive code: IEEE 754 bit manipulation, format comparisons, quantization visualisations |
| [exercises.ipynb](exercises.ipynb) | Practice problems: bit decoding, precision analysis, quantization grid computation           |

## Learning Objectives

After completing this section, you will:

- Understand why number system choice is a first-class engineering decision in LLM development
- Master IEEE 754 floating-point representation (FP32, FP16, BF16) and its special values
- Know every AI-relevant format: FP64, FP32, TF32, BF16, FP16, FP8 (E4M3/E5M2), INT8, INT4, NF4, ternary
- Derive floating-point arithmetic algorithms (addition, multiplication, FMA) and understand rounding error
- Analyse numerical stability in softmax, layer normalisation, gradient accumulation, and Adam
- Compute quantization grids, signal-to-noise ratios, and error propagation through layers
- Apply the correct number system for each operation in training and inference
- Understand hardware implications: tensor core throughput, memory bandwidth, energy cost per format

---

## Table of Contents

- [Number Systems](#number-systems)
  - [Overview](#overview)
  - [Prerequisites](#prerequisites)
  - [Companion Notebooks](#companion-notebooks)
  - [Learning Objectives](#learning-objectives)
  - [Table of Contents](#table-of-contents)
  - [1. Intuition](#1-intuition)
    - [1.1 What Are Number Systems?](#11-what-are-number-systems)
    - [1.2 Why Number Systems Matter for AI](#12-why-number-systems-matter-for-ai)
    - [1.3 The Precision-Efficiency Frontier](#13-the-precision-efficiency-frontier)
    - [1.4 Levels of Number System Usage in LLMs](#14-levels-of-number-system-usage-in-llms)
    - [1.5 Historical Timeline](#15-historical-timeline)
    - [1.6 The Role of Hardware](#16-the-role-of-hardware)
  - [2. Positional Number Systems](#2-positional-number-systems)
    - [2.1 The Positional Representation Principle](#21-the-positional-representation-principle)
    - [2.2 Binary (Base 2)](#22-binary-base-2)
    - [2.3 Hexadecimal (Base 16)](#23-hexadecimal-base-16)
    - [2.4 Two's Complement for Signed Integers](#24-twos-complement-for-signed-integers)
    - [2.5 Fixed-Point Representation](#25-fixed-point-representation)
  - [3. IEEE 754 Floating-Point Standard](#3-ieee-754-floating-point-standard)
    - [3.1 Floating-Point Representation Principle](#31-floating-point-representation-principle)
    - [3.2 IEEE 754 FP32 (Single Precision)](#32-ieee-754-fp32-single-precision)
    - [3.3 Special Values in IEEE 754](#33-special-values-in-ieee-754)
    - [3.4 FP32 Arithmetic Properties](#34-fp32-arithmetic-properties)
    - [3.5 Rounding Modes](#35-rounding-modes)
    - [3.6 Catastrophic Cancellation](#36-catastrophic-cancellation)
    - [3.7 Kahan Summation Algorithm](#37-kahan-summation-algorithm)
  - [4. Floating-Point Formats for AI](#4-floating-point-formats-for-ai)
    - [4.1 FP64 (Double Precision)](#41-fp64-double-precision)
    - [4.2 FP32 (Single Precision) — Detailed](#42-fp32-single-precision--detailed)
    - [4.3 TF32 (TensorFloat-32)](#43-tf32-tensorfloat-32)
    - [4.4 FP16 (Half Precision)](#44-fp16-half-precision)
    - [4.5 BF16 (Brain Float 16)](#45-bf16-brain-float-16)
    - [4.6 FP16 vs BF16 Head-to-Head](#46-fp16-vs-bf16-head-to-head)
    - [4.7 FP8 Formats (E4M3 and E5M2)](#47-fp8-formats-e4m3-and-e5m2)
    - [4.8 FP8 Scaling Strategies](#48-fp8-scaling-strategies)
  - [5. Integer Formats for AI](#5-integer-formats-for-ai)
    - [5.1 INT32 — Full Precision Integer](#51-int32--full-precision-integer)
    - [5.2 INT16](#52-int16)
    - [5.3 INT8](#53-int8)
    - [5.4 UINT8 — Unsigned Integer 8-bit](#54-uint8--unsigned-integer-8-bit)
    - [5.5 INT4](#55-int4)
    - [5.6 INT2 and INT1 (Binary)](#56-int2-and-int1-binary)
    - [5.7 Packed Integer Formats](#57-packed-integer-formats)
  - [6. Non-Uniform and Specialised Formats](#6-non-uniform-and-specialised-formats)
    - [6.1 Normal Float 4-bit (NF4)](#61-normal-float-4-bit-nf4)
    - [6.2 Log Number System (LNS)](#62-log-number-system-lns)
    - [6.3 Posit Number System (Unum Type III)](#63-posit-number-system-unum-type-iii)
    - [6.4 Microscaling Formats (MX — OCP Standard 2023)](#64-microscaling-formats-mx--ocp-standard-2023)
    - [6.5 Ternary Weights {−1, 0, +1}](#65-ternary-weights-1-0-1)
  - [7. Floating-Point Arithmetic Deep Dive](#7-floating-point-arithmetic-deep-dive)
    - [7.1 Floating-Point Addition](#71-floating-point-addition)
    - [7.2 Floating-Point Multiplication](#72-floating-point-multiplication)
    - [7.3 Floating-Point Division and Square Root](#73-floating-point-division-and-square-root)
    - [7.4 Fused Multiply-Add (FMA)](#74-fused-multiply-add-fma)
    - [7.5 Dot Product Accumulation](#75-dot-product-accumulation)
    - [7.6 Mixed-Precision Matmul](#76-mixed-precision-matmul)
  - [8. Numerical Precision in Neural Networks](#8-numerical-precision-in-neural-networks)
    - [8.1 Forward Pass Precision Requirements](#81-forward-pass-precision-requirements)
    - [8.2 Backward Pass Precision Requirements](#82-backward-pass-precision-requirements)
    - [8.3 Mixed Precision Training — Complete Picture](#83-mixed-precision-training--complete-picture)
    - [8.4 Numerical Stability of Softmax](#84-numerical-stability-of-softmax)
    - [8.5 Log-Sum-Exp Trick](#85-log-sum-exp-trick)
    - [8.6 Numerical Stability of Layer Normalisation](#86-numerical-stability-of-layer-normalisation)
    - [8.7 Gradient Vanishing and Exploding — Numerical View](#87-gradient-vanishing-and-exploding--numerical-view)
  - [9. Quantization Mathematics](#9-quantization-mathematics)
    - [9.1 Uniform Quantization Formulas — Derivation and Intuition](#91-uniform-quantization-formulas--derivation-and-intuition)
    - [9.2 Signal-to-Quantization-Noise Ratio (SQNR)](#92-signal-to-quantization-noise-ratio-sqnr)
    - [9.3 Block Floating-Point and Group Quantization](#93-block-floating-point-and-group-quantization)
    - [9.4 Optimal Quantization Levels — Lloyd-Max Algorithm](#94-optimal-quantization-levels--lloyd-max-algorithm)
    - [9.5 Hadamard Transform for Quantization — QuIP and QuaRot](#95-hadamard-transform-for-quantization--quip-and-quarot)
    - [9.6 Quantization Error Propagation Through Layers](#96-quantization-error-propagation-through-layers)
  - [10. Number Systems for Specific AI Operations](#10-number-systems-for-specific-ai-operations)
    - [10.1 Embedding Tables](#101-embedding-tables)
    - [10.2 Attention Score Computation](#102-attention-score-computation)
    - [10.3 KV Cache Precision](#103-kv-cache-precision)
    - [10.4 Feed-Forward Network (FFN) Precision](#104-feed-forward-network-ffn-precision)
    - [10.5 Optimizer State Precision](#105-optimizer-state-precision)
    - [10.6 Gradient Communication Precision](#106-gradient-communication-precision)
  - [11. Hardware Implementation of Number Systems](#11-hardware-implementation-of-number-systems)
    - [11.1 Integer ALUs — Simple and Fast](#111-integer-alus--simple-and-fast)
    - [11.2 Floating-Point Units — Complex and Power-Hungry](#112-floating-point-units--complex-and-power-hungry)
    - [11.3 Tensor Cores — Systolic Arrays for AI](#113-tensor-cores--systolic-arrays-for-ai)
    - [11.4 Memory Bandwidth — The True Bottleneck](#114-memory-bandwidth--the-true-bottleneck)
    - [11.5 Energy Cost per Operation](#115-energy-cost-per-operation)
    - [11.6 Format Conversion Hardware](#116-format-conversion-hardware)
  - [12. Precision and the Training Stability Connection](#12-precision-and-the-training-stability-connection)
    - [12.1 Loss Landscape and Precision](#121-loss-landscape-and-precision)
    - [12.2 Precision Cliffs — When Training Suddenly Diverges](#122-precision-cliffs--when-training-suddenly-diverges)
    - [12.3 Stochastic Rounding — A Precision Amplifier](#123-stochastic-rounding--a-precision-amplifier)
    - [12.4 Adam Optimizer Numerical Error Analysis](#124-adam-optimizer-numerical-error-analysis)
    - [12.5 Attention Logit Growth](#125-attention-logit-growth)
  - [13. Practical Guide — Choosing Number Formats](#13-practical-guide--choosing-number-formats)
    - [13.1 Decision Framework for Training](#131-decision-framework-for-training)
    - [13.2 Decision Framework for Inference](#132-decision-framework-for-inference)
    - [13.3 Per-Layer Sensitivity Table](#133-per-layer-sensitivity-table)
    - [13.4 Quantization Quality vs Bit Width](#134-quantization-quality-vs-bit-width)
  - [14. Common Mistakes and Misconceptions](#14-common-mistakes-and-misconceptions)
  - [15. Exercises](#15-exercises)
    - [Exercise 1: IEEE 754 Encoding](#exercise-1-ieee-754-encoding)
    - [Exercise 2: Quantization Calculation](#exercise-2-quantization-calculation)
    - [Exercise 3: BF16 Precision Limits](#exercise-3-bf16-precision-limits)
    - [Exercise 4: Memory Budget](#exercise-4-memory-budget)
    - [Exercise 5: Softmax Stability](#exercise-5-softmax-stability)
    - [Exercise 6: Error Propagation](#exercise-6-error-propagation)
    - [Exercise 7: Hardware Arithmetic Intensity](#exercise-7-hardware-arithmetic-intensity)
    - [Exercise 8: Stochastic Rounding Simulation](#exercise-8-stochastic-rounding-simulation)
  - [16. Why This Matters for AI/ML](#16-why-this-matters-for-aiml)
  - [17. Conceptual Bridge](#17-conceptual-bridge)

---

## 1. Intuition

### 1.1 What Are Number Systems?

A number system is a formal framework for representing, storing, and computing with numerical quantities using a finite set of symbols. At its core, a number system answers four questions:

1. **What values can be represented?** — The range of the system
2. **How precisely?** — The resolution or granularity between representable values
3. **At what memory cost?** — The number of bits required per value
4. **How fast can arithmetic be performed?** — Hardware throughput for the format

Every number stored in a computer is an approximation. Real numbers ($\mathbb{R}$) are uncountably infinite — there are infinitely many real numbers between any two real numbers. But a computer register has a fixed number of bits. A 32-bit register can represent at most $2^{32} = 4{,}294{,}967{,}296$ distinct values. A 16-bit register: $2^{16} = 65{,}536$. An 8-bit register: $2^8 = 256$. A 4-bit register: $2^4 = 16$.

The art of number system design is choosing **which** $2^b$ values to represent — how to distribute those discrete points across the real number line to minimise error for the intended application.

```
THE NUMBER SYSTEM DESIGN PROBLEM
═══════════════════════════════════════════════════════════════════════

Real number line (continuous, infinite):
───────────────────────────────────────────────────────────────────►

4-bit representation (only 16 values available):
──●──●──●──●──●──●──●──●──●──●──●──●──●──●──●──●──────────────────►

Where do you place those 16 dots?

  Uniform spacing (INT4):      ●──●──●──●──●──●──●──●──●──●──●──●──●──●──●──●
  Dense near zero (NF4):       ●●●●●●●──●──●──●────●────●────●─────●──────●
  Logarithmic spacing (FP4):   ●●●●●──●──●──●────●────●────●────────●────────●

Each choice optimises for different distributions of real-world data.
```

For neural networks, the data we need to represent — weights, activations, gradients — follows specific statistical distributions. Transformer weights are approximately normally distributed with small variance. Activations have heavier tails. Gradients span many orders of magnitude. The "right" number system is the one that best matches the statistical distribution of the data it represents.

### 1.2 Why Number Systems Matter for AI

The impact of number system choice on LLM development is immediate and quantifiable:

**Memory impact (LLaMA-3 70B model, weights only):**

| Format | Bytes/param | Total Weight Memory | GPU Requirement   |
| ------ | ----------- | ------------------- | ----------------- |
| FP32   | 4           | 280 GB              | 4× A100 80 GB     |
| BF16   | 2           | 140 GB              | 2× A100 80 GB     |
| INT8   | 1           | 70 GB               | 1× A100 80 GB     |
| INT4   | 0.5         | 35 GB               | 1× RTX 4090 24 GB |
| INT2   | 0.25        | 17.5 GB             | 1× RTX 3090 24 GB |

**Training cost impact:**

- Training a 70B model in FP32: ~280 GB just for weights; optimizer states (Adam $m$ and $v$) add another 560 GB; total ~840 GB — requires a full node of 8× A100s or more
- Training the same model in BF16 mixed precision: ~140 GB weights + 560 GB optimizer (FP32) + 140 GB gradients; total ~840 GB still (optimizer dominates), but forward/backward pass is 2× faster due to BF16 matmul throughput
- Inference in INT4: 35 GB; fits on a single consumer GPU; enables local AI on laptops

**What goes wrong with the wrong number system:**

- FP16 training without loss scaling → gradient underflow → weights stop updating → training stalls silently
- BF16 accumulation instead of FP32 → precision loss in gradient sums → training diverges after millions of steps
- Naive INT8 quantization of transformers → activation outliers clip → catastrophic quality degradation
- INT4 quantization of first/last layers → disproportionate quality loss → model produces gibberish

**The history of AI progress is partly a history of number system engineering:**

```
EVOLUTION OF NUMBER FORMATS IN DEEP LEARNING
═══════════════════════════════════════════════════════════════════════

2012 ─── FP32 ──────── AlexNet; all computation in single precision
  │
2017 ─── FP16 ──────── NVIDIA Volta tensor cores; first hardware-accelerated
  │                    reduced precision; required loss scaling
  │
2018 ─── BF16 ──────── Google Brain introduces Brain Float 16; same range
  │                    as FP32; no loss scaling needed; game changer
  │
2020 ─── Mixed ─────── BF16 forward + FP32 master weights becomes standard
  │     Precision      for all large-scale training
  │
2022 ─── FP8 ───────── NVIDIA H100 adds FP8 tensor cores; 2× throughput
  │                    vs BF16; per-tensor scaling required
  │
2023 ─── INT4 ──────── GPTQ, AWQ enable high-quality 4-bit inference;
  │                    70B models on consumer GPUs for the first time
  │
2024 ─── Ternary ───── BitNet b1.58: {-1,0,+1} weights trained from scratch;
  │                    eliminates multiplication entirely
  │
2024 ─── FP8 Train ─── DeepSeek-V3: entire training in FP8; commercial
  │                    frontier model; massive cost reduction
  │
2025-26 ─ MXFP/Sub-4 ─ Microscaling formats; sub-4-bit active research;
                        hardware co-designed with number formats
```

### 1.3 The Precision-Efficiency Frontier

The fundamental trade-off: **more bits → higher precision → better model quality**, but **fewer bits → less memory → faster compute → wider hardware access**.

The engineering challenge is finding the **minimum precision that preserves acceptable quality** for each operation. The key insight is that different operations have vastly different precision requirements:

```
PRECISION REQUIREMENTS BY OPERATION
═══════════════════════════════════════════════════════════════════════

                          More Bits Needed
                               ▲
                               │
                               │    Gradient accumulation (FP32)
                               │    ■ Small errors compound over millions
                               │      of steps; must be high precision
                               │
                               │    Optimizer states (FP32)
                               │    ■ Adam m, v track subtle gradient
                               │      statistics; precision critical
                               │
                               │    Loss computation (FP32)
                               │    ■ Cross-entropy involves log/exp;
                               │      overflow/underflow risk
                               │
                               │    Weight storage — training (BF16)
                               │    ■ Weights change slowly; moderate
                               │      precision sufficient
                               │
                               │    Activation computation (BF16/FP8)
                               │    ■ Errors localised per forward pass;
                               │      don't compound across steps
                               │
                               │    KV cache storage (INT8/INT4)
                               │    ■ Small reconstruction error per token;
                               │      acceptable quality trade-off
                               │
                               │    Weight-only inference (INT4/INT2)
                               │    ■ Weights static; error bounded;
                               │      no accumulation across steps
                               │
                               ▼
                          Fewer Bits Needed
```

**Quantitative precision requirements:**

| Operation                        | Minimum Format | Machine Epsilon                                    | Why This Precision                                    |
| -------------------------------- | -------------- | -------------------------------------------------- | ----------------------------------------------------- |
| Gradient accumulation            | FP32           | $\varepsilon = 2^{-23} \approx 1.2 \times 10^{-7}$ | Must resolve updates $\sim 10^{-7}$ over $10^6$ steps |
| Optimizer states (Adam $m$, $v$) | FP32           | $\varepsilon = 2^{-23}$                            | Tracks exponential moving averages; $\beta_2 = 0.999$ |
| Weight master copy               | FP32           | $\varepsilon = 2^{-23}$                            | Single source of truth; must accumulate tiny updates  |
| Forward/backward matmul          | BF16           | $\varepsilon = 2^{-7} \approx 7.8 \times 10^{-3}$  | Errors don't compound across training steps           |
| Inference weights                | INT4           | $\Delta \approx 0.1$ (depends on range)            | Static; bounded error; no accumulation                |
| KV cache                         | INT8/FP8       | $\varepsilon \approx 10^{-2}$                      | Per-token error; minor impact on generation quality   |

### 1.4 Levels of Number System Usage in LLMs

A modern LLM uses **multiple number systems simultaneously** in different parts of the computation:

```
MIXED-PRECISION LLM ARCHITECTURE (2026 Standard)
═══════════════════════════════════════════════════════════════════════

TRAINING:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                   │
│  │ Master Weights   │    │ Adam m (moment)  │                   │
│  │     FP32         │    │     FP32         │                   │
│  │  (4 bytes/param) │    │  (4 bytes/param) │                   │
│  └────────┬─────────┘    └──────────────────┘                   │
│           │ cast                                                │
│           ▼                ┌──────────────────┐                  │
│  ┌──────────────────┐     │ Adam v (variance)│                  │
│  │ Working Weights  │     │     FP32         │                  │
│  │     BF16         │     │  (4 bytes/param) │                  │
│  │  (2 bytes/param) │     └──────────────────┘                  │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │  Forward Pass    │────►│  Activations     │                  │
│  │     BF16         │     │     BF16         │                  │
│  │  matmul in BF16  │     │  (for backward)  │                  │
│  │  accum in FP32   │     └──────────────────┘                  │
│  └────────┬─────────┘                                           │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │  Backward Pass   │────►│  Gradient Accum  │                  │
│  │     BF16         │     │     FP32         │                  │
│  │  gradient compute│     │  (critical!)     │                  │
│  └──────────────────┘     └────────┬─────────┘                  │
│                                    │                            │
│                                    ▼                            │
│                           ┌──────────────────┐                  │
│                           │  Weight Update   │                  │
│                           │     FP32         │                  │
│                           │  θ ← θ - η·m̂/√v̂ │                  │
│                           └──────────────────┘                  │
│                                                                 │
│  Total per parameter: 4 + 4 + 4 + 2 = 14 bytes (FP32 master    │
│  + FP32 Adam m + FP32 Adam v + BF16 working copy)              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

INFERENCE:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │  Weights         │     │  KV Cache        │                  │
│  │  INT4 (GPTQ/AWQ) │     │  INT8 or FP8     │                  │
│  │  0.5 bytes/param │     │  1 byte/scalar   │                  │
│  │  dequant → BF16  │     │                  │                  │
│  └────────┬─────────┘     └──────────────────┘                  │
│           │                                                     │
│           ▼                                                     │
│  ┌──────────────────┐     ┌──────────────────┐                  │
│  │  Matmul          │     │  Softmax         │                  │
│  │  BF16 compute    │     │  FP32 (stable)   │                  │
│  │  FP32 accum      │     │                  │                  │
│  └──────────────────┘     └──────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.5 Historical Timeline

| Year    | Event                                                         | Significance for AI                                                                              |
| ------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1940s   | ENIAC: 10-digit decimal fixed-point                           | First electronic general-purpose computer; decimal arithmetic                                    |
| 1954    | IBM 704: first commercial floating-point hardware (36-bit)    | Floating-point becomes accessible; enables scientific computing                                  |
| 1985    | **IEEE 754 standard** published: defines FP32 and FP64        | Universal standard; all modern hardware and software agrees on FP format                         |
| 2008    | IEEE 754-2008 revision: adds FP16 (binary16)                  | Half-precision officially standardised; enables GPU compute                                      |
| 2012    | AlexNet wins ImageNet using FP32 on GPUs                      | Deep learning revolution begins; all training in FP32                                            |
| 2017    | **NVIDIA Volta GPU**: FP16 tensor cores                       | First hardware acceleration for reduced precision ML; 8× throughput vs FP32                      |
| 2018    | **Google Brain introduces BF16** (Brain Float 16)             | Same exponent range as FP32 with 7-bit mantissa; eliminates loss scaling                         |
| 2018    | Mixed precision training paper (Micikevicius et al.)          | FP16 forward + FP32 master weights formalised as standard practice                               |
| 2019    | INT8 inference widely deployed                                | LLM.int8() predecessor methods; production quantization begins                                   |
| 2020    | **A100 GPU**: native BF16 tensor cores                        | BF16 becomes the default training format for all large models                                    |
| 2022    | **FP8 proposed for training** (Micikevicius et al. 2022)      | Two complementary 8-bit formats: E4M3 (precision) and E5M2 (range)                               |
| 2022    | **NVIDIA H100**: first hardware with FP8 tensor cores         | 4× throughput vs BF16; enables FP8 training at scale                                             |
| 2023    | GPTQ, AWQ: INT4 post-training quantization for LLMs           | 70B models on consumer GPUs; democratises large model access                                     |
| 2023    | QLoRA introduces NF4 (Normal Float 4)                         | Quantile-based 4-bit format optimal for normally-distributed weights                             |
| 2023    | OCP Microscaling (MX) standard published                      | Industry-wide block floating-point standard (AMD, ARM, Intel, Meta, Microsoft, NVIDIA, Qualcomm) |
| 2024    | **BitNet b1.58**: ternary weights {−1,0,+1} (1.58 bits)       | Eliminates multiply; addition/subtraction only; trained from scratch                             |
| 2024    | **DeepSeek-V3**: FP8 training at scale                        | Commercial frontier model trained entirely in FP8; massive cost reduction                        |
| 2025    | NVIDIA B200 (Blackwell): MXFP8 native support                 | Hardware designed around block floating-point; format and silicon co-evolution                   |
| 2025–26 | FP8 training standard; sub-4-bit quantization active research | Industry converges on FP8 for training; INT4/INT2/ternary for inference                          |

### 1.6 The Role of Hardware

Number systems are not purely mathematical abstractions — they are **hardware capabilities**. A format that has no hardware support executes in software at full precision, gaining no speed advantage.

**GPU tensor core throughput by format (NVIDIA H100 SXM):**

| Format | Throughput   | Relative to FP32 | Hardware Unit                 |
| ------ | ------------ | ---------------- | ----------------------------- |
| FP64   | 67 TFLOPS    | 1×               | CUDA cores (double precision) |
| FP32   | 67 TFLOPS    | 1×               | CUDA cores                    |
| TF32   | 989 TFLOPS   | 15×              | Tensor cores                  |
| BF16   | 989 TFLOPS   | 15×              | Tensor cores                  |
| FP16   | 989 TFLOPS   | 15×              | Tensor cores                  |
| FP8    | 1,979 TFLOPS | 30×              | Tensor cores                  |
| INT8   | 1,979 TOPS   | 30×              | Tensor cores                  |

The throughput gap is enormous: FP8 operations are 30× faster than FP32 on the same hardware. This means a format change from FP32 to FP8 can potentially deliver a 30× speedup for matmul-bound operations — far more impactful than any algorithmic optimisation.

**Hardware design co-evolves with number format needs:**

- NVIDIA added BF16 tensor cores to A100 because the ML community needed wider dynamic range than FP16
- NVIDIA added FP8 tensor cores to H100 specifically because 8-bit training research showed viability
- NVIDIA added MXFP support to B200 because block floating-point emerged as the optimal scaling strategy
- Google designed TPU v2+ with native BF16 from the start — BF16 was literally invented for TPUs

The implication: when choosing a number format, you must check whether your target hardware has native support. Running FP8 arithmetic on A100 (which has no FP8 tensor cores) gains nothing — the computation falls back to BF16 or FP32.

```
HARDWARE-FORMAT SUPPORT MATRIX
═══════════════════════════════════════════════════════════════════════

              │ FP64 │ FP32 │ TF32 │ BF16 │ FP16 │ FP8  │ INT8 │ INT4
══════════════╪══════╪══════╪══════╪══════╪══════╪══════╪══════╪═════
V100 (2017)   │  ✓   │  ✓   │  ✗   │  ✗   │  ✓TC │  ✗   │  ✗   │  ✗
A100 (2020)   │  ✓   │  ✓   │  ✓TC │  ✓TC │  ✓TC │  ✗   │  ✓TC │  ✗
H100 (2022)   │  ✓   │  ✓   │  ✓TC │  ✓TC │  ✓TC │  ✓TC │  ✓TC │  ✗
B200 (2025)   │  ✓   │  ✓   │  ✓TC │  ✓TC │  ✓TC │  ✓TC │  ✓TC │  ✓TC
RTX 4090      │  ✓   │  ✓   │  ✓TC │  ✓TC │  ✓TC │  ✓TC │  ✓TC │  ✓TC

TC = Tensor Core accelerated (high throughput)
✓  = Supported via CUDA cores (standard throughput)
✗  = Not supported at hardware level
```

---

## 2. Positional Number Systems

### 2.1 The Positional Representation Principle

A positional number system represents any number as a weighted sum of powers of a fixed **base** (or **radix**) $b$:

$$x = \sum_{i=-\infty}^{n} d_i \cdot b^i$$

where:

- $d_i \in \{0, 1, \ldots, b-1\}$: the digit at position $i$
- $n$: the position of the most significant digit
- The **radix point** separates the integer part ($i \geq 0$) from the fractional part ($i < 0$)
- Any real number representable exactly may require infinitely many digits

The value of a digit depends on its **position** — this is what "positional" means. The digit 3 in position 2 of a base-10 number represents $3 \times 10^2 = 300$, not simply 3.

**Example in base 10:**

$$4{,}725.38_{10} = 4 \times 10^3 + 7 \times 10^2 + 2 \times 10^1 + 5 \times 10^0 + 3 \times 10^{-1} + 8 \times 10^{-2}$$

$$= 4000 + 700 + 20 + 5 + 0.3 + 0.08 = 4725.38$$

```
POSITIONAL VALUE IN BASE 10
═══════════════════════════════════════════════════════════════════════

Position:    3       2       1       0    .   -1      -2
Power:     10³     10²     10¹     10⁰       10⁻¹    10⁻²
Weight:   1000     100      10       1        0.1     0.01
Digit:      4       7       2       5    .    3       8
Value:   4000     700      20       5        0.3     0.08
                                                          = 4725.38
```

**Why positional systems matter for computing:** every digital computer uses a positional system (base 2) because transistors have two states. Understanding how positional encoding works is prerequisite to understanding how floating-point numbers allocate bits.

### 2.2 Binary (Base 2)

Binary is the foundational number system for all digital computation:

- Base $b = 2$; digits $\in \{0, 1\}$; each digit is one **bit** (binary digit)
- Natural representation for digital hardware: a transistor is either on (1) or off (0)
- Every number stored in a computer — every weight, every activation, every token ID — is ultimately a binary string

**Integer binary representation:**

$$x = \sum_{i=0}^{n} d_i \cdot 2^i$$

**Worked example:**

$$1011_2 = 1 \times 2^3 + 0 \times 2^2 + 1 \times 2^1 + 1 \times 2^0 = 8 + 0 + 2 + 1 = 11_{10}$$

```
BINARY TO DECIMAL CONVERSION  (1011₂ → 11₁₀)
═══════════════════════════════════════════════════════════════════════

Bit position:    3       2       1       0
Power of 2:      2³      2²      2¹      2⁰
Weight:          8       4       2       1
Bit value:       1       0       1       1
Contribution:    8       0       2       1     →  Total: 11₁₀
```

**Binary fraction:**

$$0.101_2 = 1 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3} = 0.5 + 0 + 0.125 = 0.625_{10}$$

**The critical limitation — infinite binary expansions:**

Most decimal fractions have **infinite** binary representations. This is the root cause of floating-point "errors" that every programmer encounters:

$$0.1_{10} = 0.0\overline{0011}_2 = 0.000110011001100110011\ldots_2$$

This is not a bug — it is a fundamental mathematical fact. The number $0.1$ cannot be represented exactly in any finite number of binary digits, just as $1/3 = 0.333\ldots$ cannot be represented exactly in any finite number of decimal digits.

**AI implication:** when a learning rate is set to `lr = 0.001`, the actual value stored in memory is the nearest binary floating-point approximation, not exactly $10^{-3}$. For FP32, this is $0.001000000047497\ldots$ — close enough that it doesn't matter, but the principle underlies all numerical precision analysis.

**Common powers of 2 (essential for parameter counting and memory estimation):**

| Power    | Value              | Common Usage                                    |
| -------- | ------------------ | ----------------------------------------------- |
| $2^{10}$ | 1,024 ≈ 1K         | Kilobyte                                        |
| $2^{20}$ | 1,048,576 ≈ 1M     | Megabyte                                        |
| $2^{30}$ | 1,073,741,824 ≈ 1B | Gigabyte                                        |
| $2^{32}$ | 4,294,967,296      | Max UINT32; max token count for most frameworks |
| $2^{40}$ | ≈ 1T               | Terabyte; large training datasets               |

### 2.3 Hexadecimal (Base 16)

Hexadecimal (hex) provides a compact human-readable representation of binary data:

- Base $b = 16$; digits $\in \{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, A, B, C, D, E, F\}$
- Each hex digit corresponds to **exactly 4 bits** — this is why hex is used
- A 32-bit value requires only 8 hex digits instead of 32 binary digits
- Standard for representing memory addresses, byte patterns, and floating-point bit representations

**Conversion — binary to hex:** group binary digits in groups of 4 from the right; convert each group:

$$10111010_2 = \underbrace{1011}_{B_{16}} \; \underbrace{1010}_{A_{16}} = \text{BA}_{16}$$

```
BINARY-HEX CONVERSION TABLE
═══════════════════════════════════════════════════════════════════════

Binary │ Hex │ Decimal     Binary │ Hex │ Decimal
───────┼─────┼─────────    ───────┼─────┼─────────
 0000  │  0  │   0          1000  │  8  │   8
 0001  │  1  │   1          1001  │  9  │   9
 0010  │  2  │   2          1010  │  A  │  10
 0011  │  3  │   3          1011  │  B  │  11
 0100  │  4  │   4          1100  │  C  │  12
 0101  │  5  │   5          1101  │  D  │  13
 0110  │  6  │   6          1110  │  E  │  14
 0111  │  7  │   7          1111  │  F  │  15
```

**AI usage:** when inspecting raw model weight files (`.safetensors`, `.bin`), memory dumps, or CUDA kernel outputs, values are displayed in hex. For example, the FP32 bit pattern for `1.0` is `0x3F800000`:

$$\underbrace{0}_{s=0} \; \underbrace{01111111}_{E=127} \; \underbrace{00000000000000000000000}_{m=0} \quad = \quad (-1)^0 \times 1.0 \times 2^{127-127} = 1.0$$

### 2.4 Two's Complement for Signed Integers

Two's complement is the universal standard for representing negative integers in hardware:

**Principle:** represent negative numbers using modular arithmetic. For an $n$-bit system:

- Non-negative: $0$ to $2^{n-1} - 1$; represented in standard binary
- Negative: $-x$ is represented as $2^n - x$
- Or equivalently: **flip all bits, then add 1**

**Range:** $[-2^{n-1}, \; 2^{n-1} - 1]$

| Bit Width      | Range                                                          | AI Usage                       |
| -------------- | -------------------------------------------------------------- | ------------------------------ |
| INT8 (8-bit)   | $[-128, 127]$                                                  | Quantized weights, activations |
| INT16 (16-bit) | $[-32{,}768, 32{,}767]$                                        | Token IDs (vocabulary < 65K)   |
| INT32 (32-bit) | $[-2{,}147{,}483{,}648, \; 2{,}147{,}483{,}647]$               | Accumulator for INT8 matmul    |
| INT64 (64-bit) | $[\approx -9.2 \times 10^{18}, \; \approx 9.2 \times 10^{18}]$ | Dataset sizes, file offsets    |

**Worked example — representing −42 in INT8:**

```
STEP 1: Start with +42
  42₁₀ = 00101010₂

STEP 2: Flip all bits (one's complement)
  00101010  →  11010101

STEP 3: Add 1
  11010101 + 1 = 11010110

RESULT: -42₁₀ = 11010110₂

VERIFICATION: 42 + (-42) should equal 0 (mod 256)
  00101010
+ 11010110
─────────
 100000000  →  Discard carry bit  →  00000000 = 0  ✓
```

**Key properties for hardware design:**

1. **Same hardware for signed and unsigned addition** — the adder doesn't need to know about signs; two's complement arithmetic "just works" with the same circuit
2. **Asymmetry:** there is one more negative value than positive — $-128$ has no positive counterpart in INT8. This matters for symmetric quantization: the range $[-127, 127]$ wastes one code point
3. **Overflow detection:** carry into sign bit $\neq$ carry out of sign bit indicates overflow

**AI relevance — quantization:** when quantizing FP32 weights to INT8, we map continuous values to $\{-128, \ldots, 127\}$. Symmetric quantization uses $\{-127, \ldots, 127\}$ (wastes one level) for simplicity; asymmetric uses all 256 levels for better coverage.

### 2.5 Fixed-Point Representation

Fixed-point represents fractional numbers with a fixed number of bits allocated to the integer and fractional parts:

**Format $Q(m.n)$:** $m$ integer bits + $n$ fractional bits; total $m + n + 1$ bits (including sign bit in two's complement)

**Value interpretation:**

$$\text{value} = \frac{\text{stored integer}}{2^n}$$

**Properties:**

- **Range:** $[-2^m, \; 2^m - 2^{-n}]$
- **Precision:** uniform spacing of $2^{-n}$ across the entire range — every representable value is exactly $2^{-n}$ apart from its neighbours
- **No exponent:** unlike floating-point, there is no exponent field; the radix point position is fixed and implicit

**Example — Q3.4 format (8-bit):**

```
Q3.4 FIXED-POINT (8-BIT)
═══════════════════════════════════════════════════════════════════════

Bit layout: [S][I₂ I₁ I₀][F₃ F₂ F₁ F₀]
             │     │            │
          Sign   Integer     Fraction
          (1)    (3 bits)    (4 bits)

Range:        [-8,  7.9375]    (= [-8, 8 - 1/16])
Resolution:   1/16 = 0.0625

Example: 01011100₂
  Sign = 0 (positive)
  Integer = 101₂ = 5
  Fraction = 1100₂ = 12/16 = 0.75
  Value = 5.75

REPRESENTABLE VALUES NEAR ZERO:
  ... -0.1875  -0.125  -0.0625  0  0.0625  0.125  0.1875 ...
       │         │        │      │     │       │       │
     Uniform spacing of 0.0625 everywhere
```

**Contrast with floating-point:** fixed-point has **uniform** spacing everywhere — the gap between representable values near zero is the same as between large values. Floating-point has **non-uniform** spacing — very dense near zero, sparse for large values. For neural network weights (which cluster near zero), floating-point is generally better.

**Fixed-point in AI:**

- **Hardware quantization:** INT8 with an implicit scale factor $S$ is effectively fixed-point: $\text{value} = \text{int8\_value} \times S$
- **DSP and edge inference:** some microcontrollers (ARM Cortex-M) lack floating-point units entirely; all neural network inference runs in fixed-point
- **Advantages:** simpler hardware; exact representation for dyadic fractions ($k/2^n$); deterministic timing
- **Disadvantages:** cannot represent values spanning many orders of magnitude simultaneously; poor for gradients ($\sim 10^{-7}$) and losses ($\sim 1$) in the same format

---

## 3. IEEE 754 Floating-Point Standard

### 3.1 Floating-Point Representation Principle

Floating-point is **scientific notation for binary**. Instead of allocating a fixed number of bits to the integer and fractional parts, floating-point uses a **mantissa** (significand) and an **exponent** to represent numbers across a vast range of magnitudes with consistent relative precision:

$$x = (-1)^s \times m \times 2^e$$

where:

- **Sign bit** $s \in \{0, 1\}$: $0$ for positive, $1$ for negative
- **Mantissa** (significand) $m \in [1, 2)$: normalised so the leading digit is always 1 (and therefore can be stored implicitly — the "hidden bit")
- **Exponent** $e \in$ integer range: stored with a **bias** so that both negative and positive exponents can be represented as unsigned integers

**The key property:** floating-point provides **the same number of significant bits** regardless of value magnitude. A number near $10^{-30}$ and a number near $10^{30}$ both have the same relative precision. This is exactly what neural networks need — weights near zero and weights near the maximum both get the same relative accuracy.

**Contrast with fixed-point:**

```
FIXED-POINT vs FLOATING-POINT DISTRIBUTION
═══════════════════════════════════════════════════════════════════════

Fixed-point (Q7.8, 16-bit):
Representable values uniformly spaced by 1/256:
│ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │ │
0                                                               128

Near zero: same spacing 1/256  →  relative precision: high
Near 128:  same spacing 1/256  →  relative precision: low


Floating-point (FP16):
Representable values DENSE near zero, SPARSE far from zero:
│││││││││ │ │ │ │ │ │  │  │  │   │   │    │    │      │      │
0                                                           65504

Near zero: spacing ~2⁻²⁴  →  very fine resolution
Near 65504: spacing ~32    →  coarse but same relative precision

Same NUMBER of representable values between 1 and 2
    as between 1024 and 2048 (mantissa bits = 10 for FP16)
```

### 3.2 IEEE 754 FP32 (Single Precision)

FP32 has been the **default numerical format for deep learning** since the field began using GPUs. Understanding its bit layout is essential.

**Bit layout: 32 bits total**

$$\underbrace{s}_{1\text{ bit}} \quad \underbrace{E_7 E_6 E_5 E_4 E_3 E_2 E_1 E_0}_{8\text{ exponent bits}} \quad \underbrace{m_{22} m_{21} \ldots m_1 m_0}_{23\text{ mantissa bits}}$$

```
FP32 BIT LAYOUT (32 BITS)
═══════════════════════════════════════════════════════════════════════

Bit:  31  30────────23  22────────────────────────────────────0
      ┌──┬────────────┬──────────────────────────────────────────┐
      │ s│  Exponent  │              Mantissa                    │
      │  │  (8 bits)  │             (23 bits)                    │
      └──┴────────────┴──────────────────────────────────────────┘
       1      8 bits                   23 bits             = 32 bits
```

**Decoding formula:**

$$x = (-1)^s \times 1.\underbrace{m_{22}m_{21}\ldots m_0}_{\text{23 mantissa bits}} \times 2^{E - 127}$$

where:

- $E$ is the **stored exponent** (unsigned 8-bit integer, $0 \leq E \leq 255$)
- **Exponent bias** = 127: the true exponent is $e = E - 127$, giving $e \in [-126, 127]$ for normal numbers
- $E = 0$ and $E = 255$ are reserved for special values (see §3.3)
- The leading $1.$ before the mantissa is **implicit** (not stored) — this gives 24 bits of significand precision using only 23 stored bits

**Numerical properties:**

| Property                      | Value                                  |
| ----------------------------- | -------------------------------------- |
| Total bits                    | 32                                     |
| Sign bits                     | 1                                      |
| Exponent bits                 | 8                                      |
| Mantissa bits                 | 23 (24 with implicit leading 1)        |
| Exponent bias                 | 127                                    |
| Exponent range (true)         | $-126$ to $127$                        |
| Max positive value            | $\approx 3.4028 \times 10^{38}$        |
| Min positive normal           | $\approx 1.1755 \times 10^{-38}$       |
| Machine epsilon $\varepsilon$ | $2^{-23} \approx 1.192 \times 10^{-7}$ |
| Decimal precision             | ~7.2 significant digits                |
| Min positive subnormal        | $\approx 1.401 \times 10^{-45}$        |

**Machine epsilon** ($\varepsilon$) is the smallest value such that $1 + \varepsilon \neq 1$ in the floating-point system. For FP32, $\varepsilon = 2^{-23}$. This determines the **relative precision** — when you add a value smaller than $\varepsilon \times x$ to $x$, the result rounds back to $x$ and the addition is lost.

**Worked example — decoding a bit pattern:**

Decode: `0 10000001 01000000000000000000000`

```
STEP-BY-STEP FP32 DECODING
═══════════════════════════════════════════════════════════════════════

Given:  0  10000001  01000000000000000000000

Step 1: Sign bit
  s = 0  →  positive

Step 2: Exponent
  E = 10000001₂ = 128 + 1 = 129
  True exponent: e = 129 - 127 = 2

Step 3: Mantissa (implicit leading 1)
  m = 1.01000000000000000000000₂
    = 1 + 0×2⁻¹ + 1×2⁻² + 0×2⁻³ + ...
    = 1 + 0.25
    = 1.25

Step 4: Combine
  x = (-1)⁰ × 1.25 × 2²
    = 1 × 1.25 × 4
    = 5.0

ANSWER: The bit pattern represents 5.0
```

**AI usage of FP32:**

- **Training master weights**: always stored in FP32; the authoritative copy of all parameters
- **Gradient accumulation**: partial gradient sums accumulated in FP32 to prevent precision loss
- **Optimizer states**: Adam $m$ (first moment) and $v$ (second moment) stored in FP32
- **Loss and metrics**: cross-entropy loss computed in FP32 to avoid overflow in exp/log
- **Cannot be used alone for large model training**: LLaMA-3 70B in pure FP32 requires ~280 GB for weights alone

### 3.3 Special Values in IEEE 754

IEEE 754 reserves certain exponent-mantissa combinations for special mathematical values:

```
IEEE 754 SPECIAL VALUES
═══════════════════════════════════════════════════════════════════════

Exponent E │ Mantissa m │ Value              │ Purpose
═══════════╪════════════╪════════════════════╪══════════════════════════
    0       │     0      │ ±0                 │ Signed zero; +0 = -0
    0       │    ≠ 0     │ ±subnormal         │ Gradual underflow
  1–254     │   any      │ ±normal number     │ Standard representation
   255      │     0      │ ±∞ (infinity)      │ Overflow result
   255      │    ≠ 0     │ NaN (Not a Number) │ Invalid operations
```

**Zero ($\pm 0$):**

- Exponent $E = 0$, mantissa $m = 0$; sign bit determines $+0$ or $-0$
- $+0 = -0$ in all comparisons (IEEE 754 mandates this)
- Both exist because $\lim_{x \to 0^+} f(x) \neq \lim_{x \to 0^-} f(x)$ for some functions (e.g., $1/x$)
- AI relevance: ReLU outputs $+0$; signed zero rarely matters in practice

**Subnormal (denormal) numbers:**

- Exponent $E = 0$, mantissa $m \neq 0$
- Value: $x = (-1)^s \times 0.m \times 2^{-126}$ (no implicit leading 1; exponent fixed at $-126$)
- Purpose: **gradual underflow** — fills the gap between zero and the smallest normal number
- Without subnormals: there would be a gap from $0$ to $1.175 \times 10^{-38}$ with nothing in between
- With subnormals: smallest representable positive value = $2^{-149} \approx 1.4 \times 10^{-45}$
- **Hardware cost:** subnormal arithmetic is slower on GPUs (10–100× penalty); many GPU "fast-math" modes **flush subnormals to zero** (`--ftz=true`)
- AI implication: gradients below the subnormal threshold become exactly zero; this is a form of gradient underflow

**Infinity ($\pm\infty$):**

- Exponent $E = 255$, mantissa $m = 0$
- Result of overflow ($x > 3.4 \times 10^{38}$) or division by zero ($x / 0$)
- Arithmetic with infinity: $\infty + x = \infty$; $\infty \times x = \pm\infty$ (for $x \neq 0$); $\infty - \infty = $ NaN
- AI relevance: loss explosion during training produces $+\infty$; must detect and handle (skip step, reduce learning rate)

**NaN (Not a Number):**

- Exponent $E = 255$, mantissa $m \neq 0$
- Two types:
  - **Quiet NaN (qNaN):** propagates silently through all arithmetic — $\text{qNaN} + x = \text{qNaN}$, $\text{qNaN} \times x = \text{qNaN}$, $\text{qNaN} > x = \text{false}$. **Extremely dangerous in training** — loss appears to be a number but all weights are corrupted
  - **Signalling NaN (sNaN):** raises a hardware exception when used in arithmetic; useful for debugging but rarely used in GPU code
- Produced by: $0/0$, $\infty - \infty$, $\sqrt{-1}$, $\log(-x)$
- AI relevance: NaN in loss → all subsequent gradients are NaN → all weights become NaN → model destroyed. Must check for NaN explicitly; PyTorch `torch.autograd.detect_anomaly()` helps

**Ordering of all special values:**

$$-\infty < \text{negative normals} < \text{negative subnormals} < -0 = +0 < \text{positive subnormals} < \text{positive normals} < +\infty$$

NaN is **unordered** — $\text{NaN} \neq \text{NaN}$ (NaN is not equal to itself!). This is the standard way to check for NaN: `x != x` is true only if `x` is NaN.

### 3.4 FP32 Arithmetic Properties

Floating-point arithmetic violates several "obvious" algebraic properties that hold for real numbers. These violations have direct consequences for neural network training:

**1. Associativity failure** — $(a + b) + c \neq a + (b + c)$ in general:

$$\text{Consider: } a = 10^8, \; b = 1, \; c = -10^8$$

$$(a + b) + c = (10^8 + 1) + (-10^8)$$

In FP32, $10^8 + 1 = 10^8$ because $1 < \varepsilon \times 10^8 = 10^8 \times 1.19 \times 10^{-7} \approx 11.9$. Wait — $1 < 11.9$, so $1$ IS below the relative precision threshold. The result is $10^8$, and then $10^8 - 10^8 = 0$.

But:

$$a + (b + c) = 10^8 + (1 + (-10^8)) = 10^8 + (1 - 10^8)$$

Here $1 - 10^8 = -99{,}999{,}999$, and $10^8 + (-99{,}999{,}999) = 1$.

**Result:** $(a + b) + c = 0$ but $a + (b + c) = 1$. The "correct" mathematical answer is 1.

**AI impact:** multi-GPU training performs parallel reduction (summing gradients across GPUs) in different orders depending on timing → different results each run. **This is why multi-GPU training is non-deterministic even with the same seed.**

**2. Commutativity — always holds:** $a + b = b + a$ and $a \times b = b \times a$ in IEEE 754. The order of two operands does not matter. (But the order of three or more does, because associativity fails.)

**3. Distributivity failure** — $a \times (b + c) \neq a \times b + a \times c$ in general. Each intermediate operation rounds independently, accumulating different errors.

### 3.5 Rounding Modes

IEEE 754 defines five rounding modes. The choice of rounding mode affects the bias and variance of numerical errors:

**1. Round to Nearest, Ties to Even (RNE) — the default:**

- Round to the nearest representable value
- When the value is exactly midway between two representable values, round to the one whose last mantissa bit is 0 (even)
- **Why ties-to-even?** Eliminates statistical bias — ties-to-up would systematically inflate values over many operations; ties-to-even has zero expected bias
- This is the default for FP32, BF16, and all standard GPU computation

**2. Round Toward Zero (Truncation):**

- Simply discard all bits beyond the mantissa width
- Always rounds toward zero: positive values round down, negative values round up
- Used in: integer conversion (`int(3.7) = 3`); some fixed-point hardware

**3. Round Toward $+\infty$ (Ceiling):**

- Always round up (toward positive infinity)
- Used in interval arithmetic for computing upper bounds

**4. Round Toward $-\infty$ (Floor):**

- Always round down (toward negative infinity)
- Used in interval arithmetic for computing lower bounds

**5. Round to Nearest, Ties Away from Zero:**

- What most people think "normal" rounding is: 0.5 always rounds up
- NOT the IEEE default; less common in hardware
- Introduces slight positive bias over many operations

**AI-specific: Stochastic Rounding:**

- Not part of IEEE 754 but increasingly important for low-precision training
- Round up or down **randomly** with probability proportional to the fractional position
- $P(\text{round up}) = x - \lfloor x \rfloor$, $P(\text{round down}) = \lceil x \rceil - x$
- **Unbiased:** $E[\text{SR}(x)] = x$ — preserves small gradient updates in expectation
- Used in: some FP8 training implementations; Graphcore IPU hardware
- Cost: requires random number generation per operation

### 3.6 Catastrophic Cancellation

**The problem:** subtracting two nearly equal floating-point numbers catastrophically reduces the number of significant bits in the result.

**Mechanism:**

Consider $a = 1.0000001$ and $b = 1.0000000$ stored in FP32 (which has ~7 decimal digits of precision):

$$a - b = 0.0000001 = 1 \times 10^{-7}$$

Both $a$ and $b$ have 7 significant digits. Their difference has only **1 significant digit** — the other 6 significant digits cancelled. The result is represented as $1.0 \times 10^{-7}$ with the remaining mantissa bits filled with garbage (the hardware doesn't know what the true value would have been beyond the stored precision).

**Formal statement:** if $a$ and $b$ agree to $k$ significant bits, then $a - b$ has at most $(\text{mantissa bits} - k)$ significant bits. For FP32 with 24-bit significand:

- If $a$ and $b$ agree to 20 bits: $a - b$ has only 4 significant bits
- If $a$ and $b$ agree to 23 bits: $a - b$ has only 1 significant bit
- If $a$ and $b$ agree to 24 bits: $a - b = 0$ (complete cancellation)

**Where cancellation occurs in neural networks:**

- **Attention logit differences:** softmax($z_i - \max(z)$) where logits are close
- **Layer normalisation:** $x - \mu$ where $\mu$ is close to $x$
- **Gradient computation:** chain rule products where terms nearly cancel
- **Residual connections:** $f(x) + x$ followed by subtraction in backward pass

**Mitigations:**

1. **Log-sum-exp trick** for softmax (§8.4): avoids computing exp of very large/small numbers
2. **Kahan summation** (§3.7): maintains error correction term to recover lost precision
3. **RMSNorm instead of LayerNorm** (§8.6): avoids mean subtraction entirely
4. **Reordering operations:** compute $(a^2 - b^2) = (a+b)(a-b)$ instead of direct subtraction when $a \approx b$

### 3.7 Kahan Summation Algorithm

**The problem:** summing $n$ floating-point numbers with naive sequential addition accumulates rounding error of $O(n\varepsilon)$, where $\varepsilon$ is machine epsilon. For a training step with millions of gradient contributions, this error can become significant.

**Kahan's solution (1965):** maintain a running **compensation term** $c$ that captures the rounding error from each addition and feeds it back into the next:

```
KAHAN SUMMATION ALGORITHM
═══════════════════════════════════════════════════════════════════════

Input: values x₁, x₂, ..., xₙ
Output: sum with O(ε) error (instead of O(nε))

    sum = 0.0          // Running total
    c   = 0.0          // Compensation for lost low-order bits

    for each xᵢ:
        y = xᵢ - c         // Compensate: add back what was lost last time
        t = sum + y         // Tentative new sum (rounding happens here)
        c = (t - sum) - y   // Recover rounding error: what was lost
        sum = t             // Update running total

    return sum
```

**How it works — step by step with FP32 (7 decimal digits):**

```
KAHAN SUMMATION TRACE
═══════════════════════════════════════════════════════════════════════

Sum: [1.0,  1e-7,  1e-7,  1e-7,  1e-7]

NAIVE SUMMATION:
  sum = 1.0
  sum = 1.0 + 1e-7 = 1.0000001     ← barely fits in FP32
  sum = 1.0000001 + 1e-7 = 1.0000002   ← may round
  sum = 1.0000002 + 1e-7 = 1.0000003   ← each step loses precision
  sum = 1.0000003 + 1e-7 = 1.0000004
  Result: 1.0000004  (limited by FP32 precision near 1.0)

KAHAN SUMMATION:
  Step 1: x₁ = 1.0
    y = 1.0 - 0 = 1.0
    t = 0 + 1.0 = 1.0
    c = (1.0 - 0) - 1.0 = 0    // no error
    sum = 1.0

  Step 2: x₂ = 1e-7
    y = 1e-7 - 0 = 1e-7
    t = 1.0 + 1e-7 = 1.0000001
    c = (1.0000001 - 1.0) - 1e-7    // recovers rounding error
    sum = 1.0000001

  Step 3: x₃ = 1e-7
    y = 1e-7 - c                // compensates for error from step 2
    t = 1.0000001 + y
    c = (t - 1.0000001) - y     // captures new error
    sum = t

  ... continues, accumulating with compensation ...
  Result: more accurate than naive sum
```

**Error analysis:**

- Naive summation: error = $O(n\varepsilon)$ — grows linearly with number of terms
- Kahan summation: error = $O(\varepsilon)$ — **independent of $n$**; dramatic improvement for large sums
- Cost: approximately 4 FLOPs per element instead of 1; worth the overhead for numerical stability

**AI applications:**

- **Gradient accumulation** across micro-batches: sum thousands of small gradient contributions
- **Loss computation** over large batches: sum cross-entropy losses across many tokens
- **Weight norm computation**: sum of squares of millions of parameters
- In practice: PyTorch and JAX gradient accumulation uses FP32 (which provides sufficient precision without Kahan); Kahan is used in specialised scenarios where even FP32 is insufficient

---

## 4. Floating-Point Formats for AI

This section covers every floating-point format relevant to modern AI systems, from the rarely-used FP64 down to the frontier FP8 formats that power 2024–2026 era training.

### 4.1 FP64 (Double Precision)

**Bit layout:** 64 = 1 sign + 11 exponent + 52 mantissa

| Property              | Value                                                 |
| --------------------- | ----------------------------------------------------- |
| Exponent bias         | 1023                                                  |
| Exponent range (true) | $-1022$ to $1023$                                     |
| Max value             | $\approx 1.798 \times 10^{308}$                       |
| Min positive normal   | $\approx 2.225 \times 10^{-308}$                      |
| Machine epsilon       | $\varepsilon = 2^{-52} \approx 2.220 \times 10^{-16}$ |
| Decimal precision     | ~15–17 significant digits                             |

**AI relevance — mostly irrelevant for neural networks:**

- GPU throughput: A100 FP64 = 19.5 TFLOPS vs FP32 = 19.5 TFLOPS (scalar) vs BF16 = 312 TFLOPS (tensor core). FP64 is 16× slower than BF16.
- No neural network operation requires 15-digit precision. The signal-to-noise ratio of gradient estimates (due to mini-batch sampling) is far larger than FP32 precision errors.
- **Niche uses:** eigenvalue decomposition in research; high-precision statistical hypothesis tests; numerical ODE/SDE solvers for diffusion models; Gram matrix condition number analysis
- **Rule of thumb:** if you think you need FP64 for neural networks, you probably have a numerical stability bug that should be fixed structurally (better algorithm) rather than with more bits

### 4.2 FP32 (Single Precision) — Detailed

FP32 was the **unquestioned standard** for deep learning from 2012 to approximately 2018. All early frameworks (Theano, Caffe, early TensorFlow, early PyTorch) used FP32 by default.

```
FP32 ROLE IN MODERN LLM TRAINING (2026)
═══════════════════════════════════════════════════════════════════════

Still used for:
  ✓ Master weights (authoritative copy)
  ✓ Optimizer states (Adam m, v)
  ✓ Gradient accumulation (sum of gradients)
  ✓ Loss and metrics computation
  ✓ Learning rate and hyperparameters
  ✓ Numerical stability fallback (softmax intermediate)

NO LONGER used for (replaced by BF16/FP8):
  ✗ Forward pass matmul computation
  ✗ Backward pass gradient computation
  ✗ Activation storage
  ✗ Weight communication across GPUs
  ✗ Inference of any kind
```

**Why FP32 persists for certain operations:** gradient accumulation sums millions of small values. With BF16 ($\varepsilon = 7.8 \times 10^{-3}$), a gradient contribution smaller than $0.78\%$ of the current sum is lost. Over millions of steps, these lost contributions compound. FP32 ($\varepsilon = 1.19 \times 10^{-7}$) resolves contributions as small as $0.0000119\%$ of the sum — 65,000× more precise.

**Memory cost for a 70B model in pure FP32:**

- Weights: $70 \times 10^9 \times 4 = 280$ GB
- Adam $m$: 280 GB
- Adam $v$: 280 GB
- Total optimizer + weights: 840 GB
- Impractical on any single GPU; requires aggressive model parallelism

### 4.3 TF32 (TensorFloat-32)

TF32 is an **NVIDIA-proprietary** format that exists only inside tensor core hardware. It is not an IEEE standard and cannot be stored in memory — it is a computational format.

**Bit layout:** 19 bits = 1 sign + 8 exponent + 10 mantissa

```
TF32 — A HYBRID FORMAT
═══════════════════════════════════════════════════════════════════════

FP32:  [1 sign][8 exponent][23 mantissa]     ← 32 bits
TF32:  [1 sign][8 exponent][10 mantissa]     ← 19 bits
BF16:  [1 sign][8 exponent][ 7 mantissa]     ← 16 bits
FP16:  [1 sign][5 exponent][10 mantissa]     ← 16 bits

TF32 takes:
  • Exponent from FP32 (8 bits → same range as FP32)
  • Mantissa from FP16 (10 bits → same precision as FP16)
  • Result: FP32 range with FP16 precision
```

**How it works in practice:**

- When you call a FP32 matmul on an A100 or H100, cuBLAS **automatically** uses TF32 internally
- Inputs are FP32; tensor core truncates mantissa to 10 bits for multiply; accumulates in FP32
- Output is FP32
- Throughput: A100 TF32 = 156 TFLOPS vs FP32 scalar = 19.5 TFLOPS — **8× faster**

**Implication:** if you're running "FP32 training" on an A100/H100, your matmuls are actually TF32 matmuls unless you explicitly disable it (`torch.backends.cuda.matmul.allow_tf32 = False`). This is fine for virtually all training — the precision loss from 23→10 mantissa bits in the multiply step is negligible because accumulation is still FP32.

### 4.4 FP16 (Half Precision)

**IEEE 754 binary16.** The first reduced-precision format to gain widespread ML use, starting with NVIDIA Volta (2017).

**Bit layout:** 16 = 1 sign + 5 exponent + 10 mantissa

| Property              | Value                                                |
| --------------------- | ---------------------------------------------------- |
| Exponent bias         | 15                                                   |
| Exponent range (true) | $-14$ to $15$                                        |
| Max value             | $65{,}504$                                           |
| Min positive normal   | $\approx 6.104 \times 10^{-5}$                       |
| Machine epsilon       | $\varepsilon = 2^{-10} \approx 9.766 \times 10^{-4}$ |
| Decimal precision     | ~3.3 significant digits                              |

**The critical limitation — narrow dynamic range:**

FP16 can represent values from $6.1 \times 10^{-5}$ to $65{,}504$. This range is **far too narrow** for LLM training:

```
FP16 FAILURE MODES IN LLM TRAINING
═══════════════════════════════════════════════════════════════════════

OVERFLOW: values > 65,504 become ∞
  • Loss values in early training often > 10
  • Attention logits (before softmax) can reach 100+
  • If any intermediate value exceeds 65,504 → ∞ → NaN → training dead

UNDERFLOW: values < 6.1×10⁻⁵ become 0
  • Gradients are often ~10⁻⁷
  • In FP16, these gradients → 0
  • Zero gradients → weights don't update → training stalls
  • This is SILENT — loss appears stable but model isn't learning

                    FP16 representable range
                    ┌──────────────────────┐
  gradient zone     │                      │     loss zone
  (10⁻⁷ to 10⁻⁴)   │  6.1×10⁻⁵  →  65504  │    (1 to 100+)
  ████████░░░░░░░░░│██████████████████████│░░░░░░░████████
  underflow!        │                      │     overflow!
```

**Loss scaling — the FP16 workaround:**

1. Before backward pass: multiply loss by a large constant $S$ (e.g., 128, 1024, or dynamic)
2. All gradients are scaled by $S$ → shifted into representable range
3. After gradient computation: divide all gradients by $S$ before optimizer step
4. Dynamic loss scaling: start with large $S$; if overflow (Inf/NaN) detected, halve $S$; if no overflow for $N$ steps, double $S$

**2026 status:** FP16 is **largely replaced by BF16** for training. Still used in some inference engines and legacy code. The loss scaling machinery made FP16 training possible but fragile — BF16 eliminates the need entirely.

### 4.5 BF16 (Brain Float 16)

BF16 is **the most important number format innovation for deep learning** in the 2018–2026 era. Invented by Google Brain for TPU hardware, it was specifically designed for neural network training.

**Bit layout:** 16 = 1 sign + 8 exponent + 7 mantissa

| Property            | Value                                               |
| ------------------- | --------------------------------------------------- |
| Exponent bits       | 8 (same as FP32)                                    |
| Mantissa bits       | 7                                                   |
| Exponent bias       | 127 (same as FP32)                                  |
| Max value           | $\approx 3.39 \times 10^{38}$ (same as FP32)        |
| Min positive normal | $\approx 1.175 \times 10^{-38}$ (same as FP32)      |
| Machine epsilon     | $\varepsilon = 2^{-7} \approx 7.813 \times 10^{-3}$ |
| Decimal precision   | ~2.4 significant digits                             |

**Why BF16 dominates — the key insight:**

BF16 has the **same 8-bit exponent as FP32**, giving it identical dynamic range ($10^{-38}$ to $10^{38}$). This single design choice eliminates both overflow and underflow problems that plague FP16:

```
BF16 vs FP16 — THE RANGE ADVANTAGE
═══════════════════════════════════════════════════════════════════════

                FP16 range                    FAILS for AI
        ┌───────────────────────┐
   6×10⁻⁵                  65,504
        │███████████████████████│

        BF16 range                            WORKS for AI
┌───────────────────────────────────────────────────────────────┐
1.2×10⁻³⁸                                              3.4×10³⁸
│███████████████████████████████████████████████████████████████│

        FP32 range (identical to BF16!)
┌───────────────────────────────────────────────────────────────┐
1.2×10⁻³⁸                                              3.4×10³⁸
│███████████████████████████████████████████████████████████████│

Gradients at 10⁻⁷?    In BF16 range ✓     In FP16? UNDERFLOW ✗
Large logits at 200?   In BF16 range ✓     In FP16? OVERFLOW  ✗
```

**The precision trade-off:**

- BF16 has only 7 mantissa bits vs FP32's 23 — it is $2^{16} = 65{,}536\times$ less precise in relative terms
- Machine epsilon $\varepsilon = 7.8 \times 10^{-3}$: any value smaller than $0.78\%$ of the current number is lost when added
- This means BF16 can only represent $\sim 2.4$ decimal digits of precision

**Why this is acceptable for ML:** the noise from mini-batch gradient estimation (which typically has variance on the order of the gradient magnitude) far exceeds BF16 precision error. The gradient itself is a noisy estimate — adding $0.78\%$ numerical noise to a signal that already has $\sim 10\%$ statistical noise is negligible.

**FP32 ↔ BF16 conversion:**

- BF16 to FP32: pad 16 zero bits to the mantissa → trivial; free in hardware
- FP32 to BF16: drop the lower 16 mantissa bits (with rounding) → trivial; 1 cycle
- This simplicity is by design: the exponent field is identical, so no exponent conversion needed

**Hardware support:**

- Google TPU v2+ (2017): BF16 native from inception
- NVIDIA A100 (2020): BF16 tensor cores at 312 TFLOPS
- NVIDIA H100 (2022): BF16 tensor cores at 989 TFLOPS
- All modern training hardware supports BF16 at full tensor core speed

**BF16 matmul accumulation:** NVIDIA tensor cores compute BF16 × BF16 but **accumulate the intermediate sums in FP32**. The result is then optionally converted back to BF16 for storage. This means the matmul output quality is much better than naive BF16 arithmetic — the FP32 accumulation prevents precision loss in long dot products.

### 4.6 FP16 vs BF16 Head-to-Head

| Property                | FP16                                           | BF16                           |
| ----------------------- | ---------------------------------------------- | ------------------------------ |
| Total bits              | 16                                             | 16                             |
| Exponent bits           | 5                                              | 8                              |
| Mantissa bits           | 10                                             | 7                              |
| Max value               | 65,504                                         | $3.4 \times 10^{38}$           |
| Min positive normal     | $6.1 \times 10^{-5}$                           | $1.2 \times 10^{-38}$          |
| Machine epsilon         | $9.77 \times 10^{-4}$                          | $7.81 \times 10^{-3}$          |
| Needs loss scaling      | **Yes** (critical)                             | **No**                         |
| Gradient underflow risk | **High** (gradients $< 6 \times 10^{-5}$ lost) | **None** (range matches FP32)  |
| Precision per value     | Higher (10-bit mantissa)                       | Lower (7-bit mantissa)         |
| FP32 conversion cost    | Non-trivial (exponent remapping)               | **Trivial** (truncate 16 LSBs) |
| GPU tensor core support | V100+, A100+, H100+                            | A100+, H100+ (TPU: v2+)        |
| 2026 training status    | **Legacy**                                     | **Standard**                   |
| 2026 inference status   | Some engines (TensorRT)                        | Standard                       |

**The verdict:** BF16 is strictly preferred for LLM training. FP16 wins only in niche inference scenarios where the extra 3 mantissa bits matter and loss scaling overhead is acceptable.

### 4.7 FP8 Formats (E4M3 and E5M2)

FP8 is the **frontier training format** as of 2024–2026. NVIDIA H100 was the first GPU with FP8 tensor cores, enabling 2× throughput over BF16. There are two complementary 8-bit floating-point formats:

**FP8 E4M3 — optimised for precision (forward pass):**

| Property                | Value                                                  |
| ----------------------- | ------------------------------------------------------ |
| Bit layout              | 1 sign + 4 exponent + 3 mantissa                       |
| Exponent bias           | 7                                                      |
| Max value               | 448                                                    |
| Machine epsilon         | $\varepsilon = 2^{-3} = 0.125$ (12.5% relative error!) |
| Decimal precision       | ~1 significant digit                                   |
| Use case                | Weights and activations in forward pass                |
| Special: no $\pm\infty$ | NaN is the only special value (S=1, E=1111, M=111)     |

**FP8 E5M2 — optimised for range (gradients):**

| Property                | Value                                               |
| ----------------------- | --------------------------------------------------- |
| Bit layout              | 1 sign + 5 exponent + 2 mantissa                    |
| Exponent bias           | 15                                                  |
| Max value               | 57,344                                              |
| Machine epsilon         | $\varepsilon = 2^{-2} = 0.25$ (25% relative error!) |
| Decimal precision       | less than 1 significant digit                       |
| Use case                | Gradients in backward pass                          |
| Has $\pm\infty$ and NaN | Like IEEE 754 FP16                                  |

```
FP8 FORMAT COMPARISON
═══════════════════════════════════════════════════════════════════════

E4M3 — more mantissa bits → better precision
┌──┬────────┬──────┐
│ s│  EEEE  │ MMM  │    4 exponent + 3 mantissa
└──┴────────┴──────┘    Range: ±448; good precision for weights
                        Use: forward pass (activations, weights)

E5M2 — more exponent bits → wider range
┌──┬──────────┬────┐
│ s│  EEEEE   │ MM │    5 exponent + 2 mantissa
└──┴──────────┴────┘    Range: ±57,344; wider range for gradients
                        Use: backward pass (gradients)

WHY TWO FORMATS?
  Forward pass: values clustered in known range; precision matters more
  Backward pass: gradients span many orders of magnitude; range matters more
```

**The FP8 challenge — extreme quantization noise:**

FP8 E4M3 has $\varepsilon = 0.125$: every value has up to 12.5% relative error. This is enormous — for comparison, BF16 has 0.78%. At this precision level, **per-tensor or per-block scaling** is mandatory to keep values within the representable range.

**Hardware throughput:** H100 FP8 = 3,958 TOPS — that's 4× BF16 throughput and 60× FP32 scalar throughput. This massive speed advantage drives the push toward FP8 training.

**DeepSeek-V3 (2024):** trained the entire model (forward and backward) in FP8 with tile-wise scaling. This was the first commercial frontier model to demonstrate FP8 training at scale, achieving massive cost reduction vs BF16 baseline.

### 4.8 FP8 Scaling Strategies

Because FP8 has such limited precision and range, every tensor needs an associated **scale factor** $S$ that maps the tensor's actual value range into the FP8 representable range:

$$\text{FP8\_value} = \text{round\_to\_FP8}\!\left(\frac{\text{real\_value}}{S}\right)$$

$$\text{real\_value} \approx \text{FP8\_value} \times S$$

**1. Per-tensor scaling — simplest:**

- One scalar $S$ per entire tensor
- $S = \max(|\text{tensor}|) / \text{max\_FP8\_value}$
- Fast; minimal overhead; poor when tensor has outliers
- If one element is $100\times$ larger than the rest, all other elements lose precision because $S$ is set by the outlier

**2. Per-block (tile) scaling — optimal for training:**

- One scale $S$ per $B \times B$ block of the matrix (e.g., $B = 128$)
- Each block has its own scale, matching its local value range
- Overhead: $O(M \times N / B^2)$ extra scale factors per matmul
- **DeepSeek-V3 uses tile-wise scaling** with $B = 128$: this was the key innovation that made FP8 training work at frontier scale

```
PER-TENSOR vs PER-BLOCK SCALING
═══════════════════════════════════════════════════════════════════════

Matrix with outlier:
┌─────────────────────────────────────────┐
│  0.01  0.02  0.01  0.03  0.02  │ 100.0 │  ← outlier
│  0.02  0.01  0.03  0.01  0.02  │  0.01 │
│  0.01  0.03  0.02  0.01  0.01  │  0.02 │
│  0.02  0.01  0.01  0.02  0.03  │  0.01 │
└─────────────────────────────────────────┘

Per-tensor scaling: S = 100.0 / 448 ≈ 0.223
  All 0.01 values → 0.01/0.223 = 0.045 → rounds to FP8: 0.0 or 0.0625
  MASSIVE precision loss for small values!

Per-block scaling (2×3 blocks shown):
  Block1 (top-left 2×3): max=0.03; S₁ = 0.03/448 ≈ 6.7×10⁻⁵
    0.01 → 0.01/6.7×10⁻⁵ ≈ 149 → FP8: 152  ← good precision!
  Block2 (top-right 2×3): max=100; S₂ = 100/448 ≈ 0.223
    100.0 → 448 → FP8: 448  ← full range for outlier
```

**3. Delayed scaling:**

- Use the scale factor computed from the **previous iteration** (slightly stale)
- Avoids an extra pass over the tensor to compute the current max
- Works because tensor statistics change slowly between iterations

**4. Just-in-time scaling:**

- Compute scale fresh each iteration by scanning the tensor
- Accurate but requires an extra read pass over the data
- Used when tensor distributions change rapidly

**5. Stochastic rounding for FP8:**

- Standard round-to-nearest introduces systematic bias at low precision
- Stochastic rounding: round up with probability = fractional position; round down otherwise
- $E[\text{SR}(x)] = x$ — unbiased in expectation
- Critical for gradient accumulation in FP8: small gradient updates are preserved probabilistically
- Cost: requires hardware random number generation per operation

---

## 5. Integer Formats for AI

Integer formats have become central to AI inference through quantization. Unlike floating-point, integers have no exponent — all values are uniformly spaced. This simplicity makes integer arithmetic faster, cheaper, and more energy-efficient.

### 5.1 INT32 — Full Precision Integer

- **32-bit two's complement**; range $[-2{,}147{,}483{,}648, \; 2{,}147{,}483{,}647]$
- **AI role: accumulator.** When multiplying two INT8 values, the product can be up to $127 \times 127 = 16{,}129$ — this fits in INT16. But a dot product of 4096-dimensional INT8 vectors produces partial sums up to $4096 \times 16{,}129 \approx 66$ million — requires INT32 to avoid overflow.
- **Pipeline:** INT8 matmul → INT32 accumulation → scale to FP32 → output as BF16 or INT8
- GPU supports INT32 natively; standard for index operations, loop counters, and token IDs in CUDA kernels

### 5.2 INT16

- **16-bit two's complement**; range $[-32{,}768, \; 32{,}767]$
- **Rare in AI** — too narrow for accumulation (INT8 × INT8 summed over $>2$ elements overflows) and too wide for weight storage (twice the memory of INT8)
- **Usage:** token IDs are often stored as INT16 when vocabulary size $< 65{,}536$ (using UINT16). GPT-2's vocabulary of 50,257 tokens fits in UINT16.
- Some DSP-based inference engines use INT16 for activations

### 5.3 INT8

INT8 is the **workhorse format for production LLM inference** in 2026:

- **8-bit two's complement**; range $[-128, 127]$; 1 byte per value
- **GPU throughput:** A100 INT8 = 624 TOPS; 2× FP16/BF16 tensor core throughput
- **Memory:** 2× compression vs BF16; 4× vs FP32

**Quantization — mapping float to INT8:**

$$q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{S}\right) + Z, \; -128, \; 127\right)$$

where $S$ (scale) and $Z$ (zero point) define the mapping:

- **Symmetric quantization** ($Z = 0$): maps $[-\alpha, \alpha]$ to $[-127, 127]$ where $\alpha = \max(|x|)$; $S = \alpha / 127$
- **Asymmetric quantization** ($Z \neq 0$): maps $[x_{\min}, x_{\max}]$ to $[-128, 127]$; $S = (x_{\max} - x_{\min}) / 255$; $Z = \text{round}(-x_{\min} / S) - 128$

**Dequantization:**

$$\hat{x} = S \cdot (q - Z)$$

**Quantization granularity:**

- **Per-tensor:** one $S$, $Z$ for entire weight matrix — simple but outliers dominate
- **Per-channel (per-row/column):** one $S$, $Z$ per output channel — standard for weight matrices
- **Per-group:** one $S$, $Z$ per group of $g$ values (e.g., $g = 128$) — better for activations
- **Per-token:** one $S$, $Z$ per token's activation vector — handles dynamic range well

**Activation quantization is harder than weight quantization:**

- Weights are static (computed once during quantization); activations change every forward pass
- Transformer activations have **outlier channels** — a few channels have values 10–100× larger than others
- **SmoothQuant (Xiao et al. 2023):** migrate difficulty from activations to weights by per-channel scaling:

  $$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X} \cdot \hat{W}$$

  where $s_j = \max(|X_{:,j}|)^\alpha / \max(|W_{j,:}|)^{1-\alpha}$ with $\alpha = 0.5$

**LLM.int8() (Dettmers et al. 2022):** handles outlier features by computing them in FP16 while the rest uses INT8 — mixed-precision decomposition at the feature level

### 5.4 UINT8 — Unsigned Integer 8-bit

- **Range:** $[0, 255]$; no negative values
- **AI uses:**
  - **Activations after ReLU:** always non-negative; UINT8 matches perfectly
  - **Image pixel values:** standard image format (0–255 per channel)
  - **Asymmetric quantization zero point:** $Z \in [0, 255]$ stored as UINT8
- **Asymmetric activation quantization** often uses UINT8: maps $[x_{\min}, x_{\max}] \to [0, 255]$ with the conceptual zero mapped to UINT8 value $Z$

### 5.5 INT4

INT4 is the **primary format for consumer-GPU LLM inference** in 2026:

- **4-bit two's complement**; range $[-8, 7]$; 0.5 bytes per value
- **Memory:** 4× compression vs BF16; 8× vs FP32
- LLaMA-3 70B in INT4: ~35 GB — fits on a single RTX 4090 (24 GB VRAM, with offloading) or 2× RTX 3090

**Hardware status:**

- Not natively supported by standard tensor cores for compute on H100 — must be **dequantized to BF16** before matmul
- NVIDIA Ada Lovelace (RTX 4090): INT4 tensor cores with 2× INT8 throughput
- Typical deployment: **W4A16** (Weight INT4, Activation BF16) — weights stored in INT4, dequantized to BF16 on the fly before each matmul

```
INT4 DEQUANTIZE-ON-THE-FLY PIPELINE
═══════════════════════════════════════════════════════════════════════

Memory (HBM)          GPU Compute (SM)
┌────────────┐        ┌──────────────────────────────┐
│ INT4 weight│──load─→│  Unpack INT4 → INT8          │
│ (0.5 B/val)│        │  Dequantize: INT8 × S → BF16 │
│            │        │  BF16 matmul with activation  │
│            │        │  FP32 accumulation            │
│            │        │  Output BF16                  │
└────────────┘        └──────────────────────────────┘

Benefit: 4× less memory bandwidth (bandwidth-bound → 4× faster)
Cost: dequantization overhead (small; amortised over matmul compute)
```

**INT4 quantization methods:**

- **GPTQ (Frantar et al. 2023):** post-training quantization using approximate second-order information (Hessian inverse); groups of 128; deterministic
- **AWQ (Lin et al. 2023, MLSys 2024 Best Paper):** activation-aware weight quantization; protects salient channels based on activation magnitudes
- **Symmetric INT4:** levels $\{-8, -7, \ldots, -1, 0, 1, \ldots, 7\}$; $S = \max(|w|) / 7$
- **Asymmetric INT4 (UINT4):** levels $\{0, 1, \ldots, 15\}$; allows asymmetric weight distributions

### 5.6 INT2 and INT1 (Binary)

**INT2 — 2-bit quantization:**

- Range: $[-2, 1]$ (signed) or $\{0, 1, 2, 3\}$ (unsigned)
- 4× compression vs INT8; 16× vs FP32
- **QuIP# (Tseng et al. 2024):** achieves usable 2-bit quantization using incoherence processing (random orthogonal transforms to distribute weight magnitudes more uniformly)
- Quality: significant degradation; 2–8 points PPL increase over BF16; research-grade only

**INT1 — binary neural networks:**

- 1 bit: $\{0, 1\}$ or $\{-1, +1\}$; 8× compression vs INT8; 32× vs FP32
- **XNOR-popcount matmul:** binary weights transform multiply-accumulate into bitwise XNOR followed by popcount (counting set bits); extremely fast hardware implementation:
  - Standard matmul: $y = \sum_i w_i x_i$ (multiply + add)
  - Binary matmul: $y = \text{popcount}(\text{XNOR}(\mathbf{w}, \mathbf{x}))$ (bitwise + count)
  - 64 binary multiplies per single 64-bit XNOR instruction; then one popcount
- **BitNet (Wang et al. 2023):** INT1 weights trained from scratch; competitive with FP16 at large scale when combined with proper training methodology
- **Practical challenge:** model quality degrades severely below INT4 without training from scratch with quantization-aware methods

### 5.7 Packed Integer Formats

Low-bit integers are packed into larger registers for efficient storage and SIMD processing:

**INT4 packing:**

- 8 INT4 values packed into a single 32-bit word; 16 values per 64-bit word
- Layout: little-endian; first value in least significant bits

```
INT4 PACKING IN A 32-BIT REGISTER
═══════════════════════════════════════════════════════════════════════

32-bit word: [v₇][v₆][v₅][v₄][v₃][v₂][v₁][v₀]
              MSB                            LSB

Each vᵢ is 4 bits (one INT4 value)

Extraction in Python/CUDA:
  value_i = (int32_word >> (4 * i)) & 0xF    # Extract i-th INT4

  # For signed INT4, sign-extend:
  if value_i >= 8:
      value_i -= 16    # Convert [0,15] → [-8, 7]
```

**INT2 packing:** 16 values per 32-bit word
**INT1 packing:** 32 values per 32-bit word — entire binary weight vector in one register

**SIMD unpacking:**

- Modern GPUs and CPUs can unpack and process multiple packed values per instruction
- AVX-512: 512-bit register → 128 packed INT4 values per register
- Key performance consideration: unpacking overhead must be amortised over computation; fine-grained per-element unpacking kills performance

---

## 6. Non-Uniform and Specialised Formats

Standard integer and floating-point formats use uniform or logarithmic spacing. Specialised formats can achieve better accuracy per bit by matching the spacing to the statistical distribution of the data.

### 6.1 Normal Float 4-bit (NF4)

NF4 is an **information-theoretically optimal** 4-bit format for normally distributed data. Introduced by Dettmers et al. (2023) as part of QLoRA (Quantization-aware Low-Rank Adaptation).

**Key insight:** transformer weights are approximately normally distributed $w \sim \mathcal{N}(0, \sigma^2)$. A uniform quantization grid (INT4) places equal numbers of levels across the range, but most weights cluster near zero. NF4 places **more levels near zero** (where the density is highest) and **fewer levels at the tails** (where few weights exist).

**Construction — quantile-based level placement:**

1. Compute the quantiles $q_1, q_2, \ldots, q_{15}$ of the standard normal distribution $\mathcal{N}(0, 1)$ that divide the probability mass into 16 equal regions
2. The level for each region is the expected value (centroid) of the distribution within that region
3. Normalise all levels to $[-1, 1]$

**NF4 levels (16 values, symmetric around 0):**

$$\{-1.0, -0.6962, -0.5251, -0.3949, -0.2840, -0.1848, -0.0922, 0,$$
$$0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1.0\}$$

```
NF4 vs INT4 LEVEL DISTRIBUTION
═══════════════════════════════════════════════════════════════════════

Normal distribution of weights:
                        ████
                      ████████
                    ████████████
                  ████████████████
               █████████████████████
           ███████████████████████████████
  ──────────────────────────────────────────────────────
 -1.0                    0                           1.0

INT4 levels (uniform spacing):
  ●     ●     ●     ●     ●     ●     ●     ●     ●     ●
 -1.0  -0.78  -0.56  -0.33  -0.11  0.11  0.33  0.56  0.78  1.0
  Wasted precision in tails; insufficient detail near zero

NF4 levels (quantile-based):
  ●  ●  ● ● ● ●●●●●●● ● ● ●  ●  ●
 -1.0            0            1.0
  Dense near zero (where most weights are); sparse in tails
```

**Quantization: nearest-level assignment**

For each weight $w$, find the NF4 level $q_k$ minimising $|w/\sigma - q_k|$ and store the 4-bit index $k$.

**Why NF4 is better than INT4 for weights:**

- NF4 minimises the expected quantization mean squared error (MSE) for $\mathcal{N}(0, \sigma^2)$
- Theory: for a Gaussian source, quantile-based quantization approaches the Lloyd-Max optimum
- Practice: NF4 achieves 0.3–1.5 PPL increase over BF16 vs 0.5–2.0 for INT4 AWQ on equivalent models

**Usage:** QLoRA fine-tuning — base model weights stored in NF4 (frozen); LoRA adapter weights in BF16 (trainable). This enables fine-tuning a 65B model on a single 48 GB GPU.

### 6.2 Log Number System (LNS)

The log number system represents numbers by their logarithm:

$$\text{Store: } \tilde{x} = \log_2(|x|) \quad \text{plus sign bit}$$

**Multiplication becomes addition:**

$$\log_2(a \times b) = \log_2(a) + \log_2(b) = \tilde{a} + \tilde{b}$$

A single addition in log space replaces a multiplication in linear space — a major hardware simplification.

**Addition becomes complex:**

$$\log_2(a + b) = \log_2(a) + \log_2\!\left(1 + 2^{\tilde{b} - \tilde{a}}\right)$$

This requires a lookup table or approximation for the function $\log_2(1 + 2^x)$, known as the **Gaussian logarithm**. Mitchell's approximation: $\log_2(1 + x) \approx x$ for small $x$, enabling fast LNS addition.

**Properties:**

- Range: theoretically unbounded in both directions (limited only by the precision of the stored logarithm)
- Precision: **uniform in log space** — constant relative precision across all magnitudes
- Contrast with floating-point: FP also has approximately logarithmic spacing, but LNS makes this exact

**AI relevance:** LNS is rarely used in mainstream ML. Research interest exists for extremely low-bit hardware where multiplication is the dominant energy cost. Some custom accelerator designs use LNS internally.

### 6.3 Posit Number System (Unum Type III)

The posit system (Gustafson, 2017) is an alternative to IEEE 754 that claims superior accuracy per bit for certain applications:

**Structure:** variable-precision encoding with four fields:

$$[\text{sign}][\text{regime}][\text{exponent}][\text{fraction}]$$

- **Regime:** unary encoding of the exponent range — runs of 0s or 1s terminated by the opposite bit
- **Variable allocation:** bits not used by the regime are available for the exponent and fraction
- Near $\pm 1$: regime is short → more bits for fraction → **higher precision near 1.0**
- Far from $\pm 1$: regime is long → fewer fraction bits → lower precision but wider range

**Properties:**

- Only one representation of zero (no $\pm 0$)
- Only one NaN (called "Not-a-Real", or NaR)
- No gradual underflow or overflow — tapers smoothly
- Higher accuracy per bit than IEEE 754 for values near 1.0; worse for extremes

**AI evaluation:** multiple research groups evaluated posits for neural network training:

- Marginal accuracy improvement over BF16 at the same bit width (16-bit posit vs BF16)
- No significant quality advantage that justifies the hardware redesign cost
- Custom posit hardware exists (Positron chip) but no mainstream GPU supports posits

**2026 status:** academic interest only. Not deployed in any production ML system.

### 6.4 Microscaling Formats (MX — OCP Standard 2023)

Microscaling (MX) is an **industry-standard block floating-point format** published by the Open Compute Project (OCP) in 2023, co-authored by AMD, ARM, Intel, Meta, Microsoft, NVIDIA, and Qualcomm.

**Key idea:** share a single exponent (scale) across a block of values, giving each value more effective precision per bit:

```
MICROSCALING BLOCK STRUCTURE
═══════════════════════════════════════════════════════════════════════

Block of N values (e.g., N = 32):

┌─────────────────┐ ┌──┬──┬──┬──┬──┬──┬──┬──┬───────┬──┬──┐
│ Shared Exponent  │ │v₁│v₂│v₃│v₄│v₅│v₆│v₇│v₈│  ...  │v₃₁│v₃₂│
│   (8 bits)       │ │  │  │  │  │  │  │  │  │       │   │   │
└─────────────────┘ └──┴──┴──┴──┴──┴──┴──┴──┴───────┴──┴──┘
                     Each vᵢ: sign + mantissa (element format)

value_i = vᵢ × 2^(shared_exponent)
```

**MX format variants:**

| Format     | Element Bits | Element Layout | Block Size | Effective Bits/Value |
| ---------- | ------------ | -------------- | ---------- | -------------------- |
| MXFP8 E4M3 | 8            | 1s + 4e + 3m   | 32         | 8.25                 |
| MXFP8 E5M2 | 8            | 1s + 5e + 2m   | 32         | 8.25                 |
| MXFP6 E2M3 | 6            | 1s + 2e + 3m   | 32         | 6.25                 |
| MXFP6 E3M2 | 6            | 1s + 3e + 2m   | 32         | 6.25                 |
| MXFP4 E2M1 | 4            | 1s + 2e + 1m   | 32         | 4.25                 |
| MXINT8     | 8            | 1s + 7 integer | 32         | 8.25                 |

**Why MX is better than per-element floating-point:**

- The exponent storage is **amortised** over $N$ values: instead of each value needing its own 4–8 bit exponent, $N$ values share one 8-bit exponent
- Effective bits per value: element bits + (8 shared exponent bits / $N$ values) ≈ element bits + 0.25
- Result: more bits available for mantissa → better precision per total storage bit

**Hardware adoption:**

- NVIDIA B200 (Blackwell, 2025): native MXFP8 support
- AMD, Intel, Qualcomm: committed to MX hardware support
- Training: MXFP8 block scaling for both forward and backward — better than per-tensor FP8

### 6.5 Ternary Weights {−1, 0, +1}

Ternary quantization represents weights using only three values:

$$w_i \in \{-1, 0, +1\}$$

**Information content:** $\log_2(3) \approx 1.585$ bits per weight — hence the name "1.58-bit" quantization.

**The arithmetic revolution — no multiplication:**

- Standard matmul: $y_j = \sum_i w_i \cdot x_i$ — requires multiply + accumulate per element
- Ternary matmul: $y_j = \sum_{w_i=+1} x_i - \sum_{w_i=-1} x_i$ — only additions, subtractions, and skips
- If $w_i = 0$: skip entirely (natural sparsity, typically ~50% of weights)
- If $w_i = +1$: add $x_i$
- If $w_i = -1$: subtract $x_i$

```
TERNARY vs STANDARD MATMUL
═══════════════════════════════════════════════════════════════════════

Standard FP16 matmul (one output element):
  y = w₁×x₁ + w₂×x₂ + w₃×x₃ + w₄×x₄ + w₅×x₅ + w₆×x₆
  Operations: 6 multiplies + 5 adds = 11 FLOPs

Ternary matmul (w = [+1, 0, -1, +1, 0, -1]):
  y = x₁ + 0 - x₃ + x₄ + 0 - x₆
  y = (x₁ + x₄) - (x₃ + x₆)
  Operations: 3 adds + 1 subtract = 4 integer ops (NO multiplies)
```

**Storage:**

- Efficient packing: 5 ternary values per 8 bits ($\log_2(3^5) = 7.92$ bits)
- Simple packing: 2 bits per value (one wasted code point); each value = $\{00, 01, 10\}$ mapping to $\{-1, 0, +1\}$

**BitNet b1.58 (Ma et al., 2024):**

- Ternary weights trained from scratch (not quantized from a float model)
- Competitive with FP16 baselines at 3B+ parameter scale
- Key innovation: uses **absmean quantization** — scale weights by $1/\bar{|w|}$ before ternarizing
- Energy advantage: ternary addition $\approx 0.9$ pJ vs FP16 multiply $\approx 3.7$ pJ — 4× energy savings per operation

**Limitation:** ternary models must be trained from scratch with ternarization-aware methods. Post-training ternarization of float models produces unusable quality.

---

## 7. Floating-Point Arithmetic Deep Dive

Understanding how floating-point arithmetic works at the bit level explains why certain operations lose precision, why GPU matmul results differ between runs, and why specific numerical tricks (FMA, compensated summation) matter for training stability.

### 7.1 Floating-Point Addition

Adding two floating-point numbers is more complex than integer addition because the operands may have different exponents. The hardware must **align** them before adding:

**Algorithm for $x + y$ (assuming $|x| \geq |y|$):**

```
FLOATING-POINT ADDITION — STEP BY STEP
═══════════════════════════════════════════════════════════════════════

Input: x = 1.000 × 2³  and  y = 1.011 × 2⁰

Step 1: ALIGN EXPONENTS
  Shift y's significand RIGHT by (e_x - e_y) = 3 - 0 = 3 positions:
  y = 1.011 × 2⁰ → 0.001011 × 2³
                          ↑↑↑ These bits shifted right
                          GRS  (Guard, Round, Sticky bits)

Step 2: ADD ALIGNED SIGNIFICANDS
    1.000000 × 2³     (x)
  + 0.001011 × 2³     (y, aligned)
  ──────────────
    1.001011 × 2³

Step 3: NORMALISE
  Already normalised (leading 1 present) → no adjustment needed
  If needed: shift significand and adjust exponent

Step 4: ROUND
  Result has more bits than mantissa allows
  Apply rounding mode (default: RNE) to fit into mantissa width
  Round 1.001011 to 24 bits (FP32) → 1.00101100000000000000000 × 2³

RESULT: x + y = 1.001011 × 2³ = 9.375₁₀
```

**Guard, Round, and Sticky bits — critical for rounding accuracy:**

When shifting $y$'s significand during alignment, bits shift beyond the mantissa width. The hardware keeps three extra bits to improve rounding decisions:

| Bit | Name   | Purpose                                                               |
| --- | ------ | --------------------------------------------------------------------- |
| G   | Guard  | First bit beyond mantissa width                                       |
| R   | Round  | Second bit beyond mantissa width                                      |
| S   | Sticky | OR of all remaining shifted-out bits (indicates if any were non-zero) |

These three bits provide enough information for the hardware to implement all IEEE 754 rounding modes correctly. Without them, rounding accuracy would be significantly worse.

**Precision loss during alignment:** when exponents differ significantly, shifting $y$ right discards its low-order bits. If $e_x - e_y > 24$ (for FP32), then $y$'s entire significand is shifted away and $x + y = x$. This is why adding a small number to a large number does nothing — the small number is below the precision of the large number.

### 7.2 Floating-Point Multiplication

Multiplication is simpler than addition because no alignment is needed:

**Algorithm for $x \times y$:**

```
FLOATING-POINT MULTIPLICATION
═══════════════════════════════════════════════════════════════════════

Input: x = (-1)^s_x × m_x × 2^e_x
       y = (-1)^s_y × m_y × 2^e_y

Step 1: XOR SIGN BITS
  s_result = s_x XOR s_y
  (positive × positive = positive; positive × negative = negative)

Step 2: ADD EXPONENTS
  e_result = e_x + e_y - bias
  (Subtract bias once because both inputs had bias added;
   without correction, result would have double bias)

Step 3: MULTIPLY SIGNIFICANDS
  m_result = m_x × m_y
  Each significand has p bits → product has 2p bits
  (For FP32: 24 × 24 = 48-bit product)

Step 4: NORMALISE AND ROUND
  If m_result ≥ 2.0: shift right by 1; increment exponent
  Round to fit mantissa width (24 bits for FP32)
```

**Hardware cost:** the significand multiplier is the largest component — an $n \times n$ bit multiplier requires $O(n^2)$ hardware area. This is why reducing mantissa width (BF16: 7 bits vs FP32: 23 bits) dramatically reduces multiplication hardware cost and energy.

**Key advantage over addition:** no alignment step → simpler circuit → faster in hardware. This is why GPU throughput for multiply is typically the same or better than for addition.

### 7.3 Floating-Point Division and Square Root

**Division:**

- Compute significand quotient: $m_{\text{result}} = m_x / m_y$
- Subtract exponents: $e_{\text{result}} = e_x - e_y + \text{bias}$
- Iterative algorithms: **Newton-Raphson** (converges quadratically; computes $1/m_y$ then multiplies) or **SRT algorithm** (produces one quotient digit per cycle)
- Hardware: slower than multiply; separate functional unit on GPU; typically 4–20× slower

**Square root:**

$$\sqrt{x} = \sqrt{m \times 2^e} = \sqrt{m} \times 2^{e/2}$$

- Even exponent: straightforward division by 2
- Odd exponent: multiply $m$ by 2 (making it $\in [2, 4)$), then halve the adjusted exponent
- Significand square root: Newton-Raphson iteration $y_{n+1} = \frac{1}{2}(y_n + m/y_n)$
  - 2 iterations sufficient for FP32; 3 for FP64

**Fast inverse square root — the famous Quake III trick (historical interest):**

```c
float Q_rsqrt(float number) {
    long i;
    float x2, y;
    x2 = number * 0.5F;
    y  = number;
    i  = *(long *)&y;                  // Interpret float bits as integer
    i  = 0x5f3759df - (i >> 1);        // Initial approximation (magic!)
    y  = *(float *)&i;                  // Convert back to float
    y  = y * (1.5F - (x2 * y * y));    // One Newton-Raphson refinement
    return y;
}
```

The "magic number" `0x5f3759df` exploits the fact that the integer representation of a float is approximately its logarithm. Shifting right by 1 halves the logarithm (approximating square root), and subtracting from the magic constant computes the inverse. The Newton-Raphson step then refines to full FP32 precision.

**Modern relevance:** modern GPUs have dedicated hardware for `rsqrt` and `sqrt`; the Q_rsqrt trick is no longer needed for performance. But understanding it demonstrates the deep connection between integer and floating-point bit representations.

### 7.4 Fused Multiply-Add (FMA)

FMA computes $a \times b + c$ as a **single operation with a single rounding** at the end, rather than two separate operations with two roundings:

$$\text{FMA}(a, b, c) = \text{round}(a \times b + c)$$

Compare to the unfused version:

$$\text{Unfused: } \text{round}(\text{round}(a \times b) + c) \quad \text{→ two rounding errors}$$

**Why FMA matters:**

1. **More accurate:** one rounding error instead of two. For a single operation, the difference is small. But a dot product of length $n$ performs $n$ FMAs — the error savings compound.

2. **Foundation of dot products:** the inner product $\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i$ is computed as a chain of FMAs:

   ```
   acc = 0
   acc = FMA(x₁, y₁, acc)    // acc = x₁·y₁ + 0
   acc = FMA(x₂, y₂, acc)    // acc = x₂·y₂ + x₁·y₁
   acc = FMA(x₃, y₃, acc)    // acc = x₃·y₃ + (x₂·y₂ + x₁·y₁)
   ...
   ```

3. **Hardware unit:** GPU tensor cores and CPU FPUs implement FMA as a single hardware functional unit — no intermediate register write, no intermediate rounding.

4. **Kahan summation:** FMA can implement compensated summation more efficiently by exploiting the exact product computation.

**CUDA intrinsics:** `fmaf(a, b, c)` for FP32 FMA; `__fmaf_rn()` for round-to-nearest FMA; `__fmaf_rd()` for round-down FMA.

### 7.5 Dot Product Accumulation

The dot product (inner product) is the **fundamental operation** underlying all matrix multiplications in neural networks. Its numerical accuracy determines the quality of every matmul output.

**Error analysis:**

For a naive dot product $s = \sum_{i=1}^{n} x_i y_i$ computed in floating-point:

$$\hat{s} = s(1 + \delta), \quad |\delta| \leq n\varepsilon + O(\varepsilon^2)$$

where $\varepsilon$ is machine epsilon. The relative error grows **linearly with the dimension** $n$.

For BF16 ($\varepsilon = 7.8 \times 10^{-3}$) with $n = 4096$ (typical hidden dimension):

$$|\delta| \leq 4096 \times 7.8 \times 10^{-3} \approx 32$$

This is a **3,200% relative error** — completely unusable for any computation. This is why BF16 dot products **must** accumulate in FP32.

**GPU practice — mixed-precision accumulation:**

| Input Format | Accumulation Format | Output Format | Error Bound                                    |
| ------------ | ------------------- | ------------- | ---------------------------------------------- |
| INT8 × INT8  | INT32 → FP32        | BF16 or INT8  | Exact integer, then scale                      |
| BF16 × BF16  | FP32                | BF16          | $O(n \times 2^{-23}) \approx 5 \times 10^{-4}$ |
| FP8 × FP8    | FP32                | BF16          | $O(n \times 2^{-23}) \approx 5 \times 10^{-4}$ |

The key insight: even though inputs are low-precision, **FP32 accumulation** ensures the sum is computed accurately. The accumulated result is then converted back to the output format. This is why GPU matmul quality is far better than naive BF16 arithmetic suggests.

**Compensated dot product (Ogita-Rump-Oishi, 2005):**

- Uses FMA to capture the exact error from each multiplication
- Accumulates the errors separately and adds them at the end
- Result: relative error $O(\varepsilon^2)$ regardless of $n$ — quadratic improvement
- Not yet standard in GPU matmul but used in scientific computing

### 7.6 Mixed-Precision Matmul

The actual computation inside NVIDIA tensor cores for a BF16 matmul:

```
MIXED-PRECISION TENSOR CORE MATMUL (BF16)
═══════════════════════════════════════════════════════════════════════

Input:  A ∈ ℝᵐˣᵏ (BF16),  B ∈ ℝᵏˣⁿ (BF16)
Output: C ∈ ℝᵐˣⁿ (FP32 or BF16)

Hardware tile: 16×16×16 (m_tile × n_tile × k_tile)
Each warp (32 threads) computes one tile output

For each K-dimension chunk of 16:
  1. Load A tile (16×16 BF16) and B tile (16×16 BF16) into registers
  2. Compute 16×16 partial products: BF16 × BF16
  3. Accumulate partial sums in FP32 registers (one FP32 per output element)

After all K chunks:
  4. FP32 accumulated result in registers
  5. Optionally convert to BF16 for storage

Key: The multiplication is BF16, but accumulation is FP32
This is what makes mixed-precision training work!
```

**FP8 tensor core operation (H100):**

```
FP8 TENSOR CORE MATMUL
═══════════════════════════════════════════════════════════════════════

Input:  A ∈ ℝᵐˣᵏ (FP8 E4M3),  B ∈ ℝᵏˣⁿ (FP8 E4M3)
Scales: S_A (per-tensor or per-block), S_B (per-tensor or per-block)

1. Dequantize: A_f = A_fp8 × S_A,  B_f = B_fp8 × S_B
2. Multiply: BF16-equivalent element products
3. Accumulate: FP32 partial sums
4. Scale output: C = (A_f × B_f) = S_A × S_B × (A_fp8 × B_fp8)
5. Output: FP32 or BF16

Throughput: H100 FP8 = 3,958 TOPS  (4× BF16)
```

**Throughput comparison (H100 SXM):**

| Input       | Accumulation | TFLOPS/TOPS | Notes                                                          |
| ----------- | ------------ | ----------- | -------------------------------------------------------------- |
| BF16 × BF16 | FP32         | 989         | Standard for training                                          |
| FP8 × FP8   | FP32         | 3,958       | 4× BF16; note: 2× from data density + 2× from simpler hardware |
| INT8 × INT8 | INT32        | 3,958       | Same throughput as FP8                                         |

The 4× throughput gain from BF16 → FP8 is the driving force behind the industry's push toward FP8 training and inference.

---

## 8. Numerical Precision in Neural Networks

This section connects number system theory to the practical precision requirements of each component in a transformer-based LLM.

### 8.1 Forward Pass Precision Requirements

Each operation in the forward pass has different sensitivity to numerical precision:

**Input embeddings (BF16 sufficient):**

- Embedding lookup is a simple table read — no arithmetic precision concern for the lookup itself
- Small errors in embedding vectors ($< 1\%$) don't compound catastrophically because they pass through layer normalisation early
- BF16 with 7-bit mantissa provides $\sim 0.78\%$ relative precision — adequate

**Attention scores $QK^T / \sqrt{d_k}$ (BF16 matmul, FP32 softmax):**

- The matmul $QK^T$ is computed in BF16 with FP32 accumulation (§7.6) — no problem
- The division by $\sqrt{d_k}$ is a simple scale; BF16 sufficient
- **However:** the subsequent softmax requires FP32 for the exponentiation (see §8.4)

**Softmax (must use FP32 or numerical tricks):**

- $\text{softmax}(z)_i = e^{z_i} / \sum_j e^{z_j}$
- $e^{z}$ overflows BF16 for $z > 88$ (because $e^{88} \approx 1.65 \times 10^{38} \approx$ BF16 max)
- Attention logits can reach 50–100 in late training — well below 88, but early layers or unstable training can exceed this
- **Solution:** compute softmax intermediate values in FP32; or use the numerically stable version (§8.4)

**Layer normalisation (FP32 for mean/variance):**

- Mean $\mu = \frac{1}{n}\sum x_i$ and variance $\sigma^2 = \frac{1}{n}\sum (x_i - \mu)^2$: sums over $n$ values must be accurate
- With BF16 inputs: accumulate $\mu$ and $\sigma^2$ in FP32 before applying normalisation
- The normalised output $(x - \mu)/\sqrt{\sigma^2 + \epsilon}$ can be stored in BF16

**FFN activations (BF16 sufficient):**

- SwiGLU gate and up projections: standard matmul; BF16 with FP32 accumulation
- Swish activation function: smooth, bounded gradient; no overflow/underflow issues in BF16
- Down projection: same as any matmul; BF16 sufficient

### 8.2 Backward Pass Precision Requirements

The backward pass is more precision-sensitive than the forward pass because errors in gradients compound through the optimizer over millions of steps:

**Gradient magnitudes:**

- Typical gradient magnitudes: $10^{-4}$ to $10^{-7}$
- These are well within BF16 range ($\text{min} \approx 1.2 \times 10^{-38}$) but near FP16 minimum ($6.1 \times 10^{-5}$)
- This is exactly why BF16 works and FP16 fails for training without loss scaling

**Gradient accumulation (MUST use FP32):**

$$\frac{\partial L}{\partial W} = \sum_{\text{batch}} x^T \delta$$

This sum aggregates many small gradient contributions. In BF16:

- Machine epsilon $= 7.8 \times 10^{-3}$
- Any gradient contribution smaller than $0.78\%$ of the running sum is lost
- Over a batch of 1024 samples, many individual contributions are small → silently discarded
- Over millions of training steps, this precision loss causes weights to stop updating → **training stalls**

In FP32:

- Machine epsilon $= 1.19 \times 10^{-7}$
- Contributions as small as $0.0000119\%$ of the running sum are preserved
- 65,000× more precise than BF16 — sufficient for stable training

**Gradient clipping (FP32 norm, BF16 application):**

- Global gradient norm $\|\mathbf{g}\|_2 = \sqrt{\sum g_i^2}$: compute in FP32 (sum of squares)
- Apply clipping scale factor: can be done in BF16 (simple multiplication)

**Loss computation (FP32):**

- Cross-entropy loss involves $\log(\text{softmax}(z))$ — both log and exp are precision-sensitive
- Must use numerically stable log-sum-exp (§8.5) in FP32

### 8.3 Mixed Precision Training — Complete Picture

The standard mixed-precision training recipe that has been used for virtually all large-scale LLM training since 2020:

```
MIXED PRECISION TRAINING PIPELINE
═══════════════════════════════════════════════════════════════════════

INITIALISATION:
  θ_master = initialise_weights()           # FP32 (4 bytes/param)
  m = zeros_like(θ_master)                  # FP32 Adam momentum
  v = zeros_like(θ_master)                  # FP32 Adam variance

FOR EACH TRAINING STEP:

  ┌─ FORWARD PASS ──────────────────────────────────────────────┐
  │  θ_bf16 = cast(θ_master, BF16)         # FP32 → BF16       │
  │  activations = forward(x, θ_bf16)       # BF16 matmul       │
  │  loss = cross_entropy(activations, y)    # FP32 (stable)     │
  └─────────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─ BACKWARD PASS ─────────────────────────────────────────────┐
  │  grads_bf16 = backward(loss, activations) # BF16 gradients  │
  │  grads_fp32 = accumulate(grads_bf16)       # FP32 accumulate │
  │  clip_grad_norm(grads_fp32)                # FP32 norm       │
  └─────────────────────────────────────────────────────────────┘
                           │
                           ▼
  ┌─ OPTIMIZER STEP ────────────────────────────────────────────┐
  │  m = β₁·m + (1-β₁)·grads_fp32             # FP32           │
  │  v = β₂·v + (1-β₂)·grads_fp32²            # FP32           │
  │  m̂ = m / (1 - β₁ᵗ)                        # FP32           │
  │  v̂ = v / (1 - β₂ᵗ)                        # FP32           │
  │  θ_master -= η · m̂ / (√v̂ + ε)             # FP32 update   │
  └─────────────────────────────────────────────────────────────┘

MEMORY PER PARAMETER:
  θ_master (FP32):     4 bytes
  Adam m (FP32):       4 bytes
  Adam v (FP32):       4 bytes
  θ_bf16 (working):    2 bytes
  ──────────────────────────
  Total:              14 bytes per parameter

  For 70B model: 70B × 14 = 980 GB (explaining why large model
  training requires many GPUs even with mixed precision)
```

### 8.4 Numerical Stability of Softmax

The softmax function is one of the most numerically sensitive operations in a transformer:

**Naive softmax — the overflow problem:**

$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

For $z_i = 100$ in BF16: $e^{100} \approx 2.69 \times 10^{43}$ — exceeds BF16 max ($3.39 \times 10^{38}$) → **overflow to Inf**. Even in FP32: $e^{89} \approx 4.5 \times 10^{38}$ overflows. Any attention logit above ~88 causes overflow.

**Numerically stable softmax — the max-subtraction trick:**

$$\text{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \quad m = \max_j(z_j)$$

**Proof that this is mathematically equivalent:**

$$\frac{e^{z_i - m}}{\sum_j e^{z_j - m}} = \frac{e^{z_i} \cdot e^{-m}}{\sum_j e^{z_j} \cdot e^{-m}} = \frac{e^{z_i} \cdot \cancel{e^{-m}}}{\cancel{e^{-m}} \cdot \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

The factor $e^{-m}$ cancels in numerator and denominator. But numerically, the subtraction ensures that the largest exponent is $e^{z_{\max} - z_{\max}} = e^0 = 1$ — no overflow possible. All other exponents are $e^{z_i - m} \leq 1$ — also safe.

**FlashAttention's online softmax:**

- Standard softmax requires two passes: (1) compute $m = \max(z)$; (2) compute $e^{z_i - m}$ and sum
- FlashAttention maintains a running max and running sum, rescaling as new blocks of keys arrive
- This enables computing attention in a single pass through the KV cache, keeping all data in SRAM

### 8.5 Log-Sum-Exp Trick

The log-sum-exp (LSE) operation appears in cross-entropy loss, softmax computation, and log-normaliser calculation:

$$\text{LSE}(z) = \log \sum_i e^{z_i}$$

**The overflow problem:** for large $z_i$, $e^{z_i}$ overflows before the log can "undo" it.

**The solution — factor out the maximum:**

$$\text{LSE}(z) = m + \log \sum_i e^{z_i - m}, \quad m = \max_i(z_i)$$

**Proof:**

$$\log \sum_i e^{z_i} = \log \left(e^m \sum_i e^{z_i - m}\right) = \log(e^m) + \log \sum_i e^{z_i - m} = m + \log \sum_i e^{z_i - m}$$

The maximum $m$ is factored out. All remaining exponents $z_i - m \leq 0$ → no overflow. The subtracted $m$ is re-added as a simple sum — no precision loss.

**Why this matters for LLM training:**

Cross-entropy loss for next-token prediction:

$$L = -\log P(t_{\text{target}}) = -\log \text{softmax}(z)_{\text{target}} = -(z_{\text{target}} - \text{LSE}(z))$$

$$= \text{LSE}(z) - z_{\text{target}}$$

This is computed millions of times per training step (once per token in the batch). If $\text{LSE}(z)$ overflows, the loss becomes Inf → gradient is NaN → training dies.

**PyTorch:** `torch.logsumexp(z, dim)` implements this correctly. Never compute `torch.log(torch.sum(torch.exp(z)))` directly.

### 8.6 Numerical Stability of Layer Normalisation

**LayerNorm:**

$$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

where $\mu = \frac{1}{n}\sum x_i$ and $\sigma^2 = \frac{1}{n}\sum (x_i - \mu)^2$.

**Numerical concerns:**

1. **Catastrophic cancellation in $x - \mu$:** when $x_i \approx \mu$ (which is common — normalisation centers data near zero), $x_i - \mu$ loses significant bits (§3.6)

2. **Division by small $\sigma$:** if all values are nearly identical, $\sigma \approx 0$; division amplifies noise
   - The $\epsilon = 10^{-5}$ ensures we divide by at least $\sqrt{10^{-5}} \approx 0.00316$; this prevents division by zero and limits amplification
   - In BF16: $\epsilon = 10^{-5}$ is representable (BF16 min normal $\approx 1.2 \times 10^{-38}$), but the precision of $\sigma^2 + \epsilon$ when $\sigma^2 \approx 0$ is limited by BF16's 7-bit mantissa

3. **FP32 accumulation for mean and variance:** even if input $x$ is BF16, compute $\mu$ and $\sigma^2$ by accumulating in FP32 before applying the normalisation

**RMSNorm — a numerically cleaner alternative:**

$$y = \frac{x}{\sqrt{\frac{1}{n}\sum x_i^2 + \epsilon}} \cdot \gamma$$

- **No mean subtraction** → no catastrophic cancellation from $x - \mu$
- Only requires computing $\text{RMS}(x) = \sqrt{\text{mean}(x^2)}$ — a sum of positive values (no cancellation)
- Used in LLaMA, Mistral, and most 2024+ transformer architectures
- Slightly cheaper to compute (one fewer reduction) and more numerically stable

### 8.7 Gradient Vanishing and Exploding — Numerical View

**The chain rule through $L$ layers:**

$$\frac{\partial L}{\partial x_0} = \prod_{l=1}^{L} \frac{\partial x_l}{\partial x_{l-1}} = \prod_{l=1}^{L} J_l$$

where $J_l$ is the Jacobian of layer $l$. If each Jacobian has a dominant eigenvalue $\lambda$:

$$\left\|\frac{\partial L}{\partial x_0}\right\| \sim \lambda^L$$

**Exploding gradients ($\lambda > 1$):**

- $\lambda = 1.01$, $L = 200$ layers: $1.01^{200} \approx 7.32$ — moderate growth
- $\lambda = 1.1$, $L = 200$ layers: $1.1^{200} \approx 1.9 \times 10^{8}$ — rapid growth
- In FP32: gradients exceed $3.4 \times 10^{38}$ → Inf → training crashes
- In BF16: same range as FP32, but precision errors amplified by large gradients → unstable updates

**Vanishing gradients ($\lambda < 1$):**

- $\lambda = 0.99$, $L = 200$: $0.99^{200} \approx 0.134$ — still meaningful
- $\lambda = 0.9$, $L = 200$: $0.9^{200} \approx 7 \times 10^{-10}$ — tiny
- In FP16: gradients below $6.1 \times 10^{-5}$ underflow to zero → **silent death** (no error, no NaN, just zero gradients)
- In BF16: gradients below $1.2 \times 10^{-38}$ underflow — this almost never happens in practice

**Residual connections — the numerical fix:**

$$x_l = x_{l-1} + f_l(x_{l-1})$$

$$\frac{\partial x_l}{\partial x_{l-1}} = I + \frac{\partial f_l}{\partial x_{l-1}}$$

The Jacobian is $I + J_f$, whose eigenvalues are $1 + \lambda_f$. Even if $\lambda_f$ is small or negative, the eigenvalue stays near 1 — **preventing both explosion and vanishing**.

**Gradient clipping — the safety net:**

- Clip the global gradient norm: $\hat{g} = g \cdot \min\!\left(1, \frac{C}{\|g\|}\right)$
- Prevents Inf from exceeding FP32 range
- Standard practice in LLM training: $C = 1.0$ typical

---

## 9. Quantization Mathematics

Quantization is the mathematical process of mapping values from a high-precision format (e.g., FP32, BF16) to a low-precision format (e.g., INT8, INT4, NF4). This section develops the mathematical foundations rigorously.

### 9.1 Uniform Quantization Formulas — Derivation and Intuition

**The fundamental problem:** map a continuous (or high-precision) value $x \in [x_{\min}, x_{\max}]$ to an integer $x_q \in \{0, 1, \ldots, 2^b - 1\}$ (unsigned) or $x_q \in \{-2^{b-1}, \ldots, 2^{b-1}-1\}$ (signed), where $b$ is the bit width.

**Step 1 — Define the scale factor $s$:**

$$s = \frac{x_{\max} - x_{\min}}{2^b - 1}$$

This is the "width" of one quantization bin. For symmetric quantization around zero:

$$s = \frac{2 \cdot \max(|x_{\min}|, |x_{\max}|)}{2^b - 1}$$

**Step 2 — Define the zero-point $z$ (asymmetric only):**

$$z = \text{round}\!\left(-\frac{x_{\min}}{s}\right)$$

The zero-point ensures that the floating-point value 0.0 maps exactly to an integer. This is critical because:

- Padding zeros in convolutions must remain exactly zero after quantization
- Skip connections that add 0 must not introduce bias

For symmetric quantization: $z = 0$ (or $z = 2^{b-1}$ for unsigned).

**Step 3 — Quantize (FP → INT):**

$$x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s} + z\right),\; 0,\; 2^b - 1\right)$$

**Step 4 — Dequantize (INT → FP approximation):**

$$\hat{x} = s \cdot (x_q - z)$$

**Quantization error:**

$$\epsilon_q = x - \hat{x} = x - s \cdot (\text{round}(x/s + z) - z)$$

For uniform quantization with round-to-nearest, the error is bounded:

$$|\epsilon_q| \leq \frac{s}{2} = \frac{x_{\max} - x_{\min}}{2(2^b - 1)}$$

**Worked example — INT8 symmetric quantization of a weight tensor:**

```
Given:  weights = [-0.45, 0.12, -0.03, 0.67, -0.89, 0.34]
Bit width: b = 8 (signed: range [-128, 127])

Step 1: α = max(|-0.89|, |0.67|) = 0.89
        s = 2α / (2^8 - 1) = 1.78 / 255 = 0.006980

Step 2: z = 0 (symmetric)

Step 3: Quantize each weight:
  -0.45 → round(-0.45 / 0.006980) = round(-64.47) = -64
   0.12 → round(0.12 / 0.006980)  = round(17.19)  =  17
  -0.03 → round(-0.03 / 0.006980) = round(-4.30)  =  -4
   0.67 → round(0.67 / 0.006980)  = round(95.99)  =  96
  -0.89 → round(-0.89 / 0.006980) = round(-127.51)= -128  ← clips to -128
   0.34 → round(0.34 / 0.006980)  = round(48.71)  =  49

Step 4: Dequantize:
  -64 → 0.006980 × (-64) = -0.44672   (error: 0.00328)
   17 → 0.006980 × 17    =  0.11866   (error: 0.00134)
   -4 → 0.006980 × (-4)  = -0.02792   (error: 0.00208)
   96 → 0.006980 × 96    =  0.67008   (error: 0.00008)
 -128 → 0.006980 × (-128)= -0.89344   (error: 0.00344)  ← clipping error
   49 → 0.006980 × 49    =  0.34202   (error: 0.00202)

Maximum error: 0.00344 ≈ s/2 = 0.00349  ✓
```

### 9.2 Signal-to-Quantization-Noise Ratio (SQNR)

SQNR measures the quality of quantization — how much signal is preserved relative to the quantization noise introduced.

**Definition:**

$$\text{SQNR} = \frac{\mathbb{E}[x^2]}{\mathbb{E}[(x - \hat{x})^2]} = \frac{\text{signal power}}{\text{quantization noise power}}$$

In dB:

$$\text{SQNR}_{\text{dB}} = 10 \log_{10} \text{SQNR}$$

**For uniform quantization of a uniformly distributed signal:**

The quantisation error is approximately uniformly distributed on $[-s/2, s/2]$ with variance:

$$\sigma_{\epsilon}^2 = \frac{s^2}{12}$$

For a signal uniformly distributed on $[-\alpha, \alpha]$ quantized to $b$ bits:

$$s = \frac{2\alpha}{2^b - 1} \approx \frac{2\alpha}{2^b}$$

$$\sigma_x^2 = \frac{\alpha^2}{3}$$

$$\text{SQNR} = \frac{\sigma_x^2}{\sigma_\epsilon^2} = \frac{\alpha^2/3}{s^2/12} = \frac{4\alpha^2}{s^2} = \frac{4\alpha^2}{(2\alpha/2^b)^2} = 2^{2b}$$

$$\text{SQNR}_{\text{dB}} = 10 \log_{10}(2^{2b}) = 20b \log_{10}(2) \approx 6.02b$$

> **The 6 dB per bit rule:** each additional bit of precision adds approximately 6 dB of SQNR.

| Bit Width | SQNR (dB) | Noise Ratio    | Typical Use               |
| --------- | --------- | -------------- | ------------------------- |
| 2         | 12.0      | 6.25%          | Binary/ternary weights    |
| 4         | 24.1      | 0.39%          | Inference (GPTQ, AWQ)     |
| 8         | 48.2      | 0.0015%        | PTQ inference standard    |
| 16        | 96.3      | $< 10^{-7}\%$  | Training weights          |
| 32        | 192.7     | $< 10^{-17}\%$ | Master weights, optimizer |

**For non-uniform distributions (which neural network weights follow):**

Neural network weights typically follow a bell-shaped distribution (roughly Gaussian with heavy tails after training). This means:

- Most values cluster near zero
- A few outlier values stretch the range
- Uniform quantization "wastes" most levels on the sparse tails

This is why calibration (choosing $x_{\min}$, $x_{\max}$ to clip outliers) dramatically improves SQNR, and why non-uniform formats like NF4 outperform uniform INT4.

### 9.3 Block Floating-Point and Group Quantization

Instead of a single $(s, z)$ pair for an entire tensor, **group quantization** assigns one $(s, z)$ pair per block of $G$ elements:

```
BLOCK/GROUP QUANTIZATION
═════════════════════════

Full tensor T with N elements, block size G:

  Block 0         Block 1         Block 2
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ w₀...w_{G-1}│ │ w_G...w_{2G-1}│ │ ...        │
  │ s₀, z₀      │ │ s₁, z₁      │ │ s₂, z₂      │
  └──────────┘   └──────────┘   └──────────┘

Each block has its own scale s_k and zero-point z_k:

  s_k = (max(block_k) - min(block_k)) / (2^b - 1)
  z_k = round(-min(block_k) / s_k)

Memory overhead:
  Without grouping:  N×b bits + 1×(32+32) bits for (s,z)
  With grouping:     N×b bits + (N/G)×(32+32) bits for (s_k,z_k)
  Overhead ratio:    64/G bits per element ≈ 64/G extra bits per parameter
```

**Common group sizes and their overhead (for INT4 — 4 bits per weight):**

| Group Size $G$ | Overhead per param    | Effective bits/param | Used In              |
| -------------- | --------------------- | -------------------- | -------------------- |
| 32             | +1.0 bit (FP16 scale) | 4.5                  | QLoRA (NF4)          |
| 64             | +0.5 bit              | 4.25                 | Common middle ground |
| 128            | +0.25 bit             | 4.125                | GPTQ default         |
| Per-channel    | +negligible           | ~4.0                 | Weight-only quant    |
| Per-tensor     | +negligible           | ~4.0                 | Least accurate       |

**Double quantization (QLoRA innovation):**

The scales $s_k$ themselves are FP16 (2 bytes each). With group size $G = 64$, this adds $2/64 = 0.03125$ bytes = 0.25 bits per parameter. QLoRA further quantizes these FP16 scales to FP8, reducing the overhead to 0.125 bits per parameter:

$$\text{Effective bits per param (QLoRA)} = 4 + \frac{16}{64} \cdot \frac{8}{16} = 4 + 0.125 = 4.125$$

### 9.4 Optimal Quantization Levels — Lloyd-Max Algorithm

Uniform quantization is suboptimal when the input distribution is non-uniform (which it always is for neural network weights). **Lloyd-Max quantization** finds the $2^b$ reproduction levels $\{r_i\}$ and decision boundaries $\{d_i\}$ that minimise $\text{MSE} = \mathbb{E}[(x - \hat{x})^2]$ for a given distribution $f(x)$.

**The optimality conditions:**

1. **Nearest-neighbour condition** — each value maps to its closest reproduction level:
   $$d_i = \frac{r_i + r_{i+1}}{2}$$

2. **Centroid condition** — each reproduction level is the conditional mean of values in its bin:
   $$r_i = \frac{\int_{d_{i-1}}^{d_i} x \, f(x) \, dx}{\int_{d_{i-1}}^{d_i} f(x) \, dx} = \mathbb{E}[x \mid d_{i-1} \leq x < d_i]$$

**Iterative algorithm:**

1. Initialise $\{r_i\}$ (e.g., uniformly spaced or k-means++ style)
2. Compute $\{d_i\}$ as midpoints of adjacent $\{r_i\}$
3. Recompute $\{r_i\}$ as centroids of bins defined by $\{d_i\}$
4. Repeat until convergence (MSE decreases monotonically)

**Connection to NF4:** the 16 NF4 levels (§6.1) are the Lloyd-Max optimal quantization levels for a standard normal distribution $\mathcal{N}(0, 1)$ with $b = 4$ bits. The weight tensor is first normalised to zero mean and unit variance, making the normal assumption a good fit.

### 9.5 Hadamard Transform for Quantization — QuIP and QuaRot

**The outlier problem:**
Transformer activations and certain weight columns contain **outlier channels** — values 10–100× larger than the rest. These stretch the quantization range, wasting precision for all other values.

**The Hadamard solution:**
The Hadamard matrix $H_n$ is an $n \times n$ orthogonal matrix with entries $\pm 1/\sqrt{n}$:

$$H_1 = [1], \quad H_2 = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}, \quad H_n = \frac{1}{\sqrt{2}}\begin{bmatrix} H_{n/2} & H_{n/2} \\ H_{n/2} & -H_{n/2} \end{bmatrix}$$

**Key property:** $H_n H_n^T = I$ (orthogonal) — applying $H_n$ and $H_n^T$ in sequence is an identity.

**How it helps quantization:**

For a weight matrix $W$ and input $x$:

$$Wx = (WH_n^T)(H_n x)$$

Let $W' = WH_n^T$ and $x' = H_n x$. Then:

- $W'$ and $x'$ have much more uniform magnitudes (Hadamard "spreads" outliers across all dimensions)
- Quantizing $W'$ and $x'$ yields much lower error than quantizing $W$ and $x$ directly

**This is free at inference** (for weights):

- Pre-compute $W' = WH_n^T$ once during quantization
- Apply $H_n$ to activations at runtime (cost: $O(n \log n)$ via Fast Walsh-Hadamard Transform, much cheaper than the $O(n^2)$ matmul it enables)

**QuIP (Chee et al., 2023):** uses random orthogonal matrices (including Hadamard) to "incohere" weight matrices before quantization, achieving near-optimal 2-bit quantization.

**QuaRot (Ashkboos et al., 2024):** applies Hadamard rotations to both weights and activations, enabling 4-bit quantization of all linear layers including KV cache with minimal accuracy loss.

### 9.6 Quantization Error Propagation Through Layers

A critical question: if each layer introduces quantization error $\epsilon_l$, how does the total error grow through $L$ layers?

**Single layer error model:**

$$\hat{y}_l = W_l^q x_l = (W_l + \Delta W_l) x_l = W_l x_l + \Delta W_l x_l = y_l + \epsilon_l$$

where $\Delta W_l = W_l^q - W_l$ is the per-layer weight perturbation and $\epsilon_l = \Delta W_l x_l$.

**Through $L$ layers (without activations, for intuition):**

$$\hat{y}_L = \prod_{l=1}^{L} (W_l + \Delta W_l) \cdot x_0$$

Expanding the product (first-order approximation, dropping cross-terms of $\Delta W$):

$$\hat{y}_L \approx \left(\prod_{l=1}^{L} W_l\right) x_0 + \sum_{l=1}^{L} \left(\prod_{k=l+1}^{L} W_k\right) \Delta W_l \left(\prod_{k=1}^{l-1} W_k\right) x_0$$

The total error is a sum of $L$ terms, each amplified by the products of weight matrices of the layers above and below.

**Practical implications:**

| Layer Position           | Amplification                            | Recommendation                            |
| ------------------------ | ---------------------------------------- | ----------------------------------------- |
| First layer (near input) | Error amplified by all subsequent layers | Use higher precision or skip quantization |
| Middle layers            | Moderate amplification                   | Standard quantization (INT4/INT8) is fine |
| Last layer (near output) | Error directly affects logits            | Use higher precision or skip quantization |
| Attention layers         | Errors in QK^T amplified by softmax      | More sensitive; use INT8 minimum          |

**Empirical observation from GPTQ, AWQ, and similar methods:**

- Quantizing all layers to INT4: moderate perplexity increase (~0.5–1.0 on WikiText)
- Keeping first and last layers in FP16: recovers most of the loss (~0.1–0.3 increase)
- This is consistent with the error amplification analysis above

---

## 10. Number Systems for Specific AI Operations

Different operations within a transformer have radically different precision requirements. This section provides a per-operation analysis.

### 10.1 Embedding Tables

**The operation:** look up a learned vector $\mathbf{e}_t \in \mathbb{R}^d$ for each token $t$ in the vocabulary.

**Precision analysis:**

- Embedding lookup is a **table read** — no arithmetic; precision only matters for storage
- Embedding vectors are typically small in magnitude ($|e_{t,i}| < 1$ after weight decay)
- The embedding is immediately fed into layer normalisation, which re-centres and re-scales

**Storage formats:**

| Format | Memory for V=128K, d=4096 | Quality Impact                 |
| ------ | ------------------------- | ------------------------------ |
| FP32   | 2.0 GB                    | Baseline                       |
| BF16   | 1.0 GB                    | No measurable loss             |
| INT8   | 0.5 GB + scales           | < 0.1% perplexity increase     |
| INT4   | 0.25 GB + scales          | Slight degradation; acceptable |

**Autoregressive inference:** embeddings are accessed one token at a time — memory-bandwidth-bound. Smaller formats directly reduce first-token latency.

**Training:** always FP32/BF16 (embeddings need to receive precise gradient updates since the vocabulary is large and each token's embedding updates are sparse).

### 10.2 Attention Score Computation

**The operation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**Precision requirements by component:**

**$QK^T$ matmul:**

- BF16 × BF16 with FP32 accumulation is standard and sufficient
- FP8 with FP32 accumulation is emerging (H100 tensor cores support this natively)
- INT8 × INT8 with INT32 accumulation works for inference with calibrated quantization

**$\div \sqrt{d_k}$ scaling:**

- A single multiplication by $1/\sqrt{d_k}$ — precision irrelevant
- Often fused into the Q or K projection ($W_Q \leftarrow W_Q / \sqrt{d_k}$) to avoid a separate operation

**Softmax (see §8.4):**

- Must use FP32 intermediate values (or at minimum the numerically stable BF16 version)
- The exponentiation and normalisation are the precision bottleneck

**Attention × V matmul:**

- Same precision profile as $QK^T$: BF16 with FP32 accumulation
- The attention weights after softmax are in $[0, 1]$ — well-suited for any format

**Multi-query / grouped-query attention (MQA/GQA):**

- Fewer K, V heads (typically 1 or 8 groups vs 32-128 query heads)
- Doesn't change precision requirements, but dramatically reduces KV cache memory (see §10.3)

### 10.3 KV Cache Precision

The KV cache stores past key and value vectors for autoregressive generation. For long-context models, it dominates inference memory:

**Memory calculation:**

$$\text{KV cache} = 2 \times L \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{seq\_len} \times \text{batch} \times \text{bytes\_per\_element}$$

**Example — LLaMA 3 70B with 128K context:**

- $L = 80$ layers, $n_{\text{kv\_heads}} = 8$ (GQA), $d_{\text{head}} = 128$
- In BF16: $2 \times 80 \times 8 \times 128 \times 131072 \times 2 = 34.4\text{ GB}$ per request
- In INT8: $17.2\text{ GB}$ per request
- In INT4: $8.6\text{ GB}$ per request

**KV cache quantization is uniquely challenging:**

- Keys and values are computed on-the-fly during generation; you can't calibrate offline
- Outlier channels in keys cause accuracy degradation under naive quantization
- Per-channel quantization (different scale per head dimension) works well because outlier channels are consistent across tokens

**Effective approaches:**

- **Per-channel INT8:** scale per head dimension; negligible accuracy loss for most models
- **Per-token INT4 with group size 128:** achievable with careful calibration; small perplexity increase
- **KIVI (Liu et al., 2024):** Key cache in INT2 (per-channel), Value cache in INT2 (per-token), with residual FP16 for recent tokens — 8× compression with minimal loss
- **SqueezeLLM / GEAR:** mixed-precision KV cache with outlier tokens kept in higher precision

### 10.4 Feed-Forward Network (FFN) Precision

The FFN in modern transformers (SwiGLU variant):

$$\text{FFN}(x) = (\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$$

**Precision analysis:**

| Component             | Operation           | Format    | Rationale                           |
| --------------------- | ------------------- | --------- | ----------------------------------- |
| Gate projection       | $xW_{\text{gate}}$  | BF16/INT8 | Standard matmul                     |
| Up projection         | $xW_{\text{up}}$    | BF16/INT8 | Standard matmul                     |
| Swish activation      | $x \cdot \sigma(x)$ | BF16      | Smooth function; no overflow risk   |
| Element-wise multiply | $\odot$             | BF16      | Simple multiplication               |
| Down projection       | $W_{\text{down}}$   | BF16/INT8 | Standard matmul, but more sensitive |

**The SwiGLU precision advantage over ReLU:**

- ReLU creates exact zeros, which slightly help quantization (zero is perfectly representable)
- SwiGLU produces smooth, non-zero activations — slightly harder to quantize but better model quality
- In practice, the difference in quantization difficulty is negligible

**FFN weight quantization sensitivity:**

- $W_{\text{gate}}$ and $W_{\text{up}}$: moderately sensitive (they determine what information passes through)
- $W_{\text{down}}$: most sensitive (final projection directly affects residual stream and all subsequent layers)
- **Practical strategy:** quantize $W_{\text{gate}}$, $W_{\text{up}}$ to INT4; keep $W_{\text{down}}$ in INT8 or higher

### 10.5 Optimizer State Precision

The Adam optimizer maintains two state variables per parameter:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment — momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment — adaptive learning rate)}$$

**Why optimizer states are precision-critical:**

1. **Exponential moving average requires cumulative precision:**
   - With $\beta_2 = 0.999$, the effective window is $\sim 1000$ steps
   - Each step contributes $(1 - \beta_2) = 0.001$ of its gradient to the running average
   - In BF16: contributions smaller than $7.8 \times 10^{-3}$ relative to the running sum are lost
   - Since $0.001 < 7.8 \times 10^{-3}$, individual gradient contributions are frequently lost → **optimizer blindness**
   - In FP32: contributions as small as $1.19 \times 10^{-7}$ relative to running sum are preserved → 65,000× more room

2. **$v_t$ stores squared gradients:**
   - Typical gradient: $10^{-4}$ → squared: $10^{-8}$
   - These tiny values need precise accumulation over thousands of steps
   - BF16 would round $v_t$ values to zero for small gradients → divide-by-zero or extremely large updates

**Memory-efficient optimizer approaches:**

| Approach              | Memory/param     | Quality                       | Used In                     |
| --------------------- | ---------------- | ----------------------------- | --------------------------- |
| Adam FP32             | 12 bytes (θ+m+v) | Baseline                      | Standard training           |
| Adam BF16 states      | 6 bytes          | **Fails** — training diverges | Don't use                   |
| 8-bit Adam (Dettmers) | 4 bytes          | Minimal loss                  | bitsandbytes                |
| Adafactor             | 4-8 bytes        | Slight loss for some tasks    | T5, PaLM                    |
| LOMO                  | ~0 bytes         | Limited quality               | Fine-tuning only            |
| GaLore                | ~4 bytes         | Near-baseline                 | Memory-constrained training |

**8-bit Adam (bitsandbytes) — how it works:**

- Stores $m$ and $v$ in INT8 with dynamic exponent (block-wise FP8 effectively)
- Maintains a per-block scale factor in FP32
- Dequantizes to FP32 for the update step, then re-quantizes
- Round-trip error is small enough not to affect training for models up to 175B parameters

### 10.6 Gradient Communication Precision

In distributed training, gradients are communicated between GPUs via all-reduce. For a 70B model, each gradient sync transmits $\sim 70\text{B} \times 2 = 140\text{ GB}$ in BF16.

**Gradient compression techniques:**

| Method          | Format                 | Compression | Quality                |
| --------------- | ---------------------- | ----------- | ---------------------- |
| Full BF16       | BF16                   | 1×          | Baseline               |
| FP8 all-reduce  | FP8                    | 2×          | < 0.1% loss            |
| INT8 all-reduce | INT8                   | 2×          | < 0.1% loss            |
| 1-bit Adam/LAMB | 1-bit + error feedback | 16-32×      | Slight loss, converges |
| TopK + INT8     | Sparse INT8            | 10-100×     | Depends on sparsity    |

**Error feedback mechanism (for aggressive compression):**

1. Compress gradient: $\tilde{g} = \text{compress}(g)$
2. Compute error: $e = g - \tilde{g}$
3. Accumulate error in FP32 buffer: $e_{\text{acc}} += e$
4. Next step, add accumulated error: $g' = g + e_{\text{acc}}$, then compress $g'$
5. The accumulated error ensures that over many steps, the true gradient is communicated on average

**DeepSpeed ZeRO and precision:**

- ZeRO-1/2/3 partitions optimizer states/gradients/parameters across GPUs
- Reduces per-GPU memory but increases communication volume
- Communication precision becomes even more important at scale
- ZeRO++ uses INT8 for all-to-all communication with negligible quality loss

---

## 11. Hardware Implementation of Number Systems

Understanding hardware constraints explains why certain number formats exist and dictates the achievable performance for each format.

### 11.1 Integer ALUs — Simple and Fast

An integer arithmetic logic unit (ALU) for $b$-bit operations:

**Gate count and area:**

- $b$-bit adder: $\sim 5b$ gates (carry-lookahead) → completes in $O(\log b)$ gate delays
- $b$-bit multiplier: $\sim b^2$ gates (Wallace tree) → completes in $O(\log^2 b)$ gate delays
- An INT8 multiplier uses $\sim 64$ gates; an INT32 multiplier uses $\sim 1024$ gates — **16× more silicon**

**Relative cost (approximate, normalised to INT8 multiply):**

| Operation      | Relative Area | Relative Energy | Relative Latency |
| -------------- | ------------- | --------------- | ---------------- |
| INT8 multiply  | 1×            | 1×              | 1×               |
| INT16 multiply | 4×            | 4×              | 1.3×             |
| INT32 multiply | 16×           | 16×             | 1.6×             |
| INT4 multiply  | 0.25×         | 0.25×           | 0.8×             |
| INT2 multiply  | 0.0625×       | 0.0625×         | 0.6×             |
| INT1 XNOR      | 0.01×         | 0.01×           | 0.2×             |

This is why 1-bit networks (BitNet) are so attractive for edge deployment: the multiply-accumulate (MAC) unit — the most replicated component — becomes nearly free.

### 11.2 Floating-Point Units — Complex and Power-Hungry

An FP multiply involves:

1. XOR the sign bits (1 gate)
2. Add the exponents (integer adder)
3. Multiply the mantissas (integer multiplier of mantissa width)
4. Normalise and round (shift + round logic)

The mantissa multiplier dominates cost. Since mantissa width determines silicon area:

| Format   | Mantissa bits | Multiplier gates | Relative to FP32     |
| -------- | ------------- | ---------------- | -------------------- |
| FP32     | 23+1=24       | ~576             | 1×                   |
| BF16     | 7+1=8         | ~64              | 0.11× (9× cheaper)   |
| FP16     | 10+1=11       | ~121             | 0.21× (5× cheaper)   |
| TF32     | 10+1=11       | ~121             | 0.21×                |
| FP8 E4M3 | 3+1=4         | ~16              | 0.028× (36× cheaper) |
| FP8 E5M2 | 2+1=3         | ~9               | 0.016× (64× cheaper) |

**Key insight:** BF16 is only slightly more expensive than FP8 in multiplier area, but the 9× savings over FP32 is why BF16 replaced FP32 as the default training format. The jump from BF16 to FP8 saves another 4×.

### 11.3 Tensor Cores — Systolic Arrays for AI

Tensor cores (NVIDIA terminology; Google calls them "matrix multiply units" in TPUs) are specialised hardware that compute small matrix multiplies in a single clock cycle.

**Architecture of a tensor core:**

```
TENSOR CORE (NVIDIA H100 — one core)
════════════════════════════════════

  Computes: D = A × B + C

  A: 16×16 matrix (FP16/BF16/FP8/INT8)    input
  B: 16×16 matrix (FP16/BF16/FP8/INT8)    input
  C: 16×16 matrix (FP32/FP16)              accumulator (read)
  D: 16×16 matrix (FP32/FP16)              accumulator (write)

  ┌────────────────────────────────────────┐
  │  16×16 array of fused multiply-add     │
  │  units, pipelined as a systolic array  │
  │                                        │
  │  For FP16: each unit has an 11-bit     │
  │  multiplier + FP32 adder              │
  │                                        │
  │  For INT8: each unit has an 8-bit      │
  │  multiplier + INT32 adder             │
  │                                        │
  │  For FP8: each unit has a 4-bit        │
  │  multiplier + FP32 adder              │
  └────────────────────────────────────────┘
```

**Number of tensor cores per GPU generation:**

| GPU         | Tensor Cores | Formats Supported                  | Peak TOPS (INT8) |
| ----------- | ------------ | ---------------------------------- | ---------------- |
| V100 (2017) | 640          | FP16                               | 130              |
| A100 (2020) | 432          | FP16, BF16, TF32, INT8, INT4, FP64 | 624              |
| H100 (2022) | 528          | FP16, BF16, TF32, FP8, INT8        | 1,979            |
| B200 (2024) | 640+         | FP16, BF16, TF32, FP8, FP4, INT8   | 4,500+           |

Each new generation adds support for lower-precision formats, enabling higher throughput without proportionally increasing power or die area.

### 11.4 Memory Bandwidth — The True Bottleneck

For large language model inference, the bottleneck is almost never compute — it's **memory bandwidth**. Loading model weights from HBM (High-Bandwidth Memory) to the compute units limits throughput.

**The arithmetic intensity argument:**

For a single token in autoregressive generation:

- **Compute:** one matrix-vector multiply per layer: $2 \times d^2$ FLOPs (for a square weight matrix)
- **Memory:** load weight matrix: $d^2 \times \text{bytes\_per\_weight}$
- **Arithmetic intensity** $= \frac{\text{FLOPs}}{\text{bytes}} = \frac{2d^2}{d^2 \times B} = \frac{2}{B}$

For different formats:

| Format | Bytes $B$ | Arithmetic Intensity | Bottleneck                         |
| ------ | --------- | -------------------- | ---------------------------------- |
| FP32   | 4         | 0.5 FLOP/byte        | Memory (severely)                  |
| BF16   | 2         | 1.0 FLOP/byte        | Memory                             |
| INT8   | 1         | 2.0 FLOP/byte        | Memory (still)                     |
| INT4   | 0.5       | 4.0 FLOP/byte        | Memory or compute (depends on GPU) |

**H100 SXM5 specs:**

- HBM3 bandwidth: 3.35 TB/s
- FP8 tensor core compute: 1,979 TFLOPS

To be compute-bound: need arithmetic intensity $> 1979 / 3350 \approx 0.59$ FLOP/byte. Even INT8 (2 FLOP/byte) exceeds this threshold, meaning single-token inference is **compute-bound** at INT8 on H100 for sufficiently small batch sizes. But at batch size 1, the overhead of loading the full weight matrix dominates.

**Practical implications:**

- Reducing model size from BF16 → INT4 doesn't just halve memory — it halves the time to load weights, almost doubling inference speed
- This is why quantization provides near-linear speedups for inference — the improvement comes from reduced memory traffic, not faster arithmetic

### 11.5 Energy Cost per Operation

Energy consumption is a critical constraint for data centre deployment and edge AI:

**Energy per arithmetic operation (approximate, 7nm process):**

| Operation                 | Energy (pJ) | Relative |
| ------------------------- | ----------- | -------- |
| INT8 MAC                  | 0.2         | 1×       |
| INT16 MAC                 | 0.9         | 4.5×     |
| FP16 FMA                  | 1.0         | 5×       |
| BF16 FMA                  | 0.8         | 4×       |
| FP32 FMA                  | 3.7         | 18.5×    |
| FP8 FMA                   | 0.4         | 2×       |
| INT4 MAC                  | 0.05        | 0.25×    |
| 1-bit XNOR+popcount       | 0.02        | 0.1×     |
| HBM3 DRAM read (64 bytes) | 12.5        | 62.5×    |

**Critical observation:** DRAM access costs 60× more energy than an INT8 MAC. For large models:

- **Most energy is spent moving data, not computing**
- Smaller number formats reduce both memory traffic and compute energy — a double benefit
- This energy analysis is the fundamental driver behind the industry's push toward lower precision

**Energy for one forward pass of a 70B model (estimated):**

| Format | Compute Energy | Memory Energy | Total  | Relative |
| ------ | -------------- | ------------- | ------ | -------- |
| FP32   | ~80 J          | ~150 J        | ~230 J | 1×       |
| BF16   | ~17 J          | ~75 J         | ~92 J  | 0.40×    |
| INT8   | ~4 J           | ~38 J         | ~42 J  | 0.18×    |
| INT4   | ~1 J           | ~19 J         | ~20 J  | 0.087×   |

INT4 inference uses approximately **11× less energy** than FP32 — enabling deployment at 1/11th the power budget.

### 11.6 Format Conversion Hardware

GPUs include dedicated hardware for converting between number formats:

**Conversions that are "free" (no precision loss, handled by wiring):**

- FP32 → FP64: zero-extend mantissa, adjust exponent bias
- BF16 → FP32: zero-extend 16 bits of mantissa, copy exponent and sign
- INT8 → INT32: sign-extend 24 bits

**Conversions that require rounding (1-cycle latency on modern GPUs):**

- FP32 → BF16: truncate 16 mantissa bits, apply rounding (§3.5)
- FP32 → FP16: truncate + check for overflow (may produce Inf/NaN)
- FP32 → FP8: significant truncation + overflow check + rounding
- FP32 → INT8: multiply by scale, round, clamp

**Key conversion in mixed-precision training:**

```
FP32 master weight → BF16 working copy:
  Simply drop the lower 16 mantissa bits and round
  This truncation introduces up to 0.39% error per weight

  But the FP32 master copy is never lost — it receives the
  precise gradient update and only the BF16 working copy
  is re-derived fresh each forward pass
```

---

## 12. Precision and the Training Stability Connection

Training instability in large language models is intimately connected to number format limitations. This section explores the mechanisms by which precision failures manifest as training failures.

### 12.1 Loss Landscape and Precision

The loss landscape of a large neural network is a high-dimensional surface $L(\theta)$ where $\theta \in \mathbb{R}^N$ ($N \sim 10^{10}$ for a 70B model).

**Curvature and precision requirements:**

The second-order Taylor expansion around the current point $\theta$ gives:

$$L(\theta + \delta) \approx L(\theta) + \nabla L^T \delta + \frac{1}{2} \delta^T H \delta$$

where $H$ is the Hessian matrix. The curvature eigenvalues $\{\lambda_i\}$ of $H$ determine precision requirements:

- **High curvature direction** ($\lambda_i \gg 1$): the loss changes rapidly; even small perturbations from quantization cause large loss changes → **needs high precision**
- **Low curvature direction** ($\lambda_i \approx 0$): the loss is flat; quantization noise barely affects loss → **can tolerate low precision**

**Implications for mixed precision:**

- Most directions in parameter space are low-curvature (the loss landscape is approximately flat in most dimensions)
- Only a small fraction of directions are "sharp" — these correspond to critical features
- This is why BF16 training works: the quantization noise ($\sim 0.4\%$ per weight) is smaller than the gradient step size in most directions, and the few sharp directions are protected by FP32 master weights

### 12.2 Precision Cliffs — When Training Suddenly Diverges

A **precision cliff** occurs when training appears stable for many steps, then suddenly produces NaN or Inf losses. The mechanism:

**Phase 1 — Slow drift (invisible):**

- Gradient accumulation in BF16 silently drops small gradient contributions (§8.2)
- Certain weight matrices slowly drift from their optimal values
- Loss appears stable because the drift is small compared to normal training noise

**Phase 2 — Amplification:**

- Drifted weights cause slightly larger activations in deep layers
- Attention logits grow (§12.5), pushing softmax inputs toward overflow
- Gradients start hitting gradient clipping more frequently
- The model is now operating at the edge of numerical stability

**Phase 3 — Collapse:**

- A single unlucky batch pushes attention logits past BF16/FP32 overflow → Inf
- Inf propagates through softmax → NaN
- NaN gradients update all weights → all weights become NaN → **training dies**

This typically happens after 50K–200K training steps, which is why short training runs may appear fine in low precision while long runs fail.

**Diagnostic signals:**

- Gradient norm spikes (> 10× baseline) increasing in frequency
- Learning rate warmup completing without issue, then instability at decay phase
- Loss spikes that don't recover to baseline

### 12.3 Stochastic Rounding — A Precision Amplifier

When rounding $x$ to the nearest representable value in format $F$:

- **Round-to-nearest-even (RNE):** always rounds to the same value → systematic bias when many values round in the same direction
- **Stochastic rounding:** rounds up with probability $p = (x - \lfloor x \rfloor_F) / ({\lceil x \rceil_F - \lfloor x \rfloor_F})$, otherwise rounds down

**Mathematical property of stochastic rounding:**

$$\mathbb{E}[\text{SR}(x)] = x$$

Stochastic rounding is **unbiased** — the expected value of the rounded result equals the true value. Over many steps, the rounding errors cancel out on average:

$$\mathbb{E}\left[\sum_{t=1}^{T} \text{SR}(g_t)\right] = \sum_{t=1}^{T} g_t$$

This means that even if individual gradient contributions are too small to represent in BF16, stochastic rounding ensures they contribute to the weight update over time.

**Convergence guarantee:**
With RNE, gradients smaller than $\frac{1}{2}$ ULP are always rounded to zero — the weight provably never updates if all gradients are this small. With stochastic rounding, even infinitesimally small gradients have a non-zero probability of causing an update.

**Hardware support:**

- NVIDIA H100 supports stochastic rounding for FP8 operations
- This is one reason why FP8 training is feasible despite the very low precision — stochastic rounding compensates for the coarse representation

### 12.4 Adam Optimizer Numerical Error Analysis

The Adam update rule:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t / (1 - \beta_2^t)} + \epsilon} \cdot \frac{m_t}{1 - \beta_1^t}$$

**Numerical failure mode 1 — $\epsilon$ too small:**

- Standard $\epsilon = 10^{-8}$
- If $v_t \approx 0$ (parameter with near-zero gradients): update $\approx \eta m_t / \epsilon$
- For $\eta = 10^{-4}$ and $m_t = 10^{-4}$: update $= 10^{-4} \times 10^{-4} / 10^{-8} = 1.0$ — a **massive** weight update
- This is why $\epsilon$ should be $10^{-6}$ or even $10^{-4}$ for BF16 training

**Numerical failure mode 2 — $v_t$ overflow in low precision:**

- $v_t$ accumulates $g_t^2$; if gradients are large ($g \sim 1$), then $v_t \sim 1$, fine
- But if gradient clipping isn't applied and $g \sim 100$ (loss spike): $g^2 = 10^4$ → $v_t$ grows rapidly
- In FP32: max $\approx 3.4 \times 10^{38}$ — fine
- In 8-bit optimizer states: overflow can occur, corrupting the optimizer state silently

**Numerical failure mode 3 — bias correction precision:**

- Bias correction factor $(1 - \beta_2^t)$ for $\beta_2 = 0.999$:
  - $t = 1$: $1 - 0.999 = 0.001$
  - $t = 100$: $1 - 0.999^{100} = 1 - 0.905 = 0.095$
  - $t = 10000$: $1 - 0.999^{10000} = 1 - 0.0000454 \approx 1.0$
- Early in training ($t < 100$), dividing by $0.001$ amplifies $v_t$ by 1000× — this amplification can push values out of representable range in low precision

### 12.5 Attention Logit Growth

A common failure mode in large transformer training:

**The mechanism:**

1. Attention logits $a = QK^T / \sqrt{d_k}$ grow slowly during training as the model learns sharper attention patterns
2. The softmax temperature effectively decreases: $\text{softmax}(a/T)$ with decreasing $T$
3. As logits grow, the softmax output becomes more peaked ("spiky")
4. Eventually, a single attention logit dominates: $\text{softmax}([50, 1, 1, \ldots]) \approx [1, 0, 0, \ldots]$
5. The gradient of softmax when it's nearly one-hot is very small → **vanishing gradients for attention**
6. Meanwhile, the large logit values approach the overflow boundary of the number format

**Numerical progression (BF16 example):**

| Training Step | Max Logit | Softmax Max    | Gradient Magnitude | Risk      |
| ------------- | --------- | -------------- | ------------------ | --------- |
| 10K           | 5.0       | 0.73           | Normal             | Safe      |
| 50K           | 15.0      | 0.995          | Reduced            | Low       |
| 100K          | 30.0      | 0.9999         | Very small         | Medium    |
| 200K          | 60.0      | ~1.0           | Near zero          | High      |
| 250K          | 89+       | Overflow → NaN | N/A                | **Crash** |

**Mitigation strategies:**

- **QK-Norm (Dehghani et al., 2023):** normalise $Q$ and $K$ before computing attention: $a = \text{norm}(Q) \cdot \text{norm}(K)^T / \sqrt{d_k}$. This bounds logit magnitudes by $\sqrt{d_k}$ regardless of training step.
- **Logit capping:** clamp attention logits to $[-C, C]$ before softmax ($C = 50$ in PaLM). Simple but effective.
- **Softmax temperature:** explicitly maintain temperature: $\text{softmax}(a / T_{\text{learned}})$ where $T$ is a learnable parameter, constrained to be positive.

---

## 13. Practical Guide — Choosing Number Formats

### 13.1 Decision Framework for Training

```
TRAINING FORMAT DECISION TREE
══════════════════════════════

Start: What is your model size?

├── < 1B parameters (fits in one GPU)
│   ├── Research/prototyping → FP32 (simplest, no mixed-precision bugs)
│   └── Production training  → BF16 mixed precision (2× speedup)
│
├── 1B–13B parameters
│   ├── Full training  → BF16 mixed precision (mandatory for speed)
│   └── Fine-tuning    → QLoRA (NF4 base + BF16 adapters)
│
├── 13B–70B parameters
│   ├── Full training  → BF16 + ZeRO-3 across multiple GPUs
│   ├── Fine-tuning    → QLoRA NF4 (fits on single 80GB GPU)
│   └── Continued PT   → BF16 + gradient checkpointing
│
└── 70B+ parameters
    ├── Full training  → BF16 + 3D parallelism (need cluster)
    ├── FP8 training   → if H100+ hardware (emerging 2024+)
    └── Fine-tuning    → QLoRA NF4 (4-bit base + 16-bit LoRA)

KEY RULES:
  1. Master weights ALWAYS in FP32
  2. Optimizer states ALWAYS in FP32 (or 8-bit Adam)
  3. Forward/backward: BF16 (or FP8 on H100+)
  4. Loss computation: FP32
  5. Gradient accumulation: FP32
```

### 13.2 Decision Framework for Inference

```
INFERENCE FORMAT DECISION TREE
═══════════════════════════════

Start: What is your latency/quality tradeoff?

├── Maximum quality (no degradation acceptable)
│   └── BF16 / FP16 (same as training format)
│
├── Balanced quality/speed (< 1% perplexity increase)
│   ├── NVIDIA GPU → INT8 (W8A8) with SmoothQuant
│   ├── Apple Silicon → INT8 (W8A8) via MLX
│   └── CPU → INT8 with ONNX Runtime
│
├── High compression (1-3% perplexity increase, 4× speedup)
│   ├── Batch inference    → GPTQ INT4 (W4A16)
│   ├── Real-time serving  → AWQ INT4 (W4A16)
│   └── Memory-constrained → GGML/GGUF Q4_K_M
│
├── Maximum compression (edge/mobile deployment)
│   ├── INT3 (W3A16) → aggressive but usable with GPTQ
│   ├── INT2 (W2A16) → significant quality loss; only for small models
│   └── 1.58-bit (ternary) → BitNet models (trained from scratch only)
│
└── Speculative/KV cache optimisation
    ├── KV cache INT8 → per-channel quantization
    └── KV cache INT4 → with group quantization + recent-token FP16
```

### 13.3 Per-Layer Sensitivity Table

Different layers have different tolerance for quantization. Based on empirical findings from AWQ, GPTQ, SqueezeLLM:

| Layer Type                  | INT8 | INT4 | INT3 | INT2 | Notes                                 |
| --------------------------- | ---- | ---- | ---- | ---- | ------------------------------------- |
| Embedding                   | ✅   | ✅   | ⚠️   | ❌   | Vocabulary coverage matters           |
| Attention Q, K projections  | ✅   | ✅   | ⚠️   | ❌   | Attention pattern quality degrades    |
| Attention V projection      | ✅   | ✅   | ✅   | ⚠️   | Slightly more robust than Q, K        |
| Attention output projection | ✅   | ✅   | ⚠️   | ❌   | Affects residual stream               |
| FFN gate & up projection    | ✅   | ✅   | ✅   | ⚠️   | Relatively robust                     |
| FFN down projection         | ✅   | ⚠️   | ❌   | ❌   | **Most sensitive** — affects residual |
| LM head (final projection)  | ✅   | ⚠️   | ❌   | ❌   | Directly affects token probabilities  |
| Layer norm / RMSNorm        | ✅   | ❌   | ❌   | ❌   | Keep in FP32 or BF16 always           |

Legend: ✅ = safe, ⚠️ = measurable degradation, ❌ = significant quality loss

### 13.4 Quantization Quality vs Bit Width

Empirical perplexity results (approximate, varies by model and method):

| Bit Width | Method             | 7B Model PPL | 13B Model PPL | 70B Model PPL |
| --------- | ------------------ | ------------ | ------------- | ------------- |
| 16 (BF16) | Baseline           | 5.68         | 5.09          | 3.56          |
| 8 (INT8)  | SmoothQuant        | 5.69 (+0.01) | 5.10 (+0.01)  | 3.56 (+0.00)  |
| 4 (INT4)  | GPTQ               | 5.85 (+0.17) | 5.20 (+0.11)  | 3.60 (+0.04)  |
| 4 (NF4)   | QLoRA/bitsandbytes | 5.80 (+0.12) | 5.16 (+0.07)  | 3.58 (+0.02)  |
| 4 (INT4)  | AWQ                | 5.78 (+0.10) | 5.15 (+0.06)  | 3.58 (+0.02)  |
| 3 (INT3)  | GPTQ               | 6.29 (+0.61) | 5.51 (+0.42)  | 3.72 (+0.16)  |
| 2 (INT2)  | QuIP#              | 7.85 (+2.17) | 6.43 (+1.34)  | 4.15 (+0.59)  |

**Key observation:** larger models are more tolerant of quantization — a 70B INT4 model often outperforms a 13B BF16 model. This is because larger models have more redundancy in their weight matrices.

---

## 14. Common Mistakes and Misconceptions

| #   | Mistake                                      | Why It's Wrong                                                                       | Correct Understanding                                             |
| --- | -------------------------------------------- | ------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| 1   | "FP16 and BF16 are interchangeable"          | FP16 overflows at 65504; BF16 at 3.4×10³⁸. FP16 needs loss scaling; BF16 doesn't.    | BF16 matches FP32 range; FP16 does not. Choose BF16 for training. |
| 2   | "More bits always means better quality"      | 70B at INT4 outperforms 13B at FP16 on most benchmarks.                              | Total model capacity matters more than per-parameter precision.   |
| 3   | "Quantization only saves memory"             | Quantization also reduces memory bandwidth (the actual bottleneck) → direct speedup. | Memory savings → bandwidth savings → latency reduction.           |
| 4   | "INT8 inference loses accuracy"              | With proper calibration (SmoothQuant, GPTQ), INT8 is lossless for nearly all models. | INT8 is the "free lunch" of inference optimisation.               |
| 5   | "I should quantize my model during training" | Quantization-aware training is expensive and unnecessary for PTQ with 4+ bits.       | Use PTQ unless deploying at INT2 or lower.                        |
| 6   | "FP32 master weights waste memory"           | Without FP32 master weights, BF16 training diverges after ~100K steps.               | FP32 masters are essential, not optional.                         |
| 7   | "Stochastic rounding is just noisy"          | SR is an unbiased estimator that enables convergence in low precision.               | SR is mathematically principled, not a hack.                      |
| 8   | "FLOPS determines inference speed"           | Memory bandwidth is the bottleneck for autoregressive LLM inference.                 | Optimise for memory bandwidth, not compute.                       |
| 9   | "Quantizing KV cache is dangerous"           | Per-channel INT8 KV cache quantization is nearly lossless.                           | KV cache quantization is safe and critical for long-context.      |
| 10  | "All layers should use the same precision"   | Output projection and FFN down projection are much more sensitive.                   | Use mixed-precision per-layer quantization.                       |

---

## 15. Exercises

### Exercise 1: IEEE 754 Encoding

Encode the decimal value $-13.625$ in IEEE 754 FP32 format. Show all steps: sign bit, binary conversion, normalisation, biased exponent, and mantissa. Verify by decoding your result back to decimal.

### Exercise 2: Quantization Calculation

Given a weight tensor $W = [-1.2, 0.3, -0.7, 0.9, 0.0, -0.1, 0.5, -0.4]$, compute the INT8 symmetric quantization:

- (a) Calculate the scale factor $s$
- (b) Quantize all values to INT8
- (c) Dequantize and compute the maximum absolute error
- (d) Compute the SQNR in dB

### Exercise 3: BF16 Precision Limits

Two FP32 values: $a = 1.0$ and $b = 0.001$.

- (a) Add $a + b$ in FP32. What is the result?
- (b) Cast both values to BF16, add, and cast back to FP32. What is the result?
- (c) Repeat the BF16 addition 1000 times (adding $b$ to a running sum starting at $a$). Compare with the FP32 result. What is the relative error?
- (d) How does this relate to gradient accumulation in training?

### Exercise 4: Memory Budget

You have a single NVIDIA A100 80GB GPU. Calculate the maximum model size (in parameters) you can:

- (a) Train with full FP32 (weights + Adam optimizer states)
- (b) Train with BF16 mixed precision (FP32 master + BF16 working + FP32 Adam)
- (c) Fine-tune with QLoRA (NF4 base + BF16 LoRA with rank 16, hidden dim 4096, 32 layers × 4 projection matrices)
- (d) Serve for inference in INT4 (weights only, no optimizer states)

### Exercise 5: Softmax Stability

Given attention logits $z = [88.5, 88.7, 88.3, 88.6]$:

- (a) Compute $e^{z_i}$ for each value. Can these be represented in FP32? In BF16?
- (b) Apply the max-subtraction trick and recompute. Show that no overflow occurs.
- (c) Compute the final softmax probabilities.
- (d) What would happen if $z = [200, 200.1, 199.9]$? How does log-sum-exp help?

### Exercise 6: Error Propagation

A 3-layer network with weight matrices $W_1, W_2, W_3$ (each $d \times d$, where $d = 1024$). Each weight is quantized to INT8 with scale $s = 0.01$. Assuming input $\|x\| = 1$ and $\|W_i\| = 1$ (operator norm):

- (a) Bound the quantization error $\|\Delta W_i\|$ for each layer.
- (b) Using the first-order error approximation from §9.6, bound the total output error.
- (c) At what bit width does the output error drop below 1% of the output magnitude?

### Exercise 7: Hardware Arithmetic Intensity

For a decoder-only transformer with $L = 32$ layers, $d = 4096$, $d_{\text{ffn}} = 11008$ (LLaMA-7B architecture):

- (a) Calculate the total FLOPs for generating one token (attention + FFN, ignore KV cache read).
- (b) Calculate the total weight bytes loaded from memory in INT4, INT8, and BF16.
- (c) Compute the arithmetic intensity for each format.
- (d) Given the H100's 3.35 TB/s bandwidth and 1979 TOPS (INT8), determine which format is compute-bound vs memory-bound.

### Exercise 8: Stochastic Rounding Simulation

Implement a simple stochastic rounding function in Python. Given a "true" gradient $g = 0.001$ and a simulated BF16 format where values snap to multiples of 0.0078125 (= $2^{-7}$, the BF16 ULP near 1.0):

- (a) What does round-to-nearest produce for $g$? (Hint: $g$ is below half the ULP.)
- (b) Implement stochastic rounding. Over 10,000 steps of accumulating $g$ into a running sum, what is the expected sum? Simulate and verify.
- (c) Compare with deterministic RNE accumulation over the same 10,000 steps. What is the relative error?

---

## 16. Why This Matters for AI/ML

| Concept              | Where It Appears                     | Why You Need It                                                       |
| -------------------- | ------------------------------------ | --------------------------------------------------------------------- |
| IEEE 754 FP32/FP64   | Loss computation, optimizer states   | Understanding overflow/underflow prevents silent training failures    |
| BF16                 | Default training format (2020+)      | Know when and why to use it; understand its precision limits          |
| FP8                  | Next-gen training (H100+)            | Critical for cost-efficient training at scale                         |
| INT8 quantization    | Standard inference optimisation      | 2× speedup with no quality loss — mandatory knowledge                 |
| INT4 quantization    | High-compression inference           | Enables serving 70B models on consumer hardware                       |
| NF4                  | QLoRA fine-tuning                    | Enables 70B fine-tuning on a single GPU                               |
| Softmax stability    | Every forward pass                   | Prevents the most common NaN crash in transformers                    |
| Mixed precision      | Every training pipeline              | Halves memory, doubles speed — used in all modern training            |
| KV cache formats     | Long-context inference               | Enables 128K+ context without OOM                                     |
| Error propagation    | Model debugging                      | Understanding why certain layers can/cannot be quantized aggressively |
| Hardware constraints | Deployment planning                  | Choose the right format for your target hardware                      |
| Stochastic rounding  | FP8 training, low-precision research | Key enabler for sub-8-bit training                                    |

---

## 17. Conceptual Bridge

This chapter covered the **representation** of numbers — how mathematical quantities are encoded in finite hardware. The key insight is that every number format is a **tradeoff between range, precision, and cost**, and AI engineering is the art of choosing the right tradeoff for each part of the pipeline.

```
CONCEPTUAL FLOW
═══════════════

  Number Systems (this chapter)
  ├── How are values represented in hardware?
  ├── What are the precision limits?
  └── How do these limits affect AI systems?
          │
          ▼
  Sets and Logic (next chapter)
  ├── How do we formalise collections of objects?
  ├── What are the logical foundations of proofs?
  └── How does set theory underpin probability and statistics?
          │
          ▼
  Functions and Mappings
  ├── How do we formalise input-output relationships?
  ├── What properties must a function have?
  └── Neural networks as compositions of functions
          │
          ▼
  Summation and Product Notation
  ├── Compact notation for sums and products
  ├── Used everywhere: loss functions, gradients, attention
  └── Connection to vectorised computation
```

**How number systems connect to every subsequent chapter:**

- **Linear Algebra:** matrix operations are chains of multiply-accumulate; the number format determines both the speed and accuracy of every matrix operation in the model
- **Calculus:** derivatives are computed via finite differences or analytical rules; numerical precision determines whether gradient computation is meaningful or noise
- **Probability:** probability values in $[0, 1]$ are represented as floating-point numbers; underflow in small probabilities ($P < 10^{-38}$) causes silent failures unless you use log-probabilities
- **Optimisation:** every optimiser update involves division, square roots, and accumulation — all precision-sensitive operations
- **Information Theory:** cross-entropy loss involves $\log$ and $\exp$ — the most overflow/underflow-prone elementary functions

> **The fundamental lesson:** a deep understanding of number systems is not optional for AI practitioners. It's the foundation that determines whether your model trains at all, how much it costs, how fast it runs, and whether you can deploy it on your target hardware.

---

[← Back to Mathematical Foundations](../README.md) | [Next: Sets and Logic →](../02-Sets-and-Logic/notes.md)
