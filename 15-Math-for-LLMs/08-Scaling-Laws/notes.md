# Scaling Laws

[← Fine-Tuning Math](../07-Fine-Tuning-Math/notes.md) | [Home](../../README.md) | [Efficient Attention and Inference →](../09-Efficient-Attention-and-Inference/notes.md)

> _"Scaling laws are the closest thing deep learning has to a physics — reliable, quantitative, predictive. They answer: given a budget of compute C, what is the best model size N and dataset size D?"_

## Overview

Scaling laws describe **empirical power-law relationships** between a model's performance (cross-entropy loss) and its key resources: parameters $N$, training data $D$ (tokens), and compute $C$ (FLOPs). These are not mere observations — they are predictive tools that guide multi-million-dollar training decisions. Given a fixed compute budget, scaling laws tell you how large your model should be, how much data to train on, and what loss to expect. This section covers the Kaplan et al. (2020) and Chinchilla (Hoffmann et al. 2022) scaling laws, derives compute-optimal allocation via Lagrange multipliers, explores inference-optimal scaling, data quality effects, emergent abilities, MoE and multimodal scaling, test-time compute, and connects everything to the practical economics of building frontier AI systems.

## Prerequisites

- Calculus: partial derivatives, Lagrange multipliers, optimisation
- Probability: cross-entropy loss, perplexity
- Logarithms: log-log plots, power laws, curve fitting
- Completed: [07-Fine-Tuning-Math](../07-Fine-Tuning-Math/notes.md) — fine-tuning objectives, LoRA, RLHF

## Companion Notebooks

| Notebook                           | Description                                                                                                     |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| [theory.ipynb](theory.ipynb)       | Power-law fitting, Chinchilla-optimal allocation, loss prediction, IsoFLOP profiles, MoE scaling, visualisation |
| [exercises.ipynb](exercises.ipynb) | Compute-optimal sizing, loss prediction, FLOPs budgeting, power-law fitting, MoE effective parameters           |

## Learning Objectives

After completing this section, you will:

- State the power-law scaling relationships between loss, parameters, data, and compute
- Derive the Kaplan and Chinchilla compute-optimal allocation formulas using Lagrange multipliers
- Compute the optimal model size and data requirement for any given compute budget
- Explain why Chinchilla overturned Kaplan's recommendations and quantify the under-training of GPT-3
- Apply inference-optimal scaling to balance training cost against deployment cost
- Analyse data quality and repeated-data scaling laws
- Evaluate the emergent abilities debate and the metric artefact hypothesis
- Apply scaling laws to MoE models, multimodal systems, and test-time compute
- Make practical training budget decisions using scaling law predictions

---

## Table of Contents

1. [Intuition](#1-intuition)
2. [Formal Definitions](#2-formal-definitions)
3. [Kaplan et al. (2020) — OpenAI Scaling Laws](#3-kaplan-et-al-2020--openai-scaling-laws)
4. [Chinchilla (Hoffmann et al. 2022) — Revised Scaling Laws](#4-chinchilla-hoffmann-et-al-2022--revised-scaling-laws)
5. [Inference-Optimal Scaling](#5-inference-optimal-scaling)
6. [Data Scaling Laws](#6-data-scaling-laws)
7. [Emergent Abilities](#7-emergent-abilities)
8. [Compute-Optimal Scaling — Full Mathematical Treatment](#8-compute-optimal-scaling--full-mathematical-treatment)
9. [Scaling Laws for Downstream Tasks](#9-scaling-laws-for-downstream-tasks)
10. [Test-Time Compute Scaling](#10-test-time-compute-scaling)
11. [Scaling Laws for MoE Models](#11-scaling-laws-for-moe-models)
12. [Multimodal Scaling Laws](#12-multimodal-scaling-laws)
13. [Practical Scaling Law Estimation](#13-practical-scaling-law-estimation)
14. [Limitations and Critiques](#14-limitations-and-critiques)
15. [Common Mistakes](#15-common-mistakes)
16. [Exercises](#16-exercises)
17. [Why This Matters for AI](#17-why-this-matters-for-ai-2026-perspective)

---

## 1. Intuition

### 1.1 What Are Scaling Laws?

Scaling laws are **empirical mathematical relationships** describing how model performance changes as you increase compute, data, or parameters. The core discovery: loss decreases as a smooth, predictable **power law** across many orders of magnitude.

$$L = \frac{A}{X^\alpha} + L_\infty$$

where $X$ is the resource (parameters, tokens, or FLOPs), $A$ is a coefficient, $\alpha$ is the scaling exponent, and $L_\infty$ is the irreducible loss (entropy of the data).

This was **not obvious** — neural networks could have plateaued, shown diminishing returns, or behaved erratically. Instead: remarkably clean relationships that hold from 10M to 10T parameters. Scaling laws are the closest thing to a "physics of deep learning": reliable, quantitative, predictive.

They answer: **given a budget of compute $C$, what is the best model size $N$ and dataset size $D$?**

### 1.2 Why Scaling Laws Matter

```
SCALING LAWS AS ENGINEERING TOOLS
═══════════════════════════════════════════════════════════════════════

Before scaling laws:
┌─────────────────────────────────────────────────────────────────────┐
│  "Let's train a big model and hope it works"                        │
│  → $10M spent → model undertrained → wasted budget                  │
│  → Or: model too small → data wasted → suboptimal loss              │
└─────────────────────────────────────────────────────────────────────┘

After scaling laws:
┌─────────────────────────────────────────────────────────────────────┐
│  Budget: $10M → compute: 10²⁴ FLOPs                                │
│  Chinchilla-optimal: N* = 67B params, D* = 1.4T tokens              │
│  Expected loss: L ≈ 1.69 nats                                       │
│  → BEFORE spending a dollar, you know what you'll get               │
└─────────────────────────────────────────────────────────────────────┘
```

| Use Case                    | What Scaling Laws Provide                                         |
| --------------------------- | ----------------------------------------------------------------- |
| **Planning**                | Predict final loss before spending millions on training           |
| **Resource allocation**     | Determine optimal N/D split for fixed compute budget              |
| **Capability forecasting**  | Extrapolate future model capabilities from trend                  |
| **Architecture comparison** | Compare architectures fairly at equal compute                     |
| **Research prioritisation** | Identify whether an improvement scales or only helps small models |
| **ROI estimation**          | Is 10× more compute worth the loss improvement?                   |

Without scaling laws: trial-and-error at enormous cost. With them: principled extrapolation.

### 1.3 The Central Empirical Finding

Test loss $L$ as a function of parameters $N$, data $D$, compute $C$:

$$\boxed{L \sim N^{-\alpha}, \quad L \sim D^{-\beta}, \quad L \sim C^{-\gamma}}$$

Each relationship is a **power law**: straight line on log-log plot.

```
LOG-LOG PLOT (Schematic)
═══════════════════════════════════════════════════════════════════════

  Loss (log)
  3.0 ┤●
      │ ●
  2.5 ┤  ●
      │    ●
  2.0 ┤      ●
      │        ●●
  1.8 ┤           ●●●
      │               ●●●●
  1.7 ┤                    ●●●●●●●
      │                            ●●●●●●●●●●●  ← diminishing returns
  1.6 ┤─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ L∞ (irreducible loss)
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──▶
       10⁶ 10⁷ 10⁸ 10⁹ 10¹⁰           Parameters (log)

  Slope = −α (scaling exponent); steeper = faster improvement
  Key: STRAIGHT LINE on log-log → power law
```

Exponents are small (0.05–0.1): large increases in scale yield modest but consistent loss reduction. No sign of saturation across observed range (up to 2026).

### 1.4 Historical Timeline

| Year    | Milestone                    | Key Contribution                                                                |
| ------- | ---------------------------- | ------------------------------------------------------------------------------- |
| 1951    | Shannon                      | Entropy rate of English; implicit scaling intuition                             |
| 1989    | Baum & Haussler              | Sample complexity theory; early scaling ideas                                   |
| 2017    | Hestness et al.              | First systematic power-law observations across NLP domains                      |
| 2020    | Kaplan et al. (OpenAI)       | Foundational scaling laws paper; power law in N, D, C; compute-optimal frontier |
| 2022    | Hoffmann et al. (Chinchilla) | Revised optimal N/D ratio; 20 tokens/param rule                                 |
| 2022    | Wei et al.                   | Emergent abilities; discontinuous capability jumps at scale                     |
| 2022    | Zoph et al.                  | Scaling laws for transfer learning                                              |
| 2022    | Hernandez et al.             | Scaling laws for fine-tuning                                                    |
| 2022    | Clark et al.                 | Unified scaling laws across modalities                                          |
| 2023    | Schaeffer et al.             | "Emergent abilities are a mirage" — metric artefact hypothesis                  |
| 2023    | Muennighoff et al.           | Scaling laws for repeated data (multi-epoch)                                    |
| 2023    | Sardana & Frankle            | Scaling laws for inference-optimal training                                     |
| 2024    | Gadre et al.                 | DataComp-LM; data quality scaling; not all tokens equal                         |
| 2024    | Snell et al.                 | Scaling laws for test-time compute                                              |
| 2024–25 | DeepSeek                     | Scaling beyond standard Chinchilla; MoE scaling; data quality matters           |
| 2025–26 | Frontier labs                | Scaling laws for reasoning, multimodal, MoE; active frontier                    |

### 1.5 What Scaling Laws Cover

```
SCOPE OF SCALING LAW RESEARCH
═══════════════════════════════════════════════════════════════════════

  Pretraining loss vs parameters, data, compute    ← §3, §4, §8
  Downstream task performance vs scale              ← §9
  Inference-optimal allocation                      ← §5
  Data quality, mixture, and repeated data           ← §6
  Emergent abilities and phase transitions           ← §7
  Test-time compute (reasoning tokens)               ← §10
  MoE models                                         ← §11
  Multimodal models                                  ← §12

  Pipeline position:
  [Architecture] → [Scaling Analysis] → [Budget] → [Training] → [Serving]
                    ^^^^^^^^^^^^^^^^^^
                      THIS section
```

---

## 2. Formal Definitions

### 2.1 The Loss Function

$L(N, D)$: **cross-entropy loss** (nats or bits) on held-out test set, measured per token — average negative log-likelihood per token.

$$L = -\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})$$

All scaling law analysis uses this as the primary metric. Lower is better; irreducible entropy $H^*$ sets a floor.

**Relationship to perplexity:**

$$\text{PPL} = e^L$$

A loss of 1.69 nats corresponds to perplexity $e^{1.69} \approx 5.42$; a loss reduction of 0.1 nats lowers perplexity by ≈10%.

### 2.2 Key Variables

|       Symbol        | Name             | Unit   | Definition                                         |
| :-----------------: | ---------------- | ------ | -------------------------------------------------- |
|         $N$         | Parameters       | count  | Non-embedding parameters (exclude embedding table) |
|         $D$         | Data             | tokens | Number of training tokens                          |
|         $C$         | Compute          | FLOPs  | Total training compute                             |
|         $B$         | Batch size       | tokens | Tokens per gradient step                           |
|         $S$         | Steps            | count  | Number of gradient update steps: $S = D/B$         |
|       $\eta$        | Learning rate    | —      | Step size for optimisation                         |
|        $L^*$        | Optimal loss     | nats   | Lowest achievable loss at given $(N, D, C)$        |
| $E$ (or $L_\infty$) | Irreducible loss | nats   | Entropy of true data distribution                  |

**The fundamental compute constraint:**

$$\boxed{C \approx 6ND}$$

Factor 6: 2 (multiply-add) × 3 (forward ×1, backward ×2 for gradient computation).

```
          Parameters (N)
              ▲
             /|\
            / | \
           /  |  \
          / C≈6ND  \
         /    |     \
        /     |      \
       ────────┼────────▶ Data (D)
               |
           Compute (C)
```

**More precise estimate (accounting for attention):**

$$C = 6ND + 12 \cdot n_{\text{layers}} \cdot n_{\text{ctx}} \cdot d_{\text{model}}$$

For most models: second term is small (< 5%) relative to first.

### 2.3 Power Law Definition

A function $f(x)$ follows a power law if:

$$\boxed{f(x) = ax^{-\alpha}}$$

Linear on log-log scale: $\log f = \log a - \alpha \log x$.

| Property                    | Meaning                                                  |
| --------------------------- | -------------------------------------------------------- |
| $\alpha$ (scaling exponent) | Larger $\alpha$ = faster improvement with scale          |
| **Scale-free**              | Doubling $x$ always yields same proportional improvement |
| **Log-linear**              | Straight line on log-log plot                            |

**Offset power law** (with irreducible loss):

$$f(x) = \left(\frac{x_0}{x}\right)^{\alpha} + L_\infty$$

where $L_\infty$ = irreducible loss floor.

### 2.4 Irreducible Loss

$L_\infty$ (or $E$): minimum possible loss — the **entropy of the true data distribution**.

$$H(\text{text}) = -\sum_t P(t)\log P(t)$$

No amount of scale can reduce loss below $L_\infty$.

| Source                |    Estimated $H$    | Method                           |
| --------------------- | :-----------------: | -------------------------------- |
| Shannon (1951)        |   ~1.0 bits/char    | Human prediction game            |
| Brown et al. (1992)   |   ~1.2 bits/char    | Trigram models                   |
| Chinchilla fit        |  ~1.69 nats/token   | Scaling law extrapolation        |
| Modern estimate (BPE) | ~1.5–1.8 nats/token | Depends on tokeniser, domain     |
| Code (structured)     | ~1.0–1.3 nats/token | Lower entropy (more predictable) |
| Math (LaTeX)          | ~1.3–1.6 nats/token | Structured but technical         |

### 2.5 The Scaling Hypothesis

The broader conjecture:

1. Most cognitive tasks reduce to prediction
2. Prediction improves smoothly and predictably with scale
3. Sufficiently large, well-trained LLMs may approach human-level performance on many tasks
4. No fundamental barrier has been observed — only resource constraints

Not proven; contested; but motivates continued scaling investment. Counter-arguments: reasoning may require fundamentally different architectures; distribution shift limits; alignment difficulties.

---

## 3. Kaplan et al. (2020) — OpenAI Scaling Laws

### 3.1 Core Results

Kaplan et al. trained ~70 Transformer language models ranging from 768 parameters to 1.5B parameters on WebText2, measuring test loss. Found clean power laws for each axis independently:

**Parameters (infinite data limit):**

$$\boxed{L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad N_c = 8.8 \times 10^{13}, \quad \alpha_N \approx 0.076}$$

**Data (infinite model limit):**

$$\boxed{L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad D_c = 5.4 \times 10^{13}, \quad \alpha_D \approx 0.095}$$

**Compute (optimal model/data for each C):**

$$\boxed{L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad C_c = 3.1 \times 10^8, \quad \alpha_C \approx 0.050}$$

| Law    | Exponent | Meaning            | 10× Scale Effect                                 |
| ------ | :------: | ------------------ | ------------------------------------------------ |
| $L(N)$ |  0.076   | Loss vs parameters | $10^{-0.076} \approx 0.84$ (~16% loss reduction) |
| $L(D)$ |  0.095   | Loss vs data       | $10^{-0.095} \approx 0.80$ (~20% loss reduction) |
| $L(C)$ |  0.050   | Loss vs compute    | $10^{-0.050} \approx 0.89$ (~11% loss reduction) |

**Interpretation**: 10× more parameters reduces loss by ~16%. 10× more data reduces loss by ~20%. 10× more compute reduces loss by ~11%.

### 3.2 Combined Loss Formula

Unified formula treating $L$ as a function of both $N$ and $D$:

$$\boxed{L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\beta} + \left(\frac{D_c}{D}\right)^{\alpha_D/\beta}\right]^\beta}$$

$\beta \approx 1$; both terms add contributions to loss. Reduces to $L(N)$ when $D \to \infty$; reduces to $L(D)$ when $N \to \infty$.

```
HOW N AND D INTERACT (Iso-Loss Contours)
═══════════════════════════════════════════════════════════════════════

  D (tokens, log)
  10¹³ ┤              ╱ L=1.8
       │            ╱
  10¹² ┤          ╱      ╱ L=2.0
       │        ╱      ╱
  10¹¹ ┤      ╱      ╱     ╱ L=2.5
       │    ╱      ╱     ╱
  10¹⁰ ┤  ╱      ╱     ╱
       └──┴──────┴─────┴────────────▶ N (params, log)
        10⁸    10⁹   10¹⁰   10¹¹

  Trade-off: same loss achievable with many (N, D) combinations
  Iso-loss curves are convex → unique optimum for each C
```

### 3.3 Compute-Optimal Allocation (Kaplan)

Kaplan found that for fixed compute $C$, optimal allocation **heavily favours parameters over data**:

$$\boxed{N_{\text{opt}} \propto C^{0.73}, \quad D_{\text{opt}} \propto C^{0.27}}$$

Implication: given 10× more compute, make model ~5× larger but only ~2× more data. This led to the design philosophy behind GPT-3: a very large model (175B) trained on relatively few tokens (300B).

| Compute $C$ | Kaplan $N^*$ | Kaplan $D^*$ | $D/N$ ratio |
| :---------: | :----------: | :----------: | :---------: |
|  $10^{18}$  |     ~10M     |     ~2B      |    ~200     |
|  $10^{20}$  |    ~300M     |     ~6B      |     ~20     |
|  $10^{22}$  |     ~10B     |     ~17B     |    ~1.7     |
|  $10^{24}$  |    ~300B     |     ~50B     |    ~0.17    |

**Later shown to be wrong by Chinchilla** — Kaplan under-estimated the importance of data.

### 3.4 Critical Batch Size

Kaplan observed that optimal batch size scales with the current loss:

$$B_{\text{crit}} \propto L^{-1/\alpha_B}, \quad \alpha_B \approx 0.21$$

- Training at $B \ll B_{\text{crit}}$: wasteful (gradient noise too high; could use larger batch without degrading quality)
- Training at $B \gg B_{\text{crit}}$: diminishing returns per token (computation spent on redundant gradient information)
- At $B = B_{\text{crit}}$: optimal trade-off between time and compute efficiency

**Practical rule**: $B_{\text{crit}}$ increases during training as loss decreases; modern practice: warm up batch size or use gradient accumulation.

### 3.5 Sample Efficiency

A key Kaplan finding: **larger models are more sample-efficient**. They reach any given loss with fewer tokens.

$$D_{\min}(L, N) \propto N^{-\gamma}, \quad \gamma > 0$$

- Larger model extracts more signal from each token
- But: per-token compute cost also grows with $N$ ($C_{\text{per-token}} \propto N$)
- Trade-off: more parameters = fewer tokens needed, but each token costs more

This observation led Kaplan to conclude that parameters matter more than data — a conclusion Chinchilla later reversed by showing their models were undertrained.

### 3.6 Architectural Irrelevance (at Fixed N, C)

Key Kaplan finding: **architectural details matter little** for performance at fixed $N$ and $C$.

| Hyperparameter            | Effect on Loss at Fixed N            |
| ------------------------- | ------------------------------------ |
| Depth vs width            | Negligible (within reasonable range) |
| Number of attention heads | Negligible                           |
| Context window length     | Small effect                         |
| Activation function       | Negligible                           |

Performance primarily determined by total parameter count $N$ and total compute $C$, not specific architecture shape. This enables fair comparison: compare models at equal compute budget.

**Caveat**: this holds within the Transformer family. Radically different architectures (SSMs, state-space models) may have different scaling constants.

---

## 4. Chinchilla (Hoffmann et al. 2022) — Revised Scaling Laws

### 4.1 The Chinchilla Correction

Kaplan held data fixed while varying model size, fitting power laws in a biased region of $(N, D)$ space. Hoffmann trained models with **jointly varying** $N$ and $D$, using three complementary approaches:

1. **Fixed model sizes**: train each model on varying tokens
2. **IsoFLOP profiles**: fix compute, vary $N$ and $D$
3. **Parametric fit**: fit $L(N, D) = E + A/N^\alpha + B/D^\beta$

All three converged on the same revised conclusion:

$$\boxed{N_{\text{opt}} \propto C^{0.50}, \quad D_{\text{opt}} \propto C^{0.50}}$$

$N$ and $D$ should scale **equally** with compute.

```
KAPLAN vs CHINCHILLA — THE FUNDAMENTAL DISAGREEMENT
═══════════════════════════════════════════════════════════════════════

                Kaplan (2020)              Chinchilla (2022)
                ─────────────              ─────────────────
  N_opt ∝       C^0.73                     C^0.50
  D_opt ∝       C^0.27                     C^0.50
  D/N ratio     Decreasing with C          Constant (~20)

  Interpretation:
  Kaplan:     "Make model BIG; data doesn't matter as much"
              → GPT-3: 175B params, 300B tokens (D/N = 1.7)
  Chinchilla: "Scale model AND data EQUALLY"
              → Chinchilla: 70B params, 1.4T tokens (D/N = 20)

  Why Kaplan was wrong:
  ┌──────────────────────────────────────────────────────────────────┐
  │ • Trained models for too few tokens (≤300B for all sizes)       │
  │ • Never explored regime where bigger models are undertrained    │
  │ • Power law fitted in a biased region of (N, D) space           │
  │ • Models hadn't reached data-limited regime                     │
  └──────────────────────────────────────────────────────────────────┘
```

### 4.2 Chinchilla Loss Formula

$$\boxed{L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}}$$

- $E$ = irreducible entropy ($L_\infty$); fixed floor from data distribution
- $A/N^\alpha$: **parametric loss**; decreases as model gets larger
- $B/D^\beta$: **data loss**; decreases as more training tokens seen

Fitted values from Hoffmann et al.:

| Parameter | Value | Meaning                         |
| --------- | :---: | ------------------------------- |
| $A$       | 406.4 | Parameter scaling coefficient   |
| $B$       | 410.7 | Data scaling coefficient        |
| $\alpha$  | 0.34  | Parameter scaling exponent      |
| $\beta$   | 0.28  | Data scaling exponent           |
| $E$       | 1.69  | Irreducible loss (data entropy) |

**Example calculation** for GPT-3 ($N = 175$B, $D = 300$B):

$$L(175\text{B}, 300\text{B}) = 1.69 + \frac{406.4}{(175 \times 10^9)^{0.34}} + \frac{410.7}{(300 \times 10^9)^{0.28}}$$

$$= 1.69 + \frac{406.4}{4,\!584} + \frac{410.7}{2,\!636} \approx 1.69 + 0.089 + 0.156 = 1.93$$

Data term (0.156) dominates parametric term (0.089) → model is **under-trained** on data.

### 4.3 Optimal Allocation

Minimise $L(N, D)$ subject to $C = 6ND$ using Lagrange multipliers (full derivation in §8):

$$\frac{A\alpha}{N^{\alpha+1}} = \frac{B\beta}{D^{\beta+1}}$$

Result:

$$\boxed{N^* = G\left(\frac{C}{6}\right)^a, \quad D^* = \frac{1}{G}\left(\frac{C}{6}\right)^b}$$

where $a = \frac{\beta}{\alpha+\beta} \approx 0.45$, $b = \frac{\alpha}{\alpha+\beta} \approx 0.55$, and:

$$G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)} \cdot 6^{-\beta/(\alpha+\beta)}$$

| Compute $C$ | $N^*$  | $D^*$ | $D^*/N^*$ | Example               |
| :---------: | :----: | :---: | :-------: | --------------------- |
|  $10^{18}$  |  ~4M   | ~80M  |    20     | Tiny model            |
|  $10^{20}$  |  ~60M  | ~1.2B |    20     | Small research model  |
|  $10^{22}$  | ~700M  | ~14B  |    20     | Medium model          |
|  $10^{23}$  |  ~7B   | ~140B |    20     | LLaMA-1 7B scale      |
|  $10^{24}$  |  ~67B  | ~1.4T |    20     | **Chinchilla**        |
|  $10^{25}$  | ~500B+ | ~10T+ |    20     | Hypothetical frontier |

### 4.4 The 20 Tokens Per Parameter Rule

The practical summary of Chinchilla-optimal training:

$$\boxed{D_{\text{opt}} \approx 20 \times N}$$

This emerges directly from $a \approx b \approx 0.5$ and the fitted constants. "Compute-optimal" = 20 tokens per parameter.

| Model      | $N$  | Chinchilla $D_{\text{opt}}$ | Actual $D$ | Status                           |
| ---------- | ---- | --------------------------- | ---------- | -------------------------------- |
| GPT-3      | 175B | 3.5T                        | 300B       | **Severely under-trained**       |
| LLaMA-1 7B | 7B   | 140B                        | 1T         | Over-trained (inference-optimal) |
| Chinchilla | 70B  | 1.4T                        | 1.4T       | ✓ Optimal                        |
| PaLM       | 540B | 10.8T                       | 780B       | Under-trained                    |

### 4.5 Chinchilla Model Details

- $N = 70$B parameters; $D = 1.4$T tokens
- Same compute as Gopher (280B params, 300B tokens)
- Chinchilla **outperforms Gopher, GPT-3, Megatron** despite being 4× smaller
- Validated the scaling law: smaller model + more data beats larger under-trained model

| Benchmark | Gopher (280B) | Chinchilla (70B) |   Winner   |
| --------- | :-----------: | :--------------: | :--------: |
| MMLU      |     60.0%     |    **67.5%**     | Chinchilla |
| HellaSwag |     79.2%     |    **80.8%**     | Chinchilla |
| LAMBADA   |     74.5%     |    **77.4%**     | Chinchilla |
| BoolQ     |     79.3%     |    **83.7%**     | Chinchilla |

4× fewer parameters → cheaper to serve → cheaper to fine-tune → better results.

### 4.6 Implications: Under-Training of Prior Models

| Model      | Parameters | Tokens | Chinchilla-Optimal Tokens | Under-Training Ratio |
| ---------- | ---------- | ------ | ------------------------- | -------------------- |
| GPT-3      | 175B       | 300B   | 3.5T                      | **11.7×**            |
| Gopher     | 280B       | 300B   | 5.6T                      | **18.7×**            |
| PaLM       | 540B       | 780B   | 10.8T                     | **13.8×**            |
| MT-NLG     | 530B       | 270B   | 10.6T                     | **39.3×**            |
| Chinchilla | 70B        | 1.4T   | 1.4T                      | 1.0× (optimal)       |

Billions of dollars in compute was spent training over-sized, under-trained models.

### 4.7 Limitations of Chinchilla Scaling Laws

1. **Training-compute optimal only**: ignores inference cost entirely
2. **Data source**: trained on MassiveText; scaling constants may differ for other data
3. **Architecture**: fitted for standard dense Transformers; MoE models differ
4. **Tokeniser dependence**: exponents may shift with different tokenisers
5. **Scale range**: fitted up to ~280B parameters; extrapolation beyond uncertain
6. **Ignores learning rate scheduling**: assumes optimal LR schedule already used

---

## 5. Inference-Optimal Scaling

### 5.1 The Inference Cost Perspective

A model is trained **once**; served **millions or billions** of times.

$$\text{Total cost} = C_{\text{train}} + C_{\text{inference}} \times n_{\text{queries}}$$

$$= 6ND + 2N \cdot T_{\text{output}} \cdot n_{\text{queries}}$$

- Training cost: fixed; amortised over model lifetime
- Inference cost: $\propto N$ (parameters determine FLOPs per forward pass)
- For high-traffic deployment: **inference cost dominates within weeks**

```
COST CROSSOVER POINT
═══════════════════════════════════════════════════════════════════════

  Cumulative Cost ($)
  ┤
  │                                      ╱ 70B model (Chinchilla-optimal)
  │                                   ╱     → cheaper to TRAIN
  │                                ╱        → expensive to SERVE
  │                             ╱
  │                          ╱
  │                    ×── ╱─── ← crossover point
  │                 ╱  ╱
  │              ╱  ╱       7B model (inference-optimal)
  │           ╱  ╱          → more expensive to TRAIN (more tokens)
  │        ╱  ╱             → much cheaper to SERVE
  │     ╱  ╱
  │  ╱  ╱
  └──────────────────────────────────────▶ Queries served
           Weeks    Months    Years
```

### 5.2 Inference-Optimal Frontier (Sardana & Frankle 2023)

For fixed inference budget (FLOPs per query), what is the best model?

Smaller model trained on more data reaches same loss at lower inference cost:

$$N_{\text{inference-opt}} < N_{\text{chinchilla-opt}}, \quad D_{\text{inference-opt}} > D_{\text{chinchilla-opt}}$$

**Formal trade-off**: define total lifecycle cost:

$$C_{\text{lifecycle}} = 6ND + 2N \cdot Q$$

where $Q = T_{\text{output}} \times n_{\text{queries}}$ (total inference tokens over lifetime).

Minimise $L(N, D)$ subject to $C_{\text{lifecycle}}$ = constant → different $N^*, D^*$ depending on $Q$.

### 5.3 The LLaMA Philosophy

Train smaller models on **far more data** than Chinchilla recommends:

| Strategy           | Model Size      | Tokens           | Tokens/Param | Optimised For             |
| ------------------ | --------------- | ---------------- | :----------: | ------------------------- |
| Chinchilla         | Compute-optimal | ~20× params      |     ~20      | Training cost             |
| Moderate overtrain | 2–5× smaller    | 50–100× params   |    50–100    | Balanced                  |
| Extreme overtrain  | 5–25× smaller   | 200–2000× params |   200–2000   | Inference cost            |
| LLaMA-3 8B         | 8B              | 15T              |  **1,875**   | Extreme inference savings |

LLaMA-1 7B: Chinchilla-optimal at ~140B tokens; trained to 1T = **7× Chinchilla** ratio. Result: LLaMA-7B matches or beats GPT-3 (175B) on many benchmarks — at 25× lower inference cost.

### 5.4 Tokens Per Parameter in Practice (2024–2026)

| Model               | Parameters        | Training Tokens |              Tokens/Param | Philosophy           |
| ------------------- | ----------------- | --------------- | ------------------------: | -------------------- |
| Chinchilla (2022)   | 70B               | 1.4T            |                        20 | Compute-optimal      |
| LLaMA-1 7B (2023)   | 7B                | 1T              |                       143 | Overtrained          |
| LLaMA-2 7B (2023)   | 7B                | 2T              |                       286 | More overtrained     |
| Mistral 7B (2023)   | 7B                | ~8T (est.)      |                    ~1,143 | High overtrain       |
| Phi-3 Mini (2024)   | 3.8B              | 3.3T            |                       868 | Inference-first      |
| LLaMA-3 8B (2024)   | 8B                | 15T             |                 **1,875** | Extreme overtrain    |
| DeepSeek-V3 (2024)  | 671B (37B active) | 14.8T           | 22 (total) / 400 (active) | MoE                  |
| Qwen-2.5 72B (2025) | 72B               | 18T             |                       250 | Aggressive overtrain |

Trend: tokens/param increasing every generation — from 20 (2022) to 1,875 (2024).

### 5.5 The Overtrain Trend and Its Limits

2023–2026: systematic shift toward inference-optimal training:

1. **Inference cost economics**: serving cost dominates for deployed models
2. **Open-source deployment**: users run models locally; smaller = accessible
3. **Edge computing**: mobile, on-device inference requires small models
4. **Quantisation synergy**: smaller models quantise more gracefully

**The key question**: does loss keep improving with more data, or does it plateau?

Evidence (2024–2025): LLaMA-3 8B still improving at 15T tokens. No clear plateau observed yet for models in the 3–8B range. Loss improvements slow but remain measurable up to at least 2,000 tokens/param.

**Diminishing returns**: the marginal improvement from the 15th trillion token is far smaller than the 1st trillion. Whether the improvement is worth the training cost depends on inference volume.

---

## 6. Data Scaling Laws

### 6.1 Not All Tokens Are Equal

Kaplan and Chinchilla assume uniform data quality. Reality: web data quality varies enormously. A supervised textbook example is worth far more than a random forum post.

**Data quality multiplier**: effective tokens $D_{\text{eff}} = q \cdot D_{\text{raw}}$ where quality multiplier $q$ varies 0.01–1.0.

$$L(N, D_{\text{eff}}) = E + \frac{A}{N^\alpha} + \frac{B}{D_{\text{eff}}^\beta}$$

Highly curated data shifts the entire scaling curve downward.

### 6.2 Data Quality Scaling (Gadre et al. 2024)

DataComp-LM: systematic study comparing data curation strategies at multiple scales.

Key findings:

- Better data filtering → steeper scaling slope
- A 10× data quality improvement can be worth 10× more raw tokens
- Optimal filtering aggressiveness changes with scale: larger models tolerate (and benefit from) noisier data

```
DATA QUALITY EFFECT ON SCALING
═══════════════════════════════════════════════════════════════════════

  Loss
  3.0 ┤●
      │  ●  Low-quality data (noisy web scrape)
  2.5 ┤●  ●●
      │  ●     ●●●●●●
  2.0 ┤●       ●●●●●●●●●●●●●●●●●●●  ← slow improvement
      │  ●
  1.8 ┤    ●  High-quality data (curated, filtered)
      │      ●●
  1.6 ┤         ●●●●●●●●  ← faster improvement
      │               ●●●●●●●  ← even at large scale
  1.5 ┤
      └───────────────────────────────────────▶ Tokens (log)

  Key: same model + better data → steeper scaling curve
       Data quality IS a scaling axis
```

### 6.3 Repeated Data Scaling (Muennighoff et al. 2023)

What happens when you run out of unique data? Training with $R$ epochs over $D$ unique tokens $\neq$ training on $R \times D$ unique tokens.

Scaling law for repeated tokens:

$$L(N, D, R) \approx L\left(N, D \cdot f(R)\right)$$

where $f(R)$ is an effective data multiplier:

$$f(R) = R^{1-c}, \quad c \approx 0.28$$

Effective tokens scale sub-linearly with repetitions: 4 epochs on $D$ tokens ≈ $4^{0.72} \approx 2.9$ effective epochs.

| Repetitions $R$ | Effective Data Multiplier $f(R)$ | Loss Degradation      | Practical Assessment |
| :-------------: | :------------------------------: | --------------------- | -------------------- |
|        1        |               1.00               | 0%                    | Baseline             |
|        2        |               1.66               | Small (~1–3%)         | Acceptable           |
|        4        |               2.84               | Moderate (~3–5%)      | Usually acceptable   |
|        8        |               4.85               | Notable (~5–10%)      | Quality degrades     |
|       16        |               8.30               | Significant (~10–15%) | Last resort          |
|       50+       |          Very sublinear          | Severe                | Memorisation risk    |

**Rule of thumb**: up to 4 epochs is acceptable; beyond 4, significant diminishing returns; beyond 16, training becomes inefficient and risks memorisation.

### 6.4 Data Mixture Scaling

Multiple data sources with different quality and domain distributions:

- **DoReMi** (Xie et al. 2023): learns mixing weights via group distributionally robust optimisation (group DRO)
- **DOGE** (2024): domain-optimised mixture for downstream tasks
- **SlimPajama** (2023): data deduplication analysis; removing duplicates improves scaling

Key insight: optimal mixing weights **change with scale**. At small scale: diverse data helps; at large scale: domain-specific data (code, math) becomes more valuable.

| Data Source       | Small Model Weight | Large Model Weight | Direction |
| ----------------- | :----------------: | :----------------: | --------- |
| Web text          |        70%         |        50%         | Decreases |
| Books             |        10%         |        10%         | Stable    |
| Code              |        10%         |        25%         | Increases |
| Scientific papers |         5%         |        10%         | Increases |
| Math              |         5%         |         5%         | Stable    |

### 6.5 The Data Wall Hypothesis

High-quality unique text data on the internet is **finite**:

| Estimate Source           | Available Unique Tokens    | Notes                        |
| ------------------------- | -------------------------- | ---------------------------- |
| Villalobos et al. (2022)  | 10–100T (high-quality)     | Human-generated text         |
| Common Crawl (total)      | ~200T (mostly low quality) | Unfiltered web               |
| Current frontier training | 15–20T per model           | Already significant fraction |
| Projected exhaustion      | 2026–2030                  | At current growth rate       |

Responses to the data wall:

1. **Synthetic data generation** (§6.6)
2. **Multi-epoch training** with careful scheduling
3. **Multimodal data** (images, video, audio)
4. **Data quality improvements** (better filtering = more effective tokens)
5. **Curriculum learning** (order data for maximum learning signal)

### 6.6 Synthetic Data Scaling

Generate training data using existing LLMs; filter for quality:

| Approach             | Method                                       | Example                  | Quality                    |
| -------------------- | -------------------------------------------- | ------------------------ | -------------------------- |
| **Self-play**        | Model generates data; trains on own outputs  | STaR, ReST               | Good for reasoning         |
| **Distillation**     | Strong model generates data for weaker model | Alpaca, Orca             | Good for instruction       |
| **Textbook-quality** | Curated synthetic for specific domains       | Phi series (Microsoft)   | Excellent for small models |
| **Instruction data** | Synthetic instruction-response pairs         | WizardLM, Evol-Instruct  | Good for chat              |
| **Code generation**  | Models generate code + tests                 | CodeAlpaca, OSS-Instruct | Good for code              |

**Risk: model collapse**. Training on own outputs across generations: distribution narrows; tail knowledge lost. Shumailov et al. (2023): after ~5 generations, quality degrades significantly.

Mitigation: mix synthetic data with real data (10–30% synthetic max); use stronger teacher model for generation.

---

## 7. Emergent Abilities

### 7.1 Definition

An **emergent ability** is a capability absent in smaller models and present in larger models — not predicted by smooth extrapolation of the loss curve; appears discontinuously at scale.

Wei et al. (2022) catalogued dozens of emergent abilities appearing at specific compute thresholds. "Emergent" because they were not present at smaller scale: qualitatively different behaviour, not just quantitatively better.

### 7.2 Examples of Emergent Abilities

| Ability                    | Approximate Emergence Scale     | Benchmark           |
| -------------------------- | ------------------------------- | ------------------- |
| 3-digit addition           | ~$10^{22}$ FLOPs / ~10B params  | GSM8K, arithmetic   |
| Chain-of-thought reasoning | ~$10^{23}$ FLOPs / ~100B params | various             |
| Instruction following      | ~$10^{23}$ FLOPs                | FLAN tasks          |
| Multi-step reasoning       | ~$10^{23}$–$10^{24}$ FLOPs      | BIG-Bench Hard      |
| Analogical reasoning       | ~$10^{23}$ FLOPs                | SCAN, analogy tasks |
| Code generation            | ~$10^{22}$ FLOPs                | HumanEval           |
| Translation (low-resource) | ~$10^{23}$ FLOPs                | FLORES              |
| Word unscrambling          | ~$10^{22}$ FLOPs                | BIG-Bench           |

```
EMERGENT ABILITIES (SCHEMATIC)
═══════════════════════════════════════════════════════════════════════

  Accuracy (e.g., 3-digit addition)
  100% ┤                                    ●●●●●●  ← sudden jump
       │                                 ●●
   80% ┤                              ●●
       │                           ●
   60% ┤                        ●
       │
   40% ┤
       │
   20% ┤
       │
    0% ┤●  ●  ●  ●  ●  ●  ●  ●  ← seemingly no ability
       └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──▶
         10⁷ 10⁸ 10⁹ 10¹⁰        Parameters

  Key: ability appears to emerge suddenly at threshold scale
  Question: is this real, or an artefact of the metric?
```

### 7.3 The Metric Artefact Hypothesis (Schaeffer et al. 2023)

Schaeffer, Miranda & Koyejo argued that emergence is an artefact of **discontinuous evaluation metrics**, not a property of the model:

| Metric Type                    | Observation            | Explanation                                        |
| ------------------------------ | ---------------------- | -------------------------------------------------- |
| **Exact match** accuracy       | Sharp phase transition | 0 or 1 per sample. Averaging creates step function |
| **Token-level log-likelihood** | Smooth improvement     | Continuous metric; captures partial knowledge      |
| **Brier score**                | Smooth improvement     | Probabilistic; captures calibration                |
| **Partial credit**             | Smooth improvement     | Awards credit for partial correctness              |

**The argument**: accuracy on 3-digit addition is exactly 0% until the model gets all 3 digits right (exact match). With partial credit (e.g., fraction of digits correct), improvement is **smooth and predictable**.

```
SAME DATA, DIFFERENT METRICS
═══════════════════════════════════════════════════════════════════════

  Exact Match                    Partial Credit
  100% ┤         ●●●●             1.0 ┤              ●●●●
       │       ●●                     │           ●●●
   50% ┤     ●                    0.5 ┤       ●●●
       │   ●                          │    ●●●
    0% ┤● ●● ●                   0.0 ┤●●●
       └──────────▶ Scale             └──────────▶ Scale

  Same underlying model capability
  Different metric → different apparent scaling behaviour
```

### 7.4 Phase Transitions in Training

Some capabilities appear suddenly **during a single training run**:

| Phenomenon          | Description                                                 | Mechanism                       |
| ------------------- | ----------------------------------------------------------- | ------------------------------- |
| **Induction heads** | Attention heads that copy patterns; appear at specific step | Circuit formation               |
| **Grokking**        | Model suddenly generalises after extended overfit           | Regularisation delayed learning |
| **Loss spikes**     | Sudden loss drops during training                           | Feature learning transitions    |

Mechanism: internal circuits form discretely, not via gradual accumulation. Even if loss decreases smoothly, internal representations may reorganise at discrete steps.

### 7.5 Inverse Scaling

Some tasks get **worse** as models get larger:

| Task                  | Effect                                       | Explanation                                    |
| --------------------- | -------------------------------------------- | ---------------------------------------------- |
| **TruthfulQA**        | Larger models more confidently wrong         | Learn common misconceptions from training data |
| **Sycophancy**        | Larger models more likely to agree with user | Learn to be agreeable from RLHF                |
| **Hindsight neglect** | Larger models worse at ignoring hindsight    | Over-reliance on most common patterns          |

Not all scaling is positive; emergent **negative** capabilities exist. Inverse scaling prize (2022) identified dozens of such tasks.

### 7.6 U-Shaped Scaling

Some tasks: performance **decreases then increases** with scale.

- BIG-Bench: ~10% of tasks show U-shaped scaling
- Interpretation: models first learn a **wrong heuristic** (memorised pattern); at larger scale, develop capability to override it
- Example: a model might initially learn to answer "yes" to most questions (high accuracy on yes-biased benchmarks); then at larger scale, learn to actually reason

```
U-SHAPED SCALING
═══════════════════════════════════════════════════════════════════════

  Accuracy
  80% ┤●                                    ●●●●●●
      │  ●                               ●●●
  60% ┤    ●                          ●●●
      │      ●●                   ●●●
  40% ┤         ●●●          ●●●●
      │             ●●●●●●●●●  ← "valley" where wrong heuristic
  30% ┤                         dominates
      └────────────────────────────────────────▶ Scale

  Phase 1: correct by chance or simple heuristic
  Phase 2: wrong heuristic dominates; performance drops
  Phase 3: model powerful enough to learn correct strategy
```

---

## 8. Compute-Optimal Scaling — Full Mathematical Treatment

### 8.1 Problem Setup

Given: compute budget $C$ (total FLOPs).

Choose: $N$ (parameters) and $D$ (training tokens).

Objective: minimise $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$

Subject to: $6ND = C$

The constraint expresses that with a fixed compute budget, increasing $N$ requires decreasing $D$, and vice versa.

### 8.2 Lagrange Multiplier Derivation

**Step 1**: Form the Lagrangian.

$$\mathcal{L}(N, D, \lambda) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta} + \lambda(6ND - C)$$

**Step 2**: First-order conditions (set partial derivatives to zero).

$$\frac{\partial \mathcal{L}}{\partial N} = -\frac{A\alpha}{N^{\alpha+1}} + 6\lambda D = 0 \quad \Rightarrow \quad \lambda = \frac{A\alpha}{6DN^{\alpha+1}} \tag{1}$$

$$\frac{\partial \mathcal{L}}{\partial D} = -\frac{B\beta}{D^{\beta+1}} + 6\lambda N = 0 \quad \Rightarrow \quad \lambda = \frac{B\beta}{6ND^{\beta+1}} \tag{2}$$

$$\frac{\partial \mathcal{L}}{\partial \lambda} = 6ND - C = 0 \quad \Rightarrow \quad D = \frac{C}{6N} \tag{3}$$

**Step 3**: Equate (1) and (2) to eliminate $\lambda$.

$$\frac{A\alpha}{6DN^{\alpha+1}} = \frac{B\beta}{6ND^{\beta+1}}$$

$$\frac{A\alpha}{DN^{\alpha+1}} = \frac{B\beta}{ND^{\beta+1}}$$

$$A\alpha \cdot N \cdot D^{\beta+1} = B\beta \cdot D \cdot N^{\alpha+1}$$

$$A\alpha \cdot D^\beta = B\beta \cdot N^\alpha$$

$$\boxed{\frac{A\alpha}{N^\alpha} = \frac{B\beta}{D^\beta}} \tag{4}$$

**Interpretation**: at the optimum, the **marginal return from adding a parameter** (scaled by $\alpha$) equals the **marginal return from adding a token** (scaled by $\beta$). This is the equal marginal benefit condition.

**Step 4**: Solve for $N^*$ using constraint (3) and optimality (4).

From (4): $D^\beta = \frac{B\beta}{A\alpha} N^\alpha$

From (3): $D = \frac{C}{6N}$, so $D^\beta = \left(\frac{C}{6N}\right)^\beta = \frac{C^\beta}{6^\beta N^\beta}$

Equating:

$$\frac{C^\beta}{6^\beta N^\beta} = \frac{B\beta}{A\alpha} N^\alpha$$

$$N^{\alpha+\beta} = \frac{A\alpha \cdot C^\beta}{B\beta \cdot 6^\beta}$$

$$\boxed{N^* = \left(\frac{A\alpha}{B\beta}\right)^{1/(\alpha+\beta)} \cdot \left(\frac{C}{6}\right)^{\beta/(\alpha+\beta)}}$$

**Step 5**: Solve for $D^*$.

$$D^* = \frac{C}{6N^*} = \frac{C}{6} \cdot \left(\frac{B\beta}{A\alpha}\right)^{1/(\alpha+\beta)} \cdot \left(\frac{C}{6}\right)^{-\beta/(\alpha+\beta)}$$

$$\boxed{D^* = \left(\frac{B\beta}{A\alpha}\right)^{1/(\alpha+\beta)} \cdot \left(\frac{C}{6}\right)^{\alpha/(\alpha+\beta)}}$$

### 8.3 Scaling Exponents

Define:

$$a = \frac{\beta}{\alpha+\beta}, \quad b = \frac{\alpha}{\alpha+\beta}, \quad a+b=1$$

Then $N^* \propto C^a$ and $D^* \propto C^b$.

| Scaling Law    | $\alpha$ | $\beta$ | $a$ ($N$ exponent) | $b$ ($D$ exponent) | $D^*/N^*$         |
| -------------- | :------: | :-----: | :----------------: | :----------------: | ----------------- |
| **Kaplan**     |  0.076   |  0.095  |        0.56        |        0.44        | Decreasing with C |
| **Chinchilla** |   0.34   |  0.28   |        0.45        |        0.55        | ~20 (constant)    |

Chinchilla's larger exponents ($\alpha, \beta$) yield $a \approx b \approx 0.5$ (equal scaling). Kaplan's smaller exponents + the choice $\alpha_N/\alpha_D \approx 0.8$ distorted the ratio toward parameters.

### 8.4 Optimal Loss as Function of Compute

Substituting $N^*$ and $D^*$ back into $L$:

At optimum, from (4): $\frac{A}{N^{*\alpha}} = \frac{\beta}{\alpha} \cdot \frac{B}{D^{*\beta}}$

Combined contribution:

$$L^*(C) = E + \frac{A}{N^{*\alpha}} + \frac{B}{D^{*\beta}} = E + \left(1 + \frac{\alpha}{\beta}\right) \cdot \frac{B}{D^{*\beta}}$$

Since $D^* \propto C^{\alpha/(\alpha+\beta)}$:

$$\frac{B}{D^{*\beta}} \propto C^{-\alpha\beta/(\alpha+\beta)}$$

$$\boxed{L^*(C) = E + K \cdot C^{-\gamma}, \quad \gamma = \frac{\alpha\beta}{\alpha+\beta}}$$

|                                |               Chinchilla                |              Kaplan              |
| ------------------------------ | :-------------------------------------: | :------------------------------: |
| $\gamma$                       | $(0.34 \times 0.28)/0.62 \approx 0.154$ |              ~0.050              |
| Loss reduction per 10× compute |    $10^{-0.154} \approx 0.70$ (30%)     | $10^{-0.050} \approx 0.89$ (11%) |

Chinchilla‐optimal training extracts **3× more loss reduction** per 10× compute than Kaplan's approach.

### 8.5 Verification and Numerical Example

**Example**: $C = 10^{24}$ FLOPs

With Chinchilla parameters ($A=406.4$, $B=410.7$, $\alpha=0.34$, $\beta=0.28$, $E=1.69$):

$$N^* = \left(\frac{406.4 \times 0.34}{410.7 \times 0.28}\right)^{1/0.62} \cdot \left(\frac{10^{24}}{6}\right)^{0.45}$$

$$= (1.20)^{1.61} \times (1.67 \times 10^{23})^{0.45}$$

$$= 1.33 \times 5.04 \times 10^{10} \approx 6.7 \times 10^{10} = 67\text{B}$$

$$D^* = \frac{C}{6N^*} = \frac{10^{24}}{6 \times 6.7 \times 10^{10}} = \frac{10^{24}}{4.02 \times 10^{11}} \approx 2.5 \times 10^{12} \approx 1.4\text{T}$$

Verification: $C = 6 \times 67\text{B} \times 1.4\text{T} = 6 \times 9.38 \times 10^{22} \approx 10^{24}$ ✓

$D^*/N^* = 1.4\text{T}/67\text{B} \approx 21 \approx 20$ ✓

---

## 9. Scaling Laws for Downstream Tasks

### 9.1 From Loss to Accuracy

Cross-entropy loss is continuous; benchmark accuracy is discrete. The relationship between loss and task accuracy is typically sigmoid-shaped:

$$\text{acc}(L) \approx \sigma\left(\frac{L_0 - L}{\tau}\right) = \frac{1}{1 + e^{-(L_0 - L)/\tau}}$$

where $L_0$ is the loss threshold for 50% accuracy and $\tau$ controls the steepness.

Small loss improvements can yield large accuracy jumps near the inflection point (where $L \approx L_0$).

```
LOSS → ACCURACY RELATIONSHIP
═══════════════════════════════════════════════════════════════════════

  Task Accuracy
  100% ┤                          ●●●●●●●●●●●●
       │                      ●●●●
   80% ┤                   ●●●
       │                ●●●       ← steepest region
   60% ┤             ●●●            (small ΔL → large Δacc)
       │          ●●●
   40% ┤       ●●●
       │    ●●●
   20% ┤  ●●
       │●●
    0% ┤●
       └────────────────────────────────────────▶
       3.0    2.5    2.0    1.5    1.0   Loss (decreasing →)

  Smooth loss improvement → sigmoid-like accuracy curve
  Explains why "emergence" appears with discontinuous metrics
```

### 9.2 Task-Specific Scaling Exponents

Different tasks scale at different rates:

| Task Category               | Scaling Behaviour             | Effective $\alpha_{\text{task}}$ | Notes                                 |
| --------------------------- | ----------------------------- | :------------------------------: | ------------------------------------- |
| Easy factual recall         | Saturates quickly             |              ~0.15               | Diminishing returns at moderate scale |
| Hard reasoning              | Slow, then fast               |        ~0.05, then jumps         | "Emergent"-like behaviour             |
| Translation (high-resource) | Smooth                        |              ~0.10               | Consistent improvement                |
| Translation (low-resource)  | Delayed then fast             |        ~0.03, then ~0.12         | Threshold behaviour                   |
| Code generation             | Smooth                        |              ~0.10               | Consistent with scale                 |
| Math                        | Slow start, then accelerating |        ~0.04, then ~0.08         | Benefits from reasoning capability    |
| Summarisation               | Quick saturation              |              ~0.12               | Easier task                           |

Each task can be characterised by its own scaling curve and effective loss-to-accuracy mapping.

### 9.3 Scaling Laws for Code (Chen et al. 2021)

HumanEval pass@k metric: generate $k$ code samples, check if any passes all tests.

$$\text{pass@}k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} \approx 1 - (1-p)^k$$

where $n$ = total generated samples, $c$ = correct samples, $p = c/n$.

Code generation scales smoothly and predictably:

|  Model Size  | pass@1 | pass@10 | pass@100 |
| :----------: | :----: | :-----: | :------: |
|     300M     |  0.6%  |   3%    |   10%    |
|      1B      |   2%   |   8%    |   22%    |
|     12B      |  28%   |   47%   |   72%    |
| 175B (GPT-3) |  47%   |   75%   |   92%    |

### 9.4 Scaling Laws for Reasoning

Chain-of-thought reasoning quality scales with both model size **and** chain length:

$$L_{\text{reasoning}}(N, T) \propto N^{-\alpha_r} \cdot T^{-\delta}$$

$T$ = number of thinking tokens (reasoning chain length); $\delta \approx 0.1$–$0.3$.

Implication: **trading inference compute for capability** via longer reasoning chains (see §10).

| Model | Params | Without CoT | With CoT | CoT Improvement |
| ----- | ------ | :---------: | :------: | :-------------: |
| PaLM  | 8B     |     4%      |    6%    |       +2%       |
| PaLM  | 62B    |     17%     |   33%    |      +16%       |
| PaLM  | 540B   |     33%     |   58%    |      +25%       |

CoT benefit scales **super-linearly** with model size; small models don't benefit.

### 9.5 Transfer Scaling Laws (Hernandez et al. 2022)

How much pretraining data $D_{\text{pre}}$ is "worth" in fine-tuning data $D_{\text{fine}}$?

$$L_{\text{fine-tune}}(D_{\text{fine}}, D_{\text{pre}}) = L_0 + \frac{A}{(D_{\text{fine}} + \epsilon \cdot D_{\text{pre}})^\alpha}$$

- $\epsilon \approx 0$ for unrelated domains (pretraining on web text, fine-tuning on medical reports)
- $\epsilon \approx 0.01$–$0.1$ for weakly related domains
- $\epsilon \approx 1$ for closely related domains

Result: pretraining is an effective **data amplifier** for fine-tuning. 1T tokens of pretraining ≈ 1–100B tokens of fine-tuning data (depending on domain relatedness).

---

## 10. Test-Time Compute Scaling

### 10.1 The New Scaling Axis (2024–2026)

Traditional scaling: more training compute → better model (pretraining scaling). **Test-time scaling**: more inference compute per query → better answer.

- OpenAI o1 (2024): allocate variable tokens for "thinking" before answering
- DeepSeek-R1 (2025): RL-trained to reason; dramatic improvement with more thinking tokens
- New question: are test-time compute improvements governed by power laws?

```
TWO AXES OF SCALING
═══════════════════════════════════════════════════════════════════════

  Quality
    ▲
    │                     ●  ← more test-time compute
    │                  ●       (same model, more thinking)
    │               ●
    │            ●
    │         ●  ← base model
    │      ●
    │   ●  ← more training compute
    │●  (larger model, more data)
    └───────────────────────────────▶ Total compute

  Training compute: paid once; improves all queries
  Test-time compute: paid per query; improves that query
  Optimal: balance both types of scaling
```

### 10.2 Best-of-N Sampling (Snell et al. 2024)

Generate $N$ responses; select the best using a verifier or reward model.

$$\text{pass@1}_{\text{best-of-N}} \approx 1 - (1-p)^N$$

where $p$ = probability of correct answer per sample.

| $N$ | Effective pass@1 (if $p=0.3$) | Compute Cost |
| :-: | :---------------------------: | :----------: |
|  1  |              30%              |      1×      |
|  4  |              76%              |      4×      |
| 16  |             98.3%             |     16×      |
| 64  |            99.99%             |     64×      |

Compute-optimal crossover: at some $N$, a **larger base model** beats more samples from a smaller model. Snell et al. found this crossover at $N \approx 8$–$32$ for typical tasks.

### 10.3 Scaling Law for Chain-of-Thought Length

For math and reasoning tasks (observed in o1, R1, QwQ):

$$\text{acc}(T) \propto T^\delta, \quad \delta \approx 0.1\text{–}0.3$$

$T$ = number of reasoning/thinking tokens.

| Thinking Tokens | Relative Quality    | Compute Cost |
| :-------------: | ------------------- | :----------: |
|       100       | Baseline            |      1×      |
|       400       | ~$1.15$–$1.4\times$ |      4×      |
|      1,600      | ~$1.32$–$2.0\times$ |     16×      |
|      6,400      | ~$1.50$–$2.8\times$ |     64×      |

Diminishing returns: quality scales as $T^{0.1\text{–}0.3}$, but cost scales linearly with $T$.

### 10.4 Compute-Optimal Test-Time Allocation

Given fixed total compute budget $C_{\text{total}}$:

$$C_{\text{total}} = C_{\text{train}} + C_{\text{inference}} \times n_{\text{queries}}$$

$$= 6ND + (2N \cdot T + \text{overhead}) \times n_{\text{queries}}$$

Trade-offs:

- Larger model (more $C_{\text{train}}$) → needs less $T$ per query for same quality
- Smaller model + more thinking tokens → cheaper to train, more expensive per query
- Optimal depends on $n_{\text{queries}}$: high-traffic deployment favours larger model; low-traffic favours smaller model with more test-time compute

### 10.5 Verifier-Based Scaling

Generate $N$ candidate solutions; use verifier (reward model) to select best:

**Outcome Reward Model (ORM)**: scores complete answer.

**Process Reward Model (PRM)**: scores each reasoning step; better selection.

Best-of-N improvement from extreme value theory (assuming Gaussian score distribution):

$$\text{best score from } N \text{ samples} \approx \mu + \sigma\sqrt{2\ln N}$$

| $N$ | Expected Improvement ($\sigma$ units) | Compute Cost |
| :-: | :-----------------------------------: | :----------: |
|  1  |             0 (baseline)              |      1×      |
|  4  |           $\sim 1.67\sigma$           |      4×      |
| 16  |           $\sim 2.35\sigma$           |     16×      |
| 64  |           $\sim 2.88\sigma$           |     64×      |
| 256 |           $\sim 3.33\sigma$           |     256×     |

Very sublinear ($\sqrt{\ln N}$) — expensive to keep improving via sampling alone.

Key insight: **verifier quality bounds the gain**. Bad verifier → wrong selection → no improvement above ~$N = 16$.

```
VERIFIER QUALITY MATTERS
═══════════════════════════════════════════════════════════════════════

  Accuracy
  90% ┤                        ●●●●●●  Perfect verifier
      │                  ●●●●●●
  80% ┤             ●●●●●
      │       ●●●●●●         ●●●●●●●●●  Good verifier
  70% ┤  ●●●●●          ●●●●●
      │●●           ●●●●●
  60% ┤        ●●●●●●      ●●●●●●●●●●●●  Mediocre verifier
      │   ●●●●●        ●●●●●
  50% ┤●●●         ●●●●●
      │        ●●●●●
  40% ┤●●●●●●●●                          No verifier (random)
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──▶
         1  4  8  16 32 64 128    N (samples)
```

---

## 11. Scaling Laws for MoE Models

### 11.1 MoE Scaling Adjustment

Standard scaling laws use $N$ = total parameters. MoE models have **active** vs **total** parameters:

- $N_{\text{total}}$: all parameters across all experts (determines memory)
- $N_{\text{active}}$: parameters activated per token (determines compute cost)
- Top-$k$ routing: $k$ out of $E$ experts activated per token
- $N_{\text{active}} = N_{\text{shared}} + k \cdot N_{\text{expert}}$

| Component                 |           Per-Token FLOPs           |           Memory            |
| ------------------------- | :---------------------------------: | :-------------------------: |
| Attention layers (shared) |     $\propto N_{\text{shared}}$     |     $N_{\text{shared}}$     |
| Active experts            | $\propto k \cdot N_{\text{expert}}$ | $E \cdot N_{\text{expert}}$ |
| Router                    |             Negligible              |         Negligible          |
| **Total**                 |     $\propto N_{\text{active}}$     | $\propto N_{\text{total}}$  |

Compute cost $\propto N_{\text{active}}$; memory cost $\propto N_{\text{total}}$; $N_{\text{total}} \gg N_{\text{active}}$.

### 11.2 MoE Scaling Laws (Clark et al. 2022, Artetxe et al. 2021)

MoE with $E$ experts at fixed active compute:

$$L_{\text{MoE}}(N_{\text{active}}, E) < L_{\text{dense}}(N_{\text{active}})$$

MoE consistently beats dense model with same active parameters. The improvement follows:

$$\boxed{L_{\text{MoE}} \propto (N_{\text{active}} \cdot E^\eta)^{-\alpha}, \quad \eta \approx 0.3\text{–}0.5}$$

Each expert adds information; but less than doubling active parameters would.

### 11.3 Effective Parameter Count

Define effective parameters for scaling comparison:

$$\boxed{N_{\text{eff}} = N_{\text{active}} \cdot E^\eta, \quad \eta < 1}$$

With $\eta = 0.4$:

| Configuration     | $N_{\text{active}}$ | Experts ($E$) | $N_{\text{eff}}$                            | Equivalent Dense |
| ----------------- | ------------------- | :-----------: | ------------------------------------------- | ---------------- |
| Dense baseline    | 7B                  |       1       | 7B                                          | 7B               |
| MoE (8 experts)   | 7B                  |       8       | $7 \times 8^{0.4} = 7 \times 2.30 = 16.1$B  | ~16B             |
| MoE (64 experts)  | 7B                  |      64       | $7 \times 64^{0.4} = 7 \times 6.96 = 48.7$B | ~49B             |
| MoE (256 experts) | 7B                  |      256      | $7 \times 256^{0.4} = 7 \times 13.2 = 92$B  | ~92B             |

**Rule of thumb**: doubling experts ($E$) gains $\sim 2^{0.4} \approx 1.32\times$ effective parameters.

```
MoE SCALING ADVANTAGE
═══════════════════════════════════════════════════════════════════════

  Loss
  2.5 ┤●
      │  ●   Dense (all params active)
  2.0 ┤    ●●
      │        ●●●●
  1.8 ┤●             ●●●●●●
      │  ●   MoE (same active FLOPs,
  1.6 ┤    ●● more total params via experts)
      │       ●●●●●●●●●  ← same loss at LOWER active compute
  1.5 ┤
      └──────────────────────────────────────▶
                Active Compute per Token (FLOPs)

  DeepSeek-V3: 671B total, 37B active per token
  Comparable to ~200B+ dense model at 5× lower inference cost
```

### 11.4 Expert Count vs Expert Size

Fixed compute budget: many small experts or few large experts?

| More Experts (smaller each)     | Fewer Experts (larger each)   |
| ------------------------------- | ----------------------------- |
| ✓ Better specialisation         | ✓ Each expert is more capable |
| ✓ More routing options          | ✓ Simpler routing             |
| ✗ Higher communication overhead | ✗ Less specialisation         |
| ✗ Load balancing harder         | ✓ Easier to balance           |
| ✗ Higher memory bandwidth       | ✓ Lower memory overhead       |

Empirical finding: **more, smaller experts** generally better (up to a point).

| Research                  | Experts | Size Per Expert | Result                             |
| ------------------------- | :-----: | --------------- | ---------------------------------- |
| GShard (2021)             |  2,048  | Small           | Good but hard to distribute        |
| Switch Transformer (2021) | 64–128  | Medium          | Strong results                     |
| DeepSeek-MoE (2024)       |   256   | Fine-grained    | Strong results with shared experts |
| Mixtral (2024)            |    8    | Large           | Practical, good results            |

Diminishing returns beyond ~64–256 experts; practical constraint: communication overhead in distributed training.

---

## 12. Multimodal Scaling Laws

### 12.1 Vision-Language Scaling

Clark et al. (2022): unified scaling laws across text, images, video, audio.

Key findings:

- Power-law exponents are **similar** across modalities ($\alpha \approx 0.05$–$0.10$)
- Multimodal models: combined scaling with modality-specific terms

$$L_{\text{multimodal}} = E_{\text{multi}} + \frac{A_{\text{text}}}{N^\alpha} + \frac{A_{\text{img}}}{N^{\alpha_{\text{img}}}} + \frac{B}{D_{\text{total}}^\beta}$$

| Modality |       Scaling Exponent       | Data Requirement | Token Representation      |
| -------- | :--------------------------: | ---------------- | ------------------------- |
| Text     |    $\alpha \approx 0.076$    | BPE tokens       | 1 token = ~4 chars        |
| Images   | $\alpha \approx 0.05$–$0.08$ | Image patches    | 1 image = 256–1024 tokens |
| Video    | $\alpha \approx 0.04$–$0.06$ | Frame patches    | 1 sec = 1000+ tokens      |
| Audio    | $\alpha \approx 0.06$–$0.08$ | Audio tokens     | 1 sec = 25–50 tokens      |

### 12.2 Image Token Scaling

Images tokenised into patches; scaling law applies to image tokens:

- More tokens per image → finer visual detail → better understanding
- Resolution scaling: $L \propto (\text{patches})^{-\alpha_{\text{img}}}$; $\alpha_{\text{img}} \approx 0.05$–$0.10$

| Image Resolution | Patches (ViT-B/16) | Relative Loss |
| :--------------: | :----------------: | ------------- |
|     224×224      |        196         | Baseline      |
|     384×384      |        576         | ~8% lower     |
|     512×512      |        1024        | ~12% lower    |
|    1024×1024     |        4096        | ~18% lower    |

Diminishing returns: 4× resolution → ~6% loss improvement. Cost scales quadratically with resolution.

### 12.3 Cross-Modal Transfer

Key insight: modalities benefit from each other.

- Text pretraining helps image understanding (concepts transfer)
- Image data helps text models (visual grounding)
- Transfer coefficient similar to text-to-text domain transfer
- Joint training typically better than separate training for each modality

Practical implication: multimodal models should be trained on **all modalities simultaneously** rather than sequentially; joint scaling is more compute-efficient.

---

## 13. Practical Scaling Law Estimation

### 13.1 How to Fit Your Own Scaling Laws

```
SCALING LAW WORKFLOW
═══════════════════════════════════════════════════════════════════════

Step 1: Train small proxy models (10M – 1B params)
  ├── 5–10 model sizes spanning at least 2 orders of magnitude
  ├── Each trained at 3+ different token counts
  ├── Same architecture family as target model
  └── Total cost: ~$1,000–$10,000 (1000× cheaper than flagship)

Step 2: Fit scaling law
  ├── Model: L(N, D) = A/N^α + B/D^β + E
  ├── Method: non-linear least squares (scipy.optimize.curve_fit)
  ├── Alternative: log-linear regression on log-log transformed data
  └── Quality check: R² > 0.99 on training data; good fit on log-log plot

Step 3: Validate
  ├── Train a ~3B model (not used in fitting)
  ├── Compare predicted loss vs actual loss
  └── Acceptable error: < 5% relative error

Step 4: Extrapolate to target
  ├── Input: target compute budget C
  ├── Output: optimal N*, D*, predicted loss L*
  ├── Report: confidence interval from parameter uncertainty
  └── Check: N* and D* should be feasible (enough data? enough GPUs?)

Step 5: Train flagship model
  ├── Use predicted N*, D*
  ├── Monitor loss curve against prediction
  ├── If loss deviates > 10%: diagnose (data quality? LR schedule? bugs?)
  └── Cost: $1M–$100M+ (but well-planned)
```

### 13.2 IsoFLOP Profiles

Fix $C$; vary $N$ and $D$ subject to $ND = C/6$; measure final loss for each $(N, D)$ pair.

Plot loss vs $N$ at fixed $C$: reveals optimal $N^*$ as the minimum of a U-shaped curve.

```
IsoFLOP PROFILE
═══════════════════════════════════════════════════════════════════════

  Loss                        C = 10²¹ FLOPs
  2.8 ┤●                                     ●
      │  ●                                 ●
  2.6 ┤    ●                             ●
      │      ●                        ●
  2.4 ┤        ●                   ●
      │          ●●             ●●
  2.2 ┤            ●●        ●●
      │              ●●●  ●●
  2.0 ┤                 ★  ← optimal N* for this C
      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──▶
        10⁷ 10⁸     10⁹     10¹⁰     10¹¹
                        N (params)

  Left of minimum: model too small (data-rich, parameter-poor)
  Right of minimum: model too large (parameter-rich, data-poor)
  Minimum: compute-optimal allocation
```

Repeat for multiple $C$ values; fit $N^*(C)$ and $D^*(C)$ scaling laws.

**Most rigorous method**: requires many training runs but gives reliable scaling law estimates. This is the method Chinchilla used.

### 13.3 Extrapolation Reliability

| Extrapolation Distance | Reliability                                   | Action                           |
| ---------------------- | --------------------------------------------- | -------------------------------- |
| 1–10× beyond fit range | High; power law holds                         | Trust prediction                 |
| 10–100×                | Moderate; increasing uncertainty              | Use confidence intervals         |
| 100–1000×              | Low; architecture changes may shift constants | Validate with intermediate model |
| >1000×                 | Very low; fundamental changes possible        | Treat as rough estimate only     |

**Best practice**: train proxy models at 0.001×, 0.01×, 0.1× of target compute; fit and extrapolate. Maximum ~1000× extrapolation.

### 13.4 Confidence Intervals

1. **Bootstrap resampling**: resample training runs; refit scaling law; compute confidence bands
2. **Profile likelihood**: vary each parameter; find confidence region where log-likelihood is within $\chi^2$ threshold
3. **Propagated uncertainty**: uncertainty in $\alpha$ → uncertainty in $N^*$; grows with extrapolation distance

Typical uncertainties from Chinchilla-scale fits:

| Parameter | Point Estimate | Standard Error | Relative Uncertainty |
| --------- | :------------: | :------------: | :------------------: |
| $\alpha$  |      0.34      |     ±0.02      |         ±6%          |
| $\beta$   |      0.28      |     ±0.02      |         ±7%          |
| $E$       |      1.69      |     ±0.05      |         ±3%          |

At 100× extrapolation: $N^*$ uncertain by factor ~1.5×; $L^*$ uncertain by ~±0.1 nats.

### 13.5 The μP Advantage for Proxy Tuning

Standard problem: hyperparameters (learning rate, $\beta_2$, init scale) optimised for small proxy models **don't transfer** to large models.

**μP parametrisation** (Yang et al. 2021): choose parametrisation so optimal hyperparameters transfer exactly:

| Hyperparameter | Standard Parametrisation | μP Parametrisation     |
| -------------- | ------------------------ | ---------------------- |
| Learning rate  | Retune at each scale     | **Transfers directly** |
| Init scale     | Retune at each scale     | **Transfers directly** |
| $\beta_2$      | Retune at each scale     | **Transfers directly** |

Result: tune hyperparameters on a tiny model (e.g., 40M params); apply directly to 7B+ model. No expensive hyperparameter search at scale.

Cost savings: hyperparameter tuning at large scale costs 5–10× base training. μP eliminates this.

---

## 14. Limitations and Critiques

### 14.1 What Scaling Laws Don't Capture

| Factor                        | Why It Matters                                            | Scaling Law Coverage    |
| ----------------------------- | --------------------------------------------------------- | ----------------------- |
| **Architecture efficiency**   | Different architectures at same $N$ → different loss      | Not captured            |
| **Data quality**              | Same $N$, $D$ but different quality → very different loss | Partially captured (§6) |
| **Tokeniser**                 | Different tokenisers → different effective $D$            | Not captured            |
| **Training stability**        | Large models may fail to train (loss spikes, NaN)         | Not captured            |
| **Task-specific performance** | Loss $\neq$ downstream accuracy                           | Loosely captured (§9)   |
| **Emergent abilities**        | Discontinuous; not predicted by smooth loss               | Not captured            |
| **Alignment**                 | Capability $\neq$ helpfulness or safety                   | Not captured            |
| **Memorisation**              | Model may memorise training data                          | Not captured            |

### 14.2 The Benchmark Contamination Problem

Large training corpora may contain test set data → **inflated benchmark scores**.

- Not predicted by scaling laws; breaks fair comparison
- Contamination increases with $D$ (more tokens → higher overlap probability)
- Solutions: contamination-free benchmarks (LiveBench 2024); dynamic evaluation; held-out test sets
- Active problem: exact contamination analysis difficult for closed-source models

### 14.3 Distribution Shift

- Scaling laws fitted on specific data distribution (e.g., MassiveText, WebText2)
- Change distribution (web → code → medical → legal): scaling constants shift
- Exponents may also change across domains
- Domain-specific scaling laws needed for specialised models

### 14.4 The End of Scaling Debate

**Arguments for continued scaling:**

- No breakdown observed across 7+ orders of magnitude
- New axes of scaling (test-time compute, synthetic data)
- Architectural improvements (MoE) extend effective scale

**Arguments for scaling limits:**

- Data wall: running out of unique high-quality text
- Diminishing returns: small exponents ($\alpha \approx 0.05$–$0.34$) mean enormous cost for modest improvement
- Energy and cost constraints: training frontier models already costs $100M+
- No guarantee power laws hold beyond observed range
- Possible fundamental limits unrelated to resources (reasoning, planning)

2025–2026 consensus: scaling continues to work but at increasing cost; efficiency improvements (architecture, data, algorithms) may matter as much as raw scale.

### 14.5 Beyond Loss — What we Really Care About

| What Scaling Laws Predict           | What We Actually Want                   |
| ----------------------------------- | --------------------------------------- |
| Cross-entropy loss on held-out data | Helpful, harmless, honest responses     |
| Next-token prediction accuracy      | Reasoning, creativity, factual accuracy |
| Average performance                 | Worst-case safety guarantees            |
| Pre-training loss                   | Post-fine-tuning, post-RLHF performance |

A model with lower loss may be worse on specific tasks (distribution mismatch). Need **capability-specific scaling laws** for practical planning — active research area.

---

## 15. Common Mistakes

|  #  | Mistake                                         | Why It's Wrong                                                                      | Fix                                                                        |
| :-: | ----------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
|  1  | "Chinchilla optimal = best model"               | Optimal for training compute only; over-trained smaller models better for inference | Use inference-optimal allocation for deployed models                       |
|  2  | "Scaling laws are universal constants"          | Exponents depend on architecture, data distribution, tokeniser                      | Fit your own scaling laws; don't blindly apply Kaplan/Chinchilla constants |
|  3  | "Power laws extrapolate indefinitely"           | Emergent abilities, data walls, architecture changes can break extrapolation        | Validate with intermediate model; limit extrapolation to ~100×             |
|  4  | "Lower loss = better model"                     | Loss measures next-token prediction; tasks need task-specific eval                  | Always benchmark downstream; loss is a necessary but not sufficient metric |
|  5  | "More compute always beats better architecture" | Efficient architectures (MoE, SSM) can outperform at fixed compute                  | Compare at equal compute budget; consider effective parameters             |
|  6  | "Emergent abilities are real phase transitions" | May be metric artefacts; underlying capability scales smoothly                      | Use continuous metrics; check log-likelihood alongside accuracy            |
|  7  | "Repeated data is as good as new data"          | After ~4 epochs, diminishing returns significant ($R^{0.72}$ effective)             | Prioritise unique data; treat repeated epochs as last resort               |
|  8  | "Scaling laws apply directly to fine-tuning"    | Pretraining and fine-tuning follow different scaling laws                           | Use transfer scaling laws (Hernandez et al.) for fine-tuning estimates     |
|  9  | "$N$ includes embedding parameters"             | Standard convention excludes embeddings; including distorts comparisons             | Always specify: non-embedding parameters only                              |
| 10  | "Test-time compute doesn't follow scaling laws" | Early evidence shows power law in reasoning tokens ($T^\delta$)                     | Monitor o1/R1-style results; incorporate test-time into compute budget     |

---

## 16. Exercises

1. **Power law fitting** — given loss measurements: $L(1\text{B}) = 2.8$, $L(10\text{B}) = 2.4$, $L(100\text{B}) = 2.1$. Fit $L(N) = A \cdot N^{-\alpha}$. Estimate $L(1\text{T})$. How reliable is this extrapolation?

2. **Chinchilla optimal allocation** — given compute budget $C = 10^{23}$ FLOPs: compute $N^*$ and $D^*$ using the Chinchilla formula from §8.3. Verify that $C = 6N^*D^*$.

3. **Under-training ratio** — GPT-3: $N = 175$B params, $D = 300$B tokens. Compute (a) Chinchilla-optimal $D$ for this $N$; (b) the under-training ratio; (c) estimated loss improvement if GPT-3 had been trained to Chinchilla-optimal.

4. **IsoFLOP curve** — at $C = 10^{21}$ FLOPs, compute $L$ for three configurations: $(N = 1\text{B}, D = 167\text{B})$, $(N = 3\text{B}, D = 56\text{B})$, $(N = 10\text{B}, D = 17\text{B})$ using the Chinchilla loss formula. Which is most compute-efficient? Find optimal $N^*$.

5. **Inference-optimal trade-off** — Model A: 70B params, 1T token training. Model B: 7B params, 10T token training. Both achieve similar loss. If inference cost is proportional to $N$: (a) which is cheaper to serve? (b) by how much? (c) at what query volume does Model A's higher training cost get offset?

6. **Emergent ability metric** — design a continuous partial-credit metric for 3-digit addition (e.g., fraction of correct digits). Sketch the expected scaling curve with this metric vs exact-match accuracy. Which shows "emergence"?

7. **MoE effective parameters** — Dense model: $N = 7$B. MoE model: $N_{\text{active}} = 7$B, $E = 8$ experts, $\eta = 0.4$. (a) Compute $N_{\text{eff}}$ for the MoE model. (b) What dense model size is the MoE equivalent to? (c) What's the memory cost ratio?

8. **Scaling law extrapolation with error** — $\alpha_N = 0.076 \pm 0.01$, and loss at $N_0 = 1\text{B}$ is $L_0 = 3.0$. (a) Compute $L$ at $N = 1\text{T}$ using central estimate. (b) Compute $L$ at $N = 1\text{T}$ using $\alpha_N = 0.066$ and $\alpha_N = 0.086$. (c) Express the range as percentage uncertainty.

---

## 17. Why This Matters for AI (2026 Perspective)

| Aspect                     | Impact                                                                                                               |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **Research planning**      | Predict final loss before expensive runs; save millions in misallocated compute                                      |
| **Capability forecasting** | Extrapolate when frontier models reach human-level on specific benchmarks                                            |
| **Compute allocation**     | $N/D$ trade-off determines training efficiency; wrong split wastes 2–20× compute                                     |
| **Inference economics**    | Inference-optimal training (LLaMA philosophy) reduces serving cost by 10–100×                                        |
| **Safety planning**        | If emergent abilities are predictable, safety measures can be prepared in advance                                    |
| **Open source**            | Efficient scaling (Phi, LLaMA) makes frontier-quality models accessible on consumer hardware                         |
| **Data strategy**          | Data wall and data quality scaling laws guide when to invest in synthetic data vs curation                           |
| **Architecture research**  | Every proposed architecture must beat baseline on compute-normalised benchmark; scaling laws provide fair comparison |
| **Test-time compute**      | New scaling axis; may fundamentally change cost/capability trade-offs for reasoning tasks                            |
| **Investment decisions**   | Scaling law extrapolations underpin billion-dollar compute infrastructure and data center decisions                  |
| **Regulation**             | Governments use scaling laws (compute thresholds) to define "frontier" models requiring oversight                    |

---

## Conceptual Bridge

Scaling laws are the **map of the territory**: they tell you where you are, where you're going, and what it will cost to get there. Every other section of this curriculum — tokenisation, embeddings, attention, training, fine-tuning — feeds into the loss function that scaling laws describe.

This section showed that model performance follows remarkably predictable power laws in parameters, data, and compute. The Chinchilla correction fixed Kaplan's parameter-heavy bias, establishing the 20-tokens-per-parameter rule. Inference-optimal scaling then showed that even Chinchilla isn't optimal for deployment — we should train smaller models on more data. Emergent abilities, MoE scaling, test-time compute, and data quality all add nuance to the basic picture.

Next: **Efficient Attention and Inference** — how to make trained models serve queries efficiently. Where scaling laws meet real-world deployment economics: KV-cache optimization, attention approximations, speculative decoding, and quantisation.

```
This Section:
Parameters, Data, Compute → [SCALING LAWS] → Predicted Loss → Optimal N*, D*

Next Section:
Trained Model → [EFFICIENT INFERENCE] → Fast, Cheap Serving → Real-World Deployment
```

---

[← Fine-Tuning Math](../07-Fine-Tuning-Math/notes.md) | [Home](../../README.md) | [Efficient Attention and Inference →](../09-Efficient-Attention-and-Inference/notes.md)
