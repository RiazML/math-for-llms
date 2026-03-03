# Quantization and Distillation

[← Mixture of Experts and Routing](../10-Mixture-of-Experts-and-Routing/notes.md) | [Home](../../README.md) | [RAG Math and Retrieval →](../12-RAG-Math-and-Retrieval/notes.md)

---

## 1. Intuition

### 1.1 What Is Quantization?

Quantization reduces the numerical precision of model weights and activations.
Instead of storing each number in 32 bits (FP32) or 16 bits (BF16), use 8, 4, 2,
or even 1 bit.

Analogy: instead of recording audio at 24-bit studio quality, compress to 8-bit
telephone quality. The art: compress as much as possible while preserving enough
precision for the model to remain useful.

Two distinct benefits:

- **Smaller memory footprint** — more model fits in GPU → lower cost → more accessible AI
- **Faster computation** — quantization-aware hardware executes integer ops at higher throughput

Every byte saved = more model fits in GPU = lower cost = more accessible AI.

### 1.2 What Is Knowledge Distillation?

Distillation transfers knowledge from a large, expensive "teacher" model to a
smaller "student" model. Not just training the student on data — training it to
mimic the teacher's internal representations and output distributions.

Analogy: an expert professor (teacher) teaches a student not just what the answers
are but _how to think about problems_.

Key insight: the teacher's soft probability distribution over all classes contains
far more information than the hard one-hot label. "Paris" is the correct answer,
but the probabilities over "Lyon", "Rome", "Berlin" encode similarity structure
that one-hot labels discard.

Result: student models that punch above their weight — performing closer to the
teacher than training on data alone would achieve.

### 1.3 Why Both Matter for AI

The best AI models are too large to run on most hardware:

- GPT-4 class models: estimated 1–2T parameters; requires multiple high-end server GPUs
- Quantization + distillation together enable:
  - Running 70B-class models on a single consumer GPU
  - Running 7B-class models on a smartphone
  - 10× reduction in inference cost for the same quality
  - Enabling privacy-preserving on-device AI

2026 reality: quantization and distillation are not optional optimisations — they
are core to AI deployment.

### 1.4 The Quality-Efficiency Frontier

Every compression technique trades quality for efficiency:

| Technique               | Quality Impact        | Efficiency Gain       |
| ----------------------- | --------------------- | --------------------- |
| INT8 quantization       | <0.5 PPL increase     | 2× memory, ~2× speed  |
| INT4 (GPTQ/AWQ)         | 0.5–2.0 PPL increase  | 4× memory, ~3× speed  |
| 2-bit quantization      | 5–20 PPL increase     | 8× memory             |
| Distillation 70B→7B     | 5–15% benchmark loss  | 10× compute reduction |
| Combined (distil+quant) | 10–20% benchmark loss | 40× total compression |

**Pareto frontier:** the set of (quality, cost) points where no improvement in one
dimension is possible without degrading the other. Goal: push the Pareto frontier —
same quality at lower cost, or higher quality at same cost.

### 1.5 Historical Timeline

| Year    | Work                   | Key Contribution                                                             |
| ------- | ---------------------- | ---------------------------------------------------------------------------- |
| 2008    | Widrow & Kollár        | "Quantization Noise" book; signal processing foundations                     |
| 2015    | Hinton, Vinyals & Dean | "Distilling the Knowledge in a Neural Network" — modern distillation         |
| 2018    | Jacob et al.           | Quantization-aware training for integer inference (Google)                   |
| 2021    | Gholami et al.         | Survey of quantization methods for efficient inference                       |
| 2022    | Dettmers et al.        | LLM.int8() — mixed-precision INT8 for large language models                  |
| 2022    | Frantar et al.         | GPTQ — one-shot weight quantization using second-order information           |
| 2022    | Xiao et al.            | SmoothQuant — migrating quantization difficulty from activations to weights  |
| 2023    | Lin et al.             | AWQ — activation-aware weight quantization                                   |
| 2023    | Dettmers et al.        | QLoRA — 4-bit quantization + LoRA fine-tuning                                |
| 2024    | Ma et al.              | BitNet b1.58 — ternary weight LLMs trained from scratch                      |
| 2024    | Gu et al.              | MiniCPM; knowledge distillation for small efficient LLMs                     |
| 2024    | DeepSeek               | FP8 training + quantized inference; cost-efficient frontier model            |
| 2025–26 | Industry-wide          | Sub-4-bit quantization, quantization-aware pretraining, on-device 70B models |

### 1.6 Pipeline Position

```
Large Teacher Model
        ↓ (distillation)
Smaller Student Model
        ↓ (quantization)
Compressed Quantized Model
        ↓ (deployment)
Fast Efficient Inference on Target Hardware
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            THIS section
```

---

## 2. Quantization Fundamentals

### 2.1 Number Representation Systems

| Format       | Layout                       | Range        | Bytes | Use Case                   |
| ------------ | ---------------------------- | ------------ | ----- | -------------------------- |
| **FP32**     | 1 sign + 8 exp + 23 mantissa | ±3.4×10³⁸    | 4     | Training (legacy)          |
| **BF16**     | 1 sign + 8 exp + 7 mantissa  | Same as FP32 | 2     | Standard LLM training      |
| **FP16**     | 1 sign + 5 exp + 10 mantissa | ±65504       | 2     | Narrower range than BF16   |
| **FP8 E4M3** | 1 sign + 4 exp + 3 mantissa  | ±448         | 1     | Forward pass               |
| **FP8 E5M2** | 1 sign + 5 exp + 2 mantissa  | ±57344       | 1     | Gradients                  |
| **INT8**     | Signed 8-bit integer         | [−128, 127]  | 1     | Inference standard         |
| **INT4**     | Signed 4-bit integer         | [−8, 7]      | 0.5   | Memory-efficient inference |
| **INT2**     | Signed 2-bit                 | [−2, 1]      | 0.25  | Extreme compression        |
| **INT1**     | Binary {−1, +1}              | {−1, +1}     | 1 bit | Theoretical minimum        |

### 2.2 Uniform Quantization — Formal Definition

Map floating-point value $x \in [x_{\min}, x_{\max}]$ to integer $x_q \in [-2^{b-1}, 2^{b-1}-1]$:

$$x_q = \text{clamp}\!\left(\text{round}\!\left(\frac{x}{s} + z\right),\; -2^{b-1},\; 2^{b-1}-1\right)$$

- **Scale factor:** $s = \frac{x_{\max} - x_{\min}}{2^b - 1}$
- **Zero point:** $z = -\text{round}(x_{\min} / s)$ (integer offset for asymmetric quantization)
- **Dequantization:** $\hat{x} = s \cdot (x_q - z)$
- **Quantization error:** $\varepsilon = x - \hat{x}$; bounded by $|\varepsilon| \leq s/2$

### 2.3 Symmetric vs Asymmetric Quantization

**Symmetric** ($z = 0$):

- Range: $[-s \cdot 2^{b-1},\; s \cdot (2^{b-1}-1)]$
- Simpler arithmetic: $\hat{x} = s \cdot x_q$; no zero point subtraction at inference
- Slightly larger error when distribution is not symmetric around zero
- Standard for **weight** quantization (weights typically symmetric)

**Asymmetric** ($z \neq 0$):

- Range $[x_{\min}, x_{\max}]$ mapped exactly
- Better for activations which are often non-negative (after ReLU/GELU)
- Extra cost: zero point addition at every multiply-accumulate
- Standard for **activation** quantization

### 2.4 Quantization Granularity

| Granularity     | Description                               | Scale Factors                           | Quality                     |
| --------------- | ----------------------------------------- | --------------------------------------- | --------------------------- |
| **Per-tensor**  | Single $(s, z)$ for entire matrix         | 1                                       | Poorest                     |
| **Per-channel** | One $(s, z)$ per output channel/row       | $m$ for $W \in \mathbb{R}^{m \times n}$ | Good; standard for weights  |
| **Per-group**   | One $(s, z)$ per $g$ consecutive elements | $mn/g$                                  | Very good; GPTQ/AWQ default |
| **Per-token**   | One $(s, z)$ per token vector             | $T$ per batch                           | Best for activations        |

Group size $g$: smaller $g$ → better quality (finer scale factors); larger $g$ → less overhead.
Typical $g \in \{64, 128, 256\}$.

### 2.5 Quantization Error Analysis

- **Rounding error:** $\varepsilon_{\text{round}} \in [-s/2, s/2]$; uniform distribution
- **Clipping error:** occurs when $|x|$ > clipping range; potentially large
- **Total error:** $\varepsilon = \varepsilon_{\text{round}} + \varepsilon_{\text{clip}}$

Mean squared error for uniform quantization over range $[-R, R]$:

$$\text{MSE}_{\text{round}} = \frac{s^2}{12} = \frac{R^2}{3 \cdot 2^{2b}}$$

Each additional bit halves $s$, reducing MSE by 4×.

Clipping MSE depends on weight distribution tail; heavier tails → more clipping error.

### 2.6 Optimal Clipping Range

Tradeoff: wider range → less clipping but coarser quantization grid (larger $s$).
Narrower range → finer grid but more clipping of outliers.

Optimal clipping: minimise total $\text{MSE} = \text{MSE}_{\text{round}} + \text{MSE}_{\text{clip}}$.

| Distribution                        | Optimal Clip $c \cdot \sigma$ | Bit Width |
| ----------------------------------- | ----------------------------- | --------- |
| Gaussian $\mathcal{N}(0, \sigma^2)$ | $c \approx 2.5\text{–}3.0$    | INT8      |
| Laplacian (common for weights)      | $c \approx 2.83$              | INT4      |

**Entropy calibration:** find range that minimises KL divergence between original and
quantized distribution.

---

## 3. Post-Training Quantization (PTQ)

### 3.1 What Is PTQ?

Quantize a fully trained model without any further training:

- Only requires a small calibration dataset (typically 128–512 samples)
- No gradient computation; fast; practical for deployment
- Quality degrades more than QAT especially at low bits; acceptable for INT8, challenging for INT4

### 3.2 Calibration

Run calibration dataset through model; collect activation statistics per layer:

- Statistics collected: min, max, mean, variance, percentiles (99th, 99.9th)
- Purpose: determine appropriate scale factors and zero points per layer
- Calibration set size: 128–512 samples; larger = better statistics; diminishing returns
- **Distribution shift:** if calibration data ≠ inference data → poor quantization

### 3.3 Min-Max Calibration

Simplest method: $s = (\max(x) - \min(x)) / (2^b - 1)$.

- Captures full range but sensitive to outliers
- One extreme outlier inflates $s$; all other values quantized too coarsely
- **Percentile calibration:** use $p$-th percentile instead of absolute min/max (e.g. 99.9th)
- Reduces outlier sensitivity; small clipping error in exchange for finer grid

### 3.4 KL Divergence Calibration (Entropy Calibration)

Find scale $s$ that minimises KL divergence between original and quantized distribution:

$$s^* = \arg\min_s D_{KL}(P_{\text{original}} \| P_{\text{quantized}}(s))$$

- TensorRT default calibration method
- Better than min-max for asymmetric activation distributions
- Computationally more expensive; requires histogram computation

### 3.5 GPTQ — Generative Pre-Trained Transformer Quantization (Frantar et al. 2022)

One-shot post-training quantization using second-order information. Key insight:
minimise layer-wise **reconstruction error** rather than weight error:

$$\min_{\hat{W}} \|WX - \hat{W}X\|_F^2$$

Uses Hessian matrix $H = 2XX^\top$ to identify which weights are most sensitive.

**Optimal Brain Quantization (OBQ) foundation:**

- Quantize weights one at a time
- After each quantization, update remaining weights to compensate
- Weight update: $\delta w = -(w_q - w) / H^{-1}_{qq} \cdot H^{-1}_{:,q}$

### 3.6 GPTQ Algorithm — Step by Step

```
For each layer weight matrix W ∈ ℝᵈˣⁿ:
  1. Compute Hessian: H = 2XX^T + λI   (λ = damping factor)
  2. Compute Cholesky factorisation of H⁻¹
  3. For each column group (e.g. 128 columns at a time):
     a. Quantize column j:  w_q,j = quantize(w_j)
     b. Compute error:      δj = w_q,j − w_j
     c. Update remaining:   W[:,j+1:] += δj · (H⁻¹[j,j+1:] / H⁻¹[j,j])
  4. Return quantized W_q
```

- Complexity: $O(d \times n^2 / \text{block\_size})$ per layer
- Result: INT4 quantization with minimal quality loss vs BF16; 4× memory reduction

### 3.7 AWQ — Activation-Aware Weight Quantization (Lin et al. 2023)

Key observation: not all weights are equally important. **Salient weights** are
connected to channels with large activation magnitudes.

**Scaling trick:**

1. Find per-channel activation scale $s_i = \text{mean}(|X_i|)$ from calibration data
2. Scale weights: $\hat{W} = W \cdot \text{diag}(s)$; scale inputs: $\hat{X} = X \cdot \text{diag}(s)^{-1}$
3. Net effect: same output $WX = \hat{W}\hat{X}$; but salient channels now quantize more precisely
4. Optimal scale: $s^* = \arg\min \|Q(W \cdot \text{diag}(s)) \cdot \text{diag}(s)^{-1} X\|$

### 3.8 AWQ vs GPTQ Comparison

| Property             | GPTQ                       | AWQ                           |
| -------------------- | -------------------------- | ----------------------------- |
| **Method**           | Second-order weight update | Activation-guided scaling     |
| **Calibration cost** | Higher (Hessian)           | Lower (activation stats)      |
| **Quality at INT4**  | Very good                  | Slightly better on some tasks |
| **Speed**            | Slower calibration         | Faster calibration            |
| **Hardware**         | GPU required               | GPU or CPU                    |
| **Adoption**         | llama.cpp, TGI             | llama.cpp, MLX, Ollama        |

### 3.9 SmoothQuant (Xiao et al. 2022)

Problem: activations have outliers in specific channels; weights are smooth.
Quantizing activations naively: outlier channels dominate scale → poor precision elsewhere.

Solution: mathematically migrate quantization difficulty from activations to weights.

For linear layer $Y = XW$:

- Introduce per-channel scale $s$: $Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W) = \hat{X}\hat{W}$
- Choose $s$ to equalise difficulty: $s_i = \max(|X_i|)^\alpha / \max(|W_i|)^{1-\alpha}$
- $\alpha \in [0,1]$: migration strength; $\alpha=0.5$ balances equally

Both $\hat{X}$ and $\hat{W}$ now quantizable to INT8 without significant error.
Enables **W8A8** quantization (weights and activations both INT8).

### 3.10 LLM.int8() (Dettmers et al. 2022)

Observation: ~0.1% of activation values are extreme outliers (100× larger than typical).
These outliers appear in specific dimensions; consistent across tokens.

**Mixed-precision decomposition:**

1. Identify outlier dimensions (threshold = 6.0 typical)
2. Compute matrix multiply in two parts:
   - Outlier dimensions: FP16 matmul for those columns
   - Non-outlier dimensions: INT8 matmul for everything else
3. Combine results: $Y = Y_{\text{FP16}} + Y_{\text{INT8}}$

Result: near-lossless INT8 inference even for 175B models.
Memory: ~8 GB for 7B model (vs 14 GB BF16); 2× compression.

---

## 4. Quantization-Aware Training (QAT)

### 4.1 What Is QAT?

Simulate quantization effects during the training forward pass:

- Model learns to be robust to quantization noise _before_ deployment
- Significantly better quality than PTQ at the same bit width
- Cost: requires retraining or fine-tuning; more expensive than PTQ
- Essential for very low bit widths (2–3 bit) where PTQ quality is unacceptable

### 4.2 Fake Quantization

Quantize then immediately dequantize during forward pass:

$$\hat{x} = \text{dequantize}(\text{quantize}(x)) = s \cdot \text{round}(x/s)$$

- Model sees quantization effects; gradients flow through dequantized values
- Weights stored in FP32 during training; quantized at each forward pass
- Inference: apply real quantization; weights stored as integers; faster computation

### 4.3 Straight-Through Estimator (STE)

Problem: $\text{round}(\cdot)$ has zero gradient almost everywhere; backpropagation fails.

STE (Bengio et al. 2013): approximate gradient of quantize as identity function:

$$\frac{\partial \hat{x}}{\partial x} \approx \mathbb{1}[x_{\min} \leq x \leq x_{\max}]$$

- Gradient passes through quantization operation unchanged (within clipping range)
- Zero gradient outside clipping range (clipped values don't update)
- Widely used; simple; surprisingly effective despite being an approximation

### 4.4 Learned Step Size Quantization (LSQ — Esser et al. 2020)

Make scale factor $s$ a learnable parameter; optimise via gradient descent:

$$\frac{\partial \mathcal{L}}{\partial s} = \sum_i \frac{\partial \mathcal{L}}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial s}$$

$$\frac{\partial \hat{x}_i}{\partial s} = \begin{cases} -x_i/s + \text{round}(x_i/s) & \text{if clipped} \\ \text{round}(x_i/s) & \text{otherwise} \end{cases}$$

Scale and weights co-optimised; better than fixed scale calibration.
LSQ+: also learn zero point; further improvement for asymmetric distributions.

### 4.5 PACT — Parameterised Clipping Activation Quantization

Learn optimal clipping value $\alpha$ for activations:

$$x_{\text{clip}} = \begin{cases} x & 0 \leq x \leq \alpha \\ \alpha & x > \alpha \end{cases}$$

- $\alpha$ learned via gradient descent alongside model weights
- $L_2$ regularisation on $\alpha$: prevents $\alpha$ from growing too large
- Better than fixed percentile clipping; adapts to each layer's optimal range

### 4.6 QAT for LLMs — Practical Considerations

| Approach                  | Description                                                                           | Cost                                    |
| ------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------- |
| **Full QAT from scratch** | Train entire model with fake quantization throughout                                  | Highest; only for targeted small models |
| **QAT fine-tuning**       | Fine-tune pretrained model with fake quantization for 1–5% of pretraining steps       | Moderate; good quality recovery         |
| **Subset QAT**            | Only apply fake quantization to sensitive layers (first, last, attention projections) | Lowest; targeted recovery               |

---

## 5. Extreme Quantization

### 5.1 Binary Neural Networks (BNN)

Weights $w \in \{-1, +1\}$: 1 bit per weight.

$$\hat{W} = \alpha \cdot \text{sign}(W), \quad \alpha = \frac{1}{n}\|W\|_1$$

- Matrix multiply becomes XNOR + popcount: replace multiply-accumulate
- XNOR-Net (Rastegari et al. 2016): BNN with per-channel scaling; ImageNet competitive
- Limitation: significant quality gap vs full precision for LLMs; not yet practical for text

### 5.2 Ternary Neural Networks

Weights $w \in \{-1, 0, +1\}$: $\sim 1.58$ bits per weight ($\log_2 3 \approx 1.585$).

$$w_t = \begin{cases} +1 & w > \Delta \\ 0 & |w| \leq \Delta \\ -1 & w < -\Delta \end{cases}$$

- Optimal threshold: $\Delta \approx 0.7 \times \mathbb{E}[|w|]$ (empirical)
- Sparse multiply: zero weights skip computation; ~50% sparsity typical

### 5.3 BitNet (Wang et al. 2023)

Train LLM with 1-bit weights from scratch (not quantize a pretrained model):

- Replace LayerNorm with BitNorm (normalisation before binarization)
- AbsMax quantization for activations to INT8
- Learning rate and initialization adjusted for binary gradient flow
- Result: competitive with FP16 LLaMA at 7B+ scale on perplexity
- Inference: XNOR + popcount replaces matrix multiply; extreme energy efficiency

### 5.4 BitNet b1.58 (Ma et al. 2024)

Ternary weights $\{-1, 0, +1\}$; $\log_2 3 \approx 1.58$ bits per weight.

- Zero weight allowed; natural sparsity emerges
- Quantization: $\text{round}(W / \text{mean}(|W|))$; clip to $\{-1, 0, +1\}$
- Activations: INT8 per-token quantization
- Claimed to match FP16 LLaMA at 3B+ parameters

Energy: matrix multiply becomes addition/subtraction only. Theoretical savings:

| vs Format | Energy Savings |
| --------- | -------------- |
| vs INT8   | 3.7×           |
| vs FP16   | 71.4×          |

### 5.5 2-Bit Quantization

Weights in $\{-1.5, -0.5, +0.5, +1.5\}$ or non-uniform levels: 2 bits = 4 levels.

**QuIP# (Chee et al. 2023):** Hadamard incoherence processing + 2-bit quantization.

- Apply random Hadamard transform to weights before quantization
- Reduces outlier problem; weights become more Gaussian after transform
- 2-bit quality comparable to 4-bit standard methods

**AQLM (Egiazarian et al. 2024):** additive quantization for LLMs.

- Represent each weight as sum of $k$ codewords from learned codebooks
- 2-bit average with $k=2$ codebooks; better quality than scalar 2-bit

### 5.6 Vector Quantization for Weights

Quantize groups of weights jointly using a learned codebook:

- Codebook $C = \{c_1, c_2, \ldots, c_K\}$: $K$ learned centroid vectors in $\mathbb{R}^g$
- Each group of $g$ weights assigned to nearest centroid: $\arg\min_k \|w_{\text{group}} - c_k\|^2$
- Storage: $\log_2 K$ bits per group (vs $g \times b$ bits for scalar quantization)
- Rate: if $K=256$ and $g=8$: $\log_2 256 / 8 = 1$ bit per weight vs 8 bits scalar
- **Product quantization (PQ):** partition weight vector into subvectors; quantize each independently
- **Residual quantization (RQ):** apply PQ iteratively on residuals; AQLM uses this approach

---

## 6. Quantization Formats in Practice

### 6.1 GGUF Format (llama.cpp)

General-purpose quantization format for the llama.cpp ecosystem:

| Format  | Bits | Description                          | Quality                        |
| ------- | ---- | ------------------------------------ | ------------------------------ |
| Q4_0    | 4    | Symmetric per-block (block=32)       | Fast; acceptable               |
| Q4_K_M  | 4–6  | K-quants; mixed for sensitive layers | Best quality/speed             |
| Q5_K_M  | 5    | 5-bit variant                        | Better quality; larger         |
| Q8_0    | 8    | Symmetric; near-lossless             | Recommended when memory allows |
| IQ2_XXS | ~2   | Extreme compression                  | Significant quality loss       |

"K-quants": key layers (attention projections, embeddings) in higher precision; rest lower.
Standard for local inference: Ollama, LMStudio, Jan use GGUF.

### 6.2 GPTQ Format

- 4-bit group quantization using GPTQ algorithm
- Group size: 64 or 128 (quality vs overhead tradeoff)
- Asymmetric: separate scale and zero point per group
- Activation order: quantize weights in order of Hessian diagonal (activation magnitude)
- Used by: TheBloke models on HuggingFace; TGI; AutoGPTQ library

### 6.3 AWQ Format

- 4-bit group quantization using AWQ algorithm
- Per-channel scaling + group quantization; slightly better than GPTQ on most benchmarks
- Faster calibration: minutes vs hours for GPTQ
- Used by: vLLM native support; AutoAWQ library; LMDeploy

### 6.4 EXL2 Format

- Variable bit-width within model: sensitive layers in higher precision; others lower
- Allows e.g. 4.5 bits/weight average vs fixed 4-bit; better quality at same model size
- Optimises per-layer bit allocation to minimise reconstruction error
- ExLlamaV2 framework; popular for enthusiast/research inference

### 6.5 FP8 Inference (Production 2024–2026)

- E4M3 for weights and activations; hardware-accelerated on H100/H200/B200
- Static quantization: scale factors computed offline; no per-token overhead
- Dynamic quantization: per-token scales computed online; better accuracy; small overhead
- Block-wise FP8: per-128-element scaling; much better quality than tensor-wise
- DeepSeek-V3: FP8 throughout; TensorRT-LLM FP8 mode; standard for datacenter 2025+

### 6.6 Quantization for KV Cache

| Format        | Compression | Quality Impact                              |
| ------------- | ----------- | ------------------------------------------- |
| KV8 (INT8)    | 2× vs BF16  | Near-lossless                               |
| KV4 (INT4)    | 4× vs BF16  | 0.5–1% PPL degradation                      |
| FP8 KV (E4M3) | 2× vs BF16  | Near-lossless; hardware accelerated on H100 |

Per-token dynamic scaling: each token's K,V quantized with its own scale factor.
Challenge: keys have more outliers than values; different strategies per tensor.

---

## 7. Knowledge Distillation — Foundations

### 7.1 The Core Insight — Soft Labels

- Hard label for "Paris": one-hot $[0, 0, \ldots, 1, \ldots, 0]$
- Teacher soft label: $[0.001, 0.003, \ldots, 0.85, 0.08, 0.04, \ldots]$
- Soft label says: "Paris is most likely, but Lyon and Rome are plausible"
- This **"dark knowledge"** encodes similarity structure between outputs
- Training on soft labels: exponentially more information per example than hard labels

Hinton's insight (2015): knowledge in a neural network is not just the weights but
the output distributions.

### 7.2 Temperature Softening

Teacher logits $z$; student logits $\hat{z}$. Softmax at temperature $\tau$:

$$p_i = \frac{\exp(z_i / \tau)}{\sum_j \exp(z_j / \tau)}$$

| Temperature                   | Effect                                                          |
| ----------------------------- | --------------------------------------------------------------- |
| $\tau = 1$                    | Standard distribution; sharp peak at correct class              |
| $\tau > 1$                    | Softer distribution; more information in non-peak probabilities |
| $\tau \to \infty$             | Uniform distribution; all classes equally likely                |
| Typical: $\tau = 2\text{–}10$ | Amplifies "dark knowledge" in non-target outputs                |

### 7.3 Distillation Loss — Original Formulation (Hinton 2015)

$$\mathcal{L}_{\text{distil}} = \alpha\, \mathcal{L}_{\text{CE}}(y, y_{\text{true}}) + (1-\alpha)\, \tau^2\, \mathcal{L}_{\text{KL}}\!\left(\sigma(z_T/\tau) \,\|\, \sigma(z_S/\tau)\right)$$

- First term: standard cross-entropy with ground truth labels (hard targets)
- Second term: KL divergence between teacher and student soft distributions at temperature $\tau$
- $\tau^2$ scaling: compensates for reduced gradient magnitude at high temperature
- $\alpha = 0.1\text{–}0.5$ typical; balances task loss and distillation signal
- Without ground truth: $\alpha = 0$; purely match teacher

### 7.4 KL Divergence as Distillation Loss

$$\mathcal{L}_{\text{KL}} = D_{KL}(P_T \| P_S) = \sum_{v \in V} P_T(v) \log \frac{P_T(v)}{P_S(v)}$$

| Direction                   | Behaviour                                             |
| --------------------------- | ----------------------------------------------------- |
| Forward KL $D_{KL}(T \| S)$ | Student covers all teacher modes; may be oversmoothed |
| Reverse KL $D_{KL}(S \| T)$ | Student mode-seeking; sharper; may miss some modes    |
| Jensen-Shannon              | Symmetric; bounded in $[0, 1]$; sometimes preferred   |

### 7.5 Token-Level vs Sequence-Level Distillation

**Token-level:** match teacher and student probability at each token independently.

- Standard approach; computationally efficient
- Loss: $\sum_i D_{KL}(P_T(\cdot | \text{context}_i) \| P_S(\cdot | \text{context}_i))$

**Sequence-level:** match sequence probability $P(y_1, \ldots, y_n | x)$.

- Requires sampling from teacher; more expensive
- Better for capturing long-range dependencies
- Sequence KD (Kim & Rush 2016): student trained on teacher-generated sequences

---

## 8. Distillation Methods for LLMs

### 8.1 Standard Response Distillation

Teacher generates responses; student trained on those responses ("knowledge transfer
via synthetic data"):

1. Sample prompts from dataset
2. Teacher generates responses (one per prompt)
3. Student trained on (prompt, teacher_response) pairs with standard cross-entropy

Cheaper than soft-label distillation: no teacher forward pass per student training step.
Used by: Alpaca (GPT-3.5), Vicuna, WizardLM, Orca.

### 8.2 Soft-Label / Logit Distillation

Teacher and student both process same input; student matches teacher logit distribution:

- Requires running teacher forward pass for every training batch
- Most expensive but highest quality information transfer
- **Vocabulary alignment problem:** teacher and student may have different vocabularies
  - Solution: use same tokenizer; project logits to shared vocabulary
- **Top-k logit distillation:** only match top-$k$ teacher probabilities; reduces computation
  - $k = 10$ or $100$; captures most information; ignores long tail

### 8.3 Feature/Intermediate Distillation

Match not just final output but intermediate representations:

$$\mathcal{L}_{\text{feat}} = \sum_l \left\|h_S^{(l)} - W_l h_T^{(l')}\right\|_F^2$$

- $W_l$: learned linear transformation from teacher to student dimension (if $d_T \neq d_S$)
- Layer mapping strategies:
  - Uniform spacing: if teacher has $L$ layers, student $M$: match every $L/M$ layers
  - Last-$k$: match last $k$ teacher layers to last $k$ student layers
- PKD (Patient Knowledge Distillation): match every $k$-th teacher layer

### 8.4 Attention Transfer Distillation

$$\mathcal{L}_{\text{attn}} = \sum_l \sum_h \left\|A_{T,l,h} - A_{S,l,h}\right\|_F^2$$

- $A_{T,l,h} \in \mathbb{R}^{n \times n}$: teacher attention matrix at layer $l$, head $h$
- Head mapping: teacher may have more heads; average or select heads
- Relation-based: match attention relation matrices $R = \text{softmax}(QK^\top / \sqrt{d})$

### 8.5 Contrastive Distillation

$$\mathcal{L}_{\text{contrast}} = -\log \frac{\exp(\text{sim}(h_S, h_T) / \tau)}{\sum_j \exp(\text{sim}(h_S, h_{T,j}^-) / \tau)}$$

- Pull student representations closer to teacher for same input
- Push apart for different inputs
- CRD (Tian et al. 2020): contrastive representation distillation; strong for small students

### 8.6 GKD — Generalised Knowledge Distillation (Agarwal et al. 2023)

Key insight: standard distillation trains on teacher-preferred outputs; student may
not generate those. **On-policy distillation:** sample from student during training:

$$\mathcal{L}_{\text{GKD}} = \mathbb{E}_{y \sim \pi_S} \left[D_{KL}(P_T(\cdot|x,y_{<t}) \| P_S(\cdot|x,y_{<t}))\right]$$

Student trains on its own generation distribution; avoids exposure bias.

### 8.7 MiniLLM (Gu et al. 2023)

Reverse KL distillation: minimise $D_{KL}(P_S \| P_T)$ instead of $D_{KL}(P_T \| P_S)$.

- Forward KL: student must cover all teacher modes; oversmoothed for generation
- Reverse KL: student is mode-seeking; generates sharper outputs
- REINFORCE-based optimisation:

$$\nabla \mathcal{L} = \mathbb{E}_{y \sim P_S}\left[\nabla \log P_S(y) \cdot \log \frac{P_S(y)}{P_T(y)}\right]$$

Better for open-ended text generation.

### 8.8 DistiLLM (Ko et al. 2024)

Skew divergence: interpolate between forward and reverse KL:

$$D_\alpha(P_T \| P_S) = D_{KL}(P_T \| \alpha P_S + (1-\alpha)P_T)$$

- $\alpha = 0$: zero loss; $\alpha = 1$: standard forward KL
- Better stability than pure reverse KL; better quality than pure forward KL

---

## 9. Distillation for Specific Capabilities

### 9.1 Reasoning Distillation

- Teacher generates chain-of-thought reasoning traces
- Student trained to reproduce reasoning _process_ not just final answer
- Step-by-step distillation: match teacher at each reasoning step
- Rationale distillation (Magister et al. 2022): GPT-3 generates explanations; small student learns
- DeepSeek-R1 (2025): large reasoning model distilled into 7B, 14B, 32B; strong math/code

### 9.2 Instruction Following Distillation

| Model                  | Teacher | Key Technique                            |
| ---------------------- | ------- | ---------------------------------------- |
| Alpaca (Stanford 2023) | GPT-3.5 | 52K instruction demonstrations           |
| Orca (Microsoft 2023)  | GPT-4   | System prompt augmentation; step-by-step |
| Orca-2 (2023)          | GPT-4   | Task-specific prompting strategies       |
| WizardLM               | GPT-4   | Evolve instructions simple → complex     |

### 9.3 Code Distillation

- Teacher (GPT-4, Claude) generates code solutions with explanations
- Student trained on (problem, explanation, code) triples
- **Execution-guided distillation:** filter teacher responses by correctness; only train on passing
- CodeLLaMA (2023): LLaMA distilled and fine-tuned on code; strong at fill-in-the-middle

### 9.4 Multilingual Distillation

- Cross-lingual distillation: align student and teacher representations across languages
- Language-neutral teacher embeddings: teacher's representation should be language-agnostic
- Same meaning in French and English should have similar student representations

### 9.5 Speculative Decoding as Implicit Distillation

- Draft model in speculative decoding is a form of distillation
- Draft model must approximate target distribution with acceptance rate $\alpha$
- Training draft model to maximise $\alpha$ ≡ minimising reverse KL $D_{KL}(P_{\text{draft}} \| P_{\text{target}})$
- EAGLE (2024): draft on target's hidden states; implicit feature distillation

---

## 10. Architecture Design for Distillation

### 10.1 Student Architecture Choices

| Reduction | What Changes              | Impact                  |
| --------- | ------------------------- | ----------------------- |
| Width     | Reduce $d_{\text{model}}$ | Fewer params per layer  |
| Depth     | Fewer layers $L$          | Less compute per token  |
| Heads     | Fewer attention heads $H$ | Less attention capacity |
| FFN       | Smaller $d_{ff}$          | Less FFN capacity       |
| Combined  | Reduce all proportionally | Standard approach       |

### 10.2 Capacity Gap Problem

Very large teacher → very small student: hard to distil. Student lacks capacity to
represent teacher's knowledge.

Solutions:

- **Teacher assistant**: distil giant → medium → small sequentially
- **Progressive distillation:** gradually shrink student in stages
- Mirzadeh et al. 2020: large teacher → medium assistant → small student; significantly
  better than direct

### 10.3 DistilBERT (Sanh et al. 2019)

- Student: 6 layers (vs BERT 12); 768 hidden dim same; 40% fewer parameters
- Triple distillation loss:
  - Soft logit distillation: match BERT output distribution
  - Cosine embedding distillation: match hidden state directions
  - MLM task loss: standard masked language modelling
- Layer initialisation: student layers from every other teacher layer
- Result: 97% of BERT performance; 60% faster; 40% smaller

### 10.4 TinyBERT (Jiao et al. 2020)

- Two-stage: pre-training distillation + task-specific distillation
- Generalised: match embedding, attention, hidden states at every layer
- Layer mapping with learned linear projections for dimension mismatch
- Result: 7.5× smaller; 9.4× faster than BERT; 96.8% performance

### 10.5 Small Language Models (SLMs) — 2024–2026

| Model                     | Size      | Key Technique                                       |
| ------------------------- | --------- | --------------------------------------------------- |
| Phi-2 (Microsoft 2023)    | 2.7B      | "Textbook" data; GPT-4 distillation                 |
| Phi-3 (Microsoft 2024)    | 3.8B      | Beats GPT-3.5; "textbooks are all you need"         |
| MiniCPM (2024)            | 2B–4B     | WSD schedule; careful data curation + distillation  |
| Gemma-2 (Google 2024)     | 2B–9B     | Distillation from Gemini; teacher logit training    |
| Qwen2.5 (2024)            | 0.5B–72B  | Smaller models distilled from larger family members |
| SmolLM (HuggingFace 2024) | 135M–1.7B | Distilled for on-device use                         |

Common pattern: all SLMs use some form of teacher data generation or logit distillation.

---

## 11. Quantization + Distillation Combined

### 11.1 QLoRA — Quantized Low-Rank Adaptation (Dettmers et al. 2023)

Quantize base model to 4-bit NF4; add LoRA adapters in BF16:

- Training: frozen 4-bit base + trainable BF16 LoRA adapters
- Gradient computation: dequantize weights for backward pass; update only LoRA
- **Double quantization:** quantize the quantization constants themselves
  - Block size 64: one FP32 scale per 64 weights = 0.5 bits overhead per weight
  - Double quantize: quantize FP32 scales to FP8; reduces to 0.127 bits overhead
- **Paged optimiser:** offload optimiser states to CPU when GPU memory full; swap in as needed
- Result: fine-tune LLaMA-3 70B on single A100 80GB; 4× memory reduction vs BF16 LoRA

### 11.2 NF4 — Normal Float 4-bit

Optimal quantization for normally-distributed data:

- Compute $2^b - 1 = 15$ quantile levels of $\mathcal{N}(0,1)$ distribution
- Use these as fixed quantization levels (not uniformly spaced)
- More levels near the mean (where most weights cluster); fewer in tails
- Theoretically optimal for Gaussian weights; better than INT4 for same bit budget

NF4 levels:

```
{−1, −0.6962, −0.5251, −0.3949, −0.2840, −0.1848, −0.0922, 0,
  0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7229, 1}
```

### 11.3 Quantization-Aware Distillation

Distil teacher into quantized student simultaneously:

$$\mathcal{L} = \alpha\, \mathcal{L}_{\text{CE}}(y, y_{\text{true}}) + \beta\, D_{KL}(P_T \| P_S^{\text{quant}}) + \gamma\, \mathcal{L}_{\text{feat}}$$

Joint optimisation: student learns both to match teacher AND be robust to quantization.
Result: significantly better than PTQ of distilled model.

### 11.4 Data-Free Quantization via Distillation

No real data needed: generate synthetic calibration data using teacher.

- Zero-shot quantization (Cai et al. 2020): calibration data from batch normalisation statistics
- GFPQ (2023): GAN-based synthetic data generation for quantization calibration
- Useful when real data unavailable (privacy)

### 11.5 Speculative Decoding + Quantization

- Quantize draft model aggressively (INT4); keep target model in higher precision (INT8/BF16)
- Draft model cheap, fast; quantization acceptable since exact match not required
- Acceptance rate $\alpha$ may decrease slightly; monitor carefully
- End-to-end: quantized draft + target = 3–5× speedup vs unquantized target alone

---

## 12. Evaluation and Quality Assessment

### 12.1 Perplexity as Quantization Metric

| Bit Width     | Typical PPL Increase | Quality               |
| ------------- | -------------------- | --------------------- |
| INT8          | < 0.5                | Near-lossless         |
| INT4 GPTQ/AWQ | 0.5–2.0              | Acceptable            |
| INT4 naive    | 5–50                 | Unacceptable          |
| 2-bit         | 10–100               | Highly task-dependent |

Limitation: PPL captures average case; task-specific degradation may be larger.

### 12.2 Task-Specific Benchmarks for Quantization

| Task Category                 | Sensitivity     | Why                               |
| ----------------------------- | --------------- | --------------------------------- |
| Math (GSM8K, MATH)            | Most sensitive  | Reasoning degrades first          |
| Code (HumanEval, MBPP)        | Sensitive       | Requires precise token prediction |
| Knowledge (MMLU)              | Moderate        | Factual recall less affected      |
| Common sense (HellaSwag, ARC) | Least sensitive | Simple pattern matching           |

Rule of thumb: if PPL increase < 1.0, task accuracy typically within 2–3% of original.

### 12.3 Distillation Evaluation

Key questions:

- Does student match teacher on **task**, not just loss value?
- Is student **calibrated** (does confidence match accuracy)?
- Does student generate as **diverse** outputs as teacher?
- Where specifically does student **underperform** teacher?

### 12.4 Sensitivity Analysis

Quantize one layer at a time; measure PPL increase per layer:

- **Most sensitive:** first and last layers (embedding + LM head), early transformer layers
- **Least sensitive:** middle layers of deep transformers
- Mixed-precision allocation: assign bits based on sensitivity
- Sensitivity metric: Fisher information, Hessian trace, or simple PPL delta per layer

### 12.5 Quantization Error Propagation

- Error at layer $l$ propagates to all subsequent layers
- Accumulated error: $\varepsilon_{\text{total}} \approx \sum_l \alpha_l \varepsilon_l$ where $\alpha_l$ depends on sensitivity
- Non-linear amplification through activation functions (especially at boundaries)
- Attention mechanism: error in Q or K amplified by softmax sharpening
- Mitigation: quantize layers in reverse order (last first); compensate errors upstream

---

## 13. Hardware Considerations

### 13.1 Integer Arithmetic on GPUs

| Format | H100 Throughput | Speedup vs FP16 |
| ------ | --------------- | --------------- |
| FP16   | 1979 TFLOPS     | 1×              |
| INT8   | 3958 TOPS       | 2×              |
| INT4   | 7918 TOPS       | 4×              |

Requirements:

- Weight dimensions must be multiples of 16 (INT8) or 32 (INT4) for tensor core utilisation
- INT8 GEMM: mixed-precision; accumulate in INT32; scale and convert to FP16 output
- Dequantization overhead usually fused into the GEMM kernel

### 13.2 ARM and Mobile Hardware

| Platform             | INT Support | Peak TOPS | Notes                 |
| -------------------- | ----------- | --------- | --------------------- |
| Apple Neural Engine  | INT8 native | 38        | Powers Core ML models |
| Qualcomm Hexagon DSP | INT4/INT8   | 45+       | Snapdragon X Elite    |
| ARM Cortex-A         | SIMD INT8   | —         | llama.cpp ARM NEON    |

Mobile constraint: 4–8 GB RAM total; model must fit entirely in RAM + OS overhead.
LLaMA-3 8B INT4: ~4.5 GB; fits in 8 GB iPhone Pro; enables on-device generation.

### 13.3 Roofline Analysis for Quantized Models

- INT4 models: 4× more weights fit in cache; 4× higher arithmetic intensity per byte
- At batch=1: INT4 decode ≈ 4× faster than FP16 (bandwidth-bound regime)
- At large batch: both become compute-bound; INT4 and FP16 may have similar throughput

### 13.4 Quantization for Different Hardware Targets

| Target        | Formats         | Framework                |
| ------------- | --------------- | ------------------------ |
| NVIDIA GPU    | FP8, INT8, INT4 | TensorRT-LLM, vLLM       |
| AMD GPU       | FP8, INT8       | ROCm; less mature        |
| Apple Silicon | INT4, INT8      | Metal, mlx-lm, llama.cpp |
| CPU (x86)     | INT8 AVX-512    | llama.cpp                |
| CPU (ARM)     | INT4/INT8 NEON  | llama.cpp ARM backend    |
| NPU/DSP       | INT4/INT8       | Fixed-function; edge     |

---

## 14. Advanced Topics

### 14.1 Sparsity + Quantization Combined

- Prune + quantize: unstructured 50% sparsity + INT4 = 8× compression over FP32
- **SparseGPT** (Frantar & Alistarh 2023): prune + quantize using second-order info
  - Achieve 50% sparsity + INT4 with minimal quality loss
  - Same OBQ framework as GPTQ; extended for zero-weight constraint
- **2:4 structured sparsity** (NVIDIA A100+): every 4 elements, exactly 2 are zero
  - Hardware-accelerated; 2× speedup for sparse matmul
  - Combined with INT8: 4× speedup vs dense FP16

### 14.2 Outlier-Free Quantization

Root cause of quantization difficulty: outlier activations in specific channels.

**QuaRot** (Ashkboos et al. 2024): rotate weight and activation matrices to eliminate outliers:

- Apply random orthogonal matrix $R$: $W \to WR^\top$; $X \to XR$
- Rotation preserves output: $WX = (WR^\top)(RX)$
- After rotation: activations more Gaussian; fewer outliers
- Enables **W4A4** (both weights and activations at INT4) with acceptable quality

### 14.3 Quantization for MoE Models

- Expert weights: quantize independently per expert (different distributions)
- Router: keep in FP16 or BF16 (routing decisions sensitive to precision)
- Shared experts: higher precision (always active; most impactful)
- Memory benefit: quantized MoE can fit on fewer GPUs; reduces expert parallelism cost

### 14.4 Online Distillation

No pretrained teacher required:

- **Mutual learning** (Zhang et al. 2018): two networks teach each other; both improve
- **Deep mutual learning:** ensemble of students; each teaches others
- Advantage: no need to first train large teacher
- Disadvantage: quality bounded by mutual improvement

### 14.5 Selective Distillation

Not all teacher outputs are equally valuable:

- High-confidence outputs: teacher very certain → reliable signal
- Low-confidence outputs: teacher uncertain → noisy signal
- **Confidence-based filtering:** only distil tokens where teacher confidence > threshold
- **Curriculum distillation:** start with easy tokens; gradually add harder ones

### 14.6 Self-Distillation

Use model itself as both teacher and student:

- **Born-again networks:** train model; use as teacher for identical architecture; repeat
- Each generation improves over previous
- **Layer-wise self-distillation:** earlier layers distilled from later layers in same network
- BYOT (Be Your Own Teacher): internal representation alignment across depths

---

## 15. Common Mistakes

| Mistake                                         | Why It's Wrong                                                   | Fix                                                     |
| ----------------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------- |
| "INT8 is always lossless"                       | INT8 degrades on reasoning and math tasks measurably             | Benchmark on task-specific metrics not just PPL         |
| "More bits always better"                       | FP16 has narrower range than BF16; loss spikes more common       | Choose format based on hardware and dynamic range       |
| "PTQ at INT4 works without calibration"         | Without calibration, scale factors wrong; massive quality loss   | Always calibrate with representative data; 128+ samples |
| "Distillation only helps small models"          | Large models also benefit from distillation from larger teachers | Distillation useful at all scales                       |
| "Soft labels need same vocabulary"              | Vocabulary projection allows cross-tokenizer distillation        | Align vocabularies or use response distillation         |
| "GPTQ and AWQ give identical results"           | Different algorithms; different quality on different tasks       | Benchmark both; task-specific choice                    |
| "Quantization is reversible"                    | Quantization is lossy; dequantized ≠ originals                   | Treat quantized model as distinct; keep original        |
| "Student can exceed teacher"                    | Bounded by teacher quality and own capacity                      | If student beats teacher, improve teacher               |
| "One calibration dataset for all"               | Domain mismatch → suboptimal quantization                        | Use calibration data representative of inference        |
| "Distillation eliminates need for quality data" | Data quality affects teacher response AND student learning       | Curate high-quality diverse prompts                     |

---

## 16. Exercises

1. **Uniform quantization by hand** — given weights $[-1.2, 0.3, 0.8, -0.5, 1.5, 0.1]$
   and $b=3$ bits: compute $s$ and $z$ for symmetric quantization; quantize each weight;
   compute dequantized values; compute MSE.

2. **NF4 vs INT4** — for Gaussian weight distribution $\mathcal{N}(0, 0.02^2)$: compute
   optimal INT4 uniform levels; compare to NF4 levels; compute theoretical MSE; which is better?

3. **GPTQ compensation** — weight row $w = [1.2, -0.8, 0.4]$; quantize $w_1 = 1.2$ to
   nearest INT4 level ($s=0.2$); compute error $\delta$; given Hessian
   $H = [[2,1,0.5],[1,2,1],[0.5,1,2]]$: compute weight update for remaining weights.

4. **Distillation loss** — teacher logits $z_T = [3.0, 1.0, -0.5]$ at $\tau=2$;
   student logits $z_S = [2.5, 0.8, 0.0]$ at $\tau=2$; compute softmax distributions;
   compute KL divergence; compute $\tau^2$ scaled distillation loss.

5. **Temperature effect** — given logits $[4.0, 2.0, 0.5, -1.0]$; compute softmax at
   $\tau = 0.5, 1.0, 2.0, 5.0$; compute entropy at each; interpret "dark knowledge" emergence.

6. **QLoRA memory** — LLaMA-3 8B; BF16 fine-tuning memory (weights + gradients + Adam);
   QLoRA memory (INT4 weights + LoRA rank=16 on Q,V in 32 layers + BF16 adapter gradients + Adam);
   compute savings.

7. **Sensitivity analysis** — 12-layer model; quantizing layer 1 to INT4 → PPL +3.2;
   layer 6 → PPL +0.4; layer 12 → PPL +1.8; design mixed-precision with average 5 bits.

8. **Distillation capacity gap** — teacher 70B, student 1B: estimate whether direct
   distillation is feasible; design teacher assistant cascade with intermediate sizes.

---

## 17. Why This Matters for AI (2026 Perspective)

| Aspect                   | Impact                                                                              |
| ------------------------ | ----------------------------------------------------------------------------------- |
| **Accessibility**        | INT4 quantization enables 70B models on consumer hardware; democratises frontier AI |
| **Cost**                 | 4-bit inference costs 4× less per token; directly reduces API prices                |
| **On-device AI**         | Quantization enables Apple Intelligence, Android AI, privacy-preserving inference   |
| **Speed**                | INT4 decode 4× faster than BF16 in bandwidth-bound regime                           |
| **Environment**          | 4× energy reduction from INT4 vs FP32; significant carbon impact at scale           |
| **Small models**         | Phi-3, Gemma-2, Qwen2.5 at 2–7B outperform GPT-3.5; distillation enables this       |
| **Reasoning**            | DeepSeek-R1 distilled reasoning into 7B–32B; accessible without frontier cost       |
| **Customisation**        | QLoRA enables fine-tuning 70B on single GPU; personalised AI without data centre    |
| **Deployment diversity** | Same model family: cloud (BF16), edge (INT8), mobile (INT4)                         |
| **Research**             | 1.58-bit may change future hardware; dedicated ternary network hardware             |

---

## Conceptual Bridge

Quantization compresses the mathematical representation of knowledge stored in weights.
Distillation transfers knowledge from a complex high-capacity model to a simpler one.
Together they solve the deployment problem: making the mathematics of large models
accessible at small cost.

Next: **RAG Math and Retrieval** — how to augment language models with external
knowledge through retrieval, vector search, and mathematical grounding.

```
Large Pretrained Model
    ↓ [Distillation]
Smaller Student Model
    ↓ [Quantization]
Efficient Compressed Model          ← THIS section
    ↓ [Retrieval Augmentation]      ← NEXT section
Knowledge-Augmented Model
```

---

[← Mixture of Experts and Routing](../10-Mixture-of-Experts-and-Routing/notes.md) | [Home](../../README.md) | [RAG Math and Retrieval →](../12-RAG-Math-and-Retrieval/notes.md)
