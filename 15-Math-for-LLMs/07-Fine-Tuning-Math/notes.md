# Fine-Tuning Math

[← Training at Scale](../06-Training-at-Scale/notes.md) | [Home](../../README.md) | [Scaling Laws →](../08-Scaling-Laws/notes.md)

---

## 1. Intuition

### 1.1 What Is Fine-Tuning?

Pretraining learns a general distribution over language from massive data. Fine-tuning **shifts** that distribution toward a specific task, domain, or behaviour.

$$\theta^* = \arg\min_\theta \;\mathcal{L}_{\text{task}}(\theta) \quad\text{starting from } \theta_0 = \theta_{\text{pretrained}}$$

Fine-tuning is **not** retraining from scratch — it exploits the rich representations already learned. The core tension: **learn new behaviour without forgetting pretrained knowledge**.

### 1.2 Why Fine-Tuning Works

- Pretrained model already encodes syntax, semantics, world knowledge, reasoning patterns
- Fine-tuning only adjusts the final mapping from representations → desired outputs
- The loss landscape near $\theta_0$ is relatively smooth for related tasks
- Small gradient steps from a good initialisation reach good solutions with **far less data and compute** than pretraining
- Linear probing result: even frozen pretrained representations are highly linearly separable for many tasks

### 1.3 The Fine-Tuning Spectrum

```
← Less compute, less data              More compute, more data →

Prompt       Zero/Few    Linear     Adapter    LoRA      Full
Engineering   -shot      Probing    Tuning     Tuning    Fine-tune
    |           |          |          |          |           |
No update   No update  1 layer    <1% params  1-5%     100% params
```

### 1.4 When to Fine-Tune vs Other Approaches

| Approach                  | When to Use                                  | Gradient Updates? | Cost      |
| ------------------------- | -------------------------------------------- | ----------------- | --------- |
| **Prompt engineering**    | Well-defined tasks; capable base model       | No                | Zero      |
| **Few-shot / ICL**        | Small number of examples; no training infra  | No                | Zero      |
| **LoRA / PEFT**           | Consistent behaviour needed; limited compute | Yes (partial)     | Low       |
| **Full fine-tune**        | Maximum quality; ample compute and data      | Yes (all)         | High      |
| **Pretrain from scratch** | Fundamentally OOD domain (proteins, music)   | Yes (all)         | Very high |

### 1.5 Historical Timeline

| Year | Milestone                        | Contribution                                    |
| ---- | -------------------------------- | ----------------------------------------------- |
| 2018 | BERT (Devlin et al.)             | Pretrain-then-fine-tune paradigm established    |
| 2018 | ULMFiT (Howard & Ruder)          | Discriminative fine-tuning; layer-wise LR decay |
| 2019 | Adapter modules (Houlsby et al.) | First PEFT method                               |
| 2021 | LoRA (Hu et al.)                 | Low-rank adaptation; dominant PEFT method       |
| 2021 | Instruction tuning (Wei et al.)  | FLAN — fine-tune on 60+ tasks with instructions |
| 2022 | InstructGPT (Ouyang et al.)      | RLHF pipeline; reward-based fine-tuning         |
| 2023 | QLoRA (Dettmers et al.)          | 4-bit quantised fine-tuning; 65B on single GPU  |
| 2023 | DPO (Rafailov et al.)            | Direct preference optimisation; replaces PPO    |
| 2024 | DoRA (Liu et al.)                | Weight decomposition for better LoRA            |
| 2025 | DeepSeek-R1 / GRPO               | Reasoning emerges from RL fine-tuning           |

### 1.6 Pipeline Position

```
θ_pretrained → [Fine-Tuning] → θ_finetuned → [Inference] → Task outputs
                ^^^^^^^^^^^^^
                 THIS section
```

---

## 2. Formal Definitions

### 2.1 The Fine-Tuning Objective

Pretrained parameters $\theta_0 \in \mathbb{R}^d$ serve as the starting point. The fine-tuning objective:

$$\boxed{\theta^* = \arg\min_\theta \;\mathcal{L}_{\text{task}}(\theta) + \mathcal{R}(\theta, \theta_0)}$$

- $\mathcal{L}_{\text{task}}$: task-specific loss (cross-entropy, preference loss, reward)
- $\mathcal{R}(\theta, \theta_0)$: regularisation penalising deviation from $\theta_0$ (prevents forgetting)

### 2.2 Catastrophic Forgetting

After fine-tuning on $\mathcal{D}_{\text{task}}$, performance on original distribution $\mathcal{D}_{\text{pretrain}}$ degrades:

$$\Delta_{\text{forget}} = \text{acc}(\theta_0, \mathcal{D}_{\text{pretrain}}) - \text{acc}(\theta^*, \mathcal{D}_{\text{pretrain}})$$

Severity increases with: learning rate, number of steps, distance $\|\theta^* - \theta_0\|$.

### 2.3 Task Distribution Shift

| Quantity                               | Definition              | Interpretation                            |
| -------------------------------------- | ----------------------- | ----------------------------------------- |
| $P_{\theta_0}(t \mid \text{ctx})$      | Pretrained distribution | General language model                    |
| $P_{\theta^*}(t \mid \text{ctx})$      | Fine-tuned distribution | Task-specialised model                    |
| $D_{KL}(P_{\theta^*} \| P_{\theta_0})$ | KL divergence           | How far fine-tuning has shifted the model |

- Too small $D_{KL}$: model not learning task
- Too large $D_{KL}$: catastrophic forgetting

### 2.4 Parameter Space vs Function Space

- **Parameter space**: distance $\|\theta^* - \theta_0\|_2$
- **Function space**: $D_{KL}(P_{\theta^*} \| P_{\theta_0})$ over input distribution
- Two models can be close in parameter space but far in function space (and vice versa)
- RLHF KL penalty operates in function space — the right choice

### 2.5 Intrinsic Dimensionality

Li et al. (2018) showed that most fine-tuning happens in a **low-dimensional subspace**:

$$d_{\text{int}} = \text{smallest } d \text{ such that fine-tuning in random } d\text{-dim subspace } \approx \text{ full fine-tune}$$

| Model     | Task | $d_{\text{int}}$ | Total Params |
| --------- | ---- | :--------------: | :----------: |
| BERT-base | MRPC |       ~200       |     110M     |
| BERT-base | QQP  |      ~1,000      |     110M     |
| BERT-base | MNLI |      ~2,000      |     110M     |

This **justifies LoRA**: if updates live in a low-dim subspace, low-rank approximation is sufficient.

---

## 3. Full Fine-Tuning

### 3.1 Standard Full Fine-Tuning

Update **all** parameters with task gradient:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_{\text{task}}(\theta_t)$$

| Setting       |     Pretraining     |      Fine-tuning       |
| ------------- | :-----------------: | :--------------------: |
| Learning rate | $3 \times 10^{-4}$  | $10^{-5}$ to $10^{-6}$ |
| Epochs        |     1–2 passes      |       1–5 passes       |
| Data size     | Trillions of tokens |    10K–1M examples     |
| Optimiser     |        AdamW        |      AdamW (same)      |

### 3.2 Discriminative Fine-Tuning (ULMFiT)

Different learning rates **per layer**:

$$\boxed{\eta^l = \frac{\eta}{\gamma^{N-l}}, \quad l = 1, \ldots, N}$$

- $\gamma = 2.6$ (Howard & Ruder 2018)
- Earlier layers → lower LR → preserve general features (syntax, morphology)
- Later layers → higher LR → adapt to task-specific features

### 3.3 Gradual Unfreezing

- Start with **all layers frozen** except final layer
- Gradually unfreeze lower layers over training
- Layer $l$ unfrozen at step $t_l = l \times T_{\text{unfreeze}}$
- Prevents early catastrophic forgetting of low-level representations

### 3.4 Full Fine-Tuning Memory Cost

Same as pretraining (mixed precision):

$$M = \underbrace{2N}_{\text{fp16 params}} + \underbrace{2N}_{\text{fp16 grads}} + \underbrace{4N + 4N + 4N}_{\text{fp32 copy + m + v}} = 16N \text{ bytes}$$

| Model        | Params | Memory (16N) | Min GPUs (A100 80GB) |
| ------------ | :----: | :----------: | :------------------: |
| LLaMA-3 8B   |   8B   |    128 GB    |          2           |
| LLaMA-3 70B  |  70B   |   1,120 GB   |          14          |
| LLaMA-3 405B |  405B  |   6,480 GB   |          81          |

### 3.5 Overfitting in Fine-Tuning

Fine-tuning datasets are often **tiny** relative to model capacity:

| Signal                   | Meaning                 |
| ------------------------ | ----------------------- |
| Train loss ↓, val loss ↓ | Learning — continue     |
| Train loss ↓, val loss ↑ | Overfitting — stop      |
| Train loss flat          | LR too low or converged |

Solutions: early stopping, weight decay, label smoothing, data augmentation.

---

## 4. Adapter Tuning

### 4.1 Architecture (Houlsby et al. 2019)

Insert small trainable modules between transformer sublayers; freeze all original parameters:

```
Standard:     x → [Attention] → [Add&Norm] → [FFN] → [Add&Norm] → output
With adapter:  x → [Attention] → [Add&Norm] → [Adapter] → [FFN] → [Add&Norm] → [Adapter] → output
```

### 4.2 Adapter Module Design

$$\boxed{\text{Adapter}(\mathbf{x}) = \mathbf{x} + W_{\text{up}}\;\sigma(W_{\text{down}}\;\mathbf{x})}$$

| Component         |        Dimensions         | Purpose                                           |
| ----------------- | :-----------------------: | ------------------------------------------------- |
| $W_{\text{down}}$ | $\mathbb{R}^{r \times d}$ | Down-project to bottleneck                        |
| $\sigma$          |             —             | Nonlinearity (ReLU/GELU)                          |
| $W_{\text{up}}$   | $\mathbb{R}^{d \times r}$ | Up-project back                                   |
| Residual          |      $+ \mathbf{x}$       | Near-identity at init ($W_{\text{up}} \approx 0$) |

Parameters per adapter: $2rd$. Two adapters per block → $4rd$ per layer.

### 4.3 Adapter Parameter Count

$$P_{\text{adapters}} = 4rd \times L$$

| Model     | $d$  | $r$ | $L$ | Adapter Params | % of Total |
| --------- | :--: | :-: | :-: | :------------: | :--------: |
| BERT-base | 768  | 64  | 12  |      2.4M      |    1.5%    |
| LLaMA-7B  | 4096 | 64  | 32  |     33.6M      |    0.5%    |
| LLaMA-70B | 8192 | 64  | 80  |     167.8M     |    0.2%    |

### 4.4 Adapter Variants

| Method               |          What's Trained          |     Params per Layer      | Key Idea                              |
| -------------------- | :------------------------------: | :-----------------------: | ------------------------------------- |
| **Serial adapter**   | $W_{\text{up}}, W_{\text{down}}$ |      $2 \times 2rd$       | Sequential bottleneck after sublayer  |
| **Parallel adapter** |               Same               |      $2 \times 2rd$       | Runs in parallel with sublayer        |
| **Prefix tuning**    |       $P_K, P_V$ per layer       |          $2md_k$          | Prepend $m$ trainable vectors to K, V |
| **Prompt tuning**    |           Soft tokens            | $m \times d$ (input only) | Prepend trainable tokens to input     |
| **P-Tuning v2**      |       Prefixes all layers        |     $2md_k \times L$      | Prefix tuning at every layer          |

---

## 5. LoRA — Low-Rank Adaptation

### 5.1 Core Idea (Hu et al. 2021)

Decompose weight update as low-rank product:

$$\boxed{W' = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r},\; A \in \mathbb{R}^{r \times d},\; r \ll d}$$

- Forward pass: $\mathbf{h} = (W_0 + BA)\mathbf{x} = W_0\mathbf{x} + BA\mathbf{x}$
- Freeze $W_0$; train only $B$ and $A$

### 5.2 Initialisation

| Matrix | Init                         | Why                                          |
| ------ | ---------------------------- | -------------------------------------------- |
| $A$    | $\mathcal{N}(0, 1/\sqrt{r})$ | Random projection                            |
| $B$    | **Zero**                     | $\Delta W = BA = 0$ at start → no disruption |

### 5.3 Scaling Factor

$$\mathbf{h} = W_0\mathbf{x} + \frac{\alpha}{r}\;BA\mathbf{x}$$

- $\alpha$: scaling hyperparameter (not learned)
- Setting $\alpha = r$: unit scaling (effective scale = 1)
- Setting $\alpha = 2r$: effective scale = 2
- Decouples rank from update magnitude

### 5.4 LoRA Parameter Count

Per weight matrix ($d \times d$):

$$P_{\text{LoRA}} = r \times d + d \times r = 2dr \quad\text{vs}\quad d^2 \text{ for full}$$

**Reduction factor:** $\frac{d^2}{2dr} = \frac{d}{2r}$

| Config     | Matrices | Params/Block  |          Total ($L{=}32$)          | % of 8B |
| ---------- | :------: | :-----------: | :--------------------------------: | :-----: |
| Q, V only  |    2     |     $4dr$     | $4 \times 4096 \times r \times 32$ |    —    |
| Q, K, V, O |    4     |     $8dr$     | $8 \times 4096 \times r \times 32$ |    —    |
| All linear |   6–7    | $12dr$–$14dr$ |           Depends on FFN           |    —    |

| $r$ | Q+V Params | All Linear Params |  % of 8B   |
| :-: | :--------: | :---------------: | :--------: |
|  4  |    3.1M    |       9.4M        | 0.04–0.12% |
| 16  |    8.4M    |       25.2M       | 0.11–0.32% |
| 64  |   33.6M    |      100.7M       | 0.42–1.26% |

### 5.5 Which Matrices to Apply LoRA To

- Original paper: Q, V only → strong results
- 2024 practice: **all linear layers** (Q, K, V, O, up_proj, down_proj, gate_proj) → best quality
- Attention matrices > FFN matrices for most tasks
- More matrices → more parameters → better task fit → slightly more forgetting

### 5.6 Rank Selection

| Rank $r$ | Use Case                         | Quality vs Full FT |
| :------: | -------------------------------- | :----------------: |
|    1     | Extreme compression; format only |        Low         |
|   4–16   | Standard instruction tuning      |        Good        |
|  64–128  | Complex domain adaptation        |     Near-full      |
|  $d/2$   | Full rank (no compression)       |       Equal        |

### 5.7 LoRA Merging

After fine-tuning, merge for **zero-overhead inference**:

$$\boxed{W' = W_0 + \frac{\alpha}{r}\;BA}$$

- Single weight matrix; no additional computation at inference
- Unmerge: $W_0 = W' - (\alpha/r)BA$; recovers base model

### 5.8 Multi-Task LoRA

Train separate LoRA adapters per task; share base model:

$$\Delta W_{\text{combined}} = \sum_i w_i \Delta W_i = \sum_i w_i B_i A_i$$

- LoRAHub (2023): combine multiple adapters via learned weights
- Task arithmetic (Ilharco et al. 2022): add/subtract weight deltas for task composition
- Per-task storage: $2dr$ per matrix per task $\ll d^2$ for full fine-tune

---

## 6. QLoRA — Quantised Low-Rank Adaptation

### 6.1 Core Idea (Dettmers et al. 2023)

- Quantise pretrained weights to **4-bit NF4**
- Add LoRA adapters in **BF16**; train only adapters
- Dequantise on-the-fly for forward pass: 4-bit stored → 16-bit compute → 4-bit stored
- Enables **70B model fine-tuning on single 80GB A100**

### 6.2 NF4 — Normal Float 4-bit

Standard int4 uses **uniform** quantisation — poor for normally-distributed weights. NF4 places 16 quantisation levels to **equalise probability mass**:

$$q_i^* = \arg\min_q \;\mathbb{E}_{x \sim \mathcal{N}(0,1)}\!\left[(x - q)^2 \cdot \mathbb{1}[q_i \le x < q_{i+1}]\right]$$

Result: lower quantisation error than int4 for neural network weights.

### 6.3 Double Quantisation

| What           |           Standard            |        Double Quantised        |
| -------------- | :---------------------------: | :----------------------------: |
| Weight bits    |              4.0              |              4.0               |
| Scale overhead | 0.5 bits/weight (fp32 per 64) | 0.127 bits/weight (fp8 scales) |
| **Total**      |      **4.5 bits/weight**      |     **4.127 bits/weight**      |

~8% additional memory saving.

### 6.4 QLoRA Memory Formula

$$\boxed{M = N \times 0.5\text{ B (NF4)} + N_{\text{LoRA}} \times 2\text{ B (BF16)} + N_{\text{LoRA}} \times 8\text{ B (Adam fp32)}}$$

| Model                  | Base (NF4) | LoRA (BF16+Adam) |  Total   | Fits A100 80GB? |
| ---------------------- | :--------: | :--------------: | :------: | :-------------: |
| LLaMA-3 8B, $r{=}16$   |    4 GB    |      0.1 GB      | ~4.1 GB  |   ✓ (easily)    |
| LLaMA-3 70B, $r{=}32$  |   35 GB    |      0.3 GB      | ~35.3 GB |        ✓        |
| LLaMA-3 70B, $r{=}64$  |   35 GB    |     0.67 GB      | ~35.7 GB |        ✓        |
| LLaMA-3 405B, $r{=}16$ |   202 GB   |      0.5 GB      | ~203 GB  |   ✗ (3 GPUs)    |

### 6.5 QLoRA vs LoRA Quality

- Quantisation noise from NF4 introduces small error (~1–2% degradation)
- For sensitive tasks (maths, code): gap can be larger
- Mitigations: larger $r$; longer training; higher quality data

---

## 7. Advanced PEFT Methods

### 7.1 DoRA — Weight-Decomposed Low-Rank Adaptation (Liu et al. 2024)

Decompose weight $W$ into **magnitude** and **direction**:

$$W = m \cdot \frac{V}{\|V\|_c}, \quad m \in \mathbb{R}^{1 \times n},\; V \in \mathbb{R}^{d \times n}$$

Fine-tune via:

$$\boxed{W' = (m + \Delta m) \cdot \frac{V_0 + \Delta V}{\|V_0 + \Delta V\|_c}}$$

- Magnitude $m$: fully trainable (tiny — $n$ params)
- Direction $V$: updated via LoRA ($2dr$ params)
- More expressive than LoRA at same rank; better mimics full fine-tuning

### 7.2 LoRA+ (Hayou et al. 2024)

- Set $\eta_B = \lambda \times \eta_A$ where $\lambda = 16$
- Theoretical finding: optimal LR for $B$ should be much larger than for $A$
- Improves convergence speed by ~2× on some tasks
- Zero cost modification

### 7.3 Other Advanced Methods

| Method        | Key Idea                                        |   Parameters    | Notes                    |
| ------------- | ----------------------------------------------- | :-------------: | ------------------------ |
| **LoRA-FA**   | Freeze $A$; train only $B$                      | $dr$ per matrix | Half of LoRA params      |
| **VeRA**      | Shared frozen $A, B$; per-layer scaling vectors |  $2r \times L$  | 10–100× fewer than LoRA  |
| **GaLore**    | Project gradients to low-rank subspace          |  Full weights   | Adam-like efficiency     |
| **IA³**       | Scale K, V, FFN activations                     | $3d$ per layer  | ~0.01% of model          |
| **Sparse FT** | Update only high-Fisher parameters              |    top-$k$%     | Fisher-based selection   |
| **BitFit**    | Fine-tune only bias terms                       |      ~0.1%      | Surprisingly competitive |

### 7.4 Fisher Information for Sparse Selection

$$\boxed{F_i = \mathbb{E}\!\left[\left(\frac{\partial \log P_\theta}{\partial \theta_i}\right)^2\right]}$$

High $F_i$ → parameter $i$ is important → protect or prioritise for update.

---

## 8. Instruction Tuning

### 8.1 Training Objective

Cross-entropy loss on **response tokens only** (mask instruction tokens):

$$\boxed{\mathcal{L}_{\text{SFT}} = -\sum_{i \in \text{response}} \log P_\theta(t_i \mid \text{instruction}, t_{<i})}$$

> **Critical**: do NOT compute loss on instruction tokens — they are conditioning context.

### 8.2 Data Format (LLaMA-3 Chat)

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant.
<|start_header_id|>user<|end_header_id|>
Explain Newton's second law.
<|start_header_id|>assistant<|end_header_id|>
Newton's second law states that F = ma...
```

### 8.3 Data Quality vs Quantity

LIMA (Zhou et al. 2023): **1,000 carefully curated examples** match 50K noisy examples.

**IFD Score** — Instruction Following Difficulty:

$$\boxed{\text{IFD}(x, y) = \frac{\mathcal{L}(y \mid x)}{\mathcal{L}(y)}}$$

- $\mathcal{L}(y \mid x)$: loss on response given instruction
- $\mathcal{L}(y)$: loss on response unconditionally
- High IFD → response is surprising given instruction → more informative training example

### 8.4 Instruction Following Evaluation

| Benchmark   | What It Measures                                  | Method         |
| ----------- | ------------------------------------------------- | -------------- |
| MT-Bench    | Multi-turn quality (80 questions, 8 categories)   | GPT-4 judge    |
| AlpacaEval  | Win rate vs text-davinci-003 (805 instructions)   | LLM comparison |
| IFEval      | Verifiable constraints ("respond in 3 sentences") | Automated      |
| FollowBench | Multi-level constraint difficulty                 | Automated      |

---

## 9. RLHF — Reinforcement Learning from Human Feedback

### 9.1 Pipeline

```
Stage 1: SFT          →  θ_SFT (fine-tune on demonstrations)
Stage 2: Reward Model  →  R(x, y) (learn from human preferences)
Stage 3: RL (PPO)      →  θ_RLHF (maximise reward, constrained by KL)
```

### 9.2 Preference Model (Bradley-Terry)

$$\boxed{P(y_w \succ y_l \mid x) = \sigma\!\left(R(x, y_w) - R(x, y_l)\right)}$$

Reward model loss:

$$\mathcal{L}_R = -\mathbb{E}\!\left[\log \sigma\!\left(R(x, y_w) - R(x, y_l)\right)\right]$$

where $y_w$ = preferred response, $y_l$ = dispreferred.

### 9.3 PPO Objective for LLMs

$$\boxed{\mathcal{L}_{\text{PPO}} = \mathbb{E}\!\left[\min\!\left(\rho_t A_t,\; \text{clip}(\rho_t, 1{-}\epsilon, 1{+}\epsilon)\,A_t\right)\right] - \beta\,D_{KL}(\pi_\theta \| \pi_{\text{ref}})}$$

where $\rho_t = \pi_\theta(a|s) / \pi_{\text{old}}(a|s)$.

| Component                            | Role                                             |
| ------------------------------------ | ------------------------------------------------ |
| $\rho_t$                             | Policy ratio — how much policy changed           |
| $A_t$                                | Advantage — how much better than baseline        |
| $\text{clip}(\cdot, 1{\pm}\epsilon)$ | Prevents large policy updates ($\epsilon = 0.2$) |
| $\beta D_{KL}$                       | KL penalty — stay near reference model           |

### 9.4 PPO Memory Requirement

Four models loaded simultaneously:

| Model                        | Purpose                   |  Memory  |
| ---------------------------- | ------------------------- | :------: |
| Policy $\pi_\theta$          | Being optimised           |  $16N$   |
| Reference $\pi_{\text{ref}}$ | KL anchor (frozen)        |   $2N$   |
| Reward $R$                   | Scores responses (frozen) |   $2N$   |
| Value $V_\phi$               | Advantage estimation      |  $16N$   |
| **Total**                    |                           | **~36N** |

70B PPO: ~5 TB memory → 60+ H100s.

### 9.5 Reward Hacking

Model finds ways to **maximise reward** without actual quality improvement:

- Verbosity (longer = higher reward)
- Sycophancy (agree with user)
- Format gaming (bullet points always score higher)

Mitigations: diverse reward models; adversarial red-teaming; Constitutional AI.

---

## 10. DPO — Direct Preference Optimisation

### 10.1 Core Insight (Rafailov et al. 2023)

The optimal policy under KL-constrained RL has a **closed form**:

$$\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)}{Z(x)} \exp\!\left(\frac{1}{\beta}\,R(x, y)\right)$$

Solve for $R$:

$$R(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

### 10.2 DPO Loss

Substitute into Bradley-Terry; $Z(x)$ cancels:

$$\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}\!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]}$$

### 10.3 DPO Gradient

$$\nabla_\theta \mathcal{L}_{\text{DPO}} \propto -\underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{weight}}\left[\nabla_\theta \log \pi_\theta(y_w) - \nabla_\theta \log \pi_\theta(y_l)\right]$$

- $\sigma(\hat{r}_l - \hat{r}_w)$: larger when model gets preference **wrong** → harder examples get bigger gradients
- Increases $\log \pi_\theta(y_w)$; decreases $\log \pi_\theta(y_l)$

### 10.4 β Parameter

|   $\beta$    | Effect                                     | Use                  |
| :----------: | ------------------------------------------ | -------------------- |
|   $\to 0$    | Ignore KL; maximize preference; forgetting | Aggressive alignment |
| $0.01$–$0.1$ | Typical range                              | Standard DPO         |
| $0.1$–$1.0$  | Conservative                               | LoRA DPO             |
| $\to \infty$ | No movement from reference                 | Useless              |

### 10.5 DPO vs PPO

| Property           |     PPO     |         DPO          |
| ------------------ | :---------: | :------------------: |
| Reward model       |  Required   |     Not required     |
| Value model        |  Required   |     Not required     |
| Models in memory   |      4      |          2           |
| Training stability |  Sensitive  |     More stable      |
| Hyperparameters    |    Many     | Few (mainly $\beta$) |
| Online vs offline  |   Online    |       Offline        |
| 2026 adoption      | Less common |       Dominant       |

### 10.6 DPO Variants

| Method    | Year | Key Idea                                           |
| --------- | :--: | -------------------------------------------------- |
| **IPO**   | 2023 | Regularises to avoid overfit to preferences        |
| **KTO**   | 2023 | Prospect theory; unpaired good/bad examples        |
| **ORPO**  | 2024 | Combines SFT + DPO in one loss; no reference model |
| **SimPO** | 2024 | Length-normalised implicit reward; no ref model    |
| **TDPO**  | 2024 | Token-level DPO; finer-grained signal              |

**ORPO Loss:**

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} - \lambda \cdot \mathbb{E}\!\left[\log \sigma\!\left(\log \frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}\right)\right]$$

---

## 11. GRPO and Reasoning Fine-Tuning

### 11.1 GRPO — Group Relative Policy Optimisation (DeepSeek 2024)

For each prompt $x$, sample $G$ responses $\{y_1, \ldots, y_G\}$ from current policy. Normalise advantages:

$$\boxed{A_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}}$$

Policy update:

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\!\left[\frac{1}{G}\sum_{i=1}^{G} \min\!\left(\rho_i A_i,\;\text{clip}(\rho_i, 1{\pm}\epsilon)\,A_i\right)\right] + \beta\,D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**Key advantage**: eliminates separate value model → saves one full model of memory.

### 11.2 Reward Design for Reasoning

| Reward Type       | What It Checks                          | Pros             | Cons                      |
| ----------------- | --------------------------------------- | ---------------- | ------------------------- |
| **Format reward** | Correct structure (e.g. `<think>` tags) | Easy to verify   | Doesn't check correctness |
| **ORM** (Outcome) | Final answer correct                    | Simple; scalable | May reward shortcuts      |
| **PRM** (Process) | Each reasoning step correct             | Better signal    | Expensive to label        |

### 11.3 Self-Improvement Methods

| Method          | Year | Key Idea                                                     |
| --------------- | :--: | ------------------------------------------------------------ |
| **STaR**        | 2022 | Generate rationales; keep correct ones; iterate              |
| **ReST**        | 2023 | Generate K responses; filter by reward; SFT on best; iterate |
| **DeepSeek-R1** | 2025 | Cold-start SFT → GRPO → rejection sampling → GRPO alignment  |

DeepSeek-R1 key finding: **reasoning emerges from RL** without supervised reasoning chains. Model spontaneously develops self-verification and backtracking.

---

## 12. Continual Learning and Catastrophic Forgetting

### 12.1 Elastic Weight Consolidation (EWC)

$$\boxed{\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2}\sum_i F_i\,(\theta_i - \theta_i^*)^2}$$

- $F_i$: Fisher information — importance of parameter $i$ for old task
- High $F_i$ → penalise changing parameter $i$ heavily
- $\lambda$: controls forgetting vs plasticity tradeoff

### 12.2 Simpler Regularisation

**L2 / Proximal fine-tuning:**

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \frac{\lambda}{2}\|\theta - \theta_0\|_2^2$$

Equivalent to Gaussian prior on $\theta$ centred at $\theta_0$. Less principled than EWC (uniform importance), but effective and easy.

### 12.3 Replay Methods

| Method                 |  What's Stored   | Forgetting |  Cost   |
| ---------------------- | :--------------: | :--------: | :-----: |
| Exact replay           | Old data samples |    Low     | Storage |
| Generative replay      | Generative model |   Medium   | Compute |
| Dark experience replay | Soft labels (KD) |    Low     | Storage |

### 12.4 LoRA for Continual Learning

- Separate LoRA adapter per task; base model **never changes**
- No forgetting: base frozen; old adapters preserved
- **O-LoRA** (2023): orthogonal constraint between consecutive LoRA updates:

$$\Delta W_k \perp \Delta W_{k-1}: \quad \Delta W_k^\top \Delta W_{k-1} = 0$$

### 12.5 Model Merging

Combine multiple fine-tuned models **without retraining**:

| Method               | Formula                                              | Key Idea                       |
| -------------------- | ---------------------------------------------------- | ------------------------------ |
| **Weight averaging** | $\theta_{\text{merge}} = \frac{1}{K}\sum_i \theta_i$ | Simple; surprisingly effective |
| **SLERP**            | Spherical interpolation on unit sphere               | Preserves norm                 |
| **TIES**             | Resolve sign conflicts before merge                  | Better than naive average      |
| **DARE**             | Randomly prune LoRA before merge                     | Reduces interference           |
| **Model soup**       | Average multiple fine-tunes from same base           | Better than any single         |

**SLERP:**

$$\text{SLERP}(\theta_0, \theta_1; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,\theta_0 + \frac{\sin(t\Omega)}{\sin\Omega}\,\theta_1$$

---

## 13. Evaluation of Fine-Tuned Models

### 13.1 Backward Transfer

$$\boxed{\text{BWT} = \frac{1}{T-1}\sum_{i=1}^{T-1}(R_{T,i} - R_{i,i})}$$

- $R_{T,i}$: performance on task $i$ after learning task $T$
- Negative BWT = forgetting; positive BWT = backward transfer (rare)

### 13.2 Alignment Tax

Run standard benchmarks **before and after** fine-tuning:

$$\Delta_{\text{tax}} = \text{score}_{\text{finetuned}} - \text{score}_{\text{pretrained}}$$

Modern finding: well-executed SFT/DPO causes minimal alignment tax (<1–2%).

### 13.3 Overfitting & Calibration

| Issue              | Signal                            | Fix                          |
| ------------------ | --------------------------------- | ---------------------------- |
| Overfitting        | Val loss increases                | Early stopping; weight decay |
| Reward hacking     | Reward ↑ but human pref stagnates | Diverse reward models        |
| Overconfidence     | High confidence, low accuracy     | Temperature scaling          |
| Format overfitting | Perfect format, bad content       | More diverse training data   |

---

## 14. Common Mistakes

|  #  | Mistake                            | Why It's Wrong                                             | Fix                                          |
| :-: | ---------------------------------- | ---------------------------------------------------------- | -------------------------------------------- |
|  1  | "Fine-tuning replaces pretraining" | Fine-tuning succeeds because of pretrained representations | Choose strongest base model                  |
|  2  | "More LoRA rank always better"     | High $r$ → approaches full FT cost and forgetting          | Grid search $r \in \{8, 16, 32, 64\}$        |
|  3  | "Loss on instruction tokens too"   | Instructions are conditioning; wrong prediction target     | Mask instruction tokens                      |
|  4  | "Same LR as pretraining"           | Fine-tuning LR should be 10–100× smaller                   | Start at $10^{-5}$; tune on val set          |
|  5  | "QLoRA is lossless"                | 4-bit quantisation introduces ~1–2% degradation            | Full LoRA when quality critical              |
|  6  | "DPO β doesn't matter"             | $\beta$ controls entire forgetting/alignment tradeoff      | Sweep $\beta \in \{0.01, 0.05, 0.1, 0.5\}$   |
|  7  | "One epoch always enough"          | Complex tasks need more steps; simple ones less            | Monitor validation loss                      |
|  8  | "LoRA only on attention"           | FFN critical for knowledge-intensive tasks                 | Apply to all linear layers                   |
|  9  | "SFT eliminates hallucination"     | SFT teaches format; factuality needs separate treatment    | RAG; factuality data; TruthfulQA             |
| 10  | "Merged LoRA ≠ trained model"      | Merge is exact: $W' = W_0 + (\alpha/r)BA$                  | Safe to merge; verify with output comparison |

---

## 15. Exercises

1. **LoRA parameter count** — $d{=}4096$, $r{=}16$, applied to Q, K, V, O, up_proj, down_proj (6 matrices), $L{=}32$ layers: compute trainable params; percentage of 8B.
2. **DPO loss by hand** — $\beta{=}0.1$; $\log \pi_\theta(y_w)/\pi_{\text{ref}}(y_w) = 0.8$; $\log \pi_\theta(y_l)/\pi_{\text{ref}}(y_l) = -0.4$: compute DPO loss; compute gradient weight $\sigma(\hat{r}_l - \hat{r}_w)$.
3. **QLoRA memory** — LLaMA-3 70B, $r{=}32$, Q+V only, $L{=}80$: compute base model memory (NF4); LoRA memory (BF16+Adam); total; does it fit A100 80GB?
4. **EWC loss** — $\theta = [1.0, 2.0]$, $\theta^* = [0.8, 2.3]$, $F = [10.0, 1.0]$, $\lambda{=}1$: compute EWC penalty; which parameter is more constrained?
5. **Adapter parameter count** — BERT-base ($d{=}768$, $L{=}12$), $r{=}64$: compute adapter params; percentage of 125M total.
6. **IFD score** — $\mathcal{L}(y|x) = 2.3$, $\mathcal{L}(y) = 1.5$: compute IFD; is this high or low quality?
7. **GRPO advantage** — $G{=}8$, rewards $= [1, 0, 1, 1, 0, 0, 1, 0]$: compute mean, std, all advantages; highest advantage?
8. **Model merging** — $\theta_1 = [1.2, 0.8, 1.5]$, $\theta_2 = [0.9, 1.1, 1.3]$, $\theta_0 = [1.0, 1.0, 1.0]$: compute task vectors; merged $\theta = \theta_0 + 0.5(\Delta_1 + \Delta_2)$.

---

## 16. Why This Matters for AI

| Aspect                 | Impact                                                                       |
| ---------------------- | ---------------------------------------------------------------------------- |
| **Accessibility**      | QLoRA enables 70B fine-tuning on consumer hardware; democratises alignment   |
| **Alignment**          | DPO + GRPO are how frontier models are aligned to human values               |
| **Reasoning**          | GRPO + process rewards produced DeepSeek-R1 / o1-class reasoning             |
| **Specialisation**     | Domain-specific models (medical, legal, code) produced via fine-tuning       |
| **Safety**             | Constitutional AI, RLAIF use fine-tuning to instil safety behaviours         |
| **Cost**               | LoRA fine-tuning costs \$10–\$1000 vs millions for pretraining               |
| **Personalisation**    | Per-user LoRA adapters: personalise without sharing data                     |
| **Model merging**      | Combine capabilities without retraining; enables modular AI                  |
| **Continual learning** | Fine-tune on new data without forgetting old; essential for deployed systems |
| **Evaluation**         | Fine-tuning makes evaluation harder; models overfit benchmarks               |

---

## Conceptual Bridge

Fine-tuning is the bridge between a general pretrained model and a deployed, aligned, task-specific system. All techniques share one core tension: **learn new behaviour vs preserve existing knowledge**.

Next: **Scaling Laws** — how model size, data, and compute interact to determine performance; Chinchilla optimality; emergent abilities.

```
θ_pretrained → [Fine-Tuning] → θ_aligned → [Quantisation] → θ_quantised → [Serving] → Responses
                ^^^^^^^^^^^^^
                 THIS section
```

---

[← Training at Scale](../06-Training-at-Scale/notes.md) | [Home](../../README.md) | [Scaling Laws →](../08-Scaling-Laws/notes.md)
