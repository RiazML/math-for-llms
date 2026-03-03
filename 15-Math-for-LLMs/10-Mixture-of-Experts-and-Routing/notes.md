# Mixture of Experts and Routing

[← Efficient Attention and Inference](../09-Efficient-Attention-and-Inference/notes.md) | [Home](../../README.md) | [Quantization and Distillation →](../11-Quantization-and-Distillation/notes.md)

---

## 1. Intuition

### 1.1 What Is Mixture of Experts?

A neural network architecture where different subnetworks ("experts") specialise on
different inputs. Instead of every parameter being active for every token, only a
small subset is activated per token.

Core idea: divide the model's capacity into specialised modules; route each input to
the most relevant ones. MoE decouples two quantities that are coupled in dense models:
**total parameters** vs **compute per token**.

A 100B MoE model can have the quality of a 100B dense model at the compute cost of a
10B dense model. Every token only "sees" a fraction of the network; different tokens
see different fractions.

### 1.2 The Specialisation Hypothesis

Different tokens require different types of knowledge and computation:

- Syntactic tokens ("the", "of") → different processing than semantic tokens ("quantum", "photosynthesis")
- Code tokens require different transformations than natural language tokens
- MoE allows the model to develop specialised processing pathways without explicit supervision
- Emerges naturally from routing training: experts differentiate through gradient descent

### 1.3 The Core Tradeoff

| Property                      | Dense Model      | MoE Model                                    |
| ----------------------------- | ---------------- | -------------------------------------------- |
| **Parameters active**         | All N every step | Only k/N per token                           |
| **Compute per token**         | High             | Low (proportional to k)                      |
| **Training complexity**       | Standard         | Complex (load balancing, instability)        |
| **Capacity at fixed compute** | N params         | N × (N_experts/k) params                     |
| **Memory**                    | N params         | All N_total params (even though only k used) |

The gain: at fixed compute budget, MoE can have N/k times more parameters.
The cost: load balancing, communication overhead, training instability, expert collapse.

**Key insight:** parameters are "free" in terms of inference compute but NOT memory.

### 1.4 Historical Timeline

| Year    | Work            | Key Contribution                                                           |
| ------- | --------------- | -------------------------------------------------------------------------- |
| 1991    | Jacobs et al.   | Original MoE; soft gating for regression                                   |
| 1994    | Jordan & Jacobs | Hierarchical MoE; EM training algorithm                                    |
| 2014    | Eigen et al.    | First neural MoE with backpropagation                                      |
| 2017    | Shazeer et al.  | "Outrageously Large NNs"; sparsely-gated MoE for NLP; top-k + noise        |
| 2020    | Lepikhin et al. | GShard; MoE at 600B for multilingual translation                           |
| 2021    | Fedus et al.    | Switch Transformer; top-1 routing; 1.6T parameters                         |
| 2022    | Zoph et al.     | ST-MoE; stability improvements; Z-loss; auxiliary loss design              |
| 2022    | Mustafa et al.  | V-MoE; MoE for vision                                                      |
| 2023    | Mistral AI      | Mixtral 8×7B; first widely-deployed open MoE LLM                           |
| 2024    | DeepSeek        | DeepSeekMoE; fine-grained experts; shared expert concept                   |
| 2024    | DeepSeek-V2/V3  | MLA + MoE; 671B total / 37B active; auxiliary-loss-free balancing          |
| 2024    | xAI             | Grok-1; 314B MoE; open-sourced                                             |
| 2025–26 | Industry-wide   | MoE standard for frontier models; nearly all leading models use sparse MoE |

### 1.5 Pipeline Position

```
Token Embedding → [Attention] → Residual → [MoE FFN Layer] → Residual → … → LM Head
                                             ^^^^^^^^^^^^^^^^
                                              THIS section
       Standard FFN replaced by:
       Router → [Expert 1]  ↘
               [Expert 2]  → Weighted Sum → Output
               [Expert k]  ↗
               (top-k selected from N total)
```

---

## 2. Formal Definitions

### 2.1 Expert Network

An expert $E_i$ is a feed-forward network:

$$E_i(x) = W_{2,i} \cdot \sigma(W_{1,i}\, x + b_{1,i}) + b_{2,i}$$

- Each expert has independent parameters $W_{1,i}, W_{2,i}$
- In transformers: experts are typically the FFN sublayer; attention is shared (dense)
- Expert hidden dimension: $d_{ff}$ (same as standard FFN, e.g. $4d$)
- $N$ experts total; typically $N \in \{8, 16, 64, 128, 256\}$

### 2.2 Router / Gating Function

The router computes a distribution over experts given input token $x \in \mathbb{R}^d$:

$$G(x) = \text{softmax}(W_g\, x) \in \mathbb{R}^N$$

- $W_g \in \mathbb{R}^{N \times d}$: learned gating weight matrix
- $G(x)_i \in [0,1]$: probability (or weight) assigned to expert $i$
- Full soft MoE: use all experts weighted by $G(x)$ — computationally expensive

### 2.3 Sparse Top-k Routing

Select only the $k$ experts with highest gate values:

$$\text{TopK}(G(x), k) = \text{indices of } k \text{ largest values in } G(x)$$

Sparse gate:

$$\tilde{G}(x)_i = \begin{cases} G(x)_i & \text{if } i \in \text{TopK}(G(x), k) \\ 0 & \text{otherwise} \end{cases}$$

Renormalise selected gates:

$$\hat{G}(x)_i = \frac{\tilde{G}(x)_i}{\sum_{j \in \text{TopK}} \tilde{G}(x)_j}$$

MoE output:

$$y = \sum_{i \in \text{TopK}(G(x), k)} \hat{G}(x)_i \cdot E_i(x)$$

### 2.4 Sparsity Ratio

Active parameters per token: $k \times$ (expert params) out of $N \times$ (expert params).

$$\rho = \frac{k}{N}$$

| Configuration | k   | N   | ρ     |
| ------------- | --- | --- | ----- |
| Mixtral 8×7B  | 2   | 8   | 0.25  |
| DeepSeekMoE   | 6   | 64  | 0.094 |
| DeepSeek-V3   | 8   | 256 | 0.031 |

Compute per token scales with $k$, not $N$. Memory scales with $N$: must load all
$N$ expert weights regardless of which $k$ are used.

### 2.5 MoE Layer vs Dense Layer

| Property          | Dense FFN                  | MoE FFN                             |
| ----------------- | -------------------------- | ----------------------------------- |
| **Output**        | $y = \text{FFN}(x)$        | $y = \sum_i \hat{G}(x)_i E_i(x)$    |
| **Active params** | All                        | Only $k$ experts                    |
| **FLOPs**         | $2 \times d \times d_{ff}$ | $2 \times k \times d \times d_{ff}$ |
| **Total params**  | $2 \times d \times d_{ff}$ | $N \times 2 \times d \times d_{ff}$ |

Same FLOPs if $d_{ff}^{(\text{MoE})} = d_{ff}^{(\text{dense})}$: MoE has $N\times$ more
capacity at same compute.

---

## 3. Routing Algorithms — Complete Taxonomy

### 3.1 Token Choice Routing (Standard)

Each token independently selects its top-k experts:

```
x → W_g x → softmax → argsort → take top-k → renormalise → dispatch
```

Problem: tokens may choose the same experts → **load imbalance**. Some experts overflow
(too many tokens); some underflow.

Overflow handling: drop excess tokens OR use expert capacity buffers.

### 3.2 Expert Capacity

Capacity $C$: maximum tokens each expert can process per batch.

$$\text{Capacity Factor } CF = \frac{C \times N}{T \times k}$$

| CF   | Meaning                                                                     |
| ---- | --------------------------------------------------------------------------- |
| 1.0  | Perfectly uniform load → each expert handles exactly $T \cdot k / N$ tokens |
| 1.25 | 25% slack to absorb imbalance without dropping tokens                       |
| 1.5  | 50% slack; wasteful but safe                                                |

**Token dropping:** tokens exceeding expert capacity are skipped — either passed through
via identity mapping or zero-padded. Critical failure mode if too many tokens dropped.

### 3.3 Expert Choice Routing (Zhou et al. 2022)

Reverse the selection: each expert selects its top-$C$ tokens.

$$\text{Selected by expert } i = \text{TopC}\!\left(\{G(x_t)_i : t \in \text{batch}\}, C\right)$$

Guarantees:

- **Perfect load balance by construction** — no dropped tokens; no auxiliary loss needed
- Problem: some tokens may not be processed by any expert (underselected)
- Some tokens processed by multiple experts; some by zero
- Not suitable for autoregressive generation (cannot guarantee every token processed)
- Strong for encoder models and offline batch processing

### 3.4 Global vs Local Routing

| Type       | Description                                  | Use Case                         |
| ---------- | -------------------------------------------- | -------------------------------- |
| **Local**  | Routing decisions made per-batch             | Most common in practice          |
| **Global** | Aggregate routing statistics across batches  | Research; adjust router globally |
| **Online** | Routing changes dynamically during inference | Adaptive serving                 |
| **Static** | Fixed routing after training; deterministic  | Fast inference; no routing cost  |

### 3.5 Soft Routing (Soft MoE — Puigcerver et al. 2023)

Dispatch weighted combinations of tokens to each expert:

$$\tilde{x}_i = \sum_t D(x_t)_i \cdot x_t \quad\text{(dispatch)}$$

$$y_t = \sum_i C(x_t)_i \cdot E_i(\tilde{x}_i) \quad\text{(combine)}$$

- $D$: dispatch weight matrix; $C$: combine weight matrix
- No token dropping; fully differentiable; no discrete routing decisions
- Disadvantage: all experts active for all tokens (soft) → compute = dense model
- Used in Soft MoE ViT (Google 2023); research setting

### 3.6 Hash-Based Routing

Deterministic routing: assign token to expert based on $\text{hash}(\text{token\_id}) \bmod N$.

- No learned router; no load balancing issues; perfect uniformity by construction
- Disadvantage: no specialisation learning; token content doesn't influence routing
- Surprisingly competitive with learned routing on some benchmarks (Roller et al. 2021)
- Use case: when training stability is more important than maximal specialisation

### 3.7 Random Routing

Assign tokens to random experts at each step. Used in ablation studies to disentangle
"routing quality" from "expert capacity."

Near-competitive with learned routing in some settings → questions whether routing matters
at all. Provides lower bound on routing quality.

### 3.8 Learned Routing with Discrete Optimisation

Standard softmax routing: gradients flow through selected experts only. Discrete
selection (argmax) is not differentiable.

**Gumbel-softmax:** add Gumbel noise; temperature annealing; differentiable approximation:

$$G(x) = \text{softmax}\!\left(\frac{W_g\, x + \text{Gumbel}(0,1)}{\tau}\right)$$

- $\tau \to 0$: approaches one-hot (argmax)
- $\tau = 1$: standard softmax
- **Straight-through estimator:** use argmax in forward; softmax gradient in backward

---

## 4. Load Balancing — The Central Problem

### 4.1 Expert Collapse

Without explicit load balancing, the router rapidly converges to always sending tokens
to the same 1–2 experts.

**Self-reinforcing loop:**

```
Popular expert → more gradient updates → improves → becomes more popular → …
```

"Rich-get-richer" dynamics. Result: most experts receive zero tokens; effectively a
dense model with less capacity. Catastrophic for training. Detected by monitoring per-expert
load distribution.

### 4.2 Importance Loss (Shazeer et al. 2017)

Auxiliary loss penalising imbalanced importance (sum of gate values) across experts:

$$\mathcal{L}_{\text{importance}} = \text{CV}(\text{Importance})^2$$

$$\text{Importance}(x) = \sum_{t=1}^{T} G(x_t) \in \mathbb{R}^N$$

- $\text{CV} = \sigma / \mu$ (coefficient of variation)
- High CV → high imbalance → high loss
- Encourages equal total gate weight across all experts

### 4.3 Load Loss / Auxiliary Loss (Standard)

Shazeer et al. (2017) combined importance with load:

$$\mathcal{L}_{\text{aux}} = \alpha \sum_{i=1}^{N} f_i \cdot P_i$$

Where:

$$f_i = \frac{1}{T} \sum_{t=1}^{T} \mathbb{1}\!\left[i \in \text{TopK}(G(x_t))\right]$$

$$P_i = \frac{1}{T} \sum_{t=1}^{T} G(x_t)_i$$

- $f_i$ = fraction of tokens dispatched to expert $i$ (load)
- $P_i$ = mean gate probability for expert $i$ over batch (importance)
- Minimise $f_i \cdot P_i$ product: penalises experts that are both frequently chosen
  AND have high probability
- $\alpha = 0.01\text{–}0.1$: auxiliary loss coefficient

### 4.4 Z-Loss (Zoph et al. 2022 — ST-MoE)

Auxiliary loss penalising large logit magnitudes before softmax:

$$\mathcal{L}_z = \frac{1}{B} \sum_{b=1}^{B} \left(\log \sum_{i=1}^{N} e^{z_b^{(i)}}\right)^2$$

- $z_b$ = router logits for token $b$ before softmax
- Large logits → near-one-hot softmax → hard routing → instability
- Z-loss encourages moderate logit values; softer distributions
- Stabilises training; reduces loss spikes; combined with auxiliary load loss

### 4.5 Router Load Monitoring

Track per-expert load fraction $f_i$ throughout training:

| Status   | Condition                      | Action                              |
| -------- | ------------------------------ | ----------------------------------- |
| Healthy  | $f_i \approx 1/N$ for all $i$  | Continue training                   |
| Warning  | $f_i > 3/N$ for any expert     | Increase $\alpha$; check routing    |
| Critical | $f_i \approx 0$ for any expert | Expert collapsed; restart or reinit |

Plot: expert load histogram per layer per step. Standard monitoring in MoE training runs.

### 4.6 Expert Dropout

Randomly drop entire experts during training (set output to zero):

- Forces model to not rely on any single expert; improves redundancy
- Expert dropout rate: 10–40% during training; none at inference
- Analogy: dropout for neurons extended to expert level

### 4.7 Jitter Noise (Shazeer 2017)

Add tunable Gaussian noise to router logits before top-k selection:

$$G(x) = \text{softmax}\!\left(W_g\, x + \epsilon \cdot \text{Softplus}(W_n\, x)\right), \quad \epsilon \sim \mathcal{N}(0,1)$$

- Noise scale is learned; provides exploration of expert assignments during training
- Reduces routing collapse; encourages diverse expert usage
- Not used at inference (noise removed)

---

## 5. Expert Architecture Variants

### 5.1 Standard MoE FFN

Replace dense FFN in transformer block with MoE layer:

```
Dense Block:  x → Attention → x + Δ → FFN(x) → x + Δ
MoE Block:    x → Attention → x + Δ → Router → top-k Experts → weighted sum → x + Δ
```

- Attention is always dense (shared across all tokens)
- Only the FFN sublayer is sparse/MoE
- Ratio of MoE to dense layers: every layer (Mixtral), every other layer (Jamba), custom

### 5.2 Expert Size Choices

**Equal-size experts:** each expert = standard FFN; $N$ experts × $d_{ff}$ hidden dim.

- Mixtral 8×7B: each expert has hidden dim = 14,336 (same as dense 7B FFN)

**Fine-grained experts (DeepSeekMoE 2024):** smaller individual experts; many more of them.

- Hypothesis: finer granularity → more combinatorial flexibility in routing
- DeepSeek-V2: $N=160$ routed experts, each 1/16th size of standard; $k=6$
- More expert combinations: $\binom{160}{6} = 1.2 \times 10^{10}$ vs $\binom{8}{2} = 28$ for Mixtral

### 5.3 Shared + Routed Experts (DeepSeekMoE)

Divide $N$ experts into:

- $N_s$ **shared experts**: always active for every token (like dense FFN component)
- $N_r$ **routed experts**: top-k selected per token

$$y = \sum_{i=1}^{N_s} E_i(x) + \sum_{i \in \text{TopK}(\text{routed})} \hat{G}(x)_i\, E_i(x)$$

- Shared experts capture common/universal patterns; routed experts capture specialised patterns
- Reduces expert redundancy: shared experts prevent multiple routed experts learning the same thing
- DeepSeek-V3: $N_s=1$ shared; $N_r=256$ routed; top-8 selected

### 5.4 Expert Merging and Recycling

- **Dead experts:** experts that receive near-zero load consistently
- **Expert merging:** combine weights of dead expert with similar active expert; reinitialise
- **Expert recycling (Zoph et al. 2022):** periodically reinitialise underutilised experts
- Prevents capacity waste; all experts contribute to model quality

### 5.5 Conditional Computation Beyond FFN

| Variant                | Description                                            |
| ---------------------- | ------------------------------------------------------ |
| MoE attention heads    | Different attention patterns for different tokens      |
| MoE embedding layers   | Different embeddings per language/domain               |
| MoE transformer blocks | Route tokens to entirely different transformer stacks  |
| Modular networks       | Extreme; entirely separate subnetworks per task/domain |

### 5.6 Number of Experts Per Layer

| Strategy              | Description                                | Example                               |
| --------------------- | ------------------------------------------ | ------------------------------------- |
| **Uniform**           | All layers have same $N$                   | Mixtral, Switch                       |
| **Variable**          | More experts in later layers               | Deeper = more semantic specialisation |
| **Sparse MoE layers** | Only every $k$-th layer is MoE; rest dense | Jamba (MoE + Mamba)                   |
| **All-MoE**           | Every FFN layer is MoE                     | DeepSeek-V3                           |

---

## 6. Training MoE Models

### 6.1 MoE Training Objective

Total loss = task loss + auxiliary losses:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \alpha\, \mathcal{L}_{\text{aux}} + \beta\, \mathcal{L}_z$$

| $\alpha$ | Effect                                                     |
| -------- | ---------------------------------------------------------- |
| Too high | Router dominates training signal; experts don't specialise |
| Too low  | Expert collapse; most experts idle                         |
| Optimal  | Auxiliary loss ≈ 5–10% of total loss                       |

### 6.2 Gradient Flow Through Router

- Only selected top-k experts receive gradients for a given token
- Router receives gradient signal from all tokens it routes (via gate values)
- Gradient for expert $i$ parameters: non-zero only when expert $i$ is selected
- Low-traffic experts: sparse gradient updates; slow learning; may not specialise
- **Implication:** initial load balancing is critical; experts that start unused stay unused

### 6.3 Initialisation for MoE

- Expert weights: initialised identically (or with small perturbation to break symmetry slightly)
- If all experts initialised identically: routing is random initially → good starting diversity
- Router weights: small random initialisation; avoid early collapse to deterministic routing
- Warmup routing: some works use random routing or high noise for first $N$ steps; gradually reduce

### 6.4 Training Instability in MoE

MoE training is significantly less stable than dense model training:

- Loss spikes more frequent and more severe
- **Router logit explosion:** $W_g$ values grow → softmax saturates → near-one-hot → collapse
- Z-loss addresses this; QK-Norm equivalent needed for router
- Gradient checkpointing: activations for all $N$ experts must be stored during forward; expensive

### 6.5 Data and Expert Specialisation

Experts naturally specialise on different token types, languages, domains:

- Syntax/function words; rare words; specific languages; code vs text
- Specialisation emerges without explicit supervision; purely from routing gradient
- Measuring specialisation: for each expert, compute entropy of token-type distribution
  - Low entropy → high specialisation; high entropy → generalist

### 6.6 Curriculum Learning for MoE

**Upcycling (Komatsuzaki et al. 2022):**

1. Start with a well-trained dense model
2. Copy dense FFN weights to all $N$ experts
3. Initialise router randomly
4. Fine-tune with MoE routing; ~5–10% of original training compute

Much cheaper than training MoE from scratch. Dense model provides strong initialisation.

---

## 7. Expert Parallelism

### 7.1 Why Expert Parallelism Is Needed

At scale: all $N$ expert weight matrices must fit in GPU memory.

- DeepSeek-V3: 256 routed experts; if each expert = standard FFN, 256× more FFN memory
- Solution: distribute experts across GPUs; each GPU holds $N/\text{EP}$ experts
- **Expert parallelism (EP):** GPUs specialise in specific experts; tokens routed to correct GPU

### 7.2 All-to-All Communication

Forward pass with EP:

```
Step 1: Each GPU holds subset of expert weights
Step 2: Router computes expert assignments for local token batch
Step 3: *** All-to-All Dispatch *** — send tokens to GPU holding assigned expert
Step 4: Each GPU runs its experts on received tokens
Step 5: *** All-to-All Combine *** — return processed tokens to originating GPU
Step 6: Aggregate weighted expert outputs
```

Two all-to-all operations per MoE layer (dispatch + combine).

### 7.3 All-to-All Mathematics

- EP = number of GPUs in expert parallel group
- Each GPU sends $T/\text{EP}$ tokens to each other GPU (if perfectly balanced)
- Total data per GPU: $T \times d \times \text{bytes\_per\_elem}$ sent/received

**Worked example:**

- $T=4096$ tokens, $d=4096$, EP=64, InfiniBand 400 Gb/s = 50 GB/s
- Data per GPU: $4096 \times 4096 \times 4 / 64 = 1$ MB sent
- Time: $1\text{ MB} / 50\text{ GB/s} = 0.02$ ms per all-to-all
- Two all-to-all per layer × 32 layers = 1.28 ms total → manageable

### 7.4 Communication-Compute Overlap

All-to-all can be overlapped with expert computation on other tokens:

- Pipeline: while GPU A computes expert on batch B tokens, receive batch C tokens for next step
- DeepSeek-V3 training: **DualPipe** algorithm overlaps computation and communication; near-zero overhead
- Key requirement: computation time >> communication time per layer

### 7.5 Hierarchical Expert Parallelism

Two-level routing: coarse routing to node → fine routing to GPU within node.

| Level      | Interconnect | Bandwidth   |
| ---------- | ------------ | ----------- |
| Intra-node | NVLink       | 600 GB/s    |
| Inter-node | InfiniBand   | 50–400 GB/s |

Locality-aware routing: prefer experts on same node or GPU when quality allows.
Reduces expensive inter-node all-to-all communication.

### 7.6 Expert Parallelism + Other Parallelism

| Combination | Description                                            |
| ----------- | ------------------------------------------------------ |
| EP + DP     | Each DP replica has full expert set; EP within replica |
| EP + TP     | Each expert further split across tensor-parallel GPUs  |
| EP + PP     | Expert layers in different pipeline stages             |

DeepSeek-V3 training: EP=32 or EP=64; combined with DP; no TP (communication overhead).

---

## 8. MoE Inference Challenges

### 8.1 Memory vs Compute Asymmetry

| Metric              | Dense 37B    | MoE 671B (DeepSeek-V3) |
| ------------------- | ------------ | ---------------------- |
| Active params/token | 37B          | 37B                    |
| Total memory needed | 74 GB (BF16) | 1.34 TB (BF16)         |
| Compute per token   | Same         | Same                   |
| Min GPUs (80 GB)    | 1            | 17                     |

Naive: load all $N$ experts → memory = $N \times$ (expert size). Requires multi-GPU
inference even though compute ≈ small dense model.

### 8.2 Expert Offloading for Inference

- Keep only most frequently used experts in GPU HBM; rest on CPU DRAM or NVMe
- Load expert on demand when routed to; significant latency cost
- **Pre-fetching:** predict which experts will be needed next; load ahead of time
- **Speculative expert prefetch (2024):** use small router to predict expert assignments
  several steps ahead
- Works well for offline/batch; too slow for interactive latency SLOs

### 8.3 Expert Activation Patterns

Key observation: expert routing is highly consistent across similar inputs.

- Same token in similar contexts → same expert selected with high probability
- Expert activation locality: for a given domain, same expert subset repeatedly activated
- Exploit for caching: cache recently used experts in fast memory; evict cold experts
- Expert cache hit rate: 70–90% for typical production workloads (2024 empirical results)

### 8.4 Batch Routing Efficiency

| Batch Size | Behaviour                                                              |
| ---------- | ---------------------------------------------------------------------- |
| $B=1$      | Only $k$ experts active per token; many experts never touched per step |
| Large $B$  | More diversity in routing; all experts likely used; better utilisation |

Throughput scales more steeply with batch size for MoE than dense.
Minimum efficient batch size: $B \geq N/k$.

### 8.5 Expert Dropping at Inference

- At inference: no training signal; some works simply drop overloaded experts
- Token importance scoring: if expert is overloaded, route lower-importance tokens to backup
- Second-choice routing: each token maintains ranked expert list; use second choice if first full
- Quality impact depends on how critical the dropped expert is for that token

### 8.6 MoE Quantisation

| Component            | Quantisation Strategy                              |
| -------------------- | -------------------------------------------------- |
| Expert weights (FFN) | INT4/INT8 with GPTQ/AWQ; per-expert calibration    |
| Router weights       | Keep full precision (small; critical for accuracy) |
| Shared expert        | Quantise like dense FFN                            |
| Attention            | Standard quantisation (same as dense model)        |

Challenge: each expert's weight distribution may differ; one-size calibration less effective.

---

## 9. Routing Analysis and Interpretability

### 9.1 Measuring Expert Utilisation

Load distribution: histogram of tokens per expert over evaluation set.

Gini coefficient for expert load:

$$G = \frac{\sum_i \sum_j |f_i - f_j|}{2N \sum_i f_i}$$

- $G=0$: perfectly uniform; $G=1$: all tokens to one expert
- Monitor per layer; layer-wise routing patterns differ significantly

### 9.2 Expert Specialisation Analysis

For each expert $i$, collect all tokens routed to it over a large corpus. Compute
token distribution and specialisation metrics:

| Metric                   | What It Measures                                         |
| ------------------------ | -------------------------------------------------------- |
| Token entropy $H$        | $-\sum_t P(t \mid i)\log P(t \mid i)$; low = specialised |
| Language specialisation  | Fraction of tokens from each language                    |
| Syntactic specialisation | Fraction of POS tags (noun, verb, etc.)                  |
| Domain specialisation    | Fraction from code, math, natural language               |

### 9.3 Observed Specialisation Patterns

- Multilingual MoE: clear language-specific experts emerge (French expert, Chinese expert, code expert)
- Token frequency: some experts specialise in rare/OOV tokens; others in common tokens
- Positional patterns: some experts handle early-position tokens; others end-of-sentence
- Layer depth: shallow layers → syntactic specialisation; deep layers → semantic specialisation
- Switch Transformer analysis: experts cluster by input characteristics visible in embedding space

### 9.4 Routing Consistency

Same token in different contexts → same expert? Test routing consistency:

- Take token $t$; place in 1000 different contexts; record which expert selected
- High consistency: **token-level** routing dominant (token identity determines expert)
- Low consistency: **context-level** routing dominant (surrounding context determines expert)
- Empirical finding: routing is ~70% consistent for common tokens; less for rare tokens

### 9.5 Expert Attribution

| Method           | Description                                                           |
| ---------------- | --------------------------------------------------------------------- |
| Expert ablation  | Zero out single expert; measure capability degradation                |
| Expert probing   | Train linear probe on expert output for syntactic/semantic properties |
| Circuit analysis | Trace information flow through specific experts                       |

Active research area (2024–2026): mechanistic interpretability of MoE routing.

### 9.6 Router Confidence

Distribution of top gate value across dataset:

$$\text{Confidence} = \max_i G(x)_i$$

- High confidence: near-one-hot routing; model sure about expert assignment
- Low confidence: spread across experts; ambiguous token
- Confidence correlates with token frequency: common tokens → high routing confidence
- Routing confidence increases with training (router becomes more decisive over time)

---

## 10. Advanced Routing Methods

### 10.1 Two-Stage Routing

- First stage: coarse routing to group of experts (e.g. "code group" vs "language group")
- Second stage: fine routing within selected group
- Hierarchical MoE: tree-structured routing; reduces routing search space
- Benefit: $O(\log N)$ routing cost vs $O(N)$ for large $N$
- Used in very large $N$ settings ($N=512, 1024$)

### 10.2 Input-Conditioned Routing (Conditional MoE)

Router conditioned beyond just the token embedding:

| Condition   | Description                                                       |
| ----------- | ----------------------------------------------------------------- |
| Layer index | Routing aware of which transformer layer is being computed        |
| Position    | Position-aware routing; different experts for different positions |
| Task/domain | Conditioning on task embedding; explicit domain routing           |
| History     | Routing conditioned on previous token's expert assignment         |

### 10.3 Reinforcement Learning for Routing

- Train router with RL objective: maximise task reward subject to load balance constraint
- Router as policy: action = expert assignment; reward = quality + load balance
- Advantage: directly optimises task metric; not proxy auxiliary loss
- Challenge: discrete action space; high variance; computationally expensive
- Research area (2024–2026): RL routing not yet mainstream

### 10.4 Mixture of Depths (MoD — Raposo et al. 2024)

Route tokens through different numbers of layers, not just different experts.

- "Compute allocation" problem: some tokens need deep processing; others need shallow
- Router decides: process token at this layer OR skip (identity mapping)?
- Simple tokens skip many layers; complex tokens use full depth
- Combines with MoE: joint MoD + MoE routing for maximum efficiency

### 10.5 Dynamic Expert Count

Adaptive top-k: not fixed $k$, but variable number of experts per token.

| Strategy         | Description                                                      |
| ---------------- | ---------------------------------------------------------------- |
| Confidence-based | High confidence → $k=1$; uncertain → $k=3$                       |
| Budget-based     | Given compute budget $B$, allocate more experts to harder tokens |
| Token difficulty | Lightweight scorer predicts optimal $k$ per token                |

### 10.6 Recurrent Routing

- Multiple rounds of routing per MoE layer (iterative routing)
- Round 1: initial assignment; Round 2: refine based on expert feedback
- Soft assignment in early rounds; hard top-k in final round
- Research prototype; not yet mainstream

---

## 11. Specific MoE Architectures

### 11.1 Switch Transformer (Fedus et al. 2021)

| Property         | Value                                                    |
| ---------------- | -------------------------------------------------------- |
| Total parameters | 1.6T                                                     |
| Routing          | Top-1 ($k=1$): single expert per token; maximum sparsity |
| Auxiliary loss   | Load balancing via $f_i \cdot P_i$ sum                   |
| Capacity factor  | CF = 1.25; token dropping for overflow                   |
| Key finding      | Top-1 competitive with top-2; simpler routing works      |
| Stability        | FP32 router; BF16 experts                                |

### 11.2 GShard (Lepikhin et al. 2020)

- Top-2 routing; first expert mandatory, second random with probability ∝ gate value
- Local dispatch groups: balance within local group of tokens (not globally)
- Scaled to 600B parameters for multilingual translation
- First demonstration of MoE outperforming dense at equivalent compute

### 11.3 Mixtral 8×7B (Mistral AI 2023)

| Property              | Value                                                      |
| --------------------- | ---------------------------------------------------------- |
| Experts per MoE layer | 8                                                          |
| Routing               | Top-2 (2 experts active per token)                         |
| Expert size           | Each expert = standard 7B FFN (hidden dim 14,336)          |
| Total parameters      | ~46.7B                                                     |
| Active per token      | ~12.9B                                                     |
| Architecture          | Every FFN layer is MoE; attention dense; no shared experts |
| Quality               | Matches or exceeds LLaMA-2 70B at 12.9B active compute     |

### 11.4 Mixtral 8×22B (Mistral AI 2024)

8 experts; top-2 routing; 22B-scale experts. Total: ~141B parameters; active: ~39B.
Extended context: 65K tokens; strong multilingual and code performance.

### 11.5 DeepSeekMoE (Dai et al. 2024)

- Fine-grained experts: $N=64$ small experts instead of 8 large
- Shared experts: $N_s=2$ always-active + $N_r=62$ routed; top-6 from routed
- Better expert utilisation; reduced load imbalance; stronger specialisation

### 11.6 DeepSeek-V2 (2024)

- MLA (low-rank KV compression) + DeepSeekMoE
- $N=160$ routed; $N_s=2$ shared; top-6 routed per token
- Total: 236B; active: 21B
- Device-level load balance loss (not just expert level)

### 11.7 DeepSeek-V3 (2024)

| Property           | Value                                              |
| ------------------ | -------------------------------------------------- |
| Routed experts     | 256                                                |
| Shared experts     | 1                                                  |
| Top-k              | 8                                                  |
| Total parameters   | 671B                                               |
| Active per token   | 37B                                                |
| Load balancing     | Auxiliary-loss-free; dynamic bias on router logits |
| Training precision | FP8 throughout; DualPipe overlap                   |
| Training cost      | ~2.79M H800 GPU hours (~$5.5M)                     |

### 11.8 Grok-1 (xAI 2024)

314B total; 8 experts; top-2 routing; ~86B active. Open-sourced (Apache 2.0);
largest open MoE at release.

### 11.9 Jamba (AI21 Labs 2024)

Hybrid: alternates Transformer (MoE) + Mamba (SSM) layers.

- MoE layers: 8 experts; top-2 routing
- 256K context window; KV cache only for attention layers (Mamba has fixed state)
- Demonstrates MoE + SSM hybrid viability

---

## 12. Mathematical Analysis

### 12.1 Capacity and Quality Tradeoff

For fixed compute $C$ and quality target $Q$:

- Dense model: $N_{\text{dense}}$ parameters; quality $Q(N_{\text{dense}}, C)$
- MoE model: $N_{\text{moe}} = N_{\text{dense}} \times N/k$ total parameters; active $= N_{\text{dense}}$
- Empirical scaling: quality improves roughly as $\log(N_{\text{moe}}/k) = \log(N_{\text{moe}}) - \log(k)$
- Diminishing returns from more experts at fixed active count

### 12.2 Expert Routing as Discrete Latent Variable

Token $x$; expert assignment $z \in \{1, \ldots, N\}$; expert output $E_i(x)$.

Marginalised output (soft MoE):

$$y = \sum_{i=1}^{N} P(z=i \mid x) \cdot E_i(x) = \mathbb{E}_{z \sim G(x)}[E_z(x)]$$

- Top-k routing: approximate posterior with sparse distribution
- **EM interpretation:** E-step = compute routing probabilities; M-step = update expert parameters

### 12.3 Information-Theoretic View of Routing

Router computes: $I(x \to \text{expert})$ = mutual information between input and expert assignment.

- High $I$: routing is input-dependent; strong specialisation
- Low $I$: routing is near-uniform; weak specialisation
- Maximising MI between routing and input characteristics encourages specialisation
- **Balancing objective:** maximise $I$ while constraining load uniformity

### 12.4 Load Balancing as Constrained Optimisation

$$\min_\theta \mathcal{L}_{\text{CE}}(\theta) \quad \text{s.t.} \quad f_i \leq \frac{1+\epsilon}{N} \quad \forall\, i$$

Lagrangian relaxation:

$$\mathcal{L} = \mathcal{L}_{\text{CE}} + \sum_i \lambda_i \max\!\left(0,\; f_i - \frac{1+\epsilon}{N}\right)$$

Standard auxiliary loss is an approximation of this constrained optimisation.

### 12.5 Expert Capacity as Bin Packing

$T$ tokens; $N$ bins (experts); capacity $C$ per bin. Token $i$ has weight 1; assign to
exactly $k$ bins.

Expected overflow with uniform random routing (Poisson approximation):

$$P(\text{overflow}) \approx 1 - \text{PoissonCDF}\!\left(C;\; T \cdot k / N\right)$$

For $CF = 1.25$: overflow probability ≈ 5–10% of tokens.

### 12.6 Auxiliary Loss Weight Sensitivity

| $\alpha$ Range      | Effect                                                            |
| ------------------- | ----------------------------------------------------------------- |
| Too high (>0.1)     | Auxiliary loss dominates; uniform routing but poor specialisation |
| Too low (<0.001)    | Expert collapse; most capacity wasted                             |
| Optimal (0.01–0.05) | Auxiliary loss ≈ 5–10% of total loss                              |

### 12.7 Auxiliary-Loss-Free Load Balancing (DeepSeek-V3)

Add per-expert learnable bias $b_i$ to router logits:

$$\text{Modified logit for expert } i = G(x)_i + b_i$$

Bias update rule: monitor load per expert; increase $b_i$ for underloaded; decrease for overloaded:

$$b_i \leftarrow b_i + \gamma \cdot \text{sign}(c_i - \bar{c})$$

- $c_i$ = token count for expert $i$ in recent batch; $\bar{c} = T/N$ target count
- No auxiliary loss in total objective; load balancing purely through bias adaptation
- Result: better task quality (no auxiliary loss conflict); similar load balance

---

## 13. Common Mistakes

| Mistake                                    | Why It's Wrong                                                                                | Fix                                                                        |
| ------------------------------------------ | --------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| "MoE is always cheaper than dense"         | Memory cost = all N expert weights; compute = k active experts; memory dominates at inference | Account for total memory, not just active compute                          |
| "Expert collapse is rare"                  | Without auxiliary loss, collapse happens in first few thousand steps reliably                 | Monitor expert load from step 1; tune α before full run                    |
| "More experts always better"               | Routing harder; load balance harder; communication overhead grows                             | Tune N and k together; fine-grained experts better than coarse             |
| "Top-1 routing is inferior to top-2"       | Switch Transformer showed top-1 competitive; simpler routing has advantages                   | Evaluate both; task-specific choice                                        |
| "Expert specialisation can be controlled"  | Specialisation emerges from training; cannot be explicitly assigned                           | Design auxiliary objectives to encourage desired specialisation indirectly |
| "Auxiliary loss weight α is not sensitive" | α is highly sensitive; wrong value → collapse or degraded quality                             | Grid search α on small proxy model; use μP to transfer                     |
| "All-to-all communication is negligible"   | At large EP, all-to-all can be 30–50% of step time without overlap                            | Implement compute-communication overlap; measure ratio explicitly          |
| "MoE models fine-tune like dense models"   | Expert routing changes under distribution shift; different experts activate                   | Lower LR for router; monitor routing distribution shift                    |
| "Quantising MoE is same as dense"          | Each expert has different weight distribution; shared calibration insufficient                | Per-expert calibration; expert-aware quantisation                          |
| "Expert capacity CF=1.0 is fine"           | CF=1.0 means any imbalance drops tokens; realistic imbalance ~20%                             | Use CF=1.25–1.5 for training; CF=1.0 only if load near-perfect             |

---

## 14. Exercises

1. **Routing by hand** — given router weights $W_g \in \mathbb{R}^{4 \times 3}$, input
   $x \in \mathbb{R}^3$: compute $G(x) = \text{softmax}(W_g x)$; identify top-2 experts;
   compute renormalised gates; compute weighted expert output.

2. **Auxiliary loss** — batch of $T=8$ tokens; 4 experts; $f_i = [0.5, 0.3, 0.15, 0.05]$;
   $P_i = [0.4, 0.3, 0.2, 0.1]$; compute $L_{\text{aux}} = \sum f_i P_i$; compare to
   perfectly uniform case.

3. **Capacity and overflow** — $T=1024$ tokens; $N=8$ experts; $k=2$; $CF=1.25$;
   compute capacity $C$; if expert 1 receives 200 tokens, how many are dropped?

4. **Parameter count** — dense model: $d=4096$, $d_{ff}=16384$; MoE replacement: $N=8$
   experts, same $d_{ff}$; compute total MoE params vs dense params; active params per
   token for $k=2$.

5. **Expert utilisation** — load distribution $[0.35, 0.25, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01]$
   for 8 experts; compute Gini coefficient; is this healthy?

6. **All-to-all latency** — $EP=32$ GPUs; $T=4096$ tokens; $d=4096$; BF16; NVLink 600 GB/s;
   compute data sent per GPU; estimate all-to-all time; compare to expert compute time.

7. **MoE vs dense scaling** — 10B active params; MoE has $N=64$ experts ($k=2$);
   estimate relative quality improvement using $\log$ scaling; is 64× extra capacity
   worth the routing overhead?

8. **Z-loss calculation** — router logits $z = [3.2, 1.1, -0.5, 2.8]$ (4 experts):
   compute $\mathcal{L}_z = \left(\log \sum e^{z_i}\right)^2$; compare to
   $z_{\text{uniform}} = [1, 1, 1, 1]$; interpret the regularisation effect.

---

## 15. Why This Matters for AI (2026 Perspective)

| Aspect                  | Impact                                                                                   |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| **Frontier capability** | Nearly all leading models (GPT-4, Gemini, Claude, DeepSeek) believed to use MoE          |
| **Cost efficiency**     | 5–10× more parameters at same training compute; dramatically better $/quality            |
| **Specialisation**      | Experts specialise by language, domain, task; single model excels everywhere             |
| **Open source**         | Mixtral and DeepSeek open-sourced competitive MoE models; democratises access            |
| **Inference cost**      | Active compute ≈ small dense model despite massive total capacity                        |
| **Multilingual**        | Language-specific experts emerge naturally; better multilingual performance              |
| **Research frontier**   | Routing algorithms, expert specialisation, auxiliary-loss-free training all active       |
| **Hardware design**     | MoE's all-to-all pattern drives NVLink bandwidth requirements                            |
| **Scaling laws**        | MoE shifts scaling laws; expert count is now a fourth axis alongside N, D, C             |
| **Interpretability**    | Expert routing provides natural modularity; mechanistic interpretability active research |

---

## Conceptual Bridge

MoE replaces the monolithic FFN with a conditional computation graph: every token
activates a different specialised subnetwork. The model is simultaneously many models.
Routing is the bridge between the input's content and the model's specialised capacity.

Next: **Quantization and Distillation** — how to compress model capacity
for efficient deployment.

```
… → [Attention] → h → [Router] → [Expert k₁] + [Expert k₂] → weighted sum → h' → …
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                      THIS section
```

---

[← Efficient Attention and Inference](../09-Efficient-Attention-and-Inference/notes.md) | [Home](../../README.md) | [Quantization and Distillation →](../11-Quantization-and-Distillation/notes.md)
