# Efficient Attention and Inference

[← Scaling Laws](../08-Scaling-Laws/notes.md) | [Home](../../README.md) | [Mixture of Experts and Routing →](../10-Mixture-of-Experts-and-Routing/notes.md)

---

## 1. Intuition

### 1.1 What Is Inference?

Inference is the process of running a trained model to generate outputs from new inputs.
Training happens once; inference happens billions of times per day across all deployed
models. Every ChatGPT response, every Copilot suggestion, every Claude reply is one
inference call.

Inference cost at scale dominates total AI compute spend — often 10× training cost over
a model's lifetime. Optimising inference is therefore the highest-leverage engineering
problem in deployed AI.

### 1.2 Why Inference Is Different from Training

| Aspect          | Training                      | Inference                  |
| --------------- | ----------------------------- | -------------------------- |
| **Batch size**  | Large (thousands)             | Small (often 1)            |
| **Pass**        | Forward + backward            | Forward only               |
| **Activations** | All stored for backward       | Minimal; discard after use |
| **Duration**    | Weeks of computation          | Milliseconds per token     |
| **Orientation** | Throughput (tokens/sec total) | Latency (time per token)   |
| **Bottleneck**  | Activations + gradient memory | KV cache + weight loading  |

Training is throughput-oriented: maximise tokens/second overall.
Inference is latency-oriented: minimise **time-to-first-token (TTFT)** and
**time-per-output-token (TPOT)**.

### 1.3 The Two Phases of Autoregressive Inference

**Prefill phase** — process entire input prompt in parallel; compute KV cache for all
input tokens.

- Compute-bound: many tokens processed simultaneously; GPU utilisation is high
- Duration: proportional to prompt length; can be seconds for long prompts

**Decode phase** — generate one token at a time; each step attends to all previous tokens.

- Memory-bandwidth-bound: load all weights + KV cache for every single token generated
- Duration: proportional to output length; typically 10–100 ms per token

```
┌────────────────────────────────────────────────────────────────┐
│  Prefill                           │  Decode                  │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┐             │  ┌─┐ ┌─┐ ┌─┐ ┌─┐       │
│  │1│2│3│4│5│6│7│8│9│  parallel    │  │A│→│B│→│C│→│D│ serial  │
│  └─┴─┴─┴─┴─┴─┴─┴─┴─┘             │  └─┘ └─┘ └─┘ └─┘       │
│  ← compute-bound →                │  ← bandwidth-bound →    │
└────────────────────────────────────────────────────────────────┘
```

Most latency optimisation targets the decode phase.

### 1.4 The Memory Bandwidth Wall

A100 SXM specifications:

- Compute: 312 TFLOPS (BF16)
- Bandwidth: 2 TB/s HBM

**Arithmetic intensity** is the ratio of compute work to data movement:

$$I = \frac{\text{FLOPs}}{\text{bytes accessed}}$$

- Dense matrix multiply (large batch): high I → compute-bound
- Decode attention (batch=1): read all weights + KV cache for 1 token → extremely low I → bandwidth-bound

```
                  Peak FLOPS
Throughput  ──────────────────────────────
(FLOPS)    /
          /
         /       bandwidth-bound     compute-bound
        /         region              region
       /
      /
     /─────────────────────────────────────
     0              I*              Arithmetic Intensity
                    ↑
              Ridge Point
         I* = 312 TFLOPS / 2 TB/s = 156 FLOP/byte
```

Operations below the ridge point are bandwidth-limited regardless of GPU compute power.

### 1.5 Key Metrics

| Metric               | Definition                                             | Typical Target     |
| -------------------- | ------------------------------------------------------ | ------------------ |
| **TTFT**             | Time from request arrival to first output token        | < 500 ms           |
| **TPOT**             | Time between output tokens (inverse decode throughput) | < 50 ms            |
| **Throughput**       | Total tokens generated per second (all requests)       | Hardware-dependent |
| **Memory footprint** | GPU memory consumed by model + KV cache                | Fit in GPU HBM     |
| **$/M tokens**       | Cost per million tokens                                | Business metric    |

### 1.6 Historical Timeline

| Year | Development                                     | Impact                                 |
| ---- | ----------------------------------------------- | -------------------------------------- |
| 2020 | GPT-3 inference; KV cache first widely used     | Baseline autoregressive                |
| 2022 | FlashAttention (Dao et al.)                     | 2–4× speedup; IO-aware exact attention |
| 2022 | Continuous batching (Orca)                      | 23× throughput improvement             |
| 2023 | FlashAttention-2; PagedAttention (vLLM)         | Production inference systems           |
| 2023 | AWQ, GPTQ; speculative decoding                 | 4-bit inference; 2–3× decode speedup   |
| 2024 | FlashAttention-3; FP8 inference; MLA (DeepSeek) | KV compression; hardware-optimal       |
| 2024 | SGLang, TensorRT-LLM                            | Production inference frameworks        |
| 2025 | Multi-token prediction; spec-decode at scale    | Disaggregated prefill/decode           |
| 2026 | Near-hardware-optimal inference                 | 1M+ token context in production        |

### 1.7 Pipeline Position

```
User Request → [Tokeniser] → Prompt Tokens → [Prefill] → KV Cache
                                                           ↓
KV Cache → [Decode Loop] → Output Tokens → [Detokeniser] → Response
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                THIS section
```

---

## 2. Formal Definitions

### 2.1 Autoregressive Generation — Formal

Model parameters θ; input prompt **x** = (x₁, …, xₘ).
Generate output **y** = (y₁, y₂, …, yₙ) token by token:

$$y_t \sim P_\theta(y_t \mid x_1, \ldots, x_m, y_1, \ldots, y_{t-1})$$

Full sequence probability:

$$P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{n} P_\theta(y_t \mid \mathbf{x}, y_{<t})$$

Each step requires one full forward pass through all L transformer layers.

### 2.2 KV Cache — Formal Definition

At layer l, position i:

$$k_i^l = x_i W_K^l, \quad v_i^l = x_i W_V^l$$

KV cache C at step t:

$$\mathcal{C}_t = \{(k_i^l, v_i^l) : l \in [1,L],\; i \in [1, m+t-1]\}$$

At step t, compute only the new key/value for the current token; reuse all previous.

| Mode             | Compute per generation | Approach                             |
| ---------------- | ---------------------- | ------------------------------------ |
| Without KV cache | O(t²L) per generation  | Recompute all keys/values every step |
| With KV cache    | O(tL) per generation   | Store and reuse previous keys/values |

KV cache grows linearly with sequence length — the primary memory bottleneck at
inference time.

### 2.3 Attention During Decode — Single Token

At decode step t, the new query:

$$q = y_t W^Q$$

Attend over all cached keys/values:

$$o_t = \text{Attention}(q, K_{\text{cache}}, V_{\text{cache}}) = \text{softmax}\!\left(\frac{q\,K_{\text{cache}}^\top}{\sqrt{d_k}}\right) V_{\text{cache}}$$

Dimensions:

- q ∈ ℝ^{1×d_k} (single token query)
- K_cache ∈ ℝ^{s×d_k} (all past tokens)
- s = current sequence length

Attention is now a **matrix-vector** multiply, not matrix-matrix: much lower arithmetic
intensity.

### 2.4 Roofline Model

For an operation with F FLOPs accessing B bytes:

$$I = \frac{F}{B} \quad \text{(FLOPs/byte)}$$

Ridge point:

$$I^* = \frac{\text{Peak FLOPS}}{\text{Peak Bandwidth}}$$

**A100**: I\* = 312 × 10¹² / (2 × 10¹²) = **156 FLOPs/byte**

| Condition | Regime          | Actual throughput |
| --------- | --------------- | ----------------- |
| I < I\*   | Bandwidth-bound | Bandwidth × I     |
| I > I\*   | Compute-bound   | Peak FLOPS        |

**Decode attention example** (batch=1, s=2048, d_k=128, h=32):

- FLOPs ≈ 2 × s × d_k × h = 2 × 2048 × 128 × 32 = 16.8 MFLOPs
- Bytes ≈ 2 × (s × d_k × h × 2) = 2 × 2048 × 128 × 32 × 2 = 33.6 MB
- I = 16.8M / 33.6M = **0.5 FLOPs/byte** → deeply bandwidth-bound

---

## 3. FlashAttention — Complete Treatment

### 3.1 The Problem: Memory Bandwidth in Standard Attention

Standard attention materialises the full n×n attention matrix S in HBM (GPU main memory).

Memory access pattern:

1. Write Q, K, V to HBM: 3 × n × d bytes
2. Read Q, K for scores: 2 × n × d bytes; write S: n² bytes
3. Read S for softmax: n² bytes; write P: n² bytes
4. Read P, V for output: n² + n×d bytes; write O: n×d bytes

Total HBM accesses: **O(n²)** — dominated by the attention matrix for large n.

Compute: O(n²d) — but the GPU is bandwidth-limited, not compute-limited.

### 3.2 FlashAttention Core Idea (Dao et al. 2022)

Never materialise the full n×n attention matrix in HBM.

Instead: tile Q, K, V into blocks; compute attention block by block entirely within
**SRAM** (on-chip shared memory).

| Memory         | Bandwidth | Size   |
| -------------- | --------- | ------ |
| SRAM (on-chip) | ~20 TB/s  | ~20 MB |
| HBM (GPU main) | ~2 TB/s   | 80 GB  |

SRAM is ~100× faster than HBM but very small (~20 MB vs 80 GB).

**Key challenge**: softmax requires the full row sum for normalisation — cannot tile naively.

### 3.3 Online Softmax — The Mathematical Foundation

**Standard softmax** is a two-pass algorithm:

- Pass 1: compute max m*i = max_j S*{ij} (numerical stability)
- Pass 2: compute sum l*i = Σ_j exp(S*{ij} − m*i); compute A*{ij} = exp(S\_{ij} − m_i) / l_i

**Online softmax**: update max and sum incrementally as new blocks arrive.

For block b with scores S_b:

$$m_{\text{new}} = \max(m_{\text{old}},\;\max(S_b))$$

$$l_{\text{new}} = l_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) + \sum_j \exp(S_{b,j} - m_{\text{new}})$$

$$O_{\text{new}} = \frac{l_{\text{old}} \cdot \exp(m_{\text{old}} - m_{\text{new}}) \cdot O_{\text{old}} + \exp(S_b - m_{\text{new}})\,V_b}{l_{\text{new}}}$$

This rescaling trick allows **exact** softmax computation without ever storing the full
S matrix.

**Worked example** — two blocks:

Block 1: scores [3, 1, 4], values [v₁, v₂, v₃]

```
m₁ = 4
l₁ = exp(3−4) + exp(1−4) + exp(4−4) = 0.368 + 0.050 + 1.0 = 1.418
O₁ = (0.368·v₁ + 0.050·v₂ + 1.0·v₃) / 1.418
```

Block 2: scores [5, 2], values [v₄, v₅]

```
m₂ = max(4, 5) = 5
l₂ = 1.418 · exp(4−5) + exp(5−5) + exp(2−5)
   = 1.418 · 0.368 + 1.0 + 0.050 = 0.522 + 1.0 + 0.050 = 1.572
O₂ = (0.522 · O₁ + 1.0·v₄ + 0.050·v₅) / 1.572
```

Result is identical to computing softmax over all 5 scores at once.

### 3.4 FlashAttention Algorithm

```
Tile Q into blocks Q₁, Q₂, …, Q_{T_q}     (T_q = ⌈n/B_q⌉)
Tile K, V into blocks (K₁,V₁), …, (K_{T_k},V_{T_k})  (T_k = ⌈n/B_k⌉)

For each query block Qᵢ:
    Initialise: O = 0,  m = −∞,  l = 0
    For each key/value block (Kⱼ, Vⱼ):
        Load Qᵢ, Kⱼ, Vⱼ from HBM to SRAM
        Compute Sᵢⱼ = Qᵢ Kⱼᵀ / √dₖ     (in SRAM)
        Apply mask if needed              (in SRAM)
        Update (O, m, l) using online softmax rescaling  (in SRAM)
    Write final Oᵢ to HBM
```

HBM accesses: **O(nd)** — read Q, K, V once each; never write intermediate matrices.

This is a reduction from O(n²) to O(nd) memory accesses: a massive IO improvement
for large n.

### 3.5 FlashAttention Complexity

|                   | Standard Attention | FlashAttention |
| ----------------- | ------------------ | -------------- |
| Compute           | O(n²d)             | O(n²d)         |
| HBM memory        | O(n²)              | O(nd)          |
| HBM IO            | O(n² + nd)         | O(n²d / M)     |
| Practical speedup | 1×                 | 2–4×           |

M = SRAM size. IO complexity is reduced by a factor of approximately n/M.

No asymptotic compute improvement — the IO improvement is the win.

**Numerical example** — n = 8192, d = 128, M = 100 KB:

| Metric     | Standard          | FlashAttention       | Ratio         |
| ---------- | ----------------- | -------------------- | ------------- |
| HBM IO     | n² = 67M elements | n²d/M ≈ 86M elements | ~0.8×         |
| HBM memory | n² = 67M stored   | 0 intermediate       | ∞ improvement |

The real win is eliminating the O(n²) intermediate storage.

### 3.6 FlashAttention-2 (Dao 2023)

Improvements over FlashAttention-1:

- Better work partitioning: minimise non-matmul FLOPs (exp, rescaling)
- Parallelise across **sequence** dimension (not just batch/heads)
- Fewer synchronisation barriers across warps
- Causal attention optimisation: skip upper-triangle K blocks → ~2× for causal masks

Result: ~2× faster than FA-1; ~6× faster than standard PyTorch attention.

### 3.7 FlashAttention-3 (Shah et al. 2024)

Hopper architecture (H100) specific optimisations:

| Technique             | Mechanism                                                  |
| --------------------- | ---------------------------------------------------------- |
| Asynchronous pipeline | Overlap WGMMA (tensor core matmul) with GMEM loads         |
| Ping-pong scheduling  | Alternate between two warpgroups; hide memory latency      |
| FP8 support           | E4M3 for Q, K; E5M2 for V; FP32 accumulation for softmax   |
| Incoherent processing | Random orthogonal transform reduces FP8 quantisation error |

Result: ~1.5–2× faster than FA-2 on H100; up to 75% of theoretical FP16 throughput.

### 3.8 Flash-Decoding (Tri Dao et al. 2023)

**Problem**: standard FA parallelises over batch and heads. During decode (batch=1,
heads=H), parallelism is low → GPU underutilised.

**Solution**: split KV cache along **sequence** dimension across thread blocks.

- Each thread block computes partial attention over its KV slice
- Reduce partial results with online softmax rescaling

```
KV Cache:  [──────────────────────────────────────────]
            ↓           ↓           ↓           ↓
         Block 0     Block 1     Block 2     Block 3
            ↓           ↓           ↓           ↓
       Partial attn  Partial     Partial     Partial
            \           |           |           /
             \          |           |          /
              ──── Online softmax reduce ────
                         ↓
                    Final output
```

Speedup: up to 8× for long sequences (s = 32K+) during decode.
Standard in vLLM, TensorRT-LLM as of 2024.

---

## 4. KV Cache Management

### 4.1 KV Cache Memory Formula

$$\text{KV size (bytes)} = 2 \times L \times H_{\text{kv}} \times d_k \times s \times b$$

where:

- Factor 2: K and V
- L: number of layers
- H_kv: number of KV heads (= H for MHA, = G for GQA, = 1 for MQA)
- d_k: head dimension
- s: sequence length (grows during generation)
- b: bytes per element (2 for BF16, 1 for INT8, 0.5 for INT4)

**Worked examples:**

**LLaMA-3 8B** (L=32, H_kv=8 GQA groups, d_k=128, s=8192, BF16):

$$2 \times 32 \times 8 \times 128 \times 8192 \times 2 = 1.07\text{ GB}$$

**LLaMA-3 70B** (L=80, H_kv=8, d_k=128, s=128K, BF16):

$$2 \times 80 \times 8 \times 128 \times 131072 \times 2 = 42.9\text{ GB}$$

A single 128K-context request on the 70B model consumes almost an entire A100's HBM
just for KV cache.

### 4.2 PagedAttention (Kwon et al. 2023 — vLLM)

**Problem**: KV cache for different requests has variable and unknown final length.

- Naive allocation: pre-allocate max_length KV cache per request
- Result: 60–80% GPU memory wasted on internal and external fragmentation

**Solution**: physical pages of fixed block size (e.g., 16 tokens per block).

- KV cache stored in non-contiguous physical blocks
- Logical-to-physical mapping table (like OS page tables)
- Allocate new blocks only when needed; no pre-allocation waste

```
Logical view (Request A):  [Block 0] [Block 1] [Block 2] [Block 3]
                              ↓          ↓          ↓          ↓
Physical GPU memory:       [  3   ] [  7   ] [  1   ] [ 12  ]
                              ↑          ↑          ↑
Logical view (Request B):  [Block 0] [Block 1] [Block 2]
                              ↓          ↓          ↓
Physical GPU memory:       [  5   ] [  9   ] [  4   ]
```

Result: near-zero fragmentation; **2–4× more requests** fit in same GPU memory.

### 4.3 PagedAttention Mathematics

Block size B_k tokens. Request of length s needs ⌈s/B_k⌉ blocks.

**Memory utilisation comparison:**

| Allocation          | Formula                                       | Typical utilisation |
| ------------------- | --------------------------------------------- | ------------------- |
| Static (max length) | s_max × per_token_KV × num_requests           | ~40%                |
| PagedAttention      | actual_length × per_token_KV + block_overhead | >95%                |

Block table: maps logical block index → physical block address in GPU memory.

Memory sharing: multiple requests can share physical blocks (e.g., common system prompt).

**Fragmentation example** — 8 requests, max_length=2048, avg actual=512:

Static: 8 × 2048 = 16384 token-slots allocated; 8 × 512 = 4096 used → **25% utilisation**

PagedAttention (B_k=16): 8 × ⌈512/16⌉ = 8 × 32 = 256 blocks allocated;
waste = at most 8 × 15 = 120 tokens → **97% utilisation**

### 4.4 Prefix Caching

System prompts and common prefixes appear across many requests.

- Compute KV cache for shared prefix once; reuse across all requests with that prefix
- Hash-based prefix identification: hash(token_ids) → check if KV already cached
- Savings: 100% of prefill compute for cached portion

```
Request 1: [System Prompt] + [User query A]
Request 2: [System Prompt] + [User query B]      ← share KV for system prompt
Request 3: [System Prompt] + [User query C]
```

**RadixAttention** (SGLang 2024): tree-structured prefix cache; automatic sharing of
any common prefix in a radix tree data structure.

### 4.5 KV Cache Quantisation

Store KV cache in lower precision:

| Format          | Memory reduction | Quality impact                     |
| --------------- | ---------------- | ---------------------------------- |
| BF16 (baseline) | 1×               | —                                  |
| INT8            | 2×               | Minimal                            |
| INT4 (NF4/GPTQ) | 4×               | Small degradation                  |
| FP8 E4M3        | 2×               | Minimal; H100 hardware accelerated |

Per-token quantisation: compute scale factor per token before storing K, V.
Grouped quantisation: share scale factor across g tokens; balances quality and overhead.

Result: **2–4× more sequence length or batch size** for same GPU memory.

### 4.6 KV Cache Eviction Strategies

For streaming contexts that exceed any fixed KV cache budget:

**StreamingLLM** (Xiao et al. 2023):

- Observation: BOS token always receives high attention ("attention sink")
- Keep attention sink tokens (first 4) + recent window of size w
- Evict middle tokens
- Memory: **O(w)** constant regardless of sequence length → infinite streaming

```
Full KV cache:   [sink₁..₄] [old tokens .... to evict ....] [recent window w]
After eviction:  [sink₁..₄] [recent window w]
                 ← kept →                    ← kept →
```

**H2O — Heavy Hitter Oracle** (Zhang et al. 2023):

- Tokens that accumulate high attention scores ("heavy hitters") are kept
- Greedy eviction: remove lowest-score token when budget exceeded

**ScissorHands** (Liu et al. 2023):

- Attention patterns persist — tokens important early stay important
- Use first few decode steps to identify important KV pairs; fix eviction early

**SnapKV** (Li et al. 2024):

- Observe which keys receive attention during prefill; keep those
- Cluster-based selection: keep representative KV pairs per head

**PyramidKV** (2024):

- Different KV budget per layer: lower layers need more context than upper layers
- Allocate more KV slots to early layers; fewer to later layers

### 4.7 Cross-Layer KV Sharing

Adjacent transformer layers often have similar attention patterns.

**CLA — Cross-Layer Attention** (Brandon et al. 2024):

- Pairs of adjacent layers share the same KV cache
- Odd layers compute new KV; even layers reuse from previous layer
- **2× KV cache reduction**; small quality loss

**MLA — Multi-head Latent Attention** (DeepSeek-V2 2024):

- Compress KV into low-rank latent c ∈ ℝ^{d_c} per token
- Cache only compressed latent instead of full K, V ∈ ℝ^d
- d_c ≪ d; at attention time: K = W_K^U · c, V = W_V^U · c (upproject from latent)

DeepSeek-V2: d_c = 512 vs d = 5120 → **5.75× KV cache reduction vs MHA**

Key trick: absorb the up-projection into W^Q so no extra compute at inference time:

$$q^\top K = q^\top W_K^U c = (W_K^{U\top} q)^\top c$$

Pre-compute the absorbed query; attention scores computed directly from compressed latent.

---

## 5. Batching Strategies

### 5.1 Static Batching

Group a fixed number of requests into a batch; process together.

- All requests in batch run for max(output_lengths) steps
- Short requests padded with dummy tokens; GPU compute wasted on padding

```
Request A: [████████████░░░░░░░░░░░░]   (12 tokens, padded to 24)
Request B: [████████████████████████]   (24 tokens, no padding)
Request C: [████░░░░░░░░░░░░░░░░░░░░]   (4 tokens, padded to 24)
                                   pad waste ──▲
```

GPU utilisation: often <20% due to padding waste.

### 5.2 Continuous Batching (Orca — Yu et al. 2022)

**Key insight**: different requests finish at different times.

After each decode step: check if any request is done; immediately insert new requests.
"Iteration-level scheduling": batch composition changes every token step.

```
Step 1:  [A₁ B₁ C₁ D₁]   ← all 4 active
Step 2:  [A₂ B₂ C₂ D₂]
Step 3:  [A₃ B₃ ── D₃]   ← C finishes; slot empty
Step 4:  [A₄ B₄ E₁ D₄]   ← E joins immediately
Step 5:  [── B₅ E₂ D₅]   ← A finishes; F joins next step
```

No padding waste: every GPU operation processes a real token.

Result: up to **23× throughput improvement** over static batching.

Standard in all production inference systems (vLLM, TensorRT-LLM, SGLang).

### 5.3 Dynamic Batching Mathematics

At time t: batch contains requests at different positions in their generation.
Mixed batch: some requests in prefill phase; others in decode phase.

**Chunked prefill** (2024): split long prefill into chunks of size C; interleave with
decode steps.

- Prevents long prefill from blocking decode for other requests
- Reduces TTFT variance at the cost of slightly higher average TTFT
- Chunk size C: tradeoff between prefill latency and decode throughput

### 5.4 Batch Size and Throughput

Decode throughput ∝ batch size (up to memory limit).

Larger batch: same weight-loading cost amortised over more tokens.

Arithmetic intensity with batch size B, sequence length s, hidden dimension d:

$$I(B) = \frac{2 \cdot s \cdot d \cdot B}{(2 \cdot s \cdot d + d^2) \cdot \text{bytes\_per\_element}}$$

As B increases, I(B) increases until it crosses the ridge point → transition from
bandwidth-bound to compute-bound.

**Optimal batch size**: where arithmetic intensity crosses I\*.

**A100 example** (d=4096, s=2048, BF16):

- B=1: I = 2×2048×4096 / (2×2048×4096 + 4096²) × 2 ≈ 0.5 → bandwidth-bound
- B=64: I ≈ 31 → still bandwidth-bound
- B=256: I ≈ 95 → approaching compute-bound
- B=512: I ≈ 128 → near ridge point

### 5.5 Prefill-Decode Disaggregation (2024–2026)

**Problem**: prefill (compute-bound) and decode (bandwidth-bound) have different
optimal hardware configurations.

**Solution**: run prefill and decode on separate GPU pools.

- Prefill servers: high compute density (H100); process prompts
- Decode servers: high bandwidth GPUs; generate tokens
- KV cache transfer: after prefill, move KV cache to decode server via NVLink / InfiniBand

```
                    ┌──────────────┐
User Request ──────│ Router/LB    │
                    └──────┬───────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
    ┌─────────────────┐       ┌─────────────────┐
    │  Prefill Pool   │──KV──▶│  Decode Pool    │
    │  (compute-opt)  │ xfer  │  (bandwidth-opt)│
    └─────────────────┘       └─────────────────┘
```

Better hardware utilisation; lower TTFT without sacrificing decode throughput.

Adopted by: Anthropic, Google, ByteDance (2024–2026).

### 5.6 Scheduling Policies

| Policy              | Mechanism                                     | Tradeoff                                         |
| ------------------- | --------------------------------------------- | ------------------------------------------------ |
| **FCFS**            | First-come first-served                       | Simple; head-of-line blocking                    |
| **SJF**             | Shortest job first                            | Optimal average latency; needs length prediction |
| **SLO-aware**       | Prioritise requests at risk of violating SLOs | Complex; business-optimal                        |
| **Priority queues** | Different SLOs for different API tiers        | Standard in production                           |

Output length prediction: use small classifier to predict output length; schedule short
jobs first for better average latency.

---

## 6. Speculative Decoding

### 6.1 The Core Idea

**Observation**: a small model (draft) generates tokens fast; a large model (target) is
slow but accurate.

**Speculation**: use draft model to propose K tokens at once; verify all K with target
model in a single parallel forward pass.

- If draft is correct: accept K tokens at cost of ~1 target forward pass
- If draft is wrong at position j: accept tokens 0…j−1; reject from j onward; resample

```
Draft model proposes: [The] [cat] [sat] [on] [the]     (5 tokens, fast)
Target model verifies: [The]✓ [cat]✓ [sat]✓ [in]✗      (parallel verify)
Accept: [The] [cat] [sat]                               (3 accepted)
Resample position 4 from target distribution             (1 corrected token)
Net: 4 tokens from 1 target forward pass                 (vs 4 separate passes)
```

### 6.2 Mathematics of Speculative Decoding (Leviathan et al. 2023)

Draft model distribution **p**(x); target model distribution **q**(x).

Accept token x at position i with probability:

$$P(\text{accept}) = \min\!\left(1, \frac{q(x_i \mid x_{<i})}{p(x_i \mid x_{<i})}\right)$$

If rejected, resample from the adjusted distribution:

$$p'(x) = \text{normalise}\!\left(\max\!\left(0,\; q(x) - p(x)\right)\right)$$

**Key theorem**: the output distribution is **exactly** q (the target distribution) — not
an approximation.

**Proof sketch**: For any token x:

- With probability p(x): draft proposes x; accepted with prob min(1, q(x)/p(x))
- If q(x) ≥ p(x): always accept → contribution = p(x) · 1 = p(x)
- If q(x) < p(x): accept with q(x)/p(x) → contribution = p(x) · q(x)/p(x) = q(x)
- Rejected tokens resampled from max(0, q−p) normalised → fills remaining probability
- Total = target distribution q (exact)

### 6.3 Expected Tokens Per Step

Expected tokens accepted from K draft tokens:

$$\mathbb{E}[\text{tokens accepted}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

where α = expected acceptance rate per token (α close to 1 → nearly K tokens per step).

| α    | K=4  | K=8  | K=16  |
| ---- | ---- | ---- | ----- |
| 0.5  | 1.94 | 2.00 | 2.00  |
| 0.7  | 2.83 | 3.20 | 3.33  |
| 0.8  | 3.36 | 4.16 | 4.76  |
| 0.9  | 3.81 | 5.70 | 8.23  |
| 0.95 | 4.15 | 6.63 | 11.50 |

### 6.4 Speedup Analysis

| Component             | Standard     | Speculative                     |
| --------------------- | ------------ | ------------------------------- |
| Generate K tokens     | K × C_target | K × C_draft + 1 × C_target      |
| Effective tokens/step | 1            | E[accepted] ≈ (1−α^{K+1})/(1−α) |

Speedup ≈ E[accepted] × C_target / (C_target + K × C_draft)

Typical: **2–3× speedup** for well-matched draft/target pairs.

### 6.5 Draft Model Selection

| Strategy            | Example                  | Tradeoff                  |
| ------------------- | ------------------------ | ------------------------- |
| Smaller same family | LLaMA-3 8B → LLaMA-3 70B | High α; extra memory      |
| Same family ratio   | 7–15% of target params   | Sweet spot                |
| Self-speculative    | Early exit from target   | No extra memory; lower α  |
| Prompt lookup       | n-gram match in input    | Zero cost; task-dependent |

Quality requirement: acceptance rate α > 0.7 for meaningful speedup.

### 6.6 Self-Speculative Decoding and Medusa

**Self-speculative**: use early exit from the target model itself as the draft.
First l layers → draft token; all L layers → verify. No separate draft model; saves memory.

**Medusa** (Cai et al. 2024): add multiple decoding heads to target model.

- Head k predicts token at position t+k; trained jointly
- Tree-based verification: evaluate multiple candidate continuations in one pass
- 2–3× speedup with no additional model

### 6.7 EAGLE (Li et al. 2024)

Draft model uses target model's **hidden states** as input (not just token embeddings).
Autoregressive draft on feature space instead of token space.

- Higher acceptance rate than token-level draft
- **3–4× speedup** typical
- EAGLE-2: dynamic draft tree; adaptively expand tree based on confidence

Standard in production speculative decoding as of 2025–2026.

### 6.8 Prompt Lookup Decoding

For inputs with repetition (documents, code): find matching n-gram in prompt; propose
as draft.

- No draft model needed; zero extra memory
- High acceptance for summarisation, editing tasks; 2–4× speedup
- Falls back to standard decoding when no match found

---

## 7. Quantisation for Inference

### 7.1 Why Quantise?

| Precision | Bytes/param | LLaMA-3 70B | Hardware needed  |
| --------- | ----------- | ----------- | ---------------- |
| FP32      | 4           | 280 GB      | 4× A100 80GB     |
| BF16      | 2           | 140 GB      | 2× A100 80GB     |
| INT8      | 1           | 70 GB       | 1× A100 80GB     |
| INT4      | 0.5         | 35 GB       | 2× RTX 4090 24GB |

Bandwidth reduction: INT4 weights load 4× faster than FP32 → direct decode speedup.

Key tradeoff: lower precision → lower memory/latency vs potential quality degradation.

### 7.2 Quantisation Fundamentals

Map floating-point values to integers:

$$x_q = \text{round}\!\left(\frac{x - z}{s}\right), \quad \hat{x} = s \cdot x_q + z$$

where s = scale factor, z = zero point.

Quantisation error: ε = x − x̂; minimise ‖ε‖ over tensor.

**Clipping**: values outside [min, max] are clipped before quantisation → outliers cause
large error.

**Granularity levels:**

| Granularity | Scale factors              | Quality | Overhead |
| ----------- | -------------------------- | ------- | -------- |
| Per-tensor  | 1 (s, z) for entire tensor | Lowest  | Minimal  |
| Per-channel | 1 (s, z) per row/column    | Good    | Small    |
| Per-group   | 1 (s, z) per g elements    | Best    | Moderate |

### 7.3 Post-Training Quantisation (PTQ)

Quantise after training; no retraining required.

**GPTQ** (Frantar et al. 2022):

- Layer-wise quantisation; minimise ‖WX − W_qX‖²
- Second-order optimisation using inverse Hessian H⁻¹ to compensate quantisation error
- Quantise weights column by column; update remaining columns to compensate
- INT4 with minimal quality loss; standard for consumer deployment

**AWQ — Activation-Aware Weight Quantisation** (Lin et al. 2023):

- Not all weights are equally important; salient weights determined by activation magnitudes
- Scale salient channels up before quantisation; scale back at inference
- Better than GPTQ on many benchmarks; faster calibration
- Default quantisation for llama.cpp, Ollama

### 7.4 Quantisation Formats

| Format      | Bits | Range       | Key Property                            |
| ----------- | ---- | ----------- | --------------------------------------- |
| FP32        | 32   | ±3.4×10³⁸   | Full precision; training master weights |
| BF16        | 16   | ±3.4×10³⁸   | Same range as FP32; standard training   |
| FP16        | 16   | ±65504      | Narrower range; needs loss scaling      |
| FP8 E4M3    | 8    | ±448        | Forward pass; better precision          |
| FP8 E5M2    | 8    | ±57344      | Gradients; better range                 |
| INT8        | 8    | [−128, 127] | Weights + activations inference         |
| INT4        | 4    | [−8, 7]     | Weights-only inference; GPTQ/AWQ        |
| NF4         | 4    | non-uniform | Optimal for normal distributions; QLoRA |
| GGUF Q4_K_M | ~4.5 | mixed       | llama.cpp format; mixed 4/6-bit         |

### 7.5 Activation Quantisation

Weights quantisation is easy — weights are static after training. Activation quantisation
is harder — activations have dynamic range that changes per input.

**Outlier problem**: transformer activations have extreme outliers in specific channels,
~100× larger than typical values.

**LLM.int8()** (Dettmers 2022): mixed-precision decomposition.

- Identify outlier channels (> threshold); keep those in FP16
- Quantise remaining channels to INT8
- Matrix multiply: FP16 for outlier columns + INT8 for rest; combine

**SmoothQuant** (Xiao et al. 2022):

- Migrate quantisation difficulty from activations to weights
- Divide activation by per-channel scale s; multiply weights by same s:

$$Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \, W) = X_{\text{smooth}} \cdot W_{\text{smooth}}$$

Activations become smoother (easier to quantise); weights slightly harder → both
quantisable to INT8.

### 7.6 KV Cache Quantisation (Revisited)

| Format   | Memory       | Quality            | Notes                      |
| -------- | ------------ | ------------------ | -------------------------- |
| BF16     | 1×           | Baseline           | —                          |
| FP8 E4M3 | 2× reduction | Near-lossless      | H100 hardware accelerated  |
| INT8     | 2× reduction | Near-lossless      | Widely deployed            |
| INT4     | 4× reduction | Slight degradation | Per-group scaling required |

Per-token scaling: compute scale before storing; dequantise before attention.
Grouped quantisation: share scale across g=64 or g=128 tokens.

### 7.7 Quantisation-Aware Training (QAT)

Simulate quantisation during training; straight-through estimator for gradients.

Forward: quantise then dequantise ("fake quantisation"):

$$\hat{W} = Q(W), \quad z = \hat{W}x \quad \text{(quantised forward)}$$

Backward: straight-through estimator ignores quantisation:

$$\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial \hat{W}}$$

More expensive than PTQ but significantly better quality at low bits (2–3 bit).

**BitNet** (Wang et al. 2023): 1-bit weights {−1, +1} trained from scratch.

**BitNet b1.58** (Ma et al. 2024): ternary weights {−1, 0, +1}; matches FP16 quality
at 3B+ parameters.

Extreme quantisation becoming viable for smaller models in 2025–2026.

---

## 8. Model Compression — Beyond Quantisation

### 8.1 Pruning

Remove weights or structures that contribute little to output quality.

**Unstructured pruning**: zero out individual weights; sparse matrix.

- High sparsity possible (90%+) with small quality loss
- Problem: sparse matrices not efficiently supported by modern GPUs (dense ops are faster)

**Structured pruning**: remove entire attention heads, FFN neurons, or layers.

- Hardware-friendly: remaining structure is dense
- Remove heads with lowest importance score; retrain to recover quality
- **Layer pruning** — ShortGPT (2024): 25% of layers removable with acceptable quality loss

### 8.2 Knowledge Distillation

Train small student model to mimic large teacher model.

$$\mathcal{L} = \alpha \,\mathcal{L}_{\text{CE}}(y, y_{\text{true}}) + (1-\alpha) \,\mathcal{L}_{\text{KL}}(P_{\text{student}} \| P_{\text{teacher}})$$

Student learns from **soft labels** (teacher probability distributions) not just hard
labels. Soft labels contain "dark knowledge": relative probabilities among wrong answers.

**Temperature distillation**: use τ > 1 to soften teacher distribution:

$$\mathcal{L}_{\text{distil}} = \tau^2 \cdot D_{KL}\!\left(\text{softmax}(z_T/\tau) \,\|\, \text{softmax}(z_S/\tau)\right)$$

The τ² factor compensates for the reduced gradient magnitudes when softening
distributions.

Examples: DistilBERT (66% size, 97% performance), TinyLLaMA, MiniLLM.

### 8.3 Structured Decomposition

Decompose weight matrix W ≈ UV where U ∈ ℝ^{m×r}, V ∈ ℝ^{r×n}, r ≪ min(m,n).

- Reduces compute: m×n → r(m+n)
- Reduces parameters similarly
- SVD-LLM (2024): apply SVD to attention and FFN weights; truncate small singular values
- Limitation: quality degrades faster than parameter reduction

### 8.4 Layer Sharing and Weight Tying

**ALBERT**: all transformer layers share same weights → 89% parameter reduction.
Quality degrades significantly for generation tasks; mainly used for encoders.

**Weight tying**: embedding E and LM head W^T share same matrix.

- Saves 2 × V × d parameters (V = vocab size, d = hidden dim)
- Standard in most LLMs
- Constraint: input and output embedding spaces must be identical

---

## 9. Inference Frameworks and Systems

### 9.1 vLLM (Kwon et al. 2023)

| Feature     | Detail                          |
| ----------- | ------------------------------- |
| KV cache    | PagedAttention                  |
| Batching    | Continuous (iteration-level)    |
| Multi-GPU   | Tensor + pipeline parallelism   |
| Speculative | Integrated speculative decoding |
| Caching     | Automatic prefix caching        |

Standard open-source inference framework; most widely deployed.

### 9.2 TensorRT-LLM (NVIDIA 2023)

- NVIDIA-specific; compiles model to optimised TensorRT engine
- Fused kernels: attention + layernorm + activation in single CUDA kernel
- INT8/INT4/FP8 inference with hardware acceleration
- Best raw throughput on NVIDIA hardware; less flexible than vLLM

### 9.3 SGLang (Zheng et al. 2024)

- **RadixAttention**: tree-structured prefix caching; automatic KV reuse
- Efficient constraint decoding: structured output (JSON, regex) via FSM
- **5× throughput improvement** over vLLM for multi-call workloads

### 9.4 llama.cpp

- C++ inference engine; CPU + GPU hybrid computation
- GGUF quantisation format: mixed-precision per-layer quantisation
- Runs LLaMA-3 70B on consumer hardware (64 GB RAM + GPU)
- ARM NEON, AVX-512, Metal (Apple Silicon) backends
- Primary framework for edge/local deployment

### 9.5 Inference on Apple Silicon (2024–2026)

- **Unified memory**: CPU and GPU share same memory pool; no PCIe transfer
- LLaMA-3 8B: 16 GB unified memory; runs entirely on M2/M3 Mac
- Metal Performance Shaders (MPS): GPU-accelerated matrix operations
- Memory bandwidth: M3 Max = 300 GB/s; comparable to A100 for bandwidth-bound decode
- Ollama: one-click deployment on Apple Silicon via llama.cpp backend

### 9.6 Key Kernel Optimisations

| Optimisation            | Mechanism                                                 | Impact                    |
| ----------------------- | --------------------------------------------------------- | ------------------------- |
| **Fused operations**    | Combine layernorm + projection → single kernel            | Reduce HBM round-trips    |
| **In-place operations** | Reuse buffers; avoid unnecessary allocation               | Lower memory footprint    |
| **CUDA graphs**         | Capture op sequence as graph; replay without CPU overhead | 10–30% latency reduction  |
| **Persistent kernels**  | Keep computation on GPU between tokens                    | Reduce launch overhead    |
| **Custom kernels**      | Hand-tuned matmul/attention for specific shapes           | Best-possible performance |

---

## 10. Multi-Token Prediction

### 10.1 Standard Autoregressive Limitation

Standard LM: predict one token per forward pass; O(n) passes for n tokens.

Every forward pass loads all model weights from HBM → the bandwidth bottleneck is
repeated n times. Can we predict multiple tokens per forward pass?

### 10.2 Multi-Token Prediction (Gloeckle et al. 2024, Meta)

Add k independent LM heads at the final layer; each predicts token at offset +1, +2, …, +k.

Training: sum losses across all k heads:

$$\mathcal{L} = \sum_{i=1}^{k} \lambda_i \cdot \mathcal{L}_{\text{CE}}(\text{head}_i, y_{t+i})$$

Inference: use additional heads as draft tokens for speculative decoding.

Benefits:

- Improved representations from multi-step prediction signal
- Built-in draft for speculation (no separate draft model)
- LLaMA-3 models include 4-token multi-token prediction heads
- **2× inference speedup** reported

### 10.3 Jacobi Decoding

Iterative parallel decoding:

1. Initialise all n output tokens randomly
2. Update all token predictions simultaneously in parallel
3. Repeat until convergence (typically 1–3 iterations)

Problem: requires n forward passes through full model per iteration; gains limited.

**CLLMs** (Consistency Large Language Models, 2024): train model to converge in 1
iteration of Jacobi decoding.

### 10.4 Parallel Decoding Methods Comparison

| Method           | Exact Distribution | Speedup | Extra Memory |
| ---------------- | ------------------ | ------- | ------------ |
| Standard         | ✅                 | 1×      | 0            |
| Speculative      | ✅                 | 2–4×    | Draft model  |
| Medusa           | ≈ (tree verify)    | 2–3×    | Extra heads  |
| EAGLE            | ≈ (tree verify)    | 3–4×    | Small model  |
| Multi-token pred | ≈                  | 2×      | Extra heads  |
| Jacobi           | ❌                 | 1.5–2×  | 0            |

---

## 11. Long-Context Inference

### 11.1 Memory vs Compute Tradeoff at Long Context

| Sequence length | KV cache (70B, BF16) | Attention FLOPs/layer |
| --------------- | -------------------- | --------------------- |
| 4K              | 1.3 GB               | 0.13 GFLOPs           |
| 32K             | 10.5 GB              | 8.6 GFLOPs            |
| 128K            | 42 GB                | 134 GFLOPs            |
| 1M              | 335 GB               | 8.2 TFLOPs            |

At n = 1M tokens: KV cache alone exceeds a single GPU's HBM; O(n²) attention is
infeasible on a single device. Must use approximate or sparse attention.

### 11.2 Sliding Window + Global Tokens (Mistral, Longformer)

Window size w: each token attends to w past tokens only → **O(nw)** memory + compute.

Global tokens (BOS, special): attend to all positions; provide long-range information flow.

```
          Token positions
          1  2  3  4  5  6  7  8  9  10
Token 1  [G  .  .  .  .  .  .  .  .  . ]  ← global
Token 5  [.  .  . (x)(x)(x)(x) .  .  . ]  ← window=3
Token 10 [.  .  .  .  .  .  . (x)(x)(x)]  ← window=3
```

Mistral 7B: w=4096 window; effective context limited to window despite 32K nominal
context length.

### 11.3 Ring Attention for Training and Inference

Distribute sequence across k GPUs: each GPU holds n/k tokens of KV cache.

```
GPU 0: [tokens 0..n/k]  ──KV──→ GPU 1 ──KV──→ GPU 2 ──KV──→ GPU 3
                                                                │
GPU 0  ←──KV── GPU 3 ←──KV── GPU 2 ←──KV── GPU 1  ←──────────┘
                            (ring topology)
```

- Each GPU computes attention over its local KV and received remote KV
- Full attention computed without any single GPU holding full n×n matrix
- Memory per GPU: **O(n/k)** KV cache; scales linearly with GPU count
- Communication hidden behind compute with async implementation

### 11.4 Retrieval-Augmented Generation (RAG) vs Long Context

| Aspect          | Long Context                                    | RAG                                   |
| --------------- | ----------------------------------------------- | ------------------------------------- |
| Coherence       | ✅ Model sees all information                   | ⚠️ Retrieval may miss relevant pieces |
| Cost            | O(n²) attention; expensive                      | O(k) for k retrieved chunks; cheap    |
| Corpus size     | Limited by context window                       | Arbitrary — database can be huge      |
| Quality         | Better for complex reasoning                    | Better for factual lookup             |
| Trend 2024–2026 | Becoming cheap (Flash-Decoding, KV compression) | Being replaced for many use cases     |

Gemini 1.5 Pro at 1M context; LLaMA-3.1 at 128K context.

### 11.5 Needle-in-a-Haystack Analysis

Standard benchmark: insert specific fact ("needle") at various positions in a long
document ("haystack"). Measure retrieval accuracy vs needle position and context length.

Key findings:

- Most models show **U-shaped performance**: strong at start/end; weak in middle
- "Lost in the Middle" (Liu et al. 2023): recency + primacy bias in attention
- **RULER** (Hsieh et al. 2024): multi-hop retrieval; harder than single needle

```
Accuracy
100% ████                                          ████
 80% ████▄                                      ▄██████
 60% ██████▄                                  ▄████████
 40% ████████▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄██████████
 20%
     Start          Middle                    End
              Needle Position →
```

---

## 12. Inference for Special Architectures

### 12.1 MoE Inference

Expert routing: each token selects top-k experts; only k of N experts computed.

- Active parameters per token: (k/N) × total params
- Example: DeepSeek-V3: 37B active / 671B total
- **All experts must be loaded** in GPU memory — cannot predict which will be selected
- Expert parallelism: distribute experts across GPUs; all-to-all routing
- Expert offloading (CPU/SSD): load on demand; high latency; offline only

Mixtral 8×7B: 12.9B active params; inference cost ≈ 12.9B dense; quality ≈ 47B dense.

### 12.2 SSM / Mamba Inference

Recurrent form at inference: **O(1)** compute and memory per token (no KV cache).

| Step         | Operation                | Complexity     |
| ------------ | ------------------------ | -------------- |
| State update | h*t = A·h*{t-1} + B·x_t  | O(d²) per step |
| Output       | y_t = C·h_t              | O(d) per step  |
| Memory       | Fixed-size state h ∈ ℝ^d | O(d) total     |

No attention matrix; no KV cache; fixed memory regardless of sequence length.

Tradeoff: O(1) inference but cannot do exact retrieval at arbitrary position (no random
access to past tokens).

Mamba-2 (Dao & Gu 2024): structured SSM with semi-separable matrices; faster hardware
implementation.

### 12.3 Hybrid Attention-SSM Inference

| Model             | Architecture                           | KV cache savings |
| ----------------- | -------------------------------------- | ---------------- |
| Jamba (AI21 2024) | Alternating Transformer + Mamba layers | 2–4×             |
| Zamba             | Sparse attention + Mamba               | 3–5×             |
| Hymba             | Different mixing ratios                | Variable         |

Attention layers: use KV cache (expensive for long sequences).
Mamba layers: use fixed-size recurrent state (cheap for long sequences).
KV cache only needed for attention layers → significant KV cache reduction.

### 12.4 Diffusion Language Models (2024–2026)

- Mask-based discrete diffusion: MDLM, PLAID (2024)
- Generate all tokens simultaneously; iteratively refine via masked prediction
- Not autoregressive: O(T) passes for T refinement steps (typically 10–50)
- Quality gap vs autoregressive: closing rapidly in 2025–2026
- All tokens generated simultaneously → different latency profile

---

## 13. Hardware for Inference

### 13.1 GPU Inference (NVIDIA)

| GPU         | BF16 TFLOPS | HBM BW (TB/s) | HBM (GB) | Ridge Point |
| ----------- | ----------- | ------------- | -------- | ----------- |
| A100 SXM    | 312         | 2.0           | 80       | 156         |
| H100 SXM    | 989         | 3.35          | 80       | 295         |
| H200 SXM    | 989         | 4.8           | 141      | 206         |
| B200 (2025) | ~4500 (FP8) | ~8.0          | 192      | ~563        |

Key metric for inference: bandwidth / model_size_bytes = maximum decode tokens/second.

### 13.2 Memory Bandwidth Bound — Decode Speed Limit

Upper bound on decode speed (bandwidth-limited, batch=1):

$$\text{max tokens/sec} = \frac{\text{HBM bandwidth}}{\text{model size (bytes)} + \text{KV cache bytes per step}}$$

**Examples** (ignoring KV cache overhead for simplicity):

| Hardware | Model       | Precision     | Max tok/s (batch=1) |
| -------- | ----------- | ------------- | ------------------- |
| H100     | LLaMA-3 70B | BF16 (140 GB) | 3350/140 ≈ 24       |
| H200     | LLaMA-3 70B | BF16 (140 GB) | 4800/140 ≈ 34       |
| H200     | LLaMA-3 70B | INT4 (35 GB)  | 4800/35 ≈ 137       |
| H100     | LLaMA-3 8B  | BF16 (16 GB)  | 3350/16 ≈ 209       |

INT4 quantisation: 4× decode speedup at same hardware (plus fits in fewer GPUs).

### 13.3 TPU Inference (Google)

| TPU           | TFLOPS   | HBM BW (TB/s) | HBM (GB) |
| ------------- | -------- | ------------- | -------- |
| TPU v4        | 275 BF16 | 1.2           | 32       |
| TPU v5e       | 393 INT8 | —             | —        |
| TPU v5p       | 459 BF16 | —             | —        |
| Trillium (v6) | ~4× v5e  | —             | —        |

All Gemini inference runs on TPUs; estimated millions of chips deployed.

### 13.4 Custom Inference ASICs

| ASIC                    | Key Property                                          | Use Case           |
| ----------------------- | ----------------------------------------------------- | ------------------ |
| **Groq LPU**            | Deterministic; weights in SRAM; 750 tok/s LLaMA-2 70B | Ultra-low latency  |
| **Cerebras WSE-3**      | 900K cores; entire model on chip                      | Extreme throughput |
| **Inferentia (AWS)**    | Cost-optimised INT8/FP8                               | Amazon Bedrock     |
| **Apple Neural Engine** | 38 TOPS on-device                                     | Apple Intelligence |
| **Qualcomm NPU**        | INT4; 45 TOPS                                         | Mobile/edge        |

### 13.5 Memory Hierarchy for Inference

| Memory Type      | Bandwidth | Size      | Latency |
| ---------------- | --------- | --------- | ------- |
| SRAM (L1/Shared) | ~20 TB/s  | ~20 MB    | ~1 ns   |
| HBM (GPU main)   | 3–8 TB/s  | 40–192 GB | ~100 ns |
| GDDR7 (consumer) | ~1 TB/s   | 16–32 GB  | ~100 ns |
| CPU DRAM         | ~100 GB/s | 64–512 GB | ~100 ns |
| PCIe 5.0         | ~64 GB/s  | —         | ~1 μs   |
| NVLink 4.0       | ~900 GB/s | —         | ~1 μs   |
| NVMe SSD         | ~7 GB/s   | TBs       | ~100 μs |

The entire FlashAttention approach exploits the 100× bandwidth gap between SRAM and HBM.

---

## 14. Cost Optimisation

### 14.1 Cost Model for Inference

$$\text{Cost per token} = \frac{\text{GPU cost/hour}}{\text{tokens/hour}} = \frac{\text{GPU cost/hour}}{\text{tokens/sec} \times 3600}$$

Tokens/sec depends on: batch size, model size, hardware, quantisation, speculative
decoding.

**Example** — H100 at \$3/hour, 500 tokens/sec throughput:
$$\text{Cost} = \frac{3.00}{500 \times 3600} = \$1.67 \times 10^{-6} \text{ per token} = \$1.67 \text{ per million tokens}$$

### 14.2 Cost Breakdown

| Phase    | Cost Driver                             | Scalability                     |
| -------- | --------------------------------------- | ------------------------------- |
| Prefill  | input_length × model_size               | Parallelisable; cheap per token |
| Decode   | output_length × model_size / batch_size | Sequential; expensive           |
| KV cache | (input + output) length × KV_per_token  | Memory, not compute             |

Decode dominates total cost for generation-heavy workloads.

### 14.3 Cost Reduction Strategies

| Strategy             | Typical Savings            | Quality Impact |
| -------------------- | -------------------------- | -------------- |
| INT8 quantisation    | 1.5–2×                     | Minimal        |
| INT4 quantisation    | 3–4×                       | Small          |
| Speculative decoding | 2–3×                       | None (exact)   |
| GQA (vs MHA)         | 1.5–2× KV                  | Minimal        |
| Continuous batching  | 5–23× throughput           | None           |
| Prefix caching       | 2–10× for repeated prompts | None           |
| Flash-Decoding       | 2–8× decode latency        | None           |
| Model distillation   | 3–10×                      | Moderate       |

### 14.4 Prompt Optimisation

Shorter prompts = less prefill compute + less KV cache memory.

**Prompt compression** (LLMLingua 2023):

- Compress prompt by 2–20× with small proxy LM
- Remove tokens with low conditional probability; preserve high-information tokens
- Near-lossless at 2–4× compression; moderate degradation at 10–20×

**Prompt caching**: charge less for cached prefix tokens (Anthropic, OpenAI both offer this;
typical 50% discount on cached input tokens).

---

## 15. Common Mistakes

| #   | Mistake                                        | Why It's Wrong                                                        | Fix                                                                        |
| --- | ---------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1   | "Larger batch always faster per token"         | Bandwidth-bound region: larger batch doesn't help until compute-bound | Profile arithmetic intensity; find optimal batch for your hardware         |
| 2   | "INT4 is always good enough"                   | INT4 degrades on reasoning, math, code more than simple tasks         | Evaluate on task-specific benchmarks; use INT8 for quality-sensitive cases |
| 3   | "Speculative decoding is always faster"        | Requires draft model memory + compute; low α → no speedup             | Measure acceptance rate α; deploy only if α > 0.7                          |
| 4   | "KV cache is just a memory concern"            | KV cache size also limits batch size → limits throughput              | Jointly optimise architecture and KV budget                                |
| 5   | "Pre-allocate max KV cache per request"        | Massive fragmentation; 60–80% memory waste                            | Use PagedAttention / vLLM for dynamic allocation                           |
| 6   | "TTFT and TPOT are independent"                | Chunked prefill trades TTFT for TPOT; must optimise jointly           | Define SLOs for both; tune chunk size accordingly                          |
| 7   | "Quantisation errors are uniform"              | Outlier channels cause disproportionate error                         | Use SmoothQuant or LLM.int8()                                              |
| 8   | "FlashAttention only helps for long sequences" | FA helps for any sequence length; IO-bound even at n=512              | Always use FlashAttention                                                  |
| 9   | "All inference frameworks are equivalent"      | vLLM, TRT-LLM, SGLang have very different performance profiles        | Benchmark on your workload                                                 |
| 10  | "Edge inference is a minor use case"           | Apple Intelligence serves millions of queries on-device               | Quantisation + compression critical for edge deployment                    |

---

## 16. Exercises

1. **Roofline analysis** — A100: 312 TFLOPS, 2 TB/s bandwidth. Compute ridge point I\*.
   For decode at batch=1, s=4096, d=4096: compute arithmetic intensity. Is it compute-
   or bandwidth-bound?

2. **KV cache sizing** — LLaMA-3 8B (L=32, H_kv=8, d_k=128). Compute KV cache for
   s=8192 in BF16 and INT8. How many concurrent requests fit in 80 GB GPU (16 GB reserved
   for weights)?

3. **Speculative decoding speedup** — draft acceptance rate α=0.8, K=4 draft tokens.
   Compute expected tokens per target call. Compute speedup vs standard decoding.

4. **FlashAttention IO** — n=4096, d=64, H=32. Compute HBM IO for standard attention
   vs FlashAttention. Compute speedup assuming bandwidth-bound.

5. **Quantisation tradeoff** — LLaMA-3 70B. Compute model size in BF16, INT8, INT4.
   Compute max decode tokens/sec on H100 for each (bandwidth = 3.35 TB/s).

6. **PagedAttention fragmentation** — 8 requests, max_length=2048, avg actual=512.
   Compute memory waste for static allocation vs PagedAttention (block_size=16).

7. **Continuous batching throughput** — static batching: batch=8, mean output=256,
   max output=1024, GPU = 2000 tokens/sec. Estimate waste and throughput improvement
   with continuous batching.

8. **Prefill-decode disaggregation** — prefill throughput 1000 tok/sec/GPU; decode
   throughput 50 tok/sec/GPU (batch=1). Workload: 100-token prompts + 500-token outputs.
   Compute optimal ratio of prefill to decode GPUs.

---

## 17. Why This Matters for AI (2026 Perspective)

| Aspect                    | Impact                                                                                        |
| ------------------------- | --------------------------------------------------------------------------------------------- |
| **Cost**                  | Inference optimisation directly reduces $/API call; 10× optimisation = 10× more accessible AI |
| **Latency**               | TTFT and TPOT determine user experience; sub-100ms TPOT feels real-time                       |
| **Scale**                 | Continuous batching + PagedAttention → 20× more concurrent users on same hardware             |
| **Long context**          | Flash-Decoding + KV compression → 1M-token context practical in production                    |
| **Edge AI**               | Quantisation + efficient kernels → frontier-class models on phones and laptops                |
| **Sustainability**        | 4× efficiency = 4× fewer GPUs = significant energy and carbon reduction                       |
| **Reasoning models**      | o1/R1-style models generate 10–100× more tokens; efficiency is existential                    |
| **Access**                | Efficient inference = lower prices = broader access globally                                  |
| **Safety**                | Faster inference → real-time safety filtering; latency no longer an excuse                    |
| **Architecture research** | MLA, SSM hybrids, multi-token prediction all driven by inference efficiency                   |

---

## Conceptual Bridge

Efficient inference is where mathematical theory meets hardware reality.
FlashAttention is applied linear algebra; speculative decoding is probability theory;
quantisation is numerical analysis.

Next: **Mixture of Experts and Routing** — how to scale model capacity while keeping
inference cost constant.

```
θ* → [Quantise/Compress] → θ_q → [Inference Engine] → Responses at scale
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                       THIS section
```

---

[← Scaling Laws](../08-Scaling-Laws/notes.md) | [Home](../../README.md) | [Mixture of Experts and Routing →](../10-Mixture-of-Experts-and-Routing/notes.md)
